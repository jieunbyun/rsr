"""
igraph-accelerated system functions for rsr.

Drop-in replacements for ndtools.fun_binary_graph functions that use igraph
(C-based) instead of NetworkX for graph operations. Typically 10-100x faster
for connectivity computations.

Requires: pip install python-igraph
"""
import warnings
import igraph as ig
import networkx as nx
from typing import Dict, Tuple, Any, Optional, List


def nx_to_igraph(G: nx.Graph) -> Tuple[ig.Graph, Dict[str, int], Dict[str, int]]:
    """
    Convert a NetworkX graph to igraph, preserving node names and edge IDs.

    Returns:
        (ig_graph, node_to_idx, eid_to_edge_idx, edge_endpoints)
        - node_to_idx: {node_name: igraph vertex index}
        - eid_to_edge_idx: {edge_id: igraph edge index}
        - edge_endpoints: {edge_id: (source_name, target_name)} for fast lookup
    """
    nodes = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    ig_g = ig.Graph(n=len(nodes), directed=False)
    ig_g.vs["name"] = nodes

    edges = []
    eids = []
    for u, v, data in G.edges(data=True):
        edges.append((node_to_idx[u], node_to_idx[v]))
        eids.append(data.get("eid"))

    ig_g.add_edges(edges)
    ig_g.es["eid"] = eids

    eid_to_edge_idx = {}
    edge_endpoints = {}
    for idx, eid in enumerate(eids):
        if eid is not None:
            eid_to_edge_idx[eid] = idx
            e = ig_g.es[idx]
            edge_endpoints[eid] = (ig_g.vs[e.source]["name"], ig_g.vs[e.target]["name"])

    return ig_g, node_to_idx, eid_to_edge_idx, edge_endpoints


def _build_subgraph(ig_graph, comps_state, node_to_idx, eid_to_edge_idx, edge_endpoints):
    """Build igraph subgraph from component states. Shared by global_conn and conn."""
    node_off = {cid for cid, st in comps_state.items() if st == 0 and cid in node_to_idx}
    edge_on = {cid for cid, st in comps_state.items() if st == 1}

    edges_to_keep = []
    for eid, idx in eid_to_edge_idx.items():
        if eid not in edge_on:
            continue
        u_name, v_name = edge_endpoints[eid]
        if u_name not in node_off and v_name not in node_off:
            edges_to_keep.append(idx)

    return ig_graph.subgraph_edges(edges_to_keep, delete_vertices=False)


def eval_global_conn_igraph(
    comps_state: Dict[str, int],
    ig_graph: ig.Graph,
    node_to_idx: Dict[str, int],
    eid_to_edge_idx: Dict[str, int],
    edge_endpoints: Dict[str, Tuple[str, str]],
    target_g_conn: int = 1,
) -> Tuple[int, int, None]:
    """
    igraph-accelerated version of eval_global_conn_k.

    For target_g_conn=1, uses is_connected() (BFS, O(V+E)) instead of
    vertex_connectivity() (max-flow, much more expensive).
    """
    H = _build_subgraph(ig_graph, comps_state, node_to_idx, eid_to_edge_idx, edge_endpoints)

    if H.vcount() <= 1:
        return 0, 0, None

    if target_g_conn <= 1:
        # Fast path: just check if connected (BFS) instead of computing exact k
        connected = H.is_connected()
        k_val = 1 if connected else 0
    else:
        k_val = H.vertex_connectivity()

    sys_st = 1 if k_val >= target_g_conn else 0
    return k_val, sys_st, None


def eval_1od_connectivity_igraph(
    comps_state: Dict[str, int],
    ig_graph: ig.Graph,
    node_to_idx: Dict[str, int],
    eid_to_edge_idx: Dict[str, int],
    edge_endpoints: Dict[str, Tuple[str, str]],
    orig_node: str,
    dest_node: str,
) -> Tuple[str, int, Dict[str, Any]]:
    """
    igraph-accelerated version of eval_1od_connectivity.

    Same semantics: check if path exists between orig and dest under component states.
    """
    # Quick failure if origin or destination is off
    if comps_state.get(orig_node) == 0 or comps_state.get(dest_node) == 0:
        return "disconnected", 0, {"connected": False, "path_nodes": None, "path_edge_ids": None}

    H = _build_subgraph(ig_graph, comps_state, node_to_idx, eid_to_edge_idx, edge_endpoints)

    orig_idx = node_to_idx[orig_node]
    dest_idx = node_to_idx[dest_node]

    # Check connectivity and find shortest path
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Couldn't reach some vertices")
        paths = H.get_shortest_paths(orig_idx, to=dest_idx, output="vpath")
    path_v = paths[0] if paths else []

    if path_v:
        connected = True
        path_nodes = [ig_graph.vs[v]["name"] for v in path_v]
        # Extract edge IDs along the path
        path_eids = []
        for i in range(len(path_v) - 1):
            eid_found = H.get_eid(path_v[i], path_v[i + 1], error=False)
            if eid_found >= 0:
                path_eids.append(H.es[eid_found]["eid"])
            else:
                path_eids.append(None)
    else:
        connected = False
        path_nodes = None
        path_eids = None

    info = {
        "connected": connected,
        "path_nodes": path_nodes,
        "path_edge_ids": path_eids,
    }
    return ("connected" if connected else "disconnected"), (1 if connected else 0), info


def make_igraph_sfun_global_conn(G: nx.Graph, target_g_conn: int = 1):
    """
    Create an igraph-accelerated sfun for global connectivity.

    For target_g_conn=1, uses is_connected() (BFS) instead of
    vertex_connectivity() (max-flow) — typically 10-50x faster per call.

    Usage:
        sfun = make_igraph_sfun_global_conn(G, target_g_conn=1)
        k, sys_st, _ = sfun(comps_st)
    """
    ig_graph, node_to_idx, eid_to_edge_idx, edge_endpoints = nx_to_igraph(G)

    def sfun(comps_st):
        return eval_global_conn_igraph(
            comps_st, ig_graph, node_to_idx, eid_to_edge_idx,
            edge_endpoints, target_g_conn)

    return sfun


def make_igraph_sfun_conn(G: nx.Graph, orig_node: str, dest_node: str):
    """
    Create an igraph-accelerated sfun for 1OD connectivity.

    Usage:
        sfun = make_igraph_sfun_conn(G, hub, dest)
        fval, sys_st, info = sfun(comps_st)
    """
    ig_graph, node_to_idx, eid_to_edge_idx, edge_endpoints = nx_to_igraph(G)

    def sfun(comps_st):
        return eval_1od_connectivity_igraph(
            comps_st, ig_graph, node_to_idx, eid_to_edge_idx,
            edge_endpoints, orig_node, dest_node)

    return sfun
