"""Tests for igraph-accelerated system functions.

Verifies that igraph-based sfun produces identical results to NetworkX-based sfun.
Skipped if python-igraph is not installed.
"""
import pytest
import networkx as nx

try:
    import igraph
    HAS_IGRAPH = True
except ImportError:
    HAS_IGRAPH = False

pytestmark = pytest.mark.skipif(not HAS_IGRAPH, reason="python-igraph not installed")


def _make_test_graph():
    """Create a small test graph with edge IDs and known connectivity."""
    G = nx.Graph()
    # Simple path: A -- B -- C -- D with a branch B -- E
    edges = [
        ("A", "B", {"eid": "e1"}),
        ("B", "C", {"eid": "e2"}),
        ("C", "D", {"eid": "e3"}),
        ("B", "E", {"eid": "e4"}),
    ]
    G.add_edges_from(edges)
    return G


def _all_on(G):
    """Return comps_state with all edges on and all nodes on."""
    comps = {}
    for u, v, data in G.edges(data=True):
        comps[data["eid"]] = 1
    for n in G.nodes():
        comps[n] = 1
    return comps


class TestNxToIgraph:
    def test_conversion_preserves_structure(self):
        from rsr.igraph_sfun import nx_to_igraph

        G = _make_test_graph()
        ig_g, node_to_idx, eid_to_edge_idx, edge_endpoints = nx_to_igraph(G)

        assert ig_g.vcount() == G.number_of_nodes()
        assert ig_g.ecount() == G.number_of_edges()
        assert set(node_to_idx.keys()) == set(G.nodes())
        assert set(eid_to_edge_idx.keys()) == {"e1", "e2", "e3", "e4"}
        assert set(edge_endpoints.keys()) == {"e1", "e2", "e3", "e4"}


class TestGlobalConnectivity:
    def test_all_on(self):
        from rsr.igraph_sfun import make_igraph_sfun_global_conn

        G = _make_test_graph()
        sfun = make_igraph_sfun_global_conn(G, target_g_conn=1)
        comps = _all_on(G)
        k, sys_st, _ = sfun(comps)
        assert sys_st == 1
        assert k >= 1

    def test_edge_off_disconnects(self):
        from rsr.igraph_sfun import make_igraph_sfun_global_conn

        G = _make_test_graph()
        sfun = make_igraph_sfun_global_conn(G, target_g_conn=1)
        comps = _all_on(G)
        # Turn off edge e2 (B-C), disconnecting C and D from A,B,E
        comps["e2"] = 0
        k, sys_st, _ = sfun(comps)
        assert sys_st == 0
        assert k == 0

    def test_node_off(self):
        from rsr.igraph_sfun import make_igraph_sfun_global_conn

        G = _make_test_graph()
        sfun = make_igraph_sfun_global_conn(G, target_g_conn=1)
        comps = _all_on(G)
        # Turn off node B, which disconnects the graph
        comps["B"] = 0
        k, sys_st, _ = sfun(comps)
        assert sys_st == 0


class TestODConnectivity:
    def test_connected_path(self):
        from rsr.igraph_sfun import make_igraph_sfun_conn

        G = _make_test_graph()
        sfun = make_igraph_sfun_conn(G, "A", "D")
        comps = _all_on(G)
        label, sys_st, info = sfun(comps)
        assert label == "connected"
        assert sys_st == 1
        assert info["connected"] is True
        assert info["path_nodes"] is not None
        assert info["path_nodes"][0] == "A"
        assert info["path_nodes"][-1] == "D"

    def test_disconnected_path(self):
        from rsr.igraph_sfun import make_igraph_sfun_conn

        G = _make_test_graph()
        sfun = make_igraph_sfun_conn(G, "A", "D")
        comps = _all_on(G)
        comps["e2"] = 0  # Remove B-C edge
        label, sys_st, info = sfun(comps)
        assert label == "disconnected"
        assert sys_st == 0
        assert info["connected"] is False

    def test_origin_off(self):
        from rsr.igraph_sfun import make_igraph_sfun_conn

        G = _make_test_graph()
        sfun = make_igraph_sfun_conn(G, "A", "D")
        comps = _all_on(G)
        comps["A"] = 0
        label, sys_st, info = sfun(comps)
        assert label == "disconnected"
        assert sys_st == 0
