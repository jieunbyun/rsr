import json
from pathlib import Path
from ndtools.network_generator import GenConfig, generate_and_save
from ndtools.io import load_json
from ndtools.graphs import build_graph
from ndtools.fun_binary_graph import eval_1od_connectivity, eval_global_conn_k
import networkx as nx
from mbnpy import brc
import torch
import numpy as np
import gc

import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from rsr import rsr

def generate_random_network_data(name: str = "rg", 
                                 generator = "rg",
                                 generator_params={"n_nodes": 60, "radius": 0.25, "p_fail": 0.1}, 
                                 target_g_conn = 1,
                                 min_g_conn = 0, # Minimum global connectivity (k) required for the generated graph. Only used if find_connected_graph=True.
                                 out_base: Path = None,
                                 seed: int = 7,
                                 find_connected_graph: bool = True) -> Path:
    print(f"Generating random network data {name} with params: {generator_params} ..")

    if find_connected_graph: # Try multiple times with different seeds until we find a connected graph (or give up after max_tries)
        max_tries = 100  # safety guard

        for i in range(max_tries):
            seed_i = (seed + i) if seed is not None else None
            print(f"Try {i+1}/{max_tries} with seed={seed_i} ...")

            cfg = GenConfig(
                name=name,
                generator=generator,
                description=", ".join(f"{k}={v}" for k, v in generator_params.items()),
                generator_params=generator_params,
                seed=seed_i,
            )

            ds_root = generate_and_save(out_base, cfg, draw_graph=True)

            nodes = load_json(ds_root / "data" / "nodes.json")
            edges = load_json(ds_root / "data" / "edges.json")
            probs = load_json(ds_root / "data" / "probs.json")

            G = build_graph(nodes, edges, probs)

            if G.number_of_nodes() > 0 and nx.is_connected(G):
                
                # Check if global connectivity meets the target
                _, k, _ = eval_global_conn_k({x:1 for x in probs.keys()}, G)

                if k >= min_g_conn:
                    print(f"Found connected graph on try {i+1} (seed={seed_i}). Wrote: {ds_root}")
                    break
        else:
            print(f"No connected graph found after {max_tries} tries.")
            return None, None
        
    else: # Just generate once with the given seed (which may or may not yield a connected graph)
        cfg = GenConfig(
            name=name,
            generator=generator,
            description=", ".join(f"{k}={v}" for k, v in generator_params.items()),
            generator_params=generator_params,
            seed=seed,
        )

        ds_root = generate_and_save(out_base, cfg, draw_graph=True)

        nodes = load_json(ds_root / "data" / "nodes.json")
        edges = load_json(ds_root / "data" / "edges.json")
        probs = load_json(ds_root / "data" / "probs.json")

        G = build_graph(nodes, edges, probs)

    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    # Decide the hub node (node with highest degree)
    hub = max(G.degree, key=lambda x: x[1])
    print(f"Hub node is {hub[0]} with degree {hub[1]}")

    # Decide the destination node (node farthest from hub)
    dist = nx.single_source_shortest_path_length(G, hub[0])
    dest = max(dist.items(), key=lambda x: x[1])
    print(f"Destination node is {dest[0]} at distance {dest[1]} from hub")

    # Record hub and destination nodes
    for n in nodes:
        if n == hub[0]:
            nodes[n]['is_od'] = True
        elif n == dest[0]:
            nodes[n]['is_od'] = True
        else:
            nodes[n]['is_od'] = False
    with open(ds_root / "data" / "nodes.json", "w") as f:
        json.dump(nodes, f, indent=4)

    # Build system function
    ## Connectivity of one origin-destination pair
    sys_func_conn = lambda comps_st: eval_1od_connectivity(comps_st, G, hub[0], dest[0])

    ## Global connectivity
    def sys_func_global_conn_long(comps_st):
        _, k, _ = eval_global_conn_k(comps_st, G)
        if k >= target_g_conn:
            sys_st = 1
        else:
            sys_st = 0
        return k, sys_st, None
    sys_func_global_conn = lambda comps_st: sys_func_global_conn_long(comps_st)

    ## System functions for BRC (it accepts only 's'/'f' states for system event)
    def brc_wrapper(sys_func, func_option: str = "conn"):
        """Convert (k, sys_st (0/1), extra) -> (k, 's'/'f', None)."""
        assert func_option in ["conn", "global_conn"], f"Invalid func_option: {func_option}"
        
        if func_option == "conn":
            def f(comps_st):
                k, sys_st, info = sys_func(comps_st)
                if sys_st == 1: 
                    # survival case -> return minimum survival component state
                    # BRC's efficiency relies on the availability of minimum survival component state
                    min_comps_st = {comp: 1 for comp in info['path_edge_ids']}
                else:
                    min_comps_st = None
                return k, ('s' if sys_st == 1 else 'f'), min_comps_st
            return f
        else:
            def f(comps_st):
                k, sys_st, _ = sys_func(comps_st)
                return k, ('s' if sys_st == 1 else 'f'), None
            return f

    # System functions for RSR (remove the minimum comps state, which provides little benefit in RSR)
    def rsr_wrapper(sys_func):
        """Convert (k, sys_st (0/1), extra) -> (k, sys_st (0/1), None)."""
        def f(comps_st):
            k, sys_st, _ = sys_func(comps_st)
            return k, sys_st, None
        return f
    
    # Return random graph data
    rg_data = {
        "nodes": nodes,
        "edges": edges,
        "probs": probs,
        "hub": hub[0],
        "dest": dest[0],
        "sys_func_conn_brc": brc_wrapper(sys_func_conn),
        "sys_func_global_conn_brc": brc_wrapper(sys_func_global_conn),
        "sys_func_conn_rsr": rsr_wrapper(sys_func_conn),
        "sys_func_global_conn_rsr": rsr_wrapper(sys_func_global_conn),
        "graph": G
    }

    return ds_root, rg_data

if __name__ == "__main__":

    repo_root = Path(__file__).resolve().parents[1]
    out_base = repo_root / "results"
    brc_rss_max_gb = 20.0  # Max RSS for BRC in GB

    # Generate a random network 
    # smaller example
    generator = "rg"
    name = "rg1" # Random Geometric Graph
    gen_params1 = {"n_nodes": 60, "radius": 0.25, "p_fail": 0.05}
    target_g_conn1 = 1
    ds_root1, rg_data1 = generate_random_network_data(name, out_base=out_base, 
                                                    generator=generator,
                                                    generator_params=gen_params1, 
                                                    target_g_conn=target_g_conn1,
                                                    seed=7,
                                                    find_connected_graph=False)

    # Larger example
    """name = "rg2" # Random Geometric Graph
    gen_params2 = {"n_nodes": 120, "radius": 0.12, "p_fail": 0.05}
    target_g_conn2 = 1
    ds_root2, rg_data2 = generate_random_network_data(name, out_base=out_base,
                                                    generator=generator,
                                                    generator_params=gen_params2, 
                                                    target_g_conn=target_g_conn2,
                                                    seed=7)"""
    """
    # example in between (1)
    name = "rg3" # Random Geometric Graph
    gen_params3 = {"n_nodes": 80, "radius": 0.20, "p_fail": 0.05}
    target_g_conn3 = 1
    ds_root3, rg_data3 = generate_random_network_data(name, out_base=out_base,
                                                    generator=generator,
                                                    generator_params=gen_params3, 
                                                    target_g_conn=target_g_conn3,
                                                    seed=7)
    
    # example in between (2)
    name = "rg4" # Random Geometric Graph
    gen_params4 = {"n_nodes": 100, "radius": 0.16, "p_fail": 0.05}
    target_g_conn4 = 1
    ds_root4, rg_data4 = generate_random_network_data(name, out_base=out_base,
                                                    generator=generator,
                                                    generator_params=gen_params4, 
                                                    target_g_conn=target_g_conn4,
                                                    min_g_conn=2,
                                                    seed=7)
    """

    # Run BRC algorithm on the generated data
    edges1 = rg_data1['edges']
    probs1 = rg_data1['probs']
    
    #"""
    probs_brc1 = {e: {0: probs1[e]['0']['p'], 1: probs1[e]['1']['p']} for e in edges1}
    brs1, refs1, sys_res1, monitor1 = brc.run(probs_brc1, rg_data1['sys_func_conn_brc'], 
                                               max_rules=np.inf,
                                               pf_bnd_wr = 1e-5,
                                               max_memory_gb=brc_rss_max_gb)
    brc_path1 = Path(ds_root1 / "brc")
    brc_path1_rel = brc_path1.relative_to(Path.cwd())
    brc.save_brc_data(refs1, brs1, sys_res1, monitor1, output_folder = str(brc_path1_rel), fname_suffix='conn')
    
    del brs1, refs1, sys_res1, monitor1 # to not interfere with memory measurement of the next runs
    gc.collect()
    #"""

    
    """edges2 = rg_data2['edges']
    probs2 = rg_data2['probs']"""
    """probs_brc2 = {e: {0: probs2[e]['0']['p'], 1: probs2[e]['1']['p']} for e in edges2}
    
    brs2, refs2, sys_res2, monitor2 = brc.run(probs_brc2, rg_data2['sys_func_conn_brc'], 
                                               max_rules=np.inf,
                                               max_memory_gb=brc_rss_max_gb)
    brc_path2 = Path(ds_root2 / "brc")
    brc_path2_rel = brc_path2.relative_to(Path.cwd())
    brc.save_brc_data(refs2, brs2, sys_res2, monitor2, output_folder = str(brc_path2_rel), fname_suffix='conn')

    del brs2, refs2, sys_res2, monitor2 # to not interfere with memory measurement of the next runs
    gc.collect()
    """


    # Run RSR
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ## First example - 1OD connectivity
    row_names = list(edges1.keys()) 
    n_state = 2  # binary states: 0, 1
    probs = [[rg_data1['probs'][n]['0']['p'], rg_data1['probs'][n]['1']['p']] for n in row_names]
    probs = torch.tensor(probs, dtype=torch.float32, device=device)

    """
    _ = rsr.run_ref_extraction_by_mcs(
        # Problem-specific callables / data
        sfun=rg_data1['sys_func_conn_rsr'],
        probs=probs,
        row_names=row_names,
        n_state=n_state,
        sys_upper_st=1,
        unk_prob_thres = 1e-5,
        unk_prob_opt = 'abs',
        output_dir=ds_root1 / "rsr_conn",
    )
    """
    
    ## First example - Global connectivity
    """_ = rsr.run_ref_extraction_by_mcs( # to not interfere with memory measurement of the next runs
        # Problem-specific callables / data
        sfun=rg_data1['sys_func_global_conn_rsr'],
        probs=probs,
        row_names=row_names,
        n_state=n_state,
        sys_upper_st=1,
        unk_prob_thres = 1e-5,
        unk_prob_opt = 'abs',
        output_dir=ds_root1 / "rsr_global_conn",
    )"""

    ## First example - 1OD connectivity solved by RSR without boundary reference search
    """_ = rsr.run_ref_extraction_by_mcs(
        # Problem-specific callables / data
        sfun=rg_data1['sys_func_conn_rsr'],
        probs=probs,
        row_names=row_names,
        n_state=n_state,
        sys_upper_st=1,
        unk_prob_thres = 1e-5,
        unk_prob_opt = 'abs',
        output_dir=ds_root1 / "rsr_conn_no_min_rsr",
        min_ref_search = False
    )"""

    ## First example - Global connectivity solved by RSR without boundary reference search
    """_ = rsr.run_ref_extraction_by_mcs( # to not interfere with memory measurement of the next runs
        # Problem-specific callables / data
        sfun=rg_data1['sys_func_global_conn_rsr'],
        probs=probs,
        row_names=row_names,
        n_state=n_state,
        sys_upper_st=1,
        unk_prob_thres = 1e-5,
        unk_prob_opt = 'abs',
        output_dir=ds_root1 / "rsr_global_conn_no_min_rsr",
        min_ref_search = False
    )"""

    ## Second example - 1OD connectivity
    """row_names = list(edges2.keys())
    n_state = 2  # binary states: 0, 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    probs = [[rg_data2['probs'][n]['0']['p'], rg_data2['probs'][n]['1']['p']] for n in row_names]
    probs = torch.tensor(probs, dtype=torch.float32, device=device)"""
    """_ = rsr.run_ref_extraction_by_mcs( # to not interfere with memory measurement of the next run
        # Problem-specific callables / data
        sfun=rg_data2['sys_func_conn_rsr'],
        probs=probs,
        row_names=row_names,
        n_state=n_state,
        sys_upper_st=1,
        unk_prob_thres = 1e-5,
        unk_prob_opt = 'abs',
        output_dir=ds_root2 / "rsr_conn",
    )

    ## Second example - Global connectivity
    _ = rsr.run_ref_extraction_by_mcs( # to not use up residence set size
        # Problem-specific callables / data
        sfun=rg_data2['sys_func_global_conn_rsr'],
        probs=probs,
        row_names=row_names,
        n_state=n_state,
        sys_upper_st=1,
        unk_prob_thres = 1e-5,
        unk_prob_opt = 'abs',
        output_dir=ds_root2 / "rsr_global_conn",
    )"""

    ## Second example - 1OD connectivity solved by RSR without boundary reference search
    """_ = rsr.run_ref_extraction_by_mcs( # to not interfere with memory measurement of the next run
        # Problem-specific callables / data
        sfun=rg_data2['sys_func_conn_rsr'],
        probs=probs,
        row_names=row_names,
        n_state=n_state,
        sys_upper_st=1,
        unk_prob_thres = 1e-5,
        unk_prob_opt = 'abs',
        output_dir=ds_root2 / "rsr_conn_no_min_rsr",
        min_ref_search = False
    )"""

    ## Second example - Global connectivity solved by RSR without boundary reference search
    """_ = rsr.run_ref_extraction_by_mcs( # to not use up residence set size
        # Problem-specific callables / data
        sfun=rg_data2['sys_func_global_conn_rsr'],
        probs=probs,
        row_names=row_names,
        n_state=n_state,
        sys_upper_st=1,
        unk_prob_thres = 1e-5,
        unk_prob_opt = 'abs',
        output_dir=ds_root2 / "rsr_global_no_min_rsr",
        min_ref_search = False
    )"""

