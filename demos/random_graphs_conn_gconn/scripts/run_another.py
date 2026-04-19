import json
from pathlib import Path
import networkx as nx
import torch
import numpy as np
import gc
import os
import subprocess
import sys
import pdb
import typer

HOME = Path(__file__).parent
sys.path.append(str(HOME.joinpath('../../../../network-datasets/')))

from ndtools.network_generator import GenConfig, generate_and_save
from ndtools.io import load_json
from ndtools.graphs import build_graph
from ndtools.fun_binary_graph import eval_1od_connectivity, eval_global_conn_k

#sys.path.append(str(HOME.joinpath('../../../../mbnpy/')))
#from mbnpy import brc


app = typer.Typer()


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from rsr import rsr

try:
    from rsr.igraph_sfun import make_igraph_sfun_global_conn, make_igraph_sfun_conn
    HAS_IGRAPH = True
except ImportError:
    HAS_IGRAPH = False

repo_root = Path(__file__).resolve().parents[1]
out_base = repo_root / "results"


def generate_random_network_data(name: str = "rg",
                                 generator = "rg",
                                 generator_params={"n_nodes": 60, "radius": 0.25, "p_fail": 0.1},
                                 target_g_conn = 1,
                                 min_g_conn = 0, # Minimum global connectivity (k) required for the generated graph. Only used if find_connected_graph=True.
                                 out_base: Path = None,
                                 seed: int = 7,
                                 find_connected_graph: bool = True,
                                 use_igraph: bool = False) -> Path:
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
                    print(f"k: {k} vs target_g_conn: {target_g_conn}")
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
    if use_igraph:
        print("Using igraph-accelerated system functions")
        sys_func_conn = make_igraph_sfun_conn(G, hub[0], dest[0])
        sys_func_global_conn = make_igraph_sfun_global_conn(G, target_g_conn)
    else:
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
        "sys_func_global_conn_rsr": rsr_wrapper(sys_func_global_conn) if not use_igraph else sys_func_global_conn,
        "graph": G
    }

    return ds_root, rg_data


def _run_example(
    name: str,
    gen_params: dict,
    devices: str,
    n_workers: int,
    n_sample: int,
    sample_batch_size: int,
    max_search_loops: int,
    target_g_conn: int = 1,
    min_g_conn: int = 0,
    find_connected_graph: bool = True,
    run_conn: bool = True,
    run_global_conn: bool = True,
    global_conn_dir: str = "rsr_global_conn",
    save_every: int = 0,
    seed: int = 7,
    use_igraph: bool = False,
):
    """Common logic for all examples."""
    if use_igraph and not HAS_IGRAPH:
        raise RuntimeError("--use-igraph requires python-igraph: pip install python-igraph")
    generator = "rg"
    ds_root, rg_data = generate_random_network_data(
        name, out_base=out_base,
        generator=generator,
        generator_params=gen_params,
        target_g_conn=target_g_conn,
        min_g_conn=min_g_conn,
        seed=seed,
        find_connected_graph=find_connected_graph,
        use_igraph=use_igraph,
    )

    device_list = [d.strip() for d in devices.split(",") if d.strip()] if devices else []
    device = torch.device(device_list[0] if device_list else ('cuda' if torch.cuda.is_available() else 'cpu'))
    multi_devices = device_list if len(device_list) > 1 else None

    row_names = list(rg_data['edges'].keys())
    n_state = 2
    probs = [[rg_data['probs'][n]['0']['p'], rg_data['probs'][n]['1']['p']] for n in row_names]
    probs = torch.tensor(probs, dtype=torch.float32, device=device)

    common_kwargs = dict(
        probs=probs,
        row_names=row_names,
        n_state=n_state,
        sys_upper_st=1,
        unk_prob_thres=1e-5,
        unk_prob_opt='abs',
        n_sample=n_sample,
        sample_batch_size=sample_batch_size,
        max_search_loops=max_search_loops,
        n_workers=n_workers,
        devices=multi_devices,
    )

    if run_conn:
        rsr.run_ref_extraction_by_mcs(
            sfun=rg_data['sys_func_conn_rsr'],
            output_dir=ds_root / "rsr_conn",
            **common_kwargs,
        )

    if run_global_conn:
        extra = {}
        if save_every > 0:
            extra['save_every'] = save_every
        global_dir_name = global_conn_dir
        rsr.run_ref_extraction_by_mcs(
            sfun=rg_data['sys_func_global_conn_rsr'],
            output_dir=ds_root / global_dir_name,
            **common_kwargs,
            **extra,
        )


# -- Common CLI options shared by all example commands --
_common_opts = dict(
    n_workers=typer.Option(1, help="Number of CPU workers for parallel sfun + minimization"),
    devices=typer.Option("", help="Comma-separated GPU devices for multi-GPU sampling, e.g. 'cuda:0,cuda:1'. Empty = single device."),
    n_sample=typer.Option(10_000_000, help="Total number of samples for probability estimation"),
    sample_batch_size=typer.Option(100_000, help="Samples per GPU batch. Must fit in GPU VRAM."),
    max_search_loops=typer.Option(0, help="Max batches per round for searching unknowns. 0 = use n_sample/sample_batch_size. Set e.g. 100 to cap search and avoid long empty rounds."),
    use_igraph=typer.Option(False, help="Use igraph (C-based) instead of NetworkX for connectivity. 10-100x faster. Requires: pip install python-igraph"),
)


@app.command()
def example1(
    n_workers: int = _common_opts['n_workers'],
    devices: str = _common_opts['devices'],
    n_sample: int = _common_opts['n_sample'],
    sample_batch_size: int = _common_opts['sample_batch_size'],
    max_search_loops: int = _common_opts['max_search_loops'],
    use_igraph: bool = _common_opts['use_igraph'],
    output_str: str = typer.Option("", help="str for output folder"),
):
    _run_example(
        name="rg1",
        gen_params={"n_nodes": 60, "radius": 0.25, "p_fail": 0.05},
        find_connected_graph=False,
        run_conn=False,
        global_conn_dir="rsr_global" + output_str,
        devices=devices, n_workers=n_workers, n_sample=n_sample,
        sample_batch_size=sample_batch_size, max_search_loops=max_search_loops,
        use_igraph=use_igraph,
    )


@app.command()
def example2(
    n_workers: int = _common_opts['n_workers'],
    devices: str = _common_opts['devices'],
    n_sample: int = _common_opts['n_sample'],
    sample_batch_size: int = _common_opts['sample_batch_size'],
    max_search_loops: int = _common_opts['max_search_loops'],
    use_igraph: bool = _common_opts['use_igraph'],
    output_str: str = typer.Option("", help="str for output folder"),
):
    _run_example(
        name="rg2",
        gen_params={"n_nodes": 120, "radius": 0.12, "p_fail": 0.05},
        find_connected_graph=True,
        run_conn=True,
        global_conn_dir="rsr_global_conn" + output_str,
        devices=devices, n_workers=n_workers, n_sample=n_sample,
        sample_batch_size=sample_batch_size, max_search_loops=max_search_loops,
        use_igraph=use_igraph,
    )

@app.command()
def run_parallel(
    examples: str = typer.Argument("1,2,3,4", help="Comma-separated example numbers to run, e.g. '1,2,3,4'"),
    n_gpus: int = typer.Option(0, help="Number of GPUs available. 0 = auto-detect."),
    n_workers: int = typer.Option(1, help="Number of CPU workers per example for parallel sfun + minimization"),
    gpus_per_example: int = typer.Option(1, help="Number of GPUs per example for multi-GPU sampling. >1 enables multi-GPU within each example."),
    n_sample: int = typer.Option(10_000_000, help="Total number of samples for probability estimation"),
    sample_batch_size: int = typer.Option(100_000, help="Samples per GPU batch. Must fit in GPU VRAM."),
    max_search_loops: int = typer.Option(0, help="Max batches per round for searching unknowns. 0 = use n_sample/sample_batch_size. Set e.g. 100 to cap search and avoid long empty rounds."),
    use_igraph: bool = typer.Option(False, help="Use igraph (C-based) instead of NetworkX for connectivity. 10-100x faster."),
):
    """Run multiple examples in parallel, each pinned to one or more GPUs."""

    example_nums = [int(x.strip()) for x in examples.split(",")]

    if n_gpus <= 0:
        n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        print("No GPUs detected. Running all examples on CPU in parallel.")

    script = str(Path(__file__).resolve())
    procs = []

    for i, ex_num in enumerate(example_nums):
        cmd_name = f"example{ex_num}"
        env = os.environ.copy()
        if n_gpus > 0:
            # Assign gpus_per_example GPUs to each example (round-robin)
            assigned = []
            for g in range(gpus_per_example):
                assigned.append((i * gpus_per_example + g) % n_gpus)
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in assigned)
            # When CUDA_VISIBLE_DEVICES remaps, devices appear as cuda:0, cuda:1, ...
            device_str = ",".join(f"cuda:{j}" for j in range(len(assigned)))
            print(f"Launching {cmd_name} on physical GPU(s) {assigned}")
        else:
            env["CUDA_VISIBLE_DEVICES"] = ""
            device_str = ""
            print(f"Launching {cmd_name} on CPU")

        cmd = [sys.executable, script, cmd_name,
               "--n-workers", str(n_workers),
               "--n-sample", str(n_sample),
               "--sample-batch-size", str(sample_batch_size),
               "--max-search-loops", str(max_search_loops)]
        if use_igraph:
            cmd.append("--use-igraph")
        if device_str and gpus_per_example > 1:
            cmd += ["--devices", device_str]
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        procs.append((cmd_name, proc))

    # Wait for all to finish
    failed = []
    for cmd_name, proc in procs:
        rc = proc.wait()
        if rc != 0:
            failed.append((cmd_name, rc))
            print(f"{cmd_name} failed with return code {rc}")
        else:
            print(f"{cmd_name} completed successfully")

    if failed:
        print(f"\n{len(failed)}/{len(procs)} examples failed.")
        sys.exit(1)
    else:
        print(f"\nAll {len(procs)} examples completed successfully.")


if __name__ == "__main__":

    app()



