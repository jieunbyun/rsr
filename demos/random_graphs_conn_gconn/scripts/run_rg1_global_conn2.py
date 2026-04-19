"""Run rsr on rg1/v1 data with global connectivity target g_conn=2 and sys_upper_st=2."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch
from ndtools.io import load_json
from ndtools.graphs import build_graph
from ndtools.fun_binary_graph import eval_global_conn_k
from rsr import rsr

repo_root = Path(__file__).resolve().parents[1]
ds_root1 = repo_root / "results" / "rg1" / "v1"

# Load existing rg1/v1 data
data_dir = ds_root1 / "data"
nodes = load_json(data_dir / "nodes.json")
edges = load_json(data_dir / "edges.json")
probs = load_json(data_dir / "probs.json")

G = build_graph(nodes, edges, probs)
print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

# Build sys_func_global_conn with target_g_conn=2
target_g_conn = 2

def sys_func_global_conn_long(comps_st):
    _, k, _ = eval_global_conn_k(comps_st, G)
    sys_st = k
    return k, sys_st, None

def rsr_wrapper(sys_func):
    def f(comps_st):
        k, sys_st, _ = sys_func(comps_st)
        return k, sys_st, None
    return f

sys_func_global_conn_rsr = rsr_wrapper(sys_func_global_conn_long)

# Prepare probs tensor
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
row_names = list(edges.keys())
n_state = 2
probs_tensor = [[probs[n]['0']['p'], probs[n]['1']['p']] for n in row_names]
probs_tensor = torch.tensor(probs_tensor, dtype=torch.float32, device=device)

# Run RSR
rsr.run_ref_extraction_by_mcs(
    sfun=sys_func_global_conn_rsr,
    probs=probs_tensor,
    row_names=row_names,
    n_state=n_state,
    sys_upper_st=2,
    unk_prob_thres=1e-5,
    unk_prob_opt='abs',
    output_dir=ds_root1 / "rsr_global_conn",
    metrics_path = ds_root1 / "rsr_global_conn" / "metrics2.json",
)
