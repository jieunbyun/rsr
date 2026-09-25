"""
Run RSR reference extraction on the Hebei transportation multistate network
(Liu et al. 2021), Case 2 of Table 4, for demands d = 3 and d = 4.

Case 2 is the distribution stored in ``data/probs.json``:
    p = (0.05, 0.70, 0.25) for every arc (i.i.d.), states 0/1/2 carry capacity 0/1/2.

System function (binary, per demand d):
  sfun(comps_st) -> (max_flow, sys_st, min_comps_st)
    max_flow = maximum n1 -> n22 flow with arc capacities = arc states
    sys_st   = 1 if max_flow >= d (survival), else 0 (failure)
  => sys_upper_st = 1

On survival, sfun also returns a flow of value exactly d as the seed reference
(arcs carrying flow f -> ('>=', f)), which rsr then minimises further.
Arcs missing from ``comps_st`` are treated as state 0, since rsr's minimiser
may call sfun on partial dicts.

Reference results (Liu et al. 2021): 3596 d-MPs for d=3, 5665 for d=4.

Usage:
    python run_rsr_case2.py                      # d = 3 and 4
    python run_rsr_case2.py --demands 3          # d = 3 only
    python run_rsr_case2.py --check-only         # sanity-check sfun only
    python run_rsr_case2.py --unk-prob-thres 1e-4 --n-sample 2000000
"""

import sys
import os
import time
import json
import argparse
from collections import deque
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

import torch

# --- paths -----------------------------------------------------------------
HERE = Path(__file__).resolve().parent          # .../hebei_transport_liu2021/v1/scripts
DATA_DIR = HERE.parent / "data"                 # .../hebei_transport_liu2021/v1/data

# rsr is normally pip-installed (editable); fall back to the repo checkout.
try:
    from rsr import rsr
except ImportError:
    REPO_ROOT = HERE.parents[3]                 # .../rsr  (repo root)
    sys.path.insert(0, str(REPO_ROOT))
    from rsr import rsr


PROBS_FILE = "probs.json"   # Case 2 of Table 4: p = (0.05, 0.70, 0.25)
SOURCE = "n1"
SINK = "n22"
SYS_UPPER_ST = 1            # binary system: survival = sys_st >= 1


def parse_args():
    p = argparse.ArgumentParser(description="RSR on Hebei transport network, Case 2")
    p.add_argument("--demands", type=int, nargs="+", default=[3, 4],
                   help="System demands d to run (default: 3 4)")
    p.add_argument("--unk-prob-thres", type=float, default=1e-6,
                   help="Convergence threshold on unknown-region probability (default: 1e-6)")
    p.add_argument("--unk-prob-opt", choices=["abs", "rel"], default="abs",
                   help="Threshold interpretation (default: abs)")
    p.add_argument("--n-sample", type=int, default=1_000_000,
                   help="Samples per probability update (default: 1,000,000)")
    p.add_argument("--sample-batch-size", type=int, default=100_000,
                   help="Samples per batch (default: 100,000)")
    p.add_argument("--prob-update-every", type=int, default=1,
                   help="Re-estimate probabilities with the full n_sample every this many "
                        "rounds (default: 1, i.e. every round)")
    p.add_argument("--max-rounds", type=int, default=10_000,
                   help="Hard cap on rounds (default: 10,000)")
    p.add_argument("--output-dir", type=str, default=str(HERE.parent / "results"),
                   help="Parent directory; each demand writes to <dir>/case2_d<d> "
                        "(default: ../results)")
    p.add_argument("--device", type=str, default="",
                   help="Torch device, e.g. 'cuda:0' or 'cpu' (default: auto)")
    p.add_argument("--check-only", action="store_true",
                   help="Only run sfun sanity checks, then exit")
    return p.parse_args()


def load_inputs(device):
    """Load edges.json / probs.json and build the probability tensor."""
    edges = json.loads((DATA_DIR / "edges.json").read_text(encoding="utf-8"))
    probs_dict = json.loads((DATA_DIR / PROBS_FILE).read_text(encoding="utf-8"))

    row_names = list(edges.keys())
    n_state = max(e["n_states"] for e in edges.values())   # 3

    probs_list = [[probs_dict[n][str(s)]["p"] for s in range(n_state)] for n in row_names]
    probs_tensor = torch.tensor(probs_list, dtype=torch.float32, device=device)

    return edges, probs_tensor, row_names, n_state


def max_flow(arcs, comps_st, source, sink, limit=None):
    """Edmonds-Karp max flow on directed arcs with capacity = component state.

    Args:
        arcs: dict eid -> (from, to).
        comps_st: dict eid -> state (= capacity); missing arcs have capacity 0.
        limit: stop once this much flow has been pushed (None = no limit).

    Returns:
        (flow_value, arc_flows) where arc_flows maps eid -> flow (> 0 only).
    """
    # residual[u][v] -> list of (eid, sign); keep per-arc flow to handle parallel arcs
    adj = {}
    for eid, (u, v) in arcs.items():
        adj.setdefault(u, []).append((eid, v, +1))
        adj.setdefault(v, []).append((eid, u, -1))
    flow = {eid: 0 for eid in arcs}

    def residual(eid, sign):
        return comps_st.get(eid, 0) - flow[eid] if sign > 0 else flow[eid]

    total = 0
    while limit is None or total < limit:
        parent = {source: None}
        q = deque([source])
        while q and sink not in parent:
            u = q.popleft()
            for eid, v, sign in adj.get(u, ()):
                if v not in parent and residual(eid, sign) > 0:
                    parent[v] = (u, eid, sign)
                    q.append(v)
        if sink not in parent:
            break

        # bottleneck along the augmenting path
        path = []
        v = sink
        while parent[v] is not None:
            u, eid, sign = parent[v]
            path.append((eid, sign))
            v = u
        aug = min(residual(eid, sign) for eid, sign in path)
        if limit is not None:
            aug = min(aug, limit - total)
        for eid, sign in path:
            flow[eid] += sign * aug
        total += aug

    return total, {eid: f for eid, f in flow.items() if f > 0}


def make_sfun(arcs, demand):
    def sfun(comps_st):
        # Full max flow value (fval), then a flow of exactly `demand` as the seed ref
        fval, _ = max_flow(arcs, comps_st, SOURCE, SINK)
        if fval >= demand:
            _, arc_flows = max_flow(arcs, comps_st, SOURCE, SINK, limit=demand)
            min_comps_st = {eid: ('>=', f) for eid, f in arc_flows.items()}
            min_comps_st['sys'] = ('>=', 1)
            return fval, 1, min_comps_st
        return fval, 0, None
    return sfun


def run_one(demand, args, arcs, probs_tensor, row_names, n_state):
    print("\n" + "=" * 60)
    print(f"Demand d = {demand}")
    print("=" * 60)

    sfun = make_sfun(arcs, demand)

    # Sanity checks: all arcs at max capacity vs all failed
    all_ok = {n: n_state - 1 for n in row_names}
    fval, sys_st, _ = sfun(all_ok)
    print(f"  All arcs at capacity {n_state-1}: max_flow={fval}, sys_st={sys_st}  (expect 6, 1)")
    all_fail = {n: 0 for n in row_names}
    fval, sys_st, _ = sfun(all_fail)
    print(f"  All arcs failed (state 0): max_flow={fval}, sys_st={sys_st}  (expect 0, 0)")

    if args.check_only:
        return

    output_dir = str(Path(args.output_dir) / f"case2_d{demand}")
    print(f"\n  Output:      {output_dir}")
    print(f"  Samples:     {args.n_sample:,} per round (batch {args.sample_batch_size:,}), "
          f"full estimate every {args.prob_update_every} round(s)")
    print(f"  Convergence: unk_prob ({args.unk_prob_opt}) < {args.unk_prob_thres:.0e}")
    print("\nStarting reference extraction...\n", flush=True)

    t0 = time.time()
    result = rsr.run_ref_extraction_by_mcs(
        sfun=sfun,
        probs=probs_tensor,
        row_names=row_names,
        n_state=n_state,
        sys_upper_st=SYS_UPPER_ST,
        unk_prob_thres=args.unk_prob_thres,
        unk_prob_opt=args.unk_prob_opt,
        max_rounds=args.max_rounds,
        n_sample=args.n_sample,
        sample_batch_size=args.sample_batch_size,
        prob_update_every=args.prob_update_every,
        output_dir=output_dir,
    )
    elapsed = time.time() - t0

    metrics_log = result.get("metrics_log", [])
    last = metrics_log[-1] if metrics_log else {}
    print(f"\nd = {demand} completed in {elapsed:.1f}s")
    print(f"  Rounds:                {len(metrics_log)}")
    print(f"  Survival (upper) refs: {last.get('n_refs_upper', '?')}")
    print(f"  Failure  (lower) refs: {last.get('n_refs_lower', '?')}")
    print(f"  Unk prob:              {last.get('unk_prob', last.get('p_unknown', '?'))}")
    print(f"  Reference (Liu et al. 2021): {3596 if demand == 3 else 5665 if demand == 4 else '?'} d-MPs")
    print(f"  Results saved to: {output_dir}")


def main():
    args = parse_args()

    print("=" * 60)
    print("RSR on Hebei transport network (Liu et al. 2021), Case 2")
    print("=" * 60)

    device = torch.device(args.device) if args.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")

    edges, probs_tensor, row_names, n_state = load_inputs(device)
    arcs = {eid: (e["from"], e["to"]) for eid, e in edges.items()}
    print(f"  Arcs:        {len(row_names)} ({n_state}-state)")
    print(f"  Probs:       {PROBS_FILE}  p = {probs_tensor[0].tolist()}")
    print(f"  Source/sink: {SOURCE} -> {SINK}")
    print(f"  Demands:     {args.demands}")
    print(f"  Device:      {device}")

    for demand in args.demands:
        run_one(demand, args, arcs, probs_tensor, row_names, n_state)


if __name__ == "__main__":
    main()
