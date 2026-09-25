"""
Run RSR reference extraction on the IEEE 118-bus DC-OPF blackout model.

This is the RSR counterpart to ``run_case118.py`` (which uses the ``tsum``
package).  Here we drive ``rsr.run_ref_extraction_by_mcs`` instead, using the
same pure-Python DC-OPF system function (``sfun_dcopt.make_dcopt_sfun``).

Components: 304 total (118 buses + 186 branches)
  State convention: 0 = failed/removed (worst), highest state = full capacity.
  - 54 generator buses: 4-state (0=removed ... 3=full cap)
  - 64 ordinary buses:  2-state (0=failed, 1=operational)
  - 186 branches:       2-state (0=failed, 1=operational)

System function (binary):
  sfun(comps_st) -> (blackout_size, sys_st, None)
    sys_st = 1  -> survival   (blackout_size < blackout_threshold)
    sys_st = 0  -> failure
  => sys_upper_st = 1   (upper/survival reference set: sys_st >= 1)

Usage:
    python run_rsr_case118.py                       # full run
    python run_rsr_case118.py --check-only          # sanity-check sfun only
    python run_rsr_case118.py --unk-prob-thres 1e-3
    python run_rsr_case118.py --n-sample 200000 --sample-batch-size 50000
"""

import sys
import os
import time
import argparse
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

import json
import torch

# --- paths -----------------------------------------------------------------
HERE = Path(__file__).resolve().parent          # .../ieee118/v1/scripts
DATA_DIR = HERE.parent / "data"                 # .../ieee118/v1/data
sys.path.insert(0, str(HERE))                   # so 'sfun_dcopt' / 'func_dcopt_py' import

from sfun_dcopt import make_dcopt_sfun

# rsr is normally pip-installed (editable); fall back to the repo checkout.
try:
    from rsr import rsr
except ImportError:
    REPO_ROOT = HERE.parents[3]                 # .../rsr  (repo root)
    sys.path.insert(0, str(REPO_ROOT))
    from rsr import rsr


# Scenario 1 from Chan et al. (2024): blackout_threshold = 13.8%
BLACKOUT_THRESHOLD = 13.8
ALPHA = 2.0
SYS_UPPER_ST = 1            # binary system: survival = sys_st >= 1


def parse_args():
    p = argparse.ArgumentParser(description="RSR on IEEE 118-bus DC-OPF")
    p.add_argument("--unk-prob-thres", type=float, default=1e-3,
                   help="Convergence threshold on unknown-region probability (default: 1e-3)")
    p.add_argument("--unk-prob-opt", choices=["abs", "rel"], default="abs",
                   help="Threshold interpretation (default: abs)")
    p.add_argument("--n-sample", type=int, default=1_000_000,
                   help="Samples per probability update (default: 1,000,000)")
    p.add_argument("--sample-batch-size", type=int, default=100_000,
                   help="Samples per batch (default: 100,000)")
    p.add_argument("--max-rounds", type=int, default=10_000,
                   help="Hard cap on rounds (default: 10,000)")
    p.add_argument("--output-dir", type=str, default=str(HERE / "rsr_results"),
                   help="Where references/metrics are written (default: ./rsr_results)")
    p.add_argument("--device", type=str, default="",
                   help="Torch device, e.g. 'cuda:0' or 'cpu' (default: auto)")
    p.add_argument("--check-only", action="store_true",
                   help="Only run sfun sanity checks, then exit")
    return p.parse_args()


def load_inputs(device):
    """Load probs.json and build the (padded) probability tensor + row names."""
    with open(DATA_DIR / "probs.json") as f:
        probs_dict = json.load(f)

    row_names = list(probs_dict.keys())
    n_state = max(len(v) for v in probs_dict.values())   # 4

    n_gen_bus = sum(1 for n in row_names if n.startswith("vbus") and len(probs_dict[n]) == 4)
    n_ord_bus = sum(1 for n in row_names if n.startswith("vbus") and len(probs_dict[n]) == 2)
    n_branch = sum(1 for n in row_names if n.startswith("br"))

    # Pad each component's distribution to length n_state (missing states -> 0.0)
    probs_list = []
    for name in row_names:
        p = probs_dict[name]
        probs_list.append([p[str(s)]["p"] if str(s) in p else 0.0 for s in range(n_state)])
    probs_tensor = torch.tensor(probs_list, dtype=torch.float32, device=device)

    info = dict(n_state=n_state, n_gen_bus=n_gen_bus,
                n_ord_bus=n_ord_bus, n_branch=n_branch, probs_dict=probs_dict)
    return probs_tensor, row_names, info


def main():
    args = parse_args()

    print("=" * 60)
    print("RSR on IEEE 118-bus DC-OPF (branches + buses)")
    print("=" * 60)

    device = torch.device(args.device) if args.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Inputs ----------------------------------------------------------
    probs_tensor, row_names, meta = load_inputs(device)
    n_state = meta["n_state"]
    print(f"\n  Components:  {len(row_names)} total")
    print(f"    Generator buses: {meta['n_gen_bus']} (4-state)")
    print(f"    Ordinary buses:  {meta['n_ord_bus']} (2-state)")
    print(f"    Branches:        {meta['n_branch']} (2-state)")
    print(f"  Max states:  {n_state}")
    print(f"  Threshold:   {BLACKOUT_THRESHOLD}% blackout (Scenario 1)")
    print(f"  sys_upper_st:{SYS_UPPER_ST}  (survival = sys_st >= {SYS_UPPER_ST})")
    print(f"  Device:      {device}")

    # 2. System function -------------------------------------------------
    print("\nInitialising DC-OPF system function...")
    case_path = str(DATA_DIR / "ieee118.m")
    sfun = make_dcopt_sfun(case_path=case_path,
                           blackout_threshold=BLACKOUT_THRESHOLD,
                           alpha=ALPHA)

    # Sanity checks: all best-state (max = operational) vs all worst-state (0 = failed)
    all_ok = {name: max(int(s) for s in meta['probs_dict'][name].keys())
              for name in row_names}                               # highest state = full capacity
    fval, sys_st, _ = sfun(all_ok)
    print(f"  All operational (max state): blackout={fval:.4f}%, sys_st={sys_st}  (expect sys_st=1)")

    all_fail = {name: 0 for name in row_names}                     # state 0 = failed/removed
    fval, sys_st, _ = sfun(all_fail)
    print(f"  All failed (state 0):        blackout={fval:.4f}%, sys_st={sys_st}  (expect sys_st=0)")

    if args.check_only:
        print("\n--check-only set; exiting before reference extraction.")
        return

    # 3. RSR reference extraction ---------------------------------------
    output_dir = args.output_dir
    print(f"\n  Output:      {output_dir}")
    print(f"  Samples:     {args.n_sample:,} per round (batch {args.sample_batch_size:,})")
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
        output_dir=output_dir,
        upper_json_name="refs_upper.json",
        lower_json_name="refs_lower.json",
    )
    elapsed = time.time() - t0

    metrics_log = result.get("metrics_log", [])
    last = metrics_log[-1] if metrics_log else {}
    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"  Survival (upper) refs: {last.get('n_refs_upper', '?')}")
    print(f"  Failure  (lower) refs: {last.get('n_refs_lower', '?')}")
    print(f"  Sys values found:      {result.get('sys_vals', [])}")
    print(f"  Results saved to: {output_dir}")
    print(f"    upper refs -> {result.get('refs_upper_path')}")
    print(f"    lower refs -> {result.get('refs_lower_path')}")

    # 4. Summary --------------------------------------------------------
    if last:
        print(f"\n--- Summary ---")
        print(f"  Rounds:      {len(metrics_log)}")
        print(f"  Unk prob:    {last.get('unk_prob', last.get('p_unknown', '?'))}")
        print(f"  Reference (Chan et al. Table 2): p_f ~ 1.0e-4")


if __name__ == "__main__":
    main()
