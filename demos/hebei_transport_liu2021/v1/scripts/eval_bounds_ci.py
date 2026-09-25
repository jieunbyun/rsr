"""
Find when an RSR run reaches a 99%-confidence bound gap below a tolerance
(default 1e-4), to compare with the stopping rule of Liu et al. (2021).

RSR gives, per round, Monte Carlo estimates from ``n = n_sample_actual`` samples of
    u   = P(sample covered by an upper (survival) reference)   -> lower bound on R
    unk = P(sample covered by no reference)
so the reliability R = P(max_flow >= d) satisfies  u <= R <= u + unk.

With Wald standard errors  sigma_x = sqrt(x (1 - x) / n)  and z = Z_99 (two-sided):

  gap criterion (the MC counterpart of the paper's exact U - L < tol):
      unk_ucb = unk + z sigma_unk                       < tol
  R-interval criterion (bounds on R including MC error in u):
      R in [u - z sigma_u,  u + unk + z sigma_u + z sigma_unk]
      width = unk + 2 z sigma_u + z sigma_unk           < tol
    (This width is dominated by 2 z sigma_u, which needs n ~ 1e8-1e9 to go below 1e-4.)

For each criterion the script reports the first round where it holds, and the
first round from which it holds for every later round ("sustained"), with the
cumulative wall time, reference counts, and sfun calls at that round.

``metrics.json`` is appended to across runs; runs are split where the round
counter restarts, and the last run is used unless ``--run`` is given.

Usage:
    python eval_bounds_ci.py                               # results/case2_d3, results/case2_d4
    python eval_bounds_ci.py ../results/case2_d3 --tol 1e-4
    python eval_bounds_ci.py ../results/case2_d3 --run 0   # first run in the file
"""

import sys
import csv
import json
import math
import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE.parent / "results"

Z_99 = 2.5758293035489  # two-sided 99%
ALPHA = 0.01
TABLE_TOLS = [10.0 ** -k for k in range(1, 9)]   # 1e-1 ... 1e-8


def parse_args():
    p = argparse.ArgumentParser(description="99% CI bound-gap check on RSR metrics")
    p.add_argument("result_dirs", nargs="*",
                   default=[str(RESULTS_DIR / "case2_d3"), str(RESULTS_DIR / "case2_d4")],
                   help="RSR output directories holding metrics.json "
                        "(default: ../results/case2_d3 ../results/case2_d4)")
    p.add_argument("--tol", type=float, default=1e-4,
                   help="Tolerance on the interval width (default: 1e-4)")
    p.add_argument("--z", type=float, default=Z_99,
                   help=f"Critical value (default: Z_99 = {Z_99:.4f})")
    p.add_argument("--run", type=int, default=-1,
                   help="Which run in metrics.json to use, by index (default: -1 = last)")
    p.add_argument("--no-csv", action="store_true",
                   help="Do not write the per-round bounds_ci.csv")
    return p.parse_args()


def load_runs(metrics_path):
    """Read JSON-lines metrics and split into runs where `round` restarts."""
    runs, cur, prev_round = [], [], None
    with open(metrics_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            if prev_round is not None and e["round"] <= prev_round:
                runs.append(cur)
                cur = []
            cur.append(e)
            prev_round = e["round"]
    if cur:
        runs.append(cur)
    return runs


def se(p, n):
    return math.sqrt(max(p * (1.0 - p), 0.0) / n) if n > 0 else float("inf")


def compute_rows(run, z):
    rows = []
    t_cum = 0.0
    n_sfun_cum = 0
    for e in run:
        t_cum += e["time_sec"]
        n_sfun_cum += e.get("n_sfun_upper", 0) + e.get("n_sfun_lower", 0)
        n = e["n_sample_actual"]
        u, unk = e["p_upper"], e["p_unknown"]
        s_u, s_unk = se(u, n), se(unk, n)

        if round(unk * n) == 0:
            # Wald collapses to 0 with no unknowns; use the exact zero-count bound
            unk_ucb = -math.log(ALPHA / 2) / n
        else:
            unk_ucb = unk + z * s_unk
        r_lo = u - z * s_u
        r_hi = u + unk + z * s_u + z * s_unk
        rows.append({
            "round": e["round"],
            "t_cum_sec": t_cum,
            "n_sfun_cum": n_sfun_cum,
            "n_refs_upper": e["n_refs_upper"],
            "n_refs_lower": e["n_refs_lower"],
            "n_sample": n,
            "probs_updated": e.get("probs_updated", False),
            "u_hat": u,
            "unk_hat": unk,
            "sigma_u": s_u,
            "sigma_unk": s_unk,
            "unk_ucb": unk_ucb,
            "R_lo": r_lo,
            "R_hi": r_hi,
            "R_width": r_hi - r_lo,
        })
    return rows


def first_met(rows, key, tol):
    """(first index where rows[key] < tol, first index from which it always holds)."""
    first = next((i for i, r in enumerate(rows) if r[key] < tol), None)
    sustained = None
    for i in range(len(rows) - 1, -1, -1):
        if rows[i][key] < tol:
            sustained = i
        else:
            break
    return first, sustained


def describe(rows, idx, label):
    if idx is None:
        print(f"    {label:<10} not reached")
        return
    r = rows[idx]
    print(f"    {label:<10} round {r['round']:>6}  t = {r['t_cum_sec']:9.1f} s  "
          f"refs up/low = {r['n_refs_upper']}/{r['n_refs_lower']}  "
          f"sfun calls = {r['n_sfun_cum']}  (n = {r['n_sample']:,})")
    print(f"               u = {r['u_hat']:.6f}  unk = {r['unk_hat']:.3e}  "
          f"unk_ucb = {r['unk_ucb']:.3e}  R in [{r['R_lo']:.6f}, {r['R_hi']:.6f}]")


def tol_table(rows, title):
    """Markdown table: for each tolerance, where the gap criterion holds for good."""
    lines = [
        f"**{title}**",
        "",
        "| Gap tol. | Round | Time (s) | Refs (upper) | Refs (lower) | Refs (total) "
        "| Lower bound | Upper bound | +/- z*sigma_u | n |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for tol in TABLE_TOLS:
        _, idx = first_met(rows, "unk_ucb", tol)
        if idx is None:
            lines.append(f"| 1e-{round(-math.log10(tol))} | not reached | | | | | | | | |")
            continue
        r = rows[idx]
        lo, hi = r["u_hat"], r["u_hat"] + r["unk_ucb"]
        lines.append(
            f"| 1e-{round(-math.log10(tol))} | {r['round']} | {r['t_cum_sec']:.1f} "
            f"| {r['n_refs_upper']} | {r['n_refs_lower']} "
            f"| {r['n_refs_upper'] + r['n_refs_lower']} "
            f"| {lo:.6f} | {hi:.6f} | {Z_99 * r['sigma_u']:.1e} | {r['n_sample']:,} |")
    return "\n".join(lines)


def main():
    args = parse_args()

    print(f"Tolerance {args.tol:.0e}, z = {args.z:.4f}")
    for d in args.result_dirs:
        d = Path(d)
        metrics_path = d / "metrics.json"
        print("\n" + "=" * 70)
        print(f"{d}")
        print("=" * 70)
        if not metrics_path.exists():
            print(f"  metrics.json not found; skipping")
            continue

        runs = load_runs(metrics_path)
        run = runs[args.run]
        run_idx = args.run % len(runs)
        print(f"  {len(runs)} run(s) in metrics.json; using run {run_idx} "
              f"({len(run)} rounds, last round {run[-1]['round']})")

        rows = compute_rows(run, args.z)
        last = rows[-1]
        print(f"  Final: u = {last['u_hat']:.6f}, unk = {last['unk_hat']:.3e}, "
              f"R in [{last['R_lo']:.6f}, {last['R_hi']:.6f}], "
              f"t = {last['t_cum_sec']:.1f} s")

        print(f"\n  Gap criterion: unk + z*sigma_unk < {args.tol:.0e}")
        first, sustained = first_met(rows, "unk_ucb", args.tol)
        describe(rows, first, "first")
        describe(rows, sustained, "sustained")

        print(f"\n  R-interval criterion: (u + unk + z*sigma_u + z*sigma_unk) - (u - z*sigma_u) < {args.tol:.0e}")
        first, sustained = first_met(rows, "R_width", args.tol)
        describe(rows, first, "first")
        describe(rows, sustained, "sustained")
        if first is None:
            # samples needed for 2 z sigma_u alone to fit in the tolerance, at the final u
            u = last["u_hat"]
            n_req = (2 * args.z) ** 2 * u * (1 - u) / args.tol ** 2
            print(f"    (min width at the last round: {min(r['R_width'] for r in rows):.3e}; "
                  f"2*z*sigma_u < tol alone needs n >= {n_req:.2e} at u = {u:.4f})")

        table = tol_table(rows, d.name)
        print("\n" + table)

        if not args.no_csv:
            table_path = d / "bounds_table.md" if len(runs) == 1 else d / f"bounds_table_run{run_idx}.md"
            table_path.write_text(table + "\n", encoding="utf-8")
            print(f"\n  Table -> {table_path}")
            out = d / "bounds_ci.csv" if len(runs) == 1 else d / f"bounds_ci_run{run_idx}.csv"
            with open(out, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            print(f"\n  Per-round values -> {out}")


if __name__ == "__main__":
    main()
