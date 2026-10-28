"""
Report, for an RSR run on the Hebei network (Liu et al. 2021), the round at which
the coverage criterion reaches each tolerance, together with a 99% credible
interval on the reliability that includes the Monte Carlo sampling error.

RSR gives, per round, Monte Carlo estimates from ``n = n_sample_actual`` samples of
    u   = P(sample covered by an upper (survival) reference)   -> lower bound on R
    unk = P(sample covered by no reference)
so the reliability R = P(max_flow >= d) satisfies  u <= R <= u + unk.

All bounds use the Bayesian post-processing of Betz et al. (2022): with k of n
samples in a category and a uniform prior, the posterior is Beta(k+1, n-k+1).

  Coverage criterion (Sec. 3.4.2, Eqs. 31-33):
      unk_ucb = 0.99-quantile of the posterior of unk          <= tol
  Credible interval on R (each side at 99.5%, joint coverage >= 99%):
      R_lo = 0.005-quantile of the posterior of u
      R_hi = 0.995-quantile of the posterior of (u + unk) = 1 - p^L
  The width R_hi - R_lo is dominated by the sampling error in u and does not
  shrink with more reference states; it shrinks only with more samples.

For each tolerance, the reported round is the first from which the coverage
criterion holds for every later round ("sustained").

``metrics.json`` is appended to across runs; runs are split where the round
counter restarts, and the last run is used unless ``--run`` is given.

Usage:
    python eval_bounds_ci.py                               # results/case2_d3, results/case2_d4
    python eval_bounds_ci.py ../results/case2_d3 --tol 1e-4
    python eval_bounds_ci.py ../results/case2_d3 --run 0   # first run in the file
"""

import csv
import json
import math
import argparse
from pathlib import Path

from scipy.stats import beta

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE.parent / "results"

Q_COVERAGE = 0.99    # one-sided level of the coverage criterion
Q_SIDE = 0.995       # one-sided level of each side of the interval on R
TABLE_TOLS = [10.0 ** -k for k in range(1, 9)]   # 1e-1 ... 1e-8


def parse_args():
    p = argparse.ArgumentParser(description="Coverage criterion and 99% credible interval on RSR metrics")
    p.add_argument("result_dirs", nargs="*",
                   default=[str(RESULTS_DIR / "case2_d3"), str(RESULTS_DIR / "case2_d4")],
                   help="RSR output directories holding metrics.json "
                        "(default: ../results/case2_d3 ../results/case2_d4)")
    p.add_argument("--tol", type=float, default=1e-4,
                   help="Tolerance for the detailed printout (default: 1e-4)")
    p.add_argument("--run", type=int, default=-1,
                   help="Which run in metrics.json to use, by index (default: -1 = last)")
    p.add_argument("--no-csv", action="store_true",
                   help="Do not write the per-round CSV and the tables")
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


def beta_upper(k, n, q):
    """q-quantile of the Beta(k+1, n-k+1) posterior (uniform prior, Betz et al. 2022)."""
    return beta.ppf(q, k + 1, n - k + 1)


def beta_lower(k, n, q):
    """(1-q)-quantile of the same posterior, i.e. a one-sided lower bound at level q."""
    return beta.ppf(1 - q, k + 1, n - k + 1)


def to_count(p, n):
    """Sample count recovered from a stored proportion."""
    return min(max(round(p * n), 0), n)


def compute_rows(run):
    rows = []
    t_cum = 0.0
    n_sfun_cum = 0
    for e in run:
        t_cum += e["time_sec"]
        n_sfun_cum += e.get("n_sfun_upper", 0) + e.get("n_sfun_lower", 0)
        n = e["n_sample_actual"]
        u, unk = e["p_upper"], e["p_unknown"]

        k_unk = to_count(unk, n)
        k_u = to_count(u, n)
        k_notlow = to_count(u + unk, n)          # not classified as failure

        unk_ucb = beta_upper(k_unk, n, Q_COVERAGE)
        r_lo = beta_lower(k_u, n, Q_SIDE)        # lower bound on p^up
        r_hi = beta_upper(k_notlow, n, Q_SIDE)   # upper bound on p^up + p^u = 1 - p^L

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
            "unk_ucb": unk_ucb,
            "R_lo": r_lo,
            "R_hi": r_hi,
            "R_width": r_hi - r_lo,
        })
    return rows


def first_met(rows, key, tol):
    """(first index where rows[key] <= tol, first index from which it always holds)."""
    first = next((i for i, r in enumerate(rows) if r[key] <= tol), None)
    sustained = None
    for i in range(len(rows) - 1, -1, -1):
        if rows[i][key] <= tol:
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
          f"refs low/up = {r['n_refs_lower']}/{r['n_refs_upper']}  "
          f"sfun calls = {r['n_sfun_cum']}  (n = {r['n_sample']:,})")
    print(f"               u = {r['u_hat']:.6f}  unk = {r['unk_hat']:.3e}  "
          f"unk_ucb = {r['unk_ucb']:.3e}  "
          f"99% CI on R = [{r['R_lo']:.6f}, {r['R_hi']:.6f}] (width {r['R_width']:.2e})")


def tol_exp(tol):
    return round(-math.log10(tol))


def tol_table(rows, title):
    """Markdown table: for each tolerance, the sustained round and the 99% CI on R."""
    lines = [
        f"**{title}**",
        "",
        "| unk_ucb <= | Round | Time (s) | Refs (lower) | Refs (upper) | Refs (total) "
        "| Sfun calls | u_hat | unk_ucb | 99% CI lower | 99% CI upper | CI width | n |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for tol in TABLE_TOLS:
        _, idx = first_met(rows, "unk_ucb", tol)
        if idx is None:
            lines.append(f"| 1e-{tol_exp(tol)} | not reached |" + " |" * 11)
            continue
        r = rows[idx]
        lines.append(
            f"| 1e-{tol_exp(tol)} | {r['round']} | {r['t_cum_sec']:.1f} "
            f"| {r['n_refs_lower']} | {r['n_refs_upper']} "
            f"| {r['n_refs_lower'] + r['n_refs_upper']} | {r['n_sfun_cum']} "
            f"| {r['u_hat']:.6f} | {r['unk_ucb']:.2e} "
            f"| {r['R_lo']:.6f} | {r['R_hi']:.6f} | {r['R_width']:.2e} | {r['n_sample']:,} |")
    return "\n".join(lines)


def tex_int(x):
    """Integer with LaTeX thousands separators, e.g. 1375 -> 1{,}375."""
    return f"{x:,}".replace(",", "{,}")


def latex_rows(rows, max_exp=5):
    """Rows for tab:hebei (coverage level, reference counts, 99% CI on R)."""
    lines = []
    for tol in TABLE_TOLS[:max_exp]:
        _, idx = first_met(rows, "unk_ucb", tol)
        if idx is None:
            lines.append(f"        & $\\le 10^{{-{tol_exp(tol)}}}$ & not reached & & \\\\")
            continue
        r = rows[idx]
        lo, up = r["n_refs_lower"], r["n_refs_upper"]
        lines.append(
            f"        & $\\le 10^{{-{tol_exp(tol)}}}$ "
            f"& {tex_int(lo + up)} {{\\small ({tex_int(lo)} / {tex_int(up)})}} "
            f"& $[{r['R_lo']:.6f},\\ {r['R_hi']:.6f}]$ & \\\\")
    return "\n".join(lines)


def main():
    args = parse_args()

    for d in args.result_dirs:
        d = Path(d)
        metrics_path = d / "metrics.json"
        print("\n" + "=" * 70)
        print(f"{d}")
        print("=" * 70)
        if not metrics_path.exists():
            print("  metrics.json not found; skipping")
            continue

        runs = load_runs(metrics_path)
        run = runs[args.run]
        run_idx = args.run % len(runs)
        print(f"  {len(runs)} run(s) in metrics.json; using run {run_idx} "
              f"({len(run)} rounds, last round {run[-1]['round']})")

        rows = compute_rows(run)
        last = rows[-1]
        print(f"  Final: u = {last['u_hat']:.6f}, unk = {last['unk_hat']:.3e}, "
              f"99% CI on R = [{last['R_lo']:.6f}, {last['R_hi']:.6f}] "
              f"(width {last['R_width']:.2e}), t = {last['t_cum_sec']:.1f} s")

        print(f"\n  Coverage criterion: 0.99-quantile of the posterior of unk <= {args.tol:.0e}")
        first, sustained = first_met(rows, "unk_ucb", args.tol)
        describe(rows, first, "first")
        describe(rows, sustained, "sustained")

        print(f"\n  Width of the 99% credible interval on R <= {args.tol:.0e}")
        first, sustained = first_met(rows, "R_width", args.tol)
        describe(rows, first, "first")
        describe(rows, sustained, "sustained")
        if first is None:
            # normal approximation to the samples needed for the u-interval alone
            u = last["u_hat"]
            z = 2.5758293035489   # 0.995 quantile of the standard normal
            n_req = (2 * z) ** 2 * u * (1 - u) / args.tol ** 2
            print(f"    (min width: {min(r['R_width'] for r in rows):.3e}; the interval on u alone "
                  f"needs n >= {n_req:.2e} to fit in the tolerance)")

        table = tol_table(rows, d.name)
        tex = latex_rows(rows)
        print("\n" + table)
        print("\n  LaTeX rows for tab:hebei:\n" + tex)

        if not args.no_csv:
            suffix = "" if len(runs) == 1 else f"_run{run_idx}"
            (d / f"bounds_table{suffix}.md").write_text(table + "\n", encoding="utf-8")
            (d / f"bounds_table{suffix}.tex").write_text(tex + "\n", encoding="utf-8")
            out = d / f"bounds_ci{suffix}.csv"
            with open(out, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            print(f"\n  Tables and per-round values written to {d}")


if __name__ == "__main__":
    main()