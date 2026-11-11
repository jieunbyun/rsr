"""Component posteriors from MCMC against the reference: one panel per origin and run.

Each point is one edge: x = P(e closed | F) from the weighted-IS reference, y = the same from
MCMC. Points on the dashed diagonal agree; each panel title gives the largest gap and how many
edges are off by > 0.05. A results file is {origin: result} (run_ema.py), {configuration:
{origin: result}} (run_bridge.py), or {"reference": ..., configuration: result} for several
systems observed together (run_pair.py, where F is the joint failure), or {case/start: result}
(run_sweep.py, here or in toy_ca_mcmc, whose reference is exact). Runs are rows, origins
columns; with a single origin the runs sit side by side. The figure is written beside the
results file.

    python demos/ema_mcmc/plot_posterior.py [--results results_bridge.json | results_pair.json |
                                             results_sweep.json | ../toy_ca_mcmc/results_sweep.json]
"""
import argparse
import json
import textwrap
from pathlib import Path

import torch  # noqa: F401  (in the ieee-rsr env, matplotlib's text rendering crashes unless torch's DLLs load first)
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
INK, MUTED, GRID, POINT, BG = "#1f2328", "#59636e", "#e9edf1", "#2a78d6", "#ffffff"

parser = argparse.ArgumentParser()
parser.add_argument("--results", default="results_ema_perbox10.json")
args = parser.parse_args()
res_path = (HERE / args.results).resolve()
results = json.loads(res_path.read_text())
first = next(iter(results.values()))
ref_name = "exact" if "closed_exact" in first else "weighted-IS"
if "closed_ref" in first or "closed_exact" in first:          # run_sweep.py: {case/start: result}
    runs = {}
    for key, r in results.items():
        case, _, start = key.partition("/")
        r = {**r, "closed_is": r.get("closed_ref", r.get("closed_exact")), "P_IS": r.get("P_ref", r.get("P_E")),
             "n_chain": r.get("n_chain", 1000), "n_refs": "+".join(map(str, r["n_refs"]))}
        runs.setdefault(f"{start} start" if start else "", {})[case] = r
elif "reference" in results:                                  # several systems observed together
    ref = results.pop("reference")
    joint = " + ".join(ref["systems"])
    runs = {cfg: {joint: r} for cfg, r in results.items()}
elif "closed_is" in next(iter(results.values())):
    runs = {Path(args.results).stem: results}
else:
    runs = results

origins = list(next(iter(runs.values())))
cells = [(run, o) for run in runs for o in origins]
n_row, n_col = (1, len(runs)) if len(origins) == 1 else (len(runs), len(origins))
wrapped = len(runs) == 1 and len(origins) > 4                   # one run: rows of four origins
if wrapped:
    n_row, n_col = -(-len(origins) // 4), 4
fig, axes = plt.subplots(n_row, n_col, figsize=(2.9 * n_col + 0.8, 3.0 * n_row + 0.7),
                         sharex=True, sharey=True, squeeze=False)
fig.patch.set_facecolor(BG)
for ax in axes.flat[len(cells):]:
    ax.set_visible(False)
for k, (run, o) in enumerate(cells):
    i, j = divmod(k, n_col)
    ax, r = axes[i, j], runs[run][o]
    ref, est = np.array(r["closed_is"]), np.array(r["closed_mcmc"])
    err = np.abs(est - ref)
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=MUTED, zorder=1)
    ax.scatter(ref, est, s=16, color=POINT, edgecolor=BG, linewidth=0.6, zorder=2)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_aspect("equal")
    ax.grid(color=GRID, lw=0.6)
    ax.tick_params(colors=MUTED, labelsize=7)
    for s in ax.spines.values():
        s.set_visible(False)
    if len(origins) == 1:
        head = textwrap.fill(run, 40) + "\n"
    else:
        head = f"{o}: P(F) = {r['P_IS']:.1e}, {r['n_refs']} ref. states\n" if i == 0 or wrapped else ""
    ax.set_title(f"{head}max error {err.max():.3f}; {int((err > 0.05).sum())}/{len(err)} edges off > 0.05",
                 fontsize=8, color=INK, loc="left")
    if i == n_row - 1:
        ax.set_xlabel(f"{ref_name} P(e closed | F)", fontsize=8, color=MUTED)
    if j == 0:
        ax.set_ylabel(("" if len(origins) == 1 or not run else f"{run}\n\n") + "MCMC P(e closed | F)",
                      fontsize=8, color=MUTED)
chains = {r["n_chain"] for res in runs.values() for r in res.values()}
n_chain = f"{min(chains):,}" if len(chains) == 1 else f"{min(chains):,}-{max(chains):,}"
what = (f"F = {origins[0]} all failed, P(F) = {r['P_IS']:.1e}; " if len(origins) == 1 else "")
fig.suptitle(f"Edge posteriors, MCMC against the {ref_name} {'answer' if ref_name == 'exact' else 'reference'}\n"
             f"({what}{res_path.parent.name}/{res_path.name}; {n_chain} chains)", x=0.01, ha="left",
             fontsize=10, color=INK)
fig.subplots_adjust(left=0.07, right=0.99, top=1 - (1.15 if len(origins) == 1 else 0.95) / fig.get_figheight(),
                    bottom=0.5 / fig.get_figheight(), wspace=0.12, hspace=0.35)
out = res_path.parent / f"posterior_{res_path.stem}.png"
fig.savefig(out, dpi=200, facecolor=BG)
print("wrote", out)
