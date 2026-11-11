"""Trace plots for EMA origins, one figure per origin (traces_ema_<origin>.png).

Top: the failure reference state each traced chain proposed from (gamma), as a step line over
the steps; the y axis spans all R reference states. Bottom: the 25 edges most often closed given
failure, one strip per chain (chain 1 on top), dark = closed, light = open.

    python demos/ema_mcmc/plot_ema.py [--traces traces_ema_perbox10.pt] [n61 n32 ...]
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "toy_ca_mcmc"))
from plot_traces import GAP, INK, MUTED, strips  # noqa: E402

CHAIN_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]                              # categorical slots 1-3
N_EDGE = 25

parser = argparse.ArgumentParser()
parser.add_argument("--traces", default="traces_ema_perbox10.pt")
parser.add_argument("origins", nargs="*")
args = parser.parse_args()
data = torch.load(HERE / args.traces)
edges = data["edges"]

for o in args.origins or list(data["traces"]):
    tr, r = data["traces"][o], data["results"][o]
    gamma, x = tr["gamma"].numpy(), tr["x"]                                    # (T, C), (T, C, n)
    T, C = gamma.shape
    keep = tr["closed"].argsort(descending=True)[:N_EDGE].sort().values
    rows = N_EDGE * (C + 1) - 1
    fig, (ax_g, ax_x) = plt.subplots(2, 1, figsize=(11, 2.2 + 0.085 * rows + 1.6), sharex=True,
                                     gridspec_kw={"height_ratios": [2.2, 0.085 * rows], "hspace": 0.12})
    fig.patch.set_facecolor(GAP)

    for k in range(C):
        ax_g.plot(range(T), gamma[:, k], drawstyle="steps-post", lw=1.2, color=CHAIN_COLORS[k],
                  label=f"chain {k + 1} (starts at reference state {gamma[0, k]})")
    ax_g.set_ylim(-0.02 * r["n_refs"], 1.02 * r["n_refs"])
    ax_g.set_ylabel(f"failure reference\nstate (0 to {r['n_refs'] - 1})", fontsize=8, color=MUTED)
    ax_g.legend(loc="upper right", fontsize=7, frameon=False, labelcolor=INK)
    ax_g.grid(axis="y", color="#e9edf1", lw=0.6)
    ax_g.tick_params(colors=MUTED, labelsize=7)
    for s in ax_g.spines.values():
        s.set_visible(False)
    ax_g.set_title(f"Selected reference state γ ({r['boxes_visited']} of {r['n_refs']} visited over all "
                   f"{r['n_chain']:,} chains)", loc="left", fontsize=9, color=INK)

    strips(ax_x, (x[:, :, keep] == 0).numpy(), [edges[i] for i in keep])
    ax_x.set_title("Edge state: dark = closed, light = open (two strips per edge, chain 1 on top)",
                   loc="left", fontsize=9, color=INK)
    ax_x.set_xlabel("step after burn-in", fontsize=9, color=MUTED)
    fig.suptitle(f"EMA {o} observed failed:  P(F) = {r['P_IS']:.2e},  {r['n_refs']} failure reference states,  "
                 f"{r['n_chain']:,} chains\nP(closed|F) max error {r['closed_err']:.4f},  "
                 f"max edge lag-1 autocorrelation {r['rho1_max']:.3f}",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.subplots_adjust(left=0.13, right=0.97, top=1 - 0.75 / fig.get_figheight(),
                        bottom=0.45 / fig.get_figheight())
    out = HERE / f"traces_ema_{o}.png"
    fig.savefig(out, dpi=200, facecolor=GAP)
    plt.close(fig)
    print("wrote", out)
