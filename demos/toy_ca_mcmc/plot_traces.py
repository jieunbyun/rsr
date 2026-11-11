"""Trace plots of the selected boxes (gamma) and every edge state, one figure per case:
traces_<origin>.png from traces.pt (one observed system) and traces_<n4+n6...>.png from
traces_multi.pt (several), whichever exist.

All panels use the same strips: one row per box or edge, one strip per chain (chain 1 on
top), dark over the steps where it holds.
    Selected box, one panel per observed system: dark where the chain held that box. The right
        margin gives the exact pi(gamma = r) and the share of steps both chains spent there.
    Edge state: dark = closed, light = open.
A well-mixed chain visits the boxes in proportion to pi(gamma); long solid blocks show a
chain staying in one box, and its free edges turn to speckle while the box's edges stay dark.

    python demos/toy_ca_mcmc/plot_traces.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap

HERE = Path(__file__).resolve().parent
DARK, LIGHT, GAP, INK, MUTED = "#26323d", "#e9edf1", "#ffffff", "#1f2328", "#59636e"
cmap = ListedColormap([LIGHT, DARK])
cmap.set_bad(GAP)


def strips(ax, dark, labels):
    """dark: (T, C, K) bool. Rows: item 1 chain 1, item 1 chain 2, gap, item 2 chain 1, ..."""
    T, C, K = dark.shape
    img = np.full((K * (C + 1) - 1, T), np.nan)
    for k in range(K):
        img[k * (C + 1): k * (C + 1) + C] = dark[:, :, k].T
    ax.imshow(img, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="antialiased",
              extent=(0, T, img.shape[0], 0))
    ys = [k * (C + 1) + C / 2 for k in range(K)]
    ax.set_yticks(ys, labels, fontsize=7, color=MUTED)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=MUTED, labelsize=8)
    for s in ax.spines.values():
        s.set_visible(False)
    return ys


def plot(name, x, gammas, boxes, label_exact, systems, title, edges, out_dir=HERE, header="π(γ)    chains"):
    """x (T, C, n) edge states, 1 = open; gammas, boxes, label_exact: one entry per system.
    label_exact is the left margin column (header names both columns); writes out_dir/traces_<name>.png."""
    T, C, n = x.shape
    rows = [len(b) * (C + 1) - 1 for b in boxes] + [n * (C + 1) - 1]
    suptitle = f"{title}\n{C} chains (two strips per row, chain 1 on top)"
    top_in, gap_in, bottom_in = 0.75, 0.55, 0.45                             # margins and gaps, inches
    plot_in = 0.085 * sum(rows)
    fig, axes = plt.subplots(len(rows), 1, figsize=(11, top_in + bottom_in + plot_in + gap_in * (len(rows) - 1)),
                             sharex=True, gridspec_kw={"height_ratios": rows,
                                                       "hspace": gap_in / (plot_in / len(rows))})
    fig.patch.set_facecolor(GAP)
    for ax, gamma, b, pi, system in zip(axes, gammas, boxes, label_exact, systems):
        selected = torch.nn.functional.one_hot(gamma.long(), len(b)).numpy().astype(bool)
        ys = strips(ax, selected, [e if isinstance(e, str) else f"{r + 1}: {' '.join(e)}"   # ready label or edges
                                   for r, e in enumerate(b)])
        for y, p, q in zip(ys, pi, selected.mean(axis=(0, 1))):
            ax.text(T * 1.005, y, f"{p:.3f}   {q:.3f}", va="center", fontsize=7, color=MUTED)
        ax.text(T * 1.005, -0.6, header, va="bottom", fontsize=7, color=MUTED)
        ax.set_title(f"{system}: selected box γ (edges the box holds closed)", loc="left",
                     fontsize=9, color=INK)
    strips(axes[-1], x == 0, edges)
    axes[-1].set_title("Edge state: dark = closed, light = open", loc="left", fontsize=9, color=INK)
    axes[-1].set_xlabel("step after burn-in", fontsize=9, color=MUTED)
    fig.suptitle(suptitle, x=0.01, ha="left", fontsize=10, color=INK)
    fig.subplots_adjust(left=0.13, right=0.90, top=1 - top_in / fig.get_figheight(),
                        bottom=bottom_in / fig.get_figheight())
    out = Path(out_dir) / f"traces_{name}.png"
    fig.savefig(out, dpi=200, facecolor=GAP)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    if (HERE / "traces.pt").exists():                                       # one observed system
        data = torch.load(HERE / "traces.pt")
        for o, tr in data["traces"].items():
            r = data["results"][o]
            title = (f"{o} observed failed:  P(F) = {r['P_F']:.4f},  {r['n_refs']} failure "
                     f"box{'es' * (r['n_refs'] > 1)},  max edge lag-1 autocorrelation {r['rho1_edge_max']:.3f}")
            plot(o, tr["x"].numpy(), [tr["gamma"]], [tr["boxes"]], [tr["label_exact"]], [o], title,
                 data["edges"])

    if (HERE / "traces_multi.pt").exists():                                 # several observed systems
        data = torch.load(HERE / "traces_multi.pt")
        for name, tr in data["traces"].items():
            r = data["results"][name]
            title = (f"{', '.join(r['systems'])} observed failed:  P(all fail) = {r['P_all_fail']:.4f},  "
                     f"boxes {r['n_refs']},  max edge lag-1 autocorrelation {r['rho1_edge_max']:.3f}")
            plot(name, tr["x"].numpy(), tr["gamma"], tr["boxes"], tr["label_exact"], tr["systems"],
                 title, data["edges"])
