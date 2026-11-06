"""Trace plots of the selected box (gamma) and every edge state, one figure per origin
(traces_<origin>.png, from traces.pt).

Both panels use the same strips: one row per box or edge, one strip per chain (chain 1 on
top), dark over the steps where it holds.
    Top, selected box:  dark where the chain proposed from that box. The right margin gives
                        the exact pi(gamma = r) and the share of steps both chains spent there.
    Bottom, edge state: dark = closed, light = open.
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
data = torch.load(HERE / "traces.pt")
edges, traces, results = data["edges"], data["traces"], data["results"]

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
    ax.set_yticks([k * (C + 1) + C / 2 for k in range(K)], labels, fontsize=7, color=MUTED)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=MUTED, labelsize=8)
    for s in ax.spines.values():
        s.set_visible(False)
    return [k * (C + 1) + C / 2 for k in range(K)]


for origin, tr in traces.items():
    x, gamma = tr["x"].numpy(), tr["gamma"].long()
    T, C, n = x.shape
    R = len(tr["boxes"])
    rows_g, rows_x = R * (C + 1) - 1, n * (C + 1) - 1
    fig, (ax_g, ax_x) = plt.subplots(
        2, 1, figsize=(11, 1.0 + 0.085 * (rows_g + rows_x)), sharex=True,
        gridspec_kw={"height_ratios": [rows_g, rows_x], "hspace": 0.25})
    fig.patch.set_facecolor(GAP)

    selected = torch.nn.functional.one_hot(gamma, R).numpy().astype(bool)     # (T, C, R)
    box_labels = [f"{r + 1}: {' '.join(b)}" for r, b in enumerate(tr["boxes"])]
    ys = strips(ax_g, selected, box_labels)
    share = selected.mean(axis=(0, 1))
    for y, p, q in zip(ys, tr["label_exact"], share):
        ax_g.text(T * 1.005, y, f"{p:.3f}   {q:.3f}", va="center", fontsize=7, color=MUTED)
    ax_g.text(T * 1.005, -0.6, "π(γ)    chains", va="bottom", fontsize=7, color=MUTED)
    ax_g.set_title("Selected box γ (edges the box holds closed)", loc="left", fontsize=9, color=INK)

    strips(ax_x, x == 0, edges)
    ax_x.set_title("Edge state: dark = closed, light = open", loc="left", fontsize=9, color=INK)
    ax_x.set_xlabel("step after burn-in", fontsize=9, color=MUTED)

    r = results[origin]
    fig.suptitle(f"{origin}:  P(F) = {r['P_F']:.4f},  {R} failure box{'es' * (R > 1)},  {C} chains "
                 f"(two strips per row, chain 1 on top),  max edge lag-1 autocorrelation "
                 f"{r['rho1_edge_max']:.3f}", x=0.01, ha="left", fontsize=10, color=INK)
    fig.subplots_adjust(left=0.13, right=0.90, top=1 - 0.55 / fig.get_figheight(),
                        bottom=0.45 / fig.get_figheight())
    out = HERE / f"traces_{origin}.png"
    fig.savefig(out, dpi=200, facecolor=GAP)
    plt.close(fig)
    print("wrote", out)
