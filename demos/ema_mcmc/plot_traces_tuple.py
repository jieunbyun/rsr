"""Trace plots of run_tuple.py, two figures per start.

traces_tuple_<start>.png, the first chains after burn-in:
    Top, one panel per observed system: the box (reference state) each traced chain holds after
    every sweep and its tuple jump, as a step line; the y axis lists only the boxes the traced
    chains visit, by number. A marker at the top shows each sweep whose tuple jump was accepted
    from a partner tuple; the other box changes come from the sweep (a single-system jump or the
    label refresh). Bottom: the N_EDGE edges with the most uncertain posterior (largest p (1 - p)
    under the weighted-IS reference), one strip per chain (chain 1 on top), dark = closed.

traces_tuple_all_<start>.png, every chain, burn-in included:
    One panel per observed system: one row per chain, coloured by the box it holds at each
    sweep; the N_BOX boxes most occupied after burn-in (with at least MIN_SHARE of it) have their
    own colours, every other box is grey. Bottom: for the same edges, the share of all chains with the edge closed at each
    sweep, with the weighted-IS reference in the right-hand column; at stationarity each row is
    flat at its reference.

    python demos/ema_mcmc/plot_traces_tuple.py
"""
import sys
from pathlib import Path

import torch  # noqa: F401  (in the ieee-rsr env, matplotlib's text rendering crashes unless torch's DLLs load first)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "toy_ca_mcmc"))
from plot_traces import GAP, INK, MUTED, strips  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]   # categorical 1-7
OTHER = "#c9ced4"
SEQ = LinearSegmentedColormap.from_list("blue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
N_EDGE, N_BOX, MIN_SHARE = 25, 7, 1e-3

data = torch.load(HERE / "traces_tuple.pt")
edges, systems, ref = data["edges"], data["systems"], data["closed_ref"]
keep = (ref * (1 - ref)).argsort(descending=True)[:N_EDGE].sort().values


def clean(ax):
    ax.tick_params(colors=MUTED, labelsize=7)
    for s in ax.spines.values():
        s.set_visible(False)


def finish(fig, title, out, legend=None, right=0.97):
    H_in = fig.get_figheight()
    if legend:
        fig.legend(*legend, loc="upper left", bbox_to_anchor=(0.01, 1 - 0.62 / H_in), ncol=4, fontsize=8,
                   frameon=False, labelcolor=INK)
    fig.suptitle(title, x=0.01, ha="left", fontsize=10, color=INK)
    fig.subplots_adjust(left=0.13, right=right, top=1 - 1.15 / H_in, bottom=0.45 / H_in)
    fig.savefig(out, dpi=200, facecolor=GAP)
    plt.close(fig)
    print("wrote", out)


def header(r, start, extra):
    return (f"EMA {' + '.join(systems)} observed failed, {start} start:  P(F) = {r['P_ref']:.2e},  "
            f"{r['n_chain']:,} chains,  tuple jump m = {r['n_partner']}, gamma = {r['gamma']:g}, random order\n"
            f"P(closed|F) max error {r['closed_err']:.4f},  R-hat max {r['rhat_max']:.3f},  "
            f"tuple jumps to a partner {r['tuple_accept'] * r['tuple_moved']:.3%} of sweeps{extra}")


def two_chains(start, tr):
    r, B = tr["result"], tr["n_burn"]
    x = tr["x"][B:]
    C = x.shape[1]
    t, moved = tr["t"][B:, :C].numpy(), tr["tuple_changed"][B:, :C].numpy()
    T, _, J = t.shape
    rows = N_EDGE * (C + 1) - 1
    heights = [0.6 + 0.22 * (len(np.unique(t[:, :, j])) + 1) for j in range(J)] + [0.085 * rows]
    fig, axes = plt.subplots(J + 1, 1, figsize=(11, sum(heights) + 0.55 * J + 1.6), sharex=True,
                             gridspec_kw={"height_ratios": heights, "hspace": 0.3})
    fig.patch.set_facecolor(GAP)
    for j, ax in enumerate(axes[:J]):
        R = r["n_refs"][j]
        visited = np.unique(t[:, :, j])                                        # rows: only the boxes visited
        row = np.searchsorted(visited, t[:, :, j])
        V = len(visited)
        for k in range(C):
            ax.plot(range(T), row[:, k] + 0.06 * (k - (C - 1) / 2), drawstyle="steps-post", lw=1.4,
                    color=SERIES[k], label=f"chain {k + 1}" if j == 0 else None)
            hit = moved[:, k].nonzero()[0]
            ax.scatter(hit, [V - 0.35 + 0.3 * k] * len(hit), marker="v", s=16, color=SERIES[k],
                       label=f"chain {k + 1}: tuple jump to a partner" if j == 0 else None, zorder=3)
        ax.set_ylim(-0.6, V + 0.4)
        ax.set_yticks(range(V), [str(b) for b in visited])
        ax.set_ylabel(f"{systems[j]} box\n(visited by these\nchains, of 0 to {R - 1})", fontsize=8, color=MUTED)
        ax.grid(axis="y", color="#e9edf1", lw=0.6)
        clean(ax)
        ax.set_title(f"Box held for {systems[j]} ({r['occupied'][j]} of {R} occupied over all "
                     f"{r['n_chain']:,} chains)", loc="left", fontsize=9, color=INK)
    strips(axes[J], (x[:, :, keep] == 0).numpy(), [edges[i] for i in keep])
    axes[J].set_title(f"Edge state of the {N_EDGE} edges with the most uncertain posterior: dark = closed, "
                      "light = open (two strips per edge, chain 1 on top)", loc="left", fontsize=9, color=INK)
    axes[J].set_xlabel("sweep after burn-in (each followed by a tuple jump)", fontsize=9, color=MUTED)
    finish(fig, header(r, start, ""), HERE / f"traces_tuple_{start}.png", axes[0].get_legend_handles_labels())


def all_chains(start, tr):
    r, B = tr["result"], tr["n_burn"]
    t, share = tr["t"].long(), tr["closed_share"][:, keep].numpy()
    T, C, J = t.shape
    heights = [2.6] * J + [0.17 * N_EDGE]
    fig, axes = plt.subplots(J + 1, 1, figsize=(11, sum(heights) + 0.6 * J + 1.9), sharex=True,
                             gridspec_kw={"height_ratios": heights, "hspace": 0.35})
    fig.patch.set_facecolor(GAP)
    for j, ax in enumerate(axes[:J]):
        R = r["n_refs"][j]
        occ = torch.bincount(t[B:, :, j].flatten(), minlength=R).double() / t[B:, :, j].numel()
        top = occ.argsort(descending=True)[:N_BOX]
        top = top[occ[top] >= MIN_SHARE]
        lut = np.tile(np.array(to_rgb(OTHER)), (R, 1))
        for k, b in enumerate(top.tolist()):
            lut[b] = to_rgb(SERIES[k])
        ax.imshow(lut[t[:, :, j].T.numpy()], aspect="auto", interpolation="antialiased", extent=(0, T, C, 0))
        ax.axvline(B, color=INK, lw=1, ls="--")
        ax.set_ylabel(f"{systems[j]}: chain\n(1 to {C:,})", fontsize=8, color=MUTED)
        clean(ax)
        rest = 1 - occ[top].sum().item()
        handles = [Patch(color=SERIES[k], label=f"box {b} ({occ[b]:.1%})") for k, b in enumerate(top.tolist())]
        handles.append(Patch(color=OTHER, label=f"other boxes ({rest:.1%})"))
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.005, 1), fontsize=7, frameon=False,
                  labelcolor=INK, title="share after burn-in", title_fontsize=7)
        ax.set_title(f"Box held for {systems[j]} by every chain ({r['occupied'][j]} of {R} boxes occupied after "
                     "burn-in; dashed line: end of burn-in)", loc="left", fontsize=9, color=INK)
    ax = axes[J]
    gap = max(T // 60, 5)
    img = np.full((N_EDGE, T + gap + max(T // 25, 20)), np.nan)
    img[:, :T] = share.T
    img[:, T + gap:] = ref[keep].numpy()[:, None]
    cmap = SEQ.copy()
    cmap.set_bad(GAP)
    im = ax.imshow(img, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest",
                   extent=(0, img.shape[1], N_EDGE, 0))
    ax.axvline(B, color=INK, lw=1, ls="--")
    ax.set_yticks(np.arange(N_EDGE) + 0.5, [edges[i] for i in keep], fontsize=7, color=MUTED)
    ax.tick_params(axis="y", length=0)
    ax.set_xticks([v for v in ax.get_xticks() if v <= T])
    ax.text(T + gap + (img.shape[1] - T - gap) / 2, -0.4, "reference", ha="center", va="bottom", fontsize=7,
            color=MUTED)
    clean(ax)
    ax.set_title(f"Share of all {C:,} chains with the edge closed, at each sweep, for the {N_EDGE} edges with the "
                 "most uncertain posterior", loc="left", fontsize=9, color=INK)
    ax.set_xlabel("sweep, burn-in included (each followed by a tuple jump)", fontsize=9, color=MUTED)
    H_in = fig.get_figheight()
    fig.subplots_adjust(left=0.13, right=0.82, top=1 - 1.15 / H_in, bottom=0.45 / H_in)    # as finish(), to place the bar
    pos = ax.get_position()
    cax = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.012, pos.height * 0.7])
    fig.colorbar(im, cax=cax).ax.tick_params(labelsize=7, colors=MUTED)
    cax.set_title("share\nclosed", fontsize=7, color=MUTED)
    for s in cax.spines.values():
        s.set_visible(False)
    finish(fig, header(r, start, ""), HERE / f"traces_tuple_all_{start}.png", right=0.82)


for start, tr in data["traces"].items():
    two_chains(start, tr)
    all_chains(start, tr)
