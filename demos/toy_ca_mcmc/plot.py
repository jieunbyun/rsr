"""Figures of run.py (results.json and traces.pt in the demo's folder); ../ema_mcmc/plot.py uses
the same functions.

posterior.png   Each point is one edge: x = reference P(e closed | E) (exact for toy-ca,
                weighted IS with +-2 standard-error bars for EMA), y = MCMC after burn-in.
traces.png      The first chains after burn-in. One panel per observed system: the box each
                chain holds after every sweep and its tuple jump (the y axis lists only the boxes
                these chains visit); a marker on top shows a tuple jump to a partner tuple, the
                other box changes come from the sweep. Bottom: the edge states (dark = closed),
                one strip per chain, for the edges with the most uncertain posterior.
traces_all.png  Every chain, burn-in included. One panel per observed system: one row per chain,
                coloured by the box it holds at each sweep (the most occupied boxes have their
                own colours, the rest are grey). Bottom: the share of all chains with each edge
                closed at each sweep, and the reference in the right-hand column; at stationarity
                every row is flat at its reference.

    python demos/toy_ca_mcmc/plot.py
"""
import json
from pathlib import Path

import torch  # noqa: F401  (load torch's DLLs before matplotlib's text rendering on some Windows envs)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_rgb
from matplotlib.patches import Patch

DARK, LIGHT, GAP, INK, MUTED, GRID = "#26323d", "#e9edf1", "#ffffff", "#1f2328", "#59636e", "#e9edf1"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]   # categorical 1-7
OTHER = "#c9ced4"
SEQ = LinearSegmentedColormap.from_list("blue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
N_EDGE, N_BOX, MIN_SHARE = 25, 7, 1e-3


def clean(ax):
    ax.tick_params(colors=MUTED, labelsize=7)
    for s in ax.spines.values():
        s.set_visible(False)


def strips(ax, dark, labels):
    """dark: (T, C, K) bool. Rows: item 1 chain 1, item 1 chain 2, gap, item 2 chain 1, ..."""
    T, C, K = dark.shape
    img = np.full((K * (C + 1) - 1, T), np.nan)
    for k in range(K):
        img[k * (C + 1): k * (C + 1) + C] = dark[:, :, k].T
    cmap = ListedColormap([LIGHT, DARK])
    cmap.set_bad(GAP)
    ax.imshow(img, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="antialiased",
              extent=(0, T, img.shape[0], 0))
    ax.set_yticks([k * (C + 1) + C / 2 for k in range(K)], labels, fontsize=7, color=MUTED)
    ax.tick_params(axis="y", length=0)
    clean(ax)


def header(r):
    obs = ", ".join(f"{o} {'failed' if s == 0 else 'survived'}" for o, s in zip(r["systems"], r["observed"]))
    return (f"{r['title']}, {obs}:  P(E) = {r['P_E']:.2e},  {r['n_chain']:,} chains,  "
            f"tuple jump m = {r['n_partner']}, gamma = {r['gamma']:g}, random order\n"
            f"P(e closed | E) max error {r['closed_err']:.4f} against the {r['ref_name']} reference,  "
            f"R-hat max {r['rhat_max']:.3f},  tuple jump to a partner in "
            f"{r['tuple_accept'] * r['tuple_moved']:.2%} of sweeps,  {r['ms']['sweep_median']:.0f} ms per sweep")


def finish(fig, r, out, top_in=1.15, right=0.97, left=0.13):
    H_in = fig.get_figheight()
    fig.suptitle(header(r), x=0.01, ha="left", fontsize=9, color=INK)
    fig.subplots_adjust(left=left, right=right, top=1 - top_in / H_in, bottom=0.5 / H_in)
    fig.savefig(out, dpi=170, facecolor=GAP)
    plt.close(fig)
    print("wrote", out)


def posterior(r, out):
    ref, se, est = (np.array(r[k]) for k in ("closed_ref", "closed_ref_se", "closed_mcmc"))
    err = np.abs(est - ref)
    fig, ax = plt.subplots(figsize=(5.2, 5.6))
    fig.patch.set_facecolor(GAP)
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=MUTED, zorder=1)
    if se.any():
        ax.errorbar(ref, est, xerr=2 * se, fmt="none", ecolor=OTHER, elinewidth=0.8, zorder=2)
    ax.scatter(ref, est, s=18, color=SERIES[0], edgecolor=GAP, linewidth=0.6, zorder=3)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_aspect("equal")
    ax.grid(color=GRID, lw=0.6)
    clean(ax)
    ax.set_xlabel(f"{r['ref_name']} P(e closed | E)", fontsize=8, color=MUTED)
    ax.set_ylabel("MCMC P(e closed | E)", fontsize=8, color=MUTED)
    obs = " + ".join(r["systems"])
    bars = "; grey bars: +-2 reference SE" if se.any() else ""
    ax.set_title(f"{r['title']}, {obs} observed: edge posteriors\nmax error {err.max():.4f}; "
                 f"{int((err > 0.05).sum())}/{len(err)} edges off by > 0.05\n"
                 f"({r['n_chain']:,} chains, {r['n_keep']:,} sweeps after {r['n_burn']} burn-in{bars})",
                 loc="left", fontsize=8, color=INK)
    fig.subplots_adjust(left=0.13, right=0.97, top=0.85, bottom=0.09)
    fig.savefig(out, dpi=170, facecolor=GAP)
    plt.close(fig)
    print("wrote", out)


def two_chains(r, tr, keep, out):
    B = tr["n_burn"]
    x = tr["x"][B:]
    C = x.shape[1]
    t, moved = tr["t"][B:, :C].long().numpy(), tr["tuple_changed"][B:, :C].numpy()
    T, _, J = t.shape
    rows = len(keep) * (C + 1) - 1
    heights = [0.6 + 0.22 * (len(np.unique(t[:, :, j])) + 1) for j in range(J)] + [0.085 * rows]
    fig, axes = plt.subplots(J + 1, 1, figsize=(11, sum(heights) + 0.55 * J + 2.0), sharex=True,
                             gridspec_kw={"height_ratios": heights, "hspace": 0.35})
    fig.patch.set_facecolor(GAP)
    for j, ax in enumerate(axes[:J]):
        R = r["n_refs"][j]
        visited = np.unique(t[:, :, j])
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
        ax.set_ylabel(f"{r['systems'][j]} box\n(of 0 to {R - 1})", fontsize=8, color=MUTED)
        ax.grid(axis="y", color=GRID, lw=0.6)
        clean(ax)
        ax.set_title(f"Box held for {r['systems'][j]} by these chains ({r['occupied'][j]} of {R} boxes occupied "
                     f"over all {r['n_chain']:,} chains)", loc="left", fontsize=9, color=INK)
    strips(axes[J], (x[:, :, keep] == 0).numpy(), [r["edges"][i] for i in keep])
    what = "every edge" if len(keep) == len(r["edges"]) else f"the {len(keep)} edges with the most uncertain posterior"
    axes[J].set_title(f"Edge state of {what}: dark = closed, light = open (one strip per chain, chain 1 on top)",
                      loc="left", fontsize=9, color=INK)
    axes[J].set_xlabel("sweep after burn-in (each followed by a tuple jump)", fontsize=9, color=MUTED)
    H_in = fig.get_figheight()
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.01, 1 - 0.68 / H_in),
               ncol=4, fontsize=8, frameon=False, labelcolor=INK)
    finish(fig, r, out, top_in=1.3)


def all_chains(r, tr, keep, out):
    B = tr["n_burn"]
    t, share = tr["t"].long(), tr["closed_share"][:, keep].numpy()
    T, C, J = t.shape
    heights = [2.2] * J + [max(0.17 * len(keep), 1.5)]
    fig, axes = plt.subplots(J + 1, 1, figsize=(11, sum(heights) + 0.6 * J + 2.0), sharex=True,
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
        ax.imshow(lut[t[:, :, j].T.numpy()], aspect="auto", interpolation="nearest", extent=(0, T, C, 0))
        ax.axvline(B, color=INK, lw=1, ls="--")
        ax.set_ylabel(f"{r['systems'][j]}: chain\n(1 to {C:,})", fontsize=8, color=MUTED)
        clean(ax)
        rest = 1 - occ[top].sum().item()
        handles = [Patch(color=SERIES[k], label=f"box {b} ({occ[b]:.1%})") for k, b in enumerate(top.tolist())]
        if rest > 0:
            handles.append(Patch(color=OTHER, label=f"other boxes ({rest:.1%})"))
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.005, 1), fontsize=7, frameon=False,
                  labelcolor=INK, title="share after burn-in", title_fontsize=7)
        ax.set_title(f"Box held for {r['systems'][j]} by every chain ({r['occupied'][j]} of {R} boxes occupied after "
                     "burn-in; dashed line: end of burn-in)", loc="left", fontsize=9, color=INK)
    ax = axes[J]
    gap = max(T // 60, 5)
    img = np.full((len(keep), T + gap + max(T // 25, 20)), np.nan)
    img[:, :T] = share.T
    img[:, T + gap:] = np.array(r["closed_ref"])[keep][:, None]
    cmap = SEQ.copy()
    cmap.set_bad(GAP)
    im = ax.imshow(img, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest",
                   extent=(0, img.shape[1], len(keep), 0))
    ax.axvline(B, color=INK, lw=1, ls="--")
    ax.set_yticks(np.arange(len(keep)) + 0.5, [r["edges"][i] for i in keep], fontsize=7, color=MUTED)
    ax.tick_params(axis="y", length=0)
    ax.set_xticks([v for v in ax.get_xticks() if v <= T])
    ax.text(T + gap + (img.shape[1] - T - gap) / 2, -0.4, "reference", ha="center", va="bottom", fontsize=7,
            color=MUTED)
    clean(ax)
    what = "every edge" if len(keep) == len(r["edges"]) else f"the {len(keep)} edges with the most uncertain posterior"
    ax.set_title(f"Share of all {C:,} chains with the edge closed at each sweep, for {what}", loc="left", fontsize=9,
                 color=INK)
    ax.set_xlabel("sweep, burn-in included (each followed by a tuple jump)", fontsize=9, color=MUTED)
    H_in = fig.get_figheight()
    fig.subplots_adjust(left=0.13, right=0.82, top=1 - 1.15 / H_in, bottom=0.5 / H_in)    # as finish(), for the bar
    pos = ax.get_position()
    cax = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.012, pos.height * 0.8])
    fig.colorbar(im, cax=cax).ax.tick_params(labelsize=7, colors=MUTED)
    cax.set_title("share\nclosed", fontsize=7, color=MUTED)
    for s in cax.spines.values():
        s.set_visible(False)
    finish(fig, r, out, right=0.82)


def main(here):
    r = json.loads((here / "results.json").read_text())
    tr = torch.load(here / "traces.pt")
    ref = torch.tensor(r["closed_ref"])
    keep = (ref * (1 - ref)).argsort(descending=True)[:N_EDGE].sort().values
    posterior(r, here / "posterior.png")
    two_chains(r, tr, keep, here / "traces.png")
    all_chains(r, tr, keep, here / "traces_all.png")


if __name__ == "__main__":
    main(Path(__file__).resolve().parent)
