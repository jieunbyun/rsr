"""Posterior comparison and trace plots of run_scale.py, scored against the pair-anchored
reference of reference_scale.py (reference_scale_pair.pt).

posterior_scale.png: one panel per case (rows: all failed J = 1-5, 6-10; half failed J = 2-6,
    7-10). Each point is one edge: x = reference P(e closed | E), y = MCMC; the bars are +-2
    reference standard errors. A case whose reference has fewer than ESS_MIN effective draws has
    no usable reference and its panel says so.

traces_scale/<series>_<J>.png: every chain, burn-in included. One panel per observed system:
    one row per chain, coloured by the box it holds at each sweep; the N_BOX boxes most occupied
    after burn-in (with at least MIN_SHARE of it) have their own colours, every other box is grey,
    so a chain that never leaves its box is a flat stripe. Bottom: for the N_EDGE edges with the
    most uncertain posterior, the share of all chains with the edge closed at each sweep, with the
    reference in the right-hand column (blank without a usable reference); at stationarity each
    row is flat at its reference.

    python demos/ema_mcmc/plot_scale_detail.py
"""
import json
import sys
from pathlib import Path

import torch  # noqa: F401  (load torch's DLLs before matplotlib's text rendering in the ieee-rsr env)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "toy_ca_mcmc"))
from plot_traces import GAP, INK, MUTED  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]   # categorical 1-7
OTHER, GRID, POINT = "#c9ced4", "#e9edf1", "#2a78d6"
SEQ = LinearSegmentedColormap.from_list("blue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
N_EDGE, N_BOX, MIN_SHARE, ESS_MIN = 25, 7, 1e-3, 100
LABEL = {"fail": "all failed", "mixed": "half failed, half surviving"}

res = json.loads((HERE / "results_scale_pair.json").read_text())
refs = torch.load(HERE / "reference_scale_pair.pt")


def usable(name):
    return refs[name]["ess"] >= ESS_MIN


def clean(ax):
    ax.tick_params(colors=MUTED, labelsize=7)
    for s in ax.spines.values():
        s.set_visible(False)


def describe(r):
    obs = ", ".join(f"{o} {'F' if s == 'fail' else 'S'}" for o, s in zip(r["systems"], r["states"]))
    return f"{LABEL[r['series']]}, J = {r['J']} ({obs})"


# ---------------------------------------------------------------- posterior comparison
layout = [[f"fail/{J}" for J in range(1, 6)], [f"fail/{J}" for J in range(6, 11)],
          [f"mixed/{J}" for J in range(2, 7)], [f"mixed/{J}" for J in range(7, 11)] + [None]]
fig, axes = plt.subplots(4, 5, figsize=(15, 13.2), sharex=True, sharey=True)
fig.patch.set_facecolor(GAP)
for i, row in enumerate(layout):
    for j, name in enumerate(row):
        ax = axes[i, j]
        if name is None:
            ax.set_visible(False)
            continue
        r, ref = res[name], refs[name]
        est = np.array(r["closed_mcmc"])
        ax.plot([0, 1], [0, 1], ls="--", lw=1, color=MUTED, zorder=1)
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.set_aspect("equal")
        ax.grid(color=GRID, lw=0.6)
        clean(ax)
        head = f"{LABEL[r['series']]}, J = {r['J']}\n"
        if usable(name):
            x, se = ref["closed"].numpy(), ref["closed_se"].numpy()
            err = np.abs(est - x)
            ax.errorbar(x, est, xerr=2 * se, fmt="none", ecolor=OTHER, elinewidth=0.8, zorder=2)
            ax.scatter(x, est, s=14, color=POINT, edgecolor=GAP, linewidth=0.5, zorder=3)
            ax.set_title(f"{head}max error {err.max():.3f}; {int((err > 0.05).sum())} edges off > 0.05\n"
                         f"P(E) = {ref['P']:.1e}, ref. ESS {ref['ess']:,.0f}, R-hat {r['rhat_max']:.2f}",
                         fontsize=8, color=INK, loc="left")
        else:
            ax.text(0.5, 0.5, f"no usable reference\n(ref. ESS {ref['ess']:.0f})\n\nR-hat {r['rhat_max']:.2f}",
                    ha="center", va="center", fontsize=9, color=MUTED, transform=ax.transAxes)
            ax.set_title(f"{head}\n", fontsize=8, color=INK, loc="left")
        if i == 3 or (i == 2 and j == 4):
            ax.set_xlabel("reference P(e closed | E)", fontsize=8, color=MUTED)
        if j == 0:
            ax.set_ylabel("MCMC P(e closed | E)", fontsize=8, color=MUTED)
axes[2, 4].xaxis.set_tick_params(labelbottom=True)
fig.suptitle("EMA scaling: edge posteriors, MCMC (1,000 chains, 1,000 sweeps after 500 burn-in, tuple jump m = 8) "
             "against the pair-anchored weighted-IS reference (4M draws)\n"
             "Each point is one of the 129 edges; grey bars: +-2 reference standard errors",
             x=0.01, ha="left", fontsize=10, color=INK)
fig.subplots_adjust(left=0.05, right=0.99, top=0.91, bottom=0.04, wspace=0.08, hspace=0.38)
fig.savefig(HERE / "posterior_scale.png", dpi=170, facecolor=GAP)
plt.close(fig)
print("wrote", HERE / "posterior_scale.png")


# ---------------------------------------------------------------- traces, all chains
def traces(name):
    tr = torch.load(HERE / "traces_scale" / f"{name.replace('/', '_')}.pt")
    r, B, edges = res[name], tr["n_burn"], tr["edges"]
    t, T = tr["t"].long(), tr["t"].shape[0]
    C, J = t.shape[1:]
    has_ref = usable(name)
    p = refs[name]["closed"] if has_ref else torch.tensor(r["closed_mcmc"])
    keep = (p * (1 - p)).argsort(descending=True)[:N_EDGE].sort().values
    share = tr["closed_share"][:, keep].numpy()
    heights = [1.5] * J + [0.17 * N_EDGE]
    fig, axes = plt.subplots(J + 1, 1, figsize=(11, sum(heights) + 0.45 * J + 2.0), sharex=True,
                             gridspec_kw={"height_ratios": heights, "hspace": 0.45})
    fig.patch.set_facecolor(GAP)
    for j, ax in enumerate(axes[:J]):
        R = r["n_refs"][j]
        occ = torch.bincount(t[B:, :, j].flatten(), minlength=R).double() / t[B:, :, j].numel()
        top = occ.argsort(descending=True)[:N_BOX]
        top = top[occ[top] >= MIN_SHARE]
        spread = occ[top].sum().item() < 0.5          # many boxes, none dominant: colour by box number
        if spread:
            lut = plt.get_cmap("viridis")(np.linspace(0, 1, R))[:, :3]
        else:
            lut = np.tile(np.array(to_rgb(OTHER)), (R, 1))
            for k, b in enumerate(top.tolist()):
                lut[b] = to_rgb(SERIES[k])
        ax.imshow(lut[t[:, :, j].T.numpy()], aspect="auto", interpolation="nearest", extent=(0, T, C, 0))
        ax.axvline(B, color=INK, lw=1, ls="--")
        ax.set_ylabel(f"{r['systems'][j]}\nchain", fontsize=8, color=MUTED)
        clean(ax)
        if spread:
            handles = [Patch(color=lut[b], label=f"box {b}") for b in np.linspace(0, R - 1, 4).round().astype(int)]
            handles.append(Patch(color=GAP, label=f"(by box number;\nlargest share {occ.max():.1%})"))
        else:
            rest = 1 - occ[top].sum().item()
            handles = [Patch(color=SERIES[k], label=f"box {b} ({occ[b]:.1%})") for k, b in enumerate(top.tolist())]
            handles.append(Patch(color=OTHER, label=f"other ({rest:.1%})"))
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.005, 1.05), fontsize=6, frameon=False,
                  labelcolor=INK, ncol=2 if len(handles) > 4 else 1, columnspacing=0.8, handlelength=1.2)
        state = "failed" if r["states"][j] == "fail" else "surviving"
        ax.set_title(f"Box held for {r['systems'][j]} ({state}; {r['occupied'][j]} of {R} boxes occupied after "
                     "burn-in)", loc="left", fontsize=8, color=INK)
    ax = axes[J]
    gap = max(T // 60, 5)
    img = np.full((N_EDGE, T + gap + max(T // 25, 20)), np.nan)
    img[:, :T] = share.T
    if has_ref:
        img[:, T + gap:] = p[keep].numpy()[:, None]
    cmap = SEQ.copy()
    cmap.set_bad(GAP)
    im = ax.imshow(img, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest",
                   extent=(0, img.shape[1], N_EDGE, 0))
    ax.axvline(B, color=INK, lw=1, ls="--")
    ax.set_yticks(np.arange(N_EDGE) + 0.5, [edges[i] for i in keep], fontsize=6, color=MUTED)
    ax.tick_params(axis="y", length=0)
    ax.set_xticks([v for v in ax.get_xticks() if v <= T])
    ax.text(T + gap + (img.shape[1] - T - gap) / 2, -0.4, "reference" if has_ref else "no ref.", ha="center",
            va="bottom", fontsize=7, color=MUTED)
    clean(ax)
    ax.set_title(f"Share of all {C:,} chains with the edge closed, for the {N_EDGE} edges with the most uncertain "
                 f"posterior ({'reference' if has_ref else 'MCMC'})", loc="left", fontsize=8, color=INK)
    ax.set_xlabel("sweep, burn-in included (dashed line: end of burn-in; each sweep followed by a tuple jump)",
                  fontsize=8, color=MUTED)
    H_in = fig.get_figheight()
    fig.subplots_adjust(left=0.08, right=0.8, top=1 - 1.3 / H_in, bottom=0.45 / H_in)
    pos = ax.get_position()
    cax = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.012, pos.height * 0.7])
    fig.colorbar(im, cax=cax).ax.tick_params(labelsize=7, colors=MUTED)
    cax.set_title("share\nclosed", fontsize=7, color=MUTED)
    for s in cax.spines.values():
        s.set_visible(False)
    acc = f"max error {r['closed_err']:.4f}" if has_ref else f"no usable reference (ESS {refs[name]['ess']:.0f})"
    fig.suptitle(f"EMA {describe(r)}\nP(E) = {r['P_ref']:.1e}, {acc}, R-hat max {r['rhat_max']:.2f}, "
                 f"tuple jump to a partner in {r['tuple_accept'] * r['tuple_moved']:.2%} of sweeps, "
                 f"{r['ms']['sweep_median']:.0f} ms/sweep", x=0.01, ha="left", fontsize=9, color=INK)
    out = HERE / "traces_scale" / f"{name.replace('/', '_')}.png"
    fig.savefig(out, dpi=150, facecolor=GAP)
    plt.close(fig)
    print("wrote", out)


for name in sorted(res, key=lambda k: (k.split("/")[0], int(k.split("/")[1]))):
    traces(name)
