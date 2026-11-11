"""Summary of run_scale.py against J: the largest error in P(edge closed) against the pair-anchored
reference (reference_scale.py), the largest R-hat, the share of sweeps whose tuple jump moved to a partner,
and the compute time per sweep. Writes scale.png.

    python demos/ema_mcmc/plot_scale.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "toy_ca_mcmc"))
from plot_traces import GAP, INK, MUTED  # noqa: E402

COLOR = {"fail": "#2a78d6", "mixed": "#eb6834"}
LABEL = {"fail": "all failed", "mixed": "half failed, half surviving"}
ESS_MIN = 100   # a reference with fewer effective draws is marked unreliable (hollow marker)

res = json.loads((HERE / "results_scale_pair.json").read_text())
series = {s: sorted((r for r in res.values() if r["series"] == s), key=lambda r: r["J"]) for s in COLOR}

fig, axes = plt.subplots(2, 2, figsize=(9, 6.2))
fig.patch.set_facecolor(GAP)
panels = [
    ("largest |error| in P(edge closed)\n(against the pair-anchored reference)", "log",
     lambda r: r["closed_err"] if r["ess_ref"] > 0 else None),
    ("largest R-hat over edges", "log", lambda r: r["rhat_max"]),
    ("share of sweeps whose tuple jump moved to a partner", "log", lambda r: r["tuple_accept"] * r["tuple_moved"] or None),
    ("median ms per sweep (1,000 chains, incl. tuple jump)", "linear", lambda r: r["ms"]["sweep_median"]),
]
for ax, (title, scale, f) in zip(axes.flat, panels):
    for s, rs in series.items():
        pts = [(r["J"], f(r), r["ess_ref"] >= ESS_MIN) for r in rs if f(r) is not None]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=COLOR[s], lw=2, label=LABEL[s], zorder=2)
        for J, y, good in pts:
            ax.plot(J, y, "o", ms=6, color=COLOR[s], mfc=COLOR[s] if (good or ax is not axes[0, 0]) else GAP,
                    mec=GAP if (good or ax is not axes[0, 0]) else COLOR[s], mew=1.5, zorder=3)
    ax.set_yscale(scale)
    ax.set_title(title, loc="left", fontsize=9, color=INK)
    ax.set_xticks(range(1, 11))
    ax.set_xlim(0.6, 10.4)
    ax.set_xlabel("J, number of observed systems", fontsize=8, color=MUTED)
    ax.grid(color="#e9edf1", lw=0.6)
    ax.set_facecolor(GAP)
    ax.tick_params(colors=MUTED, labelsize=7)
    for sp in ax.spines.values():
        sp.set_visible(False)
axes[0, 0].axhline(0.05, color=MUTED, lw=1, ls="--")
axes[0, 1].axhline(1.01, color=MUTED, lw=1, ls="--")
axes[0, 0].text(0.7, 0.058, "0.05", fontsize=7, color=MUTED, va="bottom")
axes[0, 1].text(0.7, 1.04, "1.01", fontsize=7, color=MUTED, va="bottom")
axes[0, 1].yaxis.set_major_locator(FixedLocator([1, 1.5, 2, 3, 4, 6]))
axes[0, 1].yaxis.set_minor_locator(NullLocator())
axes[0, 1].yaxis.set_major_formatter(ScalarFormatter())
fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.01, 0.955), ncol=2,
           fontsize=8, frameon=False, labelcolor=INK)
fig.suptitle("EMA: scaling of rsr.mcmc with the number of observed systems (sweep + tuple jump, m = 8)",
             x=0.01, ha="left", fontsize=10, color=INK)
fig.text(0.01, 0.01, f"Hollow markers: reference ESS < {ESS_MIN}. Missing points: no reference draw lands in the "
         "event (error), or no tuple jump moved.", fontsize=7, color=MUTED)
fig.subplots_adjust(left=0.07, right=0.96, top=0.85, bottom=0.1, hspace=0.55, wspace=0.2)
fig.savefig(HERE / "scale.png", dpi=200, facecolor=GAP)
print("wrote", HERE / "scale.png")
