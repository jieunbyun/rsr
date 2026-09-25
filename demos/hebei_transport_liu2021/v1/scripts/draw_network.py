"""
Draw the Hebei transportation multistate network (Liu et al. 2021, Fig. 7)
as a publication-quality figure, without the underlying province map.

Node positions come from ``data/nodes.json`` and arcs from ``data/edges.json``.
The source (n1, CD) and sink (n22, HD) are shaded grey as in the paper.

Usage:
    python draw_network.py                       # writes ../data/hebei_network.{pdf,png}
    python draw_network.py --out my_fig --width 3.5
"""

import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"

# Elsevier-like typography: serif text, STIX maths
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

NODE_R = 1.9            # node radius in data units
ARC_LABEL_OFFSET = 1.6  # perpendicular offset of a_i labels from the arc

# City-label offsets (dx, dy) in data units; default is up-left of the node.
CITY_OFFSET = {
    "n1": (0.0, 3.6),    # CD
    "n2": (-3.4, 2.6),   # ZJK
    "n3": (-3.8, 0.0),   # BJ
    "n4": (2.2, 3.0),    # QHD
    "n5": (2.6, 2.6),    # TS
    "n7": (2.6, 2.4),    # TJ
    "n8": (-3.2, -2.0),  # BZ
    "n10": (-0.8, 3.2),  # BD
    "n11": (3.4, -2.4),  # HH
    "n12": (2.4, -2.8),  # CZ
    "n14": (2.6, -2.8),  # HS
    "n15": (-3.6, 2.2),  # SJZ
    "n18": (3.0, 1.0),   # XT
    "n22": (0.0, -3.4),  # HD
}

# Which side of the arc its a_i label sits on (+1 = left of travel direction).
ARC_LABEL_SIDE = {
    "a1": +1, "a2": -1, "a3": -1, "a4": +1, "a5": -1, "a6": +1, "a7": +1,
    "a8": +1, "a9": -1, "a10": -1, "a11": +1, "a12": -1, "a13": -1, "a14": -1,
    "a15": +1, "a16": -1, "a17": +1, "a18": +1, "a19": -1, "a20": -1, "a21": +1,
    "a22": -1, "a23": +1, "a24": -1, "a25": -1, "a26": -1, "a27": -1, "a28": -1,
    "a29": -1, "a30": -1, "a31": -1, "a32": -1, "a33": -1,
}


def parse_args():
    p = argparse.ArgumentParser(description="Draw the Hebei transport network")
    p.add_argument("--out", type=str, default=str(DATA_DIR / "hebei_network"),
                   help="Output path without extension (default: ../data/hebei_network)")
    p.add_argument("--width", type=float, default=3.5,
                   help="Figure width in inches (default: 3.5, single journal column)")
    p.add_argument("--no-city", action="store_true", help="Omit city abbreviations")
    return p.parse_args()


def draw(nodes, edges, width, show_city=True):
    xs = np.array([v["x"] for v in nodes.values()])
    ys = np.array([v["y"] for v in nodes.values()])
    pad = 5.0
    xlim = (xs.min() - pad, xs.max() + pad)
    ylim = (ys.min() - pad, ys.max() + pad)
    aspect = (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])

    fig, ax = plt.subplots(figsize=(width, width * aspect))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")

    pos = {k: np.array([v["x"], v["y"]]) for k, v in nodes.items()}

    # Arcs: straight arrows trimmed to node boundaries
    for e in edges.values():
        p0, p1 = pos[e["from"]], pos[e["to"]]
        d = p1 - p0
        u = d / np.linalg.norm(d)
        start, end = p0 + u * NODE_R, p1 - u * NODE_R
        ax.add_patch(FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=7,
            lw=0.7, color="black", shrinkA=0, shrinkB=0, zorder=1))

        lab = e["label"]
        normal = np.array([-u[1], u[0]]) * ARC_LABEL_SIDE.get(lab, 1)
        mid = (p0 + p1) / 2 + normal * ARC_LABEL_OFFSET
        idx = lab[1:]
        ax.text(*mid, rf"$a_{{{idx}}}$", fontsize=6.5, ha="center", va="center",
                zorder=3)

    # Nodes
    for nid, v in nodes.items():
        terminal = v.get("role") in ("source", "sink")
        ax.add_patch(Circle(pos[nid], NODE_R, facecolor="0.6" if terminal else "white",
                            edgecolor="black", lw=0.8, zorder=2))
        num = nid[1:]
        txt = {"source": "s", "sink": "t"}.get(v.get("role"), num)
        ax.text(*pos[nid], txt, fontsize=6.5 if not terminal else 7.5,
                ha="center", va="center", zorder=4,
                fontweight="bold", style="italic" if terminal else "normal")

        if show_city and v.get("label"):
            dx, dy = CITY_OFFSET.get(nid, (-3.0, 2.6))
            ax.text(pos[nid][0] + dx, pos[nid][1] + dy, v["label"], fontsize=6.5,
                    ha="center", va="center", zorder=4)

    fig.tight_layout(pad=0.1)
    return fig


def main():
    args = parse_args()
    nodes = json.loads((DATA_DIR / "nodes.json").read_text(encoding="utf-8"))
    edges = json.loads((DATA_DIR / "edges.json").read_text(encoding="utf-8"))

    fig = draw(nodes, edges, args.width, show_city=not args.no_city)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=600, bbox_inches="tight")
    print(f"Saved {out.with_suffix('.pdf')} and {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
