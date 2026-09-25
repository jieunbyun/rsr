"""Draw the Hebei transportation multistate network.

Uses ``ndtools.graphs.draw_graph_from_data``, which reads ``nodes.json`` and
``edges.json`` from the data directory and lays the nodes out at their stored
x/y coordinates (the schematic positions digitised from Fig. 7 of
Liu et al. 2021). The result is written next to the data files.

Run from anywhere:

    python datasets/hebei_transport_liu2021/v1/scripts/draw_graph.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parents[0] / "data"
REPO_ROOT = HERE.parents[3]
sys.path.insert(0, str(REPO_ROOT))  # so ndtools resolves without an editable install

from ndtools.graphs import draw_graph_from_data


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR,
                    help="directory holding nodes.json / edges.json")
    ap.add_argument("--out", default="graph.png",
                    help="output file name, written into the data directory")
    args = ap.parse_args()

    out_path = draw_graph_from_data(
        args.data_dir,
        node_color="skyblue",
        node_size=450,
        edge_color="gray",
        with_node_labels=True,
        output_name=args.out,
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
