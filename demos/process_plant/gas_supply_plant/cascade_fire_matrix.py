"""Cascading pool-fire probability matrix for gas-supply-plant components.

Components are tanks (point coordinates from nodes.json) and pipelines
(represented by their midpoint, deduplicated by comp_id from edges.json).

For every ordered pair (i, j), computes the probability that component j
ignites given a pool fire at component i, following:

    I(L)  = alpha / L                       (heat flux, solid-flame near-field)
    P_d   = Phi(a + b * ln I)               (probit ignition model)

Diagonal entries (i == j) are set to 1.0 (the source is the one on fire).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.stats import norm


TANK_NODES: tuple[str, ...] = (
    "n5", "n8", "n10", "n12", "n14", "n16",
    "n19", "n21", "n23", "n25", "n27", "n29", "n31",
    "n34", "n36", "n38", "n40", "n42", "n43",
    "n45", "n46", "n48", "n49", "n51", "n52",
)


def load_tank_coords(
    nodes_path: Path,
    tank_ids: Sequence[str] = TANK_NODES,
) -> tuple[list[str], np.ndarray]:
    """Return tank ids (in input order) and an (n, 2) array of (x, y)."""
    nodes = json.loads(nodes_path.read_text(encoding="utf-8"))
    coords = np.array(
        [[nodes[t]["x"], nodes[t]["y"]] for t in tank_ids],
        dtype=float,
    )
    return list(tank_ids), coords


def load_pipeline_midpoints(
    nodes_path: Path,
    edges_path: Path,
) -> tuple[list[str], np.ndarray]:
    """Return pipeline labels and an (n, 2) array of midpoint (x, y).

    Pipelines are deduplicated by `comp_id` (a bidirectional pipeline appears
    as two edges sharing a comp_id). Labels are formatted as `(from, to)`
    using the first edge encountered for each comp_id, matching the
    convention in s02_cal_probs.ipynb.
    """
    nodes = json.loads(nodes_path.read_text(encoding="utf-8"))
    edges = json.loads(edges_path.read_text(encoding="utf-8"))

    labels: list[str] = []
    midpoints: list[list[float]] = []
    seen: set[str] = set()
    for edge in edges.values():
        comp_id = edge.get("comp_id")
        if comp_id is None or comp_id in seen:
            continue
        seen.add(comp_id)
        a_node, b_node = nodes[edge["from"]], nodes[edge["to"]]
        midpoints.append([(a_node["x"] + b_node["x"]) / 2.0,
                          (a_node["y"] + b_node["y"]) / 2.0])
        labels.append(f"({edge['from']}, {edge['to']})")

    return labels, np.asarray(midpoints, dtype=float)


def load_combined_coords(
    nodes_path: Path,
    edges_path: Path,
    tank_ids: Sequence[str] = TANK_NODES,
) -> tuple[list[str], np.ndarray]:
    """Concatenate tank coordinates and pipeline midpoints (tanks first)."""
    tank_labels, tank_xy = load_tank_coords(nodes_path, tank_ids)
    pipe_labels, pipe_xy = load_pipeline_midpoints(nodes_path, edges_path)
    labels = tank_labels + pipe_labels
    coords = np.vstack([tank_xy, pipe_xy])
    return labels, coords


def distance_matrix(coords: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distance matrix; diagonal is 0."""
    diff = coords[:, None, :] - coords[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def fire_probability_matrix(
    coords: np.ndarray,
    alpha: float,
    a: float,
    b: float,
) -> np.ndarray:
    """Probability matrix M[i, j] = P(tank j ignites | pool fire at tank i).

    Uses I = alpha / L and P = Phi(a + b * ln I). Diagonal set to 1.0.
    """
    L = distance_matrix(coords)
    n = L.shape[0]

    with np.errstate(divide="ignore", invalid="ignore"):
        I = alpha / L                          # inf on the diagonal
        ln_I = np.log(I)                       # inf on the diagonal
        probit_arg = a + b * ln_I              # inf on the diagonal
        P = norm.cdf(probit_arg)               # 1.0 on the diagonal automatically

    np.fill_diagonal(P, 1.0)
    return P


def save_matrix(
    matrix: np.ndarray,
    labels: Sequence[str],
    out_path: Path,
    params: dict,
) -> None:
    """Persist matrix as JSON with row/column labels and parameter record."""
    payload = {
        "params": params,
        "labels": list(labels),
        "matrix": matrix.tolist(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def sweep(
    coords: np.ndarray,
    labels: Sequence[str],
    alphas: Iterable[float],
    a_values: Iterable[float],
    b_values: Iterable[float],
    out_dir: Path | None = None,
) -> dict[tuple[float, float, float], np.ndarray]:
    """Compute the probability matrix for every (alpha, a, b) combination.

    Optionally writes one JSON per combination to `out_dir`.
    """
    results: dict[tuple[float, float, float], np.ndarray] = {}
    for alpha in alphas:
        for a in a_values:
            for b in b_values:
                M = fire_probability_matrix(coords, alpha=alpha, a=a, b=b)
                results[(alpha, a, b)] = M

                if out_dir is not None:
                    fname = f"P_alpha{alpha:g}_a{a:g}_b{b:g}.json"
                    save_matrix(
                        M,
                        labels,
                        out_dir / fname,
                        params={"alpha": alpha, "a": a, "b": b},
                    )
    return results


def summarise(M: np.ndarray) -> dict:
    """A few scalars that tend to move clearly with alpha / a / b."""
    off_diag = M[~np.eye(M.shape[0], dtype=bool)]
    return {
        "mean_off_diag": float(off_diag.mean()),
        "max_off_diag": float(off_diag.max()),
        "frac_above_0.5": float((off_diag > 0.5).mean()),
        "frac_above_0.1": float((off_diag > 0.1).mean()),
        "expected_secondary_ignitions_per_source":
            float(off_diag.reshape(M.shape[0], -1).sum(axis=1).mean()),
    }


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    nodes_path = here / "data" / "nodes.json"
    edges_path = here / "data" / "edges.json"
    out_dir = here / "fire_cascade_res"

    labels, coords = load_combined_coords(nodes_path, edges_path)
    n_tanks = len(TANK_NODES)
    n_pipes = len(labels) - n_tanks
    print(f"Components: {n_tanks} tanks + {n_pipes} pipelines "
          f"= {len(labels)} rows/cols.")

    alpha, a, b = 20.0, -6.0, 2.56
    M = fire_probability_matrix(coords, alpha=alpha, a=a, b=b)

    s = summarise(M)
    threshold = 0.03
    eye = np.eye(len(labels), dtype=bool)
    n_above = int((M[~eye] > threshold).sum())
    print(f"\nalpha={alpha}, a={a}, b={b}")
    print(f"  off-diagonal entries > {threshold:.0%} : {n_above}")
    print(f"  max off-diagonal P                : {s['max_off_diag']:.3e}")
    print(f"  mean off-diagonal P               : {s['mean_off_diag']:.3e}")
    print(f"  E[secondary ignitions per source] : "
          f"{s['expected_secondary_ignitions_per_source']:.3e}")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"P_combined_alpha{alpha:g}_a{a:g}_b{b:g}.json"
    save_matrix(
        M, labels, out_path,
        params={"alpha": alpha, "a": a, "b": b,
                "n_tanks": n_tanks, "n_pipelines": n_pipes,
                "threshold": threshold, "n_above_threshold": n_above},
    )
    print(f"\nMatrix written to: {out_path}")
