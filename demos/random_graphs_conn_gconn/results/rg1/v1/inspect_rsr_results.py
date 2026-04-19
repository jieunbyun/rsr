import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect a RSR results folder by counting reference states and plotting "
            "cumulative runtime over rounds."
        )
    )
    parser.add_argument(
        "results_dir",
        help="Folder name under this directory (for example, tsum_conn) or a full path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path for the cumulative-time plot PNG.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot window in addition to saving the PNG.",
    )
    return parser.parse_args()


def resolve_results_dir(results_dir_arg: str) -> Path:
    candidate = Path(results_dir_arg)
    if not candidate.is_absolute():
        candidate = HERE / candidate
    candidate = candidate.resolve()

    if not candidate.exists():
        raise FileNotFoundError(f"Results folder does not exist: {candidate}")
    if not candidate.is_dir():
        raise NotADirectoryError(f"Expected a directory: {candidate}")
    return candidate


def load_refs_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        refs = json.load(f)
    if not isinstance(refs, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return len(refs)


def load_metrics(path: Path) -> list[dict]:
    metrics = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                metrics.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}"
                ) from exc

    if not metrics:
        raise ValueError(f"No metrics found in {path}")
    return metrics


def build_cumulative_time(metrics: list[dict]) -> tuple[list[int], list[float]]:
    rounds = []
    cumulative_time_sec = []
    running_total = 0.0

    for item in metrics:
        round_idx = item.get("round")
        time_sec = item.get("time_sec")
        if round_idx is None or time_sec is None:
            raise ValueError("Each metrics entry must contain 'round' and 'time_sec'")

        running_total += float(time_sec)
        rounds.append(int(round_idx))
        cumulative_time_sec.append(running_total)

    return rounds, cumulative_time_sec


def plot_cumulative_time(
    rounds: list[int],
    cumulative_time_sec: list[float],
    results_dir: Path,
    output_path: Path,
    show: bool,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
        }
    )

    fig, ax = plt.subplots(figsize=(6.0, 4.0), constrained_layout=True)
    ax.plot(rounds, cumulative_time_sec, color="black", linewidth=1.5)
    ax.set_xlabel("Round")
    ax.set_ylabel("Cumulative time (seconds)")
    ax.set_title(results_dir.name)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.02)

    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results_dir = resolve_results_dir(args.results_dir)

    rules_geq_path = results_dir / "rules_geq_1.json"
    rules_leq_path = results_dir / "rules_leq_0.json"
    metrics_path = results_dir / "metrics.json"

    for required_path in [rules_geq_path, rules_leq_path, metrics_path]:
        if not required_path.exists():
            raise FileNotFoundError(f"Missing required file: {required_path}")

    n_refs_geq = load_refs_count(rules_geq_path)
    n_refs_leq = load_refs_count(rules_leq_path)
    metrics = load_metrics(metrics_path)
    rounds, cumulative_time_sec = build_cumulative_time(metrics)

    output_path = args.output or (results_dir / "cumulative_time_over_round.png")
    if not output_path.is_absolute():
        output_path = (HERE / output_path).resolve()

    print(f"Results folder: {results_dir}")
    print(f"Number of reference states in rules_geq_1.json: {n_refs_geq}")
    print(f"Number of reference states in rules_leq_0.json: {n_refs_leq}")
    print(f"Number of metric rows: {len(metrics)}")
    print(f"Final round: {rounds[-1]}")
    print(f"Final cumulative time (seconds): {cumulative_time_sec[-1]:.6f}")

    plot_cumulative_time(
        rounds=rounds,
        cumulative_time_sec=cumulative_time_sec,
        results_dir=results_dir,
        output_path=output_path,
        show=args.show,
    )
    print(f"Saved cumulative-time plot to: {output_path}")


if __name__ == "__main__":
    main()
