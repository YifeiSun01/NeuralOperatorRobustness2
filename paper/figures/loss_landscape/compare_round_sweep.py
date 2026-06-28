"""Compare moving-point and no-moving simulations across round counts."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


EXPERIMENTS = ["gradient_ascent", "no_movement_baseline"]


def read_last_round_metrics(summary_path: Path, experiment: str) -> dict[str, float]:
    with summary_path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["experiment"] == experiment]
    last = rows[-1]
    return {
        "last_start_mean": float(last["mean_start_loss"]),
        "last_end_mean": float(last["mean_end_loss"]),
        "last_path_length": float(last["mean_path_length"]),
    }


def collect_metrics(base_dir: Path) -> list[dict[str, float | str | int]]:
    rows: list[dict[str, float | str | int]] = []
    for round_dir in sorted(base_dir.glob("round_*")):
        if not round_dir.is_dir():
            continue
        rounds = int(round_dir.name.split("_")[-1])
        for experiment in EXPERIMENTS:
            data = np.load(round_dir / f"{experiment}_simulation_data.npz")
            final_loss = data["loss_grids"][-1]
            last = read_last_round_metrics(round_dir / "round_summary.csv", experiment)
            rows.append(
                {
                    "rounds": rounds,
                    "experiment": experiment,
                    "global_mean": float(final_loss.mean()),
                    "p95": float(np.percentile(final_loss, 95)),
                    "p99": float(np.percentile(final_loss, 99)),
                    "max": float(final_loss.max()),
                    **last,
                }
            )
    return rows


def add_delta_rows(rows: list[dict[str, float | str | int]]) -> list[dict[str, float | str | int]]:
    rows_with_delta = list(rows)
    by_round = {}
    for row in rows:
        by_round.setdefault(row["rounds"], {})[row["experiment"]] = row
    for rounds, grouped in sorted(by_round.items()):
        if not all(exp in grouped for exp in EXPERIMENTS):
            continue
        moving = grouped["gradient_ascent"]
        baseline = grouped["no_movement_baseline"]
        rows_with_delta.append(
            {
                "rounds": rounds,
                "experiment": "moving_minus_no_moving",
                "global_mean": moving["global_mean"] - baseline["global_mean"],
                "p95": moving["p95"] - baseline["p95"],
                "p99": moving["p99"] - baseline["p99"],
                "max": moving["max"] - baseline["max"],
                "last_start_mean": moving["last_start_mean"] - baseline["last_start_mean"],
                "last_end_mean": moving["last_end_mean"] - baseline["last_end_mean"],
                "last_path_length": moving["last_path_length"] - baseline["last_path_length"],
            }
        )
    return rows_with_delta


def write_csv(path: Path, rows: list[dict[str, float | str | int]]) -> None:
    fieldnames = [
        "rounds",
        "experiment",
        "global_mean",
        "p95",
        "p99",
        "max",
        "last_start_mean",
        "last_end_mean",
        "last_path_length",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_comparison(path: Path, rows: list[dict[str, float | str | int]]) -> None:
    real_rows = [row for row in rows if row["experiment"] in EXPERIMENTS]
    rounds = sorted({int(row["rounds"]) for row in real_rows})
    metrics = [
        ("global_mean", "Global mean L"),
        ("p95", "95th percentile L"),
        ("max", "Max L"),
    ]
    colors = {
        "gradient_ascent": "#1f77b4",
        "no_movement_baseline": "#d62728",
    }
    labels = {
        "gradient_ascent": "moving points",
        "no_movement_baseline": "no moving",
    }

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), dpi=180, constrained_layout=True)
    for ax, (metric, title) in zip(axes, metrics):
        for experiment in EXPERIMENTS:
            values = [
                next(row for row in real_rows if row["rounds"] == r and row["experiment"] == experiment)[metric]
                for r in rounds
            ]
            ax.plot(rounds, values, marker="o", linewidth=2.0, color=colors[experiment], label=labels[experiment])
        ax.set_title(title)
        ax.set_xlabel("training rounds")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.25)
        ax.set_xticks(rounds)
    axes[0].legend(frameon=False)
    fig.suptitle("Moving-point training vs no-moving baseline", fontsize=13)
    fig.savefig(path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("figures/loss_landscape/simulation_outputs/alpha_0p0015_eps_0p12"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = add_delta_rows(collect_metrics(args.base_dir))
    write_csv(args.base_dir / "round_sweep_comparison.csv", rows)
    plot_comparison(args.base_dir / "round_sweep_comparison.png", rows)
    print(f"Saved {args.base_dir / 'round_sweep_comparison.csv'}")
    print(f"Saved {args.base_dir / 'round_sweep_comparison.png'}")


if __name__ == "__main__":
    main()
