"""Plot per-round loss curves for a single simulation output directory."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


EXPERIMENTS = ["gradient_ascent", "no_movement_baseline"]
LABELS = {
    "gradient_ascent": "moving points + PGD",
    "no_movement_baseline": "no moving",
}
COLORS = {
    "gradient_ascent": "#1f77b4",
    "no_movement_baseline": "#d62728",
}


def load_round_summary(path: Path) -> dict[tuple[str, int], dict[str, float]]:
    rows: dict[tuple[str, int], dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["experiment"], int(row["round"]))
            rows[key] = {
                "mean_start_loss": float(row["mean_start_loss"]),
                "mean_end_loss": float(row["mean_end_loss"]),
                "mean_path_length": float(row["mean_path_length"]),
                "max_path_length": float(row["max_path_length"]),
                "mean_endpoint_density": float(row["mean_endpoint_density"]),
            }
    return rows


def collect_curves(out_dir: Path) -> list[dict[str, float | int | str]]:
    summary = load_round_summary(out_dir / "round_summary.csv")
    rows: list[dict[str, float | int | str]] = []
    for experiment in EXPERIMENTS:
        data = np.load(out_dir / f"{experiment}_simulation_data.npz")
        loss_grids = data["loss_grids"]
        trajectories = data["trajectories"].astype(np.float64)
        for round_id, grid in enumerate(loss_grids):
            row: dict[str, float | int | str] = {
                "round": round_id,
                "experiment": experiment,
                "global_mean": float(grid.mean()),
                "p90": float(np.percentile(grid, 90)),
                "p95": float(np.percentile(grid, 95)),
                "p99": float(np.percentile(grid, 99)),
                "max": float(grid.max()),
                "min": float(grid.min()),
            }
            if round_id > 0:
                train_key = (experiment, round_id - 1)
                round_traj = trajectories[round_id - 1]
                endpoint_displacement = np.linalg.norm(round_traj[-1] - round_traj[0], axis=1)
                endpoint_std = round_traj[-1].std(axis=0)
                row.update(summary[train_key])
                row.update(
                    {
                        "mean_endpoint_displacement": float(endpoint_displacement.mean()),
                        "max_endpoint_displacement": float(endpoint_displacement.max()),
                        "endpoint_std_x": float(endpoint_std[0]),
                        "endpoint_std_y": float(endpoint_std[1]),
                    }
                )
            else:
                row.update(
                    {
                        "mean_start_loss": np.nan,
                        "mean_end_loss": np.nan,
                        "mean_path_length": np.nan,
                        "max_path_length": np.nan,
                        "mean_endpoint_density": np.nan,
                        "mean_endpoint_displacement": np.nan,
                        "max_endpoint_displacement": np.nan,
                        "endpoint_std_x": np.nan,
                        "endpoint_std_y": np.nan,
                    }
                )
            rows.append(row)
    return rows


def add_delta_rows(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    out = list(rows)
    rounds = sorted({int(row["round"]) for row in rows})
    numeric_fields = [
        "global_mean",
        "p90",
        "p95",
        "p99",
        "max",
        "min",
        "mean_start_loss",
        "mean_end_loss",
        "mean_path_length",
        "max_path_length",
        "mean_endpoint_density",
        "mean_endpoint_displacement",
        "max_endpoint_displacement",
        "endpoint_std_x",
        "endpoint_std_y",
    ]
    for round_id in rounds:
        grouped = {row["experiment"]: row for row in rows if int(row["round"]) == round_id}
        if not all(exp in grouped for exp in EXPERIMENTS):
            continue
        moving = grouped["gradient_ascent"]
        baseline = grouped["no_movement_baseline"]
        delta: dict[str, float | int | str] = {
            "round": round_id,
            "experiment": "moving_minus_no_moving",
        }
        for field in numeric_fields:
            delta[field] = float(moving[field]) - float(baseline[field])
        out.append(delta)
    return out


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    fields = [
        "round",
        "experiment",
        "global_mean",
        "p90",
        "p95",
        "p99",
        "max",
        "min",
        "mean_start_loss",
        "mean_end_loss",
        "mean_path_length",
        "max_path_length",
        "mean_endpoint_density",
        "mean_endpoint_displacement",
        "max_endpoint_displacement",
        "endpoint_std_x",
        "endpoint_std_y",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    real_rows = [row for row in rows if row["experiment"] in EXPERIMENTS]
    rounds = sorted({int(row["round"]) for row in real_rows})
    metrics = [
        ("global_mean", "Global mean loss"),
        ("p95", "95th percentile loss"),
        ("max", "Maximum loss"),
        ("mean_endpoint_displacement", "Mean endpoint displacement"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.4), dpi=180, constrained_layout=True)
    for ax, (metric, title) in zip(axes.ravel(), metrics):
        for experiment in EXPERIMENTS:
            values = [
                next(
                    row
                    for row in real_rows
                    if int(row["round"]) == round_id and row["experiment"] == experiment
                )[metric]
                for round_id in rounds
            ]
            ax.plot(
                rounds,
                values,
                linewidth=2.1,
                color=COLORS[experiment],
                label=LABELS[experiment],
            )
        ax.set_title(title)
        ax.set_xlabel("training round")
        ax.grid(True, alpha=0.24)
    axes[0, 0].set_ylabel("L")
    axes[0, 1].set_ylabel("L")
    axes[1, 0].set_ylabel("L")
    axes[1, 1].set_ylabel("distance from sample")
    axes[1, 1].set_ylim(-0.03, 0.55)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Moving-point PGD training lowers the loss landscape faster", fontsize=13)
    fig.savefig(path)
    plt.close(fig)


def plot_final_comparison(path: Path, out_dir: Path) -> None:
    data = {
        experiment: np.load(out_dir / f"{experiment}_simulation_data.npz")["loss_grids"][-1]
        for experiment in EXPERIMENTS
    }
    global_min = min(float(grid.min()) for grid in data.values())
    global_max = max(float(grid.max()) for grid in data.values())

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), dpi=180, constrained_layout=True)
    for ax, experiment in zip(axes, EXPERIMENTS):
        grid = data[experiment]
        im = ax.imshow(
            grid,
            origin="lower",
            cmap="coolwarm",
            vmin=global_min,
            vmax=global_max,
            aspect="equal",
        )
        ax.set_title(
            f"{LABELS[experiment]}\nmean={grid.mean():.3f}, p95={np.percentile(grid, 95):.3f}, max={grid.max():.3f}"
        )
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=0.82)
    fig.suptitle("Final loss landscape comparison", fontsize=13)
    fig.savefig(path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = add_delta_rows(collect_curves(args.out_dir))
    csv_path = args.out_dir / "per_round_loss_curves.csv"
    curve_path = args.out_dir / "loss_reduction_curves.png"
    final_path = args.out_dir / "final_loss_landscape_heatmaps.png"
    write_csv(csv_path, rows)
    plot_curves(curve_path, rows)
    plot_final_comparison(final_path, args.out_dir)
    print(f"Saved {csv_path}")
    print(f"Saved {curve_path}")
    print(f"Saved {final_path}")


if __name__ == "__main__":
    main()
