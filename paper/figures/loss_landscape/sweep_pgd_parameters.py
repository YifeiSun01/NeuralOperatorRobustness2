"""Sweep PGD parameters to find early moving-point advantages."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from simulate_loss_landscape import (
    SimulationConfig,
    build_density_sampler,
    density_value,
    loss_height,
    make_grid,
    run_experiment,
)


CHECKPOINTS = [1, 5, 10, 20, 25]


def metrics_for_grid(grid: np.ndarray) -> dict[str, float]:
    return {
        "global_mean": float(grid.mean()),
        "p95": float(np.percentile(grid, 95)),
        "max": float(grid.max()),
    }


def endpoint_spread(arrays: dict[str, np.ndarray], round_id: int) -> dict[str, float]:
    trajectories = arrays["trajectories"][round_id - 1].astype(np.float64)
    endpoint_displacement = np.linalg.norm(trajectories[-1] - trajectories[0], axis=1)
    endpoints = trajectories[-1]
    return {
        "mean_endpoint_displacement": float(endpoint_displacement.mean()),
        "max_endpoint_displacement": float(endpoint_displacement.max()),
        "endpoint_std_x": float(endpoints[:, 0].std()),
        "endpoint_std_y": float(endpoints[:, 1].std()),
    }


def run_trial(
    config: SimulationConfig,
    density_grid: np.ndarray,
    base_loss_grid: np.ndarray,
    u_axis: np.ndarray,
    v_axis: np.ndarray,
    uu: np.ndarray,
    vv: np.ndarray,
    sampler: dict[str, np.ndarray],
) -> tuple[list[dict[str, float | int]], dict[str, np.ndarray], dict[str, np.ndarray]]:
    moving, _ = run_experiment(
        name="gradient_ascent",
        move_points=True,
        base_loss_grid=base_loss_grid,
        density_grid=density_grid,
        u_axis=u_axis,
        v_axis=v_axis,
        uu=uu,
        vv=vv,
        sampler=sampler,
        rng=np.random.default_rng(config.seed),
        config=config,
    )
    baseline, _ = run_experiment(
        name="no_movement_baseline",
        move_points=False,
        base_loss_grid=base_loss_grid,
        density_grid=density_grid,
        u_axis=u_axis,
        v_axis=v_axis,
        uu=uu,
        vv=vv,
        sampler=sampler,
        rng=np.random.default_rng(config.seed),
        config=config,
    )

    rows: list[dict[str, float | int]] = []
    for checkpoint in CHECKPOINTS:
        if checkpoint > config.n_rounds:
            continue
        moving_metrics = metrics_for_grid(moving["loss_grids"][checkpoint])
        baseline_metrics = metrics_for_grid(baseline["loss_grids"][checkpoint])
        spread = endpoint_spread(moving, checkpoint)
        row: dict[str, float | int] = {
            "round": checkpoint,
            "alpha": config.alpha,
            "pgd_epsilon": config.pgd_epsilon,
            "update_bandwidth": config.update_bandwidth,
            "broad_bandwidth": config.broad_bandwidth,
            "moving_global_mean": moving_metrics["global_mean"],
            "baseline_global_mean": baseline_metrics["global_mean"],
            "delta_global_mean": moving_metrics["global_mean"] - baseline_metrics["global_mean"],
            "moving_p95": moving_metrics["p95"],
            "baseline_p95": baseline_metrics["p95"],
            "delta_p95": moving_metrics["p95"] - baseline_metrics["p95"],
            "moving_max": moving_metrics["max"],
            "baseline_max": baseline_metrics["max"],
            "delta_max": moving_metrics["max"] - baseline_metrics["max"],
            **spread,
        }
        rows.append(row)
    return rows, moving, baseline


def write_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_best_curves(
    path: Path,
    moving: dict[str, np.ndarray],
    baseline: dict[str, np.ndarray],
    title: str,
) -> None:
    rounds = np.arange(moving["loss_grids"].shape[0])
    metrics = [
        ("global_mean", "Global mean loss"),
        ("p95", "95th percentile loss"),
        ("max", "Maximum loss"),
    ]
    data = {}
    for name, arrays in [("moving points + PGD", moving), ("no moving", baseline)]:
        grids = arrays["loss_grids"]
        data[(name, "global_mean")] = [float(grid.mean()) for grid in grids]
        data[(name, "p95")] = [float(np.percentile(grid, 95)) for grid in grids]
        data[(name, "max")] = [float(grid.max()) for grid in grids]

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1), dpi=180, constrained_layout=True)
    for ax, (metric, label) in zip(axes, metrics):
        ax.plot(rounds, data[("moving points + PGD", metric)], color="#1f77b4", linewidth=2.1, label="moving points + PGD")
        ax.plot(rounds, data[("no moving", metric)], color="#d62728", linewidth=2.1, label="no moving")
        ax.set_title(label)
        ax.set_xlabel("training round")
        ax.set_ylabel("L")
        ax.grid(True, alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle(title, fontsize=12)
    fig.savefig(path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("figures/loss_landscape/simulation_outputs/parameter_sweep_early_advantage"),
    )
    parser.add_argument("--n-rounds", type=int, default=25)
    parser.add_argument("--n-samples", type=int, default=120)
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument("--grid-size", type=int, default=111)
    parser.add_argument("--sampler-grid-size", type=int, default=351)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    u_axis, v_axis, uu, vv = make_grid(args.grid_size)
    density_grid = density_value(uu, vv).astype(np.float32)
    base_loss_grid = loss_height(uu, vv).astype(np.float32)
    sampler = build_density_sampler(args.sampler_grid_size)

    alpha_values = [0.00625, 0.0125, 0.01875, 0.025]
    epsilon_values = [0.25, 0.35, 0.50, 0.65]
    update_bandwidth_values = [0.24, 0.36, 0.48]

    all_rows: list[dict[str, float | int]] = []
    best_score = float("inf")
    best_config: SimulationConfig | None = None
    best_moving: dict[str, np.ndarray] | None = None
    best_baseline: dict[str, np.ndarray] | None = None

    total = len(alpha_values) * len(epsilon_values) * len(update_bandwidth_values)
    trial = 0
    for alpha in alpha_values:
        for epsilon in epsilon_values:
            for update_bandwidth in update_bandwidth_values:
                trial += 1
                broad_bandwidth = max(0.55, update_bandwidth * 1.9)
                config = SimulationConfig(
                    n_samples=args.n_samples,
                    n_steps=args.n_steps,
                    n_rounds=args.n_rounds,
                    alpha=alpha,
                    grid_size=args.grid_size,
                    sampler_grid_size=args.sampler_grid_size,
                    update_bandwidth=update_bandwidth,
                    broad_bandwidth=broad_bandwidth,
                    pgd_epsilon=epsilon,
                    seed=args.seed,
                    normalize_gradient=True,
                )
                print(
                    f"[{trial}/{total}] alpha={alpha:.5f} eps={epsilon:.2f} "
                    f"update_bw={update_bandwidth:.2f}"
                )
                rows, moving, baseline = run_trial(
                    config,
                    density_grid,
                    base_loss_grid,
                    u_axis,
                    v_axis,
                    uu,
                    vv,
                    sampler,
                )
                all_rows.extend(rows)

                by_round = {int(row["round"]): row for row in rows}
                r5 = by_round[5]
                r10 = by_round[10]
                r20 = by_round[20]
                spread_penalty = 0.0
                if min(float(r20["endpoint_std_x"]), float(r20["endpoint_std_y"])) < 0.65:
                    spread_penalty = 10.0
                score = (
                    1.50 * float(r10["delta_global_mean"])
                    + 1.00 * float(r5["delta_global_mean"])
                    + 0.45 * float(r10["delta_p95"])
                    + 0.20 * float(r20["delta_global_mean"])
                    + spread_penalty
                )
                if score < best_score:
                    best_score = score
                    best_config = config
                    best_moving = moving
                    best_baseline = baseline

    write_csv(args.out_dir / "sweep_results.csv", all_rows)

    checkpoint_rows = [row for row in all_rows if int(row["round"]) == 10]
    top_rows = sorted(
        checkpoint_rows,
        key=lambda row: (
            float(row["delta_global_mean"]),
            float(row["delta_p95"]),
            float(row["delta_max"]),
        ),
    )[:12]
    write_csv(args.out_dir / "top_round10_candidates.csv", top_rows)

    assert best_config is not None and best_moving is not None and best_baseline is not None
    with (args.out_dir / "best_config.txt").open("w", encoding="utf-8") as handle:
        for key, value in asdict(best_config).items():
            handle.write(f"{key}: {value}\n")
        handle.write(f"score: {best_score}\n")

    plot_best_curves(
        args.out_dir / "best_coarse_curves.png",
        best_moving,
        best_baseline,
        (
            f"Best coarse sweep: alpha={best_config.alpha}, eps={best_config.pgd_epsilon}, "
            f"update_bw={best_config.update_bandwidth}"
        ),
    )
    print(f"Saved {args.out_dir / 'sweep_results.csv'}")
    print(f"Saved {args.out_dir / 'top_round10_candidates.csv'}")
    print(f"Saved {args.out_dir / 'best_coarse_curves.png'}")


if __name__ == "__main__":
    main()
