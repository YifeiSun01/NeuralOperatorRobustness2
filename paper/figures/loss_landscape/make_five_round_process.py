"""Create a five-round process panel for PGD movement, loss reduction, and raw loss."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.collections import LineCollection

from simulate_loss_landscape import DOMAIN_MAX, DOMAIN_MIN


def path_ids(n_points: int, max_paths: int) -> np.ndarray:
    if max_paths >= n_points:
        return np.arange(n_points, dtype=int)
    return np.linspace(0, n_points - 1, max_paths, dtype=int)


def add_paths(ax, trajectories: np.ndarray, max_paths: int, stride: int, linewidth: float) -> None:
    ids = path_ids(trajectories.shape[1], max_paths)
    segments = []
    for point_id in ids:
        path = trajectories[::stride, point_id].astype(np.float64)
        if not np.array_equal(path[-1], trajectories[-1, point_id]):
            path = np.vstack([path, trajectories[-1, point_id]])
        for idx in range(len(path) - 1):
            segments.append([path[idx], path[idx + 1]])
    if not segments:
        return
    ax.add_collection(
        LineCollection(
            segments,
            colors=[(0.12, 0.12, 0.12, 0.66)],
            linewidths=linewidth,
            linestyles="solid",
            zorder=3,
        )
    )


def style_axis(ax, show_y: bool, show_x: bool) -> None:
    ax.set_xlim(DOMAIN_MIN, DOMAIN_MAX)
    ax.set_ylim(DOMAIN_MIN, DOMAIN_MAX)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7, length=2.0)
    if show_x:
        ax.set_xlabel("X", fontsize=8)
    else:
        ax.set_xticklabels([])
    if show_y:
        ax.set_ylabel("Y", fontsize=8)
    else:
        ax.set_yticklabels([])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("figures/loss_landscape/simulation_outputs/alpha_0p05_eps_0p50_bw_0p65_drop_0p60/round_05"),
    )
    parser.add_argument("--max-paths", type=int, default=145)
    parser.add_argument("--trajectory-stride", type=int, default=5)
    parser.add_argument("--kde-alpha", type=float, default=0.58)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = np.load(args.out_dir / "gradient_ascent_simulation_data.npz")

    loss_grids = data["loss_grids"].astype(np.float64)
    trajectories = data["trajectories"].astype(np.float64)
    endpoint_kde = data["endpoint_kde"].astype(np.float64)

    n_rounds = min(5, trajectories.shape[0])
    loss_scale_grids = loss_grids[: n_rounds + 1]
    loss_norm = colors.Normalize(vmin=float(loss_scale_grids.min()), vmax=float(loss_scale_grids.max()))
    drop_maps = np.maximum(0.0, loss_grids[:n_rounds] - loss_grids[1 : n_rounds + 1])
    drop_norm = colors.PowerNorm(gamma=0.55, vmin=0.0, vmax=float(drop_maps.max()))

    fig, axes = plt.subplots(
        3,
        n_rounds,
        figsize=(3.05 * n_rounds, 9.15),
        dpi=args.dpi,
        constrained_layout=False,
    )
    plt.subplots_adjust(left=0.060, right=0.975, bottom=0.070, top=0.910, wspace=0.10, hspace=0.25)

    for round_id in range(n_rounds):
        top_ax = axes[0, round_id]
        middle_ax = axes[1, round_id]
        bottom_ax = axes[2, round_id]
        traj = trajectories[round_id]
        endpoints = traj[-1]

        top_ax.imshow(
            loss_grids[round_id],
            origin="lower",
            extent=[DOMAIN_MIN, DOMAIN_MAX, DOMAIN_MIN, DOMAIN_MAX],
            cmap="coolwarm",
            norm=loss_norm,
            interpolation="bicubic",
            aspect="equal",
        )
        add_paths(top_ax, traj, args.max_paths, args.trajectory_stride, linewidth=1.12)
        top_ax.scatter(endpoints[:, 0], endpoints[:, 1], s=7.2, c="black", edgecolors="none", zorder=4)
        top_ax.set_title(f"Round {round_id + 1}: PGD movement", fontsize=9, pad=6)
        style_axis(top_ax, show_y=round_id == 0, show_x=False)

        # Blue indicates the region whose loss is reduced by the training update.
        middle_ax.imshow(
            drop_maps[round_id],
            origin="lower",
            extent=[DOMAIN_MIN, DOMAIN_MAX, DOMAIN_MIN, DOMAIN_MAX],
            cmap="Blues",
            norm=drop_norm,
            interpolation="bicubic",
            aspect="equal",
        )
        axis = np.linspace(DOMAIN_MIN, DOMAIN_MAX, endpoint_kde.shape[1])
        middle_ax.contour(
            axis,
            axis,
            endpoint_kde[round_id],
            levels=[0.35, 0.62],
            colors=["0.20", "0.05"],
            linewidths=[0.55, 0.75],
            alpha=0.42,
        )
        middle_ax.scatter(endpoints[:, 0], endpoints[:, 1], s=7.2, c="black", edgecolors="none", zorder=4)
        middle_ax.set_title("loss decrease after training", fontsize=9, pad=6)
        style_axis(middle_ax, show_y=round_id == 0, show_x=False)

        bottom_ax.imshow(
            loss_grids[round_id],
            origin="lower",
            extent=[DOMAIN_MIN, DOMAIN_MAX, DOMAIN_MIN, DOMAIN_MAX],
            cmap="coolwarm",
            norm=loss_norm,
            interpolation="bicubic",
            aspect="equal",
        )
        bottom_ax.scatter(endpoints[:, 0], endpoints[:, 1], s=7.2, c="black", edgecolors="none", zorder=4)
        bottom_ax.set_title("loss before training", fontsize=9, pad=6)
        style_axis(bottom_ax, show_y=round_id == 0, show_x=True)

    fig.text(0.020, 0.750, "PGD sampling", rotation=90, va="center", fontsize=10)
    fig.text(0.020, 0.485, "Loss decrease", rotation=90, va="center", fontsize=10)
    fig.text(0.020, 0.220, "Loss before training", rotation=90, va="center", fontsize=10)
    fig.suptitle("Five rounds of PGD sampling, local loss reduction, and pre-training loss", fontsize=15, y=0.980)

    output = args.output
    if output is None:
        output = args.out_dir / "five_round_pgd_loss_reduction_and_loss_process.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor="white")
    plt.close(fig)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
