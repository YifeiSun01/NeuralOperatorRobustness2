r"""
Build process-oriented GIFs for the loss-landscape simulations.

Unlike make_simulation_gifs.py, these animations explicitly separate:

    1. sample: draw points from the fixed data distribution P(x)
    2. attack/no attack: move points by gradient ascent, or keep them fixed
    3. train: update the loss landscape using the endpoint density

The loss landscape shown during the train phase is interpolated from the
pre-training grid to the post-training grid so the surface change is visible.
"""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib import cm, colors
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from PIL import Image

from make_simulation_gifs import EXPERIMENT_FILES, load_experiments, save_gif
from simulate_loss_landscape import DOMAIN_MAX, DOMAIN_MIN, bilinear_sample, kde_on_grid


FrameSpec = Tuple[int, int, str, float]


PHASE_LABELS = {
    "sample": "1. sample points from the data distribution P(x)",
    "attack": "2. attack: move sampled points by PGD ascent on L(x)",
    "no_attack": "2. no attack: keep sampled points fixed",
    "train": "3. train: dense endpoint regions lower L(x) by 20%-30%; nearby sparse regions change by 5%-10%",
}


def build_process_schedule(
    n_rounds: int,
    n_steps: int,
    attack_frames: int,
    train_frames: int,
    round_stride: int,
) -> List[FrameSpec]:
    attack_steps = np.linspace(0, n_steps, attack_frames, dtype=int)
    attack_steps = np.unique(attack_steps)
    train_lambdas = np.linspace(0.0, 1.0, train_frames + 1)[1:]

    schedule: List[FrameSpec] = []
    round_ids = list(range(0, n_rounds, max(1, round_stride)))
    if round_ids[-1] != n_rounds - 1:
        round_ids.append(n_rounds - 1)
    for round_id in round_ids:
        schedule.append((round_id, 0, "sample", 0.0))
        for step_id in attack_steps[1:]:
            schedule.append((round_id, int(step_id), "attack", 0.0))
        for lam in train_lambdas:
            schedule.append((round_id, n_steps, "train", float(lam)))
    return schedule


def display_loss_grid(arrays: Dict[str, np.ndarray], round_id: int, phase: str, lam: float) -> np.ndarray:
    before = arrays["loss_grids"][round_id].astype(np.float64)
    if phase != "train":
        return before
    after = arrays["loss_grids"][round_id + 1].astype(np.float64)
    return (1.0 - lam) * before + lam * after


def current_points(arrays: Dict[str, np.ndarray], round_id: int, step_id: int, phase: str) -> np.ndarray:
    trajectories = arrays["trajectories"][round_id].astype(np.float64)
    if phase == "train":
        return trajectories[-1]
    return trajectories[step_id]


def add_vertical_segments(ax, points: np.ndarray, z_values: np.ndarray, floor_z: float) -> None:
    segments = [
        [(float(u), float(v), floor_z), (float(u), float(v), float(z))]
        for (u, v), z in zip(points, z_values)
    ]
    ax.add_collection3d(
        Line3DCollection(
            segments,
            colors=[(0.18, 0.18, 0.18, 0.36)],
            linewidths=0.62,
            linestyles="dashed",
        )
    )


def path_ids(n_points: int, max_lines: int) -> np.ndarray:
    if max_lines >= n_points:
        return np.arange(n_points, dtype=int)
    return np.linspace(0, n_points - 1, max_lines, dtype=int)


def path_alpha_ramp(n_segments: int, low: float = 0.18, high: float = 1.0) -> np.ndarray:
    if n_segments <= 1:
        return np.array([high], dtype=np.float64)
    return np.linspace(low, high, n_segments, dtype=np.float64)


def add_attack_paths(
    ax,
    arrays: Dict[str, np.ndarray],
    round_id: int,
    step_id: int,
    loss_grid: np.ndarray,
    max_lines: int,
) -> None:
    if step_id <= 0:
        return
    trajectories = arrays["trajectories"][round_id].astype(np.float64)
    u_axis = arrays["u_axis"]
    v_axis = arrays["v_axis"]
    ids = path_ids(trajectories.shape[1], max_lines)
    stride = max(1, step_id // 58)
    segments = []
    halo_colors = []
    path_colors = []
    for point_id in ids:
        path = trajectories[: step_id + 1 : stride, point_id]
        if not np.array_equal(path[-1], trajectories[step_id, point_id]):
            path = np.vstack([path, trajectories[step_id, point_id]])
        if len(path) < 2:
            continue
        path_z = bilinear_sample(loss_grid, u_axis, v_axis, path) + 0.035
        path3 = np.column_stack([path[:, 0], path[:, 1], path_z])
        alphas = path_alpha_ramp(len(path3) - 1)
        for idx in range(len(path3) - 1):
            segments.append([path3[idx], path3[idx + 1]])
            halo_colors.append((1.0, 1.0, 1.0, min(0.86, 0.36 + 0.50 * alphas[idx])))
            path_colors.append(colors.to_rgba("#4E4E4E", alphas[idx]))
    if not segments:
        return
    ax.add_collection3d(Line3DCollection(segments, colors=halo_colors, linewidths=2.15))
    ax.add_collection3d(Line3DCollection(segments, colors=path_colors, linewidths=1.32))


def add_3d_scene(
    ax,
    arrays: Dict[str, np.ndarray],
    experiment: str,
    round_id: int,
    step_id: int,
    phase: str,
    lam: float,
    z_min: float,
    z_max: float,
    density_norm: colors.Normalize,
    loss_norm: colors.Normalize,
    surface_stride: int,
    max_path_lines: int,
) -> None:
    u_axis = arrays["u_axis"]
    v_axis = arrays["v_axis"]
    uu, vv = np.meshgrid(u_axis, v_axis, indexing="xy")
    loss_grid = display_loss_grid(arrays, round_id, phase, lam)
    density_grid = arrays["density_grid"]
    pts = current_points(arrays, round_id, step_id, phase)
    start_pts = arrays["trajectories"][round_id, 0].astype(np.float64)
    z_pts = bilinear_sample(loss_grid, u_axis, v_axis, pts)

    sl = slice(None, None, surface_stride)
    density_colors = cm.viridis_r(density_norm(density_grid[sl, sl]))
    density_colors[..., 3] = 0.73

    ax.plot_surface(
        uu[sl, sl],
        vv[sl, sl],
        loss_grid[sl, sl],
        cmap="coolwarm",
        norm=loss_norm,
        linewidth=0,
        antialiased=True,
        alpha=0.76,
    )
    ax.plot_surface(
        uu[sl, sl],
        vv[sl, sl],
        np.full_like(uu[sl, sl], z_min),
        facecolors=density_colors,
        linewidth=0,
        antialiased=False,
        shade=False,
    )

    add_vertical_segments(ax, pts, z_pts, z_min)
    if experiment == "gradient_ascent" and phase in {"attack", "train"}:
        add_attack_paths(
            ax,
            arrays,
            round_id,
            step_id if phase == "attack" else arrays["trajectories"].shape[1] - 1,
            arrays["loss_grids"][round_id],
            max_path_lines,
        )

    ax.scatter(
        pts[:, 0],
        pts[:, 1],
        np.full(pts.shape[0], z_min),
        s=25,
        c="#050505",
        edgecolors="white",
        linewidths=0.28,
        alpha=0.95,
        depthshade=False,
    )
    ax.scatter(
        pts[:, 0],
        pts[:, 1],
        z_pts,
        s=28,
        c="#050505",
        edgecolors="white",
        linewidths=0.36,
        alpha=1.0,
        depthshade=False,
    )

    if phase == "train":
        endpoint_kde = arrays["endpoint_kde"][round_id]
        high = endpoint_kde > 0.68
        if np.any(high):
            high_pts = np.column_stack([uu[high], vv[high]])
            keep = np.linspace(0, high_pts.shape[0] - 1, min(160, high_pts.shape[0]), dtype=int)
            high_pts = high_pts[keep]
            high_z = bilinear_sample(loss_grid, u_axis, v_axis, high_pts)
            ax.scatter(
                high_pts[:, 0],
                high_pts[:, 1],
                high_z + 0.018,
                s=3,
                c="#111111",
                alpha=0.18,
                depthshade=False,
            )

    ax.set_xlim(DOMAIN_MIN, DOMAIN_MAX)
    ax.set_ylim(DOMAIN_MIN, DOMAIN_MAX)
    ax.set_zlim(z_min, z_max)
    ax.set_xlabel("X", labelpad=6)
    ax.set_ylabel("Y", labelpad=6)
    ax.set_zlabel("Z", labelpad=6)
    ax.view_init(elev=25, azim=-55)
    ax.grid(False)
    ax.set_box_aspect((1.0, 1.0, 0.72))


def add_fading_2d_paths(
    ax,
    trajectories: np.ndarray | None,
    trajectory_step: int,
    max_path_lines: int,
    color: str = "#595959",
) -> None:
    if trajectories is None or trajectory_step <= 0:
        return
    ids = path_ids(trajectories.shape[1], max_path_lines)
    stride = max(1, trajectory_step // 70)
    segments = []
    halo_colors = []
    path_colors = []
    for point_id in ids:
        path = trajectories[: trajectory_step + 1 : stride, point_id]
        if not np.array_equal(path[-1], trajectories[trajectory_step, point_id]):
            path = np.vstack([path, trajectories[trajectory_step, point_id]])
        if len(path) < 2:
            continue
        alphas = path_alpha_ramp(len(path) - 1, low=0.24, high=1.0)
        for idx in range(len(path) - 1):
            segments.append([path[idx], path[idx + 1]])
            halo_colors.append((1.0, 1.0, 1.0, min(0.88, 0.36 + 0.52 * alphas[idx])))
            path_colors.append(colors.to_rgba(color, alphas[idx]))
    if not segments:
        return
    ax.add_collection(LineCollection(segments, colors=halo_colors, linewidths=3.20, zorder=3))
    ax.add_collection(LineCollection(segments, colors=path_colors, linewidths=1.72, zorder=4))


def current_kde(points: np.ndarray, grid_size: int, bandwidth: float) -> np.ndarray:
    axis = np.linspace(DOMAIN_MIN, DOMAIN_MAX, grid_size, dtype=np.float64)
    uu, vv = np.meshgrid(axis, axis, indexing="xy")
    field = kde_on_grid(points, uu, vv, bandwidth=bandwidth, chunk_size=64)
    return field / (field.max() + 1e-12)


def add_heatmap(
    ax,
    field: np.ndarray,
    points: np.ndarray,
    start_points: np.ndarray,
    title: str,
    cmap: str,
    norm: colors.Normalize,
    point_color: str,
    endpoint_kde: np.ndarray | None = None,
    trajectories: np.ndarray | None = None,
    trajectory_step: int = 0,
    max_path_lines: int = 80,
    show_start_points: bool = False,
) -> None:
    ax.imshow(
        field,
        origin="lower",
        extent=[DOMAIN_MIN, DOMAIN_MAX, DOMAIN_MIN, DOMAIN_MAX],
        cmap=cmap,
        norm=norm,
        interpolation="bicubic",
        aspect="equal",
    )
    if endpoint_kde is not None:
        axis = np.linspace(DOMAIN_MIN, DOMAIN_MAX, endpoint_kde.shape[0])
        ax.contour(
            axis,
            axis,
            endpoint_kde,
            levels=[0.35, 0.60, 0.82],
            colors=["#222222"],
            linewidths=[0.45, 0.60, 0.75],
            alpha=0.34,
        )
    add_fading_2d_paths(ax, trajectories, trajectory_step, max_path_lines)
    if show_start_points:
        ax.scatter(
            start_points[:, 0],
            start_points[:, 1],
            s=20,
            facecolors="none",
            edgecolors="0.05",
            linewidths=0.78,
            alpha=0.72,
            zorder=5,
        )
    ax.scatter(
        points[:, 0],
        points[:, 1],
        s=26,
        c=point_color,
        alpha=0.98,
        edgecolors="white",
        linewidths=0.30,
        zorder=6,
    )
    ax.set_title(title, fontsize=10, pad=5)
    ax.set_xlim(DOMAIN_MIN, DOMAIN_MAX)
    ax.set_ylim(DOMAIN_MIN, DOMAIN_MAX)
    ax.set_xlabel("X", fontsize=9)
    ax.set_ylabel("Y", fontsize=9)
    ax.tick_params(labelsize=8, length=2.5)


def add_initial_distribution_inset(
    ax,
    density_grid: np.ndarray,
    start_points: np.ndarray,
    density_norm: colors.Normalize,
) -> None:
    inset = ax.inset_axes([0.60, 0.58, 0.36, 0.36])
    inset.imshow(
        density_grid,
        origin="lower",
        extent=[DOMAIN_MIN, DOMAIN_MAX, DOMAIN_MIN, DOMAIN_MAX],
        cmap="viridis_r",
        norm=density_norm,
        interpolation="bicubic",
        aspect="equal",
    )
    inset.scatter(
        start_points[:, 0],
        start_points[:, 1],
        s=5,
        c="#050505",
        alpha=0.72,
        linewidths=0,
        zorder=3,
    )
    inset.set_title(r"initial $P(x)$", fontsize=7, pad=1)
    inset.set_xlim(DOMAIN_MIN, DOMAIN_MAX)
    inset.set_ylim(DOMAIN_MIN, DOMAIN_MAX)
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_linewidth(0.55)
        spine.set_edgecolor("0.25")


def render_frame(
    arrays: Dict[str, np.ndarray],
    experiment: str,
    round_id: int,
    step_id: int,
    phase: str,
    lam: float,
    z_min: float,
    z_max: float,
    density_norm: colors.Normalize,
    loss_norm: colors.Normalize,
    args: argparse.Namespace,
) -> Image.Image:
    actual_phase = "no_attack" if experiment == "no_movement_baseline" and phase == "attack" else phase
    fig = plt.figure(figsize=(13.2, 8.65), dpi=args.dpi)
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[2.42, 1.0],
        height_ratios=[1.0, 1.0],
        left=0.03,
        right=0.985,
        bottom=0.055,
        top=0.875,
        wspace=0.12,
        hspace=0.30,
    )
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax_loss = fig.add_subplot(gs[0, 1])
    ax_current = fig.add_subplot(gs[1, 1])

    add_3d_scene(
        ax3d,
        arrays,
        experiment,
        round_id,
        step_id,
        phase,
        lam,
        z_min,
        z_max,
        density_norm,
        loss_norm,
        args.surface_stride,
        args.max_path_lines,
    )

    loss_grid = display_loss_grid(arrays, round_id, phase, lam)
    pts = current_points(arrays, round_id, step_id, phase)
    start_pts = arrays["trajectories"][round_id, 0].astype(np.float64)
    kde = arrays["endpoint_kde"][round_id] if phase == "train" else None
    trajectory_step = step_id if phase == "attack" else arrays["trajectories"].shape[1] - 1
    path_trajectories = (
        arrays["trajectories"][round_id].astype(np.float64)
        if experiment == "gradient_ascent" and phase in {"attack", "train"}
        else None
    )

    add_heatmap(
        ax_loss,
        loss_grid,
        pts,
        start_pts,
        r"loss landscape $\mathcal{L}(x)$",
        "coolwarm",
        loss_norm,
        "#050505",
        endpoint_kde=kde,
        trajectories=path_trajectories,
        trajectory_step=trajectory_step,
        max_path_lines=args.max_path_lines,
    )
    kde_field = current_kde(pts, args.kde_grid_size, args.kde_bandwidth)
    add_heatmap(
        ax_current,
        kde_field,
        pts,
        start_pts,
        "current sample distribution",
        "viridis_r",
        colors.Normalize(vmin=0.0, vmax=1.0),
        "#050505",
        trajectories=path_trajectories,
        trajectory_step=trajectory_step,
        max_path_lines=args.max_path_lines,
        show_start_points=False,
    )
    add_initial_distribution_inset(ax_current, arrays["density_grid"], start_pts, density_norm)

    headline = "attack training process" if experiment == "gradient_ascent" else "no-attack training process"
    total_rounds = arrays["samples"].shape[0]
    fig.text(0.5, 0.965, f"{headline}: round {round_id + 1}/{total_rounds}", ha="center", va="top", fontsize=15)
    fig.text(0.5, 0.925, PHASE_LABELS[actual_phase], ha="center", va="top", fontsize=11, color="#333333")
    if phase == "attack":
        fig.text(0.5, 0.895, f"attack step {step_id}/100", ha="center", va="top", fontsize=10, color="#555555")
    elif phase == "train":
        fig.text(0.5, 0.895, f"training update progress {int(round(100 * lam))}%", ha="center", va="top", fontsize=10, color="#555555")

    buffer = BytesIO()
    fig.savefig(buffer, format="png", facecolor="white")
    plt.close(fig)
    buffer.seek(0)
    image = Image.open(buffer).convert("P", palette=Image.Palette.ADAPTIVE, colors=192)
    return image.copy()


def make_process_gif(
    experiment: str,
    arrays: Dict[str, np.ndarray],
    out_path: Path,
    z_min: float,
    z_max: float,
    density_norm: colors.Normalize,
    loss_norm: colors.Normalize,
    args: argparse.Namespace,
) -> None:
    n_rounds = arrays["samples"].shape[0]
    n_steps = arrays["trajectories"].shape[1] - 1
    schedule = build_process_schedule(
        n_rounds,
        n_steps,
        args.attack_frames,
        args.train_frames,
        args.round_stride,
    )
    frames = []
    for index, (round_id, step_id, phase, lam) in enumerate(schedule, start=1):
        shown_phase = "no_attack" if experiment == "no_movement_baseline" and phase == "attack" else phase
        print(
            f"{experiment}: frame {index}/{len(schedule)} "
            f"round={round_id + 1} phase={shown_phase} step={step_id} train={lam:.2f}"
        )
        frames.append(
            render_frame(
                arrays,
                experiment,
                round_id,
                step_id,
                phase,
                lam,
                z_min,
                z_max,
                density_norm,
                loss_norm,
                args,
            )
        )
    save_gif(frames, out_path, args.duration_ms)
    print(f"Saved {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("figures/loss_landscape/simulation_outputs"),
    )
    parser.add_argument("--attack-frames", type=int, default=7)
    parser.add_argument("--train-frames", type=int, default=7)
    parser.add_argument("--round-stride", type=int, default=1)
    parser.add_argument("--duration-ms", type=int, default=210)
    parser.add_argument("--surface-stride", type=int, default=4)
    parser.add_argument("--max-path-lines", type=int, default=200)
    parser.add_argument("--kde-grid-size", type=int, default=161)
    parser.add_argument("--kde-bandwidth", type=float, default=0.42)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=sorted(EXPERIMENT_FILES),
        default=["gradient_ascent", "no_movement_baseline"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiments = load_experiments(args.out_dir)
    z_min = min(float(arrays["loss_grids"].min()) for arrays in experiments.values()) - 0.75
    z_max = max(float(arrays["loss_grids"].max()) for arrays in experiments.values()) + 0.30
    density_grid = next(iter(experiments.values()))["density_grid"]
    density_norm = colors.Normalize(vmin=float(density_grid.min()), vmax=float(density_grid.max()))
    loss_norm = colors.Normalize(
        vmin=min(float(arrays["loss_grids"].min()) for arrays in experiments.values()),
        vmax=max(float(arrays["loss_grids"].max()) for arrays in experiments.values()),
    )

    for experiment in args.experiments:
        suffix = "attack_process" if experiment == "gradient_ascent" else "no_attack_process"
        make_process_gif(
            experiment,
            experiments[experiment],
            args.out_dir / f"{suffix}.gif",
            z_min,
            z_max,
            density_norm,
            loss_norm,
            args,
        )


if __name__ == "__main__":
    main()
