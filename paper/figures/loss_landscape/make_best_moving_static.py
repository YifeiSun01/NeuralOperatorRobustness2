"""Create a static visualization for the best early moving-point run."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from simulate_loss_landscape import DOMAIN_MAX, DOMAIN_MIN, bilinear_sample, kde_on_grid


def normalize_field(field: np.ndarray) -> np.ndarray:
    field = field.astype(np.float64)
    return (field - field.min()) / (field.max() - field.min() + 1e-12)


def make_current_kde(points: np.ndarray, uu: np.ndarray, vv: np.ndarray, bandwidth: float) -> np.ndarray:
    kde = kde_on_grid(points, uu, vv, bandwidth=bandwidth, chunk_size=64)
    return kde / (kde.max() + 1e-12)


def add_vertical_projection_lines(
    ax,
    points: np.ndarray,
    z_values: np.ndarray,
    floor_z: float,
    alpha: float = 0.16,
) -> None:
    segments = [
        [(float(x), float(y), floor_z), (float(x), float(y), float(z))]
        for (x, y), z in zip(points, z_values)
    ]
    ax.add_collection3d(
        Line3DCollection(
            segments,
            colors=[(0.15, 0.15, 0.15, alpha)],
            linewidths=0.34,
            linestyles="dashed",
        )
    )


def trajectory_subset(n_points: int, max_paths: int) -> np.ndarray:
    if max_paths >= n_points:
        return np.arange(n_points, dtype=int)
    return np.linspace(0, n_points - 1, max_paths, dtype=int)


def add_3d_trajectories(
    ax,
    trajectories: np.ndarray,
    loss_grid: np.ndarray,
    u_axis: np.ndarray,
    v_axis: np.ndarray,
    max_paths: int,
    stride: int,
    z_offset: float = 0.0,
) -> None:
    ids = trajectory_subset(trajectories.shape[1], max_paths)
    segments = []
    for point_id in ids:
        path = trajectories[::stride, point_id].astype(np.float64)
        if not np.array_equal(path[-1], trajectories[-1, point_id]):
            path = np.vstack([path, trajectories[-1, point_id]])
        z = bilinear_sample(loss_grid, u_axis, v_axis, path) + z_offset
        path3 = np.column_stack([path[:, 0], path[:, 1], z])
        for idx in range(len(path3) - 1):
            segments.append([path3[idx], path3[idx + 1]])
    if not segments:
        return
    collection = Line3DCollection(
        segments,
        colors=[(0.0, 0.0, 0.0, 1.0)],
        linewidths=0.82,
        linestyles="solid",
        zorder=200,
    )
    collection.set_sort_zpos(1.0e6)
    ax.add_collection3d(collection)


def add_2d_trajectories(ax, trajectories: np.ndarray, max_paths: int, stride: int) -> None:
    ids = trajectory_subset(trajectories.shape[1], max_paths)
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
            colors=[(0.12, 0.12, 0.12, 0.68)],
            linewidths=1.35,
            linestyles="solid",
            zorder=3,
        )
    )


def format_2d_axis(ax, title: str) -> None:
    ax.set_title(title, fontsize=13, pad=7)
    ax.set_xlim(DOMAIN_MIN, DOMAIN_MAX)
    ax.set_ylim(DOMAIN_MIN, DOMAIN_MAX)
    ax.set_aspect("equal")
    ax.set_xlabel("X", fontsize=9)
    ax.set_ylabel("Y", fontsize=9)
    ax.tick_params(labelsize=8, length=2.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("figures/loss_landscape/simulation_outputs/alpha_0p05_eps_0p50_bw_0p65_drop_0p60/round_03"),
    )
    parser.add_argument("--round-index", type=int, default=0)
    parser.add_argument("--scale-rounds", type=int, default=5)
    parser.add_argument("--kde-bandwidth", type=float, default=0.62)
    parser.add_argument("--max-paths", type=int, default=200)
    parser.add_argument("--trajectory-stride", type=int, default=1)
    parser.add_argument("--layout", choices=("full", "3d-only"), default="full")
    parser.add_argument("--surface-alpha", type=float, default=0.88)
    parser.add_argument("--density-alpha", type=float, default=0.94)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = np.load(args.out_dir / "gradient_ascent_simulation_data.npz")

    u_axis = data["u_axis"].astype(np.float64)
    v_axis = data["v_axis"].astype(np.float64)
    uu, vv = np.meshgrid(u_axis, v_axis, indexing="xy")

    round_index = int(args.round_index)
    loss_grids = data["loss_grids"].astype(np.float64)
    loss_grid = loss_grids[round_index]
    next_loss_grid = loss_grids[min(round_index + 1, loss_grids.shape[0] - 1)]
    loss_drop_grid = np.maximum(0.0, loss_grid - next_loss_grid)
    density_grid = normalize_field(data["density_grid"])
    trajectories = data["trajectories"][round_index].astype(np.float64)
    start_points = trajectories[0]
    endpoints = trajectories[-1]
    start_z = bilinear_sample(loss_grid, u_axis, v_axis, start_points)
    endpoint_z = bilinear_sample(loss_grid, u_axis, v_axis, endpoints)
    current_kde = make_current_kde(endpoints, uu, vv, args.kde_bandwidth)

    scale_count = min(max(args.scale_rounds, round_index + 1) + 1, loss_grids.shape[0])
    scale_grids = loss_grids[:scale_count]
    loss_vmin = float(scale_grids.min())
    loss_vmax = float(scale_grids.max())
    drop_scale_count = min(scale_count - 1, loss_grids.shape[0] - 1)
    drop_scale_maps = np.maximum(0.0, loss_grids[:drop_scale_count] - loss_grids[1 : drop_scale_count + 1])
    drop_vmax = float(drop_scale_maps.max() + 1e-12)
    floor_z = float(loss_vmin - 0.82)
    loss_norm = colors.Normalize(vmin=loss_vmin, vmax=loss_vmax)
    drop_norm = colors.PowerNorm(gamma=0.58, vmin=0.0, vmax=drop_vmax)
    dist_norm = colors.Normalize(vmin=0.0, vmax=1.0)
    trajectory_z_offset = 0.0

    is_3d_only = args.layout == "3d-only"
    if is_3d_only:
        fig = plt.figure(figsize=(10.6, 7.85), dpi=args.dpi)
        ax3d = fig.add_axes([-0.085, -0.085, 1.185, 1.160], projection="3d")
    else:
        fig = plt.figure(figsize=(18.2, 8.6), dpi=args.dpi)
        ax3d = fig.add_axes([-0.006, 0.022, 0.620, 0.900], projection="3d")
    ax3d.computed_zorder = False

    if not is_3d_only:
        fig_w, fig_h = fig.get_size_inches()
        side_w = 0.184
        side_h = side_w * fig_w / fig_h
        right_x = 0.606
        bottom_y = 0.067
        gap_w = 0.018
        row_gap = 0.040
        top_y = bottom_y + side_h + row_gap

        ax_loss = fig.add_axes([right_x, top_y, side_w, side_h])
        ax_drop = fig.add_axes([right_x + side_w + gap_w, top_y, side_w, side_h])
        ax_kde = fig.add_axes([right_x, bottom_y, side_w, side_h])
        ax_initial = fig.add_axes([right_x + side_w + gap_w, bottom_y, side_w, side_h])

    dist_facecolors = cm.viridis_r(dist_norm(density_grid))
    dist_facecolors[..., 3] = args.density_alpha

    loss_surface = ax3d.plot_surface(
        uu,
        vv,
        loss_grid,
        cmap="coolwarm",
        norm=loss_norm,
        rcount=260,
        ccount=260,
        linewidth=0.025,
        edgecolor=(1.0, 1.0, 1.0, 0.055),
        antialiased=True,
        alpha=args.surface_alpha,
        shade=True,
        zorder=2,
    )
    if hasattr(loss_surface, "set_sort_zpos"):
        loss_surface.set_sort_zpos(-1.0e6)
    dist_surface = ax3d.plot_surface(
        uu,
        vv,
        np.full_like(uu, floor_z),
        facecolors=dist_facecolors,
        rcount=260,
        ccount=260,
        linewidth=0,
        antialiased=False,
        shade=False,
        zorder=1,
    )
    if hasattr(dist_surface, "set_sort_zpos"):
        dist_surface.set_sort_zpos(-1.0e6)

    add_vertical_projection_lines(ax3d, start_points, start_z + trajectory_z_offset, floor_z)
    add_3d_trajectories(
        ax3d,
        trajectories,
        loss_grid,
        u_axis,
        v_axis,
        max_paths=args.max_paths,
        stride=args.trajectory_stride,
        z_offset=trajectory_z_offset,
    )
    floor_points = ax3d.scatter(
        start_points[:, 0],
        start_points[:, 1],
        np.full(start_points.shape[0], floor_z + 0.075),
        s=18.0,
        c="black",
        edgecolors="none",
        alpha=1.0,
        depthshade=False,
        zorder=210,
    )
    if hasattr(floor_points, "set_sort_zpos"):
        floor_points.set_sort_zpos(1.0e6)
    surface_points = ax3d.scatter(
        endpoints[:, 0],
        endpoints[:, 1],
        endpoint_z + trajectory_z_offset,
        s=18.0,
        c="black",
        edgecolors="none",
        alpha=1.0,
        depthshade=False,
        zorder=220,
    )
    if hasattr(surface_points, "set_sort_zpos"):
        surface_points.set_sort_zpos(1.0e6)
    ax3d.set_xlim(DOMAIN_MIN, DOMAIN_MAX)
    ax3d.set_ylim(DOMAIN_MIN, DOMAIN_MAX)
    ax3d.set_zlim(floor_z, float(loss_vmax + 0.90))
    ax3d.set_xlabel("X", labelpad=7)
    ax3d.set_ylabel("Y", labelpad=7)
    ax3d.set_zlabel("Z", labelpad=7)
    ax3d.view_init(elev=25, azim=-55)
    ax3d.set_proj_type("ortho")
    ax3d.grid(False)
    try:
        ax3d.set_box_aspect((1.0, 1.0, 0.72), zoom=1.24 if is_3d_only else 1.12)
    except TypeError:
        ax3d.set_box_aspect((1.0, 1.0, 0.72))
    ax3d.annotate(
        r"loss landscape $\mathcal{L}(x)$",
        xy=(0.42, 0.66),
        xycoords="axes fraction",
        xytext=(0.000, 0.805),
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color="0.15", lw=1.20),
        ha="left",
        va="center",
        fontsize=15,
        annotation_clip=False,
    )
    ax3d.annotate(
        r"data distribution $\mathcal{P}(x)$",
        xy=(0.34, 0.24),
        xycoords="axes fraction",
        xytext=(0.000, 0.072),
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color="0.15", lw=1.20),
        ha="left",
        va="center",
        fontsize=15,
        annotation_clip=False,
    )

    if not is_3d_only:
        ax_loss.imshow(
            loss_grid,
            origin="lower",
            extent=[DOMAIN_MIN, DOMAIN_MAX, DOMAIN_MIN, DOMAIN_MAX],
            cmap="coolwarm",
            norm=loss_norm,
            interpolation="bicubic",
            aspect="equal",
        )
        add_2d_trajectories(ax_loss, trajectories, args.max_paths, args.trajectory_stride)
        ax_loss.scatter(endpoints[:, 0], endpoints[:, 1], s=9, c="black", edgecolors="none", zorder=4)
        format_2d_axis(ax_loss, r"loss landscape $\mathcal{L}(x)$")
        ax_loss.set_xlabel("")
        ax_loss.tick_params(labelbottom=False)

        ax_drop.imshow(
            loss_drop_grid,
            origin="lower",
            extent=[DOMAIN_MIN, DOMAIN_MAX, DOMAIN_MIN, DOMAIN_MAX],
            cmap="Blues",
            norm=drop_norm,
            interpolation="bicubic",
            aspect="equal",
        )
        ax_drop.scatter(endpoints[:, 0], endpoints[:, 1], s=9, c="black", edgecolors="none", zorder=4)
        format_2d_axis(ax_drop, "loss decrease after training")
        ax_drop.set_xlabel("")
        ax_drop.tick_params(labelbottom=False)

        ax_kde.imshow(
            current_kde,
            origin="lower",
            extent=[DOMAIN_MIN, DOMAIN_MAX, DOMAIN_MIN, DOMAIN_MAX],
            cmap="viridis_r",
            norm=dist_norm,
            interpolation="bicubic",
            aspect="equal",
        )
        add_2d_trajectories(ax_kde, trajectories, args.max_paths, args.trajectory_stride)
        ax_kde.scatter(endpoints[:, 0], endpoints[:, 1], s=9, c="black", edgecolors="none", zorder=4)
        format_2d_axis(ax_kde, "current sample KDE")

        ax_initial.imshow(
            density_grid,
            origin="lower",
            extent=[DOMAIN_MIN, DOMAIN_MAX, DOMAIN_MIN, DOMAIN_MAX],
            cmap="viridis_r",
            norm=dist_norm,
            interpolation="bicubic",
            aspect="equal",
        )
        ax_initial.scatter(start_points[:, 0], start_points[:, 1], s=9, c="black", edgecolors="none", zorder=4)
        format_2d_axis(ax_initial, r"initial data distribution $\mathcal{P}(x)$")

        fig.suptitle(
            "sampled data move towards high loss region during using gradients",
            fontsize=18,
            y=0.965,
        )

    output = args.output
    if output is None:
        suffix = "3d_only" if is_3d_only else "trajectories"
        output = args.out_dir / f"best_moving_static_{suffix}_round_{round_index + 1:02d}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {"facecolor": "white"}
    if is_3d_only:
        save_kwargs.update({"bbox_inches": "tight", "pad_inches": 0.02})
    fig.savefig(output, **save_kwargs)
    plt.close(fig)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
