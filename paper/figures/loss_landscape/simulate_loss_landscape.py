r"""
Simulate sampled data moving toward high-loss regions.

This script uses the same scalar fields as the current TikZ figure:

    L(u, v): loss landscape, copied from \lossheight in loss_landscape.tex
    P(u, v): data distribution, copied from \densityvalue in loss_landscape.tex

It records two experiments:

    1. gradient_ascent:
       samples are drawn from P(u, v), then projected-gradient-ascent moves
       them on the current loss landscape for 100 small steps.

    2. no_movement_baseline:
       samples are drawn from P(u, v), but the samples do not move before the
       training-style loss-landscape update.

For each experiment and each round, the script records samples, trajectories,
loss values along trajectories, KDE fields used for updates, update
multipliers, and the full loss landscape before/after each round.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "This script requires NumPy. Install it with `python -m pip install numpy`."
    ) from exc


DOMAIN_MIN = -2.35
DOMAIN_MAX = 2.35
EPS = 1e-12


@dataclass(frozen=True)
class SimulationConfig:
    n_samples: int = 200
    n_steps: int = 100
    n_rounds: int = 5
    alpha: float = 0.0125
    grid_size: int = 321
    sampler_grid_size: int = 701
    update_bandwidth: float = 0.24
    broad_bandwidth: float = 0.55
    drop_min: float = 0.20
    drop_max: float = 0.30
    rise_min: float = 0.05
    rise_max: float = 0.10
    pgd_epsilon: float = 0.50
    seed: int = 7
    normalize_gradient: bool = True


def loss_height(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    r"""Loss landscape copied from the current LaTeX \lossheight macro."""
    return (
        1.82
        + 0.075 * (u**2 + 0.88 * v**2)
        - 0.120 * u * v
        + 0.135 * u
        - 0.055 * v
        + 1.95 * np.exp(-((u + 2.10) ** 2) / 0.28 - ((v - 1.70) ** 2) / 0.42)
        + 0.78 * np.exp(-((u - 1.76) ** 2) / 0.62 - ((v + 1.62) ** 2) / 0.52)
        + 1.05 * np.exp(-((u + 0.35) ** 2) / 0.36 - ((v + 2.05) ** 2) / 0.30)
        + 1.34 * np.exp(-((u - 0.22) ** 2) / 0.30 - ((v - 1.18) ** 2) / 0.34)
        + 0.42 * np.exp(-((u - 0.90) ** 2) / 1.30 - ((v - 0.40) ** 2) / 0.90)
        - 1.12 * np.exp(-((u - 1.24) ** 2) / 0.52 - ((v - 1.34) ** 2) / 0.48)
        - 1.04 * np.exp(-((u + 0.34) ** 2) / 0.48 - ((v + 0.20) ** 2) / 0.70)
        - 0.70 * np.exp(-((u - 1.90) ** 2) / 0.34 - ((v + 0.18) ** 2) / 0.34)
        + 0.22 * np.exp(-((u - 2.15) ** 2) / 0.55 - ((v - 2.05) ** 2) / 0.48)
    )


def density_value(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    r"""Data distribution copied from the current LaTeX \densityvalue macro."""
    return (
        0.22 * np.exp(-0.5 * (((u + 1.75) ** 2) / (0.50**2) + ((v - 0.15) ** 2) / (0.46**2)))
        + 0.19 * np.exp(-0.5 * (((u + 0.70) ** 2) / (0.58**2) + ((v + 1.55) ** 2) / (0.44**2)))
        + 0.21 * np.exp(-0.5 * (((u - 0.75) ** 2) / (0.56**2) + ((v - 1.45) ** 2) / (0.52**2)))
        + 0.18 * np.exp(-0.5 * (((u - 1.70) ** 2) / (0.48**2) + ((v + 0.55) ** 2) / (0.60**2)))
        + 0.20 * np.exp(-0.5 * (((u + 0.10) ** 2) / (0.72**2) + ((v - 0.65) ** 2) / (0.50**2)))
    )


def make_grid(n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    u_axis = np.linspace(DOMAIN_MIN, DOMAIN_MAX, n, dtype=np.float64)
    v_axis = np.linspace(DOMAIN_MIN, DOMAIN_MAX, n, dtype=np.float64)
    uu, vv = np.meshgrid(u_axis, v_axis, indexing="xy")
    return u_axis, v_axis, uu, vv


def build_density_sampler(grid_size: int) -> Dict[str, np.ndarray]:
    u_axis, v_axis, uu, vv = make_grid(grid_size)
    weights = density_value(uu, vv).astype(np.float64)
    weights = np.maximum(weights, 0.0)
    weights /= weights.sum()
    return {
        "u_axis": u_axis,
        "v_axis": v_axis,
        "weights": weights.ravel(),
        "cell_width": np.float64((DOMAIN_MAX - DOMAIN_MIN) / (grid_size - 1)),
        "shape": np.array(weights.shape, dtype=np.int64),
    }


def sample_from_density(
    rng: np.random.Generator,
    sampler: Dict[str, np.ndarray],
    n_samples: int,
) -> np.ndarray:
    """Sample from the gridded version of P(u, v), with within-cell jitter."""
    shape = tuple(int(x) for x in sampler["shape"])
    flat_idx = rng.choice(sampler["weights"].size, size=n_samples, replace=True, p=sampler["weights"])
    v_idx, u_idx = np.unravel_index(flat_idx, shape)
    cell = float(sampler["cell_width"])
    u = sampler["u_axis"][u_idx] + rng.uniform(-0.5 * cell, 0.5 * cell, size=n_samples)
    v = sampler["v_axis"][v_idx] + rng.uniform(-0.5 * cell, 0.5 * cell, size=n_samples)
    pts = np.column_stack([u, v])
    return np.clip(pts, DOMAIN_MIN, DOMAIN_MAX).astype(np.float64)


def bilinear_sample(
    field: np.ndarray,
    u_axis: np.ndarray,
    v_axis: np.ndarray,
    pts: np.ndarray,
) -> np.ndarray:
    """Bilinear interpolation on a field indexed as field[v_index, u_index]."""
    du = u_axis[1] - u_axis[0]
    dv = v_axis[1] - v_axis[0]
    fu = (pts[:, 0] - u_axis[0]) / du
    fv = (pts[:, 1] - v_axis[0]) / dv

    i0 = np.floor(fu).astype(np.int64)
    j0 = np.floor(fv).astype(np.int64)
    i0 = np.clip(i0, 0, len(u_axis) - 2)
    j0 = np.clip(j0, 0, len(v_axis) - 2)
    i1 = i0 + 1
    j1 = j0 + 1

    su = np.clip(fu - i0, 0.0, 1.0)
    sv = np.clip(fv - j0, 0.0, 1.0)

    f00 = field[j0, i0]
    f10 = field[j0, i1]
    f01 = field[j1, i0]
    f11 = field[j1, i1]

    return (
        (1 - su) * (1 - sv) * f00
        + su * (1 - sv) * f10
        + (1 - su) * sv * f01
        + su * sv * f11
    )


def loss_gradients(
    loss_grid: np.ndarray,
    u_axis: np.ndarray,
    v_axis: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    dloss_dv, dloss_du = np.gradient(loss_grid, v_axis, u_axis, edge_order=2)
    return dloss_du, dloss_dv


def projected_gradient_ascent(
    start_pts: np.ndarray,
    loss_grid: np.ndarray,
    u_axis: np.ndarray,
    v_axis: np.ndarray,
    config: SimulationConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Move points toward higher loss and record every step."""
    dloss_du, dloss_dv = loss_gradients(loss_grid, u_axis, v_axis)
    origin = start_pts.astype(np.float64).copy()
    pts = start_pts.astype(np.float64).copy()
    n_samples = pts.shape[0]
    trajectories = np.empty((config.n_steps + 1, n_samples, 2), dtype=np.float32)
    trajectory_loss = np.empty((config.n_steps + 1, n_samples), dtype=np.float32)

    trajectories[0] = pts.astype(np.float32)
    trajectory_loss[0] = bilinear_sample(loss_grid, u_axis, v_axis, pts).astype(np.float32)

    for step in range(1, config.n_steps + 1):
        grad_u = bilinear_sample(dloss_du, u_axis, v_axis, pts)
        grad_v = bilinear_sample(dloss_dv, u_axis, v_axis, pts)
        grad = np.column_stack([grad_u, grad_v])

        if config.normalize_gradient:
            grad_norm = np.linalg.norm(grad, axis=1, keepdims=True)
            grad = grad / np.maximum(grad_norm, EPS)

        pts = pts + config.alpha * grad
        if config.pgd_epsilon > 0:
            delta = pts - origin
            delta_norm = np.linalg.norm(delta, axis=1, keepdims=True)
            scale = np.minimum(1.0, config.pgd_epsilon / np.maximum(delta_norm, EPS))
            pts = origin + delta * scale
        pts = np.clip(pts, DOMAIN_MIN, DOMAIN_MAX)
        trajectories[step] = pts.astype(np.float32)
        trajectory_loss[step] = bilinear_sample(loss_grid, u_axis, v_axis, pts).astype(np.float32)

    return trajectories, trajectory_loss


def stationary_trajectory(
    pts: np.ndarray,
    loss_grid: np.ndarray,
    u_axis: np.ndarray,
    v_axis: np.ndarray,
    config: SimulationConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    repeated = np.repeat(pts[None, :, :], config.n_steps + 1, axis=0).astype(np.float32)
    loss_vals = bilinear_sample(loss_grid, u_axis, v_axis, pts).astype(np.float32)
    repeated_loss = np.repeat(loss_vals[None, :], config.n_steps + 1, axis=0)
    return repeated, repeated_loss


def kde_on_grid(
    points: np.ndarray,
    uu: np.ndarray,
    vv: np.ndarray,
    bandwidth: float,
    chunk_size: int = 48,
) -> np.ndarray:
    """Gaussian KDE evaluated on the whole grid. Normalization is not needed."""
    flat_u = uu.ravel()
    flat_v = vv.ravel()
    out = np.zeros(flat_u.shape, dtype=np.float64)

    inv_bw2 = 1.0 / (bandwidth * bandwidth)
    for start in range(0, len(points), chunk_size):
        chunk = points[start : start + chunk_size]
        du = flat_u[:, None] - chunk[None, :, 0]
        dv = flat_v[:, None] - chunk[None, :, 1]
        out += np.exp(-0.5 * (du * du + dv * dv) * inv_bw2).sum(axis=1)

    return out.reshape(uu.shape)


def update_loss_landscape(
    loss_grid: np.ndarray,
    train_points: np.ndarray,
    uu: np.ndarray,
    vv: np.ndarray,
    rng: np.random.Generator,
    config: SimulationConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Training-style update.

    Dense endpoint regions decrease loss by 20-30 percent. Nearby but sparse
    regions increase by 5-10 percent. This encodes the latest verbal rule while
    keeping the update smooth and reproducible.
    """
    kde = kde_on_grid(train_points, uu, vv, config.update_bandwidth)
    broad = kde_on_grid(train_points, uu, vv, config.broad_bandwidth)

    kde_norm = kde / (kde.max() + EPS)
    broad_norm = broad / (broad.max() + EPS)

    drop_rate = float(rng.uniform(config.drop_min, config.drop_max))
    rise_rate = float(rng.uniform(config.rise_min, config.rise_max))

    dense_drop = drop_rate * kde_norm
    sparse_nearby_rise = rise_rate * broad_norm * (1.0 - kde_norm)
    multiplier = 1.0 - dense_drop + sparse_nearby_rise

    updated = np.maximum(0.02, loss_grid * multiplier)
    return (
        updated.astype(np.float32),
        kde_norm.astype(np.float32),
        multiplier.astype(np.float32),
        drop_rate,
        rise_rate,
    )


def path_lengths(trajectories: np.ndarray) -> np.ndarray:
    diffs = np.diff(trajectories.astype(np.float64), axis=0)
    return np.linalg.norm(diffs, axis=2).sum(axis=0)


def run_experiment(
    name: str,
    move_points: bool,
    base_loss_grid: np.ndarray,
    density_grid: np.ndarray,
    u_axis: np.ndarray,
    v_axis: np.ndarray,
    uu: np.ndarray,
    vv: np.ndarray,
    sampler: Dict[str, np.ndarray],
    rng: np.random.Generator,
    config: SimulationConfig,
) -> Tuple[Dict[str, np.ndarray], Iterable[Dict[str, float]]]:
    current_loss = base_loss_grid.astype(np.float32).copy()
    all_loss_grids = [current_loss.copy()]
    all_samples = []
    all_trajectories = []
    all_trajectory_loss = []
    all_endpoint_kde = []
    all_update_multipliers = []
    round_summaries = []

    for round_idx in range(config.n_rounds):
        samples = sample_from_density(rng, sampler, config.n_samples)
        if move_points:
            trajectories, trajectory_loss = projected_gradient_ascent(
                samples, current_loss, u_axis, v_axis, config
            )
        else:
            trajectories, trajectory_loss = stationary_trajectory(
                samples, current_loss, u_axis, v_axis, config
            )

        endpoints = trajectories[-1].astype(np.float64)
        next_loss, endpoint_kde, multiplier, drop_rate, rise_rate = update_loss_landscape(
            current_loss, endpoints, uu, vv, rng, config
        )

        sample_density = density_value(samples[:, 0], samples[:, 1])
        endpoint_density = density_value(endpoints[:, 0], endpoints[:, 1])
        lengths = path_lengths(trajectories)

        round_summaries.append(
            {
                "experiment": name,
                "round": round_idx,
                "drop_rate": drop_rate,
                "rise_rate": rise_rate,
                "mean_start_loss": float(trajectory_loss[0].mean()),
                "mean_end_loss": float(trajectory_loss[-1].mean()),
                "mean_path_length": float(lengths.mean()),
                "max_path_length": float(lengths.max()),
                "mean_sample_density": float(sample_density.mean()),
                "mean_endpoint_density": float(endpoint_density.mean()),
                "loss_grid_min_before": float(current_loss.min()),
                "loss_grid_max_before": float(current_loss.max()),
                "loss_grid_min_after": float(next_loss.min()),
                "loss_grid_max_after": float(next_loss.max()),
            }
        )

        all_samples.append(samples.astype(np.float32))
        all_trajectories.append(trajectories.astype(np.float32))
        all_trajectory_loss.append(trajectory_loss.astype(np.float32))
        all_endpoint_kde.append(endpoint_kde)
        all_update_multipliers.append(multiplier)
        current_loss = next_loss
        all_loss_grids.append(current_loss.copy())

    arrays = {
        "u_axis": u_axis.astype(np.float32),
        "v_axis": v_axis.astype(np.float32),
        "density_grid": density_grid.astype(np.float32),
        "loss_grids": np.stack(all_loss_grids).astype(np.float32),
        "samples": np.stack(all_samples).astype(np.float32),
        "trajectories": np.stack(all_trajectories).astype(np.float32),
        "trajectory_loss": np.stack(all_trajectory_loss).astype(np.float32),
        "endpoint_kde": np.stack(all_endpoint_kde).astype(np.float32),
        "update_multipliers": np.stack(all_update_multipliers).astype(np.float32),
    }
    return arrays, round_summaries


def save_round_summaries(path: Path, rows: Iterable[Dict[str, float]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_trajectory_csv_gz(path: Path, experiments: Dict[str, Dict[str, np.ndarray]]) -> None:
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["experiment", "round", "point_id", "step", "u", "v", "loss", "p_density"])
        for experiment, arrays in experiments.items():
            trajectories = arrays["trajectories"]
            trajectory_loss = arrays["trajectory_loss"]
            for round_idx in range(trajectories.shape[0]):
                for step_idx in range(trajectories.shape[1]):
                    pts = trajectories[round_idx, step_idx]
                    p_vals = density_value(pts[:, 0], pts[:, 1])
                    for point_idx, ((u, v), loss_val, p_val) in enumerate(
                        zip(pts, trajectory_loss[round_idx, step_idx], p_vals)
                    ):
                        writer.writerow(
                            [
                                experiment,
                                round_idx,
                                point_idx,
                                step_idx,
                                f"{float(u):.8f}",
                                f"{float(v):.8f}",
                                f"{float(loss_val):.8f}",
                                f"{float(p_val):.8f}",
                            ]
                        )


def try_make_plots(
    out_dir: Path,
    experiments: Dict[str, Dict[str, np.ndarray]],
    config: SimulationConfig,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib import cm, colors
    except ImportError:
        print("Matplotlib is not installed; data were saved, but preview plots were skipped.")
        return

    u_axis = next(iter(experiments.values()))["u_axis"]
    v_axis = next(iter(experiments.values()))["v_axis"]
    density_grid = next(iter(experiments.values()))["density_grid"]
    uu, vv = np.meshgrid(u_axis, v_axis, indexing="xy")
    floor_z = float(min(arrays["loss_grids"].min() for arrays in experiments.values()) - 0.75)

    density_norm = colors.Normalize(vmin=float(density_grid.min()), vmax=float(density_grid.max()))
    density_facecolors = cm.viridis(density_norm(density_grid))
    density_facecolors[..., 3] = 0.72

    for name, arrays in experiments.items():
        for round_idx in [0, config.n_rounds - 1]:
            loss_grid = arrays["loss_grids"][round_idx]
            trajectories = arrays["trajectories"][round_idx]
            final_pts = trajectories[-1]

            fig = plt.figure(figsize=(10.5, 7.2), dpi=180)
            ax = fig.add_subplot(111, projection="3d")
            ax.plot_surface(
                uu,
                vv,
                loss_grid,
                cmap="cool",
                linewidth=0,
                antialiased=True,
                alpha=0.72,
                rcount=130,
                ccount=130,
            )
            ax.plot_surface(
                uu,
                vv,
                np.full_like(uu, floor_z),
                facecolors=density_facecolors,
                linewidth=0,
                antialiased=False,
                shade=False,
            )

            preview_ids = np.linspace(0, config.n_samples - 1, min(42, config.n_samples), dtype=int)
            start_pts = trajectories[0, preview_ids]
            end_pts = trajectories[-1, preview_ids]
            start_loss = bilinear_sample(loss_grid, u_axis, v_axis, start_pts)
            end_loss = bilinear_sample(loss_grid, u_axis, v_axis, end_pts)
            start_floor = np.full(len(preview_ids), floor_z)

            stride = max(1, config.n_steps // 25)
            for point_id, (u0, v0), z0 in zip(preview_ids, start_pts, start_loss):
                ax.plot(
                    [u0, u0],
                    [v0, v0],
                    [floor_z, z0],
                    color="0.55",
                    linestyle="--",
                    linewidth=0.35,
                    alpha=0.25,
                )
                if name == "gradient_ascent":
                    path = trajectories[::stride, point_id].astype(np.float64)
                    path_z = bilinear_sample(loss_grid, u_axis, v_axis, path)
                    ax.plot(
                        path[:, 0],
                        path[:, 1],
                        path_z,
                        color="#233B8E",
                        linewidth=0.45,
                        alpha=0.28,
                    )

            ax.scatter(start_pts[:, 0], start_pts[:, 1], start_floor, s=8, c="#B73757", alpha=0.70, depthshade=False)
            ax.scatter(start_pts[:, 0], start_pts[:, 1], start_loss, s=7, c="#6E6E6E", alpha=0.34, depthshade=False)
            ax.scatter(end_pts[:, 0], end_pts[:, 1], end_loss, s=9, c="#122A78", alpha=0.70, depthshade=False)

            ax.set_title(f"{name}, round {round_idx + 1}", pad=12)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.view_init(elev=25, azim=-55)
            ax.set_xlim(DOMAIN_MIN, DOMAIN_MAX)
            ax.set_ylim(DOMAIN_MIN, DOMAIN_MAX)
            ax.set_zlim(floor_z, float(arrays["loss_grids"].max()) + 0.15)
            ax.grid(False)
            fig.tight_layout()
            fig.savefig(out_dir / f"{name}_round_{round_idx + 1:02d}_preview.png")
            plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=180, constrained_layout=True)
    global_min = min(float(arrays["loss_grids"][-1].min()) for arrays in experiments.values())
    global_max = max(float(arrays["loss_grids"][-1].max()) for arrays in experiments.values())
    for ax, (name, arrays) in zip(axes, experiments.items()):
        final_loss = arrays["loss_grids"][-1]
        im = ax.imshow(
            final_loss,
            origin="lower",
            extent=[DOMAIN_MIN, DOMAIN_MAX, DOMAIN_MIN, DOMAIN_MAX],
            cmap="cool",
            vmin=global_min,
            vmax=global_max,
            aspect="equal",
        )
        mean_loss = float(final_loss.mean())
        p95_loss = float(np.percentile(final_loss, 95))
        ax.set_title(f"final L(u, v): {name}\nmean L = {mean_loss:.3f}, p95 = {p95_loss:.3f}")
        ax.set_xlabel("u")
        ax.set_ylabel("v")
        fig.colorbar(im, ax=ax, shrink=0.86)
    fig.savefig(out_dir / "final_loss_landscape_comparison.png")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("figures/loss_landscape/simulation_outputs"))
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument("--n-rounds", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.0125)
    parser.add_argument("--grid-size", type=int, default=321)
    parser.add_argument("--sampler-grid-size", type=int, default=701)
    parser.add_argument("--update-bandwidth", type=float, default=0.24)
    parser.add_argument("--broad-bandwidth", type=float, default=0.55)
    parser.add_argument("--drop-min", type=float, default=0.20)
    parser.add_argument("--drop-max", type=float, default=0.30)
    parser.add_argument("--rise-min", type=float, default=0.05)
    parser.add_argument("--rise-max", type=float, default=0.10)
    parser.add_argument("--pgd-epsilon", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--unpaired-baseline-seed",
        action="store_true",
        help="Use an independent sample/update seed for the no-moving baseline.",
    )
    parser.add_argument("--no-normalize-gradient", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--skip-trajectory-csv", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SimulationConfig(
        n_samples=args.n_samples,
        n_steps=args.n_steps,
        n_rounds=args.n_rounds,
        alpha=args.alpha,
        grid_size=args.grid_size,
        sampler_grid_size=args.sampler_grid_size,
        update_bandwidth=args.update_bandwidth,
        broad_bandwidth=args.broad_bandwidth,
        drop_min=args.drop_min,
        drop_max=args.drop_max,
        rise_min=args.rise_min,
        rise_max=args.rise_max,
        pgd_epsilon=args.pgd_epsilon,
        seed=args.seed,
        normalize_gradient=not args.no_normalize_gradient,
    )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    u_axis, v_axis, uu, vv = make_grid(config.grid_size)
    density_grid = density_value(uu, vv).astype(np.float32)
    base_loss_grid = loss_height(uu, vv).astype(np.float32)
    sampler = build_density_sampler(config.sampler_grid_size)

    experiments: Dict[str, Dict[str, np.ndarray]] = {}
    summaries = []

    for name, move_points, seed_offset in [
        ("gradient_ascent", True, 0),
        ("no_movement_baseline", False, 10_000 if args.unpaired_baseline_seed else 0),
    ]:
        rng = np.random.default_rng(config.seed + seed_offset)
        arrays, rows = run_experiment(
            name=name,
            move_points=move_points,
            base_loss_grid=base_loss_grid,
            density_grid=density_grid,
            u_axis=u_axis,
            v_axis=v_axis,
            uu=uu,
            vv=vv,
            sampler=sampler,
            rng=rng,
            config=config,
        )
        experiments[name] = arrays
        summaries.extend(rows)

        np.savez_compressed(
            out_dir / f"{name}_simulation_data.npz",
            **arrays,
            config_json=json.dumps(asdict(config), indent=2),
        )

    with (out_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "config": asdict(config),
                "domain": [DOMAIN_MIN, DOMAIN_MAX],
                "loss_function_source": "figures/loss_landscape/non-compact/loss_landscape.tex::lossheight",
                "density_function_source": "figures/loss_landscape/non-compact/loss_landscape.tex::densityvalue",
                "experiments": list(experiments.keys()),
            },
            handle,
            indent=2,
        )

    save_round_summaries(out_dir / "round_summary.csv", summaries)
    if not args.skip_trajectory_csv:
        save_trajectory_csv_gz(out_dir / "trajectories.csv.gz", experiments)

    if not args.skip_plots:
        try_make_plots(out_dir, experiments, config)

    print(f"Saved simulation outputs to: {out_dir.resolve()}")
    print("Saved experiments: gradient_ascent, no_movement_baseline")
    print(f"Rounds: {config.n_rounds}; samples per round: {config.n_samples}; steps: {config.n_steps}")


if __name__ == "__main__":
    main()
