#!/usr/bin/env python3
"""Render a saved NS2D attack step trace as an animated heat-map GIF.

This script is intentionally CPU-only. It reads the compact per-sample trace
saved by attack_ns2d_recurrent_core4.py and does not import torch or jax.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method-dir",
        type=Path,
        required=True,
        help="Directory containing step_sample_trace.npz and step_sample_trace_metrics.csv.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for GIF and manifest.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Output GIF stem. Defaults to a name inferred from the method directory.",
    )
    parser.add_argument("--duration", type=float, default=0.18, help="Seconds per GIF frame.")
    parser.add_argument("--dpi", type=int, default=90, help="Matplotlib render DPI.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional maximum number of frames for quick previews.",
    )
    return parser.parse_args()


def symmetric_limits(array: np.ndarray) -> Tuple[float, float]:
    finite = np.asarray(array)[np.isfinite(array)]
    if finite.size == 0:
        return -1.0, 1.0
    bound = float(np.max(np.abs(finite)))
    if not np.isfinite(bound) or bound <= 0.0:
        bound = 1.0
    return -bound, bound


def read_metrics(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def metric_float(rows: List[Dict[str, str]], key: str) -> np.ndarray:
    values: List[float] = []
    for row in rows:
        raw = row.get(key, "")
        try:
            values.append(float(raw))
        except ValueError:
            values.append(np.nan)
    return np.asarray(values, dtype=np.float64)


def infer_label(method_dir: Path) -> Dict[str, str]:
    parts = method_dir.parts
    label = {
        "eps_alpha": "unknown",
        "mode": "unknown",
        "loss_type": method_dir.parent.name,
        "method": method_dir.name,
    }
    for part in parts:
        if part.startswith("eps") and "_alpha" in part:
            label["eps_alpha"] = part
        if part.startswith("mode_"):
            label["mode"] = part.removeprefix("mode_").split("_p2_q2_")[0]
    return label


def add_heatmap(
    ax: plt.Axes,
    image: np.ndarray,
    title: str,
    cmap: str,
    limits: Tuple[float, float],
) -> None:
    im = ax.imshow(image, origin="lower", cmap=cmap, vmin=limits[0], vmax=limits[1])
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cbar.ax.tick_params(labelsize=7)


def plot_loss_panel(
    ax: plt.Axes,
    steps: np.ndarray,
    true_loss: np.ndarray,
    active_loss: np.ndarray,
    k: int,
    row: Dict[str, str],
) -> None:
    ax.plot(steps, active_loss, color="#b2182b", lw=1.8, label=row.get("active_loss", "active"))
    ax.plot(steps, true_loss, color="#2166ac", lw=1.5, label="true_loss")
    ax.axvline(k, color="black", lw=1.0, alpha=0.6)
    if k in steps:
        idx = int(np.where(steps == k)[0][0])
        ax.scatter([k], [active_loss[idx]], color="#b2182b", s=24, zorder=4)
        ax.scatter([k], [true_loss[idx]], color="#2166ac", s=24, zorder=4)
    ax.set_title("loss progression", fontsize=9)
    ax.set_xlabel("attack step")
    ax.set_ylabel("loss")
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=7, loc="best")

    text = (
        f"k={k}\n"
        f"true={float(row.get('true_loss', 'nan')):.4g}\n"
        f"active={float(row.get('active_loss_value', 'nan')):.4g}\n"
        f"delta_l2={float(row.get('delta_l2', 'nan')):.4g}\n"
        f"boundary={float(row.get('boundary_ratio', 'nan')):.3f}"
    )
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "0.75", "alpha": 0.88},
    )


def main() -> int:
    args = parse_args()
    method_dir = args.method_dir
    trace_path = method_dir / "step_sample_trace.npz"
    metrics_path = method_dir / "step_sample_trace_metrics.csv"
    if not trace_path.exists():
        raise FileNotFoundError(trace_path)
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    trace = np.load(trace_path)
    rows = read_metrics(metrics_path)
    label = infer_label(method_dir)
    sample_position = int(np.asarray(trace["sample_position"]))
    dataset_index = int(np.asarray(trace["dataset_index"]))
    steps = np.asarray(trace["k"], dtype=np.int64)
    if args.max_frames is not None:
        keep = min(args.max_frames, len(steps))
        steps = steps[:keep]
        rows = rows[:keep]

    x_adv = np.asarray(trace["x_adv"], dtype=np.float32)[: len(steps)]
    grad = np.asarray(trace["grad"], dtype=np.float32)[: len(steps)]
    model = np.asarray(trace["adv_model_final"], dtype=np.float32)[: len(steps)]
    solver = np.asarray(trace["adv_solver_final"], dtype=np.float32)[: len(steps)]
    diff = np.asarray(trace["adv_model_minus_solver"], dtype=np.float32)[: len(steps)]

    limits = {
        "x_adv": symmetric_limits(x_adv),
        "grad": symmetric_limits(grad),
        "model": symmetric_limits(model),
        "solver": symmetric_limits(solver),
        "diff": symmetric_limits(diff),
    }

    metric_steps = metric_float(rows, "k").astype(np.int64)
    true_loss = metric_float(rows, "true_loss")
    active_loss = metric_float(rows, "active_loss_value")

    stem = args.name
    if stem is None:
        stem = (
            f"{label['eps_alpha']}_{label['loss_type']}_{label['mode']}_"
            f"{label['method']}_sample{sample_position}_idx{dataset_index}_step_trace"
        )
    gif_path = out_dir / f"{stem}.gif"

    title_prefix = (
        f"{label['eps_alpha']} | {label['loss_type']} | mode={label['mode']} | "
        f"{label['method']} | sample_pos={sample_position} idx={dataset_index}"
    )

    with imageio.get_writer(gif_path, mode="I", duration=args.duration, loop=0) as writer:
        for frame_idx, k in enumerate(steps):
            fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.2), dpi=args.dpi, constrained_layout=True)
            fig.suptitle(title_prefix, fontsize=12)

            row = rows[frame_idx]
            add_heatmap(
                axes[0, 0],
                x_adv[frame_idx],
                "perturbed initial x + delta",
                "viridis",
                limits["x_adv"],
            )
            add_heatmap(
                axes[0, 1],
                grad[frame_idx],
                "gradient wrt initial condition",
                "coolwarm",
                limits["grad"],
            )
            add_heatmap(
                axes[0, 2],
                model[frame_idx],
                "FNO model final output",
                "viridis",
                limits["model"],
            )
            add_heatmap(
                axes[1, 0],
                solver[frame_idx],
                "solver final output",
                "viridis",
                limits["solver"],
            )
            add_heatmap(
                axes[1, 1],
                diff[frame_idx],
                "model output - solver output",
                "coolwarm",
                limits["diff"],
            )
            plot_loss_panel(axes[1, 2], metric_steps, true_loss, active_loss, int(k), row)

            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            writer.append_data(rgba[:, :, :3].copy())
            plt.close(fig)

    manifest = {
        "gif": str(gif_path),
        "source_method_dir": str(method_dir),
        "source_trace": str(trace_path),
        "source_metrics": str(metrics_path),
        "sample_position": sample_position,
        "dataset_index": dataset_index,
        "num_frames": int(len(steps)),
        "step_min": int(steps.min()),
        "step_max": int(steps.max()),
        "duration_seconds_per_frame": args.duration,
        "color_maps": {
            "perturbed_initial": "viridis",
            "gradient": "coolwarm",
            "model_output": "viridis",
            "solver_output": "viridis",
            "model_minus_solver": "coolwarm",
        },
        "color_range_policy": "Each heatmap panel uses its own global symmetric vmin/vmax across GIF frames, so zero is centered.",
        "limits": {key: [float(v[0]), float(v[1])] for key, v in limits.items()},
    }
    manifest_path = out_dir / f"{stem}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(gif_path)
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
