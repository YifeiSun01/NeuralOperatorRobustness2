#!/usr/bin/env python3
"""GIF/final-frame visualization for three-loss objective attack runs.

This intentionally does not plot loss curves. It mirrors the old Burgers GIF
style by showing clean/perturbed inputs and model/solver outputs over attack
steps. Loss/objective values are shown only as compact text for the current
frame.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


LOSS_TYPES = ("loss1", "loss2", "loss3")
OBJECTIVE_VARIANTS = ("original", "increment_ratio", "regularized")


def read_metrics(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out: list[dict[str, Any]] = []
    for row in rows:
        converted: dict[str, Any] = {}
        for key, value in row.items():
            if value == "":
                converted[key] = value
                continue
            try:
                converted[key] = float(value)
            except ValueError:
                converted[key] = value
        out.append(converted)
    return out


def squeeze_1d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    arr = np.squeeze(arr)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim > 1:
        return arr.reshape(-1)
    return arr


def finite_minmax(arrays: list[np.ndarray], pad_frac: float = 0.06) -> tuple[float, float]:
    vals = [np.asarray(a).reshape(-1) for a in arrays if a is not None and np.asarray(a).size]
    vals = [v[np.isfinite(v)] for v in vals]
    vals = [v for v in vals if v.size]
    if not vals:
        return -1.0, 1.0
    cat = np.concatenate(vals)
    lo = float(np.min(cat))
    hi = float(np.max(cat))
    if lo == hi:
        pad = max(1.0, abs(lo)) * pad_frac
        return lo - pad, hi + pad
    pad = (hi - lo) * pad_frac
    return lo - pad, hi + pad


def fmt(value: Any) -> str:
    try:
        value = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(value):
        return "nan"
    if abs(value) >= 1e3 or (0 < abs(value) < 1e-3):
        return f"{value:.2e}"
    return f"{value:.4g}"


def objective_table_text(row: dict[str, Any]) -> str:
    lines = ["3 x 3 objective evaluation"]
    for loss_name in LOSS_TYPES:
        vals = []
        for variant in OBJECTIVE_VARIANTS:
            key = f"{loss_name}_{variant}_objective"
            vals.append(f"{variant}={fmt(row.get(key, np.nan))}")
        lines.append(f"{loss_name}: " + " | ".join(vals))
    lines.append("")
    lines.append(f"optimized={fmt(row.get('optimized_loss', np.nan))}")
    lines.append(f"delta_p={fmt(row.get('delta_norm_p', np.nan))}")
    lines.append(f"budget={fmt(row.get('delta_budget_ratio', np.nan))}")
    return "\n".join(lines)


def plot_line(ax, x, y, label, color, linewidth=1.4, alpha=1.0):
    ax.plot(x, squeeze_1d(y), label=label, color=color, linewidth=linewidth, alpha=alpha)


def render_frame(
    traj: dict[str, np.ndarray],
    rows: list[dict[str, Any]],
    i: int,
    limits: dict[str, tuple[float, float]],
    title: str,
    out_path: Path | None,
) -> np.ndarray:
    k_values = traj["k"]
    k = int(k_values[i])
    row = rows[min(k, len(rows) - 1)] if rows else {}
    x_clean = squeeze_1d(traj["x_clean"])
    x_adv = squeeze_1d(traj["x_adv"][i])
    delta = squeeze_1d(traj["delta"][i])
    model_clean = squeeze_1d(traj["model_output_clean"])
    solver_clean = squeeze_1d(traj["solver_output_clean"])
    model_adv = squeeze_1d(traj["model_output_adv"][i])
    solver_adv = squeeze_1d(traj["solver_output_adv"][i])
    diff_adv = squeeze_1d(traj["difference_adv"][i])
    xgrid = np.arange(x_clean.size)
    ygrid = np.arange(model_clean.size)

    fig = plt.figure(figsize=(16, 9.5))
    gs = fig.add_gridspec(2, 3, hspace=0.28, wspace=0.18)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]

    ax = axes[0]
    plot_line(ax, xgrid, x_clean, "clean input", "black", 1.6)
    plot_line(ax, xgrid, x_adv, "perturbed input", "#1f77b4", 1.3)
    ax.set_title("Input")
    ax.set_ylim(limits["input"])
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1]
    plot_line(ax, xgrid, delta, "delta", "#d62728", 1.3)
    ax.axhline(0.0, color="0.35", linewidth=0.8)
    ax.set_title("Perturbation")
    ax.set_ylim(limits["delta"])
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[2]
    plot_line(ax, ygrid, model_clean, "model clean", "black", 1.4)
    plot_line(ax, ygrid, solver_clean, "solver clean", "0.45", 1.3)
    ax.set_title("Clean Outputs")
    ax.set_ylim(limits["output"])
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[3]
    ax.fill_between(ygrid, model_adv, solver_adv, color="#ff7f0e", alpha=0.16, linewidth=0.0, label="model-solver gap")
    plot_line(ax, ygrid, model_adv, "model perturbed", "#ff7f0e", 1.5)
    plot_line(ax, ygrid, solver_adv, "solver perturbed", "#2ca02c", 1.35)
    ax.set_title("Perturbed Outputs")
    ax.set_ylim(limits["output"])
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[4]
    plot_line(ax, ygrid, diff_adv, "model - solver", "#9467bd", 1.35)
    ax.axhline(0.0, color="0.35", linewidth=0.8)
    ax.set_title("Output Difference")
    ax.set_ylim(limits["diff"])
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[5]
    ax.axis("off")
    ax.text(0.0, 0.98, objective_table_text(row), va="top", ha="left", fontsize=10.8, family="monospace")

    fig.suptitle(f"{title} | step={k}", fontsize=13)
    fig.subplots_adjust(left=0.045, right=0.985, bottom=0.06, top=0.92)
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=145)
    plt.close(fig)
    return frame


def make_outputs(run_dir: Path, outdir: Path, fps: int, stride: int, cleanup_frames: bool, final_png_only: bool) -> None:
    traj_path = run_dir / "trajectory.npz"
    if not traj_path.exists():
        raise FileNotFoundError(f"{traj_path} not found. Re-run attack with --save_trajectory or --make_gif.")
    traj_file = np.load(traj_path)
    traj = {key: traj_file[key] for key in traj_file.files}
    rows = read_metrics(run_dir / "metrics.csv") if (run_dir / "metrics.csv").exists() else []
    outdir.mkdir(parents=True, exist_ok=True)

    title = run_dir.name
    limits = {
        "input": finite_minmax([traj["x_clean"], traj["x_adv"]]),
        "delta": finite_minmax([traj["delta"]]),
        "output": finite_minmax([traj["model_output_clean"], traj["solver_output_clean"], traj["model_output_adv"], traj["solver_output_adv"]]),
        "diff": finite_minmax([traj["difference_adv"]]),
    }
    n = len(traj["k"])
    selected = list(range(0, n, max(1, stride)))
    if selected[-1] != n - 1:
        selected.append(n - 1)

    final_png = outdir / f"{run_dir.name}_final.png"
    if final_png_only:
        render_frame(traj, rows, n - 1, limits, title, final_png)
        print(f"[ok] saved {final_png}")
        return

    frame_dir = outdir / f".{run_dir.name}_frames"
    frames = []
    for j, i in enumerate(tqdm(selected, desc=f"Rendering {run_dir.name}")):
        frame_path = frame_dir / f"frame_{j:04d}_step{int(traj['k'][i]):04d}.png"
        frames.append(render_frame(traj, rows, i, limits, title, frame_path))
    gif_path = outdir / f"{run_dir.name}.gif"
    iio.imwrite(gif_path, frames, fps=fps)
    render_frame(traj, rows, n - 1, limits, title, final_png)
    if cleanup_frames:
        shutil.rmtree(frame_dir, ignore_errors=True)
    print(f"[ok] saved {gif_path}")
    print(f"[ok] saved {final_png}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--cleanup_frames", action="store_true")
    parser.add_argument("--final_png_only", action="store_true")
    args = parser.parse_args()
    outdir = args.outdir or (args.run_dir / "gif_frames")
    make_outputs(args.run_dir, outdir, args.fps, args.stride, args.cleanup_frames, args.final_png_only)


if __name__ == "__main__":
    main()
