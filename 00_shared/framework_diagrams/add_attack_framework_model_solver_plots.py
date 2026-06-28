#!/usr/bin/env python3
"""Add model-vs-solver comparison plots to existing attack-framework results.

The original attack-framework visualizations emphasize before/after changes for
the same object.  These supplemental figures put model and solver outputs in the
same view so their discrepancy is easy to inspect before and after perturbation.
Existing figures are left untouched.
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np


COMBO_ORDER = [
    "torch_solver_torch_model",
    "torch_solver_jax_model",
    "jax_solver_torch_model",
    "jax_solver_jax_model",
]


def read_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
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


def sorted_combo_dirs(case_dir: Path) -> list[Path]:
    dirs = [p for p in case_dir.iterdir() if p.is_dir() and (p / "histories.npz").exists()]
    order = {name: i for i, name in enumerate(COMBO_ORDER)}
    return sorted(dirs, key=lambda p: (order.get(p.name, 999), p.name))


def step_label(combo_dir: Path, step: int) -> str:
    rows = read_metrics(combo_dir / "metrics.csv")
    if rows and step < len(rows):
        loss = rows[step].get("loss", rows[step].get("true_loss", math.nan))
        try:
            return f"step {step}, loss={float(loss):.3e}"
        except Exception:
            return f"step {step}"
    return f"step {step}"


def as_1d(arr: np.ndarray) -> np.ndarray:
    return np.squeeze(arr).reshape(-1)


def ns_last_frame(arr: np.ndarray) -> np.ndarray:
    arr = np.squeeze(arr)
    if arr.ndim == 3:
        return arr[..., -1]
    return arr


def save_burgers_combo_plot(combo_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = np.load(combo_dir / "histories.npz")
    inp = h["input_history"]
    model = h["model_history"]
    solver = h["solver_history"]
    final_idx = inp.shape[0] - 1
    x = np.arange(inp.shape[1])

    input_clean = as_1d(inp[0])
    input_adv = as_1d(inp[final_idx])
    model_clean = as_1d(model[0])
    solver_clean = as_1d(solver[0])
    model_adv = as_1d(model[final_idx])
    solver_adv = as_1d(solver[final_idx])

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax = axes[0]
    ax.plot(x, input_clean, label="input before attack", color="0.25", linewidth=1.2)
    ax.plot(x, input_adv, label="input after attack", color="tab:blue", linewidth=1.2)
    ax.set_title(f"{combo_dir.name}: initial condition before vs after perturbation")
    ax.set_ylabel("input")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(x, solver_clean, label="solver output before attack", color="tab:orange", linestyle="-", linewidth=1.2)
    ax.plot(x, model_clean, label="model output before attack", color="tab:green", linestyle="-", linewidth=1.2)
    ax.plot(x, solver_adv, label="solver output after attack", color="tab:red", linestyle="--", linewidth=1.2)
    ax.plot(x, model_adv, label="model output after attack", color="tab:purple", linestyle="--", linewidth=1.2)
    ax.set_title("final condition: solver vs model, before and after perturbation")
    ax.set_xlabel("grid index")
    ax.set_ylabel("output")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8, ncol=2)

    fig.tight_layout()
    out = combo_dir / "model_solver_input_output_comparison.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def save_burgers_case_frame(case_dir: Path, frame_idx: int, suffix: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    combo_dirs = sorted_combo_dirs(case_dir)
    fig, axes = plt.subplots(len(combo_dirs), 2, figsize=(13, 3.2 * len(combo_dirs)), squeeze=False)
    for row, combo_dir in enumerate(combo_dirs):
        h = np.load(combo_dir / "histories.npz")
        inp = h["input_history"]
        model = h["model_history"]
        solver = h["solver_history"]
        final_idx = inp.shape[0] - 1
        idx = final_idx if frame_idx < 0 else min(frame_idx, final_idx)
        x = np.arange(inp.shape[1])

        ax = axes[row, 0]
        ax.plot(x, as_1d(inp[0]), label="input before attack", color="0.25", linewidth=1.0)
        ax.plot(x, as_1d(inp[idx]), label=f"input at {step_label(combo_dir, idx)}", color="tab:blue", linewidth=1.0)
        ax.set_title(f"{combo_dir.name}: input before vs selected attack step")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)

        ax = axes[row, 1]
        ax.plot(x, as_1d(solver[0]), label="solver before attack", color="tab:orange", linewidth=1.0)
        ax.plot(x, as_1d(model[0]), label="model before attack", color="tab:green", linewidth=1.0)
        ax.plot(x, as_1d(solver[idx]), label="solver selected step", color="tab:red", linestyle="--", linewidth=1.0)
        ax.plot(x, as_1d(model[idx]), label="model selected step", color="tab:purple", linestyle="--", linewidth=1.0)
        ax.set_title("output: solver vs model before/after")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7, ncol=2)
    axes[-1, 0].set_xlabel("grid index")
    axes[-1, 1].set_xlabel("grid index")
    fig.suptitle(f"{case_dir.name}: model-vs-solver comparison, {suffix}", fontsize=12)
    fig.tight_layout()
    out = case_dir / f"{case_dir.name}_model_solver_comparison_{suffix}.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def save_ns_combo_plot(combo_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = np.load(combo_dir / "histories.npz")
    inp = h["input_history"]
    model = h["model_history"]
    solver = h["solver_history"]
    final_idx = inp.shape[0] - 1

    panels = [
        (ns_last_frame(inp[0]), "input before attack"),
        (ns_last_frame(inp[final_idx]), "input after attack"),
        (ns_last_frame(inp[final_idx] - inp[0]), "input perturbation"),
        (ns_last_frame(solver[0]), "solver output before attack"),
        (ns_last_frame(model[0]), "model output before attack"),
        (ns_last_frame(model[0] - solver[0]), "model - solver before attack"),
        (ns_last_frame(solver[final_idx]), "solver output after attack"),
        (ns_last_frame(model[final_idx]), "model output after attack"),
        (ns_last_frame(model[final_idx] - solver[final_idx]), "model - solver after attack"),
    ]
    field_vals = [v for v, title in panels if "perturbation" not in title and "model - solver" not in title]
    diff_vals = [v for v, title in panels if "perturbation" in title or "model - solver" in title]
    field_vmin = min(float(np.min(v)) for v in field_vals)
    field_vmax = max(float(np.max(v)) for v in field_vals)
    diff_abs = max(1e-12, *(float(np.max(np.abs(v))) for v in diff_vals))

    fig, axes = plt.subplots(3, 3, figsize=(12, 11), squeeze=False)
    for ax, (values, title) in zip(axes.ravel(), panels):
        is_diff = "perturbation" in title or "model - solver" in title
        im = ax.imshow(
            values,
            cmap="coolwarm" if is_diff else "viridis",
            vmin=-diff_abs if is_diff else field_vmin,
            vmax=diff_abs if is_diff else field_vmax,
        )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{combo_dir.name}: input/output and model-vs-solver comparison", fontsize=12)
    fig.tight_layout()
    out = combo_dir / "model_solver_input_output_comparison.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def save_ns_case_frame(case_dir: Path, frame_idx: int, suffix: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    combo_dirs = sorted_combo_dirs(case_dir)
    nrows = len(combo_dirs)
    ncols = 6
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 3.0 * nrows), squeeze=False)
    for row, combo_dir in enumerate(combo_dirs):
        h = np.load(combo_dir / "histories.npz")
        inp = h["input_history"]
        model = h["model_history"]
        solver = h["solver_history"]
        final_idx = inp.shape[0] - 1
        idx = final_idx if frame_idx < 0 else min(frame_idx, final_idx)
        panels = [
            (ns_last_frame(inp[0]), "input before", False),
            (ns_last_frame(inp[idx]), "input selected", False),
            (ns_last_frame(solver[0]), "solver before", False),
            (ns_last_frame(model[0]), "model before", False),
            (ns_last_frame(solver[idx]), "solver selected", False),
            (ns_last_frame(model[idx]), "model selected", False),
        ]
        vals = [p[0] for p in panels]
        vmin = min(float(np.min(v)) for v in vals)
        vmax = max(float(np.max(v)) for v in vals)
        for col, (values, title, _) in enumerate(panels):
            ax = axes[row, col]
            im = ax.imshow(values, cmap="viridis", vmin=vmin, vmax=vmax)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"{combo_dir.name}\n{title}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(f"{case_dir.name}: model-vs-solver before/after comparison, {suffix}", fontsize=12)
    fig.tight_layout()
    out = case_dir / f"{case_dir.name}_model_solver_comparison_{suffix}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def save_case_gif(case_dir: Path, case: str, max_frames: int, cleanup_png: bool) -> Path | None:
    try:
        import imageio.v2 as imageio
    except Exception:
        return None

    combo_dirs = sorted_combo_dirs(case_dir)
    if not combo_dirs:
        return None
    steps = min(np.load(d / "histories.npz")["input_history"].shape[0] for d in combo_dirs)
    frame_ids = np.unique(np.linspace(0, steps - 1, min(max_frames, steps), dtype=int))
    frame_dir = case_dir / ".model_solver_comparison_frames"
    shutil.rmtree(frame_dir, ignore_errors=True)
    frame_dir.mkdir(parents=True, exist_ok=True)
    frames: list[Path] = []
    for frame_id in frame_ids:
        if case == "burgers_1d":
            path = save_burgers_case_frame(case_dir, int(frame_id), f"frame_{int(frame_id):04d}")
        else:
            path = save_ns_case_frame(case_dir, int(frame_id), f"frame_{int(frame_id):04d}")
        tmp = frame_dir / path.name
        shutil.move(str(path), tmp)
        frames.append(tmp)
    gif_path = case_dir / f"{case}_model_solver_comparison.gif"
    imageio.mimsave(gif_path, [imageio.imread(frame) for frame in frames], duration=0.25)
    first = case_dir / f"{case}_model_solver_comparison_first_frame.png"
    final = case_dir / f"{case}_model_solver_comparison_final_frame.png"
    shutil.copyfile(frames[0], first)
    shutil.copyfile(frames[-1], final)
    if cleanup_png:
        shutil.rmtree(frame_dir, ignore_errors=True)
    return gif_path


def process_case_dir(case_dir: Path, max_gif_frames: int, cleanup_png: bool) -> list[Path]:
    made: list[Path] = []
    case = case_dir.name
    for combo_dir in sorted_combo_dirs(case_dir):
        if case == "burgers_1d":
            made.append(save_burgers_combo_plot(combo_dir))
        elif case == "ns_2d":
            made.append(save_ns_combo_plot(combo_dir))
    if case == "burgers_1d":
        made.append(save_burgers_case_frame(case_dir, 0, "first_frame"))
        made.append(save_burgers_case_frame(case_dir, -1, "final_frame"))
    elif case == "ns_2d":
        made.append(save_ns_case_frame(case_dir, 0, "first_frame"))
        made.append(save_ns_case_frame(case_dir, -1, "final_frame"))
    gif = save_case_gif(case_dir, case, max_gif_frames, cleanup_png)
    if gif is not None:
        made.append(gif)
    return made


def find_case_dirs(root: Path) -> list[Path]:
    out: list[Path] = []
    for case in ("burgers_1d", "ns_2d"):
        path = root / case
        if path.exists() and sorted_combo_dirs(path):
            out.append(path)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="+", type=Path, default=sorted(Path("benchmark_results").glob("attack_framework_matrix*")))
    parser.add_argument("--max-gif-frames", type=int, default=30)
    parser.add_argument("--cleanup-png", action="store_true")
    args = parser.parse_args()

    made: list[Path] = []
    for root in args.roots:
        for case_dir in find_case_dirs(root):
            made.extend(process_case_dir(case_dir, args.max_gif_frames, args.cleanup_png))
    for path in made:
        print(path)
    print(f"[done] generated {len(made)} supplemental model-vs-solver plot files")


if __name__ == "__main__":
    main()
