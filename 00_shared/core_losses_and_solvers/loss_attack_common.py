#!/usr/bin/env python3
"""Shared plotting and summary helpers for loss-comparison attack experiments."""

from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np


LOSS_KEYS = ["optimized_loss", "true_loss", "loss1", "loss2", "loss2_dict", "loss3_stopgrad", "loss3"]
SUMMARY_COLUMNS = [
    "case",
    "loss_type",
    "attack_method",
    "norm",
    "input_p",
    "output_q",
    "power_variant",
    "power_radius_mode",
    "canonical_loss_type",
    "epsilon",
    "alpha",
    "steps",
    "index",
    "frame_mode",
    "uses_solver_forward",
    "uses_solver_backward",
    "initial_optimized_loss",
    "final_optimized_loss",
    "optimized_loss_growth",
    "optimized_loss_growth_ratio",
    "initial_true_loss",
    "final_true_loss",
    "true_loss_growth",
    "true_loss_growth_ratio",
    "final_delta_l2",
    "final_delta_linf",
    "final_gradient_l2",
    "final_gradient_linf",
    "total_time",
    "mean_step_time",
    "estimated_sigma_final",
    "rayleigh_quotient_final",
    "cos_pgd_singular_vector",
    "abs_cos_pgd_singular_vector",
]


def finite_or_none(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, dict):
        return {k: finite_or_none(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_or_none(v) for v in value]
    return value


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(finite_or_none(payload), indent=2), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_metrics(path: Path) -> list[dict[str, Any]]:
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
                number = float(value)
                converted[key] = int(number) if key in {"k", "step", "index", "steps"} else number
            except ValueError:
                converted[key] = value
        out.append(converted)
    return out


def find_run_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path.parent for path in root.rglob("summary.json") if (path.parent / "metrics.csv").exists())


def load_curve(run_dir: Path, name: str) -> np.ndarray:
    path = run_dir / f"{name}.npy"
    if path.exists():
        return np.load(path)
    rows = read_metrics(run_dir / "metrics.csv")
    return np.asarray([float(row.get(name, np.nan)) for row in rows], dtype=np.float64)


def safe_last(arr: np.ndarray) -> float:
    return float(arr[-1]) if arr.size else float("nan")


def label_from_summary(run_dir: Path) -> str:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return run_dir.name
    summary = load_json(summary_path)
    return f"{summary.get('loss_type', '?')}_{summary.get('attack_method', '?')}"


def make_basic_run_plots(run_dir: Path, output_dir: Path | None = None) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = output_dir or (run_dir / "figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_metrics(run_dir / "metrics.csv")
    k = np.asarray([row["k"] for row in rows])
    made: list[Path] = []

    groups = [
        ("optimized_loss", ["optimized_loss"], "optimized loss"),
        ("true_loss", ["true_loss"], "true loss"),
        ("true_loss_growth_ratio", ["true_loss_growth_ratio", "true_loss_step_growth_ratio"], "true loss growth ratios"),
        ("all_losses", ["loss1", "loss2", "loss2_dict", "loss3_stopgrad", "loss3"], "loss comparison"),
        ("delta_norms", ["delta_norm_l2", "delta_norm_linf", "delta_budget_ratio"], "perturbation norms"),
        ("gradient_norms", ["gradient_norm_l2", "gradient_norm_linf"], "gradient norms"),
        ("timing", ["step_time", "forward_time", "backward_time", "solver_time", "model_time"], "timing"),
    ]
    for stem, keys, title in groups:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for key in keys:
            if key in rows[0]:
                ax.plot(k, [row.get(key, np.nan) for row in rows], label=key)
        ax.set_xlabel("attack step")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = output_dir / f"{stem}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        made.append(path)

    for npy_name, title in [
        ("estimated_singular_value", "estimated singular value"),
        ("rayleigh_quotient", "Rayleigh quotient"),
        ("jvp_norm", "JVP norm"),
        ("jtjv_norm", "J^T J v norm"),
    ]:
        path = run_dir / f"{npy_name}.npy"
        if not path.exists():
            continue
        values = np.load(path)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(np.arange(values.size), values, label=npy_name)
        ax.set_xlabel("power step")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        out = output_dir / f"{npy_name}.png"
        fig.savefig(out, dpi=160)
        plt.close(fig)
        made.append(out)
    return made


def plot_grouped_curves(
    run_dirs: list[Path],
    output_dir: Path,
    filename: str,
    curve_name: str,
    title: str,
    ylabel: str,
    labels: list[str] | None = None,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    labels = labels or [label_from_summary(run_dir) for run_dir in run_dirs]
    for run_dir, label in zip(run_dirs, labels):
        y = load_curve(run_dir, curve_name)
        ax.plot(np.arange(y.size), y, label=label, linewidth=1.8)
    ax.set_xlabel("attack step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = output_dir / filename
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def plot_final_fields(run_dir: Path, output_dir: Path | None = None) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = output_dir or (run_dir / "figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    names = [
        "x_clean",
        "x_adv",
        "delta_final",
        "model_output_clean",
        "solver_output_clean",
        "model_output_adv",
        "solver_output_adv",
        "difference_adv",
    ]
    arrays = {name: np.load(run_dir / f"{name}.npy") for name in names if (run_dir / f"{name}.npy").exists()}
    if not arrays:
        return []
    sample = next(iter(arrays.values()))
    made: list[Path] = []

    if sample.ndim <= 2 or (sample.ndim == 3 and sample.shape[-1] == 1):
        fig, axes = plt.subplots(len(arrays), 1, figsize=(9, 2.1 * len(arrays)), squeeze=False)
        for ax, (name, values) in zip(axes.ravel(), arrays.items()):
            flat = np.squeeze(values)
            ax.plot(np.arange(flat.size), flat.reshape(-1))
            ax.set_title(name)
            ax.grid(alpha=0.2)
        fig.tight_layout()
        path = output_dir / "spatial_1d_fields.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        made.append(path)
        return made

    ncols = min(4, len(arrays))
    nrows = int(math.ceil(len(arrays) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.4 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.set_visible(False)
    for ax, (name, values) in zip(axes.ravel(), arrays.items()):
        ax.set_visible(True)
        arr = np.squeeze(values)
        if arr.ndim == 3:
            arr = arr[..., -1]
        im = ax.imshow(arr, cmap="coolwarm" if "difference" in name or "delta" in name else "viridis")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    path = output_dir / "spatial_2d_fields.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    made.append(path)
    return made


def make_gif_from_trajectory(run_dir: Path, output_dir: Path | None = None, cleanup_png: bool = True) -> list[Path]:
    import imageio.v2 as imageio
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = output_dir or (run_dir / "figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    made: list[Path] = []
    names = ["model_output_adv", "solver_output_adv", "difference_adv"]
    for name in names:
        path = run_dir / f"{name}.npy"
        if not path.exists():
            continue
        arr = np.squeeze(np.load(path))
        if arr.ndim != 3:
            continue
        frame_dir = output_dir / f".{name}_frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        frame_paths: list[Path] = []
        vmin = float(np.min(arr))
        vmax = float(np.max(arr))
        cmap = "coolwarm" if "difference" in name else "viridis"
        for t in range(arr.shape[-1]):
            fig, ax = plt.subplots(figsize=(4.5, 4))
            im = ax.imshow(arr[..., t], cmap=cmap, vmin=vmin, vmax=vmax)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"{name} frame {t}")
            ax.set_xticks([])
            ax.set_yticks([])
            fig.tight_layout()
            frame_path = frame_dir / f"{name}_{t:03d}.png"
            fig.savefig(frame_path, dpi=130)
            plt.close(fig)
            frame_paths.append(frame_path)
        gif_path = output_dir / f"{name}.gif"
        imageio.mimsave(gif_path, [imageio.imread(frame) for frame in frame_paths], duration=0.25)
        made.append(gif_path)
        if cleanup_png:
            shutil.rmtree(frame_dir, ignore_errors=True)
    return made


def summary_row_from_run(run_dir: Path) -> dict[str, Any]:
    summary = load_json(run_dir / "summary.json")
    row = {key: summary.get(key, "") for key in SUMMARY_COLUMNS}
    for key in SUMMARY_COLUMNS:
        row.setdefault(key, "")
    return row
