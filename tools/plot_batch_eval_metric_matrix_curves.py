#!/usr/bin/env python3
"""Plot all 27 three-loss attack curves using a shared evaluation metric.

The original loss-curve plotting scripts draw each run's own optimized
objective.  This script keeps the same 3 losses x 3 objective variants x
3 methods layout, but fixes the y-axis to one recorded evaluation key such
as ``loss3_original`` or ``loss3_increment_ratio``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LOSSES = ("loss1", "loss2", "loss3")
VARIANTS = ("original", "increment_ratio", "regularized")
METHODS = ("pgd", "lp_steepest_pgd", "generalized_power")
EVAL_KEYS = tuple(f"{loss}_{variant}" for loss in LOSSES for variant in VARIANTS)

METHOD_COLORS = {
    "pgd": "#2563eb",
    "lp_steepest_pgd": "#dc2626",
    "generalized_power": "#059669",
}
METHOD_LABELS = {
    "pgd": "PGD",
    "lp_steepest_pgd": "LP-steepest PGD",
    "generalized_power": "GPI",
}
LOSS_LABELS = {
    "loss1": "optimize loss1",
    "loss2": "optimize loss2",
    "loss3": "optimize loss3",
}
VARIANT_LABELS = {
    "original": "original objective",
    "increment_ratio": "increment ratio",
    "regularized": "regularized",
}
EVAL_LABELS = {
    "loss1_original": r"$L_1(\delta)=\|f(x+\delta)-f(x)\|$",
    "loss1_increment_ratio": r"$(L_1(\delta)-L_1(0))/(\|\delta\|+\eta)$",
    "loss1_regularized": r"$L_1(\delta)-C\|\delta\|$",
    "loss2_original": r"$L_2(\delta)=\|f(x+\delta)-g(x)\|$",
    "loss2_increment_ratio": r"$(L_2(\delta)-L_2(0))/(\|\delta\|+\eta)$",
    "loss2_regularized": r"$L_2(\delta)-C\|\delta\|$",
    "loss3_original": r"$L_3(\delta)=\|f(x+\delta)-g(x+\delta)\|$",
    "loss3_increment_ratio": r"$(L_3(\delta)-L_3(0))/(\|\delta\|+\eta)$",
    "loss3_regularized": r"$L_3(\delta)-C\|\delta\|$",
}


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out: list[dict[str, Any]] = []
    for row in rows:
        converted: dict[str, Any] = {}
        for key, value in row.items():
            try:
                converted[key] = float(value)
            except (TypeError, ValueError):
                converted[key] = value
        out.append(converted)
    return out


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def load_config(root: Path) -> dict[str, Any]:
    path = root / "config.json"
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def config_summary(config: dict[str, Any]) -> str:
    if not config:
        return "config not found"
    start = int(config.get("start_index", 0))
    batch = int(config.get("batch_size", 0))
    end = start + batch - 1
    model = config.get("model_label") or config.get("model_kind", "model")
    return (
        f"{model}, Burgers nu={config.get('burgers_nu', '?')}, "
        f"batch={batch} index={start}-{end}, "
        f"epsilon={config.get('epsilon', '?')}, alpha={config.get('alpha', '?')}, "
        f"steps={config.get('steps', '?')}, p={config.get('p_order', config.get('p', '?'))}, "
        f"q={config.get('q_order', config.get('q', '?'))}"
    )


def load_runs(root: Path) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    runs: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for path in sorted(root.rglob("loss_stats.csv")):
        rows = read_csv(path)
        if not rows:
            continue
        first = rows[0]
        loss = str(first.get("optimized_loss"))
        variant = str(first.get("objective_variant"))
        method = str(first.get("attack_method"))
        if loss in LOSSES and variant in VARIANTS and method in METHODS:
            runs[(loss, variant, method)] = rows
    return runs


def load_npz_run(root: Path, optimized_loss: str, variant: str, method: str) -> tuple[Path, Any] | None:
    run_dir = root / f"{optimized_loss}_{variant}_{method}"
    values_path = run_dir / "loss_values.npz"
    if not values_path.exists():
        return None
    return values_path, np.load(values_path)


def mask_label(label: str, k: np.ndarray, mask: np.ndarray, nonfinite_count: np.ndarray | None = None) -> str:
    if nonfinite_count is not None:
        bad = np.isfinite(nonfinite_count) & (nonfinite_count > 0)
        if bad.any():
            first = int(np.flatnonzero(bad)[0])
            max_bad = int(np.nanmax(nonfinite_count[bad]))
            return f"{label} (nonfinite after k={k[first]:g}, max n={max_bad})"
    if mask.all():
        return label
    bad = np.flatnonzero(~mask)
    return f"{label} (NaN after k={k[bad[0]]:g})" if bad.size else label



def shared_mean_ylim(runs: dict[tuple[str, str, str], list[dict[str, Any]]], eval_key: str, zero_ymin: bool = True) -> tuple[float, float]:
    lows: list[np.ndarray] = []
    highs: list[np.ndarray] = []
    for rows in runs.values():
        mean = np.asarray([finite_float(row.get(f"{eval_key}_mean")) for row in rows], dtype=np.float64)
        std = np.asarray([finite_float(row.get(f"{eval_key}_std")) for row in rows], dtype=np.float64)
        mask = np.isfinite(mean) & np.isfinite(std)
        if mask.any():
            lows.append(mean[mask] - std[mask])
            highs.append(mean[mask] + std[mask])
    if not lows:
        raise RuntimeError(f"No finite data found for eval_key={eval_key}")
    y_min = 0.0 if zero_ymin else float(np.min(np.concatenate(lows)))
    y_max = float(np.max(np.concatenate(highs)))
    pad = max(0.05 * (y_max - y_min), 1.0 if y_min == y_max else 0.0)
    return (0.0, y_max + pad) if zero_ymin else (y_min - pad, y_max + pad)


def shared_index_ylim(root: Path, eval_key: str, dataset_index: int, zero_ymin: bool = True) -> tuple[float, float]:
    config = load_config(root)
    start = int(config.get("start_index", 0))
    sample_position = dataset_index - start
    lows: list[np.ndarray] = []
    highs: list[np.ndarray] = []
    for optimized_loss in LOSSES:
        for variant in VARIANTS:
            for method in METHODS:
                loaded = load_npz_run(root, optimized_loss, variant, method)
                if loaded is None:
                    continue
                _, data = loaded
                if eval_key not in data or sample_position >= data[eval_key].shape[1]:
                    continue
                y = data[eval_key][:, sample_position].astype(np.float64)
                mask = np.isfinite(y)
                if mask.any():
                    lows.append(y[mask])
                    highs.append(y[mask])
    if not lows:
        raise RuntimeError(f"No finite data found for eval_key={eval_key}, dataset_index={dataset_index}")
    y_min = 0.0 if zero_ymin else float(np.min(np.concatenate(lows)))
    y_max = float(np.max(np.concatenate(highs)))
    pad = max(0.05 * (y_max - y_min), 1.0 if y_min == y_max else 0.0)
    return (0.0, y_max + pad) if zero_ymin else (y_min - pad, y_max + pad)

def plot_mean_matrix(root: Path, eval_key: str, output_dir: Path, dpi: int, zero_ymin: bool = True) -> Path:
    runs = load_runs(root)
    if not runs:
        raise RuntimeError(f"No loss_stats.csv files found under {root}")
    y_min, y_max = shared_mean_ylim(runs, eval_key, zero_ymin=zero_ymin)
    fig, axes = plt.subplots(3, 3, figsize=(18, 13.5), sharex=True, sharey=True)
    fig.suptitle(
        f"All 27 attack trajectories evaluated by {eval_key}\n"
        f"{EVAL_LABELS[eval_key]}\n"
        f"Batch mean with +/- 1 std over samples | {config_summary(load_config(root))}",
        fontsize=15,
        fontweight="bold",
    )
    any_line = False
    for row_idx, optimized_loss in enumerate(LOSSES):
        for col_idx, variant in enumerate(VARIANTS):
            ax = axes[row_idx, col_idx]
            for method in METHODS:
                rows = runs.get((optimized_loss, variant, method))
                if not rows:
                    continue
                k = np.asarray([finite_float(row.get("k")) for row in rows], dtype=np.float64)
                mean = np.asarray([finite_float(row.get(f"{eval_key}_mean")) for row in rows], dtype=np.float64)
                std = np.asarray([finite_float(row.get(f"{eval_key}_std")) for row in rows], dtype=np.float64)
                nonfinite = np.asarray(
                    [finite_float(row.get(f"{eval_key}_nonfinite_count"), 0.0) for row in rows],
                    dtype=np.float64,
                )
                mask = np.isfinite(k) & np.isfinite(mean) & np.isfinite(std)
                if not mask.any():
                    continue
                any_line = True
                color = METHOD_COLORS[method]
                ax.plot(
                    k,
                    np.ma.masked_where(~mask, mean),
                    color=color,
                    linewidth=2.0,
                    label=mask_label(METHOD_LABELS[method], k, mask, nonfinite),
                )
                ax.fill_between(k, mean - std, mean + std, where=mask, color=color, alpha=0.16, linewidth=0)
            ax.set_title(f"{LOSS_LABELS[optimized_loss]} | {VARIANT_LABELS[variant]}", fontsize=11, loc="left")
            ax.grid(True, color="#d4d4d8", linewidth=0.8, alpha=0.75)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if col_idx == 0:
                ax.set_ylabel(eval_key)
            ax.set_ylim(y_min, y_max)
            if row_idx == 2:
                ax.set_xlabel("optimization step k")
            ax.legend(loc="best", fontsize=8, frameon=True)
    if not any_line:
        raise RuntimeError(f"No finite data found for eval_key={eval_key}")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"{eval_key}_all_27_mean_std.png"
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def plot_index_matrix(root: Path, eval_key: str, dataset_index: int, output_dir: Path, dpi: int, zero_ymin: bool = True) -> Path:
    config = load_config(root)
    start = int(config.get("start_index", 0))
    sample_position = dataset_index - start
    if sample_position < 0:
        raise ValueError(f"dataset_index={dataset_index} is before start_index={start}")
    y_min, y_max = shared_index_ylim(root, eval_key, dataset_index, zero_ymin=zero_ymin)
    fig, axes = plt.subplots(3, 3, figsize=(18, 13.5), sharex=True, sharey=True)
    fig.suptitle(
        f"All 27 attack trajectories evaluated by {eval_key} | dataset index {dataset_index}\n"
        f"{EVAL_LABELS[eval_key]}\n"
        f"{config_summary(config)}",
        fontsize=15,
        fontweight="bold",
    )
    any_line = False
    for row_idx, optimized_loss in enumerate(LOSSES):
        for col_idx, variant in enumerate(VARIANTS):
            ax = axes[row_idx, col_idx]
            for method in METHODS:
                loaded = load_npz_run(root, optimized_loss, variant, method)
                if loaded is None:
                    continue
                _, data = loaded
                if eval_key not in data:
                    continue
                values = data[eval_key]
                if sample_position >= values.shape[1]:
                    raise IndexError(f"dataset_index={dataset_index} outside batch range")
                k = data["k"].astype(np.float64)
                y = values[:, sample_position].astype(np.float64)
                mask = np.isfinite(k) & np.isfinite(y)
                if not mask.any():
                    continue
                any_line = True
                color = METHOD_COLORS[method]
                ax.plot(
                    k,
                    np.ma.masked_where(~mask, y),
                    color=color,
                    linewidth=2.0,
                    label=mask_label(METHOD_LABELS[method], k, mask),
                )
            ax.set_title(f"{LOSS_LABELS[optimized_loss]} | {VARIANT_LABELS[variant]}", fontsize=11, loc="left")
            ax.grid(True, color="#d4d4d8", linewidth=0.8, alpha=0.75)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if col_idx == 0:
                ax.set_ylabel(eval_key)
            if row_idx == 2:
                ax.set_xlabel("optimization step k")
            ax.legend(loc="best", fontsize=8, frameon=True)
    if not any_line:
        raise RuntimeError(f"No finite data found for eval_key={eval_key}, dataset_index={dataset_index}")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"{eval_key}_all_27_index{dataset_index}.png"
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def parse_eval_keys(values: list[str]) -> list[str]:
    if len(values) == 1 and values[0] == "all":
        return list(EVAL_KEYS)
    bad = [value for value in values if value not in EVAL_KEYS]
    if bad:
        raise SystemExit(f"Unknown eval key(s): {bad}. Choices: {', '.join(EVAL_KEYS)}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--eval-keys", nargs="+", default=["loss3_original"])
    parser.add_argument("--dataset-indices", nargs="*", type=int, default=[0])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=170)
    parser.add_argument(
        "--allow-negative-ymin",
        action="store_true",
        help="Use the true global lower bound instead of forcing the shared y-axis to start at 0.",
    )
    args = parser.parse_args()

    eval_keys = parse_eval_keys(args.eval_keys)
    base = args.output_dir or args.root / "figures" / "eval_metric_curves_shared_y_zero"
    zero_ymin = not args.allow_negative_ymin
    for eval_key in eval_keys:
        mean_path = plot_mean_matrix(args.root, eval_key, base / "png", args.dpi, zero_ymin=zero_ymin)
        print(mean_path)
        for dataset_index in args.dataset_indices:
            index_path = plot_index_matrix(args.root, eval_key, dataset_index, base / "index_png", args.dpi, zero_ymin=zero_ymin)
            print(index_path)


if __name__ == "__main__":
    main()
