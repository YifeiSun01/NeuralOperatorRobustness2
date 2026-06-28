#!/usr/bin/env python3
"""CPU-only loss/delta curve plots from attack per_step_metrics.csv files.

Plots method-wise mean curves with shaded mean +/- std bands where std columns
are available. This script does not import torch or jax and does not touch GPU.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_METHODS = ["raw_add", "raw_replace", "steepest_add", "steepest_replace"]


def f(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value in ("", None, "nan", "None"):
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def series(rows: list[dict[str, str]], key: str) -> np.ndarray:
    return np.asarray([f(row, key) for row in rows], dtype=np.float64)


def finite_last(values: np.ndarray) -> float:
    vals = values[np.isfinite(values)]
    return float(vals[-1]) if vals.size else float("nan")


def first_crossing_step(x: np.ndarray, y: np.ndarray, threshold: float) -> int | None:
    mask = np.isfinite(y) & (y >= threshold)
    if not np.any(mask):
        return None
    return int(x[np.argmax(mask)])


def choose_objective_key(loss_name: str, rows: list[dict[str, str]]) -> str:
    preferred = f"{loss_name}_mean"
    if rows and preferred in rows[0]:
        return preferred
    if rows and "active_loss_mean" in rows[0]:
        return "active_loss_mean"
    return preferred


def std_key_for(mean_key: str) -> str | None:
    if mean_key.endswith("_mean"):
        return mean_key[:-5] + "_std"
    return None


def shade(ax, x: np.ndarray, mean: np.ndarray, std: np.ndarray, color, alpha: float = 0.16) -> None:
    if std is None or not np.any(np.isfinite(std)):
        return
    lo = mean - std
    hi = mean + std
    mask = np.isfinite(x) & np.isfinite(lo) & np.isfinite(hi)
    if np.any(mask):
        ax.fill_between(x[mask], lo[mask], hi[mask], color=color, alpha=alpha, linewidth=0)


def plot_overview(loss_root: Path, loss_name: str, methods: list[str], out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    data: dict[str, dict[str, np.ndarray | str | int | float | None]] = {}
    for method in methods:
        path = loss_root / method / "per_step_metrics.csv"
        if not path.exists():
            print(f"[skip] missing {path}")
            continue
        rows = read_rows(path)
        if not rows:
            continue
        objective_key = choose_objective_key(loss_name, rows)
        objective_std_key = std_key_for(objective_key)
        epsilon = finite_last(series(rows, "epsilon"))
        delta_p_std = series(rows, "delta_p_std")
        boundary_std = delta_p_std / epsilon if np.isfinite(epsilon) and epsilon != 0 else np.full_like(delta_p_std, np.nan)
        true_mean = series(rows, "true_loss_mean")
        true_std = series(rows, "true_loss_std")
        true_initial = true_mean[0] if true_mean.size else float("nan")
        data[method] = {
            "path": str(path),
            "objective_key": objective_key,
            "objective_std_key": objective_std_key or "",
            "k": series(rows, "k"),
            "time_min": series(rows, "seconds_since_method_start") / 60.0,
            "objective": series(rows, objective_key),
            "objective_std": series(rows, objective_std_key) if objective_std_key and objective_std_key in rows[0] else np.full(len(rows), np.nan),
            "active": series(rows, "active_loss_mean"),
            "surrogate": series(rows, "surrogate_loss_mean"),
            "true": true_mean,
            "true_std": true_std,
            "true_increase": true_mean - true_initial,
            "true_increase_std": true_std,
            "true_ratio": series(rows, "true_loss_mean_ratio_to_k0"),
            "delta_p": series(rows, "delta_p_mean"),
            "delta_p_std": delta_p_std,
            "boundary": series(rows, "boundary_ratio_mean"),
            "boundary_std": boundary_std,
            "epsilon": epsilon,
            "alpha": finite_last(series(rows, "alpha")),
        }

    if not data:
        raise SystemExit(f"No per_step_metrics.csv files found under {loss_root}")

    colors = {
        "raw_add": "tab:blue",
        "raw_replace": "tab:orange",
        "steepest_add": "tab:green",
        "steepest_replace": "tab:red",
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    fig.suptitle(f"{loss_name} curves with mean +/- std shading | {loss_root}", fontsize=12)

    for method, d in data.items():
        k = d["k"]
        color = colors.get(method, None)
        axes[0, 0].plot(k, d["objective"], label=f"{method}", color=color, linewidth=2)
        shade(axes[0, 0], k, d["objective"], d["objective_std"], color)

        axes[0, 1].plot(k, d["true"], label=f"{method}", color=color, linewidth=2)
        shade(axes[0, 1], k, d["true"], d["true_std"], color)

        axes[1, 0].plot(k, d["boundary"], label=f"{method}", color=color, linewidth=2)
        shade(axes[1, 0], k, d["boundary"], d["boundary_std"], color)

        axes[1, 1].plot(k, d["true_increase"], label=f"{method}", color=color, linewidth=2)
        shade(axes[1, 1], k, d["true_increase"], d["true_increase_std"], color)

    axes[0, 0].set_title(f"Objective / surrogate curve ({loss_name}_mean)")
    axes[0, 0].set_xlabel("attack step k")
    axes[0, 0].set_ylabel("mean objective")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].set_title("True loss: ||F(x+delta)-G(x+delta)||")
    axes[0, 1].set_xlabel("attack step k")
    axes[0, 1].set_ylabel("true_loss_mean")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].set_title("Delta boundary ratio: ||delta||_p / epsilon")
    axes[1, 0].set_xlabel("attack step k")
    axes[1, 0].set_ylabel("boundary_ratio_mean")
    axes[1, 0].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].set_title("True loss increase from k=0")
    axes[1, 1].set_xlabel("attack step k")
    axes[1, 1].set_ylabel("true_loss_mean - true_loss_mean@k0")
    axes[1, 1].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(fontsize=8)

    overview = out_dir / f"{loss_name}_loss_curve_overview_with_std.png"
    fig.savefig(overview, dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    fig.suptitle(f"{loss_name} curves versus wall time with mean +/- std shading", fontsize=12)
    for method, d in data.items():
        t = d["time_min"]
        color = colors.get(method, None)
        axes[0, 0].plot(t, d["objective"], label=method, color=color, linewidth=2)
        shade(axes[0, 0], t, d["objective"], d["objective_std"], color)
        axes[0, 1].plot(t, d["true"], label=method, color=color, linewidth=2)
        shade(axes[0, 1], t, d["true"], d["true_std"], color)
        axes[1, 0].plot(t, d["boundary"], label=method, color=color, linewidth=2)
        shade(axes[1, 0], t, d["boundary"], d["boundary_std"], color)
        axes[1, 1].plot(t, d["delta_p"], label=method, color=color, linewidth=2)
        shade(axes[1, 1], t, d["delta_p"], d["delta_p_std"], color)
    for ax in axes.ravel():
        ax.set_xlabel("minutes since method start")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[0, 0].set_title("Objective / surrogate")
    axes[0, 1].set_title("True loss")
    axes[1, 0].set_title("Boundary ratio")
    axes[1, 0].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[1, 1].set_title("Delta p-norm")
    time_png = out_dir / f"{loss_name}_loss_curve_wall_time_with_std.png"
    fig.savefig(time_png, dpi=150)
    plt.close(fig)

    summary_rows = []
    for method, d in data.items():
        k = d["k"]
        boundary = d["boundary"]
        objective = d["objective"]
        true = d["true"]
        delta_p = d["delta_p"]
        row = {
            "method": method,
            "epsilon": d["epsilon"],
            "alpha": d["alpha"],
            "objective_key": d["objective_key"],
            "objective_std_key": d["objective_std_key"],
            "steps_recorded": int(np.isfinite(k).sum()),
            "boundary_25_step": first_crossing_step(k, boundary, 0.25),
            "boundary_50_step": first_crossing_step(k, boundary, 0.50),
            "boundary_75_step": first_crossing_step(k, boundary, 0.75),
            "boundary_100_step": first_crossing_step(k, boundary, 1.00),
            "objective_initial": float(objective[0]),
            "objective_final": finite_last(objective),
            "objective_increase": finite_last(objective) - float(objective[0]),
            "objective_final_std": finite_last(d["objective_std"]),
            "true_initial": float(true[0]),
            "true_final": finite_last(true),
            "true_increase": finite_last(true) - float(true[0]),
            "true_final_std": finite_last(d["true_std"]),
            "true_ratio": finite_last(true) / float(true[0]) if float(true[0]) != 0 else float("nan"),
            "delta_p_final": finite_last(delta_p),
            "delta_p_final_std": finite_last(d["delta_p_std"]),
            "boundary_final": finite_last(boundary),
            "boundary_final_std": finite_last(d["boundary_std"]),
            "runtime_min": finite_last(d["time_min"]),
            "source_csv": d["path"],
        }
        summary_rows.append(row)

    report = {
        "loss_root": str(loss_root),
        "loss_name": loss_name,
        "note": "Curves plot method means with shaded mean +/- std bands from per_step_metrics.csv.",
        "overview_png": str(overview),
        "wall_time_png": str(time_png),
        "summary_rows": summary_rows,
    }
    report_path = out_dir / f"{loss_name}_loss_curve_report_with_std.json"
    report_path.write_text(json.dumps(report, indent=2))

    csv_path = out_dir / f"{loss_name}_loss_curve_summary_with_std.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"[done] {overview}")
    print(f"[done] {time_png}")
    print(f"[done] {report_path}")
    print(f"[done] {csv_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loss-root", type=Path, required=True)
    parser.add_argument("--loss-name", default="loss1")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    args = parser.parse_args()
    plot_overview(args.loss_root, args.loss_name, args.methods, args.out_dir)


if __name__ == "__main__":
    main()
