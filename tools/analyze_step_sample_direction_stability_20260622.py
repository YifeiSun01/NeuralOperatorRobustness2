#!/usr/bin/env python3
"""Analyze per-step mechanism traces for optimizer direction stability.

The NS2D mechanism runner writes one ``step_sample_trace.npz`` per
batch/method.  This script turns those arrays into scalar diagnostics that test
whether replacement behaves like a stable power-style iteration:

- early perturbation direction vs final perturbation direction;
- update/gradient direction rotation across steps;
- boundary arrival time and post-boundary true-loss gain.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


METHOD_ORDER = ["raw_add", "raw_replace", "steepest_add", "steepest_replace"]


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def vector_norm(arr: np.ndarray) -> float:
    flat = np.asarray(arr, dtype=np.float64).ravel()
    return float(np.linalg.norm(flat))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    af = np.asarray(a, dtype=np.float64).ravel()
    bf = np.asarray(b, dtype=np.float64).ravel()
    an = np.linalg.norm(af)
    bn = np.linalg.norm(bf)
    if an <= 0.0 or bn <= 0.0:
        return float("nan")
    return float(np.dot(af, bf) / (an * bn))


def read_csv_by_k(path: Path) -> dict[int, dict[str, str]]:
    if not path.exists():
        return {}
    rows: dict[int, dict[str, str]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                rows[int(row["k"])] = row
            except (KeyError, ValueError):
                continue
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_summary(method_dir: Path) -> dict[str, Any]:
    summary_path = method_dir / "summary.json"
    if not summary_path.exists():
        return {}
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def parse_trace_metadata(trace_path: Path) -> dict[str, str]:
    method_dir = trace_path.parent
    return {
        "trace_path": str(trace_path),
        "method": method_dir.name,
        "loss_type": method_dir.parent.name if method_dir.parent else "",
        "batch": method_dir.parent.parent.name if method_dir.parent.parent else "",
        "run_dir": method_dir.parent.parent.parent.name if method_dir.parent.parent.parent else "",
    }


def nearest_value(rows: list[dict[str, Any]], key: str, target_k: int) -> float:
    candidates = [row for row in rows if math.isfinite(finite_float(row.get(key)))]
    if not candidates:
        return float("nan")
    best = min(candidates, key=lambda row: abs(int(row["k"]) - target_k))
    return finite_float(best.get(key))


def first_row_at_boundary(rows: list[dict[str, Any]], threshold: float = 0.99) -> dict[str, Any] | None:
    for row in rows:
        ratio = finite_float(row.get("boundary_ratio"))
        if math.isfinite(ratio) and ratio >= threshold:
            return row
    return None


def analyze_trace(trace_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    method_dir = trace_path.parent
    metadata = parse_trace_metadata(trace_path)
    summary = load_summary(method_dir)
    metrics = read_csv_by_k(method_dir / "step_sample_trace_metrics.csv")
    if not metrics:
        metrics = read_csv_by_k(method_dir / "per_sample_step_metrics.csv")

    with np.load(trace_path, allow_pickle=False) as data:
        k_values = np.asarray(data["k"], dtype=np.int64)
        delta = np.asarray(data["delta"], dtype=np.float64)
        direction = np.asarray(data["direction"], dtype=np.float64) if "direction" in data else np.zeros_like(delta)
        grad = np.asarray(data["grad"], dtype=np.float64) if "grad" in data else np.zeros_like(delta)
        direction_available = (
            np.asarray(data["direction_available"], dtype=bool)
            if "direction_available" in data
            else np.ones(len(k_values), dtype=bool)
        )
        grad_available = (
            np.asarray(data["grad_available"], dtype=bool)
            if "grad_available" in data
            else np.ones(len(k_values), dtype=bool)
        )
        dataset_index = int(np.asarray(data.get("dataset_index", -1)).item())
        sample_position = int(np.asarray(data.get("sample_position", 0)).item())

    epsilon = finite_float(summary.get("epsilon"))
    alpha = finite_float(summary.get("alpha"))
    final_delta = delta[-1]
    final_delta_norm = vector_norm(final_delta)

    rows: list[dict[str, Any]] = []
    for i, k in enumerate(k_values):
        metric = metrics.get(int(k), {})
        delta_norm = vector_norm(delta[i])
        boundary_ratio = finite_float(metric.get("boundary_ratio"))
        if not math.isfinite(boundary_ratio) and math.isfinite(epsilon) and epsilon > 0.0:
            boundary_ratio = delta_norm / epsilon

        row: dict[str, Any] = {
            **metadata,
            "dataset_index": dataset_index,
            "sample_position": sample_position,
            "k": int(k),
            "epsilon": epsilon,
            "alpha": alpha,
            "delta_l2": delta_norm,
            "delta_l2_fraction_of_epsilon": delta_norm / epsilon if math.isfinite(epsilon) and epsilon > 0.0 else float("nan"),
            "boundary_ratio": boundary_ratio,
            "true_loss": finite_float(metric.get("true_loss")),
            "loss3": finite_float(metric.get("loss3")),
            "surrogate_loss": finite_float(metric.get("surrogate_loss_value", metric.get("active_loss_value"))),
            "seconds_since_method_start": finite_float(metric.get("seconds_since_method_start")),
            "cos_delta_to_final_delta": cosine(delta[i], final_delta),
            "direction_available": bool(direction_available[i]),
            "grad_available": bool(grad_available[i]),
            "direction_l2": vector_norm(direction[i]) if direction_available[i] else float("nan"),
            "grad_l2": vector_norm(grad[i]) if grad_available[i] else float("nan"),
            "cos_direction_to_final_delta": cosine(direction[i], final_delta) if direction_available[i] else float("nan"),
            "cos_grad_to_final_delta": cosine(grad[i], final_delta) if grad_available[i] else float("nan"),
        }
        if i > 0:
            row["cos_delta_to_prev_delta"] = cosine(delta[i], delta[i - 1])
            row["cos_direction_to_prev_direction"] = (
                cosine(direction[i], direction[i - 1])
                if direction_available[i] and direction_available[i - 1]
                else float("nan")
            )
            row["cos_grad_to_prev_grad"] = (
                cosine(grad[i], grad[i - 1]) if grad_available[i] and grad_available[i - 1] else float("nan")
            )
        else:
            row["cos_delta_to_prev_delta"] = float("nan")
            row["cos_direction_to_prev_direction"] = float("nan")
            row["cos_grad_to_prev_grad"] = float("nan")
        if i + 1 < len(k_values) and direction_available[i]:
            row["cos_direction_to_actual_delta_step"] = cosine(direction[i], delta[i + 1] - delta[i])
        else:
            row["cos_direction_to_actual_delta_step"] = float("nan")
        rows.append(row)

    first_true = next((finite_float(row["true_loss"]) for row in rows if math.isfinite(finite_float(row["true_loss"]))), float("nan"))
    for row in rows:
        true_loss = finite_float(row.get("true_loss"))
        row["true_loss_increase_from_k0"] = true_loss - first_true if math.isfinite(true_loss) and math.isfinite(first_true) else float("nan")

    boundary_row = first_row_at_boundary(rows, threshold=0.99)
    final_row = rows[-1]
    final_true = finite_float(final_row.get("true_loss"))
    boundary_true = finite_float(boundary_row.get("true_loss")) if boundary_row is not None else float("nan")
    post_boundary_gain = final_true - boundary_true if math.isfinite(final_true) and math.isfinite(boundary_true) else float("nan")

    direction_prev_values = [finite_float(row.get("cos_direction_to_prev_direction")) for row in rows]
    direction_prev_values = [v for v in direction_prev_values if math.isfinite(v)]
    grad_prev_values = [finite_float(row.get("cos_grad_to_prev_grad")) for row in rows]
    grad_prev_values = [v for v in grad_prev_values if math.isfinite(v)]

    summary_row: dict[str, Any] = {
        **metadata,
        "dataset_index": dataset_index,
        "sample_position": sample_position,
        "epsilon": epsilon,
        "alpha": alpha,
        "num_recorded_steps": len(rows),
        "final_k": int(final_row["k"]),
        "final_delta_l2": final_delta_norm,
        "final_true_loss": final_true,
        "final_loss3": finite_float(final_row.get("loss3")),
        "first_boundary99_k": int(boundary_row["k"]) if boundary_row is not None else "",
        "true_loss_at_boundary99": boundary_true,
        "post_boundary_true_loss_gain": post_boundary_gain,
        "mean_cos_direction_to_prev_direction": float(np.nanmean(direction_prev_values)) if direction_prev_values else float("nan"),
        "mean_one_minus_cos_direction_to_prev_direction": float(np.nanmean([1.0 - v for v in direction_prev_values])) if direction_prev_values else float("nan"),
        "mean_cos_grad_to_prev_grad": float(np.nanmean(grad_prev_values)) if grad_prev_values else float("nan"),
        "mean_one_minus_cos_grad_to_prev_grad": float(np.nanmean([1.0 - v for v in grad_prev_values])) if grad_prev_values else float("nan"),
    }
    for target_k in (1, 5, 10, 20, 50, 100):
        summary_row[f"cos_delta_final_near_k{target_k}"] = nearest_value(rows, "cos_delta_to_final_delta", target_k)
        summary_row[f"cos_direction_final_near_k{target_k}"] = nearest_value(rows, "cos_direction_to_final_delta", target_k)
        summary_row[f"true_loss_near_k{target_k}"] = nearest_value(rows, "true_loss", target_k)
        summary_row[f"boundary_ratio_near_k{target_k}"] = nearest_value(rows, "boundary_ratio", target_k)

    return rows, summary_row


def method_sort_key(method: str) -> tuple[int, str]:
    try:
        return METHOD_ORDER.index(method), method
    except ValueError:
        return len(METHOD_ORDER), method


def aggregate_for_plot(rows: list[dict[str, Any]], metric: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    buckets: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        value = finite_float(row.get(metric))
        if math.isfinite(value):
            buckets[(str(row["method"]), int(row["k"]))].append(value)
    by_method: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for (method, k), values in buckets.items():
        by_method[method].append((k, float(np.mean(values))))
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for method, pairs in by_method.items():
        pairs.sort()
        out[method] = (np.asarray([p[0] for p in pairs]), np.asarray([p[1] for p in pairs]))
    return out


def plot_diagnostics(rows: list[dict[str, Any]], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    metrics = [
        ("cos_delta_to_final_delta", r"$\cos(\delta_k,\delta_T)$"),
        ("cos_direction_to_final_delta", r"$\cos(d_k,\delta_T)$"),
        ("cos_direction_to_prev_direction", r"$\cos(d_k,d_{k-1})$"),
        ("true_loss", "true Loss3"),
        ("boundary_ratio", r"$\|\delta_k\|/\epsilon$"),
        ("true_loss_increase_from_k0", "true Loss3 increase"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.2), constrained_layout=True)
    for ax, (metric, ylabel) in zip(axes.ravel(), metrics):
        series = aggregate_for_plot(rows, metric)
        for method in sorted(series, key=method_sort_key):
            x, y = series[method]
            linestyle = "--" if method.endswith("replace") else "-"
            ax.plot(x, y, label=method, linewidth=2.0, linestyle=linestyle)
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle("Step-Sample Direction Stability Diagnostics", fontsize=14)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("analysis_outputs/mechanism_20260622"),
        help="Root to search recursively for step_sample_trace.npz files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("analysis_outputs/mechanism_20260622/diagnostics"),
        help="Directory for CSV and plot outputs.",
    )
    args = parser.parse_args()

    trace_paths = sorted(args.root.rglob("step_sample_trace.npz"))
    if not trace_paths:
        print(f"[direction-stability] no step_sample_trace.npz files found under {args.root}")
        return

    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for trace_path in trace_paths:
        rows, summary = analyze_trace(trace_path)
        detail_rows.extend(rows)
        summary_rows.append(summary)

    detail_csv = args.out_dir / "step_sample_direction_diagnostics.csv"
    summary_csv = args.out_dir / "step_sample_direction_summary.csv"
    plot_path = args.out_dir / "step_sample_direction_diagnostics.png"
    write_csv(detail_csv, detail_rows)
    write_csv(summary_csv, summary_rows)
    try:
        plot_diagnostics(detail_rows, plot_path)
        plot_msg = f", plot={plot_path}"
    except Exception as exc:  # pragma: no cover - plotting is best-effort.
        plot_msg = f", plot skipped: {exc!r}"
    print(
        f"[direction-stability] traces={len(trace_paths)} rows={len(detail_rows)} "
        f"summary={summary_csv} detail={detail_csv}{plot_msg}"
    )


if __name__ == "__main__":
    main()
