#!/usr/bin/env python3
"""CPU-only overview plots for one completed NS2D recurrent attack pair root.

Reads per_step_metrics.csv, step_sample_trace.npz, and final_state_outputs.npz.
Does not import torch or jax and does not rerun model/solver.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METHODS = ["raw_add", "raw_replace", "steepest_add", "steepest_replace"]
METHOD_COLORS = {
    "raw_add": "tab:blue",
    "raw_replace": "tab:orange",
    "steepest_add": "tab:green",
    "steepest_replace": "tab:red",
}
THRESHOLDS = (0.25, 0.50, 0.75, 1.00)
THRESHOLD_MARKERS = {
    0.25: "o",
    0.50: "^",
    0.75: "s",
    1.00: "X",
}
MODE_NAMES = {
    "wwwwwwwwww": "all_w",
    "aaaaaaaaaw": "all_a_target_w",
    "dddddddddw": "all_d_target_w",
    "wwwwwddddw": "w1_5_d6_9_target_w",
    "dddddwwwww": "d1_5_w6_9_target_w",
    "aaaaaddddw": "a1_5_d6_9_target_w",
}
BLOCK_ORDER = [
    "loss1/all_w",
    "loss2/all_a_target_w",
    "loss3/all_w",
    "loss3/all_d_target_w",
    "loss3/w1_5_d6_9_target_w",
    "loss3/d1_5_w6_9_target_w",
    "loss3/a1_5_d6_9_target_w",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def f(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value in ("", None, "nan", "None"):
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def series(rows: list[dict[str, str]], key: str) -> np.ndarray:
    return np.asarray([f(row, key) for row in rows], dtype=np.float64)


def block_label_from_path(path: Path) -> tuple[str, str, str]:
    parts = path.parts
    mode_part = next(part for part in parts if part.startswith("mode_"))
    mode_spec = mode_part.split("_p2_q2", 1)[0][len("mode_"):]
    mode_name = MODE_NAMES.get(mode_spec, mode_spec)
    loss = next(part for part in parts if part in ("loss1", "loss2", "loss3"))
    return f"{loss}/{mode_name}", loss, mode_spec


def discover(pair_root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    blocks: dict[str, dict[str, dict[str, Any]]] = {}
    for csv_path in sorted(pair_root.glob("mode_*/batch_0000_0009/loss*/*/per_step_metrics.csv")):
        method = csv_path.parent.name
        if method not in METHODS:
            continue
        block, loss, mode_spec = block_label_from_path(csv_path)
        method_dir = csv_path.parent
        rows = read_rows(csv_path)
        if not rows:
            continue
        objective_key = f"{loss}_mean"
        objective_std_key = f"{loss}_std"
        blocks.setdefault(block, {})[method] = {
            "method": method,
            "method_dir": str(method_dir),
            "csv": str(csv_path),
            "loss": loss,
            "mode_spec": mode_spec,
            "rows": rows,
            "k": series(rows, "k"),
            "time_min": series(rows, "seconds_since_method_start") / 60.0,
            "objective_key": objective_key,
            "objective": series(rows, objective_key),
            "objective_std": series(rows, objective_std_key),
            "true": series(rows, "true_loss_mean"),
            "true_std": series(rows, "true_loss_std"),
            "delta_p": series(rows, "delta_p_mean"),
            "delta_p_std": series(rows, "delta_p_std"),
            "boundary": series(rows, "boundary_ratio_mean"),
            "boundary_std": series(rows, "delta_p_std") / max(float(series(rows, "epsilon")[-1]), 1e-12),
            "epsilon": float(series(rows, "epsilon")[-1]),
            "alpha": float(series(rows, "alpha")[-1]),
        }
    return blocks


def ordered_blocks(blocks: dict[str, Any]) -> list[str]:
    known = [b for b in BLOCK_ORDER if b in blocks]
    extra = sorted([b for b in blocks if b not in BLOCK_ORDER])
    return known + extra


def shade(ax, x: np.ndarray, y: np.ndarray, std: np.ndarray, color: str | None, alpha: float = 0.14) -> None:
    if std is None or not np.any(np.isfinite(std)):
        return
    lo = y - std
    hi = y + std
    mask = np.isfinite(x) & np.isfinite(lo) & np.isfinite(hi)
    if np.any(mask):
        ax.fill_between(x[mask], lo[mask], hi[mask], color=color, alpha=alpha, linewidth=0)


def padded_ylim(values: list[np.ndarray], pad: float = 0.05) -> tuple[float, float]:
    finite = []
    for arr in values:
        arr = np.asarray(arr, dtype=np.float64)
        finite.append(arr[np.isfinite(arr)])
    vals = np.concatenate([v for v in finite if v.size]) if any(v.size for v in finite) else np.asarray([0.0, 1.0])
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return 0.0, 1.0
    if hi <= lo:
        hi = lo + 1.0
    margin = (hi - lo) * pad
    return lo - margin, hi + margin


def first_threshold_indices(k: np.ndarray, boundary: np.ndarray, tol: float = 1e-6) -> dict[float, int | None]:
    out: dict[float, int | None] = {}
    for threshold in THRESHOLDS:
        mask = np.isfinite(boundary) & (boundary >= threshold - tol)
        out[threshold] = int(np.argmax(mask)) if np.any(mask) else None
    return out


def mark_thresholds(ax, d: dict[str, Any], y: np.ndarray, *, y_multiplier: float = 1.0) -> None:
    k = d["k"]
    color = METHOD_COLORS.get(str(d.get("method", "")), "black")
    idxs = first_threshold_indices(k, d["boundary"])
    for threshold, idx in idxs.items():
        if idx is None or idx >= len(k) or idx >= len(y):
            continue
        yy = y[idx] * y_multiplier
        if not (np.isfinite(k[idx]) and np.isfinite(yy)):
            continue
        ax.scatter(
            [k[idx]],
            [yy],
            marker=THRESHOLD_MARKERS[threshold],
            s=58,
            facecolors="none",
            edgecolors=color,
            linewidths=1.7,
            zorder=5,
        )


def threshold_legend_handles():
    from matplotlib.lines import Line2D
    return [
        Line2D([0], [0], marker=THRESHOLD_MARKERS[t], color="black", markerfacecolor="none", linestyle="None", label=f"{int(t*100)}%")
        for t in THRESHOLDS
    ]


def save_loss_curves(blocks: dict[str, Any], out_dir: Path) -> Path:
    labels = ordered_blocks(blocks)
    true_ranges = []
    for block in labels:
        for d in blocks[block].values():
            true_ranges.append(d["true"] - d["true_std"])
            true_ranges.append(d["true"] + d["true_std"])
    shared_true_ylim = padded_ylim(true_ranges, pad=0.04)

    fig, axes = plt.subplots(len(labels), 2, figsize=(16, 3.1 * len(labels)), constrained_layout=True)
    if len(labels) == 1:
        axes = np.asarray([axes])
    fig.suptitle("eps32_alpha10 baseline: objective and true loss curves (mean +/- std; markers = 25/50/75/100% boundary)", fontsize=14)
    for row, block in enumerate(labels):
        for method, d in blocks[block].items():
            color = METHOD_COLORS.get(method)
            k = d["k"]
            axes[row, 0].plot(k, d["objective"], color=color, label=method, linewidth=1.8)
            shade(axes[row, 0], k, d["objective"], d["objective_std"], color)
            mark_thresholds(axes[row, 0], d, d["objective"])
            axes[row, 1].plot(k, d["true"], color=color, label=method, linewidth=1.8)
            shade(axes[row, 1], k, d["true"], d["true_std"], color)
            mark_thresholds(axes[row, 1], d, d["true"])
        axes[row, 0].set_title(f"{block}: objective")
        axes[row, 1].set_title(f"{block}: true loss (shared y-axis)")
        axes[row, 1].set_ylim(*shared_true_ylim)
        for ax in axes[row]:
            ax.set_xlabel("attack step k")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, ncol=2, loc="upper left")
        axes[row, 1].legend(handles=threshold_legend_handles(), title="boundary", fontsize=7, loc="upper right")
    out = out_dir / "eps32_alpha10_baseline_loss_curves_with_std_shared_true_y_thresholds.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out

def save_delta_curves(blocks: dict[str, Any], out_dir: Path) -> Path:
    labels = ordered_blocks(blocks)
    max_eps = max(float(d["epsilon"]) for block in labels for d in blocks[block].values())
    fig, axes = plt.subplots(len(labels), 2, figsize=(16, 3.1 * len(labels)), constrained_layout=True)
    if len(labels) == 1:
        axes = np.asarray([axes])
    fig.suptitle("eps32_alpha10 baseline: delta norm and boundary ratio (mean +/- std; markers = 25/50/75/100% boundary)", fontsize=14)
    for row, block in enumerate(labels):
        for method, d in blocks[block].items():
            color = METHOD_COLORS.get(method)
            k = d["k"]
            axes[row, 0].plot(k, d["delta_p"], color=color, label=method, linewidth=1.8)
            shade(axes[row, 0], k, d["delta_p"], d["delta_p_std"], color)
            mark_thresholds(axes[row, 0], d, d["delta_p"])
            axes[row, 1].plot(k, d["boundary"], color=color, label=method, linewidth=1.8)
            shade(axes[row, 1], k, d["boundary"], d["boundary_std"], color)
            mark_thresholds(axes[row, 1], d, d["boundary"])
        axes[row, 0].set_title(f"{block}: ||delta||_p")
        axes[row, 1].set_title(f"{block}: ||delta||_p / epsilon")
        axes[row, 0].set_ylim(-0.02 * max_eps, 1.08 * max_eps)
        axes[row, 1].set_ylim(-0.02, 1.08)
        for threshold in THRESHOLDS:
            axes[row, 0].axhline(threshold * max_eps, color="gray", linestyle=":", linewidth=0.8)
            axes[row, 1].axhline(threshold, color="gray", linestyle=":", linewidth=0.8)
        axes[row, 1].axhline(1.0, color="black", linestyle="--", linewidth=1)
        for ax in axes[row]:
            ax.set_xlabel("attack step k")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, ncol=2, loc="lower right")
    out = out_dir / "eps32_alpha10_baseline_delta_norm_boundary_with_std_thresholds.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out

def flat(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(x.shape[0], -1)


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa = flat(a)
    bb = flat(b)
    num = np.sum(aa * bb, axis=1)
    den = np.linalg.norm(aa, axis=1) * np.linalg.norm(bb, axis=1)
    out = np.full_like(num, np.nan, dtype=np.float64)
    mask = den > 1e-12
    out[mask] = num[mask] / den[mask]
    return np.clip(out, -1.0, 1.0)


def angles_for_trace(path: Path) -> dict[str, np.ndarray]:
    z = np.load(path)
    k = z["k"].astype(np.float64)
    delta = z["delta"].astype(np.float32)
    direction = z["direction"].astype(np.float32)
    direction_available = z["direction_available"].astype(bool)
    cos_delta_direction = cosine(delta, direction)
    cos_delta_direction[~direction_available] = np.nan
    angle_delta_direction = np.degrees(np.arccos(cos_delta_direction))
    rotation = np.full_like(k, np.nan, dtype=np.float64)
    if len(delta) > 1:
        c = cosine(delta[1:], delta[:-1])
        rotation[1:] = np.degrees(np.arccos(c))
    return {
        "k": k,
        "angle_delta_direction": angle_delta_direction,
        "rotation_angle": rotation,
    }


def save_angle_curves(blocks: dict[str, Any], out_dir: Path) -> Path:
    labels = ordered_blocks(blocks)
    fig, axes = plt.subplots(len(labels), 2, figsize=(16, 3.1 * len(labels)), constrained_layout=True)
    if len(labels) == 1:
        axes = np.asarray([axes])
    fig.suptitle("eps32_alpha10 baseline: sample0 angle diagnostics", fontsize=14)
    for row, block in enumerate(labels):
        for method, d in blocks[block].items():
            trace = Path(d["method_dir"]) / "step_sample_trace.npz"
            if not trace.exists():
                continue
            a = angles_for_trace(trace)
            color = METHOD_COLORS.get(method)
            axes[row, 0].plot(a["k"], a["angle_delta_direction"], color=color, label=method, linewidth=1.8)
            axes[row, 1].plot(a["k"], a["rotation_angle"], color=color, label=method, linewidth=1.8)
        axes[row, 0].set_title(f"{block}: angle(delta, direction)")
        axes[row, 1].set_title(f"{block}: angle(delta_k, delta_(k-1))")
        for ax in axes[row]:
            ax.set_xlabel("attack step k")
            ax.set_ylabel("degrees")
            ax.set_ylim(0, 180)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, ncol=2)
    out = out_dir / "eps32_alpha10_baseline_angle_curves_sample0.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def sym_limits(arrays: list[np.ndarray], pct: float = 99.0) -> tuple[float, float]:
    vals = np.concatenate([np.asarray(a, dtype=np.float64).ravel() for a in arrays])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return -1.0, 1.0
    vmax = float(np.percentile(np.abs(vals), pct))
    if vmax <= 0:
        vmax = float(np.max(np.abs(vals))) if vals.size else 1.0
    if vmax <= 0:
        vmax = 1.0
    return -vmax, vmax


def save_final_delta_grid(blocks: dict[str, Any], out_dir: Path) -> Path:
    labels = ordered_blocks(blocks)
    arrays: dict[tuple[str, str], dict[str, Any]] = {}
    deltas = []
    for block in labels:
        for method, d in blocks[block].items():
            npz = Path(d["method_dir"]) / "final_state_outputs.npz"
            if not npz.exists():
                continue
            z = np.load(npz)
            delta = z["final_delta"][0].astype(np.float32)
            clean = float(z["clean_true_loss"][0])
            adv = float(z["adv_true_loss"][0])
            arrays[(block, method)] = {"delta": delta, "clean": clean, "adv": adv}
            deltas.append(delta)
    vlim = sym_limits(deltas, pct=99.0)
    fig, axes = plt.subplots(len(labels), len(METHODS), figsize=(4.0 * len(METHODS), 3.1 * len(labels)), constrained_layout=True)
    if len(labels) == 1:
        axes = np.asarray([axes])
    fig.suptitle("eps32_alpha10 baseline: final perturbation delta, sample position 0", fontsize=14)
    for i, block in enumerate(labels):
        for j, method in enumerate(METHODS):
            ax = axes[i, j]
            item = arrays.get((block, method))
            if not item:
                ax.axis("off")
                continue
            delta = item["delta"]
            im = ax.imshow(delta, cmap="coolwarm", origin="lower", vmin=vlim[0], vmax=vlim[1])
            l2 = float(np.linalg.norm(delta.reshape(-1)))
            linf = float(np.max(np.abs(delta)))
            inc = item["adv"] - item["clean"]
            ax.set_title(f"{block}\n{method}\nL2={l2:.2f}, Linf={linf:.3f}, trueΔ={inc:+.1f}", fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01)
    out = out_dir / "eps32_alpha10_baseline_final_delta_sample0_grid.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def save_final_metric_heatmaps(blocks: dict[str, Any], out_dir: Path) -> tuple[Path, list[dict[str, Any]]]:
    labels = ordered_blocks(blocks)
    true_inc = np.full((len(labels), len(METHODS)), np.nan)
    objective_inc = np.full_like(true_inc, np.nan)
    final_boundary = np.full_like(true_inc, np.nan)
    rows_out = []
    for i, block in enumerate(labels):
        for j, method in enumerate(METHODS):
            d = blocks[block].get(method)
            if not d:
                continue
            obj = d["objective"]
            true = d["true"]
            boundary = d["boundary"]
            true_inc[i, j] = float(true[-1] - true[0])
            objective_inc[i, j] = float(obj[-1] - obj[0])
            final_boundary[i, j] = float(boundary[-1])
            rows_out.append({
                "block": block,
                "method": method,
                "epsilon": d["epsilon"],
                "alpha": d["alpha"],
                "objective_initial": float(obj[0]),
                "objective_final": float(obj[-1]),
                "objective_increase": float(objective_inc[i, j]),
                "true_initial": float(true[0]),
                "true_final": float(true[-1]),
                "true_increase": float(true_inc[i, j]),
                "final_boundary_ratio": float(final_boundary[i, j]),
                "runtime_min": float(d["time_min"][-1]),
                "source_csv": d["csv"],
            })

    fig, axes = plt.subplots(1, 3, figsize=(18, max(6, 0.7 * len(labels))), constrained_layout=True)
    mats = [true_inc, objective_inc, final_boundary]
    titles = ["Final true loss increase", "Final objective increase", "Final boundary ratio"]
    cmaps = ["viridis", "magma", "cividis"]
    for ax, mat, title, cmap in zip(axes, mats, titles, cmaps):
        im = ax.imshow(mat, aspect="auto", cmap=cmap)
        ax.set_title(title)
        ax.set_xticks(range(len(METHODS)))
        ax.set_xticklabels(METHODS, rotation=35, ha="right")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if np.isfinite(mat[i, j]):
                    ax.text(j, i, f"{mat[i,j]:.1f}", ha="center", va="center", fontsize=7, color="white" if title != "Final boundary ratio" else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    out = out_dir / "eps32_alpha10_baseline_final_metrics_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out, rows_out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    blocks = discover(args.pair_root)
    if not blocks:
        raise SystemExit(f"No per_step_metrics.csv files found under {args.pair_root}")

    loss_png = save_loss_curves(blocks, args.out_dir)
    delta_png = save_delta_curves(blocks, args.out_dir)
    angle_png = save_angle_curves(blocks, args.out_dir)
    delta_grid_png = save_final_delta_grid(blocks, args.out_dir)
    heatmap_png, summary_rows = save_final_metric_heatmaps(blocks, args.out_dir)

    summary_csv = args.out_dir / "eps32_alpha10_baseline_final_metric_summary.csv"
    write_csv(summary_csv, summary_rows)
    report = {
        "note": "CPU-only plots from saved attack outputs; no model/solver rerun.",
        "pair_root": str(args.pair_root),
        "out_dir": str(args.out_dir),
        "blocks": ordered_blocks(blocks),
        "methods": METHODS,
        "loss_curves_png": str(loss_png),
        "delta_curves_png": str(delta_png),
        "angle_curves_png": str(angle_png),
        "final_delta_grid_png": str(delta_grid_png),
        "final_metrics_heatmap_png": str(heatmap_png),
        "summary_csv": str(summary_csv),
        "summary_rows": summary_rows,
    }
    report_path = args.out_dir / "eps32_alpha10_baseline_overview_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"[done] {loss_png}")
    print(f"[done] {delta_png}")
    print(f"[done] {angle_png}")
    print(f"[done] {delta_grid_png}")
    print(f"[done] {heatmap_png}")
    print(f"[done] {summary_csv}")
    print(f"[done] {report_path}")


if __name__ == "__main__":
    main()
