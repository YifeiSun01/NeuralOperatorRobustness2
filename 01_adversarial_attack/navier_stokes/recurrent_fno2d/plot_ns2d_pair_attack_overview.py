#!/usr/bin/env python3
"""CPU-only overview plots and tables for a completed NS2D pair-root attack.

Reads saved CSV/NPZ artifacts only. It does not import torch or jax and does
not rerun the model, solver, or attack.
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

METHODS = ["raw_add", "raw_replace", "steepest_add", "steepest_replace"]
METHOD_COLORS = {
    "raw_add": "tab:blue",
    "raw_replace": "tab:orange",
    "steepest_add": "tab:green",
    "steepest_replace": "tab:red",
}
METHOD_LABELS = {
    "raw_add": "raw_add / PGD",
    "raw_replace": "raw_replace",
    "steepest_add": "steepest_add / LP steepest PGD",
    "steepest_replace": "steepest_replace / GPI-style",
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
BLOCK_COLORS = {
    "loss1/all_w": "#1f77b4",
    "loss2/all_a_target_w": "#ff7f0e",
    "loss3/all_w": "#2ca02c",
    "loss3/all_d_target_w": "#d62728",
    "loss3/w1_5_d6_9_target_w": "#9467bd",
    "loss3/d1_5_w6_9_target_w": "#8c564b",
    "loss3/a1_5_d6_9_target_w": "#17becf",
}
THRESHOLDS = (0.25, 0.50, 0.75, 1.00)
THRESHOLD_MARKERS = {0.25: "o", 0.50: "^", 0.75: "s", 1.00: "X"}
SPECTRAL_FIELDS = [
    ("final_delta", "final delta"),
    ("adv_model_final", "adv FNO final"),
    ("adv_solver_final", "adv solver final"),
    ("adv_model_minus_solver", "adv FNO - solver"),
]
NBINS = 96
RADIAL_AXIS_CUTOFF = np.sqrt(2.0) / 3.0
RADIAL_CORNER_CUTOFF = 2.0 / 3.0


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


def infer_run_label(pair_root: Path) -> str:
    name = pair_root.name
    if name:
        return name
    return "ns2d_pair"


def block_from_path(path: Path) -> tuple[str, str, str]:
    mode_part = next(part for part in path.parts if part.startswith("mode_"))
    mode_spec = mode_part.split("_p2_q2", 1)[0][len("mode_"):]
    mode = MODE_NAMES.get(mode_spec, mode_spec)
    loss = next(part for part in path.parts if part in ("loss1", "loss2", "loss3"))
    return f"{loss}/{mode}", loss, mode_spec


def ordered_blocks(blocks: dict[str, Any]) -> list[str]:
    return [b for b in BLOCK_ORDER if b in blocks] + sorted([b for b in blocks if b not in BLOCK_ORDER])


def discover(pair_root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    blocks: dict[str, dict[str, dict[str, Any]]] = {}
    for csv_path in sorted(pair_root.glob("mode_*/batch_0000_0009/loss*/*/per_step_metrics.csv")):
        method = csv_path.parent.name
        if method not in METHODS:
            continue
        block, loss, mode_spec = block_from_path(csv_path)
        rows = read_rows(csv_path)
        if not rows:
            continue
        method_dir = csv_path.parent
        objective_key = f"{loss}_mean"
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
            "objective_std": series(rows, f"{loss}_std"),
            "true": series(rows, "true_loss_mean"),
            "true_std": series(rows, "true_loss_std"),
            "delta_p": series(rows, "delta_p_mean"),
            "delta_p_std": series(rows, "delta_p_std"),
            "boundary": series(rows, "boundary_ratio_mean"),
            "epsilon": float(series(rows, "epsilon")[-1]),
            "alpha": float(series(rows, "alpha")[-1]),
        }
    return blocks


def discover_final_outputs(pair_root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    blocks: dict[str, dict[str, dict[str, Any]]] = {}
    for npz_path in sorted(pair_root.glob("mode_*/batch_0000_0009/loss*/*/final_state_outputs.npz")):
        method = npz_path.parent.name
        if method not in METHODS:
            continue
        block, loss, mode_spec = block_from_path(npz_path)
        z = np.load(npz_path)
        item: dict[str, Any] = {"npz": str(npz_path), "loss": loss, "mode_spec": mode_spec, "method": method}
        for field, _label in SPECTRAL_FIELDS:
            if field in z.files:
                item[field] = z[field].astype(np.float32)
        if "clean_true_loss" in z.files:
            item["clean_true_loss"] = z["clean_true_loss"].astype(np.float32)
        if "adv_true_loss" in z.files:
            item["adv_true_loss"] = z["adv_true_loss"].astype(np.float32)
        blocks.setdefault(block, {})[method] = item
    return blocks


def shade(ax, x: np.ndarray, y: np.ndarray, std: np.ndarray, color: str | None, alpha: float = 0.12) -> None:
    if std is None:
        return
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(std)
    if np.any(mask):
        ax.fill_between(x[mask], y[mask] - std[mask], y[mask] + std[mask], color=color or "gray", alpha=alpha, linewidth=0)


def padded_ylim(values: list[np.ndarray], pad: float = 0.05) -> tuple[float, float]:
    vals = []
    for arr in values:
        arr = np.asarray(arr, dtype=np.float64)
        vals.append(arr[np.isfinite(arr)])
    vals = np.concatenate([v for v in vals if v.size]) if any(v.size for v in vals) else np.asarray([0.0, 1.0])
    lo, hi = float(np.min(vals)), float(np.max(vals))
    if hi <= lo:
        hi = lo + 1.0
    m = (hi - lo) * pad
    return lo - m, hi + m


def first_threshold_indices(k: np.ndarray, boundary: np.ndarray) -> dict[float, int | None]:
    out: dict[float, int | None] = {}
    for threshold in THRESHOLDS:
        mask = np.isfinite(boundary) & (boundary >= threshold - 1e-6)
        out[threshold] = int(np.argmax(mask)) if np.any(mask) else None
    return out


def mark_thresholds(ax, d: dict[str, Any], y: np.ndarray) -> None:
    k = d["k"]
    color = METHOD_COLORS.get(d["method"], "black")
    for threshold, idx in first_threshold_indices(k, d["boundary"]).items():
        if idx is None or idx >= len(k) or idx >= len(y):
            continue
        if np.isfinite(k[idx]) and np.isfinite(y[idx]):
            ax.scatter([k[idx]], [y[idx]], marker=THRESHOLD_MARKERS[threshold], s=55,
                       facecolors="none", edgecolors=color, linewidths=1.6, zorder=5)


def threshold_legend_handles():
    from matplotlib.lines import Line2D
    return [Line2D([0], [0], marker=THRESHOLD_MARKERS[t], color="black", markerfacecolor="none", linestyle="None", label=f"{int(t*100)}%") for t in THRESHOLDS]


def flat(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(x.shape[0], -1)


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa, bb = flat(a), flat(b)
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
    cd = cosine(delta, direction)
    cd[~direction_available] = np.nan
    angle_delta_direction = np.degrees(np.arccos(cd))
    rotation = np.full_like(k, np.nan, dtype=np.float64)
    if len(delta) > 1:
        rotation[1:] = np.degrees(np.arccos(cosine(delta[1:], delta[:-1])))
    return {"k": k, "angle_delta_direction": angle_delta_direction, "rotation_angle": rotation}


def sym_limits(arrays: list[np.ndarray], pct: float = 99.0) -> tuple[float, float]:
    if not arrays:
        return -1.0, 1.0
    vals = np.concatenate([np.asarray(a, dtype=np.float64).ravel() for a in arrays])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return -1.0, 1.0
    vmax = float(np.percentile(np.abs(vals), pct))
    if vmax <= 0:
        vmax = float(np.max(np.abs(vals))) if vals.size else 1.0
    return -max(vmax, 1e-12), max(vmax, 1e-12)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_loss_curves(blocks: dict[str, Any], out_dir: Path, run_label: str) -> Path:
    labels = ordered_blocks(blocks)
    true_ranges = []
    for block in labels:
        for d in blocks[block].values():
            true_ranges += [d["true"] - d["true_std"], d["true"] + d["true_std"]]
    shared_true_ylim = padded_ylim(true_ranges, pad=0.04)
    fig, axes = plt.subplots(len(labels), 2, figsize=(16, 3.1 * len(labels)), constrained_layout=True)
    if len(labels) == 1:
        axes = np.asarray([axes])
    fig.suptitle(f"{run_label}: objective and true loss curves (mean +/- std; markers = 25/50/75/100% boundary)", fontsize=14)
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
        axes[row, 0].set_title(f"{block}: active objective")
        axes[row, 1].set_title(f"{block}: true loss (shared y-axis)")
        axes[row, 1].set_ylim(*shared_true_ylim)
        for ax in axes[row]:
            ax.set_xlabel("attack step k")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, ncol=2, loc="upper left")
        axes[row, 1].legend(handles=threshold_legend_handles(), title="boundary", fontsize=7, loc="upper right")
    out = out_dir / f"{run_label}_loss_curves_with_std_shared_true_y_thresholds.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def save_delta_curves(blocks: dict[str, Any], out_dir: Path, run_label: str) -> Path:
    labels = ordered_blocks(blocks)
    max_eps = max(float(d["epsilon"]) for block in labels for d in blocks[block].values())
    fig, axes = plt.subplots(len(labels), 2, figsize=(16, 3.1 * len(labels)), constrained_layout=True)
    if len(labels) == 1:
        axes = np.asarray([axes])
    fig.suptitle(f"{run_label}: delta norm and boundary ratio (mean +/- std)", fontsize=14)
    for row, block in enumerate(labels):
        for method, d in blocks[block].items():
            color = METHOD_COLORS.get(method)
            k = d["k"]
            boundary_std = d["delta_p_std"] / max(float(d["epsilon"]), 1e-12)
            axes[row, 0].plot(k, d["delta_p"], color=color, label=method, linewidth=1.8)
            shade(axes[row, 0], k, d["delta_p"], d["delta_p_std"], color)
            mark_thresholds(axes[row, 0], d, d["delta_p"])
            axes[row, 1].plot(k, d["boundary"], color=color, label=method, linewidth=1.8)
            shade(axes[row, 1], k, d["boundary"], boundary_std, color)
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
    out = out_dir / f"{run_label}_delta_norm_boundary_with_std_thresholds.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def save_angle_curves(blocks: dict[str, Any], out_dir: Path, run_label: str) -> Path:
    labels = ordered_blocks(blocks)
    fig, axes = plt.subplots(len(labels), 2, figsize=(16, 3.1 * len(labels)), constrained_layout=True)
    if len(labels) == 1:
        axes = np.asarray([axes])
    fig.suptitle(f"{run_label}: sample0 angle diagnostics", fontsize=14)
    for row, block in enumerate(labels):
        for method, d in blocks[block].items():
            trace = Path(d["method_dir"]) / "step_sample_trace.npz"
            if not trace.exists():
                continue
            a = angles_for_trace(trace)
            color = METHOD_COLORS.get(method)
            axes[row, 0].plot(a["k"], a["angle_delta_direction"], color=color, label=method, linewidth=1.8)
            axes[row, 1].plot(a["k"], a["rotation_angle"], color=color, label=method, linewidth=1.8)
        axes[row, 0].set_title(f"{block}: angle(delta, update direction)")
        axes[row, 1].set_title(f"{block}: angle(delta_k, delta_(k-1))")
        for ax in axes[row]:
            ax.set_xlabel("attack step k")
            ax.set_ylabel("degrees")
            ax.set_ylim(0, 180)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, ncol=2)
    out = out_dir / f"{run_label}_angle_curves_sample0.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def save_final_delta_grid(blocks: dict[str, Any], out_dir: Path, run_label: str) -> Path:
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
            arrays[(block, method)] = {
                "delta": delta,
                "clean": float(z["clean_true_loss"][0]) if "clean_true_loss" in z.files else float("nan"),
                "adv": float(z["adv_true_loss"][0]) if "adv_true_loss" in z.files else float("nan"),
            }
            deltas.append(delta)
    vlim = sym_limits(deltas, pct=99.0)
    fig, axes = plt.subplots(len(labels), len(METHODS), figsize=(4.0 * len(METHODS), 3.1 * len(labels)), constrained_layout=True)
    if len(labels) == 1:
        axes = np.asarray([axes])
    fig.suptitle(f"{run_label}: final perturbation delta, sample position 0", fontsize=14)
    im = None
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
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01)
    out = out_dir / f"{run_label}_final_delta_sample0_grid.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def save_final_metric_heatmaps(blocks: dict[str, Any], out_dir: Path, run_label: str) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
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
            obj, true, boundary = d["objective"], d["true"], d["boundary"]
            true_inc[i, j] = float(true[-1] - true[0])
            objective_inc[i, j] = float(obj[-1] - obj[0])
            final_boundary[i, j] = float(boundary[-1])
            rows_out.append({
                "block": block,
                "loss": d["loss"],
                "mode_spec": d["mode_spec"],
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
    titles = ["Final true loss increase", "Final active-objective increase", "Final boundary ratio"]
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
                    ax.text(j, i, f"{mat[i,j]:.1f}", ha="center", va="center", fontsize=7,
                            color="white" if title != "Final boundary ratio" else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    out = out_dir / f"{run_label}_final_metrics_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)

    def row_max(key: str) -> dict[str, Any] | None:
        valid = [r for r in rows_out if np.isfinite(float(r[key]))]
        return max(valid, key=lambda r: float(r[key])) if valid else None

    best_by_block = []
    for block in labels:
        rows = [r for r in rows_out if r["block"] == block]
        if rows:
            best_by_block.append({
                "block": block,
                "best_true_increase": max(rows, key=lambda r: float(r["true_increase"])),
                "best_objective_increase": max(rows, key=lambda r: float(r["objective_increase"])),
            })
    method_counts: dict[str, int] = {m: 0 for m in METHODS}
    for item in best_by_block:
        method_counts[item["best_true_increase"]["method"]] += 1
    winners = {
        "overall_best_true_increase": row_max("true_increase"),
        "overall_best_objective_increase": row_max("objective_increase"),
        "best_by_block": best_by_block,
        "best_true_increase_method_counts_by_block": method_counts,
    }
    return out, rows_out, winners


def save_loss_curves_by_method(blocks: dict[str, Any], out_dir: Path, run_label: str, *, logy: bool) -> Path:
    labels = ordered_blocks(blocks)
    fig, axes = plt.subplots(len(METHODS), 2, figsize=(16, 3.15 * len(METHODS)), constrained_layout=True)
    fig.suptitle(f"{run_label}: curves grouped by optimizer; all loss/mode blocks overlaid" + (" (log y)" if logy else " (linear y)"), fontsize=14)
    for row, method in enumerate(METHODS):
        for block in labels:
            item = blocks.get(block, {}).get(method)
            if not item:
                continue
            color = BLOCK_COLORS.get(block)
            k = item["k"]
            axes[row, 0].plot(k, item["objective"], color=color, label=block, linewidth=1.8)
            shade(axes[row, 0], k, item["objective"], item["objective_std"], color, alpha=0.10)
            axes[row, 1].plot(k, item["true"], color=color, label=block, linewidth=1.8)
            shade(axes[row, 1], k, item["true"], item["true_std"], color, alpha=0.10)
        axes[row, 0].set_title(f"{METHOD_LABELS[method]}: active objective")
        axes[row, 1].set_title(f"{METHOD_LABELS[method]}: true all-W final loss")
        for ax in axes[row]:
            ax.set_xlabel("attack step k")
            ax.grid(True, alpha=0.28)
            if logy:
                ax.set_yscale("log")
            ax.legend(fontsize=7, ncol=2, loc="best")
    suffix = "logy" if logy else "linear"
    out = out_dir / f"{run_label}_loss_curves_by_method_all_blocks_{suffix}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def frequency_grid(n: int) -> np.ndarray:
    freq = np.fft.fftshift(np.fft.fftfreq(n))
    fx, fy = np.meshgrid(freq, freq, indexing="xy")
    return np.sqrt(fx * fx + fy * fy) / np.sqrt(0.5 * 0.5 + 0.5 * 0.5)


def fft_power(arr: np.ndarray) -> np.ndarray:
    f = np.fft.fftshift(np.fft.fft2(arr, axes=(-2, -1)), axes=(-2, -1))
    return np.abs(f) ** 2


def radial_profile(power: np.ndarray, rho: np.ndarray, nbins: int = NBINS) -> tuple[np.ndarray, np.ndarray]:
    bins = np.linspace(0.0, 1.0, nbins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    flat_rho = rho.reshape(-1)
    profiles = []
    for sample_power in power:
        flat = sample_power.reshape(-1)
        prof = np.zeros(nbins, dtype=np.float64)
        for i in range(nbins):
            mask = (flat_rho >= bins[i]) & (flat_rho < bins[i + 1])
            prof[i] = np.mean(flat[mask]) if np.any(mask) else np.nan
        total = np.nansum(prof)
        if total > 0:
            prof = prof / total
        profiles.append(prof)
    return centers, np.asarray(profiles)


def save_radial_spectrum_by_method(final_blocks: dict[str, Any], field: str, label: str, out_dir: Path, run_label: str) -> Path | None:
    labels = ordered_blocks(final_blocks)
    candidates = [item for block in final_blocks.values() for item in block.values() if field in item]
    if not candidates:
        return None
    rho = frequency_grid(int(candidates[0][field].shape[-1]))
    fig, axes = plt.subplots(len(METHODS), 1, figsize=(11.5, 3.0 * len(METHODS)), constrained_layout=True)
    fig.suptitle(f"{run_label} {label}: radial FFT spectra grouped by optimizer", fontsize=14)
    for row, method in enumerate(METHODS):
        ax = axes[row]
        for block in labels:
            item = final_blocks.get(block, {}).get(method)
            if not item or field not in item:
                continue
            centers, profs = radial_profile(fft_power(item[field]), rho)
            mean = np.nanmean(profs, axis=0)
            std = np.nanstd(profs, axis=0)
            color = BLOCK_COLORS.get(block)
            ax.plot(centers, mean, color=color, label=block, linewidth=1.7)
            ax.fill_between(centers, mean - std, mean + std, color=color, alpha=0.08, linewidth=0)
        ax.axvline(RADIAL_AXIS_CUTOFF, color="black", linestyle=":", linewidth=1.2, alpha=0.85, label="axis cutoff rho=sqrt(2)/3" if row == 0 else None)
        ax.axvline(RADIAL_CORNER_CUTOFF, color="black", linestyle="--", linewidth=1.2, alpha=0.85, label="box-corner rho=2/3" if row == 0 else None)
        ax.set_yscale("log")
        ax.set_title(METHOD_LABELS[method])
        ax.set_ylabel("normalized radial power")
        ax.grid(True, alpha=0.28)
        ax.legend(fontsize=7, ncol=2, loc="best")
    axes[-1].set_xlabel("normalized radial frequency rho")
    safe = field.replace("_", "-")
    out = out_dir / f"{run_label}_{safe}_radial_fft_by_method_all_blocks.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--run-label", default=None)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_label = args.run_label or infer_run_label(args.pair_root)

    blocks = discover(args.pair_root)
    if not blocks:
        raise SystemExit(f"No per_step_metrics.csv files found under {args.pair_root}")
    final_blocks = discover_final_outputs(args.pair_root)

    pngs: list[str] = []
    pngs.append(str(save_loss_curves(blocks, args.out_dir, run_label)))
    pngs.append(str(save_delta_curves(blocks, args.out_dir, run_label)))
    pngs.append(str(save_angle_curves(blocks, args.out_dir, run_label)))
    pngs.append(str(save_final_delta_grid(blocks, args.out_dir, run_label)))
    heatmap_png, summary_rows, winners = save_final_metric_heatmaps(blocks, args.out_dir, run_label)
    pngs.append(str(heatmap_png))
    pngs.append(str(save_loss_curves_by_method(blocks, args.out_dir, run_label, logy=False)))
    pngs.append(str(save_loss_curves_by_method(blocks, args.out_dir, run_label, logy=True)))
    for field, label in SPECTRAL_FIELDS:
        out = save_radial_spectrum_by_method(final_blocks, field, label, args.out_dir, run_label)
        if out:
            pngs.append(str(out))

    summary_csv = args.out_dir / f"{run_label}_final_metric_summary.csv"
    write_csv(summary_csv, summary_rows)
    winners_json = args.out_dir / f"{run_label}_winners.json"
    winners_json.write_text(json.dumps(winners, indent=2))
    report = {
        "note": "CPU-only plots/tables from saved attack outputs; no model/solver rerun.",
        "pair_root": str(args.pair_root),
        "out_dir": str(args.out_dir),
        "run_label": run_label,
        "blocks": ordered_blocks(blocks),
        "methods": METHODS,
        "pngs": pngs,
        "summary_csv": str(summary_csv),
        "winners_json": str(winners_json),
        "winners": winners,
    }
    report_path = args.out_dir / f"{run_label}_overview_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
