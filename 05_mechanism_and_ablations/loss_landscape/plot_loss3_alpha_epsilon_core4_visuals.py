#!/usr/bin/env python3
"""Visualize the loss3 alpha/epsilon core-four sweep.

The figures produced here are intentionally tied to the existing sweep
artifacts.  They do not rerun any experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SWEEP = PROJECT_ROOT / "forensics" / "loss3_alpha_epsilon_core4_sweep_20260519"
DEFAULT_ANALYSIS = PROJECT_ROOT / "forensics" / "loss3_alpha_epsilon_core4_analysis_20260519"
DEFAULT_OUT = PROJECT_ROOT / "forensics" / "loss3_alpha_epsilon_core4_visuals_20260520"
DEFAULT_DOC = PROJECT_ROOT / "docs" / "loss3_alpha_epsilon_core4_visuals_20260520.md"
CORE4 = ("raw_add", "raw_replace", "steepest_add", "steepest_replace")
COLORS = {
    "raw_add": "#1f77b4",
    "raw_replace": "#d62728",
    "steepest_add": "#2ca02c",
    "steepest_replace": "#9467bd",
}
LINESTYLES = {
    "raw_add": "-",
    "raw_replace": "--",
    "steepest_add": "-.",
    "steepest_replace": ":",
}
METHOD_LABELS = {
    "raw_add": "raw add",
    "raw_replace": "raw replace",
    "steepest_add": "steepest add",
    "steepest_replace": "steepest replace",
}
BOUNDARY_MARKERS = (
    (0.25, "o", "25% boundary"),
    (0.50, "^", "50% boundary"),
    (0.75, "s", "75% boundary"),
    (0.99, "x", "99% boundary"),
)
REPRESENTATIVE_SETTINGS = ((4.0, 0.4), (8.0, 0.3), (8.0, 1.6), (16.0, 1.6))


def metric_grid_cols(n_panels: int) -> int:
    if n_panels >= 20:
        return 5
    if n_panels >= 12:
        return 4
    return min(3, max(1, n_panels))


def fnum(value) -> float:
    if value in {None, "", "nan", "NaN"}:
        return math.nan
    try:
        return float(value)
    except Exception:
        return math.nan


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def setting_key(epsilon: float, alpha: float) -> tuple[float, float]:
    return (round(float(epsilon), 10), round(float(alpha), 10))


def setting_label(epsilon: float, alpha: float) -> str:
    return f"eps={epsilon:g}, alpha={alpha:g}"


def setting_label_short(epsilon: float, alpha: float) -> str:
    return f"e{epsilon:g}/a{alpha:g}"


def setting_slug(epsilon: float, alpha: float) -> str:
    def tag(value: float) -> str:
        return f"{value:g}".replace("-", "m").replace(".", "p")

    return f"eps{tag(epsilon)}_alpha{tag(alpha)}"


def load_completed_roots(sweep_root: Path) -> list[Path]:
    manifest_path = sweep_root / "sweep_manifest.json"
    roots: list[Path] = []
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item in payload.get("completed", []):
            root = Path(item.get("out_root", ""))
            if not root.is_absolute():
                root = PROJECT_ROOT / root
            if (root / "per_step_metrics.csv").exists():
                roots.append(root)
    if not roots:
        roots = sorted(p.parent for p in sweep_root.glob("*/per_step_metrics.csv"))

    def sort_key(root: Path) -> tuple[float, float, str]:
        eps, alpha = root_setting(root)
        return (eps, alpha, root.name)

    return sorted(dict.fromkeys(roots), key=sort_key)


def root_setting(root: Path) -> tuple[float, float]:
    for name in ("config.json", "manifest.json"):
        path = root / name
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if "epsilon" in payload and "alpha" in payload:
                return float(payload["epsilon"]), float(payload["alpha"])
    rows = read_csv(root / "per_step_metrics.csv")
    return float(rows[0]["epsilon"]), float(rows[0]["alpha"])


def root_pq(root: Path) -> tuple[str, str]:
    for name in ("config.json", "manifest.json"):
        path = root / name
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            p_order = payload.get("p_order", payload.get("p"))
            q_order = payload.get("q_order", payload.get("q"))
            if p_order is not None and q_order is not None:
                return str(p_order), str(q_order)
    rows = read_csv(root / "per_step_metrics.csv")
    return str(rows[0].get("p_order", "unknown")), str(rows[0].get("q_order", "unknown"))


def grouped_per_step(root: Path) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(root / "per_step_metrics.csv"):
        groups[row["method"]].append(row)
    for method in groups:
        groups[method].sort(key=lambda r: int(r["k"]))
    return groups


def first_boundary_step(rows: list[dict[str, str]], threshold: float) -> int | None:
    for row in rows:
        value = fnum(row.get("boundary_ratio_mean"))
        if math.isfinite(value) and value >= threshold:
            return int(row["k"])
    return None


def row_at_step(rows: list[dict[str, str]], step: int) -> dict[str, str] | None:
    for row in rows:
        if int(row["k"]) == step:
            return row
    return None


def plot_metric_curves(
    roots: list[Path],
    metric: str,
    ylabel: str,
    out_path: Path,
    boundary_threshold: float,
    mark_boundary: bool = False,
    shade_std: bool = True,
) -> None:
    cols = metric_grid_cols(len(roots))
    rows_count = math.ceil(len(roots) / cols)
    fig, axes = plt.subplots(rows_count, cols, figsize=(5.2 * cols, 3.8 * rows_count), squeeze=False)
    for ax, root in zip(axes.flat, roots):
        epsilon, alpha = root_setting(root)
        groups = grouped_per_step(root)
        for method in CORE4:
            if method not in groups:
                continue
            group = groups[method]
            xs = np.array([int(r["k"]) for r in group], dtype=float)
            ys = np.array([fnum(r.get(metric)) for r in group], dtype=float)
            if not np.isfinite(ys).any():
                continue
            ax.plot(
                xs,
                ys,
                label=METHOD_LABELS[method],
                color=COLORS[method],
                linestyle=LINESTYLES[method],
                linewidth=1.8,
            )
            if shade_std and metric.endswith("_mean"):
                std_key = metric[: -len("_mean")] + "_std"
                std = np.array([fnum(r.get(std_key)) for r in group], dtype=float)
                finite = np.isfinite(ys) & np.isfinite(std)
                if finite.any():
                    ax.fill_between(
                        xs[finite],
                        ys[finite] - std[finite],
                        ys[finite] + std[finite],
                        color=COLORS[method],
                        alpha=0.12,
                        linewidth=0,
                    )
            if mark_boundary:
                for threshold, marker, _label in BOUNDARY_MARKERS:
                    hit = first_boundary_step(group, threshold)
                    if hit is None:
                        continue
                    hit_row = row_at_step(group, hit)
                    if hit_row is None:
                        continue
                    y_hit = fnum(hit_row.get(metric))
                    if not math.isfinite(y_hit):
                        continue
                    if marker == "x":
                        ax.scatter(
                            [hit],
                            [y_hit],
                            marker=marker,
                            s=82,
                            color=COLORS[method],
                            linewidths=2.2,
                            zorder=5,
                        )
                    else:
                        ax.scatter(
                            [hit],
                            [y_hit],
                            marker=marker,
                            s=58,
                            facecolors="none",
                            edgecolors=COLORS[method],
                            linewidths=1.5,
                            zorder=5,
                        )
        if metric == "boundary_ratio_mean":
            ax.axhline(boundary_threshold, color="#666666", linestyle="--", linewidth=0.9, alpha=0.65)
            ax.set_ylim(-0.02, 1.05)
        ax.set_title(setting_label(epsilon, alpha), fontsize=11, pad=8)
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25, linewidth=0.7)
    for ax in axes.flat[len(roots) :]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=len(CORE4),
        frameon=True,
        fontsize=10,
        borderaxespad=0.0,
    )
    bottom = 0.085
    title = f"{ylabel} by alpha/epsilon setting"
    if metric.endswith("_mean"):
        title += " (mean +/- std)" if shade_std else " (mean only, no std band)"
    if mark_boundary:
        marker_handles = [
            Line2D(
                [0],
                [0],
                marker=marker,
                linestyle="None",
                color="#333333",
                markerfacecolor="none" if marker != "x" else "#333333",
                markeredgecolor="#333333",
                markeredgewidth=1.4,
                markersize=6.5,
                label=label,
            )
            for _threshold, marker, label in BOUNDARY_MARKERS
        ]
        fig.legend(
            marker_handles,
            [h.get_label() for h in marker_handles],
            loc="lower center",
            bbox_to_anchor=(0.5, 0.047),
            ncol=len(marker_handles),
            frameon=True,
            fontsize=9,
            borderaxespad=0.0,
        )
        bottom = 0.125
        title = f"{ylabel}: markers show first mean boundary-ratio hits"
        if metric.endswith("_mean"):
            title += " (mean +/- std)" if shade_std else " (mean only, no std band)"
    fig.suptitle(title, y=0.985, fontsize=15)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.925, bottom=bottom, hspace=0.46, wspace=0.28)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def root_has_metric(root: Path, metric: str) -> bool:
    for row in read_csv(root / "per_step_metrics.csv"):
        if metric in row and math.isfinite(fnum(row.get(metric))):
            return True
    return False


def has_metric(roots: list[Path], metric: str) -> bool:
    return any(root_has_metric(root, metric) for root in roots)


def angle_axis_limit(values: list[float]) -> tuple[float, float]:
    finite = np.array([v for v in values if math.isfinite(v)], dtype=float)
    if finite.size == 0:
        return (0.0, 10.0)
    top = float(np.nanmax(finite))
    if top <= 0:
        return (0.0, 10.0)
    # Round up to readable 10-degree ticks with a little headroom.
    ymax = math.ceil((top * 1.08) / 10.0) * 10.0
    ymax = max(10.0, min(180.0, ymax))
    return (0.0, ymax)


def plot_dynamics_triptych(root: Path, out_path: Path, shade_std: bool = True) -> None:
    epsilon, alpha = root_setting(root)
    groups = grouped_per_step(root)
    panels = [
        ("loss3_q_mean", "loss3 q mean", None),
        ("boundary_ratio_mean", r"$||delta||_p / epsilon$", (0.0, 1.05)),
        ("delta_prev_angle_degrees_mean", "angle(delta_k, delta_{k-1}) degrees", "auto_angle"),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(10.5, 8.3), sharex=True)
    for ax, (metric, ylabel, ylim) in zip(axes, panels):
        plotted = False
        angle_ylim_values: list[float] = []
        for method in CORE4:
            if method not in groups:
                continue
            group = groups[method]
            xs = np.array([int(r["k"]) for r in group], dtype=float)
            ys = np.array([fnum(r.get(metric)) for r in group], dtype=float)
            if not np.isfinite(ys).any():
                continue
            plotted = True
            ax.plot(xs, ys, label=METHOD_LABELS[method], color=COLORS[method], linestyle=LINESTYLES[method], linewidth=1.9)
            if metric == "delta_prev_angle_degrees_mean":
                angle_ylim_values.extend(float(v) for v in ys[np.isfinite(ys)])
            if shade_std and metric.endswith("_mean"):
                std_key = metric[: -len("_mean")] + "_std"
                std = np.array([fnum(r.get(std_key)) for r in group], dtype=float)
                finite = np.isfinite(ys) & np.isfinite(std)
                if finite.any():
                    ax.fill_between(xs[finite], ys[finite] - std[finite], ys[finite] + std[finite], color=COLORS[method], alpha=0.14, linewidth=0)
                    if metric == "delta_prev_angle_degrees_mean":
                        angle_ylim_values.extend(float(v) for v in (ys[finite] + std[finite]) if math.isfinite(float(v)))
        if metric == "boundary_ratio_mean":
            for threshold, _marker, _label in BOUNDARY_MARKERS:
                ax.axhline(threshold, color="#777777", linestyle="--", linewidth=0.6, alpha=0.30)
        if ylim == "auto_angle" and plotted:
            ax.set_ylim(*angle_axis_limit(angle_ylim_values))
        elif ylim is not None and ylim != "auto_angle":
            ax.set_ylim(*ylim)
        if not plotted:
            ax.text(
                0.5,
                0.5,
                f"{metric} was not recorded for this setting",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=10,
                color="#666666",
            )
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25, linewidth=0.7)
    axes[-1].set_xlabel("step")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(handles), frameon=True, fontsize=9)
    band_label = "mean +/- std" if shade_std else "mean only, no std band"
    fig.suptitle(f"Loss, boundary ratio, and delta angular speed: {setting_label(epsilon, alpha)} ({band_label})", y=0.985, fontsize=14)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.93, bottom=0.09, hspace=0.28)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def summary_matrix(summary_rows: list[dict[str, str]], metric: str) -> tuple[list[str], np.ndarray]:
    settings = sorted(
        {
            setting_key(row["epsilon"], row["alpha"])
            for row in summary_rows
        }
    )
    labels = [setting_label_short(eps, alpha) for eps, alpha in settings]
    matrix = np.full((len(CORE4), len(settings)), np.nan, dtype=float)
    by_key = {
        (row["method"], setting_key(row["epsilon"], row["alpha"])): row
        for row in summary_rows
    }
    for i, method in enumerate(CORE4):
        for j, key in enumerate(settings):
            row = by_key.get((method, key))
            if row is not None:
                matrix[i, j] = fnum(row.get(metric))
    return labels, matrix


def plot_heatmap(
    summary_rows: list[dict[str, str]],
    metric: str,
    title: str,
    out_path: Path,
    cmap: str = "viridis",
    log10: bool = False,
    fmt: str = ".2g",
) -> None:
    labels, matrix = summary_matrix(summary_rows, metric)
    plot_matrix = np.array(matrix, copy=True)
    annotation_matrix = np.array(matrix, copy=True)
    if log10:
        plot_matrix = np.where(plot_matrix > 0, np.log10(plot_matrix), np.nan)
    fig, ax = plt.subplots(figsize=(max(15.5, 1.55 * len(labels) + 4.0), 5.8))
    im = ax.imshow(plot_matrix, aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=42, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(CORE4)))
    ax.set_yticklabels([METHOD_LABELS[m] for m in CORE4], fontsize=10)
    ax.set_title(title, fontsize=15, pad=18)
    finite_plot = plot_matrix[np.isfinite(plot_matrix)]
    threshold = float(np.nanmean(finite_plot)) if finite_plot.size else math.nan
    for i in range(annotation_matrix.shape[0]):
        for j in range(annotation_matrix.shape[1]):
            value = annotation_matrix[i, j]
            plot_value = plot_matrix[i, j]
            text_color = "black" if math.isfinite(plot_value) and math.isfinite(threshold) and plot_value > threshold else "white"
            if math.isfinite(value):
                ax.text(j, i, format(value, fmt), ha="center", va="center", fontsize=8, color=text_color)
            else:
                ax.text(j, i, "nan", ha="center", va="center", fontsize=8, color="black")
    fig.subplots_adjust(left=0.105, right=0.865, top=0.82, bottom=0.28)
    cax = fig.add_axes([0.895, 0.28, 0.018, 0.54])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(("log10 " if log10 else "") + metric, fontsize=10, labelpad=10)
    cbar.ax.tick_params(labelsize=9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def finite_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else math.nan


def finite_std(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.std()) if values.size else math.nan


def peakiness(delta: np.ndarray) -> np.ndarray:
    flat = np.asarray(delta, dtype=float).reshape(delta.shape[0], -1)
    rms = np.sqrt(np.mean(flat * flat, axis=1))
    peak = np.max(np.abs(flat), axis=1)
    return peak / np.maximum(rms, 1e-12)


def total_variation(delta: np.ndarray) -> np.ndarray:
    flat = np.asarray(delta, dtype=float).reshape(delta.shape[0], -1)
    return np.sum(np.abs(np.diff(flat, axis=1)), axis=1)


def derivative_l2(delta: np.ndarray) -> np.ndarray:
    flat = np.asarray(delta, dtype=float).reshape(delta.shape[0], -1)
    diff = np.diff(flat, axis=1)
    return np.sqrt(np.sum(diff * diff, axis=1))


def high_frequency_ratio(delta: np.ndarray, cutoff_fraction: float = 0.25) -> np.ndarray:
    flat = np.asarray(delta, dtype=float).reshape(delta.shape[0], -1)
    spectrum = np.fft.rfft(flat, axis=1)
    power = np.abs(spectrum) ** 2
    cutoff = int(math.ceil(cutoff_fraction * power.shape[1]))
    total = power.sum(axis=1)
    high = power[:, cutoff:].sum(axis=1)
    return high / np.maximum(total, 1e-30)


def compute_boundary_threshold_summary(roots: list[Path], out_csv: Path, legacy_99_csv: Path | None = None) -> list[dict]:
    rows: list[dict] = []
    for root in roots:
        epsilon, alpha = root_setting(root)
        groups = grouped_per_step(root)
        for method in CORE4:
            group = groups.get(method, [])
            if not group:
                continue
            final = group[-1]
            final_loss = fnum(final.get("loss3_q_mean"))
            final_ratio = fnum(final.get("boundary_ratio_mean"))
            for threshold, _marker, label in BOUNDARY_MARKERS:
                hit_step = first_boundary_step(group, threshold)
                hit_row = row_at_step(group, hit_step) if hit_step is not None else None
                hit_loss = fnum(hit_row.get("loss3_q_mean")) if hit_row is not None else math.nan
                hit_ratio = fnum(hit_row.get("boundary_ratio_mean")) if hit_row is not None else math.nan
                gain = final_loss - hit_loss if math.isfinite(final_loss) and math.isfinite(hit_loss) else math.nan
                rows.append(
                    {
                        "source_root": str(root),
                        "epsilon": epsilon,
                        "alpha": alpha,
                        "method": method,
                        "threshold": threshold,
                        "threshold_label": label,
                        "boundary_hit_step_mean": hit_step if hit_step is not None else "nan",
                        "loss3_q_mean_at_threshold_hit": hit_loss if math.isfinite(hit_loss) else "nan",
                        "boundary_ratio_mean_at_threshold_hit": hit_ratio if math.isfinite(hit_ratio) else "nan",
                        "final_loss3_q_mean": final_loss if math.isfinite(final_loss) else "nan",
                        "post_threshold_loss_gain": gain if math.isfinite(gain) else "nan",
                        "post_threshold_loss_gain_fraction_of_final": (gain / final_loss) if math.isfinite(gain) and math.isfinite(final_loss) and final_loss else "nan",
                        "final_boundary_ratio_mean": final_ratio if math.isfinite(final_ratio) else "nan",
                    }
                )
    write_csv(out_csv, rows)
    if legacy_99_csv is not None:
        legacy = [row for row in rows if abs(fnum(row["threshold"]) - 0.99) < 1e-12]
        legacy_rows = []
        for row in legacy:
            legacy_rows.append(
                {
                    "source_root": row["source_root"],
                    "epsilon": row["epsilon"],
                    "alpha": row["alpha"],
                    "method": row["method"],
                    "boundary_hit_step_mean_0.99": row["boundary_hit_step_mean"],
                    "loss3_q_mean_at_boundary_hit": row["loss3_q_mean_at_threshold_hit"],
                    "final_loss3_q_mean": row["final_loss3_q_mean"],
                    "post_boundary_loss_gain": row["post_threshold_loss_gain"],
                    "post_boundary_loss_gain_fraction_of_final": row["post_threshold_loss_gain_fraction_of_final"],
                    "final_boundary_ratio_mean": row["final_boundary_ratio_mean"],
                }
            )
        write_csv(legacy_99_csv, legacy_rows)
    return rows


def compute_peakiness_summary(roots: list[Path], out_csv: Path) -> list[dict]:
    rows: list[dict] = []
    for root in roots:
        epsilon, alpha = root_setting(root)
        for method in CORE4:
            path = root / method / "final_deltas.npz"
            if not path.exists():
                continue
            data = np.load(path)
            delta = data["final_delta"]
            dataset_index = data["dataset_index"]
            p = peakiness(delta)
            tv = total_variation(delta)
            d1 = derivative_l2(delta)
            hf = high_frequency_ratio(delta)
            max_abs = np.max(np.abs(delta.reshape(delta.shape[0], -1)), axis=1)
            rms = np.sqrt(np.mean(delta.reshape(delta.shape[0], -1).astype(float) ** 2, axis=1))
            sample40_idx = np.where(dataset_index == 40)[0]
            sample40 = int(sample40_idx[0]) if sample40_idx.size else None
            rows.append(
                {
                    "source_root": str(root),
                    "epsilon": epsilon,
                    "alpha": alpha,
                    "method": method,
                    "sample_count": int(delta.shape[0]),
                    "mean_peakiness_max_abs_over_rms": finite_mean(p),
                    "std_peakiness_max_abs_over_rms": finite_std(p),
                    "max_peakiness_max_abs_over_rms": float(np.nanmax(p)),
                    "mean_max_abs_delta": finite_mean(max_abs),
                    "mean_rms_delta": finite_mean(rms),
                    "mean_total_variation": finite_mean(tv),
                    "mean_first_derivative_l2": finite_mean(d1),
                    "mean_high_frequency_ratio_from_final_delta": finite_mean(hf),
                    "sample40_peakiness_max_abs_over_rms": float(p[sample40]) if sample40 is not None else "nan",
                    "sample40_max_abs_delta": float(max_abs[sample40]) if sample40 is not None else "nan",
                    "sample40_total_variation": float(tv[sample40]) if sample40 is not None else "nan",
                    "sample40_high_frequency_ratio": float(hf[sample40]) if sample40 is not None else "nan",
                }
            )
    write_csv(out_csv, rows)
    return rows


def plot_peakiness_heatmap(rows: list[dict], out_path: Path) -> None:
    settings = sorted({setting_key(row["epsilon"], row["alpha"]) for row in rows})
    labels = [setting_label_short(eps, alpha) for eps, alpha in settings]
    matrix = np.full((len(CORE4), len(settings)), np.nan)
    by_key = {(row["method"], setting_key(row["epsilon"], row["alpha"])): row for row in rows}
    for i, method in enumerate(CORE4):
        for j, key in enumerate(settings):
            row = by_key.get((method, key))
            if row is not None:
                matrix[i, j] = fnum(row["mean_peakiness_max_abs_over_rms"])
    fig, ax = plt.subplots(figsize=(max(15.5, 1.55 * len(labels) + 4.0), 5.8))
    im = ax.imshow(matrix, aspect="auto", cmap="magma")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=42, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(CORE4)))
    ax.set_yticklabels([METHOD_LABELS[m] for m in CORE4], fontsize=10)
    ax.set_title("Final delta peakiness: max(abs(delta)) / RMS(delta)", fontsize=15, pad=18)
    finite_matrix = matrix[np.isfinite(matrix)]
    threshold = float(np.nanmean(finite_matrix)) if finite_matrix.size else math.nan
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            text_color = "black" if math.isfinite(value) and math.isfinite(threshold) and value > threshold else "white"
            ax.text(j, i, f"{value:.2f}" if math.isfinite(value) else "nan", ha="center", va="center", fontsize=8, color=text_color)
    fig.subplots_adjust(left=0.105, right=0.865, top=0.82, bottom=0.28)
    cax = fig.add_axes([0.895, 0.28, 0.018, 0.54])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("peakiness", fontsize=10, labelpad=10)
    cbar.ax.tick_params(labelsize=9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def load_traj(root: Path, method: str) -> np.lib.npyio.NpzFile:
    return np.load(root / method / "trajectory_samples.npz")


def sample_position(data: np.lib.npyio.NpzFile, dataset_index: int) -> int:
    matches = np.where(data["dataset_index"] == dataset_index)[0]
    if matches.size:
        return int(matches[0])
    return 0


def plot_representative_sample(root: Path, out_path: Path, dataset_index: int) -> None:
    epsilon, alpha = root_setting(root)
    fig, axes = plt.subplots(len(CORE4), 4, figsize=(18, 10.5), squeeze=False)
    for i, method in enumerate(CORE4):
        data = load_traj(root, method)
        pos = sample_position(data, dataset_index)
        clean = data["x_adv"][0, pos].reshape(-1).astype(float)
        adv = data["x_adv"][-1, pos].reshape(-1).astype(float)
        delta = data["delta"][-1, pos].reshape(-1).astype(float)
        k_values = data["k"].astype(int)
        boundary = data["boundary_ratio"][:, pos].astype(float)
        hit_indices = np.where(boundary >= 0.99)[0]
        hit_k = int(k_values[hit_indices[0]]) if hit_indices.size else None
        x = np.arange(clean.size)
        p = peakiness(delta.reshape(1, -1))[0]
        hf = high_frequency_ratio(delta.reshape(1, -1))[0]
        d1 = derivative_l2(delta.reshape(1, -1))[0]
        tv = total_variation(delta.reshape(1, -1))[0]
        loss = float(data["loss3_q"][-1, pos])
        norm = float(data["delta_pnorm"][-1, pos])

        ax = axes[i, 0]
        ax.plot(x, clean, color="#333333", linewidth=1.15, label="clean")
        ax.plot(x, adv, color=COLORS[method], linewidth=1.0, alpha=0.9, label="clean+delta")
        ax.set_ylabel(method)
        ax.set_title("initial condition and perturbed")
        ax.grid(True, alpha=0.2)

        ax = axes[i, 1]
        ax.plot(x, delta, color=COLORS[method], linewidth=1.0)
        ax.set_title("final delta")
        ax.grid(True, alpha=0.2)

        ax = axes[i, 2]
        freqs = np.fft.rfftfreq(delta.size)
        amp = np.abs(np.fft.rfft(delta))
        ax.semilogy(freqs[1:], amp[1:] + 1e-12, color=COLORS[method], linewidth=1.0)
        ax.axvline(0.25 * freqs.max(), color="#666666", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_title("delta spectrum")
        ax.grid(True, alpha=0.2)

        ax = axes[i, 3]
        ax.plot(k_values, data["loss3_q"][:, pos], color=COLORS[method], linewidth=1.5, label="sample loss3")
        if hit_k is not None:
            hit_pos = np.where(k_values == hit_k)[0][0]
            ax.scatter([hit_k], [data["loss3_q"][hit_pos, pos]], marker="x", s=80, color="black", linewidths=2.0)
            ax.annotate(f"boundary k={hit_k}", (hit_k, data["loss3_q"][hit_pos, pos]), textcoords="offset points", xytext=(5, 6), fontsize=8)
        ax.set_title("sample loss curve")
        ax.grid(True, alpha=0.2)
        ax.text(
            0.03,
            0.04,
            f"final loss={loss:.3g}\n||d||={norm:.3g}, eps={epsilon:g}\npeak={p:.2f}, hf={hf:.2e}\nd1={d1:.2g}, TV={tv:.2g}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.82, "pad": 3},
        )
    for ax in axes[-1, :]:
        ax.set_xlabel("grid index / step")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.015), ncol=2, frameon=True, fontsize=9)
    fig.suptitle(f"Representative sample {dataset_index}: {setting_label(epsilon, alpha)}", fontsize=15, y=0.982)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.91, bottom=0.08, hspace=0.45, wspace=0.22)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_delta_method_grid(root: Path, out_path: Path, dataset_indices: list[int]) -> None:
    epsilon, alpha = root_setting(root)
    fig, axes = plt.subplots(len(CORE4), len(dataset_indices), figsize=(4.3 * len(dataset_indices), 9.5), squeeze=False, sharex=True)
    for i, method in enumerate(CORE4):
        data = load_traj(root, method)
        for j, dataset_index in enumerate(dataset_indices):
            pos = sample_position(data, dataset_index)
            delta = data["delta"][-1, pos].reshape(-1).astype(float)
            ax = axes[i, j]
            ax.plot(delta, color=COLORS[method], linewidth=0.9)
            p = peakiness(delta.reshape(1, -1))[0]
            hf = high_frequency_ratio(delta.reshape(1, -1))[0]
            ax.set_title(f"sample {int(data['dataset_index'][pos])}\npeak={p:.2f}, hf={hf:.1e}", fontsize=9)
            if j == 0:
                ax.set_ylabel(method)
            ax.grid(True, alpha=0.2)
    fig.suptitle(f"Final delta shapes: {setting_label(epsilon, alpha)}", fontsize=15, y=0.982)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.90, bottom=0.07, hspace=0.58, wspace=0.22)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_doc(
    doc_path: Path,
    out_dir: Path,
    roots: list[Path],
    generated: dict[str, Path | list[Path]],
    boundary_threshold: float,
) -> None:
    lines = [
        "# Loss3 Core-Four Alpha/Epsilon Sweep Visualizations",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Scope",
        "",
        "Observed from existing completed artifacts under `forensics/loss3_alpha_epsilon_core4_sweep_20260519/`; no experiment was rerun for these plots.",
        "",
        "Geometry note: these alpha/epsilon plots are scoped to the completed roots listed below. At generation time the completed roots are `p=2,q=2`; all-PQ alpha/epsilon comparison requires additional P/Q runs and should be plotted per P/Q pair.",
        "",
        "Boundary markers on loss curves use the first step where mean `boundary_ratio` reaches `0.25`, `0.50`, `0.75`, and `0.99`. The `0.99` marker is the near-boundary marker for `||delta||_p ~= epsilon`, chosen to avoid floating-point exact-equality issues.",
        "",
        "## Main Figures",
        "",
        f"- Loss curves with 25/50/75/99% boundary-hit markers, mean +/- std: `{rel(generated['loss_curves'])}`",
        f"- Loss curves with 25/50/75/99% boundary-hit markers, mean only/no std band: `{rel(generated['loss_curves_no_std'])}`",
        f"- Boundary ratio curves, mean +/- std: `{rel(generated['boundary_curves'])}`",
        f"- Boundary ratio curves, mean only/no std band: `{rel(generated['boundary_curves_no_std'])}`",
        f"- Boundary ratio sample-standard-deviation curves: `{rel(generated['boundary_std_curves'])}`",
        f"- Delta angular-speed curves, mean +/- std over batch: `{rel(generated['angle_curves'])}`" if generated.get("angle_curves") is not None else "- Delta angular-speed curves: not generated because `delta_prev_angle_degrees_mean` was not present in these artifacts.",
        f"- Delta angular-speed curves, mean only/no std band: `{rel(generated['angle_curves_no_std'])}`" if generated.get("angle_curves_no_std") is not None else "- Delta angular-speed mean-only curves: not generated because `delta_prev_angle_degrees_mean` was not present in these artifacts.",
        f"- Final loss heatmap: `{rel(generated['final_loss_heatmap'])}`",
        f"- Mean 99% boundary-arrival step heatmap: `{rel(generated['boundary_heatmap'])}`",
        f"- Final high-frequency energy heatmap: `{rel(generated['hf_heatmap'])}`",
        f"- Final first-derivative smoothness heatmap: `{rel(generated['d1_heatmap'])}`",
        f"- Final delta peakiness heatmap: `{rel(generated['peakiness_heatmap'])}`",
        f"- Final delta peakiness source table: `{rel(generated['peakiness_csv'])}`",
        f"- Boundary-threshold loss-gain source table: `{rel(generated['boundary_threshold_csv'])}`",
        f"- 99% boundary-hit loss-gain compatibility table: `{rel(generated['boundary_hit_loss_gain_csv'])}`",
        "",
        "Note: every mean curve family is now emitted twice: one mean +/- sample-std version for variability, and one mean-only/no-std-band version so the central trajectories and boundary markers remain readable when the std band expands the y-axis.",
        "",
        "## Representative Sample Panels",
        "",
        "Each representative panel shows clean initial condition, final delta, clean+delta, delta spectrum, and a sample-level loss curve with a boundary-hit marker.",
        "",
    ]
    for path in generated["representative_panels"]:  # type: ignore[index]
        lines.append(f"- `{rel(path)}`")
    lines.extend(
        [
            "",
            "## Delta Shape Grids",
            "",
            "These grids show final delta traces for the stored trajectory samples. No cross markers are used on delta plots; cross markers are reserved for boundary hits on loss curves.",
            "",
        ]
    )
    for path in generated["delta_grids"]:  # type: ignore[index]
        lines.append(f"- `{rel(path)}`")
    lines.extend(
        [
            "",
            "## Dynamics Triptychs",
            "",
            "Triptychs are also emitted in paired versions: mean +/- std and mean-only/no-std-band. The angular panel uses `angle(delta_k, delta_{k-1})` in degrees and a data-driven y-axis instead of a fixed 0..180 degree range.",
            "",
        ]
    )
    for path in generated.get("dynamics_triptychs", []):  # type: ignore[union-attr]
        lines.append(f"- `{rel(path)}`")
    for path in generated.get("dynamics_triptychs_no_std", []):  # type: ignore[union-attr]
        lines.append(f"- `{rel(path)}`")
    angle_available = generated.get("angle_available_triptychs", [])
    if angle_available:
        lines.extend(
            [
                "",
                "## Angle-Available Dynamics Triptychs",
                "",
                "These additional triptychs are generated only for settings whose `per_step_metrics.csv` actually contains `delta_prev_angle_degrees_mean`. This avoids hiding usable angular-motion data when older completed settings did not record that metric.",
                "",
            ]
        )
        for path in angle_available:  # type: ignore[union-attr]
            lines.append(f"- `{rel(path)}`")
        for path in generated.get("angle_available_triptychs_no_std", []):  # type: ignore[union-attr]
            lines.append(f"- `{rel(path)}`")
    lines.extend(
        [
            "",
            "## Inputs",
            "",
        ]
    )
    for root in roots:
        epsilon, alpha = root_setting(root)
        p_order, q_order = root_pq(root)
        lines.append(f"- `{rel(root)}` ({setting_label(epsilon, alpha)}, p={p_order}, q={q_order})")
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_setting(value: str) -> tuple[float, float]:
    for sep in (":", ","):
        if sep in value:
            left, right = value.split(sep, 1)
            return float(left), float(right)
    raise argparse.ArgumentTypeError("settings must be formatted epsilon:alpha, e.g. 4:0.4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-root", type=Path, default=DEFAULT_SWEEP)
    parser.add_argument("--analysis-root", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    parser.add_argument("--boundary-threshold", type=float, default=0.99)
    parser.add_argument("--p-filter", default=None, help="Optional p value to plot when the sweep root contains multiple P/Q pairs.")
    parser.add_argument("--q-filter", default=None, help="Optional q value to plot when the sweep root contains multiple P/Q pairs.")
    parser.add_argument("--representative-settings", nargs="+", type=parse_setting, default=list(REPRESENTATIVE_SETTINGS))
    parser.add_argument("--dataset-index", type=int, default=40)
    parser.add_argument("--delta-grid-indices", nargs="+", type=int, default=[0, 7, 40, 47])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sweep_root = args.sweep_root if args.sweep_root.is_absolute() else PROJECT_ROOT / args.sweep_root
    analysis_root = args.analysis_root if args.analysis_root.is_absolute() else PROJECT_ROOT / args.analysis_root
    out_dir = args.out_dir if args.out_dir.is_absolute() else PROJECT_ROOT / args.out_dir
    doc_path = args.doc if args.doc.is_absolute() else PROJECT_ROOT / args.doc
    out_dir.mkdir(parents=True, exist_ok=True)

    roots = load_completed_roots(sweep_root)
    if args.p_filter is not None or args.q_filter is not None:
        roots = [
            root
            for root in roots
            if (args.p_filter is None or root_pq(root)[0] == str(args.p_filter))
            and (args.q_filter is None or root_pq(root)[1] == str(args.q_filter))
        ]
    pq_set = sorted({root_pq(root) for root in roots})
    if len(pq_set) > 1 and args.p_filter is None and args.q_filter is None:
        raise SystemExit(
            "Multiple P/Q pairs are present in the sweep root. Use --p-filter and --q-filter, "
            "and write each P/Q pair to a separate --out-dir to avoid mixing alpha/epsilon plots."
        )
    if not roots:
        raise SystemExit("No completed roots matched the requested sweep/PQ filters.")
    summary_rows = read_csv(analysis_root / "core4_alpha_epsilon_method_summary.csv")
    if args.p_filter is not None or args.q_filter is not None:
        summary_rows = [
            row
            for row in summary_rows
            if (args.p_filter is None or str(row.get("p_order", row.get("p", ""))) == str(args.p_filter))
            and (args.q_filter is None or str(row.get("q_order", row.get("q", ""))) == str(args.q_filter))
        ]
    root_by_setting = {setting_key(*root_setting(root)): root for root in roots}

    figures_dir = out_dir / "figures"
    tables_dir = out_dir / "tables"
    loss_curves = figures_dir / "loss3_q_mean_curves_with_boundary_markers.png"
    loss_curves_no_std = figures_dir / "loss3_q_mean_curves_with_boundary_markers_no_std.png"
    boundary_curves = figures_dir / "boundary_ratio_mean_curves.png"
    boundary_curves_no_std = figures_dir / "boundary_ratio_mean_curves_no_std.png"
    boundary_std_curves = figures_dir / "boundary_ratio_std_curves.png"
    angle_curves = figures_dir / "delta_prev_angle_degrees_mean_curves.png"
    angle_curves_no_std = figures_dir / "delta_prev_angle_degrees_mean_curves_no_std.png"
    plot_metric_curves(
        roots,
        "loss3_q_mean",
        "loss3 q mean",
        loss_curves,
        args.boundary_threshold,
        mark_boundary=True,
        shade_std=True,
    )
    plot_metric_curves(
        roots,
        "loss3_q_mean",
        "loss3 q mean",
        loss_curves_no_std,
        args.boundary_threshold,
        mark_boundary=True,
        shade_std=False,
    )
    plot_metric_curves(
        roots,
        "boundary_ratio_mean",
        "mean boundary ratio",
        boundary_curves,
        args.boundary_threshold,
        mark_boundary=False,
        shade_std=True,
    )
    plot_metric_curves(
        roots,
        "boundary_ratio_mean",
        "mean boundary ratio",
        boundary_curves_no_std,
        args.boundary_threshold,
        mark_boundary=False,
        shade_std=False,
    )
    plot_metric_curves(
        roots,
        "boundary_ratio_std",
        "boundary ratio sample std",
        boundary_std_curves,
        args.boundary_threshold,
        mark_boundary=False,
        shade_std=False,
    )
    angle_curves_generated = False
    if has_metric(roots, "delta_prev_angle_degrees_mean"):
        plot_metric_curves(
            roots,
            "delta_prev_angle_degrees_mean",
            "mean delta angle from previous step (degrees)",
            angle_curves,
            args.boundary_threshold,
            mark_boundary=False,
            shade_std=True,
        )
        plot_metric_curves(
            roots,
            "delta_prev_angle_degrees_mean",
            "mean delta angle from previous step (degrees)",
            angle_curves_no_std,
            args.boundary_threshold,
            mark_boundary=False,
            shade_std=False,
        )
        angle_curves_generated = True

    final_loss_heatmap = figures_dir / "heatmap_final_loss3_q_mean.png"
    boundary_heatmap = figures_dir / "heatmap_step_to_boundary_ratio_mean_0p99.png"
    hf_heatmap = figures_dir / "heatmap_final_high_frequency_energy_ratio_mean_log10.png"
    d1_heatmap = figures_dir / "heatmap_final_first_derivative_l2_mean.png"
    plot_heatmap(summary_rows, "final_loss3_q_mean", "Final loss3 q mean", final_loss_heatmap, cmap="viridis", fmt=".3g")
    plot_heatmap(summary_rows, "step_to_boundary_ratio_mean_0.99", "First mean step to 99% boundary", boundary_heatmap, cmap="cividis", fmt=".0f")
    plot_heatmap(summary_rows, "final_high_frequency_energy_ratio_mean", "Final high-frequency energy ratio", hf_heatmap, cmap="magma", log10=True, fmt=".1e")
    plot_heatmap(summary_rows, "final_first_derivative_l2_mean", "Final first-derivative L2", d1_heatmap, cmap="plasma", fmt=".2g")

    boundary_threshold_csv = tables_dir / "boundary_threshold_loss_gain_summary.csv"
    boundary_hit_loss_gain_csv = tables_dir / "boundary_hit_loss_gain_summary.csv"
    compute_boundary_threshold_summary(roots, boundary_threshold_csv, boundary_hit_loss_gain_csv)

    peakiness_csv = tables_dir / "final_delta_peakiness_summary.csv"
    peak_rows = compute_peakiness_summary(roots, peakiness_csv)
    peakiness_heatmap = figures_dir / "heatmap_final_delta_peakiness.png"
    plot_peakiness_heatmap(peak_rows, peakiness_heatmap)

    representative_panels: list[Path] = []
    delta_grids: list[Path] = []
    dynamics_triptychs: list[Path] = []
    dynamics_triptychs_no_std: list[Path] = []
    angle_available_triptychs: list[Path] = []
    angle_available_triptychs_no_std: list[Path] = []
    for epsilon, alpha in args.representative_settings:
        root = root_by_setting.get(setting_key(epsilon, alpha))
        if root is None:
            continue
        panel = figures_dir / "representative_samples" / f"sample{args.dataset_index}_{setting_slug(epsilon, alpha)}.png"
        plot_representative_sample(root, panel, args.dataset_index)
        representative_panels.append(panel)
        grid = figures_dir / "delta_shape_grids" / f"delta_grid_{setting_slug(epsilon, alpha)}.png"
        plot_delta_method_grid(root, grid, args.delta_grid_indices)
        delta_grids.append(grid)
        triptych = figures_dir / "dynamics_triptychs" / f"dynamics_triptych_{setting_slug(epsilon, alpha)}.png"
        plot_dynamics_triptych(root, triptych, shade_std=True)
        dynamics_triptychs.append(triptych)
        triptych_no_std = figures_dir / "dynamics_triptychs_no_std" / f"dynamics_triptych_{setting_slug(epsilon, alpha)}_no_std.png"
        plot_dynamics_triptych(root, triptych_no_std, shade_std=False)
        dynamics_triptychs_no_std.append(triptych_no_std)

    for root in roots:
        if not root_has_metric(root, "delta_prev_angle_degrees_mean"):
            continue
        epsilon, alpha = root_setting(root)
        triptych = figures_dir / "dynamics_triptychs_angle_available" / f"dynamics_triptych_{setting_slug(epsilon, alpha)}.png"
        plot_dynamics_triptych(root, triptych, shade_std=True)
        angle_available_triptychs.append(triptych)
        triptych_no_std = figures_dir / "dynamics_triptychs_angle_available_no_std" / f"dynamics_triptych_{setting_slug(epsilon, alpha)}_no_std.png"
        plot_dynamics_triptych(root, triptych_no_std, shade_std=False)
        angle_available_triptychs_no_std.append(triptych_no_std)

    manifest = {
        "status": "completed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sweep_root": str(sweep_root),
        "analysis_root": str(analysis_root),
        "out_dir": str(out_dir),
        "boundary_marker_definition": "loss curves mark first mean boundary_ratio hits at 0.25, 0.50, 0.75, and 0.99",
        "completed_setting_count": len(roots),
        "pq_pairs_plotted": [{"p": p, "q": q} for p, q in pq_set],
        "figures": {
            "loss_curves": str(loss_curves),
            "loss_curves_no_std": str(loss_curves_no_std),
            "boundary_curves": str(boundary_curves),
            "boundary_curves_no_std": str(boundary_curves_no_std),
            "boundary_std_curves": str(boundary_std_curves),
            "angle_curves": str(angle_curves) if angle_curves_generated else None,
            "angle_curves_no_std": str(angle_curves_no_std) if angle_curves_generated else None,
            "final_loss_heatmap": str(final_loss_heatmap),
            "boundary_heatmap": str(boundary_heatmap),
            "hf_heatmap": str(hf_heatmap),
            "d1_heatmap": str(d1_heatmap),
            "peakiness_heatmap": str(peakiness_heatmap),
            "representative_panels": [str(p) for p in representative_panels],
            "delta_grids": [str(p) for p in delta_grids],
            "dynamics_triptychs": [str(p) for p in dynamics_triptychs],
            "dynamics_triptychs_no_std": [str(p) for p in dynamics_triptychs_no_std],
            "angle_available_triptychs": [str(p) for p in angle_available_triptychs],
            "angle_available_triptychs_no_std": [str(p) for p in angle_available_triptychs_no_std],
        },
        "tables": {
            "peakiness_summary": str(peakiness_csv),
            "boundary_threshold_loss_gain_summary": str(boundary_threshold_csv),
            "boundary_hit_loss_gain_summary_0.99": str(boundary_hit_loss_gain_csv),
        },
        "doc": str(doc_path),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_doc(
        doc_path,
        out_dir,
        roots,
        {
            "loss_curves": loss_curves,
            "loss_curves_no_std": loss_curves_no_std,
            "boundary_curves": boundary_curves,
            "boundary_curves_no_std": boundary_curves_no_std,
            "boundary_std_curves": boundary_std_curves,
            "angle_curves": angle_curves if angle_curves_generated else None,
            "angle_curves_no_std": angle_curves_no_std if angle_curves_generated else None,
            "final_loss_heatmap": final_loss_heatmap,
            "boundary_heatmap": boundary_heatmap,
            "hf_heatmap": hf_heatmap,
            "d1_heatmap": d1_heatmap,
            "peakiness_heatmap": peakiness_heatmap,
            "peakiness_csv": peakiness_csv,
            "boundary_threshold_csv": boundary_threshold_csv,
            "boundary_hit_loss_gain_csv": boundary_hit_loss_gain_csv,
            "representative_panels": representative_panels,
            "delta_grids": delta_grids,
            "dynamics_triptychs": dynamics_triptychs,
            "dynamics_triptychs_no_std": dynamics_triptychs_no_std,
            "angle_available_triptychs": angle_available_triptychs,
            "angle_available_triptychs_no_std": angle_available_triptychs_no_std,
        },
        args.boundary_threshold,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
