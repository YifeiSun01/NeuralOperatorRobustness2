#!/usr/bin/env python3
"""Plot loss1/loss2/loss3 core-four baseline metrics on shared figures.

No experiment is rerun.  This script combines:

- loss1/loss2 corrected 0..100 marked-angle output
- loss3 p2q2 baseline output, filtered to k=0..100

The script writes both orientations:

- by method: four method panels, each overlaying loss1/loss2/loss3
- by objective: three objective panels, each overlaying the four methods
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOSS12_ROOT = PROJECT_ROOT / "forensics" / "loss1_loss2_core4_p2q2_baseline_marked_angles_20260521" / "fno_nu0p001_eps4_alpha0p4_batch100_steps100_p2_q2"
DEFAULT_LOSS3_ROOT = PROJECT_ROOT / "forensics" / "loss3_alpha_epsilon_core4_baseline_giftrace_20260520" / "fno_nu0p001_eps4_alpha0p4_batch100_steps300_p2_q2"
DEFAULT_OUT = PROJECT_ROOT / "forensics" / "loss1_loss2_loss3_core4_combined_0to100_20260521"
DEFAULT_DOC = PROJECT_ROOT / "docs" / "loss1_loss2_loss3_core4_combined_0to100_20260521.md"
CORE4 = ("raw_add", "raw_replace", "steepest_add", "steepest_replace")
OBJECTIVES = ("loss1", "loss2", "loss3")
OBJ_COLORS = {"loss1": "#1f77b4", "loss2": "#d62728", "loss3": "#2ca02c"}
OBJ_LABELS = {"loss1": "loss1", "loss2": "loss2", "loss3": "loss3"}
METHOD_COLORS = {
    "raw_add": "#1f77b4",
    "raw_replace": "#ff7f0e",
    "steepest_add": "#2ca02c",
    "steepest_replace": "#9467bd",
}
METHOD_LABELS = {
    "raw_add": "raw add",
    "raw_replace": "raw replace",
    "steepest_add": "steepest add",
    "steepest_replace": "steepest replace",
}


def fnum(value: Any) -> float:
    if value in {None, "", "nan", "NaN"}:
        return math.nan
    try:
        return float(value)
    except Exception:
        return math.nan


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
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


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def angle_from_cos(cos_value: float) -> float:
    if not math.isfinite(cos_value):
        return math.nan
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_value))))


def angle_std_from_cos(cos_mean: float, cos_std: float) -> float:
    if not math.isfinite(cos_mean) or not math.isfinite(cos_std):
        return math.nan
    lower_angle = angle_from_cos(cos_mean + cos_std)
    upper_angle = angle_from_cos(cos_mean - cos_std)
    if not math.isfinite(lower_angle) or not math.isfinite(upper_angle):
        return math.nan
    return max(0.0, (upper_angle - lower_angle) / 2.0)


def std_column(metric: str) -> str:
    if metric == "loss_mean":
        return "loss_std"
    if metric.endswith("_mean"):
        return metric[: -len("_mean")] + "_std"
    return metric + "_std"


def load_loss12(loss12_root: Path, objective: str, method: str) -> list[dict[str, Any]]:
    obj_dir = "loss1_original" if objective == "loss1" else "loss2_original"
    rows = read_csv(loss12_root / obj_dir / method / "per_step_metrics.csv")
    out: list[dict[str, Any]] = []
    for row in rows:
        k = int(row["k"])
        if k > 100:
            continue
        out.append(
            {
                "objective": objective,
                "method": method,
                "k": k,
                "loss_mean": fnum(row.get("optimized_loss_mean")),
                "loss_std": fnum(row.get("optimized_loss_std")),
                "delta_pnorm_mean": fnum(row.get("delta_pnorm_mean")),
                "delta_pnorm_std": fnum(row.get("delta_pnorm_std")),
                "boundary_ratio_mean": fnum(row.get("boundary_ratio_mean")),
                "boundary_ratio_std": fnum(row.get("boundary_ratio_std")),
                "delta_step_l2_mean": fnum(row.get("delta_step_l2_mean")),
                "delta_step_l2_std": fnum(row.get("delta_step_l2_std")),
                "delta_prev_angle_degrees_mean": fnum(row.get("delta_prev_angle_degrees_mean")),
                "delta_prev_angle_degrees_std": fnum(row.get("delta_prev_angle_degrees_std")),
                "direction_prev_angle_degrees_mean": fnum(row.get("direction_prev_angle_degrees_mean")),
                "direction_prev_angle_degrees_std": fnum(row.get("direction_prev_angle_degrees_std")),
                "delta_direction_angle_degrees_mean": fnum(row.get("delta_direction_angle_degrees_mean")),
                "delta_direction_angle_degrees_std": fnum(row.get("delta_direction_angle_degrees_std")),
                "high_frequency_energy_ratio_mean": fnum(row.get("high_frequency_energy_ratio_mean")),
                "high_frequency_energy_ratio_std": fnum(row.get("high_frequency_energy_ratio_std")),
                "source_csv": rel(loss12_root / obj_dir / method / "per_step_metrics.csv"),
            }
        )
    return out


def load_loss3(loss3_root: Path, method: str) -> list[dict[str, Any]]:
    rows = read_csv(loss3_root / method / "per_step_metrics.csv")
    out: list[dict[str, Any]] = []
    for row in rows:
        k = int(row["k"])
        if k > 100:
            continue
        out.append(
            {
                "objective": "loss3",
                "method": method,
                "k": k,
                "loss_mean": fnum(row.get("loss3_q_mean")),
                "loss_std": fnum(row.get("loss3_q_std")),
                "delta_pnorm_mean": fnum(row.get("delta_pnorm_mean")),
                "delta_pnorm_std": fnum(row.get("delta_pnorm_std")),
                "boundary_ratio_mean": fnum(row.get("boundary_ratio_mean")),
                "boundary_ratio_std": fnum(row.get("boundary_ratio_std")),
                "delta_step_l2_mean": fnum(row.get("delta_step_l2_mean")),
                "delta_step_l2_std": fnum(row.get("delta_step_l2_std")),
                "delta_prev_angle_degrees_mean": fnum(row.get("delta_prev_angle_degrees_mean")),
                "delta_prev_angle_degrees_std": fnum(row.get("delta_prev_angle_degrees_std")),
                "direction_prev_angle_degrees_mean": angle_from_cos(fnum(row.get("cos_direction_prev_mean"))),
                "direction_prev_angle_degrees_std": angle_std_from_cos(fnum(row.get("cos_direction_prev_mean")), fnum(row.get("cos_direction_prev_std"))),
                "delta_direction_angle_degrees_mean": angle_from_cos(fnum(row.get("cos_delta_direction_mean"))),
                "delta_direction_angle_degrees_std": angle_std_from_cos(fnum(row.get("cos_delta_direction_mean")), fnum(row.get("cos_delta_direction_std"))),
                "high_frequency_energy_ratio_mean": fnum(row.get("high_frequency_energy_ratio_mean")),
                "high_frequency_energy_ratio_std": fnum(row.get("high_frequency_energy_ratio_std")),
                "source_csv": rel(loss3_root / method / "per_step_metrics.csv"),
            }
        )
    return out


def load_combined(loss12_root: Path, loss3_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in CORE4:
        rows.extend(load_loss12(loss12_root, "loss1", method))
        rows.extend(load_loss12(loss12_root, "loss2", method))
        rows.extend(load_loss3(loss3_root, method))
    return rows


def grouped(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    out: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for method in CORE4:
        for objective in OBJECTIVES:
            group = [r for r in rows if r["method"] == method and r["objective"] == objective]
            out[(method, objective)] = sorted(group, key=lambda r: int(r["k"]))
    return out


def metric_values(
    rows: list[dict[str, Any]],
    metric: str,
    *,
    normalize_loss: bool = False,
) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    groups = grouped(rows)
    out: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    std_metric = std_column(metric)
    for method in CORE4:
        for objective in OBJECTIVES:
            group = groups[(method, objective)]
            xs = np.asarray([r["k"] for r in group], dtype=float)
            ys = np.asarray([r[metric] for r in group], dtype=float)
            ystd = np.asarray([r.get(std_metric, math.nan) for r in group], dtype=float)
            if normalize_loss:
                finite = np.isfinite(ys)
                if finite.any():
                    denom = np.nanmax(ys[finite])
                    if math.isfinite(float(denom)) and denom > 0:
                        ys = ys / denom
                        ystd = ystd / denom
            out[(method, objective)] = (xs, ys, ystd)
    return out


def auto_ylim(series: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]], *, include_std: bool = False) -> tuple[float, float]:
    chunks: list[np.ndarray] = []
    for _, ys, ystd in series.values():
        if include_std:
            lower = ys - ystd
            upper = ys + ystd
            if np.isfinite(lower).any():
                chunks.append(lower[np.isfinite(lower)])
            if np.isfinite(upper).any():
                chunks.append(upper[np.isfinite(upper)])
        elif np.isfinite(ys).any():
            chunks.append(ys[np.isfinite(ys)])
    values = np.concatenate(chunks) if chunks else np.asarray([], dtype=float)
    if values.size == 0:
        return (-1.0, 1.0)
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    if lo == hi:
        pad = 1.0 if lo == 0 else abs(lo) * 0.08
    else:
        pad = (hi - lo) * 0.08
    if lo >= 0:
        lo = max(0.0, lo - pad)
    else:
        lo -= pad
    hi += pad
    return (lo, hi)


def plot_by_method(
    rows: list[dict[str, Any]],
    metric: str,
    ylabel: str,
    title: str,
    out_path: Path,
    *,
    normalize_loss: bool = False,
    show_std: bool = False,
    ylim: tuple[float, float] | None = None,
) -> Path:
    series = metric_values(rows, metric, normalize_loss=normalize_loss)
    use_ylim = ylim if ylim is not None else auto_ylim(series, include_std=show_std)
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.6), sharex=True, sharey=True)
    for ax, method in zip(axes.ravel(), CORE4):
        for objective in OBJECTIVES:
            xs, ys, ystd = series[(method, objective)]
            if show_std and np.isfinite(ystd).any():
                lower = ys - ystd
                upper = ys + ystd
                ax.fill_between(xs, lower, upper, color=OBJ_COLORS[objective], alpha=0.18, linewidth=0)
            ax.plot(xs, ys, color=OBJ_COLORS[objective], linewidth=2.0, label=OBJ_LABELS[objective])
        ax.set_title(METHOD_LABELS[method])
        ax.grid(alpha=0.25)
        ax.set_xlim(0, 100)
        ax.set_ylim(*use_ylim)
    for ax in axes[-1, :]:
        ax.set_xlabel("optimization step k")
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle(title, y=0.965)
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.025))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.13, hspace=0.32, wspace=0.16)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_by_objective(
    rows: list[dict[str, Any]],
    metric: str,
    ylabel: str,
    title: str,
    out_path: Path,
    *,
    normalize_loss: bool = False,
    show_std: bool = False,
    ylim: tuple[float, float] | None = None,
) -> Path:
    series = metric_values(rows, metric, normalize_loss=normalize_loss)
    use_ylim = ylim if ylim is not None else auto_ylim(series, include_std=show_std)
    fig, axes = plt.subplots(3, 1, figsize=(12.8, 10.2), sharex=True, sharey=True)
    for ax, objective in zip(axes, OBJECTIVES):
        for method in CORE4:
            xs, ys, ystd = series[(method, objective)]
            if show_std and np.isfinite(ystd).any():
                lower = ys - ystd
                upper = ys + ystd
                ax.fill_between(xs, lower, upper, color=METHOD_COLORS[method], alpha=0.16, linewidth=0)
            ax.plot(xs, ys, color=METHOD_COLORS[method], linewidth=2.0, label=METHOD_LABELS[method])
        ax.set_title(OBJ_LABELS[objective], loc="left")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.set_xlim(0, 100)
        ax.set_ylim(*use_ylim)
    axes[-1].set_xlabel("optimization step k")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(title, y=0.965)
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.025))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.91, bottom=0.12, hspace=0.36)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def final_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for method in CORE4:
        for objective in OBJECTIVES:
            group = [r for r in rows if r["method"] == method and r["objective"] == objective]
            if not group:
                continue
            group = sorted(group, key=lambda r: r["k"])
            final = group[-1]

            def last_finite(metric: str) -> tuple[int | float, float]:
                for row in reversed(group):
                    value = row.get(metric, math.nan)
                    if isinstance(value, (int, float)) and math.isfinite(float(value)):
                        return int(row["k"]), float(value)
                return math.nan, math.nan

            delta_prev_k, delta_prev_angle = last_finite("delta_prev_angle_degrees_mean")
            direction_prev_k, direction_prev_angle = last_finite("direction_prev_angle_degrees_mean")
            delta_direction_k, delta_direction_angle = last_finite("delta_direction_angle_degrees_mean")
            out.append(
                {
                    "objective": objective,
                    "method": method,
                    "final_k": final["k"],
                    "final_loss_mean": final["loss_mean"],
                    "final_boundary_ratio_mean": final["boundary_ratio_mean"],
                    "final_delta_pnorm_mean": final["delta_pnorm_mean"],
                    "last_finite_delta_prev_angle_k": delta_prev_k,
                    "last_finite_delta_prev_angle_degrees_mean": delta_prev_angle,
                    "last_finite_direction_prev_angle_k": direction_prev_k,
                    "last_finite_direction_prev_angle_degrees_mean": direction_prev_angle,
                    "last_finite_delta_direction_angle_k": delta_direction_k,
                    "last_finite_delta_direction_angle_degrees_mean": delta_direction_angle,
                    "final_high_frequency_energy_ratio_mean": final["high_frequency_energy_ratio_mean"],
                    "source_csv": final["source_csv"],
                }
            )
    return out


def boundary_gain_summary(rows: list[dict[str, Any]], threshold: float = 0.99) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for method in CORE4:
        for objective in OBJECTIVES:
            group = sorted(
                [r for r in rows if r["method"] == method and r["objective"] == objective],
                key=lambda r: r["k"],
            )
            if not group:
                continue
            initial = group[0]
            final = group[-1]
            hit = next((r for r in group if r["boundary_ratio_mean"] >= threshold), None)
            if hit is None:
                hit_k = math.nan
                hit_loss = math.nan
                post_gain = math.nan
                steps_after_hit = math.nan
            else:
                hit_k = int(hit["k"])
                hit_loss = float(hit["loss_mean"])
                post_gain = float(final["loss_mean"] - hit["loss_mean"])
                steps_after_hit = int(final["k"] - hit["k"])
            total_gain = float(final["loss_mean"] - initial["loss_mean"])
            out.append(
                {
                    "objective": objective,
                    "method": method,
                    "boundary_threshold": threshold,
                    "first_boundary_k": hit_k,
                    "initial_loss_mean": initial["loss_mean"],
                    "loss_at_first_boundary_mean": hit_loss,
                    "final_loss_mean": final["loss_mean"],
                    "gain_initial_to_final": total_gain,
                    "gain_boundary_to_final": post_gain,
                    "post_boundary_gain_fraction_of_total": post_gain / total_gain if total_gain != 0 and math.isfinite(post_gain) else math.nan,
                    "steps_after_boundary": steps_after_hit,
                    "final_boundary_ratio_mean": final["boundary_ratio_mean"],
                    "source_csv": final["source_csv"],
                }
            )
    return out


def write_doc(
    doc: Path,
    out_dir: Path,
    loss12_root: Path,
    loss3_root: Path,
    figures: list[Path],
    summary_rows: list[dict[str, Any]],
    gain_rows: list[dict[str, Any]],
) -> None:
    fields = [
        "objective",
        "method",
        "final_loss_mean",
        "final_boundary_ratio_mean",
        "final_delta_pnorm_mean",
        "last_finite_delta_prev_angle_k",
        "last_finite_delta_prev_angle_degrees_mean",
        "last_finite_direction_prev_angle_k",
        "last_finite_direction_prev_angle_degrees_mean",
    ]
    lines = [
        "# Loss1/Loss2/Loss3 Combined Core-Four 0..100 Visuals - 2026-05-21",
        "",
        "Status: generated from existing completed outputs. No attack was rerun.",
        "",
        "## Source Data",
        "",
        f"- Loss1/Loss2 source: `{rel(loss12_root)}`",
        f"- Loss3 source: `{rel(loss3_root)}`",
        f"- Combined output: `{rel(out_dir)}`",
        "- Scope: FNO / 1D Burgers `nu=0.001`, `epsilon=4`, `alpha=0.4`, `p=q=2`, `k=0..100`.",
        "",
        "## Figures",
        "",
    ]
    lines.extend(f"- `{rel(p)}`" for p in figures)
    lines.extend(
        [
            "",
            "Figure naming convention:",
            "- `*_no_std_by_method_*.png`: four method panels; each panel overlays `loss1`, `loss2`, and `loss3` as mean lines only.",
            "- `*_with_std_by_method_*.png`: same method-panel layout with mean +/- std shading.",
            "- `*_no_std_by_objective_*.png`: three objective rows; each row overlays the four optimizer methods as mean lines only.",
            "- `*_with_std_by_objective_*.png`: same objective-row layout with mean +/- std shading. Rows share the same x/y axis ranges within each metric figure.",
        ]
    )
    gain_fields = [
        "objective",
        "method",
        "first_boundary_k",
        "loss_at_first_boundary_mean",
        "final_loss_mean",
        "gain_boundary_to_final",
        "steps_after_boundary",
    ]
    lines.extend(
        [
            "",
            "## Observed Boundary-Gain Evidence",
            "",
            "Observed from `combined_boundary_gain_summary_0to100.csv`, using first mean boundary ratio >= 0.99.",
            "",
            "| " + " | ".join(gain_fields) + " |",
            "| " + " | ".join(["---"] * len(gain_fields)) + " |",
        ]
    )
    for row in gain_rows:
        vals = []
        for field in gain_fields:
            value = row.get(field, "")
            if isinstance(value, float):
                vals.append("nan" if not math.isfinite(value) else f"{value:.4g}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    lines.extend(["", "## Final/Last-Finite Summary", "", "| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"])
    for row in summary_rows:
        vals = []
        for field in fields:
            value = row.get(field, "")
            if isinstance(value, float):
                vals.append("nan" if not math.isfinite(value) else f"{value:.4g}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "Both orientations are preserved: `by_method` keeps one panel per optimizer method, and `by_objective` keeps one panel per objective with four method curves.",
            "Both no-std and with-std versions are generated. The with-std plots draw mean +/- std as translucent bands.",
            "The `combined_loss_normalized_*_0to100.png` plots normalize each objective/method curve by its own max over `k=0..100`; use them to compare shape rather than scale.",
            "Loss/boundary/delta-norm summary values are from `k=100`; angle summary values use the last finite value in `k=0..100`, because the final logged row can have no following direction for a step-to-step comparison.",
            "Loss3 angle means for direction/delta-direction are derived from the saved cosine columns in the Loss3 baseline CSV; their shaded std bands are obtained by converting cosine mean +/- std into an approximate angle std band.",
            "Inference from the overlaid curves and boundary-gain table: under this baseline p=q=2 setting, loss1/loss2 reach the epsilon boundary early and then add only a small amount of objective value by k=100; the normalized plot is the cleanest way to compare this shape against loss3 because the raw objective scales differ.",
        ]
    )
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--loss12-root", type=Path, default=DEFAULT_LOSS12_ROOT)
    ap.add_argument("--loss3-root", type=Path, default=DEFAULT_LOSS3_ROOT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    for attr in ("loss12_root", "loss3_root", "out_dir", "doc"):
        value = getattr(args, attr)
        if not value.is_absolute():
            setattr(args, attr, PROJECT_ROOT / value)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_combined(args.loss12_root, args.loss3_root)
    write_csv(args.out_dir / "combined_per_step_metrics_0to100.csv", rows)
    summary_rows = final_summary(rows)
    write_csv(args.out_dir / "combined_final_summary_0to100.csv", summary_rows)
    gain_rows = boundary_gain_summary(rows)
    write_csv(args.out_dir / "combined_boundary_gain_summary_0to100.csv", gain_rows)

    figure_specs = [
        ("loss_mean", "objective mean", "Loss1/Loss2/Loss3 objective means, k=0..100", "combined_loss_mean", False, None),
        ("loss_mean", "normalized objective mean", "Normalized objective shape, k=0..100", "combined_loss_normalized", True, (-0.03, 1.05)),
        ("boundary_ratio_mean", "mean ||delta||_2 / epsilon", "Boundary ratio comparison, k=0..100", "combined_boundary_ratio", False, (-0.03, 1.05)),
        ("delta_pnorm_mean", "mean ||delta||_2", "Delta L2 norm comparison, k=0..100", "combined_delta_pnorm", False, None),
        ("delta_prev_angle_degrees_mean", "mean angle(delta_k, delta_{k-1}) degrees", "Delta step-to-step angle comparison, k=0..100", "combined_delta_prev_angle", False, None),
        ("direction_prev_angle_degrees_mean", "mean angle(direction_k, direction_{k-1}) degrees", "Direction step-to-step angle comparison, k=0..100", "combined_direction_prev_angle", False, None),
        ("delta_direction_angle_degrees_mean", "mean angle(delta_k, direction_k) degrees", "Delta vs current direction angle comparison, k=0..100", "combined_delta_direction_angle", False, None),
        ("high_frequency_energy_ratio_mean", "mean high-frequency energy ratio", "Delta high-frequency ratio comparison, k=0..100", "combined_high_frequency_ratio", False, None),
    ]
    figs: list[Path] = []
    for metric, ylabel, title, stem, normalize_loss, ylim in figure_specs:
        for show_std, suffix in ((False, "no_std"), (True, "with_std")):
            figs.append(
                plot_by_method(
                    rows,
                    metric,
                    ylabel,
                    title + (" - by method, mean +/- std" if show_std else " - by method, mean only"),
                    args.out_dir / f"{stem}_{suffix}_by_method_0to100.png",
                    normalize_loss=normalize_loss,
                    show_std=show_std,
                    ylim=ylim,
                )
            )
            figs.append(
                plot_by_objective(
                    rows,
                    metric,
                    ylabel,
                    title + (" - by objective, mean +/- std" if show_std else " - by objective, mean only"),
                    args.out_dir / f"{stem}_{suffix}_by_objective_0to100.png",
                    normalize_loss=normalize_loss,
                    show_std=show_std,
                    ylim=ylim,
                )
            )
    manifest = {
        "status": "completed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "loss12_root": str(args.loss12_root),
        "loss3_root": str(args.loss3_root),
        "out_dir": str(args.out_dir),
        "doc": str(args.doc),
        "figure_paths": [str(p) for p in figs],
        "row_count": len(rows),
        "summary_row_count": len(summary_rows),
        "boundary_gain_summary_row_count": len(gain_rows),
        "note": "Generated from existing outputs; no attack rerun.",
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_doc(args.doc, args.out_dir, args.loss12_root, args.loss3_root, figs, summary_rows, gain_rows)
    print(f"[done] wrote {args.out_dir}")
    print(f"[done] wrote {args.doc}")


if __name__ == "__main__":
    main()
