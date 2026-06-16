#!/usr/bin/env python3
"""Fix Darcy cflow organized-release plot layout and wall-hour ranges.

This script edits only the organized release payload. It removes unwanted
work-hour main curves, redraws epoch/wall-hour curves with legends outside the
title area, adds transparent-line variants for the old required raw figures,
and writes a small manifest.
"""

from __future__ import annotations

import json
import argparse
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "outputs" / "darcy_cflow_timematched_organized_release_20260614"
METRICS_CSV = RELEASE / "data" / "source_tables" / "six_method_common_range_eval_metrics.csv"
SPLIT_CSV = RELEASE / "data" / "source_tables" / "six_method_common_range_eval_split_summary.csv"

MAIN_CURVES = RELEASE / "figures" / "main_curves"
RAW_ALPHA = RELEASE / "figures" / "diagnostic_existing" / "required_raw_figures_alpha"
POLISHED = RELEASE / "figures" / "polished_report"
MANIFEST = RELEASE / "manifests" / "cflow_plot_layout_fix_20260614.json"

WALL_XMAX_HOURS = 4.0
LINE_ALPHA = 0.58
BASELINE_ALPHA = 0.62
LEGEND_NCOL = 4

METHOD_LABEL = {
    "baseline": "baseline",
    "loss1": "loss1",
    "loss2": "loss2",
    "loss3": "loss3",
    "physics": "physics loss",
    "random_clean": "random clean",
    "random_solver": "random solver",
}

METHOD_COLOR = {
    "baseline": "#4b5563",
    "loss1": "#2563eb",
    "loss2": "#f97316",
    "loss3": "#16a34a",
    "physics": "#7c3aed",
    "random_clean": "#db2777",
    "random_solver": "#0891b2",
}

VARIANTS = {
    "all_six_models": ["loss1", "loss2", "loss3", "physics", "random_clean", "random_solver"],
    "no_random_clean": ["loss1", "loss2", "loss3", "physics", "random_solver"],
    "loss123_only": ["loss1", "loss2", "loss3"],
}


def metric_col(df: pd.DataFrame, metric: str) -> str:
    for col in [f"{metric}_plot", f"{metric}_corrected", metric]:
        if col in df.columns:
            return col
    raise KeyError(metric)


def dataset_label(dataset_id: str, max_len: int = 32) -> str:
    text = str(dataset_id)
    replacements = [
        ("darcy_lossdrop_pool_", ""),
        ("train_screen_binary_grf_", "train "),
        ("test_screen_binary_grf_", "test "),
        ("original_binary_grf_", "original "),
        ("binary_", "binary "),
        ("soft_", "soft "),
        ("maternfine", "matern fine"),
        ("highpass", "high pass"),
        ("bandpass", "band pass"),
        ("_", " "),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    text = " ".join(text.split())
    if len(text) > max_len:
        text = text[: max_len - 1].rstrip() + "."
    return text


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in ["epoch", "wall_seconds", "work_seconds", "manual_rank"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for metric in ["rmse", "relative_l2", "rmse_plot", "relative_l2_plot", "rmse_corrected", "relative_l2_corrected"]:
        if metric in df.columns:
            df[metric] = pd.to_numeric(df[metric], errors="coerce")
    return df


def load_eval() -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics_df = coerce_numeric(pd.read_csv(METRICS_CSV, low_memory=False))
    split_df = coerce_numeric(pd.read_csv(SPLIT_CSV, low_memory=False))
    return metrics_df, split_df


def delete_work_hour_figures() -> list[str]:
    removed: list[str] = []
    if not MAIN_CURVES.exists():
        return removed
    for p in MAIN_CURVES.rglob("*work_hours.png"):
        removed.append(str(p.relative_to(RELEASE)))
        p.unlink()
    return removed


def legend_lines(methods: list[str]) -> list[plt.Line2D]:
    lines = [
        plt.Line2D([0], [0], color=METHOD_COLOR["baseline"], lw=1.6, ls="--", alpha=BASELINE_ALPHA, label="baseline")
    ]
    lines.extend(
        plt.Line2D([0], [0], color=METHOD_COLOR[m], lw=2.1, alpha=LINE_ALPHA, label=METHOD_LABEL[m]) for m in methods
    )
    return lines


def x_values(df: pd.DataFrame, x_axis: str) -> tuple[pd.Series, str, tuple[float, float] | None]:
    if x_axis == "epoch":
        return pd.to_numeric(df["epoch"], errors="coerce"), "Epoch", None
    x = pd.to_numeric(df["wall_seconds"], errors="coerce") / 3600.0
    return x, "Wall-clock time (hours)", (0.0, WALL_XMAX_HOURS)


def filtered_xy(df: pd.DataFrame, x_axis: str, y_col: str) -> tuple[pd.Series, pd.Series]:
    x, _, _ = x_values(df, x_axis)
    y = pd.to_numeric(df[y_col], errors="coerce")
    mask = x.notna() & y.notna()
    if x_axis == "wall_hours":
        mask &= x <= WALL_XMAX_HOURS
    return x[mask], y[mask]


def baseline_value(df: pd.DataFrame, split: str | None, dataset_id: str | None, y_col: str) -> float | None:
    b = df[df["phase"].eq("baseline_before_adversarial_training")]
    if split is not None:
        b = b[b["split"].eq(split)]
    if dataset_id is not None:
        b = b[b["dataset_id"].eq(dataset_id)]
    values = pd.to_numeric(b[y_col], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.iloc[0])


def plot_split_means(df: pd.DataFrame, metric: str, x_axis: str, variant: str, yscale: str, out: Path) -> None:
    methods = VARIANTS[variant]
    y_col = metric_col(df, metric)
    data = df[
        df["phase"].eq("during_adversarial_training")
        & df["method"].isin(methods)
        & df["split"].isin(["train", "test", "generalization"])
    ].copy()
    if data.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18.5, 6.0), sharex=False, sharey=False)
    for ax, split in zip(axes, ["train", "test", "generalization"]):
        yb = baseline_value(df, split, None, y_col)
        if yb is not None:
            ax.axhline(
                yb,
                color=METHOD_COLOR["baseline"],
                linestyle="--",
                linewidth=1.35,
                alpha=BASELINE_ALPHA,
                label="baseline",
                zorder=1,
            )
        for method in methods:
            m = data[data["method"].eq(method) & data["split"].eq(split)].sort_values("epoch")
            x, y = filtered_xy(m, x_axis, y_col)
            if x.empty:
                continue
            ax.plot(
                x,
                y,
                color=METHOD_COLOR[method],
                linewidth=1.18,
                alpha=LINE_ALPHA,
                label=METHOD_LABEL[method],
                zorder=2,
            )
        _, xlabel, xlim = x_values(data, x_axis)
        if xlim:
            ax.set_xlim(*xlim)
            ax.set_xticks(np.arange(0, WALL_XMAX_HOURS + 0.001, 0.5))
        ax.set_title(split.title(), fontsize=12, pad=8)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(metric.replace("_", " ").upper() if metric == "rmse" else metric.replace("_", " ").title(), fontsize=10)
        ax.grid(True, color="#e5e7eb", linewidth=0.7, alpha=0.85)
        ax.tick_params(labelsize=8.5)
        if yscale == "log":
            ax.set_yscale("log")

    fig.suptitle(
        f"DarcyFlow {metric.replace('_', ' ').title()} Split Means ({variant.replace('_', ' ')}, {yscale})",
        fontsize=15,
        fontweight="semibold",
        y=0.975,
    )
    handles = legend_lines(methods)
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=min(LEGEND_NCOL, len(handles)),
        frameon=False,
        fontsize=9.5,
        handlelength=2.6,
        columnspacing=1.45,
        labelspacing=0.75,
    )
    fig.subplots_adjust(left=0.065, right=0.985, top=0.82, bottom=0.27, wspace=0.23)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def generalization_order(df: pd.DataFrame) -> list[str]:
    g = df[df["split"].eq("generalization")]
    order = (
        g[["dataset_id", "manual_rank"]]
        .drop_duplicates()
        .sort_values(["manual_rank", "dataset_id"], na_position="last")["dataset_id"]
        .astype(str)
        .tolist()
    )
    return order


def plot_generalization_grid(df: pd.DataFrame, metric: str, x_axis: str, variant: str, yscale: str, part: int, out: Path) -> None:
    methods = VARIANTS[variant]
    y_col = metric_col(df, metric)
    ids = generalization_order(df)[(part - 1) * 25 : part * 25]
    if not ids:
        return
    data = df[
        df["phase"].eq("during_adversarial_training")
        & df["method"].isin(methods)
        & df["dataset_id"].isin(ids)
        & df["split"].eq("generalization")
    ].copy()
    if data.empty:
        return

    fig, axes = plt.subplots(5, 5, figsize=(25.5, 17.2), sharex=False, sharey=False)
    axes = axes.ravel()
    for ax, dataset_id in zip(axes, ids):
        yb = baseline_value(df, "generalization", dataset_id, y_col)
        if yb is not None:
            ax.axhline(yb, color=METHOD_COLOR["baseline"], linestyle="--", linewidth=0.9, alpha=BASELINE_ALPHA, zorder=1)
        for method in methods:
            m = data[data["method"].eq(method) & data["dataset_id"].eq(dataset_id)].sort_values("epoch")
            x, y = filtered_xy(m, x_axis, y_col)
            if x.empty:
                continue
            ax.plot(x, y, color=METHOD_COLOR[method], linewidth=0.78, alpha=LINE_ALPHA, zorder=2)
        _, _, xlim = x_values(data, x_axis)
        if xlim:
            ax.set_xlim(*xlim)
            ax.set_xticks([0, 1, 2, 3, 4])
        ax.set_title(dataset_label(dataset_id), fontsize=8.4, pad=4)
        ax.grid(True, color="#e5e7eb", linewidth=0.45, alpha=0.8)
        ax.tick_params(labelsize=6.8, length=2)
        if yscale == "log":
            ax.set_yscale("log")
    for ax in axes[len(ids) :]:
        ax.axis("off")

    handles = legend_lines(methods)
    _, xlabel, _ = x_values(data, x_axis)
    fig.suptitle(
        f"DarcyFlow Generalization {metric.replace('_', ' ').title()} Part {part} ({variant.replace('_', ' ')}, {yscale})",
        fontsize=16,
        fontweight="semibold",
        y=0.984,
    )
    fig.supxlabel(xlabel, y=0.085, fontsize=11)
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.018),
        ncol=min(LEGEND_NCOL, len(handles)),
        frameon=False,
        fontsize=10,
        handlelength=2.5,
        columnspacing=1.45,
        labelspacing=0.75,
    )
    fig.subplots_adjust(left=0.045, right=0.99, top=0.925, bottom=0.145, hspace=0.54, wspace=0.25)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def redraw_main_curves(metrics_df: pd.DataFrame, split_df: pd.DataFrame, *, include_grids: bool = True) -> list[str]:
    written: list[str] = []
    for variant in VARIANTS:
        for yscale in ["linear", "log"]:
            for metric in ["rmse", "relative_l2"]:
                for x_axis in ["epoch", "wall_hours"]:
                    out = MAIN_CURVES / variant / yscale / f"{metric}_split_means_vs_{x_axis}.png"
                    plot_split_means(split_df, metric, x_axis, variant, yscale, out)
                    written.append(str(out.relative_to(RELEASE)))
                    if include_grids:
                        for part in [1, 2]:
                            out = MAIN_CURVES / variant / yscale / f"{metric}_generalization_part{part:02d}_vs_{x_axis}.png"
                            plot_generalization_grid(metrics_df, metric, x_axis, variant, yscale, part, out)
                            written.append(str(out.relative_to(RELEASE)))
    return written


def plot_required_raw_alpha(metrics_df: pd.DataFrame, split_df: pd.DataFrame, *, include_grids: bool = True) -> list[str]:
    RAW_ALPHA.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    # Alpha replacements for the old split-mean raw figures, using epoch and the requested wall-clock axis.
    for metric in ["rmse", "relative_l2"]:
        for x_axis, tag in [("epoch", "epoch"), ("wall_hours", "wall_clock")]:
            out = RAW_ALPHA / f"six_method_{tag}_{metric}_train_test_generalization_alpha.png"
            plot_split_means(split_df, metric, x_axis, "all_six_models", "linear", out)
            written.append(str(out.relative_to(RELEASE)))
        for x_axis, tag in [("epoch", "epoch"), ("wall_hours", "wall")]:
            for part in [1, 2]:
                if not include_grids:
                    continue
                out = RAW_ALPHA / f"six_method_{metric}_generalization_part{part:02d}_vs_{tag}_alpha.png"
                plot_generalization_grid(metrics_df, metric, x_axis, "all_six_models", "linear", part, out)
                written.append(str(out.relative_to(RELEASE)))
    return written


def plot_polished_overview(df: pd.DataFrame) -> list[str]:
    POLISHED.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for variant, methods in VARIANTS.items():
        for metric in ["rmse", "relative_l2"]:
            y_col = metric_col(df, metric)
            data = df[
                df["phase"].eq("during_adversarial_training")
                & df["method"].isin(methods)
                & df["split"].isin(["train", "test", "generalization"])
            ].copy()
            if data.empty:
                continue
            fig, axes = plt.subplots(2, 3, figsize=(19, 9.3), sharey="row")
            for row, x_axis in enumerate(["epoch", "wall_hours"]):
                for col, split in enumerate(["train", "test", "generalization"]):
                    ax = axes[row, col]
                    yb = baseline_value(df, split, None, y_col)
                    if yb is not None:
                        ax.axhline(
                            yb,
                            color=METHOD_COLOR["baseline"],
                            linestyle="--",
                            linewidth=1.4,
                            alpha=BASELINE_ALPHA,
                            label="baseline",
                        )
                    for method in methods:
                        m = data[data["method"].eq(method) & data["split"].eq(split)].sort_values("epoch")
                        x, y = filtered_xy(m, x_axis, y_col)
                        if x.empty:
                            continue
                        ax.plot(x, y, color=METHOD_COLOR[method], linewidth=1.25, alpha=LINE_ALPHA, label=METHOD_LABEL[method])
                    _, xlabel, xlim = x_values(data, x_axis)
                    if xlim:
                        ax.set_xlim(*xlim)
                        ax.set_xticks(np.arange(0, WALL_XMAX_HOURS + 0.001, 0.5))
                    ax.set_title(f"{split.title()} - {xlabel}", fontsize=11, pad=7)
                    ax.grid(True, color="#e5e7eb", linewidth=0.7, alpha=0.82)
                    ax.tick_params(labelsize=8)
                    ax.set_xlabel(xlabel, fontsize=9.5)
                    if col == 0:
                        ax.set_ylabel(metric.replace("_", " ").title(), fontsize=10)
            fig.suptitle(
                f"Polished DarcyFlow {metric.replace('_', ' ').title()} Overview - {variant.replace('_', ' ')}",
                fontsize=16,
                fontweight="semibold",
                y=0.982,
            )
            handles = legend_lines(methods)
            fig.legend(
                handles=handles,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.018),
                ncol=min(LEGEND_NCOL, len(handles)),
                frameon=False,
                fontsize=10,
                handlelength=2.6,
                columnspacing=1.45,
                labelspacing=0.75,
            )
            fig.subplots_adjust(left=0.06, right=0.985, top=0.88, bottom=0.18, hspace=0.38, wspace=0.18)
            out = POLISHED / f"polished_{variant}_{metric}_split_means_epoch_wall.png"
            fig.savefig(out, dpi=220)
            plt.close(fig)
            written.append(str(out.relative_to(RELEASE)))
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split-polished-only",
        action="store_true",
        help="Only redraw split-mean and polished overview plots; keep existing 5x5 generalization panels.",
    )
    args = parser.parse_args()
    metrics_df, split_df = load_eval()
    removed = delete_work_hour_figures()
    include_grids = not args.split_polished_only
    main_written = redraw_main_curves(metrics_df, split_df, include_grids=include_grids)
    raw_alpha_written = plot_required_raw_alpha(metrics_df, split_df, include_grids=include_grids)
    polished_written = plot_polished_overview(split_df)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "release_dir": str(RELEASE.relative_to(ROOT)),
        "actions": {
            "deleted_work_hour_main_curve_png": len(removed),
            "redrawn_epoch_and_wall_hour_main_curve_png": len(main_written),
            "added_required_raw_alpha_png": len(raw_alpha_written),
            "added_polished_report_png": len(polished_written),
        },
        "wall_hour_xmax": WALL_XMAX_HOURS,
        "line_alpha": LINE_ALPHA,
        "baseline_alpha": BASELINE_ALPHA,
        "mode": "split_polished_only" if args.split_polished_only else "full",
        "deleted_files": removed,
        "redrawn_files": main_written,
        "required_raw_alpha_files": raw_alpha_written,
        "polished_report_files": polished_written,
        "notes": [
            "Main curves now keep only epoch and wall_hours axes.",
            "All main wall_hours plots are truncated to 0-4 wall-clock hours.",
            "Legends are placed outside the axes at the bottom; titles are at the top with reserved spacing.",
            "Original required_raw_figures_previous files are preserved; transparent-line variants are written separately.",
            "Random clean/random solver have no wall_seconds in the source split/eval tables; wall-hour legends still list them, but no wall-hour curve is fabricated when data is missing.",
        ],
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["actions"], indent=2))
    print(MANIFEST)


if __name__ == "__main__":
    main()
