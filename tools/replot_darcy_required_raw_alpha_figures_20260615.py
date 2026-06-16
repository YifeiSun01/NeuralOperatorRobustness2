#!/usr/bin/env python3
"""Rebuild Darcy required-raw alpha figures with all six trained methods.

The original non-alpha required raw figures are preserved. This script writes a
separate transparent-line set whose filenames mirror the previous raw figures
with ``_alpha`` appended before ``.png``.
"""

from __future__ import annotations

import json
import math
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "outputs" / "darcy_cflow_timematched_organized_release_20260614"
SOURCE = RELEASE / "data" / "source_tables"
FIG_ROOT = RELEASE / "figures" / "diagnostic_existing"
PREVIOUS_DIR = FIG_ROOT / "required_raw_figures_previous"
ALPHA_DIR = FIG_ROOT / "required_raw_figures_alpha"
RETIRED_DIR = FIG_ROOT / "required_raw_figures_alpha_retired_wall_axis_20260615"
MANIFEST = RELEASE / "manifests" / "required_raw_alpha_full_six_20260615.json"

SPLIT_CSV = SOURCE / "six_method_common_range_eval_split_summary.csv"
METRICS_CSV = SOURCE / "six_method_common_range_eval_metrics.csv"

METHOD_ORDER = ["loss1", "loss2", "loss3", "physics", "random_clean", "random_solver"]
METHOD_LABELS = {
    "loss1": "loss1",
    "loss2": "loss2",
    "loss3": "loss3",
    "physics": "Physics Loss",
    "random_clean": "random clean",
    "random_solver": "random solver",
}
METHOD_COLORS = {
    "loss1": "#1b6ca8",
    "loss2": "#d95f02",
    "loss3": "#2ca25f",
    "physics": "#7b3294",
    "random_clean": "#c77dbb",
    "random_solver": "#8c564b",
}
METRICS = ("rmse", "relative_l2")
SPLITS = ("train", "test", "generalization")
PHASES = ("baseline_before_adversarial_training", "during_adversarial_training")
TARGET_MAX_EPOCH = 3500
LINE_ALPHA = 0.58
BASELINE_ALPHA = 0.72


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 240,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.facecolor": "white",
            "figure.facecolor": "#fbfaf7",
            "axes.edgecolor": "#aaa59b",
            "axes.grid": True,
            "grid.color": "#d8d4c8",
            "grid.alpha": 0.42,
            "grid.linewidth": 0.55,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "axes.titleweight": "semibold",
            "axes.titlepad": 6.0,
            "xtick.major.pad": 2.5,
            "ytick.major.pad": 2.5,
        }
    )


def metric_col(metric: str) -> str:
    plot_col = f"{metric}_plot"
    return plot_col


def metric_label(metric: str) -> str:
    if metric == "rmse":
        return "RMSE"
    if metric == "relative_l2":
        return "Relative L2"
    return metric


def short_dataset_name(dataset_id: str) -> str:
    name = str(dataset_id)
    for prefix in ("darcy_binary_loss3targeted_20260611_", "darcy_lossdrop_pool_", "darcy_"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    name = name.replace("_", " ")
    if len(name) > 40:
        name = f"{name[:37]}..."
    return name


def dataset_sort_key(row: pd.Series) -> tuple[float, int, str]:
    rank = row.get("manual_rank", np.nan)
    try:
        rank_key = float(rank)
    except (TypeError, ValueError):
        rank_key = math.inf
    dataset_id = str(row["dataset_id"])
    match = re.search(r"_(\d{2})_", dataset_id)
    number = int(match.group(1)) if match else 10_000
    return (rank_key if math.isfinite(rank_key) else math.inf, number, dataset_id)


def read_split_summary() -> pd.DataFrame:
    usecols = [
        "method",
        "phase",
        "epoch",
        "global_step",
        "split",
        "rmse_plot",
        "relative_l2_plot",
        "work_seconds",
    ]
    df = pd.read_csv(SPLIT_CSV, usecols=usecols)
    df = df[df["method"].isin(METHOD_ORDER) & df["phase"].isin(PHASES) & df["split"].isin(SPLITS)].copy()
    return df


def read_eval_metrics() -> pd.DataFrame:
    usecols = [
        "method",
        "phase",
        "epoch",
        "global_step",
        "split",
        "dataset_id",
        "manual_rank",
        "rmse_plot",
        "relative_l2_plot",
        "work_seconds",
    ]
    df = pd.read_csv(METRICS_CSV, usecols=usecols)
    df = df[df["method"].isin(METHOD_ORDER) & df["phase"].isin(PHASES) & (df["split"] == "generalization")].copy()
    df["dataset_id"] = df["dataset_id"].astype(str)
    return df


def retire_wall_alpha_files() -> list[str]:
    """Move old wall-axis alpha files aside because random methods lack wall time."""
    ALPHA_DIR.mkdir(parents=True, exist_ok=True)
    RETIRED_DIR.mkdir(parents=True, exist_ok=True)
    moved: list[str] = []
    patterns = ("*_vs_wall_alpha.png", "six_method_wall_clock_*_alpha.png")
    for pattern in patterns:
        for path in sorted(ALPHA_DIR.glob(pattern)):
            target = RETIRED_DIR / path.name
            if target.exists():
                target.unlink()
            shutil.move(str(path), str(target))
            moved.append(str(path.relative_to(RELEASE)))
    return moved


def common_work_limit(*frames: pd.DataFrame) -> float:
    max_by_method = []
    for frame in frames:
        phase_frame = frame[frame["phase"].isin(PHASES)]
        per = phase_frame.groupby("method")["work_seconds"].max()
        for method in METHOD_ORDER:
            value = per.get(method)
            if value is None or not math.isfinite(float(value)):
                raise ValueError(f"missing work_seconds for {method}")
        max_by_method.append(per.loc[METHOD_ORDER].astype(float))
    merged = pd.concat(max_by_method, axis=1).min(axis=1)
    return float(merged.min())


def baseline_split_values(split_df: pd.DataFrame, metric: str) -> dict[str, float]:
    col = metric_col(metric)
    base = split_df[split_df["phase"] == "baseline_before_adversarial_training"]
    return base.groupby("split")[col].mean().to_dict()


def baseline_dataset_values(metrics_df: pd.DataFrame, metric: str) -> dict[str, float]:
    col = metric_col(metric)
    base = metrics_df[metrics_df["phase"] == "baseline_before_adversarial_training"]
    return base.groupby("dataset_id")[col].mean().to_dict()


def legend_handles(linewidth: float = 2.0) -> list[plt.Line2D]:
    handles = [
        plt.Line2D([0], [0], color=METHOD_COLORS[m], lw=linewidth, alpha=LINE_ALPHA, label=METHOD_LABELS[m])
        for m in METHOD_ORDER
    ]
    handles.append(
        plt.Line2D(
            [0],
            [0],
            color="#474747",
            lw=1.25,
            alpha=BASELINE_ALPHA,
            linestyle=(0, (4, 3)),
            label="baseline",
        )
    )
    return handles


def plot_split_mean(
    split_df: pd.DataFrame,
    metric: str,
    x_axis: str,
    max_work_seconds: float,
    output: Path,
) -> dict[str, object]:
    col = metric_col(metric)
    if x_axis == "epoch":
        x_col = "epoch"
        x_label = "epoch"
        x_limit = TARGET_MAX_EPOCH
    else:
        x_col = "work_seconds"
        x_label = "work-clock seconds"
        x_limit = max_work_seconds

    data = split_df.copy()
    if x_axis == "epoch":
        data = data[data["epoch"] <= TARGET_MAX_EPOCH]
    else:
        data = data[data["work_seconds"] <= max_work_seconds]

    baseline = baseline_split_values(split_df, metric)
    fig, axes = plt.subplots(1, 3, figsize=(22.8, 6.3), sharey=False)
    subplot_counts: dict[str, list[str]] = {}
    for ax, split in zip(axes, SPLITS):
        panel = data[data["split"] == split]
        drawn: list[str] = []
        for method in METHOD_ORDER:
            sub = panel[panel["method"] == method].sort_values(x_col)
            sub = sub[np.isfinite(sub[x_col]) & np.isfinite(sub[col])]
            if sub.empty:
                continue
            ax.plot(
                sub[x_col].to_numpy(),
                sub[col].to_numpy(),
                color=METHOD_COLORS[method],
                lw=1.55,
                alpha=LINE_ALPHA,
                label=METHOD_LABELS[method],
            )
            drawn.append(method)
        base_value = baseline.get(split)
        if base_value is not None and math.isfinite(float(base_value)):
            ax.axhline(
                float(base_value),
                color="#474747",
                linewidth=1.05,
                alpha=BASELINE_ALPHA,
                linestyle=(0, (4, 3)),
            )
        ax.set_title(split if split != "generalization" else "generalization mean", fontsize=12.2)
        ax.set_xlabel(x_label)
        ax.set_ylabel(metric_label(metric))
        ax.set_yscale("log")
        ax.tick_params(labelsize=9.3)
        ax.set_xlim(0, x_limit)
        subplot_counts[split] = drawn

    fig.suptitle(f"Darcy six-method {metric_label(metric)}", fontsize=16.4, y=0.982)
    fig.text(
        0.5,
        0.928,
        f"train / test / generalization mean by {x_label}; transparent alpha={LINE_ALPHA}",
        ha="center",
        va="center",
        fontsize=10.3,
        color="#4f4b45",
    )
    fig.legend(
        handles=legend_handles(2.1),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=7,
        fontsize=9.2,
        handlelength=2.6,
        columnspacing=1.28,
    )
    fig.subplots_adjust(top=0.835, bottom=0.19, left=0.055, right=0.988, wspace=0.27)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.16, facecolor=fig.get_facecolor())
    plt.close(fig)
    return {"file": str(output.relative_to(RELEASE)), "subplot_methods": subplot_counts}


def ordered_generalization_ids(metrics_df: pd.DataFrame) -> list[str]:
    base = metrics_df[metrics_df["phase"] == "baseline_before_adversarial_training"].copy()
    base = base.drop_duplicates("dataset_id")
    base["_sort"] = base.apply(dataset_sort_key, axis=1)
    ids = base.sort_values("_sort")["dataset_id"].tolist()
    return ids[:50]


def plot_generalization_grid(
    metrics_df: pd.DataFrame,
    metric: str,
    x_axis: str,
    dataset_ids: list[str],
    part: int,
    max_work_seconds: float,
    output: Path,
) -> dict[str, object]:
    col = metric_col(metric)
    if x_axis == "epoch":
        x_col = "epoch"
        x_label = "epoch"
        x_limit = TARGET_MAX_EPOCH
    else:
        x_col = "work_seconds"
        x_label = "work-clock seconds"
        x_limit = max_work_seconds

    data = metrics_df[metrics_df["dataset_id"].isin(dataset_ids)].copy()
    if x_axis == "epoch":
        data = data[data["epoch"] <= TARGET_MAX_EPOCH]
    else:
        data = data[data["work_seconds"] <= max_work_seconds]

    baseline = baseline_dataset_values(metrics_df, metric)
    fig, axes = plt.subplots(5, 5, figsize=(24.0, 18.0), sharex=False, sharey=False)
    axes = axes.reshape(-1)
    subplot_counts: dict[str, list[str]] = {}

    for ax, dataset_id in zip(axes, dataset_ids):
        panel = data[data["dataset_id"] == dataset_id]
        drawn: list[str] = []
        for method in METHOD_ORDER:
            sub = panel[panel["method"] == method].sort_values(x_col)
            sub = sub[np.isfinite(sub[x_col]) & np.isfinite(sub[col])]
            if sub.empty:
                continue
            ax.plot(
                sub[x_col].to_numpy(),
                sub[col].to_numpy(),
                color=METHOD_COLORS[method],
                linewidth=1.0,
                alpha=LINE_ALPHA,
            )
            drawn.append(method)
        base_value = baseline.get(dataset_id)
        if base_value is not None and math.isfinite(float(base_value)):
            ax.axhline(
                float(base_value),
                color="#474747",
                linewidth=0.78,
                alpha=BASELINE_ALPHA,
                linestyle=(0, (4, 3)),
            )
        ax.set_title(short_dataset_name(dataset_id), fontsize=7.3)
        ax.set_yscale("log")
        ax.grid(alpha=0.22, linewidth=0.45)
        ax.tick_params(labelsize=6.8)
        ax.set_xlim(0, x_limit)
        subplot_counts[dataset_id] = drawn

    for ax in axes[len(dataset_ids) :]:
        ax.axis("off")

    start = 1 + (part - 1) * 25
    end = start + len(dataset_ids) - 1
    fig.suptitle(f"Darcy six-method generalization {metric_label(metric)}", fontsize=16.0, y=0.992)
    fig.text(
        0.5,
        0.968,
        f"datasets {start}-{end} by {x_label}; transparent alpha={LINE_ALPHA}",
        ha="center",
        va="center",
        fontsize=9.8,
        color="#4f4b45",
    )
    fig.legend(
        handles=legend_handles(1.5),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.946),
        ncol=7,
        fontsize=8.4,
        handlelength=2.4,
        columnspacing=1.15,
    )
    fig.text(0.5, 0.028, x_label, ha="center", va="center", fontsize=10.0, color="#37342f")
    fig.text(0.012, 0.5, metric_label(metric), ha="center", va="center", rotation="vertical", fontsize=10.0, color="#37342f")
    fig.subplots_adjust(top=0.902, bottom=0.06, left=0.055, right=0.992, hspace=0.54, wspace=0.25)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.18, facecolor=fig.get_facecolor())
    plt.close(fig)
    return {"file": str(output.relative_to(RELEASE)), "subplot_methods": subplot_counts}


def validate_manifest(records: list[dict[str, object]]) -> dict[str, object]:
    incomplete: list[dict[str, object]] = []
    for record in records:
        for panel, methods in record["subplot_methods"].items():  # type: ignore[index,union-attr]
            missing = [m for m in METHOD_ORDER if m not in methods]
            if missing:
                incomplete.append({"file": record["file"], "panel": panel, "missing_methods": missing})
    return {
        "expected_methods": METHOD_ORDER,
        "expected_display_names": [METHOD_LABELS[m] for m in METHOD_ORDER],
        "expected_method_count": len(METHOD_ORDER),
        "incomplete_panels": incomplete,
        "all_panels_have_six_methods": not incomplete,
    }


def main() -> None:
    setup_style()
    if not PREVIOUS_DIR.exists():
        raise FileNotFoundError(PREVIOUS_DIR)
    ALPHA_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    retired = retire_wall_alpha_files()
    split_df = read_split_summary()
    metrics_df = read_eval_metrics()
    max_work_seconds = common_work_limit(split_df, metrics_df)
    gen_ids = ordered_generalization_ids(metrics_df)

    records: list[dict[str, object]] = []
    for metric in METRICS:
        records.append(
            plot_split_mean(
                split_df,
                metric,
                "epoch",
                max_work_seconds,
                ALPHA_DIR / f"six_method_epoch_{metric}_train_test_generalization_alpha.png",
            )
        )
        records.append(
            plot_split_mean(
                split_df,
                metric,
                "work",
                max_work_seconds,
                ALPHA_DIR / f"six_method_work_clock_{metric}_train_test_generalization_alpha.png",
            )
        )
        for part, ids in enumerate((gen_ids[:25], gen_ids[25:50]), start=1):
            records.append(
                plot_generalization_grid(
                    metrics_df,
                    metric,
                    "epoch",
                    ids,
                    part,
                    max_work_seconds,
                    ALPHA_DIR / f"six_method_{metric}_generalization_part{part:02d}_vs_epoch_alpha.png",
                )
            )
            records.append(
                plot_generalization_grid(
                    metrics_df,
                    metric,
                    "work",
                    ids,
                    part,
                    max_work_seconds,
                    ALPHA_DIR / f"six_method_{metric}_generalization_part{part:02d}_vs_work_alpha.png",
                )
            )

    validation = validate_manifest(records)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "release_dir": str(RELEASE.relative_to(ROOT)),
        "source_tables": [str(SPLIT_CSV.relative_to(ROOT)), str(METRICS_CSV.relative_to(ROOT))],
        "previous_raw_dir_preserved": str(PREVIOUS_DIR.relative_to(RELEASE)),
        "alpha_dir": str(ALPHA_DIR.relative_to(RELEASE)),
        "retired_old_wall_alpha_files": retired,
        "retired_note": (
            "Old wall-axis alpha files were moved aside because source wall_seconds is NaN "
            "for random_clean and random_solver; the rebuilt required-raw alpha set mirrors "
            "the previous work-clock raw figures and therefore contains all six trained methods."
        ),
        "line_alpha": LINE_ALPHA,
        "baseline_alpha": BASELINE_ALPHA,
        "target_max_epoch": TARGET_MAX_EPOCH,
        "common_max_work_seconds": max_work_seconds,
        "generated_files": [record["file"] for record in records],
        "validation": validation,
    }
    MANIFEST.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
