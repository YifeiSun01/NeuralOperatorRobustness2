#!/usr/bin/env python3
"""Build moving-average Darcy required figures from the archived curve CSVs."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "outputs" / "darcy_sir20_required_figures_only_20260614"
DATA = BUNDLE / "data"
METHOD_ORDER = ["loss1", "loss2", "loss3", "physics", "random_clean", "random_solver"]
LABELS = {
    "loss1": "loss1",
    "loss2": "loss2",
    "loss3": "loss3",
    "physics": "physics",
    "random_clean": "random clean",
    "random_solver": "random solver",
}
COLORS = {
    "loss1": "#1b6ca8",
    "loss2": "#d95f02",
    "loss3": "#2ca25f",
    "physics": "#7b3294",
    "random_clean": "#c77dbb",
    "random_solver": "#8c564b",
}
METRICS = ["rmse", "relative_l2"]
SPLITS = ["train", "test", "generalization"]
PLOT_PHASES = {"baseline_before_adversarial_training", "during_adversarial_training"}


def setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 230,
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
            "axes.titlepad": 7.0,
            "xtick.major.pad": 2.5,
            "ytick.major.pad": 2.5,
        }
    )


def metric_label(metric: str) -> str:
    return "RMSE" if metric == "rmse" else "Relative L2"


def smooth_series(values: pd.Series, window: int) -> pd.Series:
    window = max(1, int(window))
    min_periods = max(1, min(window, max(3, window // 5)))
    return values.rolling(window=window, center=True, min_periods=min_periods).mean()


def add_smoothed_column(df: pd.DataFrame, metric: str, group_cols: list[str], window: int) -> pd.DataFrame:
    out = df.sort_values(group_cols + ["epoch"]).copy()
    value_col = f"{metric}_plot"
    smooth_col = f"{metric}_ma"
    out[smooth_col] = out.groupby(group_cols, dropna=False)[value_col].transform(lambda s: smooth_series(s, window))
    out[smooth_col] = out[smooth_col].fillna(out[value_col])
    return out


def common_limits(split_df: pd.DataFrame, metrics_df: pd.DataFrame) -> tuple[int, float]:
    split_per = split_df[split_df["phase"].isin(PLOT_PHASES)].groupby("method").agg(
        max_epoch=("epoch", "max"), max_work=("work_seconds", "max")
    )
    metrics_per = metrics_df[metrics_df["phase"].isin(PLOT_PHASES)].groupby("method").agg(
        max_epoch=("epoch", "max"), max_work=("work_seconds", "max")
    )
    max_epoch = int(min(split_per["max_epoch"].min(), metrics_per["max_epoch"].min()))
    max_work = float(min(split_per["max_work"].min(), metrics_per["max_work"].min()))
    return max_epoch, max_work


def baseline_split_values(split_df: pd.DataFrame, metric: str) -> dict[str, float]:
    base = split_df[(split_df["phase"] == "baseline_before_adversarial_training") & split_df["split"].isin(SPLITS)]
    return {str(r.split): float(getattr(r, f"{metric}_plot")) for r in base.drop_duplicates("split").itertuples(index=False)}


def baseline_dataset_values(metrics_df: pd.DataFrame, metric: str) -> dict[str, float]:
    base = metrics_df[(metrics_df["phase"] == "baseline_before_adversarial_training") & (metrics_df["split"] == "generalization")].copy()
    base["dataset_id"] = base["dataset_id"].astype(str)
    return base.groupby("dataset_id")[f"{metric}_plot"].mean().to_dict()


def dataset_order(df: pd.DataFrame) -> list[str]:
    base = df[(df["phase"] == "baseline_before_adversarial_training") & (df["split"] == "generalization")].copy()
    base["dataset_id"] = base["dataset_id"].astype(str)

    def key(dataset_id: str) -> tuple[int, str]:
        match = re.search(r"_(\d{2})_", dataset_id)
        return (int(match.group(1)) if match else 10_000, dataset_id)

    return sorted(base["dataset_id"].drop_duplicates().tolist(), key=key)[:50]


def short_name(dataset_id: str) -> str:
    for prefix in ["darcy_binary_loss3targeted_20260611_", "darcy_"]:
        if dataset_id.startswith(prefix):
            dataset_id = dataset_id[len(prefix) :]
    name = dataset_id[:38]
    if len(dataset_id) > 38:
        name = f"{name}..."
    return name


def plot_split_mean(df: pd.DataFrame, metric: str, x_axis: str, max_epoch: int, max_work: float, window: int, out: Path) -> None:
    x_col = "epoch" if x_axis == "epoch" else "work_seconds"
    x_label = "epoch" if x_axis == "epoch" else "work-clock seconds"
    y_col = f"{metric}_ma"
    base = baseline_split_values(df, metric)
    plot_df = df[df["phase"].isin(PLOT_PHASES) & df["split"].isin(SPLITS)].copy()
    if x_axis == "epoch":
        plot_df = plot_df[plot_df["epoch"] <= max_epoch]
    else:
        plot_df = plot_df[plot_df["work_seconds"] <= max_work]
    fig, axes = plt.subplots(1, 3, figsize=(22.8, 6.35), sharey=False)
    for ax, split in zip(axes, SPLITS):
        s = plot_df[plot_df["split"] == split]
        for method in METHOD_ORDER:
            sub = s[s["method"] == method].sort_values(x_col)
            if sub.empty:
                continue
            ax.plot(sub[x_col], sub[y_col], color=COLORS[method], lw=1.55, alpha=0.96, label=LABELS[method])
        b = base.get(split)
        if b is not None and math.isfinite(b):
            ax.axhline(b, color="#555555", linewidth=0.95, alpha=0.68)
        ax.set_title(split if split != "generalization" else "generalization mean", fontsize=12.5)
        ax.set_xlabel(x_label)
        ax.set_ylabel(metric_label(metric))
        ax.set_yscale("log")
        ax.tick_params(labelsize=9.5)
        if x_axis == "epoch":
            ax.set_xlim(0, max_epoch)
        else:
            ax.set_xlim(0, max_work)
    handles = [plt.Line2D([0], [0], color=COLORS[m], lw=2.1, label=LABELS[m]) for m in METHOD_ORDER]
    handles.append(plt.Line2D([0], [0], color="#555555", lw=1.4, label="baseline"))
    fig.suptitle(f"Darcy six-method {metric_label(metric)} moving average", fontsize=16.5, y=0.982)
    fig.text(
        0.5,
        0.925,
        f"centered moving average, window={window} epochs; train / test / generalization mean by {x_label}",
        ha="center",
        va="center",
        fontsize=10.3,
        color="#4f4b45",
    )
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.888), ncol=7, fontsize=9.0, handlelength=2.5, columnspacing=1.35)
    fig.subplots_adjust(top=0.79, bottom=0.145, left=0.055, right=0.988, wspace=0.27)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.16, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_generalization_grid(
    df: pd.DataFrame,
    metric: str,
    x_axis: str,
    ids: list[str],
    part: int,
    max_epoch: int,
    max_work: float,
    window: int,
    out: Path,
) -> None:
    x_col = "epoch" if x_axis == "epoch" else "work_seconds"
    x_label = "epoch" if x_axis == "epoch" else "work-clock seconds"
    y_col = f"{metric}_ma"
    base = baseline_dataset_values(df, metric)
    plot_df = df[df["phase"].isin(PLOT_PHASES) & (df["split"] == "generalization") & df["dataset_id"].astype(str).isin(ids)].copy()
    if x_axis == "epoch":
        plot_df = plot_df[plot_df["epoch"] <= max_epoch]
    else:
        plot_df = plot_df[plot_df["work_seconds"] <= max_work]
    fig, axes = plt.subplots(5, 5, figsize=(23.5, 17.7), sharex=False, sharey=False)
    axes = axes.reshape(-1)
    for ax, dataset_id in zip(axes, ids):
        for method in METHOD_ORDER:
            sub = plot_df[(plot_df["method"] == method) & (plot_df["dataset_id"].astype(str) == dataset_id)].sort_values(x_col)
            if sub.empty:
                continue
            ax.plot(sub[x_col], sub[y_col], color=COLORS[method], linewidth=1.12, alpha=0.95)
        b = base.get(dataset_id)
        if b is not None and math.isfinite(float(b)):
            ax.axhline(float(b), color="#555555", linewidth=0.76, alpha=0.64)
        ax.set_title(short_name(dataset_id), fontsize=7.4)
        ax.set_yscale("log")
        ax.grid(alpha=0.22)
        ax.tick_params(labelsize=6.9)
        if x_axis == "epoch":
            ax.set_xlim(0, max_epoch)
        else:
            ax.set_xlim(0, max_work)
    handles = [plt.Line2D([0], [0], color=COLORS[m], lw=1.6, label=LABELS[m]) for m in METHOD_ORDER]
    handles.append(plt.Line2D([0], [0], color="#555555", lw=1.0, label="baseline"))
    fig.suptitle(f"Darcy six-method generalization {metric_label(metric)} moving average", fontsize=16.0, y=0.992)
    fig.text(
        0.5,
        0.966,
        f"datasets {1 + (part - 1) * 25}-{part * 25} by {x_label}; centered window={window} epochs",
        ha="center",
        va="center",
        fontsize=9.8,
        color="#4f4b45",
    )
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.943), ncol=7, fontsize=8.4, handlelength=2.4, columnspacing=1.25)
    fig.text(0.5, 0.026, x_label, ha="center", va="center", fontsize=10.0, color="#37342f")
    fig.text(0.012, 0.5, metric_label(metric), ha="center", va="center", rotation="vertical", fontsize=10.0, color="#37342f")
    fig.subplots_adjust(top=0.905, bottom=0.058, left=0.055, right=0.992, hspace=0.54, wspace=0.25)
    fig.savefig(out, dpi=230, bbox_inches="tight", pad_inches=0.18, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=int, default=51)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    window = int(args.window)
    out_dir = args.out_dir or BUNDLE / f"figures_moving_average_ma{window}"
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_plot_style()

    split_df = pd.read_csv(DATA / "six_method_common_range_eval_split_summary.csv")
    metrics_df = pd.read_csv(DATA / "six_method_common_range_eval_metrics.csv")
    max_epoch, max_work = common_limits(split_df, metrics_df)
    ids = dataset_order(metrics_df)
    for metric in METRICS:
        split_s = add_smoothed_column(split_df, metric, ["method", "split"], window)
        metrics_s = add_smoothed_column(metrics_df, metric, ["method", "split", "dataset_id"], window)
        plot_split_mean(split_s, metric, "epoch", max_epoch, max_work, window, out_dir / f"ma{window}_epoch_{metric}_train_test_generalization.png")
        plot_split_mean(split_s, metric, "work", max_epoch, max_work, window, out_dir / f"ma{window}_work_clock_{metric}_train_test_generalization.png")
        for part, subset in enumerate((ids[:25], ids[25:50]), 1):
            plot_generalization_grid(metrics_s, metric, "epoch", subset, part, max_epoch, max_work, window, out_dir / f"ma{window}_{metric}_generalization_part{part:02d}_vs_epoch.png")
            plot_generalization_grid(metrics_s, metric, "work", subset, part, max_epoch, max_work, window, out_dir / f"ma{window}_{metric}_generalization_part{part:02d}_vs_work.png")

    config = {
        "source_split_csv": str((DATA / "six_method_common_range_eval_split_summary.csv").relative_to(ROOT)),
        "source_metrics_csv": str((DATA / "six_method_common_range_eval_metrics.csv").relative_to(ROOT)),
        "window_epochs": window,
        "max_epoch": max_epoch,
        "max_work_seconds": max_work,
        "figures": sorted(p.name for p in out_dir.glob("*.png")),
    }
    (out_dir / "moving_average_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(out_dir)


if __name__ == "__main__":
    main()
