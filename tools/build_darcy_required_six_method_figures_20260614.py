#!/usr/bin/env python3
"""Build only the required Darcy six-method figures over the common available range."""

from __future__ import annotations

import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "darcy_sir20_required_figures_only_20260614"
FIG = OUT / "figures"
DATA = OUT / "data"

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
TARGET_MAX_EPOCH = 3500

RUN_TRAIN_STEPS = {
    "loss1": [
        ROOT / "adversarial_training_runs/darcy_binary_loss3targeted_loss1_1000ep_full50_timematched_20260612_full50_timematched_1000c/darcy/train_steps.csv",
        ROOT / "adversarial_training_runs/darcy_binary_loss3targeted_loss1_continue2000ep_from_1000ep_full50_timematched_20260612_stage2_2000_from_1000c/darcy/train_steps.csv",
    ],
    "loss2": [
        ROOT / "adversarial_training_runs/darcy_binary_loss3targeted_loss2_1026ep_full50_timematched_20260612_full50_timematched_1000c/darcy/train_steps.csv",
        ROOT / "adversarial_training_runs/darcy_binary_loss3targeted_loss2_continue2053ep_from_1026ep_full50_timematched_20260612_stage2_2000_from_1000c/darcy/train_steps.csv",
    ],
    "loss3": [
        ROOT / "adversarial_training_runs/darcy_binary_loss3targeted_loss3_1011ep_full50_timematched_20260612_full50_timematched_1000c/darcy/train_steps.csv",
        ROOT / "adversarial_training_runs/darcy_binary_loss3targeted_loss3_continue2022ep_from_1011ep_full50_timematched_20260612_stage2_2000_from_1000c/darcy/train_steps.csv",
    ],
    "physics": [
        ROOT / "adversarial_training_runs/darcy_binary_loss3targeted_physics_1040ep_full50_timematched_20260612_full50_timematched_1000c/darcy/train_steps.csv",
        ROOT / "adversarial_training_runs/darcy_binary_loss3targeted_physics_continue2081ep_from_1040ep_full50_timematched_20260612_stage2_2000_from_1000c/darcy/train_steps.csv",
    ],
    "random_clean": [
        ROOT / "adversarial_training_runs/darcy_binary_random_binary_fixed_y_3000ep_full50_20260614_random_binary_source_3000_supervised/darcy/train_steps.csv",
        ROOT / "adversarial_training_runs/darcy_binary_random_binary_fixed_y_continue_to3500_from_3000_20260614_supervised/darcy/train_steps.csv",
    ],
    "random_solver": [
        ROOT / "adversarial_training_runs/darcy_binary_random_binary_solver_y_3000ep_full50_20260614_random_binary_source_3000_supervised/darcy/train_steps.csv",
        ROOT / "adversarial_training_runs/darcy_binary_random_binary_solver_y_continue_to3500_from_3000_20260614_supervised/darcy/train_steps.csv",
    ],
}

RANDOM_EVAL_SPLIT = {
    "random_clean": [
        ROOT / "adversarial_training_runs/darcy_binary_random_binary_fixed_y_3000ep_full50_20260614_random_binary_source_3000_supervised/darcy/eval_split_summary.csv",
        ROOT / "adversarial_training_runs/darcy_binary_random_binary_fixed_y_continue_to3500_from_3000_20260614_supervised/darcy/eval_split_summary.csv",
    ],
    "random_solver": [
        ROOT / "adversarial_training_runs/darcy_binary_random_binary_solver_y_3000ep_full50_20260614_random_binary_source_3000_supervised/darcy/eval_split_summary.csv",
        ROOT / "adversarial_training_runs/darcy_binary_random_binary_solver_y_continue_to3500_from_3000_20260614_supervised/darcy/eval_split_summary.csv",
    ],
}
RANDOM_EVAL_METRICS = {
    "random_clean": [
        ROOT / "adversarial_training_runs/darcy_binary_random_binary_fixed_y_3000ep_full50_20260614_random_binary_source_3000_supervised/darcy/eval_metrics.csv",
        ROOT / "adversarial_training_runs/darcy_binary_random_binary_fixed_y_continue_to3500_from_3000_20260614_supervised/darcy/eval_metrics.csv",
    ],
    "random_solver": [
        ROOT / "adversarial_training_runs/darcy_binary_random_binary_solver_y_3000ep_full50_20260614_random_binary_source_3000_supervised/darcy/eval_metrics.csv",
        ROOT / "adversarial_training_runs/darcy_binary_random_binary_solver_y_continue_to3500_from_3000_20260614_supervised/darcy/eval_metrics.csv",
    ],
}


def setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 220,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.facecolor": "white",
            "figure.facecolor": "#fbfaf7",
            "axes.edgecolor": "#aaa59b",
            "axes.grid": True,
            "grid.color": "#d8d4c8",
            "grid.alpha": 0.46,
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


def work_seconds_for_method(method: str) -> dict[int, float]:
    frames = []
    for path in RUN_TRAIN_STEPS[method]:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        frames.append(df[["epoch", "global_step", "step_wall_sec"]].copy())
    if not frames:
        raise FileNotFoundError(f"no train_steps files found for {method}")
    train = pd.concat(frames, ignore_index=True, sort=False)
    train = train.sort_values(["epoch", "global_step"]).drop_duplicates("epoch", keep="last")
    out = {0: 0.0}
    total = 0.0
    for row in train.sort_values("epoch").itertuples(index=False):
        total += float(row.step_wall_sec)
        out[int(row.epoch)] = total
    return out


def add_work_seconds(df: pd.DataFrame, method: str) -> pd.DataFrame:
    work = work_seconds_for_method(method)
    out = df.copy()
    out["work_seconds"] = out["epoch"].map(lambda e: work.get(int(e), np.nan))
    out["work_minutes"] = out["work_seconds"] / 60.0
    return out


def load_split_summary() -> pd.DataFrame:
    frames = []
    corrected = pd.read_csv(ROOT / "outputs/darcy_eval_artifact_corrected_20260614/data/eval_split_summary_artifact_corrected.csv")
    corrected = corrected[corrected["method"].isin(["loss1", "loss2", "loss3", "physics"])].copy()
    corrected["rmse_plot"] = corrected["rmse_corrected"]
    corrected["relative_l2_plot"] = corrected["relative_l2_corrected"]
    frames.append(corrected)
    for method, paths in RANDOM_EVAL_SPLIT.items():
        method_frames = []
        for path in paths:
            if path.exists():
                method_frames.append(pd.read_csv(path))
        if not method_frames:
            raise FileNotFoundError(f"no eval_split_summary files found for {method}")
        df = pd.concat(method_frames, ignore_index=True, sort=False).rename(
            columns={
                "rmse_dataset_mean": "rmse",
                "relative_l2_dataset_mean": "relative_l2",
                "mae_dataset_mean": "mae",
                "accuracy_score_dataset_mean": "accuracy_score",
            }
        )
        df = df.sort_values(["epoch", "phase", "split"]).drop_duplicates(["epoch", "phase", "split"], keep="last")
        df.insert(0, "method", method)
        df["rmse_plot"] = df["rmse"]
        df["relative_l2_plot"] = df["relative_l2"]
        frames.append(add_work_seconds(df, method))
    out = pd.concat(frames, ignore_index=True, sort=False)
    for method in ["loss1", "loss2", "loss3", "physics"]:
        idx = out["method"] == method
        out.loc[idx, ["work_seconds", "work_minutes"]] = add_work_seconds(out.loc[idx], method)[["work_seconds", "work_minutes"]].to_numpy()
    return out


def load_eval_metrics() -> pd.DataFrame:
    frames = []
    corrected = pd.read_csv(ROOT / "outputs/darcy_generalization50_artifact_corrected_20260614/data/eval_metrics_artifact_corrected.csv")
    corrected = corrected[corrected["method"].isin(["loss1", "loss2", "loss3", "physics"])].copy()
    corrected["rmse_plot"] = corrected["rmse_corrected"]
    corrected["relative_l2_plot"] = corrected["relative_l2_corrected"]
    frames.append(corrected)
    for method, paths in RANDOM_EVAL_METRICS.items():
        method_frames = []
        for path in paths:
            if path.exists():
                method_frames.append(pd.read_csv(path))
        if not method_frames:
            raise FileNotFoundError(f"no eval_metrics files found for {method}")
        df = pd.concat(method_frames, ignore_index=True, sort=False)
        df = df.sort_values(["epoch", "phase", "split", "dataset_id"]).drop_duplicates(["epoch", "phase", "split", "dataset_id"], keep="last")
        df.insert(0, "method", method)
        df["rmse_plot"] = df["rmse"]
        df["relative_l2_plot"] = df["relative_l2"]
        frames.append(add_work_seconds(df, method))
    out = pd.concat(frames, ignore_index=True, sort=False)
    for method in ["loss1", "loss2", "loss3", "physics"]:
        idx = out["method"] == method
        out.loc[idx, ["work_seconds", "work_minutes"]] = add_work_seconds(out.loc[idx], method)[["work_seconds", "work_minutes"]].to_numpy()
    return out


def common_limits(df: pd.DataFrame) -> tuple[int, float]:
    per = df[df["phase"].isin(PLOT_PHASES)].groupby("method").agg(max_epoch=("epoch", "max"), max_work=("work_seconds", "max"))
    return int(per["max_epoch"].min()), float(per["max_work"].min())


def metric_label(metric: str) -> str:
    return "RMSE" if metric == "rmse" else "Relative L2"


def baseline_split_values(split_df: pd.DataFrame, metric: str) -> dict[str, float]:
    base = split_df[(split_df["phase"] == "baseline_before_adversarial_training") & split_df["split"].isin(SPLITS)]
    return {str(r.split): float(getattr(r, f"{metric}_plot")) for r in base.drop_duplicates("split").itertuples(index=False)}


def baseline_dataset_values(metrics_df: pd.DataFrame, metric: str) -> dict[str, float]:
    base = metrics_df[(metrics_df["phase"] == "baseline_before_adversarial_training") & (metrics_df["split"] == "generalization")].copy()
    base["dataset_id"] = base["dataset_id"].astype(str)
    return base.groupby("dataset_id")[f"{metric}_plot"].mean().to_dict()


def plot_split_mean(df: pd.DataFrame, metric: str, x_axis: str, max_epoch: int, max_work: float, out: Path) -> None:
    x_col = "epoch" if x_axis == "epoch" else "work_seconds"
    x_label = "epoch" if x_axis == "epoch" else "work-clock seconds"
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
            ax.plot(sub[x_col], sub[f"{metric}_plot"], color=COLORS[method], lw=1.35, alpha=0.94, label=LABELS[method])
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
    handles = [plt.Line2D([0], [0], color=COLORS[m], lw=2.0, label=LABELS[m]) for m in METHOD_ORDER]
    handles.append(plt.Line2D([0], [0], color="#555555", lw=1.4, label="baseline"))
    fig.suptitle(f"Darcy six-method {metric_label(metric)}", fontsize=16.5, y=0.982)
    fig.text(0.5, 0.925, f"train / test / generalization mean by {x_label}", ha="center", va="center", fontsize=10.5, color="#4f4b45")
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.888), ncol=7, frameon=False, fontsize=9.0, handlelength=2.5, columnspacing=1.35)
    fig.subplots_adjust(top=0.79, bottom=0.145, left=0.055, right=0.988, wspace=0.27)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.16, facecolor=fig.get_facecolor())
    plt.close(fig)


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


def plot_generalization_grid(df: pd.DataFrame, metric: str, x_axis: str, ids: list[str], part: int, max_epoch: int, max_work: float, out: Path) -> None:
    x_col = "epoch" if x_axis == "epoch" else "work_seconds"
    x_label = "epoch" if x_axis == "epoch" else "work-clock seconds"
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
            ax.plot(sub[x_col], sub[f"{metric}_plot"], color=COLORS[method], linewidth=1.0, alpha=0.92)
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
    handles = [plt.Line2D([0], [0], color=COLORS[m], lw=1.4, label=LABELS[m]) for m in METHOD_ORDER]
    handles.append(plt.Line2D([0], [0], color="#555555", lw=1.0, label="baseline"))
    fig.suptitle(f"Darcy six-method generalization {metric_label(metric)}", fontsize=16.0, y=0.992)
    fig.text(0.5, 0.966, f"datasets {1 + (part - 1) * 25}-{part * 25} by {x_label}", ha="center", va="center", fontsize=9.8, color="#4f4b45")
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.943), ncol=7, frameon=False, fontsize=8.4, handlelength=2.4, columnspacing=1.25)
    fig.text(0.5, 0.026, x_label, ha="center", va="center", fontsize=10.0, color="#37342f")
    fig.text(0.012, 0.5, metric_label(metric), ha="center", va="center", rotation="vertical", fontsize=10.0, color="#37342f")
    fig.subplots_adjust(top=0.905, bottom=0.058, left=0.055, right=0.992, hspace=0.54, wspace=0.25)
    fig.savefig(out, dpi=230, bbox_inches="tight", pad_inches=0.18, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    setup_plot_style()
    FIG.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    split_df = load_split_summary()
    metrics_df = load_eval_metrics()
    split_max_epoch, split_max_work = common_limits(split_df)
    metrics_max_epoch, metrics_max_work = common_limits(metrics_df)
    max_epoch = TARGET_MAX_EPOCH
    max_work = min(split_max_work, metrics_max_work)
    split_df.to_csv(DATA / "six_method_common_range_eval_split_summary.csv", index=False)
    metrics_df.to_csv(DATA / "six_method_common_range_eval_metrics.csv", index=False)
    pd.DataFrame([{"common_max_epoch": max_epoch, "common_max_work_seconds": max_work}]).to_csv(DATA / "six_method_common_range_limits.csv", index=False)
    ids = dataset_order(metrics_df)
    for metric in METRICS:
        plot_split_mean(split_df, metric, "epoch", max_epoch, max_work, FIG / f"six_method_epoch_{metric}_train_test_generalization.png")
        plot_split_mean(split_df, metric, "work", max_epoch, max_work, FIG / f"six_method_work_clock_{metric}_train_test_generalization.png")
        for part, subset in enumerate((ids[:25], ids[25:50]), 1):
            plot_generalization_grid(metrics_df, metric, "epoch", subset, part, max_epoch, max_work, FIG / f"six_method_{metric}_generalization_part{part:02d}_vs_epoch.png")
            plot_generalization_grid(metrics_df, metric, "work", subset, part, max_epoch, max_work, FIG / f"six_method_{metric}_generalization_part{part:02d}_vs_work.png")
    print(FIG)


if __name__ == "__main__":
    main()
