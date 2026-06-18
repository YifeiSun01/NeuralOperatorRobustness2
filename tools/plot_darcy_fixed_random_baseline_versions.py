#!/usr/bin/env python3
"""Draw fixed/random overlay and separate Darcy plots with baseline lines."""

from __future__ import annotations

import math
import re
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = PROJECT_ROOT / "analysis_outputs" / "darcy_fixed_vs_random_eps_loss123_20260618"
OUT_DIR = ANALYSIS_DIR / "figures_fixed_random_with_baselines_versions"
DOWNLOAD_DIR = ANALYSIS_DIR / "darcy_fixed_random_generalization_figures_download_only_20260618"
MAX_EPOCH = 6000
METHODS = ("loss1", "loss2", "loss3", "physics")
SPLITS = ("train", "test", "generalization")
METRICS = {"rmse": "RMSE", "relative_l2": "Relative L2"}
SPLIT_METRICS = {"rmse": "rmse_dataset_mean", "relative_l2": "relative_l2_dataset_mean"}
RUN_COLORS = {"fixed": "#4C78A8", "random": "#F58518"}

OLD_FIXED = {
    "loss1": [
        PROJECT_ROOT / "adversarial_training_runs/darcy_binary_loss3targeted_loss1_1000ep_full50_timematched_20260612_full50_timematched_1000c/darcy",
        PROJECT_ROOT / "adversarial_training_runs/darcy_binary_loss3targeted_loss1_continue2000ep_from_1000ep_full50_timematched_20260612_stage2_2000_from_1000c/darcy",
    ],
    "loss2": [
        PROJECT_ROOT / "adversarial_training_runs/darcy_binary_loss3targeted_loss2_1026ep_full50_timematched_20260612_full50_timematched_1000c/darcy",
        PROJECT_ROOT / "adversarial_training_runs/darcy_binary_loss3targeted_loss2_continue2053ep_from_1026ep_full50_timematched_20260612_stage2_2000_from_1000c/darcy",
    ],
    "loss3": [
        PROJECT_ROOT / "adversarial_training_runs/darcy_binary_loss3targeted_loss3_1011ep_full50_timematched_20260612_full50_timematched_1000c/darcy",
        PROJECT_ROOT / "adversarial_training_runs/darcy_binary_loss3targeted_loss3_continue2022ep_from_1011ep_full50_timematched_20260612_stage2_2000_from_1000c/darcy",
    ],
    "physics": [
        PROJECT_ROOT / "adversarial_training_runs/darcy_binary_loss3targeted_physics_1040ep_full50_timematched_20260612_full50_timematched_1000c/darcy",
        PROJECT_ROOT / "adversarial_training_runs/darcy_binary_loss3targeted_physics_continue2081ep_from_1040ep_full50_timematched_20260612_stage2_2000_from_1000c/darcy",
    ],
}

NEW_RANDOM = {
    "loss1": PROJECT_ROOT / "outputs/darcy_sir20_timematched_full_serial_double_budget_full_delta_budget_records_20260617/data/training_runs/darcy_sir20_full_loss1_2986to5972ep_epsj0.25to1.75_workmatched/darcy",
    "loss2": PROJECT_ROOT / "outputs/darcy_sir20_timematched_full_serial_double_budget_full_delta_budget_records_20260617/data/training_runs/darcy_sir20_full_loss2_3320to6640ep_epsj0.25to1.75_workmatched/darcy",
    "loss3": PROJECT_ROOT / "outputs/darcy_sir20_timematched_full_serial_double_budget_full_delta_budget_records_20260617/data/training_runs/darcy_sir20_full_loss3_3000to6000ep_epsj0.25to1.75/darcy",
    "physics": PROJECT_ROOT / "outputs/darcy_sir20_timematched_full_serial_double_budget_full_delta_budget_records_20260617/data/training_runs/darcy_sir20_full_physics_3860to7720ep_epsj0.25to1.75_workmatched/darcy",
}


def dataset_index(dataset_id: str) -> int:
    match = re.search(r"_(\d{2})_", str(dataset_id))
    return int(match.group(1)) if match else 9999


def dataset_title(dataset_id: str) -> str:
    text = re.sub(r"^darcy_binary_loss3targeted_20260611_", "", str(dataset_id)).replace("_", " ")
    return text[:54] + ("..." if len(text) > 54 else "")


def setup_axes(n: int) -> tuple[plt.Figure, list[plt.Axes]]:
    rows = math.ceil(n / 5)
    fig, axes = plt.subplots(rows, 5, figsize=(19, 3.0 * rows), squeeze=False)
    return fig, axes.ravel().tolist()


def smooth(values: pd.Series, window: int = 41) -> pd.Series:
    return values.rolling(window=window, center=True, min_periods=max(5, window // 5)).median()


def read_eval_metrics(root: Path) -> pd.DataFrame:
    cols = ["epoch", "dataset_id", "split", "rmse", "relative_l2"]
    df = pd.read_csv(root / "eval_metrics.csv", usecols=cols)
    df = df[df["split"].eq("generalization")].copy()
    for col in ("epoch", "rmse", "relative_l2"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["epoch", "dataset_id"]).copy()
    df["epoch"] = df["epoch"].astype(int)
    return df[df["epoch"] <= MAX_EPOCH]


def read_split_summary(root: Path) -> pd.DataFrame:
    cols = ["epoch", "split", "rmse_dataset_mean", "relative_l2_dataset_mean"]
    df = pd.read_csv(root / "eval_split_summary.csv", usecols=cols)
    df = df[df["split"].isin(SPLITS)].copy()
    for col in ("epoch", "rmse_dataset_mean", "relative_l2_dataset_mean"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["epoch"]).copy()
    df["epoch"] = df["epoch"].astype(int)
    return df[df["epoch"] <= MAX_EPOCH]


def load_method_frames(reader, method: str, run: str) -> pd.DataFrame:
    roots = OLD_FIXED[method] if run == "fixed" else [NEW_RANDOM[method]]
    frames = [reader(root) for root in roots]
    df = pd.concat(frames, ignore_index=True)
    dedup_cols = ["epoch", "dataset_id"] if "dataset_id" in df.columns else ["epoch", "split"]
    df = df.sort_values("epoch").drop_duplicates(dedup_cols, keep="last")
    df["method"] = method
    df["run"] = run
    return df


def load_generalization_data() -> pd.DataFrame:
    frames = []
    for method in METHODS:
        frames.append(load_method_frames(read_eval_metrics, method, "fixed"))
        frames.append(load_method_frames(read_eval_metrics, method, "random"))
    return pd.concat(frames, ignore_index=True)


def load_split_data() -> pd.DataFrame:
    frames = []
    for method in METHODS:
        frames.append(load_method_frames(read_split_summary, method, "fixed"))
        frames.append(load_method_frames(read_split_summary, method, "random"))
    return pd.concat(frames, ignore_index=True)


def plot_generalization_version(
    gen_df: pd.DataFrame,
    method: str,
    metric: str,
    dataset_ids: list[str],
    part: int,
    version: str,
    out: Path,
) -> None:
    selected = dataset_ids[(part - 1) * 25 : part * 25]
    fig, axes = setup_axes(len(selected))
    runs = ("fixed", "random") if version == "overlay" else (version.removesuffix("_only"),)
    max_fixed = int(gen_df[(gen_df["method"].eq(method)) & (gen_df["run"].eq("fixed"))]["epoch"].max())
    for ax, dataset_id in zip(axes, selected, strict=False):
        for run in runs:
            sub = gen_df[
                gen_df["method"].eq(method)
                & gen_df["run"].eq(run)
                & gen_df["dataset_id"].eq(dataset_id)
            ].sort_values("epoch")
            if sub.empty:
                continue
            baseline = sub[sub["epoch"].eq(0)]
            if not baseline.empty:
                ax.axhline(float(baseline.iloc[0][metric]), color=RUN_COLORS[run], lw=0.75, ls=":", alpha=0.65)
            style = "--" if run == "fixed" else "-"
            ax.plot(sub["epoch"], sub[metric], color=RUN_COLORS[run], lw=0.95 if run == "fixed" else 1.08, ls=style, alpha=0.9, label=run)
        if version == "overlay":
            ax.axvline(max_fixed, color="#777777", lw=0.55, ls=":", alpha=0.35)
        ax.set_xlim(0, MAX_EPOCH)
        ax.set_yscale("log")
        ax.set_title(dataset_title(dataset_id), fontsize=7)
        ax.grid(alpha=0.2)
        ax.tick_params(labelsize=7)
    for ax in axes[len(selected) :]:
        ax.axis("off")
    handles = []
    for run in runs:
        handles.append(plt.Line2D([0], [0], color=RUN_COLORS[run], lw=1.4, ls="--" if run == "fixed" else "-", label=run))
        handles.append(plt.Line2D([0], [0], color=RUN_COLORS[run], lw=1.0, ls=":", label=f"{run} baseline"))
    metric_label = METRICS[metric]
    title_version = {"overlay": "fixed+random overlay", "fixed_only": "fixed only", "random_only": "random only"}[version]
    fig.suptitle(f"Darcy {method}: {metric_label} generalization {title_version}, x-axis to 6000, part {part:02d}", y=0.992, fontsize=14)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.958), ncol=min(4, len(handles)), frameon=False, fontsize=8)
    fig.text(0.5, 0.932, "dotted horizontal line = that run's epoch-0 baseline; lower is better", ha="center", fontsize=8, color="#555555")
    fig.text(0.012, 0.50, metric_label, va="center", rotation="vertical", fontsize=10)
    fig.text(0.50, 0.018, "epoch", ha="center", fontsize=10)
    fig.tight_layout(rect=(0.030, 0.040, 0.995, 0.905), h_pad=1.15, w_pad=0.75)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_split_version(split_df: pd.DataFrame, method: str, metric: str, version: str, out: Path) -> None:
    metric_col = SPLIT_METRICS[metric]
    metric_label = METRICS[metric]
    runs = ("fixed", "random") if version == "overlay" else (version.removesuffix("_only"),)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), sharey=False)
    max_fixed = int(split_df[(split_df["method"].eq(method)) & (split_df["run"].eq("fixed"))]["epoch"].max())
    for ax, split in zip(axes, SPLITS, strict=True):
        for run in runs:
            sub = split_df[
                split_df["method"].eq(method)
                & split_df["run"].eq(run)
                & split_df["split"].eq(split)
            ].sort_values("epoch")
            if sub.empty:
                continue
            y = pd.to_numeric(sub[metric_col], errors="coerce")
            base = sub[sub["epoch"].eq(0)]
            if not base.empty:
                ax.axhline(float(base.iloc[0][metric_col]), color=RUN_COLORS[run], lw=0.8, ls=":", alpha=0.65)
            style = "--" if run == "fixed" else "-"
            ax.plot(sub["epoch"], y, color=RUN_COLORS[run], lw=0.5, alpha=0.24, ls=style)
            ax.plot(sub["epoch"], smooth(y), color=RUN_COLORS[run], lw=1.7, ls=style, label=run)
        if version == "overlay":
            ax.axvline(max_fixed, color="#777777", lw=0.6, ls=":", alpha=0.35)
        ax.set_xlim(0, MAX_EPOCH)
        ax.set_yscale("log")
        ax.set_title(split)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    axes[0].set_ylabel(metric_label)
    title_version = {"overlay": "fixed+random overlay", "fixed_only": "fixed only", "random_only": "random only"}[version]
    fig.suptitle(f"Darcy {method}: {metric_label} split means {title_version}, x-axis to 6000", y=0.99, fontsize=14)
    fig.text(0.5, 0.925, "dotted horizontal line = that run's epoch-0 baseline; lower is better", ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0.02, 0.03, 1, 0.90))
    fig.savefig(out, dpi=220)
    plt.close(fig)


def main() -> None:
    gen_df = load_generalization_data()
    split_df = load_split_data()
    dataset_ids = sorted(gen_df["dataset_id"].dropna().unique().tolist(), key=dataset_index)
    outputs: list[Path] = []
    for version in ("overlay", "fixed_only", "random_only"):
        for metric in METRICS:
            for method in METHODS:
                for part in (1, 2):
                    out_dir = OUT_DIR / "generalization" / version
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out = out_dir / f"darcy_{method}_{metric}_{version}_baseline_to6000_part{part:02d}.png"
                    plot_generalization_version(gen_df, method, metric, dataset_ids, part, version, out)
                    outputs.append(out)
                out_dir = OUT_DIR / "split_curves" / version
                out_dir.mkdir(parents=True, exist_ok=True)
                out = out_dir / f"darcy_{method}_{metric}_train_test_generalization_{version}_baseline_to6000.png"
                plot_split_version(split_df, method, metric, version, out)
                outputs.append(out)

    download_subdir = DOWNLOAD_DIR / "fixed_random_with_baselines_versions"
    download_subdir.mkdir(parents=True, exist_ok=True)
    for path in outputs:
        rel = path.relative_to(OUT_DIR)
        dest = download_subdir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        print(path)


if __name__ == "__main__":
    main()
