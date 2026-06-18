#!/usr/bin/env python3
"""Plot fixed-vs-random Darcy train/test/generalization split curves."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = PROJECT_ROOT / "analysis_outputs" / "darcy_fixed_vs_random_eps_loss123_20260618"
OUT_DIR = ANALYSIS_DIR / "figures_fixed_random_split_curves"

METHODS = ("loss1", "loss2", "loss3")
SPLITS = ("train", "test", "generalization")
METRICS = {
    "rmse": ("rmse_dataset_mean", "RMSE"),
    "relative_l2": ("relative_l2_dataset_mean", "Relative L2"),
}
COMMON_END = {"loss1": 3000, "loss2": 3070, "loss3": 3030}
COLORS = {"fixed": "#4C78A8", "random": "#F58518"}

OLD_FIXED = {
    "loss1": [
        PROJECT_ROOT / "adversarial_training_runs/darcy_binary_loss3targeted_loss1_1000ep_full50_timematched_20260612_full50_timematched_1000c/darcy/eval_split_summary.csv",
        PROJECT_ROOT / "adversarial_training_runs/darcy_binary_loss3targeted_loss1_continue2000ep_from_1000ep_full50_timematched_20260612_stage2_2000_from_1000c/darcy/eval_split_summary.csv",
    ],
    "loss2": [
        PROJECT_ROOT / "adversarial_training_runs/darcy_binary_loss3targeted_loss2_1026ep_full50_timematched_20260612_full50_timematched_1000c/darcy/eval_split_summary.csv",
        PROJECT_ROOT / "adversarial_training_runs/darcy_binary_loss3targeted_loss2_continue2053ep_from_1026ep_full50_timematched_20260612_stage2_2000_from_1000c/darcy/eval_split_summary.csv",
    ],
    "loss3": [
        PROJECT_ROOT / "adversarial_training_runs/darcy_binary_loss3targeted_loss3_1011ep_full50_timematched_20260612_full50_timematched_1000c/darcy/eval_split_summary.csv",
        PROJECT_ROOT / "adversarial_training_runs/darcy_binary_loss3targeted_loss3_continue2022ep_from_1011ep_full50_timematched_20260612_stage2_2000_from_1000c/darcy/eval_split_summary.csv",
    ],
}

NEW_RANDOM = {
    "loss1": PROJECT_ROOT / "outputs/darcy_sir20_timematched_full_serial_double_budget_full_delta_budget_records_20260617/data/training_runs/darcy_sir20_full_loss1_2986to5972ep_epsj0.25to1.75_workmatched/darcy/eval_split_summary.csv",
    "loss2": PROJECT_ROOT / "outputs/darcy_sir20_timematched_full_serial_double_budget_full_delta_budget_records_20260617/data/training_runs/darcy_sir20_full_loss2_3320to6640ep_epsj0.25to1.75_workmatched/darcy/eval_split_summary.csv",
    "loss3": PROJECT_ROOT / "outputs/darcy_sir20_timematched_full_serial_double_budget_full_delta_budget_records_20260617/data/training_runs/darcy_sir20_full_loss3_3000to6000ep_epsj0.25to1.75/darcy/eval_split_summary.csv",
}


def load_eval(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df = df[df["split"].isin(SPLITS)].copy()
    for col in ("epoch", "rmse_dataset_mean", "relative_l2_dataset_mean"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["epoch"]).copy()
    df["epoch"] = df["epoch"].astype(int)
    return df


def load_fixed(method: str) -> pd.DataFrame:
    frames = [load_eval(path) for path in OLD_FIXED[method]]
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("epoch").drop_duplicates(["epoch", "split"], keep="last")
    df["run"] = "fixed"
    df["method"] = method
    return df


def load_random(method: str) -> pd.DataFrame:
    df = load_eval(NEW_RANDOM[method])
    df = df.sort_values("epoch").drop_duplicates(["epoch", "split"], keep="last")
    df["run"] = "random"
    df["method"] = method
    return df


def smooth(values: pd.Series, window: int = 41) -> pd.Series:
    return values.rolling(window=window, center=True, min_periods=max(5, window // 5)).median()


def plot_method_metric(data: pd.DataFrame, method: str, metric_key: str, out: Path) -> None:
    metric_col, metric_label = METRICS[metric_key]
    end = COMMON_END[method]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), sharey=False)
    for ax, split in zip(axes, SPLITS, strict=True):
        for run, style in (("fixed", "--"), ("random", "-")):
            sub = data[(data["method"].eq(method)) & (data["run"].eq(run)) & (data["split"].eq(split)) & (data["epoch"] <= end)].sort_values("epoch")
            if sub.empty:
                continue
            y = pd.to_numeric(sub[metric_col], errors="coerce")
            ax.plot(sub["epoch"], y, color=COLORS[run], lw=0.55, alpha=0.25, ls=style)
            ax.plot(sub["epoch"], smooth(y), color=COLORS[run], lw=1.8, ls=style, label=f"{run}")
        ax.set_title(split)
        ax.set_xlabel("epoch")
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    axes[0].set_ylabel(metric_label)
    fig.suptitle(
        f"Darcy {method}: fixed eps=1 vs random 0.25-1.75 {metric_label} split means",
        y=0.99,
        fontsize=14,
    )
    fig.text(0.5, 0.925, f"train/test/generalization split means, same-epoch comparison, epoch <= {end}; lower is better", ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0.02, 0.03, 1, 0.90))
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_metric_grid(data: pd.DataFrame, metric_key: str, out: Path) -> None:
    metric_col, metric_label = METRICS[metric_key]
    fig, axes = plt.subplots(3, 3, figsize=(16, 11.5), sharey=False)
    for row, method in enumerate(METHODS):
        end = COMMON_END[method]
        for col, split in enumerate(SPLITS):
            ax = axes[row, col]
            for run, style in (("fixed", "--"), ("random", "-")):
                sub = data[(data["method"].eq(method)) & (data["run"].eq(run)) & (data["split"].eq(split)) & (data["epoch"] <= end)].sort_values("epoch")
                if sub.empty:
                    continue
                y = pd.to_numeric(sub[metric_col], errors="coerce")
                ax.plot(sub["epoch"], y, color=COLORS[run], lw=0.5, alpha=0.22, ls=style)
                ax.plot(sub["epoch"], smooth(y), color=COLORS[run], lw=1.5, ls=style, label=run)
            ax.set_title(f"{method} / {split}")
            ax.set_yscale("log")
            ax.grid(alpha=0.25)
            if row == 2:
                ax.set_xlabel("epoch")
            if col == 0:
                ax.set_ylabel(metric_label)
            if row == 0 and col == 2:
                ax.legend(frameon=False)
    fig.suptitle(f"Darcy fixed vs random {metric_label}: train/test/generalization split means", y=0.99, fontsize=15)
    fig.text(0.5, 0.955, "dashed=fixed eps=1, solid=random 0.25-1.75; faint=raw, bold=rolling median; lower is better", ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0.02, 0.025, 1, 0.94))
    fig.savefig(out, dpi=220)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.concat([load_fixed(m) for m in METHODS] + [load_random(m) for m in METHODS], ignore_index=True)
    outputs: list[Path] = []
    for metric_key in METRICS:
        out = OUT_DIR / f"darcy_loss123_fixed_vs_random_{metric_key}_train_test_generalization_grid.png"
        plot_metric_grid(data, metric_key, out)
        outputs.append(out)
        for method in METHODS:
            out = OUT_DIR / f"darcy_{method}_fixed_vs_random_{metric_key}_train_test_generalization.png"
            plot_method_metric(data, method, metric_key, out)
            outputs.append(out)
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
