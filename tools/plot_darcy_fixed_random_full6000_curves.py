#!/usr/bin/env python3
"""Plot Darcy fixed-vs-random curves with the x-axis extended to epoch 6000."""

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
MAX_EPOCH = 6000
METHODS = ("loss1", "loss2", "loss3")
SPLITS = ("train", "test", "generalization")
METRICS = {
    "rmse": ("rmse", "RMSE"),
    "relative_l2": ("relative_l2", "Relative L2"),
}
SPLIT_METRICS = {
    "rmse": ("rmse_dataset_mean", "RMSE"),
    "relative_l2": ("relative_l2_dataset_mean", "Relative L2"),
}
METHOD_COLORS = {"loss1": "#4C78A8", "loss2": "#F58518", "loss3": "#54A24B"}
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
}

NEW_RANDOM = {
    "loss1": PROJECT_ROOT / "outputs/darcy_sir20_timematched_full_serial_double_budget_full_delta_budget_records_20260617/data/training_runs/darcy_sir20_full_loss1_2986to5972ep_epsj0.25to1.75_workmatched/darcy",
    "loss2": PROJECT_ROOT / "outputs/darcy_sir20_timematched_full_serial_double_budget_full_delta_budget_records_20260617/data/training_runs/darcy_sir20_full_loss2_3320to6640ep_epsj0.25to1.75_workmatched/darcy",
    "loss3": PROJECT_ROOT / "outputs/darcy_sir20_timematched_full_serial_double_budget_full_delta_budget_records_20260617/data/training_runs/darcy_sir20_full_loss3_3000to6000ep_epsj0.25to1.75/darcy",
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
    dedup_cols = ["epoch", "split"] if "split" in df.columns and "dataset_id" not in df.columns else ["epoch", "dataset_id"]
    df = df.sort_values("epoch").drop_duplicates(dedup_cols, keep="last")
    df["method"] = method
    df["run"] = run
    return df


def add_delta_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for metric_key, (metric_col, _) in METRICS.items():
        base = (
            out[out["epoch"].eq(0)][["method", "run", "dataset_id", metric_col]]
            .rename(columns={metric_col: f"{metric_key}_epoch0"})
            .drop_duplicates(["method", "run", "dataset_id"])
        )
        out = out.merge(base, on=["method", "run", "dataset_id"], how="left")
        out[f"{metric_key}_delta_vs_epoch0"] = out[metric_col] - out[f"{metric_key}_epoch0"]
    return out


def load_generalization_data() -> pd.DataFrame:
    frames = []
    for method in METHODS:
        frames.append(load_method_frames(read_eval_metrics, method, "fixed"))
        frames.append(load_method_frames(read_eval_metrics, method, "random"))
    return add_delta_columns(pd.concat(frames, ignore_index=True))


def load_split_data() -> pd.DataFrame:
    frames = []
    for method in METHODS:
        frames.append(load_method_frames(read_split_summary, method, "fixed"))
        frames.append(load_method_frames(read_split_summary, method, "random"))
    return pd.concat(frames, ignore_index=True)


def fixed_end_epoch(gen_df: pd.DataFrame, method: str) -> int:
    sub = gen_df[(gen_df["method"].eq(method)) & (gen_df["run"].eq("fixed"))]
    return int(sub["epoch"].max()) if not sub.empty else 0


def plot_generalization_method(
    gen_df: pd.DataFrame,
    method: str,
    metric_key: str,
    dataset_ids: list[str],
    part: int,
    out: Path,
    *,
    delta: bool,
) -> None:
    metric_col, metric_label = METRICS[metric_key]
    y_col = f"{metric_key}_delta_vs_epoch0" if delta else metric_col
    selected = dataset_ids[(part - 1) * 25 : part * 25]
    fig, axes = setup_axes(len(selected))
    fend = fixed_end_epoch(gen_df, method)
    for ax, dataset_id in zip(axes, selected, strict=False):
        for run, style in (("fixed", "--"), ("random", "-")):
            sub = gen_df[
                gen_df["method"].eq(method)
                & gen_df["run"].eq(run)
                & gen_df["dataset_id"].eq(dataset_id)
                & (gen_df["epoch"] <= MAX_EPOCH)
            ].sort_values("epoch")
            if sub.empty:
                continue
            ax.plot(sub["epoch"], sub[y_col], color=RUN_COLORS[run], lw=0.95 if run == "fixed" else 1.08, ls=style, alpha=0.9, label=run)
        if fend:
            ax.axvline(fend, color="#777777", lw=0.55, ls=":", alpha=0.35)
        ax.set_xlim(0, MAX_EPOCH)
        if not delta:
            ax.set_yscale("log")
        else:
            ax.axhline(0.0, color="#333333", lw=0.65, alpha=0.5)
        ax.set_title(dataset_title(dataset_id), fontsize=7)
        ax.grid(alpha=0.2)
        ax.tick_params(labelsize=7)
    for ax in axes[len(selected) :]:
        ax.axis("off")
    handles = [
        plt.Line2D([0], [0], color=RUN_COLORS["fixed"], lw=1.3, ls="--", label=f"fixed eps=1, ends at {fend}"),
        plt.Line2D([0], [0], color=RUN_COLORS["random"], lw=1.5, label="random 0.25-1.75"),
    ]
    title_kind = "change vs own epoch 0" if delta else "absolute"
    better = "more negative is better" if delta else "lower is better"
    fig.suptitle(f"Darcy {method}: {metric_label} generalization {title_kind}, x-axis to 6000, part {part:02d}", y=0.992, fontsize=14)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.958), ncol=2, frameon=False, fontsize=9)
    fig.text(0.5, 0.932, f"fixed has no data after its last epoch; random continues to available epoch <= {MAX_EPOCH}; {better}", ha="center", fontsize=8, color="#555555")
    fig.text(0.012, 0.50, f"{metric_label}{' delta vs own epoch 0' if delta else ''}", va="center", rotation="vertical", fontsize=10)
    fig.text(0.50, 0.018, "epoch", ha="center", fontsize=10)
    fig.tight_layout(rect=(0.030, 0.040, 0.995, 0.905), h_pad=1.15, w_pad=0.75)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_loss1_loss2_generalization(gen_df: pd.DataFrame, metric_key: str, dataset_ids: list[str], part: int, out: Path) -> None:
    metric_col, metric_label = METRICS[metric_key]
    selected = dataset_ids[(part - 1) * 25 : part * 25]
    fig, axes = setup_axes(len(selected))
    for ax, dataset_id in zip(axes, selected, strict=False):
        for method in ("loss1", "loss2"):
            for run, style in (("fixed", "--"), ("random", "-")):
                sub = gen_df[
                    gen_df["method"].eq(method)
                    & gen_df["run"].eq(run)
                    & gen_df["dataset_id"].eq(dataset_id)
                    & (gen_df["epoch"] <= MAX_EPOCH)
                ].sort_values("epoch")
                if sub.empty:
                    continue
                ax.plot(sub["epoch"], sub[metric_col], color=METHOD_COLORS[method], lw=0.82 if run == "fixed" else 1.02, ls=style, alpha=0.82 if run == "fixed" else 0.92)
        ax.set_xlim(0, MAX_EPOCH)
        ax.set_yscale("log")
        ax.set_title(dataset_title(dataset_id), fontsize=7)
        ax.grid(alpha=0.2)
        ax.tick_params(labelsize=7)
    for ax in axes[len(selected) :]:
        ax.axis("off")
    handles = [
        plt.Line2D([0], [0], color=METHOD_COLORS["loss1"], lw=1.4, ls="--", label="loss1 fixed"),
        plt.Line2D([0], [0], color=METHOD_COLORS["loss1"], lw=1.6, label="loss1 random"),
        plt.Line2D([0], [0], color=METHOD_COLORS["loss2"], lw=1.4, ls="--", label="loss2 fixed"),
        plt.Line2D([0], [0], color=METHOD_COLORS["loss2"], lw=1.6, label="loss2 random"),
    ]
    fig.suptitle(f"Darcy loss1/loss2 fixed vs random {metric_label} generalization, x-axis to 6000, part {part:02d}", y=0.992, fontsize=14)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.958), ncol=4, frameon=False, fontsize=9)
    fig.text(0.5, 0.932, "fixed lines stop when fixed runs end; dashed=fixed, solid=random; lower is better", ha="center", fontsize=8, color="#555555")
    fig.text(0.012, 0.50, metric_label, va="center", rotation="vertical", fontsize=10)
    fig.text(0.50, 0.018, "epoch", ha="center", fontsize=10)
    fig.tight_layout(rect=(0.030, 0.040, 0.995, 0.905), h_pad=1.15, w_pad=0.75)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_split_method(split_df: pd.DataFrame, method: str, metric_key: str, out: Path) -> None:
    metric_col, metric_label = SPLIT_METRICS[metric_key]
    fend = int(split_df[(split_df["method"].eq(method)) & (split_df["run"].eq("fixed"))]["epoch"].max())
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), sharey=False)
    for ax, split in zip(axes, SPLITS, strict=True):
        for run, style in (("fixed", "--"), ("random", "-")):
            sub = split_df[
                split_df["method"].eq(method)
                & split_df["run"].eq(run)
                & split_df["split"].eq(split)
                & (split_df["epoch"] <= MAX_EPOCH)
            ].sort_values("epoch")
            if sub.empty:
                continue
            y = pd.to_numeric(sub[metric_col], errors="coerce")
            ax.plot(sub["epoch"], y, color=RUN_COLORS[run], lw=0.5, alpha=0.23, ls=style)
            ax.plot(sub["epoch"], smooth(y), color=RUN_COLORS[run], lw=1.7, ls=style, label=run)
        ax.axvline(fend, color="#777777", lw=0.6, ls=":", alpha=0.35)
        ax.set_xlim(0, MAX_EPOCH)
        ax.set_yscale("log")
        ax.set_title(split)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    axes[0].set_ylabel(metric_label)
    fig.suptitle(f"Darcy {method}: fixed vs random {metric_label} split means, x-axis to 6000", y=0.99, fontsize=14)
    fig.text(0.5, 0.925, f"fixed ends at {fend}; random continues to available epoch <= {MAX_EPOCH}; lower is better", ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0.02, 0.03, 1, 0.90))
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_split_grid(split_df: pd.DataFrame, metric_key: str, out: Path) -> None:
    metric_col, metric_label = SPLIT_METRICS[metric_key]
    fig, axes = plt.subplots(3, 3, figsize=(16, 11.5), sharey=False)
    for row, method in enumerate(METHODS):
        fend = int(split_df[(split_df["method"].eq(method)) & (split_df["run"].eq("fixed"))]["epoch"].max())
        for col, split in enumerate(SPLITS):
            ax = axes[row, col]
            for run, style in (("fixed", "--"), ("random", "-")):
                sub = split_df[
                    split_df["method"].eq(method)
                    & split_df["run"].eq(run)
                    & split_df["split"].eq(split)
                    & (split_df["epoch"] <= MAX_EPOCH)
                ].sort_values("epoch")
                if sub.empty:
                    continue
                y = pd.to_numeric(sub[metric_col], errors="coerce")
                ax.plot(sub["epoch"], y, color=RUN_COLORS[run], lw=0.5, alpha=0.22, ls=style)
                ax.plot(sub["epoch"], smooth(y), color=RUN_COLORS[run], lw=1.45, ls=style, label=run)
            ax.axvline(fend, color="#777777", lw=0.55, ls=":", alpha=0.35)
            ax.set_xlim(0, MAX_EPOCH)
            ax.set_yscale("log")
            ax.set_title(f"{method} / {split}")
            ax.grid(alpha=0.25)
            if row == 2:
                ax.set_xlabel("epoch")
            if col == 0:
                ax.set_ylabel(metric_label)
            if row == 0 and col == 2:
                ax.legend(frameon=False)
    fig.suptitle(f"Darcy fixed vs random {metric_label}: train/test/generalization split means, x-axis to 6000", y=0.99, fontsize=15)
    fig.text(0.5, 0.955, "dashed=fixed eps=1, solid=random 0.25-1.75; fixed stops near epoch 3000; lower is better", ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0.02, 0.025, 1, 0.94))
    fig.savefig(out, dpi=220)
    plt.close(fig)


def copy_pngs_to_download(src_dir: Path, dest_subdir: str) -> None:
    download_root = ANALYSIS_DIR / "darcy_fixed_random_generalization_figures_download_only_20260618" / dest_subdir
    download_root.mkdir(parents=True, exist_ok=True)
    for path in sorted(src_dir.glob("*.png")):
        shutil.copy2(path, download_root / path.name)


def main() -> None:
    gen_dir = ANALYSIS_DIR / "figures_fixed_random_generalization_panels_6000"
    split_dir = ANALYSIS_DIR / "figures_fixed_random_split_curves_6000"
    gen_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    gen_df = load_generalization_data()
    split_df = load_split_data()
    dataset_ids = sorted(gen_df["dataset_id"].dropna().unique().tolist(), key=dataset_index)
    outputs: list[Path] = []

    for metric_key in METRICS:
        for method in METHODS:
            for part in (1, 2):
                out = gen_dir / f"darcy_{method}_fixed_vs_random_{metric_key}_generalization_to6000_part{part:02d}.png"
                plot_generalization_method(gen_df, method, metric_key, dataset_ids, part, out, delta=False)
                outputs.append(out)
                out = gen_dir / f"darcy_{method}_fixed_vs_random_{metric_key}_delta_vs_epoch0_to6000_part{part:02d}.png"
                plot_generalization_method(gen_df, method, metric_key, dataset_ids, part, out, delta=True)
                outputs.append(out)
        for part in (1, 2):
            out = gen_dir / f"darcy_loss1_loss2_fixed_vs_random_{metric_key}_generalization_to6000_part{part:02d}.png"
            plot_loss1_loss2_generalization(gen_df, metric_key, dataset_ids, part, out)
            outputs.append(out)

    for metric_key in SPLIT_METRICS:
        out = split_dir / f"darcy_loss123_fixed_vs_random_{metric_key}_train_test_generalization_to6000_grid.png"
        plot_split_grid(split_df, metric_key, out)
        outputs.append(out)
        for method in METHODS:
            out = split_dir / f"darcy_{method}_fixed_vs_random_{metric_key}_train_test_generalization_to6000.png"
            plot_split_method(split_df, method, metric_key, out)
            outputs.append(out)

    copy_pngs_to_download(gen_dir, "figures_fixed_random_generalization_panels_6000")
    copy_pngs_to_download(split_dir, "figures_fixed_random_split_curves_6000")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
