#!/usr/bin/env python3
"""Plot Burgers round03 long-training loss1/loss2/loss3 comparison figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMPARISON_DIR = PROJECT_ROOT / "forensics/burgers_loss3_selective_round03_long_training_comparison_20260605"
DEFAULT_OUT_DIR = PROJECT_ROOT / "visualizations/burgers_loss3_selective_round03_long_training_comparison_20260605"

COLORS = {
    "loss1": "#1b6ca8",
    "loss2": "#d95f02",
    "loss3": "#2ca25f",
}
MARKERS = {"loss1": "o", "loss2": "s", "loss3": "^"}
BG = "#fbfaf7"
GRID = "#d8d4c8"
TEXT = "#202124"


def setup() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 230,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.facecolor": "white",
            "figure.facecolor": BG,
            "axes.edgecolor": "#aaa59b",
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "text.color": TEXT,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.alpha": 0.48,
            "grid.linewidth": 0.55,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def savefig(fig: plt.Figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return str(path)


def metric_label(metric: str) -> str:
    return "Relative L2" if metric == "relative_l2" else "RMSE"


def plot_epoch_curves(epoch_df: pd.DataFrame, out_dir: Path, metric: str) -> str:
    d = epoch_df[(epoch_df["comparison"] == "same_epoch") & (epoch_df["split"].isin(["test", "generalization"]))].copy()
    d = d[d["epoch"] >= 0]
    fig, axes = plt.subplots(1, 2, figsize=(14.8, 5.2), sharey=False)
    for ax, split in zip(axes, ["test", "generalization"]):
        s = d[d["split"] == split]
        for loss in ["loss1", "loss2", "loss3"]:
            g = s[s["loss"] == loss].sort_values("epoch")
            if g.empty:
                continue
            ax.plot(g["epoch"], g[metric], color=COLORS[loss], marker=MARKERS[loss], ms=3.8, lw=1.8, label=loss)
        ax.set_title(split)
        ax.set_xlabel("epoch")
        ax.set_ylabel(metric_label(metric))
        ax.set_yscale("log")
        ax.legend()
    fig.suptitle(f"Round03 same-epoch {metric_label(metric)} comparison")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return savefig(fig, out_dir / f"round03_same_epoch_{metric}.png")


def plot_wall_curves(wall_df: pd.DataFrame, out_dir: Path, metric: str) -> str:
    d = wall_df[wall_df["split"].isin(["test", "generalization"])].copy()
    d = d[d["target_reached_by_run"].astype(str).isin(["True", "true", "1"])]
    fig, axes = plt.subplots(1, 2, figsize=(14.8, 5.2), sharey=False)
    for ax, split in zip(axes, ["test", "generalization"]):
        s = d[d["split"] == split]
        for loss in ["loss1", "loss2", "loss3"]:
            g = s[s["loss"] == loss].sort_values("selected_checkpoint_wall_hours")
            if g.empty:
                continue
            ax.plot(g["selected_checkpoint_wall_hours"], g[metric], color=COLORS[loss], marker=MARKERS[loss], ms=3.8, lw=1.8, label=loss)
        ax.set_title(split)
        ax.set_xlabel("wall-clock hours")
        ax.set_ylabel(metric_label(metric))
        ax.set_yscale("log")
        ax.legend()
    fig.suptitle(f"Round03 wall-clock-aligned {metric_label(metric)} comparison")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return savefig(fig, out_dir / f"round03_wall_clock_{metric}.png")


def plot_final_bars(epoch_df: pd.DataFrame, out_dir: Path, metric: str) -> str:
    d = epoch_df[(epoch_df["comparison"] == "run_final") & (epoch_df["split"].isin(["train", "test", "generalization"]))].copy()
    splits = ["train", "test", "generalization"]
    losses = ["loss1", "loss2", "loss3"]
    x = np.arange(len(splits))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10.8, 5.6))
    for i, loss in enumerate(losses):
        vals = []
        for split in splits:
            g = d[(d["loss"] == loss) & (d["split"] == split)]
            vals.append(float(g[metric].iloc[0]) if not g.empty else np.nan)
        ax.bar(x + (i - 1) * width, vals, width=width, label=loss, color=COLORS[loss], alpha=0.88)
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_yscale("log")
    ax.set_ylabel(metric_label(metric))
    ax.set_title(f"Round03 final-checkpoint {metric_label(metric)}")
    ax.legend()
    fig.tight_layout()
    return savefig(fig, out_dir / f"round03_final_{metric}_bars.png")


def plot_advantage_distribution(per_dataset_df: pd.DataFrame, out_dir: Path) -> str:
    d = per_dataset_df[per_dataset_df["comparison"] == "run_final"].copy()
    fig, ax = plt.subplots(figsize=(10.8, 5.6))
    plotted = False
    for base_loss, color in [("loss1", COLORS["loss1"]), ("loss2", COLORS["loss2"] )]:
        col = f"loss3_rmse_advantage_pct_vs_{base_loss}"
        if col not in d:
            continue
        vals = pd.to_numeric(d[col], errors="coerce").dropna().to_numpy(dtype=float)
        if vals.size == 0:
            continue
        ax.hist(vals, bins=16, alpha=0.54, color=color, label=f"loss3 vs {base_loss}")
        plotted = True
    ax.axvline(0.0, color="#202124", lw=1.1, alpha=0.75)
    ax.set_xlabel("loss3 RMSE advantage percent on generated datasets")
    ax.set_ylabel("dataset count")
    ax.set_title("Round03 generated50 final per-dataset advantage distribution")
    if plotted:
        ax.legend()
    fig.tight_layout()
    return savefig(fig, out_dir / "round03_final_generated50_loss3_advantage_hist.png")


def write_manifest(out_dir: Path, outputs: list[str]) -> Path:
    manifest = out_dir / "round03_long_training_comparison_plot_manifest.txt"
    manifest.write_text("\n".join(outputs) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-dir", type=Path, default=DEFAULT_COMPARISON_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    epoch_df = pd.read_csv(args.comparison_dir / "epoch_aligned_split_metrics.csv")
    wall_df = pd.read_csv(args.comparison_dir / "wall_clock_aligned_split_metrics.csv")
    per_dataset_df = pd.read_csv(args.comparison_dir / "per_dataset_generated_advantage.csv")
    outputs: list[str] = []
    for metric in ["rmse", "relative_l2"]:
        outputs.append(plot_epoch_curves(epoch_df, args.out_dir, metric))
        outputs.append(plot_wall_curves(wall_df, args.out_dir, metric))
        outputs.append(plot_final_bars(epoch_df, args.out_dir, metric))
    outputs.append(plot_advantage_distribution(per_dataset_df, args.out_dir))
    manifest = write_manifest(args.out_dir, outputs)
    print("\n".join(outputs + [str(manifest)]), flush=True)


if __name__ == "__main__":
    main()
