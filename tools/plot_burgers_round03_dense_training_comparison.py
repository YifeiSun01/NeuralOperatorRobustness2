#!/usr/bin/env python3
"""Dense epoch/wall-clock comparison plots for Burgers round03 long runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "loss1": PROJECT_ROOT / "adversarial_training_runs/burgers_loss3_selective_round03_loss1_1000ep_long_20260605",
    "loss2": PROJECT_ROOT / "adversarial_training_runs/burgers_loss3_selective_round03_loss2_500ep_long_20260605",
    "loss3": PROJECT_ROOT / "adversarial_training_runs/burgers_loss3_selective_round03_loss3_500ep_long_20260605",
}
DEFAULT_OUT_DIR = PROJECT_ROOT / "visualizations/burgers_loss3_selective_round03_long_training_comparison_dense_20260605"

COLORS = {"loss1": "#1b6ca8", "loss2": "#d95f02", "loss3": "#2ca25f"}
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
            "grid.alpha": 0.46,
            "grid.linewidth": 0.55,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def metric_label(metric: str) -> str:
    return "Relative L2" if metric == "relative_l2" else "RMSE"


def run_burgers_dir(run_dir: Path) -> Path:
    return run_dir / "burgers"


def reconstruct_wall_seconds(run_dir: Path) -> dict[int, float]:
    task_dir = run_burgers_dir(run_dir)
    train = pd.read_csv(task_dir / "train_steps.csv")
    evals = pd.read_csv(task_dir / "evaluation_passes.csv")
    step_by_epoch = train.groupby("epoch")["step_wall_sec"].sum().to_dict()
    eval_by_epoch = evals.groupby("epoch")["eval_wall_sec"].sum().to_dict()
    max_epoch = int(max(step_by_epoch))
    wall: dict[int, float] = {0: 0.0}
    total = 0.0
    for epoch in range(1, max_epoch + 1):
        total += float(step_by_epoch.get(epoch, 0.0)) + float(eval_by_epoch.get(epoch, 0.0))
        wall[epoch] = total

    # Pin available checkpoint epochs to recorded wall_elapsed_seconds. Dense non-checkpoint
    # epochs remain reconstructed from the per-step logs; checkpoint deltas are only seconds.
    ckpt_path = task_dir / "checkpoints.csv"
    if ckpt_path.exists():
        ckpt = pd.read_csv(ckpt_path)
        for row in ckpt.itertuples(index=False):
            wall[int(row.epoch)] = float(row.wall_elapsed_seconds)
    return wall


def load_dense_frame() -> pd.DataFrame:
    frames = []
    for loss, run_dir in RUNS.items():
        task_dir = run_burgers_dir(run_dir)
        wall = reconstruct_wall_seconds(run_dir)
        df = pd.read_csv(task_dir / "eval_split_summary.csv")
        df = df.rename(
            columns={
                "rmse_dataset_mean": "rmse",
                "mae_dataset_mean": "mae",
                "relative_l2_dataset_mean": "relative_l2",
                "accuracy_score_dataset_mean": "accuracy_score",
            }
        )
        keep = [
            "epoch",
            "global_step",
            "progress_fraction",
            "split",
            "dataset_count",
            "total_samples_evaluated",
            "rmse",
            "mae",
            "relative_l2",
            "accuracy_score",
        ]
        df = df[keep].copy()
        df.insert(0, "loss", loss)
        df["wall_seconds"] = df["epoch"].map(lambda e: wall.get(int(e), np.nan))
        df["wall_hours"] = df["wall_seconds"] / 3600.0
        df["wall_source"] = "train_steps_plus_eval_passes_reconstructed;checkpoint_epochs_pinned"
        frames.append(df)
    dense = pd.concat(frames, ignore_index=True)
    dense = dense.sort_values(["loss", "epoch", "split"]).reset_index(drop=True)
    return dense


def downsample(df: pd.DataFrame, every: int) -> pd.DataFrame:
    if every <= 1:
        return df.copy()
    pieces = []
    for loss, g in df.groupby("loss"):
        max_epoch = int(g["epoch"].max())
        mask = (g["epoch"] == 0) | (g["epoch"] % every == 0) | (g["epoch"] == max_epoch)
        pieces.append(g[mask].copy())
    return pd.concat(pieces, ignore_index=True).sort_values(["loss", "epoch", "split"])


def savefig(fig: plt.Figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return str(path)


def plot_curves(df: pd.DataFrame, out_dir: Path, *, metric: str, x: str, splits: list[str], every: int) -> str:
    title_x = "same-epoch" if x == "epoch" else "wall-clock"
    suffix_splits = "_".join(splits)
    fig, axes = plt.subplots(1, len(splits), figsize=(7.2 * len(splits), 5.2), sharey=False)
    if len(splits) == 1:
        axes = [axes]
    for ax, split in zip(axes, splits):
        s = df[df["split"] == split].copy()
        for loss in ["loss1", "loss2", "loss3"]:
            g = s[s["loss"] == loss].sort_values(x)
            if g.empty:
                continue
            marker = MARKERS[loss] if every > 1 else None
            ms = 3.2 if every > 1 else 0
            lw = 1.15 if every == 1 else 1.55
            alpha = 0.84 if every == 1 else 0.9
            ax.plot(g[x], g[metric], color=COLORS[loss], marker=marker, ms=ms, lw=lw, alpha=alpha, label=loss)
        ax.set_title(split)
        ax.set_xlabel("epoch" if x == "epoch" else "wall-clock hours")
        ax.set_ylabel(metric_label(metric))
        ax.set_yscale("log")
        ax.legend()
    fig.suptitle(f"Round03 dense {title_x} {metric_label(metric)} every {every} epoch")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    name = f"round03_dense_{title_x.replace('-', '_')}_{metric}_every{every}_{suffix_splits}.png"
    return savefig(fig, out_dir / name)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dense = load_dense_frame()
    outputs: list[str] = []
    for every in [1, 5]:
        d = downsample(dense, every)
        d_path = args.out_dir / f"round03_dense_epoch_metrics_every{every}.csv"
        d.to_csv(d_path, index=False)
        outputs.append(str(d_path))
        for metric in ["rmse", "relative_l2"]:
            for splits in [["train", "test", "generalization"]]:
                outputs.append(plot_curves(d, args.out_dir, metric=metric, x="epoch", splits=splits, every=every))
                outputs.append(plot_curves(d, args.out_dir, metric=metric, x="wall_hours", splits=splits, every=every))
    manifest = args.out_dir / "round03_dense_training_comparison_plot_manifest.txt"
    manifest.write_text("# Dense outputs\n" + "\n".join(outputs) + "\n", encoding="utf-8")
    print("\n".join(outputs + [str(manifest)]), flush=True)


if __name__ == "__main__":
    main()
