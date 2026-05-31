#!/usr/bin/env python3
"""Plot evaluation metrics and training/attack losses for an adversarial training run."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--task", default="burgers")
    p.add_argument("--out-path", type=Path, required=True)
    p.add_argument("--title", default=None)
    p.add_argument("--dpi", type=int, default=180)
    return p.parse_args()


def mean_eval(eval_df: pd.DataFrame, split_filter: str | None, metric: str) -> pd.Series:
    df = eval_df.copy()
    if split_filter == "generalization":
        df = df[df["split"] == "generalization"]
    elif split_filter == "original":
        df = df[df["split"].isin(["train", "test"])]
    return df.groupby("epoch")[metric].mean().sort_index()


def smooth(s: pd.Series, window: int = 15) -> pd.Series:
    if len(s) < 3:
        return s
    return s.rolling(window=min(window, max(3, len(s)//10)), min_periods=1, center=True).mean()


def main() -> None:
    args = parse_args()
    task_dir = args.run_dir / args.task
    eval_df = pd.read_csv(task_dir / "eval_metrics.csv")
    train_df = pd.read_csv(task_dir / "train_steps.csv")

    fig, axes = plt.subplots(3, 1, figsize=(11.5, 12.0), sharex=False)

    ax = axes[0]
    for metric, style in [("relative_l2", "-"), ("rmse", "--")]:
        all_s = mean_eval(eval_df, None, metric)
        gen_s = mean_eval(eval_df, "generalization", metric)
        orig_s = mean_eval(eval_df, "original", metric)
        label_metric = "Relative L2" if metric == "relative_l2" else "RMSE"
        ax.plot(all_s.index, all_s.values, style, marker="o", linewidth=2.2, label=f"{label_metric}: all eval datasets")
        ax.plot(gen_s.index, gen_s.values, style, marker="s", linewidth=1.9, alpha=0.82, label=f"{label_metric}: generalization only")
        ax.plot(orig_s.index, orig_s.values, style, marker="^", linewidth=1.7, alpha=0.78, label=f"{label_metric}: original train+test")
    ax.set_title("Evaluation loss, saved every evaluation epoch")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.32)
    ax.legend(ncol=2, fontsize=8)

    ax = axes[1]
    for col, label in [
        ("train_loss_on_adv", "Optimizer update MSE on training batch"),
        ("clean_loss_before_attack", "Clean training batch MSE before attack"),
        ("adv_loss_after_attack", "Adversarial training batch MSE after attack"),
    ]:
        if col in train_df:
            s = train_df.set_index("epoch")[col]
            ax.plot(s.index, smooth(s).values, linewidth=1.9, label=label)
    ax.set_title("Smoothed training-batch attack diagnostics, not evaluation")
    ax.set_ylabel("MSE loss")
    ax.grid(True, alpha=0.32)
    ax.legend(fontsize=8)

    ax = axes[2]
    if {"clean_loss_before_attack", "adv_loss_after_attack"}.issubset(train_df.columns):
        gap = train_df["adv_loss_after_attack"] - train_df["clean_loss_before_attack"]
        ax.plot(train_df["epoch"], smooth(gap), color="#7a3db8", linewidth=2.0, label="Attack gain = adversarial MSE - clean MSE")
    if "train_loss_on_adv" in train_df:
        s = train_df.set_index("epoch")["train_loss_on_adv"]
        ax.plot(s.index, smooth(s), color="#2f6f9f", linewidth=1.8, alpha=0.72, label="Optimizer update MSE on training batch")
    ax.set_title("Attack gain and optimizer update loss")
    ax.set_xlabel("evaluation/training epoch")
    ax.set_ylabel("MSE loss")
    ax.grid(True, alpha=0.32)
    ax.legend(fontsize=8)

    title = args.title or f"{args.task}: evaluation and train loss progress"
    fig.suptitle(title, fontsize=14, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] wrote {args.out_path}")


if __name__ == "__main__":
    main()
