#!/usr/bin/env python3
"""Draw clear Darcy fixed-vs-random generalization panels."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANALYSIS = PROJECT_ROOT / "analysis_outputs" / "darcy_fixed_vs_random_eps_loss123_20260618"

METHOD_COMMON_END = {"loss1": 3000, "loss2": 3070, "loss3": 3030}
METHOD_COLORS = {"loss1": "#4C78A8", "loss2": "#F58518", "loss3": "#54A24B"}


def dataset_index(dataset_id: str) -> int:
    match = re.search(r"_(\d{2})_", dataset_id)
    if match:
        return int(match.group(1))
    return 9999


def dataset_title(dataset_id: str) -> str:
    text = re.sub(r"^darcy_binary_loss3targeted_20260611_", "", dataset_id)
    text = text.replace("_", " ")
    return text[:54] + ("..." if len(text) > 54 else "")


def load_pairs(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = [
        "method",
        "epoch",
        "dataset_id",
        "old_fixed_relative_l2",
        "new_random_relative_l2",
        "old_delta_vs_own_epoch0",
        "new_delta_vs_own_epoch0",
    ]
    missing = [col for col in needed if col not in df.columns]
    if missing:
        raise ValueError(f"missing columns in {path}: {missing}")
    for col in needed:
        if col not in {"method", "dataset_id"}:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["epoch", "old_fixed_relative_l2", "new_random_relative_l2"]).copy()


def ordered_datasets(df: pd.DataFrame) -> list[str]:
    ids = sorted(df["dataset_id"].dropna().unique().tolist(), key=dataset_index)
    return ids


def setup_axes(n: int) -> tuple[plt.Figure, list[plt.Axes]]:
    rows = math.ceil(n / 5)
    fig, axes = plt.subplots(rows, 5, figsize=(19, 3.0 * rows), squeeze=False)
    return fig, axes.ravel().tolist()


def plot_method_absolute(df: pd.DataFrame, method: str, dataset_ids: list[str], part: int, out: Path) -> None:
    common_end = METHOD_COMMON_END[method]
    selected = dataset_ids[(part - 1) * 25 : part * 25]
    fig, axes = setup_axes(len(selected))
    for ax, dataset_id in zip(axes, selected, strict=False):
        sub = df[(df["method"].eq(method)) & (df["dataset_id"].eq(dataset_id)) & (df["epoch"] <= common_end)].sort_values("epoch")
        if sub.empty:
            ax.axis("off")
            continue
        ax.plot(sub["epoch"], sub["old_fixed_relative_l2"], color="#4C78A8", lw=0.95, ls="--", alpha=0.88, label="fixed eps=1")
        ax.plot(sub["epoch"], sub["new_random_relative_l2"], color="#F58518", lw=1.1, alpha=0.9, label="random 0.25-1.75")
        ax.set_yscale("log")
        ax.set_title(dataset_title(dataset_id), fontsize=7)
        ax.grid(alpha=0.2)
        ax.tick_params(labelsize=7)
    for ax in axes[len(selected) :]:
        ax.axis("off")
    handles = [
        plt.Line2D([0], [0], color="#4C78A8", lw=1.3, ls="--", label="fixed eps=1"),
        plt.Line2D([0], [0], color="#F58518", lw=1.5, label="random 0.25-1.75"),
    ]
    fig.suptitle(
        f"Darcy {method}: fixed vs random budget generalization Relative L2, part {part:02d}",
        y=0.992,
        fontsize=14,
    )
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.958), ncol=2, frameon=False, fontsize=9)
    fig.text(0.5, 0.932, f"same-epoch comparison only, epoch <= {common_end}; lower is better", ha="center", fontsize=8, color="#555555")
    fig.text(0.012, 0.50, "Relative L2", va="center", rotation="vertical", fontsize=10)
    fig.text(0.50, 0.018, "epoch", ha="center", fontsize=10)
    fig.tight_layout(rect=(0.028, 0.040, 0.995, 0.905), h_pad=1.15, w_pad=0.75)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_method_delta(df: pd.DataFrame, method: str, dataset_ids: list[str], part: int, out: Path) -> None:
    common_end = METHOD_COMMON_END[method]
    selected = dataset_ids[(part - 1) * 25 : part * 25]
    fig, axes = setup_axes(len(selected))
    for ax, dataset_id in zip(axes, selected, strict=False):
        sub = df[(df["method"].eq(method)) & (df["dataset_id"].eq(dataset_id)) & (df["epoch"] <= common_end)].sort_values("epoch")
        if sub.empty:
            ax.axis("off")
            continue
        ax.axhline(0.0, color="#333333", lw=0.75, alpha=0.55)
        ax.plot(sub["epoch"], sub["old_delta_vs_own_epoch0"], color="#4C78A8", lw=0.95, ls="--", alpha=0.88, label="fixed eps=1")
        ax.plot(sub["epoch"], sub["new_delta_vs_own_epoch0"], color="#F58518", lw=1.1, alpha=0.9, label="random 0.25-1.75")
        ax.set_title(dataset_title(dataset_id), fontsize=7)
        ax.grid(alpha=0.2)
        ax.tick_params(labelsize=7)
    for ax in axes[len(selected) :]:
        ax.axis("off")
    handles = [
        plt.Line2D([0], [0], color="#4C78A8", lw=1.3, ls="--", label="fixed eps=1"),
        plt.Line2D([0], [0], color="#F58518", lw=1.5, label="random 0.25-1.75"),
    ]
    fig.suptitle(
        f"Darcy {method}: generalization Relative L2 change vs own epoch 0, part {part:02d}",
        y=0.992,
        fontsize=14,
    )
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.958), ncol=2, frameon=False, fontsize=9)
    fig.text(0.5, 0.932, f"same-epoch comparison only, epoch <= {common_end}; more negative is better", ha="center", fontsize=8, color="#555555")
    fig.text(0.012, 0.50, "Relative L2 delta vs own epoch 0", va="center", rotation="vertical", fontsize=10)
    fig.text(0.50, 0.018, "epoch", ha="center", fontsize=10)
    fig.tight_layout(rect=(0.028, 0.040, 0.995, 0.905), h_pad=1.15, w_pad=0.75)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_loss1_loss2_absolute(df: pd.DataFrame, dataset_ids: list[str], part: int, out: Path) -> None:
    selected = dataset_ids[(part - 1) * 25 : part * 25]
    common_end = min(METHOD_COMMON_END["loss1"], METHOD_COMMON_END["loss2"])
    fig, axes = setup_axes(len(selected))
    for ax, dataset_id in zip(axes, selected, strict=False):
        for method in ("loss1", "loss2"):
            sub = df[(df["method"].eq(method)) & (df["dataset_id"].eq(dataset_id)) & (df["epoch"] <= common_end)].sort_values("epoch")
            if sub.empty:
                continue
            color = METHOD_COLORS[method]
            ax.plot(sub["epoch"], sub["old_fixed_relative_l2"], color=color, lw=0.85, ls="--", alpha=0.82)
            ax.plot(sub["epoch"], sub["new_random_relative_l2"], color=color, lw=1.05, alpha=0.9)
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
    fig.suptitle(
        f"Darcy loss1/loss2 fixed vs random budget generalization Relative L2, part {part:02d}",
        y=0.992,
        fontsize=14,
    )
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.958), ncol=4, frameon=False, fontsize=9)
    fig.text(0.5, 0.932, f"same-epoch comparison only, epoch <= {common_end}; dashed=fixed, solid=random; lower is better", ha="center", fontsize=8, color="#555555")
    fig.text(0.012, 0.50, "Relative L2", va="center", rotation="vertical", fontsize=10)
    fig.text(0.50, 0.018, "epoch", ha="center", fontsize=10)
    fig.tight_layout(rect=(0.028, 0.040, 0.995, 0.905), h_pad=1.15, w_pad=0.75)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS)
    args = parser.parse_args()

    analysis_dir = args.analysis_dir.resolve()
    df = load_pairs(analysis_dir / "paired_dataset_comparison_all_common_epochs.csv")
    dataset_ids = ordered_datasets(df)
    out_dir = analysis_dir / "figures_fixed_random_generalization_panels_clear"
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []
    for method in ("loss1", "loss2", "loss3"):
        for part in (1, 2):
            out = out_dir / f"darcy_{method}_fixed_vs_random_relative_l2_generalization_part{part:02d}.png"
            plot_method_absolute(df, method, dataset_ids, part, out)
            outputs.append(out)
            out = out_dir / f"darcy_{method}_fixed_vs_random_delta_vs_epoch0_generalization_part{part:02d}.png"
            plot_method_delta(df, method, dataset_ids, part, out)
            outputs.append(out)
    for part in (1, 2):
        out = out_dir / f"darcy_loss1_loss2_fixed_vs_random_relative_l2_generalization_part{part:02d}.png"
        plot_loss1_loss2_absolute(df, dataset_ids, part, out)
        outputs.append(out)

    manifest = pd.DataFrame({"figure": [str(path.relative_to(analysis_dir)) for path in outputs]})
    manifest.to_csv(out_dir / "figure_manifest.csv", index=False)
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
