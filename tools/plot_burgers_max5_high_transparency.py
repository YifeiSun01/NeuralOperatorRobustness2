#!/usr/bin/env python3
"""High-transparency Max5 grouped/shared-y dataset line panels."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_burgers_zero_adv_training_visualizations import (  # noqa: E402
    TIER_ORDER,
    metric_limits,
    setup_matplotlib,
    shorten_dataset_id,
    sort_eval_rows,
)

RUN_DIR = Path("adversarial_training_runs/burgers_zero_advonly_random_jitter_1000ep_bs480_steps5_eps5bucket_20260601")
BURGERS_DIR = RUN_DIR / "burgers"
OUT_DIR = Path("visualizations/burgers_zero_adv_training_20260601")


def plot_high_transparency_max5(
    eval_df: pd.DataFrame,
    metric: str,
    out_path: Path,
    max_lines_per_panel: int = 5,
    smooth_window: int | None = None,
) -> pd.DataFrame:
    ylo, yhi = metric_limits(eval_df, metric)
    sorted_df = sort_eval_rows(eval_df)
    present = [tier for tier in TIER_ORDER if tier in set(eval_df["manual_tier"])]
    panel_defs: list[tuple[str, int, list[str]]] = []
    for tier in present:
        d_tier = sorted_df[sorted_df["manual_tier"] == tier]
        datasets = list(d_tier.drop_duplicates("dataset_id")["dataset_id"])
        for chunk_idx, start in enumerate(range(0, len(datasets), max_lines_per_panel), start=1):
            panel_defs.append((tier, chunk_idx, datasets[start : start + max_lines_per_panel]))

    cols = 3
    rows = math.ceil(len(panel_defs) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(18.8, 4.15 * rows), sharex=True, sharey=True)
    axes = np.asarray(axes).ravel()
    cmap = plt.get_cmap("tab10")
    index_rows: list[dict[str, object]] = []
    suffix = f", {smooth_window}-epoch moving average" if smooth_window else ", raw epoch values"

    for panel_idx, (ax, (tier, chunk_idx, chunk)) in enumerate(zip(axes, panel_defs), start=1):
        line_alpha = 0.26 if len(chunk) >= 2 else 0.80
        label_alpha = 0.52 if len(chunk) >= 2 else 0.80
        for local_idx, dataset_id in enumerate(chunk):
            d = sorted_df[sorted_df["dataset_id"] == dataset_id].sort_values("epoch")
            x = d["epoch"].to_numpy(dtype=float)
            y = d[metric].astype(float)
            y_plot = y.rolling(smooth_window, min_periods=1, center=True).mean() if smooth_window else y
            y_arr = y_plot.to_numpy(dtype=float)
            color = cmap(local_idx % 10)
            lw = 1.85 if tier in {"train", "test"} else 1.45
            label = f"{local_idx + 1:02d} {shorten_dataset_id(dataset_id)}"
            ax.plot(x, y_arr, color=color, alpha=line_alpha, lw=lw, label=label)
            ax.text(
                1005,
                float(y_arr[-1]),
                f"{local_idx + 1}",
                color=color,
                alpha=label_alpha,
                fontsize=6.3,
                va="center",
                ha="left",
                clip_on=False,
            )
            index_rows.append(
                {
                    "metric": metric,
                    "panel": panel_idx,
                    "tier": tier,
                    "tier_chunk": chunk_idx,
                    "line_on_panel": local_idx + 1,
                    "dataset_id": dataset_id,
                    "manual_rank": d["manual_rank"].iloc[0],
                    "smooth_window": smooth_window or 0,
                    "line_alpha": line_alpha,
                    "figure": out_path.name,
                }
            )

        for xline in [200, 400, 600, 800]:
            ax.axvline(xline, color="#555555", alpha=0.13, lw=0.9)
        ax.set_title(f"{tier} chunk {chunk_idx}: {len(chunk)} datasets", fontsize=9.8)
        ax.set_xlim(0, 1035)
        ax.set_ylim(ylo, yhi)
        ax.legend(
            loc="upper right",
            fontsize=5.5,
            frameon=True,
            framealpha=0.50,
            facecolor="white",
            edgecolor="none",
            handlelength=1.05,
            handletextpad=0.25,
            borderaxespad=0.22,
        )

    for ax in axes[len(panel_defs) :]:
        ax.axis("off")
    fig.supxlabel("evaluation epoch")
    fig.supylabel(metric)
    metric_label = "Relative L2 loss" if metric == "relative_l2" else "RMSE"
    trend_prefix = "Smoothed" if smooth_window else "Raw"
    fig.suptitle(
        f"{trend_prefix} {metric_label} trajectories by dataset family during adversarial training",
        y=0.997,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out_path)
    plt.close(fig)
    return pd.DataFrame(index_rows)


def main() -> None:
    setup_matplotlib()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    eval_df = pd.read_csv(BURGERS_DIR / "eval_metrics.csv")
    outputs = []
    for metric in ["relative_l2", "rmse"]:
        for smooth_window, suffix in [(None, "raw"), (25, "ma25")]:
            out = OUT_DIR / f"{metric}_grouped_shared_y_distinct_datasets_max5_high_transparency_{suffix}.png"
            index = plot_high_transparency_max5(eval_df, metric, out, max_lines_per_panel=5, smooth_window=smooth_window)
            csv = out.with_suffix(".csv")
            index.to_csv(csv, index=False)
            outputs.append(str(out))
            outputs.append(str(csv))
    print("\n".join(outputs))


if __name__ == "__main__":
    main()
