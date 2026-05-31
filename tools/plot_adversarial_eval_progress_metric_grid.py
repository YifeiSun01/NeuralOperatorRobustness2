#!/usr/bin/env python3
"""Plot 2x3 per-metric eval progress grids for adversarial training runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from plot_adversarial_eval_progress_bars import (
    CHANGE_ALPHA,
    FLAT_COLOR,
    HATCH_EDGE,
    PREV_ALPHA,
    aligned_styles,
    aligned_values,
    draw_styled_bar,
    family_hatch,
    lighten_color,
    load_task_eval,
    load_total_epoch,
    order_datasets,
    pass_label,
    short_label,
    style_legend_handles,
    task_passes,
    _clamp01,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--task", default="burgers")
    parser.add_argument("--metric", default="relative_l2", choices=["relative_l2", "rmse", "mae"])
    parser.add_argument("--out-path", type=Path, required=True)
    parser.add_argument("--run-label", default=None)
    parser.add_argument("--max-label-len", type=int, default=26)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--base-alpha", type=float, default=PREV_ALPHA)
    parser.add_argument("--change-alpha", type=float, default=CHANGE_ALPHA)
    parser.add_argument("--initial-alpha", type=float, default=CHANGE_ALPHA)
    parser.add_argument("--change-lighten", type=float, default=0.0)
    parser.add_argument("--arrow-alpha", type=float, default=1.0)
    parser.add_argument("--arrow-color", default="black")
    parser.add_argument("--fig-width", type=float, default=18.6)
    parser.add_argument("--fig-height", type=float, default=11.1)
    return parser.parse_args()


def metric_label(metric: str) -> str:
    return "Relative L2 loss" if metric == "relative_l2" else metric.upper()


def draw_panel(
    ax,
    *,
    values: np.ndarray,
    prev_values: np.ndarray | None,
    x: np.ndarray,
    colors: list[object],
    hatches: list[str],
    title: str,
    ylabel: str,
    ymax: float,
    base_alpha: float,
    change_alpha: float,
    initial_alpha: float,
    change_lighten: float,
    arrow_alpha: float,
    arrow_color: str,
) -> tuple[int, int, int, float]:
    if prev_values is None:
        for i, value in enumerate(values):
            draw_styled_bar(
                ax,
                float(i),
                float(value),
                bottom=0.0,
                color=colors[i],
                hatch=hatches[i],
                alpha=initial_alpha,
                zorder=3,
            )
        up = down = flat = 0
        mean_delta = 0.0
    else:
        for i, old in enumerate(prev_values):
            draw_styled_bar(
                ax,
                float(i),
                float(old),
                bottom=0.0,
                color=colors[i],
                hatch=hatches[i],
                alpha=base_alpha,
                zorder=2,
            )
        deltas = values - prev_values
        up = int(np.sum(deltas > 1e-12))
        down = int(np.sum(deltas < -1e-12))
        flat = int(np.sum(np.abs(deltas) <= 1e-12))
        mean_delta = float(np.nanmean(deltas))
        for i, (old, new, delta) in enumerate(zip(prev_values, values, deltas)):
            if not (math.isfinite(float(old)) and math.isfinite(float(new))):
                continue
            if abs(float(delta)) > 1e-12:
                draw_styled_bar(
                    ax,
                    float(i),
                    float(delta),
                    bottom=float(old),
                    color=colors[i],
                    hatch=hatches[i],
                    alpha=change_alpha,
                    lighten=change_lighten,
                    zorder=4,
                )
                ax.annotate(
                    "",
                    xy=(i, new),
                    xytext=(i, old),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        lw=1.25,
                        color=arrow_color,
                        alpha=arrow_alpha,
                        mutation_scale=8,
                        shrinkA=0,
                        shrinkB=0,
                    ),
                    zorder=8,
                )
            else:
                ax.plot(i, new, marker="_", color=FLAT_COLOR, markersize=4, markeredgewidth=1.2, zorder=8)

    ax.set_title(f"{title}\nup={up}, down={down}, flat={flat}, mean delta={mean_delta:+.3g}", fontsize=9)
    ax.set_ylim(0, ymax)
    ax.set_xlim(-0.6, len(x) - 0.4)
    ax.grid(axis="y", linestyle="--", linewidth=0.55, alpha=0.35)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=4.4, pad=1)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_ylabel(ylabel, fontsize=8)
    return up, down, flat, mean_delta


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    df = load_task_eval(run_dir, args.task)
    passes = task_passes(df)
    if len(passes) != 6:
        print(f"[warn] expected 6 eval passes, got {len(passes)}")
    dataset_order = order_datasets(df, args.task, args.metric)
    total_epoch = load_total_epoch(run_dir, args.task)
    values_by_pass = [aligned_values(df, dataset_order, args.metric, phase, step)[0] for phase, step, epoch in passes]
    finite = np.concatenate([v[np.isfinite(v)] for v in values_by_pass])
    ymax = max(float(finite.max()) * 1.12 if finite.size else 1.0, 1e-8)

    x = np.arange(len(dataset_order))
    labels = [short_label(dataset_id, args.task, args.max_label_len) for dataset_id in dataset_order]
    colors, hatches, range_labels, family_labels, range_palette = aligned_styles(args.task, df, dataset_order)

    fig, axes = plt.subplots(2, 3, figsize=(args.fig_width, args.fig_height), sharey=True)
    axes_flat = axes.ravel()
    prev_values: np.ndarray | None = None
    ylabel = metric_label(args.metric)
    for idx, ((phase, step, epoch), values) in enumerate(zip(passes, values_by_pass)):
        ax = axes_flat[idx]
        draw_panel(
            ax,
            values=values,
            prev_values=prev_values,
            x=x,
            colors=colors,
            hatches=hatches,
            title=pass_label(phase, step, epoch, total_epoch),
            ylabel=ylabel if idx % 3 == 0 else "",
            ymax=ymax,
            base_alpha=args.base_alpha,
            change_alpha=args.change_alpha,
            initial_alpha=args.initial_alpha,
            change_lighten=args.change_lighten,
            arrow_alpha=args.arrow_alpha,
            arrow_color=args.arrow_color,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=73, ha="right")
        prev_values = values

    for idx in range(len(passes), len(axes_flat)):
        axes_flat[idx].axis("off")

    run_label = args.run_label or run_dir.name
    fig.suptitle(
        f"{args.task} {run_label}: {metric_label(args.metric)} eval 00-05, common y-range\n"
        f"base dark alpha={args.base_alpha:g}; change light alpha={args.change_alpha:g}; arrow alpha={args.arrow_alpha:g}",
        fontsize=13,
        y=0.992,
    )

    legend = style_legend_handles(range_labels, family_labels, range_palette)
    sample_range_color = next(iter(range_palette.values()), matplotlib.colormaps.get_cmap("coolwarm")(0.5))
    sample_family = next((x for x in family_labels if x not in {"train", "test"}), "gaussian")
    sample_hatch = family_hatch(sample_family)
    legend.extend(
        [
            Patch(
                facecolor=to_rgba(sample_range_color, _clamp01(args.base_alpha)),
                edgecolor=to_rgba(HATCH_EDGE, _clamp01(args.base_alpha)) if sample_hatch else "none",
                hatch=sample_hatch or None,
                linewidth=0.0,
                label="base/previous height",
            ),
            Patch(
                facecolor=to_rgba(lighten_color(sample_range_color, args.change_lighten), _clamp01(args.change_alpha)),
                edgecolor=to_rgba(HATCH_EDGE, _clamp01(args.change_alpha)) if sample_hatch else "none",
                hatch=sample_hatch or None,
                linewidth=0.0,
                label="changed amount",
            ),
            Line2D([0], [0], color=to_rgba(args.arrow_color, _clamp01(args.arrow_alpha)), lw=1.5, marker=">", markersize=6, label="arrow: previous -> current"),
        ]
    )
    fig.legend(handles=legend, loc="lower center", ncol=5, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, 0.006))
    fig.subplots_adjust(left=0.045, right=0.995, top=0.92, bottom=0.17, wspace=0.065, hspace=0.36)

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    manifest_path = args.out_path.with_suffix(".json")
    manifest_path.write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "task": args.task,
                "metric": args.metric,
                "plot": str(args.out_path),
                "base_alpha": args.base_alpha,
                "change_alpha": args.change_alpha,
                "initial_alpha": args.initial_alpha,
                "change_lighten": args.change_lighten,
                "arrow_alpha": args.arrow_alpha,
                "arrow_color": args.arrow_color,
                "ymax": ymax,
                "eval_passes": [{"phase": phase, "global_step": step, "epoch": epoch} for phase, step, epoch in passes],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[done] wrote {args.out_path}")


if __name__ == "__main__":
    main()
