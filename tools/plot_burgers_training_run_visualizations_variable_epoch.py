#!/usr/bin/env python3
"""Variable-epoch Burgers adversarial-training visualizations.

The older polished Burgers plotting scripts were written for a 1000-epoch run.
This script keeps the same visual vocabulary, but reads the maximum epoch from
the CSV files so loss1 2000-epoch and shorter aligned runs are plotted without
truncating the x-axis.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TIER_ORDER = [
    "train",
    "test",
    "near_param_shift",
    "mid_kernel_spectrum",
    "far_range_pattern",
    "target_loss_param_shift",
    "target_loss_kernel_shift",
]
TIER_COLORS = {
    "train": "#202124",
    "test": "#D55E00",
    "near_param_shift": "#0072B2",
    "mid_kernel_spectrum": "#009E73",
    "far_range_pattern": "#CC79A7",
    "target_loss_param_shift": "#E69F00",
    "target_loss_kernel_shift": "#6A3D9A",
}
ATTACK_COLORS = {
    "clean_loss_before_attack": "#0072B2",
    "adv_loss_after_attack": "#D55E00",
    "attack_loss_gain": "#009E73",
}
BG = "#fbfaf7"
AX_BG = "#ffffff"
GRID = "#d9d6cc"
TEXT = "#202124"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--suffix",
        default="",
        help="Optional filename suffix without leading underscore, for example to2000.",
    )
    parser.add_argument("--max-lines-per-panel", type=int, default=5)
    parser.add_argument("--skip-fft", action="store_true")
    return parser.parse_args()


def setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 240,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.facecolor": AX_BG,
            "figure.facecolor": BG,
            "axes.edgecolor": "#b8b3a7",
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "text.color": TEXT,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.alpha": 0.45,
            "grid.linewidth": 0.55,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "legend.fontsize": 8,
        }
    )


def name_with_suffix(filename: str, suffix: str) -> str:
    if not suffix:
        return filename
    stem, ext = filename.rsplit(".", 1)
    return f"{stem}_{suffix}.{ext}"


def savefig(fig: plt.Figure, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def shorten(s: str, n: int = 42) -> str:
    out = s.replace("burgers_", "")
    out = out.replace("gaussian_", "gauss_").replace("matern_", "mat_")
    out = out.replace("target_", "tgt_").replace("kernel_", "ker_").replace("corr", "c")
    return out if len(out) <= n else out[: n - 1] + "..."


def tier_sorted(df: pd.DataFrame) -> pd.DataFrame:
    rank = {tier: i for i, tier in enumerate(TIER_ORDER)}
    d = df.copy()
    d["_tier_rank"] = d["manual_tier"].map(rank).fillna(999)
    d["_manual_rank"] = pd.to_numeric(d["manual_rank"], errors="coerce").fillna(999)
    return d.sort_values(["_tier_rank", "_manual_rank", "dataset_id", "epoch"]).drop(
        columns=["_tier_rank", "_manual_rank"]
    )


def group_mean_std(eval_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    g = eval_df.groupby(["manual_tier", "epoch"], as_index=False)[metric].agg(["mean", "std", "count"]).reset_index()
    g["std"] = g["std"].fillna(0.0)
    return g


def major_ticks(max_epoch: int) -> list[int]:
    if max_epoch <= 100:
        step = 10
    elif max_epoch <= 500:
        step = 50
    elif max_epoch <= 1000:
        step = 100
    else:
        step = 200
    ticks = list(range(0, max_epoch + 1, step))
    if ticks[-1] != max_epoch:
        ticks.append(max_epoch)
    return ticks


def checkpoint_epochs(max_epoch: int, available_epochs: set[int]) -> list[int]:
    requested = [int(round(max_epoch * i / 10.0)) for i in range(11)]
    out: list[int] = []
    for epoch in requested:
        if epoch in available_epochs:
            chosen = epoch
        else:
            chosen = min(available_epochs, key=lambda e: abs(e - epoch))
        if chosen not in out:
            out.append(chosen)
    return out


def selected_fft_epochs(max_epoch: int, available_epochs: np.ndarray) -> list[int]:
    if max_epoch >= 2000:
        requested = [50, 200, 400, 800, 1200, 1600, 2000]
    else:
        requested = [int(round(max_epoch * f)) for f in [0.05, 0.20, 0.40, 0.60, 0.80, 1.00]]
    out: list[int] = []
    for epoch in requested:
        chosen = int(available_epochs[np.argmin(np.abs(available_epochs - epoch))])
        if chosen not in out:
            out.append(chosen)
    return out


def positive_floor(values: np.ndarray) -> float:
    finite = values[np.isfinite(values) & (values > 0)]
    return max(float(np.nanmin(finite)) * 0.08, 1e-12) if finite.size else 1e-12


def positive_band(mean: np.ndarray, std: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)
    floor = positive_floor(np.concatenate([mean, mean + std]))
    y = np.maximum(mean, floor)
    lower = np.maximum(mean - std, floor)
    upper = np.maximum(mean + std, floor * 1.2)
    return y, lower, upper


def pooled_epoch_std(bucket_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    rows = []
    mean_col = f"{prefix}_mean"
    std_col = f"{prefix}_std"
    if mean_col not in bucket_df or std_col not in bucket_df:
        return pd.DataFrame({"epoch": [], f"{prefix}_pooled_mean": [], f"{prefix}_pooled_std": []})
    for epoch, d in bucket_df.groupby("epoch"):
        n = d["sample_count"].to_numpy(dtype=float)
        means = d[mean_col].to_numpy(dtype=float)
        stds = d[std_col].fillna(0.0).to_numpy(dtype=float)
        total_n = float(np.sum(n))
        if total_n <= 1:
            pooled_mean = float(np.nanmean(means))
            pooled_std = 0.0
        else:
            pooled_mean = float(np.sum(n * means) / total_n)
            ss_within = np.sum(np.maximum(n - 1.0, 0.0) * stds * stds)
            ss_between = np.sum(n * (means - pooled_mean) ** 2)
            pooled_std = float(np.sqrt(max((ss_within + ss_between) / (total_n - 1.0), 0.0)))
        rows.append({"epoch": int(epoch), f"{prefix}_pooled_mean": pooled_mean, f"{prefix}_pooled_std": pooled_std})
    return pd.DataFrame(rows).sort_values("epoch")


def make_attack_summary(attack_df: pd.DataFrame, bucket_df: pd.DataFrame) -> pd.DataFrame:
    out = attack_df[
        ["epoch", "clean_loss_before_attack_mean", "adv_loss_after_attack_mean", "attack_loss_gain_mean"]
    ].copy()
    for prefix in ["clean_loss_before_attack", "adv_loss_after_attack", "attack_loss_gain"]:
        pooled = pooled_epoch_std(bucket_df, prefix)
        if not pooled.empty:
            out = out.merge(pooled, on="epoch", how="left")
        else:
            out[f"{prefix}_pooled_std"] = 0.0
    return out


def bucket_label(d: pd.DataFrame) -> str:
    lo = float(d["bucket_value_low"].iloc[0])
    hi = float(d["bucket_value_high"].iloc[0])
    eps_lo = float(d["epsilon_min"].min())
    eps_hi = float(d["epsilon_max"].max())
    return f"{lo:.2f}-{hi:.2f}x eps ({eps_lo:.3f}-{eps_hi:.3f})"


def plot_attack_loss(attack_df: pd.DataFrame, bucket_df: pd.DataFrame, polished_dir: Path, suffix: str) -> Path:
    summary = make_attack_summary(attack_df, bucket_df)
    max_epoch = int(summary["epoch"].max())
    ticks = major_ticks(max_epoch)

    fig = plt.figure(figsize=(19, 12.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.9, 1.05], hspace=0.28, wspace=0.18)
    ax_top = fig.add_subplot(gs[0, :])
    axes_bottom = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]

    x = summary["epoch"].to_numpy(dtype=float)
    bounds = []
    for prefix, label in [
        ("clean_loss_before_attack", "clean before attack"),
        ("adv_loss_after_attack", "adv after attack"),
        ("attack_loss_gain", "attack gain = adv - clean"),
    ]:
        mean = summary[f"{prefix}_mean"].to_numpy(dtype=float)
        std = summary[f"{prefix}_pooled_std"].fillna(0.0).to_numpy(dtype=float)
        y, lower, upper = positive_band(mean, std)
        color = ATTACK_COLORS[prefix]
        ax_top.plot(x, y, color=color, lw=1.65, alpha=0.92, label=label)
        ax_top.fill_between(x, lower, upper, color=color, alpha=0.13, lw=0)
        bounds.extend([lower, upper])

    values = np.concatenate(bounds)
    values = values[np.isfinite(values) & (values > 0)]
    log_min, log_max = np.log10([float(np.nanmin(values)), float(np.nanmax(values))])
    pad = 0.06 * (log_max - log_min)
    ax_top.set_yscale("log")
    ax_top.set_ylim(10 ** (log_min - pad), 10 ** (log_max + pad))
    ax_top.set_xlim(1, max_epoch)
    ax_top.set_xticks([t for t in ticks if t > 0])
    ax_top.set_xlabel("training epoch")
    ax_top.set_ylabel("MSE, log scale")
    ax_top.set_title("Attack loss before perturbation, after perturbation, and attack gain", loc="left", fontweight="bold")
    ax_top.legend(loc="upper right", ncol=3)

    bucket_df = bucket_df.sort_values(["bucket_index", "epoch"])
    cmap = plt.get_cmap("viridis")
    n_buckets = int(bucket_df["bucket_index"].nunique())
    bottom_specs = [
        (axes_bottom[0], "attack_loss_gain", "Attack gain by epsilon bucket", "MSE gain, log scale", True),
        (axes_bottom[1], "attack_loss_gain_relative", "Relative attack gain by epsilon bucket", "gain / clean", False),
    ]
    for bpos, (_bucket_idx, d) in enumerate(bucket_df.groupby("bucket_index")):
        color = cmap(0.10 + 0.80 * bpos / max(1, n_buckets - 1))
        bx = d["epoch"].to_numpy(dtype=float)
        label = bucket_label(d)
        for ax, prefix, title, ylabel, use_log in bottom_specs:
            mean = d[f"{prefix}_mean"].to_numpy(dtype=float)
            std = d[f"{prefix}_std"].fillna(0.0).to_numpy(dtype=float)
            if use_log:
                y, lower, upper = positive_band(mean, std)
                ax.plot(bx, y, color=color, lw=1.45, alpha=0.92, label=label)
            else:
                lower = np.maximum(mean - std, 0.0)
                upper = mean + std
                ax.plot(bx, mean, color=color, lw=1.45, alpha=0.92, label=label)
            ax.fill_between(bx, lower, upper, color=color, alpha=0.13, lw=0)
            if use_log:
                ax.set_yscale("log")
            ax.set_xlim(1, max_epoch)
            ax.set_xticks([t for t in ticks if t > 0])
            ax.set_xlabel("training epoch")
            ax.set_ylabel(ylabel)
            ax.set_title(title, loc="left", fontweight="bold")

    finite_rel = bucket_df["attack_loss_gain_relative_mean"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(finite_rel):
        axes_bottom[1].set_ylim(0, min(10.0, max(1.0, float(finite_rel.quantile(0.995)) * 1.1)))
    axes_bottom[1].legend(title="epsilon jitter bucket", loc="upper right", fontsize=7.6, title_fontsize=8)
    fig.suptitle(f"Attack loss and attack gain during {max_epoch:,} epochs of adversarial training", fontsize=18, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return savefig(fig, polished_dir / name_with_suffix("corrected_attack_loss_three_lines_plus_buckets.png", suffix))


def metric_limits(eval_df: pd.DataFrame, metric: str) -> tuple[float, float]:
    vals = eval_df[metric].to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    lo = max(0.0, float(np.nanpercentile(vals, 0.5)))
    hi = float(np.nanpercentile(vals, 99.5))
    if hi <= lo:
        hi = float(np.nanmax(vals))
    pad = 0.06 * (hi - lo if hi > lo else 1.0)
    return max(0.0, lo - pad), hi + pad


def plot_corrected_loss(eval_df: pd.DataFrame, metric: str, polished_dir: Path, suffix: str) -> Path:
    label = "Relative L2" if metric == "relative_l2" else "RMSE"
    d = tier_sorted(eval_df)
    max_epoch = int(d["epoch"].max())
    ticks = major_ticks(max_epoch)
    order = d.drop_duplicates("dataset_id")["dataset_id"].tolist()
    tiers = d.drop_duplicates("dataset_id")["manual_tier"].tolist()
    pivot = d.pivot_table(index="dataset_id", columns="epoch", values=metric, aggfunc="mean").reindex(order)
    epochs = np.asarray(sorted(pivot.columns.astype(int)), dtype=int)
    matrix = pivot.reindex(columns=epochs).to_numpy(dtype=float)
    tier_nums = np.array([TIER_ORDER.index(t) if t in TIER_ORDER else -1 for t in tiers]).reshape(-1, 1)

    fig = plt.figure(figsize=(19.5, 13.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.24], width_ratios=[0.30, 5.8, 0.22], hspace=0.22, wspace=0.045)
    ax_strip = fig.add_subplot(gs[0, 0])
    ax_heat = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])
    ax_line = fig.add_subplot(gs[1, :])

    tier_cmap = plt.matplotlib.colors.ListedColormap([TIER_COLORS[t] for t in TIER_ORDER])
    ax_strip.imshow(tier_nums, aspect="auto", cmap=tier_cmap, vmin=0, vmax=len(TIER_ORDER) - 1)
    ax_strip.set_xticks([])
    ax_strip.set_yticks(np.arange(len(order)))
    ax_strip.set_yticklabels([shorten(x, 42) for x in order], fontsize=6.3)
    ax_strip.tick_params(axis="y", length=0, pad=5)
    ax_strip.yaxis.tick_left()
    ax_strip.set_title("group", fontsize=9)

    finite = matrix[np.isfinite(matrix)]
    vmin, vmax = np.nanpercentile(finite, [1, 99.5]) if finite.size else (0.0, 1.0)
    im = ax_heat.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap="magma",
        vmin=float(vmin),
        vmax=float(vmax),
        extent=[float(epochs[0]), float(epochs[-1]), len(order) - 0.5, -0.5],
    )
    cbar = fig.colorbar(im, cax=cax, label=label)
    cbar.ax.tick_params(labelsize=8)
    ax_heat.set_title(f"Raw {label} heatmap: {len(order)} datasets x epochs 0..{max_epoch}", loc="left", fontweight="bold")
    ax_heat.set_xlabel("evaluation epoch")
    ax_heat.set_ylabel("")
    ax_heat.set_xticks(ticks)
    ax_heat.set_yticks([])

    for i in range(1, len(tiers)):
        if tiers[i] != tiers[i - 1]:
            ax_heat.axhline(i - 0.5, color="white", lw=1.25, alpha=0.78)
            ax_strip.axhline(i - 0.5, color="white", lw=1.25, alpha=0.78)

    g = group_mean_std(eval_df, metric)
    for tier in TIER_ORDER:
        dd = g[g["manual_tier"] == tier].sort_values("epoch")
        if dd.empty:
            continue
        x = dd["epoch"].to_numpy(dtype=float)
        mean = dd["mean"].to_numpy(dtype=float)
        std = dd["std"].to_numpy(dtype=float)
        color = TIER_COLORS[tier]
        ax_line.plot(x, mean, color=color, lw=1.6, alpha=0.80, label=tier)
        ax_line.fill_between(x, np.maximum(mean - std, 0.0), mean + std, color=color, alpha=0.09, lw=0)
    ax_line.set_xlim(0, max_epoch)
    ax_line.set_xticks(ticks)
    ax_line.set_xlabel("evaluation epoch")
    ax_line.set_ylabel(label)
    ax_line.set_title("Group mean trajectory over the full available training horizon", loc="left", fontweight="bold", fontsize=10.5)
    ax_line.legend(ncol=4, loc="upper right", fontsize=7.6)

    fig.suptitle(f"{label} trajectories by dataset family during adversarial training", fontsize=18, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    filename = f"corrected_{metric}_full_heatmap_raw_group_line.png"
    return savefig(fig, polished_dir / name_with_suffix(filename, suffix))


def plot_checkpoint_style_loss(eval_df: pd.DataFrame, metric: str, polished_dir: Path, suffix: str) -> Path:
    label = "Relative L2" if metric == "relative_l2" else "RMSE"
    max_epoch = int(eval_df["epoch"].max())
    cps = checkpoint_epochs(max_epoch, set(int(e) for e in eval_df["epoch"].unique()))
    d = tier_sorted(eval_df[eval_df["epoch"].isin(cps)])
    order = d.drop_duplicates("dataset_id")["dataset_id"].tolist()
    tiers = d.drop_duplicates("dataset_id")["manual_tier"].tolist()
    pivot = d.pivot_table(index="dataset_id", columns="epoch", values=metric, aggfunc="mean").reindex(order)
    pivot = pivot.reindex(columns=cps)
    matrix = pivot.to_numpy(dtype=float)
    csv_name = name_with_suffix(f"polished_checkpoint_style_{metric}_absolute11_heatmap_line_below.csv", suffix)
    pivot.to_csv(polished_dir / csv_name)

    tier_nums = np.array([TIER_ORDER.index(t) if t in TIER_ORDER else -1 for t in tiers]).reshape(-1, 1)
    fig = plt.figure(figsize=(19.5, 13.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.22], width_ratios=[0.30, 5.8, 0.22], hspace=0.22, wspace=0.045)
    ax_strip = fig.add_subplot(gs[0, 0])
    ax_heat = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])
    ax_line = fig.add_subplot(gs[1, :])

    tier_cmap = plt.matplotlib.colors.ListedColormap([TIER_COLORS[t] for t in TIER_ORDER])
    ax_strip.imshow(tier_nums, aspect="auto", cmap=tier_cmap, vmin=0, vmax=len(TIER_ORDER) - 1)
    ax_strip.set_xticks([])
    ax_strip.set_yticks(np.arange(len(order)))
    ax_strip.set_yticklabels([shorten(x, 42) for x in order], fontsize=6.3)
    ax_strip.tick_params(axis="y", length=0, pad=5)
    ax_strip.yaxis.tick_left()
    ax_strip.set_title("group", fontsize=9)

    im = ax_heat.imshow(matrix, aspect="auto", interpolation="nearest", cmap="magma")
    cbar = fig.colorbar(im, cax=cax, label=label)
    cbar.ax.tick_params(labelsize=8)
    ax_heat.set_title(f"Absolute {label} at {len(cps)} checkpoints: epochs {cps[0]}..{cps[-1]}", loc="left", fontweight="bold")
    ax_heat.set_xlabel("evaluation epoch")
    ax_heat.set_ylabel("")
    ax_heat.set_xticks(np.arange(len(cps)))
    ax_heat.set_xticklabels([str(x) for x in cps], rotation=0)
    ax_heat.set_yticks([])

    for i in range(1, len(tiers)):
        if tiers[i] != tiers[i - 1]:
            ax_heat.axhline(i - 0.5, color="white", lw=1.25, alpha=0.78)
            ax_strip.axhline(i - 0.5, color="white", lw=1.25, alpha=0.78)

    g = group_mean_std(eval_df[eval_df["epoch"].isin(cps)], metric)
    y_values: list[np.ndarray] = []
    for tier in TIER_ORDER:
        dd = g[g["manual_tier"] == tier].sort_values("epoch")
        if dd.empty:
            continue
        x = dd["epoch"].to_numpy(dtype=float)
        mean = dd["mean"].to_numpy(dtype=float)
        std = dd["std"].to_numpy(dtype=float)
        color = TIER_COLORS[tier]
        ax_line.plot(x, mean, color=color, lw=1.7, alpha=0.78, marker="o", ms=2.6, label=tier)
        ax_line.fill_between(x, np.maximum(mean - std, 0.0), mean + std, color=color, alpha=0.09, lw=0)
        y_values.extend([mean - std, mean + std])
    if y_values:
        vals = np.concatenate([v[np.isfinite(v)] for v in y_values])
        ymin = max(0.0, float(np.nanmin(vals)))
        ymax = float(np.nanmax(vals))
        pad = 0.05 * (ymax - ymin if ymax > ymin else 1.0)
        ax_line.set_ylim(max(0.0, ymin - pad), ymax + pad)
    ax_line.set_xlim(0, max_epoch)
    ax_line.set_xticks(cps)
    ax_line.set_title("Group mean lineplot at the same checkpoints; no epoch moving average", loc="left", fontweight="bold", fontsize=10.5)
    ax_line.set_xlabel("evaluation epoch")
    ax_line.set_ylabel(label)
    ax_line.legend(ncol=4, loc="upper right", fontsize=7.6)

    fig.suptitle(f"Checkpoint-style {label} heatmap with short group lineplot below", fontsize=18, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    png_name = name_with_suffix(f"polished_checkpoint_style_{metric}_absolute11_heatmap_line_below.png", suffix)
    return savefig(fig, polished_dir / png_name)


def plot_high_transparency_max5(
    eval_df: pd.DataFrame,
    metric: str,
    out_path: Path,
    max_lines_per_panel: int,
    smooth_window: int | None,
) -> pd.DataFrame:
    ylo, yhi = metric_limits(eval_df, metric)
    sorted_df = tier_sorted(eval_df)
    present = [tier for tier in TIER_ORDER if tier in set(eval_df["manual_tier"])]
    panel_defs: list[tuple[str, int, list[str]]] = []
    for tier in present:
        d_tier = sorted_df[sorted_df["manual_tier"] == tier]
        datasets = list(d_tier.drop_duplicates("dataset_id")["dataset_id"])
        for chunk_idx, start in enumerate(range(0, len(datasets), max_lines_per_panel), start=1):
            panel_defs.append((tier, chunk_idx, datasets[start : start + max_lines_per_panel]))

    max_epoch = int(eval_df["epoch"].max())
    ticks = major_ticks(max_epoch)
    x_text = max_epoch + 0.01 * max_epoch
    x_right = max_epoch + 0.07 * max_epoch
    guide_lines = [int(round(max_epoch * f)) for f in [0.2, 0.4, 0.6, 0.8]]

    cols = 3
    rows = math.ceil(len(panel_defs) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(18.8, 4.15 * rows), sharex=True, sharey=True)
    axes = np.asarray(axes).ravel()
    cmap = plt.get_cmap("tab10")
    index_rows: list[dict[str, object]] = []

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
            label = f"{local_idx + 1:02d} {shorten(dataset_id)}"
            ax.plot(x, y_arr, color=color, alpha=line_alpha, lw=lw, label=label)
            ax.text(x_text, float(y_arr[-1]), f"{local_idx + 1}", color=color, alpha=label_alpha, fontsize=6.3, va="center", ha="left", clip_on=False)
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

        for xline in guide_lines:
            ax.axvline(xline, color="#555555", alpha=0.13, lw=0.9)
        ax.set_title(f"{tier} chunk {chunk_idx}: {len(chunk)} datasets", fontsize=9.8)
        ax.set_xlim(0, x_right)
        ax.set_xticks(ticks)
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
    fig.suptitle(f"{trend_prefix} {metric_label} trajectories by dataset family during adversarial training", y=0.997)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    savefig(fig, out_path)
    return pd.DataFrame(index_rows)


def load_probe_npz(burgers_dir: Path, epoch: int) -> dict[str, np.ndarray]:
    matches = sorted((burgers_dir / "attack_probe_samples").glob(f"burgers_epoch{epoch:03d}_step*_attack_probe.npz"))
    if not matches:
        matches = sorted((burgers_dir / "attack_probe_samples").glob(f"burgers_epoch{epoch:04d}_step*_attack_probe.npz"))
    if not matches:
        raise FileNotFoundError(f"No attack probe npz for epoch {epoch}")
    return dict(np.load(matches[0], allow_pickle=True))


def epoch_from_npz_path(path: str) -> int | None:
    match = re.search(r"epoch(\d+)_step", str(path))
    return int(match.group(1)) if match else None


def fft_power(delta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = np.asarray(delta, dtype=float).reshape(-1)
    d = d - d.mean()
    power = np.abs(np.fft.rfft(d)) ** 2
    freq = np.fft.rfftfreq(d.size, d=1.0 / d.size)
    total = power[1:].sum()
    if total > 0:
        power = power / total
    return freq, power


def compute_delta_fft_matrix(burgers_dir: Path, epochs: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = []
    freq_modes = None
    for epoch in sorted(int(e) for e in epochs):
        data = load_probe_npz(burgers_dir, epoch)
        powers = []
        for sample_idx in range(data["delta"].shape[0]):
            freq, power = fft_power(data["delta"][sample_idx])
            if freq_modes is None:
                freq_modes = freq[1:]
            powers.append(power[1:])
        rows.append(np.nanmean(np.asarray(powers, dtype=float), axis=0))
    return np.asarray(sorted(epochs), dtype=int), np.asarray(freq_modes, dtype=float), np.asarray(rows, dtype=float)


def rolling_epoch_matrix(matrix: np.ndarray, window: int = 25) -> np.ndarray:
    return pd.DataFrame(matrix).rolling(window, center=True, min_periods=1).mean().to_numpy(dtype=float)


def plot_delta_fft(burgers_dir: Path, probe_df: pd.DataFrame, polished_dir: Path, suffix: str) -> Path:
    raw_epochs = sorted(int(e) for e in probe_df["epoch"].dropna().unique())
    epochs, freq_modes, raw_matrix = compute_delta_fft_matrix(burgers_dir, raw_epochs)
    epoch_smooth = rolling_epoch_matrix(raw_matrix, 25)
    log_power = np.log10(np.clip(raw_matrix, 1e-18, None))
    finite = log_power[np.isfinite(log_power)]
    vmin, vmax = np.nanpercentile(finite, [2, 99.5])

    max_epoch = int(epochs[-1])
    selected = selected_fft_epochs(max_epoch, epochs)
    fig = plt.figure(figsize=(18, 12.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.28], hspace=0.24)
    ax_heat = fig.add_subplot(gs[0, 0])
    ax_line = fig.add_subplot(gs[1, 0])

    im = ax_heat.imshow(
        log_power,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[float(freq_modes[0]), float(freq_modes[-1]), float(epochs[0]), float(epochs[-1])],
        cmap="magma",
        vmin=float(vmin),
        vmax=float(vmax),
    )
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.024, pad=0.018)
    cbar.set_label("log10 normalized FFT power")
    ax_heat.set_title("Raw delta FFT power heatmap, no moving average", loc="left", fontweight="bold")
    ax_heat.set_xlabel("Fourier mode")
    ax_heat.set_ylabel("training epoch")
    ax_heat.set_xlim(1, min(512, float(freq_modes[-1])))
    ax_heat.set_yticks(major_ticks(max_epoch))

    colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.95, len(selected)))
    for epoch, color in zip(selected, colors):
        idx = int(np.argmin(np.abs(epochs - epoch)))
        ax_line.plot(freq_modes, epoch_smooth[idx] + 1e-18, color=color, lw=1.55, alpha=0.84, label=f"epoch {epochs[idx]}")
    ax_line.set_yscale("log")
    ax_line.set_xlim(1, min(512, float(freq_modes[-1])))
    ax_line.set_title("Selected spectra from 25-epoch moving-average matrix; top heatmap remains raw", loc="left", fontweight="bold", fontsize=10.5)
    ax_line.set_xlabel("Fourier mode")
    ax_line.set_ylabel("normalized power")
    ax_line.legend(ncol=min(7, len(selected)), loc="upper right", fontsize=7.6)

    fig.suptitle("Delta frequency content during adversarial training", fontsize=18, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    return savefig(fig, polished_dir / name_with_suffix("polished_checkpoint_style_delta_fft_raw_heatmap_25epoch_smoothed_lines.png", suffix))


def main() -> None:
    args = parse_args()
    setup_matplotlib()
    burgers_dir = args.run_dir / "burgers"
    out_dir = args.out_dir
    polished_dir = out_dir / "polished_report"
    out_dir.mkdir(parents=True, exist_ok=True)
    polished_dir.mkdir(parents=True, exist_ok=True)

    eval_df = pd.read_csv(burgers_dir / "eval_metrics.csv")
    attack_df = pd.read_csv(burgers_dir / "attack_epoch_summary.csv")
    bucket_df = pd.read_csv(burgers_dir / "attack_epsilon_bucket_summary.csv")
    probe_df = pd.read_csv(burgers_dir / "attack_probe_samples.csv")

    outputs: list[str] = []
    outputs.append(str(plot_attack_loss(attack_df, bucket_df, polished_dir, args.suffix)))
    for metric in ["rmse", "relative_l2"]:
        outputs.append(str(plot_corrected_loss(eval_df, metric, polished_dir, args.suffix)))
        outputs.append(str(plot_checkpoint_style_loss(eval_df, metric, polished_dir, args.suffix)))
        for smooth_window, short in [(None, "raw"), (25, "ma25")]:
            out = out_dir / name_with_suffix(
                f"{metric}_grouped_shared_y_distinct_datasets_max5_high_transparency_{short}.png",
                args.suffix,
            )
            index = plot_high_transparency_max5(eval_df, metric, out, args.max_lines_per_panel, smooth_window)
            csv_path = out.with_suffix(".csv")
            index.to_csv(csv_path, index=False)
            outputs.extend([str(out), str(csv_path)])

    if not args.skip_fft:
        outputs.append(str(plot_delta_fft(burgers_dir, probe_df, polished_dir, args.suffix)))

    manifest = {
        "run_dir": str(args.run_dir),
        "out_dir": str(out_dir),
        "max_eval_epoch": int(eval_df["epoch"].max()),
        "max_attack_epoch": int(attack_df["epoch"].max()),
        "suffix": args.suffix,
        "outputs": outputs,
    }
    manifest_path = polished_dir / name_with_suffix("variable_epoch_visualization_manifest.json", args.suffix)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\n".join(outputs + [str(manifest_path)]))


if __name__ == "__main__":
    main()
