#!/usr/bin/env python3
"""Variable-epoch Darcy adversarial-training visualizations in Burgers style.

This mirrors ``plot_burgers_training_run_visualizations_variable_epoch.py``:
attack loss with bucket panels, raw dataset heatmaps with group lines,
checkpoint heatmaps with a short line plot below, grouped dataset trajectories,
and delta FFT heatmaps when attack-probe NPZ files are present.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


GROUP_ORDER = [
    "train",
    "test",
    "matern_smooth",
    "matern_fine",
    "highpass_grf",
    "bandpass_grf",
    "wave_mix",
    "blocky_tiles",
    "rectangles",
    "cellular_blobs",
    "other_generalization",
]
GROUP_COLORS = {
    "train": "#202124",
    "test": "#D55E00",
    "matern_smooth": "#0072B2",
    "matern_fine": "#009E73",
    "highpass_grf": "#CC79A7",
    "bandpass_grf": "#E69F00",
    "wave_mix": "#6A3D9A",
    "blocky_tiles": "#56B4E9",
    "rectangles": "#8C564B",
    "cellular_blobs": "#7F7F7F",
    "other_generalization": "#999999",
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
    parser.add_argument("--suffix", default="")
    parser.add_argument("--task-subdir", default="darcy")
    parser.add_argument("--max-lines-per-panel", type=int, default=5)
    parser.add_argument("--max-epoch", type=int, default=None)
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


def filter_max_epoch(df: pd.DataFrame, max_epoch: int | None) -> pd.DataFrame:
    if max_epoch is None or "epoch" not in df.columns:
        return df
    epochs = pd.to_numeric(df["epoch"], errors="coerce")
    return df[epochs <= max_epoch].copy()


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
    out = str(s)
    for prefix in [
        "darcy_binary_loss3targeted_20260611_",
        "darcy_binary_diverse_20260611_",
        "darcy_binary_diverse_preview_20260611_",
        "train_original_binary_grf_",
        "test_original_binary_grf_",
    ]:
        out = out.replace(prefix, "")
    out = out.replace("matern_", "mat_").replace("highpass_", "hi_")
    out = out.replace("bandpass_", "band_").replace("cellular_", "cell_")
    out = out.replace("frac0p", "f.").replace("alpha", "a")
    return out if len(out) <= n else out[: n - 1] + "..."


def dataset_rank(dataset_id: str, split: str) -> float:
    if split == "train":
        return 0.0
    if split == "test":
        return 0.1
    match = re.search(r"_(\d{2,3})_", str(dataset_id))
    return 10.0 + float(match.group(1)) if match else 999.0


def infer_group(row: pd.Series) -> str:
    split = str(row.get("split", "")).lower()
    if split in {"train", "test"}:
        return split
    dataset_id = str(row.get("dataset_id", "")).lower()
    family = str(row.get("family", "")).lower()
    text = f"{dataset_id} {family}"
    for group in [
        "matern_smooth",
        "matern_fine",
        "highpass_grf",
        "bandpass_grf",
        "wave_mix",
        "blocky_tiles",
        "rectangles",
        "cellular_blobs",
    ]:
        if group in text:
            return group
    if "highpass" in text:
        return "highpass_grf"
    if "bandpass" in text:
        return "bandpass_grf"
    if "wave" in text or "sine" in text or "sin" in text:
        return "wave_mix"
    if "block" in text or "tile" in text:
        return "blocky_tiles"
    if "rect" in text:
        return "rectangles"
    if "cell" in text or "blob" in text:
        return "cellular_blobs"
    if "matern" in text and "fine" in text:
        return "matern_fine"
    if "matern" in text:
        return "matern_smooth"
    return "other_generalization"


def prepare_eval_df(eval_df: pd.DataFrame) -> pd.DataFrame:
    d = eval_df.copy()
    d["darcy_group"] = d.apply(infer_group, axis=1)
    if "manual_rank" not in d.columns:
        d["manual_rank"] = np.nan
    d["dataset_order"] = [
        dataset_rank(dataset_id, split)
        for dataset_id, split in zip(d["dataset_id"].astype(str), d["split"].astype(str))
    ]
    return d


def tier_sorted(df: pd.DataFrame) -> pd.DataFrame:
    rank = {tier: i for i, tier in enumerate(GROUP_ORDER)}
    d = df.copy()
    d["_tier_rank"] = d["darcy_group"].map(rank).fillna(999)
    d["_manual_rank"] = pd.to_numeric(d["manual_rank"], errors="coerce").fillna(999)
    d["_dataset_order"] = pd.to_numeric(d["dataset_order"], errors="coerce").fillna(999)
    return d.sort_values(["_tier_rank", "_dataset_order", "_manual_rank", "dataset_id", "epoch"]).drop(
        columns=["_tier_rank", "_manual_rank", "_dataset_order"]
    )


def group_mean_std(eval_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    g = eval_df.groupby(["darcy_group", "epoch"], as_index=False)[metric].agg(["mean", "std", "count"]).reset_index()
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
    if not ticks or ticks[-1] != max_epoch:
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
    return max(float(np.nanmin(finite)) * 0.08, 1e-18) if finite.size else 1e-18


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
    needed = {"epoch", "sample_count", mean_col, std_col}
    if not needed.issubset(bucket_df.columns):
        return pd.DataFrame({"epoch": [], f"{prefix}_pooled_mean": [], f"{prefix}_pooled_std": []})
    usable = bucket_df[bucket_df["sample_count"].fillna(0) > 0].copy()
    if usable.empty:
        return pd.DataFrame({"epoch": [], f"{prefix}_pooled_mean": [], f"{prefix}_pooled_std": []})
    for epoch, d in usable.groupby("epoch"):
        n = d["sample_count"].to_numpy(dtype=float)
        means = d[mean_col].to_numpy(dtype=float)
        stds = d[std_col].fillna(0.0).to_numpy(dtype=float)
        mask = np.isfinite(n) & np.isfinite(means) & np.isfinite(stds)
        n, means, stds = n[mask], means[mask], stds[mask]
        total_n = float(np.sum(n))
        if total_n <= 1:
            pooled_mean = float(np.nanmean(means)) if means.size else np.nan
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


def bucket_summary_is_usable(bucket_df: pd.DataFrame) -> bool:
    return (
        not bucket_df.empty
        and {"bucket_index", "sample_count", "epoch"}.issubset(bucket_df.columns)
        and float(bucket_df["sample_count"].fillna(0).sum()) > 0
    )


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
    if values.size:
        log_min, log_max = np.log10([float(np.nanmin(values)), float(np.nanmax(values))])
        pad = 0.06 * (log_max - log_min if log_max > log_min else 1.0)
        ax_top.set_ylim(10 ** (log_min - pad), 10 ** (log_max + pad))
    ax_top.set_yscale("log")
    ax_top.set_xlim(1 if max_epoch > 1 else 0, max_epoch)
    ax_top.set_xticks([t for t in ticks if t > 0])
    ax_top.set_xlabel("training epoch")
    ax_top.set_ylabel("MSE, log scale")
    ax_top.set_title("Attack loss before perturbation, after perturbation, and attack gain", loc="left", fontweight="bold")
    ax_top.legend(loc="upper right", ncol=3)

    if bucket_summary_is_usable(bucket_df):
        usable = bucket_df[bucket_df["sample_count"].fillna(0) > 0].sort_values(["bucket_index", "epoch"])
        cmap = plt.get_cmap("viridis")
        n_buckets = int(usable["bucket_index"].nunique())
        bottom_specs = [
            (axes_bottom[0], "attack_loss_gain", "Attack gain by epsilon bucket", "MSE gain, log scale", True),
            (axes_bottom[1], "attack_loss_gain_relative", "Relative attack gain by epsilon bucket", "gain / clean", False),
        ]
        for bpos, (_bucket_idx, d) in enumerate(usable.groupby("bucket_index")):
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
                ax.set_xlim(1 if max_epoch > 1 else 0, max_epoch)
                ax.set_xticks([t for t in ticks if t > 0])
                ax.set_xlabel("training epoch")
                ax.set_ylabel(ylabel)
                ax.set_title(title, loc="left", fontweight="bold")

        finite_rel = usable["attack_loss_gain_relative_mean"].replace([np.inf, -np.inf], np.nan).dropna()
        if len(finite_rel):
            axes_bottom[1].set_ylim(0, min(10.0, max(1.0, float(finite_rel.quantile(0.995)) * 1.1)))
        axes_bottom[1].legend(title="epsilon jitter bucket", loc="upper right", fontsize=7.6, title_fontsize=8)
    else:
        for ax, title in zip(axes_bottom, ["Attack gain by epsilon bucket", "Relative attack gain by epsilon bucket"]):
            ax.set_title(title, loc="left", fontweight="bold")
            ax.set_axis_off()
            ax.text(
                0.5,
                0.52,
                "epsilon-bucket samples unavailable for this Darcy run\nfixed epsilon/no attack-probe bucket capture in the saved CSV",
                ha="center",
                va="center",
                fontsize=13,
                color="#555555",
                transform=ax.transAxes,
            )

    fig.suptitle(f"Attack loss and attack gain during {max_epoch:,} epochs of Darcy adversarial training", fontsize=18, fontweight="bold", y=0.995)
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


def metric_label(metric: str) -> str:
    return "Relative L2" if metric == "relative_l2" else "RMSE" if metric == "rmse" else metric


def plot_corrected_loss(eval_df: pd.DataFrame, metric: str, polished_dir: Path, suffix: str) -> Path:
    label = metric_label(metric)
    d = tier_sorted(eval_df)
    max_epoch = int(d["epoch"].max())
    ticks = major_ticks(max_epoch)
    order = d.drop_duplicates("dataset_id")["dataset_id"].tolist()
    tiers = d.drop_duplicates("dataset_id")["darcy_group"].tolist()
    pivot = d.pivot_table(index="dataset_id", columns="epoch", values=metric, aggfunc="mean").reindex(order)
    epochs = np.asarray(sorted(pivot.columns.astype(int)), dtype=int)
    matrix = pivot.reindex(columns=epochs).to_numpy(dtype=float)
    tier_nums = np.array([GROUP_ORDER.index(t) if t in GROUP_ORDER else len(GROUP_ORDER) - 1 for t in tiers]).reshape(-1, 1)

    fig = plt.figure(figsize=(19.5, 13.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.24], width_ratios=[0.30, 5.8, 0.22], hspace=0.22, wspace=0.045)
    ax_strip = fig.add_subplot(gs[0, 0])
    ax_heat = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])
    ax_line = fig.add_subplot(gs[1, :])

    tier_cmap = plt.matplotlib.colors.ListedColormap([GROUP_COLORS[t] for t in GROUP_ORDER])
    ax_strip.imshow(tier_nums, aspect="auto", cmap=tier_cmap, vmin=0, vmax=len(GROUP_ORDER) - 1)
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
    for tier in GROUP_ORDER:
        dd = g[g["darcy_group"] == tier].sort_values("epoch")
        if dd.empty:
            continue
        x = dd["epoch"].to_numpy(dtype=float)
        mean = dd["mean"].to_numpy(dtype=float)
        std = dd["std"].to_numpy(dtype=float)
        color = GROUP_COLORS[tier]
        ax_line.plot(x, mean, color=color, lw=1.6, alpha=0.80, label=tier)
        ax_line.fill_between(x, np.maximum(mean - std, 0.0), mean + std, color=color, alpha=0.09, lw=0)
    ax_line.set_xlim(0, max_epoch)
    ax_line.set_xticks(ticks)
    ax_line.set_xlabel("evaluation epoch")
    ax_line.set_ylabel(label)
    ax_line.set_title("Group mean trajectory over the full available training horizon", loc="left", fontweight="bold", fontsize=10.5)
    ax_line.legend(ncol=4, loc="upper right", fontsize=7.6)

    fig.suptitle(f"Darcy {label} trajectories by binary coefficient-field family during adversarial training", fontsize=18, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    filename = f"corrected_{metric}_full_heatmap_raw_group_line.png"
    return savefig(fig, polished_dir / name_with_suffix(filename, suffix))


def plot_checkpoint_style_loss(eval_df: pd.DataFrame, metric: str, polished_dir: Path, suffix: str) -> Path:
    label = metric_label(metric)
    max_epoch = int(eval_df["epoch"].max())
    cps = checkpoint_epochs(max_epoch, set(int(e) for e in eval_df["epoch"].unique()))
    d = tier_sorted(eval_df[eval_df["epoch"].isin(cps)])
    order = d.drop_duplicates("dataset_id")["dataset_id"].tolist()
    tiers = d.drop_duplicates("dataset_id")["darcy_group"].tolist()
    pivot = d.pivot_table(index="dataset_id", columns="epoch", values=metric, aggfunc="mean").reindex(order)
    pivot = pivot.reindex(columns=cps)
    matrix = pivot.to_numpy(dtype=float)
    csv_name = name_with_suffix(f"polished_checkpoint_style_{metric}_absolute11_heatmap_line_below.csv", suffix)
    pivot.to_csv(polished_dir / csv_name)

    tier_nums = np.array([GROUP_ORDER.index(t) if t in GROUP_ORDER else len(GROUP_ORDER) - 1 for t in tiers]).reshape(-1, 1)
    fig = plt.figure(figsize=(19.5, 13.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.22], width_ratios=[0.30, 5.8, 0.22], hspace=0.22, wspace=0.045)
    ax_strip = fig.add_subplot(gs[0, 0])
    ax_heat = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])
    ax_line = fig.add_subplot(gs[1, :])

    tier_cmap = plt.matplotlib.colors.ListedColormap([GROUP_COLORS[t] for t in GROUP_ORDER])
    ax_strip.imshow(tier_nums, aspect="auto", cmap=tier_cmap, vmin=0, vmax=len(GROUP_ORDER) - 1)
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
    for tier in GROUP_ORDER:
        dd = g[g["darcy_group"] == tier].sort_values("epoch")
        if dd.empty:
            continue
        x = dd["epoch"].to_numpy(dtype=float)
        mean = dd["mean"].to_numpy(dtype=float)
        std = dd["std"].to_numpy(dtype=float)
        color = GROUP_COLORS[tier]
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

    fig.suptitle(f"Checkpoint-style Darcy {label} heatmap with short group lineplot below", fontsize=18, fontweight="bold", y=0.995)
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
    present = [tier for tier in GROUP_ORDER if tier in set(eval_df["darcy_group"])]
    panel_defs: list[tuple[str, int, list[str]]] = []
    for tier in present:
        d_tier = sorted_df[sorted_df["darcy_group"] == tier]
        datasets = list(d_tier.drop_duplicates("dataset_id")["dataset_id"])
        for chunk_idx, start in enumerate(range(0, len(datasets), max_lines_per_panel), start=1):
            panel_defs.append((tier, chunk_idx, datasets[start : start + max_lines_per_panel]))

    max_epoch = int(eval_df["epoch"].max())
    ticks = major_ticks(max_epoch)
    x_text = max_epoch + max(1.0, 0.01 * max_epoch)
    x_right = max_epoch + max(1.0, 0.07 * max_epoch)
    guide_lines = [int(round(max_epoch * f)) for f in [0.2, 0.4, 0.6, 0.8]]

    cols = 3
    rows = max(1, math.ceil(len(panel_defs) / cols))
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
                    "group": tier,
                    "group_chunk": chunk_idx,
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
    metric_title = "Relative L2 loss" if metric == "relative_l2" else "RMSE"
    trend_prefix = "Smoothed" if smooth_window else "Raw"
    fig.suptitle(f"{trend_prefix} Darcy {metric_title} trajectories by binary coefficient-field family", y=0.997)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    savefig(fig, out_path)
    return pd.DataFrame(index_rows)


def load_probe_npz(darcy_dir: Path, epoch: int) -> dict[str, np.ndarray]:
    probe_dir = darcy_dir / "attack_probe_samples"
    matches = sorted(probe_dir.glob(f"darcy_epoch{epoch:03d}_step*_attack_probe.npz"))
    if not matches:
        matches = sorted(probe_dir.glob(f"darcy_epoch{epoch:04d}_step*_attack_probe.npz"))
    if not matches:
        matches = sorted(probe_dir.glob(f"*epoch{epoch:03d}_step*_attack_probe.npz"))
    if not matches:
        raise FileNotFoundError(f"No Darcy attack probe npz for epoch {epoch}")
    return dict(np.load(matches[0], allow_pickle=True))


def probe_delta(data: dict[str, np.ndarray]) -> np.ndarray:
    if "delta" in data:
        return np.asarray(data["delta"])
    if "x_adv" in data and "x_clean" in data:
        return np.asarray(data["x_adv"]) - np.asarray(data["x_clean"])
    raise KeyError("attack probe npz has neither delta nor x_adv/x_clean")


def fft_power(delta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = np.asarray(delta, dtype=float).squeeze()
    if d.ndim <= 1:
        d = d.reshape(-1)
        d = d - d.mean()
        power = np.abs(np.fft.rfft(d)) ** 2
        freq = np.fft.rfftfreq(d.size, d=1.0 / d.size)
        total = power[1:].sum()
        if total > 0:
            power = power / total
        return freq, power

    d = d.reshape(d.shape[-2], d.shape[-1])
    d = d - d.mean()
    fft = np.fft.fftshift(np.fft.fftn(d))
    power2 = np.abs(fft) ** 2
    fy = np.fft.fftshift(np.fft.fftfreq(d.shape[0], d=1.0 / d.shape[0]))
    fx = np.fft.fftshift(np.fft.fftfreq(d.shape[1], d=1.0 / d.shape[1]))
    yy, xx = np.meshgrid(fy, fx, indexing="ij")
    radius = np.sqrt(xx * xx + yy * yy)
    radial_bin = np.floor(radius).astype(int)
    max_bin = int(radial_bin.max())
    radial_power = np.bincount(radial_bin.ravel(), weights=power2.ravel(), minlength=max_bin + 1).astype(float)
    freq = np.arange(max_bin + 1, dtype=float)
    total = radial_power[1:].sum()
    if total > 0:
        radial_power = radial_power / total
    return freq, radial_power


def compute_delta_fft_matrix(darcy_dir: Path, epochs: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = []
    freq_modes = None
    for epoch in sorted(int(e) for e in epochs):
        data = load_probe_npz(darcy_dir, epoch)
        deltas = probe_delta(data)
        powers = []
        for sample_idx in range(deltas.shape[0]):
            freq, power = fft_power(deltas[sample_idx])
            if freq_modes is None:
                freq_modes = freq[1:]
            powers.append(power[1:])
        rows.append(np.nanmean(np.asarray(powers, dtype=float), axis=0))
    return np.asarray(sorted(epochs), dtype=int), np.asarray(freq_modes, dtype=float), np.asarray(rows, dtype=float)


def rolling_epoch_matrix(matrix: np.ndarray, window: int = 25) -> np.ndarray:
    return pd.DataFrame(matrix).rolling(window, center=True, min_periods=1).mean().to_numpy(dtype=float)


def plot_delta_fft(darcy_dir: Path, probe_df: pd.DataFrame, polished_dir: Path, suffix: str) -> Path:
    raw_epochs = sorted(int(e) for e in probe_df["epoch"].dropna().unique())
    epochs, freq_modes, raw_matrix = compute_delta_fft_matrix(darcy_dir, raw_epochs)
    epoch_smooth = rolling_epoch_matrix(raw_matrix, 25)
    log_power = np.log10(np.clip(raw_matrix, 1e-18, None))
    finite = log_power[np.isfinite(log_power)]
    vmin, vmax = np.nanpercentile(finite, [2, 99.5]) if finite.size else (-18.0, 0.0)

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
    cbar.set_label("log10 normalized radial FFT power")
    ax_heat.set_title("Raw Darcy delta FFT power heatmap, no moving average", loc="left", fontweight="bold")
    ax_heat.set_xlabel("radial Fourier mode")
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
    ax_line.set_xlabel("radial Fourier mode")
    ax_line.set_ylabel("normalized power")
    ax_line.legend(ncol=min(7, len(selected)), loc="upper right", fontsize=7.6)

    fig.suptitle("Darcy attack delta frequency content during adversarial training", fontsize=18, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    return savefig(fig, polished_dir / name_with_suffix("polished_checkpoint_style_delta_fft_raw_heatmap_25epoch_smoothed_lines.png", suffix))


def write_fft_missing_note(path: Path, reason: str) -> None:
    path.write_text(
        "Darcy delta FFT plot was not generated for this run.\n"
        f"Reason: {reason}\n\n"
        "The Burgers-style delta FFT figure requires attack_probe_samples.csv plus per-epoch NPZ files under "
        "darcy/attack_probe_samples/. Re-run with --attack-probe-samples > 0 and "
        "--attack-probe-every-n-epochs 1; tools/run_darcy_loss123physics_full_logging_20260611.sh already sets this.\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    setup_matplotlib()
    darcy_dir = args.run_dir / args.task_subdir
    if not darcy_dir.exists():
        darcy_dir = args.run_dir
    out_dir = args.out_dir
    polished_dir = out_dir / "polished_report"
    out_dir.mkdir(parents=True, exist_ok=True)
    polished_dir.mkdir(parents=True, exist_ok=True)

    eval_df = prepare_eval_df(filter_max_epoch(pd.read_csv(darcy_dir / "eval_metrics.csv"), args.max_epoch))
    attack_df = filter_max_epoch(pd.read_csv(darcy_dir / "attack_epoch_summary.csv"), args.max_epoch)
    bucket_path = darcy_dir / "attack_epsilon_bucket_summary.csv"
    bucket_df = filter_max_epoch(pd.read_csv(bucket_path), args.max_epoch) if bucket_path.exists() else pd.DataFrame()
    probe_path = darcy_dir / "attack_probe_samples.csv"
    probe_df = filter_max_epoch(pd.read_csv(probe_path), args.max_epoch) if probe_path.exists() and probe_path.stat().st_size else pd.DataFrame()

    outputs: list[str] = []
    fft_status = "skipped_by_argument" if args.skip_fft else "not_attempted"
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
        if probe_df.empty:
            fft_status = "missing_attack_probe_samples_csv"
            note = polished_dir / name_with_suffix("delta_fft_missing_note.txt", args.suffix)
            write_fft_missing_note(note, "attack_probe_samples.csv is missing or empty")
            outputs.append(str(note))
        else:
            try:
                outputs.append(str(plot_delta_fft(darcy_dir, probe_df, polished_dir, args.suffix)))
                fft_status = "generated"
            except Exception as exc:
                fft_status = f"failed: {type(exc).__name__}: {exc}"
                note = polished_dir / name_with_suffix("delta_fft_missing_note.txt", args.suffix)
                write_fft_missing_note(note, fft_status)
                outputs.append(str(note))

    dataset_count = int(eval_df["dataset_id"].nunique())
    generalization_count = int(eval_df[eval_df["split"] == "generalization"]["dataset_id"].nunique())
    manifest = {
        "run_dir": str(args.run_dir),
        "darcy_dir": str(darcy_dir),
        "out_dir": str(out_dir),
        "max_eval_epoch": int(eval_df["epoch"].max()),
        "max_attack_epoch": int(attack_df["epoch"].max()),
        "dataset_count_logged_per_epoch": dataset_count,
        "generalization_dataset_count_logged_per_epoch": generalization_count,
        "suffix": args.suffix,
        "fft_status": fft_status,
        "outputs": outputs,
    }
    manifest_path = polished_dir / name_with_suffix("variable_epoch_visualization_manifest.json", args.suffix)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\n".join(outputs + [str(manifest_path)]))


if __name__ == "__main__":
    main()
