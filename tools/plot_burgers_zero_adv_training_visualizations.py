#!/usr/bin/env python3
"""Plot diagnostics for the Burgers zero adversarial-training run."""

from __future__ import annotations

import argparse
import json
import math
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
    "train": "#111111",
    "test": "#D55E00",
    "near_param_shift": "#0072B2",
    "mid_kernel_spectrum": "#009E73",
    "far_range_pattern": "#CC79A7",
    "target_loss_param_shift": "#E69F00",
    "target_loss_kernel_shift": "#6A3D9A",
}

CHECKPOINT_EPOCHS = [0, 200, 400, 600, 800, 1000]
FINE_CHANGE_EPOCHS = list(range(0, 1001, 50))
DELTA_EPOCHS = [200, 400, 600, 800, 1000]
HIGH_FREQ_TOP_FRACTIONS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]


def setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 220,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def shorten_dataset_id(dataset_id: str) -> str:
    s = dataset_id
    replacements = [
        ("burgers_", ""),
        ("target_", "tgt_"),
        ("gaussian_", "gauss_"),
        ("matern_", "mat_"),
        ("corr", "c"),
        ("kernel_", "ker_"),
        ("param_", "par_"),
        ("loss_", "loss_"),
    ]
    for old, new in replacements:
        s = s.replace(old, new)
    return s[:42]


def sort_eval_rows(df: pd.DataFrame) -> pd.DataFrame:
    tier_rank = {tier: i for i, tier in enumerate(TIER_ORDER)}
    out = df.copy()
    out["_tier_rank"] = out["manual_tier"].map(tier_rank).fillna(999)
    out["_rank"] = pd.to_numeric(out["manual_rank"], errors="coerce").fillna(9999)
    out = out.sort_values(["_tier_rank", "_rank", "dataset_id", "epoch"])
    return out.drop(columns=["_tier_rank", "_rank"])


def metric_limits(df: pd.DataFrame, metric: str) -> tuple[float, float]:
    values = df[metric].to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    lo = float(np.nanmin(finite))
    hi = float(np.nanmax(finite))
    pad = (hi - lo) * 0.05 if hi > lo else max(abs(hi) * 0.05, 1e-6)
    return max(0.0, lo - pad), hi + pad


def plot_all_dataset_lines(eval_df: pd.DataFrame, metric: str, out_path: Path) -> None:
    ylo, yhi = metric_limits(eval_df, metric)
    fig, ax = plt.subplots(figsize=(14, 7.5))
    sorted_df = sort_eval_rows(eval_df)
    for dataset_id, d in sorted_df.groupby("dataset_id", sort=False):
        tier = str(d["manual_tier"].iloc[0])
        color = TIER_COLORS.get(tier, "#666666")
        lw = 1.8 if tier in {"train", "test"} else 0.9
        alpha = 0.95 if tier in {"train", "test"} else 0.58
        ax.plot(d["epoch"], d[metric], color=color, alpha=alpha, lw=lw)

    handles = []
    labels = []
    for tier in TIER_ORDER:
        if tier in set(eval_df["manual_tier"]):
            handles.append(plt.Line2D([0], [0], color=TIER_COLORS[tier], lw=2.5))
            labels.append(tier)
    ax.legend(handles, labels, ncol=4, loc="upper right", frameon=False)
    ax.set_title(f"Burgers adversarial training: {metric} on all 52 datasets")
    ax.set_xlabel("evaluation epoch")
    ax.set_ylabel(metric)
    ax.set_xlim(0, 1000)
    ax.set_ylim(ylo, yhi)
    ax.axvline(200, color="#555555", alpha=0.18, lw=1)
    ax.axvline(400, color="#555555", alpha=0.18, lw=1)
    ax.axvline(600, color="#555555", alpha=0.18, lw=1)
    ax.axvline(800, color="#555555", alpha=0.18, lw=1)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_grouped_dataset_panels(eval_df: pd.DataFrame, metric: str, out_path: Path) -> None:
    ylo, yhi = metric_limits(eval_df, metric)
    present = [tier for tier in TIER_ORDER if tier in set(eval_df["manual_tier"])]
    rows, cols = 3, 3
    fig, axes = plt.subplots(rows, cols, figsize=(15.5, 10.5), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, tier in zip(axes, present):
        d_tier = sort_eval_rows(eval_df[eval_df["manual_tier"] == tier])
        for dataset_id, d in d_tier.groupby("dataset_id", sort=False):
            color = TIER_COLORS.get(tier, "#666666")
            lw = 1.8 if tier in {"train", "test"} else 1.0
            alpha = 0.95 if tier in {"train", "test"} else 0.72
            ax.plot(d["epoch"], d[metric], color=color, alpha=alpha, lw=lw)
        ax.set_title(f"{tier} ({d_tier['dataset_id'].nunique()} datasets)")
        ax.set_xlim(0, 1000)
        ax.set_ylim(ylo, yhi)
        ax.axvline(200, color="#555555", alpha=0.15, lw=1)
        ax.axvline(400, color="#555555", alpha=0.15, lw=1)
        ax.axvline(600, color="#555555", alpha=0.15, lw=1)
        ax.axvline(800, color="#555555", alpha=0.15, lw=1)
    for ax in axes[len(present) :]:
        ax.axis("off")
    fig.supxlabel("evaluation epoch")
    fig.supylabel(metric)
    fig.suptitle(f"{metric} by dataset group, shared y-axis", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_grouped_dataset_panels_distinct_datasets(eval_df: pd.DataFrame, metric: str, out_path: Path) -> None:
    ylo, yhi = metric_limits(eval_df, metric)
    present = [tier for tier in TIER_ORDER if tier in set(eval_df["manual_tier"])]
    rows, cols = 3, 3
    fig, axes = plt.subplots(rows, cols, figsize=(17.2, 11.8), sharex=True, sharey=True)
    axes = axes.ravel()
    cmap_names = ["tab20", "tab20b", "tab20c"]
    for ax, tier in zip(axes, present):
        d_tier = sort_eval_rows(eval_df[eval_df["manual_tier"] == tier])
        datasets = list(d_tier.drop_duplicates("dataset_id")["dataset_id"])
        n = max(len(datasets), 1)
        cmap = plt.get_cmap(cmap_names[present.index(tier) % len(cmap_names)])
        for idx, dataset_id in enumerate(datasets):
            d = d_tier[d_tier["dataset_id"] == dataset_id]
            if tier in {"train", "test"}:
                color = TIER_COLORS.get(tier, "#111111")
                lw = 2.4
                alpha = 0.98
            else:
                color = cmap(idx / max(n - 1, 1))
                lw = 1.45
                alpha = 0.92
            label = f"{idx + 1:02d} {shorten_dataset_id(dataset_id)}"
            ax.plot(d["epoch"], d[metric], color=color, alpha=alpha, lw=lw, label=label)
            # Add a tiny number tag at the final point so dense panels remain traceable.
            end = d.sort_values("epoch").iloc[-1]
            ax.text(
                1004,
                float(end[metric]),
                f"{idx + 1}",
                color=color,
                fontsize=5.6,
                va="center",
                ha="left",
                clip_on=False,
            )
        ax.set_title(f"{tier}: distinct dataset colors ({len(datasets)} datasets)")
        ax.set_xlim(0, 1035)
        ax.set_ylim(ylo, yhi)
        for x in [200, 400, 600, 800]:
            ax.axvline(x, color="#555555", alpha=0.13, lw=1)
        legend_cols = 1 if len(datasets) <= 8 else 2
        ax.legend(
            loc="upper right",
            ncol=legend_cols,
            fontsize=4.8 if len(datasets) > 12 else 5.8,
            handlelength=1.1,
            handletextpad=0.25,
            borderaxespad=0.2,
            frameon=True,
            framealpha=0.68,
            facecolor="white",
            edgecolor="none",
        )
    for ax in axes[len(present) :]:
        ax.axis("off")
    fig.supxlabel("evaluation epoch")
    fig.supylabel(metric)
    fig.suptitle(f"{metric} by dataset group, shared y-axis, distinct color per dataset", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.982])
    fig.savefig(out_path)
    plt.close(fig)


def plot_group_mean_std(eval_df: pd.DataFrame, metric: str, out_path: Path) -> pd.DataFrame:
    grouped = (
        eval_df.groupby(["manual_tier", "epoch"], as_index=False)[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grouped["std"] = grouped["std"].fillna(0.0)
    ylo, yhi = metric_limits(eval_df, metric)
    fig, ax = plt.subplots(figsize=(13, 7))
    for tier in TIER_ORDER:
        d = grouped[grouped["manual_tier"] == tier]
        if d.empty:
            continue
        color = TIER_COLORS.get(tier, "#666666")
        x = d["epoch"].to_numpy(dtype=float)
        mean = d["mean"].to_numpy(dtype=float)
        std = d["std"].to_numpy(dtype=float)
        ax.plot(x, mean, color=color, lw=2.2, label=f"{tier} mean")
        if np.nanmax(std) > 0:
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15, lw=0)
    ax.set_title(f"{metric}: mean with standard-deviation shadow by dataset group")
    ax.set_xlabel("evaluation epoch")
    ax.set_ylabel(metric)
    ax.set_xlim(0, 1000)
    ax.set_ylim(ylo, yhi)
    ax.legend(ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return grouped


def plot_heatmap(eval_df: pd.DataFrame, metric: str, out_path: Path) -> None:
    sorted_df = sort_eval_rows(eval_df)
    dataset_order = sorted_df.drop_duplicates("dataset_id")["dataset_id"].tolist()
    pivot = sorted_df.pivot_table(index="dataset_id", columns="epoch", values=metric, aggfunc="mean")
    pivot = pivot.reindex(dataset_order)
    matrix = pivot.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(15, 11))
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis")
    cbar = fig.colorbar(im, ax=ax, fraction=0.024, pad=0.02)
    cbar.set_label(metric)
    ax.set_title(f"{metric} heatmap across 52 datasets and 1001 evaluation epochs")
    ax.set_xlabel("evaluation epoch")
    ax.set_ylabel("dataset")
    xticks = [0, 200, 400, 600, 800, 1000]
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(x) for x in xticks])
    yticks = np.arange(len(dataset_order))
    ax.set_yticks(yticks)
    ax.set_yticklabels([shorten_dataset_id(x) for x in dataset_order], fontsize=6)
    # Draw tier separators.
    tiers = sorted_df.drop_duplicates("dataset_id")["manual_tier"].tolist()
    for i in range(1, len(tiers)):
        if tiers[i] != tiers[i - 1]:
            ax.axhline(i - 0.5, color="white", lw=1.1, alpha=0.8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_reduction_bars(eval_df: pd.DataFrame, metric: str, out_path: Path) -> pd.DataFrame:
    checkpoints = eval_df[eval_df["epoch"].isin([0, 1000])]
    wide = checkpoints.pivot_table(index="dataset_id", columns="epoch", values=metric, aggfunc="mean")
    meta = eval_df.drop_duplicates("dataset_id").set_index("dataset_id")[["split", "manual_tier", "manual_rank"]]
    wide = wide.join(meta)
    wide = wide.rename(columns={0: "epoch0", 1000: "epoch1000"})
    wide["absolute_drop"] = wide["epoch0"] - wide["epoch1000"]
    wide["relative_drop_fraction"] = wide["absolute_drop"] / wide["epoch0"].replace(0, np.nan)
    wide = wide.sort_values("absolute_drop", ascending=True)

    colors = [TIER_COLORS.get(t, "#666666") for t in wide["manual_tier"]]
    fig, ax = plt.subplots(figsize=(13, 11))
    ax.barh(np.arange(len(wide)), wide["absolute_drop"], color=colors, alpha=0.85)
    ax.axvline(0, color="#333333", lw=1)
    ax.set_yticks(np.arange(len(wide)))
    ax.set_yticklabels([shorten_dataset_id(x) for x in wide.index], fontsize=6)
    ax.set_xlabel(f"epoch0 - epoch1000 {metric}")
    ax.set_title(f"Final loss reduction by dataset: {metric}")
    handles = [
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=TIER_COLORS[t], markersize=8)
        for t in TIER_ORDER
        if t in set(wide["manual_tier"])
    ]
    labels = [t for t in TIER_ORDER if t in set(wide["manual_tier"])]
    ax.legend(handles, labels, ncol=2, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return wide.reset_index()


def plot_checkpoint_change_heatmap(eval_df: pd.DataFrame, metric: str, out_path: Path) -> pd.DataFrame:
    d = sort_eval_rows(eval_df[eval_df["epoch"].isin(CHECKPOINT_EPOCHS)])
    order = d.drop_duplicates("dataset_id")["dataset_id"].tolist()
    pivot = d.pivot_table(index="dataset_id", columns="epoch", values=metric, aggfunc="mean").reindex(order)
    diffs = pivot.diff(axis=1).iloc[:, 1:]
    diffs.columns = [f"{CHECKPOINT_EPOCHS[i-1]}->{CHECKPOINT_EPOCHS[i]}" for i in range(1, len(CHECKPOINT_EPOCHS))]
    max_abs = float(np.nanmax(np.abs(diffs.to_numpy(dtype=float))))
    fig, ax = plt.subplots(figsize=(10, 11))
    im = ax.imshow(diffs.to_numpy(dtype=float), aspect="auto", cmap="RdBu_r", vmin=-max_abs, vmax=max_abs)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label(f"change in {metric}; blue is lower loss")
    ax.set_xticks(np.arange(len(diffs.columns)))
    ax.set_xticklabels(diffs.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([shorten_dataset_id(x) for x in order], fontsize=6)
    ax.set_title(f"Checkpoint-to-checkpoint change in {metric}")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    out = diffs.reset_index()
    return out


def plot_fine_checkpoint_change_heatmap(
    eval_df: pd.DataFrame,
    metric: str,
    out_path: Path,
    epochs: list[int] | None = None,
) -> pd.DataFrame:
    epochs = epochs or FINE_CHANGE_EPOCHS
    d = sort_eval_rows(eval_df[eval_df["epoch"].isin(epochs)])
    order = d.drop_duplicates("dataset_id")["dataset_id"].tolist()
    pivot = d.pivot_table(index="dataset_id", columns="epoch", values=metric, aggfunc="mean").reindex(order)
    pivot = pivot.reindex(columns=epochs)
    diffs = pivot.diff(axis=1).iloc[:, 1:]
    diffs.columns = [f"{epochs[i-1]}->{epochs[i]}" for i in range(1, len(epochs))]
    vals = diffs.to_numpy(dtype=float)
    max_abs = float(np.nanmax(np.abs(vals)))
    fig, ax = plt.subplots(figsize=(16.5, 11.5))
    im = ax.imshow(vals, aspect="auto", cmap="RdBu_r", vmin=-max_abs, vmax=max_abs, interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.018)
    cbar.set_label(f"change in {metric}; blue = lower loss, red = higher loss")
    ax.set_xticks(np.arange(len(diffs.columns)))
    ax.set_xticklabels(diffs.columns, rotation=42, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([shorten_dataset_id(x) for x in order], fontsize=6)
    ax.set_title(f"50-epoch checkpoint-to-checkpoint change in {metric}")
    # Add separators between dataset groups.
    tiers = d.drop_duplicates("dataset_id")["manual_tier"].tolist()
    for i in range(1, len(tiers)):
        if tiers[i] != tiers[i - 1]:
            ax.axhline(i - 0.5, color="#111111", lw=0.75, alpha=0.35)
    for x in [3.5, 7.5, 11.5, 15.5]:
        ax.axvline(x, color="#111111", lw=0.7, alpha=0.18)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return diffs.reset_index()


def plot_attack_losses(attack_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 7))
    specs = [
        ("clean_loss_before_attack_mean", "#0072B2", "clean before attack"),
        ("adv_loss_after_attack_mean", "#D55E00", "adv after attack"),
        ("attack_loss_gain_mean", "#009E73", "attack gain"),
        ("train_loss_used_for_optimizer_updates_mean", "#6A3D9A", "optimizer microbatch loss"),
    ]
    for col, color, label in specs:
        ax.plot(attack_df["epoch"], attack_df[col], color=color, lw=1.6, alpha=0.75, label=label)
        roll = attack_df[col].rolling(25, min_periods=1, center=True).mean()
        ax.plot(attack_df["epoch"], roll, color=color, lw=3.0, alpha=0.95)
    ax.set_yscale("log")
    ax.set_title("Attack loss progress: raw line plus 25-epoch moving average")
    ax.set_xlabel("training epoch")
    ax.set_ylabel("MSE loss, log scale")
    ax.set_xlim(1, 1000)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_attack_relative_gain(attack_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))
    y = attack_df["attack_loss_gain_relative_mean"].astype(float)
    ax.plot(attack_df["epoch"], y, color="#D55E00", lw=1.2, alpha=0.55, label="raw")
    ax.plot(
        attack_df["epoch"],
        y.rolling(25, min_periods=1, center=True).mean(),
        color="#D55E00",
        lw=3,
        label="25-epoch moving average",
    )
    ax.set_title("Relative attack gain over training")
    ax.set_xlabel("training epoch")
    ax.set_ylabel("attack gain / clean loss before attack")
    ax.set_xlim(1, 1000)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_epsilon_bucket_gain(bucket_df: pd.DataFrame, value_col: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 7))
    cmap = plt.get_cmap("tab10")
    for bucket_idx, d in bucket_df.groupby("bucket_index"):
        d = d.sort_values("epoch")
        color = cmap(int(bucket_idx) % 10)
        x = d["epoch"].to_numpy(dtype=float)
        mean = d[f"{value_col}_mean"].to_numpy(dtype=float)
        std_col = f"{value_col}_std"
        std = d[std_col].to_numpy(dtype=float) if std_col in d else np.zeros_like(mean)
        label = f"bucket {bucket_idx}: {d['bucket_value_low'].iloc[0]:.2f}-{d['bucket_value_high'].iloc[0]:.2f}"
        ax.plot(x, mean, color=color, lw=2.0, label=label)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.14, lw=0)
    ax.set_title(f"Epsilon bucket {value_col}, mean with standard-deviation shadow")
    ax.set_xlabel("training epoch")
    ax.set_ylabel(value_col)
    ax.set_xlim(1, 1000)
    ax.legend(ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_probe_metric_trends(probe_df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    metrics = [
        "attack_loss_gain_sample",
        "attack_loss_gain_relative_sample",
        "delta_fft_high_freq_ratio",
        "delta_fft_spectral_centroid",
        "delta_total_variation",
        "delta_sign_change_fraction",
        "delta_l2_rms",
        "delta_abs_mean",
    ]
    grouped = probe_df.groupby("epoch")[metrics].agg(["mean", "std"])
    fig, axes = plt.subplots(4, 2, figsize=(15, 13), sharex=True)
    axes = axes.ravel()
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#6A3D9A", "#56B4E9", "#8C564B"]
    x = grouped.index.to_numpy(dtype=float)
    for ax, metric, color in zip(axes, metrics, colors):
        mean = grouped[(metric, "mean")].to_numpy(dtype=float)
        std = grouped[(metric, "std")].fillna(0.0).to_numpy(dtype=float)
        ax.plot(x, mean, color=color, lw=2.0)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.16, lw=0)
        ax.set_title(metric)
        ax.set_xlim(1, 1000)
    fig.supxlabel("training epoch")
    fig.supylabel("mean across 5 fixed probe samples, shaded +/- std")
    fig.suptitle("Fixed-probe attack delta and frequency metrics", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

    rows = []
    for metric in metrics:
        y = grouped[(metric, "mean")].to_numpy(dtype=float)
        finite = np.isfinite(y)
        xx = x[finite]
        yy = y[finite]
        if len(xx) > 1 and np.nanstd(yy) > 0:
            slope = float(np.polyfit(xx, yy, 1)[0])
            corr = float(np.corrcoef(xx, yy)[0, 1])
        else:
            slope = 0.0
            corr = 0.0
        rows.append(
            {
                "metric": metric,
                "epoch1_mean": float(grouped.loc[1, (metric, "mean")]) if 1 in grouped.index else np.nan,
                "epoch1000_mean": float(grouped.loc[1000, (metric, "mean")]) if 1000 in grouped.index else np.nan,
                "absolute_change": (
                    float(grouped.loc[1000, (metric, "mean")] - grouped.loc[1, (metric, "mean")])
                    if 1 in grouped.index and 1000 in grouped.index
                    else np.nan
                ),
                "linear_slope_per_epoch": slope,
                "pearson_corr_with_epoch": corr,
            }
        )
    return pd.DataFrame(rows)


def fft_power(delta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = np.asarray(delta, dtype=float).reshape(-1)
    d = d - np.mean(d)
    fft = np.fft.rfft(d)
    power = np.abs(fft) ** 2
    freq = np.fft.rfftfreq(d.size, d=1.0 / d.size)
    total = np.sum(power)
    if total > 0:
        power = power / total
    return freq, power


def load_probe_npz(run_dir: Path, epoch: int) -> dict[str, np.ndarray]:
    probe_dir = run_dir / "burgers" / "attack_probe_samples"
    matches = sorted(probe_dir.glob(f"burgers_epoch{epoch:03d}_step*_attack_probe.npz"))
    if not matches:
        raise FileNotFoundError(f"No probe NPZ for epoch {epoch} under {probe_dir}")
    return dict(np.load(matches[0], allow_pickle=True))


def high_frequency_energy_shares(delta: np.ndarray, fractions: list[float]) -> dict[float, float]:
    d = np.asarray(delta, dtype=float).reshape(-1)
    d = d - np.mean(d)
    power = np.abs(np.fft.rfft(d)) ** 2
    non_dc_power = power[1:]
    total = float(np.sum(non_dc_power))
    if total <= 0 or not np.isfinite(total):
        return {fraction: np.nan for fraction in fractions}
    shares = {}
    mode_count = non_dc_power.size
    for fraction in fractions:
        top_count = max(1, int(math.ceil(mode_count * fraction)))
        shares[fraction] = float(np.sum(non_dc_power[-top_count:]) / total)
    return shares


def compute_high_frequency_energy_share_trends(
    run_dir: Path,
    epochs: list[int],
    fractions: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_rows = []
    for epoch in epochs:
        data = load_probe_npz(run_dir, int(epoch))
        deltas = data["delta"]
        probe_ranks = [int(x) for x in data["probe_rank"]]
        source_indices = [int(x) for x in data["source_index"]]
        for sample_idx, (probe_rank, source_index) in enumerate(zip(probe_ranks, source_indices)):
            shares = high_frequency_energy_shares(deltas[sample_idx], fractions)
            for fraction, share in shares.items():
                sample_rows.append(
                    {
                        "epoch": int(epoch),
                        "probe_rank": probe_rank,
                        "source_index": source_index,
                        "top_frequency_fraction": fraction,
                        "top_frequency_percent": fraction * 100.0,
                        "energy_share": share,
                    }
                )
    sample_df = pd.DataFrame(sample_rows)
    summary_df = (
        sample_df.groupby(["epoch", "top_frequency_fraction", "top_frequency_percent"], as_index=False)["energy_share"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )
    summary_df["std"] = summary_df["std"].fillna(0.0)
    return sample_df, summary_df


def plot_high_frequency_energy_share_progress(
    summary_df: pd.DataFrame,
    out_path: Path,
) -> pd.DataFrame:
    colors = ["#0072B2", "#009E73", "#E69F00", "#D55E00", "#6A3D9A", "#CC79A7"]
    fig, ax = plt.subplots(figsize=(13.5, 7.2))
    trend_rows = []
    for color, (fraction, d) in zip(colors, summary_df.groupby("top_frequency_fraction", sort=True)):
        d = d.sort_values("epoch")
        x = d["epoch"].to_numpy(dtype=float)
        mean = d["mean"].to_numpy(dtype=float)
        std = d["std"].to_numpy(dtype=float)
        label = f"top {fraction * 100:g}% highest modes"
        ax.plot(x, mean, color=color, lw=2.2, label=label)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.14, lw=0)
        finite = np.isfinite(mean)
        xx = x[finite]
        yy = mean[finite]
        if len(xx) > 1 and np.nanstd(yy) > 0:
            slope = float(np.polyfit(xx, yy, 1)[0])
            corr = float(np.corrcoef(xx, yy)[0, 1])
        else:
            slope = 0.0
            corr = 0.0
        epoch1 = d[d["epoch"] == 1]["mean"]
        epoch1000 = d[d["epoch"] == 1000]["mean"]
        start = float(epoch1.iloc[0]) if not epoch1.empty else np.nan
        end = float(epoch1000.iloc[0]) if not epoch1000.empty else np.nan
        trend_rows.append(
            {
                "top_frequency_fraction": float(fraction),
                "top_frequency_percent": float(fraction * 100.0),
                "epoch1_mean": start,
                "epoch1000_mean": end,
                "absolute_change": end - start if np.isfinite(start) and np.isfinite(end) else np.nan,
                "linear_slope_per_epoch": slope,
                "pearson_corr_with_epoch": corr,
            }
        )
    ax.set_title("High-frequency energy share of fixed-probe adversarial deltas")
    ax.set_xlabel("training epoch")
    ax.set_ylabel("energy share in highest Fourier modes")
    ax.set_xlim(1, 1000)
    ax.set_ylim(0, 0.05)
    ax.legend(ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return pd.DataFrame(trend_rows)


def plot_delta_checkpoint_grid(run_dir: Path, out_path: Path) -> None:
    data = {epoch: load_probe_npz(run_dir, epoch) for epoch in DELTA_EPOCHS}
    probe_ranks = [int(x) for x in data[DELTA_EPOCHS[0]]["probe_rank"]]
    all_deltas = []
    for epoch in DELTA_EPOCHS:
        all_deltas.append(data[epoch]["delta"].reshape(len(probe_ranks), -1))
    all_deltas_arr = np.concatenate(all_deltas, axis=0)
    ylim = float(np.nanmax(np.abs(all_deltas_arr))) * 1.08
    xgrid = np.linspace(0, 1, all_deltas_arr.shape[1])

    fig, axes = plt.subplots(len(probe_ranks), len(DELTA_EPOCHS), figsize=(16, 10), sharex=True, sharey=True)
    for r_idx, probe_rank in enumerate(probe_ranks):
        for c_idx, epoch in enumerate(DELTA_EPOCHS):
            ax = axes[r_idx, c_idx]
            delta = data[epoch]["delta"][r_idx].reshape(-1)
            ax.plot(xgrid, delta, color="#0072B2", lw=1.0)
            ax.axhline(0, color="#222222", lw=0.6, alpha=0.55)
            ax.set_ylim(-ylim, ylim)
            if r_idx == 0:
                ax.set_title(f"epoch {epoch}")
            if c_idx == 0:
                source_index = int(data[epoch]["source_index"][r_idx])
                ax.set_ylabel(f"probe {probe_rank}\nidx {source_index}")
    fig.supxlabel("spatial coordinate")
    fig.supylabel("delta = x_adv - x_clean")
    fig.suptitle("Fixed-probe delta shapes at saved checkpoints, shared y-axis", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_delta_probe_detail(run_dir: Path, probe_rank: int, out_shape_path: Path, out_fft_path: Path) -> None:
    data = {epoch: load_probe_npz(run_dir, epoch) for epoch in DELTA_EPOCHS}
    ranks = list(map(int, data[DELTA_EPOCHS[0]]["probe_rank"]))
    if probe_rank not in ranks:
        raise ValueError(f"probe_rank {probe_rank} not found; available {ranks}")
    idx = ranks.index(probe_rank)
    all_delta = np.stack([data[e]["delta"][idx].reshape(-1) for e in DELTA_EPOCHS])
    ylim = float(np.nanmax(np.abs(all_delta))) * 1.08
    xgrid = np.linspace(0, 1, all_delta.shape[1])
    colors = ["#0072B2", "#009E73", "#E69F00", "#D55E00", "#6A3D9A"]

    fig, axes = plt.subplots(len(DELTA_EPOCHS), 1, figsize=(14, 9), sharex=True, sharey=True)
    source_index = int(data[DELTA_EPOCHS[0]]["source_index"][idx])
    for ax, epoch, color in zip(axes, DELTA_EPOCHS, colors):
        delta = data[epoch]["delta"][idx].reshape(-1)
        ax.plot(xgrid, delta, color=color, lw=1.1)
        ax.axhline(0, color="#222222", lw=0.6, alpha=0.55)
        ax.set_ylabel(f"ep {epoch}")
        ax.set_ylim(-ylim, ylim)
    fig.supxlabel("spatial coordinate")
    fig.supylabel("delta")
    fig.suptitle(f"Probe {probe_rank}, source index {source_index}: delta at checkpoints", y=0.995)
    fig.tight_layout()
    fig.savefig(out_shape_path)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 7))
    for epoch, color in zip(DELTA_EPOCHS, colors):
        delta = data[epoch]["delta"][idx].reshape(-1)
        freq, power = fft_power(delta)
        ax.plot(freq[1:], power[1:] + 1e-18, color=color, lw=1.8, label=f"epoch {epoch}")
    ax.set_yscale("log")
    ax.set_xlim(1, 512)
    ax.set_title(f"Probe {probe_rank}, source index {source_index}: normalized FFT power of delta")
    ax.set_xlabel("Fourier mode")
    ax.set_ylabel("normalized power, log scale")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_fft_path)
    plt.close(fig)


def rolling_mean_matrix(matrix: np.ndarray, window: int = 25) -> np.ndarray:
    return pd.DataFrame(matrix).rolling(window, min_periods=1, center=True).mean().to_numpy(dtype=float)


def plot_delta_fft_power_moving_average(
    run_dir: Path,
    epochs: list[int],
    out_path: Path,
    out_selected_csv: Path,
    window: int = 25,
) -> None:
    epochs = sorted(int(e) for e in epochs)
    all_power = []
    probe_ranks = None
    freq_modes = None
    for epoch in epochs:
        data = load_probe_npz(run_dir, epoch)
        deltas = data["delta"]
        if probe_ranks is None:
            probe_ranks = [int(x) for x in data["probe_rank"]]
        probe_power = []
        for sample_idx in range(deltas.shape[0]):
            freq, power = fft_power(deltas[sample_idx])
            if freq_modes is None:
                freq_modes = freq[1:]
            probe_power.append(power[1:])
        all_power.append(probe_power)
    power_arr = np.asarray(all_power, dtype=float)  # epoch x probe x Fourier mode
    mean_power = np.nanmean(power_arr, axis=1)
    mean_power_smooth = rolling_mean_matrix(mean_power, window=window)

    if probe_ranks and 0 in probe_ranks:
        probe0_idx = probe_ranks.index(0)
    else:
        probe0_idx = 0
    probe0_power_smooth = rolling_mean_matrix(power_arr[:, probe0_idx, :], window=window)

    selected_modes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512]
    selected_rows = []
    for label, matrix in [("mean_5_fixed_probes", mean_power_smooth), (f"probe_{probe_ranks[probe0_idx] if probe_ranks else probe0_idx}", probe0_power_smooth)]:
        for row_idx, epoch in enumerate(epochs):
            row = {"epoch": epoch, "series": label}
            for mode in selected_modes:
                mode_idx = min(max(mode - 1, 0), matrix.shape[1] - 1)
                row[f"mode_{mode}"] = float(matrix[row_idx, mode_idx])
            selected_rows.append(row)
    pd.DataFrame(selected_rows).to_csv(out_selected_csv, index=False)

    log_power = np.log10(np.clip(mean_power_smooth, 1e-18, None))
    finite = log_power[np.isfinite(log_power)]
    vmin, vmax = np.nanpercentile(finite, [2, 99.5])

    fig = plt.figure(figsize=(15.5, 10.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.35, 0.9], hspace=0.28)
    ax0 = fig.add_subplot(gs[0, 0])
    im = ax0.imshow(
        log_power,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[float(freq_modes[0]), float(freq_modes[-1]), float(epochs[0]), float(epochs[-1])],
        cmap="magma",
        vmin=float(vmin),
        vmax=float(vmax),
    )
    cbar = fig.colorbar(im, ax=ax0, fraction=0.023, pad=0.018)
    cbar.set_label("log10 normalized FFT power")
    ax0.set_title(f"Delta normalized FFT power, {window}-epoch moving average, mean across 5 fixed probes")
    ax0.set_xlabel("Fourier mode")
    ax0.set_ylabel("training epoch")
    ax0.set_xlim(1, 512)

    ax1 = fig.add_subplot(gs[1, 0])
    selected_epochs = [50, 200, 400, 600, 800, 1000]
    colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.95, len(selected_epochs)))
    for epoch, color in zip(selected_epochs, colors):
        idx = int(np.argmin(np.abs(np.asarray(epochs) - epoch)))
        ax1.plot(freq_modes, mean_power_smooth[idx] + 1e-18, color=color, lw=1.8, label=f"epoch {epochs[idx]}")
    ax1.set_yscale("log")
    ax1.set_xlim(1, 512)
    ax1.set_title("Smoothed spectra at selected epochs, same 25-epoch moving average")
    ax1.set_xlabel("Fourier mode")
    ax1.set_ylabel("normalized power, log scale")
    ax1.legend(ncol=3, frameon=False)

    fig.suptitle("Moving-average view of fixed-probe delta FFT power", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out_path)
    plt.close(fig)


def write_summary(
    out_dir: Path,
    reduction_rel: pd.DataFrame,
    reduction_rmse: pd.DataFrame,
    probe_trends: pd.DataFrame,
    high_freq_band_trends: pd.DataFrame,
    final_split: pd.DataFrame,
) -> None:
    def markdown_table(frame: pd.DataFrame) -> str:
        headers = [str(col) for col in frame.columns]
        rows = []
        for _, row in frame.iterrows():
            rendered = []
            for value in row:
                if isinstance(value, float):
                    rendered.append(f"{value:.6g}")
                else:
                    rendered.append(str(value))
            rows.append(rendered)
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        lines.extend("| " + " | ".join(row) + " |" for row in rows)
        return "\n".join(lines)

    rel_mean_drop = float(reduction_rel["absolute_drop"].mean())
    rel_mean_drop_frac = float(reduction_rel["relative_drop_fraction"].mean())
    rmse_mean_drop = float(reduction_rmse["absolute_drop"].mean())
    high_freq = probe_trends[probe_trends["metric"] == "delta_fft_high_freq_ratio"].iloc[0]
    attack_gain = probe_trends[probe_trends["metric"] == "attack_loss_gain_sample"].iloc[0]
    top_1 = high_freq_band_trends[high_freq_band_trends["top_frequency_fraction"] == 0.01].iloc[0]
    top_10 = high_freq_band_trends[high_freq_band_trends["top_frequency_fraction"] == 0.10].iloc[0]
    top_50 = high_freq_band_trends[high_freq_band_trends["top_frequency_fraction"] == 0.50].iloc[0]
    lines = [
        "# Burgers zero adversarial-training visualization summary",
        "",
        "Generated from:",
        "",
        "```text",
        "adversarial_training_runs/burgers_zero_advonly_random_jitter_1000ep_bs480_steps5_eps5bucket_20260601",
        "```",
        "",
        "## Final split metrics at epoch 1000",
        "",
        markdown_table(final_split),
        "",
        "## Loss reduction summary",
        "",
        f"- Mean absolute Relative L2 drop across 52 datasets: `{rel_mean_drop:.6g}`.",
        f"- Mean fractional Relative L2 drop across 52 datasets: `{rel_mean_drop_frac:.3%}`.",
        f"- Mean absolute RMSE drop across 52 datasets: `{rmse_mean_drop:.6g}`.",
        "",
        "## Fixed-probe attack/frequency trend",
        "",
        f"- Mean fixed-probe attack gain changed from `{attack_gain['epoch1_mean']:.6g}` to `{attack_gain['epoch1000_mean']:.6g}`.",
        f"- Mean fixed-probe high-frequency ratio changed from `{high_freq['epoch1_mean']:.6g}` to `{high_freq['epoch1000_mean']:.6g}`.",
        f"- High-frequency ratio slope per epoch: `{high_freq['linear_slope_per_epoch']:.6g}`.",
        f"- High-frequency ratio Pearson correlation with epoch: `{high_freq['pearson_corr_with_epoch']:.6g}`.",
        f"- Top 1% highest-mode energy share changed from `{top_1['epoch1_mean']:.6g}` to `{top_1['epoch1000_mean']:.6g}`.",
        f"- Top 10% highest-mode energy share changed from `{top_10['epoch1_mean']:.6g}` to `{top_10['epoch1000_mean']:.6g}`.",
        f"- Top 50% highest-mode energy share changed from `{top_50['epoch1_mean']:.6g}` to `{top_50['epoch1000_mean']:.6g}`.",
        "",
        "## Main figures",
        "",
        "- `relative_l2_all_52_datasets.png`",
        "- `rmse_all_52_datasets.png`",
        "- `relative_l2_grouped_shared_y.png`",
        "- `rmse_grouped_shared_y.png`",
        "- `relative_l2_grouped_shared_y_distinct_datasets.png`",
        "- `rmse_grouped_shared_y_distinct_datasets.png`",
        "- `relative_l2_group_mean_std.png`",
        "- `rmse_group_mean_std.png`",
        "- `relative_l2_heatmap_52_datasets.png`",
        "- `relative_l2_reduction_by_dataset.png`",
        "- `relative_l2_checkpoint_change_heatmap.png`",
        "- `relative_l2_checkpoint_change_heatmap_50epoch.png`",
        "- `rmse_checkpoint_change_heatmap_50epoch.png`",
        "- `attack_losses_progress.png`",
        "- `attack_relative_gain_progress.png`",
        "- `epsilon_bucket_attack_loss_gain.png`",
        "- `epsilon_bucket_attack_loss_gain_relative.png`",
        "- `fixed_probe_delta_frequency_metrics.png`",
        "- `delta_high_frequency_energy_share_progress.png`",
        "- `delta_checkpoint_shapes_all_probes.png`",
        "- `delta_checkpoint_shapes_probe0.png`",
        "- `delta_checkpoint_fft_probe0.png`",
        "- `delta_fft_power_25epoch_moving_average.png`",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("adversarial_training_runs/burgers_zero_advonly_random_jitter_1000ep_bs480_steps5_eps5bucket_20260601"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("visualizations/burgers_zero_adv_training_20260601"),
    )
    args = parser.parse_args()

    setup_matplotlib()
    run_dir: Path = args.run_dir
    out_dir: Path = args.out_dir
    ensure_dir(out_dir)

    burgers_dir = run_dir / "burgers"
    eval_df = pd.read_csv(burgers_dir / "eval_metrics.csv")
    eval_df = eval_df.sort_values(["epoch", "split", "manual_tier", "dataset_id"])
    attack_df = pd.read_csv(burgers_dir / "attack_epoch_summary.csv")
    bucket_df = pd.read_csv(burgers_dir / "attack_epsilon_bucket_summary.csv")
    probe_df = pd.read_csv(burgers_dir / "attack_probe_samples.csv")
    split_df = pd.read_csv(burgers_dir / "eval_split_summary.csv")

    # Core 52-dataset visualizations.
    plot_all_dataset_lines(eval_df, "relative_l2", out_dir / "relative_l2_all_52_datasets.png")
    plot_all_dataset_lines(eval_df, "rmse", out_dir / "rmse_all_52_datasets.png")
    plot_grouped_dataset_panels(eval_df, "relative_l2", out_dir / "relative_l2_grouped_shared_y.png")
    plot_grouped_dataset_panels(eval_df, "rmse", out_dir / "rmse_grouped_shared_y.png")
    plot_grouped_dataset_panels_distinct_datasets(
        eval_df, "relative_l2", out_dir / "relative_l2_grouped_shared_y_distinct_datasets.png"
    )
    plot_grouped_dataset_panels_distinct_datasets(
        eval_df, "rmse", out_dir / "rmse_grouped_shared_y_distinct_datasets.png"
    )
    rel_group = plot_group_mean_std(eval_df, "relative_l2", out_dir / "relative_l2_group_mean_std.png")
    rmse_group = plot_group_mean_std(eval_df, "rmse", out_dir / "rmse_group_mean_std.png")
    rel_group.to_csv(out_dir / "relative_l2_group_mean_std.csv", index=False)
    rmse_group.to_csv(out_dir / "rmse_group_mean_std.csv", index=False)
    plot_heatmap(eval_df, "relative_l2", out_dir / "relative_l2_heatmap_52_datasets.png")
    reduction_rel = plot_reduction_bars(eval_df, "relative_l2", out_dir / "relative_l2_reduction_by_dataset.png")
    reduction_rmse = plot_reduction_bars(eval_df, "rmse", out_dir / "rmse_reduction_by_dataset.png")
    reduction_rel.to_csv(out_dir / "relative_l2_reduction_by_dataset.csv", index=False)
    reduction_rmse.to_csv(out_dir / "rmse_reduction_by_dataset.csv", index=False)
    checkpoint_change = plot_checkpoint_change_heatmap(
        eval_df, "relative_l2", out_dir / "relative_l2_checkpoint_change_heatmap.png"
    )
    checkpoint_change.to_csv(out_dir / "relative_l2_checkpoint_change_by_dataset.csv", index=False)
    fine_checkpoint_rel = plot_fine_checkpoint_change_heatmap(
        eval_df, "relative_l2", out_dir / "relative_l2_checkpoint_change_heatmap_50epoch.png"
    )
    fine_checkpoint_rmse = plot_fine_checkpoint_change_heatmap(
        eval_df, "rmse", out_dir / "rmse_checkpoint_change_heatmap_50epoch.png"
    )
    fine_checkpoint_rel.to_csv(out_dir / "relative_l2_checkpoint_change_by_dataset_50epoch.csv", index=False)
    fine_checkpoint_rmse.to_csv(out_dir / "rmse_checkpoint_change_by_dataset_50epoch.csv", index=False)

    # Attack and epsilon-bucket visualizations.
    plot_attack_losses(attack_df, out_dir / "attack_losses_progress.png")
    plot_attack_relative_gain(attack_df, out_dir / "attack_relative_gain_progress.png")
    plot_epsilon_bucket_gain(bucket_df, "attack_loss_gain", out_dir / "epsilon_bucket_attack_loss_gain.png")
    plot_epsilon_bucket_gain(
        bucket_df, "attack_loss_gain_relative", out_dir / "epsilon_bucket_attack_loss_gain_relative.png"
    )

    # Fixed-probe delta and frequency visualizations.
    probe_trends = plot_probe_metric_trends(probe_df, out_dir / "fixed_probe_delta_frequency_metrics.png")
    probe_trends.to_csv(out_dir / "fixed_probe_delta_frequency_trend_summary.csv", index=False)
    high_freq_sample_shares, high_freq_band_summary = compute_high_frequency_energy_share_trends(
        run_dir, sorted(probe_df["epoch"].unique()), HIGH_FREQ_TOP_FRACTIONS
    )
    high_freq_sample_shares.to_csv(out_dir / "delta_high_frequency_energy_share_samples.csv", index=False)
    high_freq_band_summary.to_csv(out_dir / "delta_high_frequency_energy_share_by_epoch.csv", index=False)
    high_freq_band_trends = plot_high_frequency_energy_share_progress(
        high_freq_band_summary, out_dir / "delta_high_frequency_energy_share_progress.png"
    )
    high_freq_band_trends.to_csv(out_dir / "delta_high_frequency_energy_share_trend_summary.csv", index=False)
    plot_delta_checkpoint_grid(run_dir, out_dir / "delta_checkpoint_shapes_all_probes.png")
    plot_delta_probe_detail(
        run_dir,
        probe_rank=0,
        out_shape_path=out_dir / "delta_checkpoint_shapes_probe0.png",
        out_fft_path=out_dir / "delta_checkpoint_fft_probe0.png",
    )
    plot_delta_fft_power_moving_average(
        run_dir,
        sorted(probe_df["epoch"].unique()),
        out_path=out_dir / "delta_fft_power_25epoch_moving_average.png",
        out_selected_csv=out_dir / "delta_fft_power_25epoch_moving_average_selected_modes.csv",
        window=25,
    )

    final_split = split_df[split_df["epoch"] == 1000].copy()
    final_split.to_csv(out_dir / "epoch1000_split_summary.csv", index=False)
    write_summary(out_dir, reduction_rel, reduction_rmse, probe_trends, high_freq_band_trends, final_split)

    manifest = {
        "run_dir": str(run_dir),
        "out_dir": str(out_dir),
        "figure_count": len(list(out_dir.glob("*.png"))),
        "csv_count": len(list(out_dir.glob("*.csv"))),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

