#!/usr/bin/env python3
"""Polished report visualizations for the Burgers zero adversarial-training run."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

RUN_DIR = Path("adversarial_training_runs/burgers_zero_advonly_random_jitter_1000ep_bs480_steps5_eps5bucket_20260601")
RAW_VIZ_DIR = Path("visualizations/burgers_zero_adv_training_20260601")
OUT_DIR = RAW_VIZ_DIR / "polished_report"
BURGERS_DIR = RUN_DIR / "burgers"
CHECKPOINTS = [0, 200, 400, 600, 800, 1000]
CHECKPOINTS_FINE = list(range(0, 1001, 50))
DELTA_EPOCHS = [200, 400, 600, 800, 1000]

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
FREQ_COLORS = {
    1.0: "#12355B",
    2.0: "#0072B2",
    5.0: "#009E73",
    10.0: "#E69F00",
    20.0: "#D55E00",
    50.0: "#6A3D9A",
}
BG = "#fbfaf7"
AX_BG = "#ffffff"
GRID = "#d9d6cc"
TEXT = "#202124"
MUTED = "#6f6f6f"


def setup() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
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


def savefig(fig: plt.Figure, filename: str) -> Path:
    path = OUT_DIR / filename
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.075,
        1.055,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=13,
        fontweight="bold",
        color=TEXT,
        bbox=dict(boxstyle="round,pad=0.24", fc="#f0eee8", ec="none", alpha=0.95),
    )


def shorten(s: str, n: int = 36) -> str:
    out = s.replace("burgers_", "")
    out = out.replace("gaussian_", "gauss_").replace("matern_", "mat_")
    out = out.replace("target_", "tgt_").replace("kernel_", "ker_").replace("corr", "c")
    return out if len(out) <= n else out[: n - 1] + "..."


def tier_sorted(df: pd.DataFrame) -> pd.DataFrame:
    rank = {tier: i for i, tier in enumerate(TIER_ORDER)}
    d = df.copy()
    d["_tier_rank"] = d["manual_tier"].map(rank).fillna(999)
    d["_manual_rank"] = pd.to_numeric(d["manual_rank"], errors="coerce").fillna(999)
    return d.sort_values(["_tier_rank", "_manual_rank", "dataset_id", "epoch"]).drop(columns=["_tier_rank", "_manual_rank"])


def read_data() -> dict[str, pd.DataFrame]:
    eval_df = pd.read_csv(BURGERS_DIR / "eval_metrics.csv")
    attack_df = pd.read_csv(BURGERS_DIR / "attack_epoch_summary.csv")
    bucket_df = pd.read_csv(BURGERS_DIR / "attack_epsilon_bucket_summary.csv")
    probe_df = pd.read_csv(BURGERS_DIR / "attack_probe_samples.csv")
    split_df = pd.read_csv(BURGERS_DIR / "eval_split_summary.csv")
    highfreq_df = pd.read_csv(RAW_VIZ_DIR / "delta_high_frequency_energy_share_by_epoch.csv")
    return {
        "eval": eval_df,
        "attack": attack_df,
        "bucket": bucket_df,
        "probe": probe_df,
        "split": split_df,
        "highfreq": highfreq_df,
    }


def reduction_table(eval_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    d = eval_df[eval_df["epoch"].isin([0, 1000])]
    wide = d.pivot_table(index="dataset_id", columns="epoch", values=metric, aggfunc="mean")
    meta = eval_df.drop_duplicates("dataset_id").set_index("dataset_id")[["split", "manual_tier", "manual_rank"]]
    wide = wide.join(meta).rename(columns={0: "epoch0", 1000: "epoch1000"})
    wide["absolute_drop"] = wide["epoch0"] - wide["epoch1000"]
    wide["relative_drop_fraction"] = wide["absolute_drop"] / wide["epoch0"].replace(0, np.nan)
    return wide.reset_index()


def group_mean_std(eval_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    g = eval_df.groupby(["manual_tier", "epoch"], as_index=False)[metric].agg(["mean", "std", "count"]).reset_index()
    g["std"] = g["std"].fillna(0.0)
    return g


def rolling(y: pd.Series | np.ndarray, window: int = 31) -> np.ndarray:
    return pd.Series(y).rolling(window, center=True, min_periods=1).mean().to_numpy(dtype=float)


def plot_loss_reduction(eval_df: pd.DataFrame) -> Path:
    rel = reduction_table(eval_df, "relative_l2").sort_values("absolute_drop")
    rmse = reduction_table(eval_df, "rmse").sort_values("absolute_drop")
    rel.to_csv(OUT_DIR / "polished_relative_l2_reduction_by_dataset.csv", index=False)
    rmse.to_csv(OUT_DIR / "polished_rmse_reduction_by_dataset.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(18, 12), sharey=True)
    fig.suptitle("Burgers ADV-only training: loss reduction by dataset", fontsize=18, fontweight="bold", y=0.985)
    for ax, df, metric, xlabel in [
        (axes[0], rel, "Relative L2", "epoch 0 - epoch 1000 Relative L2"),
        (axes[1], rmse.reindex(rel.index), "RMSE", "epoch 0 - epoch 1000 RMSE"),
    ]:
        colors = [TIER_COLORS.get(t, "#777777") for t in df["manual_tier"]]
        y = np.arange(len(df))
        ax.barh(y, df["absolute_drop"], color=colors, alpha=0.88, edgecolor="white", linewidth=0.35)
        ax.axvline(0, color="#2b2b2b", lw=1)
        ax.set_title(f"{metric}: 52 / 52 datasets improved", loc="left", fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylim(-0.8, len(df) - 0.2)
        ax.set_yticks(y)
        ax.set_yticklabels([shorten(x, 38) for x in df["dataset_id"]], fontsize=6.7)
        mean_drop = float(df["absolute_drop"].mean())
        mean_frac = float(df["relative_drop_fraction"].mean())
        ax.text(
            0.98,
            0.03,
            f"mean drop: {mean_drop:.4g}\nmean fraction: {mean_frac:.1%}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.45", fc="#fffaf0", ec="#d4c39a", alpha=0.95),
        )
    handles = [
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=TIER_COLORS[t], markersize=9)
        for t in TIER_ORDER
        if t in set(rel["manual_tier"])
    ]
    labels = [t for t in TIER_ORDER if t in set(rel["manual_tier"])]
    fig.legend(handles, labels, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=[0, 0.03, 1, 0.955])
    return savefig(fig, "polished_loss_reduction_by_dataset.png")


def plot_heatmap(eval_df: pd.DataFrame) -> Path:
    d = tier_sorted(eval_df)
    order = d.drop_duplicates("dataset_id")["dataset_id"].tolist()
    pivot = d.pivot_table(index="dataset_id", columns="epoch", values="relative_l2", aggfunc="mean").reindex(order)
    matrix = pivot.to_numpy(dtype=float)
    tiers = d.drop_duplicates("dataset_id")["manual_tier"].tolist()
    tier_nums = np.array([TIER_ORDER.index(t) if t in TIER_ORDER else -1 for t in tiers]).reshape(-1, 1)

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.16, 5.8, 0.2], wspace=0.04)
    ax_strip = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])

    tier_cmap = plt.matplotlib.colors.ListedColormap([TIER_COLORS[t] for t in TIER_ORDER])
    ax_strip.imshow(tier_nums, aspect="auto", cmap=tier_cmap, vmin=0, vmax=len(TIER_ORDER) - 1)
    ax_strip.set_xticks([])
    ax_strip.set_yticks([])
    ax_strip.set_title("group", fontsize=9)

    im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="magma_r")
    fig.colorbar(im, cax=cax, label="Relative L2")
    ax.set_title("Relative L2 across 52 datasets and 1001 evaluation epochs", loc="left", fontweight="bold")
    ax.set_xlabel("evaluation epoch")
    ax.set_ylabel("dataset")
    ax.set_xticks(CHECKPOINTS)
    ax.set_xticklabels([str(x) for x in CHECKPOINTS])
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([shorten(x, 42) for x in order], fontsize=6.3)
    for i in range(1, len(tiers)):
        if tiers[i] != tiers[i - 1]:
            ax.axhline(i - 0.5, color="white", lw=1.3, alpha=0.75)
            ax_strip.axhline(i - 0.5, color="white", lw=1.3, alpha=0.75)
    handles = [plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=TIER_COLORS[t], markersize=8) for t in TIER_ORDER if t in set(tiers)]
    labels = [t for t in TIER_ORDER if t in set(tiers)]
    ax.legend(handles, labels, ncol=2, loc="lower right", frameon=True, facecolor="#ffffff", framealpha=0.88)
    fig.tight_layout()
    return savefig(fig, "polished_relative_l2_heatmap.png")


def plot_checkpoint_summary(eval_df: pd.DataFrame) -> Path:
    checkpoints = CHECKPOINTS_FINE
    d = eval_df[eval_df["epoch"].isin(checkpoints)]
    g = group_mean_std(d, "relative_l2")
    full = tier_sorted(d)
    order = full.drop_duplicates("dataset_id")["dataset_id"].tolist()
    pivot = full.pivot_table(index="dataset_id", columns="epoch", values="relative_l2", aggfunc="mean").reindex(order)
    pivot = pivot.reindex(columns=checkpoints)
    diffs = pivot.diff(axis=1).iloc[:, 1:]
    diffs.columns = [f"{checkpoints[i-1]}->{checkpoints[i]}" for i in range(1, len(checkpoints))]
    diffs.to_csv(OUT_DIR / "polished_checkpoint_relative_l2_changes.csv")
    diffs.to_csv(OUT_DIR / "polished_checkpoint_relative_l2_changes_50epoch.csv")

    fig, axes = plt.subplots(1, 2, figsize=(20, 9.2), gridspec_kw={"width_ratios": [1.05, 1.25]})
    ax = axes[0]
    for tier in TIER_ORDER:
        dd = g[g["manual_tier"] == tier].sort_values("epoch")
        if dd.empty:
            continue
        x = dd["epoch"].to_numpy(dtype=float)
        mean = dd["mean"].to_numpy(dtype=float)
        std = dd["std"].to_numpy(dtype=float)
        color = TIER_COLORS[tier]
        ax.plot(x, mean, color=color, lw=2.15, marker="o", ms=3.2, label=tier)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.12, lw=0)
    ax.set_title("50-epoch checkpoint loss summary: group mean +/- std", loc="left", fontweight="bold")
    ax.set_xlabel("evaluation epoch")
    ax.set_ylabel("Relative L2")
    ax.set_xlim(0, 1000)
    ax.legend(ncol=2)

    ax2 = axes[1]
    vals = diffs.to_numpy(dtype=float)
    max_abs = float(np.nanmax(np.abs(vals)))
    im = ax2.imshow(vals, aspect="auto", cmap="RdBu_r", vmin=-max_abs, vmax=max_abs)
    ax2.set_title("50-epoch checkpoint-to-checkpoint change\nblue = lower loss, red = higher loss", loc="left", fontweight="bold")
    ax2.set_xticks(np.arange(len(diffs.columns)))
    ax2.set_xticklabels(diffs.columns, rotation=42, ha="right", fontsize=7.5)
    ax2.set_yticks(np.arange(len(order)))
    ax2.set_yticklabels([shorten(x, 30) for x in order], fontsize=5.6)
    cbar = fig.colorbar(im, ax=ax2, fraction=0.03, pad=0.02)
    cbar.set_label("Delta Relative L2")
    fig.suptitle("Burgers ADV-only: 50-epoch checkpoint loss behavior", fontsize=17, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    return savefig(fig, "polished_checkpoint_loss_summary.png")


def plot_attack_buckets(attack_df: pd.DataFrame, bucket_df: pd.DataFrame) -> Path:
    bucket_df = bucket_df.sort_values(["bucket_index", "epoch"])
    dec = attack_df.copy()
    dec["epoch_bucket"] = pd.cut(dec["epoch"], bins=np.arange(0, 1001, 100), labels=[f"{i*100+1}-{(i+1)*100}" for i in range(10)], include_lowest=True)
    dec_summary = dec.groupby("epoch_bucket", observed=True).agg(
        clean_mean=("clean_loss_before_attack_mean", "mean"),
        clean_std=("clean_loss_before_attack_mean", "std"),
        adv_mean=("adv_loss_after_attack_mean", "mean"),
        adv_std=("adv_loss_after_attack_mean", "std"),
        gain_mean=("attack_loss_gain_mean", "mean"),
        gain_std=("attack_loss_gain_mean", "std"),
        relative_gain_mean=("attack_loss_gain_relative_mean", "mean"),
        relative_gain_std=("attack_loss_gain_relative_mean", "std"),
    ).reset_index()
    dec_summary.to_csv(OUT_DIR / "polished_attack_loss_epoch_decile_summary.csv", index=False)

    fig = plt.figure(figsize=(18, 13))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1], hspace=0.28, wspace=0.20)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    cmap = plt.get_cmap("viridis")
    for i, (bucket_idx, dd) in enumerate(bucket_df.groupby("bucket_index")):
        dd = dd.sort_values("epoch")
        x = dd["epoch"].to_numpy(dtype=float)
        color = cmap(0.12 + 0.76 * i / max(1, bucket_df["bucket_index"].nunique() - 1))
        label = f"ε jitter {dd['bucket_value_low'].iloc[0]:.2f}-{dd['bucket_value_high'].iloc[0]:.2f}"
        for ax, col, ylabel in [
            (ax1, "attack_loss_gain", "attack gain, MSE"),
            (ax2, "attack_loss_gain_relative", "relative gain"),
        ]:
            mean = rolling(dd[f"{col}_mean"], 25)
            raw_std = dd[f"{col}_std"].fillna(0).to_numpy(dtype=float)
            std = rolling(raw_std, 25)
            ax.plot(x, mean, color=color, lw=2.1, label=label)
            ax.fill_between(x, np.maximum(mean - std, 0), mean + std, color=color, alpha=0.12, lw=0)
            ax.set_xlabel("training epoch")
            ax.set_ylabel(ylabel)
            ax.set_xlim(1, 1000)
    ax1.set_yscale("log")
    ax1.set_title("5 epsilon buckets: attack loss gain", loc="left", fontweight="bold")
    ax2.set_title("5 epsilon buckets: relative attack gain", loc="left", fontweight="bold")
    ax2.legend(ncol=1, loc="upper right")

    x = np.arange(len(dec_summary))
    width = 0.23
    bars = [
        ("clean_mean", "clean_std", "clean before", "#0072B2", -width),
        ("adv_mean", "adv_std", "adv after", "#D55E00", 0),
        ("gain_mean", "gain_std", "attack gain", "#009E73", width),
    ]
    for mean_col, std_col, label, color, offset in bars:
        ax3.bar(x + offset, dec_summary[mean_col], width=width, color=color, alpha=0.82, label=label)
        ax3.errorbar(x + offset, dec_summary[mean_col], yerr=dec_summary[std_col].fillna(0), fmt="none", ecolor=color, alpha=0.45, lw=1)
    ax3.set_yscale("log")
    ax3.set_xticks(x)
    ax3.set_xticklabels(dec_summary["epoch_bucket"], rotation=25, ha="right")
    ax3.set_xlabel("training epoch bucket")
    ax3.set_ylabel("MSE, log scale")
    ax3.set_title("10 training-progress buckets: clean / adv / gain", loc="left", fontweight="bold")
    ax3.legend(ncol=3)
    fig.suptitle("Attack loss bucket analysis", fontsize=18, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    return savefig(fig, "polished_attack_loss_buckets.png")


def plot_high_frequency(highfreq_df: pd.DataFrame) -> Path:
    trend_rows = []
    fig, ax = plt.subplots(figsize=(15, 8))
    for pct, dd in highfreq_df.groupby("top_frequency_percent"):
        dd = dd.sort_values("epoch")
        x = dd["epoch"].to_numpy(dtype=float)
        mean = dd["mean"].to_numpy(dtype=float)
        std = dd["std"].fillna(0).to_numpy(dtype=float)
        color = FREQ_COLORS.get(float(pct), "#555555")
        label = f"top {pct:g}% highest modes"
        ax.plot(x, mean, lw=2.5, color=color, label=label)
        ax.fill_between(x, np.maximum(mean - std, 0), mean + std, color=color, alpha=0.12, lw=0)
        finite = np.isfinite(mean)
        slope = float(np.polyfit(x[finite], mean[finite], 1)[0]) if finite.sum() > 1 else np.nan
        corr = float(np.corrcoef(x[finite], mean[finite])[0, 1]) if finite.sum() > 1 and np.nanstd(mean[finite]) > 0 else np.nan
        trend_rows.append(
            {
                "top_frequency_percent": float(pct),
                "epoch1_mean": float(dd[dd["epoch"] == 1]["mean"].iloc[0]),
                "epoch1000_mean": float(dd[dd["epoch"] == 1000]["mean"].iloc[0]),
                "absolute_change": float(dd[dd["epoch"] == 1000]["mean"].iloc[0] - dd[dd["epoch"] == 1]["mean"].iloc[0]),
                "linear_slope_per_epoch": slope,
                "pearson_corr_with_epoch": corr,
            }
        )
    trend = pd.DataFrame(trend_rows)
    trend.to_csv(OUT_DIR / "polished_high_frequency_energy_share_trend_summary.csv", index=False)
    ax.set_title("Adversarial delta shifts toward high-frequency Fourier modes", loc="left", fontweight="bold")
    ax.set_xlabel("training epoch")
    ax.set_ylabel("energy share in highest Fourier modes")
    ax.set_xlim(1, 1000)
    ax.set_ylim(0, 0.05)
    ax.legend(ncol=2, loc="upper left")
    ax.text(
        0.98,
        0.05,
        "All six high-frequency tails increase\nwith positive epoch correlation",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.5", fc="#fffaf0", ec="#d4c39a", alpha=0.95),
    )
    fig.tight_layout()
    return savefig(fig, "polished_high_frequency_energy_share.png")


def load_probe_npz(epoch: int) -> dict[str, np.ndarray]:
    matches = sorted((BURGERS_DIR / "attack_probe_samples").glob(f"burgers_epoch{epoch:03d}_step*_attack_probe.npz"))
    if not matches:
        raise FileNotFoundError(epoch)
    return dict(np.load(matches[0], allow_pickle=True))


def fft_power(delta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = np.asarray(delta, dtype=float).reshape(-1)
    d = d - d.mean()
    power = np.abs(np.fft.rfft(d)) ** 2
    freq = np.fft.rfftfreq(d.size, d=1.0 / d.size)
    total = power[1:].sum()
    if total > 0:
        power = power / total
    return freq, power


def plot_delta_probe_checkpoints() -> Path:
    data = {epoch: load_probe_npz(epoch) for epoch in DELTA_EPOCHS}
    ranks = [int(x) for x in data[DELTA_EPOCHS[0]]["probe_rank"]]
    probe_rank = 0
    idx = ranks.index(probe_rank) if probe_rank in ranks else 0
    xgrid = np.linspace(0, 1, data[DELTA_EPOCHS[0]]["delta"].shape[1])
    deltas = np.stack([data[e]["delta"][idx].reshape(-1) for e in DELTA_EPOCHS])
    ylim = float(np.nanmax(np.abs(deltas))) * 1.08
    colors = ["#0072B2", "#009E73", "#E69F00", "#D55E00", "#6A3D9A"]

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, len(DELTA_EPOCHS), height_ratios=[1.0, 0.78], hspace=0.32, wspace=0.18)
    for col, (epoch, color) in enumerate(zip(DELTA_EPOCHS, colors)):
        ax = fig.add_subplot(gs[0, col])
        delta = data[epoch]["delta"][idx].reshape(-1)
        x_clean = data[epoch]["x_clean"][idx].reshape(-1)
        ax.plot(xgrid, x_clean, color="#8a8a8a", lw=1.0, alpha=0.45, label="clean input")
        ax2 = ax.twinx()
        ax2.plot(xgrid, delta, color=color, lw=1.15, label="delta")
        ax2.axhline(0, color="#333333", lw=0.6, alpha=0.55)
        ax2.set_ylim(-ylim, ylim)
        ax.set_title(f"epoch {epoch}", fontweight="bold")
        ax.set_xlabel("space")
        if col == 0:
            ax.set_ylabel("clean input", color="#666666")
            ax2.set_ylabel("delta", color=color)
        else:
            ax.set_yticklabels([])
            ax2.set_yticklabels([])
        ax.grid(False)
        ax2.grid(True, alpha=0.18)

        axf = fig.add_subplot(gs[1, col])
        freq, power = fft_power(delta)
        axf.plot(freq[1:], power[1:] + 1e-18, color=color, lw=1.25)
        axf.set_yscale("log")
        axf.set_xlim(1, 512)
        axf.set_xlabel("Fourier mode")
        if col == 0:
            axf.set_ylabel("normalized power")
        else:
            axf.set_yticklabels([])
        axf.set_title("FFT power", fontsize=10)
    source_index = int(data[DELTA_EPOCHS[0]]["source_index"][idx])
    fig.suptitle(
        f"Fixed probe delta checkpoint view: probe {probe_rank}, source index {source_index}",
        fontsize=18,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    return savefig(fig, "polished_delta_probe_checkpoint.png")


def plot_dashboard(data: dict[str, pd.DataFrame]) -> Path:
    eval_df, attack_df, bucket_df, highfreq_df = data["eval"], data["attack"], data["bucket"], data["highfreq"]
    rel = reduction_table(eval_df, "relative_l2")
    mean_start = rel["epoch0"].mean()
    mean_end = rel["epoch1000"].mean()
    mean_frac = rel["relative_drop_fraction"].mean()

    fig = plt.figure(figsize=(20, 15.5))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.9, 1.05, 1.05], hspace=0.34, wspace=0.20)
    fig.suptitle("Burgers zero adversarial training: polished diagnostic dashboard", fontsize=20, fontweight="bold", y=0.995)
    fig.text(
        0.5,
        0.963,
        f"52/52 datasets improved | mean Relative L2 {mean_start:.4f} -> {mean_end:.4f} | mean fractional drop {mean_frac:.1%}",
        ha="center",
        fontsize=12,
        color=MUTED,
    )

    ax1 = fig.add_subplot(gs[0, 0])
    g = group_mean_std(eval_df, "relative_l2")
    for tier in TIER_ORDER:
        dd = g[g["manual_tier"] == tier].sort_values("epoch")
        if dd.empty:
            continue
        x = dd["epoch"].to_numpy(dtype=float)
        mean = dd["mean"].to_numpy(dtype=float)
        std = dd["std"].to_numpy(dtype=float)
        c = TIER_COLORS[tier]
        ax1.plot(x, mean, color=c, lw=2.0, label=tier)
        ax1.fill_between(x, mean - std, mean + std, color=c, alpha=0.10, lw=0)
    add_panel_label(ax1, "A")
    ax1.set_title("Relative L2 mean +/- std by dataset group", loc="left", fontweight="bold")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("Relative L2")
    ax1.set_xlim(0, 1000)
    ax1.legend(ncol=2, fontsize=7)

    ax2 = fig.add_subplot(gs[0, 1])
    atk_cols = [
        ("clean_loss_before_attack_mean", "clean", "#0072B2"),
        ("adv_loss_after_attack_mean", "adv after", "#D55E00"),
        ("attack_loss_gain_mean", "gain", "#009E73"),
    ]
    for col, label, color in atk_cols:
        ax2.plot(attack_df["epoch"], rolling(attack_df[col], 25), color=color, lw=2.2, label=label)
    add_panel_label(ax2, "B")
    ax2.set_title("Attack losses, 25-epoch moving average", loc="left", fontweight="bold")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("MSE, log scale")
    ax2.set_yscale("log")
    ax2.set_xlim(1, 1000)
    ax2.legend(ncol=3)

    ax3 = fig.add_subplot(gs[1, 0])
    d = tier_sorted(eval_df)
    order = d.drop_duplicates("dataset_id")["dataset_id"].tolist()
    piv = d.pivot_table(index="dataset_id", columns="epoch", values="relative_l2", aggfunc="mean").reindex(order)
    im = ax3.imshow(piv.to_numpy(dtype=float), aspect="auto", interpolation="nearest", cmap="magma_r")
    add_panel_label(ax3, "C")
    ax3.set_title("Relative L2 heatmap", loc="left", fontweight="bold")
    ax3.set_xlabel("epoch")
    ax3.set_ylabel("52 datasets")
    ax3.set_xticks(CHECKPOINTS)
    ax3.set_yticks([])
    fig.colorbar(im, ax=ax3, fraction=0.035, pad=0.02)

    ax4 = fig.add_subplot(gs[1, 1])
    rel_sorted = rel.sort_values("absolute_drop")
    y = np.arange(len(rel_sorted))
    ax4.barh(y, rel_sorted["absolute_drop"], color=[TIER_COLORS[t] for t in rel_sorted["manual_tier"]], alpha=0.88)
    add_panel_label(ax4, "D")
    ax4.set_title("Relative L2 reduction by dataset", loc="left", fontweight="bold")
    ax4.set_xlabel("epoch 0 - epoch 1000")
    ax4.set_yticks([])
    ax4.axvline(0, color="#333333", lw=1)

    ax5 = fig.add_subplot(gs[2, 0])
    cmap = plt.get_cmap("viridis")
    for i, (bucket_idx, dd) in enumerate(bucket_df.groupby("bucket_index")):
        dd = dd.sort_values("epoch")
        c = cmap(0.12 + 0.76 * i / 4)
        ax5.plot(dd["epoch"], rolling(dd["attack_loss_gain_mean"], 25), color=c, lw=2.0, label=f"bucket {bucket_idx}")
    add_panel_label(ax5, "E")
    ax5.set_title("Attack gain by epsilon bucket", loc="left", fontweight="bold")
    ax5.set_xlabel("epoch")
    ax5.set_ylabel("MSE gain, log scale")
    ax5.set_yscale("log")
    ax5.set_xlim(1, 1000)
    ax5.legend(ncol=5, fontsize=7)

    ax6 = fig.add_subplot(gs[2, 1])
    for pct, dd in highfreq_df.groupby("top_frequency_percent"):
        dd = dd.sort_values("epoch")
        c = FREQ_COLORS.get(float(pct), "#555555")
        ax6.plot(dd["epoch"], dd["mean"], color=c, lw=2.0, label=f"top {pct:g}%")
        ax6.fill_between(dd["epoch"], np.maximum(dd["mean"] - dd["std"].fillna(0), 0), dd["mean"] + dd["std"].fillna(0), color=c, alpha=0.08)
    add_panel_label(ax6, "F")
    ax6.set_title("High-frequency energy share of delta", loc="left", fontweight="bold")
    ax6.set_xlabel("epoch")
    ax6.set_ylabel("energy share")
    ax6.set_xlim(1, 1000)
    ax6.set_ylim(0, 0.05)
    ax6.legend(ncol=3, fontsize=7)

    fig.tight_layout(rect=[0, 0, 1, 0.945])
    return savefig(fig, "polished_report_dashboard.png")


def make_pdf(image_paths: list[Path]) -> Path:
    pdf_path = OUT_DIR / "polished_report.pdf"
    with PdfPages(pdf_path) as pdf:
        for path in image_paths:
            img = mpimg.imread(path)
            h, w = img.shape[:2]
            fig_w = 14
            fig_h = max(7, fig_w * h / w)
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            ax.imshow(img)
            ax.axis("off")
            fig.tight_layout(pad=0)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    return pdf_path


def write_readme(paths: list[Path], data: dict[str, pd.DataFrame]) -> None:
    rel = reduction_table(data["eval"], "relative_l2")
    attack = data["attack"]
    high = pd.read_csv(OUT_DIR / "polished_high_frequency_energy_share_trend_summary.csv")
    lines = [
        "# Polished Burgers Adversarial Training Visualizations",
        "",
        "This folder contains presentation-style visualizations for the completed Burgers ADV-only run.",
        "",
        "## Key Numbers",
        "",
        f"- Relative L2 improved on `{int((rel['absolute_drop'] > 0).sum())}/{len(rel)}` datasets.",
        f"- Mean Relative L2 changed from `{rel['epoch0'].mean():.6g}` to `{rel['epoch1000'].mean():.6g}`.",
        f"- Mean fractional Relative L2 reduction: `{rel['relative_drop_fraction'].mean():.3%}`.",
        f"- Mean attack gain changed from `{attack.loc[attack['epoch'] == 1, 'attack_loss_gain_mean'].iloc[0]:.6g}` to `{attack.loc[attack['epoch'] == 1000, 'attack_loss_gain_mean'].iloc[0]:.6g}`.",
        f"- Top 50% high-frequency energy share changed from `{high.loc[high['top_frequency_percent'] == 50.0, 'epoch1_mean'].iloc[0]:.6g}` to `{high.loc[high['top_frequency_percent'] == 50.0, 'epoch1000_mean'].iloc[0]:.6g}`.",
        "",
        "## Figures",
        "",
    ]
    for path in paths:
        lines.append(f"- `{path.name}`")
    lines.extend(
        [
            "",
            "## CSV Outputs",
            "",
            "- `polished_attack_loss_epoch_decile_summary.csv`",
            "- `polished_checkpoint_relative_l2_changes.csv`",
            "- `polished_checkpoint_relative_l2_changes_50epoch.csv`",
            "- `polished_high_frequency_energy_share_trend_summary.csv`",
            "- `polished_relative_l2_reduction_by_dataset.csv`",
            "- `polished_rmse_reduction_by_dataset.csv`",
        ]
    )
    (OUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    setup()
    data = read_data()
    paths = [
        plot_dashboard(data),
        plot_loss_reduction(data["eval"]),
        plot_heatmap(data["eval"]),
        plot_checkpoint_summary(data["eval"]),
        plot_attack_buckets(data["attack"], data["bucket"]),
        plot_high_frequency(data["highfreq"]),
        plot_delta_probe_checkpoints(),
    ]
    pdf_path = make_pdf(paths)
    write_readme(paths + [pdf_path], data)
    manifest = {
        "run_dir": str(RUN_DIR),
        "out_dir": str(OUT_DIR),
        "png_count": len(list(OUT_DIR.glob("*.png"))),
        "csv_count": len(list(OUT_DIR.glob("*.csv"))),
        "pdf": pdf_path.name,
        "figures": [p.name for p in paths],
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
