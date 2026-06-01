#!/usr/bin/env python3
"""Checkpoint-style hybrid visualizations requested after the final composite set.

Loss figures follow the checkpoint-summary layout: a large absolute-value heatmap
on top and a short group-mean lineplot below. Delta FFT keeps the raw heatmap on
top and uses a 25-epoch moving average only for the bottom selected spectra.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUN_DIR = Path("adversarial_training_runs/burgers_zero_advonly_random_jitter_1000ep_bs480_steps5_eps5bucket_20260601")
BURGERS_DIR = RUN_DIR / "burgers"
OUT_DIR = Path("visualizations/burgers_zero_adv_training_20260601/polished_report")
EVAL_EPOCHS_11 = list(range(0, 1001, 100))
SELECTED_EPOCHS = [50, 200, 400, 600, 800, 1000]

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
BG = "#fbfaf7"
AX_BG = "#ffffff"
GRID = "#d9d6cc"
TEXT = "#202124"
MUTED = "#666666"


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
    return d.sort_values(["_tier_rank", "_manual_rank", "dataset_id", "epoch"]).drop(columns=["_tier_rank", "_manual_rank"])


def group_mean_std(eval_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    g = eval_df.groupby(["manual_tier", "epoch"], as_index=False)[metric].agg(["mean", "std", "count"]).reset_index()
    g["std"] = g["std"].fillna(0.0)
    return g


def plot_checkpoint_style_loss(eval_df: pd.DataFrame, metric: str, filename: str) -> Path:
    label = "Relative L2" if metric == "relative_l2" else "RMSE"
    d = tier_sorted(eval_df[eval_df["epoch"].isin(EVAL_EPOCHS_11)])
    order = d.drop_duplicates("dataset_id")["dataset_id"].tolist()
    tiers = d.drop_duplicates("dataset_id")["manual_tier"].tolist()
    pivot = d.pivot_table(index="dataset_id", columns="epoch", values=metric, aggfunc="mean").reindex(order)
    pivot = pivot.reindex(columns=EVAL_EPOCHS_11)
    matrix = pivot.to_numpy(dtype=float)
    pivot.to_csv(OUT_DIR / filename.replace(".png", ".csv"))

    tier_nums = np.array([TIER_ORDER.index(t) if t in TIER_ORDER else -1 for t in tiers]).reshape(-1, 1)
    fig = plt.figure(figsize=(19.5, 13.2))
    gs = fig.add_gridspec(
        2,
        3,
        height_ratios=[1.0, 0.22],
        width_ratios=[0.30, 5.8, 0.22],
        hspace=0.22,
        wspace=0.045,
    )
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
    ax_heat.set_title(f"Absolute {label} at 11 checkpoints: 52 datasets x epochs 0,100,...,1000", loc="left", fontweight="bold")
    ax_heat.set_xlabel("evaluation epoch")
    ax_heat.set_ylabel("")
    ax_heat.set_xticks(np.arange(len(EVAL_EPOCHS_11)))
    ax_heat.set_xticklabels([str(x) for x in EVAL_EPOCHS_11], rotation=0)
    ax_heat.set_yticks([])

    for i in range(1, len(tiers)):
        if tiers[i] != tiers[i - 1]:
            ax_heat.axhline(i - 0.5, color="white", lw=1.25, alpha=0.78)
            ax_strip.axhline(i - 0.5, color="white", lw=1.25, alpha=0.78)

    g = group_mean_std(eval_df[eval_df["epoch"].isin(EVAL_EPOCHS_11)], metric)
    y_values = []
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
    vals = np.concatenate([v[np.isfinite(v)] for v in y_values])
    ymin = max(0.0, float(np.nanmin(vals)))
    ymax = float(np.nanmax(vals))
    pad = 0.05 * (ymax - ymin if ymax > ymin else 1.0)
    ax_line.set_xlim(0, 1000)
    ax_line.set_ylim(max(0.0, ymin - pad), ymax + pad)
    ax_line.set_title("Group mean lineplot at the same 11 checkpoints; no epoch moving average", loc="left", fontweight="bold", fontsize=10.5)
    ax_line.set_xlabel("evaluation epoch")
    ax_line.set_ylabel(label)
    ax_line.legend(ncol=4, loc="upper right", fontsize=7.6)

    fig.suptitle(f"Checkpoint-style {label} heatmap with short group lineplot below", fontsize=18, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    return savefig(fig, filename)


def load_probe_npz(epoch: int) -> dict[str, np.ndarray]:
    matches = sorted((BURGERS_DIR / "attack_probe_samples").glob(f"burgers_epoch{epoch:03d}_step*_attack_probe.npz"))
    if not matches:
        raise FileNotFoundError(f"No attack probe npz for epoch {epoch}")
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


def compute_delta_fft_matrix(epochs: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    epochs = sorted(int(e) for e in epochs)
    rows = []
    freq_modes = None
    for epoch in epochs:
        data = load_probe_npz(epoch)
        powers = []
        for sample_idx in range(data["delta"].shape[0]):
            freq, power = fft_power(data["delta"][sample_idx])
            if freq_modes is None:
                freq_modes = freq[1:]
            powers.append(power[1:])
        rows.append(np.nanmean(np.asarray(powers, dtype=float), axis=0))
    return np.asarray(epochs, dtype=int), np.asarray(freq_modes, dtype=float), np.asarray(rows, dtype=float)


def rolling_epoch_matrix(matrix: np.ndarray, window: int = 25) -> np.ndarray:
    return pd.DataFrame(matrix).rolling(window, center=True, min_periods=1).mean().to_numpy(dtype=float)


def plot_delta_fft_raw_heatmap_epoch_smoothed_lines(probe_df: pd.DataFrame) -> Path:
    epochs, freq_modes, raw_matrix = compute_delta_fft_matrix(sorted(probe_df["epoch"].unique()))
    epoch_smooth = rolling_epoch_matrix(raw_matrix, 25)
    log_power = np.log10(np.clip(raw_matrix, 1e-18, None))
    finite = log_power[np.isfinite(log_power)]
    vmin, vmax = np.nanpercentile(finite, [2, 99.5])

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
    ax_heat.set_xlim(1, 512)

    colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.95, len(SELECTED_EPOCHS)))
    for epoch, color in zip(SELECTED_EPOCHS, colors):
        idx = int(np.argmin(np.abs(epochs - epoch)))
        ax_line.plot(freq_modes, epoch_smooth[idx] + 1e-18, color=color, lw=1.55, alpha=0.84, label=f"epoch {epochs[idx]}")
    ax_line.set_yscale("log")
    ax_line.set_xlim(1, 512)
    ax_line.set_title("Selected spectra from 25-epoch moving-average matrix; top heatmap remains raw", loc="left", fontweight="bold", fontsize=10.5)
    ax_line.set_xlabel("Fourier mode")
    ax_line.set_ylabel("normalized power")
    ax_line.legend(ncol=6, loc="upper right", fontsize=7.6)

    fig.suptitle("Delta FFT spectrum: raw heatmap with short 25-epoch-smoothed lineplot below", fontsize=18, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    return savefig(fig, "polished_checkpoint_style_delta_fft_raw_heatmap_25epoch_smoothed_lines.png")


def update_readme(paths: list[Path]) -> None:
    readme = OUT_DIR / "README.md"
    text = readme.read_text(encoding="utf-8") if readme.exists() else "# Polished Burgers Adversarial Training Visualizations\n"
    marker = "## Checkpoint-Style Hybrid Figures"
    section = [
        "",
        marker,
        "",
        "These figures follow the checkpoint-summary layout: a large absolute-value heatmap on top and a short, wide lineplot underneath. Loss lineplots use group means at the same 11 checkpoints and do not use epoch moving average. The FFT figure keeps the heatmap raw and uses 25-epoch moving average only for the bottom selected spectra.",
        "",
    ]
    for path in paths:
        section.append(f"- `{path.name}`")
    if marker in text:
        text = text.split(marker)[0].rstrip()
    text = text.rstrip() + "\n" + "\n".join(section) + "\n"
    readme.write_text(text, encoding="utf-8")


def main() -> None:
    setup()
    eval_df = pd.read_csv(BURGERS_DIR / "eval_metrics.csv")
    probe_df = pd.read_csv(BURGERS_DIR / "attack_probe_samples.csv")
    paths = [
        plot_checkpoint_style_loss(eval_df, "relative_l2", "polished_checkpoint_style_relative_l2_absolute11_heatmap_line_below.png"),
        plot_checkpoint_style_loss(eval_df, "rmse", "polished_checkpoint_style_rmse_absolute11_heatmap_line_below.png"),
        plot_delta_fft_raw_heatmap_epoch_smoothed_lines(probe_df),
    ]
    update_readme(paths)
    manifest = {
        "run_dir": str(RUN_DIR),
        "out_dir": str(OUT_DIR),
        "figures": [p.name for p in paths],
        "loss_checkpoint_epochs": EVAL_EPOCHS_11,
        "delta_line_smoothing": "25-epoch moving average for bottom spectra only; heatmap is raw",
        "notes": "Attack geometry is run-specific; check run config and delta geometry validator.",
    }
    (OUT_DIR / "polished_checkpoint_style_hybrid_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
