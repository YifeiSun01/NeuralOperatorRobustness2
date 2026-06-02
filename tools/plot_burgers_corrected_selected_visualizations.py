#!/usr/bin/env python3
"""Corrected selected visualizations after layout feedback."""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUN_DIR = Path("adversarial_training_runs/burgers_zero_advonly_random_jitter_1000ep_bs480_steps5_eps5bucket_20260601")
BURGERS_DIR = RUN_DIR / "burgers"
RAW_VIZ_DIR = Path("visualizations/burgers_zero_adv_training_20260601")
OUT_DIR = RAW_VIZ_DIR / "polished_report"
REPO_SELECTED = RAW_VIZ_DIR / "polished_selected_download_20260601"
TOP_SELECTED = Path("/workspace/polished_selected_download_20260601")
TOP_ZIP = Path("/workspace/polished_selected_download_20260601.zip")
REPO_ZIP = RAW_VIZ_DIR / "polished_selected_download_20260601.zip"

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


def positive_band(mean: np.ndarray, std: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)
    upper = mean + std
    lower = mean - std
    positive = mean[np.isfinite(mean) & (mean > 0)]
    floor = max(float(np.nanmin(positive)) * 0.08, 1e-12) if positive.size else 1e-12
    lower = np.maximum(lower, floor)
    upper = np.maximum(upper, floor * 1.2)
    return lower, upper


def pooled_epoch_std(bucket_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    rows = []
    mean_col = f"{prefix}_mean"
    std_col = f"{prefix}_std"
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
    out = attack_df[["epoch", "clean_loss_before_attack_mean", "adv_loss_after_attack_mean", "attack_loss_gain_mean"]].copy()
    for prefix in ["clean_loss_before_attack", "adv_loss_after_attack", "attack_loss_gain"]:
        out = out.merge(pooled_epoch_std(bucket_df, prefix), on="epoch", how="left")
    return out


def bucket_label(d: pd.DataFrame) -> str:
    lo = float(d["bucket_value_low"].iloc[0])
    hi = float(d["bucket_value_high"].iloc[0])
    eps_lo = float(d["epsilon_min"].min())
    eps_hi = float(d["epsilon_max"].max())
    return f"{lo:.2f}-{hi:.2f}x eps ({eps_lo:.3f}-{eps_hi:.3f})"


def plot_corrected_attack(summary: pd.DataFrame, bucket_df: pd.DataFrame) -> Path:
    fig = plt.figure(figsize=(19, 12.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.9, 1.05], hspace=0.28, wspace=0.18)
    ax_top = fig.add_subplot(gs[0, :])
    axes_bottom = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]

    x = summary["epoch"].to_numpy(dtype=float)
    specs_top = [
        ("clean_loss_before_attack", "clean before attack"),
        ("adv_loss_after_attack", "adv after attack"),
        ("attack_loss_gain", "attack gain = adv - clean"),
    ]
    bounds = []
    for prefix, label in specs_top:
        mean = summary[f"{prefix}_mean"].to_numpy(dtype=float)
        std = summary[f"{prefix}_pooled_std"].fillna(0.0).to_numpy(dtype=float)
        lower, upper = positive_band(mean, std)
        color = ATTACK_COLORS[prefix]
        ax_top.plot(x, mean, color=color, lw=1.65, alpha=0.92, label=label)
        ax_top.fill_between(x, lower, upper, color=color, alpha=0.13, lw=0)
        bounds.extend([lower, upper])
    values = np.concatenate(bounds)
    values = values[np.isfinite(values) & (values > 0)]
    log_min, log_max = np.log10([float(np.nanmin(values)), float(np.nanmax(values))])
    pad = 0.06 * (log_max - log_min)
    ax_top.set_yscale("log")
    ax_top.set_ylim(10 ** (log_min - pad), 10 ** (log_max + pad))
    ax_top.set_xlim(1, 1000)
    ax_top.set_xlabel("training epoch")
    ax_top.set_ylabel("MSE, log scale")
    ax_top.set_title("Attack loss before perturbation, attack loss after perturbation, and attack gain", loc="left", fontweight="bold")
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
                lower, upper = positive_band(mean, std)
            else:
                lower = np.maximum(mean - std, 0.0)
                upper = mean + std
            ax.plot(bx, mean, color=color, lw=1.45, alpha=0.92, label=label)
            ax.fill_between(bx, lower, upper, color=color, alpha=0.13, lw=0)
            if use_log:
                ax.set_yscale("log")
            ax.set_xlim(1, 1000)
            ax.set_xlabel("training epoch")
            ax.set_ylabel(ylabel)
            ax.set_title(title, loc="left", fontweight="bold")
    axes_bottom[1].set_ylim(0, 10)
    axes_bottom[1].legend(title="epsilon jitter bucket", loc="upper right", fontsize=7.6, title_fontsize=8)
    fig.suptitle("Attack loss and attack gain during 1,000 epochs of adversarial training", fontsize=18, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return savefig(fig, "corrected_attack_loss_three_lines_plus_buckets.png")


def plot_corrected_loss(eval_df: pd.DataFrame, metric: str, filename: str) -> Path:
    label = "Relative L2" if metric == "relative_l2" else "RMSE"
    d = tier_sorted(eval_df)
    order = d.drop_duplicates("dataset_id")["dataset_id"].tolist()
    tiers = d.drop_duplicates("dataset_id")["manual_tier"].tolist()
    pivot = d.pivot_table(index="dataset_id", columns="epoch", values=metric, aggfunc="mean").reindex(order)
    matrix = pivot.to_numpy(dtype=float)
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
    ax_strip.set_yticklabels([shorten(x, 42) for x in order], fontsize=6.2)
    ax_strip.tick_params(axis="y", length=0, pad=5)
    ax_strip.yaxis.tick_left()
    ax_strip.set_title("group", fontsize=9)

    im = ax_heat.imshow(matrix, aspect="auto", interpolation="nearest", cmap="magma")
    cbar = fig.colorbar(im, cax=cax, label=label)
    cbar.ax.tick_params(labelsize=8)
    ax_heat.set_title(f"{label} across 52 train, test, and generalization datasets", loc="left", fontweight="bold")
    ax_heat.set_xlabel("evaluation epoch")
    ax_heat.set_ylabel("")
    xticks = [0, 200, 400, 600, 800, 1000]
    ax_heat.set_xticks(xticks)
    ax_heat.set_xticklabels([str(x) for x in xticks])
    ax_heat.set_yticks([])
    for i in range(1, len(tiers)):
        if tiers[i] != tiers[i - 1]:
            ax_heat.axhline(i - 0.5, color="white", lw=1.25, alpha=0.78)
            ax_strip.axhline(i - 0.5, color="white", lw=1.25, alpha=0.78)

    g = group_mean_std(eval_df, metric)
    y_values = []
    for tier in TIER_ORDER:
        dd = g[g["manual_tier"] == tier].sort_values("epoch")
        if dd.empty:
            continue
        x = dd["epoch"].to_numpy(dtype=float)
        mean = dd["mean"].to_numpy(dtype=float)
        std = dd["std"].to_numpy(dtype=float)
        color = TIER_COLORS[tier]
        ax_line.plot(x, mean, color=color, lw=1.35, alpha=0.78, label=tier)
        ax_line.fill_between(x, np.maximum(mean - std, 0.0), mean + std, color=color, alpha=0.08, lw=0)
        y_values.extend([mean - std, mean + std])
    vals = np.concatenate([v[np.isfinite(v)] for v in y_values])
    ymin = max(0.0, float(np.nanmin(vals)))
    ymax = float(np.nanmax(vals))
    pad = 0.05 * (ymax - ymin if ymax > ymin else 1.0)
    ax_line.set_xlim(0, 1000)
    ax_line.set_ylim(max(0.0, ymin - pad), ymax + pad)
    ax_line.set_title("Group mean loss trends by dataset family", loc="left", fontweight="bold", fontsize=10.5)
    ax_line.set_xlabel("evaluation epoch")
    ax_line.set_ylabel(label)
    ax_line.legend(ncol=4, loc="upper right", fontsize=7.5)

    fig.suptitle(f"{label} decrease across train, test, and generalization datasets during 1,000 epochs of adversarial training", fontsize=18, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    return savefig(fig, filename)


def reset_selected_folder(paths: list[Path]) -> None:
    for folder in [REPO_SELECTED, TOP_SELECTED]:
        folder.mkdir(parents=True, exist_ok=True)
        for p in folder.iterdir():
            if p.is_file():
                p.unlink()
        for src in paths:
            shutil.copy2(src, folder / src.name)
    for zip_path, folder in [(REPO_ZIP, REPO_SELECTED), (TOP_ZIP, TOP_SELECTED)]:
        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(folder.glob("*.png")):
                zf.write(p, arcname=str(Path(folder.name) / p.name))


def main() -> None:
    setup()
    eval_df = pd.read_csv(BURGERS_DIR / "eval_metrics.csv")
    attack_df = pd.read_csv(BURGERS_DIR / "attack_epoch_summary.csv")
    bucket_df = pd.read_csv(BURGERS_DIR / "attack_epsilon_bucket_summary.csv")
    summary = make_attack_summary(attack_df, bucket_df)
    paths = [
        plot_corrected_attack(summary, bucket_df),
        plot_corrected_loss(eval_df, "relative_l2", "corrected_relative_l2_full_heatmap_raw_group_line.png"),
        plot_corrected_loss(eval_df, "rmse", "corrected_rmse_full_heatmap_raw_group_line.png"),
    ]
    # Keep the delta spectrum figure the user had accepted, but put it into the selected download with the corrected set.
    delta = OUT_DIR / "polished_checkpoint_style_delta_fft_raw_heatmap_25epoch_smoothed_lines.png"
    if delta.exists():
        paths.append(delta)
    reset_selected_folder(paths)
    manifest = {
        "run_dir": str(RUN_DIR),
        "out_dir": str(OUT_DIR),
        "selected_folder_repo": str(REPO_SELECTED),
        "selected_folder_top": str(TOP_SELECTED),
        "figures": [p.name for p in paths],
        "notes": "Corrected selected set. Removed 100-epoch checkpoint loss figures and removed 25-epoch moving-average loss lineplots. Attack geometry is run-specific; check run config and delta geometry validator.",
    }
    (OUT_DIR / "corrected_selected_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
