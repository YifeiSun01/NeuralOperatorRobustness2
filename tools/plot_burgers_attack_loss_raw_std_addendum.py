#!/usr/bin/env python3
"""Raw attack-loss visualizations with standard-deviation shading."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUN_DIR = Path("adversarial_training_runs/burgers_zero_advonly_random_jitter_1000ep_bs480_steps5_eps5bucket_20260601")
BURGERS_DIR = RUN_DIR / "burgers"
OUT_DIR = Path("visualizations/burgers_zero_adv_training_20260601/polished_report")

BG = "#fbfaf7"
AX_BG = "#ffffff"
TEXT = "#202124"
GRID = "#d9d6cc"
COLORS = {
    "clean_loss_before_attack": "#0072B2",
    "adv_loss_after_attack": "#D55E00",
    "attack_loss_gain": "#009E73",
    "attack_loss_gain_relative": "#6A3D9A",
}


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
        }
    )


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


def make_pooled_summary(attack_df: pd.DataFrame, bucket_df: pd.DataFrame) -> pd.DataFrame:
    out = attack_df[["epoch", "clean_loss_before_attack_mean", "adv_loss_after_attack_mean", "attack_loss_gain_mean"]].copy()
    for prefix in ["clean_loss_before_attack", "adv_loss_after_attack", "attack_loss_gain"]:
        pooled = pooled_epoch_std(bucket_df, prefix)
        out = out.merge(pooled, on="epoch", how="left")
    out.to_csv(OUT_DIR / "polished_attack_loss_raw_std_summary.csv", index=False)
    return out


def positive_band(mean: np.ndarray, std: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    upper = mean + std
    lower = mean - std
    positive = mean[mean > 0]
    floor = max(float(np.nanmin(positive)) * 0.08, 1e-12) if positive.size else 1e-12
    lower = np.maximum(lower, floor)
    upper = np.maximum(upper, floor * 1.2)
    return lower, upper


def plot_raw_attack_loss(summary: pd.DataFrame) -> Path:
    specs = [
        ("clean_loss_before_attack", "Clean loss before attack", "clean training-batch loss keeps falling"),
        ("adv_loss_after_attack", "ADV loss after attack", "attacked loss also becomes smaller"),
        ("attack_loss_gain", "Attack gain = adv - clean", "attackable gap shrinks over training"),
    ]
    shared_bounds = []
    for prefix, _, _ in specs:
        mean = summary[f"{prefix}_mean"].to_numpy(dtype=float)
        std = summary[f"{prefix}_pooled_std"].to_numpy(dtype=float)
        lower, upper = positive_band(mean, std)
        shared_bounds.extend([lower, upper])
    shared_values = np.concatenate(shared_bounds)
    shared_values = shared_values[np.isfinite(shared_values) & (shared_values > 0)]
    y_min = float(np.nanmin(shared_values))
    y_max = float(np.nanmax(shared_values))
    log_pad = 0.06
    log_min, log_max = np.log10([y_min, y_max])
    y_min = 10 ** (log_min - log_pad * (log_max - log_min))
    y_max = 10 ** (log_max + log_pad * (log_max - log_min))

    fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True, sharey=True)
    x = summary["epoch"].to_numpy(dtype=float)
    for ax, (prefix, title, subtitle) in zip(axes, specs):
        mean = summary[f"{prefix}_mean"].to_numpy(dtype=float)
        std = summary[f"{prefix}_pooled_std"].to_numpy(dtype=float)
        color = COLORS[prefix]
        lower, upper = positive_band(mean, std)
        ax.plot(x, mean, color=color, lw=1.45, alpha=0.92, label="raw epoch mean")
        ax.fill_between(x, lower, upper, color=color, alpha=0.16, lw=0, label="pooled sample std")
        ax.set_yscale("log")
        ax.set_ylim(y_min, y_max)
        ax.set_ylabel("MSE, log scale")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.text(0.995, 0.80, subtitle, transform=ax.transAxes, ha="right", va="top", color="#626262", fontsize=10)
        ax.legend(loc="upper right")
        ax.set_xlim(1, 1000)
    axes[-1].set_xlabel("training epoch")
    fig.suptitle("Raw attack loss progress, no moving average", fontsize=18, fontweight="bold", y=0.995)
    fig.text(0.5, 0.963, "Solid line is the raw per-epoch mean; shaded band is pooled mean +/- std from epsilon buckets; all three panels share the same y range.", ha="center", color="#666666")
    fig.tight_layout(rect=[0, 0, 1, 0.945])
    out = OUT_DIR / "polished_attack_loss_raw_std_no_moving_average.png"
    fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def plot_raw_attack_loss_single_axis(summary: pd.DataFrame) -> Path:
    specs = [
        ("clean_loss_before_attack", "clean before attack"),
        ("adv_loss_after_attack", "adv after attack"),
        ("attack_loss_gain", "attack gain = adv - clean"),
    ]
    bounds = []
    for prefix, _ in specs:
        mean = summary[f"{prefix}_mean"].to_numpy(dtype=float)
        std = summary[f"{prefix}_pooled_std"].to_numpy(dtype=float)
        lower, upper = positive_band(mean, std)
        bounds.extend([lower, upper])
    values = np.concatenate(bounds)
    values = values[np.isfinite(values) & (values > 0)]
    y_min = float(np.nanmin(values))
    y_max = float(np.nanmax(values))
    log_min, log_max = np.log10([y_min, y_max])
    pad = 0.06 * (log_max - log_min)

    fig, ax = plt.subplots(figsize=(15.5, 7.8))
    x = summary["epoch"].to_numpy(dtype=float)
    for prefix, label in specs:
        mean = summary[f"{prefix}_mean"].to_numpy(dtype=float)
        std = summary[f"{prefix}_pooled_std"].to_numpy(dtype=float)
        lower, upper = positive_band(mean, std)
        color = COLORS[prefix]
        ax.plot(x, mean, color=color, lw=1.65, alpha=0.92, label=label)
        ax.fill_between(x, lower, upper, color=color, alpha=0.13, lw=0)
    ax.set_yscale("log")
    ax.set_ylim(10 ** (log_min - pad), 10 ** (log_max + pad))
    ax.set_xlim(1, 1000)
    ax.set_xlabel("training epoch")
    ax.set_ylabel("MSE, log scale")
    ax.set_title("Raw attack loss progress, one axis, no moving average", loc="left", fontweight="bold")
    ax.text(
        0.995,
        0.965,
        "solid lines: raw per-epoch means; shaded bands: pooled mean +/- std from epsilon buckets",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color="#626262",
        fontsize=10,
    )
    ax.legend(loc="upper right", ncol=3)
    fig.tight_layout()
    out = OUT_DIR / "polished_attack_loss_raw_std_three_lines_one_axis.png"
    fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out



def bucket_label(d: pd.DataFrame) -> str:
    lo = float(d["bucket_value_low"].iloc[0])
    hi = float(d["bucket_value_high"].iloc[0])
    eps_lo = float(d["epsilon_min"].min())
    eps_hi = float(d["epsilon_max"].max())
    return f"{lo:.2f}-{hi:.2f}x eps\n({eps_lo:.3f}-{eps_hi:.3f})"


def plot_epsilon_bucket_adv_and_gain(bucket_df: pd.DataFrame) -> Path:
    bucket_df = bucket_df.sort_values(["bucket_index", "epoch"])
    fig, axes = plt.subplots(1, 2, figsize=(17, 7.5), sharex=True)
    cmap = plt.get_cmap("viridis")
    n_buckets = bucket_df["bucket_index"].nunique()
    specs = [
        (axes[0], "adv_loss_after_attack", "ADV loss after attack by epsilon bucket"),
        (axes[1], "attack_loss_gain", "Attack gain by epsilon bucket"),
    ]
    for bpos, (bucket_idx, d) in enumerate(bucket_df.groupby("bucket_index")):
        color = cmap(0.10 + 0.80 * bpos / max(1, n_buckets - 1))
        x = d["epoch"].to_numpy(dtype=float)
        label = bucket_label(d)
        for ax, prefix, title in specs:
            mean = d[f"{prefix}_mean"].to_numpy(dtype=float)
            std = d[f"{prefix}_std"].fillna(0.0).to_numpy(dtype=float)
            lower, upper = positive_band(mean, std)
            ax.plot(x, mean, color=color, lw=1.55, alpha=0.92, label=label)
            ax.fill_between(x, lower, upper, color=color, alpha=0.13, lw=0)
            ax.set_yscale("log")
            ax.set_xlim(1, 1000)
            ax.set_xlabel("training epoch")
            ax.set_ylabel("MSE, log scale")
            ax.set_title(title, loc="left", fontweight="bold")
    axes[0].legend(title="epsilon jitter bucket\n(actual epsilon range)", loc="upper right", fontsize=8, title_fontsize=8)
    fig.suptitle("Raw epsilon-bucket attack curves with standard-deviation shading", fontsize=18, fontweight="bold", y=0.995)
    fig.text(0.5, 0.952, "No moving average: every line is the raw epoch mean for that bucket; shading is mean +/- std inside the bucket.", ha="center", color="#666666")
    fig.tight_layout(rect=[0, 0, 1, 0.925])
    out = OUT_DIR / "polished_epsilon_bucket_adv_after_and_gain_raw_std.png"
    fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def plot_epsilon_bucket_gain_and_relative_raw_std(bucket_df: pd.DataFrame) -> Path:
    bucket_df = bucket_df.sort_values(["bucket_index", "epoch"])
    fig, axes = plt.subplots(1, 2, figsize=(17.5, 7.5), sharex=True)
    cmap = plt.get_cmap("viridis")
    n_buckets = bucket_df["bucket_index"].nunique()
    specs = [
        (axes[0], "attack_loss_gain", "Attack gain by epsilon bucket", "MSE, log scale", True),
        (axes[1], "attack_loss_gain_relative", "Relative attack gain by epsilon bucket", "gain / clean", False),
    ]
    for bpos, (bucket_idx, d) in enumerate(bucket_df.groupby("bucket_index")):
        color = cmap(0.10 + 0.80 * bpos / max(1, n_buckets - 1))
        x = d["epoch"].to_numpy(dtype=float)
        label = bucket_label(d)
        for ax, prefix, title, ylabel, use_log in specs:
            mean = d[f"{prefix}_mean"].to_numpy(dtype=float)
            std = d[f"{prefix}_std"].fillna(0.0).to_numpy(dtype=float)
            if use_log:
                lower, upper = positive_band(mean, std)
            else:
                lower = np.maximum(mean - std, 0.0)
                upper = mean + std
            ax.plot(x, mean, color=color, lw=1.55, alpha=0.92, label=label)
            ax.fill_between(x, lower, upper, color=color, alpha=0.13, lw=0)
            if use_log:
                ax.set_yscale("log")
            ax.set_xlim(1, 1000)
            ax.set_xlabel("training epoch")
            ax.set_ylabel(ylabel)
            ax.set_title(title, loc="left", fontweight="bold")
    axes[1].legend(title="epsilon jitter bucket\n(actual epsilon range)", loc="upper right", fontsize=8, title_fontsize=8)
    fig.suptitle("Raw epsilon-bucket attack gain and relative gain, no moving average", fontsize=18, fontweight="bold", y=0.995)
    fig.text(0.5, 0.952, "No epoch moving average: every line is the raw epoch mean for that epsilon bucket; shading is mean +/- std inside the bucket.", ha="center", color="#666666")
    fig.tight_layout(rect=[0, 0, 1, 0.925])
    out = OUT_DIR / "polished_attack_loss_buckets_gain_relative_raw_std_no_moving_average.png"
    fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out



def update_readme(paths: list[Path]) -> None:
    readme = OUT_DIR / "README.md"
    text = readme.read_text(encoding="utf-8") if readme.exists() else "# Polished Burgers Adversarial Training Visualizations\n"
    section = """

## Raw Attack-Loss Supplement: No Moving Average

These figures remove the 25-epoch moving average and show raw epoch means with standard-deviation shading.

- `polished_attack_loss_raw_std_no_moving_average.png`
- `polished_epsilon_bucket_adv_after_and_gain_raw_std.png`
- `polished_attack_loss_raw_std_summary.csv`
"""
    if "## Raw Attack-Loss Supplement: No Moving Average" not in text:
        text = text.rstrip() + section
    for name in [
        "polished_attack_loss_raw_std_three_lines_one_axis.png",
        "polished_attack_loss_buckets_gain_relative_raw_std_no_moving_average.png",
    ]:
        bullet = f"- `{name}`"
        if bullet not in text:
            text = text.rstrip() + "\n" + bullet + "\n"
    readme.write_text(text.rstrip() + "\n", encoding="utf-8")


def main() -> None:
    setup()
    attack_df = pd.read_csv(BURGERS_DIR / "attack_epoch_summary.csv")
    bucket_df = pd.read_csv(BURGERS_DIR / "attack_epsilon_bucket_summary.csv")
    summary = make_pooled_summary(attack_df, bucket_df)
    paths = [
        plot_raw_attack_loss(summary),
        plot_raw_attack_loss_single_axis(summary),
        plot_epsilon_bucket_adv_and_gain(bucket_df),
        plot_epsilon_bucket_gain_and_relative_raw_std(bucket_df),
    ]
    update_readme(paths)
    manifest_path = OUT_DIR / "raw_attack_loss_addendum_manifest.json"
    manifest = {
        "run_dir": str(RUN_DIR),
        "out_dir": str(OUT_DIR),
        "figures": [p.name for p in paths],
        "csv": "polished_attack_loss_raw_std_summary.csv",
        "bucket_count": int(bucket_df["bucket_index"].nunique()),
        "notes": "No moving average; shaded bands are mean +/- std. Overall std is pooled from epsilon-bucket sample stats. The three raw attack-loss panels share one y-axis range.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
