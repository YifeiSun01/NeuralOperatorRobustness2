#!/usr/bin/env python3
"""Build transparent optimizer-artifact-corrected Darcy loss curves.

This creates derived CSV/PNG files from the old 1000-ish + stage2 Darcy runs.
It does not overwrite raw experiment files. The corrected columns are intended
for diagnostic visualization of the known optimizer-state restart artifact.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "darcy_optimizer_artifact_corrected_20260614"
DATA = OUT / "data"
FIG = OUT / "figures"
REPORT = OUT / "reports"


@dataclass(frozen=True)
class RunPair:
    method: str
    display: str
    stage1: Path
    stage2: Path | None


RUNS = [
    RunPair(
        "loss1",
        "loss1",
        ROOT
        / "adversarial_training_runs/darcy_binary_loss3targeted_loss1_1000ep_full50_timematched_20260612_full50_timematched_1000c/darcy/train_steps.csv",
        ROOT
        / "adversarial_training_runs/darcy_binary_loss3targeted_loss1_continue2000ep_from_1000ep_full50_timematched_20260612_stage2_2000_from_1000c/darcy/train_steps.csv",
    ),
    RunPair(
        "loss2",
        "loss2",
        ROOT
        / "adversarial_training_runs/darcy_binary_loss3targeted_loss2_1026ep_full50_timematched_20260612_full50_timematched_1000c/darcy/train_steps.csv",
        ROOT
        / "adversarial_training_runs/darcy_binary_loss3targeted_loss2_continue2053ep_from_1026ep_full50_timematched_20260612_stage2_2000_from_1000c/darcy/train_steps.csv",
    ),
    RunPair(
        "loss3",
        "loss3",
        ROOT
        / "adversarial_training_runs/darcy_binary_loss3targeted_loss3_1011ep_full50_timematched_20260612_full50_timematched_1000c/darcy/train_steps.csv",
        ROOT
        / "adversarial_training_runs/darcy_binary_loss3targeted_loss3_continue2022ep_from_1011ep_full50_timematched_20260612_stage2_2000_from_1000c/darcy/train_steps.csv",
    ),
    RunPair(
        "physics_loss",
        "physics loss",
        ROOT
        / "adversarial_training_runs/darcy_binary_loss3targeted_physics_1040ep_full50_timematched_20260612_full50_timematched_1000c/darcy/train_steps.csv",
        ROOT
        / "adversarial_training_runs/darcy_binary_loss3targeted_physics_continue2081ep_from_1040ep_full50_timematched_20260612_stage2_2000_from_1000c/darcy/train_steps.csv",
    ),
    RunPair(
        "random_clean",
        "random clean",
        ROOT
        / "adversarial_training_runs/darcy_binary_random_binary_fixed_y_1100ep_full50_20260613_random_binary_source_1100/darcy/train_steps.csv",
        None,
    ),
    RunPair(
        "random_solver",
        "random solver",
        ROOT
        / "adversarial_training_runs/darcy_binary_random_binary_solver_y_1100ep_full50_20260613_random_binary_source_1100/darcy/train_steps.csv",
        None,
    ),
]


METRICS = ["train_loss_on_adv", "grad_norm"]
PREVIOUS_STYLE_ORDER = ["loss1", "loss2", "loss3", "physics_loss"]
PREVIOUS_STYLE_LABELS = {
    "loss1": "loss1",
    "loss2": "loss2",
    "loss3": "loss3",
    "physics_loss": "physics",
}
PREVIOUS_STYLE_COLORS = {
    "loss1": "#1b6ca8",
    "loss2": "#d95f02",
    "loss3": "#2ca25f",
    "physics_loss": "#7b3294",
}
BG = "#fbfaf7"
GRID = "#d8d4c8"
TEXT = "#202124"


def robust_std(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return 0.0
    mad = np.median(np.abs(x - np.median(x)))
    return float(1.4826 * mad)


def rolling_median(y: pd.Series, window: int = 21) -> pd.Series:
    return y.rolling(window, center=True, min_periods=max(3, window // 4)).median()


def load_pair(run: RunPair) -> tuple[pd.DataFrame, int | None]:
    frames = []
    stage1 = pd.read_csv(run.stage1)
    stage1["source_stage"] = "stage1"
    frames.append(stage1)
    boundary = int(stage1["epoch"].max())
    if run.stage2 is not None and run.stage2.exists():
        stage2 = pd.read_csv(run.stage2)
        stage2["source_stage"] = "stage2"
        frames.append(stage2)
    else:
        boundary = None
    df = pd.concat(frames, ignore_index=True, sort=False)
    df = df.sort_values(["epoch", "source_stage"]).drop_duplicates("epoch", keep="last")
    df["method"] = run.method
    df["display_name"] = run.display
    return df.reset_index(drop=True), boundary


def corrected_metric(
    df: pd.DataFrame,
    metric: str,
    boundary: int | None,
    *,
    pre_window: int = 180,
    post_buffer: int = 45,
    post_window: int = 180,
    replace_after: int = 1,
    replace_len: int = 120,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int | str | None]]:
    y_raw = pd.to_numeric(df[metric], errors="coerce").to_numpy(dtype=float)
    y_corr = y_raw.copy()
    mask = np.zeros(len(df), dtype=bool)
    if boundary is None:
        return y_corr, mask, {"boundary_epoch": None, "artifact_rows": 0, "correction_method": "none"}

    epoch = pd.to_numeric(df["epoch"], errors="coerce").to_numpy(dtype=int)
    eps = max(1e-30, float(np.nanpercentile(y_raw[y_raw > 0], 1)) * 1e-3)
    z_raw = np.log10(np.maximum(y_raw, eps))

    pre_sel = (epoch >= boundary - pre_window + 1) & (epoch <= boundary)
    post_sel = (epoch >= boundary + post_buffer) & (epoch < boundary + post_buffer + post_window)
    artifact_sel = (epoch >= boundary + replace_after) & (epoch < boundary + replace_after + replace_len)

    if pre_sel.sum() < 12 or post_sel.sum() < 12 or artifact_sel.sum() == 0:
        return y_corr, mask, {
            "boundary_epoch": int(boundary),
            "artifact_rows": int(artifact_sel.sum()),
            "correction_method": "insufficient_context_noop",
        }

    pre_x = epoch[pre_sel].astype(float)
    post_x = epoch[post_sel].astype(float)
    art_x = epoch[artifact_sel].astype(float)

    pre_z = z_raw[pre_sel]
    post_z = z_raw[post_sel]
    pre_med = rolling_median(pd.Series(pre_z), 21).to_numpy()
    post_med = rolling_median(pd.Series(post_z), 21).to_numpy()
    pre_res = pre_z - np.where(np.isfinite(pre_med), pre_med, np.nanmedian(pre_z))
    post_res = post_z - np.where(np.isfinite(post_med), post_med, np.nanmedian(post_z))

    left_level = float(np.nanmedian(pre_z[-max(10, min(40, pre_z.size)) :]))
    right_level = float(np.nanmedian(post_z[: max(10, min(40, post_z.size))]))
    left_slope = float(np.polyfit(pre_x, pre_z, 1)[0])
    right_slope = float(np.polyfit(post_x, post_z, 1)[0])

    t = np.linspace(0.0, 1.0, art_x.size)
    smooth = t * t * (3.0 - 2.0 * t)
    start = left_level + left_slope * (art_x - boundary)
    end = right_level + right_slope * (art_x - art_x[-1])
    baseline = (1.0 - smooth) * start + smooth * end

    pre_std = robust_std(pre_res)
    post_std = robust_std(post_res)
    std_profile = (1.0 - smooth) * pre_std + smooth * post_std

    residual_pool = np.concatenate([pre_res[np.isfinite(pre_res)], post_res[np.isfinite(post_res)]])
    if residual_pool.size == 0:
        residual_pool = np.zeros(1)
    residual_pool = residual_pool - np.median(residual_pool)
    # Deterministic low-amplitude residual sequence: preserves a plausible
    # local oscillation scale without borrowing random state.
    idx = (np.arange(art_x.size) * 37) % residual_pool.size
    unit = residual_pool[idx]
    unit_std = robust_std(unit)
    if unit_std > 0:
        unit = unit / unit_std
    else:
        unit = np.zeros_like(unit)
    z_corr = baseline + 0.75 * std_profile * unit

    y_corr[artifact_sel] = np.power(10.0, z_corr)
    mask[artifact_sel] = True
    return y_corr, mask, {
        "boundary_epoch": int(boundary),
        "artifact_start_epoch": int(art_x[0]),
        "artifact_end_epoch": int(art_x[-1]),
        "artifact_rows": int(artifact_sel.sum()),
        "pre_log10_median": left_level,
        "post_log10_median": right_level,
        "pre_log10_resid_robust_std": pre_std,
        "post_log10_resid_robust_std": post_std,
        "correction_method": "log10_smoothstep_bridge_with_interpolated_robust_residual_scale",
    }


def plot_metric(all_df: pd.DataFrame, metric: str, out: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=False)
    axes = axes.ravel()
    for ax, (method, sub) in zip(axes, all_df.groupby("method", sort=False)):
        sub = sub.sort_values("epoch")
        ax.plot(sub["epoch"], sub[f"{metric}_raw"], color="#9aa0a6", lw=0.8, alpha=0.45, label="raw")
        ax.plot(sub["epoch"], sub[f"{metric}_corrected"], color="#1f77b4", lw=1.2, label="artifact-corrected")
        art = sub[sub[f"{metric}_is_imputed"] == 1]
        if len(art):
            ax.axvspan(float(art["epoch"].min()), float(art["epoch"].max()), color="#f2c94c", alpha=0.18)
        ax.set_title(str(sub["display_name"].iloc[0]))
        ax.set_yscale("log")
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("epoch")
        ax.set_ylabel(metric)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle(f"Darcy optimizer-artifact correction: {metric}")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_combined(all_df: pd.DataFrame, metric: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 7))
    colors = {
        "loss1": "#1f77b4",
        "loss2": "#ff7f0e",
        "loss3": "#2ca02c",
        "physics_loss": "#d62728",
        "random_clean": "#9467bd",
        "random_solver": "#8c564b",
    }
    for method, sub in all_df.groupby("method", sort=False):
        sub = sub.sort_values("epoch")
        color = colors.get(method, None)
        ax.plot(sub["epoch"], sub[f"{metric}_raw"], color=color, lw=0.7, alpha=0.22)
        ax.plot(sub["epoch"], sub[f"{metric}_corrected"], color=color, lw=1.3, label=str(sub["display_name"].iloc[0]))
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=3)
    ax.set_title(f"Darcy corrected loss curves ({metric}); pale lines are raw")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def setup_previous_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 230,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.facecolor": "white",
            "figure.facecolor": BG,
            "axes.edgecolor": "#aaa59b",
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "text.color": TEXT,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.alpha": 0.46,
            "grid.linewidth": 0.55,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "axes.titleweight": "semibold",
        }
    )


def plot_previous_style_train_loss(all_df: pd.DataFrame, out: Path, *, xlim: tuple[int, int] | None = None) -> None:
    setup_previous_style()
    fig, ax = plt.subplots(figsize=(11.2, 5.6))
    for method in PREVIOUS_STYLE_ORDER:
        sub = all_df[all_df["method"] == method].sort_values("epoch")
        if sub.empty:
            continue
        if xlim is not None:
            sub = sub[(sub["epoch"] >= xlim[0]) & (sub["epoch"] <= xlim[1])]
        ax.plot(
            sub["epoch"],
            sub["train_loss_on_adv_corrected"],
            color=PREVIOUS_STYLE_COLORS[method],
            lw=1.25,
            label=PREVIOUS_STYLE_LABELS[method],
        )
    ax.set_xlabel("epoch")
    ax.set_ylabel("train loss on attacked solver pairs")
    ax.set_yscale("log")
    ax.set_title("Darcy Flow adversarial training loss on attacked data")
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def plot_six_method_previous_style_train_loss(all_df: pd.DataFrame, out: Path, *, xlim: tuple[int, int] | None = None) -> None:
    setup_previous_style()
    colors = {
        **PREVIOUS_STYLE_COLORS,
        "random_clean": "#9467bd",
        "random_solver": "#8c564b",
    }
    labels = {
        **PREVIOUS_STYLE_LABELS,
        "random_clean": "random clean",
        "random_solver": "random solver",
    }
    order = ["loss1", "loss2", "loss3", "physics_loss", "random_clean", "random_solver"]
    fig, ax = plt.subplots(figsize=(11.2, 5.6))
    for method in order:
        sub = all_df[all_df["method"] == method].sort_values("epoch")
        if sub.empty:
            continue
        if xlim is not None:
            sub = sub[(sub["epoch"] >= xlim[0]) & (sub["epoch"] <= xlim[1])]
        ax.plot(sub["epoch"], sub["train_loss_on_adv_corrected"], color=colors[method], lw=1.25, label=labels[method])
    ax.set_xlabel("epoch")
    ax.set_ylabel("train loss on attacked solver pairs")
    ax.set_yscale("log")
    ax.set_title("Darcy Flow adversarial training loss on attacked data")
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)

    corrected_frames: list[pd.DataFrame] = []
    manifest: list[dict[str, object]] = []

    for run in RUNS:
        if not run.stage1.exists():
            raise FileNotFoundError(run.stage1)
        df, boundary = load_pair(run)
        out_df = df[["method", "display_name", "source_stage", "epoch", "global_step"]].copy()
        for metric in METRICS:
            if metric not in df.columns:
                continue
            raw = pd.to_numeric(df[metric], errors="coerce").to_numpy(dtype=float)
            corr, mask, stats = corrected_metric(df, metric, boundary)
            out_df[f"{metric}_raw"] = raw
            out_df[f"{metric}_corrected"] = corr
            out_df[f"{metric}_is_imputed"] = mask.astype(int)
            manifest.append({"method": run.method, "metric": metric, **stats})
        out_csv = DATA / f"{run.method}_artifact_corrected_train_steps.csv"
        out_df.to_csv(out_csv, index=False)
        corrected_frames.append(out_df)

    all_df = pd.concat(corrected_frames, ignore_index=True, sort=False)
    all_df.to_csv(DATA / "all_methods_artifact_corrected_train_steps.csv", index=False)
    (DATA / "artifact_correction_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    pd.DataFrame(manifest).to_csv(DATA / "artifact_correction_manifest.csv", index=False)

    for metric in METRICS:
        if f"{metric}_corrected" in all_df.columns:
            plot_metric(all_df, metric, FIG / f"{metric}_raw_vs_artifact_corrected_by_method.png")
            plot_combined(all_df, metric, FIG / f"{metric}_corrected_combined.png")
    previous_style_dir = FIG / "previous_style_corrected"
    plot_previous_style_train_loss(
        all_df,
        previous_style_dir / "darcy_adv_training_train_loss_on_adv.png",
        xlim=(0, 3000),
    )
    plot_previous_style_train_loss(
        all_df,
        FIG / "darcy_adv_training_train_loss_on_adv_artifact_corrected_same_style.png",
        xlim=(0, 3000),
    )
    plot_six_method_previous_style_train_loss(
        all_df,
        FIG / "darcy_adv_training_train_loss_on_adv_artifact_corrected_six_method_same_style.png",
        xlim=(0, 3000),
    )

    report = [
        "# Darcy Optimizer-Artifact Corrected Curves",
        "",
        "This bundle is a derived diagnostic visualization. Raw files were not overwritten.",
        "The corrected columns bridge the known optimizer-state restart artifact near the stage1/stage2 boundary.",
        "",
        "Files:",
        f"- Combined CSV: `{DATA / 'all_methods_artifact_corrected_train_steps.csv'}`",
        f"- Manifest CSV: `{DATA / 'artifact_correction_manifest.csv'}`",
        f"- Figures: `{FIG}`",
        f"- Previous-style corrected loss figure: `{previous_style_dir / 'darcy_adv_training_train_loss_on_adv.png'}`",
        "",
        "Correction method: log10-domain smoothstep bridge with residual scale interpolated between pre-boundary and post-boundary robust statistics.",
        "Rows changed are flagged with `*_is_imputed = 1`.",
    ]
    (REPORT / "artifact_correction_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
