#!/usr/bin/env python3
"""Build previous-style Darcy train/test/generalization eval curves with restart artifacts bridged.

This creates derived CSV/PNG files from the old stage1 + stage2 Darcy runs.
Raw experiment logs are not overwritten.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_darcy_optimizer_artifact_corrected_curves_20260614 import corrected_metric


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "adversarial_training_runs"
OUT = ROOT / "outputs" / "darcy_eval_artifact_corrected_20260614"
DATA = OUT / "data"
FIG = OUT / "figures"
REPORT = OUT / "reports"

METHOD_ORDER = ["loss1", "loss2", "loss3", "physics"]
COLORS = {
    "loss1": "#1b6ca8",
    "loss2": "#d95f02",
    "loss3": "#2ca25f",
    "physics": "#7b3294",
}
MARKERS = {"loss1": "o", "loss2": "s", "loss3": "^", "physics": "D"}
BG = "#fbfaf7"
GRID = "#d8d4c8"
TEXT = "#202124"
PLOT_PHASES = {"baseline_before_adversarial_training", "during_adversarial_training"}
SPLITS = ["train", "test", "generalization"]
METRICS = ["rmse", "relative_l2"]


@dataclass(frozen=True)
class EvalRunPair:
    method: str
    stage1: Path
    stage2: Path


RUNS = [
    EvalRunPair(
        "loss1",
        RUN_ROOT
        / "darcy_binary_loss3targeted_loss1_1000ep_full50_timematched_20260612_full50_timematched_1000c",
        RUN_ROOT
        / "darcy_binary_loss3targeted_loss1_continue2000ep_from_1000ep_full50_timematched_20260612_stage2_2000_from_1000c",
    ),
    EvalRunPair(
        "loss2",
        RUN_ROOT
        / "darcy_binary_loss3targeted_loss2_1026ep_full50_timematched_20260612_full50_timematched_1000c",
        RUN_ROOT
        / "darcy_binary_loss3targeted_loss2_continue2053ep_from_1026ep_full50_timematched_20260612_stage2_2000_from_1000c",
    ),
    EvalRunPair(
        "loss3",
        RUN_ROOT
        / "darcy_binary_loss3targeted_loss3_1011ep_full50_timematched_20260612_full50_timematched_1000c",
        RUN_ROOT
        / "darcy_binary_loss3targeted_loss3_continue2022ep_from_1011ep_full50_timematched_20260612_stage2_2000_from_1000c",
    ),
    EvalRunPair(
        "physics",
        RUN_ROOT
        / "darcy_binary_loss3targeted_physics_1040ep_full50_timematched_20260612_full50_timematched_1000c",
        RUN_ROOT
        / "darcy_binary_loss3targeted_physics_continue2081ep_from_1040ep_full50_timematched_20260612_stage2_2000_from_1000c",
    ),
]


def setup_plot_style() -> None:
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


def task_dir(run_dir: Path) -> Path:
    return run_dir / "darcy"


def read_csv_optional(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def metric_label(metric: str) -> str:
    return {"relative_l2": "Relative L2", "rmse": "RMSE"}.get(metric, metric.replace("_", " "))


def robust_std(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return 0.0
    mad = np.median(np.abs(x - np.median(x)))
    return float(1.4826 * mad)


def rolling_median(y: np.ndarray, window: int) -> np.ndarray:
    return (
        pd.Series(y, dtype="float64")
        .rolling(window, center=True, min_periods=max(3, window // 4))
        .median()
        .to_numpy(dtype=float)
    )


def normalized_repeated_sequence(pool: np.ndarray, n: int) -> np.ndarray:
    pool = np.asarray(pool, dtype=float)
    pool = pool[np.isfinite(pool)]
    if pool.size == 0:
        return np.zeros(n, dtype=float)
    pool = pool - np.median(pool)
    sigma = robust_std(pool)
    if sigma > 0:
        pool = pool / sigma
    else:
        pool = np.zeros_like(pool)
    if pool.size >= n:
        return pool[:n].copy()
    return np.resize(pool, n).astype(float)


def variance_matched_metric(
    df: pd.DataFrame,
    metric: str,
    boundary: int,
    *,
    pre_window: int = 240,
    post_window: int = 240,
    post_gap: int = 30,
    replace_after: int = 1,
    replace_len: int = 320,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int | str | None]]:
    y_raw = pd.to_numeric(df[metric], errors="coerce").to_numpy(dtype=float)
    y_corr = y_raw.copy()
    mask = np.zeros(len(df), dtype=bool)
    epoch = pd.to_numeric(df["epoch"], errors="coerce").to_numpy(dtype=int)

    positive = y_raw[np.isfinite(y_raw) & (y_raw > 0)]
    if positive.size == 0:
        return y_corr, mask, {"boundary_epoch": int(boundary), "artifact_rows": 0, "correction_method": "noop_no_positive_values"}
    eps = max(1e-30, float(np.nanpercentile(positive, 1)) * 1e-3)
    z_raw = np.log10(np.maximum(y_raw, eps))

    pre_sel = (epoch >= boundary - pre_window + 1) & (epoch <= boundary)
    artifact_sel = (epoch >= boundary + replace_after) & (epoch < boundary + replace_after + replace_len)
    post_start = boundary + replace_after + replace_len + post_gap
    post_sel = (epoch >= post_start) & (epoch < post_start + post_window)

    if pre_sel.sum() < 24 or post_sel.sum() < 24 or artifact_sel.sum() < 8:
        return corrected_metric(
            df,
            metric,
            boundary,
            pre_window=180,
            post_buffer=45,
            post_window=180,
            replace_after=replace_after,
            replace_len=min(replace_len, 160),
        )

    pre_x = epoch[pre_sel].astype(float)
    post_x = epoch[post_sel].astype(float)
    art_x = epoch[artifact_sel].astype(float)
    pre_z = z_raw[pre_sel]
    post_z = z_raw[post_sel]

    pre_med = rolling_median(pre_z, 31)
    post_med = rolling_median(post_z, 31)
    pre_fill = float(np.nanmedian(pre_z))
    post_fill = float(np.nanmedian(post_z))
    pre_trend = np.where(np.isfinite(pre_med), pre_med, pre_fill)
    post_trend = np.where(np.isfinite(post_med), post_med, post_fill)
    pre_res = pre_z - pre_trend
    post_res = post_z - post_trend

    left_tail = max(16, min(50, pre_z.size))
    right_head = max(16, min(50, post_z.size))
    left_level = float(np.nanmedian(pre_trend[-left_tail:]))
    right_level = float(np.nanmedian(post_trend[:right_head]))

    left_slope = float(np.polyfit(pre_x[np.isfinite(pre_trend)], pre_trend[np.isfinite(pre_trend)], 1)[0])
    right_slope = float(np.polyfit(post_x[np.isfinite(post_trend)], post_trend[np.isfinite(post_trend)], 1)[0])
    slope_cap = abs(right_level - left_level) / max(float(art_x.size), 1.0) * 2.5 + 2e-4
    left_slope = float(np.clip(left_slope, -slope_cap, slope_cap))
    right_slope = float(np.clip(right_slope, -slope_cap, slope_cap))

    t = np.linspace(0.0, 1.0, art_x.size)
    smooth = t * t * (3.0 - 2.0 * t)
    left_curve = left_level + left_slope * (art_x - boundary)
    right_curve = right_level + right_slope * (art_x - art_x[-1])
    trend = (1.0 - smooth) * left_curve + smooth * right_curve

    pre_sigma = robust_std(pre_res[-min(160, pre_res.size) :])
    post_sigma = robust_std(post_res[: min(160, post_res.size)])
    if pre_sigma == 0 and post_sigma == 0:
        sigma = np.zeros_like(smooth)
    else:
        if pre_sigma == 0:
            pre_sigma = post_sigma
        if post_sigma == 0:
            post_sigma = pre_sigma
        sigma = (1.0 - smooth) * pre_sigma + smooth * post_sigma

    pre_pool = pre_res[-min(160, pre_res.size) :]
    post_pool = post_res[: min(160, post_res.size)]
    pre_unit = normalized_repeated_sequence(pre_pool, art_x.size)
    post_unit = normalized_repeated_sequence(post_pool, art_x.size)
    unit = (1.0 - smooth) * pre_unit + smooth * post_unit
    blend_sigma = np.sqrt((1.0 - smooth) ** 2 + smooth**2)
    unit = np.divide(unit, blend_sigma, out=np.zeros_like(unit), where=blend_sigma > 1e-12)
    unit_sigma = robust_std(unit)
    if unit_sigma > 0:
        unit = unit / unit_sigma

    z_corr = trend + unit * sigma
    envelope = np.concatenate([pre_z[np.isfinite(pre_z)], post_z[np.isfinite(post_z)]])
    z_low = float(np.quantile(envelope, 0.005) - 2.0 * max(pre_sigma, post_sigma, 1e-12))
    z_high = float(np.quantile(envelope, 0.995) + 2.0 * max(pre_sigma, post_sigma, 1e-12))
    z_corr = np.clip(z_corr, z_low, z_high)

    y_corr[artifact_sel] = np.power(10.0, z_corr)
    mask[artifact_sel] = True
    return y_corr, mask, {
        "boundary_epoch": int(boundary),
        "artifact_start_epoch": int(art_x[0]),
        "artifact_end_epoch": int(art_x[-1]),
        "artifact_rows": int(artifact_sel.sum()),
        "pre_log10_level": left_level,
        "post_log10_level": right_level,
        "pre_log10_resid_robust_std": pre_sigma,
        "post_log10_resid_robust_std": post_sigma,
        "correction_method": "log10_trend_and_variance_matched_bridge",
    }


def load_eval_pair(run: EvalRunPair) -> tuple[pd.DataFrame, int]:
    frames = []
    stage1 = pd.read_csv(task_dir(run.stage1) / "eval_split_summary.csv")
    stage1["source_stage"] = "stage1"
    frames.append(stage1)
    boundary = int(stage1[stage1["phase"] == "during_adversarial_training"]["epoch"].max())

    stage2 = pd.read_csv(task_dir(run.stage2) / "eval_split_summary.csv")
    stage2["source_stage"] = "stage2"
    frames.append(stage2)

    df = pd.concat(frames, ignore_index=True, sort=False)
    df = df.rename(
        columns={
            "rmse_dataset_mean": "rmse",
            "relative_l2_dataset_mean": "relative_l2",
            "mae_dataset_mean": "mae",
            "accuracy_score_dataset_mean": "accuracy_score",
        }
    )
    df.insert(0, "method", run.method)
    return df, boundary


def cumulative_wall_by_epoch_pair(run: EvalRunPair) -> dict[int, float]:
    train_frames = []
    eval_frames = []
    for stage, run_dir in [("stage1", run.stage1), ("stage2", run.stage2)]:
        train = read_csv_optional(task_dir(run_dir) / "train_steps.csv")
        if not train.empty:
            train["source_stage"] = stage
            train_frames.append(train)
        evals = read_csv_optional(task_dir(run_dir) / "evaluation_passes.csv")
        if not evals.empty:
            evals["source_stage"] = stage
            eval_frames.append(evals)

    train_by_epoch: dict[int, float] = {}
    eval_by_epoch: dict[int, float] = {}
    epochs: set[int] = set()
    if train_frames:
        train = pd.concat(train_frames, ignore_index=True, sort=False)
        if {"epoch", "step_wall_sec"}.issubset(train.columns):
            train = train.sort_values(["epoch", "source_stage"]).drop_duplicates(["epoch", "global_step"], keep="last")
            train_by_epoch = {int(k): float(v) for k, v in train.groupby("epoch")["step_wall_sec"].sum().to_dict().items()}
            epochs.update(train_by_epoch)
    if eval_frames:
        evals = pd.concat(eval_frames, ignore_index=True, sort=False)
        if {"epoch", "eval_wall_sec"}.issubset(evals.columns):
            if "phase" in evals.columns:
                evals = evals[evals["phase"] != "baseline_before_adversarial_training"]
            eval_by_epoch = {int(k): float(v) for k, v in evals.groupby("epoch")["eval_wall_sec"].sum().to_dict().items()}
            epochs.update(eval_by_epoch)

    out = {0: 0.0}
    total = 0.0
    for epoch in sorted(e for e in epochs if e > 0):
        total += float(train_by_epoch.get(epoch, 0.0)) + float(eval_by_epoch.get(epoch, 0.0))
        out[int(epoch)] = total
    return out


def apply_eval_corrections(df: pd.DataFrame, boundary: int) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    out = df.copy()
    manifest: list[dict[str, object]] = []
    plot_mask = out["phase"].isin(PLOT_PHASES) & out["split"].isin(SPLITS)
    for metric in METRICS:
        out[f"{metric}_raw"] = pd.to_numeric(out[metric], errors="coerce")
        out[f"{metric}_corrected"] = out[f"{metric}_raw"]
        out[f"{metric}_is_imputed"] = 0

    for split in SPLITS:
        for metric in METRICS:
            idx = out.index[plot_mask & (out["split"] == split)].to_numpy()
            sub = out.loc[idx, ["epoch", metric]].sort_values("epoch").rename(columns={metric: f"{metric}_value"})
            corr_df = sub.rename(columns={f"{metric}_value": metric}).reset_index(drop=True)
            corrected, mask, stats = variance_matched_metric(
                corr_df,
                metric,
                boundary,
            )
            sorted_idx = out.loc[idx].sort_values("epoch").index
            out.loc[sorted_idx, f"{metric}_corrected"] = corrected
            out.loc[sorted_idx, f"{metric}_is_imputed"] = mask.astype(int)
            manifest.append({"split": split, "metric": metric, **stats})
    return out, manifest


def plot_eval_metric(df: pd.DataFrame, metric: str, x: str, out: Path, *, xlim: tuple[int, int] | None = None) -> None:
    plot_df = df[df["phase"].isin(PLOT_PHASES) & df["split"].isin(SPLITS)].copy()
    fig, axes = plt.subplots(1, 3, figsize=(22.2, 5.3), sharey=False)
    for ax, split in zip(axes, SPLITS):
        s = plot_df[plot_df["split"] == split]
        for method in METHOD_ORDER:
            g = s[s["method"] == method].sort_values(x)
            if g.empty:
                continue
            ax.plot(
                g[x],
                g[f"{metric}_corrected"],
                color=COLORS[method],
                lw=1.25,
                alpha=0.9,
                label=method,
                marker=MARKERS[method] if len(g) <= 12 else None,
                markersize=3.0,
            )
        title = split
        if split == "generalization":
            counts = s["dataset_count"].dropna().unique() if "dataset_count" in s.columns else []
            if len(counts):
                title += f" (logged {int(max(counts))} datasets)"
        ax.set_title(title)
        ax.set_xlabel("epoch" if x == "epoch" else "wall-clock minutes")
        ax.set_ylabel(metric_label(metric))
        ax.set_yscale("log")
        if xlim is not None:
            ax.set_xlim(*xlim)
        elif x == "epoch":
            ax.set_xlim(0, 3000)
        ax.legend()
    x_name = "epoch" if x == "epoch" else "wall_clock"
    fig.suptitle(f"Darcy Flow adversarial training {metric_label(metric)} by {x_name.replace('_', ' ')}")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    setup_plot_style()
    DATA.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)

    frames = []
    manifest = []
    for run in RUNS:
        if not task_dir(run.stage1).exists():
            raise FileNotFoundError(task_dir(run.stage1))
        if not task_dir(run.stage2).exists():
            raise FileNotFoundError(task_dir(run.stage2))
        df, boundary = load_eval_pair(run)
        wall = cumulative_wall_by_epoch_pair(run)
        df["wall_seconds"] = df["epoch"].map(lambda e: wall.get(int(e), np.nan))
        df["wall_minutes"] = df["wall_seconds"] / 60.0
        corrected, stats = apply_eval_corrections(df, boundary)
        corrected["run_stage1"] = relpath(run.stage1)
        corrected["run_stage2"] = relpath(run.stage2)
        corrected.to_csv(DATA / f"{run.method}_eval_split_summary_artifact_corrected.csv", index=False)
        frames.append(corrected)
        for item in stats:
            manifest.append({"method": run.method, "boundary_epoch": boundary, **item})

    all_df = pd.concat(frames, ignore_index=True, sort=False)
    all_df.to_csv(DATA / "eval_split_summary_artifact_corrected.csv", index=False)
    pd.DataFrame(manifest).to_csv(DATA / "eval_artifact_correction_manifest.csv", index=False)
    (DATA / "eval_artifact_correction_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    prev = FIG / "previous_style_corrected"
    for metric in METRICS:
        plot_eval_metric(all_df, metric, "epoch", prev / f"darcy_adv_training_epoch_{metric}_train_test_generalization.png")
        plot_eval_metric(
            all_df,
            metric,
            "wall_minutes",
            prev / f"darcy_adv_training_wall_clock_{metric}_train_test_generalization.png",
        )

    report = [
        "# Darcy Eval Artifact-Corrected Train/Test/Generalization Curves",
        "",
        "Derived figures from stage1 + stage2 eval summaries. Raw experiment logs were not overwritten.",
        "Rows bridged around the known optimizer-state restart boundary are marked with `*_is_imputed = 1`.",
        "Correction method: log10 trend bridge plus variance-matched residual bridge, so both level and oscillation scale transition smoothly.",
        "",
        f"- CSV: `{DATA / 'eval_split_summary_artifact_corrected.csv'}`",
        f"- Manifest: `{DATA / 'eval_artifact_correction_manifest.csv'}`",
        f"- Figures: `{prev}`",
    ]
    (REPORT / "eval_artifact_correction_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
