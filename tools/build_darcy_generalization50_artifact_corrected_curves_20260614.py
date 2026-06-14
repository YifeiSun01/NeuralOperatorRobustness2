#!/usr/bin/env python3
"""Build 50-Darcy-generalization 5x5 corrected eval curve pages.

The figures mirror the SIR20 visualization layout: 50 generalization datasets
split into two 25-panel pages, with one subplot per dataset.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_darcy_eval_artifact_corrected_curves_20260614 import (
    COLORS,
    DATA,
    FIG,
    METHOD_ORDER,
    METRICS,
    PLOT_PHASES,
    RUNS,
    SPLITS,
    cumulative_wall_by_epoch_pair,
    relpath,
    setup_plot_style,
    task_dir,
    variance_matched_metric,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "darcy_generalization50_artifact_corrected_20260614"
OUT_DATA = OUT / "data"
OUT_FIG = OUT / "figures" / "previous_style_corrected"
OUT_REPORT = OUT / "reports"
REVIEW = ROOT / "outputs" / "darcy_corrected_loss_figures_review_20260614"

LABELS = {
    "loss1": "loss1",
    "loss2": "loss2",
    "loss3": "loss3",
    "physics": "physics",
}


def load_eval_metrics_pair(run) -> tuple[pd.DataFrame, int]:
    stage1 = pd.read_csv(task_dir(run.stage1) / "eval_metrics.csv")
    stage1["source_stage"] = "stage1"
    stage2 = pd.read_csv(task_dir(run.stage2) / "eval_metrics.csv")
    stage2["source_stage"] = "stage2"
    boundary = int(stage1[stage1["phase"] == "during_adversarial_training"]["epoch"].max())
    df = pd.concat([stage1, stage2], ignore_index=True, sort=False)
    df.insert(0, "method", run.method)
    return df, boundary


def apply_dataset_corrections(df: pd.DataFrame, boundary: int) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    out = df.copy()
    for metric in METRICS:
        out[f"{metric}_raw"] = pd.to_numeric(out[metric], errors="coerce")
        out[f"{metric}_corrected"] = out[f"{metric}_raw"]
        out[f"{metric}_is_imputed"] = 0

    manifest: list[dict[str, object]] = []
    plot_mask = out["phase"].isin(PLOT_PHASES) & (out["split"] == "generalization")
    dataset_ids = sorted(out.loc[plot_mask, "dataset_id"].dropna().astype(str).unique().tolist())
    for dataset_id in dataset_ids:
        for metric in METRICS:
            idx = out.index[plot_mask & (out["dataset_id"].astype(str) == dataset_id)].to_numpy()
            if idx.size == 0:
                continue
            sub = out.loc[idx].sort_values("epoch")[["epoch", metric]].reset_index(drop=True)
            corrected, mask, stats = variance_matched_metric(sub, metric, boundary)
            sorted_idx = out.loc[idx].sort_values("epoch").index
            out.loc[sorted_idx, f"{metric}_corrected"] = corrected
            out.loc[sorted_idx, f"{metric}_is_imputed"] = mask.astype(int)
            manifest.append({"dataset_id": dataset_id, "metric": metric, **stats})
    return out, manifest


def dataset_order(all_df: pd.DataFrame) -> list[str]:
    base = all_df[
        (all_df["phase"] == "baseline_before_adversarial_training")
        & (all_df["split"] == "generalization")
    ].copy()
    if base.empty:
        base = all_df[all_df["split"] == "generalization"].copy()
    base["dataset_id"] = base["dataset_id"].astype(str)
    base = base.drop_duplicates("dataset_id")

    def key(dataset_id: str) -> tuple[int, str]:
        match = re.search(r"_(\d{2})_", dataset_id)
        if match:
            return int(match.group(1)), dataset_id
        return 10_000, dataset_id

    return sorted(base["dataset_id"].tolist(), key=key)[:50]


def baseline_by_dataset(all_df: pd.DataFrame, metric: str) -> dict[str, float]:
    base = all_df[
        (all_df["phase"] == "baseline_before_adversarial_training")
        & (all_df["split"] == "generalization")
    ].copy()
    base["dataset_id"] = base["dataset_id"].astype(str)
    values = base.groupby("dataset_id")[metric].mean().to_dict()
    return {str(k): float(v) for k, v in values.items() if math.isfinite(float(v))}


def short_name(dataset_id: str) -> str:
    name = dataset_id
    for prefix in [
        "darcy_binary_loss3targeted_20260611_",
        "darcy_lossdrop_pool_",
        "darcy_",
    ]:
        if name.startswith(prefix):
            name = name[len(prefix) :]
    return name[:46]


def plot_grid(all_df: pd.DataFrame, metric: str, x_axis: str, ids: list[str], part: int, out: Path) -> None:
    x_col = "epoch" if x_axis == "epoch" else "wall_minutes"
    base = baseline_by_dataset(all_df, metric)
    plot_df = all_df[
        all_df["phase"].isin(PLOT_PHASES)
        & (all_df["split"] == "generalization")
        & all_df["dataset_id"].astype(str).isin(ids)
    ].copy()
    fig, axes = plt.subplots(5, 5, figsize=(19.0, 14.2), sharex=False, sharey=False)
    axes = axes.reshape(-1)
    for ax, dataset_id in zip(axes, ids):
        for method in METHOD_ORDER:
            sub = plot_df[
                (plot_df["method"] == method)
                & (plot_df["dataset_id"].astype(str) == dataset_id)
            ].sort_values(x_col)
            if sub.empty:
                continue
            ax.plot(sub[x_col], sub[f"{metric}_corrected"], color=COLORS[method], linewidth=0.95, alpha=0.92)
        b = base.get(dataset_id, float("nan"))
        if math.isfinite(b):
            ax.axhline(b, color="#555555", linewidth=0.75, alpha=0.62)
        ax.set_title(short_name(dataset_id), fontsize=7.0)
        ax.grid(alpha=0.2)
        ax.tick_params(labelsize=6.8)
        ax.set_yscale("log")
        if x_axis == "epoch":
            ax.set_xlim(0, 3000)
        ax.margins(x=0.01)
    for ax in axes[len(ids) :]:
        ax.axis("off")
    handles = [plt.Line2D([0], [0], color=COLORS[m], lw=1.5, label=LABELS[m]) for m in METHOD_ORDER]
    handles.append(plt.Line2D([0], [0], color="#555555", lw=1.1, label="baseline"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.972), ncol=5, frameon=False, fontsize=9)
    x_label = "epoch" if x_axis == "epoch" else "wall-clock minutes"
    fig.suptitle(
        f"Darcy generalization {metric.replace('_', ' ')} by {x_label}, datasets {1 + (part - 1) * 25}-{part * 25}",
        y=0.998,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def write_report(figures: list[Path]) -> None:
    OUT_REPORT.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Darcy 50-Generalization Corrected Curves",
        "",
        "Derived from stage1 + stage2 eval_metrics.csv files. Raw logs were not overwritten.",
        "Each page has 25 generalization datasets; rows bridged near the optimizer restart boundary are flagged in the CSV.",
        "",
    ]
    for fig in figures:
        lines.append(f"- `{relpath(fig)}`")
    (OUT_REPORT / "generalization50_figures.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    setup_plot_style()
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    frames = []
    manifest = []
    for run in RUNS:
        df, boundary = load_eval_metrics_pair(run)
        wall = cumulative_wall_by_epoch_pair(run)
        df["wall_seconds"] = df["epoch"].map(lambda e: wall.get(int(e), np.nan))
        df["wall_minutes"] = df["wall_seconds"] / 60.0
        corrected, stats = apply_dataset_corrections(df, boundary)
        corrected["run_stage1"] = relpath(run.stage1)
        corrected["run_stage2"] = relpath(run.stage2)
        corrected.to_csv(OUT_DATA / f"{run.method}_eval_metrics_artifact_corrected.csv", index=False)
        frames.append(corrected)
        for item in stats:
            manifest.append({"method": run.method, "boundary_epoch": boundary, **item})

    all_df = pd.concat(frames, ignore_index=True, sort=False)
    all_df.to_csv(OUT_DATA / "eval_metrics_artifact_corrected.csv", index=False)
    pd.DataFrame(manifest).to_csv(OUT_DATA / "generalization50_artifact_correction_manifest.csv", index=False)

    ids = dataset_order(all_df)
    figures: list[Path] = []
    for metric in METRICS:
        for x_axis in ("epoch", "work"):
            for part, subset in enumerate((ids[:25], ids[25:50]), 1):
                out = OUT_FIG / f"{metric}_generalization_part{part:02d}_vs_{x_axis}.png"
                plot_grid(all_df, metric, x_axis, subset, part, out)
                figures.append(out)

    write_report(figures)
    REVIEW.mkdir(parents=True, exist_ok=True)
    for fig in figures:
        target = REVIEW / f"generalization50_{fig.name}"
        target.write_bytes(fig.read_bytes())
    print(OUT_FIG)


if __name__ == "__main__":
    main()
