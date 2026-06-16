#!/usr/bin/env python3
"""Build transparent raw-vs-artifact-corrected Darcy audit figures."""

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


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "darcy_raw_vs_corrected_audit_20260614"
FIG = OUT / "figures"
DATA = OUT / "data"
REPORT = OUT / "reports"

SPLIT_CSV = ROOT / "outputs/darcy_eval_artifact_corrected_20260614/data/eval_split_summary_artifact_corrected.csv"
METRICS_CSV = ROOT / "outputs/darcy_generalization50_artifact_corrected_20260614/data/eval_metrics_artifact_corrected.csv"

METHOD_ORDER = ["loss1", "loss2", "loss3", "physics"]
LABELS = {"loss1": "loss1", "loss2": "loss2", "loss3": "loss3", "physics": "physics"}
COLORS = {"loss1": "#1b6ca8", "loss2": "#d95f02", "loss3": "#2ca25f", "physics": "#7b3294"}
METRICS = ["rmse", "relative_l2"]
SPLITS = ["train", "test", "generalization"]
PLOT_PHASES = {"baseline_before_adversarial_training", "during_adversarial_training"}


def setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 220,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.facecolor": "white",
            "figure.facecolor": "#fbfaf7",
            "axes.edgecolor": "#aaa59b",
            "axes.grid": True,
            "grid.color": "#d8d4c8",
            "grid.alpha": 0.42,
            "grid.linewidth": 0.55,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "axes.titleweight": "semibold",
            "axes.titlepad": 7.0,
        }
    )


def metric_label(metric: str) -> str:
    return "RMSE" if metric == "rmse" else "Relative L2"


def baseline_split_values(df: pd.DataFrame, metric: str, value_kind: str) -> dict[str, float]:
    col = f"{metric}_{value_kind}"
    base = df[(df["phase"] == "baseline_before_adversarial_training") & df["split"].isin(SPLITS)]
    return {str(r.split): float(getattr(r, col)) for r in base.drop_duplicates("split").itertuples(index=False)}


def artifact_spans(df: pd.DataFrame, metric: str) -> list[tuple[int, int]]:
    flag = f"{metric}_is_imputed"
    if flag not in df.columns:
        return []
    spans = []
    flagged = df[pd.to_numeric(df[flag], errors="coerce").fillna(0).astype(int) == 1]
    for method, sub in flagged.groupby("method", sort=False):
        if sub.empty:
            continue
        spans.append((int(sub["epoch"].min()), int(sub["epoch"].max())))
    return spans


def y_limits_for_pair(df: pd.DataFrame, metric: str, split: str | None = None, dataset_id: str | None = None) -> tuple[float, float] | None:
    sub = df[df["phase"].isin(PLOT_PHASES)].copy()
    if split is not None:
        sub = sub[sub["split"] == split]
    if dataset_id is not None:
        sub = sub[sub["dataset_id"].astype(str) == dataset_id]
    vals = []
    for col in (f"{metric}_raw", f"{metric}_corrected"):
        if col in sub.columns:
            vals.append(pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=float))
    if not vals:
        return None
    arr = np.concatenate(vals)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if arr.size == 0:
        return None
    lo = float(np.nanpercentile(arr, 0.8))
    hi = float(np.nanpercentile(arr, 99.2))
    if not math.isfinite(lo) or not math.isfinite(hi) or lo <= 0 or hi <= lo:
        lo = float(np.nanmin(arr))
        hi = float(np.nanmax(arr))
    pad = 0.10
    return lo / (1.0 + pad), hi * (1.0 + pad)


def plot_split_raw_vs_corrected(df: pd.DataFrame, metric: str, out: Path) -> None:
    plot_df = df[df["phase"].isin(PLOT_PHASES) & df["split"].isin(SPLITS)].copy()
    fig, axes = plt.subplots(2, 3, figsize=(22.5, 10.5), sharex=True, sharey=False)
    panel_specs = [("raw", "RAW old logs"), ("corrected", "DERIVED artifact-corrected")]
    for row_idx, (value_kind, row_title) in enumerate(panel_specs):
        base = baseline_split_values(df, metric, value_kind)
        for col_idx, split in enumerate(SPLITS):
            ax = axes[row_idx, col_idx]
            split_df = plot_df[plot_df["split"] == split]
            for method in METHOD_ORDER:
                sub = split_df[split_df["method"] == method].sort_values("epoch")
                if sub.empty:
                    continue
                ax.plot(sub["epoch"], sub[f"{metric}_{value_kind}"], color=COLORS[method], lw=1.15, alpha=0.94, label=LABELS[method])
            b = base.get(split)
            if b is not None and math.isfinite(b):
                ax.axhline(b, color="#555555", linewidth=0.8, alpha=0.68)
            for lo, hi in artifact_spans(split_df, metric):
                ax.axvspan(lo, hi, color="#f2c94c", alpha=0.13, linewidth=0)
            ax.set_title(f"{row_title} - {split if split != 'generalization' else 'generalization mean'}", fontsize=11.3)
            ax.set_xlabel("epoch")
            ax.set_ylabel(metric_label(metric))
            ax.set_yscale("log")
            ylim = y_limits_for_pair(plot_df, metric, split=split)
            if ylim is not None:
                ax.set_ylim(*ylim)
            ax.set_xlim(0, 3150)
            ax.tick_params(labelsize=8.7)
    handles = [plt.Line2D([0], [0], color=COLORS[m], lw=1.7, label=LABELS[m]) for m in METHOD_ORDER]
    handles.extend(
        [
            plt.Line2D([0], [0], color="#555555", lw=1.1, label="baseline"),
            plt.Rectangle((0, 0), 1, 1, color="#f2c94c", alpha=0.18, label="corrected interval"),
        ]
    )
    fig.suptitle(f"Darcy {metric_label(metric)} raw vs derived artifact-corrected audit", fontsize=16.0, y=0.986)
    fig.text(
        0.5,
        0.949,
        "Left-row source values are raw old logs; right-row values are derived visualization columns, not raw experiment measurements.",
        ha="center",
        fontsize=10.2,
        color="#4f4b45",
    )
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.922), ncol=6, fontsize=8.8, handlelength=2.4)
    fig.subplots_adjust(top=0.86, bottom=0.07, left=0.055, right=0.99, hspace=0.34, wspace=0.23)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.18, facecolor=fig.get_facecolor())
    plt.close(fig)


def dataset_order(df: pd.DataFrame) -> list[str]:
    base = df[(df["phase"] == "baseline_before_adversarial_training") & (df["split"] == "generalization")].copy()
    base["dataset_id"] = base["dataset_id"].astype(str)

    def key(dataset_id: str) -> tuple[int, str]:
        match = re.search(r"_(\d{2})_", dataset_id)
        return (int(match.group(1)) if match else 10_000, dataset_id)

    return sorted(base["dataset_id"].drop_duplicates().tolist(), key=key)[:50]


def short_name(dataset_id: str) -> str:
    for prefix in ["darcy_binary_loss3targeted_20260611_", "darcy_"]:
        if dataset_id.startswith(prefix):
            dataset_id = dataset_id[len(prefix) :]
    name = dataset_id[:38]
    if len(dataset_id) > 38:
        name = f"{name}..."
    return name


def baseline_dataset_values(df: pd.DataFrame, metric: str, value_kind: str) -> dict[str, float]:
    col = f"{metric}_{value_kind}"
    base = df[(df["phase"] == "baseline_before_adversarial_training") & (df["split"] == "generalization")].copy()
    base["dataset_id"] = base["dataset_id"].astype(str)
    return base.groupby("dataset_id")[col].mean().to_dict()


def plot_grid_row(
    axes: np.ndarray,
    df: pd.DataFrame,
    metric: str,
    value_kind: str,
    ids: list[str],
    row_title: str,
) -> None:
    base = baseline_dataset_values(df, metric, value_kind)
    plot_df = df[df["phase"].isin(PLOT_PHASES) & (df["split"] == "generalization") & df["dataset_id"].astype(str).isin(ids)].copy()
    for ax, dataset_id in zip(axes, ids):
        sub_dataset = plot_df[plot_df["dataset_id"].astype(str) == dataset_id]
        for method in METHOD_ORDER:
            sub = sub_dataset[sub_dataset["method"] == method].sort_values("epoch")
            if sub.empty:
                continue
            ax.plot(sub["epoch"], sub[f"{metric}_{value_kind}"], color=COLORS[method], linewidth=0.78, alpha=0.92)
        b = base.get(dataset_id)
        if b is not None and math.isfinite(float(b)):
            ax.axhline(float(b), color="#555555", linewidth=0.62, alpha=0.62)
        for lo, hi in artifact_spans(sub_dataset, metric):
            ax.axvspan(lo, hi, color="#f2c94c", alpha=0.12, linewidth=0)
        ax.set_title(short_name(dataset_id), fontsize=6.4)
        ax.set_yscale("log")
        ylim = y_limits_for_pair(df, metric, split="generalization", dataset_id=dataset_id)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_xlim(0, 3150)
        ax.grid(alpha=0.18)
        ax.tick_params(labelsize=5.8)
    axes[0].set_ylabel(f"{row_title}\n{metric_label(metric)}", fontsize=8.0)


def plot_generalization_raw_vs_corrected(df: pd.DataFrame, metric: str, ids: list[str], part: int, out: Path) -> None:
    fig, axes = plt.subplots(10, 5, figsize=(23.5, 25.6), sharex=False, sharey=False)
    raw_axes = axes[:5].reshape(-1)
    corr_axes = axes[5:].reshape(-1)
    plot_grid_row(raw_axes, df, metric, "raw", ids, "RAW")
    plot_grid_row(corr_axes, df, metric, "corrected", ids, "DERIVED")
    handles = [plt.Line2D([0], [0], color=COLORS[m], lw=1.5, label=LABELS[m]) for m in METHOD_ORDER]
    handles.extend(
        [
            plt.Line2D([0], [0], color="#555555", lw=1.0, label="baseline"),
            plt.Rectangle((0, 0), 1, 1, color="#f2c94c", alpha=0.18, label="corrected interval"),
        ]
    )
    fig.suptitle(
        f"Darcy generalization {metric_label(metric)} raw vs derived artifact-corrected, datasets {1 + (part - 1) * 25}-{part * 25}",
        fontsize=15.5,
        y=0.994,
    )
    fig.text(
        0.5,
        0.974,
        "Top 5 rows: raw old logs. Bottom 5 rows: derived artifact-corrected visualization columns.",
        ha="center",
        fontsize=9.6,
        color="#4f4b45",
    )
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.958), ncol=6, fontsize=8.4, handlelength=2.3)
    fig.text(0.5, 0.022, "epoch", ha="center", va="center", fontsize=10)
    fig.subplots_adjust(top=0.935, bottom=0.045, left=0.045, right=0.992, hspace=0.62, wspace=0.24)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight", pad_inches=0.18, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_generalization_paired_raw_vs_corrected(df: pd.DataFrame, metric: str, ids: list[str], part: int, out: Path) -> None:
    fig, axes = plt.subplots(5, 10, figsize=(29.5, 15.8), sharex=False, sharey=False)
    base_raw = baseline_dataset_values(df, metric, "raw")
    base_corr = baseline_dataset_values(df, metric, "corrected")
    plot_df = df[df["phase"].isin(PLOT_PHASES) & (df["split"] == "generalization") & df["dataset_id"].astype(str).isin(ids)].copy()
    for idx, dataset_id in enumerate(ids):
        row = idx // 5
        pair_col = idx % 5
        raw_ax = axes[row, pair_col * 2]
        corr_ax = axes[row, pair_col * 2 + 1]
        sub_dataset = plot_df[plot_df["dataset_id"].astype(str) == dataset_id]
        ylim = y_limits_for_pair(df, metric, split="generalization", dataset_id=dataset_id)
        for ax, value_kind, title_prefix, base in [
            (raw_ax, "raw", "raw", base_raw),
            (corr_ax, "corrected", "corrected", base_corr),
        ]:
            for method in METHOD_ORDER:
                sub = sub_dataset[sub_dataset["method"] == method].sort_values("epoch")
                if sub.empty:
                    continue
                ax.plot(sub["epoch"], sub[f"{metric}_{value_kind}"], color=COLORS[method], linewidth=0.82, alpha=0.92)
            b = base.get(dataset_id)
            if b is not None and math.isfinite(float(b)):
                ax.axhline(float(b), color="#555555", linewidth=0.6, alpha=0.62)
            for lo, hi in artifact_spans(sub_dataset, metric):
                ax.axvspan(lo, hi, color="#f2c94c", alpha=0.13, linewidth=0)
            if ylim is not None:
                ax.set_ylim(*ylim)
            ax.set_xlim(0, 3150)
            ax.set_yscale("log")
            ax.grid(alpha=0.18)
            ax.tick_params(labelsize=5.6)
            ax.set_title(title_prefix, fontsize=6.7, pad=3)
        raw_ax.set_ylabel(metric_label(metric), fontsize=7.0)
        raw_ax.text(
            0.0,
            1.23,
            short_name(dataset_id),
            transform=raw_ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=6.5,
            fontweight="semibold",
            clip_on=False,
        )
    handles = [plt.Line2D([0], [0], color=COLORS[m], lw=1.5, label=LABELS[m]) for m in METHOD_ORDER]
    handles.extend(
        [
            plt.Line2D([0], [0], color="#555555", lw=1.0, label="baseline"),
            plt.Rectangle((0, 0), 1, 1, color="#f2c94c", alpha=0.18, label="corrected interval"),
        ]
    )
    fig.suptitle(
        f"Darcy generalization {metric_label(metric)} paired raw/corrected audit, datasets {1 + (part - 1) * 25}-{part * 25}",
        fontsize=15.5,
        y=0.994,
    )
    fig.text(
        0.5,
        0.972,
        "Each dataset is an adjacent pair: left raw old logs, right derived artifact-corrected visualization column.",
        ha="center",
        fontsize=9.5,
        color="#4f4b45",
    )
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.952), ncol=6, fontsize=8.3, handlelength=2.2)
    fig.text(0.5, 0.022, "epoch", ha="center", va="center", fontsize=10)
    fig.subplots_adjust(top=0.905, bottom=0.055, left=0.04, right=0.992, hspace=0.72, wspace=0.18)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight", pad_inches=0.18, facecolor=fig.get_facecolor())
    plt.close(fig)


def write_report(figures: list[Path]) -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Darcy Raw vs Artifact-Corrected Audit Figures",
        "",
        "These figures are transparency/audit plots. They compare raw old logs with derived artifact-corrected visualization columns.",
        "",
        "- Raw experiment CSVs were not overwritten.",
        "- Yellow spans mark rows where corrected columns were imputed.",
        "- The corrected panels must not be represented as raw experiment measurements.",
        "",
        "## Figures",
        "",
    ]
    for fig in figures:
        lines.append(f"- `{fig.relative_to(ROOT)}`")
    (REPORT / "raw_vs_corrected_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-only", action="store_true", help="Only generate adjacent raw/corrected paired 25-dataset grids.")
    args = parser.parse_args()

    setup_plot_style()
    FIG.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    split_df = pd.read_csv(SPLIT_CSV)
    metrics_df = pd.read_csv(METRICS_CSV)
    if not args.paired_only:
        split_df.to_csv(DATA / "source_eval_split_summary_artifact_corrected.csv", index=False)
        metrics_df.to_csv(DATA / "source_eval_metrics_artifact_corrected.csv", index=False)

    figures: list[Path] = []
    if not args.paired_only:
        for metric in METRICS:
            out = FIG / f"raw_vs_corrected_epoch_{metric}_train_test_generalization.png"
            plot_split_raw_vs_corrected(split_df, metric, out)
            figures.append(out)

    ids = dataset_order(metrics_df)
    for metric in METRICS:
        for part, subset in enumerate((ids[:25], ids[25:50]), 1):
            if not args.paired_only:
                out = FIG / f"raw_vs_corrected_{metric}_generalization_part{part:02d}_epoch.png"
                plot_generalization_raw_vs_corrected(metrics_df, metric, subset, part, out)
                figures.append(out)
            paired_out = FIG / f"paired_raw_vs_corrected_{metric}_generalization_part{part:02d}_epoch.png"
            plot_generalization_paired_raw_vs_corrected(metrics_df, metric, subset, part, paired_out)
            figures.append(paired_out)

    manifest = {
        "split_csv": str(SPLIT_CSV.relative_to(ROOT)),
        "metrics_csv": str(METRICS_CSV.relative_to(ROOT)),
        "figures": [str(p.relative_to(ROOT)) for p in figures],
        "note": "Audit comparison only: corrected columns are derived visualization values, not raw experiment measurements.",
    }
    (DATA / "raw_vs_corrected_audit_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_report(figures)
    print(FIG)


if __name__ == "__main__":
    main()
