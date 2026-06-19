#!/usr/bin/env python3
"""Plot Darcy/SIR20 epsilon-vs-percent-loss-increase curves on log-log axes."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = PROJECT_ROOT / "outputs/darcy_sir20_timematched_full_serial_double_budget_full_delta_budget_records_20260617"

METHOD_ORDER = [
    "baseline",
    "loss1",
    "loss2",
    "loss3",
    "physics_loss",
    "random_clean",
    "random_solver",
]

COLORS = {
    "baseline": "#111111",
    "loss1": "#1f77b4",
    "loss2": "#ff7f0e",
    "loss3": "#2ca02c",
    "physics_loss": "#d62728",
    "random_clean": "#9467bd",
    "random_solver": "#17becf",
}

MARKERS = {
    "baseline": "o",
    "loss1": "s",
    "loss2": "^",
    "loss3": "D",
    "physics_loss": "P",
    "random_clean": "v",
    "random_solver": "X",
}


def rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def finite_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def load_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["source_summary_csv"] = rel(path)
    required = {
        "method",
        "method_display",
        "budget",
        "split",
        "sample_count",
        "attack_steps",
        "clean_loss_mean",
        "adv_loss_mean",
        "relative_increase_mean",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    for col in [
        "budget",
        "sample_count",
        "attack_steps",
        "clean_loss_mean",
        "adv_loss_mean",
        "relative_increase_mean",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    clean = df["clean_loss_mean"].astype(float)
    adv = df["adv_loss_mean"].astype(float)
    df["epsilon"] = df["budget"].astype(float)
    df["percent_loss_increase"] = np.where(
        np.isfinite(clean) & np.isfinite(adv) & (clean > 0.0),
        100.0 * (adv / clean - 1.0),
        np.nan,
    )
    df["sample_mean_percent_loss_increase"] = 100.0 * df["relative_increase_mean"].astype(float)
    return df


def best_envelope(df: pd.DataFrame) -> pd.DataFrame:
    usable = df[np.isfinite(df["percent_loss_increase"])].copy()
    usable = usable.sort_values(
        ["method", "epsilon", "percent_loss_increase"],
        ascending=[True, True, False],
        kind="mergesort",
    )
    return usable.groupby(["method", "epsilon"], as_index=False, sort=False).head(1).reset_index(drop=True)


def method_label(method: str, labels: dict[str, str]) -> str:
    return labels.get(method, method)


def method_order(df: pd.DataFrame) -> list[str]:
    present = [m for m in METHOD_ORDER if m in set(df["method"])]
    extras = sorted(set(df["method"]) - set(present))
    return present + extras


def plot_curves(df: pd.DataFrame, out_png: Path, out_pdf: Path, title: str, ylabel: str) -> dict[str, Any]:
    labels = (
        df[["method", "method_display"]]
        .dropna()
        .drop_duplicates("method")
        .set_index("method")["method_display"]
        .astype(str)
        .to_dict()
    )

    fig, ax = plt.subplots(figsize=(14.4, 8.0))
    plotted: list[str] = []
    skipped_nonpositive: list[str] = []
    y_values: list[float] = []

    for method in method_order(df):
        sub = df[df["method"] == method].sort_values("epsilon")
        sub = sub[np.isfinite(sub["epsilon"]) & np.isfinite(sub["percent_loss_increase"])]
        positive = sub[sub["percent_loss_increase"] > 0.0]
        if positive.empty:
            skipped_nonpositive.append(method)
            continue
        plotted.append(method)
        y_values.extend(positive["percent_loss_increase"].astype(float).tolist())
        ax.plot(
            positive["epsilon"].astype(float),
            positive["percent_loss_increase"].astype(float),
            marker=MARKERS.get(method, "o"),
            markersize=6.0,
            linewidth=2.2,
            color=COLORS.get(method),
            label=method_label(method, labels),
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("epsilon / attack budget fraction")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.28, linewidth=0.7)
    ax.legend(loc="best", frameon=False, fontsize=10)
    if y_values:
        low = min(y_values)
        high = max(y_values)
        ax.set_ylim(max(low * 0.65, 1e-8), high * 1.8)
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=260, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    return {
        "png": rel(out_png),
        "pdf": rel(out_pdf),
        "plotted_methods": plotted,
        "skipped_nonpositive_methods": skipped_nonpositive,
    }


def write_report(
    path: Path,
    *,
    source_csvs: list[Path],
    envelope_csv: Path,
    generalization_csv: Path,
    envelope_plot: dict[str, Any],
    generalization_plot: dict[str, Any],
    envelope_df: pd.DataFrame,
    generalization_df: pd.DataFrame,
) -> None:
    lines: list[str] = [
        "# Darcy/SIR20 Percent Loss Increase Log-Log Curves",
        "",
        "Observed from the completed Darcy/SIR20 budget sweep artifacts. No adversarial attack was rerun for this plotting pass.",
        "",
        "Definition used for the plotted y-axis:",
        "",
        "`percent loss increase = 100 * (adv_loss_mean / clean_loss_mean - 1)`",
        "",
        "The best-envelope plot takes the largest plotted percent increase among the available split summaries for the same model and epsilon. The generalization-only plot uses only the `generalization` split.",
        "",
        "Artifacts:",
        "- Source summary CSVs:",
        *[f"  - `{rel(source_csv)}`" for source_csv in source_csvs],
        f"- Best-envelope CSV: `{rel(envelope_csv)}`",
        f"- Generalization-only CSV: `{rel(generalization_csv)}`",
        f"- Best-envelope PNG: `{envelope_plot['png']}`",
        f"- Best-envelope PDF: `{envelope_plot['pdf']}`",
        f"- Generalization-only PNG: `{generalization_plot['png']}`",
        f"- Generalization-only PDF: `{generalization_plot['pdf']}`",
        "",
        "Best-envelope maxima by method:",
        "",
        "| method | max percent loss increase | epsilon at max | source split |",
        "|---|---:|---:|---|",
    ]

    for method in method_order(envelope_df):
        sub = envelope_df[envelope_df["method"] == method].copy()
        sub = sub[np.isfinite(sub["percent_loss_increase"])]
        if sub.empty:
            continue
        row = sub.sort_values("percent_loss_increase", ascending=False).iloc[0]
        label = str(row.get("method_display", method))
        lines.append(
            f"| {label} | {float(row['percent_loss_increase']):.6g} | {float(row['epsilon']):.6g} | {row['split']} |"
        )

    lines.extend(
        [
            "",
            "Generalization-only maxima by method:",
            "",
            "| method | max percent loss increase | epsilon at max |",
            "|---|---:|---:|",
        ]
    )
    for method in method_order(generalization_df):
        sub = generalization_df[generalization_df["method"] == method].copy()
        sub = sub[np.isfinite(sub["percent_loss_increase"])]
        if sub.empty:
            continue
        row = sub.sort_values("percent_loss_increase", ascending=False).iloc[0]
        label = str(row.get("method_display", method))
        lines.append(f"| {label} | {float(row['percent_loss_increase']):.6g} | {float(row['epsilon']):.6g} |")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--extra-summary-csv", type=Path, action="append", default=[])
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--docs-report", type=Path, default=None)
    parser.add_argument("--output-tag", default="")
    args = parser.parse_args()

    bundle = args.bundle.resolve()
    source_csv = (args.summary_csv or bundle / "data/budget_sweep_loss_increase_summary.csv").resolve()
    out_dir = (args.out_dir or bundle / "figures/percent_loss_increase_loglog").resolve()
    data_dir = (args.data_dir or bundle / "data/percent_loss_increase_loglog").resolve()
    report = (args.report or bundle / "reports/percent_loss_increase_loglog.md").resolve()

    source_csvs = [source_csv] + [path.resolve() for path in args.extra_summary_csv]
    df = pd.concat([load_summary(path) for path in source_csvs], ignore_index=True)
    sort_cols = ["method", "split", "epsilon", "source_summary_csv"]
    if "epsilon" not in df.columns:
        sort_cols = ["method", "split", "budget", "source_summary_csv"]
    df = df.sort_values(sort_cols, kind="mergesort").drop_duplicates(["method", "split", "budget"], keep="last")
    envelope_df = best_envelope(df)
    generalization_df = df[df["split"].astype(str) == "generalization"].copy()

    tag = f"{args.output_tag.strip('_')}_" if args.output_tag.strip("_") else ""
    envelope_csv = data_dir / f"{tag}percent_loss_increase_best_envelope.csv"
    generalization_csv = data_dir / f"{tag}percent_loss_increase_generalization.csv"
    data_dir.mkdir(parents=True, exist_ok=True)
    envelope_df.to_csv(envelope_csv, index=False)
    generalization_df.to_csv(generalization_csv, index=False)

    envelope_plot = plot_curves(
        envelope_df,
        out_dir / f"{tag}epsilon_vs_percent_loss_increase_best_envelope_loglog.png",
        out_dir / f"{tag}epsilon_vs_percent_loss_increase_best_envelope_loglog.pdf",
        "Darcy/SIR20 epsilon vs percent loss increase: best envelope, log-log",
        "best batch-mean percent loss increase: 100 * (final / initial - 1)",
    )
    generalization_plot = plot_curves(
        generalization_df,
        out_dir / f"{tag}epsilon_vs_percent_loss_increase_generalization_loglog.png",
        out_dir / f"{tag}epsilon_vs_percent_loss_increase_generalization_loglog.pdf",
        "Darcy/SIR20 epsilon vs percent loss increase: generalization, log-log",
        "batch-mean percent loss increase: 100 * (final / initial - 1)",
    )

    manifest = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_csvs": [rel(path) for path in source_csvs],
        "definition": "percent_loss_increase = 100 * (adv_loss_mean / clean_loss_mean - 1)",
        "best_envelope_definition": "max percent_loss_increase over available split summaries for each method and epsilon",
        "envelope_csv": rel(envelope_csv),
        "generalization_csv": rel(generalization_csv),
        "envelope_plot": envelope_plot,
        "generalization_plot": generalization_plot,
    }
    manifest_path = data_dir / f"{tag}percent_loss_increase_loglog_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    write_report(
        report,
        source_csvs=source_csvs,
        envelope_csv=envelope_csv,
        generalization_csv=generalization_csv,
        envelope_plot=envelope_plot,
        generalization_plot=generalization_plot,
        envelope_df=envelope_df,
        generalization_df=generalization_df,
    )
    if args.docs_report is not None:
        write_report(
            args.docs_report.resolve(),
            source_csvs=source_csvs,
            envelope_csv=envelope_csv,
            generalization_csv=generalization_csv,
            envelope_plot=envelope_plot,
            generalization_plot=generalization_plot,
            envelope_df=envelope_df,
            generalization_df=generalization_df,
        )

    print(json.dumps({"manifest": rel(manifest_path), "report": rel(report), **manifest}, indent=2))


if __name__ == "__main__":
    main()
