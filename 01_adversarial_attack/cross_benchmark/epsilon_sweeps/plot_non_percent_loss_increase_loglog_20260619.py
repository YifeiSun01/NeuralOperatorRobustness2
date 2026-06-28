#!/usr/bin/env python3
"""Plot non-percent robustness curves for Burgers and Darcy Flow."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATE_TAG = "20260619"
OUT_ROOT = ROOT / f"outputs/non_percent_loss_increase_loglog_{DATE_TAG}"
DOC_PATH = ROOT / f"docs/non_percent_loss_increase_loglog_{DATE_TAG}.md"


@dataclass(frozen=True)
class ProblemSpec:
    key: str
    display: str
    folder: str
    source_csv: Path
    extra_source_csvs: tuple[Path, ...]
    epsilon_col: str
    xlabel: str
    method_order: tuple[str, ...]
    colors: dict[str, str]
    markers: dict[str, str]


COMMON_COLORS = {
    "baseline": "#111111",
    "loss1": "#1f77b4",
    "loss2": "#ff7f0e",
    "loss3": "#2ca02c",
    "physics_loss": "#d62728",
    "random_clean": "#9467bd",
    "random_solver": "#17becf",
    "random_clean_y": "#9467bd",
    "random_solver_y": "#17becf",
}

COMMON_MARKERS = {
    "baseline": "o",
    "loss1": "s",
    "loss2": "^",
    "loss3": "D",
    "physics_loss": "P",
    "random_clean": "v",
    "random_solver": "X",
    "random_clean_y": "v",
    "random_solver_y": "X",
}

SPECS = [
    ProblemSpec(
        key="burgers",
        display="Burgers",
        folder="burgers",
        source_csv=ROOT
        / "outputs/burgers_elisa_percent_loss_increase_loglog_20260619/data/percent_loss_increase_loglog/budget_sweep_loss_increase_summary.csv",
        extra_source_csvs=(),
        epsilon_col="epsilon",
        xlabel="epsilon / RMS-L2 attack budget",
        method_order=("baseline", "loss1", "loss2", "loss3", "random_clean_y", "random_solver_y"),
        colors=COMMON_COLORS,
        markers=COMMON_MARKERS,
    ),
    ProblemSpec(
        key="darcy_flow",
        display="Darcy Flow",
        folder="darcy_flow",
        source_csv=ROOT
        / "outputs/darcy_sir20_timematched_full_serial_double_budget_full_delta_budget_records_20260617/data/budget_sweep_loss_increase_summary.csv",
        extra_source_csvs=(
            ROOT
            / "outputs/darcy_sir20_dense_extra_budgets_generalization_batched_20260619/data/budget_sweep_loss_increase_summary.csv",
        ),
        epsilon_col="budget",
        xlabel="epsilon / binary attack budget fraction",
        method_order=("baseline", "loss1", "loss2", "loss3", "physics_loss", "random_clean", "random_solver"),
        colors=COMMON_COLORS,
        markers=COMMON_MARKERS,
    ),
]


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def method_order(spec: ProblemSpec, methods: list[str]) -> list[str]:
    present = set(methods)
    ordered = [m for m in spec.method_order if m in present]
    return ordered + sorted(present - set(ordered))


def load_summary(spec: ProblemSpec) -> pd.DataFrame:
    frames = []
    for source_csv in (spec.source_csv, *spec.extra_source_csvs):
        if not source_csv.exists():
            continue
        frame = pd.read_csv(source_csv)
        frame["source_summary_csv"] = rel(source_csv)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(spec.source_csv)
    df = pd.concat(frames, ignore_index=True)
    required = {"method", "method_display", "split", "clean_loss_mean", "adv_loss_mean", spec.epsilon_col}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{spec.source_csv} missing required columns: {missing}")

    df = df.copy()
    df["epsilon"] = pd.to_numeric(df[spec.epsilon_col], errors="coerce")
    df["clean_loss_mean"] = pd.to_numeric(df["clean_loss_mean"], errors="coerce")
    df["adv_loss_mean"] = pd.to_numeric(df["adv_loss_mean"], errors="coerce")
    if "loss_increase_mean" in df.columns:
        df["absolute_loss_increase"] = pd.to_numeric(df["loss_increase_mean"], errors="coerce")
    else:
        df["absolute_loss_increase"] = df["adv_loss_mean"] - df["clean_loss_mean"]
    if "relative_increase_mean" in df.columns:
        df["relative_increase_mean"] = pd.to_numeric(df["relative_increase_mean"], errors="coerce")
    else:
        df["relative_increase_mean"] = df["absolute_loss_increase"] / df["clean_loss_mean"]
    df["batch_ratio_percent"] = np.where(
        np.isfinite(df["clean_loss_mean"]) & (df["clean_loss_mean"] > 0.0),
        100.0 * (df["adv_loss_mean"] / df["clean_loss_mean"] - 1.0),
        np.nan,
    )
    df["problem"] = spec.key
    df = df.sort_values(["method", "split", "epsilon", "source_summary_csv"], kind="mergesort")
    df = df.drop_duplicates(["method", "split", "epsilon"], keep="last").reset_index(drop=True)
    return df


def max_across_splits(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    usable = df[np.isfinite(df[metric])].copy()
    usable = usable.sort_values(["method", "epsilon", metric], ascending=[True, True, False], kind="mergesort")
    return usable.groupby(["method", "epsilon"], as_index=False, sort=False).head(1).reset_index(drop=True)


def cumulative_worst(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    parts = []
    for method, group in df.sort_values(["method", "epsilon"]).groupby("method", sort=False):
        g = group.copy()
        g[metric] = g[metric].cummax()
        parts.append(g)
    return pd.concat(parts, ignore_index=True) if parts else df.copy()


def plot_curves(
    spec: ProblemSpec,
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    out_png: Path,
) -> dict[str, object]:
    labels = (
        df[["method", "method_display"]]
        .dropna()
        .drop_duplicates("method")
        .set_index("method")["method_display"]
        .astype(str)
        .to_dict()
    )

    fig, ax = plt.subplots(figsize=(14.4, 8.0))
    plotted = []
    skipped_nonpositive = []
    y_values: list[float] = []
    for method in method_order(spec, df["method"].astype(str).unique().tolist()):
        sub = df[df["method"].eq(method)].sort_values("epsilon")
        sub = sub[np.isfinite(sub["epsilon"]) & np.isfinite(sub[metric])]
        sub = sub[sub[metric] > 0.0]
        if sub.empty:
            skipped_nonpositive.append(method)
            continue
        plotted.append(method)
        xs = sub["epsilon"].astype(float).to_numpy()
        ys = sub[metric].astype(float).to_numpy()
        y_values.extend(float(v) for v in ys)
        ax.plot(
            xs,
            ys,
            marker=spec.markers.get(method, "o"),
            markersize=6.0,
            linewidth=2.2,
            color=spec.colors.get(method),
            label=labels.get(method, method),
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(spec.xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.28, linewidth=0.7)
    ax.legend(loc="best", frameon=False, fontsize=10)
    if y_values:
        ax.set_ylim(max(min(y_values) * 0.65, 1e-14), max(y_values) * 1.8)
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return {
        "png": rel(out_png),
        "metric": metric,
        "plotted_methods": plotted,
        "skipped_nonpositive_methods": skipped_nonpositive,
    }


def rank_summary(df: pd.DataFrame, metrics: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        for epsilon, group in df.groupby("epsilon"):
            ranked = group.sort_values(metric, ascending=True).reset_index(drop=True)
            for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
                rows.append({"metric": metric, "epsilon": float(epsilon), "method": row["method"], "rank": rank})
    ranks = pd.DataFrame(rows)
    if ranks.empty:
        return ranks
    return ranks.groupby(["metric", "method"], as_index=False)["rank"].mean().sort_values(["metric", "rank", "method"])


def markdown_table(df: pd.DataFrame, cols: list[str], max_rows: int | None = None) -> str:
    if df.empty:
        return "_No rows._"
    shown = df[cols].head(max_rows) if max_rows else df[cols]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in shown.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, (float, np.floating)):
                vals.append(f"{float(val):.6g}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def build_problem(spec: ProblemSpec) -> dict[str, object]:
    df = load_summary(spec)
    out_dir = OUT_ROOT / spec.folder
    data_dir = out_dir / "data"
    fig_dir = out_dir / "figures"
    report_path = out_dir / "README.md"
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    generalization = df[df["split"].eq("generalization")].copy()
    if generalization.empty:
        raise RuntimeError(f"{spec.display}: no generalization rows")

    envelope = max_across_splits(df, "absolute_loss_increase")
    generalization_cumulative = cumulative_worst(generalization, "absolute_loss_increase")
    envelope_cumulative = cumulative_worst(envelope, "absolute_loss_increase")

    raw_csv = data_dir / "source_with_absolute_metrics.csv"
    gen_csv = data_dir / "generalization_loss_metrics.csv"
    envelope_csv = data_dir / "best_envelope_loss_increase.csv"
    gen_cum_csv = data_dir / "generalization_cumulative_worst_loss_increase.csv"
    envelope_cum_csv = data_dir / "best_envelope_cumulative_worst_loss_increase.csv"
    rank_csv = data_dir / "generalization_mean_ranks_by_metric.csv"

    df.to_csv(raw_csv, index=False)
    generalization.to_csv(gen_csv, index=False)
    envelope.to_csv(envelope_csv, index=False)
    generalization_cumulative.to_csv(gen_cum_csv, index=False)
    envelope_cumulative.to_csv(envelope_cum_csv, index=False)
    ranks = rank_summary(generalization, ("absolute_loss_increase", "adv_loss_mean", "clean_loss_mean", "batch_ratio_percent"))
    ranks.to_csv(rank_csv, index=False)

    plots = {
        "loss_increase_generalization": plot_curves(
            spec,
            generalization,
            "absolute_loss_increase",
            "batch-mean absolute loss increase: attacked - clean",
            f"{spec.display}: epsilon vs absolute loss increase, generalization, log-log",
            fig_dir / "epsilon_vs_loss_increase_generalization_loglog.png",
        ),
        "loss_increase_generalization_cumulative_worst": plot_curves(
            spec,
            generalization_cumulative,
            "absolute_loss_increase",
            "cumulative worst-case absolute loss increase",
            f"{spec.display}: cumulative worst-case loss increase, generalization, log-log",
            fig_dir / "epsilon_vs_loss_increase_generalization_cumulative_worst_loglog.png",
        ),
        "loss_increase_best_envelope": plot_curves(
            spec,
            envelope,
            "absolute_loss_increase",
            "max split-level absolute loss increase",
            f"{spec.display}: epsilon vs absolute loss increase, best split envelope, log-log",
            fig_dir / "epsilon_vs_loss_increase_best_envelope_loglog.png",
        ),
        "loss_increase_best_envelope_cumulative_worst": plot_curves(
            spec,
            envelope_cumulative,
            "absolute_loss_increase",
            "cumulative worst-case max split-level loss increase",
            f"{spec.display}: cumulative worst-case best envelope, log-log",
            fig_dir / "epsilon_vs_loss_increase_best_envelope_cumulative_worst_loglog.png",
        ),
        "final_adversarial_loss_generalization": plot_curves(
            spec,
            generalization,
            "adv_loss_mean",
            "batch-mean final adversarial loss",
            f"{spec.display}: epsilon vs final adversarial loss, generalization, log-log",
            fig_dir / "epsilon_vs_final_adversarial_loss_generalization_loglog.png",
        ),
        "clean_loss_generalization": plot_curves(
            spec,
            generalization,
            "clean_loss_mean",
            "batch-mean clean loss",
            f"{spec.display}: clean loss denominator by epsilon row, generalization, log-log",
            fig_dir / "epsilon_vs_clean_loss_generalization_loglog.png",
        ),
    }

    max_eps = float(generalization["epsilon"].max())
    eps_grid = sorted(float(x) for x in generalization["epsilon"].dropna().unique())
    final_rows = generalization[generalization["epsilon"].eq(max_eps)].sort_values("absolute_loss_increase")
    final_cols = [
        "method",
        "epsilon",
        "clean_loss_mean",
        "adv_loss_mean",
        "absolute_loss_increase",
        "relative_increase_mean",
        "batch_ratio_percent",
    ]
    final_csv = data_dir / "generalization_max_epsilon_sorted_by_loss_increase.csv"
    final_rows[final_cols].to_csv(final_csv, index=False)

    report_lines = [
        f"# {spec.display} Non-Percent Robustness Curves",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "These plots replace percent loss increase with absolute loss increase or final attacked loss. No attack or training run was launched for this plotting pass; it only reads existing budget-sweep CSVs.",
        "",
        "Primary interpretation:",
        "- Use `epsilon_vs_loss_increase_generalization_loglog.png` for the raw generalization loss-increase curve.",
        "- Use `epsilon_vs_loss_increase_generalization_cumulative_worst_loglog.png` when the figure should represent a monotone worst-case-over-budget envelope.",
        "- Use `epsilon_vs_clean_loss_generalization_loglog.png` only as a denominator diagnostic for why percent plots can invert rankings.",
        "",
        "Artifacts:",
        f"- Source CSV: `{rel(spec.source_csv)}`",
        *[f"- Extra source CSV: `{rel(path)}`" for path in spec.extra_source_csvs if path.exists()],
        f"- Raw enriched CSV: `{rel(raw_csv)}`",
        f"- Generalization CSV: `{rel(gen_csv)}`",
        f"- Best-envelope CSV: `{rel(envelope_csv)}`",
        f"- Generalization cumulative-worst CSV: `{rel(gen_cum_csv)}`",
        f"- Mean-rank CSV: `{rel(rank_csv)}`",
        f"- Max-epsilon sorted CSV: `{rel(final_csv)}`",
        f"- Generalization epsilon grid: `{', '.join(f'{eps:.6g}' for eps in eps_grid)}`",
        "",
        "PNG figures:",
        *[f"- `{plot['png']}`" for plot in plots.values()],
        "",
        f"Generalization max epsilon `{max_eps}` sorted by absolute loss increase:",
        "",
        markdown_table(final_rows, final_cols),
        "",
        "Generalization mean rank by metric; lower is better:",
        "",
        markdown_table(ranks, ["metric", "method", "rank"]),
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    manifest = {
        "problem": spec.key,
        "display": spec.display,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_csv": rel(spec.source_csv),
        "extra_source_csvs": [rel(path) for path in spec.extra_source_csvs if path.exists()],
        "generalization_epsilon_grid": eps_grid,
        "folder": rel(out_dir),
        "data": {
            "raw_enriched": rel(raw_csv),
            "generalization": rel(gen_csv),
            "best_envelope": rel(envelope_csv),
            "generalization_cumulative_worst": rel(gen_cum_csv),
            "best_envelope_cumulative_worst": rel(envelope_cum_csv),
            "mean_ranks": rel(rank_csv),
            "max_epsilon_sorted": rel(final_csv),
        },
        "plots": plots,
        "report": rel(report_path),
    }
    manifest_path = data_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifests = [build_problem(spec) for spec in SPECS]
    top_manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "out_root": rel(OUT_ROOT),
        "problems": manifests,
    }
    manifest_path = OUT_ROOT / "manifest.json"
    manifest_path.write_text(json.dumps(top_manifest, indent=2) + "\n", encoding="utf-8")

    doc_lines = [
        "# Non-Percent Loss-Increase Log-Log Curves",
        "",
        f"Generated: {top_manifest['created_utc']}",
        "",
        "This bundle redraws Burgers and Darcy Flow without percent loss increase on the y-axis.",
        "",
        "Main files:",
    ]
    for manifest in manifests:
        doc_lines.extend(
            [
                f"- {manifest['display']} folder: `{manifest['folder']}`",
                f"- {manifest['display']} report: `{manifest['report']}`",
            ]
        )
    doc_lines.extend(
        [
            "",
            "Recommended figure for robustness ranking: raw or cumulative-worst absolute loss increase, not percent loss increase. The clean-loss plots are included only to show why the percent denominator can distort rankings.",
        ]
    )
    DOC_PATH.write_text("\n".join(doc_lines) + "\n", encoding="utf-8")

    print(json.dumps({"status": "done", "out_root": rel(OUT_ROOT), "doc": rel(DOC_PATH), "manifest": rel(manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
