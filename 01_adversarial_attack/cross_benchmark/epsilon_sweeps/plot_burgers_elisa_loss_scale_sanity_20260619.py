#!/usr/bin/env python3
"""Plot absolute-loss sanity checks for the Burgers Elisa percent curves."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO / "outputs/burgers_elisa_percent_loss_increase_loglog_20260619"
DATE_TAG = "20260619"

MODEL_ORDER = ["baseline", "loss1", "loss2", "loss3", "random_clean_y", "random_solver_y"]
DISPLAY = {
    "baseline": "Burgers baseline",
    "loss1": "Burgers loss1",
    "loss2": "Burgers loss2",
    "loss3": "Burgers loss3",
    "random_clean_y": "Burgers random clean",
    "random_solver_y": "Burgers random solver",
}
COLORS = {
    "baseline": "#111111",
    "loss1": "#1f77b4",
    "loss2": "#ff7f0e",
    "loss3": "#2ca02c",
    "random_clean_y": "#9467bd",
    "random_solver_y": "#17becf",
}
MARKERS = {
    "baseline": "o",
    "loss1": "s",
    "loss2": "^",
    "loss3": "D",
    "random_clean_y": "v",
    "random_solver_y": "X",
}


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO.resolve()))
    except ValueError:
        return str(path)


def method_key(method: str) -> tuple[int, str]:
    try:
        return (MODEL_ORDER.index(method), method)
    except ValueError:
        return (999, method)


def plot_metric(df: pd.DataFrame, metric: str, ylabel: str, title: str, out_base: Path) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(14.4, 8.0))
    vals: list[float] = []
    for method in sorted(df["method"].unique(), key=method_key):
        sub = df[df["method"].eq(method)].sort_values("epsilon")
        sub = sub[sub[metric] > 0.0]
        if sub.empty:
            continue
        xs = sub["epsilon"].astype(float).to_numpy()
        ys = sub[metric].astype(float).to_numpy()
        vals.extend(float(v) for v in ys)
        ax.plot(
            xs,
            ys,
            marker=MARKERS.get(method, "o"),
            markersize=6.0,
            linewidth=2.2,
            color=COLORS.get(method),
            label=DISPLAY.get(method, method),
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("epsilon / RMS-L2 attack budget")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.28, linewidth=0.7)
    ax.legend(loc="best", frameon=False, fontsize=10)
    if vals:
        ax.set_ylim(max(min(vals) * 0.65, 1e-12), max(vals) * 1.8)
    fig.tight_layout()
    out_base.parent.mkdir(parents=True, exist_ok=True)
    png = out_base.with_suffix(".png")
    pdf = out_base.with_suffix(".pdf")
    fig.savefig(png, dpi=260, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {"png": rel(png), "pdf": rel(pdf)}


def maxima(df: pd.DataFrame, metric: str, mode: str) -> list[dict[str, object]]:
    rows = []
    for method in sorted(df["method"].unique(), key=method_key):
        sub = df[df["method"].eq(method)]
        if sub.empty:
            continue
        idx = sub[metric].idxmax() if mode == "max" else sub[metric].idxmin()
        row = sub.loc[idx]
        rows.append(
            {
                "method": method,
                "method_display": DISPLAY.get(method, method),
                f"{mode}_{metric}": float(row[metric]),
                "epsilon": float(row["epsilon"]),
                "clean_loss_mean": float(row["clean_loss_mean"]),
                "adv_loss_mean": float(row["adv_loss_mean"]),
                "percent_loss_increase": float(row["percent_loss_increase"]),
            }
        )
    return rows


def rank_table(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    records = []
    for epsilon, sub in df.groupby("epsilon"):
        ranked = sub.sort_values(metric, ascending=True).reset_index(drop=True)
        for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
            records.append({"epsilon": epsilon, "metric": metric, "method": row["method"], "rank": rank})
    ranks = pd.DataFrame(records)
    return (
        ranks.groupby(["metric", "method"], as_index=False)["rank"]
        .mean()
        .sort_values(["metric", "rank", "method"], ascending=[True, True, True])
    )


def markdown_table(rows: list[dict[str, object]], cols: list[str]) -> str:
    if not rows:
        return "_No rows._"
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in rows:
        vals = []
        for col in cols:
            val = row.get(col, "")
            if isinstance(val, float):
                vals.append(f"{val:.6g}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()

    root = args.root.resolve()
    summary_csv = root / "data/percent_loss_increase_loglog/budget_sweep_loss_increase_summary.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(summary_csv)

    df = pd.read_csv(summary_csv)
    df["abs_loss_increase"] = df["adv_loss_mean"] - df["clean_loss_mean"]
    gen = df[df["split"].eq("generalization")].copy()
    if gen.empty:
        raise RuntimeError("no generalization rows found")

    data_dir = root / "data/loss_scale_sanity"
    fig_dir = root / "figures/loss_scale_sanity"
    report_dir = root / "reports"
    data_dir.mkdir(parents=True, exist_ok=True)

    gen_csv = data_dir / "generalization_absolute_loss_sanity.csv"
    gen.to_csv(gen_csv, index=False)
    rank_csv = data_dir / "generalization_mean_ranks_by_metric.csv"
    rank_df = pd.concat(
        [rank_table(gen, metric) for metric in ["abs_loss_increase", "adv_loss_mean", "percent_loss_increase"]],
        ignore_index=True,
    )
    rank_df.to_csv(rank_csv, index=False)

    plots = {
        "abs_loss_increase_generalization": plot_metric(
            gen,
            "abs_loss_increase",
            "batch-mean absolute loss increase: final - initial",
            "Burgers epsilon vs absolute loss increase: generalization, log-log",
            fig_dir / "epsilon_vs_absolute_loss_increase_generalization_loglog",
        ),
        "adv_loss_mean_generalization": plot_metric(
            gen,
            "adv_loss_mean",
            "batch-mean final adversarial loss",
            "Burgers epsilon vs final adversarial loss: generalization, log-log",
            fig_dir / "epsilon_vs_final_adversarial_loss_generalization_loglog",
        ),
        "clean_loss_mean_generalization": plot_metric(
            gen,
            "clean_loss_mean",
            "batch-mean clean loss",
            "Burgers clean loss by epsilon row: generalization, log-log",
            fig_dir / "epsilon_vs_clean_loss_generalization_loglog",
        ),
    }

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_summary_csv": rel(summary_csv),
        "generalization_csv": rel(gen_csv),
        "rank_csv": rel(rank_csv),
        "plots": plots,
        "max_abs_loss_increase": maxima(gen, "abs_loss_increase", "max"),
        "max_adv_loss_mean": maxima(gen, "adv_loss_mean", "max"),
        "max_percent_loss_increase": maxima(gen, "percent_loss_increase", "max"),
    }
    manifest_path = data_dir / "loss_scale_sanity_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    eps_max = gen["epsilon"].max()
    final = gen[gen["epsilon"].eq(eps_max)].sort_values("abs_loss_increase")
    final_rows = [
        {
            "method": row["method"],
            "clean_loss_mean": float(row["clean_loss_mean"]),
            "adv_loss_mean": float(row["adv_loss_mean"]),
            "abs_loss_increase": float(row["abs_loss_increase"]),
            "percent_loss_increase": float(row["percent_loss_increase"]),
        }
        for _, row in final.iterrows()
    ]
    rank_rows = [
        {"metric": row["metric"], "method": row["method"], "mean_rank_lower_better": float(row["rank"])}
        for _, row in rank_df.iterrows()
    ]
    report_lines = [
        "# Burgers Elisa Loss-Scale Sanity Plots",
        "",
        f"Generated: {manifest['created_utc']}",
        "",
        "These plots reuse the completed Burgers Elisa attack sweep CSV. No attack was rerun.",
        "",
        "Purpose: check whether the percent-loss plot is ranking true robustness or mostly reflecting clean-loss denominators.",
        "",
        "Artifacts:",
        f"- Source summary CSV: `{rel(summary_csv)}`",
        f"- Generalization sanity CSV: `{rel(gen_csv)}`",
        f"- Mean-rank CSV: `{rel(rank_csv)}`",
        f"- Absolute loss increase PNG: `{plots['abs_loss_increase_generalization']['png']}`",
        f"- Final adversarial loss PNG: `{plots['adv_loss_mean_generalization']['png']}`",
        f"- Clean loss PNG: `{plots['clean_loss_mean_generalization']['png']}`",
        "",
        f"Generalization rows at epsilon `{eps_max}` sorted by absolute loss increase:",
        "",
        markdown_table(final_rows, ["method", "clean_loss_mean", "adv_loss_mean", "abs_loss_increase", "percent_loss_increase"]),
        "",
        "Mean ranks over all epsilon values; lower is better:",
        "",
        markdown_table(rank_rows, ["metric", "method", "mean_rank_lower_better"]),
        "",
        "Interpretation:",
        "- The percent-loss ranking is strongly affected by the clean-loss denominator.",
        "- Absolute loss increase and final adversarial loss both rank `loss3` best on this compact generalization sweep.",
        "- Therefore the existing percent-loss figure is arithmetically consistent with the CSV, but it is not a good standalone cross-model robustness ranking for Burgers.",
    ]
    report_path = report_dir / "loss_scale_sanity.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    doc_path = REPO / f"docs/burgers_elisa_loss_scale_sanity_{DATE_TAG}.md"
    doc_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(json.dumps({"status": "done", "manifest": rel(manifest_path), "report": rel(report_path), "doc": rel(doc_path), "plots": plots}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
