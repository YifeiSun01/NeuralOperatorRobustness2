#!/usr/bin/env python3
"""Audit robustness metric consistency for the Burgers and Darcy percent plots."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATE_TAG = "20260619"
OUT_DIR = ROOT / f"analysis_outputs/robustness_metric_consistency_{DATE_TAG}"

BURGERS_CSV = ROOT / "outputs/burgers_elisa_percent_loss_increase_loglog_20260619/data/percent_loss_increase_loglog/budget_sweep_loss_increase_summary.csv"
DARCY_CSV = ROOT / "outputs/darcy_sir20_timematched_full_serial_double_budget_full_delta_budget_records_20260617/data/budget_sweep_loss_increase_summary.csv"
DARCY_CLEAN_CSV = ROOT / "outputs/darcy_sir20_timematched_full_serial_double_budget_full_delta_budget_records_20260617/data/final_eval_summary_by_model_split.csv"


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def load_problem(path: Path, *, epsilon_col: str, has_loss_increase: bool) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["epsilon"] = pd.to_numeric(df[epsilon_col], errors="coerce")
    df["clean_loss_mean"] = pd.to_numeric(df["clean_loss_mean"], errors="coerce")
    df["adv_loss_mean"] = pd.to_numeric(df["adv_loss_mean"], errors="coerce")
    if has_loss_increase:
        df["absolute_loss_increase"] = pd.to_numeric(df["loss_increase_mean"], errors="coerce")
    else:
        df["absolute_loss_increase"] = df["adv_loss_mean"] - df["clean_loss_mean"]
    if "relative_increase_mean" in df.columns:
        df["relative_increase_mean"] = pd.to_numeric(df["relative_increase_mean"], errors="coerce")
    else:
        df["relative_increase_mean"] = df["absolute_loss_increase"] / df["clean_loss_mean"]
    df["batch_ratio_percent"] = 100.0 * (df["adv_loss_mean"] / df["clean_loss_mean"] - 1.0)
    return df


def method_order(methods: list[str]) -> list[str]:
    preferred = [
        "baseline",
        "loss1",
        "loss2",
        "loss3",
        "physics_loss",
        "random_clean",
        "random_solver",
        "random_clean_y",
        "random_solver_y",
    ]
    return [m for m in preferred if m in set(methods)] + sorted(set(methods) - set(preferred))


def nonmonotone_rows(df: pd.DataFrame, problem: str, metric: str) -> list[dict[str, object]]:
    rows = []
    sub = df[df["split"].eq("generalization")].sort_values(["method", "epsilon"])
    for method, group in sub.groupby("method"):
        vals = group[metric].astype(float).to_numpy()
        eps = group["epsilon"].astype(float).to_numpy()
        for i in range(len(vals) - 1):
            if np.isfinite(vals[i]) and np.isfinite(vals[i + 1]) and vals[i + 1] < vals[i] - 1e-12:
                rows.append(
                    {
                        "problem": problem,
                        "metric": metric,
                        "method": method,
                        "epsilon_from": float(eps[i]),
                        "epsilon_to": float(eps[i + 1]),
                        "value_from": float(vals[i]),
                        "value_to": float(vals[i + 1]),
                        "drop": float(vals[i] - vals[i + 1]),
                    }
                )
    return rows


def max_epsilon_table(df: pd.DataFrame, problem: str) -> pd.DataFrame:
    sub = df[df["split"].eq("generalization")].copy()
    eps = float(sub["epsilon"].max())
    out = sub[sub["epsilon"].eq(eps)].copy()
    out.insert(0, "problem", problem)
    out = out[
        [
            "problem",
            "method",
            "method_display",
            "epsilon",
            "clean_loss_mean",
            "adv_loss_mean",
            "absolute_loss_increase",
            "relative_increase_mean",
            "batch_ratio_percent",
        ]
    ].sort_values(["problem", "absolute_loss_increase", "adv_loss_mean"])
    return out


def mean_rank_table(df: pd.DataFrame, problem: str) -> pd.DataFrame:
    sub = df[df["split"].eq("generalization")].copy()
    metrics = ["adv_loss_mean", "absolute_loss_increase", "relative_increase_mean", "batch_ratio_percent"]
    rows = []
    for metric in metrics:
        rank_frames = []
        for _, group in sub.groupby("epsilon"):
            rank_frames.append(group.set_index("method")[metric].rank(ascending=True))
        ranks = pd.concat(rank_frames, axis=1).mean(axis=1).sort_values()
        for method, rank in ranks.items():
            rows.append({"problem": problem, "metric": metric, "method": method, "mean_rank_lower_better": float(rank)})
    return pd.DataFrame(rows)


def baseline_compare_table(df: pd.DataFrame, problem: str) -> pd.DataFrame:
    sub = df[df["split"].eq("generalization")].copy()
    eps = float(sub["epsilon"].max())
    last = sub[sub["epsilon"].eq(eps)].set_index("method")
    if "baseline" not in last.index:
        return pd.DataFrame()
    baseline = last.loc["baseline"]
    metrics = ["adv_loss_mean", "absolute_loss_increase", "relative_increase_mean", "batch_ratio_percent"]
    rows = []
    for method, row in last.iterrows():
        if method == "baseline":
            continue
        for metric in metrics:
            value = float(row[metric])
            base = float(baseline[metric])
            rows.append(
                {
                    "problem": problem,
                    "epsilon": eps,
                    "method": method,
                    "metric": metric,
                    "value": value,
                    "baseline_value": base,
                    "worse_than_baseline": bool(value > base),
                    "relative_change_vs_baseline_pct": float((value - base) / base * 100.0) if base != 0 else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def darcy_delta_monotonicity_table(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df["split"].eq("generalization")].copy()
    needed = {"delta_l2_rms_mean", "adv_loss_mean", "loss_increase_mean"}
    if not needed.issubset(sub.columns):
        return pd.DataFrame()

    rows = []
    for method, group in sub.sort_values(["method", "epsilon"]).groupby("method"):
        delta = pd.to_numeric(group["delta_l2_rms_mean"], errors="coerce").to_numpy(dtype=float)
        adv = pd.to_numeric(group["adv_loss_mean"], errors="coerce").to_numpy(dtype=float)
        inc = pd.to_numeric(group["loss_increase_mean"], errors="coerce").to_numpy(dtype=float)
        if len(delta) == 0:
            continue
        rows.append(
            {
                "problem": "darcy",
                "method": method,
                "delta_l2_monotone": bool(np.all(delta[1:] + 1e-12 >= delta[:-1])),
                "adv_loss_monotone": bool(np.all(adv[1:] + 1e-12 >= adv[:-1])),
                "loss_increase_monotone": bool(np.all(inc[1:] + 1e-12 >= inc[:-1])),
                "delta_l2_first": float(delta[0]),
                "delta_l2_last": float(delta[-1]),
            }
        )
    return pd.DataFrame(rows)


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


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    burgers = load_problem(BURGERS_CSV, epsilon_col="epsilon", has_loss_increase=False)
    darcy = load_problem(DARCY_CSV, epsilon_col="budget", has_loss_increase=True)

    max_table = pd.concat(
        [max_epsilon_table(burgers, "burgers"), max_epsilon_table(darcy, "darcy")],
        ignore_index=True,
    )
    rank_table = pd.concat(
        [mean_rank_table(burgers, "burgers"), mean_rank_table(darcy, "darcy")],
        ignore_index=True,
    )
    baseline_table = pd.concat(
        [baseline_compare_table(burgers, "burgers"), baseline_compare_table(darcy, "darcy")],
        ignore_index=True,
    )
    delta_table = darcy_delta_monotonicity_table(darcy)
    nonmono = pd.DataFrame(
        nonmonotone_rows(burgers, "burgers", "adv_loss_mean")
        + nonmonotone_rows(burgers, "burgers", "absolute_loss_increase")
        + nonmonotone_rows(burgers, "burgers", "relative_increase_mean")
        + nonmonotone_rows(burgers, "burgers", "batch_ratio_percent")
        + nonmonotone_rows(darcy, "darcy", "adv_loss_mean")
        + nonmonotone_rows(darcy, "darcy", "absolute_loss_increase")
        + nonmonotone_rows(darcy, "darcy", "relative_increase_mean")
        + nonmonotone_rows(darcy, "darcy", "batch_ratio_percent")
    )

    max_csv = OUT_DIR / "generalization_max_epsilon_metrics.csv"
    rank_csv = OUT_DIR / "generalization_mean_ranks_by_metric.csv"
    baseline_csv = OUT_DIR / "generalization_baseline_comparison_at_max_epsilon.csv"
    delta_csv = OUT_DIR / "darcy_generalization_delta_vs_loss_monotonicity.csv"
    nonmono_csv = OUT_DIR / "generalization_nonmonotone_segments.csv"
    max_table.to_csv(max_csv, index=False)
    rank_table.to_csv(rank_csv, index=False)
    baseline_table.to_csv(baseline_csv, index=False)
    delta_table.to_csv(delta_csv, index=False)
    nonmono.to_csv(nonmono_csv, index=False)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": {
            "burgers_budget_summary": rel(BURGERS_CSV),
            "darcy_budget_summary": rel(DARCY_CSV),
            "darcy_clean_summary": rel(DARCY_CLEAN_CSV),
        },
        "outputs": {
            "max_epsilon_metrics": rel(max_csv),
            "mean_ranks_by_metric": rel(rank_csv),
            "baseline_comparison": rel(baseline_csv),
            "darcy_delta_vs_loss_monotonicity": rel(delta_csv),
            "nonmonotone_segments": rel(nonmono_csv),
        },
    }
    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    report = OUT_DIR / "README.md"
    lines = [
        "# Robustness Metric Consistency Audit",
        "",
        f"Generated: {manifest['created_utc']}",
        "",
        "This audit checks the finite-budget robustness metrics behind the Burgers and Darcy log-log plots.",
        "",
        "Metrics checked:",
        "- `adv_loss_mean`: final attacked loss.",
        "- `absolute_loss_increase`: `adv_loss_mean - clean_loss_mean` for Burgers, recorded `loss_increase_mean` for Darcy.",
        "- `relative_increase_mean`: mean per-sample relative increase when present.",
        "- `batch_ratio_percent`: `100 * (adv_loss_mean / clean_loss_mean - 1)`.",
        "",
        "Key finding:",
        "- Burgers generalization curves are monotone in the compact sweep, but the relative/percent metrics invert the cross-model ranking because clean-loss denominators differ strongly. Absolute increase and final attacked loss rank `loss3` best.",
        "- Darcy generalization curves are not monotone for several methods and metrics in the original budget sweep. The source attack is `binary_darcy_replace_attack`, which recomputes an exact `k = round(epsilon_fraction * n_pix)` replacement set for each budget rather than preserving a nested lower-budget solution. Larger budget therefore means more flipped pixels, not guaranteed larger loss under this heuristic.",
        "- Darcy `delta_l2_rms_mean` is monotone increasing for every method in the generalization sweep, while final attacked loss and loss increase are not monotone for most methods. This supports the interpretation that the budget/delta grows, but the exact-k replacement heuristic does not produce a nested best-loss envelope.",
        "",
        "Max-epsilon generalization metrics:",
        "",
        markdown_table(
            max_table,
            [
                "problem",
                "method",
                "epsilon",
                "clean_loss_mean",
                "adv_loss_mean",
                "absolute_loss_increase",
                "relative_increase_mean",
                "batch_ratio_percent",
            ],
        ),
        "",
        "Mean rank by metric over epsilon grid; lower is better:",
        "",
        markdown_table(rank_table, ["problem", "metric", "method", "mean_rank_lower_better"]),
        "",
        "Non-monotone segments:",
        "",
        markdown_table(nonmono, ["problem", "metric", "method", "epsilon_from", "epsilon_to", "value_from", "value_to", "drop"], max_rows=80),
        "",
        "Darcy delta-vs-loss monotonicity:",
        "",
        markdown_table(
            delta_table,
            [
                "problem",
                "method",
                "delta_l2_monotone",
                "adv_loss_monotone",
                "loss_increase_monotone",
                "delta_l2_first",
                "delta_l2_last",
            ],
        ),
        "",
        "Output CSVs:",
        f"- `{rel(max_csv)}`",
        f"- `{rel(rank_csv)}`",
        f"- `{rel(baseline_csv)}`",
        f"- `{rel(delta_csv)}`",
        f"- `{rel(nonmono_csv)}`",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    docs_report = ROOT / f"docs/robustness_metric_consistency_audit_{DATE_TAG}.md"
    docs_report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"status": "done", "report": rel(report), "docs_report": rel(docs_report), "manifest": rel(manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
