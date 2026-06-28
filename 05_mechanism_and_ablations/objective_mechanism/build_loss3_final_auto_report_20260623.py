#!/usr/bin/env python3
"""Build the final automated Loss3 report/bundle index."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def fnum(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def copy_asset(src: Path, dst_root: Path, kind: str, rows: list[dict[str, Any]]) -> None:
    if not src.exists():
        rows.append({"kind": kind, "source": rel(src), "bundle_path": "", "status": "missing"})
        return
    target = dst_root / kind / src.name
    target.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(src, target, ignore=shutil.ignore_patterns("*.npz", "*.pt", "*.pth", "__pycache__"))
    else:
        shutil.copy2(src, target)
    rows.append({"kind": kind, "source": rel(src), "bundle_path": rel(target), "status": "copied"})


def summarize_main_curves(curves_path: Path) -> list[dict[str, Any]]:
    df = read_csv(curves_path)
    if df.empty:
        return []
    for col in ("step", "mean_true_loss3", "n"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    rows: list[dict[str, Any]] = []
    for problem, group in df.groupby("problem", dropna=False):
        if group.empty:
            continue
        final_step = int(group["step"].max())
        final = group[group["step"] == final_step].copy()
        final = final.sort_values("mean_true_loss3", ascending=False)
        best = final.iloc[0] if not final.empty else {}
        for rec in final.to_dict("records"):
            rows.append(
                {
                    "problem": problem,
                    "final_step": final_step,
                    "optimizer": rec.get("optimizer"),
                    "mean_true_loss3": fnum(rec.get("mean_true_loss3")),
                    "n": int(fnum(rec.get("n"))) if math.isfinite(fnum(rec.get("n"))) else "",
                    "best_optimizer": best.get("optimizer", ""),
                }
            )
    return rows


def summarize_jvp_candidates(path: Path) -> list[dict[str, Any]]:
    df = read_csv(path)
    if df.empty:
        return []
    for col in ("true_gain_mean", "relative_abs_error_mean", "cos_step_with_grad_mean"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    focus = df[df["candidate"].isin(["add_top_singular", "replace_top_singular", "add_saved_direction", "replace_saved_direction", "add_grad", "replace_grad"])].copy()
    rows: list[dict[str, Any]] = []
    for rec in focus.to_dict("records"):
        rows.append(
            {
                "source_run": path.parent.name,
                "method": rec.get("method", ""),
                "candidate": rec.get("candidate", ""),
                "true_gain_mean": fnum(rec.get("true_gain_mean")),
                "relative_abs_error_mean": fnum(rec.get("relative_abs_error_mean")),
                "cos_step_with_grad_mean": fnum(rec.get("cos_step_with_grad_mean")),
            }
        )
    return rows


def mechanism_short_read(main_rows: list[dict[str, Any]], jvp_rows: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    if main_rows:
        lines.append("Main averaged Loss3 curves determine the empirical optimizer ranking; mechanism plots are interpreted only against that averaged result.")
        by_problem: dict[str, str] = {}
        for row in main_rows:
            if row["problem"] not in by_problem and row.get("best_optimizer"):
                by_problem[row["problem"]] = str(row["best_optimizer"])
        for problem, best in sorted(by_problem.items()):
            lines.append(f"- `{problem}` final-step best optimizer in the current mean curve table: `{best}`.")
    if jvp_rows:
        add_vals = [r["true_gain_mean"] for r in jvp_rows if str(r["candidate"]).startswith("add_") and math.isfinite(fnum(r["true_gain_mean"]))]
        rep_vals = [r["true_gain_mean"] for r in jvp_rows if str(r["candidate"]).startswith("replace_") and math.isfinite(fnum(r["true_gain_mean"]))]
        if add_vals and rep_vals:
            add_mean = sum(add_vals) / len(add_vals)
            rep_mean = sum(rep_vals) / len(rep_vals)
            if add_mean > rep_mean:
                lines.append(f"- NS2D JVP/VJP candidate probe: add-style candidates have larger average true gain ({add_mean:.4g}) than replace-style candidates ({rep_mean:.4g}), supporting the path-following/add explanation.")
            else:
                lines.append(f"- NS2D JVP/VJP candidate probe: replace-style candidates have comparable or larger average true gain ({rep_mean:.4g}) than add-style candidates ({add_mean:.4g}); check boundary/topology evidence before claiming local-Jacobian instability.")
    if not lines:
        lines.append("Final automated interpretation is pending because one or more summary tables are missing.")
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "analysis_outputs/loss3_final_auto_pipeline_20260623")
    parser.add_argument("--optimizer-root", type=Path, default=PROJECT_ROOT / "analysis_outputs/optimizer_ablation_20260622")
    parser.add_argument("--mechanism-root", type=Path, default=PROJECT_ROOT / "analysis_outputs/mechanism_20260622/full_mechanism_validation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir if args.out_dir.is_absolute() else PROJECT_ROOT / args.out_dir
    optimizer_root = args.optimizer_root if args.optimizer_root.is_absolute() else PROJECT_ROOT / args.optimizer_root
    mechanism_root = args.mechanism_root if args.mechanism_root.is_absolute() else PROJECT_ROOT / args.mechanism_root
    out_dir.mkdir(parents=True, exist_ok=True)

    assets: list[dict[str, Any]] = []
    key_assets = [
        (optimizer_root / "figures/loss3_three_system_optimizer_mean_curves.png", "figures"),
        (optimizer_root / "figures/loss3_three_system_optimizer_mean_curves.pdf", "figures"),
        (optimizer_root / "tables/loss3_three_system_optimizer_mean_curves.csv", "tables"),
        (optimizer_root / "tables/loss3_three_system_optimizer_mean_curves_selection.csv", "tables"),
        (optimizer_root / "tables/aggregate_summary.csv", "tables"),
        (mechanism_root / "cross_system_summary/mechanism_summary_rows.csv", "tables"),
        (mechanism_root / "cross_system_summary/ns2d_jvp_vjp_key_metrics.csv", "tables"),
        (mechanism_root / "figures/ns2d_jvp_vjp", "figures"),
        (mechanism_root / "burgers_landscape_ridge_probe_N20/figures", "figures"),
        (PROJECT_ROOT / "docs/loss3_full_mechanism_validation_summary_20260623.md", "docs"),
        (PROJECT_ROOT / "docs/loss3_ns2d_jvp_vjp_integration_summary_20260623.md", "docs"),
        (PROJECT_ROOT / "docs/loss3_jvp_vjp_integration_experiment_plan_20260623.md", "docs"),
        (PROJECT_ROOT / "docs/loss3_mechanism_run_coverage_20260623.md", "docs"),
    ]
    for src, kind in key_assets:
        copy_asset(src, out_dir, kind, assets)

    main_rows = summarize_main_curves(optimizer_root / "tables/loss3_three_system_optimizer_mean_curves.csv")
    jvp_rows: list[dict[str, Any]] = []
    for directory in (mechanism_root / "ns2d_jvp_vjp_spectrum_N3", mechanism_root / "ns2d_jvp_vjp_spectrum_N2_topup"):
        jvp_rows.extend(summarize_jvp_candidates(directory / "candidate_comparison_aggregate.csv"))

    write_csv(out_dir / "tables/final_main_curve_ranking.csv", main_rows)
    write_csv(out_dir / "tables/final_ns2d_jvp_candidate_read.csv", jvp_rows)
    write_csv(out_dir / "asset_index.csv", assets)

    lines = [
        "# Loss3 Final Automated Report - 2026-06-23",
        "",
        "This report is generated by the final automation pipeline after the optimizer curves and mechanism probes finish.",
        "",
        "## Automated Read",
        "",
    ]
    lines.extend(mechanism_short_read(main_rows, jvp_rows))
    lines.extend(
        [
            "",
            "## Main Curve Ranking",
            "",
            "| Problem | Final step | Optimizer | Mean true Loss3 | N | Best optimizer |",
            "| --- | ---: | --- | ---: | ---: | --- |",
        ]
    )
    for row in main_rows:
        lines.append(
            f"| {row['problem']} | {row['final_step']} | {row['optimizer']} | {row['mean_true_loss3']:.6g} | {row['n']} | {row['best_optimizer']} |"
        )
    lines.extend(
        [
            "",
            "## Key Outputs",
            "",
            f"- Main three-system curve: `{rel(optimizer_root / 'figures/loss3_three_system_optimizer_mean_curves.png')}`",
            f"- Cross-system mechanism CSV: `{rel(mechanism_root / 'cross_system_summary/mechanism_summary_rows.csv')}`",
            f"- NS2D JVP/VJP key metrics: `{rel(mechanism_root / 'cross_system_summary/ns2d_jvp_vjp_key_metrics.csv')}`",
            f"- Bundle asset index: `{rel(out_dir / 'asset_index.csv')}`",
            "",
            "## Interpretation Contract",
            "",
            "The final claim should be based on the averaged optimizer curves first, then supported by the mechanism evidence.  The NS2D JVP/VJP block specifically tests whether replace has a stable generalized-power/top-singular-direction explanation at the full attack radius.",
            "",
        ]
    )
    (out_dir / "FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    manifest = {
        "status": "completed",
        "out_dir": str(out_dir),
        "num_assets": len(assets),
        "num_main_rows": len(main_rows),
        "num_jvp_candidate_rows": len(jvp_rows),
        "report": str(out_dir / "FINAL_REPORT.md"),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
