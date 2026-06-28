#!/usr/bin/env python3
"""Summarize the loss3 alpha/epsilon core-four optimizer sweep."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = PROJECT_ROOT / "forensics" / "loss3_alpha_epsilon_core4_sweep_20260519"
DEFAULT_OUT = PROJECT_ROOT / "forensics" / "loss3_alpha_epsilon_core4_analysis_20260519"
DEFAULT_DOC = PROJECT_ROOT / "docs" / "loss3_alpha_epsilon_core4_sweep_result_20260519.md"
CORE4 = ("raw_add", "raw_replace", "steepest_add", "steepest_replace")
BOUNDARY_THRESHOLDS = (0.25, 0.50, 0.75, 0.95, 0.99)


def fnum(value: str | int | float | None) -> float:
    if value in {None, "", "nan"}:
        return math.nan
    try:
        return float(value)
    except Exception:
        return math.nan


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def manifest_completed(root: Path) -> bool:
    path = root / "manifest.json"
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "completed"
    except Exception:
        return False


def config_value(root: Path, key: str, fallback=None):
    for name in ("config.json", "manifest.json"):
        path = root / name
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if key in payload:
                    return payload[key]
            except Exception:
                pass
    return fallback


def finite_values(values) -> list[float]:
    out: list[float] = []
    for value in values:
        fv = fnum(value)
        if math.isfinite(fv):
            out.append(fv)
    return out


def first_step_metric_at(rows: list[dict[str, str]], key: str, threshold: float) -> int | str:
    if not math.isfinite(threshold):
        return "nan"
    for row in sorted(rows, key=lambda r: int(r["k"])):
        value = fnum(row.get(key))
        if math.isfinite(value) and value >= threshold:
            return int(row["k"])
    return "nan"


def first_step_at(rows: list[dict[str, str]], threshold: float) -> int | str:
    return first_step_metric_at(rows, "loss3_q_mean", threshold)


def boundary_sample_stats(sample_rows: list[dict[str, str]], method: str, threshold: float) -> dict[str, float | int | str]:
    by_sample: dict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)
    for row in sample_rows:
        if row.get("method") != method:
            continue
        key = (int(row.get("sample_position", 0)), int(row.get("dataset_index", 0)))
        by_sample[key].append(row)

    first_steps: list[int] = []
    missing = 0
    for group in by_sample.values():
        hit = first_step_metric_at(group, "boundary_ratio", threshold)
        if isinstance(hit, int):
            first_steps.append(hit)
        else:
            missing += 1

    if first_steps:
        return {
            f"sample_step_to_{threshold:g}_boundary_mean": float(mean(first_steps)),
            f"sample_step_to_{threshold:g}_boundary_median": float(median(first_steps)),
            f"sample_step_to_{threshold:g}_boundary_max": int(max(first_steps)),
            f"sample_step_to_{threshold:g}_boundary_reached_count": int(len(first_steps)),
            f"sample_step_to_{threshold:g}_boundary_not_reached_count": int(missing),
        }
    return {
        f"sample_step_to_{threshold:g}_boundary_mean": "nan",
        f"sample_step_to_{threshold:g}_boundary_median": "nan",
        f"sample_step_to_{threshold:g}_boundary_max": "nan",
        f"sample_step_to_{threshold:g}_boundary_reached_count": 0,
        f"sample_step_to_{threshold:g}_boundary_not_reached_count": int(missing),
    }


def summarize_root(root: Path) -> tuple[list[dict], list[dict]]:
    rows = read_csv(root / "per_step_metrics.csv")
    sample_rows = read_csv(root / "per_sample_step_metrics.csv") if (root / "per_sample_step_metrics.csv").exists() else []
    epsilon = float(config_value(root, "epsilon"))
    alpha = float(config_value(root, "alpha"))
    p_order = str(config_value(root, "p_order", "2"))
    q_order = str(config_value(root, "q_order", "2"))
    steps = int(config_value(root, "steps", max(int(r["k"]) for r in rows)))
    nominal_boundary_steps = epsilon / alpha if alpha > 0 else math.nan

    by_method: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)

    final_losses = {}
    for method, group in by_method.items():
        final = max(group, key=lambda r: int(r["k"]))
        final_losses[method] = fnum(final.get("loss3_q_mean"))
    best_final = max(finite_values(final_losses.values()), default=math.nan)

    out_rows = []
    for method, group in sorted(by_method.items()):
        group_sorted = sorted(group, key=lambda r: int(r["k"]))
        final = group_sorted[-1]
        k0 = group_sorted[0]
        k10 = next((r for r in group_sorted if int(r["k"]) >= min(10, steps)), group_sorted[-1])
        final_loss = fnum(final.get("loss3_q_mean"))
        early_growth = fnum(k10.get("loss3_q_mean")) - fnum(k0.get("loss3_q_mean"))
        final_delta_pnorm = fnum(final.get("delta_pnorm_mean"))
        row = {
            "source_root": str(root),
            "epsilon": epsilon,
            "alpha": alpha,
            "nominal_l2_steepest_boundary_steps": nominal_boundary_steps,
            "p_order": p_order,
            "q_order": q_order,
            "method": method,
            "steps": steps,
            "final_loss3_q_mean": final_loss,
            "final_loss3_q_std": fnum(final.get("loss3_q_std")),
            "final_loss3_q_nonfinite_count": fnum(final.get("loss3_q_nonfinite_count")),
            "best_final_loss3_q_mean_in_setting": best_final,
            "final_loss_fraction_of_setting_best": final_loss / best_final if math.isfinite(final_loss) and math.isfinite(best_final) and best_final else math.nan,
            "step_to_90pct_setting_best": first_step_at(group_sorted, 0.90 * best_final),
            "step_to_95pct_setting_best": first_step_at(group_sorted, 0.95 * best_final),
            "loss3_q_growth_0_to_10": early_growth,
            "loss3_q_growth_0_to_10_per_step": early_growth / max(1, int(k10["k"]) - int(k0["k"])) if math.isfinite(early_growth) else math.nan,
            "final_delta_pnorm_mean": final_delta_pnorm,
            "final_delta_pnorm_std": fnum(final.get("delta_pnorm_std")),
            "final_boundary_ratio_mean": fnum(final.get("boundary_ratio_mean")),
            "final_boundary_ratio_std": fnum(final.get("boundary_ratio_std")),
            "final_boundary_gap_mean": epsilon - final_delta_pnorm if math.isfinite(final_delta_pnorm) else math.nan,
            "final_high_frequency_energy_ratio_mean": fnum(final.get("high_frequency_energy_ratio_mean")),
            "final_spectral_centroid_mean": fnum(final.get("spectral_centroid_mean")),
            "final_first_derivative_l2_mean": fnum(final.get("first_derivative_l2_mean")),
            "final_total_variation_mean": fnum(final.get("total_variation_mean")),
            "final_zero_crossing_count_mean": fnum(final.get("zero_crossing_count_mean")),
        }
        for threshold in BOUNDARY_THRESHOLDS:
            row[f"step_to_boundary_ratio_mean_{threshold:g}"] = first_step_metric_at(group_sorted, "boundary_ratio_mean", threshold)
            row.update(boundary_sample_stats(sample_rows, method, threshold))
        out_rows.append(row)

    winner_rows = []
    metrics = [
        ("largest_final_loss", "final_loss3_q_mean", True),
        ("fastest_to_95pct_loss", "step_to_95pct_setting_best", False),
        ("fastest_mean_boundary_99pct", "step_to_boundary_ratio_mean_0.99", False),
        ("fastest_all_samples_boundary_99pct", "sample_step_to_0.99_boundary_max", False),
        ("lowest_high_frequency_ratio", "final_high_frequency_energy_ratio_mean", False),
        ("lowest_first_derivative_l2", "final_first_derivative_l2_mean", False),
        ("lowest_total_variation", "final_total_variation_mean", False),
    ]
    for label, key, high_wins in metrics:
        candidates = []
        for row in out_rows:
            value = fnum(row.get(key))
            if math.isfinite(value):
                candidates.append((value, row))
        if candidates:
            value, row = (max if high_wins else min)(candidates, key=lambda item: item[0])
            winner_rows.append(
                {
                    "source_root": str(root),
                    "epsilon": epsilon,
                    "alpha": alpha,
                    "p_order": p_order,
                    "q_order": q_order,
                    "criterion": label,
                    "winning_method": row["method"],
                    "winning_value": value,
                }
            )
    return out_rows, winner_rows


def method_rollup(summary_rows: list[dict]) -> list[dict]:
    rows = []
    by_method: dict[str, list[dict]] = defaultdict(list)
    for row in summary_rows:
        by_method[row["method"]].append(row)
    for method in sorted(by_method):
        group = by_method[method]
        rows.append(
            {
                "method": method,
                "setting_count": len(group),
                "mean_final_loss3_q": mean(finite_values([r["final_loss3_q_mean"] for r in group])) if finite_values([r["final_loss3_q_mean"] for r in group]) else math.nan,
                "median_step_to_95pct_setting_best": median(finite_values([r["step_to_95pct_setting_best"] for r in group])) if finite_values([r["step_to_95pct_setting_best"] for r in group]) else "nan",
                "median_step_to_boundary_ratio_mean_0.99": median(finite_values([r["step_to_boundary_ratio_mean_0.99"] for r in group])) if finite_values([r["step_to_boundary_ratio_mean_0.99"] for r in group]) else "nan",
                "median_all_samples_boundary_0.99_step": median(finite_values([r["sample_step_to_0.99_boundary_max"] for r in group])) if finite_values([r["sample_step_to_0.99_boundary_max"] for r in group]) else "nan",
                "mean_final_delta_pnorm": mean(finite_values([r["final_delta_pnorm_mean"] for r in group])) if finite_values([r["final_delta_pnorm_mean"] for r in group]) else math.nan,
                "mean_final_boundary_ratio": mean(finite_values([r["final_boundary_ratio_mean"] for r in group])) if finite_values([r["final_boundary_ratio_mean"] for r in group]) else math.nan,
                "mean_final_high_frequency_ratio": mean(finite_values([r["final_high_frequency_energy_ratio_mean"] for r in group])) if finite_values([r["final_high_frequency_energy_ratio_mean"] for r in group]) else math.nan,
                "mean_final_first_derivative_l2": mean(finite_values([r["final_first_derivative_l2_mean"] for r in group])) if finite_values([r["final_first_derivative_l2_mean"] for r in group]) else math.nan,
                "mean_final_total_variation": mean(finite_values([r["final_total_variation_mean"] for r in group])) if finite_values([r["final_total_variation_mean"] for r in group]) else math.nan,
            }
        )
    return rows


def boundary_rows(summary_rows: list[dict]) -> list[dict]:
    fields = [
        "source_root",
        "epsilon",
        "alpha",
        "nominal_l2_steepest_boundary_steps",
        "method",
        "final_delta_pnorm_mean",
        "final_boundary_ratio_mean",
        "final_boundary_gap_mean",
        "step_to_boundary_ratio_mean_0.95",
        "step_to_boundary_ratio_mean_0.99",
        "sample_step_to_0.95_boundary_max",
        "sample_step_to_0.99_boundary_max",
        "sample_step_to_0.99_boundary_reached_count",
        "sample_step_to_0.99_boundary_not_reached_count",
    ]
    return [{field: row.get(field, "") for field in fields} for row in summary_rows]


def fmt(value) -> str:
    value = fnum(value)
    return "nan" if not math.isfinite(value) else f"{value:.4g}"


def markdown_table(rows: list[dict], fields: list[str], limit: int | None = None) -> str:
    rows = rows[:limit] if limit else rows
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        vals = []
        for field in fields:
            value = row.get(field, "")
            vals.append(fmt(value) if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_doc(path: Path, *, base: Path, out_dir: Path, summary_rows: list[dict], winner_rows: list[dict], rollup_rows: list[dict]) -> None:
    completed_roots = sorted({row["source_root"] for row in summary_rows})
    boundary_preview = boundary_rows(summary_rows)
    text = [
        "# Loss3 Alpha/Epsilon Core-Four Sweep Result - 2026-05-19",
        "",
        "Status: generated from completed local sweep outputs.",
        "",
        "## Source Data",
        "",
        "Observed from local files:",
        "",
        f"- Sweep root: `{base}`",
        f"- Analysis root: `{out_dir}`",
        f"- Completed setting roots analyzed: `{len(completed_roots)}`",
        "- Per-setting source: `per_step_metrics.csv`, `per_sample_step_metrics.csv`, `manifest.json`, and `config.json`.",
        "",
        "Methods compared:",
        "",
        "- `raw_add`",
        "- `raw_replace`",
        "- `steepest_add`",
        "- `steepest_replace`",
        "",
        "Metrics used:",
        "",
        "- final `loss3_q` mean/std/nonfinite count",
        "- step to reach 90% and 95% of the best final mean loss within the same epsilon/alpha setting",
        "- `delta_pnorm_mean`, `boundary_ratio_mean`, final boundary gap, and first step to 95%/99% boundary",
        "- per-sample first step to 95%/99% boundary, including the max step needed for all reached samples",
        "- final high-frequency energy ratio, spectral centroid, first-derivative L2, and total variation",
        "",
        "## Method Rollup",
        "",
        markdown_table(
            rollup_rows,
            [
                "method",
                "setting_count",
                "mean_final_loss3_q",
                "median_step_to_95pct_setting_best",
                "median_step_to_boundary_ratio_mean_0.99",
                "median_all_samples_boundary_0.99_step",
                "mean_final_boundary_ratio",
                "mean_final_high_frequency_ratio",
            ],
        ),
        "",
        "## Boundary Arrival Preview",
        "",
        markdown_table(
            boundary_preview,
            [
                "epsilon",
                "alpha",
                "method",
                "nominal_l2_steepest_boundary_steps",
                "final_delta_pnorm_mean",
                "final_boundary_ratio_mean",
                "step_to_boundary_ratio_mean_0.99",
                "sample_step_to_0.99_boundary_max",
                "sample_step_to_0.99_boundary_not_reached_count",
            ],
            limit=80,
        ),
        "",
        "## Per-Setting Winners",
        "",
        markdown_table(winner_rows, ["epsilon", "alpha", "criterion", "winning_method", "winning_value"]),
        "",
        "## Evidence And Interpretation",
        "",
        "Observed from the generated CSV tables above: final-loss, boundary-arrival speed, and smoothness winners are computed within each fixed `(epsilon, alpha, p, q)` setting.",
        "",
        "Inference from these summaries should be made within the fixed `p=2,q=2` scope unless additional P/Q roots are added and analyzed.",
        "",
        "Boundary arrival uses `boundary_ratio = ||delta||_p / epsilon`. A 99% threshold is used as the main practical boundary test to avoid numerical roundoff around exactly `1.0`.",
        "",
        "The smoothness metrics are numerical proxies for whether the final perturbation is high-frequency or sharp. They should be read together with the saved delta trajectory/heatmap figures under each setting root.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.base.is_absolute():
        args.base = PROJECT_ROOT / args.base
    if not args.out_dir.is_absolute():
        args.out_dir = PROJECT_ROOT / args.out_dir
    if not args.doc.is_absolute():
        args.doc = PROJECT_ROOT / args.doc
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    winner_rows: list[dict] = []
    for root in sorted(args.base.glob("fno_nu0p001_eps*_alpha*_batch*_steps*_p*_q*")):
        if not manifest_completed(root):
            print(f"[skip] {root}: manifest is not completed")
            continue
        per_step = root / "per_step_metrics.csv"
        if not per_step.exists():
            print(f"[skip] {root}: missing per_step_metrics.csv")
            continue
        rows, winners = summarize_root(root)
        summary_rows.extend(rows)
        winner_rows.extend(winners)

    rollup_rows = method_rollup(summary_rows)
    boundary_summary_rows = boundary_rows(summary_rows)
    write_csv(args.out_dir / "core4_alpha_epsilon_method_summary.csv", summary_rows)
    write_csv(args.out_dir / "core4_boundary_arrival_summary.csv", boundary_summary_rows)
    write_csv(args.out_dir / "core4_alpha_epsilon_winner_summary.csv", winner_rows)
    write_csv(args.out_dir / "core4_alpha_epsilon_method_rollup.csv", rollup_rows)
    manifest = {
        "status": "completed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_base": str(args.base),
        "summary_rows": len(summary_rows),
        "winner_rows": len(winner_rows),
        "boundary_summary_rows": len(boundary_summary_rows),
        "doc": str(args.doc),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_doc(args.doc, base=args.base, out_dir=args.out_dir, summary_rows=summary_rows, winner_rows=winner_rows, rollup_rows=rollup_rows)
    print(f"[done] wrote {args.out_dir} and {args.doc}")


if __name__ == "__main__":
    main()
