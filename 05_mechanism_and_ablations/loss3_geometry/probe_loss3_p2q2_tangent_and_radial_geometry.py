#!/usr/bin/env python3
"""Probe p=2,q=2 tangent-gradient residual and radial-growth geometry.

This is post-processing only. It reads existing per-sample step metrics from the
Loss 3 p2q2 300-step sweep and computes two diagnostics:

1. p=2 boundary tangent gradient residual
   ||(I - uu^T) grad L|| / ||grad L|| = sqrt(1 - cos(delta, grad)^2)

2. additive radial-growth geometry
   r_{k+1}^2 = r_k^2 + 2 alpha r_k ||d_k|| cos(theta_k) + alpha^2 ||d_k||^2

The goal is to test whether post-boundary loss gain is associated with tangent
KKT residual and whether LP-steepest norm growth is straighter because its
radial step component is more stable than raw PGD's.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SWEEP = PROJECT_ROOT / "forensics" / "loss3_alpha_epsilon_core4_sweep_p2q2_300steps_20260520"
DEFAULT_OUT = PROJECT_ROOT / "forensics" / "loss3_p2q2_tangent_radial_geometry_probe_20260520"
DEFAULT_DOC = PROJECT_ROOT / "docs" / "loss3_p2q2_tangent_radial_geometry_probe_20260520.md"
CORE4 = ("raw_add", "raw_replace", "steepest_add", "steepest_replace")
ADDITIVE = {"raw_add", "steepest_add"}
REPLACEMENT = {"raw_replace", "steepest_replace"}


def fnum(x: Any) -> float:
    if x in {None, "", "nan", "NaN"}:
        return math.nan
    try:
        return float(x)
    except Exception:
        return math.nan


def finite(values: list[Any]) -> list[float]:
    out = []
    for v in values:
        fv = fnum(v)
        if math.isfinite(fv):
            out.append(fv)
    return out


def fmean(values: list[Any]) -> float:
    vals = finite(values)
    return float(mean(vals)) if vals else math.nan


def fmedian(values: list[Any]) -> float:
    vals = finite(values)
    return float(median(vals)) if vals else math.nan


def fstd(values: list[Any]) -> float:
    vals = np.asarray(finite(values), dtype=np.float64)
    return float(np.std(vals, ddof=0)) if vals.size else math.nan


def cv(values: list[Any]) -> float:
    m = fmean(values)
    s = fstd(values)
    return s / abs(m) if math.isfinite(m) and abs(m) > 1e-12 and math.isfinite(s) else math.nan


def pearson(xs: list[Any], ys: list[Any]) -> tuple[float, int]:
    pairs = [(fnum(x), fnum(y)) for x, y in zip(xs, ys)]
    pairs = [(x, y) for x, y in pairs if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return math.nan, len(pairs)
    x = np.asarray([p[0] for p in pairs], dtype=np.float64)
    y = np.asarray([p[1] for p in pairs], dtype=np.float64)
    if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return math.nan, len(pairs)
    return float(np.corrcoef(x, y)[0, 1]), len(pairs)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def setting_from_root(root: Path) -> tuple[float, float]:
    rows = read_csv(root / "per_sample_step_metrics.csv")
    if not rows:
        return math.nan, math.nan
    return fnum(rows[0].get("epsilon")), fnum(rows[0].get("alpha"))


def tangent_ratio_from_cos(c: float) -> float:
    if not math.isfinite(c):
        return math.nan
    c = max(-1.0, min(1.0, c))
    return float(math.sqrt(max(0.0, 1.0 - c * c)))


def load_roots(sweep: Path) -> list[Path]:
    return sorted([p.parent for p in sweep.glob("*/per_sample_step_metrics.csv")], key=lambda p: p.name)


def group_sample_rows(rows: list[dict[str, str]]) -> dict[tuple[str, int], list[dict[str, str]]]:
    groups: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        method = row["method"]
        sample = int(float(row["sample_position"]))
        groups[(method, sample)].append(row)
    for key in groups:
        groups[key].sort(key=lambda r: int(float(r["k"])))
    return groups


def summarize_setting(root: Path, threshold: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = read_csv(root / "per_sample_step_metrics.csv")
    epsilon, alpha = setting_from_root(root)
    groups = group_sample_rows(rows)
    tangent_rows: list[dict[str, Any]] = []
    radial_rows: list[dict[str, Any]] = []

    by_method: dict[str, list[list[dict[str, str]]]] = defaultdict(list)
    for (method, _sample), group in groups.items():
        if method in CORE4:
            by_method[method].append(group)

    for method, sample_groups in by_method.items():
        sample_summaries = []
        radial_step_values = []
        radial_cos_values = []
        radial_pred_values = []
        actual_inc_values = []
        exact_err_values = []
        approx_err_values = []
        direction_norm_values = []
        grad_norm_values = []
        pre_boundary_projection_shrink = []

        for group in sample_groups:
            hit_idx = None
            for i, row in enumerate(group):
                if fnum(row.get("boundary_ratio")) >= threshold:
                    hit_idx = i
                    break
            final = group[-1]
            last_update = group[-2] if len(group) >= 2 else final
            final_loss = fnum(final.get("loss3_q"))
            if hit_idx is None:
                hit = None
                after = []
                hit_loss = math.nan
            else:
                hit = group[hit_idx]
                after = group[hit_idx:]
                hit_loss = fnum(hit.get("loss3_q"))

            tangent_ratios_after = [tangent_ratio_from_cos(fnum(r.get("cos_delta_grad"))) for r in after]
            tangent_norms_after = []
            radial_grad_cos_after = []
            for r in after:
                c = fnum(r.get("cos_delta_grad"))
                g = fnum(r.get("grad_l2"))
                tr = tangent_ratio_from_cos(c)
                if math.isfinite(g) and math.isfinite(tr):
                    tangent_norms_after.append(g * tr)
                if math.isfinite(c):
                    radial_grad_cos_after.append(c)

            # Radial-growth diagnostics for additive methods before projection is active.
            if method in ADDITIVE:
                rows_by_k = {int(float(r["k"])): r for r in group}
                for r in group:
                    k = int(float(r["k"]))
                    nxt = rows_by_k.get(k + 1)
                    if nxt is None:
                        continue
                    boundary = fnum(r.get("boundary_ratio"))
                    next_boundary = fnum(nxt.get("boundary_ratio"))
                    if not (math.isfinite(boundary) and math.isfinite(next_boundary)):
                        continue
                    # Stay away from projection-dominated rows.
                    if boundary >= 0.95 or next_boundary >= 0.995:
                        continue
                    radius = fnum(r.get("delta_l2"))
                    next_radius = fnum(nxt.get("delta_l2"))
                    cos_theta = fnum(r.get("cos_delta_direction"))
                    direction_l2 = fnum(r.get("direction_l2"))
                    grad_l2 = fnum(r.get("grad_l2"))
                    shrink = fnum(r.get("projection_shrink_factor"))
                    if not all(math.isfinite(v) for v in [radius, next_radius, cos_theta, direction_l2, alpha]):
                        continue
                    exact_sq = radius * radius + 2.0 * alpha * radius * direction_l2 * cos_theta + (alpha * direction_l2) ** 2
                    if exact_sq < 0:
                        continue
                    exact_pred = math.sqrt(exact_sq)
                    approx_inc = alpha * direction_l2 * cos_theta
                    actual_inc = next_radius - radius
                    radial_step_values.append(approx_inc)
                    radial_cos_values.append(cos_theta)
                    radial_pred_values.append(exact_pred - radius)
                    actual_inc_values.append(actual_inc)
                    exact_err_values.append(actual_inc - (exact_pred - radius))
                    approx_err_values.append(actual_inc - approx_inc)
                    direction_norm_values.append(direction_l2)
                    grad_norm_values.append(grad_l2)
                    pre_boundary_projection_shrink.append(shrink)

            sample_summaries.append(
                {
                    "hit_step": int(float(hit["k"])) if hit is not None else math.nan,
                    "hit_loss": hit_loss,
                    "final_loss": final_loss,
                    "post_boundary_gain": final_loss - hit_loss if math.isfinite(final_loss) and math.isfinite(hit_loss) else math.nan,
                    "tangent_ratio_at_hit": tangent_ratio_from_cos(fnum(hit.get("cos_delta_grad"))) if hit is not None else math.nan,
                    "tangent_ratio_after_mean": fmean(tangent_ratios_after),
                    "tangent_ratio_last_update": tangent_ratio_from_cos(fnum(last_update.get("cos_delta_grad"))),
                    "tangent_norm_after_mean": fmean(tangent_norms_after),
                    "radial_grad_cos_after_mean": fmean(radial_grad_cos_after),
                    "delta_angle_after_mean": fmean([r.get("delta_prev_angle_degrees") for r in after]),
                    "loss_gain_10_after_hit": fnum(group[hit_idx + 10].get("loss3_q")) - hit_loss if hit_idx is not None and hit_idx + 10 < len(group) and math.isfinite(hit_loss) else math.nan,
                }
            )

        tangent_rows.append(
            {
                "source_root": str(root),
                "epsilon": epsilon,
                "alpha": alpha,
                "method": method,
                "sample_count": len(sample_summaries),
                "mean_hit_step": fmean([s["hit_step"] for s in sample_summaries]),
                "mean_post_boundary_gain": fmean([s["post_boundary_gain"] for s in sample_summaries]),
                "mean_loss_gain_10_after_hit": fmean([s["loss_gain_10_after_hit"] for s in sample_summaries]),
                "mean_tangent_grad_ratio_at_hit": fmean([s["tangent_ratio_at_hit"] for s in sample_summaries]),
                "mean_tangent_grad_ratio_after_hit": fmean([s["tangent_ratio_after_mean"] for s in sample_summaries]),
                "mean_tangent_grad_ratio_last_update": fmean([s["tangent_ratio_last_update"] for s in sample_summaries]),
                "mean_tangent_grad_norm_after_hit": fmean([s["tangent_norm_after_mean"] for s in sample_summaries]),
                "mean_radial_grad_cos_after_hit": fmean([s["radial_grad_cos_after_mean"] for s in sample_summaries]),
                "mean_delta_angle_after_hit": fmean([s["delta_angle_after_mean"] for s in sample_summaries]),
            }
        )

        if method in ADDITIVE:
            radial_rows.append(
                {
                    "source_root": str(root),
                    "epsilon": epsilon,
                    "alpha": alpha,
                    "method": method,
                    "pre_boundary_step_count": len(actual_inc_values),
                    "mean_cos_theta_delta_direction": fmean(radial_cos_values),
                    "std_cos_theta_delta_direction": fstd(radial_cos_values),
                    "cv_cos_theta_delta_direction": cv(radial_cos_values),
                    "mean_direction_l2": fmean(direction_norm_values),
                    "std_direction_l2": fstd(direction_norm_values),
                    "cv_direction_l2": cv(direction_norm_values),
                    "mean_grad_l2": fmean(grad_norm_values),
                    "std_grad_l2": fstd(grad_norm_values),
                    "cv_grad_l2": cv(grad_norm_values),
                    "mean_predicted_radial_increment_exact": fmean(radial_pred_values),
                    "std_predicted_radial_increment_exact": fstd(radial_pred_values),
                    "cv_predicted_radial_increment_exact": cv(radial_pred_values),
                    "mean_predicted_radial_increment_first_order": fmean(radial_step_values),
                    "std_predicted_radial_increment_first_order": fstd(radial_step_values),
                    "cv_predicted_radial_increment_first_order": cv(radial_step_values),
                    "mean_actual_radial_increment": fmean(actual_inc_values),
                    "std_actual_radial_increment": fstd(actual_inc_values),
                    "cv_actual_radial_increment": cv(actual_inc_values),
                    "mean_exact_prediction_error": fmean(exact_err_values),
                    "mae_exact_prediction_error": fmean([abs(v) for v in exact_err_values]),
                    "mean_first_order_prediction_error": fmean(approx_err_values),
                    "mae_first_order_prediction_error": fmean([abs(v) for v in approx_err_values]),
                    "mean_pre_boundary_projection_shrink": fmean(pre_boundary_projection_shrink),
                }
            )

    return tangent_rows, radial_rows


def rollup(rows: list[dict[str, Any]], key: str = "method") -> list[dict[str, Any]]:
    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    metric_keys = [k for k in rows[0].keys() if k not in {"source_root", "method"}] if rows else []
    out = []
    for group_key, group in sorted(groups.items(), key=lambda x: str(x[0])):
        row = {key: group_key, "row_count": len(group)}
        for metric in metric_keys:
            row[f"mean_{metric}"] = fmean([g.get(metric) for g in group])
            row[f"median_{metric}"] = fmedian([g.get(metric) for g in group])
        out.append(row)
    return out


def correlation_rows(tangent_rows: list[dict[str, Any]], radial_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for method in CORE4:
        group = [r for r in tangent_rows if r["method"] == method]
        for metric in [
            "mean_tangent_grad_ratio_at_hit",
            "mean_tangent_grad_ratio_after_hit",
            "mean_tangent_grad_ratio_last_update",
            "mean_delta_angle_after_hit",
        ]:
            corr, n = pearson([r["mean_post_boundary_gain"] for r in group], [r.get(metric) for r in group])
            out.append({"scope": "setting_level_by_method", "method": method, "x": "post_boundary_gain", "y": metric, "pearson_r": corr, "pair_count": n})
    for method in ADDITIVE:
        group = [r for r in radial_rows if r["method"] == method]
        for metric in [
            "cv_cos_theta_delta_direction",
            "cv_direction_l2",
            "cv_predicted_radial_increment_exact",
            "cv_actual_radial_increment",
            "mae_exact_prediction_error",
        ]:
            corr, n = pearson([r["cv_actual_radial_increment"] for r in group], [r.get(metric) for r in group])
            out.append({"scope": "radial_growth_setting_level", "method": method, "x": "cv_actual_radial_increment", "y": metric, "pearson_r": corr, "pair_count": n})
    return out


def md_table(rows: list[dict[str, Any]], fields: list[str], max_rows: int | None = None) -> str:
    show = rows[:max_rows] if max_rows is not None else rows
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in show:
        vals = []
        for field in fields:
            value = row.get(field, "")
            fv = fnum(value)
            if math.isfinite(fv):
                vals.append(f"{fv:.4g}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_doc(doc: Path, out_dir: Path, tangent_roll: list[dict[str, Any]], radial_roll: list[dict[str, Any]], corr: list[dict[str, Any]]) -> None:
    lines = [
        "# Loss 3 p2q2 Tangent and Radial Geometry Probe - 2026-05-20",
        "",
        "Status: generated from existing p2q2 300-step per-sample metrics; no optimizer/model experiment was rerun.",
        "",
        "## Tables",
        "",
        f"- Tangent diagnostics by setting/method: `{rel(out_dir / 'tables' / 'tangent_kkt_by_setting_method.csv')}`",
        f"- Tangent diagnostics rollup: `{rel(out_dir / 'tables' / 'tangent_kkt_rollup_by_method.csv')}`",
        f"- Radial growth diagnostics by setting/method: `{rel(out_dir / 'tables' / 'radial_growth_by_setting_method.csv')}`",
        f"- Radial growth diagnostics rollup: `{rel(out_dir / 'tables' / 'radial_growth_rollup_by_method.csv')}`",
        f"- Correlations: `{rel(out_dir / 'tables' / 'geometry_probe_correlations.csv')}`",
        "",
        "## Tangent Gradient Projection Diagnostic",
        "",
        "For p=2, with `u = delta / ||delta||_2`, the normalized tangent-gradient residual is computed from the recorded gradient cosine:",
        "",
        "```text",
        "||(I - u u^T) grad L|| / ||grad L|| = sqrt(1 - cos(delta, grad)^2)",
        "```",
        "",
        md_table(
            tangent_roll,
            [
                "method",
                "row_count",
                "mean_mean_hit_step",
                "mean_mean_post_boundary_gain",
                "mean_mean_tangent_grad_ratio_at_hit",
                "mean_mean_tangent_grad_ratio_after_hit",
                "mean_mean_tangent_grad_ratio_last_update",
                "mean_mean_delta_angle_after_hit",
            ],
        ),
        "",
        "## Radial Growth Diagnostic for Additive Methods",
        "",
        "For p=2 additive updates, the exact unprojected radius formula is:",
        "",
        "```text",
        "r_next^2 = r^2 + 2 alpha r ||d|| cos(theta) + alpha^2 ||d||^2",
        "```",
        "",
        "Rows below use pre-boundary steps only, so projection should not dominate the comparison.",
        "",
        md_table(
            radial_roll,
            [
                "method",
                "row_count",
                "mean_pre_boundary_step_count",
                "mean_mean_cos_theta_delta_direction",
                "mean_cv_cos_theta_delta_direction",
                "mean_mean_direction_l2",
                "mean_cv_direction_l2",
                "mean_mean_actual_radial_increment",
                "mean_cv_actual_radial_increment",
                "mean_mae_exact_prediction_error",
            ],
        ),
        "",
        "## Correlations",
        "",
        md_table(corr, ["scope", "method", "x", "y", "pearson_r", "pair_count"], max_rows=40),
        "",
        "## Interpretation",
        "",
        "Observed evidence here directly tests the p=2 tangent-plane story using gradients recorded during the original runs. A high tangent residual after boundary hit means the optimizer is not boundary-stationary and can still improve by rotating on the L2 sphere.",
        "",
        "The radial-growth table tests the LP-steepest straight-line explanation. If `steepest_add` has `direction_l2` near 1 and lower CV of actual/predicted radial increments than `raw_add`, then its straighter boundary-ratio curves are explained by normalized step size plus reasonably stable radial alignment. Raw PGD can curve because both direction norm and radial alignment vary.",
        "",
        "Caveat: these diagnostics use metrics recorded at each step, not newly recomputed gradients. They are still based on the actual autograd gradients saved during the experiments, but full vector projection plots would require storing gradient vectors or rerunning a GPU diagnostic.",
    ]
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-root", type=Path, default=DEFAULT_SWEEP)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    parser.add_argument("--boundary-threshold", type=float, default=0.99)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sweep = args.sweep_root if args.sweep_root.is_absolute() else PROJECT_ROOT / args.sweep_root
    out_dir = args.out_dir if args.out_dir.is_absolute() else PROJECT_ROOT / args.out_dir
    doc = args.doc if args.doc.is_absolute() else PROJECT_ROOT / args.doc
    roots = load_roots(sweep)
    tangent_rows: list[dict[str, Any]] = []
    radial_rows: list[dict[str, Any]] = []
    for root in roots:
        trows, rrows = summarize_setting(root, args.boundary_threshold)
        tangent_rows.extend(trows)
        radial_rows.extend(rrows)
    tangent_roll = rollup(tangent_rows)
    radial_roll = rollup(radial_rows)
    corr = correlation_rows(tangent_rows, radial_rows)
    tables = out_dir / "tables"
    write_csv(tables / "tangent_kkt_by_setting_method.csv", tangent_rows)
    write_csv(tables / "tangent_kkt_rollup_by_method.csv", tangent_roll)
    write_csv(tables / "radial_growth_by_setting_method.csv", radial_rows)
    write_csv(tables / "radial_growth_rollup_by_method.csv", radial_roll)
    write_csv(tables / "geometry_probe_correlations.csv", corr)
    manifest = {
        "status": "completed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sweep_root": str(sweep),
        "out_dir": str(out_dir),
        "doc": str(doc),
        "setting_root_count": len(roots),
        "tangent_row_count": len(tangent_rows),
        "radial_row_count": len(radial_rows),
        "boundary_threshold": args.boundary_threshold,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_doc(doc, out_dir, tangent_roll, radial_roll, corr)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
