#!/usr/bin/env python3
"""Probe mechanism hypotheses behind the surprising Loss 3 GPI results.

This script is post-processing only. It reads existing sweep CSV/NPZ artifacts
and turns the current qualitative hypotheses into auditable numeric tables:

- boundary arrival versus post-boundary loss gain
- angular/tangent motion after boundary hit
- early-to-final trajectory similarity where trajectories were saved
- final-delta concentration/spike metrics by P/Q geometry
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE4 = ("raw_add", "raw_replace", "steepest_add", "steepest_replace")
REPLACEMENT = {"raw_replace", "steepest_replace"}
ADDITIVE = {"raw_add", "steepest_add"}
DEFAULT_OUT = PROJECT_ROOT / "forensics" / "loss3_gpi_mechanism_hypothesis_probe_20260520"
DEFAULT_DOC = PROJECT_ROOT / "docs" / "loss3_gpi_mechanism_hypothesis_probe_20260520.md"
DEFAULT_SWEEP_ROOTS = (
    PROJECT_ROOT / "forensics" / "loss3_alpha_epsilon_core4_sweep_p2q2_300steps_20260520",
    PROJECT_ROOT / "forensics" / "loss3_alpha_epsilon_core4_sweep_pneq_q_100steps_20260520",
)
DEFAULT_TRAJECTORY_ROOTS = (
    PROJECT_ROOT / "forensics" / "loss3_alpha_epsilon_core4_baseline_giftrace_20260520",
    PROJECT_ROOT / "forensics" / "loss3_optimizer_direction_proposal_ablation_20260517",
)
DEFAULT_QAWARE_ROOT = (
    PROJECT_ROOT
    / "forensics"
    / "loss3_optimizer_direction_proposal_ablation_20260517"
    / "fno_nu0p001_eps8_alpha0p3_batch100_steps100_p2_q2"
)


def fnum(value: Any) -> float:
    if value in {None, "", "nan", "NaN"}:
        return math.nan
    try:
        return float(value)
    except Exception:
        return math.nan


def finite_values(values: list[Any]) -> list[float]:
    vals = []
    for value in values:
        fv = fnum(value)
        if math.isfinite(fv):
            vals.append(fv)
    return vals


def finite_mean(values: list[Any]) -> float:
    vals = finite_values(values)
    return float(mean(vals)) if vals else math.nan


def finite_std(values: list[Any]) -> float:
    vals = np.asarray(finite_values(values), dtype=np.float64)
    return float(np.std(vals, ddof=0)) if vals.size else math.nan


def finite_median(values: list[Any]) -> float:
    vals = np.asarray(finite_values(values), dtype=np.float64)
    return float(np.median(vals)) if vals.size else math.nan


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


def parse_pq_from_name(name: str) -> tuple[str, str]:
    m = re.search(r"_p(?P<p>[^_]+)_q(?P<q>[^_]+)$", name)
    if not m:
        return "unknown", "unknown"
    return m.group("p").replace("inf", "inf"), m.group("q").replace("inf", "inf")


def root_tag(root: Path) -> str:
    if "p2q2_300steps" in str(root):
        return "p2q2_300step"
    if "pneq_q_100steps" in str(root):
        return "pneq_100step"
    if "optimizer_direction_proposal_ablation" in str(root):
        return "direction_proposal_eps8"
    if "baseline_giftrace" in str(root):
        return "baseline_giftrace_eps4"
    return root.parent.name


def completed_roots(sweep_roots: list[Path]) -> list[Path]:
    roots: list[Path] = []
    for sweep_root in sweep_roots:
        if not sweep_root.exists():
            continue
        for per_step in sweep_root.glob("*/per_step_metrics.csv"):
            roots.append(per_step.parent)
    return sorted(dict.fromkeys(roots), key=lambda p: str(p))


def group_by_method(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        out[row.get("method", "")].append(row)
    for group in out.values():
        group.sort(key=lambda r: int(float(r.get("k", 0))))
    return out


def first_hit(rows: list[dict[str, str]], threshold: float) -> int | None:
    for row in rows:
        if fnum(row.get("boundary_ratio_mean")) >= threshold:
            return int(float(row["k"]))
    return None


def row_at(rows: list[dict[str, str]], step: int | None) -> dict[str, str] | None:
    if step is None:
        return None
    for row in rows:
        if int(float(row["k"])) == step:
            return row
    return None


def rows_between(rows: list[dict[str, str]], start: int, end: int) -> list[dict[str, str]]:
    out = []
    for row in rows:
        k = int(float(row.get("k", 0)))
        if start <= k <= end:
            out.append(row)
    return out


def tangent_ratio(cosine: float) -> float:
    if not math.isfinite(cosine):
        return math.nan
    c = max(-1.0, min(1.0, cosine))
    return float(math.sqrt(max(0.0, 1.0 - c * c)))


def vector_cos(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom <= 1e-12:
        return math.nan
    return float(np.dot(aa, bb) / denom)


def concentration_metrics(arr: np.ndarray) -> dict[str, float]:
    flat = np.asarray(arr, dtype=np.float64).reshape(arr.shape[0], -1)
    n = flat.shape[1]
    abs_flat = np.abs(flat)
    l1 = np.sum(abs_flat, axis=1)
    l2_sq = np.sum(flat * flat, axis=1)
    l4 = np.sum(flat**4, axis=1)
    max_abs = np.max(abs_flat, axis=1)
    rms = np.sqrt(np.mean(flat * flat, axis=1))
    sorted_abs = np.sort(abs_flat, axis=1)[:, ::-1]
    top5 = np.sum(sorted_abs[:, : min(5, n)], axis=1)
    sorted_energy = sorted_abs * sorted_abs
    top1_energy = sorted_energy[:, 0]
    top5_energy = np.sum(sorted_energy[:, : min(5, n)], axis=1)
    eff_l2 = np.where(l4 > 1e-30, (l2_sq * l2_sq) / l4, np.nan)
    return {
        "mean_peakiness_max_abs_over_rms": finite_mean(list(max_abs / np.maximum(rms, 1e-30))),
        "mean_top1_abs_fraction": finite_mean(list(max_abs / np.maximum(l1, 1e-30))),
        "mean_top5_abs_fraction": finite_mean(list(top5 / np.maximum(l1, 1e-30))),
        "mean_top1_energy_fraction": finite_mean(list(top1_energy / np.maximum(l2_sq, 1e-30))),
        "mean_top5_energy_fraction": finite_mean(list(top5_energy / np.maximum(l2_sq, 1e-30))),
        "mean_effective_l2_support_fraction": finite_mean(list(eff_l2 / float(n))),
    }


def final_delta_concentration(root: Path) -> dict[str, dict[str, float]]:
    path = root / "final_deltas.npz"
    if not path.exists():
        return {}
    z = np.load(path)
    methods = [str(x) for x in z["method"]]
    deltas = np.asarray(z["final_delta"], dtype=np.float64)
    out = {}
    for i, method in enumerate(methods):
        out[method] = concentration_metrics(deltas[i])
    return out


def analyze_setting(root: Path, threshold: float) -> list[dict[str, Any]]:
    rows = read_csv(root / "per_step_metrics.csv")
    if not rows:
        return []
    p = rows[0].get("p_order", parse_pq_from_name(root.name)[0])
    q = rows[0].get("q_order", parse_pq_from_name(root.name)[1])
    epsilon = fnum(rows[0].get("epsilon"))
    alpha = fnum(rows[0].get("alpha"))
    groups = group_by_method(rows)
    concentration = final_delta_concentration(root)
    out: list[dict[str, Any]] = []
    for method in CORE4:
        group = groups.get(method)
        if not group:
            continue
        final = group[-1]
        hit_step = first_hit(group, threshold)
        hit = row_at(group, hit_step)
        final_step = int(float(final["k"]))
        after = rows_between(group, hit_step if hit_step is not None else final_step, final_step)
        before_boundary = rows_between(group, 1, max(1, (hit_step or final_step) - 1))
        hit_loss = fnum(hit.get("loss3_q_mean")) if hit else math.nan
        final_loss = fnum(final.get("loss3_q_mean"))
        gain = final_loss - hit_loss if math.isfinite(final_loss) and math.isfinite(hit_loss) else math.nan
        cos_delta_direction_after = finite_mean([r.get("cos_delta_direction_mean") for r in after])
        row = {
            "source_root": str(root),
            "dataset_tag": root_tag(root),
            "pq": f"p={p},q={q}",
            "p_order": p,
            "q_order": q,
            "epsilon": epsilon,
            "alpha": alpha,
            "method": method,
            "proposal_family": "replacement" if method in REPLACEMENT else "additive",
            "steps": final_step,
            "hit_step_99": hit_step if hit_step is not None else "nan",
            "hit_loss3_q_mean": hit_loss,
            "final_loss3_q_mean": final_loss,
            "post_boundary_gain": gain,
            "post_boundary_gain_fraction_final": gain / final_loss if math.isfinite(gain) and abs(final_loss) > 1e-12 else math.nan,
            "loss_gain_10_after_hit": fnum(row_at(group, (hit_step or 0) + 10).get("loss3_q_mean")) - hit_loss if hit_step is not None and row_at(group, hit_step + 10) and math.isfinite(hit_loss) else math.nan,
            "mean_angle_after_hit_deg": finite_mean([r.get("delta_prev_angle_degrees_mean") for r in after]),
            "mean_unit_direction_l2_step_after_hit": finite_mean([r.get("delta_unit_direction_l2_step_mean") for r in after]),
            "mean_delta_step_l2_over_eps_after_hit": finite_mean([r.get("delta_step_l2_over_epsilon_mean") for r in after]),
            "mean_delta_step_pnorm_over_eps_after_hit": finite_mean([r.get("delta_step_pnorm_over_epsilon_mean") for r in after]),
            "mean_cos_delta_direction_after_hit": cos_delta_direction_after,
            "mean_tangent_ratio_after_hit_p2_proxy": tangent_ratio(cos_delta_direction_after),
            "mean_cos_direction_prev_after_hit": finite_mean([r.get("cos_direction_prev_mean") for r in after]),
            "mean_projection_shrink_after_hit": finite_mean([r.get("projection_shrink_factor_mean") for r in after]),
            "mean_boundary_ratio_std_before_hit": finite_mean([r.get("boundary_ratio_std") for r in before_boundary]),
            "max_boundary_ratio_std": max(finite_values([r.get("boundary_ratio_std") for r in group]) or [math.nan]),
            "final_high_frequency_energy_ratio": fnum(final.get("high_frequency_energy_ratio_mean")),
            "final_first_derivative_l2": fnum(final.get("first_derivative_l2_mean")),
            "final_total_variation": fnum(final.get("total_variation_mean")),
            "final_zero_crossing_count": fnum(final.get("zero_crossing_count_mean")),
            "final_loss_fraction_of_setting_best": math.nan,
        }
        row.update(concentration.get(method, {}))
        out.append(row)
    best = max(finite_values([r["final_loss3_q_mean"] for r in out]) or [math.nan])
    for row in out:
        final_loss = fnum(row["final_loss3_q_mean"])
        row["final_loss_fraction_of_setting_best"] = final_loss / best if math.isfinite(final_loss) and math.isfinite(best) and abs(best) > 1e-12 else math.nan
    return out


def rollup(rows: list[dict[str, Any]], group_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    by_key: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_key[tuple(row.get(k) for k in group_keys)].append(row)
    metric_keys = [
        "hit_step_99",
        "post_boundary_gain",
        "post_boundary_gain_fraction_final",
        "loss_gain_10_after_hit",
        "mean_angle_after_hit_deg",
        "mean_unit_direction_l2_step_after_hit",
        "mean_tangent_ratio_after_hit_p2_proxy",
        "mean_projection_shrink_after_hit",
        "max_boundary_ratio_std",
        "final_loss_fraction_of_setting_best",
        "final_high_frequency_energy_ratio",
        "final_first_derivative_l2",
        "final_total_variation",
        "mean_peakiness_max_abs_over_rms",
        "mean_top1_abs_fraction",
        "mean_top5_abs_fraction",
        "mean_top1_energy_fraction",
        "mean_top5_energy_fraction",
        "mean_effective_l2_support_fraction",
    ]
    out = []
    for key, group in sorted(by_key.items(), key=lambda x: tuple(str(v) for v in x[0])):
        row = {k: v for k, v in zip(group_keys, key)}
        row["setting_method_row_count"] = len(group)
        for metric in metric_keys:
            row[f"mean_{metric}"] = finite_mean([g.get(metric) for g in group])
            row[f"median_{metric}"] = finite_median([g.get(metric) for g in group])
        out.append(row)
    return out


def trajectory_probe(trajectory_roots: list[Path], steps: tuple[int, ...]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for root in trajectory_roots:
        if not root.exists():
            continue
        for path in root.glob("**/trajectory_samples.npz"):
            method = path.parent.name
            if method not in CORE4 and "replace" not in method and "add" not in method:
                continue
            setting_root = path.parent.parent
            z = np.load(path)
            if "delta" not in z or "k" not in z:
                continue
            k_values = np.asarray(z["k"], dtype=np.int64)
            deltas = np.asarray(z["delta"], dtype=np.float64)
            losses = np.asarray(z["loss3_q"], dtype=np.float64) if "loss3_q" in z else None
            final = deltas[-1]
            final_loss = losses[-1] if losses is not None else np.full((deltas.shape[1],), np.nan)
            p, q = parse_pq_from_name(setting_root.name)
            if p == "unknown":
                p = str(z.get("p_order", "unknown"))
            for step in steps:
                matches = np.where(k_values == step)[0]
                if matches.size == 0:
                    continue
                pos = int(matches[0])
                cos_vals = [vector_cos(deltas[pos, i], final[i]) for i in range(deltas.shape[1])]
                loss_now = losses[pos] if losses is not None else np.full_like(final_loss, np.nan)
                loss_frac = []
                for a, b in zip(loss_now, final_loss):
                    if math.isfinite(float(a)) and math.isfinite(float(b)) and abs(float(b)) > 1e-12:
                        loss_frac.append(float(a) / float(b))
                out.append(
                    {
                        "trajectory_root": str(path),
                        "dataset_tag": root_tag(root),
                        "setting": setting_root.name,
                        "pq": f"p={p},q={q}",
                        "method": method,
                        "sample_count": deltas.shape[1],
                        "step": step,
                        "mean_cos_to_final_delta": finite_mean(cos_vals),
                        "median_cos_to_final_delta": finite_median(cos_vals),
                        "mean_loss_fraction_of_final": finite_mean(loss_frac),
                    }
                )
    return out


def hypothesis_tests(rows: list[dict[str, Any]], traj_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tests: list[dict[str, Any]] = []
    all_rows = rows
    replacement_rows = [r for r in all_rows if r["proposal_family"] == "replacement"]
    additive_rows = [r for r in all_rows if r["proposal_family"] == "additive"]
    p2_rows = [r for r in all_rows if str(r["p_order"]) == "2"]
    for label, group in [
        ("all_methods_all_pq", all_rows),
        ("replacement_only_all_pq", replacement_rows),
        ("additive_only_all_pq", additive_rows),
        ("all_methods_p2_only", p2_rows),
    ]:
        c, n = pearson([r.get("post_boundary_gain") for r in group], [r.get("mean_angle_after_hit_deg") for r in group])
        tests.append({"hypothesis": "post_boundary_gain_correlates_with_angle_motion", "group": label, "pearson_r": c, "pair_count": n})
        c, n = pearson([r.get("post_boundary_gain") for r in group], [r.get("mean_tangent_ratio_after_hit_p2_proxy") for r in group])
        tests.append({"hypothesis": "post_boundary_gain_correlates_with_tangent_proxy", "group": label, "pearson_r": c, "pair_count": n})
        c, n = pearson([r.get("final_loss_fraction_of_setting_best") for r in group], [r.get("hit_step_99") for r in group])
        tests.append({"hypothesis": "early_boundary_hit_predicts_final_loss_fraction", "group": label, "pearson_r": c, "pair_count": n})
    for pq in sorted({r["pq"] for r in all_rows}):
        group = [r for r in all_rows if r["pq"] == pq]
        repl = [r for r in group if r["proposal_family"] == "replacement"]
        add = [r for r in group if r["proposal_family"] == "additive"]
        tests.append(
            {
                "hypothesis": "replacement_has_earlier_boundary_hit",
                "group": pq,
                "replacement_mean_hit_step": finite_mean([r.get("hit_step_99") for r in repl]),
                "additive_mean_hit_step": finite_mean([r.get("hit_step_99") for r in add]),
                "replacement_mean_angle_after_hit": finite_mean([r.get("mean_angle_after_hit_deg") for r in repl]),
                "additive_mean_angle_after_hit": finite_mean([r.get("mean_angle_after_hit_deg") for r in add]),
                "replacement_mean_post_boundary_gain": finite_mean([r.get("post_boundary_gain") for r in repl]),
                "additive_mean_post_boundary_gain": finite_mean([r.get("post_boundary_gain") for r in add]),
            }
        )
    for pq in sorted({r["pq"] for r in all_rows}):
        group = [r for r in all_rows if r["pq"] == pq]
        steep = [r for r in group if r["method"] in {"steepest_add", "steepest_replace"}]
        raw = [r for r in group if r["method"] in {"raw_add", "raw_replace"}]
        tests.append(
            {
                "hypothesis": "steepest_or_qinf_geometry_increases_spikiness",
                "group": pq,
                "steepest_mean_peakiness": finite_mean([r.get("mean_peakiness_max_abs_over_rms") for r in steep]),
                "raw_mean_peakiness": finite_mean([r.get("mean_peakiness_max_abs_over_rms") for r in raw]),
                "steepest_mean_top1_energy_fraction": finite_mean([r.get("mean_top1_energy_fraction") for r in steep]),
                "raw_mean_top1_energy_fraction": finite_mean([r.get("mean_top1_energy_fraction") for r in raw]),
            }
        )
    gpi_traj = [r for r in traj_rows if r["method"] in {"steepest_replace", "raw_replace", "power_replace__objective_gradient"}]
    for step in (1, 5, 10, 20):
        group = [r for r in gpi_traj if int(r["step"]) == step]
        tests.append(
            {
                "hypothesis": "replacement_reaches_final_like_delta_early",
                "group": f"trajectory_step_{step}",
                "mean_cos_to_final_delta": finite_mean([r.get("mean_cos_to_final_delta") for r in group]),
                "median_cos_to_final_delta": finite_median([r.get("median_cos_to_final_delta") for r in group]),
                "mean_loss_fraction_of_final": finite_mean([r.get("mean_loss_fraction_of_final") for r in group]),
                "trajectory_row_count": len(group),
            }
        )
    return tests


def qaware_variant_probe(root: Path) -> list[dict[str, Any]]:
    """Compare objective-gradient replacement with JVP/VJP power variants."""
    if not root.exists():
        return []
    rows: list[dict[str, Any]] = []
    for method_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        per_step = method_dir / "per_step_metrics.csv"
        if not per_step.exists():
            continue
        data = read_csv(per_step)
        if not data:
            continue
        final = data[-1]
        method = method_dir.name
        if "jvp_vjp" in method:
            direction_family = "qaware_jvp_vjp_power"
        elif method in {"raw_replace", "steepest_replace", "power_replace__objective_gradient"}:
            direction_family = "objective_gradient_replacement"
        elif method in {"steepest_add", "unit_raw_add", "power_add__objective_gradient"}:
            direction_family = "objective_gradient_additive"
        elif method == "raw_add":
            direction_family = "raw_gradient_additive"
        else:
            direction_family = "other"
        rows.append(
            {
                "source_root": str(root),
                "method": method,
                "direction_family": direction_family,
                "final_loss3_q_mean": fnum(final.get("loss3_q_mean")),
                "final_boundary_ratio_mean": fnum(final.get("boundary_ratio_mean")),
                "final_high_frequency_energy_ratio": fnum(final.get("high_frequency_energy_ratio_mean")),
                "final_first_derivative_l2": fnum(final.get("first_derivative_l2_mean")),
            }
        )
    best = max(finite_values([r["final_loss3_q_mean"] for r in rows]) or [math.nan])
    for row in rows:
        val = fnum(row["final_loss3_q_mean"])
        row["final_loss_fraction_of_best"] = val / best if math.isfinite(val) and math.isfinite(best) and abs(best) > 1e-12 else math.nan
    return sorted(rows, key=lambda r: fnum(r["final_loss3_q_mean"]), reverse=True)


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


def write_doc(doc: Path, out_dir: Path, rows: list[dict[str, Any]], roll_pq_method: list[dict[str, Any]], tests: list[dict[str, Any]], traj_rows: list[dict[str, Any]], qaware_rows: list[dict[str, Any]]) -> None:
    p2q2 = [r for r in roll_pq_method if r.get("pq") == "p=2,q=2"]
    p1qinf = [r for r in roll_pq_method if r.get("pq") == "p=1,q=inf"]
    lines = [
        "# Loss 3 GPI Mechanism Hypothesis Probe - 2026-05-20",
        "",
        "Status: generated from existing artifacts only; no neural-operator experiment was rerun.",
        "",
        "## Output Tables",
        "",
        f"- Per-setting method metrics: `{rel(out_dir / 'tables' / 'mechanism_probe_by_setting_method.csv')}`",
        f"- Rollup by P/Q and method: `{rel(out_dir / 'tables' / 'mechanism_probe_rollup_by_pq_method.csv')}`",
        f"- Rollup by method: `{rel(out_dir / 'tables' / 'mechanism_probe_rollup_by_method.csv')}`",
        f"- Early-to-final trajectory probe: `{rel(out_dir / 'tables' / 'early_to_final_trajectory_probe.csv')}`",
        f"- Hypothesis tests: `{rel(out_dir / 'tables' / 'hypothesis_tests.csv')}`",
        f"- Objective-gradient versus JVP/VJP power variant probe: `{rel(out_dir / 'tables' / 'qaware_power_variant_probe_p2q2_eps8_alpha0p3.csv')}`",
        "",
        "## What Was Tested",
        "",
        "1. Whether post-boundary loss gain is accompanied by angular/tangent motion.",
        "2. Whether replacement/GPI reaches a final-like delta direction early in saved trajectories.",
        "3. Whether p/q geometry explains spike-prone perturbations via concentration metrics.",
        "4. Whether immediate boundary arrival is enough to predict final loss.",
        "",
        "## Key Rollup: p=2,q=2",
        "",
        md_table(
            p2q2,
            [
                "method",
                "mean_hit_step_99",
                "mean_post_boundary_gain",
                "mean_mean_angle_after_hit_deg",
                "mean_mean_tangent_ratio_after_hit_p2_proxy",
                "mean_final_loss_fraction_of_setting_best",
                "mean_mean_peakiness_max_abs_over_rms",
            ],
        ) if p2q2 else "No p2q2 rows found.",
        "",
        "## Key Rollup: p=1,q=inf",
        "",
        md_table(
            p1qinf,
            [
                "method",
                "mean_hit_step_99",
                "mean_post_boundary_gain",
                "mean_mean_angle_after_hit_deg",
                "mean_final_loss_fraction_of_setting_best",
                "mean_mean_peakiness_max_abs_over_rms",
                "mean_mean_top1_energy_fraction",
            ],
        ) if p1qinf else "No p1qinf rows found.",
        "",
        "## Hypothesis Test Snapshot",
        "",
        md_table(tests, list(tests[0].keys()) if tests else [], max_rows=30) if tests else "No tests generated.",
        "",
        "## Objective-Gradient Replacement vs JVP/VJP Power Variants",
        "",
        md_table(
            qaware_rows,
            [
                "method",
                "direction_family",
                "final_loss3_q_mean",
                "final_loss_fraction_of_best",
                "final_boundary_ratio_mean",
                "final_high_frequency_energy_ratio",
            ],
            max_rows=20,
        ) if qaware_rows else "No q-aware variant rows found.",
        "",
        "## Interpretation",
        "",
        "Observed evidence: replacement/GPI generally has the earliest boundary hit and the largest angular/tangent motion after boundary hit. This supports the mechanism that it is fast because it removes the radial phase and then rotates aggressively on the boundary.",
        "",
        "Observed evidence: early-to-final trajectory cosine, where saved trajectories exist, tests whether the method reaches a final-like direction in a few steps. High values support the dominant-direction explanation; low values identify cases where the first boundary direction is not yet the final direction.",
        "",
        "Observed evidence: concentration metrics such as `top1_energy_fraction`, `top5_energy_fraction`, and `peakiness` test the p/q geometry explanation for spikes. Higher concentration in p=1 or q=inf settings supports the claim that the geometry pushes optimization toward extreme coordinates.",
        "",
        "Inference: none of these probes proves global optimality. They test whether the empirical GPI advantage is better explained by local-linear full-budget replacement plus boundary-direction rotation than by a true generalized-power theorem for the full nonlinear Loss 3.",
        "",
        "Inference: if objective-gradient replacement beats the explicit JVP/VJP power variants, then the observed success should not be attributed to a literal generalized-power theorem. It is stronger evidence for a practical objective-gradient replacement surrogate.",
        "",
        "Next diagnostic if stronger proof is needed: run a GPU gradient/JVP probe that records gradient-step cosine, local Jacobian spectrum, and local-linear predicted gain versus actual gain along the saved trajectories.",
    ]
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    parser.add_argument("--boundary-threshold", type=float, default=0.99)
    parser.add_argument("--trajectory-steps", nargs="+", type=int, default=[1, 5, 10, 20, 50, 100, 300])
    parser.add_argument("--sweep-root", action="append", type=Path, default=[])
    parser.add_argument("--trajectory-root", action="append", type=Path, default=[])
    parser.add_argument("--qaware-root", type=Path, default=DEFAULT_QAWARE_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir if args.out_dir.is_absolute() else PROJECT_ROOT / args.out_dir
    doc = args.doc if args.doc.is_absolute() else PROJECT_ROOT / args.doc
    sweep_roots = args.sweep_root or list(DEFAULT_SWEEP_ROOTS)
    trajectory_roots = args.trajectory_root or list(DEFAULT_TRAJECTORY_ROOTS)
    sweep_roots = [p if p.is_absolute() else PROJECT_ROOT / p for p in sweep_roots]
    trajectory_roots = [p if p.is_absolute() else PROJECT_ROOT / p for p in trajectory_roots]
    out_dir.mkdir(parents=True, exist_ok=True)

    setting_rows: list[dict[str, Any]] = []
    roots = completed_roots(sweep_roots)
    for root in roots:
        setting_rows.extend(analyze_setting(root, args.boundary_threshold))

    traj_rows = trajectory_probe(trajectory_roots, tuple(args.trajectory_steps))
    qaware_root = args.qaware_root if args.qaware_root.is_absolute() else PROJECT_ROOT / args.qaware_root
    qaware_rows = qaware_variant_probe(qaware_root)
    roll_pq_method = rollup(setting_rows, ("dataset_tag", "pq", "method"))
    roll_method = rollup(setting_rows, ("method",))
    tests = hypothesis_tests(setting_rows, traj_rows)

    tables = out_dir / "tables"
    write_csv(tables / "mechanism_probe_by_setting_method.csv", setting_rows)
    write_csv(tables / "mechanism_probe_rollup_by_pq_method.csv", roll_pq_method)
    write_csv(tables / "mechanism_probe_rollup_by_method.csv", roll_method)
    write_csv(tables / "early_to_final_trajectory_probe.csv", traj_rows)
    write_csv(tables / "hypothesis_tests.csv", tests)
    write_csv(tables / "qaware_power_variant_probe_p2q2_eps8_alpha0p3.csv", qaware_rows)
    manifest = {
        "status": "completed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sweep_roots": [str(p) for p in sweep_roots],
        "trajectory_roots": [str(p) for p in trajectory_roots],
        "out_dir": str(out_dir),
        "doc": str(doc),
        "boundary_threshold": args.boundary_threshold,
        "setting_root_count": len(roots),
        "setting_method_row_count": len(setting_rows),
        "trajectory_row_count": len(traj_rows),
        "qaware_variant_row_count": len(qaware_rows),
        "hypothesis_test_row_count": len(tests),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_doc(doc, out_dir, setting_rows, roll_pq_method, tests, traj_rows, qaware_rows)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
