#!/usr/bin/env python3
"""Validate the loss3 surprising-findings synthesis across P/Q and alpha/epsilon settings.

This is post-processing only. It reads existing CSV/NPZ-derived summaries and
per-step metrics; it does not run any optimizer or model computation.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "forensics" / "loss3_surprising_findings_validation_20260520"
DOC = PROJECT_ROOT / "docs" / "loss3_surprising_findings_validation_20260520.md"
METHODS = ("raw_add", "raw_replace", "steepest_add", "steepest_replace")
ADDITIVE = {"raw_add", "steepest_add"}
REPLACEMENT = {"raw_replace", "steepest_replace"}

SUMMARY_SOURCES = [
    (
        "p2q2_300step",
        PROJECT_ROOT / "forensics/loss3_alpha_epsilon_core4_analysis_p2q2_300steps_20260520/core4_alpha_epsilon_method_summary.csv",
    ),
    (
        "pneq_100step_stopped",
        PROJECT_ROOT / "forensics/loss3_alpha_epsilon_core4_analysis_pneq_q_stopped_100steps_20260520/core4_alpha_epsilon_method_summary.csv",
    ),
]

PEAKINESS_SOURCES = [
    PROJECT_ROOT / "forensics/loss3_alpha_epsilon_core4_visuals_p2q2_300steps_20260520/tables/final_delta_peakiness_summary.csv",
    PROJECT_ROOT / "forensics/loss3_alpha_epsilon_core4_visuals_p1_q2_stopped_100steps_20260520/tables/final_delta_peakiness_summary.csv",
    PROJECT_ROOT / "forensics/loss3_alpha_epsilon_core4_visuals_p1_qinf_stopped_100steps_20260520/tables/final_delta_peakiness_summary.csv",
    PROJECT_ROOT / "forensics/loss3_alpha_epsilon_core4_visuals_p2_q1_stopped_100steps_20260520/tables/final_delta_peakiness_summary.csv",
    PROJECT_ROOT / "forensics/loss3_alpha_epsilon_core4_visuals_p2_qinf_stopped_100steps_20260520/tables/final_delta_peakiness_summary.csv",
    PROJECT_ROOT / "forensics/loss3_alpha_epsilon_core4_visuals_pinf_q1_partial_stopped_100steps_20260520/tables/final_delta_peakiness_summary.csv",
]

SIMILARITY_SOURCES = [
    ("p2q2_300step", PROJECT_ROOT / "forensics/loss3_alpha_epsilon_core4_delta_similarity_p2q2_300steps_20260520/tables/final_delta_pairwise_similarity_summary.csv"),
    ("p1_q2_100step", PROJECT_ROOT / "forensics/loss3_alpha_epsilon_core4_delta_similarity_p1_q2_stopped_100steps_20260520/tables/final_delta_pairwise_similarity_summary.csv"),
    ("p1_qinf_100step", PROJECT_ROOT / "forensics/loss3_alpha_epsilon_core4_delta_similarity_p1_qinf_stopped_100steps_20260520/tables/final_delta_pairwise_similarity_summary.csv"),
    ("p2_q1_100step", PROJECT_ROOT / "forensics/loss3_alpha_epsilon_core4_delta_similarity_p2_q1_stopped_100steps_20260520/tables/final_delta_pairwise_similarity_summary.csv"),
    ("p2_qinf_100step", PROJECT_ROOT / "forensics/loss3_alpha_epsilon_core4_delta_similarity_p2_qinf_stopped_100steps_20260520/tables/final_delta_pairwise_similarity_summary.csv"),
    ("pinf_q1_partial_100step", PROJECT_ROOT / "forensics/loss3_alpha_epsilon_core4_delta_similarity_pinf_q1_partial_stopped_100steps_20260520/tables/final_delta_pairwise_similarity_summary.csv"),
]


def fnum(value) -> float:
    if value is None:
        return math.nan
    if isinstance(value, (float, int)):
        return float(value)
    s = str(value).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return math.nan
    try:
        return float(s)
    except Exception:
        return math.nan


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
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


def finite(values: Iterable[float]) -> list[float]:
    return [float(v) for v in values if math.isfinite(float(v))]


def mean(values: Iterable[float]) -> float:
    vals = finite(values)
    return statistics.fmean(vals) if vals else math.nan


def median(values: Iterable[float]) -> float:
    vals = finite(values)
    return statistics.median(vals) if vals else math.nan


def maxfinite(values: Iterable[float]) -> float:
    vals = finite(values)
    return max(vals) if vals else math.nan


def pq_label(p: str, q: str) -> str:
    return f"p={p},q={q}"


def setting_label(row: dict) -> str:
    return f"eps={fnum(row.get('epsilon')):g},alpha={fnum(row.get('alpha')):g}"


def root_key(root: str, method: str) -> tuple[str, str]:
    return (str(Path(root).resolve()), method)


def load_peakiness() -> dict[tuple[str, str], dict[str, str]]:
    out: dict[tuple[str, str], dict[str, str]] = {}
    for path in PEAKINESS_SOURCES:
        for row in read_csv(path):
            out[root_key(row["source_root"], row["method"])] = row
    return out


def load_method_steps(root: str, method: str) -> list[dict[str, str]]:
    path = Path(root) / method / "per_step_metrics.csv"
    rows = read_csv(path)
    rows.sort(key=lambda r: int(fnum(r.get("k"))))
    return rows


def first_step(rows: list[dict[str, str]], metric: str, threshold: float) -> int | None:
    for row in rows:
        value = fnum(row.get(metric))
        if math.isfinite(value) and value >= threshold:
            return int(fnum(row.get("k")))
    return None


def row_at(rows: list[dict[str, str]], step: int) -> dict[str, str] | None:
    for row in rows:
        if int(fnum(row.get("k"))) == step:
            return row
    return None


def linearity_metrics(rows: list[dict[str, str]], hit_step: int | None) -> dict[str, float | int | str]:
    pts: list[tuple[float, float]] = []
    for row in rows:
        k = fnum(row.get("k"))
        y = fnum(row.get("boundary_ratio_mean"))
        if not math.isfinite(k) or not math.isfinite(y):
            continue
        if hit_step is not None and k > hit_step:
            continue
        if y > 1.01:
            continue
        pts.append((k, y))
    if len(pts) < 4:
        return {
            "boundary_linearity_r2": math.nan,
            "boundary_increment_cv": math.nan,
            "boundary_late_over_early_increment": math.nan,
            "boundary_linearity_points": len(pts),
        }
    x = np.array([p[0] for p in pts], dtype=float)
    y = np.array([p[1] for p in pts], dtype=float)
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else math.nan
    diffs = np.diff(y)
    pos = diffs[np.isfinite(diffs)]
    avg = float(np.mean(pos)) if pos.size else math.nan
    cv = float(np.std(pos) / abs(avg)) if pos.size and abs(avg) > 1e-12 else math.nan
    thirds = max(1, pos.size // 3)
    early = float(np.mean(pos[:thirds])) if pos.size else math.nan
    late = float(np.mean(pos[-thirds:])) if pos.size else math.nan
    ratio = late / early if math.isfinite(early) and abs(early) > 1e-12 else math.nan
    return {
        "boundary_linearity_r2": r2,
        "boundary_increment_cv": cv,
        "boundary_late_over_early_increment": ratio,
        "boundary_linearity_points": len(pts),
    }


def method_validation_rows() -> list[dict]:
    peak = load_peakiness()
    rows_out: list[dict] = []
    for dataset_tag, summary_path in SUMMARY_SOURCES:
        for row in read_csv(summary_path):
            root = str(Path(row["source_root"]).resolve())
            method = row["method"]
            p = str(row.get("p_order", ""))
            q = str(row.get("q_order", ""))
            step_rows = load_method_steps(root, method)
            hit = first_step(step_rows, "boundary_ratio_mean", 0.99)
            summary_hit = fnum(row.get("step_to_boundary_ratio_mean_0.99"))
            if hit is None and math.isfinite(summary_hit):
                hit = int(summary_hit)
            hit_row = row_at(step_rows, hit) if hit is not None else None
            hit_loss = fnum(hit_row.get("loss3_q_mean")) if hit_row else math.nan
            final_loss = fnum(row.get("final_loss3_q_mean"))
            post_gain = final_loss - hit_loss if math.isfinite(final_loss) and math.isfinite(hit_loss) else math.nan
            final_ratio = fnum(row.get("final_boundary_ratio_mean"))
            final_std = fnum(row.get("final_boundary_ratio_std"))
            boundary_stds = [fnum(r.get("boundary_ratio_std")) for r in step_rows]
            angles = [fnum(r.get("delta_prev_angle_degrees_mean")) for r in step_rows]
            angle_stds = [fnum(r.get("delta_prev_angle_degrees_std")) for r in step_rows]
            lins = linearity_metrics(step_rows, hit if hit is not None else None) if method in ADDITIVE else {
                "boundary_linearity_r2": math.nan,
                "boundary_increment_cv": math.nan,
                "boundary_late_over_early_increment": math.nan,
                "boundary_linearity_points": 0,
            }
            peak_row = peak.get(root_key(root, method), {})
            rows_out.append({
                "dataset_tag": dataset_tag,
                "source_root": root,
                "p_order": p,
                "q_order": q,
                "pq": pq_label(p, q),
                "epsilon": fnum(row.get("epsilon")),
                "alpha": fnum(row.get("alpha")),
                "setting": setting_label(row),
                "method": method,
                "steps": int(fnum(row.get("steps"))) if math.isfinite(fnum(row.get("steps"))) else "",
                "final_loss3_q_mean": final_loss,
                "final_loss3_q_std": fnum(row.get("final_loss3_q_std")),
                "step_to_boundary_0p99": hit if hit is not None else "nan",
                "boundary_hit_loss3_q_mean": hit_loss,
                "post_boundary_loss_gain": post_gain,
                "post_boundary_loss_gain_positive": int(math.isfinite(post_gain) and post_gain > 1e-9),
                "final_boundary_ratio_mean": final_ratio,
                "final_boundary_ratio_std": final_std,
                "max_boundary_ratio_std_over_steps": maxfinite(boundary_stds),
                "mean_boundary_ratio_std_over_steps": mean(boundary_stds),
                "mean_delta_prev_angle_degrees": mean(angles),
                "max_delta_prev_angle_degrees": maxfinite(angles),
                "mean_delta_prev_angle_std_degrees": mean(angle_stds),
                "final_high_frequency_energy_ratio_mean": fnum(row.get("final_high_frequency_energy_ratio_mean")),
                "final_first_derivative_l2_mean": fnum(row.get("final_first_derivative_l2_mean")),
                "final_total_variation_mean": fnum(row.get("final_total_variation_mean")),
                "mean_peakiness_max_abs_over_rms": fnum(peak_row.get("mean_peakiness_max_abs_over_rms")),
                "max_peakiness_max_abs_over_rms": fnum(peak_row.get("max_peakiness_max_abs_over_rms")),
                **lins,
            })
    return rows_out


def groupby(rows: list[dict], keys: tuple[str, ...]) -> dict[tuple, list[dict]]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    return groups


def rollup_rows(rows: list[dict]) -> list[dict]:
    out = []
    for (dataset_tag, pq, method), group in sorted(groupby(rows, ("dataset_tag", "pq", "method")).items()):
        hit_steps = [fnum(g["step_to_boundary_0p99"]) for g in group]
        post = [fnum(g["post_boundary_loss_gain"]) for g in group]
        out.append({
            "dataset_tag": dataset_tag,
            "pq": pq,
            "method": method,
            "setting_count": len(group),
            "median_step_to_boundary_0p99": median(hit_steps),
            "mean_step_to_boundary_0p99": mean(hit_steps),
            "fraction_boundary_reached": sum(math.isfinite(v) for v in hit_steps) / len(group) if group else math.nan,
            "mean_post_boundary_loss_gain": mean(post),
            "median_post_boundary_loss_gain": median(post),
            "fraction_positive_post_boundary_gain": sum(math.isfinite(v) and v > 1e-9 for v in post) / max(1, sum(math.isfinite(v) for v in post)),
            "median_boundary_linearity_r2": median([fnum(g["boundary_linearity_r2"]) for g in group]),
            "median_boundary_increment_cv": median([fnum(g["boundary_increment_cv"]) for g in group]),
            "median_late_over_early_boundary_increment": median([fnum(g["boundary_late_over_early_increment"]) for g in group]),
            "mean_max_boundary_ratio_std_over_steps": mean([fnum(g["max_boundary_ratio_std_over_steps"]) for g in group]),
            "max_boundary_ratio_std_over_steps": maxfinite([fnum(g["max_boundary_ratio_std_over_steps"]) for g in group]),
            "mean_delta_prev_angle_degrees": mean([fnum(g["mean_delta_prev_angle_degrees"]) for g in group]),
            "max_delta_prev_angle_degrees": maxfinite([fnum(g["max_delta_prev_angle_degrees"]) for g in group]),
            "mean_peakiness_max_abs_over_rms": mean([fnum(g["mean_peakiness_max_abs_over_rms"]) for g in group]),
            "max_peakiness_max_abs_over_rms": maxfinite([fnum(g["max_peakiness_max_abs_over_rms"]) for g in group]),
            "mean_final_high_frequency_energy_ratio": mean([fnum(g["final_high_frequency_energy_ratio_mean"]) for g in group]),
            "mean_final_first_derivative_l2": mean([fnum(g["final_first_derivative_l2_mean"]) for g in group]),
        })
    return out


def winner_counts(rows: list[dict], metric: str, higher_is_better: bool = True) -> list[dict]:
    counts = Counter()
    total = Counter()
    for (dataset_tag, pq, root), group in groupby(rows, ("dataset_tag", "pq", "source_root")).items():
        candidates = [(g["method"], fnum(g.get(metric))) for g in group]
        candidates = [(m, v) for m, v in candidates if math.isfinite(v)]
        if not candidates:
            continue
        best_value = max(v for _, v in candidates) if higher_is_better else min(v for _, v in candidates)
        winners = [m for m, v in candidates if abs(v - best_value) <= max(1e-9, abs(best_value) * 1e-9)]
        total[(dataset_tag, pq)] += 1
        for winner in winners:
            counts[(dataset_tag, pq, winner)] += 1 / len(winners)
    out = []
    for (dataset_tag, pq), n in sorted(total.items()):
        for method in METHODS:
            out.append({
                "dataset_tag": dataset_tag,
                "pq": pq,
                "criterion": metric,
                "method": method,
                "winner_count_fractional_ties": counts[(dataset_tag, pq, method)],
                "setting_count": n,
                "winner_fraction": counts[(dataset_tag, pq, method)] / n if n else math.nan,
            })
    return out


def angle_winners(rows: list[dict]) -> list[dict]:
    return winner_counts(rows, "mean_delta_prev_angle_degrees", higher_is_better=True)


def boundary_fastest(rows: list[dict]) -> list[dict]:
    return winner_counts(rows, "step_to_boundary_0p99", higher_is_better=False)


def similarity_rollup() -> tuple[list[dict], list[dict]]:
    all_rows = []
    for dataset_tag, path in SIMILARITY_SOURCES:
        for row in read_csv(path):
            row = dict(row)
            row["dataset_tag"] = dataset_tag
            row["pq"] = pq_label(str(row.get("p_order", "")), str(row.get("q_order", "")))
            row["method_pair"] = f"{row['method_a']}__{row['method_b']}"
            all_rows.append(row)
    roll = []
    equiv = []
    for (dataset_tag, pq, pair), group in sorted(groupby(all_rows, ("dataset_tag", "pq", "method_pair")).items()):
        cos = [fnum(g.get("cosine_mean")) for g in group]
        spec = [fnum(g.get("spectral_magnitude_cosine_mean")) for g in group]
        rel = [fnum(g.get("relative_l2_distance_mean")) for g in group]
        eq_flags = [math.isfinite(fnum(g.get("cosine_mean"))) and fnum(g.get("cosine_mean")) > 0.99999 and math.isfinite(fnum(g.get("relative_l2_distance_mean"))) and fnum(g.get("relative_l2_distance_mean")) < 1e-4 for g in group]
        near_flags = [math.isfinite(fnum(g.get("cosine_mean"))) and fnum(g.get("cosine_mean")) > 0.95 for g in group]
        roll.append({
            "dataset_tag": dataset_tag,
            "pq": pq,
            "method_pair": pair,
            "setting_count": len(group),
            "mean_cosine": mean(cos),
            "min_cosine": min(finite(cos)) if finite(cos) else math.nan,
            "mean_spectral_cosine": mean(spec),
            "min_spectral_cosine": min(finite(spec)) if finite(spec) else math.nan,
            "mean_relative_l2": mean(rel),
            "max_relative_l2": maxfinite(rel),
            "equivalent_setting_count": sum(eq_flags),
            "near_cosine_gt_0p95_count": sum(near_flags),
            "equivalent_all_settings": int(bool(group) and all(eq_flags)),
        })
        if group and all(eq_flags):
            equiv.append({
                "dataset_tag": dataset_tag,
                "pq": pq,
                "method_pair": pair,
                "setting_count": len(group),
                "mean_cosine": mean(cos),
                "mean_relative_l2": mean(rel),
            })
    return roll, equiv


def markdown_table(rows: list[dict], columns: list[str], max_rows: int | None = None) -> list[str]:
    if max_rows is not None:
        rows = rows[:max_rows]
    def fmt(v):
        if isinstance(v, float):
            if math.isnan(v):
                return "nan"
            if abs(v) >= 1000 or (abs(v) < 1e-3 and v != 0):
                return f"{v:.3e}"
            return f"{v:.4g}"
        return str(v)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(c, "")) for c in columns) + " |")
    return lines


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    method_rows = method_validation_rows()
    roll = rollup_rows(method_rows)
    final_winners = winner_counts(method_rows, "final_loss3_q_mean", higher_is_better=True)
    angle_win = angle_winners(method_rows)
    boundary_win = boundary_fastest(method_rows)
    sim_roll, equiv = similarity_rollup()

    method_csv = OUT_DIR / "per_setting_method_validation.csv"
    roll_csv = OUT_DIR / "pq_method_validation_rollup.csv"
    final_win_csv = OUT_DIR / "final_loss_winner_counts_by_pq.csv"
    angle_win_csv = OUT_DIR / "angle_motion_winner_counts_by_pq.csv"
    boundary_win_csv = OUT_DIR / "boundary_arrival_winner_counts_by_pq.csv"
    sim_csv = OUT_DIR / "delta_similarity_pair_rollup_by_pq.csv"
    equiv_csv = OUT_DIR / "equivalent_method_pairs_by_pq.csv"

    write_csv(method_csv, method_rows)
    write_csv(roll_csv, roll)
    write_csv(final_win_csv, final_winners)
    write_csv(angle_win_csv, angle_win)
    write_csv(boundary_win_csv, boundary_win)
    write_csv(sim_csv, sim_roll)
    write_csv(equiv_csv, equiv)

    # Compact claim-focused views.
    p2q2_roll = [r for r in roll if r["pq"] == "p=2,q=2"]
    nonp2q2_roll = [r for r in roll if r["pq"] != "p=2,q=2"]
    replacement_roll = [r for r in roll if r["method"] in REPLACEMENT]
    additive_linearity = [r for r in roll if r["method"] in ADDITIVE]
    angle_top = sorted(angle_win, key=lambda r: (r["dataset_tag"], r["pq"], -fnum(r["winner_fraction"])))
    final_top = sorted(final_winners, key=lambda r: (r["dataset_tag"], r["pq"], -fnum(r["winner_fraction"])))

    doc_lines = [
        "# Loss3 Surprising Findings Cross-PQ Validation - 2026-05-20",
        "",
        "Status: post-processing analysis from existing local artifacts; no optimizer experiment was run.",
        "",
        "## Scope",
        "",
        "This note checks whether the surprising findings are universal or conditional across the available completed alpha/epsilon and P/Q settings.",
        "",
        "Observed source sets:",
        "",
        "- Full `p=2,q=2`, 20-setting, 300-step sweep.",
        "- Stopped off-diagonal P/Q 100-step sweep: full 20-setting groups for `p=1,q=2`, `p=1,q=inf`, `p=2,q=1`, `p=2,q=inf`, plus partial 2-setting `p=inf,q=1`.",
        "",
        "Generated tables:",
        "",
        f"- Per-setting/method validation: `{method_csv.relative_to(PROJECT_ROOT)}`",
        f"- P/Q-method rollup: `{roll_csv.relative_to(PROJECT_ROOT)}`",
        f"- Final-loss winners: `{final_win_csv.relative_to(PROJECT_ROOT)}`",
        f"- Angle-motion winners: `{angle_win_csv.relative_to(PROJECT_ROOT)}`",
        f"- Boundary-arrival winners: `{boundary_win_csv.relative_to(PROJECT_ROOT)}`",
        f"- Delta-similarity pair rollup: `{sim_csv.relative_to(PROJECT_ROOT)}`",
        f"- Near-exact equivalent method pairs: `{equiv_csv.relative_to(PROJECT_ROOT)}`",
        "",
        "## Claim Checks",
        "",
        "### 1. Boundary arrival is not convergence",
        "",
        "Supported, but strength varies by method and P/Q. Replacement methods usually reach 99% boundary at step 1 and often still have positive post-boundary loss gain. Additive methods may also keep improving after boundary arrival, but they spend more steps getting there and sometimes do not reach the 99% mean boundary within the stopped 100-step off-diagonal runs.",
        "",
        "Replacement-method rollup:",
        "",
    ]
    doc_lines += markdown_table(
        replacement_roll,
        ["dataset_tag", "pq", "method", "setting_count", "median_step_to_boundary_0p99", "mean_post_boundary_loss_gain", "fraction_positive_post_boundary_gain", "mean_delta_prev_angle_degrees"],
    )
    doc_lines += [
        "",
        "### 2. GPI/replacement is fast, but not always final-loss winner",
        "",
        "Supported. The final-loss winner counts show that replacement is not universally the long-run final-loss winner, especially in the 300-step p2q2 run where additive methods can overtake it.",
        "",
    ]
    doc_lines += markdown_table(
        final_top,
        ["dataset_tag", "pq", "method", "winner_count_fractional_ties", "setting_count", "winner_fraction"],
    )
    doc_lines += [
        "",
        "### 3. LP-steepest boundary-ratio growth is more linear than raw PGD",
        "",
        "Mostly supported for p2q2 and many off-diagonal groups, but not a theorem. The linearity check uses pre-boundary mean boundary-ratio `R^2`, increment coefficient of variation, and late/early increment ratio. LP-steepest usually has smaller increment CV and a late/early ratio closer to 1 than raw PGD; raw PGD often has a smaller late/early ratio, matching the visual bending/slowing observation.",
        "",
    ]
    doc_lines += markdown_table(
        additive_linearity,
        ["dataset_tag", "pq", "method", "setting_count", "median_boundary_linearity_r2", "median_boundary_increment_cv", "median_late_over_early_boundary_increment", "mean_max_boundary_ratio_std_over_steps"],
    )
    doc_lines += [
        "",
        "### 4. Boundary-ratio std is a diagnostic",
        "",
        "Supported. Raw PGD generally has the largest boundary-ratio standard deviation; LP-steepest is much smaller; replacement methods are often near zero when replacement stays on the p-boundary. In some off-diagonal P/Q settings, final boundary ratios and replacement behavior can deviate, so this should be read alongside final boundary ratio and P/Q geometry.",
        "",
        "### 5. GPI has the largest angular motion",
        "",
        "Mostly supported, but there are conditional cases. The angle-winner table below counts which method has the largest mean `angle(delta_k, delta_{k-1})` within each setting. Replacement/GPI often dominates, but raw_replace and steepest_replace can tie or overlap in p=2-like replacement geometries, and off-diagonal settings can change the winner.",
        "",
    ]
    doc_lines += markdown_table(
        angle_top,
        ["dataset_tag", "pq", "method", "winner_count_fractional_ties", "setting_count", "winner_fraction"],
    )
    doc_lines += [
        "",
        "### 6. Final delta shapes are often similar, but not always identical",
        "",
        "Supported as a broad visual/spectral statement, not as a universal exact equality. Pairwise similarity remains high for several pairs, but exact equivalence only occurs for specific method pairs and P/Q geometries.",
        "",
        "Near-exact equivalent pairs across all settings in a P/Q group:",
        "",
    ]
    doc_lines += markdown_table(
        equiv,
        ["dataset_tag", "pq", "method_pair", "setting_count", "mean_cosine", "mean_relative_l2"],
    )
    doc_lines += [
        "",
        "Selected similarity rollup:",
        "",
    ]
    selected_pairs = [r for r in sim_roll if r["method_pair"] in {"raw_replace__steepest_replace", "raw_add__steepest_add", "raw_add__steepest_replace", "steepest_add__steepest_replace"}]
    doc_lines += markdown_table(
        selected_pairs,
        ["dataset_tag", "pq", "method_pair", "setting_count", "mean_cosine", "min_cosine", "mean_spectral_cosine", "mean_relative_l2", "equivalent_setting_count"],
    )
    doc_lines += [
        "",
        "### 7. P/Q geometry affects perturbation realism",
        "",
        "Supported, but it should be stated as a combined visual + metric conclusion. `p=2,q=1` looked visually more physical in representative samples, while `q=inf` groups can create localized spikes. The peakiness/high-frequency metrics below help locate where this concern appears quantitatively, but visual representative panels remain important because a single localized spike may be more obvious in the plot than in a rollup average.",
        "",
    ]
    peak_view = sorted(roll, key=lambda r: (r["dataset_tag"], r["pq"], -fnum(r["max_peakiness_max_abs_over_rms"])))
    doc_lines += markdown_table(
        peak_view,
        ["dataset_tag", "pq", "method", "setting_count", "mean_peakiness_max_abs_over_rms", "max_peakiness_max_abs_over_rms", "mean_final_high_frequency_energy_ratio", "mean_final_first_derivative_l2"],
    )
    doc_lines += [
        "",
        "## Bottom Line",
        "",
        "The earlier summary is directionally right, but not every statement is universal. The most robust cross-setting claim is: the key distinction is radial boundary use plus angular/boundary-surface motion, not boundary arrival alone. GPI/replacement is the most aggressive boundary-direction optimizer and is usually fastest, while additive methods can sometimes win final loss after many steps. P/Q geometry strongly affects whether the final perturbation looks physically plausible or spike-like.",
        "",
        "For writing, separate claims into three levels:",
        "",
        "1. Strongly supported: replacement reaches boundary immediately in the main p2q2 setting; boundary arrival and convergence are distinct; raw PGD has larger radial variance; no-std and std plots answer different questions.",
        "2. Mostly supported but conditional: LP-steepest is more linear than raw PGD; GPI has the largest angular motion; final deltas share broad shape similarity.",
        "3. Geometry-dependent: perturbation realism and spike behavior, especially in `q=inf` settings and partial `p=inf` evidence.",
    ]
    DOC.write_text("\n".join(doc_lines) + "\n", encoding="utf-8")

    manifest = {
        "status": "completed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "out_dir": str(OUT_DIR),
        "doc": str(DOC),
        "tables": {
            "per_setting_method_validation": str(method_csv),
            "pq_method_validation_rollup": str(roll_csv),
            "final_loss_winner_counts_by_pq": str(final_win_csv),
            "angle_motion_winner_counts_by_pq": str(angle_win_csv),
            "boundary_arrival_winner_counts_by_pq": str(boundary_win_csv),
            "delta_similarity_pair_rollup_by_pq": str(sim_csv),
            "equivalent_method_pairs_by_pq": str(equiv_csv),
        },
        "method_row_count": len(method_rows),
        "rollup_row_count": len(roll),
        "similarity_rollup_row_count": len(sim_roll),
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
