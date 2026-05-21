#!/usr/bin/env python3
"""Diagnose why loss3 can keep growing after the epsilon boundary is reached.

This is post-processing only. It reads completed loss3 core-four sweep outputs
and summarizes post-boundary loss gain, boundary-direction rotation, tangent
motion proxies, and selected-sample smoothness changes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SWEEP = PROJECT_ROOT / "forensics" / "loss3_alpha_epsilon_core4_sweep_20260519"
DEFAULT_OUT = PROJECT_ROOT / "forensics" / "loss3_post_boundary_mechanism_20260520"
DEFAULT_DOC = PROJECT_ROOT / "docs" / "loss3_post_boundary_mechanism_diagnostics_20260520.md"
CORE4 = ("raw_add", "raw_replace", "steepest_add", "steepest_replace")


def fnum(value: Any) -> float:
    if value in {None, "", "nan", "NaN"}:
        return math.nan
    try:
        return float(value)
    except Exception:
        return math.nan


def finite_mean(values: list[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(mean(vals)) if vals else math.nan


def finite_std(values: list[float]) -> float:
    vals = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)
    return float(np.std(vals, ddof=0)) if vals.size else math.nan


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def manifest_completed(root: Path) -> bool:
    path = root / "manifest.json"
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "completed"
    except Exception:
        return False


def load_completed_roots(sweep_root: Path) -> list[Path]:
    roots: list[Path] = []
    manifest = sweep_root / "sweep_manifest.json"
    if manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        for item in payload.get("completed", []):
            root = Path(item.get("out_root", ""))
            if not root.is_absolute():
                root = PROJECT_ROOT / root
            if manifest_completed(root) and (root / "per_step_metrics.csv").exists():
                roots.append(root)
    for per_step in sweep_root.glob("*/per_step_metrics.csv"):
        root = per_step.parent
        if manifest_completed(root):
            roots.append(root)
    return sorted(dict.fromkeys(roots), key=lambda p: (root_setting(p), p.name))


def config_payload(root: Path) -> dict[str, Any]:
    for name in ("config.json", "manifest.json"):
        path = root / name
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                pass
    return {}


def root_setting(root: Path) -> tuple[float, float]:
    payload = config_payload(root)
    if "epsilon" in payload and "alpha" in payload:
        return float(payload["epsilon"]), float(payload["alpha"])
    rows = read_csv(root / "per_step_metrics.csv")
    return float(rows[0]["epsilon"]), float(rows[0]["alpha"])


def root_pq(root: Path) -> tuple[str, str]:
    payload = config_payload(root)
    p_order = payload.get("p_order", payload.get("p"))
    q_order = payload.get("q_order", payload.get("q"))
    if p_order is not None and q_order is not None:
        return str(p_order), str(q_order)
    rows = read_csv(root / "per_step_metrics.csv")
    return str(rows[0].get("p_order", "unknown")), str(rows[0].get("q_order", "unknown"))


def group_rows_by_method(root: Path) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(root / "per_step_metrics.csv"):
        groups[row["method"]].append(row)
    for method in groups:
        groups[method].sort(key=lambda r: int(r["k"]))
    return groups


def first_hit(rows: list[dict[str, str]], threshold: float) -> int | None:
    for row in rows:
        ratio = fnum(row.get("boundary_ratio_mean"))
        if math.isfinite(ratio) and ratio >= threshold:
            return int(row["k"])
    return None


def row_at(rows: list[dict[str, str]], step: int | None) -> dict[str, str] | None:
    if step is None:
        return None
    for row in rows:
        if int(row["k"]) == int(step):
            return row
    return None


def row_at_or_before(rows: list[dict[str, str]], step: int | None) -> dict[str, str] | None:
    if step is None:
        return None
    candidates = [row for row in rows if int(row["k"]) <= int(step)]
    return candidates[-1] if candidates else None


def mean_metric(rows: list[dict[str, str]], key: str, start_step: int, end_step: int | None = None) -> float:
    vals = []
    for row in rows:
        k = int(row["k"])
        if k < start_step:
            continue
        if end_step is not None and k > end_step:
            continue
        vals.append(fnum(row.get(key)))
    return finite_mean(vals)


def tangent_ratio_from_cos(cosine: float) -> float:
    if not math.isfinite(cosine):
        return math.nan
    c = max(-1.0, min(1.0, cosine))
    return float(math.sqrt(max(0.0, 1.0 - c * c)))


def vector_cos(a: np.ndarray, b: np.ndarray) -> float:
    af = np.asarray(a, dtype=np.float64).reshape(-1)
    bf = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(af) * np.linalg.norm(bf))
    if denom <= 1e-12:
        return math.nan
    return float(np.dot(af, bf) / denom)


def step_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    c = vector_cos(a, b)
    if not math.isfinite(c):
        return math.nan
    return float(np.degrees(np.arccos(max(-1.0, min(1.0, c)))))


def trajectory_summary(method_dir: Path, threshold: float) -> dict[str, Any]:
    path = method_dir / "trajectory_samples.npz"
    if not path.exists():
        return {
            "trajectory_sample_count": 0,
            "traj_final_cosine_from_hit_mean": math.nan,
            "traj_angular_travel_after_hit_deg_mean": math.nan,
            "traj_angular_travel_first10_after_hit_deg_mean": math.nan,
            "traj_step_angle_after_hit_deg_mean": math.nan,
            "traj_unit_l2_travel_after_hit_mean": math.nan,
            "traj_loss_gain_after_hit_mean": math.nan,
            "traj_hf_ratio_change_hit_to_final_mean": math.nan,
            "traj_first_derivative_l2_change_hit_to_final_mean": math.nan,
            "traj_total_variation_change_hit_to_final_mean": math.nan,
        }
    z = np.load(path)
    k_values = np.asarray(z["k"], dtype=np.int64)
    deltas = np.asarray(z["delta"], dtype=np.float64)
    ratios = np.asarray(z.get("boundary_ratio", np.full((len(k_values), deltas.shape[1]), np.nan)), dtype=np.float64)
    losses = np.asarray(z.get("loss3_q", np.full((len(k_values), deltas.shape[1]), np.nan)), dtype=np.float64)
    hf = np.asarray(z.get("high_frequency_energy_ratio", np.full_like(losses, np.nan)), dtype=np.float64)
    d1 = np.asarray(z.get("first_derivative_l2", np.full_like(losses, np.nan)), dtype=np.float64)
    tv = np.asarray(z.get("total_variation", np.full_like(losses, np.nan)), dtype=np.float64)

    final_cos = []
    angular_total = []
    angular_10 = []
    step_angle_means = []
    unit_l2_total = []
    loss_gain = []
    hf_change = []
    d1_change = []
    tv_change = []

    sample_count = int(deltas.shape[1]) if deltas.ndim >= 2 else 0
    for sample_i in range(sample_count):
        hit_positions = np.where(ratios[:, sample_i] >= threshold)[0]
        if hit_positions.size == 0:
            continue
        hit_pos = int(hit_positions[0])
        final_pos = len(k_values) - 1
        hit_delta = deltas[hit_pos, sample_i]
        final_delta = deltas[final_pos, sample_i]
        final_cos.append(vector_cos(hit_delta, final_delta))
        angles = []
        unit_diffs = []
        for pos in range(hit_pos, final_pos):
            a = deltas[pos, sample_i]
            b = deltas[pos + 1, sample_i]
            angle = step_angle_deg(a, b)
            if math.isfinite(angle):
                angles.append(angle)
            an = np.linalg.norm(a.reshape(-1))
            bn = np.linalg.norm(b.reshape(-1))
            if an > 1e-12 and bn > 1e-12:
                unit_diffs.append(float(np.linalg.norm(a.reshape(-1) / an - b.reshape(-1) / bn)))
        angular_total.append(float(np.sum(angles)) if angles else math.nan)
        angular_10.append(float(np.sum(angles[:10])) if angles else math.nan)
        step_angle_means.append(finite_mean(angles))
        unit_l2_total.append(float(np.sum(unit_diffs)) if unit_diffs else math.nan)
        loss_gain.append(float(losses[final_pos, sample_i] - losses[hit_pos, sample_i]))
        hf_change.append(float(hf[final_pos, sample_i] - hf[hit_pos, sample_i]))
        d1_change.append(float(d1[final_pos, sample_i] - d1[hit_pos, sample_i]))
        tv_change.append(float(tv[final_pos, sample_i] - tv[hit_pos, sample_i]))

    return {
        "trajectory_sample_count": sample_count,
        "traj_reached_sample_count": len(final_cos),
        "traj_final_cosine_from_hit_mean": finite_mean(final_cos),
        "traj_final_cosine_from_hit_std": finite_std(final_cos),
        "traj_angular_travel_after_hit_deg_mean": finite_mean(angular_total),
        "traj_angular_travel_first10_after_hit_deg_mean": finite_mean(angular_10),
        "traj_step_angle_after_hit_deg_mean": finite_mean(step_angle_means),
        "traj_unit_l2_travel_after_hit_mean": finite_mean(unit_l2_total),
        "traj_loss_gain_after_hit_mean": finite_mean(loss_gain),
        "traj_hf_ratio_change_hit_to_final_mean": finite_mean(hf_change),
        "traj_first_derivative_l2_change_hit_to_final_mean": finite_mean(d1_change),
        "traj_total_variation_change_hit_to_final_mean": finite_mean(tv_change),
    }


def analyze_root(root: Path, threshold: float) -> list[dict[str, Any]]:
    epsilon, alpha = root_setting(root)
    p_order, q_order = root_pq(root)
    groups = group_rows_by_method(root)
    rows: list[dict[str, Any]] = []
    for method in CORE4:
        if method not in groups:
            continue
        group = groups[method]
        final = group[-1]
        final_step = int(final["k"])
        last_update = row_at_or_before(group, final_step - 1)
        hit_step = first_hit(group, threshold)
        hit_row = row_at(group, hit_step) if hit_step is not None else None
        row10 = row_at(group, hit_step + 10) if hit_step is not None else None
        row25 = row_at(group, hit_step + 25) if hit_step is not None else None
        hit_loss = fnum(hit_row.get("loss3_q_mean")) if hit_row else math.nan
        final_loss = fnum(final.get("loss3_q_mean"))
        gain = final_loss - hit_loss if math.isfinite(final_loss) and math.isfinite(hit_loss) else math.nan
        denom_steps = final_step - hit_step if hit_step is not None else 0
        cos_delta_direction_after = mean_metric(group, "cos_delta_direction_mean", hit_step if hit_step is not None else final_step, final_step - 1)
        out = {
            "source_root": str(root),
            "epsilon": epsilon,
            "alpha": alpha,
            "p_order": p_order,
            "q_order": q_order,
            "method": method,
            "boundary_threshold": threshold,
            "steps": final_step,
            "hit_step": hit_step if hit_step is not None else "nan",
            "post_boundary_step_count": denom_steps if hit_step is not None else "nan",
            "loss3_q_at_hit_mean": hit_loss,
            "final_loss3_q_mean": final_loss,
            "post_boundary_loss_gain_mean": gain,
            "post_boundary_loss_gain_per_step": gain / denom_steps if math.isfinite(gain) and denom_steps > 0 else math.nan,
            "post_boundary_loss_gain_fraction_of_final": gain / final_loss if math.isfinite(gain) and math.isfinite(final_loss) and abs(final_loss) > 1e-12 else math.nan,
            "loss_gain_10_steps_after_hit": fnum(row10.get("loss3_q_mean")) - hit_loss if row10 and math.isfinite(hit_loss) else math.nan,
            "loss_gain_25_steps_after_hit": fnum(row25.get("loss3_q_mean")) - hit_loss if row25 and math.isfinite(hit_loss) else math.nan,
            "boundary_ratio_at_hit_mean": fnum(hit_row.get("boundary_ratio_mean")) if hit_row else math.nan,
            "final_boundary_ratio_mean": fnum(final.get("boundary_ratio_mean")),
            "cos_delta_grad_at_hit_mean": fnum(hit_row.get("cos_delta_grad_mean")) if hit_row else math.nan,
            "cos_delta_direction_at_hit_mean": fnum(hit_row.get("cos_delta_direction_mean")) if hit_row else math.nan,
            "cos_grad_direction_at_hit_mean": fnum(hit_row.get("cos_grad_direction_mean")) if hit_row else math.nan,
            "cos_direction_prev_at_hit_mean": fnum(hit_row.get("cos_direction_prev_mean")) if hit_row else math.nan,
            "cos_delta_grad_last_update_mean": fnum(last_update.get("cos_delta_grad_mean")) if last_update else math.nan,
            "cos_delta_direction_last_update_mean": fnum(last_update.get("cos_delta_direction_mean")) if last_update else math.nan,
            "cos_grad_direction_last_update_mean": fnum(last_update.get("cos_grad_direction_mean")) if last_update else math.nan,
            "cos_direction_prev_last_update_mean": fnum(last_update.get("cos_direction_prev_mean")) if last_update else math.nan,
            "cos_delta_direction_after_hit_mean": cos_delta_direction_after,
            "tangent_direction_ratio_after_hit_p2_proxy": tangent_ratio_from_cos(cos_delta_direction_after),
            "cos_direction_prev_after_hit_mean": mean_metric(group, "cos_direction_prev_mean", hit_step if hit_step is not None else final_step, final_step - 1),
            "delta_prev_angle_degrees_after_hit_mean": mean_metric(group, "delta_prev_angle_degrees_mean", hit_step if hit_step is not None else final_step, final_step),
            "delta_step_l2_over_epsilon_after_hit_mean": mean_metric(group, "delta_step_l2_over_epsilon_mean", hit_step if hit_step is not None else final_step, final_step),
            "delta_step_pnorm_over_epsilon_after_hit_mean": mean_metric(group, "delta_step_pnorm_over_epsilon_mean", hit_step if hit_step is not None else final_step, final_step),
            "delta_unit_direction_l2_step_after_hit_mean": mean_metric(group, "delta_unit_direction_l2_step_mean", hit_step if hit_step is not None else final_step, final_step),
            "projection_shrink_after_hit_mean": mean_metric(group, "projection_shrink_factor_mean", hit_step if hit_step is not None else final_step, final_step - 1),
            "hf_ratio_at_hit_mean": fnum(hit_row.get("high_frequency_energy_ratio_mean")) if hit_row else math.nan,
            "final_hf_ratio_mean": fnum(final.get("high_frequency_energy_ratio_mean")),
            "first_derivative_l2_at_hit_mean": fnum(hit_row.get("first_derivative_l2_mean")) if hit_row else math.nan,
            "final_first_derivative_l2_mean": fnum(final.get("first_derivative_l2_mean")),
            "total_variation_at_hit_mean": fnum(hit_row.get("total_variation_mean")) if hit_row else math.nan,
            "final_total_variation_mean": fnum(final.get("total_variation_mean")),
        }
        out.update(trajectory_summary(root / method, threshold))
        rows.append(out)
    return rows


def rollup(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)
    keys = [
        "hit_step",
        "post_boundary_loss_gain_mean",
        "post_boundary_loss_gain_fraction_of_final",
        "loss_gain_10_steps_after_hit",
        "loss_gain_25_steps_after_hit",
        "tangent_direction_ratio_after_hit_p2_proxy",
        "cos_direction_prev_after_hit_mean",
        "delta_prev_angle_degrees_after_hit_mean",
        "delta_step_l2_over_epsilon_after_hit_mean",
        "delta_step_pnorm_over_epsilon_after_hit_mean",
        "delta_unit_direction_l2_step_after_hit_mean",
        "projection_shrink_after_hit_mean",
        "traj_final_cosine_from_hit_mean",
        "traj_angular_travel_after_hit_deg_mean",
        "traj_angular_travel_first10_after_hit_deg_mean",
        "traj_loss_gain_after_hit_mean",
        "traj_hf_ratio_change_hit_to_final_mean",
        "traj_first_derivative_l2_change_hit_to_final_mean",
    ]
    out = []
    for method in CORE4:
        group = by_method.get(method, [])
        if not group:
            continue
        row: dict[str, Any] = {"method": method, "setting_count": len(group)}
        for key in keys:
            row[f"mean_{key}"] = finite_mean([fnum(r.get(key)) for r in group])
        out.append(row)
    return out


def md_table(rows: list[dict[str, Any]], fields: list[str], max_rows: int | None = None) -> str:
    show = rows[:max_rows] if max_rows is not None else rows
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in show:
        vals = []
        for field in fields:
            value = row.get(field, "")
            fv = fnum(value)
            if isinstance(value, float) or math.isfinite(fv):
                vals.append("nan" if not math.isfinite(fv) else f"{fv:.4g}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_doc(path: Path, *, sweep_root: Path, out_dir: Path, rows: list[dict[str, Any]], rollup_rows: list[dict[str, Any]], threshold: float) -> None:
    baseline_rows = [r for r in rows if abs(float(r["epsilon"]) - 4.0) < 1e-9 and abs(float(r["alpha"]) - 0.4) < 1e-9]
    lines = [
        "# Loss3 Post-Boundary Mechanism Diagnostics - 2026-05-20",
        "",
        "Status: generated from existing completed artifacts; no optimizer experiment was rerun.",
        "",
        "## Source Data",
        "",
        f"Observed from sweep root: `{rel(sweep_root)}`",
        f"Output root: `{rel(out_dir)}`",
        f"Boundary threshold used here: `{threshold}`.",
        "",
        "Tables:",
        "",
        f"- Per-setting diagnostics: `{rel(out_dir / 'tables' / 'post_boundary_mechanism_by_setting.csv')}`",
        f"- Method rollup: `{rel(out_dir / 'tables' / 'post_boundary_mechanism_rollup.csv')}`",
        "",
        "## What The Diagnostics Mean",
        "",
        "- `post_boundary_loss_gain_mean`: final mean loss minus mean loss at the first boundary-hit step. This is the direct measurement of how much useful optimization happens after the method has already reached the epsilon boundary.",
        "- `loss_gain_10_steps_after_hit` and `loss_gain_25_steps_after_hit`: short-window post-boundary gain. These test whether the method improves quickly after hitting the boundary rather than only slowly over many steps.",
        "- `cos_delta_direction_after_hit_mean`: cosine between the current perturbation direction and the proposed update direction after boundary hit. Near `1` means the update is mostly radial; smaller values mean more boundary-direction rotation.",
        "- `tangent_direction_ratio_after_hit_p2_proxy`: for `p=2`, approximately `sqrt(1 - cos_delta_direction^2)`. Larger means more tangential/boundary-surface motion is available after the boundary is reached.",
        "- `cos_direction_prev_after_hit_mean`: cosine between successive update directions. Smaller means the optimizer is still changing direction substantially; near `1` means directions have stabilized.",
        "- `delta_prev_angle_degrees_after_hit_mean`: direct mean of `angle(delta_k, delta_{k-1})` after boundary hit, when available in newly generated runs.",
        "- `delta_step_l2_over_epsilon_after_hit_mean`: actual perturbation-space step length after boundary hit, normalized by epsilon.",
        "- `projection_shrink_after_hit_mean`: how much the proposed step is shrunk by projection. For additive PGD on the boundary, a lot of radial motion can be projected away, which can waste step length.",
        "- `traj_final_cosine_from_hit_mean`: cosine between the selected-sample delta at boundary hit and final delta. Low values mean the boundary point still rotates a lot before the final solution.",
        "- `traj_angular_travel_after_hit_deg_mean`: total selected-sample angular travel on/near the boundary after the hit step. This directly measures whether the optimizer keeps moving around the boundary.",
        "- `traj_hf_ratio_change_hit_to_final_mean` and derivative/TV changes: whether the trajectory becomes smoother or rougher after boundary hit.",
        "",
        "## Method Rollup",
        "",
        md_table(
            rollup_rows,
            [
                "method",
                "setting_count",
                "mean_hit_step",
                "mean_post_boundary_loss_gain_mean",
                "mean_loss_gain_10_steps_after_hit",
                "mean_tangent_direction_ratio_after_hit_p2_proxy",
                "mean_delta_prev_angle_degrees_after_hit_mean",
                "mean_delta_step_l2_over_epsilon_after_hit_mean",
                "mean_traj_final_cosine_from_hit_mean",
                "mean_traj_angular_travel_after_hit_deg_mean",
            ],
        ),
        "",
        "## Baseline eps=4, alpha=0.4 Preview",
        "",
        md_table(
            baseline_rows,
            [
                "method",
                "hit_step",
                "loss3_q_at_hit_mean",
                "final_loss3_q_mean",
                "post_boundary_loss_gain_mean",
                "loss_gain_10_steps_after_hit",
                "tangent_direction_ratio_after_hit_p2_proxy",
                "delta_prev_angle_degrees_after_hit_mean",
                "delta_step_l2_over_epsilon_after_hit_mean",
                "traj_final_cosine_from_hit_mean",
                "traj_angular_travel_after_hit_deg_mean",
            ],
        ) if baseline_rows else "No completed baseline rows found in this sweep root.",
        "",
        "## Working Interpretation",
        "",
        "Observed evidence from these diagnostics should be read together with the loss curves and GIF trajectories.",
        "",
        "Inference: if GPI/replacement reaches the boundary at step 1 but still has a large `post_boundary_loss_gain_mean`, then the first boundary point is not the final good direction. The method is winning because it rapidly rotates/refines the direction on the boundary, not merely because it reaches the boundary.",
        "",
        "Inference: if additive PGD methods show later boundary hit, high projection shrink, smaller tangential motion, or low post-boundary angular travel, then their post-boundary updates may be wasting effort in radial components that projection removes, or moving too slowly around the boundary surface.",
        "",
        "Inference: if GPI's high-frequency/smoothness metrics decrease after the hit step, then the suspected behavior is confirmed: it may hit the boundary with a rougher intermediate perturbation and then smooth/organize the direction while staying near the boundary. If those metrics do not decrease, then the post-boundary gain is more likely directional alignment with the loss landscape rather than smoothing.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-root", type=Path, default=DEFAULT_SWEEP)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    parser.add_argument("--p-filter", default="2")
    parser.add_argument("--q-filter", default="2")
    parser.add_argument("--boundary-threshold", type=float, default=0.99)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sweep_root = args.sweep_root if args.sweep_root.is_absolute() else PROJECT_ROOT / args.sweep_root
    out_dir = args.out_dir if args.out_dir.is_absolute() else PROJECT_ROOT / args.out_dir
    doc_path = args.doc if args.doc.is_absolute() else PROJECT_ROOT / args.doc
    out_dir.mkdir(parents=True, exist_ok=True)

    roots = load_completed_roots(sweep_root)
    roots = [root for root in roots if root_pq(root) == (str(args.p_filter), str(args.q_filter))]
    if not roots:
        raise SystemExit("No completed roots matched the requested P/Q filters.")

    all_rows: list[dict[str, Any]] = []
    for root in roots:
        all_rows.extend(analyze_root(root, args.boundary_threshold))
    rollup_rows = rollup(all_rows)

    tables_dir = out_dir / "tables"
    by_setting_csv = tables_dir / "post_boundary_mechanism_by_setting.csv"
    rollup_csv = tables_dir / "post_boundary_mechanism_rollup.csv"
    write_csv(by_setting_csv, all_rows)
    write_csv(rollup_csv, rollup_rows)
    manifest = {
        "status": "completed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sweep_root": str(sweep_root),
        "out_dir": str(out_dir),
        "doc": str(doc_path),
        "p_filter": str(args.p_filter),
        "q_filter": str(args.q_filter),
        "boundary_threshold": args.boundary_threshold,
        "completed_setting_count": len(roots),
        "row_count": len(all_rows),
        "tables": {
            "by_setting": str(by_setting_csv),
            "rollup": str(rollup_csv),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_doc(doc_path, sweep_root=sweep_root, out_dir=out_dir, rows=all_rows, rollup_rows=rollup_rows, threshold=args.boundary_threshold)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
