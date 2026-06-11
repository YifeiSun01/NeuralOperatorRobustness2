#!/usr/bin/env python3
"""Analyze biased local attack directions from full Burgers error Jacobian SVD.

For the local affine error model e(x + delta) ~= b + A delta, the pure SVD
movement direction solves max ||A delta||^2.  The finite-radius attack loss solves
max ||b + A delta||^2, so the clean residual b contributes a linear/cross term.
This script quantifies the angle/gain difference on an existing full-SVD run.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.run_burgers_full1024_svd_attack_correlation_20260611 import (  # noqa: E402
    MODEL_SPECS,
    burgers_solver_target,
    load_model,
    load_x,
)

DEFAULT_SVD_ROOT = PROJECT_ROOT / "forensics/burgers_wideparam_loss3targeted_full1024_svd_attack3_20260611"
DEFAULT_OUT_DIR = PROJECT_ROOT / "forensics/burgers_wideparam_loss3targeted_biased_local_direction_20260611"
DEFAULT_REPORT = PROJECT_ROOT / "docs/burgers_wideparam_loss3targeted_biased_local_direction_20260611.md"
EPS = 1e-12


def relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def save_json(path: Path, payload: Any) -> None:
    def conv(x: Any) -> Any:
        if isinstance(x, Path):
            return str(x)
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, dict):
            return {str(k): conv(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [conv(v) for v in x]
        return x
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(conv(payload), indent=2), encoding="utf-8")


def unit(v: np.ndarray) -> np.ndarray:
    vv = np.asarray(v, dtype=np.float64).reshape(-1)
    n = float(np.linalg.norm(vv))
    if n <= EPS:
        return np.full_like(vv, np.nan)
    return vv / n


def signed_cos(a: np.ndarray, b: np.ndarray) -> float:
    aa = unit(a)
    bb = unit(b)
    if not np.all(np.isfinite(aa)) or not np.all(np.isfinite(bb)):
        return math.nan
    return float(np.clip(np.dot(aa, bb), -1.0, 1.0))


def abs_cos(a: np.ndarray, b: np.ndarray) -> float:
    c = signed_cos(a, b)
    return float(abs(c)) if math.isfinite(c) else math.nan


def angle_deg_from_cos(c: float) -> float:
    if not math.isfinite(c):
        return math.nan
    return float(math.degrees(math.acos(float(np.clip(c, -1.0, 1.0)))))


def orient_for_positive_linear(v: np.ndarray, c: np.ndarray) -> np.ndarray:
    vv = unit(v)
    if not np.all(np.isfinite(vv)):
        return vv
    return vv if float(np.dot(c, vv)) >= 0.0 else -vv


def local_gain_mse(A: np.ndarray, b: np.ndarray, v: np.ndarray, r_l2: float) -> dict[str, float]:
    vv = unit(v)
    if not np.all(np.isfinite(vv)):
        return {"linear_gain_mse": math.nan, "quadratic_gain_mse": math.nan, "total_gain_mse": math.nan, "endpoint_mse": math.nan}
    Av = A @ vv
    n = float(b.size)
    linear = 2.0 * float(r_l2) * float(np.dot(b, Av)) / n
    quad = float(r_l2) * float(r_l2) * float(np.dot(Av, Av)) / n
    clean = float(np.dot(b, b)) / n
    return {
        "linear_gain_mse": linear,
        "quadratic_gain_mse": quad,
        "total_gain_mse": linear + quad,
        "endpoint_mse": clean + linear + quad,
    }


def affine_direction_from_svd(s: np.ndarray, U: np.ndarray, Vh: np.ndarray, b: np.ndarray, r_l2: float) -> np.ndarray:
    """Solve max ||b + A delta||^2 over ||delta|| <= r using A=U diag(s) Vh.

    In V coordinates, delta_i = c_i / (mu - s_i^2), where
    c_i = s_i * (U^T b)_i and mu > max(s_i^2) is chosen by ||delta||=r.
    """
    s = np.asarray(s, dtype=np.float64).reshape(-1)
    U = np.asarray(U, dtype=np.float64)
    Vh = np.asarray(Vh, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    q = s * s
    beta = U.T @ b
    c = s * beta
    c_norm = float(np.linalg.norm(c))
    if c_norm <= EPS:
        coeff = np.zeros_like(c)
        coeff[0] = float(r_l2)
        return Vh.T @ coeff

    qmax = float(np.max(q))
    scale = max(1.0, abs(qmax))
    low = qmax + 1e-12 * scale

    def norm_at(mu: float) -> float:
        denom = np.maximum(mu - q, EPS)
        return float(np.sqrt(np.sum((c / denom) ** 2)))

    low_norm = norm_at(low)
    if not math.isfinite(low_norm) or low_norm >= r_l2:
        high = qmax + scale
        while norm_at(high) > r_l2:
            high = qmax + 2.0 * (high - qmax)
        for _ in range(120):
            mid = 0.5 * (low + high)
            if norm_at(mid) > r_l2:
                low = mid
            else:
                high = mid
        mu = high
        coeff = c / (mu - q)
    else:
        # Nongeneric case: c has too little component in the top eigenspace.
        # Put the remaining norm into the top singular direction.
        coeff = c / np.maximum(low - q, EPS)
        remaining = math.sqrt(max(float(r_l2) ** 2 - float(np.dot(coeff, coeff)), 0.0))
        top = int(np.argmax(q))
        coeff[top] += remaining
    return Vh.T @ coeff


def rank_values(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def pearson(x: list[float], y: list[float]) -> float:
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    mask = np.isfinite(xx) & np.isfinite(yy)
    xx = xx[mask]
    yy = yy[mask]
    if len(xx) < 2 or float(np.std(xx)) <= 0.0 or float(np.std(yy)) <= 0.0:
        return math.nan
    return float(np.corrcoef(xx, yy)[0, 1])


def spearman(x: list[float], y: list[float]) -> float:
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    mask = np.isfinite(xx) & np.isfinite(yy)
    xx = xx[mask]
    yy = yy[mask]
    if len(xx) < 2:
        return math.nan
    return pearson(rank_values(xx).tolist(), rank_values(yy).tolist())


def correlation_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    x_keys = [
        "error_spectral_norm",
        "bias_gradient_norm",
        "svd_local_gain_eps_mse",
        "outward_local_gain_eps_mse",
        "affine_local_gain_eps_mse",
        "attack_delta_svd_oriented_cos",
        "attack_delta_outward_cos",
        "attack_delta_affine_eps_cos",
    ]
    y_keys = ["attack_loss_growth_abs", "attack_final_mse", "attack_log_growth"]
    groups: list[tuple[str, list[dict[str, Any]]]] = [("all_model_sample_pairs", rows)]
    for model in sorted({str(r["model_key"]) for r in rows}):
        groups.append((f"model_{model}_across_samples", [r for r in rows if str(r["model_key"]) == model]))
    out: list[dict[str, Any]] = []
    for group, grows in groups:
        for x_key in x_keys:
            for y_key in y_keys:
                n = sum(
                    math.isfinite(float(r.get(x_key, math.nan))) and math.isfinite(float(r.get(y_key, math.nan)))
                    for r in grows
                )
                out.append(
                    {
                        "analysis_set": group,
                        "x": x_key,
                        "y": y_key,
                        "n": n,
                        "pearson": pearson([float(r.get(x_key, math.nan)) for r in grows], [float(r.get(y_key, math.nan)) for r in grows]),
                        "spearman": spearman([float(r.get(x_key, math.nan)) for r in grows], [float(r.get(y_key, math.nan)) for r in grows]),
                    }
                )
    return out


def load_clean_residuals(rows: list[dict[str, str]], device: torch.device) -> dict[tuple[int, str], np.ndarray]:
    models: dict[str, torch.nn.Module] = {}
    specs = {str(s["key"]): s for s in MODEL_SPECS}
    for model_key, spec in specs.items():
        print(f"[model] load {model_key} on {device}", flush=True)
        models[model_key] = load_model(Path(spec["checkpoint"]), device)
    residuals: dict[tuple[int, str], np.ndarray] = {}
    for row in rows:
        sample_id = int(row["sample_id"])
        model_key = str(row["model_key"])
        x_np = load_x(Path(row["dataset_path"]), int(row["local_index"]))
        x = torch.from_numpy(x_np).to(device=device, dtype=torch.float32).view(1, 1024, 1)
        with torch.no_grad():
            solver = burgers_solver_target(x)
            pred = models[model_key](x)
            b = (pred - solver).detach().cpu().numpy().reshape(-1).astype(np.float64)
        residuals[(sample_id, model_key)] = b
    return residuals


def analyze(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    joined_path = args.svd_root / "svd_attack_joined_metrics.csv"
    rows_in = read_csv(joined_path)
    if args.max_rows > 0:
        rows_in = rows_in[: args.max_rows]
    device = torch.device(args.device)
    residuals = load_clean_residuals(rows_in, device)
    attack_traces = {
        key: np.load(args.svd_root / "attack_traces" / f"{key}_attack_trace.npz", allow_pickle=False)
        for key in ["baseline", "loss1", "loss2", "loss3"]
    }
    eps_rms_values = [float(x) for x in args.eps_rms_values]
    rows: list[dict[str, Any]] = []
    for row in rows_in:
        sample_id = int(row["sample_id"])
        model_key = str(row["model_key"])
        b = residuals[(sample_id, model_key)]
        z = np.load(PROJECT_ROOT / row["error_svd_npz"], allow_pickle=False)
        A = z["jacobian"].astype(np.float64)
        s = z["singular_values"].astype(np.float64)
        U = z["left_singular_vectors"].astype(np.float64)
        Vh = z["right_singular_vectors"].astype(np.float64)
        c = A.T @ b
        v_svd = orient_for_positive_linear(Vh[0], c)
        v_out = unit(c)
        delta_attack = attack_traces[model_key]["final_delta"][sample_id].astype(np.float64).reshape(-1)
        v_attack = unit(delta_attack)
        eps_rms = float(args.attack_epsilon_rms)
        r_l2 = eps_rms * math.sqrt(float(A.shape[1]))
        d_affine = affine_direction_from_svd(s, U, Vh, b, r_l2)
        v_affine = unit(d_affine)
        svd_gain = local_gain_mse(A, b, v_svd, r_l2)
        outward_gain = local_gain_mse(A, b, v_out, r_l2)
        affine_gain = local_gain_mse(A, b, v_affine, r_l2)
        out: dict[str, Any] = dict(row)
        out.update(
            {
                "clean_residual_mse_recomputed": float(np.dot(b, b) / b.size),
                "clean_residual_norm_l2": float(np.linalg.norm(b)),
                "bias_gradient_norm": float(np.linalg.norm(c)),
                "attack_epsilon_rms": eps_rms,
                "attack_radius_l2": r_l2,
                "svd_outward_signed_cos": signed_cos(v_svd, v_out),
                "svd_outward_abs_cos": abs_cos(v_svd, v_out),
                "svd_outward_oriented_angle_deg": angle_deg_from_cos(signed_cos(v_svd, v_out)),
                "svd_outward_abs_angle_deg": angle_deg_from_cos(abs_cos(v_svd, v_out)),
                "svd_affine_eps_signed_cos": signed_cos(v_svd, v_affine),
                "svd_affine_eps_abs_cos": abs_cos(v_svd, v_affine),
                "svd_affine_eps_oriented_angle_deg": angle_deg_from_cos(signed_cos(v_svd, v_affine)),
                "outward_affine_eps_signed_cos": signed_cos(v_out, v_affine),
                "outward_affine_eps_abs_cos": abs_cos(v_out, v_affine),
                "attack_delta_svd_oriented_cos": signed_cos(v_attack, v_svd),
                "attack_delta_svd_abs_cos": abs_cos(v_attack, v_svd),
                "attack_delta_outward_cos": signed_cos(v_attack, v_out),
                "attack_delta_outward_abs_cos": abs_cos(v_attack, v_out),
                "attack_delta_affine_eps_cos": signed_cos(v_attack, v_affine),
                "attack_delta_affine_eps_abs_cos": abs_cos(v_attack, v_affine),
                "svd_local_gain_eps_mse": svd_gain["total_gain_mse"],
                "svd_local_gain_eps_linear_mse": svd_gain["linear_gain_mse"],
                "svd_local_gain_eps_quadratic_mse": svd_gain["quadratic_gain_mse"],
                "outward_local_gain_eps_mse": outward_gain["total_gain_mse"],
                "outward_local_gain_eps_linear_mse": outward_gain["linear_gain_mse"],
                "outward_local_gain_eps_quadratic_mse": outward_gain["quadratic_gain_mse"],
                "affine_local_gain_eps_mse": affine_gain["total_gain_mse"],
                "affine_local_gain_eps_linear_mse": affine_gain["linear_gain_mse"],
                "affine_local_gain_eps_quadratic_mse": affine_gain["quadratic_gain_mse"],
                "affine_over_svd_gain_ratio": float(affine_gain["total_gain_mse"] / max(abs(svd_gain["total_gain_mse"]), EPS)),
                "affine_over_outward_gain_ratio": float(affine_gain["total_gain_mse"] / max(abs(outward_gain["total_gain_mse"]), EPS)),
            }
        )
        for eps_rms_i in eps_rms_values:
            r_i = eps_rms_i * math.sqrt(float(A.shape[1]))
            v_aff_i = unit(affine_direction_from_svd(s, U, Vh, b, r_i))
            gain_i = local_gain_mse(A, b, v_aff_i, r_i)
            key = f"eps{eps_rms_i:g}".replace(".", "p").replace("-", "m")
            out[f"{key}_affine_vs_outward_abs_cos"] = abs_cos(v_aff_i, v_out)
            out[f"{key}_affine_vs_svd_abs_cos"] = abs_cos(v_aff_i, v_svd)
            out[f"{key}_affine_gain_mse"] = gain_i["total_gain_mse"]
        rows.append(out)
    corr = correlation_table(rows)
    summary = summarize(rows, corr, eps_rms_values)
    return rows, corr, summary


def summarize(rows: list[dict[str, Any]], corr: list[dict[str, Any]], eps_values: list[float]) -> dict[str, Any]:
    def mean(key: str) -> float:
        vals = np.asarray([float(r.get(key, math.nan)) for r in rows], dtype=float)
        vals = vals[np.isfinite(vals)]
        return float(np.mean(vals)) if vals.size else math.nan
    def by_model(key: str) -> dict[str, float]:
        out = {}
        for model in sorted({str(r["model_key"]) for r in rows}):
            vals = np.asarray([float(r.get(key, math.nan)) for r in rows if str(r["model_key"]) == model], dtype=float)
            vals = vals[np.isfinite(vals)]
            out[model] = float(np.mean(vals)) if vals.size else math.nan
        return out
    all_corr = [r for r in corr if r["analysis_set"] == "all_model_sample_pairs"]
    corr_lookup = {f"{r['x']}__{r['y']}": r for r in all_corr}
    return {
        "n_pairs": len(rows),
        "mean_svd_outward_abs_cos": mean("svd_outward_abs_cos"),
        "mean_svd_outward_abs_angle_deg": mean("svd_outward_abs_angle_deg"),
        "mean_svd_affine_eps_abs_cos": mean("svd_affine_eps_abs_cos"),
        "mean_attack_delta_svd_abs_cos": mean("attack_delta_svd_abs_cos"),
        "mean_attack_delta_outward_abs_cos": mean("attack_delta_outward_abs_cos"),
        "mean_attack_delta_affine_eps_abs_cos": mean("attack_delta_affine_eps_abs_cos"),
        "by_model_svd_outward_abs_angle_deg": by_model("svd_outward_abs_angle_deg"),
        "by_model_attack_delta_svd_abs_cos": by_model("attack_delta_svd_abs_cos"),
        "by_model_attack_delta_outward_abs_cos": by_model("attack_delta_outward_abs_cos"),
        "by_model_attack_delta_affine_eps_abs_cos": by_model("attack_delta_affine_eps_abs_cos"),
        "correlation_all_pairs": corr_lookup,
        "eps_rms_values": eps_values,
    }


def markdown_table(rows: list[dict[str, Any]], cols: list[str], limit: int | None = None) -> list[str]:
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in rows[:limit]:
        vals = []
        for col in cols:
            val = row.get(col, "")
            if isinstance(val, float):
                vals.append("" if not math.isfinite(val) else f"{val:.6g}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return lines


def write_report(path: Path, args: argparse.Namespace, rows: list[dict[str, Any]], corr: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    all_corr = [r for r in corr if r["analysis_set"] == "all_model_sample_pairs"]
    compact_rows = []
    for r in rows:
        compact_rows.append(
            {
                "sample": int(r["sample_id"]),
                "model": r["model_key"],
                "sigma1": float(r["error_spectral_norm"]),
                "angle_svd_vs_Atb_deg": float(r["svd_outward_abs_angle_deg"]),
                "cos_attack_svd": float(r["attack_delta_svd_abs_cos"]),
                "cos_attack_Atb": float(r["attack_delta_outward_abs_cos"]),
                "cos_attack_affine": float(r["attack_delta_affine_eps_abs_cos"]),
                "actual_growth": float(r["attack_loss_growth_abs"]),
                "svd_gain": float(r["svd_local_gain_eps_mse"]),
                "affine_gain": float(r["affine_local_gain_eps_mse"]),
            }
        )
    lines = [
        "# Burgers Biased Local Attack Direction Analysis - 2026-06-11",
        "",
        "This analyzes the completed full `1024 x 1024` Burgers error-Jacobian SVD run and asks whether the local attack direction should be the pure top singular vector or the biased finite-radius direction caused by the clean residual `b`.",
        "",
        "## Local Math",
        "",
        "Let `b = f_model(x) - f_solver(x)` and `A = J_model(x) - J_solver(x)`.  The pure SVD movement metric solves",
        "",
        "```text",
        "max_{||delta|| <= r} ||A delta||^2",
        "```",
        "",
        "so its direction is the top right singular vector of `A`.  But the local affine attack-loss model is",
        "",
        "```text",
        "max_{||delta|| <= r} ||b + A delta||^2 - ||b||^2",
        "= max delta^T A^T A delta + 2 (A^T b)^T delta.",
        "```",
        "",
        "Therefore as `r -> 0`, the direction tends to `A^T b / ||A^T b||`, not the top singular vector.  At finite radius, the trust-region KKT form is",
        "",
        "```text",
        "delta(mu) = (mu I - A^T A)^(-1) A^T b,  mu > lambda_max(A^T A),  ||delta(mu)|| = r.",
        "```",
        "",
        "This script computes all three directions: `top_svd`, `outward=A^T b`, and `affine_trust_region` at the attack radius.",
        "",
        "## Files",
        "",
        f"- source SVD root: `{relpath(args.svd_root)}`",
        f"- output directory: `{relpath(args.out_dir)}`",
        f"- per-pair metrics: `{relpath(args.out_dir / 'biased_direction_metrics.csv')}`",
        f"- correlations: `{relpath(args.out_dir / 'biased_direction_correlations.csv')}`",
        f"- summary JSON: `{relpath(args.out_dir / 'biased_direction_summary.json')}`",
        "",
        "## Headline Numbers",
        "",
        f"- pairs analyzed: `{summary['n_pairs']}`",
        f"- mean abs angle between top SVD direction and `A^T b`: `{summary['mean_svd_outward_abs_angle_deg']:.3f}` degrees",
        f"- mean abs cosine, final nonlinear attack delta vs top SVD: `{summary['mean_attack_delta_svd_abs_cos']:.4f}`",
        f"- mean abs cosine, final nonlinear attack delta vs `A^T b`: `{summary['mean_attack_delta_outward_abs_cos']:.4f}`",
        f"- mean abs cosine, final nonlinear attack delta vs finite-radius affine direction: `{summary['mean_attack_delta_affine_eps_abs_cos']:.4f}`",
        "",
        "## Per Model Mean Direction Agreement",
        "",
        "| model | angle SVD vs A^T b deg | attack cos SVD | attack cos A^T b | attack cos affine eps |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in sorted(summary["by_model_svd_outward_abs_angle_deg"]):
        lines.append(
            f"| {model} | {summary['by_model_svd_outward_abs_angle_deg'][model]:.6g} | "
            f"{summary['by_model_attack_delta_svd_abs_cos'][model]:.6g} | "
            f"{summary['by_model_attack_delta_outward_abs_cos'][model]:.6g} | "
            f"{summary['by_model_attack_delta_affine_eps_abs_cos'][model]:.6g} |"
        )
    lines.extend([
        "",
        "## Compact Per Pair Table",
        "",
        *markdown_table(compact_rows, ["sample", "model", "sigma1", "angle_svd_vs_Atb_deg", "cos_attack_svd", "cos_attack_Atb", "cos_attack_affine", "actual_growth", "svd_gain", "affine_gain"]),
        "",
        "## All-Pair Correlations",
        "",
        "| x | y | n | Pearson | Spearman |",
        "|---|---|---:|---:|---:|",
    ])
    keep_x = {"error_spectral_norm", "bias_gradient_norm", "svd_local_gain_eps_mse", "outward_local_gain_eps_mse", "affine_local_gain_eps_mse"}
    keep_y = {"attack_loss_growth_abs", "attack_final_mse"}
    for r in all_corr:
        if r["x"] in keep_x and r["y"] in keep_y:
            lines.append(f"| `{r['x']}` | `{r['y']}` | {r['n']} | {float(r['pearson']):.6g} | {float(r['spearman']):.6g} |")
    lines.extend([
        "",
        "## Interpretation",
        "",
        "The angle between top SVD and `A^T b` is the clean way to measure your concern: if it is large, the infinitesimal attack direction is not the top singular vector direction.  The finite-radius affine direction is the more mathematically faithful local comparator for attack loss growth because it includes both the linear `2 b^T A delta` term and the quadratic `delta^T A^T A delta` term.",
        "",
        "For the actual 15-step nonlinear attack, the final delta can still differ from all three local directions because the model and PDE solver Jacobians change along the path.  So the right conclusion is not that SVD is wrong; it is that pure SVD measures local error sensitivity, while biased affine direction measures local attack-loss increase.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--svd-root", type=Path, default=DEFAULT_SVD_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--attack-epsilon-rms", type=float, default=0.12)
    parser.add_argument("--eps-rms-values", nargs="+", type=float, default=[1e-6, 1e-4, 1e-3, 1e-2, 0.12])
    parser.add_argument("--max-rows", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.svd_root = args.svd_root.resolve()
    args.out_dir = args.out_dir.resolve()
    args.report_md = args.report_md.resolve()
    rows, corr, summary = analyze(args)
    write_csv(args.out_dir / "biased_direction_metrics.csv", rows)
    write_csv(args.out_dir / "biased_direction_correlations.csv", corr)
    save_json(args.out_dir / "biased_direction_summary.json", summary)
    save_json(args.out_dir / "config.json", vars(args))
    write_report(args.report_md, args, rows, corr, summary)
    print(json.dumps({"out_dir": relpath(args.out_dir), "report": relpath(args.report_md), "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
