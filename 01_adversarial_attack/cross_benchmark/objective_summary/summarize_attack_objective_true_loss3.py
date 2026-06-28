#!/usr/bin/env python3
"""Summarize fixed-model fixed-data attack objective comparisons.

The script ingests raw Burgers, NS2D, and Darcy attack outputs and writes the
paper-facing tables/figures for final true loss3 comparisons. Smoke directories
are ignored by default.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


OBJECTIVES = ("loss1", "loss2", "loss3")


def fmt_token(value: Any) -> str:
    return str(value).replace(".", "p").replace("-", "m").replace("/", "_")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def skip_path(path: Path) -> bool:
    return "smoke" in str(path).lower()


def collect_burgers(analysis_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    root = analysis_root / "raw_runs" / "burgers"
    for csv_path in sorted(root.rglob("batch_true_loss_summary_all_indices.csv")):
        if skip_path(csv_path):
            continue
        df = pd.read_csv(csv_path)
        num_samples = int(df["initial_condition_index"].nunique()) if "initial_condition_index" in df.columns else 0
        for rec in df.to_dict("records"):
            method = str(rec["method"])
            objective = {"loss2_fixed": "loss2"}.get(method, method)
            if objective not in OBJECTIVES:
                continue
            protocol = (
                f"burgers_nu{fmt_token(rec['nu'])}_"
                f"{rec['norm']}_eps{fmt_token(rec['epsilon'])}_"
                f"alpha{fmt_token(rec['alpha'])}_steps{int(rec['steps'])}_N{num_samples or 'unknown'}"
            )
            rows.append(
                {
                    "system": "burgers1d",
                    "protocol_id": protocol,
                    "sample_id": int(rec["initial_condition_index"]),
                    "attack_objective": objective,
                    "update_method": "pgd",
                    "norm_or_constraint": str(rec["norm"]).upper(),
                    "epsilon": float(rec["epsilon"]),
                    "alpha": float(rec["alpha"]),
                    "steps": int(rec["steps"]),
                    "clean_true_loss3": float(rec["initial_true_loss"]),
                    "final_true_loss3": float(rec["final_true_loss"]),
                    "delta_true_loss3": float(rec["true_loss_increase"]),
                    "ratio_true_loss3": float(rec["true_loss_ratio"]),
                    "model_path": "1D_Burgers/trained_models/attack_ready/burgers_nu0.001_fno1d_500/checkpoints/pytorch_fno1d_500.pt",
                    "data_path": "1D_Burgers/datasets/1D/Burgers/batched_exponax_splits/dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45",
                    "run_dir": str(csv_path.parent),
                }
            )
    return rows


def burgers_protocol_from_summary(summary_csv: Path) -> str:
    df = pd.read_csv(summary_csv)
    if df.empty:
        return f"burgers_unknown_{summary_csv.parent.name}"
    rec = df.iloc[0]
    num_samples = int(df["initial_condition_index"].nunique()) if "initial_condition_index" in df.columns else 0
    return (
        f"burgers_nu{fmt_token(rec['nu'])}_"
        f"{rec['norm']}_eps{fmt_token(rec['epsilon'])}_"
        f"alpha{fmt_token(rec['alpha'])}_steps{int(rec['steps'])}_N{num_samples or 'unknown'}"
    )


def ns_mode_root(path: Path) -> Path:
    # .../mode_x/batch_0000_0001/loss1/steepest_add/final_state_metrics.csv
    return path.parents[3]


def ns_protocol_from_manifest(mode_root: Path, method: str) -> tuple[str, dict[str, Any]]:
    manifest = read_json(mode_root / "manifest.json")
    args = manifest.get("args", {})
    epsilon = float(args.get("epsilon", 32))
    alpha = float(args.get("alpha", 1))
    steps = int(args.get("steps", 100))
    p = str(args.get("p", "2"))
    q = str(args.get("q", "2"))
    num_samples = int(args.get("num_samples", 0) or 0)
    start_index = int(args.get("start_index", 0) or 0)
    protocol_n = num_samples
    # The NS2D budget sweep was run in two chunks: indices 0..11 first and
    # indices 12..19 as a later supplement. Normalize both chunks into the
    # intended N=20 protocol so tables report the combined experiment.
    if steps == 50 and start_index in (0, 12) and num_samples in (8, 12):
        if (epsilon, alpha) in {(8.0, 0.25), (16.0, 0.5), (32.0, 1.0), (64.0, 2.0)}:
            protocol_n = 20
    protocol = (
        f"ns2d_recurrent_eps{fmt_token(epsilon)}_alpha{fmt_token(alpha)}_"
        f"p{fmt_token(p)}_q{fmt_token(q)}_steps{steps}_method{method}_N{protocol_n or 'unknown'}_"
        "modes_loss1_allw_loss2_allatargetw_loss3_allw"
    )
    return protocol, args


def collect_ns2d(analysis_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    root = analysis_root / "raw_runs" / "ns2d"
    for csv_path in sorted(root.rglob("final_state_metrics.csv")):
        if skip_path(csv_path):
            continue
        method = csv_path.parent.name
        objective = csv_path.parent.parent.name
        if objective not in OBJECTIVES:
            continue
        mode_root = ns_mode_root(csv_path)
        protocol, args = ns_protocol_from_manifest(mode_root, method)
        epsilon = float(args.get("epsilon", 32))
        alpha = float(args.get("alpha", 1))
        steps = int(args.get("steps", 100))
        p = str(args.get("p", "2"))
        q = str(args.get("q", "2"))
        df = pd.read_csv(csv_path)
        for rec in df.to_dict("records"):
            clean = float(rec["clean_true_loss"])
            final = float(rec["adv_true_loss"])
            rows.append(
                {
                    "system": "ns2d_recurrent",
                    "protocol_id": protocol,
                    "sample_id": int(rec["dataset_index"]),
                    "attack_objective": objective,
                    "update_method": method,
                    "norm_or_constraint": f"L{p}/q{q}",
                    "epsilon": epsilon,
                    "alpha": alpha,
                    "steps": steps,
                    "clean_true_loss3": clean,
                    "final_true_loss3": final,
                    "delta_true_loss3": final - clean,
                    "ratio_true_loss3": final / clean if clean else np.nan,
                    "model_path": str(args.get("checkpoint", "")),
                    "data_path": str(args.get("test_path", "")),
                    "run_dir": str(csv_path.parent),
                    "mode_spec": str(args.get("mode_spec", mode_root.name)),
                }
            )
    return rows


def darcy_protocol_from_manifest(manifest: dict[str, Any], method: str) -> str:
    steps = int(manifest.get("steps", 100))
    epsilon_fraction = float(manifest.get("epsilon_fraction", np.nan))
    epsilon_flips = manifest.get("epsilon_flips", "")
    alpha_flips = manifest.get("alpha_flips", "")
    num_samples = int(manifest.get("num_samples", 0) or 0)
    return (
        f"darcy_binary_epsfrac{fmt_token(epsilon_fraction)}_"
        f"K{epsilon_flips}_alphaflips{alpha_flips}_steps{steps}_method{method}_N{num_samples or 'unknown'}"
    )


def collect_darcy(analysis_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    root = analysis_root / "raw_runs"
    for csv_path in sorted(root.rglob("final_per_sample.csv")):
        if skip_path(csv_path):
            continue
        method = csv_path.parent.name
        objective = csv_path.parent.parent.name
        experiment = csv_path.parent.parent.parent.name
        if experiment != "loss_objectives" or objective not in OBJECTIVES:
            continue
        run_root = csv_path.parents[3]
        manifest = read_json(run_root / "experiment_manifest.json")
        steps = int(manifest.get("steps", 100))
        epsilon_fraction = float(manifest.get("epsilon_fraction", np.nan))
        epsilon_flips = manifest.get("epsilon_flips", "")
        alpha_flips = manifest.get("alpha_flips", "")
        protocol = darcy_protocol_from_manifest(manifest, method)
        df = pd.read_csv(csv_path)
        for rec in df.to_dict("records"):
            clean = float(rec["clean_loss3"])
            final = float(rec["final_loss3_true"])
            rows.append(
                {
                    "system": "darcy_cflow",
                    "protocol_id": protocol,
                    "sample_id": int(rec["dataset_index"]),
                    "attack_objective": objective,
                    "update_method": method,
                    "norm_or_constraint": f"binary_hamming_K{epsilon_flips}",
                    "epsilon": epsilon_fraction,
                    "alpha": float(alpha_flips) if str(alpha_flips) else np.nan,
                    "steps": steps,
                    "clean_true_loss3": clean,
                    "final_true_loss3": final,
                    "delta_true_loss3": final - clean,
                    "ratio_true_loss3": final / clean if clean else np.nan,
                    "model_path": str(manifest.get("checkpoint", "")),
                    "data_path": str(manifest.get("dataset", "")),
                    "run_dir": str(csv_path.parent),
                }
            )
    return rows


def collect_darcy_physics_auxiliary(analysis_root: Path) -> pd.DataFrame:
    columns = [
        "system",
        "protocol_id",
        "attack_objective",
        "n",
        "clean_loss4_physics_mean",
        "final_loss4_physics_mean",
        "final_loss4_physics_std",
        "final_loss4_physics_var",
        "final_loss4_physics_median",
        "loss4_physics_increase_mean",
        "clean_loss4_pde_mean",
        "final_loss4_pde_mean",
        "loss4_pde_increase_mean",
        "clean_loss4_bc_mean",
        "final_loss4_bc_mean",
        "loss4_bc_increase_mean",
        "final_true_loss3_mean",
        "final_true_loss3_std",
        "loss3_increase_mean",
        "final_flip_count_mean",
    ]
    rows: list[dict[str, Any]] = []
    root = analysis_root / "raw_runs" / "darcy_physics"
    for csv_path in sorted(root.rglob("final_per_sample.csv")):
        if skip_path(csv_path):
            continue
        run_root = csv_path.parent
        manifest = read_json(run_root / "experiment_manifest.json")
        method = str(manifest.get("method", "steepest_replace"))
        protocol = darcy_protocol_from_manifest(manifest, method)
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        for rec in df.to_dict("records"):
            clean = finite_float(rec.get("clean_loss4_physics"))
            final = finite_float(rec.get("final_loss4_physics"))
            rows.append(
                {
                    "system": "darcy_cflow",
                    "protocol_id": protocol,
                    "sample_id": int(rec["dataset_index"]),
                    "attack_objective": "loss4_physics",
                    "update_method": method,
                    "norm_or_constraint": f"binary_hamming_K{manifest.get('epsilon_flips', '')}",
                    "epsilon": finite_float(manifest.get("epsilon_flips", manifest.get("epsilon_fraction", np.nan))),
                    "alpha": finite_float(manifest.get("alpha_flips", np.nan)),
                    "steps": int(manifest.get("steps", 0) or 0),
                    "physics_metric": str(manifest.get("physics_metric", "")),
                    "bc_weight": finite_float(manifest.get("bc_weight", np.nan)),
                    "clean_loss4_physics": clean,
                    "final_loss4_physics": final,
                    "loss4_physics_increase": final - clean if np.isfinite(clean) and np.isfinite(final) else np.nan,
                    "clean_loss4_pde": finite_float(rec.get("clean_loss4_pde")),
                    "final_loss4_pde": finite_float(rec.get("final_loss4_pde")),
                    "loss4_pde_increase": finite_float(rec.get("loss4_pde_increase")),
                    "clean_loss4_bc": finite_float(rec.get("clean_loss4_bc")),
                    "final_loss4_bc": finite_float(rec.get("final_loss4_bc")),
                    "loss4_bc_increase": finite_float(rec.get("loss4_bc_increase")),
                    "clean_true_loss3": finite_float(rec.get("clean_loss3")),
                    "final_true_loss3": finite_float(rec.get("final_loss3_true")),
                    "loss3_increase": finite_float(rec.get("loss3_increase")),
                    "final_flip_count": finite_float(rec.get("flip_count")),
                    "run_dir": str(run_root),
                }
            )
    if not rows:
        return pd.DataFrame(columns=columns)
    sample_df = pd.DataFrame(rows)
    out_rows = []
    for keys, g in sample_df.groupby(["system", "protocol_id", "attack_objective"], dropna=False):
        system, protocol, objective = keys
        final = g["final_loss4_physics"].astype(float)
        clean = g["clean_loss4_physics"].astype(float)
        out_rows.append(
            {
                "system": system,
                "protocol_id": protocol,
                "attack_objective": objective,
                "n": int(final.notna().sum()),
                "clean_loss4_physics_mean": float(clean.mean()),
                "final_loss4_physics_mean": float(final.mean()),
                "final_loss4_physics_std": float(final.std(ddof=1)) if len(final) > 1 else 0.0,
                "final_loss4_physics_var": float(final.var(ddof=1)) if len(final) > 1 else 0.0,
                "final_loss4_physics_median": float(final.median()),
                "loss4_physics_increase_mean": float(g["loss4_physics_increase"].astype(float).mean()),
                "clean_loss4_pde_mean": float(g["clean_loss4_pde"].astype(float).mean()),
                "final_loss4_pde_mean": float(g["final_loss4_pde"].astype(float).mean()),
                "loss4_pde_increase_mean": float(g["loss4_pde_increase"].astype(float).mean()),
                "clean_loss4_bc_mean": float(g["clean_loss4_bc"].astype(float).mean()),
                "final_loss4_bc_mean": float(g["final_loss4_bc"].astype(float).mean()),
                "loss4_bc_increase_mean": float(g["loss4_bc_increase"].astype(float).mean()),
                "final_true_loss3_mean": float(g["final_true_loss3"].astype(float).mean()),
                "final_true_loss3_std": float(g["final_true_loss3"].astype(float).std(ddof=1)) if len(g) > 1 else 0.0,
                "loss3_increase_mean": float(g["loss3_increase"].astype(float).mean()),
                "final_flip_count_mean": float(g["final_flip_count"].astype(float).mean()),
            }
        )
    return pd.DataFrame(out_rows, columns=columns).sort_values(["system", "protocol_id", "attack_objective"])


def finite_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return np.nan
    return out if np.isfinite(out) else np.nan


def collect_burgers_curves(analysis_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    root = analysis_root / "raw_runs" / "burgers"
    for history_path in sorted(root.rglob("batch_loss_history.csv")):
        if skip_path(history_path):
            continue
        summary_path = history_path.parent / "batch_true_loss_summary_all_indices.csv"
        if not summary_path.exists():
            continue
        protocol = burgers_protocol_from_summary(summary_path)
        summary_df = pd.read_csv(summary_path)
        first = summary_df.iloc[0] if not summary_df.empty else {}
        df = pd.read_csv(history_path)
        for rec in df.to_dict("records"):
            objective = {"loss2_fixed": "loss2"}.get(str(rec["method"]), str(rec["method"]))
            if objective not in OBJECTIVES:
                continue
            rows.append(
                {
                    "system": "burgers1d",
                    "protocol_id": protocol,
                    "sample_id": -1,
                    "attack_objective": objective,
                    "step": int(rec["k"]),
                    "optimized_loss": finite_float(rec.get("optimized_loss")),
                    "true_loss3": finite_float(rec.get("true_loss")),
                    "delta_norm_or_flip_count": np.nan,
                    "runtime_seconds": np.nan,
                    "is_batch_mean": True,
                    "epsilon": finite_float(first.get("epsilon", np.nan)) if hasattr(first, "get") else np.nan,
                    "alpha": finite_float(first.get("alpha", np.nan)) if hasattr(first, "get") else np.nan,
                    "steps": int(first.get("steps", 0)) if hasattr(first, "get") and pd.notna(first.get("steps", np.nan)) else np.nan,
                    "run_dir": str(history_path.parent),
                }
            )
    return rows


def collect_ns2d_curves(analysis_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    root = analysis_root / "raw_runs" / "ns2d"
    for csv_path in sorted(root.rglob("per_sample_step_metrics.csv")):
        if skip_path(csv_path):
            continue
        method = csv_path.parent.name
        objective = csv_path.parent.parent.name
        if objective not in OBJECTIVES:
            continue
        mode_root = ns_mode_root(csv_path)
        protocol, args = ns_protocol_from_manifest(mode_root, method)
        df = pd.read_csv(csv_path)
        for rec in df.to_dict("records"):
            true_loss3 = finite_float(rec.get("true_loss"))
            if not np.isfinite(true_loss3):
                true_loss3 = finite_float(rec.get("loss3"))
            rows.append(
                {
                    "system": "ns2d_recurrent",
                    "protocol_id": protocol,
                    "sample_id": int(rec["dataset_index"]),
                    "attack_objective": objective,
                    "step": int(rec["k"]),
                    "optimized_loss": finite_float(rec.get("surrogate_loss_value", rec.get("active_loss_value"))),
                    "true_loss3": true_loss3,
                    "delta_norm_or_flip_count": finite_float(rec.get("delta_p", rec.get("delta_l2"))),
                    "runtime_seconds": finite_float(rec.get("seconds_since_method_start")),
                    "is_batch_mean": False,
                    "epsilon": float(args.get("epsilon", rec.get("epsilon", np.nan))),
                    "alpha": float(args.get("alpha", rec.get("alpha", np.nan))),
                    "steps": int(args.get("steps", 0) or 0),
                    "run_dir": str(csv_path.parent),
                }
            )
    return rows


def collect_darcy_curves(analysis_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    root = analysis_root / "raw_runs"
    for csv_path in sorted(root.rglob("step_losses_per_sample.csv")):
        if skip_path(csv_path):
            continue
        method = csv_path.parent.name
        objective = csv_path.parent.parent.name
        experiment = csv_path.parent.parent.parent.name
        if experiment != "loss_objectives" or objective not in OBJECTIVES:
            continue
        run_root = csv_path.parents[3]
        manifest = read_json(run_root / "experiment_manifest.json")
        protocol = darcy_protocol_from_manifest(manifest, method)
        df = pd.read_csv(csv_path)
        df = df[df["phase"].isin(["pre_update", "final"])]
        for rec in df.to_dict("records"):
            rows.append(
                {
                    "system": "darcy_cflow",
                    "protocol_id": protocol,
                    "sample_id": int(rec["dataset_index"]),
                    "attack_objective": objective,
                    "step": int(rec["step"]),
                    "optimized_loss": finite_float(rec.get("surrogate_loss")),
                    "true_loss3": finite_float(rec.get("true_loss3")),
                    "delta_norm_or_flip_count": finite_float(rec.get("flip_count")),
                    "runtime_seconds": np.nan,
                    "is_batch_mean": False,
                    "epsilon": finite_float(manifest.get("epsilon_fraction", np.nan)),
                    "alpha": finite_float(manifest.get("alpha_flips", np.nan)),
                    "steps": int(manifest.get("steps", 0) or 0),
                    "run_dir": str(csv_path.parent),
                }
            )
    return rows


def collect_per_step_curves(analysis_root: Path) -> pd.DataFrame:
    rows = collect_burgers_curves(analysis_root) + collect_ns2d_curves(analysis_root) + collect_darcy_curves(analysis_root)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["system", "protocol_id", "attack_objective", "sample_id", "step"])


def interp_at(steps: np.ndarray, values: np.ndarray, step: float) -> float:
    mask = np.isfinite(steps) & np.isfinite(values)
    steps = steps[mask]
    values = values[mask]
    if steps.size == 0 or step < np.min(steps) or step > np.max(steps):
        return np.nan
    return float(np.interp(step, steps, values))


def curve_metrics(curve: pd.DataFrame) -> dict[str, float]:
    curve = curve.sort_values("step")
    steps = curve["step"].to_numpy(dtype=float)
    values = curve["true_loss3"].to_numpy(dtype=float)
    mask = np.isfinite(steps) & np.isfinite(values)
    steps = steps[mask]
    values = values[mask]
    if steps.size == 0:
        return {
            "clean_true_loss3": np.nan,
            "final_true_loss3": np.nan,
            "delta_true_loss3": np.nan,
            "ratio_true_loss3": np.nan,
            "early_slope_0_10": np.nan,
            "mid_slope_10_50": np.nan,
            "late_slope_50_100": np.nan,
            "auc_true_loss3": np.nan,
            "max_true_loss3": np.nan,
            "step_of_max_true_loss3": np.nan,
            "step_to_90pct_final": np.nan,
        }
    clean = float(values[0])
    final = float(values[-1])

    def slope(lo: float, hi: float) -> float:
        vlo = interp_at(steps, values, lo)
        vhi = interp_at(steps, values, hi)
        return (vhi - vlo) / (hi - lo) if np.isfinite(vlo) and np.isfinite(vhi) and hi != lo else np.nan

    max_idx = int(np.nanargmax(values))
    threshold = clean + 0.9 * (final - clean)
    if final >= clean:
        hit = steps[values >= threshold]
    else:
        hit = steps[values <= threshold]
    return {
        "clean_true_loss3": clean,
        "final_true_loss3": final,
        "delta_true_loss3": final - clean,
        "ratio_true_loss3": final / clean if clean else np.nan,
        "early_slope_0_10": slope(0, 10),
        "mid_slope_10_50": slope(10, 50),
        "late_slope_50_100": slope(50, 100),
        "auc_true_loss3": float(np.trapezoid(values, steps)) if steps.size > 1 else np.nan,
        "max_true_loss3": float(values[max_idx]),
        "step_of_max_true_loss3": float(steps[max_idx]),
        "step_to_90pct_final": float(hit[0]) if hit.size else np.nan,
    }


def growth_rate_summary(per_step: pd.DataFrame) -> pd.DataFrame:
    if per_step.empty:
        return pd.DataFrame()
    sample_rows = []
    for keys, curve in per_step.groupby(["system", "protocol_id", "attack_objective", "sample_id"], dropna=False):
        system, protocol, objective, sample_id = keys
        metrics = curve_metrics(curve)
        sample_rows.append(
            {
                "system": system,
                "protocol_id": protocol,
                "attack_objective": objective,
                "sample_id": sample_id,
                **metrics,
            }
        )
    sample_df = pd.DataFrame(sample_rows)
    rows = []
    for keys, g in sample_df.groupby(["system", "protocol_id", "attack_objective"], dropna=False):
        system, protocol, objective = keys
        finite_final = g["final_true_loss3"].notna()
        rows.append(
            {
                "system": system,
                "protocol_id": protocol,
                "attack_objective": objective,
                "n": int(finite_final.sum()),
                "early_slope_mean": float(g["early_slope_0_10"].mean()),
                "mid_slope_mean": float(g["mid_slope_10_50"].mean()),
                "late_slope_mean": float(g["late_slope_50_100"].mean()),
                "auc_mean": float(g["auc_true_loss3"].mean()),
                "max_true_loss3_mean": float(g["max_true_loss3"].mean()),
                "step_to_90pct_final_mean": float(g["step_to_90pct_final"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["system", "protocol_id", "attack_objective"])


def objective_summary(all_samples: pd.DataFrame) -> pd.DataFrame:
    if all_samples.empty:
        return pd.DataFrame()
    rows = []
    for keys, g in all_samples.groupby(["system", "protocol_id", "attack_objective"], dropna=False):
        system, protocol, objective = keys
        final = g["final_true_loss3"].astype(float)
        clean = g["clean_true_loss3"].astype(float)
        rows.append(
            {
                "system": system,
                "protocol_id": protocol,
                "attack_objective": objective,
                "n": int(final.notna().sum()),
                "final_true_loss3_mean": float(final.mean()),
                "final_true_loss3_std": float(final.std(ddof=1)) if len(final) > 1 else 0.0,
                "final_true_loss3_var": float(final.var(ddof=1)) if len(final) > 1 else 0.0,
                "final_true_loss3_median": float(final.median()),
                "final_true_loss3_se": float(final.std(ddof=1) / math.sqrt(len(final))) if len(final) > 1 else 0.0,
                "clean_true_loss3_mean": float(clean.mean()),
                "increase_mean": float(g["delta_true_loss3"].astype(float).mean()),
                "ratio_mean": float(g["ratio_true_loss3"].astype(float).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["system", "protocol_id", "attack_objective"])


def winner_summary(all_samples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if all_samples.empty:
        return pd.DataFrame(), pd.DataFrame()
    sample_rows = []
    summary_rows = []
    for (system, protocol), g in all_samples.groupby(["system", "protocol_id"], dropna=False):
        pivot = g.pivot_table(index="sample_id", columns="attack_objective", values="final_true_loss3", aggfunc="first")
        for objective in OBJECTIVES:
            if objective not in pivot.columns:
                pivot[objective] = np.nan
        complete = pivot.dropna(subset=list(OBJECTIVES), how="any")
        counts = {obj: 0 for obj in OBJECTIVES}
        ties = 0
        for sample_id, row in complete.iterrows():
            vals = {obj: float(row[obj]) for obj in OBJECTIVES}
            max_val = max(vals.values())
            winners = [obj for obj, val in vals.items() if np.isclose(val, max_val, rtol=1e-10, atol=1e-12)]
            if len(winners) == 1:
                counts[winners[0]] += 1
                tie_text = ""
            else:
                ties += 1
                tie_text = ",".join(winners)
            sample_rows.append(
                {
                    "system": system,
                    "protocol_id": protocol,
                    "sample_id": sample_id,
                    "winner": winners[0] if len(winners) == 1 else "tie",
                    "tie_objectives": tie_text,
                    **{f"{obj}_final_true_loss3": vals[obj] for obj in OBJECTIVES},
                }
            )
        n = int(len(complete))
        summary_rows.append(
            {
                "system": system,
                "protocol_id": protocol,
                "n_complete_samples": n,
                "loss1_wins": counts["loss1"],
                "loss2_wins": counts["loss2"],
                "loss3_wins": counts["loss3"],
                "ties": ties,
                "loss1_win_rate": counts["loss1"] / n if n else np.nan,
                "loss2_win_rate": counts["loss2"] / n if n else np.nan,
                "loss3_win_rate": counts["loss3"] / n if n else np.nan,
            }
        )
    return (
        pd.DataFrame(summary_rows).sort_values(["system", "protocol_id"]),
        pd.DataFrame(sample_rows).sort_values(["system", "protocol_id", "sample_id"]) if sample_rows else pd.DataFrame(),
    )


def bootstrap_ci(diff: np.ndarray, rng: np.random.Generator, reps: int = 5000) -> tuple[float, float]:
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    if diff.size < 2:
        return np.nan, np.nan
    draws = rng.choice(diff, size=(reps, diff.size), replace=True).mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def paired_tests(all_samples: pd.DataFrame) -> pd.DataFrame:
    if all_samples.empty:
        return pd.DataFrame()
    try:
        from scipy import stats
    except Exception:
        stats = None
    rows = []
    rng = np.random.default_rng(20260622)
    for (system, protocol), g in all_samples.groupby(["system", "protocol_id"], dropna=False):
        pivot = g.pivot_table(index="sample_id", columns="attack_objective", values="final_true_loss3", aggfunc="first")
        for objective in OBJECTIVES:
            if objective not in pivot.columns:
                pivot[objective] = np.nan
        complete = pivot.dropna(subset=list(OBJECTIVES), how="any")
        comparisons = [
            ("loss3_minus_loss1", "loss3", "loss1"),
            ("loss3_minus_loss2", "loss3", "loss2"),
            ("loss2_minus_loss1", "loss2", "loss1"),
        ]
        for comparison, lhs, rhs in comparisons:
            if complete.empty:
                diff = np.asarray([], dtype=float)
            else:
                diff = (complete[lhs] - complete[rhs]).to_numpy(dtype=float)
            n = int(diff.size)
            mean = float(np.mean(diff)) if n else np.nan
            std = float(np.std(diff, ddof=1)) if n > 1 else np.nan
            cohen_dz = float(mean / std) if n > 1 and np.isfinite(std) and std > 0 else np.nan
            if stats is not None and n > 1:
                t = stats.ttest_rel(complete[lhs], complete[rhs], nan_policy="omit")
                try:
                    w = stats.wilcoxon(diff)
                    wilcoxon_p = float(w.pvalue)
                except Exception:
                    wilcoxon_p = np.nan
                t_p = float(t.pvalue)
            else:
                t_p = np.nan
                wilcoxon_p = np.nan
            ci_low, ci_high = bootstrap_ci(diff, rng)
            rows.append(
                {
                    "system": system,
                    "protocol_id": protocol,
                    "comparison": comparison,
                    "n": n,
                    "mean_diff": mean,
                    "std_diff": std,
                    "cohen_dz": cohen_dz,
                    "paired_ttest_p_value": t_p,
                    "wilcoxon_p_value": wilcoxon_p,
                    "bootstrap_ci95_low": ci_low,
                    "bootstrap_ci95_high": ci_high,
                    "power_note": "low power" if n < 20 else "",
                }
            )
    return pd.DataFrame(rows).sort_values(["system", "protocol_id", "comparison"])


def mean_std_text(mean: float, std: float) -> str:
    if not np.isfinite(mean):
        return "incomplete"
    return f"{mean:.6g} +/- {std:.6g}"


def paper_ready_table(obj: pd.DataFrame, winners: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if obj.empty:
        return pd.DataFrame()
    for (system, protocol), g in obj.groupby(["system", "protocol_id"], dropna=False):
        by_obj = {r["attack_objective"]: r for r in g.to_dict("records")}
        win = winners[(winners["system"] == system) & (winners["protocol_id"] == protocol)]
        win_row = win.iloc[0].to_dict() if not win.empty else {}
        means = {name: by_obj.get(name, {}).get("final_true_loss3_mean", np.nan) for name in OBJECTIVES}
        complete = all(name in by_obj for name in OBJECTIVES)
        if complete and np.isfinite(means["loss3"]) and means["loss3"] >= max(means["loss1"], means["loss2"]):
            conclusion = "loss3 strongest by mean"
        elif complete:
            conclusion = "mixed"
        else:
            conclusion = "incomplete"
        first = next(iter(by_obj.values()))
        rows.append(
            {
                "system": system,
                "protocol_id": protocol,
                "n": int(win_row.get("n_complete_samples", 0) or 0),
                "method": first.get("update_method", ""),
                "budget": f"{first.get('norm_or_constraint', '')}, eps={first.get('epsilon', '')}, alpha={first.get('alpha', '')}, steps={first.get('steps', '')}",
                "loss1 mean+/-std": mean_std_text(by_obj.get("loss1", {}).get("final_true_loss3_mean", np.nan), by_obj.get("loss1", {}).get("final_true_loss3_std", np.nan)),
                "loss2 mean+/-std": mean_std_text(by_obj.get("loss2", {}).get("final_true_loss3_mean", np.nan), by_obj.get("loss2", {}).get("final_true_loss3_std", np.nan)),
                "loss3 mean+/-std": mean_std_text(by_obj.get("loss3", {}).get("final_true_loss3_mean", np.nan), by_obj.get("loss3", {}).get("final_true_loss3_std", np.nan)),
                "loss3 wins": int(win_row.get("loss3_wins", 0) or 0),
                "loss3 win rate": float(win_row.get("loss3_win_rate", np.nan)),
                "conclusion": conclusion,
            }
        )
    return pd.DataFrame(rows).sort_values(["system", "protocol_id"])


def parameter_sweep_summary(all_samples: pd.DataFrame, obj: pd.DataFrame, winners: pd.DataFrame) -> pd.DataFrame:
    if all_samples.empty or obj.empty:
        return pd.DataFrame()
    rows = []
    for (system, protocol), g in all_samples.groupby(["system", "protocol_id"], dropna=False):
        obj_rows = obj[(obj["system"] == system) & (obj["protocol_id"] == protocol)]
        by_obj = {r["attack_objective"]: r for r in obj_rows.to_dict("records")}
        if not all(name in by_obj for name in OBJECTIVES):
            continue
        win = winners[(winners["system"] == system) & (winners["protocol_id"] == protocol)]
        win_row = win.iloc[0].to_dict() if not win.empty else {}
        means = {name: float(by_obj[name]["final_true_loss3_mean"]) for name in OBJECTIVES}
        win_rates = {name: float(win_row.get(f"{name}_win_rate", np.nan)) for name in OBJECTIVES}
        first = g.iloc[0]
        epsilon = float(first.get("epsilon", np.nan))
        alpha = float(first.get("alpha", np.nan))
        if system == "darcy_cflow":
            match = re.search(r"binary_hamming_K([0-9.]+)", str(first.get("norm_or_constraint", "")))
            if match:
                epsilon = float(match.group(1))
        rows.append(
            {
                "system": system,
                "protocol_id": protocol,
                "epsilon": epsilon,
                "alpha": alpha,
                "steps": int(first.get("steps", 0) or 0),
                "n": int(win_row.get("n_complete_samples", 0) or 0),
                "best_mean_objective": max(means, key=means.get),
                "best_winrate_objective": max(win_rates, key=lambda name: -np.inf if not np.isfinite(win_rates[name]) else win_rates[name]),
                "loss1_mean": means["loss1"],
                "loss2_mean": means["loss2"],
                "loss3_mean": means["loss3"],
                "loss1_win_rate": win_rates["loss1"],
                "loss2_win_rate": win_rates["loss2"],
                "loss3_win_rate": win_rates["loss3"],
            }
        )
    return pd.DataFrame(rows).sort_values(["system", "epsilon", "alpha", "protocol_id"])


def write_figures(
    obj: pd.DataFrame,
    winners: pd.DataFrame,
    tests: pd.DataFrame,
    per_step: pd.DataFrame,
    sweep: pd.DataFrame,
    figures: Path,
) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        (figures / "figure_error.txt").write_text(str(exc), encoding="utf-8")
        return
    if not obj.empty:
        labels = [f"{r.system}\n{r.attack_objective}" for r in obj.itertuples()]
        vals = obj["final_true_loss3_mean"].to_numpy(float)
        errs = obj["final_true_loss3_se"].fillna(0).to_numpy(float)
        fig, ax = plt.subplots(figsize=(max(7, 0.55 * len(labels)), 4.2), constrained_layout=True)
        ax.bar(np.arange(len(vals)), vals, yerr=errs, color="#4477AA")
        ax.set_xticks(np.arange(len(vals)), labels, rotation=45, ha="right")
        ax.set_ylabel("mean final true loss3")
        fig.savefig(figures / "mean_final_true_loss3_by_objective.png", dpi=180)
        plt.close(fig)
    if not winners.empty:
        labels = winners["system"].astype(str).tolist()
        vals = winners["loss3_win_rate"].to_numpy(float)
        fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(labels)), 4), constrained_layout=True)
        ax.bar(np.arange(len(vals)), vals, color="#228833")
        ax.set_ylim(0, 1)
        ax.set_xticks(np.arange(len(vals)), labels, rotation=30, ha="right")
        ax.set_ylabel("loss3 strict win rate")
        fig.savefig(figures / "win_rate_by_objective.png", dpi=180)
        plt.close(fig)
    if not tests.empty:
        labels = [f"{r.system}\n{r.comparison}" for r in tests.itertuples()]
        vals = tests["mean_diff"].to_numpy(float)
        lows = tests["bootstrap_ci95_low"].to_numpy(float)
        highs = tests["bootstrap_ci95_high"].to_numpy(float)
        yerr = np.vstack([vals - lows, highs - vals])
        fig, ax = plt.subplots(figsize=(max(7, 0.7 * len(labels)), 4.2), constrained_layout=True)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.bar(np.arange(len(vals)), vals, yerr=yerr, color="#CC6677")
        ax.set_xticks(np.arange(len(vals)), labels, rotation=45, ha="right")
        ax.set_ylabel("paired mean final true loss3 difference")
        fig.savefig(figures / "paired_difference_loss3_minus_others.png", dpi=180)
        plt.close(fig)
    if not per_step.empty:
        curve = (
            per_step.dropna(subset=["true_loss3"])
            .groupby(["system", "protocol_id", "attack_objective", "step"], dropna=False)["true_loss3"]
            .mean()
            .reset_index()
        )
        complete_protocols = set()
        for (system, protocol), g in obj.groupby(["system", "protocol_id"], dropna=False):
            if set(g["attack_objective"]) >= set(OBJECTIVES):
                complete_protocols.add((system, protocol))
        curve = curve[curve[["system", "protocol_id"]].apply(tuple, axis=1).isin(complete_protocols)]
        if not curve.empty:
            protocols = list(curve[["system", "protocol_id"]].drop_duplicates().itertuples(index=False, name=None))
            ncols = min(3, max(1, len(protocols)))
            nrows = int(math.ceil(len(protocols) / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.2 * nrows), squeeze=False, constrained_layout=True)
            colors = {"loss1": "#4477AA", "loss2": "#EE6677", "loss3": "#228833"}
            for ax, (system, protocol) in zip(axes.ravel(), protocols):
                sub = curve[(curve["system"] == system) & (curve["protocol_id"] == protocol)]
                for objective in OBJECTIVES:
                    g = sub[sub["attack_objective"] == objective]
                    if not g.empty:
                        ax.plot(g["step"], g["true_loss3"], label=objective, color=colors.get(objective), linewidth=1.5)
                ax.set_title(f"{system}\n{protocol[:42]}", fontsize=8)
                ax.set_xlabel("step")
                ax.set_ylabel("true loss3")
                ax.legend(fontsize=7)
            for ax in axes.ravel()[len(protocols) :]:
                ax.axis("off")
            fig.savefig(figures / "growth_curve_by_objective.png", dpi=180)
            plt.close(fig)
    if not sweep.empty:
        systems = sweep["system"].drop_duplicates().tolist()
        fig, axes = plt.subplots(len(systems), 1, figsize=(8, max(3, 2.7 * len(systems))), squeeze=False, constrained_layout=True)
        for ax, system in zip(axes.ravel(), systems):
            sub = sweep[sweep["system"] == system].copy()
            sub["budget_label"] = sub.apply(lambda r: f"eps={r['epsilon']:.4g}\na={r['alpha']:.4g}", axis=1)
            x = np.arange(len(sub))
            width = 0.25
            ax.bar(x - width, sub["loss1_mean"], width, label="loss1", color="#4477AA")
            ax.bar(x, sub["loss2_mean"], width, label="loss2", color="#EE6677")
            ax.bar(x + width, sub["loss3_mean"], width, label="loss3", color="#228833")
            ax.set_title(system)
            ax.set_xticks(x, sub["budget_label"], rotation=30, ha="right")
            ax.set_ylabel("mean final true loss3")
            ax.legend()
        fig.savefig(figures / "parameter_sweep_heatmap.png", dpi=180)
        plt.close(fig)


def write_table_figure(paper: pd.DataFrame, figures: Path) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    if paper.empty:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    columns = [
        "system",
        "n",
        "budget",
        "loss1 mean+/-std",
        "loss2 mean+/-std",
        "loss3 mean+/-std",
        "loss3 win rate",
        "conclusion",
    ]
    display = paper[columns].copy()
    for col in display.columns:
        display[col] = display[col].map(lambda value: f"{value:.4g}" if isinstance(value, (float, np.floating)) else str(value))
    fig_h = max(3.0, 0.42 * (len(display) + 2))
    fig, ax = plt.subplots(figsize=(18, fig_h))
    ax.axis("off")
    table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.3)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor("#d0d7de")
        if row == 0:
            cell.set_facecolor("#1f2937")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f6f8fa")
    fig.tight_layout()
    fig.savefig(figures / "paper_ready_main_table.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def df_to_markdown(df: pd.DataFrame, max_rows: int = 40) -> str:
    if df.empty:
        return ""
    view = df.head(max_rows).copy()
    for col in view.columns:
        view[col] = view[col].map(lambda value: f"{value:.6g}" if isinstance(value, (float, np.floating)) else str(value))
    headers = list(view.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for rec in view.to_dict("records"):
        lines.append("| " + " | ".join(str(rec[col]).replace("\n", " ") for col in headers) + " |")
    if len(df) > max_rows:
        lines.append(f"\nShowing first {max_rows} of {len(df)} rows.")
    return "\n".join(lines)


def write_report(
    analysis_root: Path,
    all_samples: pd.DataFrame,
    obj: pd.DataFrame,
    winners: pd.DataFrame,
    tests: pd.DataFrame,
    growth: pd.DataFrame,
    sweep: pd.DataFrame,
    paper: pd.DataFrame,
    physics_aux: pd.DataFrame,
) -> None:
    report = analysis_root / "reports" / "attack_objective_true_loss3_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    systems = sorted(all_samples["system"].unique()) if not all_samples.empty else []
    complete = []
    incomplete = []
    for system in ("burgers1d", "ns2d_recurrent", "darcy_cflow"):
        rows = paper[paper["system"] == system] if not paper.empty else pd.DataFrame()
        if not rows.empty and all(rows["conclusion"] != "incomplete"):
            complete.append(system)
        else:
            incomplete.append(system)
    lines = [
        "# Attack Objective True Loss3 Report",
        "",
        "This is a fixed-model, fixed-data, fixed-budget attack objective comparison. It is not an adversarial-training or clean/generalization comparison.",
        "",
        f"Analysis root: `{analysis_root}`",
        f"Systems present: {', '.join(systems) if systems else 'none'}",
        f"Complete systems: {', '.join(complete) if complete else 'none'}",
        f"Incomplete systems: {', '.join(incomplete) if incomplete else 'none'}",
        "",
        "## Main Table",
        "",
    ]
    if paper.empty:
        lines.append("No formal rows were available.")
    else:
        lines.append(df_to_markdown(paper))
    lines.extend(["", "## Objective Summary", ""])
    lines.append(df_to_markdown(obj) if not obj.empty else "No objective summary rows.")
    lines.extend(["", "## Winner Summary", ""])
    lines.append(df_to_markdown(winners) if not winners.empty else "No winner rows.")
    lines.extend(["", "## Paired Tests", ""])
    lines.append(df_to_markdown(tests) if not tests.empty else "No paired-test rows.")
    lines.extend(["", "## Growth Rate Summary", ""])
    lines.append(df_to_markdown(growth) if not growth.empty else "No growth-rate rows.")
    lines.extend(["", "## Parameter Sweep Summary", ""])
    lines.append(df_to_markdown(sweep) if not sweep.empty else "No parameter-sweep rows.")
    lines.extend(["", "## Auxiliary Physics Summary", ""])
    lines.append(
        df_to_markdown(physics_aux)
        if not physics_aux.empty
        else "No auxiliary physics rows. Physics metrics are diagnostic and are not included in the main winner table."
    )
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            f"- `{analysis_root / 'tables' / 'all_samples.csv'}`",
            f"- `{analysis_root / 'tables' / 'per_step_curves.csv'}`",
            f"- `{analysis_root / 'tables' / 'objective_summary.csv'}`",
            f"- `{analysis_root / 'tables' / 'winner_summary.csv'}`",
            f"- `{analysis_root / 'tables' / 'growth_rate_summary.csv'}`",
            f"- `{analysis_root / 'tables' / 'paired_tests.csv'}`",
            f"- `{analysis_root / 'tables' / 'parameter_sweep_summary.csv'}`",
            f"- `{analysis_root / 'tables' / 'auxiliary_physics_summary.csv'}`",
            f"- `{analysis_root / 'tables' / 'paper_ready_main_table.csv'}`",
            f"- `{analysis_root / 'figures' / 'mean_final_true_loss3_by_objective.png'}`",
            f"- `{analysis_root / 'figures' / 'win_rate_by_objective.png'}`",
            f"- `{analysis_root / 'figures' / 'growth_curve_by_objective.png'}`",
            f"- `{analysis_root / 'figures' / 'paired_difference_loss3_minus_others.png'}`",
            f"- `{analysis_root / 'figures' / 'parameter_sweep_heatmap.png'}`",
        ]
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    args = parser.parse_args()
    analysis_root = args.analysis_root
    tables = analysis_root / "tables"
    figures = analysis_root / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    rows = collect_burgers(analysis_root) + collect_ns2d(analysis_root) + collect_darcy(analysis_root)
    all_samples = pd.DataFrame(rows)
    if not all_samples.empty:
        all_samples = all_samples.sort_values(["system", "protocol_id", "sample_id", "attack_objective"])
    per_step = collect_per_step_curves(analysis_root)
    obj = objective_summary(all_samples)
    winners, per_sample_winners = winner_summary(all_samples)
    tests = paired_tests(all_samples)
    growth = growth_rate_summary(per_step)
    sweep = parameter_sweep_summary(all_samples, obj, winners)
    paper = paper_ready_table(obj, winners)
    physics_aux = collect_darcy_physics_auxiliary(analysis_root)
    all_samples.to_csv(tables / "all_samples.csv", index=False)
    per_step.to_csv(tables / "per_step_curves.csv", index=False)
    obj.to_csv(tables / "objective_summary.csv", index=False)
    winners.to_csv(tables / "winner_summary.csv", index=False)
    per_sample_winners.to_csv(tables / "per_sample_winners.csv", index=False)
    tests.to_csv(tables / "paired_tests.csv", index=False)
    growth.to_csv(tables / "growth_rate_summary.csv", index=False)
    sweep.to_csv(tables / "parameter_sweep_summary.csv", index=False)
    physics_aux.to_csv(tables / "auxiliary_physics_summary.csv", index=False)
    paper.to_csv(tables / "paper_ready_main_table.csv", index=False)
    write_figures(obj, winners, tests, per_step, sweep, figures)
    write_table_figure(paper, figures)
    write_report(analysis_root, all_samples, obj, winners, tests, growth, sweep, paper, physics_aux)
    print(
        json.dumps(
            {
                "analysis_root": str(analysis_root),
                "all_samples": int(len(all_samples)),
                "per_step_rows": int(len(per_step)),
                "auxiliary_physics_rows": int(len(physics_aux)),
                "systems": sorted(all_samples["system"].unique()) if not all_samples.empty else [],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
