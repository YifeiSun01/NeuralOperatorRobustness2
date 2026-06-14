#!/usr/bin/env python3
"""Re-attack actual old4 Burgers outlier initial conditions with latest models.

The mixed 52-dataset table is useful for finding suspicious high-loss old4
cases, but the old4 source manifest is the older `generalization_datasets/burgers`
root. This script selects the actual initial conditions from that manifest,
runs a fresh attack100 with the latest loss1/loss2/loss3/random_solver_y
checkpoints, renders dense four-column figures, and writes local diagnostic
tables for those same samples.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch


REPO = Path(__file__).resolve().parents[1]
DATE = "20260614"

OLD4_ATTACK_ROOT = REPO / "forensics/burgers_first_master_full_p2q2_52datasets_4models_finalonly_20step_20260608"
RANDOM_ATTACK_ROOT = REPO / "forensics/burgers_random_solver7860_clean8000_full_suite_20260614/p2q2_attack"

TRACE_ROOT = REPO / f"forensics/burgers_old4_actual_outliers_latest_attack100_diagnostics_{DATE}"
VIS_ROOT = REPO / f"visualizations/burgers_old4_actual_outliers_latest_attack100_dense_bundle_{DATE}"
AUDIT_ROOT = REPO / "outputs/burgers_timematched_solver7860_clean8000_audit_20260614"
AUDIT_FIG_ROOT = AUDIT_ROOT / "figures/burgers_old4_actual_outliers_latest_attack100_dense_bundle_20260614"
AUDIT_DATA_ROOT = AUDIT_ROOT / "data/burgers_old4_actual_outliers_latest_attack100_diagnostics_20260614"
AUDIT_REPORT_ROOT = AUDIT_ROOT / "reports"

BASE_ATTACK_SCRIPT = REPO / "tools/plot_burgers_p2q2_baseline_vs_epoch1000_attack_gif.py"
RENDER_SCRIPT = REPO / "tools/plot_burgers_wideparam_random_field_six_model_one_row_20260613.py"
LABEL_SCRIPT = REPO / "tools/plot_burgers_round03_p2q2_combined_attack_panels.py"

MODEL_ORDER = ["loss1", "loss2", "loss3", "random_solver_y"]
TRACE_KEYS = ["step", "x_adv", "delta", "model", "solver", "diff", "loss", "delta_rms"]
EPS = 1e-12


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2) + "\n", encoding="utf-8")


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


def unit_np(v: np.ndarray) -> np.ndarray:
    vv = np.asarray(v, dtype=np.float64).reshape(-1)
    n = float(np.linalg.norm(vv))
    if n <= EPS:
        return np.full_like(vv, np.nan)
    return vv / n


def signed_cos_np(a: np.ndarray, b: np.ndarray) -> float:
    aa = unit_np(a)
    bb = unit_np(b)
    if not np.all(np.isfinite(aa)) or not np.all(np.isfinite(bb)):
        return math.nan
    return float(np.clip(np.dot(aa, bb), -1.0, 1.0))


def abs_cos_np(a: np.ndarray, b: np.ndarray) -> float:
    c = signed_cos_np(a, b)
    return float(abs(c)) if math.isfinite(c) else math.nan


def angle_deg_from_abs_cos(c: float) -> float:
    if not math.isfinite(c):
        return math.nan
    return float(math.degrees(math.acos(float(np.clip(c, 0.0, 1.0)))))


def parse_old_manifest() -> pd.DataFrame:
    manifest = json.loads((OLD4_ATTACK_ROOT / "manifest.json").read_text(encoding="utf-8"))
    z = np.load(OLD4_ATTACK_ROOT / "loss3_epoch1500/losses_and_delta_rms_by_sample.npz", allow_pickle=False)
    rows: list[dict[str, Any]] = []
    for i, item in enumerate(manifest):
        row = dict(item)
        row["global_sample_id"] = int(row["global_sample_id"])
        row["source_index"] = int(row["source_index"])
        row["dataset_sample_offset"] = int(row["dataset_sample_offset"])
        row["old_loss3_epoch1500_initial_loss_20step"] = float(z["initial_loss"][i])
        row["old_loss3_epoch1500_final_loss_20step"] = float(z["final_loss"][i])
        row["old_loss3_epoch1500_attack_increase_20step"] = float(z["final_loss"][i] - z["initial_loss"][i])
        rows.append(row)
    return pd.DataFrame(rows)


def select_outlier_samples(sample_count: int, per_dataset: bool) -> pd.DataFrame:
    df = parse_old_manifest()
    if per_dataset:
        dataset_rank = (
            df.groupby(["split", "dataset_id"], dropna=False)
            .agg(
                old_dataset_n=("global_sample_id", "size"),
                old_dataset_mean_increase_20step=("old_loss3_epoch1500_attack_increase_20step", "mean"),
                old_dataset_median_increase_20step=("old_loss3_epoch1500_attack_increase_20step", "median"),
                old_dataset_max_increase_20step=("old_loss3_epoch1500_attack_increase_20step", "max"),
            )
            .reset_index()
            .sort_values("old_dataset_mean_increase_20step", ascending=False)
            .head(sample_count)
        )
        chosen_rows = []
        for _, ds_row in dataset_rank.iterrows():
            subset = df[(df["split"] == ds_row["split"]) & (df["dataset_id"] == ds_row["dataset_id"])]
            best = subset.sort_values("old_loss3_epoch1500_attack_increase_20step", ascending=False).iloc[0].to_dict()
            best.update(ds_row.to_dict())
            chosen_rows.append(best)
        out = pd.DataFrame(chosen_rows)
        out["selection_reason"] = "top_dataset_mean_old_loss3_epoch1500_20step_then_worst_sample_in_dataset"
    else:
        out = df.sort_values("old_loss3_epoch1500_attack_increase_20step", ascending=False).head(sample_count).copy()
        out["selection_reason"] = "top_per_sample_old_loss3_epoch1500_20step"
    out = out.sort_values("old_loss3_epoch1500_attack_increase_20step", ascending=False).reset_index(drop=True)
    out["outlier_rank"] = np.arange(1, len(out) + 1)
    out["sample_id"] = [descriptive_sample_id(row["dataset_id"], int(row["source_index"])) for _, row in out.iterrows()]
    return out


def descriptive_sample_id(dataset_id: str, index: int) -> str:
    text = str(dataset_id)
    replacements = [
        ("burgers_", ""),
        ("far_", "Far_"),
        ("near_", "Near_"),
        ("mid_", "Mid_"),
        ("centered_scale_shift_", "CenteredScale_"),
        ("positive_shift_", "PositiveShift_"),
        ("negative_shift_", "NegativeShift_"),
        ("sign_centered_", "SignCentered_"),
        ("scale_shift_", "ScaleShift_"),
        ("scale", "Scale"),
        ("shiftm", "ShiftM"),
        ("shift", "Shift"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    parts = [part for part in text.split("_") if part]
    camel = "".join(part[:1].upper() + part[1:] for part in parts)
    return f"{camel}_idx{int(index)}"


def load_one_x(path: Path, index: int) -> torch.Tensor:
    data = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(data, dict) or "x" not in data:
        raise ValueError(f"{path} must contain x")
    x = data["x"][int(index)].detach().cpu().float()
    if x.ndim == 2 and x.shape[-1] == 1:
        x = x[:, 0]
    if x.ndim != 1 or x.numel() != 1024:
        raise ValueError(f"expected [1024] initial condition, got {tuple(x.shape)} from {path}")
    return x


def build_x_and_manifest(selected: pd.DataFrame) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    xs: list[torch.Tensor] = []
    manifest: list[dict[str, Any]] = []
    for _, row in selected.iterrows():
        path = Path(str(row["source_path"]))
        x = load_one_x(path, int(row["source_index"]))
        xs.append(x)
        item = {
            "sample_id": row["sample_id"],
            "outlier_rank": int(row["outlier_rank"]) if "outlier_rank" in row and pd.notna(row["outlier_rank"]) else None,
            "split": row["split"],
            "dataset_id": row["dataset_id"],
            "source_dataset_id": row["dataset_id"],
            "path": str(path),
            "index": int(row["source_index"]),
            "source_index": int(row["source_index"]),
            "old4_global_sample_id": int(row["global_sample_id"]),
            "old4_dataset_sample_offset": int(row["dataset_sample_offset"]),
            "old_loss3_epoch1500_initial_loss_20step": float(row["old_loss3_epoch1500_initial_loss_20step"]),
            "old_loss3_epoch1500_final_loss_20step": float(row["old_loss3_epoch1500_final_loss_20step"]),
            "old_loss3_epoch1500_attack_increase_20step": float(row["old_loss3_epoch1500_attack_increase_20step"]),
            "selection_reason": row["selection_reason"],
        }
        for key in (
            "old_dataset_n",
            "old_dataset_mean_increase_20step",
            "old_dataset_median_increase_20step",
            "old_dataset_max_increase_20step",
        ):
            if key in row and pd.notna(row[key]):
                item[key] = float(row[key])
        manifest.append(item)
    return torch.stack(xs).unsqueeze(-1), manifest


def configure_base_attack(base_mod: Any, attack_steps: int, frame_every: int, epsilon_rms: float) -> None:
    base_mod.ATTACK_STEPS = int(attack_steps)
    base_mod.FRAME_EVERY = int(frame_every)
    base_mod.EPSILON_RMS = float(epsilon_rms)
    base_mod.ALPHA_RMS = float(epsilon_rms) / 10.0


def run_attacks(
    base_mod: Any,
    renderer: Any,
    x_cpu: torch.Tensor,
    *,
    force: bool,
    trace_npz_path: Path,
) -> dict[str, dict[str, np.ndarray]]:
    if trace_npz_path.exists() and not force:
        with np.load(trace_npz_path, allow_pickle=False) as z:
            return {
                model_key: {key: np.asarray(z[f"{model_key}_{key}"]) for key in TRACE_KEYS}
                for model_key in MODEL_ORDER
            }
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for attack100 generation")
    device = torch.device("cuda")
    x_clean = x_cpu.to(device)
    traces: dict[str, dict[str, np.ndarray]] = {}
    for model_key in MODEL_ORDER:
        spec = renderer.MODEL_SPECS[model_key]
        checkpoint = Path(spec["checkpoint"])
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        print(f"[attack100] {model_key} checkpoint={checkpoint}", flush=True)
        model = base_mod.load_model(checkpoint, device)
        traces[model_key] = base_mod.run_attack(f"old4_outlier_latest_{model_key}", model, x_clean)
        del model
        torch.cuda.empty_cache()
    return traces


def save_trace_npz(path: Path, clean: np.ndarray, traces: dict[str, dict[str, np.ndarray]]) -> None:
    payload: dict[str, np.ndarray] = {"clean": clean}
    for model_key, trace in traces.items():
        for key, value in trace.items():
            payload[f"{model_key}_{key}"] = np.asarray(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def save_loss_curves_csv(path: Path, traces: dict[str, dict[str, np.ndarray]], manifest: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for model_key in MODEL_ORDER:
        tr = traces[model_key]
        for t, step in enumerate(tr["step"]):
            for i, item in enumerate(manifest):
                rows.append(
                    {
                        "model": model_key,
                        "step": int(step),
                        "sample_id": item["sample_id"],
                        "split": item["split"],
                        "dataset_id": item["dataset_id"],
                        "index": int(item["index"]),
                        "old4_global_sample_id": int(item["old4_global_sample_id"]),
                        "loss_mse": float(tr["loss"][t, i]),
                        "delta_rms": float(tr["delta_rms"][t, i]),
                    }
                )
    write_csv(path, rows)


def attack_summary_rows(traces: dict[str, dict[str, np.ndarray]], manifest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, item in enumerate(manifest):
        for model_key in MODEL_ORDER:
            tr = traces[model_key]
            initial = float(tr["loss"][0, i])
            final = float(tr["loss"][-1, i])
            rows.append(
                {
                    "sample_id": item["sample_id"],
                    "split": item["split"],
                    "dataset_id": item["dataset_id"],
                    "index": int(item["index"]),
                    "old4_global_sample_id": int(item["old4_global_sample_id"]),
                    "model": model_key,
                    "attack_steps": int(tr["step"][-1]),
                    "initial_loss_mse": initial,
                    "final_loss_mse": final,
                    "attack_loss_increase_mse": final - initial,
                    "relative_increase": (final - initial) / max(abs(initial), EPS),
                    "final_delta_rms": float(tr["delta_rms"][-1, i]),
                    "old_loss3_epoch1500_attack_increase_20step": float(item["old_loss3_epoch1500_attack_increase_20step"]),
                }
            )
    return rows


def pairwise_loss3_random_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    df = pd.DataFrame(summary_rows)
    rows: list[dict[str, Any]] = []
    for sample_id, group in df.groupby("sample_id", sort=False):
        l3 = group[group["model"] == "loss3"].iloc[0]
        rs = group[group["model"] == "random_solver_y"].iloc[0]
        rows.append(
            {
                "sample_id": sample_id,
                "dataset_id": l3["dataset_id"],
                "index": int(l3["index"]),
                "old4_global_sample_id": int(l3["old4_global_sample_id"]),
                "loss3_attack100_increase": float(l3["attack_loss_increase_mse"]),
                "random_solver_y_attack100_increase": float(rs["attack_loss_increase_mse"]),
                "loss3_minus_random_solver_y_attack100_increase": float(l3["attack_loss_increase_mse"] - rs["attack_loss_increase_mse"]),
                "winner_lower_increase": "loss3"
                if float(l3["attack_loss_increase_mse"]) < float(rs["attack_loss_increase_mse"])
                else "random_solver_y",
                "old_loss3_epoch1500_attack20_increase": float(l3["old_loss3_epoch1500_attack_increase_20step"]),
            }
        )
    return rows


def model_error_flat(base_mod: Any, model: Any, x: torch.Tensor) -> torch.Tensor:
    return (model(x) - base_mod.burgers_solver_target(x)).reshape(-1)


def jvp_error(base_mod: Any, model: Any, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    def fn(inp: torch.Tensor) -> torch.Tensor:
        return model_error_flat(base_mod, model, inp)

    _, out = torch.autograd.functional.jvp(fn, (x,), (v,), create_graph=False, strict=False)
    return out.reshape(-1)


def vjp_error(base_mod: Any, model: Any, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    x_req = x.detach().clone().requires_grad_(True)
    e = model_error_flat(base_mod, model, x_req)
    dot = torch.dot(e, u.reshape(-1).to(device=x.device, dtype=x.dtype))
    grad = torch.autograd.grad(dot, x_req, retain_graph=False, create_graph=False)[0]
    return grad.detach().reshape(-1)


def top_singular_power_iteration(
    base_mod: Any,
    model: Any,
    x: torch.Tensor,
    *,
    iters: int,
    seed: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    gen = torch.Generator(device=x.device)
    gen.manual_seed(int(seed))
    v = torch.randn(x.numel(), device=x.device, dtype=x.dtype, generator=gen)
    v = v / v.norm().clamp_min(EPS)
    u = torch.zeros_like(v)
    sigma = torch.tensor(0.0, device=x.device, dtype=x.dtype)
    for _ in range(int(iters)):
        y = jvp_error(base_mod, model, x, v.reshape_as(x))
        yn = y.norm().clamp_min(EPS)
        u = y / yn
        z = vjp_error(base_mod, model, x, u)
        zn = z.norm().clamp_min(EPS)
        v = z / zn
        sigma = jvp_error(base_mod, model, x, v.reshape_as(x)).norm()
    y = jvp_error(base_mod, model, x, v.reshape_as(x))
    sigma = y.norm().clamp_min(EPS)
    u = y / sigma
    return float(sigma.detach().cpu()), v.detach().cpu().numpy(), u.detach().cpu().numpy()


def compute_diagnostics(
    base_mod: Any,
    renderer: Any,
    x_cpu: torch.Tensor,
    traces: dict[str, dict[str, np.ndarray]],
    manifest: list[dict[str, Any]],
    *,
    power_iters: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for local diagnostics")
    device = torch.device("cuda")
    x_all = x_cpu.to(device)
    models: dict[str, Any] = {}
    for model_key in MODEL_ORDER:
        checkpoint = Path(renderer.MODEL_SPECS[model_key]["checkpoint"])
        models[model_key] = base_mod.load_model(checkpoint, device)

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for i, item in enumerate(manifest):
        x = x_all[i : i + 1].detach()
        for model_i, model_key in enumerate(MODEL_ORDER):
            model = models[model_key]
            print(f"[diag] sample={item['sample_id']} model={model_key}", flush=True)
            x_req = x.detach().clone().requires_grad_(True)
            e = model_error_flat(base_mod, model, x_req)
            clean_residual_np = e.detach().cpu().numpy().astype(np.float64)
            clean_residual_mse = float(np.mean(clean_residual_np * clean_residual_np))
            clean_residual_l2 = float(np.linalg.norm(clean_residual_np))
            half_sum = 0.5 * torch.dot(e, e)
            bias_grad = torch.autograd.grad(half_sum, x_req, retain_graph=False, create_graph=False)[0].detach()
            bias_grad_np = bias_grad.cpu().numpy().reshape(-1).astype(np.float64)
            delta_np = traces[model_key]["delta"][-1, i].reshape(-1).astype(np.float64)
            delta = torch.from_numpy(delta_np.astype(np.float32)).to(device=device).reshape_as(x)
            jdelta = jvp_error(base_mod, model, x.detach(), delta).detach().cpu().numpy().reshape(-1).astype(np.float64)
            sigma, top_right, top_left = top_singular_power_iteration(
                base_mod,
                model,
                x.detach(),
                iters=power_iters,
                seed=20260614 + 101 * i + model_i,
            )
            n = float(clean_residual_np.size)
            actual_initial = float(traces[model_key]["loss"][0, i])
            actual_final = float(traces[model_key]["loss"][-1, i])
            actual_increase = actual_final - actual_initial
            linear_gain = 2.0 * float(np.dot(clean_residual_np, jdelta)) / n
            quadratic_gain = float(np.dot(jdelta, jdelta)) / n
            delta_top_abs = abs_cos_np(delta_np, top_right)
            delta_bias_abs = abs_cos_np(delta_np, bias_grad_np)
            top_bias_abs = abs_cos_np(top_right, bias_grad_np)
            rows.append(
                {
                    "sample_id": item["sample_id"],
                    "split": item["split"],
                    "dataset_id": item["dataset_id"],
                    "index": int(item["index"]),
                    "old4_global_sample_id": int(item["old4_global_sample_id"]),
                    "model": model_key,
                    "attack100_initial_loss_mse": actual_initial,
                    "attack100_final_loss_mse": actual_final,
                    "attack100_loss_increase_mse": actual_increase,
                    "clean_residual_mse": clean_residual_mse,
                    "clean_residual_l2": clean_residual_l2,
                    "bias_gradient_norm_jt_error": float(np.linalg.norm(bias_grad_np)),
                    "bias_gradient_rms_jt_error": float(np.sqrt(np.mean(bias_grad_np * bias_grad_np))),
                    "j_error_delta_l2": float(np.linalg.norm(jdelta)),
                    "j_error_delta_rms": float(np.sqrt(np.mean(jdelta * jdelta))),
                    "linearized_endpoint_increase_mse": linear_gain + quadratic_gain,
                    "linearized_cross_gain_mse": linear_gain,
                    "linearized_residual_movement_mse": quadratic_gain,
                    "error_spectral_norm_power_iter": sigma,
                    "power_iteration_steps": int(power_iters),
                    "attack_delta_l2": float(np.linalg.norm(delta_np)),
                    "attack_delta_rms": float(np.sqrt(np.mean(delta_np * delta_np))),
                    "attack_delta_top_error_right_abs_cos": delta_top_abs,
                    "attack_delta_top_error_right_angle_deg": angle_deg_from_abs_cos(delta_top_abs),
                    "attack_delta_bias_gradient_abs_cos": delta_bias_abs,
                    "attack_delta_bias_gradient_angle_deg": angle_deg_from_abs_cos(delta_bias_abs),
                    "top_error_right_bias_gradient_abs_cos": top_bias_abs,
                    "top_error_right_bias_gradient_angle_deg": angle_deg_from_abs_cos(top_bias_abs),
                    "diagnostic_runtime_elapsed_sec": time.perf_counter() - started,
                    "diagnostic_note": "top singular direction is power-iteration estimate, not full SVD",
                }
            )
            torch.cuda.empty_cache()
    for model in models.values():
        del model
    torch.cuda.empty_cache()

    pair_rows: list[dict[str, Any]] = []
    df = pd.DataFrame(rows)
    for sample_id, group in df.groupby("sample_id", sort=False):
        l3 = group[group["model"] == "loss3"].iloc[0]
        rs = group[group["model"] == "random_solver_y"].iloc[0]
        sample_manifest = next(item for item in manifest if item["sample_id"] == sample_id)
        l3_delta = traces["loss3"]["delta"][-1, int(list(item["sample_id"] for item in manifest).index(sample_id))].reshape(-1)
        rs_delta = traces["random_solver_y"]["delta"][-1, int(list(item["sample_id"] for item in manifest).index(sample_id))].reshape(-1)
        delta_abs = abs_cos_np(l3_delta, rs_delta)
        pair_rows.append(
            {
                "sample_id": sample_id,
                "dataset_id": l3["dataset_id"],
                "index": int(l3["index"]),
                "old4_global_sample_id": int(sample_manifest["old4_global_sample_id"]),
                "loss3_attack100_increase": float(l3["attack100_loss_increase_mse"]),
                "random_solver_y_attack100_increase": float(rs["attack100_loss_increase_mse"]),
                "loss3_minus_random_solver_y_attack100_increase": float(l3["attack100_loss_increase_mse"] - rs["attack100_loss_increase_mse"]),
                "loss3_error_spectral_norm_power_iter": float(l3["error_spectral_norm_power_iter"]),
                "random_solver_y_error_spectral_norm_power_iter": float(rs["error_spectral_norm_power_iter"]),
                "loss3_minus_random_solver_y_error_spectral_norm_power_iter": float(l3["error_spectral_norm_power_iter"] - rs["error_spectral_norm_power_iter"]),
                "loss3_bias_gradient_norm_jt_error": float(l3["bias_gradient_norm_jt_error"]),
                "random_solver_y_bias_gradient_norm_jt_error": float(rs["bias_gradient_norm_jt_error"]),
                "loss3_minus_random_solver_y_bias_gradient_norm_jt_error": float(l3["bias_gradient_norm_jt_error"] - rs["bias_gradient_norm_jt_error"]),
                "loss3_j_error_delta_l2": float(l3["j_error_delta_l2"]),
                "random_solver_y_j_error_delta_l2": float(rs["j_error_delta_l2"]),
                "loss3_minus_random_solver_y_j_error_delta_l2": float(l3["j_error_delta_l2"] - rs["j_error_delta_l2"]),
                "loss3_random_solver_attack_delta_abs_cos": delta_abs,
                "loss3_random_solver_attack_delta_angle_deg": angle_deg_from_abs_cos(delta_abs),
            }
        )

    corr_rows = correlation_rows(df)
    return rows, pair_rows, corr_rows


def correlation_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    try:
        from scipy import stats
    except Exception:
        stats = None
    y_col = "attack100_loss_increase_mse"
    x_cols = [
        "clean_residual_mse",
        "clean_residual_l2",
        "bias_gradient_norm_jt_error",
        "j_error_delta_l2",
        "linearized_endpoint_increase_mse",
        "linearized_residual_movement_mse",
        "error_spectral_norm_power_iter",
        "attack_delta_top_error_right_abs_cos",
        "attack_delta_bias_gradient_abs_cos",
    ]
    rows: list[dict[str, Any]] = []
    scopes: list[tuple[str, pd.DataFrame]] = [("all_model_sample_rows", df)]
    scopes.extend((f"model_{model}", group) for model, group in df.groupby("model", sort=False))
    for scope, group in scopes:
        for x_col in x_cols:
            sub = group[[x_col, y_col]].apply(pd.to_numeric, errors="coerce").dropna()
            n = int(len(sub))
            row: dict[str, Any] = {"scope": scope, "x_metric": x_col, "y_metric": y_col, "n": n}
            if n >= 3:
                x = sub[x_col].to_numpy(dtype=np.float64)
                y = sub[y_col].to_numpy(dtype=np.float64)
                if np.std(x) > 0 and np.std(y) > 0:
                    if stats is not None:
                        pear = stats.pearsonr(x, y)
                        spear = stats.spearmanr(x, y)
                        row.update(
                            {
                                "pearson_r": float(pear.statistic),
                                "pearson_p": float(pear.pvalue),
                                "spearman_r": float(spear.statistic),
                                "spearman_p": float(spear.pvalue),
                            }
                        )
                    else:
                        row.update({"pearson_r": float(np.corrcoef(x, y)[0, 1]), "pearson_p": math.nan, "spearman_r": math.nan, "spearman_p": math.nan})
            rows.append(row)
    return rows


def render_figures(renderer: Any, labels: Any, clean: np.ndarray, manifest: list[dict[str, Any]], traces: dict[str, dict[str, np.ndarray]]) -> list[Path]:
    out_paths: list[Path] = []
    vis_dir = VIS_ROOT / "comparison_dense" / "outlier_group_actual_old4"
    for loss_scale in ["log", "linear"]:
        name = f"burgers_old4_actual_outliers_latest_loss1_loss2_loss3_random_solver_y_attack100_four_column_{loss_scale}_y.png"
        out_path = vis_dir / name
        renderer.render_one_row(
            labels,
            out_path,
            clean,
            manifest,
            traces,
            loss_scale=loss_scale,
            model_order=MODEL_ORDER,
            variant_label="actual old4 outlier initial conditions / latest loss1-loss2-loss3-random_solver_y",
            dpi=145,
        )
        out_paths.append(out_path)
        audit_path = AUDIT_FIG_ROOT / "comparison_dense" / "outlier_group_actual_old4" / name
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out_path, audit_path)
    return out_paths


def write_report(
    path: Path,
    manifest: list[dict[str, Any]],
    attack_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    diagnostic_rows: list[dict[str, Any]],
    corr_rows: list[dict[str, Any]],
    image_paths: list[Path],
) -> None:
    df = pd.DataFrame(attack_rows)
    pair = pd.DataFrame(pair_rows)
    lines = [
        "# Burgers Actual Old4 Outlier Re-Attack100 Diagnostics",
        "",
        "This report uses the actual old4 source manifest, not the relabeled mixed-52 dataset names.",
        "The selected initial conditions are the worst old `loss3_epoch1500` 20-step cases from the older `generalization_datasets/burgers` source, then re-attacked with the latest loss1/loss2/loss3/random_solver_y checkpoints for 100 steps.",
        "",
        "## Selected Samples",
        "",
        "| sample | dataset | idx | old loss3 e1500 20-step increase |",
        "|---|---|---:|---:|",
    ]
    for item in manifest:
        lines.append(
            f"| {item['sample_id']} | `{item['dataset_id']}` | {item['index']} | "
            f"{float(item['old_loss3_epoch1500_attack_increase_20step']):.6g} |"
        )
    lines.extend(["", "## Latest Attack100 Loss3 vs Random Solver", "", "| sample | loss3 inc | random solver inc | loss3 - random solver | winner |", "|---|---:|---:|---:|---|"])
    for _, row in pair.iterrows():
        lines.append(
            f"| {row['sample_id']} | {row['loss3_attack100_increase']:.6g} | "
            f"{row['random_solver_y_attack100_increase']:.6g} | "
            f"{row['loss3_minus_random_solver_y_attack100_increase']:.6g} | {row.get('winner_lower_increase', '')} |"
        )
    lines.extend(["", "## Model Means On These Samples", "", "| model | mean initial | mean final | mean increase | median increase | wins lower increase |", "|---|---:|---:|---:|---:|---:|"])
    if not df.empty:
        win_count = {}
        for sample_id, group in df.groupby("sample_id", sort=False):
            winner = group.sort_values("attack_loss_increase_mse", ascending=True).iloc[0]["model"]
            win_count[winner] = win_count.get(winner, 0) + 1
        for model, group in df.groupby("model", sort=False):
            lines.append(
                f"| {model} | {group['initial_loss_mse'].mean():.6g} | {group['final_loss_mse'].mean():.6g} | "
                f"{group['attack_loss_increase_mse'].mean():.6g} | {group['attack_loss_increase_mse'].median():.6g} | {win_count.get(model, 0)} |"
            )
    if diagnostic_rows:
        diag = pd.DataFrame(diagnostic_rows)
        lines.extend(["", "## Local Diagnostics Means", "", "| model | clean residual MSE | J^T error norm | J_error delta L2 | spectral norm estimate | attack-delta/top-SV abs cos | attack-delta/J^T-error abs cos |", "|---|---:|---:|---:|---:|---:|---:|"])
        for model, group in diag.groupby("model", sort=False):
            lines.append(
                f"| {model} | {group['clean_residual_mse'].mean():.6g} | "
                f"{group['bias_gradient_norm_jt_error'].mean():.6g} | "
                f"{group['j_error_delta_l2'].mean():.6g} | "
                f"{group['error_spectral_norm_power_iter'].mean():.6g} | "
                f"{group['attack_delta_top_error_right_abs_cos'].mean():.6g} | "
                f"{group['attack_delta_bias_gradient_abs_cos'].mean():.6g} |"
            )
    if corr_rows:
        corr = pd.DataFrame(corr_rows)
        all_corr = corr[corr["scope"] == "all_model_sample_rows"].copy()
        lines.extend(["", "## Correlations With Attack100 Increase", "", "| x metric | n | Pearson r | Spearman r |", "|---|---:|---:|---:|"])
        for _, row in all_corr.iterrows():
            lines.append(
                f"| {row['x_metric']} | {int(row['n'])} | {float(row.get('pearson_r', math.nan)):.4g} | {float(row.get('spearman_r', math.nan)):.4g} |"
            )
    lines.extend(["", "## Figures", ""])
    for p in image_paths:
        lines.append(f"- `{p.relative_to(REPO)}`")
    lines.extend(
        [
            "",
            "## Caveat",
            "",
            "The local singular value/vector here is a power-iteration estimate for the exact selected samples. It is not a full top20/top100 SVD table.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_data_outputs(paths: list[Path]) -> None:
    AUDIT_DATA_ROOT.mkdir(parents=True, exist_ok=True)
    for path in paths:
        if path.exists():
            shutil.copy2(path, AUDIT_DATA_ROOT / path.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-count", type=int, default=6)
    parser.add_argument("--per-dataset", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attack-steps", type=int, default=100)
    parser.add_argument("--frame-every", type=int, default=2)
    parser.add_argument("--epsilon-rms", type=float, default=0.12)
    parser.add_argument("--power-iters", type=int, default=8)
    parser.add_argument("--skip-diagnostics", action="store_true")
    parser.add_argument("--only-render", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_mod = load_module("burgers_outlier_base_attack_20260614", BASE_ATTACK_SCRIPT)
    renderer = load_module("burgers_outlier_renderer_20260614", RENDER_SCRIPT)
    labels = load_module("burgers_outlier_labels_20260614", LABEL_SCRIPT)
    configure_base_attack(base_mod, args.attack_steps, args.frame_every, args.epsilon_rms)

    selected = select_outlier_samples(int(args.sample_count), bool(args.per_dataset))
    x_cpu, manifest = build_x_and_manifest(selected)
    clean = x_cpu.numpy()[..., 0]

    TRACE_ROOT.mkdir(parents=True, exist_ok=True)
    write_json(TRACE_ROOT / "sample_manifest.json", manifest)
    selected.to_csv(TRACE_ROOT / "selected_old4_outlier_samples.csv", index=False)

    trace_npz = TRACE_ROOT / "four_model_attack100_traces.npz"
    if args.only_render:
        with np.load(trace_npz, allow_pickle=False) as z:
            traces = {model_key: {key: np.asarray(z[f"{model_key}_{key}"]) for key in TRACE_KEYS} for model_key in MODEL_ORDER}
    else:
        traces = run_attacks(base_mod, renderer, x_cpu, force=bool(args.force), trace_npz_path=trace_npz)
        save_trace_npz(trace_npz, clean, traces)
        save_loss_curves_csv(TRACE_ROOT / "attack100_loss_curves_four_models.csv", traces, manifest)

    attack_rows = attack_summary_rows(traces, manifest)
    pair_rows = pairwise_loss3_random_rows(attack_rows)
    write_csv(TRACE_ROOT / "attack100_summary_by_sample_model.csv", attack_rows)
    write_csv(TRACE_ROOT / "attack100_loss3_vs_random_solver_by_sample.csv", pair_rows)

    diagnostic_rows: list[dict[str, Any]] = []
    diagnostic_pair_rows: list[dict[str, Any]] = []
    corr_rows: list[dict[str, Any]] = []
    if not args.skip_diagnostics:
        diagnostic_rows, diagnostic_pair_rows, corr_rows = compute_diagnostics(
            base_mod,
            renderer,
            x_cpu,
            traces,
            manifest,
            power_iters=int(args.power_iters),
        )
        write_csv(TRACE_ROOT / "local_jacobian_direction_diagnostics_by_sample_model.csv", diagnostic_rows)
        write_csv(TRACE_ROOT / "local_jacobian_direction_loss3_vs_random_solver_by_sample.csv", diagnostic_pair_rows)
        write_csv(TRACE_ROOT / "local_diagnostic_correlations_with_attack100_increase.csv", corr_rows)
    else:
        write_csv(TRACE_ROOT / "local_jacobian_direction_diagnostics_by_sample_model.csv", [])
        write_csv(TRACE_ROOT / "local_jacobian_direction_loss3_vs_random_solver_by_sample.csv", [])
        write_csv(TRACE_ROOT / "local_diagnostic_correlations_with_attack100_increase.csv", [])

    image_paths = render_figures(renderer, labels, clean, manifest, traces)
    summary = {
        "trace_root": str(TRACE_ROOT),
        "visualization_root": str(VIS_ROOT),
        "audit_figure_root": str(AUDIT_FIG_ROOT),
        "audit_data_root": str(AUDIT_DATA_ROOT),
        "models": MODEL_ORDER,
        "attack_steps": int(args.attack_steps),
        "epsilon_rms": float(args.epsilon_rms),
        "sample_count": int(len(manifest)),
        "selection": "actual_old4_manifest_outliers",
        "diagnostics": "skipped" if args.skip_diagnostics else "power_iteration_jacobian_direction_diagnostics",
        "images": [str(p) for p in image_paths],
    }
    write_json(TRACE_ROOT / "summary.json", summary)

    report_path = TRACE_ROOT / "README.md"
    write_report(report_path, manifest, attack_rows, pair_rows, diagnostic_rows, corr_rows, image_paths)
    docs_report = REPO / "docs/burgers_old4_actual_outlier_attack100_diagnostics_20260614.md"
    shutil.copy2(report_path, docs_report)
    AUDIT_REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(report_path, AUDIT_REPORT_ROOT / "burgers_old4_actual_outlier_attack100_diagnostics_20260614.md")

    copy_data_outputs(
        [
            TRACE_ROOT / "sample_manifest.json",
            TRACE_ROOT / "selected_old4_outlier_samples.csv",
            TRACE_ROOT / "attack100_loss_curves_four_models.csv",
            TRACE_ROOT / "attack100_summary_by_sample_model.csv",
            TRACE_ROOT / "attack100_loss3_vs_random_solver_by_sample.csv",
            TRACE_ROOT / "local_jacobian_direction_diagnostics_by_sample_model.csv",
            TRACE_ROOT / "local_jacobian_direction_loss3_vs_random_solver_by_sample.csv",
            TRACE_ROOT / "local_diagnostic_correlations_with_attack100_increase.csv",
            TRACE_ROOT / "summary.json",
        ]
    )
    print(f"[done] traces: {TRACE_ROOT}", flush=True)
    print(f"[done] figures: {VIS_ROOT}", flush=True)
    print(f"[done] audit figures: {AUDIT_FIG_ROOT}", flush=True)
    print(f"[done] audit data: {AUDIT_DATA_ROOT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
