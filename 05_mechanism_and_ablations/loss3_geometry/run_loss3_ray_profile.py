#!/usr/bin/env python3
"""Experiment 4: Loss3 ray profile / local-to-global profile.

Scope: FNO / 1D Burgers / nu=0.001 only, GPU-only.

This experiment tests whether directions that look good locally remain good at
finite radius. It constructs several direction sources, normalizes each to a
unit L2 direction v, then evaluates x + r v for r in [0, epsilon].

Recorded curves:
  - endpoint residual norm: ||e(x+r v)||_2
  - norm growth ratio: (||e(x+r v)||_2 - ||e(x)||_2) / (r + eta)
  - residual increment ratio: ||e(x+r v)-e(x)||_2 / (r + eta)

The experiment intentionally distinguishes residual-field movement from outward
clean-residual growth and from finite-radius endpoint loss3_original.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("DDE_BACKEND", "pytorch")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.attack_framework_matrix import (  # noqa: E402
    DEFAULT_BURGERS_MODEL_DIR,
    DEFAULT_BURGERS_TEST,
    load_burgers_torch_model,
    make_burgers_jax_solver,
    make_jax_torch_bridge,
    sync_torch,
)
from tools.run_loss3_small_epsilon_sweep import (  # noqa: E402
    configure_runtime,
    finite_json,
    load_outward_direction,
    load_svd,
    normalize_l2,
    require_gpu_runtime,
)

DEFAULT_JACOBIAN_ROOT = PROJECT_ROOT / "forensics" / "fno_solver_jacobian_similarity_20260514_raw_recomputed"
DEFAULT_OUTWARD_ROOT = PROJECT_ROOT / "forensics" / "outward_growth_direction_20260515" / "fno_nu0p001"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "forensics" / "loss3_ray_profile_20260516" / "fno_nu0p001_gpu_v100"
DEFAULT_RESULT_DOC = PROJECT_ROOT / "docs" / "loss3_ray_profile_fno_nu0p001_gpu_result_20260516.md"
DEFAULT_PLAN_DOC = PROJECT_ROOT / "docs" / "loss3_ray_profile_fno_nu0p001_gpu_plan_20260516.md"
EPS = 1e-12
ATTACK_SOURCES = (
    "loss3_original",
    "loss3_increment_ratio",
    "loss3_residual_increment_ratio",
    "loss3_regularized",
)
LOCAL_SOURCES = ("local_error_svd", "local_outward_growth", "random")
ALL_SOURCES = ATTACK_SOURCES + LOCAL_SOURCES


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(finite_json(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
        writer.writerows(rows)


def finite_summary(values: list[float]) -> dict[str, float]:
    arr = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return {"mean": math.nan, "std": math.nan, "min": math.nan, "max": math.nan}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def norm_l2_np(x: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(x, dtype=np.float64).reshape(-1)))


def batch_l2(x: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(x.reshape(x.shape[0], -1), ord=2, dim=1)


def batch_l2_np(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).reshape(x.shape[0], -1)
    return np.linalg.norm(arr, axis=1)


def normalize_batch_l2(delta: torch.Tensor) -> torch.Tensor:
    flat = delta.reshape(delta.shape[0], -1)
    norms = torch.linalg.vector_norm(flat, ord=2, dim=1, keepdim=True).clamp_min(EPS)
    return (flat / norms).reshape_as(delta)


def project_l2(delta: torch.Tensor, epsilon: float) -> torch.Tensor:
    flat = delta.reshape(delta.shape[0], -1)
    norms = torch.linalg.vector_norm(flat, ord=2, dim=1, keepdim=True).clamp_min(EPS)
    scale = torch.clamp(float(epsilon) / norms, max=1.0)
    return (flat * scale).reshape_as(delta)


def sanitize(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def cosine_np(a: np.ndarray, b: np.ndarray) -> float:
    af = np.asarray(a, dtype=np.float64).reshape(-1)
    bf = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = np.linalg.norm(af) * np.linalg.norm(bf)
    if denom <= EPS:
        return math.nan
    return float(np.dot(af, bf) / denom)


def make_solver_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        burgers_nx=args.burgers_nx,
        burgers_nu=args.burgers_nu,
        burgers_t_final=args.burgers_t_final,
        burgers_dt=args.burgers_dt,
        burgers_domain=args.burgers_domain,
        burgers_jax_solver_dtype=args.burgers_jax_solver_dtype,
    )


def load_burgers_indices(path: Path, indices: list[int]) -> np.ndarray:
    data = torch.load(path, map_location="cpu", weights_only=False)
    x = data["x"][indices].float().numpy()
    return x[..., None].astype(np.float32)


def evaluate_state(model: torch.nn.Module, bridge: Any, solver_fn: Any, x: torch.Tensor, *, allow_solver_grad: bool, label: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    f = model(x)
    j = bridge(x, solver_fn, label)
    if not allow_solver_grad:
        j = j.detach()
    e = f - j
    return f, j, e


def raw_quantities(
    *,
    model: torch.nn.Module,
    bridge: Any,
    solver_fn: Any,
    x0: torch.Tensor,
    f0: torch.Tensor,
    j0: torch.Tensor,
    e0: torch.Tensor,
    delta: torch.Tensor,
    eta: float,
    allow_solver_grad: bool,
    label: str,
) -> dict[str, torch.Tensor]:
    x_adv = x0 + delta
    f_adv, j_adv, e_adv = evaluate_state(model, bridge, solver_fn, x_adv, allow_solver_grad=allow_solver_grad, label=label)
    delta_norm = batch_l2(delta)
    loss1 = batch_l2(f_adv - f0)
    loss2 = batch_l2(f_adv - j0)
    loss3 = batch_l2(e_adv)
    clean_loss3 = batch_l2(e0)
    residual_increment = batch_l2(e_adv - e0)
    solver_movement = batch_l2(j_adv - j0)
    norm_growth = loss3 - clean_loss3
    return {
        "loss1": loss1,
        "loss2": loss2,
        "loss3": loss3,
        "clean_loss3": clean_loss3,
        "norm_growth": norm_growth,
        "norm_growth_ratio": norm_growth / (delta_norm + eta),
        "residual_increment": residual_increment,
        "residual_increment_ratio": residual_increment / (delta_norm + eta),
        "model_movement": loss1,
        "model_movement_ratio": loss1 / (delta_norm + eta),
        "solver_movement": solver_movement,
        "solver_movement_ratio": solver_movement / (delta_norm + eta),
        "delta_norm": delta_norm,
        "f_adv": f_adv,
        "j_adv": j_adv,
        "e_adv": e_adv,
    }


def objective_from_quantities(q: dict[str, torch.Tensor], source: str, regularization_c: float) -> torch.Tensor:
    if source == "loss3_original":
        return q["loss3"]
    if source == "loss3_increment_ratio":
        return q["norm_growth_ratio"]
    if source == "loss3_residual_increment_ratio":
        return q["residual_increment_ratio"]
    if source == "loss3_regularized":
        return q["loss3"] - float(regularization_c) * q["delta_norm"]
    raise ValueError(source)


def tensor_from_vectors(vectors: np.ndarray, device: torch.device) -> torch.Tensor:
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[..., None]
    return torch.as_tensor(arr, device=device, dtype=torch.float32)


def build_local_direction_arrays(
    *,
    indices: list[int],
    jacobian_root: Path,
    outward_root: Path,
    seed: int,
    nx: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    rng = np.random.default_rng(seed + 404)
    dirs: dict[str, list[np.ndarray]] = {key: [] for key in LOCAL_SOURCES}
    rows: list[dict[str, Any]] = []
    for index in indices:
        svd_error = load_svd(jacobian_root, index, "error")
        v_error = normalize_l2(np.asarray(svd_error["right_singular_vectors"])[0])
        outward = load_outward_direction(outward_root, index)
        if outward is None or norm_l2_np(outward) <= EPS:
            outward = np.zeros_like(v_error)
        v_rand = normalize_l2(rng.normal(size=nx))
        dirs["local_error_svd"].append(v_error)
        dirs["local_outward_growth"].append(outward)
        dirs["random"].append(v_rand)
        rows.append(
            {
                "sample_index": index,
                "error_svd_source": str(svd_error["source_path"]),
                "outward_available": bool(norm_l2_np(outward) > EPS),
                "cos_error_svd_outward": cosine_np(v_error, outward),
            }
        )
    stacked = {key: np.stack(value, axis=0).astype(np.float32) for key, value in dirs.items()}
    return stacked, rows


def initial_delta_for_source(source: str, local_dirs: dict[str, np.ndarray], device: torch.device, initial_radius: float) -> tuple[torch.Tensor, str]:
    if source == "loss3_residual_increment_ratio":
        arr = local_dirs["local_error_svd"]
        note = "initial_radius_times_local_error_svd"
    elif source in {"loss3_original", "loss3_increment_ratio", "loss3_regularized"}:
        arr = local_dirs["local_outward_growth"]
        note = "initial_radius_times_local_outward_growth"
    else:
        arr = local_dirs["random"]
        note = "initial_radius_times_random"
    v = tensor_from_vectors(arr, device)
    return float(initial_radius) * v, note


def optimize_attack_direction(
    *,
    source: str,
    model: torch.nn.Module,
    bridge: Any,
    solver_fn: Any,
    x0: torch.Tensor,
    f0: torch.Tensor,
    j0: torch.Tensor,
    e0: torch.Tensor,
    local_dirs: dict[str, np.ndarray],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, list[dict[str, Any]], str]:
    init_delta, init_note = initial_delta_for_source(source, local_dirs, device, min(args.initial_radius, args.epsilon))
    delta = torch.nn.Parameter(project_l2(init_delta.detach().clone(), args.epsilon))
    opt = torch.optim.Adam([delta], lr=args.attack_lr)
    rows: list[dict[str, Any]] = []
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for step in range(args.attack_steps + 1):
        with torch.no_grad():
            delta.copy_(project_l2(sanitize(delta), args.epsilon))
        q = raw_quantities(
            model=model,
            bridge=bridge,
            solver_fn=solver_fn,
            x0=x0,
            f0=f0,
            j0=j0,
            e0=e0,
            delta=delta,
            eta=args.eta,
            allow_solver_grad=True,
            label=f"opt_{source}_step{step}",
        )
        obj = objective_from_quantities(q, source, args.regularization_c)
        if step % args.save_every == 0 or step == args.attack_steps:
            rows.append(
                {
                    "direction_source": source,
                    "step": step,
                    "objective_mean": float(obj.detach().mean().cpu()),
                    "objective_min": float(obj.detach().min().cpu()),
                    "objective_max": float(obj.detach().max().cpu()),
                    "loss3_mean": float(q["loss3"].detach().mean().cpu()),
                    "norm_growth_ratio_mean": float(q["norm_growth_ratio"].detach().mean().cpu()),
                    "residual_increment_ratio_mean": float(q["residual_increment_ratio"].detach().mean().cpu()),
                    "delta_norm_mean": float(q["delta_norm"].detach().mean().cpu()),
                    "seconds_since_source_start": float(time.perf_counter() - start),
                }
            )
        if step >= args.attack_steps:
            break
        loss = -torch.nan_to_num(obj, nan=0.0, posinf=0.0, neginf=0.0).sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if delta.grad is not None:
            delta.grad = sanitize(delta.grad)
        opt.step()
    with torch.no_grad():
        final_delta = project_l2(sanitize(delta.detach()), args.epsilon)
    sync_torch(torch, device)
    return final_delta, rows, init_note


def make_radii(epsilon: float, small_radius: float, num_linear: int) -> np.ndarray:
    base = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, small_radius]
    linear = np.linspace(0.0, float(epsilon), int(num_linear))
    radii = np.unique(np.asarray([r for r in [*base, *linear] if 0.0 <= r <= epsilon + 1e-12], dtype=np.float64))
    return np.sort(radii)


def profile_direction(
    *,
    source: str,
    direction: torch.Tensor,
    direction_kind: str,
    model: torch.nn.Module,
    bridge: Any,
    solver_fn: Any,
    x0: torch.Tensor,
    f0: torch.Tensor,
    j0: torch.Tensor,
    e0: torch.Tensor,
    sample_indices: list[int],
    radii: np.ndarray,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    v_unit = normalize_batch_l2(direction.detach())
    for radius in radii:
        delta = float(radius) * v_unit
        with torch.no_grad():
            q = raw_quantities(
                model=model,
                bridge=bridge,
                solver_fn=solver_fn,
                x0=x0,
                f0=f0,
                j0=j0,
                e0=e0,
                delta=delta,
                eta=args.eta,
                allow_solver_grad=False,
                label=f"profile_{source}_r{radius:g}",
            )
        f_inc = (q["f_adv"] - f0).detach().cpu().numpy().astype(np.float64)
        j_inc = (q["j_adv"] - j0).detach().cpu().numpy().astype(np.float64)
        arrays = {key: value.detach().cpu().numpy().astype(np.float64) for key, value in q.items() if isinstance(value, torch.Tensor) and value.ndim == 1}
        for pos, sample_index in enumerate(sample_indices):
            rows.append(
                {
                    "sample_position": pos,
                    "sample_index": sample_index,
                    "direction_source": source,
                    "direction_kind": direction_kind,
                    "radius": float(radius),
                    "radius_fraction": float(radius / args.epsilon) if args.epsilon > 0 else math.nan,
                    "delta_norm_l2": float(arrays["delta_norm"][pos]),
                    "clean_loss3": float(arrays["clean_loss3"][pos]),
                    "loss3_original": float(arrays["loss3"][pos]),
                    "loss3_norm_growth": float(arrays["norm_growth"][pos]),
                    "norm_growth_ratio": float(arrays["norm_growth_ratio"][pos]),
                    "residual_increment_norm": float(arrays["residual_increment"][pos]),
                    "residual_increment_ratio": float(arrays["residual_increment_ratio"][pos]),
                    "model_movement_norm": float(arrays["model_movement"][pos]),
                    "model_movement_ratio": float(arrays["model_movement_ratio"][pos]),
                    "solver_movement_norm": float(arrays["solver_movement"][pos]),
                    "solver_movement_ratio": float(arrays["solver_movement_ratio"][pos]),
                    "loss1_original": float(arrays["loss1"][pos]),
                    "loss2_original": float(arrays["loss2"][pos]),
                    "cos_delta_f_delta_j": cosine_np(f_inc[pos], j_inc[pos]),
                }
            )
    sync_torch(torch, x0.device)
    return rows


def summarize_profiles(rows: list[dict[str, Any]], epsilon: float, small_radius: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["sample_index"]), str(row["direction_source"]))].append(row)

    per_direction: list[dict[str, Any]] = []
    for (sample_index, source), group in grouped.items():
        group_sorted = sorted(group, key=lambda r: float(r["radius"]))
        endpoint = min(group_sorted, key=lambda r: abs(float(r["radius"]) - epsilon))
        small = min([r for r in group_sorted if float(r["radius"]) > 0], key=lambda r: abs(float(r["radius"]) - small_radius))
        max_row = max(group_sorted, key=lambda r: float(r["loss3_original"]))
        per_direction.append(
            {
                "sample_index": sample_index,
                "direction_source": source,
                "direction_kind": endpoint["direction_kind"],
                "clean_loss3": endpoint["clean_loss3"],
                "small_radius_used": small["radius"],
                "endpoint_radius_used": endpoint["radius"],
                "endpoint_loss3": endpoint["loss3_original"],
                "endpoint_loss3_growth": endpoint["loss3_norm_growth"],
                "endpoint_norm_growth_ratio": endpoint["norm_growth_ratio"],
                "endpoint_residual_increment_norm": endpoint["residual_increment_norm"],
                "endpoint_residual_increment_ratio": endpoint["residual_increment_ratio"],
                "endpoint_model_movement_norm": endpoint["model_movement_norm"],
                "endpoint_solver_movement_norm": endpoint["solver_movement_norm"],
                "small_norm_growth_ratio": small["norm_growth_ratio"],
                "small_residual_increment_ratio": small["residual_increment_ratio"],
                "small_loss3": small["loss3_original"],
                "max_loss3_along_ray": max_row["loss3_original"],
                "argmax_radius": max_row["radius"],
                "endpoint_over_max_loss3": endpoint["loss3_original"] / max_row["loss3_original"] if abs(max_row["loss3_original"]) > EPS else math.nan,
                "endpoint_norm_growth_over_small": endpoint["norm_growth_ratio"] / small["norm_growth_ratio"] if abs(small["norm_growth_ratio"]) > EPS else math.nan,
                "endpoint_residual_increment_over_small": endpoint["residual_increment_ratio"] / small["residual_increment_ratio"] if abs(small["residual_increment_ratio"]) > EPS else math.nan,
            }
        )

    by_sample: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in per_direction:
        by_sample[int(row["sample_index"])].append(row)
    for group in by_sample.values():
        endpoint_sorted = sorted(group, key=lambda r: float(r["endpoint_loss3"]), reverse=True)
        small_growth_sorted = sorted(group, key=lambda r: float(r["small_norm_growth_ratio"]), reverse=True)
        small_resid_sorted = sorted(group, key=lambda r: float(r["small_residual_increment_ratio"]), reverse=True)
        for rank, row in enumerate(endpoint_sorted, start=1):
            row["endpoint_loss3_rank"] = rank
        for rank, row in enumerate(small_growth_sorted, start=1):
            row["small_norm_growth_rank"] = rank
        for rank, row in enumerate(small_resid_sorted, start=1):
            row["small_residual_increment_rank"] = rank

    aggregate: list[dict[str, Any]] = []
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_direction:
        by_source[str(row["direction_source"])].append(row)
    cols = [
        "endpoint_loss3",
        "endpoint_loss3_growth",
        "endpoint_norm_growth_ratio",
        "endpoint_residual_increment_ratio",
        "small_norm_growth_ratio",
        "small_residual_increment_ratio",
        "endpoint_over_max_loss3",
        "endpoint_norm_growth_over_small",
        "endpoint_residual_increment_over_small",
        "endpoint_loss3_rank",
        "small_norm_growth_rank",
        "small_residual_increment_rank",
    ]
    for source, group in sorted(by_source.items()):
        item: dict[str, Any] = {"direction_source": source, "n": len(group)}
        for col in cols:
            stats = finite_summary([float(r[col]) for r in group])
            for name, value in stats.items():
                item[f"{col}_{name}"] = value
        item["endpoint_win_count"] = sum(1 for r in group if int(r["endpoint_loss3_rank"]) == 1)
        item["small_norm_growth_win_count"] = sum(1 for r in group if int(r["small_norm_growth_rank"]) == 1)
        item["small_residual_increment_win_count"] = sum(1 for r in group if int(r["small_residual_increment_rank"]) == 1)
        aggregate.append(item)

    winner_rows: list[dict[str, Any]] = []
    for sample_index, group in sorted(by_sample.items()):
        winner_rows.append(
            {
                "sample_index": sample_index,
                "best_endpoint_loss3_direction": min(group, key=lambda r: int(r["endpoint_loss3_rank"]))["direction_source"],
                "best_small_norm_growth_direction": min(group, key=lambda r: int(r["small_norm_growth_rank"]))["direction_source"],
                "best_small_residual_increment_direction": min(group, key=lambda r: int(r["small_residual_increment_rank"]))["direction_source"],
            }
        )
    return per_direction, aggregate, winner_rows


def aggregate_curve_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["direction_source"]), float(row["radius"]))].append(row)
    cols = ["loss3_original", "norm_growth_ratio", "residual_increment_ratio", "model_movement_ratio", "solver_movement_ratio"]
    out: list[dict[str, Any]] = []
    for (source, radius), group in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        item: dict[str, Any] = {"direction_source": source, "radius": radius, "n": len(group)}
        for col in cols:
            stats = finite_summary([float(r[col]) for r in group])
            item[f"{col}_mean"] = stats["mean"]
            item[f"{col}_std"] = stats["std"]
        out.append(item)
    return out


def plot_profiles(profile_rows: list[dict[str, Any]], aggregate_rows: list[dict[str, Any]], output_dir: Path, sample_indices: list[int]) -> list[Path]:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        "loss3_original": "#1f77b4",
        "loss3_increment_ratio": "#ff7f0e",
        "loss3_residual_increment_ratio": "#2ca02c",
        "loss3_regularized": "#d62728",
        "local_error_svd": "#9467bd",
        "local_outward_growth": "#8c564b",
        "random": "#7f7f7f",
    }
    paths: list[Path] = []
    by_sample_source: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in profile_rows:
        by_sample_source[(int(row["sample_index"]), str(row["direction_source"]))].append(row)

    for sample_index in sample_indices:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
        for source in ALL_SOURCES:
            group = sorted(by_sample_source.get((sample_index, source), []), key=lambda r: float(r["radius"]))
            if not group:
                continue
            xs = [float(r["radius"]) for r in group]
            axes[0].plot(xs, [float(r["loss3_original"]) for r in group], label=source, color=colors.get(source), linewidth=1.8)
            axes[1].plot(xs, [float(r["norm_growth_ratio"]) for r in group], label=source, color=colors.get(source), linewidth=1.8)
            axes[2].plot(xs, [float(r["residual_increment_ratio"]) for r in group], label=source, color=colors.get(source), linewidth=1.8)
        axes[0].set_title(f"index {sample_index}: endpoint residual norm")
        axes[1].set_title("clean residual norm growth ratio")
        axes[2].set_title("residual increment ratio")
        for ax in axes:
            ax.set_xlabel("radius r")
            ax.grid(True, alpha=0.25)
        axes[0].set_ylabel("value")
        axes[0].legend(fontsize=7)
        path = figures_dir / f"ray_profile_index_{sample_index:03d}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)

    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in aggregate_rows:
        by_source[str(row["direction_source"])].append(row)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    for source in ALL_SOURCES:
        group = sorted(by_source.get(source, []), key=lambda r: float(r["radius"]))
        if not group:
            continue
        xs = [float(r["radius"]) for r in group]
        axes[0].plot(xs, [float(r["loss3_original_mean"]) for r in group], label=source, color=colors.get(source), linewidth=1.8)
        axes[1].plot(xs, [float(r["norm_growth_ratio_mean"]) for r in group], label=source, color=colors.get(source), linewidth=1.8)
        axes[2].plot(xs, [float(r["residual_increment_ratio_mean"]) for r in group], label=source, color=colors.get(source), linewidth=1.8)
    axes[0].set_title("mean endpoint residual norm")
    axes[1].set_title("mean clean residual norm growth ratio")
    axes[2].set_title("mean residual increment ratio")
    for ax in axes:
        ax.set_xlabel("radius r")
        ax.grid(True, alpha=0.25)
    axes[0].legend(fontsize=7)
    path = figures_dir / "ray_profile_aggregate_mean.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    paths.append(path)
    return paths


def markdown_table(rows: list[dict[str, Any]], columns: list[str], max_rows: int | None = None) -> str:
    shown = rows if max_rows is None else rows[:max_rows]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in shown:
        vals = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                vals.append(f"{value:.4g}" if math.isfinite(value) else "nan")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_plan_doc(path: Path, args: argparse.Namespace) -> None:
    lines = [
        "# Loss3 Experiment 4 Ray Profile Plan - 2026-05-16",
        "",
        "Scope: FNO / 1D Burgers `nu=0.001`, GPU-only.",
        "",
        "## Purpose",
        "",
        "This experiment checks the local-to-global gap: a direction can look optimal at very small radius but fail to maximize the finite-radius endpoint residual. The diagnostic walks along fixed directions `v` and records the whole curve `x + r v` for `r in [0, epsilon]`.",
        "",
        "## Direction Sources",
        "",
        "- `loss3_original`: direction from finite-radius optimization of `||e(x+delta)||`.",
        "- `loss3_increment_ratio`: direction from finite-radius optimization of `(||e(x+delta)||-||e(x)||)/(||delta||+eta)`.",
        "- `loss3_residual_increment_ratio`: direction from finite-radius optimization of `||e(x+delta)-e(x)||/(||delta||+eta)`.",
        "- `loss3_regularized`: direction from finite-radius optimization of `||e(x+delta)|| - C||delta||`.",
        "- `local_error_svd`: clean local top direction of `J_e`.",
        "- `local_outward_growth`: clean local direction `normalize(J_e^T e/||e||)`.",
        "- `random`: seeded random baseline direction.",
        "",
        "## Curves Recorded",
        "",
        "For each sample, direction, and radius:",
        "",
        "```text",
        "endpoint residual norm    = ||e(x+r v)||_2",
        "norm growth ratio         = (||e(x+r v)||_2 - ||e(x)||_2) / (r + eta)",
        "residual increment ratio  = ||e(x+r v)-e(x)||_2 / (r + eta)",
        "model movement ratio      = ||f(x+r v)-f(x)||_2 / (r + eta)",
        "solver movement ratio     = ||j(x+r v)-j(x)||_2 / (r + eta)",
        "cos_delta_f_delta_j       = cos(f movement, solver movement)",
        "```",
        "",
        "## Settings",
        "",
        f"- Sample indices: `{args.sample_indices}`.",
        f"- Epsilon: `{args.epsilon}`.",
        f"- Attack steps per optimized direction: `{args.attack_steps}`.",
        f"- Attack optimizer: Adam ascent on `delta`, projected to the L2 ball after every step.",
        f"- Ray radii: linear grid with `{args.num_radii}` points plus small radii `1e-4`, `1e-3`, `1e-2`, `1e-1`.",
        "- Official run must use GPU; CPU fallback is refused.",
        "",
        "## Expected Nonlinear Evidence",
        "",
        "- If a local ratio direction has high small-radius ratio but loses at `r=epsilon`, this shows the local objective is not the finite-radius endpoint objective.",
        "- If the winner changes between the small-radius ratio and endpoint `||e(x+epsilon v)||`, this directly demonstrates local-to-global nonlinear drift.",
        "- If curves bend, saturate, or cross, then a single clean-point direction is insufficient to explain the finite-radius attack.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_result_doc(
    path: Path,
    args: argparse.Namespace,
    output_dir: Path,
    aggregate_summary: list[dict[str, Any]],
    winner_rows: list[dict[str, Any]],
    plot_paths: list[Path],
    manifest: dict[str, Any],
) -> None:
    endpoint_counts = Counter(row["best_endpoint_loss3_direction"] for row in winner_rows)
    small_growth_counts = Counter(row["best_small_norm_growth_direction"] for row in winner_rows)
    small_resid_counts = Counter(row["best_small_residual_increment_direction"] for row in winner_rows)
    compact = []
    for row in aggregate_summary:
        compact.append(
            {
                "direction": row["direction_source"],
                "endpoint loss3 mean": row["endpoint_loss3_mean"],
                "small growth mean": row["small_norm_growth_ratio_mean"],
                "small resid-ratio mean": row["small_residual_increment_ratio_mean"],
                "endpoint rank mean": row["endpoint_loss3_rank_mean"],
                "endpoint wins": row["endpoint_win_count"],
                "small growth wins": row["small_norm_growth_win_count"],
                "small resid wins": row["small_residual_increment_win_count"],
            }
        )
    compact = sorted(compact, key=lambda r: float(r["endpoint rank mean"]))
    lines = [
        "# Loss3 Experiment 4 Ray Profile Result - 2026-05-16",
        "",
        "Status: completed for FNO / 1D Burgers `nu=0.001` with GPU-only execution.",
        "",
        "## Scope And Settings",
        "",
        f"- Samples: `{args.sample_indices}`.",
        f"- Epsilon: `{args.epsilon}`.",
        f"- Attack steps per optimized direction: `{args.attack_steps}`.",
        f"- Attack learning rate: `{args.attack_lr}`.",
        f"- Regularization C: `{args.regularization_c}`.",
        f"- Output directory: `{output_dir}`.",
        f"- Runtime device: `{manifest.get('gpu_runtime', {}).get('torch_device_name', 'not recorded')}`.",
        "",
        "## What Was Measured",
        "",
        "For each direction `v`, this run evaluates `x + r v` for radii from `0` to `epsilon` and records endpoint residual norm, clean-residual norm growth ratio, residual increment ratio, model/solver movement ratios, and model-solver movement cosine.",
        "",
        "The key comparison is local versus finite-radius behavior: the best small-radius ratio direction need not be the best endpoint `loss3_original` direction at `r=epsilon`.",
        "",
        "## Aggregate Direction Summary",
        "",
        markdown_table(compact, ["direction", "endpoint loss3 mean", "small growth mean", "small resid-ratio mean", "endpoint rank mean", "endpoint wins", "small growth wins", "small resid wins"]),
        "",
        "## Winner Counts",
        "",
        f"- Best finite-radius endpoint `loss3_original`: `{dict(endpoint_counts)}`.",
        f"- Best small-radius clean residual norm growth ratio: `{dict(small_growth_counts)}`.",
        f"- Best small-radius residual increment ratio: `{dict(small_resid_counts)}`.",
        "",
        "## Visualizations",
        "",
    ]
    for plot in plot_paths:
        rel = os.path.relpath(plot, path.parent)
        lines.append(f"![{plot.stem}]({rel})")
        lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            "This experiment is the finite-radius companion to the small-epsilon sweep. The small-epsilon experiment showed that ratio diagnostics are meaningful local objects; this ray-profile experiment asks whether those local objects remain endpoint-best over a large radius.",
            "",
            "The raw evidence to inspect is:",
            "",
            "- `ray_profile.csv`: every sample, direction, and radius.",
            "- `ray_direction_summary.csv`: endpoint and small-radius summaries per sample/direction.",
            "- `ray_direction_aggregate.csv`: aggregate means/ranks/win counts by direction.",
            "- `ray_winner_summary.csv`: per-sample winners for endpoint and small-radius criteria.",
            "- `attack_trace.csv`: optimization traces for regenerated finite-radius directions.",
            "- `directions.npz`: normalized directions used by the ray profiles.",
            "- `manifest.json`: GPU/runtime/source metadata.",
            "",
            "Conclusion should be read through winner changes and curve crossings: if the small-radius ratio winners differ from endpoint winners, the experiment demonstrates the nonlinear local-to-global gap that Experiment 4 was designed to expose.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--result-doc", type=Path, default=DEFAULT_RESULT_DOC)
    parser.add_argument("--plan-doc", type=Path, default=DEFAULT_PLAN_DOC)
    parser.add_argument("--sample-indices", nargs="+", type=int, default=[0, 7, 40, 47, 115])
    parser.add_argument("--epsilon", type=float, default=8.0)
    parser.add_argument("--attack-steps", type=int, default=50)
    parser.add_argument("--attack-lr", type=float, default=0.3)
    parser.add_argument("--initial-radius", type=float, default=1e-3)
    parser.add_argument("--regularization-c", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=1e-6)
    parser.add_argument("--num-radii", type=int, default=41)
    parser.add_argument("--small-radius", type=float, default=1e-2)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260516)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--runtime-workarounds", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prepend-env-ptxas", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fno-test-path", type=Path, default=DEFAULT_BURGERS_TEST)
    parser.add_argument("--fno-checkpoint", type=Path, default=DEFAULT_BURGERS_MODEL_DIR / "checkpoints" / "pytorch_fno1d_500.pt")
    parser.add_argument("--jacobian-root", type=Path, default=DEFAULT_JACOBIAN_ROOT)
    parser.add_argument("--outward-root", type=Path, default=DEFAULT_OUTWARD_ROOT)
    parser.add_argument("--burgers-nx", type=int, default=1024)
    parser.add_argument("--burgers-nu", type=float, default=0.001)
    parser.add_argument("--burgers-t-final", type=float, default=1.0)
    parser.add_argument("--burgers-dt", type=float, default=0.001)
    parser.add_argument("--burgers-domain", type=float, default=2.0)
    parser.add_argument("--burgers-jax-solver-dtype", choices=["float32", "float64"], default="float64")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if abs(float(args.burgers_nu) - 0.001) > 1e-12:
        raise ValueError("This experiment is intentionally scoped to FNO nu=0.001.")
    configure_runtime(args)
    device, gpu_runtime = require_gpu_runtime(str(args.device or "cuda"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_plan_doc(args.plan_doc, args)

    start = time.perf_counter()
    sample_indices = [int(x) for x in args.sample_indices]
    x_np = load_burgers_indices(args.fno_test_path, sample_indices)
    x0 = torch.as_tensor(x_np, device=device, dtype=torch.float32)
    model = load_burgers_torch_model(args.fno_checkpoint, device)
    solver_fn = make_burgers_jax_solver(make_solver_args(args))
    bridge = make_jax_torch_bridge()

    with torch.no_grad():
        f0, j0, e0 = evaluate_state(model, bridge, solver_fn, x0, allow_solver_grad=False, label="clean")
    sync_torch(torch, device)
    nx = int(x0.shape[1])
    local_dirs_np, local_direction_rows = build_local_direction_arrays(
        indices=sample_indices,
        jacobian_root=args.jacobian_root,
        outward_root=args.outward_root,
        seed=args.seed,
        nx=nx,
    )
    write_csv(args.output_dir / "local_direction_sources.csv", local_direction_rows)

    attack_trace_rows: list[dict[str, Any]] = []
    direction_tensors: dict[str, torch.Tensor] = {}
    direction_kind: dict[str, str] = {}
    init_notes: dict[str, str] = {}

    for source in ATTACK_SOURCES:
        print(f"[optimize] {source}", flush=True)
        final_delta, trace_rows, init_note = optimize_attack_direction(
            source=source,
            model=model,
            bridge=bridge,
            solver_fn=solver_fn,
            x0=x0,
            f0=f0,
            j0=j0,
            e0=e0,
            local_dirs=local_dirs_np,
            args=args,
            device=device,
        )
        v = normalize_batch_l2(final_delta)
        direction_tensors[source] = v
        direction_kind[source] = "finite_radius_optimized"
        init_notes[source] = init_note
        attack_trace_rows.extend(trace_rows)
        np.save(args.output_dir / f"direction_{source}.npy", v.detach().cpu().numpy().astype(np.float32))

    for source in LOCAL_SOURCES:
        v = tensor_from_vectors(local_dirs_np[source], device)
        direction_tensors[source] = normalize_batch_l2(v)
        direction_kind[source] = "local_or_random_reference"
        init_notes[source] = "not_optimized"
        np.save(args.output_dir / f"direction_{source}.npy", direction_tensors[source].detach().cpu().numpy().astype(np.float32))

    np.savez_compressed(
        args.output_dir / "directions.npz",
        **{source: direction_tensors[source].detach().cpu().numpy().astype(np.float32) for source in ALL_SOURCES},
        sample_indices=np.asarray(sample_indices, dtype=np.int64),
    )
    write_csv(args.output_dir / "attack_trace.csv", attack_trace_rows)

    radii = make_radii(args.epsilon, args.small_radius, args.num_radii)
    profile_rows: list[dict[str, Any]] = []
    for source in ALL_SOURCES:
        print(f"[profile] {source}", flush=True)
        profile_rows.extend(
            profile_direction(
                source=source,
                direction=direction_tensors[source],
                direction_kind=direction_kind[source],
                model=model,
                bridge=bridge,
                solver_fn=solver_fn,
                x0=x0,
                f0=f0,
                j0=j0,
                e0=e0,
                sample_indices=sample_indices,
                radii=radii,
                args=args,
            )
        )
    write_csv(args.output_dir / "ray_profile.csv", profile_rows)
    per_direction, aggregate_summary, winner_rows = summarize_profiles(profile_rows, args.epsilon, args.small_radius)
    curve_aggregate = aggregate_curve_rows(profile_rows)
    write_csv(args.output_dir / "ray_direction_summary.csv", per_direction)
    write_csv(args.output_dir / "ray_direction_aggregate.csv", aggregate_summary)
    write_csv(args.output_dir / "ray_winner_summary.csv", winner_rows)
    write_csv(args.output_dir / "ray_profile_aggregate_curves.csv", curve_aggregate)
    plot_paths = plot_profiles(profile_rows, curve_aggregate, args.output_dir, sample_indices)

    seconds = time.perf_counter() - start
    manifest = {
        "experiment": "loss3_ray_profile_fno_nu0p001",
        "status": "completed",
        "scope": "FNO / 1D Burgers / nu=0.001 only",
        "sample_indices": sample_indices,
        "epsilon": args.epsilon,
        "attack_steps": args.attack_steps,
        "attack_lr": args.attack_lr,
        "regularization_c": args.regularization_c,
        "eta": args.eta,
        "num_radii": int(len(radii)),
        "radii": radii.tolist(),
        "direction_sources": list(ALL_SOURCES),
        "direction_initialization": init_notes,
        "output_dir": str(args.output_dir),
        "result_doc": str(args.result_doc),
        "plan_doc": str(args.plan_doc),
        "fno_test_path": str(args.fno_test_path),
        "fno_checkpoint": str(args.fno_checkpoint),
        "jacobian_root": str(args.jacobian_root),
        "outward_root": str(args.outward_root),
        "seconds": seconds,
        "gpu_runtime": gpu_runtime,
        "output_files": [
            "local_direction_sources.csv",
            "attack_trace.csv",
            "directions.npz",
            "ray_profile.csv",
            "ray_profile_aggregate_curves.csv",
            "ray_direction_summary.csv",
            "ray_direction_aggregate.csv",
            "ray_winner_summary.csv",
            "manifest.json",
            *[str(p.relative_to(args.output_dir)) for p in plot_paths],
        ],
    }
    write_json(args.output_dir / "manifest.json", manifest)
    write_result_doc(args.result_doc, args, args.output_dir, aggregate_summary, winner_rows, plot_paths, manifest)
    print(f"[done] output={args.output_dir} seconds={seconds:.2f}", flush=True)
    print(f"[done] result_doc={args.result_doc}", flush=True)


if __name__ == "__main__":
    main()
