#!/usr/bin/env python3
"""Corrected Experiment 4: Loss3 ray profile / local-to-global profile.

GPU-only scope: FNO / 1D Burgers / nu=0.001.

This corrected version fixes the first ray-profile run in three ways:
  1. loss3_original uses best-over-steps, not the last step.
  2. loss3_original is run from multiple restarts, including the boundary
     directions found by ratio / regularized controls.
  3. local outward growth is computed as the true clean gradient direction
     grad_x ||e(x)||, i.e. the small-radius limit of loss3_increment_ratio.

The ray-profile part is unchanged in spirit: fix a direction v, evaluate
x + r v for r in [0, epsilon], and compare local slopes to endpoint behavior.
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
    require_gpu_runtime,
)
from tools.run_loss3_ray_profile import (  # noqa: E402
    batch_l2,
    cosine_np,
    finite_summary,
    make_radii,
    normalize_batch_l2,
    objective_from_quantities,
    profile_direction,
    project_l2,
    raw_quantities,
    sanitize,
    summarize_profiles,
)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "forensics" / "loss3_ray_profile_corrected_20260516" / "fno_nu0p001_gpu_v100_batch20"
DEFAULT_RESULT_DOC = PROJECT_ROOT / "docs" / "loss3_ray_profile_corrected_fno_nu0p001_gpu_batch20_result_20260516.md"
DEFAULT_PLAN_DOC = PROJECT_ROOT / "docs" / "loss3_ray_profile_corrected_fno_nu0p001_gpu_batch20_plan_20260516.md"
DEFAULT_SAMPLE_INDICES = list(range(17)) + [40, 47, 115]
EPS = 1e-12
CONTROL_SPECS = (
    ("loss3_increment_ratio_pgd_best", "loss3_increment_ratio", ("local_outward_small",)),
    ("loss3_residual_increment_ratio_pgd_best", "loss3_residual_increment_ratio", ("local_residual_small",)),
    ("loss3_regularized_pgd_best", "loss3_regularized", ("local_outward_small",)),
)
LOCAL_SOURCES = ("local_outward_growth", "local_residual_movement", "random")
PROFILE_SOURCES = (
    "loss3_original_pgd_best",
    "loss3_increment_ratio_pgd_best",
    "loss3_residual_increment_ratio_pgd_best",
    "loss3_regularized_pgd_best",
    "local_outward_growth",
    "local_residual_movement",
    "random",
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(finite_json(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
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


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        vals = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                vals.append(f"{value:.4g}" if math.isfinite(value) else "nan")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


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
    return f, j, f - j


def random_unit_like(x: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=x.device)
    generator.manual_seed(int(seed))
    v = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
    return normalize_batch_l2(v)


def compute_local_outward_growth(
    *,
    model: torch.nn.Module,
    bridge: Any,
    solver_fn: Any,
    x0: torch.Tensor,
) -> torch.Tensor:
    x = x0.detach().clone().requires_grad_(True)
    _, _, e = evaluate_state(model, bridge, solver_fn, x, allow_solver_grad=True, label="local_outward_clean")
    loss = batch_l2(e).sum()
    loss.backward()
    if x.grad is None:
        raise RuntimeError("local outward gradient is None")
    return normalize_batch_l2(sanitize(x.grad.detach()))


def optimize_local_residual_movement(
    *,
    model: torch.nn.Module,
    bridge: Any,
    solver_fn: Any,
    x0: torch.Tensor,
    f0: torch.Tensor,
    j0: torch.Tensor,
    e0: torch.Tensor,
    args: argparse.Namespace,
    seed: int,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    best_value = torch.full((x0.shape[0],), -float("inf"), device=x0.device, dtype=x0.dtype)
    best_v = torch.zeros_like(x0)
    for restart in range(args.local_residual_restarts):
        v = torch.nn.Parameter(random_unit_like(x0, seed + 1000 + restart))
        opt = torch.optim.Adam([v], lr=args.local_residual_lr)
        for step in range(args.local_residual_steps + 1):
            with torch.no_grad():
                v.copy_(normalize_batch_l2(sanitize(v)))
            delta = float(args.local_radius) * normalize_batch_l2(v)
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
                label=f"local_residual_restart{restart}_step{step}",
            )
            value = q["residual_increment_ratio"]
            with torch.no_grad():
                mask = value > best_value
                if torch.any(mask):
                    view = mask.view(-1, *([1] * (v.ndim - 1)))
                    best_v = torch.where(view, normalize_batch_l2(v.detach()), best_v)
                    best_value = torch.where(mask, value.detach(), best_value)
            if step % args.save_every == 0 or step == args.local_residual_steps:
                rows.append(
                    {
                        "direction_source": "local_residual_movement",
                        "restart": restart,
                        "step": step,
                        "residual_increment_ratio_mean": float(value.detach().mean().cpu()),
                        "residual_increment_ratio_min": float(value.detach().min().cpu()),
                        "residual_increment_ratio_max": float(value.detach().max().cpu()),
                        "best_residual_increment_ratio_mean": float(best_value.detach().mean().cpu()),
                    }
                )
            if step >= args.local_residual_steps:
                break
            loss = -torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0).sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if v.grad is not None:
                v.grad = sanitize(v.grad)
            opt.step()
    return normalize_batch_l2(best_v.detach()), rows


def make_initial_delta(
    name: str,
    *,
    x0: torch.Tensor,
    local_dirs: dict[str, torch.Tensor],
    control_deltas: dict[str, torch.Tensor],
    args: argparse.Namespace,
    seed: int,
) -> torch.Tensor:
    if name == "zero":
        return torch.zeros_like(x0)
    if name == "local_outward_small":
        return project_l2(float(args.initial_radius) * local_dirs["local_outward_growth"], args.epsilon)
    if name == "local_residual_small":
        return project_l2(float(args.initial_radius) * local_dirs["local_residual_movement"], args.epsilon)
    if name.startswith("random_small_"):
        idx = int(name.rsplit("_", 1)[1])
        return project_l2(float(args.initial_radius) * random_unit_like(x0, seed + 2000 + idx), args.epsilon)
    if name.startswith("random_boundary_"):
        idx = int(name.rsplit("_", 1)[1])
        return project_l2(float(args.epsilon) * random_unit_like(x0, seed + 3000 + idx), args.epsilon)
    if name.startswith("boundary_from:"):
        source = name.split(":", 1)[1]
        if source not in control_deltas:
            raise KeyError(f"Missing control delta for restart {name}")
        return project_l2(float(args.epsilon) * normalize_batch_l2(control_deltas[source]), args.epsilon)
    raise ValueError(f"Unknown restart {name}")


def run_pgd_restart(
    *,
    profile_source: str,
    objective_name: str,
    restart_name: str,
    initial_delta: torch.Tensor,
    model: torch.nn.Module,
    bridge: Any,
    solver_fn: Any,
    x0: torch.Tensor,
    f0: torch.Tensor,
    j0: torch.Tensor,
    e0: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    delta = project_l2(initial_delta.detach().clone(), args.epsilon)
    best_obj = torch.full((x0.shape[0],), -float("inf"), device=x0.device, dtype=x0.dtype)
    best_loss3 = torch.full((x0.shape[0],), -float("inf"), device=x0.device, dtype=x0.dtype)
    best_step = torch.full((x0.shape[0],), -1, device=x0.device, dtype=torch.long)
    best_delta = torch.zeros_like(x0)
    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    for step in range(args.attack_steps + 1):
        delta = project_l2(sanitize(delta.detach()), args.epsilon)
        delta_var = delta.detach().clone().requires_grad_(step < args.attack_steps)
        q = raw_quantities(
            model=model,
            bridge=bridge,
            solver_fn=solver_fn,
            x0=x0,
            f0=f0,
            j0=j0,
            e0=e0,
            delta=delta_var,
            eta=args.eta,
            allow_solver_grad=True,
            label=f"corrected_{profile_source}_{restart_name}_step{step}",
        )
        obj = objective_from_quantities(q, objective_name, args.regularization_c)
        loss3 = q["loss3"]
        with torch.no_grad():
            mask = obj.detach() > best_obj
            if torch.any(mask):
                view = mask.view(-1, *([1] * (delta.ndim - 1)))
                best_delta = torch.where(view, delta_var.detach(), best_delta)
                best_obj = torch.where(mask, obj.detach(), best_obj)
                best_loss3 = torch.where(mask, loss3.detach(), best_loss3)
                best_step = torch.where(mask, torch.full_like(best_step, step), best_step)
        if step % args.save_every == 0 or step == args.attack_steps:
            rows.append(
                {
                    "direction_source": profile_source,
                    "objective_name": objective_name,
                    "restart": restart_name,
                    "step": step,
                    "objective_mean": float(obj.detach().mean().cpu()),
                    "objective_min": float(obj.detach().min().cpu()),
                    "objective_max": float(obj.detach().max().cpu()),
                    "loss3_mean": float(loss3.detach().mean().cpu()),
                    "norm_growth_ratio_mean": float(q["norm_growth_ratio"].detach().mean().cpu()),
                    "residual_increment_ratio_mean": float(q["residual_increment_ratio"].detach().mean().cpu()),
                    "delta_norm_mean": float(q["delta_norm"].detach().mean().cpu()),
                    "best_objective_mean": float(best_obj.detach().mean().cpu()),
                    "best_loss3_at_best_objective_mean": float(best_loss3.detach().mean().cpu()),
                    "seconds_since_restart_start": float(time.perf_counter() - start),
                }
            )
        if step >= args.attack_steps:
            break
        batch_obj = torch.nan_to_num(obj, nan=0.0, posinf=0.0, neginf=0.0).sum()
        batch_obj.backward()
        if delta_var.grad is None:
            raise RuntimeError(f"PGD gradient is None for {profile_source}/{restart_name}/step{step}")
        grad = sanitize(delta_var.grad.detach())
        with torch.no_grad():
            if args.update_mode == "raw":
                direction = grad
            elif args.update_mode == "unit":
                direction = normalize_batch_l2(grad)
            else:
                raise ValueError(args.update_mode)
            delta = project_l2(delta + float(args.attack_lr) * direction, args.epsilon)
    sync_torch(torch, x0.device)
    return best_delta.detach(), best_obj.detach(), best_loss3.detach(), best_step.detach(), rows


def optimize_source(
    *,
    profile_source: str,
    objective_name: str,
    restart_names: list[str],
    model: torch.nn.Module,
    bridge: Any,
    solver_fn: Any,
    x0: torch.Tensor,
    f0: torch.Tensor,
    j0: torch.Tensor,
    e0: torch.Tensor,
    local_dirs: dict[str, torch.Tensor],
    control_deltas: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, list[dict[str, Any]], list[dict[str, Any]]]:
    global_best_obj = torch.full((x0.shape[0],), -float("inf"), device=x0.device, dtype=x0.dtype)
    global_best_loss3 = torch.full((x0.shape[0],), -float("inf"), device=x0.device, dtype=x0.dtype)
    global_best_step = torch.full((x0.shape[0],), -1, device=x0.device, dtype=torch.long)
    global_best_restart = ["" for _ in range(x0.shape[0])]
    global_best_delta = torch.zeros_like(x0)
    trace_rows: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []
    for restart_id, restart_name in enumerate(restart_names):
        print(f"[optimize] {profile_source} restart={restart_name}", flush=True)
        init_delta = make_initial_delta(
            restart_name,
            x0=x0,
            local_dirs=local_dirs,
            control_deltas=control_deltas,
            args=args,
            seed=args.seed + 100 * restart_id,
        )
        best_delta, best_obj, best_loss3, best_step, rows = run_pgd_restart(
            profile_source=profile_source,
            objective_name=objective_name,
            restart_name=restart_name,
            initial_delta=init_delta,
            model=model,
            bridge=bridge,
            solver_fn=solver_fn,
            x0=x0,
            f0=f0,
            j0=j0,
            e0=e0,
            args=args,
        )
        trace_rows.extend(rows)
        with torch.no_grad():
            mask = best_obj > global_best_obj
            if torch.any(mask):
                view = mask.view(-1, *([1] * (x0.ndim - 1)))
                global_best_delta = torch.where(view, best_delta, global_best_delta)
                global_best_loss3 = torch.where(mask, best_loss3, global_best_loss3)
                global_best_step = torch.where(mask, best_step, global_best_step)
                global_best_obj = torch.where(mask, best_obj, global_best_obj)
                for i, use in enumerate(mask.detach().cpu().numpy().astype(bool).tolist()):
                    if use:
                        global_best_restart[i] = restart_name
    best_obj_np = global_best_obj.detach().cpu().numpy()
    best_loss3_np = global_best_loss3.detach().cpu().numpy()
    best_step_np = global_best_step.detach().cpu().numpy()
    norm_np = batch_l2(global_best_delta).detach().cpu().numpy()
    for pos, sample_index in enumerate(args.sample_indices):
        best_rows.append(
            {
                "direction_source": profile_source,
                "objective_name": objective_name,
                "sample_position": pos,
                "sample_index": sample_index,
                "best_restart": global_best_restart[pos],
                "best_step": int(best_step_np[pos]),
                "best_objective": float(best_obj_np[pos]),
                "loss3_at_best_objective": float(best_loss3_np[pos]),
                "best_delta_norm_l2": float(norm_np[pos]),
            }
        )
    return global_best_delta.detach(), trace_rows, best_rows


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
        "loss3_original_pgd_best": "#1f77b4",
        "loss3_increment_ratio_pgd_best": "#ff7f0e",
        "loss3_residual_increment_ratio_pgd_best": "#2ca02c",
        "loss3_regularized_pgd_best": "#d62728",
        "local_outward_growth": "#8c564b",
        "local_residual_movement": "#9467bd",
        "random": "#7f7f7f",
    }
    paths: list[Path] = []
    by_sample_source: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in profile_rows:
        by_sample_source[(int(row["sample_index"]), str(row["direction_source"]))].append(row)
    for sample_index in sample_indices:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
        for source in PROFILE_SOURCES:
            group = sorted(by_sample_source.get((sample_index, source), []), key=lambda r: float(r["radius"]))
            if not group:
                continue
            xs = [float(r["radius"]) for r in group]
            axes[0].plot(xs, [float(r["loss3_original"]) for r in group], label=source, color=colors.get(source), linewidth=1.6)
            axes[1].plot(xs, [float(r["norm_growth_ratio"]) for r in group], label=source, color=colors.get(source), linewidth=1.6)
            axes[2].plot(xs, [float(r["residual_increment_ratio"]) for r in group], label=source, color=colors.get(source), linewidth=1.6)
        axes[0].set_title(f"index {sample_index}: endpoint residual norm")
        axes[1].set_title("clean residual norm growth ratio")
        axes[2].set_title("residual increment ratio")
        for ax in axes:
            ax.set_xlabel("radius r")
            ax.grid(True, alpha=0.25)
        axes[0].set_ylabel("value")
        axes[0].legend(fontsize=6)
        path = figures_dir / f"ray_profile_corrected_index_{sample_index:03d}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in aggregate_rows:
        by_source[str(row["direction_source"])].append(row)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    for source in PROFILE_SOURCES:
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
    path = figures_dir / "ray_profile_corrected_aggregate_mean.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    paths.append(path)
    return paths


def compute_alignment_rows(directions: dict[str, torch.Tensor], sample_indices: list[int]) -> list[dict[str, Any]]:
    pairs = [
        ("loss3_increment_ratio_pgd_best", "local_outward_growth"),
        ("loss3_residual_increment_ratio_pgd_best", "local_residual_movement"),
        ("loss3_original_pgd_best", "local_outward_growth"),
        ("loss3_original_pgd_best", "loss3_increment_ratio_pgd_best"),
        ("loss3_original_pgd_best", "loss3_regularized_pgd_best"),
    ]
    arrays = {k: v.detach().cpu().numpy().astype(np.float64) for k, v in directions.items()}
    rows: list[dict[str, Any]] = []
    for a, b in pairs:
        if a not in arrays or b not in arrays:
            continue
        for pos, sample_index in enumerate(sample_indices):
            c = cosine_np(arrays[a][pos], arrays[b][pos])
            angle = math.degrees(math.acos(max(-1.0, min(1.0, abs(c))))) if math.isfinite(c) else math.nan
            rows.append(
                {
                    "source_a": a,
                    "source_b": b,
                    "sample_position": pos,
                    "sample_index": sample_index,
                    "cosine": c,
                    "abs_cosine": abs(c) if math.isfinite(c) else math.nan,
                    "angle_deg_mod_sign": angle,
                }
            )
    return rows


def write_plan_doc(path: Path, args: argparse.Namespace) -> None:
    lines = [
        "# Corrected Loss3 Experiment 4 Ray Profile Plan - 2026-05-16",
        "",
        "Scope: FNO / 1D Burgers `nu=0.001`, GPU-only, batch size 20.",
        "",
        "This corrected run supersedes the first ray-profile endpoint-winner interpretation. The first run used the final `loss3_original` step, not best-over-steps, and therefore was not a fair endpoint comparison.",
        "",
        "## Corrected Design",
        "",
        "- Keep the ray diagnostic: fix a direction `v`, evaluate `x + r v` for `r in [0, epsilon]`.",
        "- Compute `local_outward_growth = normalize(grad_x ||e(x)||)`, the small-radius limit of `loss3_increment_ratio`.",
        "- Compute a local residual-movement direction by optimizing the small-radius residual increment ratio.",
        "- Recompute finite-radius control directions with PGD: `loss3_increment_ratio`, `loss3_residual_increment_ratio`, and `loss3_regularized`.",
        "- Recompute the main endpoint direction with direct `loss3_original` PGD, best-over-steps, and multiple restarts.",
        "- The `loss3_original` restarts include the boundary-normalized control directions, so the endpoint comparison is fair: direct endpoint optimization gets a chance to improve from every control ray.",
        "",
        "## Settings",
        "",
        f"- Sample indices: `{args.sample_indices}`.",
        f"- Epsilon: `{args.epsilon}`.",
        f"- Attack steps per restart: `{args.attack_steps}`.",
        f"- Attack update: `{args.update_mode}` PGD, learning rate `{args.attack_lr}`.",
        f"- Original-loss restarts: `{args.original_restarts}`.",
        f"- Control directions: `{[spec[0] for spec in CONTROL_SPECS]}`.",
        "- Official run must use GPU; CPU fallback is refused.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_result_doc(
    path: Path,
    args: argparse.Namespace,
    output_dir: Path,
    aggregate_summary: list[dict[str, Any]],
    winner_rows: list[dict[str, Any]],
    alignment_rows: list[dict[str, Any]],
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
    align_summary = []
    by_pair: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in alignment_rows:
        by_pair[(row["source_a"], row["source_b"])].append(float(row["angle_deg_mod_sign"]))
    for (a, b), values in by_pair.items():
        stats = finite_summary(values)
        align_summary.append({"source_a": a, "source_b": b, "mean angle": stats["mean"], "std angle": stats["std"], "max angle": stats["max"]})
    lines = [
        "# Corrected Loss3 Experiment 4 Ray Profile Result - 2026-05-16",
        "",
        "Status: corrected GPU run completed for FNO / 1D Burgers `nu=0.001`, batch size 20.",
        "",
        "## Superseded Point",
        "",
        "The earlier ray-profile endpoint-winner statement is superseded. It used the last `loss3_original` step and did not give direct endpoint optimization a fair best-over-steps / restart comparison.",
        "",
        "## Scope And Settings",
        "",
        f"- Samples: `{args.sample_indices}`.",
        f"- Epsilon: `{args.epsilon}`.",
        f"- Attack steps per restart: `{args.attack_steps}`.",
        f"- Attack learning rate: `{args.attack_lr}`.",
        f"- Output directory: `{output_dir}`.",
        f"- Runtime device: `{manifest.get('gpu_runtime', {}).get('torch_device_name', 'not recorded')}`.",
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
        "## Direction Alignment Diagnostics",
        "",
        markdown_table(align_summary, ["source_a", "source_b", "mean angle", "std angle", "max angle"]),
        "",
        "## Interpretation",
        "",
        "This corrected experiment should be read as a ray diagnostic, not just a winner table. The intended evidence is: local outward growth controls the small-radius norm-growth slope, local residual movement controls the small-radius residual-increment slope, and direct `loss3_original` PGD is the fair finite-radius endpoint attack baseline.",
        "",
        "The direct `loss3_original` direction is computed after the control directions and is restarted from their boundary-normalized rays. Therefore, if a control ray has a high endpoint value, direct endpoint PGD is allowed to start there and improve it. This removes the unfair endpoint comparison in the first run.",
        "",
        "## Raw Artifacts",
        "",
        "- `ray_profile.csv`: every sample, direction, and radius.",
        "- `ray_direction_summary.csv`: endpoint and small-radius summaries per sample/direction.",
        "- `ray_direction_aggregate.csv`: aggregate means/ranks/win counts by direction.",
        "- `ray_winner_summary.csv`: per-sample endpoint and small-radius winners.",
        "- `attack_trace.csv`: PGD trace for every source/restart.",
        "- `attack_best_by_sample.csv`: best step/restart per sample.",
        "- `direction_alignment.csv`: angles between local and finite-radius directions.",
        "- `directions.npz`: normalized directions used by the ray profiles.",
        "- `deltas.npz`: best deltas before ray normalization.",
        "- `manifest.json`: GPU/runtime/source metadata.",
        "",
        "## Visualizations",
        "",
    ]
    for plot in plot_paths:
        rel = os.path.relpath(plot, path.parent)
        lines.append(f"![{plot.stem}]({rel})")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--result-doc", type=Path, default=DEFAULT_RESULT_DOC)
    parser.add_argument("--plan-doc", type=Path, default=DEFAULT_PLAN_DOC)
    parser.add_argument("--sample-indices", nargs="+", type=int, default=DEFAULT_SAMPLE_INDICES)
    parser.add_argument("--epsilon", type=float, default=8.0)
    parser.add_argument("--attack-steps", type=int, default=100)
    parser.add_argument("--attack-lr", type=float, default=0.3)
    parser.add_argument("--update-mode", choices=["raw", "unit"], default="raw")
    parser.add_argument("--initial-radius", type=float, default=1e-3)
    parser.add_argument("--regularization-c", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=1e-6)
    parser.add_argument("--num-radii", type=int, default=41)
    parser.add_argument("--small-radius", type=float, default=1e-2)
    parser.add_argument("--local-radius", type=float, default=1e-3)
    parser.add_argument("--local-residual-steps", type=int, default=24)
    parser.add_argument("--local-residual-restarts", type=int, default=2)
    parser.add_argument("--local-residual-lr", type=float, default=0.2)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260516)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--runtime-workarounds", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prepend-env-ptxas", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fno-test-path", type=Path, default=DEFAULT_BURGERS_TEST)
    parser.add_argument("--fno-checkpoint", type=Path, default=DEFAULT_BURGERS_MODEL_DIR / "checkpoints" / "pytorch_fno1d_500.pt")
    parser.add_argument("--burgers-nx", type=int, default=1024)
    parser.add_argument("--burgers-nu", type=float, default=0.001)
    parser.add_argument("--burgers-t-final", type=float, default=1.0)
    parser.add_argument("--burgers-dt", type=float, default=0.001)
    parser.add_argument("--burgers-domain", type=float, default=2.0)
    parser.add_argument("--burgers-jax-solver-dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument(
        "--original-restarts",
        nargs="+",
        default=[
            "zero",
            "local_outward_small",
            "local_residual_small",
            "random_boundary_00",
            "random_boundary_01",
            "boundary_from:loss3_increment_ratio_pgd_best",
            "boundary_from:loss3_residual_increment_ratio_pgd_best",
            "boundary_from:loss3_regularized_pgd_best",
        ],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if abs(float(args.burgers_nu) - 0.001) > 1e-12:
        raise ValueError("This corrected experiment is scoped to FNO nu=0.001.")
    configure_runtime(args)
    device, gpu_runtime = require_gpu_runtime(str(args.device or "cuda"))
    args.sample_indices = [int(x) for x in args.sample_indices]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_plan_doc(args.plan_doc, args)
    start = time.perf_counter()

    print(f"[load] samples={args.sample_indices}", flush=True)
    x_np = load_burgers_indices(args.fno_test_path, args.sample_indices)
    x0 = torch.as_tensor(x_np, device=device, dtype=torch.float32)
    model = load_burgers_torch_model(args.fno_checkpoint, device)
    solver_fn = make_burgers_jax_solver(make_solver_args(args))
    bridge = make_jax_torch_bridge()
    with torch.no_grad():
        f0, j0, e0 = evaluate_state(model, bridge, solver_fn, x0, allow_solver_grad=False, label="clean")
    sync_torch(torch, device)

    print("[local] outward growth", flush=True)
    local_outward = compute_local_outward_growth(model=model, bridge=bridge, solver_fn=solver_fn, x0=x0)
    print("[local] residual movement", flush=True)
    local_residual, local_trace_rows = optimize_local_residual_movement(
        model=model,
        bridge=bridge,
        solver_fn=solver_fn,
        x0=x0,
        f0=f0,
        j0=j0,
        e0=e0,
        args=args,
        seed=args.seed,
    )
    local_random = random_unit_like(x0, args.seed + 9000)
    local_dirs = {
        "local_outward_growth": local_outward,
        "local_residual_movement": local_residual,
        "random": local_random,
    }
    write_csv(args.output_dir / "local_direction_trace.csv", local_trace_rows)

    control_deltas: dict[str, torch.Tensor] = {}
    all_deltas: dict[str, torch.Tensor] = {}
    attack_trace_rows: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []
    for profile_source, objective_name, restarts in CONTROL_SPECS:
        delta, trace, best = optimize_source(
            profile_source=profile_source,
            objective_name=objective_name,
            restart_names=list(restarts),
            model=model,
            bridge=bridge,
            solver_fn=solver_fn,
            x0=x0,
            f0=f0,
            j0=j0,
            e0=e0,
            local_dirs=local_dirs,
            control_deltas=control_deltas,
            args=args,
        )
        control_deltas[profile_source] = delta.detach()
        all_deltas[profile_source] = delta.detach()
        attack_trace_rows.extend(trace)
        best_rows.extend(best)

    original_delta, trace, best = optimize_source(
        profile_source="loss3_original_pgd_best",
        objective_name="loss3_original",
        restart_names=list(args.original_restarts),
        model=model,
        bridge=bridge,
        solver_fn=solver_fn,
        x0=x0,
        f0=f0,
        j0=j0,
        e0=e0,
        local_dirs=local_dirs,
        control_deltas=control_deltas,
        args=args,
    )
    all_deltas["loss3_original_pgd_best"] = original_delta.detach()
    attack_trace_rows.extend(trace)
    best_rows.extend(best)

    write_csv(args.output_dir / "attack_trace.csv", attack_trace_rows)
    write_csv(args.output_dir / "attack_best_by_sample.csv", best_rows)

    directions: dict[str, torch.Tensor] = {
        "loss3_original_pgd_best": normalize_batch_l2(all_deltas["loss3_original_pgd_best"]),
        "loss3_increment_ratio_pgd_best": normalize_batch_l2(all_deltas["loss3_increment_ratio_pgd_best"]),
        "loss3_residual_increment_ratio_pgd_best": normalize_batch_l2(all_deltas["loss3_residual_increment_ratio_pgd_best"]),
        "loss3_regularized_pgd_best": normalize_batch_l2(all_deltas["loss3_regularized_pgd_best"]),
        "local_outward_growth": local_dirs["local_outward_growth"],
        "local_residual_movement": local_dirs["local_residual_movement"],
        "random": local_dirs["random"],
    }
    for source, direction in directions.items():
        np.save(args.output_dir / f"direction_{source}.npy", direction.detach().cpu().numpy().astype(np.float32))
    np.savez_compressed(
        args.output_dir / "directions.npz",
        **{k: v.detach().cpu().numpy().astype(np.float32) for k, v in directions.items()},
        sample_indices=np.asarray(args.sample_indices, dtype=np.int64),
    )
    np.savez_compressed(
        args.output_dir / "deltas.npz",
        **{k: v.detach().cpu().numpy().astype(np.float32) for k, v in all_deltas.items()},
        sample_indices=np.asarray(args.sample_indices, dtype=np.int64),
    )

    alignment_rows = compute_alignment_rows(directions, args.sample_indices)
    write_csv(args.output_dir / "direction_alignment.csv", alignment_rows)

    radii = make_radii(args.epsilon, args.small_radius, args.num_radii)
    profile_rows: list[dict[str, Any]] = []
    for source in PROFILE_SOURCES:
        print(f"[profile] {source}", flush=True)
        kind = "finite_radius_optimized" if source.startswith("loss3_") else "local_or_random_reference"
        profile_rows.extend(
            profile_direction(
                source=source,
                direction=directions[source],
                direction_kind=kind,
                model=model,
                bridge=bridge,
                solver_fn=solver_fn,
                x0=x0,
                f0=f0,
                j0=j0,
                e0=e0,
                sample_indices=args.sample_indices,
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
    plot_paths = plot_profiles(profile_rows, curve_aggregate, args.output_dir, args.sample_indices)

    seconds = time.perf_counter() - start
    manifest = {
        "experiment": "loss3_ray_profile_corrected_fno_nu0p001_batch20",
        "status": "completed",
        "supersedes": "loss3_ray_profile_fno_nu0p001_gpu_result_20260516 endpoint-winner interpretation",
        "sample_indices": args.sample_indices,
        "epsilon": args.epsilon,
        "attack_steps": args.attack_steps,
        "attack_lr": args.attack_lr,
        "update_mode": args.update_mode,
        "regularization_c": args.regularization_c,
        "eta": args.eta,
        "num_radii": int(len(radii)),
        "radii": radii.tolist(),
        "profile_sources": list(PROFILE_SOURCES),
        "original_restarts": list(args.original_restarts),
        "control_specs": [{"profile_source": a, "objective_name": b, "restarts": list(c)} for a, b, c in CONTROL_SPECS],
        "output_dir": str(args.output_dir),
        "result_doc": str(args.result_doc),
        "plan_doc": str(args.plan_doc),
        "fno_test_path": str(args.fno_test_path),
        "fno_checkpoint": str(args.fno_checkpoint),
        "seconds": seconds,
        "gpu_runtime": gpu_runtime,
        "output_files": [
            "local_direction_trace.csv",
            "attack_trace.csv",
            "attack_best_by_sample.csv",
            "direction_alignment.csv",
            "directions.npz",
            "deltas.npz",
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
    write_result_doc(args.result_doc, args, args.output_dir, aggregate_summary, winner_rows, alignment_rows, plot_paths, manifest)
    print(f"[done] output={args.output_dir} seconds={seconds:.2f}", flush=True)
    print(f"[done] result_doc={args.result_doc}", flush=True)


if __name__ == "__main__":
    main()
