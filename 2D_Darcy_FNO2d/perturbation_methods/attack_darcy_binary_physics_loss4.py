#!/usr/bin/env python3
"""Darcy binary-coefficient PGD attack with loss4 = physics residual loss.

This script mirrors the existing Darcy binary attack setup for loss1/loss2/loss3,
but optimizes only a fourth surrogate objective:

    loss4(A) = physics residual of FNO(A) for -div(A grad u)=1, u|boundary=0.

The attack itself does not use the numerical solver.  The solver is still called
at each logging step so the trace records loss1/loss2/loss3 and their deltas,
which lets us compare the physics-loss surrogate against the true solver-based
Darcy error.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import sysconfig
import time
from pathlib import Path
from typing import Iterable

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.45")


def _configure_jax_cuda_toolchain() -> None:
    purelib = sysconfig.get_paths().get("purelib")
    if not purelib:
        return
    cuda_nvcc = Path(purelib) / "nvidia" / "cuda_nvcc"
    if not cuda_nvcc.exists():
        return
    os.environ.setdefault("JAX_PLATFORMS", "cuda")
    os.environ.setdefault("XLA_FLAGS", f"--xla_gpu_cuda_data_dir={cuda_nvcc}")
    bin_dir = str(cuda_nvcc / "bin")
    path = os.environ.get("PATH", "")
    if bin_dir not in path.split(os.pathsep):
        os.environ["PATH"] = bin_dir + os.pathsep + path


_configure_jax_cuda_toolchain()

import jax
import numpy as np
import torch

THIS_FILE = Path(__file__).resolve()
DARCY_ROOT = THIS_FILE.parents[1]
PROJECT_ROOT = THIS_FILE.parents[2]
if str(DARCY_ROOT) not in sys.path:
    sys.path.insert(0, str(DARCY_ROOT))

from perturbation_methods.attack_darcy_binary import (  # noqa: E402
    JaxDarcySolver,
    cuda_memory_snapshot,
    ensure_cuda_or_die,
    load_dataset,
    load_model,
    make_valid_mask,
    save_panels,
    write_csv,
)
from perturbation_methods.attack_darcy_binary_loss_method_experiments import (  # noqa: E402
    append_step_sample_trace,
    compute_loss_values,
    per_sample_metric,
    save_step_sample_trace,
    select_top_add_mask,
    select_top_mask,
    stable_name_offset,
    summarize_values,
    update_flip_mask,
)
from solvers.darcy_jax_solver import make_darcy_solve_fn  # noqa: E402


LOSS_NAME = "loss4_physics"


def darcy_matvec_torch(a: torch.Tensor, u_full: torch.Tensor) -> torch.Tensor:
    """Apply the same second-order Darcy operator as solvers/darcy_jax_solver.py.

    Args:
        a: coefficient field, shape (B, N, N)
        u_full: full predicted solution including boundary, shape (B, N, N)

    Returns:
        Interior operator values for -div(a grad u), shape (B, N-2, N-2).
    """

    n = a.shape[-1]
    h = 1.0 / float(n - 1)
    center = u_full[:, 1:-1, 1:-1]
    a_center = a[:, 1:-1, 1:-1]
    a_e = 0.5 * (a_center + a[:, 2:, 1:-1])
    a_w = 0.5 * (a_center + a[:, :-2, 1:-1])
    a_n = 0.5 * (a_center + a[:, 1:-1, 2:])
    a_s = 0.5 * (a_center + a[:, 1:-1, :-2])
    out = (
        (a_e + a_w + a_n + a_s) * center
        - a_e * u_full[:, 2:, 1:-1]
        - a_w * u_full[:, :-2, 1:-1]
        - a_n * u_full[:, 1:-1, 2:]
        - a_s * u_full[:, 1:-1, :-2]
    )
    return out / (h * h)


def boundary_values(u: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [
            u[:, 0, :],
            u[:, -1, :],
            u[:, 1:-1, 0],
            u[:, 1:-1, -1],
        ],
        dim=1,
    )


def per_sample_physics_components(
    a: torch.Tensor,
    pred_u: torch.Tensor,
    *,
    forcing_value: float = 1.0,
    physics_metric: str = "rel_l2",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute PDE residual and homogeneous Dirichlet boundary losses.

    `rel_l2` matches the residual_norm diagnostic style for the PDE term:
    ||Au-rhs||/||rhs||. The boundary term is normalized by sqrt(number of
    boundary points), so it has the scale of an RMS boundary violation.
    """

    op_u = darcy_matvec_torch(a, pred_u)
    rhs = torch.ones_like(op_u) * forcing_value
    residual = op_u - rhs
    residual_flat = residual.reshape(residual.shape[0], -1)
    if physics_metric == "rel_l2":
        rhs_flat = rhs.reshape(rhs.shape[0], -1)
        residual_loss = torch.linalg.vector_norm(residual_flat, ord=2, dim=1) / torch.linalg.vector_norm(
            rhs_flat, ord=2, dim=1
        ).clamp_min(1e-12)
    elif physics_metric == "mse":
        residual_loss = torch.mean(residual_flat.square(), dim=1)
    else:
        raise ValueError(f"unknown physics_metric={physics_metric!r}")

    b = boundary_values(pred_u)
    if physics_metric == "rel_l2":
        bc_loss = torch.linalg.vector_norm(b, ord=2, dim=1) / np.sqrt(max(1, b.shape[1]))
    else:
        bc_loss = torch.mean(b.square(), dim=1)
    return residual_loss, bc_loss


def per_sample_physics_loss(
    a: torch.Tensor,
    pred_u: torch.Tensor,
    *,
    forcing_value: float = 1.0,
    physics_metric: str = "rel_l2",
    bc_weight: float = 1.0,
) -> torch.Tensor:
    pde_loss, bc_loss = per_sample_physics_components(
        a, pred_u, forcing_value=forcing_value, physics_metric=physics_metric
    )
    return pde_loss + float(bc_weight) * bc_loss


def evaluate_no_grad_all(
    *,
    model,
    solver,
    a: torch.Tensor,
    clean_model: torch.Tensor,
    clean_solver: torch.Tensor,
    metric: str,
    physics_metric: str,
    bc_weight: float,
    forcing_value: float,
) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        pred = model(a.unsqueeze(-1)).squeeze(-1)
        solver_adv = JaxDarcySolver.apply(a, solver)
        loss1 = compute_loss_values(
            pred=pred,
            solver_adv=solver_adv,
            clean_model=clean_model,
            clean_solver=clean_solver,
            loss_name="loss1",
            metric=metric,
        ).detach()
        loss2 = compute_loss_values(
            pred=pred,
            solver_adv=solver_adv,
            clean_model=clean_model,
            clean_solver=clean_solver,
            loss_name="loss2",
            metric=metric,
        ).detach()
        loss3 = compute_loss_values(
            pred=pred,
            solver_adv=solver_adv,
            clean_model=clean_model,
            clean_solver=clean_solver,
            loss_name="loss3",
            metric=metric,
        ).detach()
        loss4_pde, loss4_bc = per_sample_physics_components(
            a, pred, forcing_value=forcing_value, physics_metric=physics_metric
        )
        loss4 = (loss4_pde + float(bc_weight) * loss4_bc).detach()
    return {
        "pred": pred.detach(),
        "solver": solver_adv.detach(),
        "loss1": loss1,
        "loss2": loss2,
        "loss3": loss3,
        "loss4": loss4,
        "loss4_pde": loss4_pde.detach(),
        "loss4_bc": loss4_bc.detach(),
    }


def add_loss_summaries(row: dict, current: dict[str, torch.Tensor], clean: dict[str, torch.Tensor]) -> None:
    for name in ("loss1", "loss2", "loss3", "loss4"):
        vals = current[name]
        row.update(summarize_values(name, vals))
        delta = vals - clean[name]
        row.update(summarize_values(f"{name}_increase", delta))


def plot_loss_curves(path: Path, rows: list[dict]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] matplotlib unavailable, skipping loss curves: {exc}", flush=True)
        return

    steps = np.array([float(r["step"]) for r in rows], dtype=np.float64)
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.4), constrained_layout=True)
    axes = axes.ravel()
    for ax, key, title in [
        (axes[0], "loss4_mean", "optimized surrogate: loss4 physics"),
        (axes[1], "loss1_mean", "loss1: model(A)-model(A0)"),
        (axes[2], "loss2_mean", "loss2: model(A)-solver(A0)"),
        (axes[3], "loss3_mean", "loss3 true: model(A)-solver(A)"),
        (axes[4], "flip_count_mean", "binary flips"),
    ]:
        vals = np.array([float(r.get(key, np.nan)) for r in rows], dtype=np.float64)
        ax.plot(steps, vals, marker="o", markersize=2, linewidth=1.4)
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.grid(True, linestyle="--", alpha=0.35)
    axes[5].axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_loss4(args: argparse.Namespace) -> Path:
    ensure_cuda_or_die()
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()

    x_cpu, dataset_meta = load_dataset(args.dataset, args.start, args.num_samples)
    resolution = int(x_cpu.shape[-1])
    model, model_config = load_model(args, resolution, device)
    solver = make_darcy_solve_fn(tol=args.solver_tol, atol=args.solver_atol, maxiter=args.solver_maxiter)

    x0 = x_cpu.to(device, non_blocking=True)
    midpoint = 0.5 * (args.low + args.high)
    x0 = torch.where(x0 >= midpoint, torch.full_like(x0, args.high), torch.full_like(x0, args.low))
    opposite = torch.where(x0 >= midpoint, torch.full_like(x0, args.low), torch.full_like(x0, args.high))
    valid_mask = make_valid_mask(tuple(x0.shape), args.include_boundary_flips, device)
    valid_pixels = int(valid_mask[0].sum().item())
    max_flips = args.epsilon_flips if args.epsilon_flips is not None else int(np.ceil(args.epsilon_fraction * valid_pixels))
    max_flips = max(0, min(max_flips, valid_pixels))
    if args.alpha_flips is not None:
        alpha_flips = args.alpha_flips
    elif args.alpha_fraction is not None:
        alpha_flips = int(np.ceil(args.alpha_fraction * valid_pixels))
    else:
        alpha_flips = int(np.ceil(max_flips / max(1, args.steps)))
    alpha_flips = max(1, min(alpha_flips, max(1, max_flips)))

    with torch.no_grad():
        clean_model = model(x0.unsqueeze(-1)).squeeze(-1).detach()
        clean_solver = JaxDarcySolver.apply(x0, solver).detach()
        clean_loss4_pde, clean_loss4_bc = per_sample_physics_components(
            x0, clean_model, forcing_value=args.forcing_value, physics_metric=args.physics_metric
        )
        clean_eval = {
            "loss1": per_sample_metric(clean_model, clean_model, args.metric).detach(),
            "loss2": per_sample_metric(clean_model, clean_solver, args.metric).detach(),
            "loss3": per_sample_metric(clean_model, clean_solver, args.metric).detach(),
            "loss4_pde": clean_loss4_pde.detach(),
            "loss4_bc": clean_loss4_bc.detach(),
            "loss4": (clean_loss4_pde + float(args.bc_weight) * clean_loss4_bc).detach(),
        }

    timestamp = time.strftime("%Y%m%d_%H%M%S_UTC", time.gmtime())
    run_name = args.run_name or (
        f"darcy_loss4_physics_{args.method}_nx{resolution}_N{x0.shape[0]}_"
        f"eps{args.epsilon_fraction:g}_alpha{args.alpha_fraction if args.alpha_fraction is not None else 'auto'}_"
        f"steps{args.steps}_{timestamp}"
    )
    run_root = (args.output_root / run_name).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    dataset_indices = torch.arange(args.start, args.start + x0.shape[0], dtype=torch.long, device=device)

    flip_mask = torch.zeros_like(x0, dtype=torch.bool)
    score_state = torch.zeros_like(x0)
    trace_rows: list[dict] = []
    step_loss_rows: list[dict] = []
    sample_trace_records: list[dict] = []
    trace_sample_position = int(args.trace_sample_index)
    if trace_sample_position < 0 or trace_sample_position >= x0.shape[0]:
        trace_sample_position = 0
    trace_dataset_index = int(dataset_indices[trace_sample_position].item())
    start_time = time.perf_counter()

    manifest = {
        "run_root": str(run_root),
        "created_utc": timestamp,
        "project_root": str(PROJECT_ROOT),
        "dataset": str(args.dataset),
        "dataset_metadata": dataset_meta,
        "checkpoint": str(args.checkpoint),
        "model": model_config,
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(d) for d in jax.devices()],
        "torch_device": torch.cuda.get_device_name(0),
        "resolution": resolution,
        "num_samples": int(x0.shape[0]),
        "dataset_indices": [int(v.item()) for v in dataset_indices],
        "valid_pixels": valid_pixels,
        "epsilon_fraction": args.epsilon_fraction,
        "epsilon_flips": max_flips,
        "alpha_fraction": args.alpha_fraction,
        "alpha_flips": alpha_flips,
        "steps": args.steps,
        "optimized_loss": LOSS_NAME,
        "metric_for_loss1_loss2_loss3": args.metric,
        "physics_metric": args.physics_metric,
        "bc_weight": args.bc_weight,
        "forcing_value": args.forcing_value,
        "method": args.method,
        "binary_policy": "A is always projected to {low, high}; epsilon/alpha are Hamming flip budgets.",
        "stablepdenet_analogy": "attack objective is physics residual; solver is used only for trace/evaluation losses.",
    }
    (run_root / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for step in range(args.steps):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        step_start = time.perf_counter()
        current_a = torch.where(flip_mask, opposite, x0).detach().requires_grad_(True)
        pred = model(current_a.unsqueeze(-1)).squeeze(-1)
        loss4_pde, loss4_bc = per_sample_physics_components(
            current_a, pred, forcing_value=args.forcing_value, physics_metric=args.physics_metric
        )
        loss4 = loss4_pde + float(args.bc_weight) * loss4_bc
        objective = loss4.mean()
        model.zero_grad(set_to_none=True)
        objective.backward()
        grad = current_a.grad.detach()

        with torch.no_grad():
            solver_adv = JaxDarcySolver.apply(current_a.detach(), solver)
            current_eval = {
                "pred": pred.detach(),
                "solver": solver_adv.detach(),
                "loss1": compute_loss_values(
                    pred=pred.detach(),
                    solver_adv=solver_adv,
                    clean_model=clean_model,
                    clean_solver=clean_solver,
                    loss_name="loss1",
                    metric=args.metric,
                ).detach(),
                "loss2": compute_loss_values(
                    pred=pred.detach(),
                    solver_adv=solver_adv,
                    clean_model=clean_model,
                    clean_solver=clean_solver,
                    loss_name="loss2",
                    metric=args.metric,
                ).detach(),
                "loss3": compute_loss_values(
                    pred=pred.detach(),
                    solver_adv=solver_adv,
                    clean_model=clean_model,
                    clean_solver=clean_solver,
                    loss_name="loss3",
                    metric=args.metric,
                ).detach(),
                "loss4": loss4.detach(),
                "loss4_pde": loss4_pde.detach(),
                "loss4_bc": loss4_bc.detach(),
            }

        flip_mask, score_state, update_stats = update_flip_mask(
            method=args.method,
            flip_mask=flip_mask,
            grad=grad,
            x0=x0,
            opposite=opposite,
            valid_mask=valid_mask,
            max_flips=max_flips,
            alpha_flips=alpha_flips,
            positive_only=args.positive_only,
            score_state=score_state,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        flip_counts = flip_mask.reshape(flip_mask.shape[0], -1).sum(dim=1).detach().float()
        row = {
            "step": step,
            "phase": "pre_update",
            "seconds": time.perf_counter() - step_start,
            "seconds_since_run_start": time.perf_counter() - start_time,
            "loss_name": LOSS_NAME,
            "method": args.method,
            "objective": float(objective.detach().item()),
            "surrogate_mean": float(loss4.detach().mean().item()),
            "true_loss3_mean": float(current_eval["loss3"].mean().item()),
            "flip_count_mean": float(flip_counts.mean().item()),
            "flip_count_min": float(flip_counts.min().item()),
            "flip_count_max": float(flip_counts.max().item()),
            **update_stats,
            **cuda_memory_snapshot(),
        }
        add_loss_summaries(row, current_eval, clean_eval)
        row.update(summarize_values("loss4_pde", current_eval["loss4_pde"]))
        row.update(summarize_values("loss4_bc", current_eval["loss4_bc"]))
        row.update(summarize_values("loss4_pde_increase", current_eval["loss4_pde"] - clean_eval["loss4_pde"]))
        row.update(summarize_values("loss4_bc_increase", current_eval["loss4_bc"] - clean_eval["loss4_bc"]))
        trace_rows.append(row)
        for sample_position in range(x0.shape[0]):
            sample_row = {
                "step": step,
                "phase": "pre_update",
                "sample_position": int(sample_position),
                "dataset_index": int(dataset_indices[sample_position].item()),
                "loss_name": LOSS_NAME,
                "method": args.method,
                "surrogate_loss4_physics": float(loss4[sample_position].detach().float().cpu().item()),
                "flip_count": int(flip_counts[sample_position].item()),
            }
            for name in ("loss1", "loss2", "loss3", "loss4", "loss4_pde", "loss4_bc"):
                sample_row[name] = float(current_eval[name][sample_position].detach().float().cpu().item())
                sample_row[f"{name}_increase"] = float(
                    (current_eval[name][sample_position] - clean_eval[name][sample_position]).detach().float().cpu().item()
                )
            step_loss_rows.append(sample_row)
        if args.trace_sample_index >= 0:
            append_step_sample_trace(
                sample_trace_records,
                step=step,
                phase="pre_update",
                sample_position=trace_sample_position,
                dataset_index=trace_dataset_index,
                x_clean=x0,
                a_state=current_a.detach(),
                flip_mask=flip_mask,
                pred=pred.detach(),
                solver_u=current_eval["solver"],
                surrogate=loss4.detach(),
                true_loss3=current_eval["loss3"],
            )
        print(
            f"[loss4/{args.method} step {step:03d}] "
            f"physics={row['loss4_mean']:.6e} loss1={row['loss1_mean']:.6e} "
            f"loss2={row['loss2_mean']:.6e} loss3={row['loss3_mean']:.6e} "
            f"flips={row['flip_count_mean']:.1f}/{max_flips} sec={row['seconds']:.2f}",
            flush=True,
        )

    final_a = torch.where(flip_mask, opposite, x0).detach()
    final_eval = evaluate_no_grad_all(
        model=model,
        solver=solver,
        a=final_a,
        clean_model=clean_model,
        clean_solver=clean_solver,
        metric=args.metric,
        physics_metric=args.physics_metric,
        bc_weight=args.bc_weight,
        forcing_value=args.forcing_value,
    )
    final_flip_counts = flip_mask.reshape(flip_mask.shape[0], -1).sum(dim=1).detach().float()
    final_row = {
        "step": args.steps,
        "phase": "final",
        "seconds": 0.0,
        "seconds_since_run_start": time.perf_counter() - start_time,
        "loss_name": LOSS_NAME,
        "method": args.method,
        "objective": float(final_eval["loss4"].mean().item()),
        "surrogate_mean": float(final_eval["loss4"].mean().item()),
        "true_loss3_mean": float(final_eval["loss3"].mean().item()),
        "flip_count_mean": float(final_flip_counts.mean().item()),
        "flip_count_min": float(final_flip_counts.min().item()),
        "flip_count_max": float(final_flip_counts.max().item()),
        "selected_mean": 0.0,
        "selected_min": 0,
        "selected_max": 0,
        **cuda_memory_snapshot(),
    }
    add_loss_summaries(final_row, final_eval, clean_eval)
    final_row.update(summarize_values("loss4_pde", final_eval["loss4_pde"]))
    final_row.update(summarize_values("loss4_bc", final_eval["loss4_bc"]))
    final_row.update(summarize_values("loss4_pde_increase", final_eval["loss4_pde"] - clean_eval["loss4_pde"]))
    final_row.update(summarize_values("loss4_bc_increase", final_eval["loss4_bc"] - clean_eval["loss4_bc"]))
    trace_rows.append(final_row)
    for sample_position in range(x0.shape[0]):
        sample_row = {
            "step": args.steps,
            "phase": "final",
            "sample_position": int(sample_position),
            "dataset_index": int(dataset_indices[sample_position].item()),
            "loss_name": LOSS_NAME,
            "method": args.method,
            "surrogate_loss4_physics": float(final_eval["loss4"][sample_position].item()),
            "flip_count": int(final_flip_counts[sample_position].item()),
        }
        for name in ("loss1", "loss2", "loss3", "loss4", "loss4_pde", "loss4_bc"):
            sample_row[name] = float(final_eval[name][sample_position].item())
            sample_row[f"{name}_increase"] = float((final_eval[name][sample_position] - clean_eval[name][sample_position]).item())
        step_loss_rows.append(sample_row)
    if args.trace_sample_index >= 0:
        append_step_sample_trace(
            sample_trace_records,
            step=args.steps,
            phase="final",
            sample_position=trace_sample_position,
            dataset_index=trace_dataset_index,
            x_clean=x0,
            a_state=final_a,
            flip_mask=flip_mask,
            pred=final_eval["pred"],
            solver_u=final_eval["solver"],
            surrogate=final_eval["loss4"],
            true_loss3=final_eval["loss3"],
        )

    write_csv(run_root / "trace.csv", trace_rows)
    write_csv(run_root / "step_losses_per_sample.csv", step_loss_rows)
    save_step_sample_trace(run_root / "step_sample_trace.npz", sample_trace_records)
    write_csv(
        run_root / "final_per_sample.csv",
        [
            {
                "sample_position": int(i),
                "dataset_index": int(dataset_indices[i].item()),
                "flip_count": int(final_flip_counts[i].item()),
                "clean_loss1": float(clean_eval["loss1"][i].item()),
                "clean_loss2": float(clean_eval["loss2"][i].item()),
                "clean_loss3": float(clean_eval["loss3"][i].item()),
                "clean_loss4_physics": float(clean_eval["loss4"][i].item()),
                "clean_loss4_pde": float(clean_eval["loss4_pde"][i].item()),
                "clean_loss4_bc": float(clean_eval["loss4_bc"][i].item()),
                "final_loss1": float(final_eval["loss1"][i].item()),
                "final_loss2": float(final_eval["loss2"][i].item()),
                "final_loss3_true": float(final_eval["loss3"][i].item()),
                "final_loss4_physics": float(final_eval["loss4"][i].item()),
                "final_loss4_pde": float(final_eval["loss4_pde"][i].item()),
                "final_loss4_bc": float(final_eval["loss4_bc"][i].item()),
                "loss1_increase": float((final_eval["loss1"][i] - clean_eval["loss1"][i]).item()),
                "loss2_increase": float((final_eval["loss2"][i] - clean_eval["loss2"][i]).item()),
                "loss3_increase": float((final_eval["loss3"][i] - clean_eval["loss3"][i]).item()),
                "loss4_increase": float((final_eval["loss4"][i] - clean_eval["loss4"][i]).item()),
                "loss4_pde_increase": float((final_eval["loss4_pde"][i] - clean_eval["loss4_pde"][i]).item()),
                "loss4_bc_increase": float((final_eval["loss4_bc"][i] - clean_eval["loss4_bc"][i]).item()),
            }
            for i in range(x0.shape[0])
        ],
    )
    torch.save(
        {
            "dataset_indices": dataset_indices.detach().cpu(),
            "x_clean": x0.detach().cpu(),
            "x_adv": final_a.detach().cpu(),
            "final_perturbation": (final_a - x0).detach().cpu(),
            "flip_mask": flip_mask.detach().cpu(),
            "clean_model_u": clean_model.detach().cpu(),
            "clean_solver_u": clean_solver.detach().cpu(),
            "clean_model_solver_diff": (clean_model - clean_solver).detach().cpu(),
            "adv_model_u": final_eval["pred"].detach().cpu(),
            "adv_solver_u": final_eval["solver"].detach().cpu(),
            "adv_model_solver_diff": (final_eval["pred"] - final_eval["solver"]).detach().cpu(),
            "adv_model_minus_clean_model": (final_eval["pred"] - clean_model).detach().cpu(),
            "adv_solver_minus_clean_solver": (final_eval["solver"] - clean_solver).detach().cpu(),
            "clean_loss1": clean_eval["loss1"].detach().cpu(),
            "clean_loss2": clean_eval["loss2"].detach().cpu(),
            "clean_loss3": clean_eval["loss3"].detach().cpu(),
            "clean_loss4_physics": clean_eval["loss4"].detach().cpu(),
            "clean_loss4_pde": clean_eval["loss4_pde"].detach().cpu(),
            "clean_loss4_bc": clean_eval["loss4_bc"].detach().cpu(),
            "final_loss1": final_eval["loss1"].detach().cpu(),
            "final_loss2": final_eval["loss2"].detach().cpu(),
            "final_loss3": final_eval["loss3"].detach().cpu(),
            "final_loss4_physics": final_eval["loss4"].detach().cpu(),
            "final_loss4_pde": final_eval["loss4_pde"].detach().cpu(),
            "final_loss4_bc": final_eval["loss4_bc"].detach().cpu(),
            "score_state": score_state.detach().cpu(),
        },
        run_root / "final_outputs.pt",
    )
    save_panels(
        run_root / "figures" / "final_panels.png",
        {
            "A clean": x0,
            "flip mask": flip_mask.float(),
            "A adv": final_a,
            "solver U adv": final_eval["solver"],
            "model U adv": final_eval["pred"],
            "model-solver": final_eval["pred"] - final_eval["solver"],
        },
        args.plot_samples,
    )
    plot_loss_curves(run_root / "figures" / "loss_curves.png", trace_rows)

    summary = {
        "run_root": str(run_root),
        "loss_name": LOSS_NAME,
        "method": args.method,
        "metric": args.metric,
        "physics_metric": args.physics_metric,
        "bc_weight": args.bc_weight,
        "steps": args.steps,
        "num_samples": int(x0.shape[0]),
        "max_flips": max_flips,
        "alpha_flips": alpha_flips,
        "clean_loss1_mean": float(clean_eval["loss1"].mean().item()),
        "clean_loss2_mean": float(clean_eval["loss2"].mean().item()),
        "clean_loss3_mean": float(clean_eval["loss3"].mean().item()),
        "clean_loss4_physics_mean": float(clean_eval["loss4"].mean().item()),
        "clean_loss4_pde_mean": float(clean_eval["loss4_pde"].mean().item()),
        "clean_loss4_bc_mean": float(clean_eval["loss4_bc"].mean().item()),
        "final_loss1_mean": float(final_eval["loss1"].mean().item()),
        "final_loss2_mean": float(final_eval["loss2"].mean().item()),
        "final_loss3_true_mean": float(final_eval["loss3"].mean().item()),
        "final_loss4_physics_mean": float(final_eval["loss4"].mean().item()),
        "final_loss4_pde_mean": float(final_eval["loss4_pde"].mean().item()),
        "final_loss4_bc_mean": float(final_eval["loss4_bc"].mean().item()),
        "loss1_increase_mean": float((final_eval["loss1"] - clean_eval["loss1"]).mean().item()),
        "loss2_increase_mean": float((final_eval["loss2"] - clean_eval["loss2"]).mean().item()),
        "loss3_increase_mean": float((final_eval["loss3"] - clean_eval["loss3"]).mean().item()),
        "loss4_increase_mean": float((final_eval["loss4"] - clean_eval["loss4"]).mean().item()),
        "loss4_pde_increase_mean": float((final_eval["loss4_pde"] - clean_eval["loss4_pde"]).mean().item()),
        "loss4_bc_increase_mean": float((final_eval["loss4_bc"] - clean_eval["loss4_bc"]).mean().item()),
        "final_flip_count_mean": float(final_flip_counts.mean().item()),
        "seconds": time.perf_counter() - start_time,
    }
    (run_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    manifest["completed_utc"] = time.strftime("%Y%m%d_%H%M%S_UTC", time.gmtime())
    manifest["summary"] = summary
    (run_root / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return run_root


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DARCY_ROOT
        / "datasets"
        / "grf_darcy_20260528_N1500"
        / "test"
        / "dim2d_darcy_nx85_N300_solver=jaxcg_solve421_alpha2_tau3_binary3-12_f1_seed45_test.pt",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DARCY_ROOT / "saved_models" / "2D" / "darcy_N1500_nx85_m64_w60_e500_20260528" / "best.pt",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--method", choices=["pgd", "raw_add", "raw_replace", "steepest_add", "steepest_replace"], default="steepest_replace")
    parser.add_argument("--metric", choices=["rel_l2", "mse"], default="rel_l2", help="Metric used to record loss1/loss2/loss3.")
    parser.add_argument("--physics-metric", choices=["rel_l2", "mse"], default="rel_l2", help="Metric optimized by loss4 physics residual.")
    parser.add_argument("--bc-weight", type=float, default=1.0, help="Homogeneous Dirichlet boundary penalty weight added to loss4; default includes boundary condition.")
    parser.add_argument("--forcing-value", type=float, default=1.0)
    parser.add_argument("--epsilon", "--epsilon-fraction", dest="epsilon_fraction", type=float, default=0.01)
    parser.add_argument("--epsilon-flips", type=int, default=None)
    parser.add_argument("--alpha", "--alpha-fraction", dest="alpha_fraction", type=float, default=None)
    parser.add_argument("--alpha-flips", type=int, default=None)
    parser.add_argument("--positive-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-boundary-flips", action="store_true")
    parser.add_argument("--low", type=float, default=3.0)
    parser.add_argument("--high", type=float, default=12.0)
    parser.add_argument("--solver-tol", type=float, default=1e-5)
    parser.add_argument("--solver-atol", type=float, default=0.0)
    parser.add_argument("--solver-maxiter", type=int, default=None)
    parser.add_argument("--modes", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--padding", type=int, default=None)
    parser.add_argument("--plot-samples", type=int, default=4)
    parser.add_argument("--trace-sample-index", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=DARCY_ROOT / "perturbation_results" / "binary_loss4_physics")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)
    args.dataset = args.dataset.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.output_root = args.output_root.resolve()
    run_loss4(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
