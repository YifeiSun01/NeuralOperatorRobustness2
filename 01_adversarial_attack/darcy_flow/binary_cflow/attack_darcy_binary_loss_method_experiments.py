#!/usr/bin/env python3
"""Darcy binary-coefficient attack experiments.

This runner is for the two Darcy attack comparisons:

1. Optimize loss1/loss2/loss3 separately and always evaluate the final true
   loss as loss3.
2. Fix the optimized objective to loss3 and compare binary update rules.

The coefficient field is always kept binary: each valid pixel is either ``low``
or ``high``.  The attack variable is a flip mask under a Hamming budget, not a
continuous-valued delta.
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
from solvers.darcy_jax_solver import make_darcy_solve_fn  # noqa: E402


LOSS_NAMES = ("loss1", "loss2", "loss3")
METHOD_NAMES = ("pgd", "raw_add", "raw_replace", "steepest_add", "steepest_replace")


def per_sample_rel_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_flat = pred.reshape(pred.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    diff = pred_flat - target_flat
    denom = torch.linalg.vector_norm(target_flat, ord=2, dim=1).clamp_min(1e-12)
    return torch.linalg.vector_norm(diff, ord=2, dim=1) / denom


def per_sample_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target).reshape(pred.shape[0], -1).square(), dim=1)


def per_sample_metric(pred: torch.Tensor, target: torch.Tensor, metric: str) -> torch.Tensor:
    if metric == "rel_l2":
        return per_sample_rel_l2(pred, target)
    if metric == "mse":
        return per_sample_mse(pred, target)
    raise ValueError(f"unknown metric {metric!r}")


def compute_loss_values(
    *,
    pred: torch.Tensor,
    solver_adv: torch.Tensor | None,
    clean_model: torch.Tensor,
    clean_solver: torch.Tensor,
    loss_name: str,
    metric: str,
) -> torch.Tensor:
    if loss_name == "loss1":
        return per_sample_metric(pred, clean_model, metric)
    if loss_name == "loss2":
        return per_sample_metric(pred, clean_solver, metric)
    if loss_name == "loss3":
        if solver_adv is None:
            raise ValueError("loss3 requires solver_adv")
        return per_sample_metric(pred, solver_adv, metric)
    raise ValueError(f"unknown loss_name {loss_name!r}")


def summarize_values(prefix: str, values: torch.Tensor) -> dict[str, float]:
    v = values.detach().float()
    return {
        f"{prefix}_mean": float(v.mean().item()),
        f"{prefix}_std": float(v.std(unbiased=False).item()),
        f"{prefix}_min": float(v.min().item()),
        f"{prefix}_max": float(v.max().item()),
    }


def _sample_np(x: torch.Tensor, sample_position: int) -> np.ndarray:
    return x[sample_position].detach().cpu().float().numpy()


def append_step_sample_trace(
    records: list[dict[str, np.ndarray | float | int | str]],
    *,
    step: int,
    phase: str,
    sample_position: int,
    dataset_index: int,
    x_clean: torch.Tensor,
    a_state: torch.Tensor,
    flip_mask: torch.Tensor,
    pred: torch.Tensor,
    solver_u: torch.Tensor,
    surrogate: torch.Tensor,
    true_loss3: torch.Tensor,
) -> None:
    a_np = _sample_np(a_state, sample_position)
    clean_np = _sample_np(x_clean, sample_position)
    pred_np = _sample_np(pred, sample_position)
    solver_np = _sample_np(solver_u, sample_position)
    records.append(
        {
            "step": int(step),
            "phase": phase,
            "sample_position": int(sample_position),
            "dataset_index": int(dataset_index),
            "surrogate_loss": float(surrogate[sample_position].detach().float().cpu().item()),
            "true_loss3": float(true_loss3[sample_position].detach().float().cpu().item()),
            "flip_count": int(flip_mask[sample_position].detach().reshape(-1).sum().cpu().item()),
            "A_adv": a_np,
            "perturbation": a_np - clean_np,
            "flip_mask": flip_mask[sample_position].detach().cpu().numpy().astype(np.uint8),
            "model_u": pred_np,
            "solver_u": solver_np,
            "model_solver_diff": pred_np - solver_np,
        }
    )


def save_step_sample_trace(path: Path, records: list[dict[str, np.ndarray | float | int | str]]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        steps=np.array([int(r["step"]) for r in records], dtype=np.int32),
        phases=np.array([str(r["phase"]) for r in records]),
        sample_position=np.array([int(r["sample_position"]) for r in records], dtype=np.int32),
        dataset_index=np.array([int(r["dataset_index"]) for r in records], dtype=np.int64),
        surrogate_loss=np.array([float(r["surrogate_loss"]) for r in records], dtype=np.float32),
        true_loss3=np.array([float(r["true_loss3"]) for r in records], dtype=np.float32),
        flip_count=np.array([int(r["flip_count"]) for r in records], dtype=np.int32),
        A_adv=np.stack([r["A_adv"] for r in records]).astype(np.float32),
        perturbation=np.stack([r["perturbation"] for r in records]).astype(np.float32),
        flip_mask=np.stack([r["flip_mask"] for r in records]).astype(np.uint8),
        model_u=np.stack([r["model_u"] for r in records]).astype(np.float32),
        solver_u=np.stack([r["solver_u"] for r in records]).astype(np.float32),
        model_solver_diff=np.stack([r["model_solver_diff"] for r in records]).astype(np.float32),
    )


def select_top_mask(
    scores: torch.Tensor,
    *,
    k_per_sample: int,
    valid: torch.Tensor,
    positive_only: bool,
) -> tuple[torch.Tensor, list[int]]:
    selected = torch.zeros_like(valid, dtype=torch.bool)
    if k_per_sample <= 0:
        return selected, [0 for _ in range(scores.shape[0])]
    masked = scores.masked_fill(~valid, float("-inf"))
    flat_scores = masked.reshape(masked.shape[0], -1)
    flat_selected = selected.reshape(selected.shape[0], -1)
    counts: list[int] = []
    for b in range(flat_scores.shape[0]):
        finite_count = int(torch.isfinite(flat_scores[b]).sum().item())
        k = min(k_per_sample, finite_count)
        if k <= 0:
            counts.append(0)
            continue
        top_scores, top_idx = torch.topk(flat_scores[b], k=k)
        if positive_only:
            keep = top_scores > 0
            top_idx = top_idx[keep]
        if top_idx.numel() > 0:
            flat_selected[b, top_idx] = True
        counts.append(int(top_idx.numel()))
    return selected, counts


def select_top_add_mask(
    scores: torch.Tensor,
    *,
    flip_mask: torch.Tensor,
    max_flips: int,
    alpha_flips: int,
    valid: torch.Tensor,
    positive_only: bool,
) -> tuple[torch.Tensor, list[int]]:
    selected = torch.zeros_like(valid, dtype=torch.bool)
    if alpha_flips <= 0 or max_flips <= 0:
        return selected, [0 for _ in range(scores.shape[0])]
    available = valid & (~flip_mask)
    masked = scores.masked_fill(~available, float("-inf"))
    flat_scores = masked.reshape(masked.shape[0], -1)
    flat_selected = selected.reshape(selected.shape[0], -1)
    flat_flip = flip_mask.reshape(flip_mask.shape[0], -1)
    counts: list[int] = []
    for b in range(flat_scores.shape[0]):
        current = int(flat_flip[b].sum().item())
        remaining = max(0, max_flips - current)
        finite_count = int(torch.isfinite(flat_scores[b]).sum().item())
        k = min(alpha_flips, remaining, finite_count)
        if k <= 0:
            counts.append(0)
            continue
        top_scores, top_idx = torch.topk(flat_scores[b], k=k)
        if positive_only:
            keep = top_scores > 0
            top_idx = top_idx[keep]
        if top_idx.numel() > 0:
            flat_selected[b, top_idx] = True
        counts.append(int(top_idx.numel()))
    return selected, counts


def normalize_method(name: str) -> str:
    lowered = name.strip().lower()
    if lowered == "pag":
        return "pgd"
    if lowered not in METHOD_NAMES:
        raise ValueError(f"unknown method {name!r}; expected one of {METHOD_NAMES} or alias 'pag'")
    return lowered


def update_flip_mask(
    *,
    method: str,
    flip_mask: torch.Tensor,
    grad: torch.Tensor,
    x0: torch.Tensor,
    opposite: torch.Tensor,
    valid_mask: torch.Tensor,
    max_flips: int,
    alpha_flips: int,
    positive_only: bool,
    score_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | int]]:
    method = normalize_method(method)
    current = torch.where(flip_mask, opposite, x0)
    clean_flip_delta = opposite - x0
    current_flip_delta = opposite - current

    if method == "pgd":
        # Binary projected gradient ascent: accumulate first-order flip gains
        # and project the mask to the top-K accumulated coordinates.
        score_state = score_state + grad.detach() * clean_flip_delta
        selected, counts = select_top_mask(
            score_state,
            k_per_sample=max_flips,
            valid=valid_mask,
            positive_only=positive_only,
        )
        return selected, score_state, {
            "selected_mean": float(np.mean(counts)),
            "selected_min": int(np.min(counts)),
            "selected_max": int(np.max(counts)),
        }

    if method == "raw_add":
        wants_high = grad.detach() > 0
        would_flip = ((x0 < opposite) & wants_high) | ((x0 > opposite) & (~wants_high))
        available = valid_mask & (~flip_mask) & would_flip
        scores = grad.detach().abs()
        selected, counts = select_top_add_mask(
            scores,
            flip_mask=flip_mask,
            max_flips=max_flips,
            alpha_flips=alpha_flips,
            valid=available,
            positive_only=False,
        )
        return flip_mask | selected, score_state, {
            "selected_mean": float(np.mean(counts)),
            "selected_min": int(np.min(counts)),
            "selected_max": int(np.max(counts)),
        }

    if method == "raw_replace":
        wants_high = grad.detach() > 0
        would_flip = ((x0 < opposite) & wants_high) | ((x0 > opposite) & (~wants_high))
        available = valid_mask & would_flip
        scores = grad.detach().abs()
        selected, counts = select_top_mask(
            scores,
            k_per_sample=max_flips,
            valid=available,
            positive_only=False,
        )
        return selected, score_state, {
            "selected_mean": float(np.mean(counts)),
            "selected_min": int(np.min(counts)),
            "selected_max": int(np.max(counts)),
        }

    if method == "steepest_add":
        gains = grad.detach() * current_flip_delta
        available = valid_mask & (~flip_mask)
        selected, counts = select_top_add_mask(
            gains,
            flip_mask=flip_mask,
            max_flips=max_flips,
            alpha_flips=alpha_flips,
            valid=available,
            positive_only=positive_only,
        )
        return flip_mask | selected, score_state, {
            "selected_mean": float(np.mean(counts)),
            "selected_min": int(np.min(counts)),
            "selected_max": int(np.max(counts)),
        }

    if method == "steepest_replace":
        gains = grad.detach() * clean_flip_delta
        selected, counts = select_top_mask(
            gains,
            k_per_sample=max_flips,
            valid=valid_mask,
            positive_only=positive_only,
        )
        return selected, score_state, {
            "selected_mean": float(np.mean(counts)),
            "selected_min": int(np.min(counts)),
            "selected_max": int(np.max(counts)),
        }

    raise AssertionError(method)


def stable_name_offset(*parts: str) -> int:
    total = 0
    for part in parts:
        for ch in part:
            total = (total * 131 + ord(ch)) % 1_000_003
    return total


def random_initial_mask(
    *,
    shape: torch.Size,
    valid_mask: torch.Tensor,
    flips_per_sample: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    if flips_per_sample <= 0:
        return torch.zeros(shape, dtype=torch.bool, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    scores = torch.rand(shape, device=device, generator=generator)
    selected, _ = select_top_mask(
        scores,
        k_per_sample=flips_per_sample,
        valid=valid_mask,
        positive_only=False,
    )
    return selected


def evaluate_no_grad(
    *,
    model,
    solver,
    a: torch.Tensor,
    clean_model: torch.Tensor,
    clean_solver: torch.Tensor,
    metric: str,
) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        pred = model(a.unsqueeze(-1)).squeeze(-1)
        solver_adv = JaxDarcySolver.apply(a, solver)
        losses = {
            "loss1": compute_loss_values(
                pred=pred,
                solver_adv=solver_adv,
                clean_model=clean_model,
                clean_solver=clean_solver,
                loss_name="loss1",
                metric=metric,
            ).detach(),
            "loss2": compute_loss_values(
                pred=pred,
                solver_adv=solver_adv,
                clean_model=clean_model,
                clean_solver=clean_solver,
                loss_name="loss2",
                metric=metric,
            ).detach(),
            "loss3": compute_loss_values(
                pred=pred,
                solver_adv=solver_adv,
                clean_model=clean_model,
                clean_solver=clean_solver,
                loss_name="loss3",
                metric=metric,
            ).detach(),
        }
    return {
        "pred": pred.detach(),
        "solver": solver_adv.detach(),
        **losses,
    }


def plot_loss_curves(path: Path, rows: list[dict]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] matplotlib unavailable, skipping loss curves: {exc}", flush=True)
        return

    steps = np.array([float(r["step"]) for r in rows], dtype=np.float64)
    surrogate = np.array([float(r["surrogate_mean"]) for r in rows], dtype=np.float64)
    true = np.array([float(r["true_loss3_mean"]) for r in rows], dtype=np.float64)
    flips = np.array([float(r["flip_count_mean"]) for r in rows], dtype=np.float64)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), constrained_layout=True)
    axes[0].plot(steps, surrogate, marker="o", markersize=2, linewidth=1.4)
    axes[0].set_title("surrogate objective")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("mean loss")
    axes[1].plot(steps, true, marker="o", markersize=2, linewidth=1.4, color="tab:red")
    axes[1].set_title("true loss3")
    axes[1].set_xlabel("step")
    axes[2].plot(steps, flips, marker="o", markersize=2, linewidth=1.4, color="tab:green")
    axes[2].set_title("binary flips")
    axes[2].set_xlabel("step")
    axes[2].set_ylabel("mean count")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_one(
    *,
    args: argparse.Namespace,
    run_root: Path,
    model,
    solver,
    x0: torch.Tensor,
    clean_model: torch.Tensor,
    clean_solver: torch.Tensor,
    dataset_indices: torch.Tensor,
    loss_name: str,
    method: str,
    valid_mask: torch.Tensor,
    max_flips: int,
    alpha_flips: int,
) -> dict:
    run_root.mkdir(parents=True, exist_ok=True)
    opposite = torch.where(x0 >= 0.5 * (args.low + args.high), torch.full_like(x0, args.low), torch.full_like(x0, args.high))
    flip_mask = torch.zeros_like(x0, dtype=torch.bool)
    if loss_name == "loss1" and args.loss1_random_start:
        random_flips = args.loss1_random_start_flips
        if random_flips is None:
            random_flips = alpha_flips
        random_flips = max(0, min(int(random_flips), max_flips))
        flip_mask = random_initial_mask(
            shape=x0.shape,
            valid_mask=valid_mask,
            flips_per_sample=random_flips,
            seed=args.seed + stable_name_offset(loss_name, method),
            device=x0.device,
        )
    score_state = torch.zeros_like(x0)
    trace_rows: list[dict] = []
    step_loss_rows: list[dict] = []
    sample_trace_records: list[dict[str, np.ndarray | float | int | str]] = []
    trace_sample_position = int(args.trace_sample_index)
    if trace_sample_position < 0 or trace_sample_position >= x0.shape[0]:
        trace_sample_position = 0
    trace_dataset_index = int(dataset_indices[trace_sample_position].item())
    start_time = time.perf_counter()

    for step in range(args.steps):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        step_start = time.perf_counter()
        current_a = torch.where(flip_mask, opposite, x0).detach().requires_grad_(True)
        pred = model(current_a.unsqueeze(-1)).squeeze(-1)
        solver_for_objective = JaxDarcySolver.apply(current_a, solver) if loss_name == "loss3" else None
        surrogate = compute_loss_values(
            pred=pred,
            solver_adv=solver_for_objective,
            clean_model=clean_model,
            clean_solver=clean_solver,
            loss_name=loss_name,
            metric=args.metric,
        )
        objective = surrogate.mean()
        model.zero_grad(set_to_none=True)
        objective.backward()
        grad = current_a.grad.detach()

        if loss_name == "loss3":
            solver_adv = solver_for_objective.detach()
            pred_det = pred.detach()
            true_loss3 = surrogate.detach()
        else:
            pred_det = pred.detach()
            should_eval_true = (
                args.trace_true_loss3_every > 0
                and (step % args.trace_true_loss3_every == 0 or step == args.steps - 1)
            )
            if should_eval_true:
                with torch.no_grad():
                    solver_adv = JaxDarcySolver.apply(current_a.detach(), solver)
                    true_loss3 = compute_loss_values(
                        pred=pred_det,
                        solver_adv=solver_adv,
                        clean_model=clean_model,
                        clean_solver=clean_solver,
                        loss_name="loss3",
                        metric=args.metric,
                    ).detach()
            else:
                true_loss3 = torch.full_like(surrogate.detach(), float("nan"))

        flip_mask, score_state, update_stats = update_flip_mask(
            method=method,
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
            "loss_name": loss_name,
            "method": method,
            "objective": float(objective.detach().item()),
            "flip_count_mean": float(flip_counts.mean().item()),
            "flip_count_min": float(flip_counts.min().item()),
            "flip_count_max": float(flip_counts.max().item()),
            **summarize_values("surrogate", surrogate),
            **summarize_values("true_loss3", true_loss3),
            **update_stats,
            **cuda_memory_snapshot(),
        }
        trace_rows.append(row)
        for sample_position in range(x0.shape[0]):
            step_loss_rows.append(
                {
                    "step": step,
                    "phase": "pre_update",
                    "sample_position": int(sample_position),
                    "dataset_index": int(dataset_indices[sample_position].item()),
                    "loss_name": loss_name,
                    "method": method,
                    "surrogate_loss": float(surrogate[sample_position].detach().float().cpu().item()),
                    "true_loss3": float(true_loss3[sample_position].detach().float().cpu().item()),
                    "flip_count": int(flip_mask[sample_position].detach().reshape(-1).sum().cpu().item()),
                }
            )
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
                pred=pred_det,
                solver_u=solver_adv,
                surrogate=surrogate.detach(),
                true_loss3=true_loss3.detach(),
            )
        print(
            f"[{loss_name}/{method} step {step:03d}] "
            f"surrogate={row['surrogate_mean']:.6e} true_loss3={row['true_loss3_mean']:.6e} "
            f"flips={row['flip_count_mean']:.1f}/{max_flips} sec={row['seconds']:.2f}",
            flush=True,
        )

    final_a = torch.where(flip_mask, opposite, x0).detach()
    final_eval = evaluate_no_grad(
        model=model,
        solver=solver,
        a=final_a,
        clean_model=clean_model,
        clean_solver=clean_solver,
        metric=args.metric,
    )
    clean_eval = {
        "loss1": per_sample_metric(clean_model, clean_model, args.metric),
        "loss2": per_sample_metric(clean_model, clean_solver, args.metric),
        "loss3": per_sample_metric(clean_model, clean_solver, args.metric),
    }
    final_flip_counts = flip_mask.reshape(flip_mask.shape[0], -1).sum(dim=1).detach().float()
    final_surrogate = final_eval[loss_name]
    final_row = {
        "step": args.steps,
        "phase": "final",
        "seconds": 0.0,
        "seconds_since_run_start": time.perf_counter() - start_time,
        "loss_name": loss_name,
        "method": method,
        "objective": float(final_surrogate.mean().item()),
        "flip_count_mean": float(final_flip_counts.mean().item()),
        "flip_count_min": float(final_flip_counts.min().item()),
        "flip_count_max": float(final_flip_counts.max().item()),
        **summarize_values("surrogate", final_surrogate),
        **summarize_values("true_loss3", final_eval["loss3"]),
        "selected_mean": 0.0,
        "selected_min": 0,
        "selected_max": 0,
        **cuda_memory_snapshot(),
    }
    trace_rows.append(final_row)
    for sample_position in range(x0.shape[0]):
        step_loss_rows.append(
            {
                "step": args.steps,
                "phase": "final",
                "sample_position": int(sample_position),
                "dataset_index": int(dataset_indices[sample_position].item()),
                "loss_name": loss_name,
                "method": method,
                "surrogate_loss": float(final_surrogate[sample_position].detach().float().cpu().item()),
                "true_loss3": float(final_eval["loss3"][sample_position].detach().float().cpu().item()),
                "flip_count": int(final_flip_counts[sample_position].item()),
            }
        )
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
            surrogate=final_surrogate,
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
                "final_loss1": float(final_eval["loss1"][i].item()),
                "final_loss2": float(final_eval["loss2"][i].item()),
                "final_loss3_true": float(final_eval["loss3"][i].item()),
                "final_surrogate": float(final_surrogate[i].item()),
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
            "final_loss1": final_eval["loss1"].detach().cpu(),
            "final_loss2": final_eval["loss2"].detach().cpu(),
            "final_loss3": final_eval["loss3"].detach().cpu(),
            "final_surrogate": final_surrogate.detach().cpu(),
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
        "loss_name": loss_name,
        "method": method,
        "metric": args.metric,
        "steps": args.steps,
        "num_samples": int(x0.shape[0]),
        "max_flips": max_flips,
        "alpha_flips": alpha_flips,
        "final_surrogate_mean": float(final_surrogate.mean().item()),
        "final_true_loss3_mean": float(final_eval["loss3"].mean().item()),
        "clean_true_loss3_mean": float(clean_eval["loss3"].mean().item()),
        "true_loss3_increase_mean": float((final_eval["loss3"] - clean_eval["loss3"]).mean().item()),
        "final_flip_count_mean": float(final_flip_counts.mean().item()),
        "seconds": time.perf_counter() - start_time,
    }
    with (run_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def build_run_plan(args: argparse.Namespace) -> list[tuple[str, str, str]]:
    plan: list[tuple[str, str, str]] = []
    if args.experiment == "loss_method_grid":
        for loss in args.losses:
            for method in args.methods:
                plan.append(("loss_method_grid", loss, normalize_method(method)))
    if args.experiment in ("both", "loss_objectives"):
        for loss in args.losses:
            plan.append(("loss_objectives", loss, normalize_method(args.loss_objective_method)))
    if args.experiment in ("both", "loss3_methods"):
        for method in args.methods:
            plan.append(("loss3_methods", "loss3", normalize_method(method)))
    seen = set()
    deduped = []
    for item in plan:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DARCY_ROOT
        / "datasets"
        / "grf_darcy_20260528_N1500"
        / "test"
        / "dim2d_darcy_nx211_N300_solver=jaxcg_solve421_alpha2_tau3_binary3-12_f1_seed45_test.pt",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DARCY_ROOT / "saved_models" / "2D" / "darcy_N1500_nx85_m64_w60_e500_20260528" / "best.pt",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--experiment", choices=["both", "loss_objectives", "loss3_methods", "loss_method_grid"], default="both")
    parser.add_argument("--losses", nargs="+", choices=LOSS_NAMES, default=["loss1", "loss2", "loss3"])
    parser.add_argument("--loss-objective-method", default="steepest_add")
    parser.add_argument("--methods", nargs="+", default=["pgd", "raw_add", "raw_replace", "steepest_add", "steepest_replace"])
    parser.add_argument("--metric", choices=["rel_l2", "mse"], default="rel_l2")
    parser.add_argument("--epsilon", "--epsilon-fraction", dest="epsilon_fraction", type=float, default=0.01)
    parser.add_argument("--epsilon-flips", type=int, default=None)
    parser.add_argument("--alpha", "--alpha-fraction", dest="alpha_fraction", type=float, default=None)
    parser.add_argument("--alpha-flips", type=int, default=None)
    parser.add_argument("--positive-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--loss1-random-start", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--loss1-random-start-flips", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-boundary-flips", action="store_true")
    parser.add_argument("--low", type=float, default=3.0)
    parser.add_argument("--high", type=float, default=12.0)
    parser.add_argument("--solver-tol", type=float, default=1e-5)
    parser.add_argument("--solver-atol", type=float, default=0.0)
    parser.add_argument("--solver-maxiter", type=int, default=None)
    parser.add_argument("--trace-true-loss3-every", type=int, default=1, help="For loss1/loss2 objectives, evaluate true loss3 every N steps; 0 means final-only.")
    parser.add_argument("--modes", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--padding", type=int, default=None)
    parser.add_argument("--plot-samples", type=int, default=4)
    parser.add_argument("--trace-sample-index", type=int, default=0, help="Batch position for full per-step field/model/solver trace; negative disables it.")
    parser.add_argument("--output-root", type=Path, default=DARCY_ROOT / "perturbation_results" / "binary_loss_method_sweep")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)
    args.dataset = args.dataset.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.output_root = args.output_root.resolve()

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

    timestamp = time.strftime("%Y%m%d_%H%M%S_UTC", time.gmtime())
    run_name = args.run_name or (
        f"darcy_binary_loss_method_nx{resolution}_N{x0.shape[0]}_"
        f"eps{args.epsilon_fraction:g}_alpha{args.alpha_fraction if args.alpha_fraction is not None else 'auto'}_"
        f"steps{args.steps}_{timestamp}"
    )
    root = (args.output_root / run_name).resolve()
    root.mkdir(parents=True, exist_ok=True)
    dataset_indices = torch.arange(args.start, args.start + x0.shape[0], dtype=torch.long)
    plan = build_run_plan(args)

    manifest = {
        "run_root": str(root),
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
        "metric": args.metric,
        "trace_true_loss3_every": args.trace_true_loss3_every,
        "trace_sample_index": args.trace_sample_index,
        "losses": args.losses,
        "methods": [normalize_method(m) for m in args.methods],
        "loss_objective_method": normalize_method(args.loss_objective_method),
        "loss1_random_start": args.loss1_random_start,
        "loss1_random_start_flips": args.loss1_random_start_flips,
        "seed": args.seed,
        "plan": [{"experiment": exp, "loss": loss, "method": method} for exp, loss, method in plan],
        "binary_policy": "A is always projected to {low, high}; epsilon/alpha are Hamming flip budgets.",
    }
    with (root / "experiment_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    summaries = []
    for experiment_name, loss_name, method in plan:
        run_dir = root / experiment_name / loss_name / method
        summary = run_one(
            args=args,
            run_root=run_dir,
            model=model,
            solver=solver,
            x0=x0,
            clean_model=clean_model,
            clean_solver=clean_solver,
            dataset_indices=dataset_indices,
            loss_name=loss_name,
            method=method,
            valid_mask=valid_mask,
            max_flips=max_flips,
            alpha_flips=alpha_flips,
        )
        summary["experiment"] = experiment_name
        summaries.append(summary)

    write_csv(root / "all_run_summary.csv", summaries)
    manifest["completed_utc"] = time.strftime("%Y%m%d_%H%M%S_UTC", time.gmtime())
    manifest["summaries"] = summaries
    with (root / "experiment_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps({"run_root": str(root), "summary_csv": str(root / "all_run_summary.csv")}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
