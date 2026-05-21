#!/usr/bin/env python3
"""Run the core-four optimizer comparison for loss1/loss2 at one baseline setting.

This is the loss1/loss2 analogue of the loss3 core-four baseline visual run,
restricted to one p/q/epsilon/alpha setting.  It keeps the GPU-only checks from
the loss3 runner but avoids differentiating the solver during the optimization:

- loss1: ||f(x + delta) - f(x)||_q
- loss2: ||f(x + delta) - g(x)||_q, with g(x) fixed at the clean input

For loss1, a tiny random start is used by default because the zero-delta
gradient is identically zero.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("DDE_BACKEND", "pytorch")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from loss_attack_common import save_json
from tools.attack_framework_matrix import (
    DEFAULT_BURGERS_MODEL_DIR,
    DEFAULT_BURGERS_TEST,
    make_burgers_jax_solver,
    make_jax_torch_bridge,
    sync_torch,
)
from tools.run_batch_three_loss_loss_only import (
    DEFAULT_DEEPONET_RUN_DIR,
    EPS,
    batch_norm,
    finite_mean_std,
    format_model_label,
    load_model,
    norm_name,
    parse_norm,
    project_delta,
    random_delta_like,
    sanitize_tensor,
    steepest_direction,
    write_csv,
)
from tools.run_loss3_direction_proposal_ablation import (
    batch_cosine_torch,
    batch_l2_torch,
    batch_linf_torch,
    collect_gpu_evidence,
    load_burgers_samples,
    smoothness_batch,
    tensor_to_numpy,
)


CORE4 = ("raw_add", "raw_replace", "steepest_add", "steepest_replace")
COLORS = {
    "raw_add": "#1f77b4",
    "raw_replace": "#d62728",
    "steepest_add": "#2ca02c",
    "steepest_replace": "#9467bd",
}
LINESTYLES = {
    "raw_add": "-",
    "raw_replace": "--",
    "steepest_add": "-.",
    "steepest_replace": ":",
}
METHOD_LABELS = {
    "raw_add": "raw add",
    "raw_replace": "raw replace",
    "steepest_add": "steepest add",
    "steepest_replace": "steepest replace",
}
BOUNDARY_MARKERS = (
    (0.25, "o", "25% boundary"),
    (0.50, "^", "50% boundary"),
    (0.75, "s", "75% boundary"),
    (0.99, "x", "99% boundary"),
)


class Problem:
    def __init__(self, args: argparse.Namespace):
        import torch

        self.args = args
        self.device = torch.device(args.device)
        x_np, dataset_indices = load_burgers_samples(
            args.burgers_test_path,
            args.start_index,
            args.batch_size,
            args.dataset_indices,
        )
        args.batch_size = int(x_np.shape[0])
        self.dataset_indices = dataset_indices
        self.x0 = torch.as_tensor(x_np, device=self.device, dtype=torch.float32)
        self.model = load_model(args, self.device)
        self.bridge = make_jax_torch_bridge()
        self.jax_solver = make_burgers_jax_solver(args)
        self.f0 = self.model_forward(self.x0).detach()
        self.g0 = self.solver_forward(self.x0).detach()

    def model_forward(self, x):
        return self.model(x.to(dtype=x.dtype))

    def solver_forward(self, x):
        return self.bridge(x, self.jax_solver, "solver").detach()


def finite_stats(prefix: str, values: Any) -> dict[str, Any]:
    mean, std, finite_count, total_count = finite_mean_std(values)
    return {
        f"{prefix}_mean": mean,
        f"{prefix}_std": std,
        f"{prefix}_finite_count": finite_count,
        f"{prefix}_nonfinite_count": total_count - finite_count,
    }


def selected_positions(problem: Problem, requested: list[int]) -> tuple[list[int], list[int], list[int]]:
    idx_to_pos = {int(idx): pos for pos, idx in enumerate(problem.dataset_indices.tolist())}
    present: list[int] = []
    positions: list[int] = []
    missing: list[int] = []
    for idx in requested:
        if int(idx) in idx_to_pos:
            present.append(int(idx))
            positions.append(idx_to_pos[int(idx)])
        else:
            missing.append(int(idx))
    return present, positions, missing


def compute_losses(problem: Problem, delta) -> dict[str, Any]:
    f_adv = problem.model_forward(problem.x0 + delta)
    q = problem.args.q_order
    return {
        "loss1_original": batch_norm(f_adv - problem.f0, q),
        "loss2_original": batch_norm(f_adv - problem.g0, q),
    }


def angle_degrees_from_cosine(cosine):
    import torch

    return torch.acos(cosine.clamp(-1.0, 1.0)) * (180.0 / math.pi)


def valid_batch_angle_degrees(a, b):
    import torch

    out = torch.full((a.shape[0],), float("nan"), device=a.device, dtype=a.dtype)
    a_flat = a.reshape(a.shape[0], -1)
    b_flat = b.reshape(b.shape[0], -1)
    denom = torch.linalg.vector_norm(a_flat, dim=1) * torch.linalg.vector_norm(b_flat, dim=1)
    valid = denom > EPS
    if bool(valid.any().detach().cpu()):
        cos = torch.sum(a_flat[valid] * b_flat[valid], dim=1) / denom[valid]
        out[valid] = angle_degrees_from_cosine(cos)
    return out.detach()


def propose_delta(problem: Problem, method: str, delta, grad):
    if method == "raw_add":
        direction = grad
        proposal = delta + float(problem.args.alpha) * direction
    elif method == "raw_replace":
        direction = batch_norm(grad, problem.args.p_order).reshape(-1, *([1] * (grad.ndim - 1))).clamp_min(EPS)
        direction = grad / direction
        proposal = float(problem.args.epsilon) * direction
    elif method == "steepest_add":
        direction = steepest_direction(grad, problem.args.p_order)
        proposal = delta + float(problem.args.alpha) * direction
    elif method == "steepest_replace":
        direction = steepest_direction(grad, problem.args.p_order)
        proposal = float(problem.args.epsilon) * direction
    else:
        raise ValueError(method)
    return sanitize_tensor(direction.detach()), sanitize_tensor(project_delta(proposal, problem.args.epsilon, problem.args.p_order))


def init_delta(problem: Problem, objective: str):
    import torch

    if objective == "loss1_original" and problem.args.loss1_init == "random":
        return random_delta_like(
            problem.x0,
            problem.args.epsilon,
            problem.args.p_order,
            problem.args.loss1_random_start_scale,
            problem.args.seed,
        )
    return torch.zeros_like(problem.x0)


def run_one(problem: Problem, objective: str, method: str, root: Path) -> dict[str, Any]:
    import torch

    method_dir = root / objective / method
    method_dir.mkdir(parents=True, exist_ok=True)
    delta = init_delta(problem, objective)
    present_indices, selected_pos, missing = selected_positions(problem, problem.args.trajectory_indices)

    rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    trajectory: dict[str, list[Any]] = defaultdict(list)
    prev_delta = None
    prev_direction = None
    start = time.perf_counter()
    if torch.cuda.is_available() and problem.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(problem.device)
        torch.cuda.synchronize(problem.device)

    for k in range(problem.args.steps + 1):
        delta = sanitize_tensor(delta.detach())
        x_delta = delta.detach().requires_grad_(k < problem.args.steps)
        losses = compute_losses(problem, x_delta)
        objective_values = sanitize_tensor(losses[objective])
        grad = None
        direction = None
        if k < problem.args.steps:
            objective_values.sum().backward()
            grad = sanitize_tensor(x_delta.grad.detach())
            direction, next_delta = propose_delta(problem, method, x_delta.detach(), grad)

        delta_norm = batch_norm(delta, problem.args.p_order).detach()
        delta_l2 = batch_l2_torch(delta).detach()
        delta_linf = batch_linf_torch(delta).detach()
        smooth = smoothness_batch(tensor_to_numpy(delta.detach()).astype(np.float32))
        step_l2 = torch.full_like(delta_norm, float("nan"))
        delta_prev_angle = torch.full_like(delta_norm, float("nan"))
        if prev_delta is not None:
            step_l2 = batch_l2_torch(delta - prev_delta).detach()
            delta_prev_angle = valid_batch_angle_degrees(delta, prev_delta)
        direction_prev_angle = torch.full_like(delta_norm, float("nan"))
        delta_direction_angle = torch.full_like(delta_norm, float("nan"))
        grad_direction_angle = torch.full_like(delta_norm, float("nan"))
        if direction is not None:
            delta_direction_angle = valid_batch_angle_degrees(delta, direction)
            grad_direction_angle = valid_batch_angle_degrees(grad, direction)
            if prev_direction is not None:
                direction_prev_angle = valid_batch_angle_degrees(direction, prev_direction)

        arrays: dict[str, np.ndarray] = {
            "loss1_original": tensor_to_numpy(losses["loss1_original"].detach()).astype(np.float32),
            "loss2_original": tensor_to_numpy(losses["loss2_original"].detach()).astype(np.float32),
            "optimized_loss": tensor_to_numpy(objective_values.detach()).astype(np.float32),
            "delta_pnorm": tensor_to_numpy(delta_norm).astype(np.float32),
            "delta_l2": tensor_to_numpy(delta_l2).astype(np.float32),
            "delta_linf": tensor_to_numpy(delta_linf).astype(np.float32),
            "boundary_ratio": tensor_to_numpy(delta_norm / float(problem.args.epsilon)).astype(np.float32),
            "delta_step_l2": tensor_to_numpy(step_l2).astype(np.float32),
            "delta_prev_angle_degrees": tensor_to_numpy(delta_prev_angle).astype(np.float32),
            "delta_direction_angle_degrees": tensor_to_numpy(delta_direction_angle).astype(np.float32),
            "grad_direction_angle_degrees": tensor_to_numpy(grad_direction_angle).astype(np.float32),
            "direction_prev_angle_degrees": tensor_to_numpy(direction_prev_angle).astype(np.float32),
        }
        for key, values in smooth.items():
            arrays[key] = values.astype(np.float32)
        if grad is not None:
            arrays["grad_l2"] = tensor_to_numpy(batch_l2_torch(grad).detach()).astype(np.float32)
            arrays["grad_linf"] = tensor_to_numpy(batch_linf_torch(grad).detach()).astype(np.float32)
        if direction is not None:
            arrays["direction_l2"] = tensor_to_numpy(batch_l2_torch(direction).detach()).astype(np.float32)

        row: dict[str, Any] = {
            "objective": objective,
            "method": method,
            "k": k,
            "epsilon": problem.args.epsilon,
            "alpha": problem.args.alpha,
            "p_order": norm_name(problem.args.p_order),
            "q_order": norm_name(problem.args.q_order),
            "seconds_since_method_start": time.perf_counter() - start,
        }
        for key, values in arrays.items():
            row.update(finite_stats(key, values))
        rows.append(row)

        for i, dataset_index in enumerate(problem.dataset_indices.tolist()):
            sample = {
                "objective": objective,
                "method": method,
                "k": k,
                "sample_position": i,
                "dataset_index": int(dataset_index),
                "epsilon": problem.args.epsilon,
                "alpha": problem.args.alpha,
                "p_order": norm_name(problem.args.p_order),
                "q_order": norm_name(problem.args.q_order),
            }
            for key, values in arrays.items():
                sample[key] = float(values[i]) if i < len(values) else float("nan")
            sample_rows.append(sample)

        if selected_pos and (k in problem.args.selected_steps or problem.args.save_every == 1):
            trajectory["k"].append(k)
            trajectory["delta"].append(tensor_to_numpy(delta.detach()).astype(np.float32)[selected_pos])
            trajectory["x_adv"].append(tensor_to_numpy((problem.x0 + delta).detach()).astype(np.float32)[selected_pos])
            for key in ("loss1_original", "loss2_original", "optimized_loss", "delta_pnorm", "boundary_ratio"):
                trajectory[key].append(arrays[key][selected_pos])

        if k >= problem.args.steps:
            break
        prev_delta = delta.detach()
        if direction is not None:
            prev_direction = direction.detach()
        delta = next_delta.detach()

    sync_torch(torch, problem.device)
    runtime_seconds = time.perf_counter() - start
    torch_peak_allocated_mib = float("nan")
    torch_peak_reserved_mib = float("nan")
    if torch.cuda.is_available() and problem.device.type == "cuda":
        torch_peak_allocated_mib = float(torch.cuda.max_memory_allocated(problem.device) / 1024**2)
        torch_peak_reserved_mib = float(torch.cuda.max_memory_reserved(problem.device) / 1024**2)

    write_csv(method_dir / "per_step_metrics.csv", rows)
    write_csv(method_dir / "per_sample_step_metrics.csv", sample_rows)
    final_delta_np = tensor_to_numpy(delta.detach()).astype(np.float32)
    np.savez_compressed(
        method_dir / "final_delta.npz",
        dataset_index=problem.dataset_indices.astype(np.int64),
        final_delta=final_delta_np,
    )
    if selected_pos and trajectory:
        np.savez_compressed(
            method_dir / "trajectory_samples.npz",
            objective=np.asarray([objective]),
            method=np.asarray([method]),
            dataset_index=np.asarray(present_indices, dtype=np.int64),
            sample_position=np.asarray(selected_pos, dtype=np.int64),
            **{key: np.asarray(value) for key, value in trajectory.items()},
        )
    final_row = rows[-1]
    summary = {
        "objective": objective,
        "method": method,
        "epsilon": problem.args.epsilon,
        "alpha": problem.args.alpha,
        "steps": problem.args.steps,
        "p_order": norm_name(problem.args.p_order),
        "q_order": norm_name(problem.args.q_order),
        "runtime_seconds": runtime_seconds,
        "torch_peak_allocated_mib": torch_peak_allocated_mib,
        "torch_peak_reserved_mib": torch_peak_reserved_mib,
        "trajectory_indices_present": present_indices,
        "trajectory_indices_missing": missing,
        "final_optimized_loss_mean": final_row["optimized_loss_mean"],
        "final_delta_pnorm_mean": final_row["delta_pnorm_mean"],
        "final_boundary_ratio_mean": final_row["boundary_ratio_mean"],
        "final_high_frequency_energy_ratio_mean": final_row["high_frequency_energy_ratio_mean"],
        "final_first_derivative_l2_mean": final_row["first_derivative_l2_mean"],
        "final_total_variation_mean": final_row["total_variation_mean"],
    }
    save_json(method_dir / "summary.json", summary)
    return summary


def collect_rows(root: Path, objective: str) -> dict[str, list[dict[str, str]]]:
    import csv

    out: dict[str, list[dict[str, str]]] = {}
    for method in CORE4:
        path = root / objective / method / "per_step_metrics.csv"
        with path.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        out[method] = rows
    return out


def fnum(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def plot_curves(root: Path, objective: str, metric: str, ylabel: str, filename: str, *, std: bool = True) -> Path:
    groups = collect_rows(root, objective)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for method, rows in groups.items():
        xs = np.asarray([int(r["k"]) for r in rows], dtype=float)
        ys = np.asarray([fnum(r.get(f"{metric}_mean")) for r in rows], dtype=float)
        ax.plot(xs, ys, color=COLORS[method], linestyle=LINESTYLES[method], linewidth=2, label=METHOD_LABELS[method])
        if std:
            sd = np.asarray([fnum(r.get(f"{metric}_std")) for r in rows], dtype=float)
            good = np.isfinite(ys) & np.isfinite(sd)
            if good.any():
                ax.fill_between(xs[good], ys[good] - sd[good], ys[good] + sd[good], color=COLORS[method], alpha=0.12, linewidth=0)
    ax.set_xlabel("optimization step k")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{objective} core-four baseline, eps=4 alpha=0.4 p=q=2")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = root / "figures" / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def first_boundary_hit(rows: list[dict[str, str]], threshold: float) -> dict[str, str] | None:
    for row in rows:
        value = fnum(row.get("boundary_ratio_mean"))
        if math.isfinite(value) and value >= threshold:
            return row
    return None


def plot_loss_curves_marked_0to100(root: Path, objective: str, filename: str) -> Path:
    groups = collect_rows(root, objective)
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    for method, all_rows in groups.items():
        rows = [r for r in all_rows if int(r["k"]) <= 100]
        xs = np.asarray([int(r["k"]) for r in rows], dtype=float)
        ys = np.asarray([fnum(r.get("optimized_loss_mean")) for r in rows], dtype=float)
        ax.plot(xs, ys, color=COLORS[method], linestyle=LINESTYLES[method], linewidth=2, label=METHOD_LABELS[method])
        for threshold, marker, _label in BOUNDARY_MARKERS:
            hit = first_boundary_hit(rows, threshold)
            if hit is None:
                continue
            x_hit = int(hit["k"])
            y_hit = fnum(hit.get("optimized_loss_mean"))
            if not math.isfinite(y_hit):
                continue
            if marker == "x":
                ax.scatter([x_hit], [y_hit], marker=marker, s=92, color=COLORS[method], linewidths=2.3, zorder=5)
            else:
                ax.scatter([x_hit], [y_hit], marker=marker, s=62, facecolors="none", edgecolors=COLORS[method], linewidths=1.8, zorder=5)
    ax.set_xlim(0, 100)
    ax.set_xlabel("optimization step k")
    ax.set_ylabel(f"{objective} mean")
    ax.set_title(f"{objective} mean vs step, boundary hits marked (0..100)")
    ax.grid(alpha=0.25)
    method_handles = [plt.Line2D([0], [0], color=COLORS[m], linestyle=LINESTYLES[m], linewidth=2, label=METHOD_LABELS[m]) for m in CORE4]
    marker_handles = []
    for _threshold, marker, label in BOUNDARY_MARKERS:
        if marker == "x":
            handle = plt.Line2D([0], [0], marker=marker, linestyle="None", color="black", markersize=8, label=label)
        else:
            handle = plt.Line2D([0], [0], marker=marker, linestyle="None", markerfacecolor="none", markeredgecolor="black", markersize=8, label=label)
        marker_handles.append(handle)
    ax.legend(handles=method_handles + marker_handles, fontsize=8, ncol=2)
    fig.tight_layout()
    out = root / "figures_0to100_marked_angles" / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_metric_0to100(root: Path, objective: str, metric: str, ylabel: str, filename: str, *, std: bool = True) -> Path:
    groups = collect_rows(root, objective)
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    for method, all_rows in groups.items():
        rows = [r for r in all_rows if int(r["k"]) <= 100]
        xs = np.asarray([int(r["k"]) for r in rows], dtype=float)
        ys = np.asarray([fnum(r.get(f"{metric}_mean")) for r in rows], dtype=float)
        ax.plot(xs, ys, color=COLORS[method], linestyle=LINESTYLES[method], linewidth=2, label=METHOD_LABELS[method])
        if std:
            sd = np.asarray([fnum(r.get(f"{metric}_std")) for r in rows], dtype=float)
            good = np.isfinite(ys) & np.isfinite(sd)
            if good.any():
                ax.fill_between(xs[good], ys[good] - sd[good], ys[good] + sd[good], color=COLORS[method], alpha=0.12, linewidth=0)
    ax.set_xlim(0, 100)
    ax.set_xlabel("optimization step k")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{objective}: {ylabel} (0..100)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = root / "figures_0to100_marked_angles" / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_boundary_ratio_marked_0to100(root: Path, objective: str, filename: str) -> Path:
    groups = collect_rows(root, objective)
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    for method, all_rows in groups.items():
        rows = [r for r in all_rows if int(r["k"]) <= 100]
        xs = np.asarray([int(r["k"]) for r in rows], dtype=float)
        ys = np.asarray([fnum(r.get("boundary_ratio_mean")) for r in rows], dtype=float)
        ax.plot(xs, ys, color=COLORS[method], linestyle=LINESTYLES[method], linewidth=2, label=METHOD_LABELS[method])
        for threshold, marker, _label in BOUNDARY_MARKERS:
            hit = first_boundary_hit(rows, threshold)
            if hit is None:
                continue
            ax.scatter([int(hit["k"])], [fnum(hit.get("boundary_ratio_mean"))], marker=marker, s=70, color=COLORS[method], zorder=5)
    for threshold, _marker, label in BOUNDARY_MARKERS:
        ax.axhline(threshold, color="black", alpha=0.13, linewidth=1)
        ax.text(100.5, threshold, label.split()[0], va="center", fontsize=8)
    ax.set_xlim(0, 104)
    ax.set_ylim(-0.02, 1.06)
    ax.set_xlabel("optimization step k")
    ax.set_ylabel("mean ||delta||_p / epsilon")
    ax.set_title(f"{objective}: boundary arrival markers (0..100)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = root / "figures_0to100_marked_angles" / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_final_deltas(root: Path, objective: str, filename: str) -> Path:
    fig, axes = plt.subplots(len(CORE4), 1, figsize=(10, 2.4 * len(CORE4)), sharex=True)
    x_axis = None
    for ax, method in zip(np.ravel(axes), CORE4):
        payload = np.load(root / objective / method / "trajectory_samples.npz")
        deltas = np.asarray(payload["delta"])
        dataset_index = np.asarray(payload["dataset_index"])
        final = deltas[-1, :, :, 0]
        if x_axis is None:
            x_axis = np.arange(final.shape[-1])
        for i, idx in enumerate(dataset_index.tolist()):
            ax.plot(x_axis, final[i], linewidth=1.15, label=f"idx {idx}")
        ax.set_title(METHOD_LABELS[method])
        ax.grid(alpha=0.2)
        ax.legend(ncol=4, fontsize=8)
    axes[-1].set_xlabel("spatial grid index")
    fig.suptitle(f"Final delta shapes for {objective}", y=0.995)
    fig.tight_layout()
    out = root / "figures" / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_delta_heatmap(root: Path, objective: str, filename: str) -> Path:
    fig, axes = plt.subplots(1, len(CORE4), figsize=(4.1 * len(CORE4), 4.8), sharey=True)
    for ax, method in zip(np.ravel(axes), CORE4):
        payload = np.load(root / objective / method / "trajectory_samples.npz")
        delta = np.asarray(payload["delta"])[:, 0, :, 0]
        ks = np.asarray(payload["k"])
        vmax = np.nanmax(np.abs(delta))
        vmax = vmax if np.isfinite(vmax) and vmax > 0 else 1.0
        im = ax.imshow(delta, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax, origin="lower")
        ax.set_title(METHOD_LABELS[method])
        ax.set_xlabel("spatial grid index")
        tick_positions = np.linspace(0, len(ks) - 1, min(6, len(ks)), dtype=int)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels([str(int(ks[i])) for i in tick_positions])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    axes[0].set_ylabel("saved step k, dataset index 0")
    fig.suptitle(f"Delta trajectory heatmaps for {objective}", y=0.995)
    fig.tight_layout()
    out = root / "figures" / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def summarize_objective(root: Path, objective: str) -> list[dict[str, Any]]:
    rows = []
    groups = collect_rows(root, objective)
    finals = {m: fnum(groups[m][-1]["optimized_loss_mean"]) for m in CORE4}
    best_final = max(v for v in finals.values() if math.isfinite(v))
    for method in CORE4:
        group = groups[method]
        final = group[-1]
        threshold = 0.95 * best_final
        k95 = "nan"
        for row in group:
            value = fnum(row.get("optimized_loss_mean"))
            if math.isfinite(value) and value >= threshold:
                k95 = int(row["k"])
                break
        rows.append(
            {
                "objective": objective,
                "method": method,
                "final_optimized_loss_mean": finals[method],
                "final_fraction_of_best": finals[method] / best_final if best_final else math.nan,
                "step_to_95pct_best_final": k95,
                "final_delta_pnorm_mean": fnum(final.get("delta_pnorm_mean")),
                "final_boundary_ratio_mean": fnum(final.get("boundary_ratio_mean")),
                "final_high_frequency_energy_ratio_mean": fnum(final.get("high_frequency_energy_ratio_mean")),
                "final_first_derivative_l2_mean": fnum(final.get("first_derivative_l2_mean")),
                "final_total_variation_mean": fnum(final.get("total_variation_mean")),
            }
        )
    return rows


def write_doc(root: Path, doc_path: Path, summary_rows: list[dict[str, Any]], figure_paths: list[Path], manifest: dict[str, Any]) -> None:
    def rel(path: Path) -> str:
        try:
            return str(path.resolve().relative_to(PROJECT_ROOT))
        except ValueError:
            return str(path)

    fields = [
        "objective",
        "method",
        "final_optimized_loss_mean",
        "final_fraction_of_best",
        "step_to_95pct_best_final",
        "final_delta_pnorm_mean",
        "final_high_frequency_energy_ratio_mean",
    ]
    lines = [
        "# Loss1/Loss2 Core-Four P2Q2 Baseline Visuals - 2026-05-21",
        "",
        "Status: completed on GPU from a new baseline run.",
        "",
        "## Source Data",
        "",
        f"- Output root: `{rel(root)}`",
        "- Per-method source files: `per_step_metrics.csv`, `per_sample_step_metrics.csv`, `trajectory_samples.npz`, and `final_delta.npz`.",
        f"- GPU evidence: `{manifest['gpu_runtime']['torch_device_name']}`, capability `{manifest['gpu_runtime']['torch_compute_capability']}`, PyTorch `{manifest['gpu_runtime']['torch_version']}`, JAX backend `{manifest['gpu_runtime']['jax']['backend']}`.",
        "",
        "## Settings",
        "",
        f"- FNO / 1D Burgers `nu=0.001`; batch size `{manifest['batch_size']}`, dataset indices `0..99`.",
        f"- `epsilon={manifest['epsilon']}`, `alpha={manifest['alpha']}`, `steps={manifest['steps']}`, `p=q=2`.",
        "- Methods: `raw_add`, `raw_replace`, `steepest_add`, `steepest_replace`.",
        "- `loss1_original` uses tiny random start `1e-6`; `loss2_original` uses zero start.",
        "",
        "## Summary Table",
        "",
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in summary_rows:
        vals = []
        for field in fields:
            value = row.get(field, "")
            if isinstance(value, float):
                vals.append("nan" if not math.isfinite(value) else f"{value:.4g}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    lines.extend(["", "## Figures", ""])
    for path in figure_paths:
        lines.append(f"- `{rel(path)}`")
    lines.extend(
        [
            "",
            "## Observed Evidence",
            "",
            "Observed from the generated CSVs and figures: replacement methods jump to the boundary immediately by construction, while additive methods grow the perturbation over many steps. For `p=q=2`, `raw_replace` and `steepest_replace` use the same normalized direction for these objectives and therefore are expected to nearly overlap.",
            "",
            "Inference should stay within this single baseline setting unless more epsilon/alpha or P/Q combinations are run.",
        ]
    )
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=PROJECT_ROOT / "forensics" / "loss1_loss2_core4_p2q2_baseline_20260521" / "fno_nu0p001_eps4_alpha0p4_batch100_steps300_p2_q2")
    parser.add_argument("--doc", type=Path, default=PROJECT_ROOT / "docs" / "loss1_loss2_core4_p2q2_baseline_visuals_20260521.md")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--dataset-indices", nargs="+", type=int, default=None)
    parser.add_argument("--epsilon", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--p", default="2")
    parser.add_argument("--q", default="2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-kind", choices=["fno", "deeponet"], default="fno")
    parser.add_argument("--burgers-test-path", type=Path, default=DEFAULT_BURGERS_TEST)
    parser.add_argument("--burgers-torch-checkpoint", type=Path, default=DEFAULT_BURGERS_MODEL_DIR / "checkpoints" / "pytorch_fno1d_500.pt")
    parser.add_argument("--deeponet-checkpoint", type=Path, default=DEFAULT_DEEPONET_RUN_DIR / "checkpoints" / "deeponet_burgers_nu0p01.pt")
    parser.add_argument("--deeponet-output-transform-stats", type=Path, default=DEFAULT_DEEPONET_RUN_DIR / "training_logs" / "output_transform_stats.npz")
    parser.add_argument("--burgers-nx", type=int, default=1024)
    parser.add_argument("--burgers-nu", type=float, default=0.001)
    parser.add_argument("--burgers-t-final", type=float, default=1.0)
    parser.add_argument("--burgers-dt", type=float, default=0.001)
    parser.add_argument("--burgers-domain", type=float, default=2.0)
    parser.add_argument("--burgers-jax-solver-dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--loss1-init", choices=["random", "zero"], default="random")
    parser.add_argument("--loss1-random-start-scale", type=float, default=1e-6)
    parser.add_argument("--trajectory-indices", nargs="+", type=int, default=[0, 7, 40, 47])
    parser.add_argument("--selected-steps", nargs="+", type=int, default=[0, 1, 2, 5, 10, 25, 50, 75, 100, 150, 200, 250, 300])
    parser.add_argument("--save-every", type=int, default=0, help="Use 1 to save every trajectory step; default saves selected steps only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.out_root.is_absolute():
        args.out_root = PROJECT_ROOT / args.out_root
    if not args.doc.is_absolute():
        args.doc = PROJECT_ROOT / args.doc
    args.out_root.mkdir(parents=True, exist_ok=True)
    args.p_order = parse_norm(args.p)
    args.q_order = parse_norm(args.q)
    args.model_label = format_model_label(args.model_kind)

    gpu_runtime = collect_gpu_evidence(args.device)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items() if key not in {"p_order", "q_order"}}
    config["p_order"] = norm_name(args.p_order)
    config["q_order"] = norm_name(args.q_order)
    save_json(args.out_root / "config.json", config)

    problem = Problem(args)
    summaries: list[dict[str, Any]] = []
    for objective in ("loss1_original", "loss2_original"):
        for method in CORE4:
            print(f"[run] {objective} {method}", flush=True)
            summaries.append(run_one(problem, objective, method, args.out_root))

    summary_rows: list[dict[str, Any]] = []
    for objective in ("loss1_original", "loss2_original"):
        summary_rows.extend(summarize_objective(args.out_root, objective))
    write_csv(args.out_root / "method_summary.csv", summary_rows)

    figures = []
    for objective in ("loss1_original", "loss2_original"):
        figures.append(plot_curves(args.out_root, objective, "optimized_loss", f"{objective} mean", f"{objective}_mean_vs_step.png"))
        figures.append(plot_curves(args.out_root, objective, "boundary_ratio", "mean ||delta||_p / epsilon", f"{objective}_boundary_ratio_vs_step.png", std=False))
        figures.append(plot_curves(args.out_root, objective, "high_frequency_energy_ratio", "mean high-frequency energy ratio", f"{objective}_high_frequency_ratio_vs_step.png"))
        figures.append(plot_final_deltas(args.out_root, objective, f"{objective}_final_delta_selected_indices.png"))
        figures.append(plot_delta_heatmap(args.out_root, objective, f"{objective}_delta_heatmap_index0.png"))
        figures.append(plot_loss_curves_marked_0to100(args.out_root, objective, f"{objective}_mean_vs_step_0to100_boundary_marked.png"))
        figures.append(plot_boundary_ratio_marked_0to100(args.out_root, objective, f"{objective}_boundary_ratio_0to100_marked.png"))
        figures.append(
            plot_metric_0to100(
                args.out_root,
                objective,
                "delta_prev_angle_degrees",
                "mean angle(delta_k, delta_{k-1}) degrees",
                f"{objective}_delta_prev_angle_degrees_0to100.png",
            )
        )
        figures.append(
            plot_metric_0to100(
                args.out_root,
                objective,
                "direction_prev_angle_degrees",
                "mean angle(direction_k, direction_{k-1}) degrees",
                f"{objective}_direction_prev_angle_degrees_0to100.png",
            )
        )
        figures.append(
            plot_metric_0to100(
                args.out_root,
                objective,
                "delta_direction_angle_degrees",
                "mean angle(delta_k, direction_k) degrees",
                f"{objective}_delta_direction_angle_degrees_0to100.png",
            )
        )

    manifest = {
        "status": "completed",
        "experiment": "loss1_loss2_core4_p2q2_baseline_visuals",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_runtime": gpu_runtime,
        "out_root": str(args.out_root),
        "doc": str(args.doc),
        "objectives": ["loss1_original", "loss2_original"],
        "methods": list(CORE4),
        "epsilon": args.epsilon,
        "alpha": args.alpha,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "start_index": args.start_index,
        "p_order": norm_name(args.p_order),
        "q_order": norm_name(args.q_order),
        "figure_paths": [str(path) for path in figures],
        "summary_rows": len(summary_rows),
    }
    save_json(args.out_root / "manifest.json", manifest)
    write_doc(args.out_root, args.doc, summary_rows, figures, manifest)
    print(f"[done] wrote {args.out_root}")
    print(f"[done] wrote {args.doc}")


if __name__ == "__main__":
    main()
