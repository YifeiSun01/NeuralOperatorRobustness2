#!/usr/bin/env python3
"""Batch attack all 27 tags and record only per-step loss values.

The 27 tags are:
  optimized loss in {loss1, loss2, loss3}
  x objective variant in {original, increment_ratio, regularized}
  x method in {pgd, lp_steepest_pgd, generalized_power}

For every tag and every step, this records all 9 evaluated loss objectives for
each sample in the batch, plus mean/std summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("DDE_BACKEND", "pytorch")

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.attack_framework_matrix import (
    DEFAULT_BURGERS_MODEL_DIR,
    DEFAULT_BURGERS_TEST,
    load_burgers_torch_model,
    make_burgers_jax_solver,
    make_jax_torch_bridge,
    sync_torch,
)
from loss_attack_common import save_json


DEFAULT_DEEPONET_RUN_DIR = PROJECT_ROOT / "deeponet_training_runs" / "burgers_nu0p01_deeponet_lu_ref_50k"
DEFAULT_DEEPONET_TEST = (
    PROJECT_ROOT
    / "1D_Burgers"
    / "datasets"
    / "1D"
    / "Burgers"
    / "batched_exponax_splits"
    / "dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.01_t1.0_seed45"
    / "dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.01_t1.0_seed45_test.pt"
)
LOSSES = ("loss1", "loss2", "loss3")
VARIANTS = ("original", "increment_ratio", "regularized")
METHODS = ("pgd", "lp_steepest_pgd", "generalized_power")
EVAL_KEYS = tuple(f"{loss}_{variant}" for loss in LOSSES for variant in VARIANTS)
EPS = 1e-12


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


def finite_mean_std(values: Any) -> tuple[float, float, int, int]:
    arr = np.asarray(values, dtype=np.float64)
    total_count = int(arr.size)
    finite = np.isfinite(arr)
    finite_count = int(np.count_nonzero(finite))
    if finite_count == 0:
        return float("nan"), float("nan"), finite_count, total_count
    finite_values = arr[finite]
    return (
        float(np.mean(finite_values)),
        float(np.std(finite_values, ddof=0)),
        finite_count,
        total_count,
    )


def add_finite_summary(summary: dict[str, Any], prefix: str, values: Any) -> None:
    mean, std, finite_count, total_count = finite_mean_std(values)
    summary[f"{prefix}_mean"] = mean
    summary[f"{prefix}_std"] = std
    summary[f"{prefix}_finite_count"] = finite_count
    summary[f"{prefix}_nonfinite_count"] = total_count - finite_count


def parse_norm(value: str) -> float:
    text = str(value).lower()
    if text in {"inf", "linf", "infinity"}:
        return float("inf")
    return float(text)


def norm_name(value: float) -> str:
    return "inf" if math.isinf(value) else f"{value:g}"


def batch_norm(x, order: float):
    import torch

    flat = x.reshape(x.shape[0], -1)
    if math.isinf(order):
        return flat.abs().max(dim=1).values
    return torch.linalg.vector_norm(flat, ord=order, dim=1)


def normalize_to_p_ball(v, p: float):
    import torch

    flat = v.reshape(v.shape[0], -1)
    if math.isinf(p):
        denom = flat.abs().max(dim=1, keepdim=True).values.clamp_min(EPS)
        return (flat / denom).reshape_as(v)
    denom = torch.linalg.vector_norm(flat, ord=p, dim=1, keepdim=True).clamp_min(EPS)
    return (flat / denom).reshape_as(v)


def project_delta(delta, epsilon: float, p: float):
    import torch

    flat = delta.reshape(delta.shape[0], -1)
    if math.isinf(p):
        return torch.clamp(delta, -epsilon, epsilon)
    norms = torch.linalg.vector_norm(flat, ord=p, dim=1, keepdim=True).clamp_min(EPS)
    scale = torch.clamp(float(epsilon) / norms, max=1.0)
    return (flat * scale).reshape_as(delta)


def sanitize_tensor(x):
    import torch

    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def rescale_delta_to_boundary(delta, epsilon: float, p: float):
    import torch

    flat = delta.reshape(delta.shape[0], -1)
    if math.isinf(p):
        norms = flat.abs().max(dim=1, keepdim=True).values
    else:
        norms = torch.linalg.vector_norm(flat, ord=p, dim=1, keepdim=True)
    valid = norms > EPS
    scale = torch.where(valid, float(epsilon) / norms.clamp_min(EPS), torch.zeros_like(norms))
    boundary = (flat * scale).reshape_as(delta)
    boundary_norm = batch_norm(boundary, p)
    return boundary, norms.squeeze(1), boundary_norm, scale.squeeze(1), valid.squeeze(1)


def random_delta_like(x, epsilon: float, p: float, scale: float, seed: int):
    import torch

    generator = torch.Generator(device=x.device)
    generator.manual_seed(seed)
    direction = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
    direction = normalize_to_p_ball(direction, p)
    return project_delta(float(epsilon) * float(scale) * direction, epsilon, p)


def steepest_direction(grad, p: float):
    import torch

    flat = grad.reshape(grad.shape[0], -1)
    if math.isinf(p):
        return torch.sign(grad)
    if p == 1.0:
        out = torch.zeros_like(flat)
        idx = torch.argmax(flat.abs(), dim=1, keepdim=True)
        out.scatter_(1, idx, torch.sign(torch.gather(flat, 1, idx)))
        return out.reshape_as(grad)
    p_dual = p / (p - 1.0)
    mapped = torch.sign(grad) * torch.clamp(grad.abs(), min=1e-30).pow(p_dual - 1.0)
    return normalize_to_p_ball(mapped, p)


def load_burgers_batch(path: Path, start: int, batch_size: int) -> np.ndarray:
    import torch

    data = torch.load(path, map_location="cpu", weights_only=False)
    x = data["x"][start : start + batch_size].float().numpy()
    return x[..., None].astype(np.float32)


def format_model_label(model_kind: str) -> str:
    return "DeepONet/default net" if model_kind == "deeponet" else "FNO"


class DeepONetBurgersModel:
    def __init__(self, checkpoint_path: Path, stats_path: Path, device, domain: float):
        import deepxde as dde
        import torch

        from train_burgers_deeponet_deepxde import make_grid, periodic_features_torch

        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = payload.get("config", {})
        branch_layers = list(config.get("branch_layers", [1024, 128, 128, 128, 128]))
        trunk_layers = list(config.get("trunk_layers", [4, 128, 128, 128]))
        activation = config.get("activation", "tanh")
        kernel_initializer = config.get("kernel_initializer", "Glorot normal")

        self.net = dde.nn.DeepONetCartesianProd(branch_layers, trunk_layers, activation, kernel_initializer).to(device)
        if config.get("periodic_trunk", True):
            self.net.apply_feature_transform(lambda x: periodic_features_torch(x, domain))

        if config.get("output_transform", True):
            stats = np.load(stats_path)
            y_mean_t = torch.as_tensor(stats["y_mean"], device=device, dtype=torch.float32)
            y_std_t = torch.as_tensor(stats["y_std"], device=device, dtype=torch.float32)

            def output_transform(_inputs: tuple[torch.Tensor, torch.Tensor], outputs: torch.Tensor) -> torch.Tensor:
                return outputs * y_std_t.to(outputs.device) + y_mean_t.to(outputs.device)

            self.net.apply_output_transform(output_transform)

        self.net.load_state_dict(payload["model_state_dict"])
        self.net.eval()
        for param in self.net.parameters():
            param.requires_grad_(False)

        nx = int(config.get("nx", branch_layers[0]))
        trunk = make_grid(nx, float(config.get("domain", domain)))
        self.trunk = torch.as_tensor(trunk, device=device, dtype=torch.float32)

    def __call__(self, x):
        import torch

        y = self.net((x[..., 0].to(dtype=torch.float32), self.trunk))
        return y[..., None].to(dtype=x.dtype)


def load_model(args: argparse.Namespace, device):
    if args.model_kind == "fno":
        return load_burgers_torch_model(args.burgers_torch_checkpoint, device)
    if args.model_kind == "deeponet":
        return DeepONetBurgersModel(args.deeponet_checkpoint, args.deeponet_output_transform_stats, device, args.burgers_domain)
    raise ValueError(args.model_kind)


class BatchProblem:
    def __init__(self, args: argparse.Namespace):
        import torch

        self.args = args
        self.device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.bridge = make_jax_torch_bridge()
        x_np = load_burgers_batch(args.burgers_test_path, args.start_index, args.batch_size)
        self.x0 = torch.as_tensor(x_np, device=self.device, dtype=torch.float32)
        self.model = load_model(args, self.device)
        self.jax_solver = make_burgers_jax_solver(args)
        self.f0 = self.model_forward(self.x0).detach()
        self.g0 = self.solver_forward(self.x0, allow_grad=False).detach()
        self.baseline = {
            "loss1": torch.zeros(self.x0.shape[0], device=self.device, dtype=self.x0.dtype),
            "loss2": batch_norm(self.f0 - self.g0, args.q_order).detach(),
            "loss3": batch_norm(self.f0 - self.g0, args.q_order).detach(),
        }

    def model_forward(self, x):
        return self.model(x.to(dtype=x.dtype))

    def solver_forward(self, x, allow_grad: bool):
        y = self.bridge(x, self.jax_solver, "solver")
        return y if allow_grad else y.detach()


def raw_losses(problem: BatchProblem, x_adv, *, allow_solver_grad_for_loss3: bool) -> dict[str, Any]:
    f_adv = problem.model_forward(x_adv)
    g_adv = problem.solver_forward(x_adv, allow_grad=allow_solver_grad_for_loss3)
    q = problem.args.q_order
    return {
        "loss1": batch_norm(f_adv - problem.f0, q),
        "loss2": batch_norm(f_adv - problem.g0, q),
        "loss3": batch_norm(f_adv - g_adv, q),
    }


def objective_values(problem: BatchProblem, losses: dict[str, Any], delta) -> dict[str, Any]:
    delta_norm = batch_norm(delta, problem.args.p_order)
    out: dict[str, Any] = {}
    for loss in LOSSES:
        base = problem.baseline[loss]
        value = losses[loss]
        out[f"{loss}_original"] = value
        out[f"{loss}_increment_ratio"] = (value - base) / (delta_norm + problem.args.eta)
        out[f"{loss}_regularized"] = value - problem.args.regularization_c * delta_norm
    return out


def optimized_objective(problem: BatchProblem, x_adv, optimized_loss: str, variant: str):
    delta = x_adv - problem.x0
    losses = raw_losses(problem, x_adv, allow_solver_grad_for_loss3=(optimized_loss == "loss3"))
    values = objective_values(problem, losses, delta)
    return values[f"{optimized_loss}_{variant}"]


def evaluate_values_np(problem: BatchProblem, delta) -> dict[str, np.ndarray]:
    import torch

    with torch.no_grad():
        x_adv = (problem.x0 + delta).detach()
        losses = raw_losses(problem, x_adv, allow_solver_grad_for_loss3=False)
        values = objective_values(problem, losses, delta)
    return {key: values[key].detach().cpu().numpy().astype(np.float32) for key in EVAL_KEYS}


def save_final_delta_diagnostics(
    problem: BatchProblem,
    *,
    tag: str,
    optimized_loss: str,
    variant: str,
    method: str,
    initial_delta: str,
    final_delta,
    out_dir: Path,
) -> dict[str, Any]:
    final_delta = final_delta.detach()
    boundary_delta, final_norm, boundary_norm, boundary_scale, boundary_valid = rescale_delta_to_boundary(
        final_delta,
        problem.args.epsilon,
        problem.args.p_order,
    )
    final_values = evaluate_values_np(problem, final_delta)
    boundary_values = evaluate_values_np(problem, boundary_delta)

    final_norm_np = final_norm.detach().cpu().numpy().astype(np.float32)
    boundary_norm_np = boundary_norm.detach().cpu().numpy().astype(np.float32)
    boundary_scale_np = boundary_scale.detach().cpu().numpy().astype(np.float32)
    boundary_valid_np = boundary_valid.detach().cpu().numpy().astype(bool)
    final_delta_np = final_delta.detach().cpu().numpy().astype(np.float32)
    boundary_delta_np = boundary_delta.detach().cpu().numpy().astype(np.float32)

    rows: list[dict[str, Any]] = []
    for i in range(problem.args.batch_size):
        row: dict[str, Any] = {
            "tag": tag,
            "optimized_loss": optimized_loss,
            "objective_variant": variant,
            "attack_method": method,
            "initial_delta": initial_delta,
            "sample_position": i,
            "dataset_index": problem.args.start_index + i,
            "epsilon": problem.args.epsilon,
            "final_delta_pnorm": float(final_norm_np[i]),
            "boundary_delta_pnorm": float(boundary_norm_np[i]),
            "boundary_rescale_factor": float(boundary_scale_np[i]),
            "boundary_rescale_valid": bool(boundary_valid_np[i]),
        }
        for key in EVAL_KEYS:
            row[f"final_{key}"] = float(final_values[key][i])
            row[f"boundary_{key}"] = float(boundary_values[key][i])
        rows.append(row)

    write_csv(out_dir / "final_delta_diagnostics.csv", rows)
    np.savez_compressed(
        out_dir / "final_delta_diagnostics.npz",
        sample_position=np.arange(problem.args.batch_size, dtype=np.int64),
        dataset_index=np.arange(problem.args.start_index, problem.args.start_index + problem.args.batch_size, dtype=np.int64),
        final_delta_pnorm=final_norm_np,
        boundary_delta_pnorm=boundary_norm_np,
        boundary_rescale_factor=boundary_scale_np,
        boundary_rescale_valid=boundary_valid_np,
        **{f"final_{key}": value for key, value in final_values.items()},
        **{f"boundary_{key}": value for key, value in boundary_values.items()},
    )
    np.savez_compressed(
        out_dir / "final_delta.npz",
        sample_position=np.arange(problem.args.batch_size, dtype=np.int64),
        dataset_index=np.arange(problem.args.start_index, problem.args.start_index + problem.args.batch_size, dtype=np.int64),
        final_delta=final_delta_np,
        boundary_delta=boundary_delta_np,
        final_delta_pnorm=final_norm_np,
        boundary_delta_pnorm=boundary_norm_np,
        boundary_rescale_factor=boundary_scale_np,
        boundary_rescale_valid=boundary_valid_np,
    )

    summary: dict[str, Any] = {
        "tag": tag,
        "optimized_loss": optimized_loss,
        "objective_variant": variant,
        "attack_method": method,
        "initial_delta": initial_delta,
        "epsilon": problem.args.epsilon,
        "boundary_rescale_valid_count": int(np.count_nonzero(boundary_valid_np)),
    }
    add_finite_summary(summary, "final_delta_pnorm", final_norm_np)
    add_finite_summary(summary, "boundary_delta_pnorm", boundary_norm_np)
    add_finite_summary(summary, "boundary_rescale_factor", boundary_scale_np)
    for key in EVAL_KEYS:
        add_finite_summary(summary, f"final_{key}", final_values[key])
        add_finite_summary(summary, f"boundary_{key}", boundary_values[key])
    save_json(out_dir / "final_delta_summary.json", summary)
    return summary


def summarize_step(
    *,
    tag: str,
    k: int,
    optimized_loss: str,
    variant: str,
    method: str,
    initial_delta: str,
    values_np: dict[str, np.ndarray],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "tag": tag,
        "k": k,
        "optimized_loss": optimized_loss,
        "objective_variant": variant,
        "attack_method": method,
        "initial_delta": initial_delta,
    }
    for key in EVAL_KEYS:
        arr = values_np[key].astype(np.float64)
        mean, std, finite_count, total_count = finite_mean_std(arr)
        row[f"{key}_mean"] = mean
        row[f"{key}_std"] = std
        row[f"{key}_finite_count"] = finite_count
        row[f"{key}_nonfinite_count"] = total_count - finite_count
    return row


def run_one(problem: BatchProblem, optimized_loss: str, variant: str, method: str, initial_delta: str, out_dir: Path) -> None:
    import torch

    args = problem.args
    tag = f"{optimized_loss}_{variant}_{method}"
    if optimized_loss == "loss1" and args.loss1_initial_delta == "both":
        tag = f"{tag}_init_{initial_delta}"
    if optimized_loss == "loss1" and initial_delta == "random":
        delta = random_delta_like(problem.x0, args.epsilon, args.p_order, args.loss1_random_start_scale, args.seed)
    else:
        delta = torch.zeros_like(problem.x0)

    rows: list[dict[str, Any]] = []
    per_sample: dict[str, list[np.ndarray]] = {key: [] for key in EVAL_KEYS}
    if torch.cuda.is_available() and problem.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(problem.device)
        torch.cuda.synchronize(problem.device)
    start = time.perf_counter()
    for k in range(args.steps + 1):
        delta = sanitize_tensor(delta)
        x_adv = (problem.x0 + delta).detach().requires_grad_(k < args.steps)
        objective = optimized_objective(problem, x_adv, optimized_loss, variant)
        losses = raw_losses(problem, x_adv, allow_solver_grad_for_loss3=False)
        eval_values = objective_values(problem, losses, x_adv - problem.x0)
        values_np = {key: eval_values[key].detach().cpu().numpy() for key in EVAL_KEYS}
        for key, arr in values_np.items():
            per_sample[key].append(arr.astype(np.float32))
        rows.append(
            summarize_step(
                tag=tag,
                k=k,
                optimized_loss=optimized_loss,
                variant=variant,
                method=method,
                initial_delta=initial_delta,
                values_np=values_np,
            )
        )
        if k >= args.steps:
            break

        # Sum keeps each sample's gradient scale identical to running the same
        # attack on that sample alone.  A mean would divide raw PGD gradients by
        # batch_size and artificially slow projected gradient descent.
        batch_objective = sanitize_tensor(objective).sum()
        batch_objective.backward()
        grad = sanitize_tensor(x_adv.grad.detach())
        with torch.no_grad():
            if method == "pgd":
                direction = grad
                delta = project_delta(delta + args.alpha * direction, args.epsilon, args.p_order)
            elif method == "lp_steepest_pgd":
                direction = steepest_direction(grad, args.p_order)
                delta = project_delta(delta + args.alpha * direction, args.epsilon, args.p_order)
            elif method == "generalized_power":
                direction = steepest_direction(grad, args.p_order)
                delta = project_delta(args.epsilon * direction, args.epsilon, args.p_order)
            else:
                raise ValueError(method)
            delta = sanitize_tensor(delta)

    sync_torch(torch, problem.device)
    runtime_seconds = time.perf_counter() - start
    torch_peak_allocated_mib = float("nan")
    torch_peak_reserved_mib = float("nan")
    if torch.cuda.is_available() and problem.device.type == "cuda":
        torch_peak_allocated_mib = float(torch.cuda.max_memory_allocated(problem.device) / 1024**2)
        torch_peak_reserved_mib = float(torch.cuda.max_memory_reserved(problem.device) / 1024**2)

    out_dir.mkdir(parents=True, exist_ok=True)
    final_diagnostics = save_final_delta_diagnostics(
        problem,
        tag=tag,
        optimized_loss=optimized_loss,
        variant=variant,
        method=method,
        initial_delta=initial_delta,
        final_delta=delta,
        out_dir=out_dir,
    )
    write_csv(out_dir / "loss_stats.csv", rows)
    np.savez_compressed(
        out_dir / "loss_values.npz",
        k=np.arange(args.steps + 1, dtype=np.int64),
        **{key: np.stack(values, axis=0) for key, values in per_sample.items()},
    )
    summary = {
        "tag": tag,
        "model_kind": args.model_kind,
        "model_label": args.model_label,
        "optimized_loss": optimized_loss,
        "objective_variant": variant,
        "attack_method": method,
        "initial_delta": initial_delta,
        "batch_size": args.batch_size,
        "start_index": args.start_index,
        "epsilon": args.epsilon,
        "alpha": args.alpha,
        "steps": args.steps,
        "p": norm_name(args.p_order),
        "q": norm_name(args.q_order),
        "eta": args.eta,
        "regularization_c": args.regularization_c,
        "burgers_nu": args.burgers_nu,
        "burgers_test_path": args.burgers_test_path,
        "burgers_torch_checkpoint": args.burgers_torch_checkpoint,
        "deeponet_checkpoint": args.deeponet_checkpoint,
        "deeponet_output_transform_stats": args.deeponet_output_transform_stats,
        "loss1_initial_delta": args.loss1_initial_delta,
        "loss1_random_start_scale": args.loss1_random_start_scale,
        "final_delta_pnorm_mean": final_diagnostics["final_delta_pnorm_mean"],
        "final_delta_pnorm_std": final_diagnostics["final_delta_pnorm_std"],
        "final_delta_pnorm_finite_count": final_diagnostics["final_delta_pnorm_finite_count"],
        "final_delta_pnorm_nonfinite_count": final_diagnostics["final_delta_pnorm_nonfinite_count"],
        "boundary_delta_pnorm_mean": final_diagnostics["boundary_delta_pnorm_mean"],
        "boundary_delta_pnorm_std": final_diagnostics["boundary_delta_pnorm_std"],
        "boundary_delta_pnorm_finite_count": final_diagnostics["boundary_delta_pnorm_finite_count"],
        "boundary_delta_pnorm_nonfinite_count": final_diagnostics["boundary_delta_pnorm_nonfinite_count"],
        "boundary_rescale_factor_mean": final_diagnostics["boundary_rescale_factor_mean"],
        "boundary_rescale_factor_std": final_diagnostics["boundary_rescale_factor_std"],
        "boundary_rescale_factor_finite_count": final_diagnostics["boundary_rescale_factor_finite_count"],
        "boundary_rescale_factor_nonfinite_count": final_diagnostics["boundary_rescale_factor_nonfinite_count"],
        "runtime_seconds": runtime_seconds,
        "torch_peak_allocated_mib": torch_peak_allocated_mib,
        "torch_peak_reserved_mib": torch_peak_reserved_mib,
        "generalized_power_radius": "fixed_epsilon",
    }
    save_json(out_dir / "summary.json", summary)
    print(f"[done] {tag} -> {out_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=Path("results/three_loss_batch100_loss_only"))
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--losses", nargs="+", choices=LOSSES, default=list(LOSSES))
    parser.add_argument("--epsilon", type=float, default=8.0)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--p", default="2")
    parser.add_argument("--q", default="2")
    parser.add_argument("--eta", type=float, default=1e-6)
    parser.add_argument("--regularization-c", type=float, default=1.0)
    parser.add_argument("--loss1-initial-delta", choices=["zero", "random", "both"], default="random")
    parser.add_argument("--loss1-random-start-scale", type=float, default=1e-6)
    parser.add_argument("--loss1-original-random-start-scale", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-kind", choices=["fno", "deeponet"], default="fno")
    parser.add_argument("--model-label", default=None)
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
    args = parser.parse_args()
    if args.loss1_original_random_start_scale is not None:
        args.loss1_random_start_scale = args.loss1_original_random_start_scale
    if args.model_label is None:
        args.model_label = format_model_label(args.model_kind)
    args.p_order = parse_norm(args.p)
    args.q_order = parse_norm(args.q)

    problem = BatchProblem(args)
    config = vars(args).copy()
    config["p_order"] = norm_name(args.p_order)
    config["q_order"] = norm_name(args.q_order)
    save_json(args.out_root / "config.json", config)

    for optimized_loss in args.losses:
        for variant in VARIANTS:
            for method in METHODS:
                initial_modes = [args.loss1_initial_delta] if optimized_loss == "loss1" else ["zero"]
                if optimized_loss == "loss1" and args.loss1_initial_delta == "both":
                    initial_modes = ["random", "zero"]
                for initial_delta in initial_modes:
                    tag = f"{optimized_loss}_{variant}_{method}"
                    if optimized_loss == "loss1" and args.loss1_initial_delta == "both":
                        tag = f"{tag}_init_{initial_delta}"
                    run_one(problem, optimized_loss, variant, method, initial_delta, args.out_root / tag)


if __name__ == "__main__":
    main()
