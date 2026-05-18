#!/usr/bin/env python3
"""Run and visualize the loss3 direction/proposal optimizer ablation.

This script is intentionally self-contained: it runs the optimizer variants,
records per-step metrics, stores representative delta trajectories, builds the
comparison tables, and renders the static/GIF visualizations used to judge the
experiment.

It does not silently fall back to CPU.  A real run requires CUDA and records the
GPU/JAX/PyTorch evidence in manifest.json before the first optimization step.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import itertools
import json
import math
import os
import shutil
import subprocess
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
    EVAL_KEYS,
    EPS,
    add_finite_summary,
    batch_norm,
    finite_mean_std,
    format_model_label,
    load_model,
    norm_name,
    normalize_to_p_ball,
    objective_values,
    parse_norm,
    project_delta,
    raw_losses,
    rescale_delta_to_boundary,
    sanitize_tensor,
    steepest_direction,
    write_csv,
)


DEFAULT_TRAJECTORY_INDICES = (0, 7, 40, 47, 115)
DEFAULT_SELECTED_STEPS = (0, 1, 2, 5, 10, 25, 50, 100)
LOSS3_KEYS = ("loss3_q", "loss3_l2", "loss3_linf")


@dataclasses.dataclass(frozen=True)
class MethodSpec:
    label: str
    direction_rule: str
    proposal_rule: str
    power_variant: str = "none"
    official_alias: str = ""

    @property
    def uses_power_state(self) -> bool:
        return self.direction_rule in {
            "power_pure_jvp_vjp",
            "power_generalized_pq",
            "power_affine_jvp_vjp",
        }


@dataclasses.dataclass
class MethodResult:
    spec: MethodSpec
    method_dir: Path
    per_step_rows: list[dict[str, Any]]
    per_sample_rows: list[dict[str, Any]]
    smoothness_rows: list[dict[str, Any]]
    final_delta: np.ndarray
    trajectory: dict[str, Any] | None
    direction_trace: dict[str, Any] | None


METHOD_PRESETS: dict[str, list[str]] = {
    "official3": ["pgd", "lp_steepest_pgd", "generalized_power"],
    "main": [
        "raw_add",
        "unit_raw_add",
        "raw_replace",
        "steepest_add",
        "steepest_replace",
        "power_add__objective_gradient",
        "power_replace__objective_gradient",
        "power_add__pure_jvp_vjp",
        "power_replace__pure_jvp_vjp",
        "power_add__affine_jvp_vjp",
        "power_replace__affine_jvp_vjp",
    ],
    "pq_key": [
        "raw_add",
        "unit_raw_add",
        "steepest_add",
        "steepest_replace",
        "power_replace__objective_gradient",
        "power_replace__pure_jvp_vjp",
        "power_replace__generalized_pq",
    ],
    "additive_alpha": ["raw_add", "unit_raw_add", "steepest_add", "power_add__objective_gradient"],
    "replacement_key": [
        "raw_replace",
        "steepest_replace",
        "power_replace__objective_gradient",
        "power_replace__pure_jvp_vjp",
        "power_replace__affine_jvp_vjp",
    ],
}


METHOD_ALIASES: dict[str, MethodSpec] = {
    "pgd": MethodSpec("pgd", "raw_gradient", "additive", official_alias="pgd"),
    "lp_steepest_pgd": MethodSpec(
        "lp_steepest_pgd",
        "lp_steepest",
        "additive",
        official_alias="lp_steepest_pgd",
    ),
    # This mirrors tools/run_batch_three_loss_loss_only.py: steepest direction,
    # replacement/boundary proposal.
    "generalized_power": MethodSpec(
        "generalized_power",
        "lp_steepest",
        "replacement",
        power_variant="steepest_gradient",
        official_alias="generalized_power",
    ),
    "raw_add": MethodSpec("raw_add", "raw_gradient", "additive"),
    "unit_raw_add": MethodSpec("unit_raw_add", "unit_raw_gradient", "additive"),
    "raw_replace": MethodSpec("raw_replace", "unit_raw_gradient", "replacement"),
    "steepest_add": MethodSpec("steepest_add", "lp_steepest", "additive"),
    "steepest_replace": MethodSpec("steepest_replace", "lp_steepest", "replacement"),
    "power_add__steepest_gradient": MethodSpec(
        "power_add__steepest_gradient",
        "lp_steepest",
        "additive",
        power_variant="steepest_gradient",
    ),
    "power_replace__steepest_gradient": MethodSpec(
        "power_replace__steepest_gradient",
        "lp_steepest",
        "replacement",
        power_variant="steepest_gradient",
    ),
    "power_add__objective_gradient": MethodSpec(
        "power_add__objective_gradient",
        "power_objective_gradient",
        "additive",
        power_variant="objective_gradient",
    ),
    "power_replace__objective_gradient": MethodSpec(
        "power_replace__objective_gradient",
        "power_objective_gradient",
        "replacement",
        power_variant="objective_gradient",
    ),
    "power_add__pure_jvp_vjp": MethodSpec(
        "power_add__pure_jvp_vjp",
        "power_pure_jvp_vjp",
        "additive",
        power_variant="pure_jvp_vjp",
    ),
    "power_replace__pure_jvp_vjp": MethodSpec(
        "power_replace__pure_jvp_vjp",
        "power_pure_jvp_vjp",
        "replacement",
        power_variant="pure_jvp_vjp",
    ),
    "power_add__generalized_pq": MethodSpec(
        "power_add__generalized_pq",
        "power_generalized_pq",
        "additive",
        power_variant="generalized_pq",
    ),
    "power_replace__generalized_pq": MethodSpec(
        "power_replace__generalized_pq",
        "power_generalized_pq",
        "replacement",
        power_variant="generalized_pq",
    ),
    "power_add__affine_jvp_vjp": MethodSpec(
        "power_add__affine_jvp_vjp",
        "power_affine_jvp_vjp",
        "additive",
        power_variant="affine_jvp_vjp",
    ),
    "power_replace__affine_jvp_vjp": MethodSpec(
        "power_replace__affine_jvp_vjp",
        "power_affine_jvp_vjp",
        "replacement",
        power_variant="affine_jvp_vjp",
    ),
}


def parse_methods(values: list[str]) -> list[MethodSpec]:
    names: list[str] = []
    for value in values:
        if value in METHOD_PRESETS:
            names.extend(METHOD_PRESETS[value])
        else:
            names.append(value)
    out: list[MethodSpec] = []
    seen: set[str] = set()
    for name in names:
        if name not in METHOD_ALIASES:
            valid = sorted(set(METHOD_ALIASES) | set(METHOD_PRESETS))
            raise ValueError(f"Unknown method/preset {name!r}. Valid: {valid}")
        spec = METHOD_ALIASES[name]
        if spec.label in seen:
            continue
        out.append(spec)
        seen.add(spec.label)
    return out


def flatten_np(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim >= 2 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return arr.reshape(arr.shape[0], -1) if arr.ndim > 1 else arr.reshape(1, -1)


def tensor_to_numpy(x) -> np.ndarray:
    return x.detach().cpu().numpy()


def batch_cosine_torch(a, b):
    import torch

    a_flat = a.reshape(a.shape[0], -1)
    b_flat = b.reshape(b.shape[0], -1)
    denom = torch.linalg.vector_norm(a_flat, dim=1) * torch.linalg.vector_norm(b_flat, dim=1)
    return torch.sum(a_flat * b_flat, dim=1) / denom.clamp_min(EPS)


def batch_l2_torch(x):
    import torch

    return torch.linalg.vector_norm(x.reshape(x.shape[0], -1), ord=2, dim=1)


def batch_linf_torch(x):
    return x.reshape(x.shape[0], -1).abs().max(dim=1).values


def np_l2_cosine(a: np.ndarray, b: np.ndarray) -> float:
    af = np.asarray(a, dtype=np.float64).reshape(-1)
    bf = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(af) * np.linalg.norm(bf))
    if denom <= EPS:
        return float("nan")
    return float(np.dot(af, bf) / denom)


def finite_stats_dict(prefix: str, values: Any) -> dict[str, Any]:
    mean, std, finite_count, total_count = finite_mean_std(values)
    return {
        f"{prefix}_mean": mean,
        f"{prefix}_std": std,
        f"{prefix}_finite_count": finite_count,
        f"{prefix}_nonfinite_count": total_count - finite_count,
    }


def load_burgers_samples(path: Path, start: int, batch_size: int, dataset_indices: list[int] | None) -> tuple[np.ndarray, np.ndarray]:
    import torch

    data = torch.load(path, map_location="cpu", weights_only=False)
    all_x = data["x"]
    if dataset_indices:
        indices = np.asarray(dataset_indices, dtype=np.int64)
        x = all_x[indices].float().numpy()
    else:
        indices = np.arange(start, start + batch_size, dtype=np.int64)
        x = all_x[start : start + batch_size].float().numpy()
    return x[..., None].astype(np.float32), indices


class AblationProblem:
    def __init__(self, args: argparse.Namespace):
        import torch

        self.args = args
        self.device = torch.device(args.device)
        x_np, dataset_indices = load_burgers_samples(args.burgers_test_path, args.start_index, args.batch_size, args.dataset_indices)
        args.batch_size = int(x_np.shape[0])
        self.dataset_indices = dataset_indices
        self.x0 = torch.as_tensor(x_np, device=self.device, dtype=torch.float32)
        self.bridge = make_jax_torch_bridge()
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


def collect_gpu_evidence(device_arg: str) -> dict[str, Any]:
    import torch

    if not str(device_arg).startswith("cuda"):
        raise RuntimeError("GPU-only rule: --device must be cuda/cuda:N for this experiment.")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU-only rule: torch.cuda.is_available() is false; do not run the experiment on CPU.")

    device_index = torch.cuda.current_device()
    if ":" in str(device_arg):
        try:
            device_index = int(str(device_arg).split(":", 1)[1])
        except ValueError:
            device_index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_index)
    arch_list: list[str] = []
    if hasattr(torch.cuda, "get_arch_list"):
        arch_list = list(torch.cuda.get_arch_list())
    compute_capability = f"sm_{props.major}{props.minor}"
    if arch_list and compute_capability not in arch_list:
        raise RuntimeError(
            f"GPU-only rule: current GPU compute capability {compute_capability} is not in PyTorch arch list {arch_list}."
        )

    try:
        smi = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,compute_cap,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
    except Exception as exc:  # pragma: no cover - depends on host tools
        smi = f"nvidia-smi failed: {exc}"

    # Real CUDA sanity check before any official run.
    x = torch.eye(8, device=f"cuda:{device_index}")
    y = x @ x
    if not bool(torch.isfinite(y).all().detach().cpu()):
        raise RuntimeError("GPU-only rule: PyTorch CUDA sanity matmul produced nonfinite values.")

    jax_info: dict[str, Any] = {}
    try:
        import jax
        import jax.numpy as jnp

        backend = jax.default_backend()
        devices = [str(d) for d in jax.devices()]
        z = jnp.eye(8) @ jnp.eye(8)
        jax.block_until_ready(z)
        if backend != "gpu":
            raise RuntimeError(f"JAX backend is {backend!r}, expected 'gpu'.")
        jax_info = {"backend": backend, "devices": devices, "sanity_matmul": "ok"}
    except Exception as exc:
        raise RuntimeError(f"GPU-only rule: JAX GPU sanity check failed: {exc}") from exc

    return {
        "nvidia_smi": smi,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "torch_device_index": device_index,
        "torch_device_name": torch.cuda.get_device_name(device_index),
        "torch_compute_capability": compute_capability,
        "torch_arch_list": arch_list,
        "torch_cuda_sanity_matmul": "ok",
        "jax": jax_info,
    }


def dual_vector_for_output(y, q: float):
    import torch

    flat = y.reshape(y.shape[0], -1)
    if math.isinf(q):
        out = torch.zeros_like(flat)
        idx = torch.argmax(flat.abs(), dim=1, keepdim=True)
        out.scatter_(1, idx, torch.sign(torch.gather(flat, 1, idx)))
        return out.reshape_as(y)
    if q == 1.0:
        return torch.sign(y)
    return torch.sign(y) * torch.clamp(y.abs(), min=1e-30).pow(q - 1.0)


def loss3_norms(problem: AblationProblem, x_adv, *, allow_solver_grad: bool) -> tuple[dict[str, Any], Any, Any, Any]:
    f_adv = problem.model_forward(x_adv)
    g_adv = problem.solver_forward(x_adv, allow_grad=allow_solver_grad)
    residual = f_adv - g_adv
    return {
        "loss3_q": batch_norm(residual, problem.args.q_order),
        "loss3_l2": batch_norm(residual, 2.0),
        "loss3_linf": batch_norm(residual, float("inf")),
    }, f_adv, g_adv, residual


def evaluate_all_values_np(problem: AblationProblem, delta) -> dict[str, np.ndarray]:
    import torch

    with torch.no_grad():
        x_adv = (problem.x0 + delta).detach()
        losses = raw_losses(problem, x_adv, allow_solver_grad_for_loss3=False)
        values = objective_values(problem, losses, delta)
    return {key: values[key].detach().cpu().numpy().astype(np.float32) for key in EVAL_KEYS}


def residual_forward(problem: AblationProblem, inp):
    return problem.model_forward(inp) - problem.solver_forward(inp, allow_grad=True)


def qaware_power_direction(problem: AblationProblem, base_x, v, variant: str, radius: float) -> tuple[Any, dict[str, Any]]:
    import torch

    q = problem.args.q_order
    p = problem.args.p_order
    x_base = base_x.detach().clone().requires_grad_(True)

    def fn(inp):
        return residual_forward(problem, inp)

    if variant in {"pure_jvp_vjp", "generalized_pq"}:
        _, jv = torch.autograd.functional.jvp(fn, (x_base,), (v.detach(),), create_graph=True)
        output_dual = dual_vector_for_output(jv, q)
        scalar = torch.sum(jv * output_dual.detach())
        (vjp,) = torch.autograd.grad(scalar, x_base)
        effective_jv = jv
        q_side_source = "explicit_phi_q_on_jv"
    elif variant == "affine_jvp_vjp":
        residual_base, jv = torch.autograd.functional.jvp(fn, (x_base,), (v.detach(),), create_graph=True)
        affine_residual = residual_base.detach() + float(radius) * jv
        output_dual = dual_vector_for_output(affine_residual, q)
        scalar = torch.sum(jv * output_dual.detach())
        (vjp,) = torch.autograd.grad(scalar, x_base)
        effective_jv = float(radius) * jv
        q_side_source = "explicit_phi_q_on_affine_residual"
    else:
        raise ValueError(f"Unsupported q-aware power variant {variant!r}")

    direction = steepest_direction(sanitize_tensor(vjp.detach()), p)
    info = {
        "q_side_map_source": q_side_source,
        "jvp_qnorm": tensor_to_numpy(batch_norm(effective_jv.detach(), q)),
        "jvp_l2": tensor_to_numpy(batch_l2_torch(effective_jv.detach())),
        "vjp_l2": tensor_to_numpy(batch_l2_torch(vjp.detach())),
        "output_dual_l2": tensor_to_numpy(batch_l2_torch(output_dual.detach())),
    }
    return direction.detach(), info


def initialize_power_state(problem: AblationProblem, seed: int):
    import torch

    generator = torch.Generator(device=problem.x0.device)
    generator.manual_seed(int(seed))
    v = torch.randn(problem.x0.shape, device=problem.x0.device, dtype=problem.x0.dtype, generator=generator)
    return normalize_to_p_ball(v, problem.args.p_order).detach()


def direction_for_method(
    problem: AblationProblem,
    spec: MethodSpec,
    grad,
    delta,
    power_state,
) -> tuple[Any, Any, dict[str, Any]]:
    base_x = (problem.x0 + delta).detach()
    radius = float(np.nanmean(tensor_to_numpy(batch_norm(delta.detach(), problem.args.p_order))))
    info: dict[str, Any] = {
        "gradient_source": "autograd(loss3_q)",
        "steepest_direction_source": "not_used",
        "q_side_map_source": "not_used",
    }

    if spec.direction_rule == "raw_gradient":
        direction = grad
        info["direction_source"] = "raw_autograd_gradient"
    elif spec.direction_rule == "unit_raw_gradient":
        direction = normalize_to_p_ball(grad, problem.args.p_order)
        info["direction_source"] = "p_normalized_autograd_gradient"
        info["steepest_direction_source"] = "normalize_to_p_ball"
    elif spec.direction_rule == "lp_steepest":
        direction = steepest_direction(grad, problem.args.p_order)
        info["direction_source"] = "p_steepest_autograd_gradient"
        info["steepest_direction_source"] = "steepest_direction(grad,p)"
    elif spec.direction_rule == "power_objective_gradient":
        direction = steepest_direction(grad, problem.args.p_order)
        info["direction_source"] = "objective_gradient_power_replacement_direction"
        info["steepest_direction_source"] = "steepest_direction(grad,p)"
    elif spec.direction_rule in {"power_pure_jvp_vjp", "power_generalized_pq", "power_affine_jvp_vjp"}:
        if power_state is None:
            power_state = initialize_power_state(problem, problem.args.seed)
        variant = "generalized_pq" if spec.direction_rule == "power_generalized_pq" else spec.power_variant
        direction, power_info = qaware_power_direction(problem, base_x, power_state, variant, radius)
        power_state = direction.detach()
        info["direction_source"] = f"q_aware_power_{variant}"
        info["gradient_source"] = "autograd(loss3_q)_for_diagnostics"
        info["steepest_direction_source"] = "steepest_direction(vjp,p)"
        info.update(power_info)
    else:
        raise ValueError(spec.direction_rule)

    return sanitize_tensor(direction.detach()), power_state, info


def propose_delta(problem: AblationProblem, spec: MethodSpec, delta, direction):
    if spec.proposal_rule == "additive":
        proposal = delta + float(problem.args.alpha) * direction
    elif spec.proposal_rule == "replacement":
        proposal = float(problem.args.epsilon) * direction
    else:
        raise ValueError(spec.proposal_rule)
    projected = project_delta(proposal, problem.args.epsilon, problem.args.p_order)
    return sanitize_tensor(proposal), sanitize_tensor(projected)


def smoothness_1d(v: np.ndarray) -> dict[str, float]:
    arr = np.asarray(v, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return {
            "high_frequency_energy_ratio": float("nan"),
            "spectral_centroid": float("nan"),
            "total_variation": float("nan"),
            "first_derivative_l2": float("nan"),
            "second_derivative_l2": float("nan"),
            "zero_crossing_count": float("nan"),
            "max_abs_delta": float("nan"),
        }
    spectrum = np.fft.rfft(arr)
    power = np.abs(spectrum) ** 2
    total_power = float(np.sum(power))
    if total_power <= EPS:
        high_ratio = 0.0
        centroid = 0.0
    else:
        start = int(math.floor(0.75 * max(1, power.size - 1)))
        high_ratio = float(np.sum(power[start:]) / total_power)
        freqs = np.arange(power.size, dtype=np.float64)
        centroid = float(np.sum(freqs * power) / total_power)
    d1 = np.diff(arr)
    d2 = np.diff(arr, n=2) if arr.size >= 3 else np.asarray([], dtype=np.float64)
    signs = np.signbit(arr)
    return {
        "high_frequency_energy_ratio": high_ratio,
        "spectral_centroid": centroid,
        "total_variation": float(np.sum(np.abs(d1))),
        "first_derivative_l2": float(np.linalg.norm(d1)),
        "second_derivative_l2": float(np.linalg.norm(d2)) if d2.size else 0.0,
        "zero_crossing_count": int(np.count_nonzero(signs[1:] != signs[:-1])) if arr.size >= 2 else 0,
        "max_abs_delta": float(np.max(np.abs(arr))),
    }


def spectrum_1d(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(v, dtype=np.float64).reshape(-1)
    spec = np.fft.rfft(arr)
    freq = np.fft.rfftfreq(arr.size, d=1.0 / max(1, arr.size))
    return freq.astype(np.float32), np.abs(spec).astype(np.float32)


def smoothness_batch(delta_np: np.ndarray) -> dict[str, np.ndarray]:
    flat = flatten_np(delta_np)
    rows = [smoothness_1d(row) for row in flat]
    keys = rows[0].keys() if rows else []
    return {key: np.asarray([row[key] for row in rows], dtype=np.float64) for key in keys}


def selected_positions(problem: AblationProblem, requested: list[int]) -> tuple[list[int], list[int], list[int]]:
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


def append_step_rows(
    *,
    rows: list[dict[str, Any]],
    sample_rows: list[dict[str, Any]],
    smooth_rows: list[dict[str, Any]],
    problem: AblationProblem,
    spec: MethodSpec,
    k: int,
    losses_np: dict[str, np.ndarray],
    boundary_losses_np: dict[str, np.ndarray] | None,
    delta,
    grad,
    direction,
    proposal,
    projected,
    prev_direction,
    direction_info: dict[str, Any],
    elapsed: float,
) -> None:
    import torch

    delta_norm_p = batch_norm(delta, problem.args.p_order).detach()
    delta_l2 = batch_l2_torch(delta).detach()
    delta_linf = batch_linf_torch(delta).detach()
    delta_np = tensor_to_numpy(delta.detach()).astype(np.float32)
    smooth = smoothness_batch(delta_np)

    grad_l2 = torch.full_like(delta_norm_p, float("nan"))
    grad_linf = torch.full_like(delta_norm_p, float("nan"))
    direction_pnorm = torch.full_like(delta_norm_p, float("nan"))
    direction_l2 = torch.full_like(delta_norm_p, float("nan"))
    proposal_pnorm = torch.full_like(delta_norm_p, float("nan"))
    projected_pnorm = torch.full_like(delta_norm_p, float("nan"))
    projection_shrink = torch.full_like(delta_norm_p, float("nan"))
    cos_delta_grad = torch.full_like(delta_norm_p, float("nan"))
    cos_delta_direction = torch.full_like(delta_norm_p, float("nan"))
    cos_direction_prev = torch.full_like(delta_norm_p, float("nan"))
    cos_grad_direction = torch.full_like(delta_norm_p, float("nan"))

    if grad is not None:
        grad_l2 = batch_l2_torch(grad).detach()
        grad_linf = batch_linf_torch(grad).detach()
        cos_delta_grad = batch_cosine_torch(delta, grad).detach()
    if direction is not None:
        direction_pnorm = batch_norm(direction, problem.args.p_order).detach()
        direction_l2 = batch_l2_torch(direction).detach()
        cos_delta_direction = batch_cosine_torch(delta, direction).detach()
        if grad is not None:
            cos_grad_direction = batch_cosine_torch(grad, direction).detach()
        if prev_direction is not None:
            cos_direction_prev = batch_cosine_torch(direction, prev_direction).detach()
    if proposal is not None and projected is not None:
        proposal_pnorm = batch_norm(proposal, problem.args.p_order).detach()
        projected_pnorm = batch_norm(projected, problem.args.p_order).detach()
        projection_shrink = projected_pnorm / proposal_pnorm.clamp_min(EPS)

    arrays: dict[str, np.ndarray] = {
        "loss3_q": losses_np["loss3_q"],
        "loss3_l2": losses_np["loss3_l2"],
        "loss3_linf": losses_np["loss3_linf"],
        "delta_pnorm": tensor_to_numpy(delta_norm_p),
        "delta_l2": tensor_to_numpy(delta_l2),
        "delta_linf": tensor_to_numpy(delta_linf),
        "boundary_ratio": tensor_to_numpy(delta_norm_p / float(problem.args.epsilon)),
        "grad_l2": tensor_to_numpy(grad_l2),
        "grad_linf": tensor_to_numpy(grad_linf),
        "direction_pnorm": tensor_to_numpy(direction_pnorm),
        "direction_l2": tensor_to_numpy(direction_l2),
        "proposal_pnorm": tensor_to_numpy(proposal_pnorm),
        "projected_pnorm": tensor_to_numpy(projected_pnorm),
        "projection_shrink_factor": tensor_to_numpy(projection_shrink),
        "cos_delta_grad": tensor_to_numpy(cos_delta_grad),
        "cos_delta_direction": tensor_to_numpy(cos_delta_direction),
        "cos_direction_prev": tensor_to_numpy(cos_direction_prev),
        "cos_grad_direction": tensor_to_numpy(cos_grad_direction),
    }
    for key, values in smooth.items():
        arrays[key] = values
    if boundary_losses_np is not None:
        arrays["boundary_loss3_q"] = boundary_losses_np["loss3_q"]
        arrays["boundary_loss3_l2"] = boundary_losses_np["loss3_l2"]
        arrays["boundary_loss3_linf"] = boundary_losses_np["loss3_linf"]

    row: dict[str, Any] = {
        "method": spec.label,
        "direction_rule": spec.direction_rule,
        "proposal_rule": spec.proposal_rule,
        "power_variant": spec.power_variant,
        "official_alias": spec.official_alias,
        "k": k,
        "epsilon": problem.args.epsilon,
        "alpha": problem.args.alpha,
        "p_order": norm_name(problem.args.p_order),
        "q_order": norm_name(problem.args.q_order),
        "seconds_since_method_start": elapsed,
        "gradient_implementation_source": direction_info.get("gradient_source", "unknown"),
        "steepest_direction_source": direction_info.get("steepest_direction_source", "unknown"),
        "q_side_map_source": direction_info.get("q_side_map_source", "unknown"),
    }
    for key, values in arrays.items():
        row.update(finite_stats_dict(key, values))
    for info_key in ("jvp_qnorm", "jvp_l2", "vjp_l2", "output_dual_l2"):
        if info_key in direction_info:
            row.update(finite_stats_dict(info_key, direction_info[info_key]))
    rows.append(row)

    for i, dataset_index in enumerate(problem.dataset_indices.tolist()):
        sample: dict[str, Any] = {
            "method": spec.label,
            "direction_rule": spec.direction_rule,
            "proposal_rule": spec.proposal_rule,
            "power_variant": spec.power_variant,
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

        smooth_row = {
            "method": spec.label,
            "k": k,
            "sample_position": i,
            "dataset_index": int(dataset_index),
            "p_order": norm_name(problem.args.p_order),
            "q_order": norm_name(problem.args.q_order),
        }
        for key in (
            "high_frequency_energy_ratio",
            "spectral_centroid",
            "total_variation",
            "first_derivative_l2",
            "second_derivative_l2",
            "zero_crossing_count",
            "max_abs_delta",
        ):
            smooth_row[key] = float(arrays[key][i])
        smooth_rows.append(smooth_row)


def save_final_delta_diagnostics_ablation(
    problem: AblationProblem,
    spec: MethodSpec,
    final_delta,
    out_dir: Path,
) -> dict[str, Any]:
    final_delta = final_delta.detach()
    boundary_delta, final_norm, boundary_norm, boundary_scale, boundary_valid = rescale_delta_to_boundary(
        final_delta,
        problem.args.epsilon,
        problem.args.p_order,
    )
    final_values = evaluate_all_values_np(problem, final_delta)
    boundary_values = evaluate_all_values_np(problem, boundary_delta)
    with np.errstate(all="ignore"):
        final_loss3_norms, _, _, _ = loss3_norms(problem, problem.x0 + final_delta, allow_solver_grad=False)
        boundary_loss3_norms, _, _, _ = loss3_norms(problem, problem.x0 + boundary_delta, allow_solver_grad=False)
    final_extra = {key: tensor_to_numpy(value.detach()).astype(np.float32) for key, value in final_loss3_norms.items()}
    boundary_extra = {key: tensor_to_numpy(value.detach()).astype(np.float32) for key, value in boundary_loss3_norms.items()}

    final_norm_np = tensor_to_numpy(final_norm).astype(np.float32)
    boundary_norm_np = tensor_to_numpy(boundary_norm).astype(np.float32)
    boundary_scale_np = tensor_to_numpy(boundary_scale).astype(np.float32)
    boundary_valid_np = tensor_to_numpy(boundary_valid).astype(bool)
    final_delta_np = tensor_to_numpy(final_delta).astype(np.float32)
    boundary_delta_np = tensor_to_numpy(boundary_delta).astype(np.float32)

    rows: list[dict[str, Any]] = []
    for i, dataset_index in enumerate(problem.dataset_indices.tolist()):
        row: dict[str, Any] = {
            "method": spec.label,
            "direction_rule": spec.direction_rule,
            "proposal_rule": spec.proposal_rule,
            "power_variant": spec.power_variant,
            "sample_position": i,
            "dataset_index": int(dataset_index),
            "epsilon": problem.args.epsilon,
            "p_order": norm_name(problem.args.p_order),
            "q_order": norm_name(problem.args.q_order),
            "final_delta_pnorm": float(final_norm_np[i]),
            "boundary_delta_pnorm": float(boundary_norm_np[i]),
            "boundary_rescale_factor": float(boundary_scale_np[i]),
            "boundary_rescale_valid": bool(boundary_valid_np[i]),
            "final_loss3_q": float(final_extra["loss3_q"][i]),
            "final_loss3_l2": float(final_extra["loss3_l2"][i]),
            "final_loss3_linf": float(final_extra["loss3_linf"][i]),
            "boundary_loss3_q": float(boundary_extra["loss3_q"][i]),
            "boundary_loss3_l2": float(boundary_extra["loss3_l2"][i]),
            "boundary_loss3_linf": float(boundary_extra["loss3_linf"][i]),
        }
        for key in EVAL_KEYS:
            row[f"final_{key}"] = float(final_values[key][i])
            row[f"boundary_{key}"] = float(boundary_values[key][i])
        rows.append(row)

    write_csv(out_dir / "final_delta_diagnostics.csv", rows)
    np.savez_compressed(
        out_dir / "final_deltas.npz",
        sample_position=np.arange(problem.args.batch_size, dtype=np.int64),
        dataset_index=problem.dataset_indices.astype(np.int64),
        final_delta=final_delta_np,
        boundary_delta=boundary_delta_np,
        final_delta_pnorm=final_norm_np,
        boundary_delta_pnorm=boundary_norm_np,
        boundary_rescale_factor=boundary_scale_np,
        boundary_rescale_valid=boundary_valid_np,
        final_loss3_q=final_extra["loss3_q"],
        final_loss3_l2=final_extra["loss3_l2"],
        final_loss3_linf=final_extra["loss3_linf"],
        boundary_loss3_q=boundary_extra["loss3_q"],
        boundary_loss3_l2=boundary_extra["loss3_l2"],
        boundary_loss3_linf=boundary_extra["loss3_linf"],
    )

    summary: dict[str, Any] = {
        "method": spec.label,
        "direction_rule": spec.direction_rule,
        "proposal_rule": spec.proposal_rule,
        "power_variant": spec.power_variant,
        "epsilon": problem.args.epsilon,
        "alpha": problem.args.alpha,
        "p_order": norm_name(problem.args.p_order),
        "q_order": norm_name(problem.args.q_order),
        "boundary_rescale_valid_count": int(np.count_nonzero(boundary_valid_np)),
    }
    for key, values in {
        "final_delta_pnorm": final_norm_np,
        "boundary_delta_pnorm": boundary_norm_np,
        "boundary_rescale_factor": boundary_scale_np,
        **{f"final_{name}": value for name, value in final_extra.items()},
        **{f"boundary_{name}": value for name, value in boundary_extra.items()},
    }.items():
        add_finite_summary(summary, key, values)
    save_json(out_dir / "final_delta_summary.json", summary)
    return summary


def run_method(problem: AblationProblem, spec: MethodSpec, root: Path, trajectory_indices: list[int]) -> MethodResult:
    import torch

    method_dir = root / spec.label
    method_dir.mkdir(parents=True, exist_ok=True)
    if problem.args.init == "random":
        # Use a tiny random start by default only when requested, not for official zero-init comparisons.
        from tools.run_batch_three_loss_loss_only import random_delta_like

        delta = random_delta_like(problem.x0, problem.args.epsilon, problem.args.p_order, problem.args.random_start_scale, problem.args.seed)
    else:
        delta = torch.zeros_like(problem.x0)

    selected_dataset_indices, selected_pos, missing_trajectory_indices = selected_positions(problem, trajectory_indices)
    if missing_trajectory_indices:
        save_json(method_dir / "missing_trajectory_indices.json", {"missing": missing_trajectory_indices})

    per_step_rows: list[dict[str, Any]] = []
    per_sample_rows: list[dict[str, Any]] = []
    smoothness_rows: list[dict[str, Any]] = []
    trajectory: dict[str, list[Any]] = {
        "k": [],
        "delta": [],
        "x_adv": [],
        "loss3_q": [],
        "loss3_l2": [],
        "loss3_linf": [],
        "delta_pnorm": [],
        "delta_l2": [],
        "delta_linf": [],
        "boundary_ratio": [],
        "high_frequency_energy_ratio": [],
        "spectral_centroid": [],
        "total_variation": [],
        "first_derivative_l2": [],
        "second_derivative_l2": [],
        "zero_crossing_count": [],
    }
    direction_trace: dict[str, list[Any]] = {"k": [], "direction": []}

    prev_direction = None
    power_state = initialize_power_state(problem, problem.args.seed + 1009) if spec.uses_power_state else None
    if torch.cuda.is_available() and problem.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(problem.device)
        torch.cuda.synchronize(problem.device)
    start = time.perf_counter()

    for k in range(problem.args.steps + 1):
        step_start = time.perf_counter()
        delta = sanitize_tensor(delta.detach())
        x_adv = (problem.x0 + delta).detach().requires_grad_(k < problem.args.steps)
        losses, _, _, _ = loss3_norms(problem, x_adv, allow_solver_grad=(k < problem.args.steps))
        objective = sanitize_tensor(losses["loss3_q"])
        losses_np = {key: tensor_to_numpy(value.detach()).astype(np.float32) for key, value in losses.items()}

        grad = None
        direction = None
        proposal = None
        projected = None
        direction_info: dict[str, Any] = {
            "gradient_source": "not_computed_final_step" if k >= problem.args.steps else "autograd(loss3_q)",
            "steepest_direction_source": "not_used",
            "q_side_map_source": "not_used",
        }
        if k < problem.args.steps:
            batch_objective = objective.sum()
            batch_objective.backward()
            grad = sanitize_tensor(x_adv.grad.detach())
            direction, power_state, direction_info = direction_for_method(problem, spec, grad, delta, power_state)
            proposal, projected = propose_delta(problem, spec, delta, direction)

        boundary_losses_np = None
        if problem.args.save_boundary_normalized_eval:
            boundary_delta, _, _, _, boundary_valid = rescale_delta_to_boundary(delta, problem.args.epsilon, problem.args.p_order)
            if bool(boundary_valid.any().detach().cpu()):
                with torch.no_grad():
                    boundary_losses, _, _, _ = loss3_norms(problem, problem.x0 + boundary_delta, allow_solver_grad=False)
                boundary_losses_np = {key: tensor_to_numpy(value.detach()).astype(np.float32) for key, value in boundary_losses.items()}

        elapsed = time.perf_counter() - start
        append_step_rows(
            rows=per_step_rows,
            sample_rows=per_sample_rows,
            smooth_rows=smoothness_rows,
            problem=problem,
            spec=spec,
            k=k,
            losses_np=losses_np,
            boundary_losses_np=boundary_losses_np,
            delta=delta,
            grad=grad,
            direction=direction,
            proposal=proposal,
            projected=projected,
            prev_direction=prev_direction,
            direction_info=direction_info,
            elapsed=elapsed,
        )

        if selected_pos and problem.args.save_delta_trajectory:
            delta_np = tensor_to_numpy(delta.detach()).astype(np.float32)
            x_adv_np = tensor_to_numpy((problem.x0 + delta).detach()).astype(np.float32)
            delta_norm_p = tensor_to_numpy(batch_norm(delta, problem.args.p_order).detach())
            delta_l2 = tensor_to_numpy(batch_l2_torch(delta).detach())
            delta_linf = tensor_to_numpy(batch_linf_torch(delta).detach())
            smooth = smoothness_batch(delta_np)
            trajectory["k"].append(k)
            trajectory["delta"].append(delta_np[selected_pos])
            trajectory["x_adv"].append(x_adv_np[selected_pos])
            for key in LOSS3_KEYS:
                trajectory[key].append(losses_np[key][selected_pos])
            trajectory["delta_pnorm"].append(delta_norm_p[selected_pos])
            trajectory["delta_l2"].append(delta_l2[selected_pos])
            trajectory["delta_linf"].append(delta_linf[selected_pos])
            trajectory["boundary_ratio"].append(delta_norm_p[selected_pos] / float(problem.args.epsilon))
            for key in (
                "high_frequency_energy_ratio",
                "spectral_centroid",
                "total_variation",
                "first_derivative_l2",
                "second_derivative_l2",
                "zero_crossing_count",
            ):
                trajectory[key].append(smooth[key][selected_pos])
            if direction is not None:
                direction_trace["k"].append(k)
                direction_trace["direction"].append(tensor_to_numpy(direction.detach()).astype(np.float32)[selected_pos])

        if direction is not None:
            prev_direction = direction.detach()
        if k >= problem.args.steps:
            break
        delta = projected.detach()
        _ = step_start  # kept for breakpoint/debugging without changing output schema

    sync_torch(torch, problem.device)
    runtime_seconds = time.perf_counter() - start
    torch_peak_allocated_mib = float("nan")
    torch_peak_reserved_mib = float("nan")
    if torch.cuda.is_available() and problem.device.type == "cuda":
        torch_peak_allocated_mib = float(torch.cuda.max_memory_allocated(problem.device) / 1024**2)
        torch_peak_reserved_mib = float(torch.cuda.max_memory_reserved(problem.device) / 1024**2)

    final_summary = save_final_delta_diagnostics_ablation(problem, spec, delta, method_dir)
    for row in per_step_rows:
        row["method_runtime_seconds"] = runtime_seconds
        row["torch_peak_allocated_mib"] = torch_peak_allocated_mib
        row["torch_peak_reserved_mib"] = torch_peak_reserved_mib

    write_csv(method_dir / "per_step_metrics.csv", per_step_rows)
    write_csv(method_dir / "per_sample_step_metrics.csv", per_sample_rows)
    write_csv(method_dir / "delta_smoothness_by_step.csv", smoothness_rows)
    write_csv(method_dir / "physical_smoothness_metrics.csv", [row for row in smoothness_rows if row["k"] == problem.args.steps])

    trajectory_payload: dict[str, Any] | None = None
    direction_payload: dict[str, Any] | None = None
    if selected_pos and problem.args.save_delta_trajectory and trajectory["k"]:
        trajectory_payload = {
            "method": np.asarray([spec.label]),
            "dataset_index": np.asarray(selected_dataset_indices, dtype=np.int64),
            "sample_position": np.asarray(selected_pos, dtype=np.int64),
            **{key: np.asarray(value) for key, value in trajectory.items()},
        }
        np.savez_compressed(method_dir / "trajectory_samples.npz", **trajectory_payload)
        save_delta_spectra(method_dir / "delta_spectrum_by_step.npz", trajectory_payload)
    if selected_pos and direction_trace["k"]:
        direction_payload = {
            "method": np.asarray([spec.label]),
            "dataset_index": np.asarray(selected_dataset_indices, dtype=np.int64),
            "sample_position": np.asarray(selected_pos, dtype=np.int64),
            "k": np.asarray(direction_trace["k"], dtype=np.int64),
            "direction": np.asarray(direction_trace["direction"], dtype=np.float32),
        }
        np.savez_compressed(method_dir / "direction_trace_selected.npz", **direction_payload)

    summary = {
        **final_summary,
        "runtime_seconds": runtime_seconds,
        "torch_peak_allocated_mib": torch_peak_allocated_mib,
        "torch_peak_reserved_mib": torch_peak_reserved_mib,
        "gradient_implementation_source": "autograd(loss3_q)",
        "steepest_direction_source": "see per_step_metrics.csv",
        "trajectory_indices_requested": trajectory_indices,
        "trajectory_indices_present": selected_dataset_indices,
        "trajectory_indices_missing": missing_trajectory_indices,
    }
    save_json(method_dir / "summary.json", summary)

    return MethodResult(
        spec=spec,
        method_dir=method_dir,
        per_step_rows=per_step_rows,
        per_sample_rows=per_sample_rows,
        smoothness_rows=smoothness_rows,
        final_delta=tensor_to_numpy(delta.detach()).astype(np.float32),
        trajectory=trajectory_payload,
        direction_trace=direction_payload,
    )


def save_delta_spectra(path: Path, trajectory_payload: dict[str, Any]) -> None:
    deltas = np.asarray(trajectory_payload["delta"])
    # deltas: step x selected_sample x spatial x 1
    freq_ref: np.ndarray | None = None
    spectra: list[list[np.ndarray]] = []
    for step_arr in deltas:
        step_specs: list[np.ndarray] = []
        for sample_delta in step_arr:
            freq, amp = spectrum_1d(sample_delta)
            freq_ref = freq if freq_ref is None else freq_ref
            step_specs.append(amp)
        spectra.append(step_specs)
    np.savez_compressed(
        path,
        method=trajectory_payload["method"],
        dataset_index=trajectory_payload["dataset_index"],
        sample_position=trajectory_payload["sample_position"],
        k=trajectory_payload["k"],
        frequency=np.asarray(freq_ref if freq_ref is not None else [], dtype=np.float32),
        amplitude=np.asarray(spectra, dtype=np.float32),
    )


def write_pairwise_final_delta_cosines(root: Path, results: list[MethodResult], dataset_indices: np.ndarray) -> None:
    rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for left, right in itertools.combinations(results, 2):
        values: list[float] = []
        for i, dataset_index in enumerate(dataset_indices.tolist()):
            cos = np_l2_cosine(left.final_delta[i], right.final_delta[i])
            values.append(cos)
            rows.append(
                {
                    "method_a": left.spec.label,
                    "method_b": right.spec.label,
                    "sample_position": i,
                    "dataset_index": int(dataset_index),
                    "final_delta_l2_cosine": cos,
                }
            )
        summary = {"method_a": left.spec.label, "method_b": right.spec.label}
        summary.update(finite_stats_dict("final_delta_l2_cosine", values))
        summary_rows.append(summary)
    write_csv(root / "method_pairwise_final_delta_cosines.csv", rows)
    write_csv(root / "method_pairwise_final_delta_cosine_summary.csv", summary_rows)


def write_selected_direction_cosines(root: Path, results: list[MethodResult]) -> None:
    payloads = [r for r in results if r.direction_trace is not None]
    if len(payloads) < 2:
        return
    rows: list[dict[str, Any]] = []
    for left, right in itertools.combinations(payloads, 2):
        l_payload = left.direction_trace or {}
        r_payload = right.direction_trace or {}
        common_k = sorted(set(np.asarray(l_payload["k"]).tolist()) & set(np.asarray(r_payload["k"]).tolist()))
        common_idx = sorted(set(np.asarray(l_payload["dataset_index"]).tolist()) & set(np.asarray(r_payload["dataset_index"]).tolist()))
        for k in common_k:
            l_step = int(np.where(np.asarray(l_payload["k"]) == k)[0][0])
            r_step = int(np.where(np.asarray(r_payload["k"]) == k)[0][0])
            for dataset_index in common_idx:
                l_sample = int(np.where(np.asarray(l_payload["dataset_index"]) == dataset_index)[0][0])
                r_sample = int(np.where(np.asarray(r_payload["dataset_index"]) == dataset_index)[0][0])
                rows.append(
                    {
                        "method_a": left.spec.label,
                        "method_b": right.spec.label,
                        "k": int(k),
                        "dataset_index": int(dataset_index),
                        "direction_l2_cosine": np_l2_cosine(
                            np.asarray(l_payload["direction"])[l_step, l_sample],
                            np.asarray(r_payload["direction"])[r_step, r_sample],
                        ),
                    }
                )
    write_csv(root / "direction_cosines.csv", rows)


def write_root_arrays(root: Path, results: list[MethodResult], dataset_indices: np.ndarray) -> None:
    if not results:
        return
    np.savez_compressed(
        root / "final_deltas.npz",
        method=np.asarray([r.spec.label for r in results]),
        dataset_index=dataset_indices.astype(np.int64),
        final_delta=np.stack([r.final_delta for r in results], axis=0),
    )


def write_pq_geometry_summary(root: Path, results: list[MethodResult], args: argparse.Namespace) -> None:
    rows: list[dict[str, Any]] = []
    for result in results:
        final_diag = result.method_dir / "final_delta_diagnostics.csv"
        if not final_diag.exists():
            continue
        with final_diag.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            data = list(reader)
        row: dict[str, Any] = {
            "method": result.spec.label,
            "direction_rule": result.spec.direction_rule,
            "proposal_rule": result.spec.proposal_rule,
            "power_variant": result.spec.power_variant,
            "p_order": norm_name(args.p_order),
            "q_order": norm_name(args.q_order),
        }
        for key in ("final_loss3_q", "final_loss3_l2", "final_loss3_linf", "final_delta_pnorm"):
            values = [float(d[key]) for d in data if d.get(key) not in {None, ""}]
            row.update(finite_stats_dict(key, values))
        rows.append(row)
    write_csv(root / "pq_geometry_summary.csv", rows)
    write_csv(root / "cross_norm_final_metrics.csv", rows)


def plot_line_curves(root: Path, rows: list[dict[str, Any]], key: str, ylabel: str, filename: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    methods = sorted({row["method"] for row in rows})
    for method in methods:
        group = sorted([row for row in rows if row["method"] == method], key=lambda r: int(r["k"]))
        xs = [int(row["k"]) for row in group]
        ys = [float(row.get(key, float("nan"))) for row in group]
        ax.plot(xs, ys, label=method, linewidth=1.6)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    out = root / "figures" / "curves" / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    plt.close(fig)


def selected_step_indices(k_values: np.ndarray, requested_steps: list[int]) -> list[int]:
    out: list[int] = []
    for step in requested_steps:
        if step in set(k_values.tolist()):
            out.append(int(np.where(k_values == step)[0][0]))
    if len(out) < 2 and k_values.size:
        out = sorted(set(out + [0, int(k_values.size - 1)]))
    return out


def plot_delta_grids(root: Path, results: list[MethodResult], requested_steps: list[int]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    payloads = [r for r in results if r.trajectory is not None]
    if not payloads:
        return
    common_indices = sorted(set.intersection(*[set(np.asarray(r.trajectory["dataset_index"]).tolist()) for r in payloads]))
    out_dir = root / "figures" / "delta_grids"
    out_dir.mkdir(parents=True, exist_ok=True)
    for dataset_index in common_indices:
        all_vals: list[np.ndarray] = []
        for result in payloads:
            payload = result.trajectory or {}
            sample_pos = int(np.where(np.asarray(payload["dataset_index"]) == dataset_index)[0][0])
            k_values = np.asarray(payload["k"])
            for step_i in selected_step_indices(k_values, requested_steps):
                all_vals.append(np.asarray(payload["delta"])[step_i, sample_pos].reshape(-1))
        if not all_vals:
            continue
        ymin = float(np.min([np.min(v) for v in all_vals]))
        ymax = float(np.max([np.max(v) for v in all_vals]))
        if math.isclose(ymin, ymax):
            ymin -= 1.0
            ymax += 1.0
        first_payload = payloads[0].trajectory or {}
        k_values = np.asarray(first_payload["k"])
        step_indices = selected_step_indices(k_values, requested_steps)
        nrows = len(payloads)
        ncols = len(step_indices)
        fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 1.8 * nrows), squeeze=False)
        for r_i, result in enumerate(payloads):
            payload = result.trajectory or {}
            sample_pos = int(np.where(np.asarray(payload["dataset_index"]) == dataset_index)[0][0])
            k_values = np.asarray(payload["k"])
            step_indices = selected_step_indices(k_values, requested_steps)
            for c_i, step_i in enumerate(step_indices):
                ax = axes[r_i, c_i]
                arr = np.asarray(payload["delta"])[step_i, sample_pos].reshape(-1)
                ax.plot(np.arange(arr.size), arr, linewidth=1.0)
                ax.set_ylim(ymin, ymax)
                ax.grid(alpha=0.15)
                if r_i == 0:
                    ax.set_title(f"k={int(k_values[step_i])}", fontsize=9)
                if c_i == 0:
                    ax.set_ylabel(result.spec.label, fontsize=8)
        fig.suptitle(f"delta trajectory grid, dataset index {dataset_index}")
        fig.tight_layout()
        fig.savefig(out_dir / f"delta_grid_sample_{dataset_index:03d}.png", dpi=170)
        plt.close(fig)


def plot_heatmaps(root: Path, results: list[MethodResult]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = root / "figures" / "delta_heatmaps"
    out_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        payload = result.trajectory
        if payload is None:
            continue
        deltas = np.asarray(payload["delta"])
        k_values = np.asarray(payload["k"])
        for sample_i, dataset_index in enumerate(np.asarray(payload["dataset_index"]).tolist()):
            arr = deltas[:, sample_i].reshape(deltas.shape[0], -1)
            vmax = float(np.max(np.abs(arr)))
            vmax = vmax if vmax > 0 else 1.0
            fig, ax = plt.subplots(figsize=(9, 4.5))
            im = ax.imshow(arr, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax, origin="lower")
            fig.colorbar(im, ax=ax, label="delta")
            ax.set_xlabel("space index")
            ax.set_ylabel("saved step index")
            ax.set_yticks(np.arange(len(k_values)))
            ax.set_yticklabels([str(int(k)) for k in k_values], fontsize=6)
            ax.set_title(f"{result.spec.label}, dataset index {dataset_index}")
            fig.tight_layout()
            fig.savefig(out_dir / f"delta_heatmap_{result.spec.label}_sample_{int(dataset_index):03d}.png", dpi=170)
            plt.close(fig)


def plot_roughness_curves(root: Path, results: list[MethodResult]) -> None:
    all_rows = [row for result in results for row in result.smoothness_rows]
    for key, ylabel in [
        ("high_frequency_energy_ratio_mean", "high-frequency energy ratio"),
        ("spectral_centroid_mean", "spectral centroid"),
        ("total_variation_mean", "total variation"),
        ("first_derivative_l2_mean", "first-derivative L2"),
        ("second_derivative_l2_mean", "second-derivative L2"),
    ]:
        # Build temporary mean rows by method/k from the sample-level rows.
        mean_rows: list[dict[str, Any]] = []
        for method in sorted({r["method"] for r in all_rows}):
            for k in sorted({int(r["k"]) for r in all_rows if r["method"] == method}):
                values = [float(r[key.replace("_mean", "")]) for r in all_rows if r["method"] == method and int(r["k"]) == k]
                mean_rows.append({"method": method, "k": k, key: finite_mean_std(values)[0]})
        plot_line_curves(root, mean_rows, key, ylabel, f"{key}.png")


def make_delta_gifs(root: Path, results: list[MethodResult]) -> None:
    try:
        import imageio.v2 as imageio
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional plotting dependency
        print(f"[warn] GIF rendering skipped: {exc}", flush=True)
        return

    out_dir = root / "figures" / "gifs"
    frame_root = out_dir / ".frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_root.mkdir(parents=True, exist_ok=True)
    try:
        for result in results:
            payload = result.trajectory
            if payload is None:
                continue
            deltas = np.asarray(payload["delta"])
            k_values = np.asarray(payload["k"])
            losses = np.asarray(payload["loss3_q"])
            ratios = np.asarray(payload["boundary_ratio"])
            rough = np.asarray(payload["high_frequency_energy_ratio"])
            for sample_i, dataset_index in enumerate(np.asarray(payload["dataset_index"]).tolist()):
                arrs = deltas[:, sample_i].reshape(deltas.shape[0], -1)
                ymin = float(np.min(arrs))
                ymax = float(np.max(arrs))
                if math.isclose(ymin, ymax):
                    ymin -= 1.0
                    ymax += 1.0
                frames = []
                sample_frame_dir = frame_root / f"{result.spec.label}_{int(dataset_index):03d}"
                sample_frame_dir.mkdir(parents=True, exist_ok=True)
                for t, k in enumerate(k_values):
                    fig, ax = plt.subplots(figsize=(7, 3.5))
                    ax.plot(np.arange(arrs.shape[1]), arrs[t], linewidth=1.2)
                    ax.set_ylim(ymin, ymax)
                    ax.grid(alpha=0.2)
                    ax.set_title(
                        f"{result.spec.label} sample={int(dataset_index)} k={int(k)} "
                        f"loss3_q={losses[t, sample_i]:.4g} boundary={ratios[t, sample_i]:.3f} "
                        f"HF={rough[t, sample_i]:.3g}",
                        fontsize=9,
                    )
                    fig.tight_layout()
                    frame_path = sample_frame_dir / f"frame_{t:04d}.png"
                    fig.savefig(frame_path, dpi=130)
                    plt.close(fig)
                    frames.append(imageio.imread(frame_path))
                imageio.mimsave(out_dir / f"delta_evolution_{int(dataset_index):03d}_{result.spec.label}.gif", frames, duration=0.20)
    finally:
        shutil.rmtree(frame_root, ignore_errors=True)


def make_all_plots(root: Path, results: list[MethodResult], selected_steps: list[int], make_gifs: bool) -> None:
    all_step_rows = [row for result in results for row in result.per_step_rows]
    plot_line_curves(root, all_step_rows, "loss3_q_mean", "loss3 native q-norm", "loss3_q_mean_vs_step.png")
    plot_line_curves(root, all_step_rows, "boundary_ratio_mean", "||delta||_p / epsilon", "boundary_ratio_mean_vs_step.png")
    plot_line_curves(root, all_step_rows, "boundary_loss3_q_mean", "boundary-normalized loss3 q-norm", "boundary_loss3_q_mean_vs_step.png")
    plot_delta_grids(root, results, selected_steps)
    plot_heatmaps(root, results)
    plot_roughness_curves(root, results)
    if make_gifs:
        make_delta_gifs(root, results)


def write_analysis_checklist(root: Path) -> None:
    text = """# Loss3 Direction-Proposal Ablation Analysis Checklist

Run-status checks:

- `manifest.json` exists and records CUDA, PyTorch, GPU compute capability, PyTorch arch list, and JAX GPU backend/devices.
- `per_step_metrics.csv`, `per_sample_step_metrics.csv`, `final_deltas.npz`, `method_pairwise_final_delta_cosines.csv`, and `pq_geometry_summary.csv` exist at the run root.
- Each method directory contains `final_delta_diagnostics.csv`, `final_delta_summary.json`, and, when requested, `trajectory_samples.npz`.
- Figures exist under `figures/`; GIFs exist under `figures/gifs/` when `--make-gifs` was used.

Core comparisons:

- Boundary proposal effect: compare `steepest_add` vs `steepest_replace`, and `power_add__*` vs `power_replace__*`.
- Direction effect: compare matched proposal rows, especially `steepest_replace` vs q-aware power replacement variants.
- Equivalence sanity: for p=2, `unit_raw_add` should match `steepest_add`; current `generalized_power` should match steepest replacement if both use the same direction.
- Alpha limitation: if additive methods catch up only at larger alpha, report step-size limitation separately from direction quality.
- P/Q geometry: compare methods within a fixed p/q pair first; compare across p/q pairs only using common metrics (`loss3_l2`, `loss3_linf`, common delta norms, smoothness).

Visualization interpretation:

- Loss curves show speed and final objective quality.
- Delta grids show whether methods end with similar perturbation shapes.
- Delta heatmaps and GIFs show whether the path jumps, rotates, smooths, or remains spiky.
- Spectrum and roughness curves show whether high loss is tied to high-frequency or nonphysical perturbations.
"""
    (root / "analysis_checklist.md").write_text(text, encoding="utf-8")


def write_manifest(root: Path, args: argparse.Namespace, specs: list[MethodSpec], gpu_evidence: dict[str, Any]) -> None:
    manifest = {
        "status": "run_started",
        "experiment": "loss3_direction_proposal_ablation",
        "model_kind": args.model_kind,
        "model_label": args.model_label,
        "objective": "loss3_original",
        "p_order": norm_name(args.p_order),
        "q_order": norm_name(args.q_order),
        "epsilon": args.epsilon,
        "alpha": args.alpha,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "start_index": args.start_index,
        "dataset_indices": args.dataset_indices,
        "methods": [dataclasses.asdict(spec) for spec in specs],
        "gradient_implementation_source": "autograd(loss3_q) for PGD/LP-steepest/objective-gradient rows",
        "steepest_direction_source": "explicit p-ball linear maximizer",
        "gpu_runtime": gpu_evidence,
        "outputs_expected": [
            "per_step_metrics.csv",
            "per_sample_step_metrics.csv",
            "final_deltas.npz",
            "method_pairwise_final_delta_cosines.csv",
            "direction_cosines.csv",
            "pq_geometry_summary.csv",
            "figures/",
        ],
    }
    save_json(root / "manifest.json", manifest)


def update_manifest_status(root: Path, status: str, extra: dict[str, Any] | None = None) -> None:
    path = root / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    payload["status"] = status
    payload["finished_at_unix"] = time.time()
    if extra:
        payload.update(extra)
    save_json(path, payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=Path("forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3"))
    parser.add_argument("--methods", nargs="+", default=["main"], help="Method names or presets: official3, main, pq_key, additive_alpha, replacement_key.")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--dataset-indices", nargs="*", type=int, default=None, help="Optional non-contiguous dataset indices. Overrides --start-index/--batch-size.")
    parser.add_argument("--epsilon", type=float, default=8.0)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--p", default="2")
    parser.add_argument("--q", default="2")
    parser.add_argument("--eta", type=float, default=1e-6)
    parser.add_argument("--regularization-c", type=float, default=1.0)
    parser.add_argument("--init", choices=["zero", "random"], default="zero")
    parser.add_argument("--random-start-scale", type=float, default=1e-6)
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
    parser.add_argument("--trajectory-indices", nargs="*", type=int, default=list(DEFAULT_TRAJECTORY_INDICES))
    parser.add_argument("--trajectory-save-every", type=int, default=1, help="Reserved for compatibility; this script currently saves every optimized step selected by --steps.")
    parser.add_argument("--selected-steps", nargs="*", type=int, default=list(DEFAULT_SELECTED_STEPS))
    parser.add_argument("--save-boundary-normalized-eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-physical-diagnostics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-delta-trajectory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--make-gifs", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--dry-run-plan", action="store_true", help="Write the resolved method plan and exit without loading GPU/model or running attacks.")
    args = parser.parse_args()
    if args.model_label is None:
        args.model_label = format_model_label(args.model_kind)
    args.p_order = parse_norm(args.p)
    args.q_order = parse_norm(args.q)
    if args.p_order < 1.0 and not math.isinf(args.p_order):
        raise ValueError("p must be >= 1 or inf")
    if args.q_order < 1.0 and not math.isinf(args.q_order):
        raise ValueError("q must be >= 1 or inf")
    if args.dataset_indices:
        args.batch_size = len(args.dataset_indices)
    return args


def main() -> None:
    args = parse_args()
    specs = parse_methods(args.methods)
    args.out_root.mkdir(parents=True, exist_ok=True)
    save_json(
        args.out_root / "resolved_run_plan.json",
        {
            "dry_run_plan": bool(args.dry_run_plan),
            "methods": [dataclasses.asdict(spec) for spec in specs],
            "p_order": norm_name(args.p_order),
            "q_order": norm_name(args.q_order),
            "epsilon": args.epsilon,
            "alpha": args.alpha,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "dataset_indices": args.dataset_indices,
            "trajectory_indices": args.trajectory_indices,
            "visual_outputs": {
                "static_figures": not args.no_plots,
                "gifs": bool(args.make_gifs),
                "trajectory_npz": bool(args.save_delta_trajectory),
            },
        },
    )
    if args.dry_run_plan:
        print(f"[dry-run-plan] wrote {args.out_root / 'resolved_run_plan.json'}", flush=True)
        return

    gpu_evidence = collect_gpu_evidence(args.device)
    write_manifest(args.out_root, args, specs, gpu_evidence)
    write_analysis_checklist(args.out_root)

    problem = AblationProblem(args)
    config = vars(args).copy()
    config["p_order"] = norm_name(args.p_order)
    config["q_order"] = norm_name(args.q_order)
    config["dataset_indices_resolved"] = problem.dataset_indices.tolist()
    save_json(args.out_root / "config.json", config)

    results: list[MethodResult] = []
    try:
        for spec in specs:
            print(f"[run] {spec.label}: direction={spec.direction_rule}, proposal={spec.proposal_rule}, power={spec.power_variant}", flush=True)
            results.append(run_method(problem, spec, args.out_root, args.trajectory_indices))

        all_step_rows = [row for result in results for row in result.per_step_rows]
        all_sample_rows = [row for result in results for row in result.per_sample_rows]
        all_smooth_rows = [row for result in results for row in result.smoothness_rows]
        write_csv(args.out_root / "per_step_metrics.csv", all_step_rows)
        write_csv(args.out_root / "per_sample_step_metrics.csv", all_sample_rows)
        write_csv(args.out_root / "delta_smoothness_by_step.csv", all_smooth_rows)
        write_csv(args.out_root / "physical_smoothness_metrics.csv", [r for r in all_smooth_rows if int(r["k"]) == args.steps])
        write_root_arrays(args.out_root, results, problem.dataset_indices)
        write_pairwise_final_delta_cosines(args.out_root, results, problem.dataset_indices)
        write_selected_direction_cosines(args.out_root, results)
        write_pq_geometry_summary(args.out_root, results, args)
        if not args.no_plots:
            make_all_plots(args.out_root, results, args.selected_steps, args.make_gifs)
        update_manifest_status(args.out_root, "completed", {"method_count": len(results)})
    except Exception as exc:
        update_manifest_status(args.out_root, "failed", {"error": repr(exc)})
        raise

    print(f"[done] ablation outputs written under {args.out_root}", flush=True)


if __name__ == "__main__":
    main()
