#!/usr/bin/env python3
"""Core-four attacks for the 2D Navier-Stokes recurrent FNO model.

This script combines two existing code paths in this repository:

* the 1D/core optimizer split from ``tools/run_loss3_direction_proposal_ablation.py``
  (raw_add, raw_replace, steepest_add, steepest_replace), and
* the 2D NS recurrent-FNO solver/model wiring from
  ``2D_NS_FNO2d_recurrent/perturbation_methods/PGD_attack_adam_batch_adaptive.py``.

It is intentionally a runnable CLI, but importing or editing this file does not
load data, start CUDA work, or build a dictionary. A real run requires the
``adv_robust`` environment and CUDA/JAX GPU support.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Keep JAX from reserving most of the GPU up front.  The wrapper exports the
# same defaults before Python starts, but these protect direct CLI use too.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.40")

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NS_ROOT = PROJECT_ROOT / "2D_NS_FNO2d_recurrent"
PERTURBATION_ROOT = Path(__file__).resolve().parent
if str(NS_ROOT) not in sys.path:
    sys.path.insert(0, str(NS_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PERTURBATION_ROOT) not in sys.path:
    sys.path.insert(0, str(PERTURBATION_ROOT))

from models.FNO2d import FNO2d, RecurrentPredictor
from ns2d_alternative_losses import LOSS3_METRIC_CHOICES, build_loss3_metric


EPS = 1e-12
LOSS_TYPES = ("loss1", "loss2", "loss3")
CORE4_METHODS = ("raw_add", "raw_replace", "steepest_add", "steepest_replace")
MODE_PRESETS = {
    "all_w": "wwwwwwwwww",
    "all_d_target_w": "dddddddddw",
    "all_a_target_w": "aaaaaaaaaw",
    "w1_5_d6_9_target_w": "wwwwwddddw",
    "d1_5_w6_9_target_w": "dddddwwwww",
    "a1_5_d6_9_target_w": "aaaaaddddw",
}

DEFAULT_TEST_PATH = (
    NS_ROOT
    / "datasets"
    / "exponax_datasets"
    / "t20"
    / "real_initial_laxmap_single"
    / "test"
    / "dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_test_ntimepoints21_all_frames.pt"
)
DEFAULT_DICTIONARY_PATH = (
    NS_ROOT
    / "datasets"
    / "exponax_datasets"
    / "t20"
    / "dictionary"
    / "dim2d_nx256_N2000_solver=exponax_nu0.000_t20.0_dict_ntimepoints21_batch0_all_frames.pt"
)
DEFAULT_OUT_ROOT = NS_ROOT / "perturbation_results" / "ns2d_recurrent_core4_attack"


@dataclass(frozen=True)
class MethodSpec:
    name: str
    direction_rule: str
    proposal_rule: str


METHOD_SPECS = {
    "raw_add": MethodSpec("raw_add", "raw_gradient", "additive"),
    "raw_replace": MethodSpec("raw_replace", "unit_raw_gradient", "replacement"),
    "steepest_add": MethodSpec("steepest_add", "lp_steepest", "additive"),
    "steepest_replace": MethodSpec("steepest_replace", "lp_steepest", "replacement"),
}


def parse_norm_order(value: str | float | int) -> float:
    text = str(value).lower()
    if text in {"inf", "linf", "infinity"}:
        return float("inf")
    out = float(text)
    if out <= 0:
        raise ValueError(f"Norm order must be positive, got {value!r}")
    return out


def norm_name(order: float) -> str:
    return "inf" if math.isinf(order) else f"{order:g}"


def batch_flat(x):
    return x.reshape(x.shape[0], -1)


def batch_norm(x, order: float):
    import torch

    flat = batch_flat(x)
    if math.isinf(order):
        return flat.abs().amax(dim=1)
    return torch.linalg.vector_norm(flat, ord=order, dim=1)


def normalize_to_p_ball(v, p_order: float):
    flat = batch_flat(v)
    if math.isinf(p_order):
        denom = flat.abs().amax(dim=1, keepdim=True).clamp_min(EPS)
        return (flat / denom).reshape_as(v)
    norms = batch_norm(v, p_order).reshape(-1, 1).clamp_min(EPS)
    return (flat / norms).reshape_as(v)


def steepest_direction(grad, p_order: float):
    """Argmax over ||s||_p <= 1 of <grad, s>, batchwise."""
    import torch

    flat = batch_flat(grad)
    if math.isinf(p_order):
        return torch.sign(grad)
    if p_order == 1.0:
        out = torch.zeros_like(flat)
        idx = torch.argmax(flat.abs(), dim=1, keepdim=True)
        out.scatter_(1, idx, torch.sign(torch.gather(flat, 1, idx)))
        return out.reshape_as(grad)
    if p_order == 2.0:
        return normalize_to_p_ball(grad, 2.0)
    p_dual = p_order / (p_order - 1.0)
    mapped = torch.sign(grad) * torch.clamp(grad.abs(), min=1e-30).pow(p_dual - 1.0)
    return normalize_to_p_ball(mapped, p_order)


def project_delta(delta, epsilon: float, p_order: float):
    import torch

    flat = batch_flat(delta)
    if math.isinf(p_order):
        return torch.clamp(delta, -epsilon, epsilon)
    norms = batch_norm(delta, p_order).reshape(-1, 1).clamp_min(EPS)
    scale = torch.clamp(float(epsilon) / norms, max=1.0)
    return (flat * scale).reshape_as(delta)


def tensor_to_numpy(x) -> np.ndarray:
    return x.detach().cpu().numpy()


def contiguous_detached(x):
    x = x.detach()
    return x if x.is_contiguous() else x.contiguous()


def finite_stats(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def metric_diag_to_numpy(diag: dict[str, Any] | None) -> dict[str, Any]:
    if not diag:
        return {}
    out: dict[str, Any] = {}
    for key, value in diag.items():
        if hasattr(value, "detach"):
            arr = tensor_to_numpy(value.detach())
            out[key] = arr.astype(np.float64) if np.issubdtype(arr.dtype, np.number) else arr
        elif isinstance(value, (int, float, np.number)):
            out[key] = float(value)
        else:
            out[key] = str(value)
    return out


def add_loss3_metric_diag(row: dict[str, Any], diag: dict[str, Any], sample_pos: int | None = None) -> None:
    for key, value in diag.items():
        out_key = f"loss3_metric_{key}"
        if isinstance(value, str):
            row[out_key] = value
            continue
        arr = np.asarray(value)
        if arr.ndim == 0:
            row[out_key] = float(arr)
            continue
        if sample_pos is None:
            for stat_key, stat_value in finite_stats(arr).items():
                row[f"{out_key}_{stat_key}"] = stat_value
            continue
        if arr.shape[0] > sample_pos:
            sample_arr = np.asarray(arr[sample_pos], dtype=np.float64)
            finite = sample_arr[np.isfinite(sample_arr)]
            row[out_key] = float(np.mean(finite)) if finite.size else float("nan")


def ensure_cuda_or_die() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this attack; refusing to fall back to CPU.")


def gpu_manifest() -> dict[str, Any]:
    import jax
    import torch

    manifest: dict[str, Any] = {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "jax_version": jax.__version__,
        "jax_backend": jax.default_backend(),
        "xla_python_client_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
        "xla_python_client_mem_fraction": os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION"),
        "xla_python_client_allocator": os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR"),
    }
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        manifest.update(
            {
                "torch_device_index": int(idx),
                "torch_device_name": torch.cuda.get_device_name(idx),
                "torch_compute_capability": f"sm_{props.major}{props.minor}",
                "torch_total_memory_bytes": int(props.total_memory),
                "torch_arch_list": getattr(torch.cuda, "get_arch_list", lambda: [])(),
            }
        )
    try:
        manifest["jax_devices"] = [str(device) for device in jax.devices()]
    except Exception as exc:
        manifest["jax_devices_error"] = repr(exc)
    try:
        manifest["nvidia_smi"] = subprocess.check_output(["nvidia-smi"], text=True, stderr=subprocess.STDOUT, timeout=20)
    except Exception as exc:
        manifest["nvidia_smi_error"] = repr(exc)
    return manifest


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class JaxPDEWrapper:
    _function = None

    @classmethod
    def function(cls):
        if cls._function is not None:
            return cls._function

        import jax
        import torch

        class _JaxPDEFunction(torch.autograd.Function):
            _vjp_cache: dict[int, Any] = {}

            @staticmethod
            def forward(ctx, a_torch, fn):
                if not a_torch.is_cuda:
                    raise ValueError("JAX PDE bridge expects a CUDA tensor.")
                a_for_jax = contiguous_detached(a_torch)
                a_jax = jax.dlpack.from_dlpack(a_for_jax)
                out_jax = fn(a_jax)
                out_torch = torch.utils.dlpack.from_dlpack(out_jax)
                ctx.save_for_backward(a_for_jax)
                ctx.fn = fn
                return out_torch

            @staticmethod
            def backward(ctx, grad_output):
                (a_torch,) = ctx.saved_tensors
                fn = ctx.fn
                cache_key = id(fn)
                if cache_key not in _JaxPDEFunction._vjp_cache:
                    def jax_vjp(a_jax, grad_jax):
                        _, vjp_fn = jax.vjp(fn, a_jax)
                        return vjp_fn(grad_jax)

                    _JaxPDEFunction._vjp_cache[cache_key] = jax.jit(jax_vjp)
                grad_jax = jax.dlpack.from_dlpack(contiguous_detached(grad_output))
                a_jax = jax.dlpack.from_dlpack(contiguous_detached(a_torch))
                (grad_input_jax,) = _JaxPDEFunction._vjp_cache[cache_key](a_jax, grad_jax)
                grad_input = torch.utils.dlpack.from_dlpack(grad_input_jax)
                return grad_input, None

        cls._function = _JaxPDEFunction
        return cls._function


def make_ns_rollout_function(nu: float, t_final: int, fixed_step: float, remat_mode: str = "micro", remat_chunk_steps: int = 20):
    import exponax as ex
    import jax
    import jax.numpy as jnp

    if abs(round(1.0 / fixed_step) - (1.0 / fixed_step)) > 1e-12:
        raise ValueError("fixed_step must divide one second exactly.")
    steps_per_second = int(round(1.0 / fixed_step))
    remat_mode = str(remat_mode).lower()
    remat_chunk_steps = int(remat_chunk_steps)
    if remat_mode not in {"micro", "none", "chunk", "second"}:
        raise ValueError(f"Unknown remat mode {remat_mode!r}")
    if remat_mode == "chunk" and (remat_chunk_steps <= 0 or steps_per_second % remat_chunk_steps != 0):
        raise ValueError("For solver_remat=chunk, remat_chunk_steps must be a positive divisor of steps_per_second.")

    def one_sample(u0):
        u = jnp.rot90(jnp.flip(u0, axis=-2), 3, axes=(-2, -1))
        stepper = ex.stepper._navier_stokes.NavierStokesVorticityZongyi(2, 1, u.shape[0], fixed_step, diffusivity=nu, order=4)

        def step_once(carry):
            return stepper(carry[None, ...])[0]

        def scan_steps(carry, length: int):
            def body(c, _):
                return step_once(c), None
            out, _ = jax.lax.scan(body, carry, None, length=length)
            return out

        if remat_mode == "micro":
            step_once_checked = jax.checkpoint(step_once)

            def one_second_impl(carry):
                def micro(c, _):
                    return step_once_checked(c), None
                u_next, _ = jax.lax.scan(micro, carry, None, length=steps_per_second)
                return u_next
        elif remat_mode == "none":
            def one_second_impl(carry):
                return scan_steps(carry, steps_per_second)
        elif remat_mode == "chunk":
            def chunk_impl(carry):
                return scan_steps(carry, remat_chunk_steps)

            chunk_impl_checked = jax.checkpoint(chunk_impl)

            def one_second_impl(carry):
                def chunk_body(c, _):
                    return chunk_impl_checked(c), None
                u_next, _ = jax.lax.scan(chunk_body, carry, None, length=steps_per_second // remat_chunk_steps)
                return u_next
        else:
            def one_second_raw(carry):
                return scan_steps(carry, steps_per_second)

            one_second_checked = jax.checkpoint(one_second_raw)

            def one_second_impl(carry):
                return one_second_checked(carry)

        def one_second(carry, _):
            u_next = one_second_impl(carry)
            return u_next, u_next

        _, seconds = jax.lax.scan(one_second, u, None, length=int(t_final))
        seq = jnp.concatenate([u[None, ...], seconds], axis=0)
        return jnp.swapaxes(seq, -1, -2)

    return jax.jit(jax.vmap(one_sample, in_axes=0, out_axes=1))


class DifferentiableNSRollout:
    def __init__(self, nu: float, fixed_step: float, remat_mode: str = "micro", remat_chunk_steps: int = 20):
        self.nu = float(nu)
        self.fixed_step = float(fixed_step)
        self.remat_mode = str(remat_mode).lower()
        self.remat_chunk_steps = int(remat_chunk_steps)
        self._cache: dict[int, Any] = {}
        self.trace: list[dict[str, Any]] = []

    def rollout(self, x0, t_final: int, context: dict[str, Any] | None = None):
        t_final_int = int(t_final)
        cache_hit = t_final_int in self._cache
        if not cache_hit:
            self._cache[t_final_int] = make_ns_rollout_function(self.nu, t_final_int, self.fixed_step, self.remat_mode, self.remat_chunk_steps)
        out = JaxPDEWrapper.function().apply(x0.contiguous(), self._cache[t_final_int])
        steps_per_second = int(round(1.0 / self.fixed_step))
        row = {
            "call_index": len(self.trace),
            "t_final": t_final_int,
            "fixed_step": self.fixed_step,
            "steps_per_second": steps_per_second,
            "micro_steps_per_sample": t_final_int * steps_per_second,
            "solver_remat": self.remat_mode,
            "solver_remat_chunk_steps": self.remat_chunk_steps,
            "x_shape": "x".join(str(v) for v in x0.shape),
            "output_shape": "x".join(str(v) for v in out.shape),
            "requires_grad": bool(x0.requires_grad),
            "device": str(x0.device),
            "dtype": str(x0.dtype),
            "cache_hit": bool(cache_hit),
        }
        if context:
            for key, value in context.items():
                if isinstance(value, (list, tuple)):
                    row[key] = ",".join(str(v) for v in value)
                else:
                    row[key] = value
        self.trace.append(row)
        return out


class ApproximateDictionary:
    def __init__(self, path: Path, device, chunk_size: int = 32):
        import torch

        self.path = Path(path)
        self.device = device
        self.chunk_size = int(chunk_size)
        payload = torch.load(self.path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict):
            self.x_cpu = payload["x"].float().cpu()
            self.y_cpu = payload["y"].float().cpu()
        else:
            tensor = payload.float().cpu()
            if tensor.ndim != 4:
                raise ValueError(f"Expected dictionary dict or [N,H,W,T] tensor, got {tuple(tensor.shape)}")
            self.y_cpu = tensor
            self.x_cpu = tensor[..., 0]
        if self.y_cpu.ndim != 4:
            raise ValueError(f"Expected dictionary y shape [N,H,W,T], got {tuple(self.y_cpu.shape)}")
        self.x_flat_cpu = self.x_cpu.reshape(self.x_cpu.shape[0], -1).contiguous()
        self.num_grid_points = int(self.x_flat_cpu.shape[1])
        self.x_norm_cpu = self.x_flat_cpu.square().mean(dim=1).contiguous()

    def find(self, x0):
        import torch

        if x0.ndim != 3:
            raise ValueError(f"Expected x0 [B,H,W], got {tuple(x0.shape)}")
        queries = x0.detach()
        bsz = queries.shape[0]
        q_flat = queries.reshape(bsz, -1)
        q_norm = q_flat.square().mean(dim=1, keepdim=True)
        best_mse = torch.full((bsz,), float("inf"), device=self.device)
        best_idx = torch.zeros((bsz,), dtype=torch.long, device=self.device)
        with torch.no_grad():
            for start in range(0, self.x_flat_cpu.shape[0], self.chunk_size):
                end = min(start + self.chunk_size, self.x_flat_cpu.shape[0])
                chunk = self.x_flat_cpu[start:end].to(self.device, non_blocking=True)
                chunk_norm = self.x_norm_cpu[start:end].to(self.device, non_blocking=True)
                # ||q-c||^2 = ||q||^2 + ||c||^2 - 2 q.c. This avoids the
                # [B, chunk, H*W] diff tensor and cuts temporary GPU memory.
                mse = q_norm + chunk_norm.unsqueeze(0) - (2.0 / self.num_grid_points) * q_flat.matmul(chunk.t())
                mse = mse.clamp_min_(0.0)
                local_mse, local_idx = torch.min(mse, dim=1)
                replace = local_mse < best_mse
                best_mse = torch.where(replace, local_mse, best_mse)
                best_idx = torch.where(replace, local_idx + start, best_idx)
                del chunk, chunk_norm, mse
            y = self.y_cpu[best_idx.cpu()].to(self.device, non_blocking=True)
        return y.detach(), best_idx.detach(), best_mse.detach()


def resolve_mode_spec(value: str) -> str:
    mode_spec = MODE_PRESETS.get(value, value).lower()
    if len(mode_spec) != 10 or any(ch not in "adw" for ch in mode_spec):
        raise ValueError("mode_spec must be 10 characters from {a,d,w}, or a known preset.")
    return mode_spec


def method_direction(spec: MethodSpec, grad, p_order: float):
    if spec.direction_rule == "raw_gradient":
        return grad
    if spec.direction_rule == "unit_raw_gradient":
        return normalize_to_p_ball(grad, p_order)
    if spec.direction_rule == "lp_steepest":
        return steepest_direction(grad, p_order)
    raise ValueError(spec.direction_rule)


def propose_delta(spec: MethodSpec, delta, direction, epsilon: float, alpha: float, p_order: float):
    if spec.proposal_rule == "additive":
        proposal = delta + float(alpha) * direction
    elif spec.proposal_rule == "replacement":
        proposal = float(epsilon) * direction
    else:
        raise ValueError(spec.proposal_rule)
    return proposal, project_delta(proposal, epsilon, p_order)


def format_float_tag(value: float) -> str:
    text = f"{float(value):.8g}"
    return text.replace("-", "m").replace("+", "").replace(".", "p")


def parse_epsilon_alpha_pair(text: str) -> tuple[float, float]:
    for sep in (":", ",", "/"):
        if sep in text:
            left, right = text.split(sep, 1)
            return float(left), float(right)
    raise ValueError(f"Expected epsilon/alpha pair like 32:1, got {text!r}")


def resolve_parameter_sweep(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.epsilon_alpha_pairs:
        pairs = [parse_epsilon_alpha_pair(item) for item in args.epsilon_alpha_pairs]
    else:
        epsilons = list(args.epsilons) if args.epsilons is not None else [args.epsilon]
        alphas = list(args.alphas) if args.alphas is not None else [args.alpha]
        pairs = list(itertools.product(epsilons, alphas))
    out = []
    seen: set[tuple[float, float]] = set()
    for epsilon, alpha in pairs:
        epsilon = float(epsilon)
        alpha = float(alpha)
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        key = (epsilon, alpha)
        if key in seen:
            continue
        seen.add(key)
        out.append({"epsilon": epsilon, "alpha": alpha, "tag": f"eps{format_float_tag(epsilon)}_alpha{format_float_tag(alpha)}"})
    if not out:
        raise ValueError("At least one epsilon/alpha setting is required.")
    return out


def stable_int_seed(*parts: Any) -> int:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**63 - 1)


def load_initial_delta_from_npz(problem: "NS2DRecurrentProblem", dataset_indices: list[int]):
    import torch

    args = problem.args
    path = getattr(args, "initial_delta_npz", None)
    if path is None:
        return None
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"initial delta npz not found: {path}")
    key = str(getattr(args, "initial_delta_key", "final_delta"))
    with np.load(path) as data:
        if key not in data:
            raise KeyError(f"initial delta key {key!r} not found in {path}; available keys: {list(data.files)}")
        delta_all = np.asarray(data[key], dtype=np.float32)
        source_indices = np.asarray(data.get("dataset_indices", []), dtype=np.int64)
    if source_indices.size:
        index_to_pos = {int(index): pos for pos, index in enumerate(source_indices.tolist())}
        missing = [int(index) for index in dataset_indices if int(index) not in index_to_pos]
        if missing:
            raise ValueError(f"initial delta npz {path} is missing dataset indices {missing}; source has {source_indices.tolist()}")
        selected = delta_all[[index_to_pos[int(index)] for index in dataset_indices]]
    else:
        if delta_all.shape[0] != len(dataset_indices):
            raise ValueError(f"initial delta npz {path} has no dataset_indices and batch mismatch: {delta_all.shape[0]} vs {len(dataset_indices)}")
        selected = delta_all
    if tuple(selected.shape) != tuple(problem.x0.shape):
        raise ValueError(f"initial delta shape {selected.shape} does not match current batch shape {tuple(problem.x0.shape)}")
    delta = torch.as_tensor(selected, device=problem.device, dtype=problem.x0.dtype)
    return project_delta(delta, args.epsilon, args.p_order).detach()


def initial_delta_for_loss(problem: "NS2DRecurrentProblem", loss_type: str, dataset_indices: list[int]):
    import torch

    args = problem.args
    resumed = load_initial_delta_from_npz(problem, dataset_indices)
    if resumed is not None:
        return resumed

    delta = torch.zeros_like(problem.x0)
    if loss_type != "loss1" or not args.loss1_random_start:
        return delta
    radius = float(args.loss1_random_start_fraction) * float(args.epsilon)
    if radius <= 0:
        return delta
    seed = stable_int_seed(
        int(args.loss1_random_start_seed),
        "loss1_random_start",
        problem.mode_spec,
        norm_name(args.p_order),
        norm_name(args.q_order),
        ",".join(str(i) for i in dataset_indices),
    )
    generator = torch.Generator(device=problem.device)
    generator.manual_seed(seed)
    noise = torch.randn(delta.shape, device=delta.device, dtype=delta.dtype, generator=generator)
    direction = normalize_to_p_ball(noise, args.p_order)
    return project_delta(radius * direction, args.epsilon, args.p_order).detach()


def delta_stats(delta, p_order: float) -> dict[str, float]:
    l2 = tensor_to_numpy(batch_norm(delta, 2.0)).astype(np.float64)
    linf = tensor_to_numpy(batch_norm(delta, float("inf"))).astype(np.float64)
    pnorm = linf if math.isinf(p_order) else tensor_to_numpy(batch_norm(delta, p_order)).astype(np.float64)
    return {
        "initial_delta_l2_mean": float(np.nanmean(l2)),
        "initial_delta_l2_max": float(np.nanmax(l2)),
        "initial_delta_linf_mean": float(np.nanmean(linf)),
        "initial_delta_linf_max": float(np.nanmax(linf)),
        "initial_delta_p_mean": float(np.nanmean(pnorm)),
        "initial_delta_p_max": float(np.nanmax(pnorm)),
    }


class NS2DRecurrentProblem:
    def __init__(self, args: argparse.Namespace, x0, clean_target=None, model=None, rollout_solver=None, dictionary=None):
        import torch

        self.args = args
        self.device = torch.device(args.device)
        self.x0 = x0.to(self.device, dtype=torch.float32).detach()
        self.clean_target_from_data = None if clean_target is None else clean_target.to(self.device, dtype=torch.float32).detach()
        self.mode_spec = resolve_mode_spec(args.mode_spec)
        self.loss3_metric = build_loss3_metric(args).to(self.device)
        if args.require_target_w and self.mode_spec[-1] != "w":
            raise ValueError("The target-frame mode must be 'w'. Use --no-require-target-w only for ablations.")

        # These objects are shared at run scope so each attack batch does not
        # reload checkpoint weights, rebuild solver caches, or reload the large
        # approximate dictionary from CPU storage.
        self.model = model if model is not None else load_recurrent_model(args, self.device)
        self.rollout_solver = rollout_solver if rollout_solver is not None else DifferentiableNSRollout(args.nu, args.fixed_step, args.solver_remat, args.solver_remat_chunk_steps)
        self.dictionary: ApproximateDictionary | None = None
        if "a" in self.mode_spec:
            self.dictionary = dictionary if dictionary is not None else ApproximateDictionary(args.dictionary_path, self.device, args.dictionary_chunk_size)

        with torch.no_grad():
            self.f0 = self.model_prediction(self.x0, need_target=False)[0].detach()
            if args.clean_target_source == "dataset" and self.clean_target_from_data is not None:
                self.g0 = self.clean_target_from_data.detach()
            else:
                self.g0 = self.target_only(self.x0, mode="w").detach()

    def _dictionary_frames(self, x_adv):
        if self.dictionary is None:
            raise ValueError("A mode was requested but no dictionary is loaded.")
        y, idx, dist = self.dictionary.find(x_adv)
        return y, idx, dist

    def _rollout_for_modes(self, x_adv, need_target: bool):
        required: list[int] = []
        for frame_index in range(1, self.args.t_in):
            if self.mode_spec[frame_index - 1] in {"d", "w"}:
                required.append(frame_index)
        if need_target and self.mode_spec[-1] in {"d", "w"}:
            required.append(self.args.target_frame_index)
        if not required:
            return None
        return self.rollout_solver.rollout(x_adv, max(required), context={"source": "model_prediction", "mode_spec": self.mode_spec, "need_target": bool(need_target), "required_frames": required})

    def model_prediction(self, x_adv, need_target: bool):
        seq = self._rollout_for_modes(x_adv, need_target=need_target)
        dict_y = dict_idx = dict_dist = None
        if "a" in self.mode_spec:
            dict_y, dict_idx, dict_dist = self._dictionary_frames(x_adv)

        frames = [x_adv]
        for frame_index in range(1, self.args.t_in):
            mode = self.mode_spec[frame_index - 1]
            if mode == "a":
                frame = dict_y[..., frame_index]
            else:
                if seq is None or seq.shape[0] <= frame_index:
                    raise RuntimeError(f"Solver rollout does not contain frame {frame_index}.")
                frame = seq[frame_index]
                if mode == "d":
                    frame = frame.detach()
            frames.append(frame)
        model_input = torch_stack_last(frames)
        pred_final = self.model(model_input)[..., -1]

        target = None
        if need_target:
            mode = self.mode_spec[-1]
            if mode == "a":
                target = dict_y[..., self.args.target_frame_index]
            else:
                if seq is None or seq.shape[0] <= self.args.target_frame_index:
                    raise RuntimeError(f"Solver rollout does not contain target frame {self.args.target_frame_index}.")
                target = seq[self.args.target_frame_index]
                if mode == "d":
                    target = target.detach()
        return pred_final, target, {"dictionary_idx": dict_idx, "dictionary_mse": dict_dist}

    def target_only(self, x_adv, mode: str):
        if mode == "a":
            y, _, _ = self._dictionary_frames(x_adv)
            return y[..., self.args.target_frame_index].detach()
        seq = self.rollout_solver.rollout(x_adv, self.args.target_frame_index, context={"source": "target_only", "mode_spec": mode, "need_target": True, "required_frames": [self.args.target_frame_index]})
        target = seq[self.args.target_frame_index]
        if mode == "d":
            target = target.detach()
        return target

    def all_losses(self, x_adv):
        pred, g_delta, aux = self.model_prediction(x_adv, need_target=True)
        loss3_result = self.loss3_metric(pred, g_delta)
        losses = {
            "loss1": batch_norm(pred - self.f0, self.args.q_order),
            "loss2": batch_norm(pred - self.g0, self.args.q_order),
            "loss3": loss3_result.values,
        }
        aux["loss3_metric_diagnostics"] = loss3_result.diagnostics
        aux["loss3_metric_alignment_fields"] = loss3_result.alignment_fields
        return losses, pred, g_delta, aux

    def active_losses(self, x_adv, loss_type: str):
        import torch

        need_target = loss_type == "loss3"
        pred, g_delta, aux = self.model_prediction(x_adv, need_target=need_target)
        nan = torch.full((x_adv.shape[0],), float("nan"), device=x_adv.device, dtype=x_adv.dtype)
        losses = {
            "loss1": batch_norm(pred - self.f0, self.args.q_order),
            "loss2": batch_norm(pred - self.g0, self.args.q_order),
            "loss3": nan,
        }
        if need_target:
            loss3_result = self.loss3_metric(pred, g_delta)
            losses["loss3"] = loss3_result.values
            aux["loss3_metric_diagnostics"] = loss3_result.diagnostics
            aux["loss3_metric_alignment_fields"] = loss3_result.alignment_fields
        return losses, pred, g_delta, aux

    def all_w_final_outputs(self, x_adv, source: str = "all_w_final_outputs"):
        """Evaluate all-W recurrent FNO and solver final-frame outputs.

        This is the canonical final-state record path used for true-loss curves,
        full-batch final output archives, and sample trace archives. It always
        detaches from the attack graph so recording does not change gradients.
        """

        import torch

        with torch.no_grad():
            x_eval = x_adv.detach()
            seq = self.rollout_solver.rollout(
                x_eval,
                self.args.target_frame_index,
                context={
                    "source": source,
                    "mode_spec": "wwwwwwwwww",
                    "need_target": True,
                    "required_frames": list(range(1, self.args.t_in)) + [self.args.target_frame_index],
                },
            )
            frames = [x_eval] + [seq[frame_index] for frame_index in range(1, self.args.t_in)]
            pred = self.model(torch_stack_last(frames))[..., -1]
            target = seq[self.args.target_frame_index]
            true_loss = batch_norm(pred - target, self.args.q_order)
            return true_loss, pred, target

    def true_loss_all_w(self, x_adv):
        """Evaluate the full-solver true loss curve without contributing gradients."""

        return self.all_w_final_outputs(x_adv, source="true_loss_all_w")[0]


def torch_stack_last(frames: list[Any]):
    import torch

    return torch.stack(frames, dim=-1)


def load_recurrent_model(args: argparse.Namespace, device):
    import torch

    model = FNO2d(modes1=args.modes1, modes2=args.modes2, width=args.width, num_layers=args.num_layers, in_channels=args.t_in).to(device)
    # Load the full training checkpoint on CPU first. The .pt checkpoint also
    # contains optimizer state, which is much larger than the model weights and
    # should not be moved to GPU for attack/inference.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        state = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint.get("model") or checkpoint
    else:
        state = checkpoint
    cleaned = {}
    for key, value in state.items():
        new_key = key
        for prefix in ("module.", "model."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned[new_key] = value
    model.load_state_dict(cleaned, strict=not args.non_strict_checkpoint)
    recurrent = RecurrentPredictor(model, T_out=args.t_out, step=args.step).to(device)
    recurrent.eval()
    for param in recurrent.parameters():
        param.requires_grad_(False)
    return recurrent


def load_initial_conditions(args: argparse.Namespace):
    import torch

    data = torch.load(args.test_path, map_location="cpu", weights_only=False)
    indices = parse_indices(args.indices, args.start_index, args.num_samples)
    if "x" in data:
        x_all = data["x"].float()
    elif "y" in data:
        x_all = data["y"][..., 0].float()
    else:
        raise ValueError("Expected test data to contain key 'x' or 'y'.")
    y_all = data.get("y")
    clean_targets = None
    if y_all is not None and y_all.ndim == 4 and y_all.shape[-1] > args.target_frame_index:
        clean_targets = y_all[indices, ..., args.target_frame_index].float()
    return x_all[indices].float(), clean_targets, indices


def parse_indices(text: str, start_index: int, num_samples: int) -> list[int]:
    if text:
        return [int(part.strip()) for part in text.split(",") if part.strip()]
    return list(range(int(start_index), int(start_index) + int(num_samples)))


def append_metric_rows(
    step_rows,
    sample_rows,
    loss_type,
    method,
    mode_spec,
    k,
    dataset_indices,
    losses_np,
    delta_l2,
    delta_linf,
    delta_p,
    true_loss_np,
    wq_l2_np,
    grad_l2_mean,
    direction_l2_mean,
    elapsed,
    args,
    loss3_metric_diag=None,
):
    row = {
        "loss_type": loss_type,
        "method": method,
        "mode_spec": mode_spec,
        "k": int(k),
        "epsilon": float(args.epsilon),
        "alpha": float(args.alpha),
        "p_order": norm_name(args.p_order),
        "q_order": norm_name(args.q_order),
        "seconds_since_method_start": float(elapsed),
    }
    for name in LOSS_TYPES:
        for key, value in finite_stats(losses_np[name]).items():
            row[f"{name}_{key}"] = value
    for name, values in {"delta_l2": delta_l2, "delta_linf": delta_linf, "delta_p": delta_p}.items():
        for key, value in finite_stats(values).items():
            row[f"{name}_{key}"] = value
    for key, value in finite_stats(true_loss_np).items():
        row[f"true_loss_{key}"] = value
    for key, value in finite_stats(wq_l2_np).items():
        row[f"wq_l2_{key}"] = value
    row["boundary_ratio_mean"] = float(np.nanmean(delta_p / float(args.epsilon)))
    if grad_l2_mean is not None:
        row["grad_l2_mean"] = float(grad_l2_mean)
    if direction_l2_mean is not None:
        row["direction_l2_mean"] = float(direction_l2_mean)
    row["loss3_metric"] = str(getattr(args, "loss3_metric", "qnorm"))
    if loss3_metric_diag:
        add_loss3_metric_diag(row, loss3_metric_diag)
    step_rows.append(row)

    for pos, dataset_index in enumerate(dataset_indices):
        sample = {
            "loss_type": loss_type,
            "method": method,
            "mode_spec": mode_spec,
            "k": int(k),
            "sample_position": int(pos),
            "dataset_index": int(dataset_index),
            "epsilon": float(args.epsilon),
            "alpha": float(args.alpha),
            "p_order": norm_name(args.p_order),
            "q_order": norm_name(args.q_order),
            "seconds_since_method_start": float(elapsed),
            "delta_l2": float(delta_l2[pos]),
            "delta_linf": float(delta_linf[pos]),
            "delta_p": float(delta_p[pos]),
            "boundary_ratio": float(delta_p[pos] / float(args.epsilon)),
            "true_loss": float(true_loss_np[pos]),
            "wq_l2": float(wq_l2_np[pos]),
        }
        for name in LOSS_TYPES:
            sample[name] = float(losses_np[name][pos])
        sample["loss3_metric"] = str(getattr(args, "loss3_metric", "qnorm"))
        if loss3_metric_diag:
            add_loss3_metric_diag(sample, loss3_metric_diag, sample_pos=pos)
        sample_rows.append(sample)


def annotate_growth_metrics(step_rows: list[dict[str, Any]], sample_rows: list[dict[str, Any]], active_loss: str) -> None:
    """Add surrogate and true loss curve/growth metrics in-place."""

    mean_key = f"{active_loss}_mean"
    first_surrogate = None
    first_true = None
    prev_surrogate = None
    prev_true = None
    prev_seconds = None
    for row in step_rows:
        surrogate = float(row.get(mean_key, float("nan")))
        true_value = float(row.get("true_loss_mean", float("nan")))
        seconds = float(row.get("seconds_since_method_start", float("nan")))
        row["active_loss"] = active_loss
        row["active_loss_mean"] = surrogate
        row["surrogate_loss_mean"] = surrogate
        if first_surrogate is None and math.isfinite(surrogate):
            first_surrogate = surrogate
        if first_true is None and math.isfinite(true_value):
            first_true = true_value

        row["active_loss_mean_increase_from_k0"] = surrogate - first_surrogate if first_surrogate is not None and math.isfinite(surrogate) else float("nan")
        row["active_loss_mean_ratio_to_k0"] = surrogate / first_surrogate if first_surrogate not in (None, 0.0) and math.isfinite(surrogate) else float("nan")
        row["surrogate_loss_mean_increase_from_k0"] = row["active_loss_mean_increase_from_k0"]
        row["surrogate_loss_mean_ratio_to_k0"] = row["active_loss_mean_ratio_to_k0"]
        row["true_loss_mean_increase_from_k0"] = true_value - first_true if first_true is not None and math.isfinite(true_value) else float("nan")
        row["true_loss_mean_ratio_to_k0"] = true_value / first_true if first_true not in (None, 0.0) and math.isfinite(true_value) else float("nan")

        if prev_surrogate is None or not math.isfinite(surrogate) or not math.isfinite(prev_surrogate):
            row["active_loss_mean_delta_from_prev"] = float("nan")
            row["active_loss_mean_growth_per_second"] = float("nan")
            row["surrogate_loss_mean_delta_from_prev"] = float("nan")
            row["surrogate_loss_mean_growth_per_second"] = float("nan")
        else:
            delta = surrogate - prev_surrogate
            dt = seconds - prev_seconds if prev_seconds is not None else float("nan")
            row["active_loss_mean_delta_from_prev"] = delta
            row["active_loss_mean_growth_per_second"] = delta / dt if dt and dt > 0 else float("nan")
            row["surrogate_loss_mean_delta_from_prev"] = delta
            row["surrogate_loss_mean_growth_per_second"] = row["active_loss_mean_growth_per_second"]

        if prev_true is None or not math.isfinite(true_value) or not math.isfinite(prev_true):
            row["true_loss_mean_delta_from_prev"] = float("nan")
            row["true_loss_mean_growth_per_second"] = float("nan")
        else:
            delta = true_value - prev_true
            dt = seconds - prev_seconds if prev_seconds is not None else float("nan")
            row["true_loss_mean_delta_from_prev"] = delta
            row["true_loss_mean_growth_per_second"] = delta / dt if dt and dt > 0 else float("nan")

        prev_surrogate = surrogate
        prev_true = true_value
        prev_seconds = seconds

    by_sample: dict[int, list[dict[str, Any]]] = {}
    for row in sample_rows:
        by_sample.setdefault(int(row["sample_position"]), []).append(row)
    for rows in by_sample.values():
        rows.sort(key=lambda item: int(item["k"]))
        first_surrogate = None
        first_true = None
        prev_surrogate = None
        prev_true = None
        prev_seconds = None
        for row in rows:
            surrogate = float(row.get(active_loss, float("nan")))
            true_value = float(row.get("true_loss", float("nan")))
            seconds = float(row.get("seconds_since_method_start", float("nan")))
            row["active_loss"] = active_loss
            row["active_loss_value"] = surrogate
            row["surrogate_loss_value"] = surrogate
            if first_surrogate is None and math.isfinite(surrogate):
                first_surrogate = surrogate
            if first_true is None and math.isfinite(true_value):
                first_true = true_value

            row["active_loss_increase_from_k0"] = surrogate - first_surrogate if first_surrogate is not None and math.isfinite(surrogate) else float("nan")
            row["active_loss_ratio_to_k0"] = surrogate / first_surrogate if first_surrogate not in (None, 0.0) and math.isfinite(surrogate) else float("nan")
            row["surrogate_loss_increase_from_k0"] = row["active_loss_increase_from_k0"]
            row["surrogate_loss_ratio_to_k0"] = row["active_loss_ratio_to_k0"]
            row["true_loss_increase_from_k0"] = true_value - first_true if first_true is not None and math.isfinite(true_value) else float("nan")
            row["true_loss_ratio_to_k0"] = true_value / first_true if first_true not in (None, 0.0) and math.isfinite(true_value) else float("nan")

            if prev_surrogate is None or not math.isfinite(surrogate) or not math.isfinite(prev_surrogate):
                row["active_loss_delta_from_prev"] = float("nan")
                row["active_loss_growth_per_second"] = float("nan")
                row["surrogate_loss_delta_from_prev"] = float("nan")
                row["surrogate_loss_growth_per_second"] = float("nan")
            else:
                delta = surrogate - prev_surrogate
                dt = seconds - prev_seconds if prev_seconds is not None else float("nan")
                row["active_loss_delta_from_prev"] = delta
                row["active_loss_growth_per_second"] = delta / dt if dt and dt > 0 else float("nan")
                row["surrogate_loss_delta_from_prev"] = delta
                row["surrogate_loss_growth_per_second"] = row["active_loss_growth_per_second"]

            if prev_true is None or not math.isfinite(true_value) or not math.isfinite(prev_true):
                row["true_loss_delta_from_prev"] = float("nan")
                row["true_loss_growth_per_second"] = float("nan")
            else:
                delta = true_value - prev_true
                dt = seconds - prev_seconds if prev_seconds is not None else float("nan")
                row["true_loss_delta_from_prev"] = delta
                row["true_loss_growth_per_second"] = delta / dt if dt and dt > 0 else float("nan")
            prev_surrogate = surrogate
            prev_true = true_value
            prev_seconds = seconds


def build_delta_threshold_rows(
    step_rows: list[dict[str, Any]],
    sample_rows: list[dict[str, Any]],
    active_loss: str,
    thresholds: tuple[float, ...] = (0.25, 0.50, 0.75, 1.00),
) -> list[dict[str, Any]]:
    """Record when delta_p/epsilon first reaches each threshold."""

    rows: list[dict[str, Any]] = []
    sorted_steps = sorted(step_rows, key=lambda item: int(item["k"]))
    for threshold in thresholds:
        hit = next((row for row in sorted_steps if float(row.get("boundary_ratio_mean", float("nan"))) >= threshold), None)
        rows.append(
            {
                "scope": "batch_mean",
                "threshold": threshold,
                "threshold_percent": int(round(100 * threshold)),
                "first_k": None if hit is None else int(hit["k"]),
                "seconds_since_method_start": None if hit is None else float(hit.get("seconds_since_method_start", float("nan"))),
                "boundary_ratio": None if hit is None else float(hit.get("boundary_ratio_mean", float("nan"))),
                "delta_p": None if hit is None else float(hit.get("delta_p_mean", float("nan"))),
                "active_loss": active_loss,
                "surrogate_loss_value": None if hit is None else float(hit.get("surrogate_loss_mean", float("nan"))),
                "true_loss_value": None if hit is None else float(hit.get("true_loss_mean", float("nan"))),
                "active_loss_value": None if hit is None else float(hit.get("active_loss_mean", float("nan"))),
            }
        )

    by_k: dict[int, list[dict[str, Any]]] = {}
    for row in sample_rows:
        by_k.setdefault(int(row["k"]), []).append(row)
    for threshold in thresholds:
        all_hit = None
        for k in sorted(by_k):
            rows_at_k = by_k[k]
            if rows_at_k and all(float(row.get("boundary_ratio", float("nan"))) >= threshold for row in rows_at_k):
                all_hit = rows_at_k
                break
        representative = None if all_hit is None else all_hit[0]
        rows.append(
            {
                "scope": "all_samples",
                "threshold": threshold,
                "threshold_percent": int(round(100 * threshold)),
                "first_k": None if representative is None else int(representative["k"]),
                "seconds_since_method_start": None if representative is None else float(representative.get("seconds_since_method_start", float("nan"))),
                "boundary_ratio": None if all_hit is None else float(np.nanmin([row["boundary_ratio"] for row in all_hit])),
                "delta_p": None if all_hit is None else float(np.nanmin([row["delta_p"] for row in all_hit])),
                "active_loss": active_loss,
                "surrogate_loss_value": None if all_hit is None else float(np.nanmean([row.get("surrogate_loss_value", float("nan")) for row in all_hit])),
                "true_loss_value": None if all_hit is None else float(np.nanmean([row.get("true_loss", float("nan")) for row in all_hit])),
                "active_loss_value": None if all_hit is None else float(np.nanmean([row.get("active_loss_value", float("nan")) for row in all_hit])),
            }
        )

    by_sample: dict[int, list[dict[str, Any]]] = {}
    for row in sample_rows:
        by_sample.setdefault(int(row["sample_position"]), []).append(row)
    for sample_position, rows_for_sample in sorted(by_sample.items()):
        rows_for_sample.sort(key=lambda item: int(item["k"]))
        dataset_index = int(rows_for_sample[0]["dataset_index"])
        for threshold in thresholds:
            hit = next((row for row in rows_for_sample if float(row.get("boundary_ratio", float("nan"))) >= threshold), None)
            rows.append(
                {
                    "scope": "sample",
                    "sample_position": sample_position,
                    "dataset_index": dataset_index,
                    "threshold": threshold,
                    "threshold_percent": int(round(100 * threshold)),
                    "first_k": None if hit is None else int(hit["k"]),
                    "seconds_since_method_start": None if hit is None else float(hit.get("seconds_since_method_start", float("nan"))),
                    "boundary_ratio": None if hit is None else float(hit.get("boundary_ratio", float("nan"))),
                    "delta_p": None if hit is None else float(hit.get("delta_p", float("nan"))),
                    "active_loss": active_loss,
                    "surrogate_loss_value": None if hit is None else float(hit.get("surrogate_loss_value", float("nan"))),
                    "true_loss_value": None if hit is None else float(hit.get("true_loss", float("nan"))),
                    "active_loss_value": None if hit is None else float(hit.get("active_loss_value", float("nan"))),
                }
            )
    return rows



def batch_numpy_l2(values: np.ndarray) -> np.ndarray:
    flat = values.reshape(values.shape[0], -1).astype(np.float64)
    return np.linalg.norm(flat, ord=2, axis=1)


def batch_numpy_linf(values: np.ndarray) -> np.ndarray:
    flat = values.reshape(values.shape[0], -1).astype(np.float64)
    return np.max(np.abs(flat), axis=1)


def final_state_metric_rows(dataset_indices: list[int], arrays: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    delta = arrays["final_delta"]
    clean_diff = arrays["clean_model_minus_solver"]
    adv_diff = arrays["adv_model_minus_solver"]
    model_change = arrays["model_final_change"]
    solver_change = arrays["solver_final_change"]
    clean_true_loss = arrays["clean_true_loss"]
    adv_true_loss = arrays["adv_true_loss"]
    clean_loss3_metric = arrays.get("clean_loss3_metric")
    adv_loss3_metric = arrays.get("adv_loss3_metric")
    rows = []
    delta_l2 = batch_numpy_l2(delta)
    delta_linf = batch_numpy_linf(delta)
    clean_diff_l2 = batch_numpy_l2(clean_diff)
    adv_diff_l2 = batch_numpy_l2(adv_diff)
    model_change_l2 = batch_numpy_l2(model_change)
    solver_change_l2 = batch_numpy_l2(solver_change)
    for pos, dataset_index in enumerate(dataset_indices):
        rows.append(
            {
                "sample_position": int(pos),
                "dataset_index": int(dataset_index),
                "delta_l2": float(delta_l2[pos]),
                "delta_linf": float(delta_linf[pos]),
                "clean_true_loss": float(clean_true_loss[pos]),
                "adv_true_loss": float(adv_true_loss[pos]),
                "true_loss_increase": float(adv_true_loss[pos] - clean_true_loss[pos]),
                "true_loss_ratio": float(adv_true_loss[pos] / clean_true_loss[pos]) if clean_true_loss[pos] != 0 else float("nan"),
                "clean_model_solver_l2": float(clean_diff_l2[pos]),
                "adv_model_solver_l2": float(adv_diff_l2[pos]),
                "adv_minus_clean_model_solver_l2": float(adv_diff_l2[pos] - clean_diff_l2[pos]),
                "model_final_change_l2": float(model_change_l2[pos]),
                "solver_final_change_l2": float(solver_change_l2[pos]),
            }
        )
        if clean_loss3_metric is not None and adv_loss3_metric is not None:
            rows[-1]["clean_loss3_metric"] = float(clean_loss3_metric[pos])
            rows[-1]["adv_loss3_metric"] = float(adv_loss3_metric[pos])
            rows[-1]["loss3_metric_increase"] = float(adv_loss3_metric[pos] - clean_loss3_metric[pos])
            rows[-1]["loss3_metric_ratio"] = float(adv_loss3_metric[pos] / clean_loss3_metric[pos]) if clean_loss3_metric[pos] != 0 else float("nan")
    return rows


def loss3_alignment_fields_to_arrays(fields: dict[str, Any] | None, prefix: str) -> dict[str, Any]:
    if not fields:
        return {}
    return {
        f"{prefix}_loss3_aligned_model_final": tensor_to_numpy(fields["aligned_model"]).astype(np.float32),
        f"{prefix}_loss3_aligned_model_minus_solver": tensor_to_numpy(fields["aligned_model_minus_solver"]).astype(np.float32),
        f"{prefix}_loss3_alignment_warp_mag": tensor_to_numpy(fields["warp_mag"]).astype(np.float32),
        f"{prefix}_loss3_alignment_available": np.asarray(True, dtype=np.bool_),
        f"{prefix}_loss3_alignment_source": np.asarray(str(fields.get("source", "unknown"))),
    }


def loss3_alignment_arrays(problem: NS2DRecurrentProblem, pred, target, prefix: str) -> dict[str, Any]:
    if not hasattr(problem.loss3_metric, "aligned_fields"):
        return {}
    return loss3_alignment_fields_to_arrays(problem.loss3_metric.aligned_fields(pred.detach(), target.detach()), prefix)


def save_final_state_outputs(
    problem: NS2DRecurrentProblem,
    delta,
    dataset_indices: list[int],
    method_dir: Path,
    runtime_adv_alignment_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not problem.args.record_final_state_outputs:
        return {"record_final_state_outputs": False}

    final_delta = delta.detach()
    x_clean = problem.x0.detach()
    x_adv = (problem.x0 + final_delta).detach()
    clean_true_loss, clean_model, clean_solver = problem.all_w_final_outputs(x_clean, source="record_final_state_clean")
    adv_true_loss, adv_model, adv_solver = problem.all_w_final_outputs(x_adv, source="record_final_state_adv")
    clean_metric_result = problem.loss3_metric(clean_model.detach(), clean_solver.detach())
    adv_metric_result = problem.loss3_metric(adv_model.detach(), adv_solver.detach())
    arrays = {
        "dataset_indices": np.asarray(dataset_indices, dtype=np.int64),
        "x_clean": tensor_to_numpy(x_clean).astype(np.float32),
        "final_delta": tensor_to_numpy(final_delta).astype(np.float32),
        "x_adv": tensor_to_numpy(x_adv).astype(np.float32),
        "clean_model_final": tensor_to_numpy(clean_model).astype(np.float32),
        "clean_solver_final": tensor_to_numpy(clean_solver).astype(np.float32),
        "clean_model_minus_solver": tensor_to_numpy(clean_model - clean_solver).astype(np.float32),
        "adv_model_final": tensor_to_numpy(adv_model).astype(np.float32),
        "adv_solver_final": tensor_to_numpy(adv_solver).astype(np.float32),
        "adv_model_minus_solver": tensor_to_numpy(adv_model - adv_solver).astype(np.float32),
        "model_final_change": tensor_to_numpy(adv_model - clean_model).astype(np.float32),
        "solver_final_change": tensor_to_numpy(adv_solver - clean_solver).astype(np.float32),
        "clean_true_loss": tensor_to_numpy(clean_true_loss).astype(np.float32),
        "adv_true_loss": tensor_to_numpy(adv_true_loss).astype(np.float32),
        "clean_loss3_metric": tensor_to_numpy(clean_metric_result.values.detach()).astype(np.float32),
        "adv_loss3_metric": tensor_to_numpy(adv_metric_result.values.detach()).astype(np.float32),
    }
    arrays.update(loss3_alignment_arrays(problem, clean_model, clean_solver, "clean"))
    if runtime_adv_alignment_fields:
        arrays.update(loss3_alignment_fields_to_arrays(runtime_adv_alignment_fields, "adv"))
    else:
        arrays.update(loss3_alignment_arrays(problem, adv_model, adv_solver, "adv"))
    npz_path = method_dir / "final_state_outputs.npz"
    np.savez_compressed(npz_path, **arrays)
    metric_rows = final_state_metric_rows(dataset_indices, arrays)
    metrics_path = method_dir / "final_state_metrics.csv"
    write_csv(metrics_path, metric_rows)
    return {
        "record_final_state_outputs": True,
        "final_state_outputs_npz": str(npz_path),
        "final_state_metrics_csv": str(metrics_path),
        "final_state_metric_rows": len(metric_rows),
        "final_adv_true_loss_mean": float(np.nanmean(arrays["adv_true_loss"])),
        "final_clean_true_loss_mean": float(np.nanmean(arrays["clean_true_loss"])),
        "final_adv_loss3_metric_mean": float(np.nanmean(arrays["adv_loss3_metric"])),
        "final_clean_loss3_metric_mean": float(np.nanmean(arrays["clean_loss3_metric"])),
    }


def should_record_step_sample(problem: NS2DRecurrentProblem, k: int) -> bool:
    args = problem.args
    if not args.record_step_sample_outputs:
        return False
    every = max(1, int(args.record_step_sample_every))
    return k % every == 0 or k >= int(args.steps)


def init_step_sample_trace() -> dict[str, list[Any]]:
    return {
        "k": [],
        "delta": [],
        "x_adv": [],
        "grad": [],
        "grad_available": [],
        "direction": [],
        "direction_available": [],
        "adv_model_final": [],
        "adv_solver_final": [],
        "adv_model_minus_solver": [],
        "loss3_aligned_model_final": [],
        "loss3_aligned_model_minus_solver": [],
        "loss3_alignment_warp_mag": [],
        "loss3_alignment_available": [],
        "loss3_alignment_source": [],
    }


def append_step_sample_trace(
    problem: NS2DRecurrentProblem,
    trace: dict[str, list[Any]],
    k: int,
    delta,
    x_adv,
    grad,
    direction,
    all_w_pred,
    all_w_target,
    loss3_alignment_fields: dict[str, Any] | None = None,
) -> None:
    pos = int(problem.args.record_step_sample_position)
    if pos < 0 or pos >= x_adv.shape[0]:
        return
    trace["k"].append(np.asarray(k, dtype=np.int64))
    trace["delta"].append(tensor_to_numpy(delta[pos]).astype(np.float32))
    trace["x_adv"].append(tensor_to_numpy(x_adv.detach()[pos]).astype(np.float32))
    if problem.args.record_step_sample_gradients and grad is not None:
        trace["grad"].append(tensor_to_numpy(grad[pos]).astype(np.float32))
        trace["grad_available"].append(np.asarray(True, dtype=np.bool_))
    else:
        trace["grad"].append(np.zeros_like(trace["delta"][-1], dtype=np.float32))
        trace["grad_available"].append(np.asarray(False, dtype=np.bool_))
    if problem.args.record_step_sample_gradients and direction is not None:
        trace["direction"].append(tensor_to_numpy(direction[pos]).astype(np.float32))
        trace["direction_available"].append(np.asarray(True, dtype=np.bool_))
    else:
        trace["direction"].append(np.zeros_like(trace["delta"][-1], dtype=np.float32))
        trace["direction_available"].append(np.asarray(False, dtype=np.bool_))
    pred = all_w_pred.detach()[pos]
    target = all_w_target.detach()[pos]
    pred_np = tensor_to_numpy(pred).astype(np.float32)
    target_np = tensor_to_numpy(target).astype(np.float32)
    trace["adv_model_final"].append(pred_np)
    trace["adv_solver_final"].append(target_np)
    trace["adv_model_minus_solver"].append(tensor_to_numpy(pred - target).astype(np.float32))
    alignment = loss3_alignment_fields
    if alignment is None and hasattr(problem.loss3_metric, "aligned_fields"):
        alignment = problem.loss3_metric.aligned_fields(
            all_w_pred.detach()[pos : pos + 1],
            all_w_target.detach()[pos : pos + 1],
        )
    if alignment:
        aligned_all = tensor_to_numpy(alignment["aligned_model"]).astype(np.float32)
        aligned_diff_all = tensor_to_numpy(alignment["aligned_model_minus_solver"]).astype(np.float32)
        warp_mag_all = tensor_to_numpy(alignment["warp_mag"]).astype(np.float32)
        sample_pos = pos if aligned_all.shape[0] > pos else 0
        aligned = aligned_all[sample_pos]
        aligned_diff = aligned_diff_all[sample_pos]
        warp_mag = warp_mag_all[sample_pos]
        trace["loss3_aligned_model_final"].append(aligned)
        trace["loss3_aligned_model_minus_solver"].append(aligned_diff)
        trace["loss3_alignment_warp_mag"].append(warp_mag)
        trace["loss3_alignment_available"].append(np.asarray(True, dtype=np.bool_))
        trace["loss3_alignment_source"].append(str(alignment.get("source", "unknown")))
    else:
        trace["loss3_aligned_model_final"].append(pred_np.copy())
        trace["loss3_aligned_model_minus_solver"].append((pred_np - target_np).astype(np.float32))
        trace["loss3_alignment_warp_mag"].append(np.zeros_like(pred_np, dtype=np.float32))
        trace["loss3_alignment_available"].append(np.asarray(False, dtype=np.bool_))
        trace["loss3_alignment_source"].append("no_explicit_alignment")


def save_step_sample_trace(problem: NS2DRecurrentProblem, trace: dict[str, list[Any]], sample_rows: list[dict[str, Any]], dataset_indices: list[int], method_dir: Path) -> dict[str, Any]:
    if not problem.args.record_step_sample_outputs or not trace["k"]:
        return {"record_step_sample_outputs": False}
    pos = int(problem.args.record_step_sample_position)
    if pos < 0 or pos >= len(dataset_indices):
        return {"record_step_sample_outputs": False, "record_step_sample_skipped": "sample position out of range"}

    clean_loss, clean_model, clean_solver = problem.all_w_final_outputs(problem.x0[pos : pos + 1], source="record_step_sample_clean")
    trace_path = method_dir / "step_sample_trace.npz"
    payload = {
        "sample_position": np.asarray(pos, dtype=np.int64),
        "dataset_index": np.asarray(int(dataset_indices[pos]), dtype=np.int64),
        "k": np.asarray(trace["k"], dtype=np.int64),
        "x_clean": tensor_to_numpy(problem.x0[pos]).astype(np.float32),
        "clean_model_final": tensor_to_numpy(clean_model[0]).astype(np.float32),
        "clean_solver_final": tensor_to_numpy(clean_solver[0]).astype(np.float32),
        "clean_model_minus_solver": tensor_to_numpy((clean_model - clean_solver)[0]).astype(np.float32),
        "clean_true_loss": np.asarray(tensor_to_numpy(clean_loss)[0], dtype=np.float32),
        "delta": np.stack(trace["delta"], axis=0),
        "x_adv": np.stack(trace["x_adv"], axis=0),
        "grad": np.stack(trace["grad"], axis=0),
        "grad_available": np.asarray(trace["grad_available"], dtype=np.bool_),
        "direction": np.stack(trace["direction"], axis=0),
        "direction_available": np.asarray(trace["direction_available"], dtype=np.bool_),
        "adv_model_final": np.stack(trace["adv_model_final"], axis=0),
        "adv_solver_final": np.stack(trace["adv_solver_final"], axis=0),
        "adv_model_minus_solver": np.stack(trace["adv_model_minus_solver"], axis=0),
        "loss3_aligned_model_final": np.stack(trace["loss3_aligned_model_final"], axis=0),
        "loss3_aligned_model_minus_solver": np.stack(trace["loss3_aligned_model_minus_solver"], axis=0),
        "loss3_alignment_warp_mag": np.stack(trace["loss3_alignment_warp_mag"], axis=0),
        "loss3_alignment_available": np.asarray(trace["loss3_alignment_available"], dtype=np.bool_),
        "loss3_alignment_source": np.asarray(trace["loss3_alignment_source"], dtype="U128"),
    }
    np.savez_compressed(trace_path, **payload)
    metric_rows = [row for row in sample_rows if int(row.get("sample_position", -1)) == pos and int(row.get("k", -1)) in set(int(v) for v in payload["k"])]
    metrics_path = method_dir / "step_sample_trace_metrics.csv"
    write_csv(metrics_path, metric_rows)
    return {
        "record_step_sample_outputs": True,
        "step_sample_trace_npz": str(trace_path),
        "step_sample_trace_metrics_csv": str(metrics_path),
        "step_sample_position": pos,
        "step_sample_dataset_index": int(dataset_indices[pos]),
        "step_sample_trace_steps": int(len(trace["k"])),
        "record_step_sample_every": int(problem.args.record_step_sample_every),
        "record_step_sample_gradients": bool(problem.args.record_step_sample_gradients),
    }


def run_one(problem: NS2DRecurrentProblem, spec: MethodSpec, loss_type: str, dataset_indices: list[int], method_dir: Path):
    import torch

    delta = initial_delta_for_loss(problem, loss_type, dataset_indices)
    step_offset = int(getattr(problem.args, "initial_step_offset", 0) or 0)
    initial_delta_summary = delta_stats(delta, problem.args.p_order)
    step_rows = []
    sample_rows = []
    trajectory = {"k": [], "delta": [], "x_adv": []}
    step_trace = init_step_sample_trace()
    last_runtime_adv_alignment_fields = None
    start = time.perf_counter()

    for k in range(problem.args.steps + 1):
        record_k = step_offset + k
        delta = delta.detach()
        x_adv = (problem.x0 + delta).detach().requires_grad_(k < problem.args.steps)
        losses, pred_active, target_active, active_aux = problem.active_losses(x_adv, loss_type)
        loss3_metric_diag = metric_diag_to_numpy(active_aux.get("loss3_metric_diagnostics") if isinstance(active_aux, dict) else None)
        active_loss3_alignment_fields = active_aux.get("loss3_metric_alignment_fields") if isinstance(active_aux, dict) else None
        if k >= problem.args.steps and loss_type == "loss3" and problem.mode_spec == "wwwwwwwwww":
            last_runtime_adv_alignment_fields = active_loss3_alignment_fields
        objective = losses[loss_type].sum()
        grad = direction = projected = None
        if k < problem.args.steps:
            objective.backward()
            grad = x_adv.grad.detach()
            direction = method_direction(spec, grad, problem.args.p_order)
            _, projected = propose_delta(spec, delta, direction, problem.args.epsilon, problem.args.alpha, problem.args.p_order)

        true_loss_needed = int(problem.args.true_loss_every) > 0 and (record_k % int(problem.args.true_loss_every) == 0 or k >= problem.args.steps)
        record_every = max(1, int(problem.args.record_step_sample_every))
        record_trace_this_step = bool(problem.args.record_step_sample_outputs) and (record_k % record_every == 0 or k >= problem.args.steps)
        true_pred_t = true_target_t = None
        if true_loss_needed or record_trace_this_step:
            if loss_type == "loss3" and problem.mode_spec == "wwwwwwwwww":
                # For loss3/all_w, active_losses already computed the canonical
                # all-W final model output and target for the whole batch. Reuse
                # them for true-loss CSVs and the one-sample trace instead of
                # paying for a duplicate solver/model rollout every step.
                true_loss_t = losses["loss3"].detach()
                true_pred_t = pred_active.detach()
                true_target_t = target_active.detach()
            else:
                true_loss_t, true_pred_t, true_target_t = problem.all_w_final_outputs(x_adv, source="true_loss_and_step_trace")
        else:
            true_loss_t = torch.full((x_adv.shape[0],), float("nan"), device=x_adv.device, dtype=x_adv.dtype)

        with torch.no_grad():
            delta_l2_t = batch_norm(delta, 2.0)
            delta_linf_t = batch_norm(delta, float("inf"))
            delta_p_t = delta_linf_t if math.isinf(problem.args.p_order) else batch_norm(delta, problem.args.p_order)
            grad_l2_mean = None if grad is None else float(batch_norm(grad, 2.0).mean().detach().cpu().item())
            direction_l2_mean = None if direction is None else float(batch_norm(direction, 2.0).mean().detach().cpu().item())

        losses_np = {name: tensor_to_numpy(value.detach()).astype(np.float64) for name, value in losses.items()}
        if loss_type == "loss3" and problem.mode_spec == "wwwwwwwwww" and true_pred_t is not None and true_target_t is not None:
            wq_l2_t = batch_norm(true_pred_t.detach() - true_target_t.detach(), problem.args.q_order)
        else:
            wq_l2_t = true_loss_t.detach()
        true_loss_np = tensor_to_numpy(true_loss_t.detach()).astype(np.float64)
        wq_l2_np = tensor_to_numpy(wq_l2_t.detach()).astype(np.float64)
        delta_l2_np = tensor_to_numpy(delta_l2_t).astype(np.float64)
        delta_linf_np = tensor_to_numpy(delta_linf_t).astype(np.float64)
        delta_p_np = tensor_to_numpy(delta_p_t).astype(np.float64)
        append_metric_rows(
            step_rows,
            sample_rows,
            loss_type,
            spec.name,
            problem.mode_spec,
            record_k,
            dataset_indices,
            losses_np,
            delta_l2_np,
            delta_linf_np,
            delta_p_np,
            true_loss_np,
            wq_l2_np,
            grad_l2_mean,
            direction_l2_mean,
            time.perf_counter() - start,
            problem.args,
            loss3_metric_diag=loss3_metric_diag,
        )
        if record_trace_this_step:
            if true_pred_t is None or true_target_t is None:
                _, true_pred_t, true_target_t = problem.all_w_final_outputs(x_adv, source="step_trace_only")
            trace_alignment_fields = active_loss3_alignment_fields if loss_type == "loss3" and problem.mode_spec == "wwwwwwwwww" else None
            append_step_sample_trace(problem, step_trace, record_k, delta, x_adv, grad, direction, true_pred_t, true_target_t, trace_alignment_fields)
        if record_k in problem.args.save_steps:
            delta_np = tensor_to_numpy(delta).astype(np.float32)
            trajectory["k"].append(np.asarray(record_k, dtype=np.int64))
            trajectory["delta"].append(delta_np)
            trajectory["x_adv"].append(tensor_to_numpy((problem.x0 + delta).detach()).astype(np.float32))

        if x_adv.grad is not None:
            x_adv.grad = None
        del x_adv, losses, pred_active, target_active, active_aux, objective, true_loss_t, delta_l2_t, delta_linf_t, delta_p_t
        if true_pred_t is not None:
            del true_pred_t
        if true_target_t is not None:
            del true_target_t
        if grad is not None:
            del grad
        if direction is not None:
            del direction

        if k >= problem.args.steps:
            break
        delta = projected.detach()
        del projected

    annotate_growth_metrics(step_rows, sample_rows, loss_type)
    threshold_rows = build_delta_threshold_rows(step_rows, sample_rows, loss_type)

    method_dir.mkdir(parents=True, exist_ok=True)
    write_csv(method_dir / "per_step_metrics.csv", step_rows)
    write_csv(method_dir / "per_sample_step_metrics.csv", sample_rows)
    write_csv(method_dir / "delta_threshold_crossings.csv", threshold_rows)
    final_delta_np = tensor_to_numpy(delta).astype(np.float32)
    final_x_adv_np = tensor_to_numpy((problem.x0 + delta).detach()).astype(np.float32)
    np.savez_compressed(
        method_dir / "final_delta_and_metrics.npz",
        dataset_indices=np.asarray(dataset_indices, dtype=np.int64),
        final_delta=final_delta_np,
        final_x_adv=final_x_adv_np,
    )
    final_state_summary = save_final_state_outputs(problem, delta, dataset_indices, method_dir, last_runtime_adv_alignment_fields)
    step_trace_summary = save_step_sample_trace(problem, step_trace, sample_rows, dataset_indices, method_dir)
    if trajectory["k"]:
        np.savez_compressed(method_dir / "trajectory_samples.npz", k=np.asarray(trajectory["k"], dtype=np.int64), delta=np.stack(trajectory["delta"], axis=0), x_adv=np.stack(trajectory["x_adv"], axis=0))
    summary = {
        "loss_type": loss_type,
        "method": spec.name,
        "mode_spec": problem.mode_spec,
        "loss3_metric": str(problem.args.loss3_metric),
        "dataset_indices": [int(i) for i in dataset_indices],
        "steps": int(problem.args.steps),
        "initial_step_offset": int(step_offset),
        "total_steps_after_run": int(step_offset + problem.args.steps),
        "initial_delta_npz": str(getattr(problem.args, "initial_delta_npz", None)) if getattr(problem.args, "initial_delta_npz", None) is not None else None,
        "initial_delta_key": str(getattr(problem.args, "initial_delta_key", "final_delta")),
        "epsilon": float(problem.args.epsilon),
        "alpha": float(problem.args.alpha),
        "p_order": norm_name(problem.args.p_order),
        "q_order": norm_name(problem.args.q_order),
        "loss1_random_start": bool(problem.args.loss1_random_start),
        "loss1_random_start_fraction": float(problem.args.loss1_random_start_fraction),
        "loss1_random_start_seed": int(problem.args.loss1_random_start_seed),
        **initial_delta_summary,
        "runtime_seconds": float(time.perf_counter() - start),
        **final_state_summary,
        **step_trace_summary,
        "delta_threshold_crossings": threshold_rows,
        "final_step_metrics": step_rows[-1] if step_rows else {},
    }
    write_json(method_dir / "summary.json", summary)
    return summary

def cuda_memory_snapshot(torch_module) -> dict[str, int]:
    if not torch_module.cuda.is_available():
        return {}
    return {
        "allocated_bytes": int(torch_module.cuda.memory_allocated()),
        "reserved_bytes": int(torch_module.cuda.memory_reserved()),
        "max_allocated_bytes": int(torch_module.cuda.max_memory_allocated()),
        "max_reserved_bytes": int(torch_module.cuda.max_memory_reserved()),
    }


def release_cached_memory(args: argparse.Namespace) -> None:
    import gc
    import torch

    gc.collect()
    if torch.cuda.is_available() and args.empty_torch_cache_after_batch:
        torch.cuda.empty_cache()
    if args.clear_jax_caches_after_batch:
        import jax

        clear_caches = getattr(jax, "clear_caches", None)
        if callable(clear_caches):
            clear_caches()


def run(args: argparse.Namespace) -> None:
    import torch

    ensure_cuda_or_die()
    args.device = "cuda"
    args.p_order = parse_norm_order(args.p)
    args.q_order = parse_norm_order(args.q)
    args.mode_spec = resolve_mode_spec(args.mode_spec)
    args.save_steps = sorted(set(int(k) for k in args.save_steps if int(k) >= 0))
    args.parameter_sweep = resolve_parameter_sweep(args)
    args.sweep_is_multi = len(args.parameter_sweep) > 1

    x_all, clean_targets_all, indices = load_initial_conditions(args)
    run_id = time.strftime("%Y%m%d_%H%M%S_UTC", time.gmtime())
    root = args.out_root / f"mode_{args.mode_spec}_p{norm_name(args.p_order)}_q{norm_name(args.q_order)}_{run_id}"
    root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    model = load_recurrent_model(args, device)
    rollout_solver = DifferentiableNSRollout(args.nu, args.fixed_step, args.solver_remat, args.solver_remat_chunk_steps)
    dictionary = None
    if "a" in args.mode_spec:
        dictionary = ApproximateDictionary(args.dictionary_path, device, args.dictionary_chunk_size)

    write_json(
        root / "manifest.json",
        {
            "args": serializable_args(args),
            "gpu": gpu_manifest(),
            "cuda_memory_after_shared_load": cuda_memory_snapshot(torch),
        },
    )

    all_summaries = []
    batch_memory_rows = []
    for batch_start in range(0, len(indices), args.attack_batch_size):
        batch_end = min(batch_start + args.attack_batch_size, len(indices))
        batch_indices = indices[batch_start:batch_end]
        x_batch = x_all[batch_start:batch_end]
        clean_batch = None if clean_targets_all is None else clean_targets_all[batch_start:batch_end]
        problem = NS2DRecurrentProblem(args, x_batch, clean_target=clean_batch, model=model, rollout_solver=rollout_solver, dictionary=dictionary)
        batch_dir = root / f"batch_{batch_start:04d}_{batch_end - 1:04d}"
        for sweep in args.parameter_sweep:
            args.epsilon = float(sweep["epsilon"])
            args.alpha = float(sweep["alpha"])
            sweep_dir = batch_dir / str(sweep["tag"]) if args.sweep_is_multi else batch_dir
            for loss_type in args.loss_types:
                for method in args.methods:
                    method_dir = sweep_dir / loss_type / method
                    all_summaries.append(run_one(problem, METHOD_SPECS[method], loss_type, batch_indices, method_dir))
                    if args.empty_torch_cache_after_method:
                        torch.cuda.empty_cache()
        del problem
        batch_memory_rows.append(
            {
                "batch_start": int(batch_start),
                "batch_end_exclusive": int(batch_end),
                "parameter_sweep_count": int(len(args.parameter_sweep)),
                **cuda_memory_snapshot(torch),
            }
        )
        release_cached_memory(args)
    write_csv(root / "batch_memory.csv", batch_memory_rows)
    write_csv(root / "solver_rollout_trace.csv", rollout_solver.trace)
    write_json(root / "summary.json", {"run_root": str(root), "summaries": all_summaries, "solver_rollout_calls": len(rollout_solver.trace), "final_cuda_memory": cuda_memory_snapshot(torch)})
    print(f"[done] saved attack outputs under {root}", flush=True)

def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    out = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-path", type=Path, default=DEFAULT_TEST_PATH)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dictionary-path", type=Path, default=DEFAULT_DICTIONARY_PATH)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--indices", default="", help="Comma-separated dataset indices. Overrides --start-index/--num-samples.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--attack-batch-size", type=int, default=1)
    parser.add_argument("--loss-types", nargs="+", default=list(LOSS_TYPES), choices=LOSS_TYPES)
    parser.add_argument("--methods", nargs="+", default=list(CORE4_METHODS), choices=CORE4_METHODS)
    parser.add_argument("--loss3-metric", choices=LOSS3_METRIC_CHOICES, default="qnorm", help="Metric used for loss3 final-state discrepancy. qnorm preserves the original behavior.")
    parser.add_argument("--loss3-image-normalization", choices=["pair_minmax_detached", "none"], default="pair_minmax_detached", help="Normalization for image-style loss3 metrics before DISTS/MS-SSIM/scattering.")
    parser.add_argument("--loss3-metric-eps", type=float, default=1e-6)
    parser.add_argument("--loss3-scattering-j", type=int, default=3)
    parser.add_argument("--loss3-align-objective", choices=["l2", "dists"], default="l2", help="Detached inner-registration objective for affine/local-warp metrics; final content metric remains DISTS.")
    parser.add_argument("--loss3-affine-inner-steps", type=int, default=8)
    parser.add_argument("--loss3-affine-lr", type=float, default=0.05)
    parser.add_argument("--loss3-affine-max-shift-ratio", type=float, default=0.05)
    parser.add_argument("--loss3-affine-max-angle-deg", type=float, default=10.0)
    parser.add_argument("--loss3-affine-max-log-scale", type=float, default=math.log(1.1))
    parser.add_argument("--loss3-affine-reg-weight", type=float, default=0.01)
    parser.add_argument("--loss3-local-grid-size", type=int, default=8)
    parser.add_argument("--loss3-local-inner-steps", type=int, default=8)
    parser.add_argument("--loss3-local-lr", type=float, default=0.05)
    parser.add_argument("--loss3-local-max-disp-ratio", type=float, default=0.03)
    parser.add_argument("--loss3-local-mag-weight", type=float, default=0.01)
    parser.add_argument("--loss3-local-smooth-weight", type=float, default=0.05)
    parser.add_argument("--loss3-homography-inner-steps", type=int, default=8)
    parser.add_argument("--loss3-homography-lr", type=float, default=0.05)
    parser.add_argument("--loss3-homography-max-corner-ratio", type=float, default=0.05)
    parser.add_argument("--loss3-homography-reg-weight", type=float, default=0.01)
    parser.add_argument("--loss3-tps-grid-size", type=int, default=4)
    parser.add_argument("--loss3-tps-inner-steps", type=int, default=8)
    parser.add_argument("--loss3-tps-lr", type=float, default=0.05)
    parser.add_argument("--loss3-tps-max-disp-ratio", type=float, default=0.05)
    parser.add_argument("--loss3-tps-offset-weight", type=float, default=0.01)
    parser.add_argument("--loss3-tps-smooth-weight", type=float, default=0.05)
    parser.add_argument("--loss3-elastic-grid-size", type=int, default=16)
    parser.add_argument("--loss3-elastic-inner-steps", type=int, default=8)
    parser.add_argument("--loss3-elastic-lr", type=float, default=0.05)
    parser.add_argument("--loss3-elastic-max-disp-ratio", type=float, default=0.05)
    parser.add_argument("--loss3-elastic-smooth-kernel", type=int, default=9)
    parser.add_argument("--loss3-elastic-smooth-passes", type=int, default=2)
    parser.add_argument("--loss3-elastic-mag-weight", type=float, default=0.01)
    parser.add_argument("--loss3-elastic-smooth-weight", type=float, default=0.03)
    parser.add_argument("--loss3-svf-grid-size", type=int, default=8)
    parser.add_argument("--loss3-svf-inner-steps", type=int, default=8)
    parser.add_argument("--loss3-svf-lr", type=float, default=0.05)
    parser.add_argument("--loss3-svf-max-vel-ratio", type=float, default=0.04)
    parser.add_argument("--loss3-svf-int-steps", type=int, default=5)
    parser.add_argument("--loss3-svf-mag-weight", type=float, default=0.01)
    parser.add_argument("--loss3-svf-smooth-weight", type=float, default=0.05)
    parser.add_argument("--mode-spec", default="all_w", help="10 chars for frames 1..9 plus target, or preset name.")
    parser.add_argument("--require-target-w", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clean-target-source", choices=["dataset", "solver"], default="dataset")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--epsilon", type=float, default=8.0, help="Single epsilon value; ignored when --epsilons or --epsilon-alpha-pairs is supplied.")
    parser.add_argument("--alpha", type=float, default=0.3, help="Single alpha value; ignored when --alphas or --epsilon-alpha-pairs is supplied.")
    parser.add_argument("--epsilons", nargs="+", type=float, default=None, help="Run the Cartesian product of these epsilon values and --alphas/--alpha.")
    parser.add_argument("--alphas", nargs="+", type=float, default=None, help="Run the Cartesian product of these alpha values and --epsilons/--epsilon.")
    parser.add_argument("--epsilon-alpha-pairs", nargs="+", default=None, help="Run explicit epsilon:alpha pairs, e.g. 16:0.5 32:1 64:2. Overrides --epsilon/--alpha/--epsilons/--alphas.")
    parser.add_argument("--loss1-random-start", action=argparse.BooleanOptionalAction, default=True, help="Use a small random nonzero initial delta for loss1 so it does not stall at the zero residual.")
    parser.add_argument("--loss1-random-start-fraction", type=float, default=1e-3, help="Initial loss1 random-start radius as a fraction of epsilon.")
    parser.add_argument("--loss1-random-start-seed", type=int, default=12345)
    parser.add_argument("--p", default="2")
    parser.add_argument("--q", default="2")
    parser.add_argument("--save-steps", nargs="*", type=int, default=[], help="Optional trajectory checkpoints to save. Final delta is always saved separately; default saves no per-step trajectory arrays.")
    parser.add_argument("--initial-delta-npz", type=Path, default=None, help="Optional final_delta_and_metrics.npz to initialize delta for continuation runs.")
    parser.add_argument("--initial-delta-key", default="final_delta", help="Array key to load from --initial-delta-npz.")
    parser.add_argument("--initial-step-offset", type=int, default=0, help="Global step offset used when continuing from an earlier run, e.g. 25 for a 25+75 run.")
    parser.add_argument("--true-loss-every", type=int, default=1, help="Evaluate the full all-W solver true loss every N attack steps; 1 records the full true-loss curve.")
    parser.add_argument("--record-final-state-outputs", action=argparse.BooleanOptionalAction, default=True, help="Save final clean/perturbed initial states plus all-W FNO and solver final outputs for every sample in the attack batch.")
    parser.add_argument("--record-step-sample-outputs", action=argparse.BooleanOptionalAction, default=True, help="Save per-step final delta, x_adv, gradients, and all-W FNO/solver final outputs for one sample position in each attack batch.")
    parser.add_argument("--record-step-sample-position", type=int, default=0, help="Batch sample position to archive through the attack trajectory, usually 0 for the first sample in each batch.")
    parser.add_argument("--record-step-sample-every", type=int, default=1, help="Record the selected sample every N attack steps; the final step is always recorded when step-sample recording is enabled.")
    parser.add_argument("--record-step-sample-gradients", action=argparse.BooleanOptionalAction, default=True, help="Include per-step gradient and update-direction arrays in step_sample_trace.npz.")
    parser.add_argument("--modes1", type=int, default=64)
    parser.add_argument("--modes2", type=int, default=64)
    parser.add_argument("--width", type=int, default=60)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--t-in", type=int, default=10)
    parser.add_argument("--t-out", type=int, default=10)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--target-frame-index", type=int, default=19, help="Zero-based frame index; 19 is the twentieth frame.")
    parser.add_argument("--nu", type=float, default=1e-5)
    parser.add_argument("--fixed-step", type=float, default=0.005)
    parser.add_argument("--solver-remat", choices=["micro", "none", "chunk", "second"], default="micro", help="Solver activation rematerialization strategy; micro matches the original implementation.")
    parser.add_argument("--solver-remat-chunk-steps", type=int, default=20, help="Micro-steps per rematerialized chunk when --solver-remat=chunk; must divide steps_per_second.")
    parser.add_argument("--dictionary-chunk-size", type=int, default=32)
    parser.add_argument("--empty-torch-cache-after-batch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--empty-torch-cache-after-method", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--clear-jax-caches-after-batch", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--non-strict-checkpoint", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
