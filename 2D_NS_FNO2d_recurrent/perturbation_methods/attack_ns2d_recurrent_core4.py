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
if str(NS_ROOT) not in sys.path:
    sys.path.insert(0, str(NS_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.FNO2d import FNO2d, RecurrentPredictor


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


class NS2DRecurrentProblem:
    def __init__(self, args: argparse.Namespace, x0, clean_target=None, model=None, rollout_solver=None, dictionary=None):
        import torch

        self.args = args
        self.device = torch.device(args.device)
        self.x0 = x0.to(self.device, dtype=torch.float32).detach()
        self.clean_target_from_data = None if clean_target is None else clean_target.to(self.device, dtype=torch.float32).detach()
        self.mode_spec = resolve_mode_spec(args.mode_spec)
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
        residuals = {"loss1": pred - self.f0, "loss2": pred - self.g0, "loss3": pred - g_delta}
        losses = {name: batch_norm(residual, self.args.q_order) for name, residual in residuals.items()}
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
            losses["loss3"] = batch_norm(pred - g_delta, self.args.q_order)
        return losses, pred, g_delta, aux

    def true_loss_all_w(self, x_adv):
        """Evaluate the full-solver true loss curve without contributing gradients."""

        import torch

        with torch.no_grad():
            x_eval = x_adv.detach()
            seq = self.rollout_solver.rollout(
                x_eval,
                self.args.target_frame_index,
                context={
                    "source": "true_loss_all_w",
                    "mode_spec": "wwwwwwwwww",
                    "need_target": True,
                    "required_frames": list(range(1, self.args.t_in)) + [self.args.target_frame_index],
                },
            )
            frames = [x_eval] + [seq[frame_index] for frame_index in range(1, self.args.t_in)]
            pred = self.model(torch_stack_last(frames))[..., -1]
            target = seq[self.args.target_frame_index]
            return batch_norm(pred - target, self.args.q_order)


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
    grad_l2_mean,
    direction_l2_mean,
    elapsed,
    args,
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
    row["boundary_ratio_mean"] = float(np.nanmean(delta_p / float(args.epsilon)))
    if grad_l2_mean is not None:
        row["grad_l2_mean"] = float(grad_l2_mean)
    if direction_l2_mean is not None:
        row["direction_l2_mean"] = float(direction_l2_mean)
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
        }
        for name in LOSS_TYPES:
            sample[name] = float(losses_np[name][pos])
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


def run_one(problem: NS2DRecurrentProblem, spec: MethodSpec, loss_type: str, dataset_indices: list[int], method_dir: Path):
    import torch

    delta = torch.zeros_like(problem.x0)
    step_rows = []
    sample_rows = []
    trajectory = {"k": [], "delta": [], "x_adv": []}
    start = time.perf_counter()

    for k in range(problem.args.steps + 1):
        delta = delta.detach()
        x_adv = (problem.x0 + delta).detach().requires_grad_(k < problem.args.steps)
        losses, _, _, _ = problem.active_losses(x_adv, loss_type)
        objective = losses[loss_type].sum()
        grad = direction = projected = None
        if k < problem.args.steps:
            objective.backward()
            grad = x_adv.grad.detach()
            direction = method_direction(spec, grad, problem.args.p_order)
            _, projected = propose_delta(spec, delta, direction, problem.args.epsilon, problem.args.alpha, problem.args.p_order)

        if int(problem.args.true_loss_every) > 0 and (k % int(problem.args.true_loss_every) == 0 or k >= problem.args.steps):
            if loss_type == "loss3" and problem.mode_spec == "wwwwwwwwww":
                true_loss_t = losses["loss3"].detach()
            else:
                true_loss_t = problem.true_loss_all_w(x_adv)
        else:
            true_loss_t = torch.full((x_adv.shape[0],), float("nan"), device=x_adv.device, dtype=x_adv.dtype)

        with torch.no_grad():
            delta_l2_t = batch_norm(delta, 2.0)
            delta_linf_t = batch_norm(delta, float("inf"))
            delta_p_t = delta_linf_t if math.isinf(problem.args.p_order) else batch_norm(delta, problem.args.p_order)
            grad_l2_mean = None if grad is None else float(batch_norm(grad, 2.0).mean().detach().cpu().item())
            direction_l2_mean = None if direction is None else float(batch_norm(direction, 2.0).mean().detach().cpu().item())

        losses_np = {name: tensor_to_numpy(value.detach()).astype(np.float64) for name, value in losses.items()}
        true_loss_np = tensor_to_numpy(true_loss_t.detach()).astype(np.float64)
        delta_l2_np = tensor_to_numpy(delta_l2_t).astype(np.float64)
        delta_linf_np = tensor_to_numpy(delta_linf_t).astype(np.float64)
        delta_p_np = tensor_to_numpy(delta_p_t).astype(np.float64)
        append_metric_rows(
            step_rows,
            sample_rows,
            loss_type,
            spec.name,
            problem.mode_spec,
            k,
            dataset_indices,
            losses_np,
            delta_l2_np,
            delta_linf_np,
            delta_p_np,
            true_loss_np,
            grad_l2_mean,
            direction_l2_mean,
            time.perf_counter() - start,
            problem.args,
        )
        if k in problem.args.save_steps:
            delta_np = tensor_to_numpy(delta).astype(np.float32)
            trajectory["k"].append(np.asarray(k, dtype=np.int64))
            trajectory["delta"].append(delta_np)
            trajectory["x_adv"].append(tensor_to_numpy((problem.x0 + delta).detach()).astype(np.float32))

        if x_adv.grad is not None:
            x_adv.grad = None
        del x_adv, losses, objective, true_loss_t, delta_l2_t, delta_linf_t, delta_p_t
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
    if trajectory["k"]:
        np.savez_compressed(method_dir / "trajectory_samples.npz", k=np.asarray(trajectory["k"], dtype=np.int64), delta=np.stack(trajectory["delta"], axis=0), x_adv=np.stack(trajectory["x_adv"], axis=0))
    summary = {
        "loss_type": loss_type,
        "method": spec.name,
        "mode_spec": problem.mode_spec,
        "dataset_indices": [int(i) for i in dataset_indices],
        "steps": int(problem.args.steps),
        "epsilon": float(problem.args.epsilon),
        "alpha": float(problem.args.alpha),
        "p_order": norm_name(problem.args.p_order),
        "q_order": norm_name(problem.args.q_order),
        "runtime_seconds": float(time.perf_counter() - start),
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
        for loss_type in args.loss_types:
            for method in args.methods:
                method_dir = root / f"batch_{batch_start:04d}_{batch_end - 1:04d}" / loss_type / method
                all_summaries.append(run_one(problem, METHOD_SPECS[method], loss_type, batch_indices, method_dir))
                if args.empty_torch_cache_after_method:
                    torch.cuda.empty_cache()
        del problem
        batch_memory_rows.append(
            {
                "batch_start": int(batch_start),
                "batch_end_exclusive": int(batch_end),
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
    parser.add_argument("--mode-spec", default="all_w", help="10 chars for frames 1..9 plus target, or preset name.")
    parser.add_argument("--require-target-w", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clean-target-source", choices=["dataset", "solver"], default="dataset")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--epsilon", type=float, default=8.0)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--p", default="2")
    parser.add_argument("--q", default="2")
    parser.add_argument("--save-steps", nargs="*", type=int, default=[], help="Optional trajectory checkpoints to save. Final delta is always saved separately; default saves no per-step trajectory arrays.")
    parser.add_argument("--true-loss-every", type=int, default=1, help="Evaluate the full all-W solver true loss every N attack steps; 1 records the full true-loss curve.")
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
