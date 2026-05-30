#!/usr/bin/env python3
"""Online adversarial training for Burgers, Darcy flow, and 2D Navier-Stokes.

The script starts from the retained trained FNO checkpoints, attacks each
incoming training batch with a short fast attack, and trains on the attacked
batch. It also runs periodic clean evaluation on train/test/generated datasets
using the same dataset specs as ``evaluate_generalization_models.py``.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.evaluate_generalization_models import (  # noqa: E402
    DatasetSpec,
    build_specs,
    evaluate_dataset,
    load_module,
    load_burgers_model,
    load_darcy_model,
    load_ns2d_model,
    tensor_xy,
    torch_load,
)


TASK_ORDER = ("burgers", "darcy", "ns2d")


@dataclass(frozen=True)
class TaskDefaults:
    batch_size: int
    optimizer_batch_size: int
    eval_batch_size: int
    epochs: int
    train_max_samples: int | None
    attack_method: str
    attack_steps: int
    epsilon_fraction: float
    epsilon_abs: float
    alpha_ratio: float
    learning_rate: float
    weight_decay: float
    random_start_fraction: float
    eps_jitter_low: float
    eps_jitter_high: float


DEFAULTS: dict[str, TaskDefaults] = {
    "burgers": TaskDefaults(
        # Exact solver-gradient attack is memory bound; batch 256 is tested safe on the 31.7GB GPU.
        batch_size=256,
        optimizer_batch_size=32,
        eval_batch_size=512,
        epochs=500,
        train_max_samples=None,
        attack_method="fast_replace_linf",
        attack_steps=3,
        epsilon_fraction=0.06,
        epsilon_abs=0.0,
        alpha_ratio=1.0,
        learning_rate=2e-4,
        weight_decay=1e-5,
        random_start_fraction=0.35,
        eps_jitter_low=0.75,
        eps_jitter_high=1.25,
    ),
    "darcy": TaskDefaults(
        # Exact solver-gradient attack is memory bound; batch 256 is tested safe on the 31.7GB GPU.
        batch_size=256,
        optimizer_batch_size=32,
        eval_batch_size=256,
        epochs=500,
        train_max_samples=None,
        attack_method="binary_steepest_replace",
        attack_steps=1,
        epsilon_fraction=0.025,
        epsilon_abs=0.0,
        alpha_ratio=1.0,
        learning_rate=1e-4,
        weight_decay=1e-5,
        random_start_fraction=0.0,
        eps_jitter_low=0.75,
        eps_jitter_high=1.35,
    ),
    "ns2d": TaskDefaults(
        # Exact NS2D solver-gradient rollout is very memory heavy; batch 2 OOMed on the 31.7GB GPU.
        batch_size=1,
        optimizer_batch_size=1,
        eval_batch_size=5,
        epochs=500,
        train_max_samples=None,
        attack_method="fast_add_linf",
        attack_steps=5,
        epsilon_fraction=0.035,
        epsilon_abs=0.0,
        alpha_ratio=0.45,
        learning_rate=5e-5,
        weight_decay=1e-5,
        random_start_fraction=0.20,
        eps_jitter_low=0.75,
        eps_jitter_high=1.25,
    ),
}


SMOKE_LIMITS = {
    "burgers": {"train_max_samples": 8, "max_batches": 1, "eval_max_samples": 4},
    "darcy": {"train_max_samples": 4, "max_batches": 1, "eval_max_samples": 3},
    "ns2d": {"train_max_samples": 2, "max_batches": 1, "eval_max_samples": 1},
}


def now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_UTC")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_csv_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    old_keys: list[str] = []
    if exists:
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            old_keys = next(reader, [])
    keys = list(old_keys)
    for key in row:
        if key not in keys:
            keys.append(key)
    if exists and keys != old_keys:
        rows: list[dict[str, Any]] = []
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows.extend(reader)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def memory_stats(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {"cuda_allocated_mb": 0.0, "cuda_reserved_mb": 0.0, "cuda_peak_allocated_mb": 0.0}
    return {
        "cuda_allocated_mb": torch.cuda.memory_allocated(device) / (1024 * 1024),
        "cuda_reserved_mb": torch.cuda.memory_reserved(device) / (1024 * 1024),
        "cuda_peak_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024 * 1024),
    }


def safe_grad_norm(model) -> float:
    total = 0.0
    for param in model.parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach()
        mag = grad.abs() if torch.is_complex(grad) else grad
        finite = torch.isfinite(mag)
        if not finite.any():
            continue
        total += float(mag[finite].double().pow(2).sum().cpu())
    return math.sqrt(total) if math.isfinite(total) else float("inf")


def load_model(task: str, device: torch.device):
    if task == "burgers":
        return load_burgers_model(device)
    if task == "darcy":
        return load_darcy_model(device)
    if task == "ns2d":
        return load_ns2d_model(device)
    raise ValueError(task)


def baseline_checkpoint_manifest() -> dict[str, dict[str, str]]:
    return {
        "burgers": {
            "path": str(PROJECT_ROOT / "1D_Burgers/trained_models/attack_ready/burgers_nu0.001_fno1d_500/checkpoints/pytorch_fno1d_500.pt"),
            "checkpoint_role": "retained_trained_baseline",
            "note": "No separate Burgers best.pt exists in this workspace; this is the retained trained baseline loaded by evaluate_generalization_models.py.",
        },
        "darcy": {
            "path": str(PROJECT_ROOT / "2D_Darcy_FNO2d/saved_models/2D/darcy_N1500_nx85_m64_w60_e500_20260528/best.pt"),
            "checkpoint_role": "best",
            "note": "Darcy best checkpoint.",
        },
        "ns2d": {
            "path": str(PROJECT_ROOT / "2D_NS_FNO2d_recurrent/saved_models/2D/modes64_modes64_width60_epochs500_Tin10_T10_recurrent_pytorch_20260521_090136_UTC/checkpoints/best.pt"),
            "checkpoint_role": "best",
            "note": "NS2D recurrent best checkpoint.",
        },
    }


def normalize_max_samples(value: Any) -> int | None:
    if value is None:
        return None
    value = int(value)
    return None if value <= 0 else value


def load_train_xy(spec: DatasetSpec, task: str, max_samples: int | None) -> tuple[torch.Tensor, torch.Tensor]:
    data = torch_load(spec.path)
    x, y = tensor_xy(data, task)
    max_samples = normalize_max_samples(max_samples)
    if max_samples is not None:
        x = x[:max_samples].contiguous()
        y = y[:max_samples].contiguous()
    else:
        x = x.contiguous()
        y = y.contiguous()
    return x, y


def train_sample_count(spec: DatasetSpec, task: str) -> int:
    data = torch_load(spec.path)
    x, _ = tensor_xy(data, task)
    return int(x.shape[0])


def dataset_range_summary(x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    xf = x.float().reshape(x.shape[0], -1)
    yf = y.float().reshape(y.shape[0], -1)
    xr = xf.max(dim=1).values - xf.min(dim=1).values
    yr = yf.max(dim=1).values - yf.min(dim=1).values
    return {
        "x_min": float(xf.min()),
        "x_max": float(xf.max()),
        "x_mean": float(xf.mean()),
        "x_std": float(xf.std(unbiased=False)),
        "x_median_sample_range": float(xr.median()),
        "x_mean_sample_range": float(xr.mean()),
        "y_min": float(yf.min()),
        "y_max": float(yf.max()),
        "y_mean": float(yf.mean()),
        "y_std": float(yf.std(unbiased=False)),
        "y_median_sample_range": float(yr.median()),
        "y_mean_sample_range": float(yr.mean()),
    }


def per_sample_range(x: torch.Tensor) -> torch.Tensor:
    flat = x.detach().reshape(x.shape[0], -1)
    return (flat.max(dim=1).values - flat.min(dim=1).values).clamp_min(1e-6)


def expand_per_sample(values: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    shape = [values.shape[0]] + [1] * (like.ndim - 1)
    return values.reshape(shape)


def compute_batch_eps(
    x: torch.Tensor,
    epsilon_fraction: float,
    epsilon_abs: float,
    jitter_low: float,
    jitter_high: float,
) -> torch.Tensor:
    if epsilon_abs and epsilon_abs > 0:
        base = torch.full((x.shape[0],), float(epsilon_abs), device=x.device, dtype=x.dtype)
    else:
        base = per_sample_range(x).to(device=x.device, dtype=x.dtype) * float(epsilon_fraction)
    jitter = torch.empty_like(base).uniform_(float(jitter_low), float(jitter_high))
    return (base * jitter).clamp_min(1e-8)


def finite_mse(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    finite = torch.isfinite(pred) & torch.isfinite(y)
    if finite.any():
        return F.mse_loss(pred[finite], y[finite])
    return torch.nan_to_num(pred).pow(2).mean() * 0.0


@dataclass
class AttackBatchResult:
    x_train: torch.Tensor
    y_train: torch.Tensor
    info: dict[str, Any]


_MODULE_CACHE: dict[str, Any] = {}
_BURGERS_SOLVER_CACHE: dict[tuple[Any, ...], tuple[Any, int]] = {}
_NS_SOLVER_CACHE: dict[tuple[Any, ...], Any] = {}
_DARCY_SOLVER_CACHE: dict[tuple[Any, ...], Any] = {}


def load_registered_module(name: str, path: Path):
    if name in _MODULE_CACHE:
        return _MODULE_CACHE[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    _MODULE_CACHE[name] = module
    return module


def torch_pde_solvers():
    return load_registered_module("advtrain_torch_pde_solvers", PROJECT_ROOT / "solvers.py")


def darcy_bridge_module():
    # The Darcy code has its own package named `solvers`.  If the project-root
    # solvers.py was imported as `solvers` elsewhere, remove that alias before
    # loading the Darcy bridge so `from solvers.darcy_jax_solver` resolves to
    # 2D_Darcy_FNO2d/solvers.
    existing = sys.modules.get("solvers")
    if existing is not None and not hasattr(existing, "__path__"):
        sys.modules.pop("solvers", None)
    return load_registered_module(
        "advtrain_darcy_binary_bridge",
        PROJECT_ROOT / "2D_Darcy_FNO2d" / "perturbation_methods" / "attack_darcy_binary.py",
    )


def get_burgers_solver(u0: torch.Tensor) -> tuple[Any, int]:
    mod = torch_pde_solvers()
    key = ("burgers", int(u0.shape[-1]), str(u0.device), str(u0.dtype), 0.001, 1e-3)
    if key not in _BURGERS_SOLVER_CACHE:
        solver = mod.Burgers1DETDRK4(
            num_points=int(u0.shape[-1]),
            domain_extent=2.0,
            dt=0.001,
            diffusivity=1e-3,
            convection_scale=1.0,
            conservative=False,
            dealiasing_fraction=2.0 / 3.0,
            device=u0.device,
            dtype=u0.dtype,
        )
        _BURGERS_SOLVER_CACHE[key] = (solver, int(round(1.0 / 0.001)))
    return _BURGERS_SOLVER_CACHE[key]


def get_ns_solver(x0: torch.Tensor):
    mod = torch_pde_solvers()
    key = ("ns2d", int(x0.shape[-1]), str(x0.device), str(x0.dtype), 0.005, 1e-5)
    if key not in _NS_SOLVER_CACHE:
        _NS_SOLVER_CACHE[key] = mod.NavierStokesVorticity2DZongyiETDRK4(
            num_points=int(x0.shape[-1]),
            domain_extent=1.0,
            dt=0.005,
            diffusivity=1e-5,
            drag=0.0,
            vorticity_convection_scale=1.0,
            injection_scale=1.0,
            dealiasing_fraction=2.0 / 3.0,
            num_circle_points=16,
            device=x0.device,
            dtype=x0.dtype,
        )
    return _NS_SOLVER_CACHE[key]


def get_darcy_solver():
    key = ("darcy", 1e-5, 0.0, None)
    if key not in _DARCY_SOLVER_CACHE:
        bridge = darcy_bridge_module()
        _DARCY_SOLVER_CACHE[key] = bridge.make_darcy_solve_fn(tol=1e-5, atol=0.0, maxiter=None)
    return _DARCY_SOLVER_CACHE[key]


def solver_target_grad_enabled(cfg: dict[str, Any]) -> bool:
    return True


def tensor_stats(prefix: str, x: torch.Tensor) -> dict[str, float]:
    flat = x.detach()
    total = max(1, flat.numel())
    finite = torch.isfinite(flat)
    valid = flat[finite]
    if valid.numel() == 0:
        return {
            f"{prefix}_min": float("nan"),
            f"{prefix}_max": float("nan"),
            f"{prefix}_absmax": float("nan"),
            f"{prefix}_invalid_fraction": 1.0,
        }
    return {
        f"{prefix}_min": float(valid.min().cpu()),
        f"{prefix}_max": float(valid.max().cpu()),
        f"{prefix}_absmax": float(valid.abs().max().cpu()),
        f"{prefix}_invalid_fraction": float((total - int(finite.sum().cpu())) / total),
    }


def burgers_solver_target(x_model: torch.Tensor, cfg: dict[str, Any], allow_target_grad: bool) -> torch.Tensor:
    if x_model.ndim != 3 or x_model.shape[-1] != 1:
        raise ValueError(f"Burgers model input must be (B,N,1), got {tuple(x_model.shape)}")
    def _solve():
        solver, steps = get_burgers_solver(x_model[..., 0])
        y = torch_pde_solvers().solve_burgers_final_batch_with_solver(
            x_model[..., 0],
            solver,
            steps,
            remat_mode=str(cfg.get("burgers_solver_remat", "none")),
            remat_chunk_steps=int(cfg.get("burgers_solver_remat_chunk_steps", 20)),
        )
        return y.unsqueeze(-1)
    if allow_target_grad:
        return _solve()
    with torch.no_grad():
        return _solve()


def darcy_solver_target(x_model: torch.Tensor, allow_target_grad: bool) -> torch.Tensor:
    if x_model.device.type != "cuda":
        raise RuntimeError("Darcy solver-label mode requires CUDA because the existing JAX bridge uses CUDA DLPack.")
    if x_model.ndim != 4 or x_model.shape[-1] != 1:
        raise ValueError(f"Darcy model input must be (B,H,W,1), got {tuple(x_model.shape)}")
    bridge = darcy_bridge_module()
    solver = get_darcy_solver()
    def _solve():
        y = bridge.JaxDarcySolver.apply(x_model[..., 0].contiguous(), solver)
        return y.unsqueeze(-1)
    if allow_target_grad:
        return _solve()
    with torch.no_grad():
        return _solve()


def ns2d_solver_pair_from_initial(
    x0: torch.Tensor,
    cfg: dict[str, Any],
    *,
    for_attack: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if x0.ndim != 3:
        raise ValueError(f"NS2D attack initial state must be (B,H,W), got {tuple(x0.shape)}")
    mod = torch_pde_solvers()
    solver = get_ns_solver(x0)
    def _solve():
        seq = mod.solve_ns_zongyi_rollout_batch_with_solver(
            x0,
            solver,
            t_final=20,
            fixed_step=0.005,
            apply_exponax_orientation_transform=True,
            return_time_last=True,
            remat_mode=str(cfg.get("ns2d_solver_remat", "none")),
            remat_chunk_steps=int(cfg.get("ns2d_solver_remat_chunk_steps", 20)),
        )
        x_seq = seq[..., :10]
        y_seq = seq[..., 10:20]
        if not solver_target_grad_enabled(cfg):
            y_seq = y_seq.detach()
        return x_seq, y_seq
    if for_attack:
        return _solve()
    with torch.no_grad():
        x_seq, y_seq = _solve()
    return x_seq.detach(), y_seq.detach()


def solver_target_for_model_input(
    task: str,
    x_model: torch.Tensor,
    y_clean: torch.Tensor,
    cfg: dict[str, Any],
    *,
    allow_target_grad: bool,
) -> torch.Tensor:
    label_mode = str(cfg.get("label_mode", "solver"))
    if label_mode == "clean":
        return y_clean
    if label_mode != "solver":
        raise ValueError(f"unknown label_mode={label_mode!r}")
    if task == "burgers":
        return burgers_solver_target(x_model, cfg, allow_target_grad=allow_target_grad)
    if task == "darcy":
        return darcy_solver_target(x_model, allow_target_grad=allow_target_grad)
    raise ValueError(f"solver_target_for_model_input does not handle {task}; NS2D uses initial-state rollouts")


def random_start_delta(x: torch.Tensor, eps: torch.Tensor, fraction: float) -> torch.Tensor:
    if fraction <= 0:
        return torch.zeros_like(x)
    eps_view = expand_per_sample(eps * float(fraction), x)
    return torch.empty_like(x).uniform_(-1.0, 1.0) * eps_view


def ns_attack_mask(x: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "all":
        return torch.ones_like(x)
    mask = torch.zeros_like(x)
    if x.ndim >= 4:
        mask[..., 0] = 1.0
    else:
        mask[...] = 1.0
    return mask


def continuous_attack(
    model,
    xb: torch.Tensor,
    yb: torch.Tensor,
    *,
    task: str,
    method: str,
    steps: int,
    epsilon_fraction: float,
    epsilon_abs: float,
    alpha_ratio: float,
    random_start_fraction: float,
    jitter_low: float,
    jitter_high: float,
    ns_frames: str,
    cfg: dict[str, Any],
) -> AttackBatchResult:
    was_training = model.training
    model.eval()
    original_requires_grad = [param.requires_grad for param in model.parameters()]
    for param in model.parameters():
        param.requires_grad_(False)

    eps = compute_batch_eps(xb, epsilon_fraction, epsilon_abs, jitter_low, jitter_high)
    eps_view = expand_per_sample(eps, xb)
    alpha_view = expand_per_sample(eps * float(alpha_ratio), xb)
    mask = ns_attack_mask(xb, ns_frames) if task == "ns2d" else torch.ones_like(xb)
    delta = random_start_delta(xb, eps, random_start_fraction) * mask
    delta = torch.clamp(delta, -eps_view, eps_view)
    x_adv = (xb + delta).detach()

    clean_loss_value = float("nan")
    first_adv_loss = float("nan")
    final_loss_value = float("nan")
    grad_abs_mean = float("nan")
    used_steps = max(1, int(steps))

    try:
        for step in range(used_steps):
            x_adv = x_adv.detach().requires_grad_(True)
            target = solver_target_for_model_input(
                task,
                x_adv,
                yb,
                cfg,
                allow_target_grad=solver_target_grad_enabled(cfg),
            )
            pred = model(x_adv)
            loss = finite_mse(pred, target)
            if step == 0:
                with torch.no_grad():
                    clean_target = solver_target_for_model_input(task, xb, yb, cfg, allow_target_grad=False)
                    clean_loss_value = float(finite_mse(model(xb), clean_target).detach().cpu())
                    first_adv_loss = float(loss.detach().cpu())
            grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]
            grad = torch.nan_to_num(grad) * mask
            grad_abs_mean = float(grad.abs().mean().detach().cpu())
            if method.endswith("replace_linf"):
                delta = eps_view * grad.sign()
            elif method.endswith("add_linf"):
                delta = (x_adv.detach() - xb) + alpha_view * grad.sign()
                delta = torch.clamp(delta, -eps_view, eps_view)
            else:
                raise ValueError(f"unknown continuous attack method: {method}")
            delta = delta * mask
            x_adv = (xb + delta).detach()

        with torch.no_grad():
            y_train = solver_target_for_model_input(task, x_adv, yb, cfg, allow_target_grad=False).detach()
            final_loss_value = float(finite_mse(model(x_adv), y_train).detach().cpu())
            delta = x_adv - xb
            active = mask > 0
            denom = eps_view.expand_as(delta).clamp_min(1e-12)
            boundary_ratio = float((delta.detach().abs()[active] / denom[active]).mean().cpu()) if active.any() else 0.0
            linf = float(delta.detach().abs().reshape(delta.shape[0], -1).max(dim=1).values.mean().cpu())
            l2 = float(torch.sqrt(delta.detach().pow(2).reshape(delta.shape[0], -1).mean(dim=1)).mean().cpu())

    finally:
        for param, flag in zip(model.parameters(), original_requires_grad):
            param.requires_grad_(flag)
        if was_training:
            model.train()

    info = {
        "label_mode": str(cfg.get("label_mode", "solver")),
        "target_source": str(cfg.get("label_mode", "solver")),
        "full_solver_gradient": True,
        "attack_type": "continuous_linf",
        "attack_method": method,
        "attack_steps": used_steps,
        "epsilon_mean": float(eps.mean().detach().cpu()),
        "epsilon_min": float(eps.min().detach().cpu()),
        "epsilon_max": float(eps.max().detach().cpu()),
        "alpha_mean": float((eps * float(alpha_ratio)).mean().detach().cpu()),
        "alpha_min": float((eps * float(alpha_ratio)).min().detach().cpu()),
        "alpha_max": float((eps * float(alpha_ratio)).max().detach().cpu()),
        "alpha_is_epsilon_times_ratio": 1.0,
        "clean_loss_before_attack": clean_loss_value,
        "adv_loss_after_random_start": first_adv_loss,
        "adv_loss_after_attack": final_loss_value,
        "attack_loss_gain": final_loss_value - clean_loss_value,
        "grad_abs_mean_last": grad_abs_mean,
        "boundary_ratio_mean": boundary_ratio,
        "delta_linf_mean": linf,
        "delta_l2_rms_mean": l2,
    }
    info.update(tensor_stats("target", y_train))
    return AttackBatchResult(x_adv.detach(), y_train.detach(), info)


def binary_darcy_replace_attack(
    model,
    xb: torch.Tensor,
    yb: torch.Tensor,
    *,
    steps: int,
    epsilon_fraction: float,
    jitter_low: float,
    jitter_high: float,
    random_pool_multiplier: float,
    random_score_noise: float,
    cfg: dict[str, Any],
) -> AttackBatchResult:
    was_training = model.training
    model.eval()
    original_requires_grad = [param.requires_grad for param in model.parameters()]
    for param in model.parameters():
        param.requires_grad_(False)

    x0 = xb.detach()
    x_adv = x0.detach()
    b = x0.shape[0]
    spatial_shape = x0.shape[1:-1] if x0.ndim == 4 else x0.shape[1:]
    n_pix = int(np.prod(spatial_shape))
    jitter = torch.empty((b,), device=x0.device, dtype=x0.dtype).uniform_(float(jitter_low), float(jitter_high))
    budgets = torch.clamp((float(epsilon_fraction) * jitter * n_pix).round().long(), min=1, max=n_pix)
    used_steps = max(1, int(steps))

    first_loss = float("nan")
    positive_score_frac = float("nan")
    flips_total = 0

    try:
        with torch.no_grad():
            clean_target = solver_target_for_model_input("darcy", x0, yb, cfg, allow_target_grad=False)
            clean_loss = finite_mse(model(x0), clean_target)

        for step in range(used_steps):
            x_score = x_adv.detach().clone().requires_grad_(True)
            pred = model(x_score)
            score_target = solver_target_for_model_input(
                "darcy",
                x_score,
                yb,
                cfg,
                allow_target_grad=solver_target_grad_enabled(cfg),
            )
            loss = finite_mse(pred, score_target)
            if step == 0:
                first_loss = float(loss.detach().cpu())
            grad = torch.autograd.grad(loss, x_score, only_inputs=True)[0].detach()
            grad = torch.nan_to_num(grad)

            flat_x = x_score.detach().reshape(b, -1)
            flat_grad = grad.reshape(b, -1)
            lo = x0.reshape(b, -1).min(dim=1).values.reshape(b, 1)
            hi = x0.reshape(b, -1).max(dim=1).values.reshape(b, 1)
            midpoint = (lo + hi) * 0.5
            other = torch.where(flat_x > midpoint, lo.expand_as(flat_x), hi.expand_as(flat_x))
            score = flat_grad * (other - flat_x)
            score_std = score.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-12)
            if random_score_noise > 0:
                score = score + torch.randn_like(score) * score_std * float(random_score_noise)

            flat_adv = flat_x.clone()
            positive_score_frac = float((score > 0).float().mean().detach().cpu())
            flips_total = 0
            for i in range(b):
                k = int(budgets[i].item())
                pool = min(n_pix, max(k, int(math.ceil(k * float(random_pool_multiplier)))))
                top_idx = torch.topk(score[i], k=pool, largest=True).indices
                if pool > k:
                    chosen = top_idx[torch.randperm(pool, device=x0.device)[:k]]
                else:
                    chosen = top_idx
                flat_adv[i, chosen] = other[i, chosen]
                flips_total += k
            x_adv = flat_adv.reshape_as(x0).detach()

        with torch.no_grad():
            y_train = solver_target_for_model_input("darcy", x_adv, yb, cfg, allow_target_grad=False).detach()
            adv_loss = finite_mse(model(x_adv), y_train)
            delta = x_adv - x0
            changed = delta.reshape(b, -1).abs() > 1e-12
            flip_fraction = float(changed.float().mean().detach().cpu())
            delta_l2 = float(torch.sqrt(delta.pow(2).reshape(b, -1).mean(dim=1)).mean().detach().cpu())

    finally:
        for param, flag in zip(model.parameters(), original_requires_grad):
            param.requires_grad_(flag)
        if was_training:
            model.train()

    info = {
        "label_mode": str(cfg.get("label_mode", "solver")),
        "target_source": str(cfg.get("label_mode", "solver")),
        "full_solver_gradient": True,
        "attack_type": "binary_replace",
        "attack_method": "binary_steepest_replace",
        "attack_steps": used_steps,
        "epsilon_mean": float((budgets.float() / n_pix).mean().detach().cpu()),
        "epsilon_min": float((budgets.float() / n_pix).min().detach().cpu()),
        "epsilon_max": float((budgets.float() / n_pix).max().detach().cpu()),
        "alpha_mean": float((budgets.float() / n_pix).mean().detach().cpu()),
        "clean_loss_before_attack": float(clean_loss.detach().cpu()),
        "adv_loss_after_random_start": first_loss,
        "adv_loss_after_attack": float(adv_loss.detach().cpu()),
        "attack_loss_gain": float((adv_loss - clean_loss).detach().cpu()),
        "grad_abs_mean_last": float(grad.abs().mean().detach().cpu()),
        "boundary_ratio_mean": flip_fraction / max(float(epsilon_fraction), 1e-12),
        "delta_linf_mean": float(delta.abs().reshape(b, -1).max(dim=1).values.mean().detach().cpu()),
        "delta_l2_rms_mean": delta_l2,
        "darcy_budget_pixels_mean": float(budgets.float().mean().detach().cpu()),
        "darcy_flip_fraction": flip_fraction,
        "darcy_positive_score_fraction": positive_score_frac,
        "darcy_last_step_flips_total": flips_total,
    }
    info.update(tensor_stats("target", y_train))
    return AttackBatchResult(x_adv.detach(), y_train.detach(), info)


def ns2d_solver_consistent_attack(
    model,
    xb: torch.Tensor,
    yb: torch.Tensor,
    *,
    method: str,
    steps: int,
    epsilon_fraction: float,
    epsilon_abs: float,
    alpha_ratio: float,
    random_start_fraction: float,
    jitter_low: float,
    jitter_high: float,
    cfg: dict[str, Any],
) -> AttackBatchResult:
    was_training = model.training
    model.eval()
    original_requires_grad = [param.requires_grad for param in model.parameters()]
    for param in model.parameters():
        param.requires_grad_(False)

    x0 = xb[..., 0].detach().contiguous()
    eps = compute_batch_eps(x0, epsilon_fraction, epsilon_abs, jitter_low, jitter_high)
    eps_view = expand_per_sample(eps, x0)
    alpha_view = expand_per_sample(eps * float(alpha_ratio), x0)
    delta = torch.clamp(random_start_delta(x0, eps, random_start_fraction), -eps_view, eps_view)
    x0_adv = (x0 + delta).detach()

    clean_loss_value = float("nan")
    first_adv_loss = float("nan")
    final_loss_value = float("nan")
    grad_abs_mean = float("nan")
    used_steps = max(1, int(steps))

    try:
        for step in range(used_steps):
            x0_adv = x0_adv.detach().requires_grad_(True)
            x_seq_adv, y_seq_adv = ns2d_solver_pair_from_initial(x0_adv, cfg, for_attack=True)
            pred = model(x_seq_adv)
            loss = finite_mse(pred, y_seq_adv)
            if step == 0:
                with torch.no_grad():
                    x_seq_clean, y_seq_clean = ns2d_solver_pair_from_initial(x0, cfg, for_attack=False)
                    clean_loss_value = float(finite_mse(model(x_seq_clean), y_seq_clean).detach().cpu())
                    first_adv_loss = float(loss.detach().cpu())
            grad = torch.autograd.grad(loss, x0_adv, only_inputs=True)[0]
            grad = torch.nan_to_num(grad)
            grad_abs_mean = float(grad.abs().mean().detach().cpu())
            if method.endswith("replace_linf"):
                delta = eps_view * grad.sign()
            elif method.endswith("add_linf"):
                delta = (x0_adv.detach() - x0) + alpha_view * grad.sign()
                delta = torch.clamp(delta, -eps_view, eps_view)
            else:
                raise ValueError(f"unknown NS2D continuous attack method: {method}")
            x0_adv = (x0 + delta).detach()

        with torch.no_grad():
            x_train, y_train = ns2d_solver_pair_from_initial(x0_adv, cfg, for_attack=False)
            final_loss_value = float(finite_mse(model(x_train), y_train).detach().cpu())
            delta = x0_adv - x0
            denom = eps_view.expand_as(delta).clamp_min(1e-12)
            boundary_ratio = float((delta.detach().abs() / denom).mean().cpu())
            linf = float(delta.detach().abs().reshape(delta.shape[0], -1).max(dim=1).values.mean().cpu())
            l2 = float(torch.sqrt(delta.detach().pow(2).reshape(delta.shape[0], -1).mean(dim=1)).mean().cpu())

    finally:
        for param, flag in zip(model.parameters(), original_requires_grad):
            param.requires_grad_(flag)
        if was_training:
            model.train()

    info = {
        "label_mode": str(cfg.get("label_mode", "solver")),
        "target_source": "solver_rollout_from_attacked_initial_state",
        "full_solver_gradient": True,
        "attack_type": "ns2d_initial_linf_solver_rollout",
        "attack_method": method,
        "attack_steps": used_steps,
        "epsilon_mean": float(eps.mean().detach().cpu()),
        "epsilon_min": float(eps.min().detach().cpu()),
        "epsilon_max": float(eps.max().detach().cpu()),
        "alpha_mean": float((eps * float(alpha_ratio)).mean().detach().cpu()),
        "alpha_min": float((eps * float(alpha_ratio)).min().detach().cpu()),
        "alpha_max": float((eps * float(alpha_ratio)).max().detach().cpu()),
        "alpha_is_epsilon_times_ratio": 1.0,
        "clean_loss_before_attack": clean_loss_value,
        "adv_loss_after_random_start": first_adv_loss,
        "adv_loss_after_attack": final_loss_value,
        "attack_loss_gain": final_loss_value - clean_loss_value,
        "grad_abs_mean_last": grad_abs_mean,
        "boundary_ratio_mean": boundary_ratio,
        "delta_linf_mean": linf,
        "delta_l2_rms_mean": l2,
    }
    info.update(tensor_stats("target", y_train))
    info.update(tensor_stats("x_train", x_train))
    return AttackBatchResult(x_train.detach(), y_train.detach(), info)


def attack_batch(model, xb: torch.Tensor, yb: torch.Tensor, task: str, cfg: dict[str, Any]) -> AttackBatchResult:
    method = str(cfg["attack_method"])
    label_mode = str(cfg.get("label_mode", "solver"))
    if label_mode == "solver" and task == "ns2d":
        return ns2d_solver_consistent_attack(
            model,
            xb,
            yb,
            method=method,
            steps=int(cfg["attack_steps"]),
            epsilon_fraction=float(cfg["epsilon_fraction"]),
            epsilon_abs=float(cfg["epsilon_abs"]),
            alpha_ratio=float(cfg["alpha_ratio"]),
            random_start_fraction=float(cfg["random_start_fraction"]),
            jitter_low=float(cfg["eps_jitter_low"]),
            jitter_high=float(cfg["eps_jitter_high"]),
            cfg=cfg,
        )
    if task == "darcy" and method == "binary_steepest_replace":
        return binary_darcy_replace_attack(
            model,
            xb,
            yb,
            steps=int(cfg["attack_steps"]),
            epsilon_fraction=float(cfg["epsilon_fraction"]),
            jitter_low=float(cfg["eps_jitter_low"]),
            jitter_high=float(cfg["eps_jitter_high"]),
            random_pool_multiplier=float(cfg["binary_pool_multiplier"]),
            random_score_noise=float(cfg["binary_score_noise"]),
            cfg=cfg,
        )
    return continuous_attack(
        model,
        xb,
        yb,
        task=task,
        method=method,
        steps=int(cfg["attack_steps"]),
        epsilon_fraction=float(cfg["epsilon_fraction"]),
        epsilon_abs=float(cfg["epsilon_abs"]),
        alpha_ratio=float(cfg["alpha_ratio"]),
        random_start_fraction=float(cfg["random_start_fraction"]),
        jitter_low=float(cfg["eps_jitter_low"]),
        jitter_high=float(cfg["eps_jitter_high"]),
        ns_frames=str(cfg["ns_attack_frames"]),
        cfg=cfg,
    )


def clean_solver_training_pair(
    task: str,
    xb: torch.Tensor,
    yb: torch.Tensor,
    cfg: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return clean training pairs with solver-consistent labels.

    This is used by --training-data-mode clean-plus-adv so the clean half is
    also trained as model(x_clean) -> solver(x_clean), not from stale labels.
    """
    if task == "ns2d":
        if xb.ndim != 4:
            raise ValueError(f"NS2D clean-plus-adv expects clean sequence input (B,H,W,T), got {tuple(xb.shape)}")
        x0 = xb[..., 0].detach().contiguous()
        return ns2d_solver_pair_from_initial(x0, cfg, for_attack=False)
    x_clean = xb.detach()
    y_clean = solver_target_for_model_input(task, x_clean, yb, cfg, allow_target_grad=False).detach()
    return x_clean, y_clean


def combine_training_pairs(
    task: str,
    xb: torch.Tensor,
    yb: torch.Tensor,
    attack_result: AttackBatchResult,
    cfg: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    mode = str(cfg.get("training_data_mode", "adv-only"))
    x_adv = attack_result.x_train.detach()
    y_adv = attack_result.y_train.detach()
    if mode == "adv-only":
        info = {
            "training_data_mode": mode,
            "clean_train_samples": 0,
            "adv_train_samples": int(x_adv.shape[0]),
            "effective_train_samples": int(x_adv.shape[0]),
            "clean_plus_adv_multiplier": 1.0,
        }
        return x_adv, y_adv, info
    if mode == "clean-plus-adv":
        x_clean, y_clean = clean_solver_training_pair(task, xb, yb, cfg)
        x_train = torch.cat([x_clean.detach(), x_adv], dim=0)
        y_train = torch.cat([y_clean.detach(), y_adv], dim=0)
        info = {
            "training_data_mode": mode,
            "clean_train_samples": int(x_clean.shape[0]),
            "adv_train_samples": int(x_adv.shape[0]),
            "effective_train_samples": int(x_train.shape[0]),
            "clean_plus_adv_multiplier": float(x_train.shape[0]) / max(1, int(x_adv.shape[0])),
        }
        return x_train.detach(), y_train.detach(), info
    raise ValueError(f"unknown training_data_mode={mode!r}; expected adv-only or clean-plus-adv")


def task_train_spec(specs: list[DatasetSpec], task: str) -> DatasetSpec:
    matches = [s for s in specs if s.task == task and s.split == "train"]
    if not matches:
        raise ValueError(f"no train spec for {task}")
    return matches[0]


def task_eval_specs(specs: list[DatasetSpec], task: str, max_generalization_eval: int | None) -> list[DatasetSpec]:
    rows = [s for s in specs if s.task == task and s.split in {"train", "test"}]
    gen = [s for s in specs if s.task == task and s.split == "generalization"]
    gen.sort(key=lambda s: (s.manual_rank, s.dataset_id))
    if max_generalization_eval is not None:
        gen = gen[: max(0, int(max_generalization_eval))]
    return rows + gen


def validate_eval_dataset_counts(specs: list[DatasetSpec], tasks: list[str], expected_generalization: int = 50) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for task in tasks:
        task_specs = [s for s in specs if s.task == task]
        train = [s for s in task_specs if s.split == "train"]
        test = [s for s in task_specs if s.split == "test"]
        gen = [s for s in task_specs if s.split == "generalization"]
        row = {
            "task": task,
            "train_count": len(train),
            "test_count": len(test),
            "generalization_count": len(gen),
            "eval_dataset_count_per_pass": len(train) + len(test) + len(gen),
            "expected_generalization_count": expected_generalization,
            "expected_eval_dataset_count_per_pass": expected_generalization + 2,
            "generalization_ids": [s.dataset_id for s in sorted(gen, key=lambda x: x.dataset_id)],
        }
        rows.append(row)
        if len(train) != 1:
            errors.append(f"{task}: expected exactly 1 train dataset, got {len(train)}")
        if len(test) != 1:
            errors.append(f"{task}: expected exactly 1 test dataset, got {len(test)}")
        if len(gen) != expected_generalization:
            errors.append(f"{task}: expected exactly {expected_generalization} generalization datasets, got {len(gen)}")
    if errors:
        raise ValueError("preflight dataset count check failed:\n" + "\n".join(errors))
    return rows


def load_latest_smoke_summary(output_root: Path) -> dict[str, Any] | None:
    candidates = sorted(output_root.glob("smoke*/summary.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["summary_path"] = str(path)
            return payload
        except Exception:
            continue
    return None


def effective_task_cfg_for_preflight(task: str, args) -> dict[str, Any]:
    cfg = dict(asdict(DEFAULTS[task]))
    if args.epochs is not None:
        cfg["epochs"] = int(args.epochs)
    override_train_max = getattr(args, f"{task}_train_max", None)
    if override_train_max is not None:
        cfg["train_max_samples"] = int(override_train_max)
    override_batch = getattr(args, f"{task}_batch_size", None)
    if override_batch is not None:
        cfg["batch_size"] = int(override_batch)
    override_optimizer_batch = getattr(args, f"{task}_optimizer_batch_size", None)
    if override_optimizer_batch is not None:
        cfg["optimizer_batch_size"] = int(override_optimizer_batch)
    override_attack_steps = getattr(args, f"{task}_attack_steps", None)
    if override_attack_steps is not None:
        cfg["attack_steps"] = int(override_attack_steps)
    override_alpha_ratio = getattr(args, f"{task}_alpha_ratio", None)
    if override_alpha_ratio is not None:
        cfg["alpha_ratio"] = float(override_alpha_ratio)
    cfg["max_batches_per_epoch"] = args.max_batches_per_epoch
    return cfg


def build_planned_workload_estimate(
    specs: list[DatasetSpec],
    tasks: list[str],
    args,
    smoke_summary: dict[str, Any] | None,
    expected_generalization: int = 50,
) -> dict[str, Any]:
    smoke_by_task: dict[str, dict[str, Any]] = {}
    if smoke_summary:
        for row in smoke_summary.get("tasks", []):
            if isinstance(row, dict) and row.get("task"):
                smoke_by_task[row["task"]] = row

    task_rows: list[dict[str, Any]] = []
    total_seconds = 0.0
    for task in tasks:
        cfg = effective_task_cfg_for_preflight(task, args)
        train_spec = task_train_spec(specs, task)
        full_train_count = train_sample_count(train_spec, task)
        max_samples = normalize_max_samples(cfg.get("train_max_samples"))
        n_train = full_train_count if max_samples is None else min(full_train_count, max_samples)
        batch_size = int(cfg["batch_size"])
        optimizer_batch_size = int(cfg.get("optimizer_batch_size") or batch_size)
        batches_nominal = int(math.ceil(n_train / max(1, batch_size)))
        max_batches = cfg.get("max_batches_per_epoch")
        batches_per_epoch = min(batches_nominal, int(max_batches)) if max_batches is not None else batches_nominal
        total_steps = int(cfg["epochs"]) * max(1, batches_per_epoch)
        optimizer_steps_per_epoch = count_optimizer_steps_for_epoch(n_train, batch_size, optimizer_batch_size, max_batches)
        total_optimizer_steps = int(cfg["epochs"]) * max(1, optimizer_steps_per_epoch)
        eval_every_steps = max(1, int(math.ceil(total_steps * float(args.eval_every_fraction))))
        eval_steps = sorted(set(list(range(eval_every_steps, total_steps + 1, eval_every_steps)) + [total_steps]))
        eval_passes = 1 + len(eval_steps)
        eval_dataset_count = expected_generalization + 2

        smoke = smoke_by_task.get(task, {})
        smoke_est = smoke.get("smoke_based_estimate", {}) if isinstance(smoke, dict) else {}
        per_batch_sec = float(smoke_est.get("smoke_per_batch_sec", float("nan")))
        smoke_eval_sec = float(smoke_est.get("smoke_eval_sec", float("nan")))
        smoke_eval_dataset_count = 4.0
        if math.isfinite(per_batch_sec):
            estimated_train_sec = per_batch_sec * total_steps
        else:
            estimated_train_sec = float("nan")
        if math.isfinite(smoke_eval_sec):
            estimated_eval_sec = smoke_eval_sec * (eval_dataset_count / smoke_eval_dataset_count) * eval_passes
        else:
            estimated_eval_sec = float("nan")
        estimated_total = estimated_train_sec + estimated_eval_sec if math.isfinite(estimated_train_sec) and math.isfinite(estimated_eval_sec) else float("nan")
        if math.isfinite(estimated_total):
            total_seconds += estimated_total
        mem = smoke.get("memory", {}) if isinstance(smoke, dict) else {}
        row = {
            "task": task,
            "baseline_checkpoint": baseline_checkpoint_manifest()[task],
            "full_train_dataset_samples": full_train_count,
            "train_max_samples": "full" if max_samples is None else max_samples,
            "effective_train_samples": n_train,
            "batch_size": batch_size,
            "attack_batch_size": batch_size,
            "optimizer_batch_size": optimizer_batch_size,
            "optimizer_microbatches_per_full_attack_batch": int(math.ceil(min(batch_size, n_train) / max(1, optimizer_batch_size))),
            "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
            "total_optimizer_steps": total_optimizer_steps,
            "epochs": int(cfg["epochs"]),
            "attack_method": cfg["attack_method"],
            "attack_steps": int(cfg["attack_steps"]),
            "epsilon_fraction": float(cfg["epsilon_fraction"]),
            "alpha_ratio": float(cfg["alpha_ratio"]),
            "full_solver_gradient": True,
            "batches_per_epoch": batches_per_epoch,
            "total_train_steps": total_steps,
            "eval_every_steps": eval_every_steps,
            "eval_steps_after_baseline": eval_steps,
            "eval_pass_count_including_baseline": eval_passes,
            "eval_dataset_count_per_pass": eval_dataset_count,
            "estimated_train_seconds_from_smoke": estimated_train_sec,
            "estimated_eval_seconds_from_smoke_scaled_to_52_datasets": estimated_eval_sec,
            "estimated_total_seconds_from_smoke": estimated_total,
            "estimated_total_minutes_from_smoke": estimated_total / 60.0 if math.isfinite(estimated_total) else float("nan"),
            "smoke_peak_cuda_allocated_mb": mem.get("cuda_peak_allocated_mb"),
        }
        task_rows.append(row)
    return {
        "created_utc": now_stamp(),
        "expected_generalization_datasets_per_task": expected_generalization,
        "expected_eval_datasets_per_pass": expected_generalization + 2,
        "smoke_reference_summary": smoke_summary.get("summary_path") if smoke_summary else None,
        "tasks": task_rows,
        "estimated_total_seconds_from_smoke": total_seconds,
        "estimated_total_minutes_from_smoke": total_seconds / 60.0,
    }


def evaluate_task(
    model,
    task: str,
    specs: list[DatasetSpec],
    device: torch.device,
    eval_batch_size: int,
    eval_max_samples: int | None,
    out_csv: Path,
    *,
    global_step: int,
    epoch: int,
    progress_fraction: float,
    phase: str,
    max_generalization_eval: int | None,
) -> tuple[list[dict[str, Any]], float]:
    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    was_training = model.training
    model.eval()
    for spec in task_eval_specs(specs, task, max_generalization_eval):
        max_samples = eval_max_samples
        result = evaluate_dataset(model, spec, device, eval_batch_size, max_samples)
        row = {
            "phase": phase,
            "global_step": global_step,
            "epoch": epoch,
            "progress_fraction": progress_fraction,
            **result,
        }
        write_csv_row(out_csv, row)
        rows.append(row)
    if was_training:
        model.train()
    return rows, time.perf_counter() - start


def make_indices(n: int, batch_size: int, generator: torch.Generator) -> list[torch.Tensor]:
    perm = torch.randperm(n, generator=generator)
    return [perm[start : start + batch_size] for start in range(0, n, batch_size)]


def make_slices(n: int, batch_size: int) -> list[slice]:
    batch_size = max(1, int(batch_size))
    return [slice(start, min(start + batch_size, n)) for start in range(0, n, batch_size)]


def epoch_attack_batch_sizes(n: int, attack_batch_size: int, max_batches: int | None = None) -> list[int]:
    attack_batch_size = max(1, int(attack_batch_size))
    sizes = [min(attack_batch_size, n - start) for start in range(0, n, attack_batch_size)]
    if max_batches is not None:
        sizes = sizes[: int(max_batches)]
    return sizes


def count_optimizer_steps_for_epoch(
    n: int,
    attack_batch_size: int,
    optimizer_batch_size: int,
    max_batches: int | None = None,
) -> int:
    optimizer_batch_size = max(1, int(optimizer_batch_size))
    return sum(int(math.ceil(size / optimizer_batch_size)) for size in epoch_attack_batch_sizes(n, attack_batch_size, max_batches))


def checkpoint_model(model, out_dir: Path, task: str, epoch: int, global_step: int, cfg: dict[str, Any]) -> Path:
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"{task}_epoch{epoch:03d}_step{global_step:06d}.pt"
    torch.save(
        {
            "task": task,
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "config": cfg,
        },
        path,
    )
    return path


def estimate_from_smoke(
    task: str,
    train_rows: list[dict[str, Any]],
    eval_seconds: list[float],
    full_epochs: int,
    full_train_samples: int,
    batch_size: int,
    optimizer_batch_size: int,
    eval_every_fraction: float,
) -> dict[str, Any]:
    if not train_rows:
        return {"task": task, "estimate_available": False}
    per_batch = float(np.mean([float(r["step_wall_sec"]) for r in train_rows]))
    attack = float(np.mean([float(r["attack_wall_sec"]) for r in train_rows]))
    train = float(np.mean([float(r["train_wall_sec"]) for r in train_rows]))
    batches_per_epoch = int(math.ceil(full_train_samples / max(1, batch_size)))
    optimizer_microbatches_per_attack_batch = int(math.ceil(min(full_train_samples, batch_size) / max(1, optimizer_batch_size)))
    optimizer_steps_per_epoch = count_optimizer_steps_for_epoch(full_train_samples, batch_size, optimizer_batch_size)
    train_batches = full_epochs * batches_per_epoch
    optimizer_steps = full_epochs * max(1, optimizer_steps_per_epoch)
    evals = max(1, int(math.ceil(1.0 / max(eval_every_fraction, 1e-9)))) + 1
    eval_mean = float(np.mean(eval_seconds)) if eval_seconds else 0.0
    total_seconds = per_batch * train_batches + eval_mean * evals
    return {
        "task": task,
        "estimate_available": True,
        "smoke_per_batch_sec": per_batch,
        "smoke_attack_sec": attack,
        "smoke_train_sec": train,
        "smoke_eval_sec": eval_mean,
        "full_epochs": full_epochs,
        "full_train_samples": full_train_samples,
        "batch_size": batch_size,
        "attack_batch_size": batch_size,
        "optimizer_batch_size": optimizer_batch_size,
        "optimizer_microbatches_per_attack_batch": optimizer_microbatches_per_attack_batch,
        "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
        "batches_per_epoch": batches_per_epoch,
        "estimated_train_batches": train_batches,
        "estimated_optimizer_steps": optimizer_steps,
        "estimated_eval_count": evals,
        "estimated_total_seconds": total_seconds,
        "estimated_total_minutes": total_seconds / 60.0,
    }


def train_one_task(
    task: str,
    all_specs: list[DatasetSpec],
    device: torch.device,
    out_root: Path,
    args,
    run_cfg: dict[str, Any],
) -> dict[str, Any]:
    defaults = DEFAULTS[task]
    task_cfg = dict(asdict(defaults))
    task_cfg.update(
        {
            "task": task,
            "device": str(device),
            "eval_every_fraction": args.eval_every_fraction,
            "eval_max_samples": args.eval_max_samples,
            "max_generalization_eval": args.max_generalization_eval,
            "binary_pool_multiplier": args.binary_pool_multiplier,
            "binary_score_noise": args.binary_score_noise,
            "ns_attack_frames": args.ns_attack_frames,
            "label_mode": args.label_mode,
            "training_data_mode": args.training_data_mode,
            "burgers_solver_remat": args.burgers_solver_remat,
            "burgers_solver_remat_chunk_steps": args.burgers_solver_remat_chunk_steps,
            "ns2d_solver_remat": args.ns2d_solver_remat,
            "ns2d_solver_remat_chunk_steps": args.ns2d_solver_remat_chunk_steps,
            "full_solver_gradient": True,
        }
    )
    if args.smoke:
        smoke = SMOKE_LIMITS[task]
        task_cfg["epochs"] = 1
        task_cfg["train_max_samples"] = smoke["train_max_samples"]
        task_cfg["max_batches_per_epoch"] = smoke["max_batches"]
        task_cfg["eval_max_samples"] = smoke["eval_max_samples"]
        task_cfg["max_generalization_eval"] = min(args.max_generalization_eval or 3, 3)
        if task == "ns2d":
            task_cfg["attack_steps"] = min(int(task_cfg["attack_steps"]), 2)
        else:
            task_cfg["attack_steps"] = min(int(task_cfg["attack_steps"]), 2)
    else:
        task_cfg["max_batches_per_epoch"] = args.max_batches_per_epoch
        if args.epochs is not None:
            task_cfg["epochs"] = int(args.epochs)
        override_train_max = getattr(args, f"{task}_train_max", None)
        if override_train_max is not None:
            task_cfg["train_max_samples"] = normalize_max_samples(override_train_max)
        override_batch = getattr(args, f"{task}_batch_size", None)
        if override_batch is not None:
            task_cfg["batch_size"] = int(override_batch)
        override_optimizer_batch = getattr(args, f"{task}_optimizer_batch_size", None)
        if override_optimizer_batch is not None:
            task_cfg["optimizer_batch_size"] = int(override_optimizer_batch)

    override_attack_steps = getattr(args, f"{task}_attack_steps", None)
    if override_attack_steps is not None:
        task_cfg["attack_steps"] = int(override_attack_steps)
    override_alpha_ratio = getattr(args, f"{task}_alpha_ratio", None)
    if override_alpha_ratio is not None:
        task_cfg["alpha_ratio"] = float(override_alpha_ratio)
    override_optimizer_batch = getattr(args, f"{task}_optimizer_batch_size", None)
    if override_optimizer_batch is not None:
        task_cfg["optimizer_batch_size"] = int(override_optimizer_batch)

    out_dir = out_root / task
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(to_jsonable({**run_cfg, **task_cfg}), indent=2), encoding="utf-8")

    model = load_model(task, device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(task_cfg["learning_rate"]),
        weight_decay=float(task_cfg["weight_decay"]),
    )

    train_spec = task_train_spec(all_specs, task)
    train_max_samples = normalize_max_samples(task_cfg.get("train_max_samples"))
    x_cpu, y_cpu = load_train_xy(train_spec, task, train_max_samples)
    data_summary = dataset_range_summary(x_cpu, y_cpu)
    (out_dir / "data_range_summary.json").write_text(json.dumps(data_summary, indent=2), encoding="utf-8")

    n_train = int(x_cpu.shape[0])
    batch_size = int(task_cfg["batch_size"])
    optimizer_batch_size = int(task_cfg.get("optimizer_batch_size") or batch_size)
    optimizer_batch_size = max(1, optimizer_batch_size)
    batches_per_epoch_nominal = int(math.ceil(n_train / max(1, batch_size)))
    max_batches = task_cfg.get("max_batches_per_epoch")
    if max_batches is not None:
        batches_per_epoch = min(batches_per_epoch_nominal, int(max_batches))
    else:
        batches_per_epoch = batches_per_epoch_nominal
    total_steps = int(task_cfg["epochs"]) * max(1, batches_per_epoch)
    optimizer_steps_per_epoch = count_optimizer_steps_for_epoch(n_train, batch_size, optimizer_batch_size, max_batches)
    total_optimizer_steps = int(task_cfg["epochs"]) * max(1, optimizer_steps_per_epoch)
    eval_every_steps = max(1, int(math.ceil(total_steps * float(args.eval_every_fraction))))

    train_csv = out_dir / "train_steps.csv"
    attack_csv = out_dir / "attack_batches.csv"
    optimizer_csv = out_dir / "optimizer_steps.csv"
    eval_csv = out_dir / "eval_metrics.csv"
    memory_csv = out_dir / "memory.csv"

    eval_seconds: list[float] = []
    eval_rows, seconds = evaluate_task(
        model,
        task,
        all_specs,
        device,
        int(task_cfg["eval_batch_size"]),
        task_cfg["eval_max_samples"],
        eval_csv,
        global_step=0,
        epoch=0,
        progress_fraction=0.0,
        phase="baseline_before_adversarial_training",
        max_generalization_eval=task_cfg["max_generalization_eval"],
    )
    eval_seconds.append(seconds)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    global_step = 0
    optimizer_global_step = 0
    train_rows: list[dict[str, Any]] = []
    seed_base = int(args.seed)
    train_start_wall = time.perf_counter()

    for epoch in range(1, int(task_cfg["epochs"]) + 1):
        generator = torch.Generator().manual_seed(seed_base + epoch * 1009 + hash(task) % 1000)
        batches = make_indices(n_train, batch_size, generator)
        if max_batches is not None:
            batches = batches[: int(max_batches)]
        for local_batch_idx, idx in enumerate(batches, 1):
            step_start = time.perf_counter()
            global_step += 1
            xb = x_cpu[idx].to(device, non_blocking=True)
            yb = y_cpu[idx].to(device, non_blocking=True)

            attack_start = time.perf_counter()
            attack_result = attack_batch(model, xb, yb, task, task_cfg)
            attack_info = attack_result.info
            x_train_adv, y_train_adv, training_mix_info = combine_training_pairs(task, xb, yb, attack_result, task_cfg)
            attack_sec = time.perf_counter() - attack_start

            train_start = time.perf_counter()
            micro_slices = make_slices(int(x_train_adv.shape[0]), optimizer_batch_size)
            loss_weighted_sum = 0.0
            loss_weight = 0
            grad_norms: list[float] = []
            optimizer_secs: list[float] = []
            for micro_idx, micro_slice in enumerate(micro_slices, 1):
                micro_start = time.perf_counter()
                optimizer_global_step += 1
                x_micro = x_train_adv[micro_slice]
                y_micro = y_train_adv[micro_slice]
                optimizer.zero_grad(set_to_none=True)
                pred = model(x_micro)
                loss = finite_mse(pred, y_micro)
                loss.backward()
                grad_norm = safe_grad_norm(model)
                optimizer.step()
                micro_sec = time.perf_counter() - micro_start
                micro_n = int(x_micro.shape[0])
                loss_value = float(loss.detach().cpu())
                loss_weighted_sum += loss_value * micro_n
                loss_weight += micro_n
                grad_norms.append(grad_norm)
                optimizer_secs.append(micro_sec)
                write_csv_row(
                    optimizer_csv,
                    {
                        "task": task,
                        "epoch": epoch,
                        "attack_global_step": global_step,
                        "optimizer_global_step": optimizer_global_step,
                        "local_batch_idx": local_batch_idx,
                        "microbatch_idx": micro_idx,
                        "attack_batch_size": int(xb.shape[0]),
                        "optimizer_batch_size": micro_n,
                        "configured_optimizer_batch_size": optimizer_batch_size,
                        "train_loss_on_adv_microbatch": loss_value,
                        "grad_norm": grad_norm,
                        "optimizer_wall_sec": micro_sec,
                        **memory_stats(device),
                    },
                )
            train_sec = time.perf_counter() - train_start
            step_sec = time.perf_counter() - step_start
            train_loss_mean = loss_weighted_sum / max(1, loss_weight)
            grad_norm_mean = float(np.mean(grad_norms)) if grad_norms else float("nan")
            grad_norm_last = float(grad_norms[-1]) if grad_norms else float("nan")

            progress = global_step / max(1, total_steps)
            row = {
                "task": task,
                "epoch": epoch,
                "global_step": global_step,
                "optimizer_global_step_last": optimizer_global_step,
                "local_batch_idx": local_batch_idx,
                "batch_size": int(xb.shape[0]),
                "attack_batch_size": int(xb.shape[0]),
                "configured_attack_batch_size": batch_size,
                "optimizer_batch_size": optimizer_batch_size,
                "optimizer_microbatches": len(micro_slices),
                **training_mix_info,
                "progress_fraction": progress,
                "train_loss_on_adv": train_loss_mean,
                "train_loss_on_adv_mean": train_loss_mean,
                "label_mode": str(task_cfg.get("label_mode", "solver")),
                "target_source": str(attack_info.get("target_source", task_cfg.get("label_mode", "solver"))),
                "grad_norm": grad_norm_last,
                "grad_norm_mean": grad_norm_mean,
                "grad_norm_last": grad_norm_last,
                "attack_wall_sec": attack_sec,
                "attack_sec_per_sample": attack_sec / max(1, int(xb.shape[0])),
                "attack_samples_per_sec": int(xb.shape[0]) / max(attack_sec, 1e-12),
                "train_wall_sec": train_sec,
                "train_sec_per_sample": train_sec / max(1, int(xb.shape[0])),
                "step_sec_per_sample": step_sec / max(1, int(xb.shape[0])),
                "step_samples_per_sec": int(xb.shape[0]) / max(step_sec, 1e-12),
                "optimizer_wall_sec_mean": float(np.mean(optimizer_secs)) if optimizer_secs else float("nan"),
                "step_wall_sec": step_sec,
                **memory_stats(device),
            }
            row.update(attack_info)
            write_csv_row(train_csv, row)
            write_csv_row(attack_csv, {k: row[k] for k in row if k.startswith("attack") or k.startswith("epsilon") or k.startswith("alpha") or k.startswith("delta") or k.startswith("darcy") or k.startswith("target") or k.startswith("solver") or k.startswith("x_train") or k in {"task", "epoch", "global_step", "label_mode", "boundary_ratio_mean", "clean_loss_before_attack", "adv_loss_after_attack"}})
            write_csv_row(memory_csv, {"task": task, "epoch": epoch, "global_step": global_step, **memory_stats(device)})
            train_rows.append(row)

            should_eval = global_step == total_steps or global_step % eval_every_steps == 0
            if should_eval:
                ckpt = checkpoint_model(model, out_dir, task, epoch, global_step, task_cfg)
                eval_rows, seconds = evaluate_task(
                    model,
                    task,
                    all_specs,
                    device,
                    int(task_cfg["eval_batch_size"]),
                    task_cfg["eval_max_samples"],
                    eval_csv,
                    global_step=global_step,
                    epoch=epoch,
                    progress_fraction=progress,
                    phase="during_adversarial_training",
                    max_generalization_eval=task_cfg["max_generalization_eval"],
                )
                eval_seconds.append(seconds)
                write_csv_row(
                    out_dir / "checkpoints.csv",
                    {
                        "task": task,
                        "epoch": epoch,
                        "global_step": global_step,
                        "progress_fraction": progress,
                        "checkpoint_path": str(ckpt.relative_to(PROJECT_ROOT)),
                        "eval_wall_sec": seconds,
                    },
                )

    final_ckpt = checkpoint_model(model, out_dir, task, int(task_cfg["epochs"]), global_step, task_cfg)
    elapsed = time.perf_counter() - train_start_wall
    estimate = estimate_from_smoke(
        task,
        train_rows,
        eval_seconds,
        full_epochs=DEFAULTS[task].epochs if args.epochs is None else int(args.epochs),
        full_train_samples=n_train,
        batch_size=batch_size,
        optimizer_batch_size=optimizer_batch_size,
        eval_every_fraction=float(args.eval_every_fraction),
    )
    summary = {
        "task": task,
        "run_dir": str(out_dir.relative_to(PROJECT_ROOT)),
        "train_samples": n_train,
        "train_max_samples": "full" if train_max_samples is None else train_max_samples,
        "batch_size": batch_size,
        "attack_batch_size": batch_size,
        "training_data_mode": str(task_cfg.get("training_data_mode", "adv-only")),
        "effective_train_samples_per_full_attack_batch": (2 * batch_size if str(task_cfg.get("training_data_mode", "adv-only")) == "clean-plus-adv" else batch_size),
        "optimizer_batch_size": optimizer_batch_size,
        "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
        "total_optimizer_steps": total_optimizer_steps,
        "epochs": int(task_cfg["epochs"]),
        "total_steps": global_step,
        "elapsed_seconds": elapsed,
        "elapsed_minutes": elapsed / 60.0,
        "final_checkpoint": str(final_ckpt.relative_to(PROJECT_ROOT)),
        "memory": memory_stats(device),
        "data_range_summary": data_summary,
        "smoke_based_estimate": estimate,
    }
    (out_dir / "summary.json").write_text(json.dumps(to_jsonable(summary), indent=2), encoding="utf-8")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def parse_tasks(value: str) -> list[str]:
    tasks = [x.strip() for x in value.split(",") if x.strip()]
    bad = [x for x in tasks if x not in TASK_ORDER]
    if bad:
        raise ValueError(f"unknown task(s): {bad}; allowed={TASK_ORDER}")
    return [x for x in TASK_ORDER if x in set(tasks)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default="burgers,darcy,ns2d")
    parser.add_argument("--generalization-root", type=Path, default=PROJECT_ROOT / "generalization_datasets_rmse_1p5_3x_all_ns50")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "adversarial_training_runs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260530)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--eval-every-fraction", type=float, default=0.2)
    parser.add_argument("--eval-max-samples", type=int, default=50)
    parser.add_argument("--max-generalization-eval", type=int, default=None)
    parser.add_argument("--max-batches-per-epoch", type=int, default=None)
    parser.add_argument("--label-mode", choices=["solver", "clean"], default="solver")
    parser.add_argument("--training-data-mode", choices=["adv-only", "clean-plus-adv"], default="adv-only", help="adv-only trains only on attacked solver pairs; clean-plus-adv doubles each attack batch with clean solver pairs plus attacked solver pairs.")
    parser.add_argument("--allow-clean-label", action="store_true", help="Allow the intentionally non-physical x_adv -> y_clean objective for ablation/debug only.")
    parser.add_argument("--ns-attack-frames", choices=["first", "all"], default="first")
    parser.add_argument("--burgers-solver-remat", choices=["none", "micro", "step", "chunk", "all"], default="none")
    parser.add_argument("--burgers-solver-remat-chunk-steps", type=int, default=20)
    parser.add_argument("--ns2d-solver-remat", choices=["none", "micro", "step", "chunk", "second"], default="chunk")
    parser.add_argument("--ns2d-solver-remat-chunk-steps", type=int, default=20)
    parser.add_argument("--binary-pool-multiplier", type=float, default=2.5)
    parser.add_argument("--binary-score-noise", type=float, default=0.10)
    parser.add_argument("--burgers-train-max", type=int, default=None)
    parser.add_argument("--darcy-train-max", type=int, default=None)
    parser.add_argument("--ns2d-train-max", type=int, default=None)
    parser.add_argument("--burgers-batch-size", type=int, default=None)
    parser.add_argument("--darcy-batch-size", type=int, default=None)
    parser.add_argument("--ns2d-batch-size", type=int, default=None)
    parser.add_argument("--burgers-optimizer-batch-size", type=int, default=None)
    parser.add_argument("--darcy-optimizer-batch-size", type=int, default=None)
    parser.add_argument("--ns2d-optimizer-batch-size", type=int, default=None)
    parser.add_argument("--burgers-attack-steps", type=int, default=None)
    parser.add_argument("--darcy-attack-steps", type=int, default=None)
    parser.add_argument("--ns2d-attack-steps", type=int, default=None)
    parser.add_argument("--burgers-alpha-ratio", type=float, default=None)
    parser.add_argument("--darcy-alpha-ratio", type=float, default=None)
    parser.add_argument("--ns2d-alpha-ratio", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.label_mode == "clean" and not args.allow_clean_label:
        raise ValueError("--label-mode clean is the old non-physical x_adv -> y_clean objective. Pass --allow-clean-label only for an explicit ablation.")
    if args.training_data_mode == "clean-plus-adv" and args.label_mode != "solver":
        raise ValueError("--training-data-mode clean-plus-adv requires --label-mode solver so both clean and attacked pairs use solver-generated Y.")
    set_seed(int(args.seed))
    tasks = parse_tasks(args.tasks)
    device = torch.device(args.device)
    run_name = args.run_name or ("smoke_" if args.smoke else "full_") + now_stamp()
    out_root = (args.output_root / run_name).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    all_specs = build_specs(args.generalization_root.resolve())
    missing = [str(s.path) for s in all_specs if s.task in tasks and not s.path.exists()]
    if missing:
        raise FileNotFoundError("missing dataset files:\n" + "\n".join(missing[:20]))

    dataset_count_rows = validate_eval_dataset_counts(all_specs, tasks, expected_generalization=50)
    (out_root / "preflight_dataset_counts.json").write_text(json.dumps(to_jsonable(dataset_count_rows), indent=2), encoding="utf-8")
    smoke_summary = load_latest_smoke_summary(args.output_root.resolve())
    planned_estimate = build_planned_workload_estimate(all_specs, tasks, args, smoke_summary, expected_generalization=50)
    (out_root / "planned_workload_estimate.json").write_text(json.dumps(to_jsonable(planned_estimate), indent=2), encoding="utf-8")
    print(f"[preflight] dataset counts ok: each task has train=1 test=1 generalization=50", flush=True)
    print(f"[preflight] planned estimate written to {out_root / 'planned_workload_estimate.json'}", flush=True)

    run_cfg = {
        "run_name": run_name,
        "tasks": tasks,
        "project_root": str(PROJECT_ROOT),
        "generalization_root": str(args.generalization_root.resolve()),
        "output_root": str(out_root),
        "device": str(device),
        "seed": int(args.seed),
        "smoke": bool(args.smoke),
        "started_utc": now_stamp(),
        "defaults": {task: asdict(DEFAULTS[task]) for task in tasks},
        "baseline_checkpoints": baseline_checkpoint_manifest(),
        "preflight_dataset_counts": dataset_count_rows,
        "planned_workload_estimate": planned_estimate,
        "training_data_mode": args.training_data_mode,
        "remat_options": {
            "burgers_solver_remat": args.burgers_solver_remat,
            "burgers_solver_remat_chunk_steps": args.burgers_solver_remat_chunk_steps,
            "ns2d_solver_remat": args.ns2d_solver_remat,
            "ns2d_solver_remat_chunk_steps": args.ns2d_solver_remat_chunk_steps,
            "darcy_note": "Darcy/C-flow uses a JAX CG solver with implicit differentiation; this path is tuned by batch and optimizer microbatch rather than a time-rollout remat mode.",
        },
        "note": (
            "Online adversarial training. The default label mode is solver-label: "
            "the attack objective uses solver(x_adv), then training uses either "
            "model(x_adv) -> solver(x_adv) for --training-data-mode adv-only, "
            "or both model(x_clean) -> solver(x_clean) and model(x_adv) -> solver(x_adv) "
            "for --training-data-mode clean-plus-adv. For NS2D the attack variable is the "
            "initial vorticity frame and both model input frames and target frames "
            "are regenerated by solver rollout from the attacked initial state. "
            "The old clean-label x_adv -> y_clean objective requires the explicit "
            "--allow-clean-label ablation flag."
        ),
    }
    (out_root / "run_config.json").write_text(json.dumps(to_jsonable(run_cfg), indent=2), encoding="utf-8")

    summaries = []
    wall_start = time.perf_counter()
    for task in tasks:
        print(f"[run] task={task}", flush=True)
        summaries.append(train_one_task(task, all_specs, device, out_root, args, run_cfg))
    total_wall = time.perf_counter() - wall_start
    final_summary = {
        "run_dir": str(out_root.relative_to(PROJECT_ROOT)),
        "tasks": summaries,
        "total_wall_seconds": total_wall,
        "total_wall_minutes": total_wall / 60.0,
        "finished_utc": now_stamp(),
    }
    (out_root / "summary.json").write_text(json.dumps(to_jsonable(final_summary), indent=2), encoding="utf-8")

    readme_lines = [
        "# Adversarial Training Run",
        "",
        f"- Run directory: `{out_root.relative_to(PROJECT_ROOT)}`",
        f"- Tasks: `{','.join(tasks)}`",
        f"- Smoke: `{bool(args.smoke)}`",
        f"- Label mode: `{args.label_mode}`",
        f"- Training data mode: `{args.training_data_mode}`",
        "",
        "Each task subdirectory contains `train_steps.csv`, `attack_batches.csv`, `optimizer_steps.csv`, `eval_metrics.csv`, `memory.csv`, checkpoints, `data_range_summary.json`, and `summary.json`.",
        "Training data modes: `adv-only` uses only attacked solver pairs; `clean-plus-adv` trains each batch on clean solver pairs plus newly attacked solver pairs, doubling the training examples per attack batch.",
        "Default training now uses the full original train split for every epoch; pass `--<task>-train-max N` only for debugging caps, or `0` for full.",
        "",
        "Default attack policy:",
        "- Burgers: short L-infinity fast-replace attack; default attack batch covers the full 1350-sample train split, optimizer microbatch is 300, with per-sample epsilon jitter and alpha=epsilon*ratio.",
        "- Darcy: binary steepest-replace flips coefficient pixels; default attack batch is 384 because 512/1200 OOM on the 31.7GB GPU, optimizer microbatch is 80, with jittered flip budgets.",
        "- NS2D: L-infinity add attack on the initial vorticity frame; attack batch and optimizer batch stay 1:1 by default.",
        "",
        "Evaluation is clean evaluation on train/test/generated datasets at baseline and at scheduled progress fractions.",
    ]
    (out_root / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
    print(f"[done] wrote {out_root}", flush=True)


if __name__ == "__main__":
    main()
