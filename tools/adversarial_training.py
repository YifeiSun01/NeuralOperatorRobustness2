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
from dataclasses import asdict, dataclass, field
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
    build_combined_eval_cache,
    build_specs,
    checkpoint_state,
    evaluate_dataset,
    evaluate_combined_eval_cache,
    evaluate_datasets_combined,
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
    alpha_jitter_low: float
    alpha_jitter_high: float
    learning_rate: float
    weight_decay: float
    random_start_fraction: float
    eps_jitter_low: float
    eps_jitter_high: float


DEFAULTS: dict[str, TaskDefaults] = {
    "burgers": TaskDefaults(
        # Exact solver-gradient attack is memory bound; batch 256 is tested safe on the 31.7GB GPU.
        # Burgers defaults to the p=2 replace geometry used by the older Loss3 p2/q2 sweeps.
        batch_size=256,
        optimizer_batch_size=32,
        eval_batch_size=512,
        epochs=1000,
        train_max_samples=None,
        attack_method="fast_replace_l2",
        attack_steps=3,
        epsilon_fraction=0.06,
        epsilon_abs=0.0,
        alpha_ratio=1.0,
        alpha_jitter_low=1.0,
        alpha_jitter_high=1.0,
        learning_rate=2e-4,
        weight_decay=1e-5,
        random_start_fraction=0.0,
        eps_jitter_low=0.5,
        eps_jitter_high=2.5,
    ),
    "darcy": TaskDefaults(
        # Exact solver-gradient attack is memory bound; batch 256 is tested safe on the 31.7GB GPU.
        batch_size=256,
        optimizer_batch_size=32,
        eval_batch_size=512,
        epochs=1000,
        train_max_samples=None,
        attack_method="binary_steepest_replace",
        attack_steps=1,
        epsilon_fraction=0.025,
        epsilon_abs=0.0,
        alpha_ratio=1.0,
        alpha_jitter_low=1.0,
        alpha_jitter_high=1.0,
        learning_rate=1e-4,
        weight_decay=1e-5,
        random_start_fraction=0.0,
        eps_jitter_low=1.0,
        eps_jitter_high=1.0,
    ),
    "ns2d": TaskDefaults(
        # Exact NS2D solver-gradient rollout is very memory heavy. Keep the
        # generic default conservative; production launchers should pass an
        # explicit tuned batch. On the 32GB V100 path, 2026-05-30
        # chunk-remat adversarial-training records found batch 6 passed and
        # batch 8/10 OOMed. Larger A100/B200 batch records are not comparable.
        batch_size=1,
        optimizer_batch_size=1,
        eval_batch_size=5,
        epochs=1000,
        train_max_samples=None,
        attack_method="fast_add_linf",
        attack_steps=5,
        epsilon_fraction=0.035,
        epsilon_abs=0.0,
        alpha_ratio=0.45,
        alpha_jitter_low=1.0,
        alpha_jitter_high=1.0,
        learning_rate=5e-5,
        weight_decay=1e-5,
        random_start_fraction=0.0,
        eps_jitter_low=1.0,
        eps_jitter_high=1.0,
    ),
}


TASK_SEED_OFFSETS = {"burgers": 17, "darcy": 31, "ns2d": 47}

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


def project_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


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


def load_model(task: str, device: torch.device, model_checkpoint_override: Path | None = None):
    if task == "burgers":
        return load_burgers_model(device)
    if task == "darcy":
        if model_checkpoint_override is not None:
            mod = load_module("darcy_fno2d_advtrain_override", PROJECT_ROOT / "2D_Darcy_FNO2d" / "models" / "FNO2d.py")
            model = mod.FNO2d(modes1=64, modes2=64, width=60, num_layers=4, in_channels=1, out_channels=1, padding=0).to(device)
            model.load_state_dict(checkpoint_state(model_checkpoint_override.resolve()), strict=True)
            model.eval()
            return model
        return load_darcy_model(device)
    if task == "ns2d":
        return load_ns2d_model(device)
    raise ValueError(task)


def apply_dataset_path_overrides(specs: list[DatasetSpec], args: argparse.Namespace) -> list[DatasetSpec]:
    darcy_train_path = getattr(args, "darcy_train_path", None)
    darcy_test_path = getattr(args, "darcy_test_path", None)
    if darcy_train_path is None and darcy_test_path is None:
        return specs
    out: list[DatasetSpec] = []
    for spec in specs:
        if spec.task == "darcy" and spec.split == "train" and darcy_train_path is not None:
            out.append(
                DatasetSpec(
                    spec.task,
                    "train_screen_binary_grf_alpha2_tau3_n384",
                    spec.split,
                    darcy_train_path.resolve(),
                    "override_screening",
                    spec.manual_tier,
                    spec.manual_rank,
                )
            )
        elif spec.task == "darcy" and spec.split == "test" and darcy_test_path is not None:
            out.append(
                DatasetSpec(
                    spec.task,
                    "test_screen_binary_grf_alpha2_tau3_n96",
                    spec.split,
                    darcy_test_path.resolve(),
                    "override_screening",
                    spec.manual_tier,
                    spec.manual_rank,
                )
            )
        else:
            out.append(spec)
    return out


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


def per_sample_active_count(mask: torch.Tensor) -> torch.Tensor:
    return mask.detach().reshape(mask.shape[0], -1).sum(dim=1).clamp_min(1.0)


def per_sample_l2_total(delta: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    flat = (delta * mask).reshape(delta.shape[0], -1)
    return torch.sqrt(flat.pow(2).sum(dim=1).clamp_min(1e-24))


def rms_eps_to_l2_total(eps_rms: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return eps_rms * torch.sqrt(per_sample_active_count(mask).to(device=eps_rms.device, dtype=eps_rms.dtype))


def l2_rms_project(delta: torch.Tensor, mask: torch.Tensor, eps_rms: torch.Tensor) -> torch.Tensor:
    delta = delta * mask
    norm = per_sample_l2_total(delta, mask)
    max_norm = rms_eps_to_l2_total(eps_rms, mask)
    scale = torch.minimum(torch.ones_like(norm), max_norm / norm.clamp_min(1e-12))
    return delta * expand_per_sample(scale, delta)


def l2_rms_direction(direction: torch.Tensor, mask: torch.Tensor, step_rms: torch.Tensor) -> torch.Tensor:
    direction = direction * mask
    norm = per_sample_l2_total(direction, mask)
    step_norm = rms_eps_to_l2_total(step_rms, mask)
    scale = step_norm / norm.clamp_min(1e-12)
    return direction * expand_per_sample(scale, direction)


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
    low = float(jitter_low)
    high = float(jitter_high)
    if high < low:
        raise ValueError(f"epsilon jitter high must be >= low, got low={low}, high={high}")
    if low == high:
        jitter = torch.full_like(base, low)
    else:
        jitter = torch.empty_like(base).uniform_(low, high)
    return (base * jitter).clamp_min(1e-8)


def compute_alpha_values(
    eps: torch.Tensor,
    alpha_ratio: float,
    jitter_low: float,
    jitter_high: float,
) -> torch.Tensor:
    base = eps * float(alpha_ratio)
    low = float(jitter_low)
    high = float(jitter_high)
    if high < low:
        raise ValueError(f"alpha jitter high must be >= low, got low={low}, high={high}")
    if low == high:
        return (base * low).clamp_min(1e-8)
    jitter = torch.empty_like(base).uniform_(low, high)
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
    sample_info: dict[str, torch.Tensor] = field(default_factory=dict)
    probe_tensors: dict[str, torch.Tensor] = field(default_factory=dict)


def make_attack_sample_info(
    *,
    epsilon: torch.Tensor,
    epsilon_jitter_factor: torch.Tensor,
    alpha: torch.Tensor,
    clean_loss: torch.Tensor,
    adv_loss: torch.Tensor,
) -> dict[str, torch.Tensor]:
    epsilon_cpu = epsilon.detach().reshape(-1).float().cpu()
    jitter_cpu = epsilon_jitter_factor.detach().reshape(-1).float().cpu()
    alpha_cpu = alpha.detach().reshape(-1).float().cpu()
    clean_cpu = clean_loss.detach().reshape(-1).float().cpu()
    adv_cpu = adv_loss.detach().reshape(-1).float().cpu()
    gain_cpu = adv_cpu - clean_cpu
    relative_cpu = torch.where(
        clean_cpu.abs() > 1e-20,
        gain_cpu / clean_cpu.abs().clamp_min(1e-20),
        torch.full_like(gain_cpu, float("nan")),
    )
    return {
        "epsilon": epsilon_cpu,
        "epsilon_jitter_factor": jitter_cpu,
        "alpha": alpha_cpu,
        "clean_loss_before_attack": clean_cpu,
        "adv_loss_after_attack": adv_cpu,
        "attack_loss_gain": gain_cpu,
        "attack_loss_gain_relative": relative_cpu,
    }


def parse_int_list(value: str | None) -> list[int] | None:
    if value is None or not str(value).strip():
        return None
    out: list[int] = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def parse_float_list(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        parts = value
    else:
        text = str(value).strip()
        if not text:
            return []
        parts = text.split(",")
    out: list[float] = []
    for part in parts:
        if part is None:
            continue
        text = str(part).strip()
        if not text:
            continue
        out.append(float(text))
    return out


def wall_clock_checkpoint_targets(seconds_value: Any, hours_value: Any) -> list[float]:
    targets = parse_float_list(seconds_value)
    targets.extend(3600.0 * hours for hours in parse_float_list(hours_value))
    return sorted({float(target) for target in targets if float(target) > 0})


def wall_checkpoint_suffix(target_seconds: float) -> str:
    return f"wall_{int(round(float(target_seconds))):07d}s"


def select_attack_probe_indices(n_train: int, requested_count: int, explicit: str | None) -> np.ndarray:
    explicit_indices = parse_int_list(explicit)
    if explicit_indices is not None:
        valid = sorted({idx for idx in explicit_indices if 0 <= idx < n_train})
        return np.asarray(valid, dtype=np.int64)
    count = min(max(0, int(requested_count)), max(0, n_train))
    if count <= 0:
        return np.asarray([], dtype=np.int64)
    if count >= n_train:
        return np.arange(n_train, dtype=np.int64)
    return np.unique(np.rint(np.linspace(0, n_train - 1, count)).astype(np.int64))


def tensor_to_probe_array(x: torch.Tensor) -> np.ndarray:
    return x.detach().float().cpu().numpy()


def _spectral_delta_stats(delta: np.ndarray) -> dict[str, float]:
    arr = np.nan_to_num(np.asarray(delta, dtype=np.float64), copy=False)
    while arr.ndim > 1 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.size <= 1 or arr.ndim == 0:
        return {
            "delta_total_variation": 0.0,
            "delta_sign_change_fraction": 0.0,
            "delta_fft_high_freq_ratio": 0.0,
            "delta_fft_spectral_centroid": 0.0,
        }

    spatial_axes = (0,) if arr.ndim == 1 else (0, 1)
    total_variation = 0.0
    sign_change_terms: list[float] = []
    for axis in spatial_axes:
        if arr.shape[axis] <= 1:
            continue
        diff = np.diff(arr, axis=axis)
        total_variation += float(np.mean(np.abs(diff)))
        left = np.take(arr, indices=range(0, arr.shape[axis] - 1), axis=axis)
        right = np.take(arr, indices=range(1, arr.shape[axis]), axis=axis)
        valid = (np.abs(left) > 1e-12) & (np.abs(right) > 1e-12)
        if np.any(valid):
            sign_change_terms.append(float(np.mean((left[valid] * right[valid]) < 0)))

    centered = arr - np.mean(arr, axis=spatial_axes, keepdims=True)
    fft = np.fft.fftn(centered, axes=spatial_axes)
    power = np.abs(fft) ** 2
    total_power = float(np.sum(power))
    if total_power <= 1e-30:
        high_ratio = 0.0
        centroid = 0.0
    else:
        freq_grids = np.meshgrid(
            *[np.fft.fftfreq(arr.shape[axis]) for axis in spatial_axes],
            indexing="ij",
        )
        radial = np.sqrt(sum(grid ** 2 for grid in freq_grids))
        radial_norm = radial / max(1e-12, math.sqrt(len(spatial_axes)) * 0.5)
        if arr.ndim > len(spatial_axes):
            radial_norm = radial_norm.reshape(radial_norm.shape + (1,) * (arr.ndim - len(spatial_axes)))
        high_mask = radial_norm >= 0.5
        high_ratio = float(np.sum(power * high_mask) / total_power)
        centroid = float(np.sum(power * radial_norm) / total_power)

    return {
        "delta_total_variation": total_variation,
        "delta_sign_change_fraction": float(np.mean(sign_change_terms)) if sign_change_terms else 0.0,
        "delta_fft_high_freq_ratio": high_ratio,
        "delta_fft_spectral_centroid": centroid,
    }


def attack_probe_delta_stats(delta: np.ndarray) -> dict[str, float]:
    flat = np.nan_to_num(np.asarray(delta, dtype=np.float64).reshape(-1), copy=False)
    if flat.size == 0:
        return {
            "delta_linf": float("nan"),
            "delta_l2_rms": float("nan"),
            "delta_abs_mean": float("nan"),
            "delta_mean": float("nan"),
            "delta_std": float("nan"),
            **_spectral_delta_stats(delta),
        }
    return {
        "delta_linf": float(np.max(np.abs(flat))),
        "delta_l2_rms": float(np.sqrt(np.mean(flat ** 2))),
        "delta_abs_mean": float(np.mean(np.abs(flat))),
        "delta_mean": float(np.mean(flat)),
        "delta_std": float(np.std(flat)),
        **_spectral_delta_stats(delta),
    }


def per_sample_finite_mse(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    diff = pred - y
    finite = torch.isfinite(diff)
    flat_diff = torch.nan_to_num(diff).reshape(diff.shape[0], -1)
    flat_finite = finite.reshape(finite.shape[0], -1)
    numerator = (flat_diff.pow(2) * flat_finite.float()).sum(dim=1)
    denominator = flat_finite.float().sum(dim=1).clamp_min(1.0)
    return numerator / denominator


def selected_clean_pair_for_probe(
    task: str,
    xb_selected: torch.Tensor,
    yb_selected: torch.Tensor,
    cfg: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    # Same clean solver-label convention as optimizer training, so the
    # per-probe gain is clean model-vs-solver loss -> attacked model-vs-solver loss.
    return clean_solver_training_pair(task, xb_selected, yb_selected, cfg)


def collect_attack_probe_records(
    *,
    model,
    task: str,
    epoch: int,
    global_step: int,
    local_batch_idx: int,
    batch_indices: torch.Tensor,
    xb: torch.Tensor,
    yb: torch.Tensor,
    attack_result: AttackBatchResult,
    cfg: dict[str, Any],
    probe_index_to_rank: dict[int, int],
    save_targets: bool,
) -> list[dict[str, Any]]:
    if not probe_index_to_rank:
        return []
    records: list[dict[str, Any]] = []
    source_indices = [int(x) for x in batch_indices.detach().cpu().tolist()]
    selected_positions: list[int] = []
    for batch_pos, source_index in enumerate(source_indices):
        probe_rank = probe_index_to_rank.get(source_index)
        if probe_rank is None:
            continue
        selected_positions.append(batch_pos)

    clean_loss_by_pos: dict[int, float] = {}
    adv_loss_by_pos: dict[int, float] = {}
    if selected_positions:
        pos_tensor = torch.as_tensor(selected_positions, device=xb.device, dtype=torch.long)
        was_training = model.training
        model.eval()
        with torch.no_grad():
            xb_sel = xb.index_select(0, pos_tensor)
            yb_sel = yb.index_select(0, pos_tensor)
            x_adv_loss = attack_result.x_train.index_select(0, pos_tensor)
            y_adv_loss = attack_result.y_train.index_select(0, pos_tensor)
            clean_tensor, adv_tensor = attack_objective_losses_for_probe(model, task, xb_sel, yb_sel, x_adv_loss, y_adv_loss, cfg)
            clean_losses = clean_tensor.detach().cpu().tolist()
            adv_losses = adv_tensor.detach().cpu().tolist()
        if was_training:
            model.train()
        for batch_pos, clean_loss, adv_loss in zip(selected_positions, clean_losses, adv_losses):
            clean_loss_by_pos[int(batch_pos)] = float(clean_loss)
            adv_loss_by_pos[int(batch_pos)] = float(adv_loss)

    for batch_pos, source_index in enumerate(source_indices):
        probe_rank = probe_index_to_rank.get(source_index)
        if probe_rank is None:
            continue
        x_clean = tensor_to_probe_array(xb[batch_pos])
        x_adv = tensor_to_probe_array(attack_result.x_train[batch_pos])
        delta = x_adv - x_clean
        clean_loss_sample = clean_loss_by_pos.get(batch_pos, float("nan"))
        adv_loss_sample = adv_loss_by_pos.get(batch_pos, float("nan"))
        record: dict[str, Any] = {
            "task": task,
            "epoch": int(epoch),
            "attack_global_step": int(global_step),
            "local_batch_idx": int(local_batch_idx),
            "probe_rank": int(probe_rank),
            "source_index": int(source_index),
            "x_clean": x_clean,
            "x_adv": x_adv,
            "delta": delta,
            "clean_loss_before_attack_sample": clean_loss_sample,
            "adv_loss_after_attack_sample": adv_loss_sample,
            "attack_loss_gain_sample": adv_loss_sample - clean_loss_sample,
            "attack_loss_gain_relative_sample": ((adv_loss_sample - clean_loss_sample) / abs(clean_loss_sample)) if math.isfinite(clean_loss_sample) and abs(clean_loss_sample) > 1e-20 else float("nan"),
            **attack_probe_delta_stats(delta),
        }
        for extra_key, extra_tensor in attack_result.probe_tensors.items():
            if not isinstance(extra_tensor, torch.Tensor) or int(extra_tensor.shape[0]) != len(source_indices):
                continue
            record[extra_key] = tensor_to_probe_array(extra_tensor[batch_pos])
        for key in (
            "attack_type",
            "attack_method",
            "attack_loss_objective",
            "attack_objective_definition",
            "attack_target_source",
            "attack_uses_solver_forward",
            "attack_uses_solver_backward",
            "training_target_source",
            "attack_steps",
            "epsilon_mean",
            "epsilon_min",
            "epsilon_max",
            "alpha_mean",
            "alpha_min",
            "alpha_max",
            "clean_loss_before_attack",
            "adv_loss_after_attack",
            "attack_loss_gain",
        ):
            if key in attack_result.info:
                record[key] = attack_result.info[key]
        if save_targets:
            record["y_clean"] = tensor_to_probe_array(yb[batch_pos])
            record["y_adv"] = tensor_to_probe_array(attack_result.y_train[batch_pos])
        records.append(record)
    return records


def save_attack_probe_epoch(
    out_dir: Path,
    task: str,
    epoch: int,
    epoch_end_global_step: int,
    probe_indices: np.ndarray,
    records: list[dict[str, Any]],
    *,
    save_targets: bool,
) -> Path | None:
    probe_dir = out_dir / "attack_probe_samples"
    probe_dir.mkdir(parents=True, exist_ok=True)
    records = sorted(records, key=lambda row: int(row["probe_rank"]))
    captured = {int(row["source_index"]) for row in records}
    missing = [int(idx) for idx in probe_indices.tolist() if int(idx) not in captured]

    npz_path: Path | None = None
    rel_npz = ""
    if records:
        npz_path = probe_dir / f"{task}_epoch{epoch:03d}_step{epoch_end_global_step:06d}_attack_probe.npz"
        payload: dict[str, Any] = {
            "task": np.asarray(task),
            "epoch": np.asarray(int(epoch), dtype=np.int64),
            "epoch_end_global_step": np.asarray(int(epoch_end_global_step), dtype=np.int64),
            "probe_rank": np.asarray([int(row["probe_rank"]) for row in records], dtype=np.int64),
            "source_index": np.asarray([int(row["source_index"]) for row in records], dtype=np.int64),
            "attack_global_step": np.asarray([int(row["attack_global_step"]) for row in records], dtype=np.int64),
            "local_batch_idx": np.asarray([int(row["local_batch_idx"]) for row in records], dtype=np.int64),
            "x_clean": np.stack([row["x_clean"] for row in records]).astype(np.float32, copy=False),
            "x_adv": np.stack([row["x_adv"] for row in records]).astype(np.float32, copy=False),
            "delta": np.stack([row["delta"] for row in records]).astype(np.float32, copy=False),
        }
        array_keys = sorted(
            {
                key
                for row in records
                for key, value in row.items()
                if isinstance(value, np.ndarray) and key not in payload
            }
        )
        for key in array_keys:
            if all(key in row and isinstance(row[key], np.ndarray) for row in records):
                payload[key] = np.stack([row[key] for row in records]).astype(np.float32, copy=False)
        if save_targets and all("y_clean" in row and "y_adv" in row for row in records):
            payload["y_clean"] = np.stack([row["y_clean"] for row in records]).astype(np.float32, copy=False)
            payload["y_adv"] = np.stack([row["y_adv"] for row in records]).astype(np.float32, copy=False)
        np.savez_compressed(npz_path, **payload)
        try:
            rel_npz = project_path(npz_path)
        except ValueError:
            rel_npz = str(npz_path)

    write_csv_row(
        out_dir / "attack_probe_epochs.csv",
        {
            "task": task,
            "epoch": int(epoch),
            "epoch_end_global_step": int(epoch_end_global_step),
            "expected_probe_count": int(len(probe_indices)),
            "captured_probe_count": int(len(records)),
            "missing_source_indices": json.dumps(missing),
            "npz_path": rel_npz,
        },
    )
    for row in records:
        csv_row = {
            key: value
            for key, value in row.items()
            if not isinstance(value, np.ndarray)
        }
        csv_row["epoch_end_global_step"] = int(epoch_end_global_step)
        csv_row["npz_path"] = rel_npz
        write_csv_row(out_dir / "attack_probe_samples.csv", csv_row)
    return npz_path


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


def parse_csv_floats(value: Any, default: list[float]) -> list[float]:
    if value is None:
        return list(default)
    if isinstance(value, (list, tuple)):
        out = [float(x) for x in value]
    else:
        out = [float(x.strip()) for x in str(value).split(",") if x.strip()]
    return out or list(default)


def parse_csv_strings(value: Any, default: list[str]) -> list[str]:
    if value is None:
        return list(default)
    if isinstance(value, (list, tuple)):
        out = [str(x).strip().lower() for x in value if str(x).strip()]
    else:
        out = [x.strip().lower() for x in str(value).split(",") if x.strip()]
    return out or list(default)


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



def burgers_attack_loss_objective(cfg: dict[str, Any]) -> str:
    objective = str(cfg.get("attack_loss_objective", "loss3")).lower().strip()
    aliases = {
        "1": "loss1",
        "l1": "loss1",
        "loss_1": "loss1",
        "2": "loss2",
        "l2": "loss2",
        "loss_2": "loss2",
        "3": "loss3",
        "l3": "loss3",
        "loss_3": "loss3",
    }
    objective = aliases.get(objective, objective)
    if objective not in {"loss1", "loss2", "loss3"}:
        raise ValueError(f"unknown Burgers attack_loss_objective={objective!r}; expected loss1/loss2/loss3")
    return objective


def burgers_attack_loss_metadata(objective: str) -> dict[str, Any]:
    if objective == "loss1":
        return {
            "attack_loss_objective": "loss1",
            "attack_objective_definition": "MSE(model(x_adv), model(x_clean).detach())",
            "attack_uses_solver_forward": 0,
            "attack_uses_solver_backward": 0,
            "full_solver_gradient": False,
            "attack_target_source": "fixed clean model output; no solver in attack",
        }
    if objective == "loss2":
        return {
            "attack_loss_objective": "loss2",
            "attack_objective_definition": "MSE(model(x_adv), solver(x_clean).detach())",
            "attack_uses_solver_forward": 1,
            "attack_uses_solver_backward": 0,
            "full_solver_gradient": False,
            "attack_target_source": "fixed clean solver output; solver forward only in attack",
        }
    return {
        "attack_loss_objective": "loss3",
        "attack_objective_definition": "MSE(model(x_adv), solver(x_adv))",
        "attack_uses_solver_forward": 1,
        "attack_uses_solver_backward": 1,
        "full_solver_gradient": True,
        "attack_target_source": "attacked solver output; solver forward and backward in attack",
    }


def darcy_attack_loss_objective(cfg: dict[str, Any]) -> str:
    objective = str(cfg.get("attack_loss_objective", "loss3")).lower().strip()
    aliases = {
        "1": "loss1",
        "l1": "loss1",
        "loss_1": "loss1",
        "2": "loss2",
        "l2": "loss2",
        "loss_2": "loss2",
        "3": "loss3",
        "l3": "loss3",
        "loss_3": "loss3",
        "4": "physics",
        "l4": "physics",
        "loss4": "physics",
        "loss_4": "physics",
        "loss4_physics": "physics",
        "physics_loss": "physics",
        "pde": "physics",
    }
    objective = aliases.get(objective, objective)
    if objective not in {"loss1", "loss2", "loss3", "physics"}:
        raise ValueError(
            f"unknown Darcy attack_loss_objective={objective!r}; expected loss1/loss2/loss3/physics"
        )
    return objective


def darcy_attack_loss_metadata(objective: str, cfg: dict[str, Any]) -> dict[str, Any]:
    if objective == "loss1":
        return {
            "attack_loss_objective": "loss1",
            "attack_objective_definition": "MSE(model(a_adv), model(a_clean).detach())",
            "attack_uses_solver_forward": 0,
            "attack_uses_solver_backward": 0,
            "full_solver_gradient": False,
            "attack_target_source": "fixed clean model output; no solver in attack",
        }
    if objective == "loss2":
        return {
            "attack_loss_objective": "loss2",
            "attack_objective_definition": "MSE(model(a_adv), solver(a_clean).detach())",
            "attack_uses_solver_forward": 1,
            "attack_uses_solver_backward": 0,
            "full_solver_gradient": False,
            "attack_target_source": "fixed clean solver output; solver forward only in attack",
        }
    if objective == "physics":
        metric = str(cfg.get("darcy_physics_metric", "rel_l2"))
        bc_weight = float(cfg.get("darcy_physics_bc_weight", 1.0))
        return {
            "attack_loss_objective": "physics",
            "attack_objective_definition": (
                f"Darcy physics residual + {bc_weight:g} * boundary penalty, metric={metric}"
            ),
            "attack_uses_solver_forward": 0,
            "attack_uses_solver_backward": 0,
            "full_solver_gradient": False,
            "attack_target_source": "PDE residual of model(a_adv); no numerical solver in attack",
            "darcy_physics_metric": metric,
            "darcy_physics_bc_weight": bc_weight,
        }
    return {
        "attack_loss_objective": "loss3",
        "attack_objective_definition": "MSE(model(a_adv), solver(a_adv))",
        "attack_uses_solver_forward": 1,
        "attack_uses_solver_backward": int(bool(solver_target_grad_enabled(cfg))),
        "full_solver_gradient": bool(solver_target_grad_enabled(cfg)),
        "attack_target_source": "attacked solver output; solver forward/backward in attack",
    }


def darcy_spatial_field(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 4 and x.shape[-1] == 1:
        return x[..., 0]
    if x.ndim == 3:
        return x
    raise ValueError(f"Darcy field expects shape (B,H,W) or (B,H,W,1), got {tuple(x.shape)}")


def darcy_matvec_torch(a: torch.Tensor, u_full: torch.Tensor) -> torch.Tensor:
    """Apply the finite-difference Darcy operator used by the JAX solver."""
    a = darcy_spatial_field(a)
    u_full = darcy_spatial_field(u_full)
    n = int(a.shape[-1])
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


def darcy_boundary_values(u: torch.Tensor) -> torch.Tensor:
    u = darcy_spatial_field(u)
    return torch.cat([u[:, 0, :], u[:, -1, :], u[:, 1:-1, 0], u[:, 1:-1, -1]], dim=1)


def darcy_per_sample_physics_components(
    a: torch.Tensor,
    pred_u: torch.Tensor,
    *,
    forcing_value: float = 1.0,
    physics_metric: str = "rel_l2",
) -> tuple[torch.Tensor, torch.Tensor]:
    op_u = darcy_matvec_torch(a, pred_u)
    rhs = torch.ones_like(op_u) * float(forcing_value)
    residual = op_u - rhs
    residual_flat = residual.reshape(residual.shape[0], -1)
    if physics_metric == "rel_l2":
        rhs_flat = rhs.reshape(rhs.shape[0], -1)
        pde_loss = torch.linalg.vector_norm(residual_flat, ord=2, dim=1) / torch.linalg.vector_norm(
            rhs_flat, ord=2, dim=1
        ).clamp_min(1e-12)
    elif physics_metric == "mse":
        pde_loss = torch.mean(residual_flat.square(), dim=1)
    else:
        raise ValueError(f"unknown Darcy physics metric {physics_metric!r}; expected rel_l2/mse")

    boundary = darcy_boundary_values(pred_u)
    if physics_metric == "rel_l2":
        bc_loss = torch.linalg.vector_norm(boundary, ord=2, dim=1) / math.sqrt(max(1, boundary.shape[1]))
    else:
        bc_loss = torch.mean(boundary.square(), dim=1)
    return pde_loss, bc_loss


def darcy_per_sample_physics_loss(a: torch.Tensor, pred_u: torch.Tensor, cfg: dict[str, Any]) -> torch.Tensor:
    pde_loss, bc_loss = darcy_per_sample_physics_components(
        a,
        pred_u,
        forcing_value=float(cfg.get("darcy_physics_forcing_value", 1.0)),
        physics_metric=str(cfg.get("darcy_physics_metric", "rel_l2")),
    )
    return pde_loss + float(cfg.get("darcy_physics_bc_weight", 1.0)) * bc_loss


def darcy_attack_objective_loss(
    model,
    x_clean: torch.Tensor,
    x_adv: torch.Tensor,
    y_clean: torch.Tensor,
    cfg: dict[str, Any],
    *,
    allow_solver_backward: bool,
) -> tuple[torch.Tensor, torch.Tensor, str, dict[str, Any]]:
    objective = darcy_attack_loss_objective(cfg)
    pred_adv = model(x_adv)
    if objective == "loss1":
        with torch.no_grad():
            target = model(x_clean).detach()
        losses = per_sample_finite_mse(pred_adv, target)
        return losses.mean(), losses, objective, darcy_attack_loss_metadata(objective, cfg)
    if objective == "loss2":
        target = solver_target_for_model_input("darcy", x_clean, y_clean, cfg, allow_target_grad=False).detach()
        losses = per_sample_finite_mse(pred_adv, target)
        return losses.mean(), losses, objective, darcy_attack_loss_metadata(objective, cfg)
    if objective == "physics":
        losses = darcy_per_sample_physics_loss(x_adv, pred_adv, cfg)
        return losses.mean(), losses, objective, darcy_attack_loss_metadata(objective, cfg)

    target = solver_target_for_model_input(
        "darcy",
        x_adv,
        y_clean,
        cfg,
        allow_target_grad=allow_solver_backward and solver_target_grad_enabled(cfg),
    )
    losses = per_sample_finite_mse(pred_adv, target)
    return losses.mean(), losses, objective, darcy_attack_loss_metadata(objective, cfg)


def attack_objective_target(
    model,
    task: str,
    xb: torch.Tensor,
    x_adv: torch.Tensor,
    yb: torch.Tensor,
    cfg: dict[str, Any],
    *,
    allow_solver_backward: bool,
) -> tuple[torch.Tensor, str, dict[str, Any]]:
    """Return the target used only for the adversarial attack objective.

    For Burgers this implements the three historical objectives:
    loss1: no solver; fixed clean model output.
    loss2: solver forward at clean x only; target detached, no solver backward.
    loss3: solver forward at attacked x; solver gradient participates.

    The optimizer training target is intentionally handled separately below by
    solver_target_for_model_input(..., x_adv, allow_target_grad=False), because
    these objectives are attack-generation choices.
    """
    if task == "darcy":
        objective = darcy_attack_loss_objective(cfg)
        if objective == "loss1":
            with torch.no_grad():
                target = model(xb).detach()
            return target, objective, darcy_attack_loss_metadata(objective, cfg)
        if objective == "loss2":
            target = solver_target_for_model_input(task, xb, yb, cfg, allow_target_grad=False).detach()
            return target, objective, darcy_attack_loss_metadata(objective, cfg)
        if objective == "physics":
            raise ValueError("Darcy physics attack objective is not target-MSE based")

    if task != "burgers":
        target = solver_target_for_model_input(
            task,
            x_adv,
            yb,
            cfg,
            allow_target_grad=allow_solver_backward and solver_target_grad_enabled(cfg),
        )
        return target, "loss3", {
            "attack_loss_objective": "loss3",
            "attack_objective_definition": "MSE(model(x_adv), solver(x_adv))",
            "attack_uses_solver_forward": 1,
            "attack_uses_solver_backward": int(bool(allow_solver_backward and solver_target_grad_enabled(cfg))),
            "full_solver_gradient": bool(allow_solver_backward and solver_target_grad_enabled(cfg)),
            "attack_target_source": "attacked solver output",
        }

    objective = burgers_attack_loss_objective(cfg)
    if objective == "loss1":
        with torch.no_grad():
            target = model(xb).detach()
        return target, objective, burgers_attack_loss_metadata(objective)
    if objective == "loss2":
        target = solver_target_for_model_input(task, xb, yb, cfg, allow_target_grad=False).detach()
        return target, objective, burgers_attack_loss_metadata(objective)

    target = solver_target_for_model_input(
        task,
        x_adv,
        yb,
        cfg,
        allow_target_grad=allow_solver_backward and solver_target_grad_enabled(cfg),
    )
    return target, objective, burgers_attack_loss_metadata(objective)


def attack_objective_losses_for_probe(
    model,
    task: str,
    xb_selected: torch.Tensor,
    yb_selected: torch.Tensor,
    x_adv_selected: torch.Tensor,
    y_adv_training: torch.Tensor,
    cfg: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sample clean/attacked losses using the configured attack objective."""
    if str(cfg.get("training_perturbation_mode", "attack")) == "random-field":
        target_mode = random_field_target_mode(cfg)
        if task != "burgers":
            raise ValueError("random-field probe losses are currently implemented for Burgers only")
        if target_mode == "clean-y":
            clean_target = yb_selected.detach()
            adv_target = y_adv_training.detach()
        elif target_mode == "solver-y":
            clean_target = solver_target_for_model_input(task, xb_selected, yb_selected, cfg, allow_target_grad=False).detach()
            adv_target = y_adv_training.detach()
        else:
            raise ValueError(f"unknown random_field_target_mode={target_mode!r}")
        clean_losses = per_sample_finite_mse(model(xb_selected), clean_target)
        adv_losses = per_sample_finite_mse(model(x_adv_selected), adv_target)
        return clean_losses, adv_losses

    if task == "darcy":
        if is_darcy_random_source_mode(cfg):
            clean_pred = model(xb_selected)
            adv_pred = model(x_adv_selected)
            clean_losses = per_sample_finite_mse(clean_pred, yb_selected)
            adv_target = yb_selected if str(cfg.get("training_data_mode")) == "random-binary-fixed-y" else y_adv_training
            adv_losses = per_sample_finite_mse(adv_pred, adv_target)
            return clean_losses, adv_losses
        _, clean_losses, _, _ = darcy_attack_objective_loss(
            model,
            xb_selected,
            xb_selected,
            yb_selected,
            cfg,
            allow_solver_backward=False,
        )
        _, adv_losses, _, _ = darcy_attack_objective_loss(
            model,
            xb_selected,
            x_adv_selected,
            yb_selected,
            cfg,
            allow_solver_backward=False,
        )
        return clean_losses, adv_losses

    if task == "burgers":
        objective = burgers_attack_loss_objective(cfg)
        if objective == "loss1":
            clean_pred = model(xb_selected)
            clean_target = clean_pred.detach()
            adv_pred = model(x_adv_selected)
            return per_sample_finite_mse(clean_pred, clean_target), per_sample_finite_mse(adv_pred, clean_target)
        if objective == "loss2":
            fixed_solver_target = solver_target_for_model_input(task, xb_selected, yb_selected, cfg, allow_target_grad=False).detach()
            clean_pred = model(xb_selected)
            adv_pred = model(x_adv_selected)
            return per_sample_finite_mse(clean_pred, fixed_solver_target), per_sample_finite_mse(adv_pred, fixed_solver_target)

    x_clean_loss, y_clean_loss = clean_solver_training_pair(task, xb_selected, yb_selected, cfg)
    clean_losses = per_sample_finite_mse(model(x_clean_loss), y_clean_loss)
    adv_losses = per_sample_finite_mse(model(x_adv_selected), y_adv_training)
    return clean_losses, adv_losses


def random_field_target_mode(cfg: dict[str, Any]) -> str:
    mode = str(cfg.get("random_field_target_mode", "solver-y")).lower().strip().replace("_", "-")
    aliases = {
        "clean": "clean-y",
        "clean-label": "clean-y",
        "clean-y": "clean-y",
        "fixed-y": "clean-y",
        "unchanged-y": "clean-y",
        "solver": "solver-y",
        "solver-label": "solver-y",
        "solver-y": "solver-y",
        "recomputed-y": "solver-y",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"clean-y", "solver-y"}:
        raise ValueError(f"unknown random_field_target_mode={mode!r}; expected clean-y or solver-y")
    return mode


def random_field_family_choices(cfg: dict[str, Any]) -> list[str]:
    raw = str(cfg.get("random_field_families", "gaussian,matern"))
    aliases = {
        "gauss": "gaussian",
        "rbf": "gaussian",
        "squared-exponential": "gaussian",
        "squared_exponential": "gaussian",
        "mattern": "matern",
        "martin": "matern",
    }
    families: list[str] = []
    for part in raw.split(","):
        family = aliases.get(part.strip().lower(), part.strip().lower())
        if not family:
            continue
        if family not in {"gaussian", "matern"}:
            raise ValueError(f"unknown random field family {family!r}; expected gaussian/matern")
        families.append(family)
    return families or ["gaussian", "matern"]


def _positive_float_choices(value: Any, default: list[float], *, name: str) -> list[float]:
    choices = parse_float_list(value)
    if not choices:
        choices = list(default)
    out = [float(x) for x in choices if float(x) > 0]
    if not out:
        raise ValueError(f"{name} must contain at least one positive value")
    return out


def burgers_random_field_delta(
    xb: torch.Tensor,
    eps: torch.Tensor,
    cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, Any], dict[str, torch.Tensor]]:
    if xb.ndim != 3 or xb.shape[-1] != 1:
        raise ValueError(f"Burgers random-field mode expects x shape (B,N,1), got {tuple(xb.shape)}")

    bsz = int(xb.shape[0])
    n = int(xb.shape[1])
    device = xb.device
    dtype = xb.dtype
    families = random_field_family_choices(cfg)
    gaussian_lengths = _positive_float_choices(
        cfg.get("random_field_gaussian_correlation_choices"),
        [0.015, 0.03, 0.06, 0.12, 0.24],
        name="random_field_gaussian_correlation_choices",
    )
    matern_lengths = _positive_float_choices(
        cfg.get("random_field_matern_correlation_choices"),
        [0.015, 0.03, 0.06, 0.12, 0.24],
        name="random_field_matern_correlation_choices",
    )
    matern_nus = _positive_float_choices(
        cfg.get("random_field_matern_nu_choices"),
        [1.2, 2.2, 3.2, 4.2, 5.2],
        name="random_field_matern_nu_choices",
    )
    domain_extent = float(cfg.get("random_field_domain_extent", 2.0))
    if domain_extent <= 0:
        raise ValueError("random_field_domain_extent must be positive")

    family_idx = torch.randint(len(families), (bsz,), device=device)
    family_names = [families[int(i)] for i in family_idx.detach().cpu().tolist()]
    gaussian_length_tensor = torch.as_tensor(gaussian_lengths, device=device, dtype=dtype)
    matern_length_tensor = torch.as_tensor(matern_lengths, device=device, dtype=dtype)
    matern_nu_tensor = torch.as_tensor(matern_nus, device=device, dtype=dtype)
    gaussian_length = gaussian_length_tensor[torch.randint(len(gaussian_lengths), (bsz,), device=device)]
    matern_length = matern_length_tensor[torch.randint(len(matern_lengths), (bsz,), device=device)]
    matern_nu = matern_nu_tensor[torch.randint(len(matern_nus), (bsz,), device=device)]

    chosen_length = torch.empty((bsz,), device=device, dtype=dtype)
    chosen_nu = torch.full((bsz,), float("nan"), device=device, dtype=dtype)
    for i, family in enumerate(families):
        mask = family_idx == i
        if family == "gaussian":
            chosen_length = torch.where(mask, gaussian_length, chosen_length)
        elif family == "matern":
            chosen_length = torch.where(mask, matern_length, chosen_length)
            chosen_nu = torch.where(mask, matern_nu, chosen_nu)

    white = torch.randn((bsz, n), device=device, dtype=dtype)
    coeff = torch.fft.rfft(white, dim=1)
    freq = torch.fft.rfftfreq(n, d=domain_extent / float(n)).to(device=device, dtype=dtype)
    omega = (2.0 * math.pi * freq).reshape(1, -1)
    length_view = chosen_length.reshape(-1, 1).clamp_min(1e-8)
    nu_view = torch.nan_to_num(chosen_nu, nan=float(matern_nus[0])).reshape(-1, 1).clamp_min(1e-6)

    gaussian_amp = torch.exp(-0.5 * (omega * length_view).pow(2))
    matern_amp = (1.0 + (omega * length_view).pow(2) / (2.0 * nu_view)).pow(-(nu_view + 0.5) * 0.5)
    amp = torch.empty_like(gaussian_amp)
    for i, family in enumerate(families):
        mask = (family_idx == i).reshape(-1, 1)
        amp = torch.where(mask, matern_amp if family == "matern" else gaussian_amp, amp)
    amp[:, 0] = 0.0
    field = torch.fft.irfft(coeff * amp.to(dtype=coeff.dtype), n=n, dim=1)
    field = field - field.mean(dim=1, keepdim=True)
    field_rms = torch.sqrt(field.pow(2).mean(dim=1, keepdim=True).clamp_min(1e-24))
    delta = field / field_rms * eps.reshape(-1, 1)
    delta = delta.unsqueeze(-1).to(dtype=dtype)

    clip_min = cfg.get("random_field_clip_x_min")
    clip_max = cfg.get("random_field_clip_x_max")
    if clip_min is not None or clip_max is not None:
        lo = -float("inf") if clip_min is None else float(clip_min)
        hi = float("inf") if clip_max is None else float(clip_max)
        x_adv = torch.clamp(xb + delta, min=lo, max=hi)
        delta = x_adv - xb

    family_counts = {f"random_field_family_{family}_count": int(family_names.count(family)) for family in sorted(set(families))}
    finite_nu = chosen_nu[torch.isfinite(chosen_nu)]
    info = {
        "random_field_families": ",".join(families),
        "random_field_domain_extent": domain_extent,
        "random_field_gaussian_correlation_choices": ",".join(f"{x:g}" for x in gaussian_lengths),
        "random_field_matern_correlation_choices": ",".join(f"{x:g}" for x in matern_lengths),
        "random_field_matern_nu_choices": ",".join(f"{x:g}" for x in matern_nus),
        "random_field_length_mean": float(chosen_length.mean().detach().cpu()),
        "random_field_length_min": float(chosen_length.min().detach().cpu()),
        "random_field_length_max": float(chosen_length.max().detach().cpu()),
        "random_field_matern_nu_mean": float(finite_nu.mean().detach().cpu()) if finite_nu.numel() else float("nan"),
        "random_field_matern_nu_min": float(finite_nu.min().detach().cpu()) if finite_nu.numel() else float("nan"),
        "random_field_matern_nu_max": float(finite_nu.max().detach().cpu()) if finite_nu.numel() else float("nan"),
        **family_counts,
    }
    probe_tensors = {
        "random_field_family_id": family_idx.detach().to(torch.float32),
        "random_field_length": chosen_length.detach(),
        "random_field_matern_nu": chosen_nu.detach(),
        "random_field_epsilon": eps.detach(),
    }
    return delta.detach(), info, probe_tensors


def burgers_random_field_training_batch(
    model,
    xb: torch.Tensor,
    yb: torch.Tensor,
    cfg: dict[str, Any],
) -> AttackBatchResult:
    was_training = model.training
    model.eval()
    eps = compute_batch_eps(
        xb,
        float(cfg["epsilon_fraction"]),
        float(cfg["epsilon_abs"]),
        float(cfg["eps_jitter_low"]),
        float(cfg["eps_jitter_high"]),
    )
    if float(cfg.get("epsilon_abs", 0.0) or 0.0) > 0:
        eps_nominal = torch.full_like(eps, float(cfg["epsilon_abs"]))
    else:
        eps_nominal = per_sample_range(xb).to(device=xb.device, dtype=xb.dtype) * float(cfg["epsilon_fraction"])
    eps_jitter_factor = eps / eps_nominal.clamp_min(1e-12)
    delta, field_info, probe_tensors = burgers_random_field_delta(xb, eps, cfg)
    x_train = (xb + delta).detach()
    target_mode = random_field_target_mode(cfg)

    with torch.no_grad():
        if target_mode == "clean-y":
            y_train = yb.detach()
            clean_target = yb.detach()
            objective_definition = "Random field delta; optimizer trains MSE(model(x+delta), y_clean)"
            target_source = "clean_y_unchanged_random_field"
            solver_forward = 0
        elif target_mode == "solver-y":
            y_train = solver_target_for_model_input("burgers", x_train, yb, cfg, allow_target_grad=False).detach()
            clean_target = solver_target_for_model_input("burgers", xb, yb, cfg, allow_target_grad=False).detach()
            objective_definition = "Random field delta; optimizer trains MSE(model(x+delta), solver(x+delta))"
            target_source = "solver_x_random_field_training_target"
            solver_forward = 1
        else:
            raise ValueError(f"unknown random_field_target_mode={target_mode!r}")
        clean_pred = model(xb)
        adv_pred = model(x_train)
        clean_loss_samples = per_sample_finite_mse(clean_pred, clean_target)
        adv_loss_samples = per_sample_finite_mse(adv_pred, y_train)
        clean_loss_value = float(finite_mse(clean_pred, clean_target).detach().cpu())
        adv_loss_value = float(finite_mse(adv_pred, y_train).detach().cpu())
        mask = torch.ones_like(xb)
        linf = float(delta.abs().reshape(delta.shape[0], -1).max(dim=1).values.mean().cpu())
        l2 = float(torch.sqrt(delta.pow(2).reshape(delta.shape[0], -1).mean(dim=1)).mean().cpu())
        l2_total = float(per_sample_l2_total(delta, mask).mean().cpu())
        eps_l2_total = float(rms_eps_to_l2_total(eps, mask).mean().cpu())
        boundary_ratio = float((per_sample_l2_total(delta, mask) / rms_eps_to_l2_total(eps, mask).clamp_min(1e-12)).mean().cpu())

    if was_training:
        model.train()

    info = {
        "label_mode": str(cfg.get("label_mode", "solver")),
        "target_source": target_source,
        "training_target_source": target_source,
        "attack_loss_objective": f"random_field_{target_mode}",
        "attack_objective_definition": objective_definition,
        "attack_uses_solver_forward": solver_forward,
        "attack_uses_solver_backward": 0,
        "full_solver_gradient": False,
        "attack_target_source": target_source,
        "attack_type": "random_field_l2_rms",
        "attack_method": "random_field_gaussian_matern",
        "attack_p_order": 2,
        "attack_q_order": 2,
        "epsilon_semantics": "per-sample RMS L2 radius for random field delta",
        "attack_steps": 0,
        "epsilon_mean": float(eps.mean().detach().cpu()),
        "epsilon_min": float(eps.min().detach().cpu()),
        "epsilon_max": float(eps.max().detach().cpu()),
        "alpha_mean": float(eps.mean().detach().cpu()),
        "alpha_min": float(eps.min().detach().cpu()),
        "alpha_max": float(eps.max().detach().cpu()),
        "alpha_ratio_nominal": 0.0,
        "alpha_jitter_low": float("nan"),
        "alpha_jitter_high": float("nan"),
        "alpha_is_epsilon_times_ratio": 0.0,
        "clean_loss_before_attack": clean_loss_value,
        "adv_loss_after_random_start": adv_loss_value,
        "adv_loss_after_attack": adv_loss_value,
        "attack_loss_gain": adv_loss_value - clean_loss_value,
        "grad_abs_mean_last": 0.0,
        "boundary_ratio_mean": boundary_ratio,
        "delta_linf_mean": linf,
        "delta_l2_rms_mean": l2,
        "delta_l2_total_mean": l2_total,
        "epsilon_l2_total_mean": eps_l2_total,
        **field_info,
    }
    info.update(tensor_stats("target", y_train))
    sample_info = make_attack_sample_info(
        epsilon=eps,
        epsilon_jitter_factor=eps_jitter_factor,
        alpha=eps,
        clean_loss=clean_loss_samples,
        adv_loss=adv_loss_samples,
    )
    sample_info.update({key: value.detach().float().cpu() for key, value in probe_tensors.items()})
    return AttackBatchResult(
        x_train.detach(),
        y_train.detach(),
        info,
        sample_info=sample_info,
        probe_tensors=probe_tensors,
    )


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
    alpha_jitter_low: float,
    alpha_jitter_high: float,
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
    if epsilon_abs and epsilon_abs > 0:
        eps_nominal = torch.full_like(eps, float(epsilon_abs))
    else:
        eps_nominal = per_sample_range(xb).to(device=xb.device, dtype=xb.dtype) * float(epsilon_fraction)
    eps_jitter_factor = eps / eps_nominal.clamp_min(1e-12)
    alpha_values = compute_alpha_values(eps, alpha_ratio, alpha_jitter_low, alpha_jitter_high)
    eps_view = expand_per_sample(eps, xb)
    alpha_view = expand_per_sample(alpha_values, xb)
    mask = ns_attack_mask(xb, ns_frames) if task == "ns2d" else torch.ones_like(xb)
    delta = random_start_delta(xb, eps, random_start_fraction) * mask
    delta = torch.clamp(delta, -eps_view, eps_view)
    x_adv = (xb + delta).detach()

    clean_loss_value = float("nan")
    first_adv_loss = float("nan")
    final_loss_value = float("nan")
    grad_abs_mean = float("nan")
    clean_loss_samples = torch.full((xb.shape[0],), float("nan"), device=xb.device, dtype=xb.dtype)
    adv_loss_samples = torch.full((xb.shape[0],), float("nan"), device=xb.device, dtype=xb.dtype)
    used_steps = max(1, int(steps))

    try:
        for step in range(used_steps):
            x_adv = x_adv.detach().requires_grad_(True)
            target, attack_objective, attack_objective_info = attack_objective_target(
                model,
                task,
                xb,
                x_adv,
                yb,
                cfg,
                allow_solver_backward=True,
            )
            pred = model(x_adv)
            loss = finite_mse(pred, target)
            if step == 0:
                with torch.no_grad():
                    clean_target, _, _ = attack_objective_target(
                        model,
                        task,
                        xb,
                        xb,
                        yb,
                        cfg,
                        allow_solver_backward=False,
                    )
                    clean_pred = model(xb)
                    clean_loss_samples = per_sample_finite_mse(clean_pred, clean_target)
                    clean_loss_value = float(finite_mse(clean_pred, clean_target).detach().cpu())
                    first_adv_loss = float(loss.detach().cpu())
            grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]
            grad = torch.nan_to_num(grad) * mask
            grad_abs_mean = float(grad.abs().mean().detach().cpu())
            if method.endswith("replace_linf"):
                delta = eps_view * grad.sign()
            elif method.endswith("add_linf"):
                delta = (x_adv.detach() - xb) + alpha_view * grad.sign()
                delta = torch.clamp(delta, -eps_view, eps_view)
            elif method.endswith("replace_l2"):
                delta = l2_rms_direction(grad, mask, eps)
            elif method.endswith("add_l2"):
                step_delta = l2_rms_direction(grad, mask, alpha_values)
                delta = l2_rms_project((x_adv.detach() - xb) + step_delta, mask, eps)
            else:
                raise ValueError(f"unknown continuous attack method: {method}")
            delta = delta * mask
            x_adv = (xb + delta).detach()

        with torch.no_grad():
            y_train = solver_target_for_model_input(task, x_adv, yb, cfg, allow_target_grad=False).detach()
            objective_target, attack_objective, attack_objective_info = attack_objective_target(
                model,
                task,
                xb,
                x_adv,
                yb,
                cfg,
                allow_solver_backward=False,
            )
            adv_pred = model(x_adv)
            adv_loss_samples = per_sample_finite_mse(adv_pred, objective_target)
            final_loss_value = float(finite_mse(adv_pred, objective_target).detach().cpu())
            delta = (x_adv - xb) * mask
            active = mask > 0
            linf = float(delta.detach().abs().reshape(delta.shape[0], -1).max(dim=1).values.mean().cpu())
            l2 = float(torch.sqrt(delta.detach().pow(2).reshape(delta.shape[0], -1).mean(dim=1)).mean().cpu())
            l2_total = float(per_sample_l2_total(delta.detach(), mask).mean().cpu())
            eps_l2_total = float(rms_eps_to_l2_total(eps, mask).mean().cpu())
            if method.endswith("_l2"):
                boundary_ratio = float((per_sample_l2_total(delta.detach(), mask) / rms_eps_to_l2_total(eps, mask).clamp_min(1e-12)).mean().cpu())
            else:
                denom = eps_view.expand_as(delta).clamp_min(1e-12)
                boundary_ratio = float((delta.detach().abs()[active] / denom[active]).mean().cpu()) if active.any() else 0.0

    finally:
        for param, flag in zip(model.parameters(), original_requires_grad):
            param.requires_grad_(flag)
        if was_training:
            model.train()

    info = {
        "label_mode": str(cfg.get("label_mode", "solver")),
        "target_source": "solver_x_adv_training_target" if str(cfg.get("label_mode", "solver")) == "solver" else str(cfg.get("label_mode", "solver")),
        "training_target_source": "solver(x_adv).detach() for optimizer update" if str(cfg.get("label_mode", "solver")) == "solver" else str(cfg.get("label_mode", "solver")),
        **attack_objective_info,
        "attack_type": "continuous_l2_rms" if method.endswith("_l2") else "continuous_linf",
        "attack_method": method,
        "attack_p_order": 2 if method.endswith("_l2") else "inf",
        "attack_q_order": 2,
        "epsilon_semantics": "per-sample RMS L2 radius" if method.endswith("_l2") else "per-coordinate Linf radius",
        "attack_steps": used_steps,
        "epsilon_mean": float(eps.mean().detach().cpu()),
        "epsilon_min": float(eps.min().detach().cpu()),
        "epsilon_max": float(eps.max().detach().cpu()),
        "alpha_mean": float(alpha_values.mean().detach().cpu()),
        "alpha_min": float(alpha_values.min().detach().cpu()),
        "alpha_max": float(alpha_values.max().detach().cpu()),
        "alpha_ratio_nominal": float(alpha_ratio),
        "alpha_jitter_low": float(alpha_jitter_low),
        "alpha_jitter_high": float(alpha_jitter_high),
        "alpha_is_epsilon_times_ratio": 0.0 if float(alpha_jitter_low) != 1.0 or float(alpha_jitter_high) != 1.0 else 1.0,
        "clean_loss_before_attack": clean_loss_value,
        "adv_loss_after_random_start": first_adv_loss,
        "adv_loss_after_attack": final_loss_value,
        "attack_loss_gain": final_loss_value - clean_loss_value,
        "grad_abs_mean_last": grad_abs_mean,
        "boundary_ratio_mean": boundary_ratio,
        "delta_linf_mean": linf,
        "delta_l2_rms_mean": l2,
        "delta_l2_total_mean": l2_total,
        "epsilon_l2_total_mean": eps_l2_total,
    }
    info.update(tensor_stats("target", y_train))
    sample_info = make_attack_sample_info(
        epsilon=eps,
        epsilon_jitter_factor=eps_jitter_factor,
        alpha=alpha_values,
        clean_loss=clean_loss_samples,
        adv_loss=adv_loss_samples,
    )
    return AttackBatchResult(x_adv.detach(), y_train.detach(), info, sample_info=sample_info)


DARCY_RANDOM_SOURCE_TRAINING_MODES = {"random-binary-fixed-y", "random-binary-solver-y"}


def is_darcy_random_source_mode(cfg: dict[str, Any]) -> bool:
    return str(cfg.get("training_data_mode", "adv-only")) in DARCY_RANDOM_SOURCE_TRAINING_MODES


def _darcy_random_source_filter(
    freq: torch.Tensor,
    *,
    kernel: str,
    alpha: float,
    lengthscale: float,
) -> torch.Tensor:
    cutoff = 1.0 / max(float(lengthscale), 1e-6)
    scaled = freq / max(cutoff, 1e-6)
    kernel = str(kernel).lower().strip()
    if kernel in {"gaussian", "rbf", "sqexp", "squared_exponential"}:
        filt = torch.exp(-0.5 * scaled.pow(2))
    elif kernel in {"matern", "matérn"}:
        filt = torch.pow(1.0 + scaled.pow(2), -0.5 * (float(alpha) + 1.0))
    elif kernel in {"highpass", "high_pass"}:
        low = torch.exp(-0.5 * scaled.pow(2))
        filt = torch.pow((1.0 - low).clamp_min(0.0), max(float(alpha) / 2.0, 0.25))
    elif kernel in {"bandpass", "band_pass"}:
        width = max(cutoff * 0.55, 1e-6)
        filt = torch.exp(-0.5 * ((freq - cutoff).abs() / width).pow(2))
    elif kernel in {"mixed", "hybrid"}:
        low = torch.exp(-0.5 * scaled.pow(2))
        high = torch.pow((1.0 - low).clamp_min(0.0), max(float(alpha) / 2.0, 0.25))
        filt = 0.5 * low + 0.5 * high
    else:
        raise ValueError(
            f"unknown Darcy random source kernel={kernel!r}; "
            "expected gaussian,matern,highpass,bandpass,or mixed"
        )
    return torch.nan_to_num(filt, nan=0.0, posinf=0.0, neginf=0.0)


def _darcy_random_source_field(
    h: int,
    w: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    kernel: str,
    alpha: float,
    lengthscale: float,
) -> torch.Tensor:
    white = torch.randn((h, w), device=device, dtype=dtype)
    fy = torch.fft.fftfreq(h, d=1.0 / max(1, h), device=device).to(dtype=dtype)
    fx = torch.fft.rfftfreq(w, d=1.0 / max(1, w), device=device).to(dtype=dtype)
    freq = torch.sqrt(fy[:, None].pow(2) + fx[None, :].pow(2))
    filt = _darcy_random_source_filter(freq, kernel=kernel, alpha=alpha, lengthscale=lengthscale)
    field = torch.fft.irfft2(torch.fft.rfft2(white) * filt, s=(h, w))
    field = field - field.mean()
    std = field.std(unbiased=False)
    if not torch.isfinite(std) or float(std.detach().cpu()) <= 1e-12:
        field = torch.randn((h, w), device=device, dtype=dtype)
        field = field - field.mean()
        std = field.std(unbiased=False).clamp_min(1e-12)
    return field / std.clamp_min(1e-12)


def darcy_random_binary_source_batch(
    model,
    xb: torch.Tensor,
    yb: torch.Tensor,
    cfg: dict[str, Any],
) -> AttackBatchResult:
    mode = str(cfg.get("training_data_mode", "adv-only"))
    if mode not in DARCY_RANDOM_SOURCE_TRAINING_MODES:
        raise ValueError(f"Darcy random source batch got unsupported mode={mode!r}")
    if xb.ndim != 4 or xb.shape[-1] != 1:
        raise ValueError(f"Darcy random binary source expects input (B,H,W,1), got {tuple(xb.shape)}")

    was_training = model.training
    model.eval()
    x0 = xb.detach()
    b, h, w, _ = x0.shape
    n_pix = int(h * w)
    max_fraction = float(cfg.get("darcy_random_source_max_flip_fraction", 0.05))
    min_fraction = float(cfg.get("darcy_random_source_min_flip_fraction", 0.005))
    if max_fraction < 0:
        raise ValueError(f"darcy_random_source_max_flip_fraction must be non-negative, got {max_fraction}")
    if min_fraction < 0:
        raise ValueError(f"darcy_random_source_min_flip_fraction must be non-negative, got {min_fraction}")
    if max_fraction < min_fraction:
        raise ValueError(
            "darcy_random_source_max_flip_fraction must be >= "
            f"min, got min={min_fraction} max={max_fraction}"
        )
    max_fraction = min(max_fraction, 1.0)
    min_fraction = min(min_fraction, max_fraction)
    alpha_values_cfg = parse_csv_floats(cfg.get("darcy_random_source_alpha_values"), [1.2, 2.2, 3.2, 4.2, 5.2])
    kernels = parse_csv_strings(
        cfg.get("darcy_random_source_kernels"),
        ["gaussian", "matern", "highpass", "bandpass", "mixed"],
    )
    lengthscale_min = float(cfg.get("darcy_random_source_lengthscale_min", 0.035))
    lengthscale_max = float(cfg.get("darcy_random_source_lengthscale_max", 0.30))
    if lengthscale_max < lengthscale_min:
        raise ValueError(
            "darcy_random_source_lengthscale_max must be >= min, "
            f"got min={lengthscale_min} max={lengthscale_max}"
        )

    flat_x0 = x0.reshape(b, -1)
    lo = flat_x0.min(dim=1).values.reshape(b, 1)
    hi = flat_x0.max(dim=1).values.reshape(b, 1)
    midpoint = 0.5 * (lo + hi)
    flat_binary = torch.where(flat_x0 > midpoint, hi.expand_as(flat_x0), lo.expand_as(flat_x0))
    other = torch.where(flat_binary > midpoint, lo.expand_as(flat_binary), hi.expand_as(flat_binary))
    flat_rand = flat_binary.clone()

    flip_fractions: list[float] = []
    alpha_draws: list[float] = []
    lengthscale_draws: list[float] = []
    kernel_draws: list[str] = []
    budgets: list[int] = []
    field_abs_means: list[float] = []
    field_stds: list[float] = []

    with torch.no_grad():
        for i in range(b):
            kernel = random.choice(kernels)
            alpha = float(random.choice(alpha_values_cfg))
            if lengthscale_max == lengthscale_min:
                lengthscale = lengthscale_min
            else:
                lengthscale = random.uniform(lengthscale_min, lengthscale_max)
            if max_fraction == min_fraction:
                fraction = max_fraction
            else:
                fraction = random.uniform(min_fraction, max_fraction)
            k = int(round(fraction * n_pix))
            k = max(0, min(k, n_pix))
            if max_fraction > 0.0 and k == 0:
                k = 1
            field = _darcy_random_source_field(
                h,
                w,
                device=x0.device,
                dtype=x0.dtype,
                kernel=kernel,
                alpha=alpha,
                lengthscale=lengthscale,
            )
            score = field.abs().reshape(-1)
            if k > 0:
                chosen = torch.topk(score, k=k, largest=True).indices
                flat_rand[i, chosen] = other[i, chosen]
            flip_fractions.append(k / max(1, n_pix))
            alpha_draws.append(alpha)
            lengthscale_draws.append(lengthscale)
            kernel_draws.append(kernel)
            budgets.append(k)
            field_abs_means.append(float(field.abs().mean().detach().cpu()))
            field_stds.append(float(field.std(unbiased=False).detach().cpu()))

        x_rand = flat_rand.reshape_as(x0).detach()
        if mode == "random-binary-fixed-y":
            y_train = yb.detach()
            target_source = "clean_dataset_y_unchanged"
            training_target_source = "clean dataset y; solver is not rerun after random binary flips"
            attack_uses_solver_forward = 0
        else:
            y_train = darcy_solver_target(x_rand, allow_target_grad=False).detach()
            target_source = "solver(a_random).detach()"
            training_target_source = "solver(a_random).detach() after random binary flips"
            attack_uses_solver_forward = 1

        clean_target = yb.detach()
        clean_pred = model(x0)
        adv_pred = model(x_rand)
        clean_loss_samples = per_sample_finite_mse(clean_pred, clean_target)
        adv_loss_samples = per_sample_finite_mse(adv_pred, y_train)
        clean_loss = finite_mse(clean_pred, clean_target)
        adv_loss = finite_mse(adv_pred, y_train)
        clean_solver_loss_samples = clean_loss_samples
        adv_solver_loss_samples = adv_loss_samples
        clean_solver_loss = clean_loss
        adv_solver_loss = adv_loss
        delta = x_rand - x0
        changed = delta.reshape(b, -1).abs() > 1e-12
        observed_flip_fraction = float(changed.float().mean().detach().cpu())
        delta_l2 = float(torch.sqrt(delta.pow(2).reshape(b, -1).mean(dim=1)).mean().detach().cpu())
        delta_linf = float(delta.abs().reshape(b, -1).max(dim=1).values.mean().detach().cpu())
        binary_error = torch.minimum((x_rand.reshape(b, -1) - lo).abs(), (x_rand.reshape(b, -1) - hi).abs())
        binary_error_max = float(binary_error.max().detach().cpu())

    if was_training:
        model.train()

    flip_tensor = torch.as_tensor(flip_fractions, device=x0.device, dtype=x0.dtype)
    alpha_tensor = torch.as_tensor(alpha_draws, device=x0.device, dtype=x0.dtype)
    lengthscale_tensor = torch.as_tensor(lengthscale_draws, device=x0.device, dtype=x0.dtype)
    budget_tensor = torch.as_tensor(budgets, device=x0.device, dtype=x0.dtype)
    kernel_counts = {name: kernel_draws.count(name) for name in sorted(set(kernel_draws))}
    info = {
        "label_mode": str(cfg.get("label_mode", "solver")),
        "target_source": target_source,
        "training_target_source": training_target_source,
        "attack_loss_objective": mode,
        "attack_objective_definition": "random binary source perturbation; no adversarial gradient is used",
        "attack_uses_solver_forward": attack_uses_solver_forward,
        "attack_uses_solver_backward": 0,
        "full_solver_gradient": False,
        "attack_target_source": target_source,
        "attack_type": "random_binary_source",
        "attack_method": mode,
        "attack_steps": 0,
        "epsilon_mean": float(flip_tensor.mean().detach().cpu()),
        "epsilon_min": float(flip_tensor.min().detach().cpu()),
        "epsilon_max": float(flip_tensor.max().detach().cpu()),
        "alpha_mean": float(alpha_tensor.mean().detach().cpu()),
        "alpha_min": float(alpha_tensor.min().detach().cpu()),
        "alpha_max": float(alpha_tensor.max().detach().cpu()),
        "alpha_ratio_nominal": float("nan"),
        "clean_loss_before_attack": float(clean_loss.detach().cpu()),
        "adv_loss_after_random_start": float(adv_loss.detach().cpu()),
        "adv_loss_after_attack": float(adv_loss.detach().cpu()),
        "attack_loss_gain": float((adv_loss - clean_loss).detach().cpu()),
        "clean_solver_mse_before_attack": float(clean_solver_loss.detach().cpu()),
        "adv_solver_mse_after_attack": float(adv_solver_loss.detach().cpu()),
        "solver_mse_attack_gain": float((adv_solver_loss - clean_solver_loss).detach().cpu()),
        "grad_abs_mean_last": float("nan"),
        "boundary_ratio_mean": observed_flip_fraction / max(max_fraction, 1e-12),
        "delta_linf_mean": delta_linf,
        "delta_l2_rms_mean": delta_l2,
        "darcy_budget_pixels_mean": float(budget_tensor.mean().detach().cpu()),
        "darcy_flip_fraction": observed_flip_fraction,
        "darcy_random_source_mode": mode,
        "darcy_random_source_kernel_set": ",".join(kernels),
        "darcy_random_source_alpha_values": ",".join(str(x) for x in alpha_values_cfg),
        "darcy_random_source_alpha_draw_mean": float(alpha_tensor.mean().detach().cpu()),
        "darcy_random_source_alpha_draw_min": float(alpha_tensor.min().detach().cpu()),
        "darcy_random_source_alpha_draw_max": float(alpha_tensor.max().detach().cpu()),
        "darcy_random_source_lengthscale_min_cfg": lengthscale_min,
        "darcy_random_source_lengthscale_max_cfg": lengthscale_max,
        "darcy_random_source_lengthscale_draw_mean": float(lengthscale_tensor.mean().detach().cpu()),
        "darcy_random_source_lengthscale_draw_min": float(lengthscale_tensor.min().detach().cpu()),
        "darcy_random_source_lengthscale_draw_max": float(lengthscale_tensor.max().detach().cpu()),
        "darcy_random_source_min_flip_fraction_cfg": min_fraction,
        "darcy_random_source_max_flip_fraction_cfg": max_fraction,
        "darcy_random_source_observed_flip_fraction_mean": observed_flip_fraction,
        "darcy_random_source_normalized_l1_energy_mean": observed_flip_fraction,
        "darcy_random_source_normalized_squared_l2_energy_mean": observed_flip_fraction,
        "darcy_random_source_field_abs_mean": float(np.mean(field_abs_means)) if field_abs_means else float("nan"),
        "darcy_random_source_field_std_mean": float(np.mean(field_stds)) if field_stds else float("nan"),
        "darcy_random_source_binary_error_max": binary_error_max,
        "darcy_random_source_distinct_kernel_count": len(kernel_counts),
    }
    for kernel_name, count in kernel_counts.items():
        safe_name = "".join(ch if ch.isalnum() else "_" for ch in kernel_name)
        info[f"darcy_random_source_kernel_{safe_name}_count"] = int(count)
    info.update(tensor_stats("target", y_train))
    sample_info = make_attack_sample_info(
        epsilon=flip_tensor,
        epsilon_jitter_factor=flip_tensor / max(max_fraction, 1e-12),
        alpha=alpha_tensor,
        clean_loss=clean_loss_samples,
        adv_loss=adv_loss_samples,
    )
    return AttackBatchResult(x_rand.detach(), y_train.detach(), info, sample_info=sample_info)


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
    jitter_low_f = float(jitter_low)
    jitter_high_f = float(jitter_high)
    if jitter_high_f < jitter_low_f:
        raise ValueError(f"Darcy epsilon jitter high must be >= low, got low={jitter_low_f}, high={jitter_high_f}")
    if jitter_low_f == jitter_high_f:
        jitter = torch.full((b,), jitter_low_f, device=x0.device, dtype=x0.dtype)
    else:
        jitter = torch.empty((b,), device=x0.device, dtype=x0.dtype).uniform_(jitter_low_f, jitter_high_f)
    budgets = torch.clamp((float(epsilon_fraction) * jitter * n_pix).round().long(), min=1, max=n_pix)
    used_steps = max(1, int(steps))
    objective = darcy_attack_loss_objective(cfg)
    attack_objective_info = darcy_attack_loss_metadata(objective, cfg)

    loss1_random_start_enabled = bool(cfg.get("darcy_loss1_random_start", True)) and objective == "loss1"
    loss1_random_start_fraction = float(cfg.get("darcy_loss1_random_start_fraction", 1.0))
    loss1_random_start_flips_total = 0
    if loss1_random_start_enabled and loss1_random_start_fraction > 0:
        random_budgets = torch.clamp(
            (budgets.float() * loss1_random_start_fraction).round().long(),
            min=1,
            max=n_pix,
        )
        flat_x0 = x0.reshape(b, -1)
        lo0 = flat_x0.min(dim=1).values.reshape(b, 1)
        hi0 = flat_x0.max(dim=1).values.reshape(b, 1)
        midpoint0 = (lo0 + hi0) * 0.5
        other0 = torch.where(flat_x0 > midpoint0, lo0.expand_as(flat_x0), hi0.expand_as(flat_x0))
        flat_adv0 = flat_x0.clone()
        for i in range(b):
            k0 = int(random_budgets[i].item())
            if k0 <= 0:
                continue
            chosen0 = torch.randperm(n_pix, device=x0.device)[:k0]
            flat_adv0[i, chosen0] = other0[i, chosen0]
            loss1_random_start_flips_total += k0
        x_adv = flat_adv0.reshape_as(x0).detach()

    first_loss = float("nan")
    positive_score_frac = float("nan")
    flips_total = 0
    clean_loss_samples = torch.full((b,), float("nan"), device=x0.device, dtype=x0.dtype)
    adv_loss_samples = torch.full((b,), float("nan"), device=x0.device, dtype=x0.dtype)

    try:
        with torch.no_grad():
            clean_loss, clean_loss_samples, _, attack_objective_info = darcy_attack_objective_loss(
                model,
                x0,
                x0,
                yb,
                cfg,
                allow_solver_backward=False,
            )
            clean_solver_target = solver_target_for_model_input("darcy", x0, yb, cfg, allow_target_grad=False).detach()
            clean_pred = model(x0)
            clean_solver_loss_samples = per_sample_finite_mse(clean_pred, clean_solver_target)
            clean_solver_loss = finite_mse(clean_pred, clean_solver_target)

        for step in range(used_steps):
            x_score = x_adv.detach().clone().requires_grad_(True)
            loss, _, attack_objective, attack_objective_info = darcy_attack_objective_loss(
                model,
                x0,
                x_score,
                yb,
                cfg,
                allow_solver_backward=True,
            )
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
            adv_loss, adv_loss_samples, attack_objective, attack_objective_info = darcy_attack_objective_loss(
                model,
                x0,
                x_adv,
                yb,
                cfg,
                allow_solver_backward=False,
            )
            adv_pred = model(x_adv)
            adv_solver_loss_samples = per_sample_finite_mse(adv_pred, y_train)
            adv_solver_loss = finite_mse(adv_pred, y_train)
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
        "target_source": "solver_x_adv_training_target" if str(cfg.get("label_mode", "solver")) == "solver" else str(cfg.get("label_mode", "solver")),
        "training_target_source": "solver(a_adv).detach() for optimizer update" if str(cfg.get("label_mode", "solver")) == "solver" else str(cfg.get("label_mode", "solver")),
        **attack_objective_info,
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
        "clean_solver_mse_before_attack": float(clean_solver_loss.detach().cpu()),
        "adv_solver_mse_after_attack": float(adv_solver_loss.detach().cpu()),
        "solver_mse_attack_gain": float((adv_solver_loss - clean_solver_loss).detach().cpu()),
        "grad_abs_mean_last": float(grad.abs().mean().detach().cpu()),
        "boundary_ratio_mean": flip_fraction / max(float(epsilon_fraction), 1e-12),
        "delta_linf_mean": float(delta.abs().reshape(b, -1).max(dim=1).values.mean().detach().cpu()),
        "delta_l2_rms_mean": delta_l2,
        "darcy_budget_pixels_mean": float(budgets.float().mean().detach().cpu()),
        "darcy_flip_fraction": flip_fraction,
        "darcy_positive_score_fraction": positive_score_frac,
        "darcy_last_step_flips_total": flips_total,
        "darcy_loss1_random_start_enabled": int(loss1_random_start_enabled),
        "darcy_loss1_random_start_fraction": loss1_random_start_fraction,
        "darcy_loss1_random_start_flips_total": loss1_random_start_flips_total,
        "darcy_loss1_random_start_flips_mean": loss1_random_start_flips_total / max(1, b),
    }
    info.update(tensor_stats("target", y_train))
    epsilon_fraction_per_sample = budgets.float() / float(n_pix)
    sample_info = make_attack_sample_info(
        epsilon=epsilon_fraction_per_sample,
        epsilon_jitter_factor=epsilon_fraction_per_sample / max(float(epsilon_fraction), 1e-12),
        alpha=epsilon_fraction_per_sample,
        clean_loss=clean_loss_samples,
        adv_loss=adv_loss_samples,
    )
    return AttackBatchResult(x_adv.detach(), y_train.detach(), info, sample_info=sample_info)


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
    alpha_jitter_low: float,
    alpha_jitter_high: float,
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
    if epsilon_abs and epsilon_abs > 0:
        eps_nominal = torch.full_like(eps, float(epsilon_abs))
    else:
        eps_nominal = per_sample_range(x0).to(device=x0.device, dtype=x0.dtype) * float(epsilon_fraction)
    eps_jitter_factor = eps / eps_nominal.clamp_min(1e-12)
    alpha_values = compute_alpha_values(eps, alpha_ratio, alpha_jitter_low, alpha_jitter_high)
    eps_view = expand_per_sample(eps, x0)
    alpha_view = expand_per_sample(alpha_values, x0)
    delta = torch.clamp(random_start_delta(x0, eps, random_start_fraction), -eps_view, eps_view)
    x0_adv = (x0 + delta).detach()

    clean_loss_value = float("nan")
    first_adv_loss = float("nan")
    final_loss_value = float("nan")
    grad_abs_mean = float("nan")
    clean_loss_samples = torch.full((x0.shape[0],), float("nan"), device=x0.device, dtype=x0.dtype)
    adv_loss_samples = torch.full((x0.shape[0],), float("nan"), device=x0.device, dtype=x0.dtype)
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
                    clean_pred = model(x_seq_clean)
                    clean_loss_samples = per_sample_finite_mse(clean_pred, y_seq_clean)
                    clean_loss_value = float(finite_mse(clean_pred, y_seq_clean).detach().cpu())
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
            adv_pred = model(x_train)
            adv_loss_samples = per_sample_finite_mse(adv_pred, y_train)
            final_loss_value = float(finite_mse(adv_pred, y_train).detach().cpu())
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
        "alpha_mean": float(alpha_values.mean().detach().cpu()),
        "alpha_min": float(alpha_values.min().detach().cpu()),
        "alpha_max": float(alpha_values.max().detach().cpu()),
        "alpha_ratio_nominal": float(alpha_ratio),
        "alpha_jitter_low": float(alpha_jitter_low),
        "alpha_jitter_high": float(alpha_jitter_high),
        "alpha_is_epsilon_times_ratio": 0.0 if float(alpha_jitter_low) != 1.0 or float(alpha_jitter_high) != 1.0 else 1.0,
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
    sample_info = make_attack_sample_info(
        epsilon=eps,
        epsilon_jitter_factor=eps_jitter_factor,
        alpha=alpha_values,
        clean_loss=clean_loss_samples,
        adv_loss=adv_loss_samples,
    )
    return AttackBatchResult(
        x_train.detach(),
        y_train.detach(),
        info,
        sample_info=sample_info,
        probe_tensors={
            "x0_clean": x0.detach(),
            "x0_adv": x0_adv.detach(),
            "delta_initial": (x0_adv - x0).detach(),
        },
    )


def attack_batch(model, xb: torch.Tensor, yb: torch.Tensor, task: str, cfg: dict[str, Any]) -> AttackBatchResult:
    if str(cfg.get("training_data_mode", "adv-only")) == "clean-only":
        return clean_only_training_batch(model, xb, yb, task, cfg)
    if str(cfg.get("training_perturbation_mode", "attack")) == "random-field":
        if task != "burgers":
            raise ValueError("--training-perturbation-mode random-field is currently implemented for Burgers only")
        return burgers_random_field_training_batch(model, xb, yb, cfg)
    if task == "darcy" and is_darcy_random_source_mode(cfg):
        return darcy_random_binary_source_batch(model, xb, yb, cfg)
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
            alpha_jitter_low=float(cfg.get("alpha_jitter_low", 1.0)),
            alpha_jitter_high=float(cfg.get("alpha_jitter_high", 1.0)),
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
        alpha_jitter_low=float(cfg.get("alpha_jitter_low", 1.0)),
        alpha_jitter_high=float(cfg.get("alpha_jitter_high", 1.0)),
        random_start_fraction=float(cfg["random_start_fraction"]),
        jitter_low=float(cfg["eps_jitter_low"]),
        jitter_high=float(cfg["eps_jitter_high"]),
        ns_frames=str(cfg["ns_attack_frames"]),
        cfg=cfg,
    )


def clean_only_training_batch(model, xb: torch.Tensor, yb: torch.Tensor, task: str, cfg: dict[str, Any]) -> AttackBatchResult:
    x_clean, y_clean = clean_solver_training_pair(task, xb, yb, cfg)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        pred = model(x_clean)
        per_sample_loss = per_sample_finite_mse(pred, y_clean)
        loss_value = float(finite_mse(pred, y_clean).detach().cpu())
    if was_training:
        model.train()

    batch = int(x_clean.shape[0])
    zeros = torch.zeros((batch,), device=x_clean.device, dtype=x_clean.dtype)
    ones = torch.ones((batch,), device=x_clean.device, dtype=x_clean.dtype)
    delta = torch.zeros_like(x_clean)
    info = {
        "label_mode": str(cfg.get("label_mode", "solver")),
        "target_source": "solver_x_clean_training_target",
        "training_target_source": "solver(x_clean).detach() for clean-only optimizer update",
        "attack_loss_objective": "clean",
        "attack_objective_definition": "No perturbation; optimizer trains MSE(model(x_clean), solver(x_clean))",
        "attack_uses_solver_forward": 1,
        "attack_uses_solver_backward": 0,
        "attack_target_source": "clean solver output",
        "full_solver_gradient": False,
        "attack_type": "clean_only",
        "attack_method": "clean_only",
        "attack_steps": 0,
        "epsilon_mean": 0.0,
        "epsilon_min": 0.0,
        "epsilon_max": 0.0,
        "alpha_mean": 0.0,
        "alpha_min": 0.0,
        "alpha_max": 0.0,
        "alpha_ratio_nominal": float(cfg.get("alpha_ratio", 0.0)),
        "alpha_jitter_low": float(cfg.get("alpha_jitter_low", 1.0)),
        "alpha_jitter_high": float(cfg.get("alpha_jitter_high", 1.0)),
        "alpha_is_epsilon_times_ratio": 1.0,
        "clean_loss_before_attack": loss_value,
        "adv_loss_after_random_start": loss_value,
        "adv_loss_after_attack": loss_value,
        "attack_loss_gain": 0.0,
        "clean_solver_mse_before_attack": loss_value,
        "adv_solver_mse_after_attack": loss_value,
        "grad_abs_mean_last": 0.0,
        "boundary_ratio_mean": 0.0,
        "delta_linf_mean": 0.0,
        "delta_l2_rms_mean": 0.0,
    }
    info.update(tensor_stats("target", y_clean))
    info.update(tensor_stats("x_train", x_clean))
    sample_info = make_attack_sample_info(
        epsilon=zeros,
        epsilon_jitter_factor=ones,
        alpha=zeros,
        clean_loss=per_sample_loss,
        adv_loss=per_sample_loss,
    )
    return AttackBatchResult(
        x_clean.detach(),
        y_clean.detach(),
        info,
        sample_info=sample_info,
        probe_tensors={
            "x0_clean": x_clean.detach(),
            "x0_adv": x_clean.detach(),
            "delta_initial": delta.detach(),
        },
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
    if mode == "adv-only" or mode == "clean-only" or mode in DARCY_RANDOM_SOURCE_TRAINING_MODES:
        info = {
            "training_data_mode": mode,
            "clean_train_samples": int(x_adv.shape[0]) if mode == "clean-only" else 0,
            "adv_train_samples": 0 if mode == "clean-only" else int(x_adv.shape[0]),
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
    raise ValueError(
        f"unknown training_data_mode={mode!r}; expected adv-only, clean-only, clean-plus-adv, "
        "random-binary-fixed-y, or random-binary-solver-y"
    )


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
    override_attack_method = getattr(args, f"{task}_attack_method", None)
    if override_attack_method is not None:
        cfg["attack_method"] = str(override_attack_method)
    if task == "burgers":
        cfg["attack_loss_objective"] = str(getattr(args, "burgers_attack_loss_objective", "loss3"))
    if task == "darcy":
        cfg["attack_loss_objective"] = str(getattr(args, "darcy_attack_loss_objective", "loss3"))
        cfg["darcy_physics_metric"] = str(getattr(args, "darcy_physics_metric", "rel_l2"))
        cfg["darcy_physics_bc_weight"] = float(getattr(args, "darcy_physics_bc_weight", 1.0))
        cfg["darcy_physics_forcing_value"] = float(getattr(args, "darcy_physics_forcing_value", 1.0))
        cfg["darcy_loss1_random_start"] = bool(getattr(args, "darcy_loss1_random_start", True))
        cfg["darcy_loss1_random_start_fraction"] = float(getattr(args, "darcy_loss1_random_start_fraction", 1.0))
        cfg["darcy_random_source_kernels"] = str(getattr(args, "darcy_random_source_kernels", "gaussian,matern,highpass,bandpass,mixed"))
        cfg["darcy_random_source_alpha_values"] = str(getattr(args, "darcy_random_source_alpha_values", "1.2,2.2,3.2,4.2,5.2"))
        cfg["darcy_random_source_lengthscale_min"] = float(getattr(args, "darcy_random_source_lengthscale_min", 0.035))
        cfg["darcy_random_source_lengthscale_max"] = float(getattr(args, "darcy_random_source_lengthscale_max", 0.30))
        cfg["darcy_random_source_min_flip_fraction"] = float(getattr(args, "darcy_random_source_min_flip_fraction", 0.005))
        cfg["darcy_random_source_max_flip_fraction"] = float(getattr(args, "darcy_random_source_max_flip_fraction", 0.05))
    override_random_start_fraction = getattr(args, f"{task}_random_start_fraction", None)
    if override_random_start_fraction is not None:
        cfg["random_start_fraction"] = float(override_random_start_fraction)
    override_epsilon_fraction = getattr(args, f"{task}_epsilon_fraction", None)
    if override_epsilon_fraction is not None:
        cfg["epsilon_fraction"] = float(override_epsilon_fraction)
    override_epsilon_abs = getattr(args, f"{task}_epsilon_abs", None)
    if override_epsilon_abs is not None:
        cfg["epsilon_abs"] = float(override_epsilon_abs)
    override_alpha_ratio = getattr(args, f"{task}_alpha_ratio", None)
    if override_alpha_ratio is not None:
        cfg["alpha_ratio"] = float(override_alpha_ratio)
    override_alpha_jitter_low = getattr(args, f"{task}_alpha_jitter_low", None)
    if override_alpha_jitter_low is not None:
        cfg["alpha_jitter_low"] = float(override_alpha_jitter_low)
    override_alpha_jitter_high = getattr(args, f"{task}_alpha_jitter_high", None)
    if override_alpha_jitter_high is not None:
        cfg["alpha_jitter_high"] = float(override_alpha_jitter_high)
    override_eps_jitter_low = getattr(args, f"{task}_eps_jitter_low", None)
    if override_eps_jitter_low is not None:
        cfg["eps_jitter_low"] = float(override_eps_jitter_low)
    override_eps_jitter_high = getattr(args, f"{task}_eps_jitter_high", None)
    if override_eps_jitter_high is not None:
        cfg["eps_jitter_high"] = float(override_eps_jitter_high)
    cfg["max_batches_per_epoch"] = args.max_batches_per_epoch
    cfg["max_work_seconds"] = args.max_work_seconds
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
        eval_every_steps = max(1, batches_per_epoch)
        eval_steps = [epoch * max(1, batches_per_epoch) for epoch in range(1, int(cfg["epochs"]) + 1)]
        eval_passes = 1 + len(eval_steps)
        checkpoint_every_epochs = int(args.checkpoint_every_epochs or 0)
        checkpoint_fraction = args.checkpoint_every_fraction
        if checkpoint_every_epochs > 0:
            checkpoint_every_steps = max(1, checkpoint_every_epochs * max(1, batches_per_epoch))
            checkpoint_steps = sorted(set([epoch * max(1, batches_per_epoch) for epoch in range(checkpoint_every_epochs, int(cfg["epochs"]) + 1, checkpoint_every_epochs)] + [total_steps]))
        elif checkpoint_fraction is not None and float(checkpoint_fraction) > 0:
            checkpoint_every_steps = max(1, int(math.ceil(total_steps * float(checkpoint_fraction))))
            checkpoint_steps = sorted(set(list(range(checkpoint_every_steps, total_steps + 1, checkpoint_every_steps)) + [total_steps]))
        else:
            checkpoint_every_steps = None
            checkpoint_steps = [total_steps]
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
            "epsilon_abs": float(cfg["epsilon_abs"]),
            "alpha_ratio": float(cfg["alpha_ratio"]),
            "alpha_jitter_low": float(cfg.get("alpha_jitter_low", 1.0)),
            "alpha_jitter_high": float(cfg.get("alpha_jitter_high", 1.0)),
            "full_solver_gradient": True,
            "batches_per_epoch": batches_per_epoch,
            "total_train_steps": total_steps,
            "evaluation_schedule": "every_epoch",
            "eval_every_steps": eval_every_steps,
            "eval_steps_after_baseline": eval_steps,
            "eval_pass_count_including_baseline": eval_passes,
            "checkpoint_every_epochs": checkpoint_every_epochs,
            "checkpoint_every_fraction": None if checkpoint_fraction is None else float(checkpoint_fraction),
            "checkpoint_every_steps": checkpoint_every_steps,
            "checkpoint_steps": checkpoint_steps,
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


def _finite_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _weighted_mean(rows: list[dict[str, Any]], key: str, weight_key: str = "batch_size") -> float:
    total = 0.0
    weight_total = 0.0
    for row in rows:
        value = _finite_float(row.get(key))
        if not math.isfinite(value):
            continue
        weight = _finite_float(row.get(weight_key, 1.0))
        if not math.isfinite(weight) or weight <= 0:
            weight = 1.0
        total += value * weight
        weight_total += weight
    return total / weight_total if weight_total > 0 else float("nan")


def write_attack_epoch_summary(path: Path, task: str, epoch: int, global_step: int, rows: list[dict[str, Any]]) -> None:
    attack_samples = sum(int(row.get("batch_size", 0)) for row in rows)
    clean_mean = _weighted_mean(rows, "clean_loss_before_attack")
    adv_mean = _weighted_mean(rows, "adv_loss_after_attack")
    gain_mean = adv_mean - clean_mean if math.isfinite(clean_mean) and math.isfinite(adv_mean) else float("nan")
    clean_solver_mse_mean = _weighted_mean(rows, "clean_solver_mse_before_attack")
    adv_solver_mse_mean = _weighted_mean(rows, "adv_solver_mse_after_attack")
    solver_mse_gain_mean = adv_solver_mse_mean - clean_solver_mse_mean if math.isfinite(clean_solver_mse_mean) and math.isfinite(adv_solver_mse_mean) else float("nan")
    attack_wall_sec_total = sum(_finite_float(row.get("attack_wall_sec")) for row in rows if math.isfinite(_finite_float(row.get("attack_wall_sec"))))
    row = {
        "task": task,
        "epoch": int(epoch),
        "global_step_last": int(global_step),
        "attack_batches": int(len(rows)),
        "attack_samples": int(attack_samples),
        "attack_loss_objective": str(rows[0].get("attack_loss_objective", "")) if rows else "",
        "attack_objective_definition": str(rows[0].get("attack_objective_definition", "")) if rows else "",
        "attack_uses_solver_forward": rows[0].get("attack_uses_solver_forward", "") if rows else "",
        "attack_uses_solver_backward": rows[0].get("attack_uses_solver_backward", "") if rows else "",
        "clean_loss_before_attack_mean": clean_mean,
        "adv_loss_after_attack_mean": adv_mean,
        "attack_loss_gain_mean": gain_mean,
        "attack_loss_gain_relative_mean": (gain_mean / abs(clean_mean)) if math.isfinite(gain_mean) and math.isfinite(clean_mean) and abs(clean_mean) > 1e-20 else float("nan"),
        "clean_solver_mse_before_attack_mean": clean_solver_mse_mean,
        "adv_solver_mse_after_attack_mean": adv_solver_mse_mean,
        "solver_mse_attack_gain_mean": solver_mse_gain_mean,
        "train_loss_used_for_optimizer_updates_mean": _weighted_mean(rows, "train_loss_on_adv_mean"),
        "epsilon_mean": _weighted_mean(rows, "epsilon_mean"),
        "epsilon_min_observed": min((_finite_float(row.get("epsilon_min")) for row in rows), default=float("nan")),
        "epsilon_max_observed": max((_finite_float(row.get("epsilon_max")) for row in rows), default=float("nan")),
        "alpha_mean": _weighted_mean(rows, "alpha_mean"),
        "alpha_min_observed": min((_finite_float(row.get("alpha_min")) for row in rows), default=float("nan")),
        "alpha_max_observed": max((_finite_float(row.get("alpha_max")) for row in rows), default=float("nan")),
        "delta_linf_mean": _weighted_mean(rows, "delta_linf_mean"),
        "delta_l2_rms_mean": _weighted_mean(rows, "delta_l2_rms_mean"),
        "boundary_ratio_mean": _weighted_mean(rows, "boundary_ratio_mean"),
        "grad_abs_mean_last_mean": _weighted_mean(rows, "grad_abs_mean_last"),
        "darcy_loss1_random_start_flips_mean": _weighted_mean(rows, "darcy_loss1_random_start_flips_mean"),
        "attack_wall_sec_total": attack_wall_sec_total,
    }
    row["attack_samples_per_sec"] = row["attack_samples"] / max(row["attack_wall_sec_total"], 1e-12)
    write_csv_row(path, row)




def _cat_sample_tensors(sample_infos: list[dict[str, torch.Tensor]], key: str) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for info in sample_infos:
        value = info.get(key)
        if isinstance(value, torch.Tensor) and value.numel() > 0:
            chunks.append(value.detach().reshape(-1).float().cpu())
    if not chunks:
        return torch.empty((0,), dtype=torch.float32)
    return torch.cat(chunks, dim=0)


def _tensor_stat(values: torch.Tensor, op: str) -> float:
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return float("nan")
    if op == "mean":
        return float(finite.mean().item())
    if op == "std":
        return float(finite.std(unbiased=False).item())
    if op == "min":
        return float(finite.min().item())
    if op == "max":
        return float(finite.max().item())
    raise ValueError(f"unknown tensor stat op={op!r}")


def _add_metric_stats(row: dict[str, Any], prefix: str, values: torch.Tensor) -> None:
    row[f"{prefix}_mean"] = _tensor_stat(values, "mean")
    row[f"{prefix}_std"] = _tensor_stat(values, "std")
    row[f"{prefix}_min"] = _tensor_stat(values, "min")
    row[f"{prefix}_max"] = _tensor_stat(values, "max")


def write_attack_epsilon_bucket_summary(
    path: Path,
    task: str,
    epoch: int,
    global_step: int,
    sample_infos: list[dict[str, torch.Tensor]],
    cfg: dict[str, Any],
    bucket_count: int,
) -> None:
    bucket_count = int(bucket_count)
    if bucket_count <= 0:
        return

    epsilon = _cat_sample_tensors(sample_infos, "epsilon")
    if epsilon.numel() == 0:
        return
    epsilon_jitter = _cat_sample_tensors(sample_infos, "epsilon_jitter_factor")
    alpha = _cat_sample_tensors(sample_infos, "alpha")
    clean_loss = _cat_sample_tensors(sample_infos, "clean_loss_before_attack")
    adv_loss = _cat_sample_tensors(sample_infos, "adv_loss_after_attack")
    gain = _cat_sample_tensors(sample_infos, "attack_loss_gain")
    relative_gain = _cat_sample_tensors(sample_infos, "attack_loss_gain_relative")

    n = int(epsilon.numel())
    valid_lengths = {tensor.numel() for tensor in (epsilon_jitter, alpha, clean_loss, adv_loss, gain, relative_gain)}
    if valid_lengths != {n}:
        # Avoid writing misleading bucket rows if one metric vector is missing or truncated.
        return

    jitter_low = float(cfg.get("eps_jitter_low", 1.0))
    jitter_high = float(cfg.get("eps_jitter_high", 1.0))
    use_jitter_basis = bool(epsilon_jitter.numel() == n and torch.isfinite(epsilon_jitter).any() and jitter_high > jitter_low)
    if use_jitter_basis:
        bucket_values = epsilon_jitter
        bucket_low = jitter_low
        bucket_high = jitter_high
        bucket_basis = "epsilon_jitter_factor"
    else:
        bucket_values = epsilon
        finite_bucket = bucket_values[torch.isfinite(bucket_values)]
        if finite_bucket.numel() == 0:
            return
        bucket_low = float(finite_bucket.min().item())
        bucket_high = float(finite_bucket.max().item())
        bucket_basis = "epsilon_observed"

    if not math.isfinite(bucket_low) or not math.isfinite(bucket_high):
        return
    if bucket_high <= bucket_low:
        bucket_high = bucket_low + max(abs(bucket_low), 1.0) * 1e-12

    edges = np.linspace(bucket_low, bucket_high, bucket_count + 1)
    finite_basis = torch.isfinite(bucket_values)
    for bucket_idx in range(bucket_count):
        left = float(edges[bucket_idx])
        right = float(edges[bucket_idx + 1])
        if bucket_idx == bucket_count - 1:
            mask = finite_basis & (bucket_values >= left) & (bucket_values <= right)
        else:
            mask = finite_basis & (bucket_values >= left) & (bucket_values < right)
        sample_count = int(mask.sum().item())
        row: dict[str, Any] = {
            "task": task,
            "epoch": int(epoch),
            "global_step_last": int(global_step),
            "attack_loss_objective": str(cfg.get("attack_loss_objective", "loss3")),
            "bucket_basis": bucket_basis,
            "bucket_index": int(bucket_idx),
            "bucket_count": int(bucket_count),
            "bucket_fraction_low": float(bucket_idx / bucket_count),
            "bucket_fraction_high": float((bucket_idx + 1) / bucket_count),
            "bucket_value_low": left,
            "bucket_value_high": right,
            "sample_count": sample_count,
            "sample_fraction": sample_count / max(1, n),
        }
        if sample_count > 0:
            _add_metric_stats(row, "bucket_value", bucket_values[mask])
            _add_metric_stats(row, "epsilon", epsilon[mask])
            _add_metric_stats(row, "epsilon_jitter_factor", epsilon_jitter[mask])
            _add_metric_stats(row, "alpha", alpha[mask])
            _add_metric_stats(row, "clean_loss_before_attack", clean_loss[mask])
            _add_metric_stats(row, "adv_loss_after_attack", adv_loss[mask])
            _add_metric_stats(row, "attack_loss_gain", gain[mask])
            _add_metric_stats(row, "loss_increase", gain[mask])
            _add_metric_stats(row, "attack_loss_gain_relative", relative_gain[mask])
            _add_metric_stats(row, "loss_increase_relative", relative_gain[mask])
        else:
            for prefix in (
                "bucket_value",
                "epsilon",
                "epsilon_jitter_factor",
                "alpha",
                "clean_loss_before_attack",
                "adv_loss_after_attack",
                "attack_loss_gain",
                "loss_increase",
                "attack_loss_gain_relative",
                "loss_increase_relative",
            ):
                for op in ("mean", "std", "min", "max"):
                    row[f"{prefix}_{op}"] = float("nan")
        write_csv_row(path, row)


def write_eval_split_summary(path: Path, rows: list[dict[str, Any]], *, phase: str, epoch: int, global_step: int, progress_fraction: float, eval_wall_sec: float) -> None:
    groups: dict[str, list[dict[str, Any]]] = {"ALL": rows}
    for row in rows:
        groups.setdefault(str(row.get("split", "unknown")), []).append(row)
    metric_keys = ["rmse", "mae", "relative_l2", "accuracy_score", "invalid_value_fraction"]
    for split, split_rows in groups.items():
        out = {
            "task": rows[0].get("task") if rows else "",
            "phase": phase,
            "epoch": int(epoch),
            "global_step": int(global_step),
            "progress_fraction": float(progress_fraction),
            "split": split,
            "dataset_count": int(len(split_rows)),
            "total_samples_evaluated": int(sum(int(row.get("num_samples_evaluated", 0)) for row in split_rows)),
            "eval_wall_sec": float(eval_wall_sec),
        }
        for key in metric_keys:
            values = [_finite_float(row.get(key)) for row in split_rows]
            values = [value for value in values if math.isfinite(value)]
            out[f"{key}_dataset_mean"] = float(np.mean(values)) if values else float("nan")
            out[f"{key}_dataset_var"] = float(np.var(values)) if values else float("nan")
            out[f"{key}_dataset_std"] = float(np.std(values)) if values else float("nan")
            out[f"{key}_dataset_min"] = float(np.min(values)) if values else float("nan")
            out[f"{key}_dataset_max"] = float(np.max(values)) if values else float("nan")
        write_csv_row(path, out)


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
    combined_eval: bool = False,
    combined_index_map_csv: Path | None = None,
    combined_eval_cache: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], float]:
    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    was_training = model.training
    model.eval()
    eval_specs = task_eval_specs(specs, task, max_generalization_eval)
    max_samples = normalize_max_samples(eval_max_samples)
    if combined_eval_cache is not None:
        results, map_rows = evaluate_combined_eval_cache(model, combined_eval_cache, device, eval_batch_size)
        if combined_index_map_csv is not None and not combined_index_map_csv.exists():
            for map_row in map_rows:
                write_csv_row(combined_index_map_csv, map_row)
    elif combined_eval and task == "darcy":
        results, map_rows = evaluate_datasets_combined(model, eval_specs, device, eval_batch_size, max_samples)
        if combined_index_map_csv is not None and not combined_index_map_csv.exists():
            for map_row in map_rows:
                write_csv_row(combined_index_map_csv, map_row)
    else:
        results = [evaluate_dataset(model, spec, device, eval_batch_size, max_samples) for spec in eval_specs]
    for result in results:
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


def checkpoint_model(
    model,
    out_dir: Path,
    task: str,
    epoch: int,
    global_step: int,
    cfg: dict[str, Any],
    *,
    optimizer: torch.optim.Optimizer | None = None,
    optimizer_global_step: int | None = None,
    work_clock_seconds: float | None = None,
    elapsed_wall_seconds: float | None = None,
    suffix: str | None = None,
) -> Path:
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    safe_suffix = ""
    if suffix:
        safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(suffix))
        safe_suffix = f"_{safe}"
    path = ckpt_dir / f"{task}_epoch{epoch:03d}_step{global_step:06d}{safe_suffix}.pt"
    payload = {
        "task": task,
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "config": cfg,
        "checkpoint_format": "model_optimizer_v1" if optimizer is not None else "model_only_legacy",
    }
    if optimizer_global_step is not None:
        payload["optimizer_global_step"] = int(optimizer_global_step)
    if work_clock_seconds is not None:
        payload["work_clock_seconds"] = float(work_clock_seconds)
    if elapsed_wall_seconds is not None:
        payload["elapsed_wall_seconds"] = float(elapsed_wall_seconds)
    if optimizer is not None:
        payload["optimizer_class"] = optimizer.__class__.__name__
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, path)
    return path


def load_model_checkpoint_state(model, checkpoint_path: Path, device: torch.device, optimizer=None) -> dict[str, Any]:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    optimizer_state = None
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
        optimizer_state = ckpt.get("optimizer_state_dict")
        metadata = {
            k: v
            for k, v in ckpt.items()
            if k not in {"model_state_dict", "optimizer_state_dict", "scheduler_state_dict"}
        }
    elif isinstance(ckpt, dict):
        state = ckpt
        metadata = {}
    else:
        raise TypeError(f"checkpoint is not a dict-like object: {checkpoint_path}")
    model.load_state_dict(state, strict=True)
    optimizer_restored = False
    if optimizer is not None and isinstance(ckpt, dict) and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        optimizer_restored = True
    return {
        "path": project_path(checkpoint_path),
        "metadata": to_jsonable(metadata),
        "optimizer_restored": optimizer_restored,
        "optimizer_global_step": int(ckpt.get("optimizer_global_step", 0)) if isinstance(ckpt, dict) else 0,
        "work_clock_seconds": float(ckpt.get("work_clock_seconds", 0.0)) if isinstance(ckpt, dict) else 0.0,
        "optimizer_state_dict": optimizer_state,
        "optimizer_state_present": optimizer_state is not None,
    }


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
    evals = int(full_epochs) + 1
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
            "evaluation_schedule": "every_epoch",
            "eval_every_fraction": args.eval_every_fraction,
            "checkpoint_every_epochs": args.checkpoint_every_epochs,
            "checkpoint_every_fraction": args.checkpoint_every_fraction,
            "checkpoint_wall_seconds": args.checkpoint_wall_seconds,
            "checkpoint_wall_hours": args.checkpoint_wall_hours,
            "eval_max_samples": args.eval_max_samples,
            "max_generalization_eval": args.max_generalization_eval,
            "combined_eval": bool(args.combined_eval),
            "attack_probe_samples": args.attack_probe_samples,
            "attack_probe_indices": args.attack_probe_indices,
            "attack_probe_every_n_epochs": args.attack_probe_every_n_epochs,
            "attack_probe_save_targets": bool(args.attack_probe_save_targets),
            "epsilon_bucket_count": int(args.epsilon_bucket_count),
            "binary_pool_multiplier": args.binary_pool_multiplier,
            "binary_score_noise": args.binary_score_noise,
            "ns_attack_frames": args.ns_attack_frames,
            "label_mode": args.label_mode,
            "training_data_mode": args.training_data_mode,
            "training_perturbation_mode": args.training_perturbation_mode,
            "random_field_target_mode": args.random_field_target_mode,
            "random_field_families": args.random_field_families,
            "random_field_gaussian_correlation_choices": parse_float_list(args.random_field_gaussian_correlation_choices),
            "random_field_matern_correlation_choices": parse_float_list(args.random_field_matern_correlation_choices),
            "random_field_matern_nu_choices": parse_float_list(args.random_field_matern_nu_choices),
            "random_field_domain_extent": args.random_field_domain_extent,
            "random_field_clip_x_min": args.random_field_clip_x_min,
            "random_field_clip_x_max": args.random_field_clip_x_max,
            "burgers_solver_remat": args.burgers_solver_remat,
            "burgers_solver_remat_chunk_steps": args.burgers_solver_remat_chunk_steps,
            "ns2d_solver_remat": args.ns2d_solver_remat,
            "ns2d_solver_remat_chunk_steps": args.ns2d_solver_remat_chunk_steps,
            "max_wall_seconds": args.max_wall_seconds,
            "max_work_seconds": args.max_work_seconds,
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
    override_attack_method = getattr(args, f"{task}_attack_method", None)
    if override_attack_method is not None:
        task_cfg["attack_method"] = str(override_attack_method)
    if task == "burgers":
        task_cfg["attack_loss_objective"] = str(getattr(args, "burgers_attack_loss_objective", "loss3"))
    if task == "darcy":
        task_cfg["attack_loss_objective"] = str(getattr(args, "darcy_attack_loss_objective", "loss3"))
        task_cfg["darcy_physics_metric"] = str(getattr(args, "darcy_physics_metric", "rel_l2"))
        task_cfg["darcy_physics_bc_weight"] = float(getattr(args, "darcy_physics_bc_weight", 1.0))
        task_cfg["darcy_physics_forcing_value"] = float(getattr(args, "darcy_physics_forcing_value", 1.0))
        task_cfg["darcy_loss1_random_start"] = bool(getattr(args, "darcy_loss1_random_start", True))
        task_cfg["darcy_loss1_random_start_fraction"] = float(getattr(args, "darcy_loss1_random_start_fraction", 1.0))
        task_cfg["darcy_random_source_kernels"] = str(getattr(args, "darcy_random_source_kernels", "gaussian,matern,highpass,bandpass,mixed"))
        task_cfg["darcy_random_source_alpha_values"] = str(getattr(args, "darcy_random_source_alpha_values", "1.2,2.2,3.2,4.2,5.2"))
        task_cfg["darcy_random_source_lengthscale_min"] = float(getattr(args, "darcy_random_source_lengthscale_min", 0.035))
        task_cfg["darcy_random_source_lengthscale_max"] = float(getattr(args, "darcy_random_source_lengthscale_max", 0.30))
        task_cfg["darcy_random_source_min_flip_fraction"] = float(getattr(args, "darcy_random_source_min_flip_fraction", 0.005))
        task_cfg["darcy_random_source_max_flip_fraction"] = float(getattr(args, "darcy_random_source_max_flip_fraction", 0.05))
    override_random_start_fraction = getattr(args, f"{task}_random_start_fraction", None)
    if override_random_start_fraction is not None:
        task_cfg["random_start_fraction"] = float(override_random_start_fraction)
    override_epsilon_fraction = getattr(args, f"{task}_epsilon_fraction", None)
    if override_epsilon_fraction is not None:
        task_cfg["epsilon_fraction"] = float(override_epsilon_fraction)
    override_epsilon_abs = getattr(args, f"{task}_epsilon_abs", None)
    if override_epsilon_abs is not None:
        task_cfg["epsilon_abs"] = float(override_epsilon_abs)
    override_alpha_ratio = getattr(args, f"{task}_alpha_ratio", None)
    if override_alpha_ratio is not None:
        task_cfg["alpha_ratio"] = float(override_alpha_ratio)
    override_alpha_jitter_low = getattr(args, f"{task}_alpha_jitter_low", None)
    if override_alpha_jitter_low is not None:
        task_cfg["alpha_jitter_low"] = float(override_alpha_jitter_low)
    override_alpha_jitter_high = getattr(args, f"{task}_alpha_jitter_high", None)
    if override_alpha_jitter_high is not None:
        task_cfg["alpha_jitter_high"] = float(override_alpha_jitter_high)
    override_eps_jitter_low = getattr(args, f"{task}_eps_jitter_low", None)
    if override_eps_jitter_low is not None:
        task_cfg["eps_jitter_low"] = float(override_eps_jitter_low)
    override_eps_jitter_high = getattr(args, f"{task}_eps_jitter_high", None)
    if override_eps_jitter_high is not None:
        task_cfg["eps_jitter_high"] = float(override_eps_jitter_high)
    override_optimizer_batch = getattr(args, f"{task}_optimizer_batch_size", None)
    if override_optimizer_batch is not None:
        task_cfg["optimizer_batch_size"] = int(override_optimizer_batch)

    checkpoint_wall_seconds = wall_clock_checkpoint_targets(
        task_cfg.get("checkpoint_wall_seconds"),
        task_cfg.get("checkpoint_wall_hours"),
    )
    task_cfg["checkpoint_wall_seconds"] = checkpoint_wall_seconds
    task_cfg["checkpoint_wall_hours"] = [seconds / 3600.0 for seconds in checkpoint_wall_seconds]
    initial_checkpoint = getattr(args, f"{task}_initial_checkpoint", None)
    resume_epoch_offset = int(args.resume_epoch_offset or 0)
    resume_global_step_offset = int(args.resume_global_step_offset or 0)
    if initial_checkpoint is None:
        resume_epoch_offset = 0
        resume_global_step_offset = 0
    task_cfg["initial_checkpoint"] = None if initial_checkpoint is None else str(initial_checkpoint)
    task_cfg["resume_epoch_offset"] = resume_epoch_offset
    task_cfg["resume_global_step_offset"] = resume_global_step_offset

    out_dir = out_root / task
    out_dir.mkdir(parents=True, exist_ok=True)

    model_checkpoint_override = getattr(args, f"{task}_model_checkpoint", None)
    if model_checkpoint_override is not None:
        model_checkpoint_override = Path(model_checkpoint_override)
    task_cfg["model_checkpoint_override"] = None if model_checkpoint_override is None else str(model_checkpoint_override.resolve())
    model = load_model(task, device, model_checkpoint_override)
    initial_checkpoint_info: dict[str, Any] | None = None
    if initial_checkpoint is not None:
        initial_checkpoint_info = load_model_checkpoint_state(model, Path(initial_checkpoint), device)
        task_cfg["initial_checkpoint"] = initial_checkpoint_info["path"]
        task_cfg["initial_checkpoint_metadata"] = initial_checkpoint_info["metadata"]
        metadata = initial_checkpoint_info.get("metadata") or {}
        inferred_epoch = int(metadata.get("epoch") or 0)
        inferred_global_step = int(metadata.get("global_step") or 0)
        inferred_optimizer_global_step = int(initial_checkpoint_info.get("optimizer_global_step") or 0)
        if resume_epoch_offset == 0 and inferred_epoch > 0:
            resume_epoch_offset = inferred_epoch
        if resume_global_step_offset == 0:
            resume_global_step_offset = inferred_optimizer_global_step or inferred_global_step
        task_cfg["resume_epoch_offset"] = resume_epoch_offset
        task_cfg["resume_global_step_offset"] = resume_global_step_offset
        task_cfg["initial_checkpoint_optimizer_state_present"] = bool(initial_checkpoint_info.get("optimizer_state_present"))
        task_cfg["initial_checkpoint_inferred_epoch"] = inferred_epoch
        task_cfg["initial_checkpoint_inferred_global_step"] = inferred_global_step
        task_cfg["initial_checkpoint_inferred_optimizer_global_step"] = inferred_optimizer_global_step
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(task_cfg["learning_rate"]),
        weight_decay=float(task_cfg["weight_decay"]),
    )
    resumed_work_clock_seconds = (
        float(initial_checkpoint_info.get("work_clock_seconds", 0.0) or 0.0)
        if initial_checkpoint_info is not None
        else 0.0
    )
    optimizer_state_loaded = False
    optimizer_load_error = ""
    if initial_checkpoint_info is not None and initial_checkpoint_info.get("optimizer_state_dict") is not None:
        try:
            optimizer.load_state_dict(initial_checkpoint_info["optimizer_state_dict"])
            for state in optimizer.state.values():
                for key, value in list(state.items()):
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(device)
            optimizer_state_loaded = True
        except Exception as exc:  # pragma: no cover - kept explicit for run metadata
            optimizer_load_error = repr(exc)
            raise RuntimeError(f"failed to load optimizer state from {initial_checkpoint_info['path']}: {exc}") from exc
    task_cfg["initial_checkpoint_optimizer_restored"] = optimizer_state_loaded
    task_cfg["optimizer_state_loaded_from_initial_checkpoint"] = optimizer_state_loaded
    task_cfg["optimizer_state_load_error"] = optimizer_load_error
    (out_dir / "config.json").write_text(json.dumps(to_jsonable({**run_cfg, **task_cfg}), indent=2), encoding="utf-8")

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
    local_epochs = int(task_cfg["epochs"])
    max_wall_seconds = task_cfg.get("max_wall_seconds")
    max_wall_seconds = None if max_wall_seconds is None else float(max_wall_seconds)
    max_work_seconds = task_cfg.get("max_work_seconds")
    max_work_seconds = None if max_work_seconds is None else float(max_work_seconds)
    total_epochs_for_progress = resume_epoch_offset + local_epochs
    total_steps = total_epochs_for_progress * max(1, batches_per_epoch)
    optimizer_steps_per_epoch = count_optimizer_steps_for_epoch(n_train, batch_size, optimizer_batch_size, max_batches)
    total_optimizer_steps = local_epochs * max(1, optimizer_steps_per_epoch)
    eval_every_steps = max(1, batches_per_epoch)
    checkpoint_every_epochs = int(task_cfg.get("checkpoint_every_epochs") or 0)
    checkpoint_fraction = task_cfg.get("checkpoint_every_fraction")
    if checkpoint_every_epochs > 0:
        checkpoint_every_steps = max(1, checkpoint_every_epochs * max(1, batches_per_epoch))
    elif checkpoint_fraction is not None and float(checkpoint_fraction) > 0:
        checkpoint_every_steps = max(1, int(math.ceil(total_steps * float(checkpoint_fraction))))
    else:
        checkpoint_every_steps = None
    probe_indices = select_attack_probe_indices(
        n_train,
        int(task_cfg.get("attack_probe_samples", 10)),
        task_cfg.get("attack_probe_indices"),
    )
    probe_index_to_rank = {int(source_index): rank for rank, source_index in enumerate(probe_indices.tolist())}
    probe_every_n_epochs = max(1, int(task_cfg.get("attack_probe_every_n_epochs", 1)))
    probe_save_targets = bool(task_cfg.get("attack_probe_save_targets", False))
    epsilon_bucket_count = max(0, int(task_cfg.get("epsilon_bucket_count", 5)))

    train_csv = out_dir / "train_steps.csv"
    attack_csv = out_dir / "attack_batches.csv"
    optimizer_csv = out_dir / "optimizer_steps.csv"
    eval_csv = out_dir / "eval_metrics.csv"
    eval_split_csv = out_dir / "eval_split_summary.csv"
    eval_pass_csv = out_dir / "evaluation_passes.csv"
    combined_eval_map_csv = out_dir / "combined_eval_dataset_index_map.csv"
    attack_epoch_csv = out_dir / "attack_epoch_summary.csv"
    attack_epsilon_bucket_csv = out_dir / "attack_epsilon_bucket_summary.csv"
    work_epoch_csv = out_dir / "work_clock_epoch_summary.csv"
    memory_csv = out_dir / "memory.csv"
    combined_eval_cache = None
    if bool(task_cfg.get("combined_eval", False)) and task == "darcy":
        combined_eval_specs = task_eval_specs(all_specs, task, task_cfg["max_generalization_eval"])
        combined_eval_cache = build_combined_eval_cache(
            combined_eval_specs,
            normalize_max_samples(task_cfg["eval_max_samples"]),
        )
        (out_dir / "combined_eval_cache_summary.json").write_text(
            json.dumps(
                to_jsonable(
                    {
                        "enabled": True,
                        "dataset_count": len(combined_eval_specs),
                        "total_samples": int(combined_eval_cache.get("total_samples", 0)),
                        "batch_size": int(task_cfg["eval_batch_size"]),
                        "index_map_csv": project_path(combined_eval_map_csv),
                    }
                ),
                indent=2,
            ),
            encoding="utf-8",
        )

    (out_dir / "attack_probe_config.json").write_text(
        json.dumps(
            to_jsonable(
                {
                    "enabled": bool(len(probe_indices) > 0),
                    "probe_indices": probe_indices.tolist(),
                    "probe_count": int(len(probe_indices)),
                    "save_every_n_epochs": probe_every_n_epochs,
                    "save_targets": probe_save_targets,
                    "epsilon_bucket_count": epsilon_bucket_count,
                    "epsilon_bucket_summary_csv": project_path(attack_epsilon_bucket_csv),
                    "note": "For each epoch, these fixed training source indices are captured from the actual attacked training batch. NPZ files contain x_clean, x_adv, delta=x_adv-x_clean, optional y_clean/y_adv targets, and task-specific tensors such as NS2D x0_clean/x0_adv/delta_initial. Epsilon bucket summaries aggregate per-sample clean loss, attacked loss, and loss increase by epsilon jitter bucket.",
                }
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    eval_seconds: list[float] = []
    initial_phase = "resume_checkpoint_before_adversarial_training" if initial_checkpoint is not None else "baseline_before_adversarial_training"
    initial_progress = resume_global_step_offset / max(1, total_steps)
    eval_rows, seconds = evaluate_task(
        model,
        task,
        all_specs,
        device,
        int(task_cfg["eval_batch_size"]),
        task_cfg["eval_max_samples"],
        eval_csv,
        global_step=resume_global_step_offset,
        epoch=resume_epoch_offset,
        progress_fraction=initial_progress,
        phase=initial_phase,
        max_generalization_eval=task_cfg["max_generalization_eval"],
        combined_eval=bool(task_cfg.get("combined_eval", False)),
        combined_index_map_csv=combined_eval_map_csv,
        combined_eval_cache=combined_eval_cache,
    )
    eval_seconds.append(seconds)
    write_eval_split_summary(
        eval_split_csv,
        eval_rows,
        phase=initial_phase,
        epoch=resume_epoch_offset,
        global_step=resume_global_step_offset,
        progress_fraction=initial_progress,
        eval_wall_sec=seconds,
    )
    write_csv_row(
        eval_pass_csv,
        {
            "task": task,
            "phase": initial_phase,
            "epoch": resume_epoch_offset,
            "global_step": resume_global_step_offset,
            "progress_fraction": initial_progress,
            "eval_wall_sec": seconds,
            "checkpoint_saved": 0,
            "checkpoint_path": "",
        },
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    metadata_for_resume = (initial_checkpoint_info or {}).get("metadata") or {}
    global_step = resume_global_step_offset
    optimizer_global_step = int(
        (initial_checkpoint_info or {}).get("optimizer_global_step")
        or metadata_for_resume.get("optimizer_global_step")
        or 0
    )
    optimizer_global_step_start = optimizer_global_step
    work_clock_resume_offset = float(
        (initial_checkpoint_info or {}).get("work_clock_seconds")
        or metadata_for_resume.get("work_clock_seconds")
        or 0.0
    )
    work_clock_seconds = work_clock_resume_offset
    train_rows: list[dict[str, Any]] = []
    seed_base = int(args.seed)
    train_start_wall = time.perf_counter()
    last_checkpoint_path: Path | None = None
    last_checkpoint_epoch = -1
    pending_wall_checkpoints = list(checkpoint_wall_seconds)
    saved_wall_checkpoints: list[dict[str, Any]] = []
    completed_local_epochs = 0
    stop_reason = "completed_configured_epochs"
    wall_stop_seconds = float("nan")
    cumulative_attack_seconds = 0.0
    cumulative_train_seconds = 0.0
    cumulative_optimizer_seconds = 0.0

    for local_epoch in range(1, local_epochs + 1):
        epoch = resume_epoch_offset + local_epoch
        probe_this_epoch = bool(probe_index_to_rank) and (epoch % probe_every_n_epochs == 0 or local_epoch == local_epochs)
        epoch_probe_records: list[dict[str, Any]] = []
        epoch_attack_rows: list[dict[str, Any]] = []
        epoch_attack_sample_infos: list[dict[str, torch.Tensor]] = []
        epoch_work_clock_seconds = 0.0
        epoch_attack_work_seconds = 0.0
        epoch_optimizer_work_seconds = 0.0
        generator = torch.Generator().manual_seed(seed_base + epoch * 1009 + TASK_SEED_OFFSETS.get(task, 0))
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
            if attack_result.sample_info:
                epoch_attack_sample_infos.append(attack_result.sample_info)
            attack_sec = time.perf_counter() - attack_start
            cumulative_attack_seconds += float(attack_sec)
            if probe_this_epoch:
                epoch_probe_records.extend(
                    collect_attack_probe_records(
                        model=model,
                        task=task,
                        epoch=epoch,
                        global_step=global_step,
                        local_batch_idx=local_batch_idx,
                        batch_indices=idx,
                        xb=xb,
                        yb=yb,
                        attack_result=attack_result,
                        cfg=task_cfg,
                        probe_index_to_rank=probe_index_to_rank,
                        save_targets=probe_save_targets,
                    )
                )
            x_train_adv, y_train_adv, training_mix_info = combine_training_pairs(task, xb, yb, attack_result, task_cfg)

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
            optimizer_sec_total = float(sum(optimizer_secs))
            cumulative_train_seconds += float(train_sec)
            cumulative_optimizer_seconds += optimizer_sec_total
            step_sec = time.perf_counter() - step_start
            step_work_sec = attack_sec + train_sec
            work_clock_seconds += step_work_sec
            epoch_work_clock_seconds += step_work_sec
            epoch_attack_work_seconds += attack_sec
            epoch_optimizer_work_seconds += train_sec
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
                "work_clock_step_seconds": step_work_sec,
                "work_clock_seconds": work_clock_seconds,
                "work_clock_attack_seconds": attack_sec,
                "work_clock_optimizer_seconds": train_sec,
                "attack_sec_per_sample": attack_sec / max(1, int(xb.shape[0])),
                "attack_samples_per_sec": int(xb.shape[0]) / max(attack_sec, 1e-12),
                "train_wall_sec": train_sec,
                "train_sec_per_sample": train_sec / max(1, int(xb.shape[0])),
                "step_sec_per_sample": step_sec / max(1, int(xb.shape[0])),
                "step_samples_per_sec": int(xb.shape[0]) / max(step_sec, 1e-12),
                "optimizer_wall_sec_mean": float(np.mean(optimizer_secs)) if optimizer_secs else float("nan"),
                "optimizer_wall_sec_total": optimizer_sec_total,
                "step_wall_sec": step_sec,
                "work_clock_sec": float(attack_sec) + float(train_sec),
                "cumulative_attack_wall_sec": cumulative_attack_seconds,
                "cumulative_train_wall_sec": cumulative_train_seconds,
                "cumulative_optimizer_wall_sec": cumulative_optimizer_seconds,
                "cumulative_work_clock_sec": work_clock_seconds,
                **memory_stats(device),
            }
            row.update(attack_info)
            write_csv_row(train_csv, row)
            write_csv_row(attack_csv, {k: row[k] for k in row if k.startswith("attack") or k.startswith("epsilon") or k.startswith("alpha") or k.startswith("delta") or k.startswith("darcy") or k.startswith("target") or k.startswith("solver") or k.startswith("x_train") or k in {"task", "epoch", "global_step", "label_mode", "boundary_ratio_mean", "clean_loss_before_attack", "adv_loss_after_attack", "clean_solver_mse_before_attack", "adv_solver_mse_after_attack"}})
            write_csv_row(memory_csv, {"task": task, "epoch": epoch, "global_step": global_step, **memory_stats(device)})
            train_rows.append(row)
            epoch_attack_rows.append(row)


        epoch_progress = global_step / max(1, total_steps)
        write_attack_epoch_summary(attack_epoch_csv, task, epoch, global_step, epoch_attack_rows)
        write_attack_epsilon_bucket_summary(
            attack_epsilon_bucket_csv,
            task,
            epoch,
            global_step,
            epoch_attack_sample_infos,
            task_cfg,
            epsilon_bucket_count,
        )
        write_csv_row(
            work_epoch_csv,
            {
                "task": task,
                "epoch": epoch,
                "local_epoch": local_epoch,
                "global_step": global_step,
                "optimizer_global_step": optimizer_global_step,
                "work_clock_epoch_seconds": epoch_work_clock_seconds,
                "work_clock_attack_seconds": epoch_attack_work_seconds,
                "work_clock_optimizer_seconds": epoch_optimizer_work_seconds,
                "work_clock_cumulative_seconds": work_clock_seconds,
                "work_clock_cumulative_minutes": work_clock_seconds / 60.0,
                "work_clock_resume_offset_seconds": work_clock_resume_offset,
                "attack_batches": len(epoch_attack_rows),
                "optimizer_steps_this_epoch": sum(int(row.get("optimizer_microbatches", 0) or 0) for row in epoch_attack_rows),
                "max_work_seconds": "" if max_work_seconds is None else max_work_seconds,
            },
        )
        if probe_this_epoch:
            save_attack_probe_epoch(
                out_dir,
                task,
                epoch,
                global_step,
                probe_indices,
                epoch_probe_records,
                save_targets=probe_save_targets,
            )

        is_final_step = local_epoch == local_epochs
        should_checkpoint = is_final_step or (checkpoint_every_steps is not None and global_step % checkpoint_every_steps == 0)
        ckpt: Path | None = None
        checkpoint_path = ""
        if should_checkpoint:
            ckpt = checkpoint_model(
                model,
                out_dir,
                task,
                epoch,
                global_step,
                task_cfg,
                optimizer=optimizer,
                optimizer_global_step=optimizer_global_step,
                work_clock_seconds=work_clock_seconds,
                elapsed_wall_seconds=time.perf_counter() - train_start_wall,
            )
            last_checkpoint_path = ckpt
            last_checkpoint_epoch = epoch
            checkpoint_path = project_path(ckpt)

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
            progress_fraction=epoch_progress,
            phase="during_adversarial_training",
            max_generalization_eval=task_cfg["max_generalization_eval"],
            combined_eval=bool(task_cfg.get("combined_eval", False)),
            combined_index_map_csv=combined_eval_map_csv,
            combined_eval_cache=combined_eval_cache,
        )
        eval_seconds.append(seconds)
        write_eval_split_summary(
            eval_split_csv,
            eval_rows,
            phase="during_adversarial_training",
            epoch=epoch,
            global_step=global_step,
            progress_fraction=epoch_progress,
            eval_wall_sec=seconds,
        )
        write_csv_row(
            eval_pass_csv,
            {
                "task": task,
                "phase": "during_adversarial_training",
                "epoch": epoch,
                "global_step": global_step,
                "progress_fraction": epoch_progress,
                "eval_wall_sec": seconds,
                "checkpoint_saved": int(should_checkpoint),
                "checkpoint_path": checkpoint_path,
                "work_clock_seconds": work_clock_seconds,
                "work_clock_minutes": work_clock_seconds / 60.0,
            },
        )
        wall_elapsed_after_eval = time.perf_counter() - train_start_wall
        if ckpt is not None:
            write_csv_row(
                out_dir / "checkpoints.csv",
                {
                    "task": task,
                    "epoch": epoch,
                    "global_step": global_step,
                    "progress_fraction": epoch_progress,
                    "checkpoint_path": checkpoint_path,
                    "eval_wall_sec": seconds,
                    "checkpoint_reason": "final" if is_final_step else "periodic",
                    "checkpoint_wall_target_seconds": "",
                    "wall_elapsed_seconds": wall_elapsed_after_eval,
                    "work_clock_seconds": work_clock_seconds,
                },
            )
        while pending_wall_checkpoints and wall_elapsed_after_eval >= pending_wall_checkpoints[0]:
            target_seconds = pending_wall_checkpoints.pop(0)
            wall_ckpt = checkpoint_model(
                model,
                out_dir,
                task,
                epoch,
                global_step,
                task_cfg,
                optimizer=optimizer,
                optimizer_global_step=optimizer_global_step,
                work_clock_seconds=work_clock_seconds,
                elapsed_wall_seconds=wall_elapsed_after_eval,
                suffix=wall_checkpoint_suffix(target_seconds),
            )
            wall_checkpoint_path = project_path(wall_ckpt)
            wall_row = {
                "task": task,
                "epoch": epoch,
                "global_step": global_step,
                "progress_fraction": epoch_progress,
                "checkpoint_path": wall_checkpoint_path,
                "eval_wall_sec": seconds,
                "checkpoint_reason": "wall_clock",
                "checkpoint_wall_target_seconds": target_seconds,
                "wall_elapsed_seconds": wall_elapsed_after_eval,
                "work_clock_seconds": work_clock_seconds,
            }
            write_csv_row(out_dir / "checkpoints.csv", wall_row)
            saved_wall_checkpoints.append(wall_row)
        completed_local_epochs = local_epoch
        if max_work_seconds is not None and work_clock_seconds >= max_work_seconds:
            stop_reason = "max_work_seconds_reached"
            wall_stop_seconds = wall_elapsed_after_eval
            write_csv_row(
                out_dir / "work_stop.csv",
                {
                    "task": task,
                    "epoch": epoch,
                    "global_step": global_step,
                    "progress_fraction": epoch_progress,
                    "max_work_seconds": max_work_seconds,
                    "work_clock_seconds": work_clock_seconds,
                    "wall_elapsed_seconds": wall_elapsed_after_eval,
                    "eval_wall_sec": seconds,
                    "stop_reason": stop_reason,
                },
            )
            break
        if max_wall_seconds is not None and wall_elapsed_after_eval >= max_wall_seconds:
            stop_reason = "max_wall_seconds_reached"
            wall_stop_seconds = wall_elapsed_after_eval
            write_csv_row(
                out_dir / "wall_stop.csv",
                {
                    "task": task,
                    "epoch": epoch,
                    "global_step": global_step,
                    "progress_fraction": epoch_progress,
                    "max_wall_seconds": max_wall_seconds,
                    "wall_elapsed_seconds": wall_elapsed_after_eval,
                    "eval_wall_sec": seconds,
                    "stop_reason": stop_reason,
                },
            )
            break

    final_epoch = resume_epoch_offset + completed_local_epochs
    if last_checkpoint_path is not None and last_checkpoint_epoch == final_epoch:
        final_ckpt = last_checkpoint_path
    else:
        final_ckpt = checkpoint_model(
            model,
            out_dir,
            task,
            final_epoch,
            global_step,
            task_cfg,
            optimizer=optimizer,
            optimizer_global_step=optimizer_global_step,
            work_clock_seconds=work_clock_seconds,
            elapsed_wall_seconds=time.perf_counter() - train_start_wall,
        )
    elapsed = time.perf_counter() - train_start_wall
    estimate = estimate_from_smoke(
        task,
        train_rows,
        eval_seconds,
        full_epochs=completed_local_epochs if completed_local_epochs > 0 else (DEFAULTS[task].epochs if args.epochs is None else int(args.epochs)),
        full_train_samples=n_train,
        batch_size=batch_size,
        optimizer_batch_size=optimizer_batch_size,
        eval_every_fraction=float(args.eval_every_fraction),
    )
    summary = {
        "task": task,
        "run_dir": project_path(out_dir),
        "train_samples": n_train,
        "train_max_samples": "full" if train_max_samples is None else train_max_samples,
        "batch_size": batch_size,
        "attack_batch_size": batch_size,
        "training_data_mode": str(task_cfg.get("training_data_mode", "adv-only")),
        "effective_train_samples_per_full_attack_batch": (2 * batch_size if str(task_cfg.get("training_data_mode", "adv-only")) == "clean-plus-adv" else batch_size),
        "optimizer_batch_size": optimizer_batch_size,
        "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
        "total_optimizer_steps": optimizer_global_step,
        "local_optimizer_steps": optimizer_global_step - optimizer_global_step_start,
        "total_optimizer_steps_planned": total_optimizer_steps,
        "epochs": completed_local_epochs,
        "epochs_configured": local_epochs,
        "stop_reason": stop_reason,
        "max_wall_seconds": max_wall_seconds,
        "max_work_seconds": max_work_seconds,
        "wall_stop_seconds": wall_stop_seconds,
        "work_clock_seconds": work_clock_seconds,
        "work_clock_minutes": work_clock_seconds / 60.0,
        "work_clock_resume_offset_seconds": work_clock_resume_offset,
        "local_work_clock_seconds": work_clock_seconds - work_clock_resume_offset,
        "local_work_clock_minutes": (work_clock_seconds - work_clock_resume_offset) / 60.0,
        "resume_epoch_offset": resume_epoch_offset,
        "resume_global_step_offset": resume_global_step_offset,
        "optimizer_global_step_start": optimizer_global_step_start,
        "optimizer_state_loaded_from_initial_checkpoint": optimizer_state_loaded,
        "total_epochs_for_progress": total_epochs_for_progress,
        "total_steps": global_step,
        "evaluation_schedule": "every_epoch",
        "eval_pass_count_including_baseline": int(len(eval_seconds)),
        "checkpoint_every_epochs": checkpoint_every_epochs,
        "checkpoint_every_fraction": None if checkpoint_fraction is None else float(checkpoint_fraction),
        "checkpoint_every_steps": checkpoint_every_steps,
        "checkpoint_wall_seconds": checkpoint_wall_seconds,
        "checkpoint_wall_hours": [seconds / 3600.0 for seconds in checkpoint_wall_seconds],
        "wall_clock_checkpoints": saved_wall_checkpoints,
        "attack_probe_indices": probe_indices.tolist(),
        "attack_probe_count": int(len(probe_indices)),
        "attack_probe_every_n_epochs": probe_every_n_epochs,
        "attack_probe_save_targets": probe_save_targets,
        "work_clock_epoch_summary_csv": project_path(work_epoch_csv),
        "elapsed_seconds": elapsed,
        "elapsed_minutes": elapsed / 60.0,
        "cumulative_attack_wall_sec": cumulative_attack_seconds,
        "cumulative_train_wall_sec": cumulative_train_seconds,
        "cumulative_optimizer_wall_sec": cumulative_optimizer_seconds,
        "resumed_work_clock_seconds": resumed_work_clock_seconds,
        "final_checkpoint": project_path(final_ckpt),
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
    parser.add_argument("--eval-every-fraction", type=float, default=0.2, help="Deprecated scheduling knob kept for old scripts; evaluation now runs every epoch.")
    parser.add_argument("--checkpoint-every-epochs", type=int, default=200, help="Save a model checkpoint every N epochs; default 200, plus final checkpoint.")
    parser.add_argument("--checkpoint-every-fraction", type=float, default=None, help="Legacy fallback: save checkpoints every this fraction of total training progress. Used only when --checkpoint-every-epochs <= 0.")
    parser.add_argument("--checkpoint-wall-seconds", default=None, help="Comma-separated elapsed wall-clock seconds. After an epoch/eval crosses each target, save an extra checkpoint with checkpoint_reason=wall_clock.")
    parser.add_argument("--checkpoint-wall-hours", default=None, help="Comma-separated elapsed wall-clock hours; converted to --checkpoint-wall-seconds targets.")
    parser.add_argument("--max-wall-seconds", type=float, default=None, help="Stop gracefully after the completed epoch/evaluation that first reaches this training wall-clock budget.")
    parser.add_argument("--max-work-seconds", type=float, default=None, help="Stop gracefully after the completed epoch whose attack/random/solver-target plus optimizer work reaches this budget. Evaluation, checkpointing, plotting, and upload are excluded.")
    parser.add_argument("--resume-epoch-offset", type=int, default=0, help="Epoch number represented by the loaded initial checkpoint; resumed epochs are logged after this offset.")
    parser.add_argument("--resume-global-step-offset", type=int, default=0, help="Global train-step number represented by the loaded initial checkpoint.")
    parser.add_argument("--eval-max-samples", type=int, default=0, help="Max samples per dataset during evaluation; default 0 evaluates the full dataset.")
    parser.add_argument("--max-generalization-eval", type=int, default=None)
    parser.add_argument("--combined-eval", action=argparse.BooleanOptionalAction, default=True, help="Evaluate same-shaped datasets as one combined tensor, then split metrics back by dataset index ranges. Currently used for Darcy.")
    parser.add_argument("--max-batches-per-epoch", type=int, default=None)
    parser.add_argument("--label-mode", choices=["solver", "clean"], default="solver")
    parser.add_argument("--training-data-mode", choices=["adv-only", "clean-only", "clean-plus-adv", "random-binary-fixed-y", "random-binary-solver-y"], default="adv-only", help="adv-only trains only on attacked solver pairs; clean-only trains on clean solver pairs without perturbation; clean-plus-adv doubles each attack batch with clean solver pairs plus attacked solver pairs; Darcy random-binary-fixed-y/random-binary-solver-y use random binary source flips instead of adversarial attacks.")
    parser.add_argument("--training-perturbation-mode", choices=["attack", "random-field"], default="attack", help="attack uses adversarially optimized delta; random-field samples a fresh Gaussian/Matern random-field delta without attack optimization.")
    parser.add_argument("--random-field-target-mode", choices=["clean-y", "solver-y"], default="solver-y", help="For --training-perturbation-mode random-field: clean-y keeps the original clean target fixed; solver-y recomputes the solver target at x+delta.")
    parser.add_argument("--random-field-families", default="gaussian,matern", help="Comma-separated random delta kernel families. Supported: gaussian,matern.")
    parser.add_argument("--random-field-gaussian-correlation-choices", default="0.015,0.03,0.06,0.12,0.24", help="Comma-separated Gaussian kernel correlation lengths sampled per training sample.")
    parser.add_argument("--random-field-matern-correlation-choices", default="0.015,0.03,0.06,0.12,0.24", help="Comma-separated Matern kernel correlation lengths sampled per training sample.")
    parser.add_argument("--random-field-matern-nu-choices", default="1.2,2.2,3.2,4.2,5.2", help="Comma-separated Matern smoothness nu values sampled per training sample.")
    parser.add_argument("--random-field-domain-extent", type=float, default=2.0, help="Physical domain extent used when converting FFT frequencies for random-field kernels.")
    parser.add_argument("--random-field-clip-x-min", type=float, default=None, help="Optional lower clamp for x+delta in random-field mode; omitted preserves exact RMS delta budget.")
    parser.add_argument("--random-field-clip-x-max", type=float, default=None, help="Optional upper clamp for x+delta in random-field mode; omitted preserves exact RMS delta budget.")
    parser.add_argument("--attack-probe-samples", type=int, default=5, help="Number of fixed train-set source indices whose attacked x_adv and delta are saved each probe epoch; set 0 to disable.")
    parser.add_argument("--attack-probe-indices", default=None, help="Optional comma-separated explicit train-set source indices for attack probe saving. Overrides --attack-probe-samples.")
    parser.add_argument("--attack-probe-every-n-epochs", type=int, default=1, help="Save attack probe arrays every N epochs; default 1 saves every epoch.")
    parser.add_argument("--attack-probe-save-targets", dest="attack_probe_save_targets", action="store_true", default=True, help="Save y_clean and y_adv arrays in attack probe NPZ files. On by default for same-index attack diagnostics.")
    parser.add_argument("--no-attack-probe-save-targets", dest="attack_probe_save_targets", action="store_false", help="Disable y_clean/y_adv arrays in attack probe NPZ files to reduce disk usage.")
    parser.add_argument("--epsilon-bucket-count", type=int, default=5, help="Number of per-epoch epsilon jitter buckets to summarize for clean loss, attacked loss, and loss increase; set 0 to disable.")
    parser.add_argument("--allow-clean-label", action="store_true", help="Allow the intentionally non-physical x_adv -> y_clean objective for ablation/debug only.")
    parser.add_argument("--ns-attack-frames", choices=["first", "all"], default="first")
    parser.add_argument("--burgers-solver-remat", choices=["none", "micro", "step", "chunk", "all"], default="none")
    parser.add_argument("--burgers-solver-remat-chunk-steps", type=int, default=20)
    parser.add_argument("--ns2d-solver-remat", choices=["none", "micro", "step", "chunk", "second"], default="chunk")
    parser.add_argument("--ns2d-solver-remat-chunk-steps", type=int, default=20)
    parser.add_argument("--binary-pool-multiplier", type=float, default=1.0, help="Darcy binary attack candidate pool multiplier. Default 1.0 is deterministic top-k for comparable same-index probes.")
    parser.add_argument("--binary-score-noise", type=float, default=0.0, help="Darcy binary attack score noise. Default 0.0 keeps same epsilon/model changes as the main source of probe variation.")
    parser.add_argument("--burgers-initial-checkpoint", type=Path, default=None, help="Optional Burgers checkpoint to load before adversarial training; used for resume/continuation runs.")
    parser.add_argument("--darcy-initial-checkpoint", type=Path, default=None, help="Optional Darcy checkpoint to load before adversarial training; used for resume/continuation runs with epoch/global-step offsets.")
    parser.add_argument("--darcy-model-checkpoint", type=Path, default=None, help="Optional Darcy checkpoint to load as the starting model when the official default checkpoint is unavailable or a screening baseline is intended.")
    parser.add_argument("--darcy-train-path", type=Path, default=None, help="Optional Darcy train dataset override path.")
    parser.add_argument("--darcy-test-path", type=Path, default=None, help="Optional Darcy test dataset override path.")
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
    parser.add_argument("--burgers-attack-method", choices=["fast_replace_l2", "fast_add_l2", "fast_replace_linf", "fast_add_linf"], default=None)
    parser.add_argument("--burgers-attack-loss-objective", choices=["loss1", "loss2", "loss3"], default="loss3", help="Burgers attack objective: loss1=MSE(model(x_adv), model(x_clean)); loss2=MSE(model(x_adv), solver(x_clean).detach()); loss3=MSE(model(x_adv), solver(x_adv)) with solver gradient.")
    parser.add_argument("--burgers-random-start-fraction", type=float, default=None, help="Optional Burgers attack random-start radius as a fraction of epsilon. Default keeps the task default; loss1 has zero gradient at exactly delta=0, so a tiny value is useful for a dedicated loss1 run.")
    parser.add_argument("--burgers-require-p2q2", action="store_true", help="Require Burgers continuous attack geometry to be p=2,q=2; rejects *_linf methods before training starts.")
    parser.add_argument("--darcy-attack-method", choices=["binary_steepest_replace"], default=None)
    parser.add_argument("--darcy-attack-loss-objective", choices=["loss1", "loss2", "loss3", "physics", "loss4", "loss4_physics"], default="loss3", help="Darcy binary attack objective. Training target remains solver(a_adv) for solver-label self-training.")
    parser.add_argument("--darcy-physics-metric", choices=["rel_l2", "mse"], default="rel_l2", help="Physics residual metric for --darcy-attack-loss-objective physics/loss4.")
    parser.add_argument("--darcy-physics-bc-weight", type=float, default=1.0, help="Boundary-condition penalty weight for Darcy physics loss.")
    parser.add_argument("--darcy-physics-forcing-value", type=float, default=1.0, help="Right-hand-side forcing value for Darcy physics residual.")
    parser.add_argument("--darcy-loss1-random-start", action=argparse.BooleanOptionalAction, default=True, help="For Darcy loss1, start from a random binary flip mask so the zero-distance loss1 gradient does not stall.")
    parser.add_argument("--darcy-loss1-random-start-fraction", type=float, default=1.0, help="Fraction of the per-sample flip budget used for the Darcy loss1 random initial mask.")
    parser.add_argument("--darcy-random-source-kernels", default="gaussian,matern,highpass,bandpass,mixed", help="Comma-separated kernels for Darcy random binary source training. Used by random-binary-fixed-y and random-binary-solver-y.")
    parser.add_argument("--darcy-random-source-alpha-values", default="1.2,2.2,3.2,4.2,5.2", help="Comma-separated alpha/smoothness values sampled per Darcy random-source sample.")
    parser.add_argument("--darcy-random-source-lengthscale-min", type=float, default=0.035, help="Minimum spectral lengthscale sampled per Darcy random-source sample.")
    parser.add_argument("--darcy-random-source-lengthscale-max", type=float, default=0.30, help="Maximum spectral lengthscale sampled per Darcy random-source sample.")
    parser.add_argument("--darcy-random-source-min-flip-fraction", type=float, default=0.005, help="Minimum fraction of binary coefficient pixels flipped per random-source sample.")
    parser.add_argument("--darcy-random-source-max-flip-fraction", type=float, default=0.05, help="Maximum fraction of binary coefficient pixels flipped per random-source sample; default enforces the requested 5 percent normalized energy budget.")
    parser.add_argument("--ns2d-attack-method", choices=["fast_replace_linf", "fast_add_linf"], default=None)
    parser.add_argument("--burgers-epsilon-fraction", type=float, default=None)
    parser.add_argument("--darcy-epsilon-fraction", type=float, default=None)
    parser.add_argument("--ns2d-epsilon-fraction", type=float, default=None)
    parser.add_argument("--burgers-epsilon-abs", type=float, default=None)
    parser.add_argument("--darcy-epsilon-abs", type=float, default=None)
    parser.add_argument("--ns2d-epsilon-abs", type=float, default=None)
    parser.add_argument("--burgers-alpha-ratio", type=float, default=None)
    parser.add_argument("--darcy-alpha-ratio", type=float, default=None)
    parser.add_argument("--ns2d-alpha-ratio", type=float, default=None)
    parser.add_argument("--burgers-alpha-jitter-low", type=float, default=None)
    parser.add_argument("--burgers-alpha-jitter-high", type=float, default=None)
    parser.add_argument("--darcy-alpha-jitter-low", type=float, default=None)
    parser.add_argument("--darcy-alpha-jitter-high", type=float, default=None)
    parser.add_argument("--ns2d-alpha-jitter-low", type=float, default=None)
    parser.add_argument("--ns2d-alpha-jitter-high", type=float, default=None)
    parser.add_argument("--burgers-eps-jitter-low", type=float, default=None)
    parser.add_argument("--burgers-eps-jitter-high", type=float, default=None)
    parser.add_argument("--darcy-eps-jitter-low", type=float, default=None)
    parser.add_argument("--darcy-eps-jitter-high", type=float, default=None)
    parser.add_argument("--ns2d-eps-jitter-low", type=float, default=None)
    parser.add_argument("--ns2d-eps-jitter-high", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.label_mode == "clean" and not args.allow_clean_label:
        raise ValueError("--label-mode clean is the old non-physical x_adv -> y_clean objective. Pass --allow-clean-label only for an explicit ablation.")
    if args.training_data_mode == "clean-plus-adv" and args.label_mode != "solver":
        raise ValueError("--training-data-mode clean-plus-adv requires --label-mode solver so both clean and attacked pairs use solver-generated Y.")
    if args.training_perturbation_mode == "random-field" and any(task.strip() != "burgers" for task in str(args.tasks).split(",") if task.strip()):
        raise ValueError("--training-perturbation-mode random-field is currently implemented for --tasks burgers only")
    if args.burgers_require_p2q2:
        burgers_method = args.burgers_attack_method or DEFAULTS["burgers"].attack_method
        if not str(burgers_method).endswith("_l2"):
            raise ValueError("--burgers-require-p2q2 requires --burgers-attack-method fast_replace_l2 or fast_add_l2; got " + str(burgers_method))
    set_seed(int(args.seed))
    tasks = parse_tasks(args.tasks)
    if args.training_data_mode in DARCY_RANDOM_SOURCE_TRAINING_MODES and tasks != ["darcy"]:
        raise ValueError("Darcy random source training modes require --tasks darcy")
    device = torch.device(args.device)
    run_name = args.run_name or ("smoke_" if args.smoke else "full_") + now_stamp()
    out_root = (args.output_root / run_name).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    all_specs = apply_dataset_path_overrides(build_specs(args.generalization_root.resolve()), args)
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
        "model_checkpoint_overrides": {"darcy": None if args.darcy_model_checkpoint is None else str(args.darcy_model_checkpoint.resolve())},
        "dataset_path_overrides": {
            "darcy_train_path": None if args.darcy_train_path is None else str(args.darcy_train_path.resolve()),
            "darcy_test_path": None if args.darcy_test_path is None else str(args.darcy_test_path.resolve()),
        },
        "preflight_dataset_counts": dataset_count_rows,
        "planned_workload_estimate": planned_estimate,
        "training_data_mode": args.training_data_mode,
        "training_perturbation_mode": args.training_perturbation_mode,
        "random_field_training": {
            "target_mode": args.random_field_target_mode,
            "families": args.random_field_families,
            "gaussian_correlation_choices": parse_float_list(args.random_field_gaussian_correlation_choices),
            "matern_correlation_choices": parse_float_list(args.random_field_matern_correlation_choices),
            "matern_nu_choices": parse_float_list(args.random_field_matern_nu_choices),
            "domain_extent": args.random_field_domain_extent,
            "clip_x_min": args.random_field_clip_x_min,
            "clip_x_max": args.random_field_clip_x_max,
        },
        "evaluation_schedule": "every_epoch",
        "combined_eval": bool(args.combined_eval),
        "eval_every_fraction_deprecated": args.eval_every_fraction,
        "checkpoint_every_epochs": int(args.checkpoint_every_epochs),
        "checkpoint_every_fraction": args.checkpoint_every_fraction,
        "checkpoint_wall_seconds": wall_clock_checkpoint_targets(args.checkpoint_wall_seconds, args.checkpoint_wall_hours),
        "max_wall_seconds": args.max_wall_seconds,
        "max_work_seconds": args.max_work_seconds,
        "darcy_attack_loss_objective": args.darcy_attack_loss_objective,
        "darcy_physics_loss": {
            "metric": args.darcy_physics_metric,
            "bc_weight": args.darcy_physics_bc_weight,
            "forcing_value": args.darcy_physics_forcing_value,
        },
        "darcy_loss1_random_start": {
            "enabled": args.darcy_loss1_random_start,
            "fraction": args.darcy_loss1_random_start_fraction,
        },
        "darcy_random_source": {
            "kernels": args.darcy_random_source_kernels,
            "alpha_values": args.darcy_random_source_alpha_values,
            "lengthscale_min": args.darcy_random_source_lengthscale_min,
            "lengthscale_max": args.darcy_random_source_lengthscale_max,
            "min_flip_fraction": args.darcy_random_source_min_flip_fraction,
            "max_flip_fraction": args.darcy_random_source_max_flip_fraction,
            "binary_constraint": "inputs are projected back to the per-sample two Darcy coefficient values before training",
        },
        "attack_probe": {
            "samples": int(args.attack_probe_samples),
            "indices": args.attack_probe_indices,
            "every_n_epochs": int(args.attack_probe_every_n_epochs),
            "save_targets": bool(args.attack_probe_save_targets),
        },
        "epsilon_bucket_count": int(args.epsilon_bucket_count),
        "remat_options": {
            "burgers_solver_remat": args.burgers_solver_remat,
            "burgers_solver_remat_chunk_steps": args.burgers_solver_remat_chunk_steps,
            "ns2d_solver_remat": args.ns2d_solver_remat,
            "ns2d_solver_remat_chunk_steps": args.ns2d_solver_remat_chunk_steps,
            "darcy_note": "Darcy Flow uses a JAX CG solver with implicit differentiation; this path is tuned by batch and optimizer microbatch rather than a time-rollout remat mode.",
        },
        "note": (
            "Online adversarial training. The default label mode is solver-label: "
            "for Burgers the attack objective can be --burgers-attack-loss-objective loss1/loss2/loss3. "
            "Loss1 uses no solver in the attack; loss2 uses solver(x_clean).detach() forward only; "
            "loss3 uses solver(x_adv) with solver forward/backward. Training still uses either "
            "model(x_adv) -> solver(x_adv) for --training-data-mode adv-only, "
            "model(x_clean) -> solver(x_clean) for --training-data-mode clean-only, "
            "or both model(x_clean) -> solver(x_clean) and model(x_adv) -> solver(x_adv) "
            "for --training-data-mode clean-plus-adv. For NS2D the attack variable is the "
            "initial vorticity frame and both model input frames and target frames "
            "are regenerated by solver rollout from the attacked initial state. "
            "The old clean-label x_adv -> y_clean objective requires the explicit "
            "--allow-clean-label ablation flag. The separate --training-perturbation-mode "
            "random-field path is a deliberate non-adversarial delta ablation: it samples "
            "Gaussian/Matern random-field perturbations and uses either clean-y or solver-y "
            "targets according to --random-field-target-mode."
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
        "run_dir": project_path(out_root),
        "tasks": summaries,
        "total_wall_seconds": total_wall,
        "total_wall_minutes": total_wall / 60.0,
        "finished_utc": now_stamp(),
    }
    (out_root / "summary.json").write_text(json.dumps(to_jsonable(final_summary), indent=2), encoding="utf-8")

    readme_lines = [
        "# Adversarial Training Run",
        "",
        f"- Run directory: `{project_path(out_root)}`",
        f"- Tasks: `{','.join(tasks)}`",
        f"- Smoke: `{bool(args.smoke)}`",
        f"- Label mode: `{args.label_mode}`",
        f"- Training data mode: `{args.training_data_mode}`",
        f"- Burgers attack loss objective: `{args.burgers_attack_loss_objective}`",
        f"- Darcy attack loss objective: `{args.darcy_attack_loss_objective}`",
        f"- Max wall seconds: `{args.max_wall_seconds}`",
        f"- Max work seconds: `{args.max_work_seconds}`",
        "",
        "Each task subdirectory contains `train_steps.csv`, `attack_batches.csv`, `attack_epoch_summary.csv`, `attack_epsilon_bucket_summary.csv`, `work_clock_epoch_summary.csv`, `optimizer_steps.csv`, `eval_metrics.csv`, `eval_split_summary.csv`, `evaluation_passes.csv`, `memory.csv`, checkpoints, `attack_probe_samples.csv`, `attack_probe_epochs.csv`, `attack_probe_samples/*.npz`, `data_range_summary.json`, and `summary.json`.",
        "Training data modes: `adv-only` uses only perturbed solver/target pairs; `clean-only` uses clean solver pairs with zero perturbation; `clean-plus-adv` trains each batch on clean solver pairs plus newly perturbed pairs, doubling the training examples per perturbation batch. Darcy-only `random-binary-fixed-y` and `random-binary-solver-y` replace adversarial attacks with random binary coefficient flips; fixed-y keeps the clean target, solver-y recomputes solver(a_random).",
        "Perturbation modes: `attack` optimizes delta adversarially; Burgers-only `random-field` samples fresh Gaussian/Matern random-field deltas without attack optimization and uses `--random-field-target-mode clean-y` or `solver-y` for the target.",
        "Default training now uses the full original train split for every epoch; pass `--<task>-train-max N` only for debugging caps, or `0` for full.",
        "",
        "Default attack policy:",
        "- Burgers: p=2/q=2 RMS-L2 fast-replace attack in the corrected pipeline. The attack objective can be loss1, loss2, or loss3: loss1 uses no solver; loss2 uses fixed clean solver output without solver backward; loss3 uses attacked solver output with solver backward. Optimizer training remains on attacked solver pairs unless training-data-mode is changed.",
        "- Darcy: binary steepest-replace flips coefficient pixels; default attack batch is 256, optimizer microbatch is 32, flip budget is fixed by epsilon, score noise is off, and top-k replacement is deterministic by default. The attack objective can be loss1, loss2, loss3, or physics/loss4; optimizer training still uses attacked solver pairs unless training-data-mode is changed. Random-source Darcy modes sample Gaussian/Matern/high-pass/band-pass/mixed fields, flip only between the two binary coefficient values, and cap the normalized mean flip energy at `--darcy-random-source-max-flip-fraction`.",
        "- NS2D: L-infinity add attack on the initial vorticity frame; attack batch and optimizer batch stay 1:1 by default, epsilon/alpha are fixed, and random start is off for comparable same-index probes.",
        "",
        "Evaluation is clean evaluation on train/test/generated datasets at baseline and after every epoch, giving 52 dataset-level curves per task when all 50 generated sets are present; checkpoints default to every 200 epochs plus final.",
        "`attack_epsilon_bucket_summary.csv` writes one row per epsilon bucket per epoch. With the default 5 buckets, the random epsilon jitter range is split into 0-20%, 20-40%, 40-60%, 60-80%, and 80-100% bands, each with clean loss, attacked loss, loss increase, and relative loss increase statistics.",
        "Attack probes save fixed train-set source indices after their actual training attack each probe epoch, including x_clean, x_adv, delta, y_clean/y_adv targets by default, per-sample clean/adv attack loss gain, and delta high-frequency summary metrics. NS2D probes also save x0_clean, x0_adv, and delta_initial.",
    ]
    (out_root / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
    print(f"[done] wrote {out_root}", flush=True)


if __name__ == "__main__":
    main()
