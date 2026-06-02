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
        eval_batch_size=256,
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
        # Exact NS2D solver-gradient rollout is very memory heavy; batch 2 OOMed on the 31.7GB GPU.
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
    relative_cpu = gain_cpu / clean_cpu.abs().clamp_min(1e-20)
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
            x_clean_loss, y_clean_loss = selected_clean_pair_for_probe(task, xb.index_select(0, pos_tensor), yb.index_select(0, pos_tensor), cfg)
            x_adv_loss = attack_result.x_train.index_select(0, pos_tensor)
            y_adv_loss = attack_result.y_train.index_select(0, pos_tensor)
            clean_losses = per_sample_finite_mse(model(x_clean_loss), y_clean_loss).detach().cpu().tolist()
            adv_losses = per_sample_finite_mse(model(x_adv_loss), y_adv_loss).detach().cpu().tolist()
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
            "attack_loss_gain_relative_sample": (adv_loss_sample - clean_loss_sample) / max(abs(clean_loss_sample), 1e-20),
            **attack_probe_delta_stats(delta),
        }
        for extra_key, extra_tensor in attack_result.probe_tensors.items():
            if not isinstance(extra_tensor, torch.Tensor) or int(extra_tensor.shape[0]) != len(source_indices):
                continue
            record[extra_key] = tensor_to_probe_array(extra_tensor[batch_pos])
        for key in (
            "attack_type",
            "attack_method",
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
            adv_pred = model(x_adv)
            adv_loss_samples = per_sample_finite_mse(adv_pred, y_train)
            final_loss_value = float(finite_mse(adv_pred, y_train).detach().cpu())
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
        "target_source": str(cfg.get("label_mode", "solver")),
        "full_solver_gradient": True,
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

    first_loss = float("nan")
    positive_score_frac = float("nan")
    flips_total = 0
    clean_loss_samples = torch.full((b,), float("nan"), device=x0.device, dtype=x0.dtype)
    adv_loss_samples = torch.full((b,), float("nan"), device=x0.device, dtype=x0.dtype)

    try:
        with torch.no_grad():
            clean_target = solver_target_for_model_input("darcy", x0, yb, cfg, allow_target_grad=False)
            clean_pred = model(x0)
            clean_loss_samples = per_sample_finite_mse(clean_pred, clean_target)
            clean_loss = finite_mse(clean_pred, clean_target)

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
            adv_pred = model(x_adv)
            adv_loss_samples = per_sample_finite_mse(adv_pred, y_train)
            adv_loss = finite_mse(adv_pred, y_train)
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
    override_attack_method = getattr(args, f"{task}_attack_method", None)
    if override_attack_method is not None:
        cfg["attack_method"] = str(override_attack_method)
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
    attack_wall_sec_total = sum(_finite_float(row.get("attack_wall_sec")) for row in rows if math.isfinite(_finite_float(row.get("attack_wall_sec"))))
    row = {
        "task": task,
        "epoch": int(epoch),
        "global_step_last": int(global_step),
        "attack_batches": int(len(rows)),
        "attack_samples": int(attack_samples),
        "clean_loss_before_attack_mean": clean_mean,
        "adv_loss_after_attack_mean": adv_mean,
        "attack_loss_gain_mean": gain_mean,
        "attack_loss_gain_relative_mean": gain_mean / max(abs(clean_mean), 1e-20) if math.isfinite(gain_mean) and math.isfinite(clean_mean) else float("nan"),
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
) -> tuple[list[dict[str, Any]], float]:
    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    was_training = model.training
    model.eval()
    for spec in task_eval_specs(specs, task, max_generalization_eval):
        max_samples = normalize_max_samples(eval_max_samples)
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
            "eval_max_samples": args.eval_max_samples,
            "max_generalization_eval": args.max_generalization_eval,
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
    override_attack_method = getattr(args, f"{task}_attack_method", None)
    if override_attack_method is not None:
        task_cfg["attack_method"] = str(override_attack_method)
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
    attack_epoch_csv = out_dir / "attack_epoch_summary.csv"
    attack_epsilon_bucket_csv = out_dir / "attack_epsilon_bucket_summary.csv"
    memory_csv = out_dir / "memory.csv"

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
    write_eval_split_summary(
        eval_split_csv,
        eval_rows,
        phase="baseline_before_adversarial_training",
        epoch=0,
        global_step=0,
        progress_fraction=0.0,
        eval_wall_sec=seconds,
    )
    write_csv_row(
        eval_pass_csv,
        {
            "task": task,
            "phase": "baseline_before_adversarial_training",
            "epoch": 0,
            "global_step": 0,
            "progress_fraction": 0.0,
            "eval_wall_sec": seconds,
            "checkpoint_saved": 0,
            "checkpoint_path": "",
        },
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    global_step = 0
    optimizer_global_step = 0
    train_rows: list[dict[str, Any]] = []
    seed_base = int(args.seed)
    train_start_wall = time.perf_counter()
    last_checkpoint_path: Path | None = None
    last_checkpoint_epoch = -1

    for epoch in range(1, int(task_cfg["epochs"]) + 1):
        probe_this_epoch = bool(probe_index_to_rank) and (epoch % probe_every_n_epochs == 0 or epoch == int(task_cfg["epochs"]))
        epoch_probe_records: list[dict[str, Any]] = []
        epoch_attack_rows: list[dict[str, Any]] = []
        epoch_attack_sample_infos: list[dict[str, torch.Tensor]] = []
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

        is_final_step = global_step == total_steps
        should_checkpoint = is_final_step or (checkpoint_every_steps is not None and global_step % checkpoint_every_steps == 0)
        ckpt: Path | None = None
        checkpoint_path = ""
        if should_checkpoint:
            ckpt = checkpoint_model(model, out_dir, task, epoch, global_step, task_cfg)
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
            },
        )
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
                },
            )

    if last_checkpoint_path is not None and last_checkpoint_epoch == int(task_cfg["epochs"]):
        final_ckpt = last_checkpoint_path
    else:
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
        "run_dir": project_path(out_dir),
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
        "evaluation_schedule": "every_epoch",
        "eval_pass_count_including_baseline": int(len(eval_seconds)),
        "checkpoint_every_epochs": checkpoint_every_epochs,
        "checkpoint_every_fraction": None if checkpoint_fraction is None else float(checkpoint_fraction),
        "checkpoint_every_steps": checkpoint_every_steps,
        "attack_probe_indices": probe_indices.tolist(),
        "attack_probe_count": int(len(probe_indices)),
        "attack_probe_every_n_epochs": probe_every_n_epochs,
        "attack_probe_save_targets": probe_save_targets,
        "elapsed_seconds": elapsed,
        "elapsed_minutes": elapsed / 60.0,
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
    parser.add_argument("--eval-max-samples", type=int, default=0, help="Max samples per dataset during evaluation; default 0 evaluates the full dataset.")
    parser.add_argument("--max-generalization-eval", type=int, default=None)
    parser.add_argument("--max-batches-per-epoch", type=int, default=None)
    parser.add_argument("--label-mode", choices=["solver", "clean"], default="solver")
    parser.add_argument("--training-data-mode", choices=["adv-only", "clean-plus-adv"], default="adv-only", help="adv-only trains only on attacked solver pairs; clean-plus-adv doubles each attack batch with clean solver pairs plus attacked solver pairs.")
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
    parser.add_argument("--burgers-require-p2q2", action="store_true", help="Require Burgers continuous attack geometry to be p=2,q=2; rejects *_linf methods before training starts.")
    parser.add_argument("--darcy-attack-method", choices=["binary_steepest_replace"], default=None)
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
    if args.burgers_require_p2q2:
        burgers_method = args.burgers_attack_method or DEFAULTS["burgers"].attack_method
        if not str(burgers_method).endswith("_l2"):
            raise ValueError("--burgers-require-p2q2 requires --burgers-attack-method fast_replace_l2 or fast_add_l2; got " + str(burgers_method))
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
        "evaluation_schedule": "every_epoch",
        "eval_every_fraction_deprecated": args.eval_every_fraction,
        "checkpoint_every_epochs": int(args.checkpoint_every_epochs),
        "checkpoint_every_fraction": args.checkpoint_every_fraction,
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
        "",
        "Each task subdirectory contains `train_steps.csv`, `attack_batches.csv`, `attack_epoch_summary.csv`, `attack_epsilon_bucket_summary.csv`, `optimizer_steps.csv`, `eval_metrics.csv`, `eval_split_summary.csv`, `evaluation_passes.csv`, `memory.csv`, checkpoints, `attack_probe_samples.csv`, `attack_probe_epochs.csv`, `attack_probe_samples/*.npz`, `data_range_summary.json`, and `summary.json`.",
        "Training data modes: `adv-only` uses only attacked solver pairs; `clean-plus-adv` trains each batch on clean solver pairs plus newly attacked solver pairs, doubling the training examples per attack batch.",
        "Default training now uses the full original train split for every epoch; pass `--<task>-train-max N` only for debugging caps, or `0` for full.",
        "",
        "Default attack policy:",
        "- Burgers: short L-infinity fast-replace attack; default attack batch is 256, optimizer microbatch is 32, epsilon is fixed per sample from input range, alpha=epsilon*ratio, and random start is off for comparable same-index probes.",
        "- Darcy: binary steepest-replace flips coefficient pixels; default attack batch is 256, optimizer microbatch is 32, flip budget is fixed by epsilon, score noise is off, and top-k replacement is deterministic by default.",
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
