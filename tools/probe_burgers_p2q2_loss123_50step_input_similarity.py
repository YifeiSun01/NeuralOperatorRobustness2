#!/usr/bin/env python3
"""Replay early Burgers adversarial training and track input-manifold similarity.

This probe starts every variant from the same retained Burgers FNO baseline,
uses the same p=2/q=2 attack geometry as the saved p2q2 training runs, and
records the first N attack-batch updates.  It is intended to test whether the
loss3 epoch-1 jump is tied to off-range/off-manifold attacked inputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import types
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import matplotlib  # noqa: F401
except ModuleNotFoundError:
    matplotlib_stub = types.ModuleType("matplotlib")
    matplotlib_stub.use = lambda *args, **kwargs: None
    pyplot_stub = types.ModuleType("matplotlib.pyplot")
    cm_stub = types.ModuleType("matplotlib.cm")
    patches_stub = types.ModuleType("matplotlib.patches")
    cm_stub.get_cmap = lambda *args, **kwargs: None
    patches_stub.Patch = object
    sys.modules["matplotlib"] = matplotlib_stub
    sys.modules["matplotlib.pyplot"] = pyplot_stub
    sys.modules["matplotlib.cm"] = cm_stub
    sys.modules["matplotlib.patches"] = patches_stub

from tools.adversarial_training import (  # noqa: E402
    DEFAULTS,
    TASK_SEED_OFFSETS,
    attack_batch,
    finite_mse,
    load_model,
    load_train_xy,
    make_indices,
    make_slices,
    normalize_max_samples,
    safe_grad_norm,
    solver_target_for_model_input,
    task_train_spec,
)
from tools.evaluate_generalization_models import (  # noqa: E402
    DatasetSpec,
    build_specs,
    tensor_xy,
    torch_load,
)


VARIANTS = [
    {"variant": "loss1_raw", "attack_objective": "loss1", "transform": "raw"},
    {"variant": "loss2_raw", "attack_objective": "loss2", "transform": "raw"},
    {"variant": "loss3_raw", "attack_objective": "loss3", "transform": "raw"},
    {"variant": "loss3_clip01", "attack_objective": "loss3", "transform": "clip01"},
    {"variant": "loss3_lowpass", "attack_objective": "loss3", "transform": "lowpass"},
    {"variant": "loss3_lowpass_clip01", "attack_objective": "loss3", "transform": "lowpass_clip01"},
]


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    if exists:
        with path.open("r", newline="", encoding="utf-8") as f:
            old_keys = next(csv.reader(f), [])
    else:
        old_keys = []
    keys = list(old_keys)
    for key in row:
        if key not in keys:
            keys.append(key)
    if exists and keys != old_keys:
        old_rows: list[dict[str, Any]] = []
        with path.open("r", newline="", encoding="utf-8") as f:
            old_rows.extend(csv.DictReader(f))
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(old_rows)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def finite_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def split_summary(rows: list[dict[str, Any]], *, variant: str, step: int, phase: str, elapsed: float) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {"ALL": rows}
    for row in rows:
        groups.setdefault(str(row["split"]), []).append(row)
    out_rows: list[dict[str, Any]] = []
    for split, split_rows in groups.items():
        out: dict[str, Any] = {
            "variant": variant,
            "global_step": step,
            "phase": phase,
            "split": split,
            "dataset_count": len(split_rows),
            "total_samples_evaluated": int(sum(int(row["num_samples_evaluated"]) for row in split_rows)),
            "eval_wall_sec": elapsed,
        }
        for key in ("rmse", "mae", "relative_l2", "accuracy_score"):
            values = [finite_float(row[key]) for row in split_rows]
            values = [value for value in values if math.isfinite(value)]
            out[f"{key}_dataset_mean"] = float(np.mean(values)) if values else float("nan")
            out[f"{key}_dataset_min"] = float(np.min(values)) if values else float("nan")
            out[f"{key}_dataset_max"] = float(np.max(values)) if values else float("nan")
        out_rows.append(out)
    return out_rows


def load_eval_sets(
    specs: list[DatasetSpec],
    *,
    task: str,
    max_generalization_eval: int | None,
    eval_max_samples: int | None,
) -> list[dict[str, Any]]:
    selected = [s for s in specs if s.task == task and s.split in {"train", "test"}]
    gen = [s for s in specs if s.task == task and s.split == "generalization"]
    gen.sort(key=lambda s: (s.manual_rank, s.dataset_id))
    if max_generalization_eval is not None:
        gen = gen[: max(0, int(max_generalization_eval))]
    selected.extend(gen)
    out: list[dict[str, Any]] = []
    for spec in selected:
        data = torch_load(spec.path)
        x, y = tensor_xy(data, task)
        cap = normalize_max_samples(eval_max_samples)
        if cap is not None:
            x = x[:cap].contiguous()
            y = y[:cap].contiguous()
        out.append({"spec": spec, "x": x.contiguous(), "y": y.contiguous()})
    return out


def evaluate_cached_dataset(model, item: dict[str, Any], device: torch.device, batch_size: int) -> dict[str, Any]:
    spec: DatasetSpec = item["spec"]
    x: torch.Tensor = item["x"]
    y: torch.Tensor = item["y"]
    n = int(x.shape[0])
    sse = 0.0
    sae = 0.0
    target_sse = 0.0
    count = 0
    invalid_count = 0
    total_count = 0
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            xb = x[start : start + batch_size].to(device, non_blocking=True)
            yb = y[start : start + batch_size].to(device, non_blocking=True)
            pred = model(xb)
            finite = torch.isfinite(pred) & torch.isfinite(yb)
            invalid_count += int(pred.numel() - finite.sum().detach().cpu())
            total_count += int(pred.numel())
            if finite.any():
                diff = pred[finite] - yb[finite]
                sse += float((diff * diff).sum().detach().cpu())
                sae += float(diff.abs().sum().detach().cpu())
                target_sse += float((yb[finite] * yb[finite]).sum().detach().cpu())
                count += int(diff.numel())
    if was_training:
        model.train()
    rmse = math.sqrt(sse / max(1, count)) if count else float("nan")
    mae = sae / max(1, count) if count else float("nan")
    rel_l2 = math.sqrt(sse / max(target_sse, 1e-20)) if count else float("nan")
    score = 100.0 / (1.0 + rel_l2) if math.isfinite(rel_l2) else float("nan")
    return {
        "task": spec.task,
        "dataset_id": spec.dataset_id,
        "split": spec.split,
        "source": spec.source,
        "manual_tier": spec.manual_tier,
        "manual_rank": spec.manual_rank,
        "num_samples_evaluated": n,
        "rmse": rmse,
        "mae": mae,
        "relative_l2": rel_l2,
        "accuracy_score": score,
        "finite_value_count": count,
        "invalid_value_count": invalid_count,
        "invalid_value_fraction": invalid_count / max(1, total_count),
    }


def evaluate_cached(model, eval_sets: list[dict[str, Any]], device: torch.device, batch_size: int) -> tuple[list[dict[str, Any]], float]:
    start = time.perf_counter()
    rows = [evaluate_cached_dataset(model, item, device, batch_size) for item in eval_sets]
    return rows, time.perf_counter() - start


def flatten_x(x: torch.Tensor) -> torch.Tensor:
    return x.detach().float().reshape(x.shape[0], -1)


def spectrum_1d(x: torch.Tensor) -> torch.Tensor:
    arr = x.detach().float()
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    arr = arr - arr.mean(dim=1, keepdim=True)
    power = torch.fft.rfft(arr, dim=1).abs().pow(2).mean(dim=0)
    return power / power.sum().clamp_min(1e-20)


def x_feature_stats(x: torch.Tensor) -> dict[str, float]:
    flat = flatten_x(x)
    valid = torch.nan_to_num(flat)
    sample_min = valid.min(dim=1).values
    sample_max = valid.max(dim=1).values
    sample_range = sample_max - sample_min
    sample_rms = torch.sqrt(valid.pow(2).mean(dim=1).clamp_min(1e-24))
    arr = x.detach().float()
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    diff = arr[:, 1:] - arr[:, :-1]
    power = spectrum_1d(x)
    n_freq = int(power.numel())
    low_end = max(1, n_freq // 16)
    mid_end = max(low_end + 1, n_freq // 4)
    freq = torch.linspace(0.0, 1.0, n_freq, device=power.device)
    return {
        "x_min": float(valid.min().detach().cpu()),
        "x_max": float(valid.max().detach().cpu()),
        "x_mean": float(valid.mean().detach().cpu()),
        "x_std": float(valid.std(unbiased=False).detach().cpu()),
        "x_rms_mean": float(sample_rms.mean().detach().cpu()),
        "x_sample_range_mean": float(sample_range.mean().detach().cpu()),
        "x_sample_range_max": float(sample_range.max().detach().cpu()),
        "x_total_variation_mean": float(diff.abs().mean().detach().cpu()),
        "x_low_freq_frac": float(power[:low_end].sum().detach().cpu()),
        "x_mid_freq_frac": float(power[low_end:mid_end].sum().detach().cpu()),
        "x_high_freq_frac": float(power[mid_end:].sum().detach().cpu()),
        "x_spectral_centroid": float((power * freq).sum().detach().cpu()),
    }


def build_reference_banks(
    eval_sets: list[dict[str, Any]],
    *,
    device: torch.device,
    ref_samples: int,
    seed: int,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[torch.Tensor]] = {}
    for item in eval_sets:
        split = str(item["spec"].split)
        grouped.setdefault(split, []).append(item["x"])
    banks: dict[str, dict[str, Any]] = {}
    generator = torch.Generator().manual_seed(seed)
    for split, tensors in grouped.items():
        x = torch.cat(tensors, dim=0).contiguous()
        n = int(x.shape[0])
        if n > ref_samples:
            idx = torch.randperm(n, generator=generator)[:ref_samples]
            x_ref = x[idx].contiguous()
        else:
            x_ref = x
        flat = flatten_x(x_ref).to(device)
        flat_norm = flat / flat.norm(dim=1, keepdim=True).clamp_min(1e-12)
        stats = x_feature_stats(x.to(device))
        spec = spectrum_1d(x.to(device))
        banks[split] = {
            "x_ref": x_ref.to(device),
            "flat": flat,
            "flat_norm": flat_norm,
            "stats": stats,
            "spectrum": spec,
            "sample_count": n,
            "ref_sample_count": int(x_ref.shape[0]),
        }
    return banks


def stage_similarity_rows(
    *,
    variant: str,
    step: int,
    input_stage: str,
    x_stage: torch.Tensor,
    ref_banks: dict[str, dict[str, Any]],
    query_samples: int,
) -> list[dict[str, Any]]:
    n = int(x_stage.shape[0])
    if n > query_samples:
        idx = torch.linspace(0, n - 1, query_samples, device=x_stage.device).round().long()
        x_query = x_stage[idx]
    else:
        x_query = x_stage
    q_flat = flatten_x(x_query).to(x_stage.device)
    q_norm = q_flat / q_flat.norm(dim=1, keepdim=True).clamp_min(1e-12)
    q_spec = spectrum_1d(x_stage)
    q_stats = x_feature_stats(x_stage)
    rows: list[dict[str, Any]] = []
    dim_sqrt = math.sqrt(max(1, int(q_flat.shape[1])))
    for split, bank in ref_banks.items():
        ref_flat = bank["flat"]
        ref_norm = bank["flat_norm"]
        cos = q_norm @ ref_norm.T
        max_cos = cos.max(dim=1).values.clamp(-1.0, 1.0)
        dist = torch.cdist(q_flat, ref_flat) / dim_sqrt
        nn_rms = dist.min(dim=1).values
        ref_stats = bank["stats"]
        ref_spec = bank["spectrum"]
        spec_cos = float((q_spec * ref_spec).sum().div(q_spec.norm().clamp_min(1e-20) * ref_spec.norm().clamp_min(1e-20)).detach().cpu())
        spec_l1 = float((q_spec - ref_spec).abs().sum().mul(0.5).detach().cpu())
        below_ref = float((x_stage < float(ref_stats["x_min"])).float().mean().detach().cpu())
        above_ref = float((x_stage > float(ref_stats["x_max"])).float().mean().detach().cpu())
        outside_amount = torch.clamp(float(ref_stats["x_min"]) - x_stage, min=0) + torch.clamp(x_stage - float(ref_stats["x_max"]), min=0)
        row = {
            "variant": variant,
            "global_step": step,
            "input_stage": input_stage,
            "reference_split": split,
            "query_sample_count": int(x_query.shape[0]),
            "reference_total_samples": int(bank["sample_count"]),
            "reference_sample_count": int(bank["ref_sample_count"]),
            "nearest_rms_mean": float(nn_rms.mean().detach().cpu()),
            "nearest_rms_min": float(nn_rms.min().detach().cpu()),
            "nearest_rms_max": float(nn_rms.max().detach().cpu()),
            "max_cosine_mean": float(max_cos.mean().detach().cpu()),
            "max_cosine_min": float(max_cos.min().detach().cpu()),
            "max_cosine_angle_deg_mean": float(torch.rad2deg(torch.acos(max_cos)).mean().detach().cpu()),
            "spectrum_cosine": spec_cos,
            "spectrum_l1_distance": spec_l1,
            "below_reference_min_frac": below_ref,
            "above_reference_max_frac": above_ref,
            "outside_reference_range_mean": float(outside_amount.mean().detach().cpu()),
            "x_rms_ratio_to_reference": q_stats["x_rms_mean"] / max(abs(float(ref_stats["x_rms_mean"])), 1e-12),
            "x_std_ratio_to_reference": q_stats["x_std"] / max(abs(float(ref_stats["x_std"])), 1e-12),
            "x_range_ratio_to_reference": q_stats["x_sample_range_mean"] / max(abs(float(ref_stats["x_sample_range_mean"])), 1e-12),
            "x_min_minus_reference_min": q_stats["x_min"] - float(ref_stats["x_min"]),
            "x_max_minus_reference_max": q_stats["x_max"] - float(ref_stats["x_max"]),
        }
        row.update(q_stats)
        rows.append(row)
    return rows


def geometry_stats(x_clean: torch.Tensor, x_adv: torch.Tensor, y_adv: torch.Tensor | None, prefix: str) -> dict[str, float]:
    delta = x_adv - x_clean
    flat_delta = flatten_x(delta)
    flat_x = flatten_x(x_adv)
    l2_rms = torch.sqrt(flat_delta.pow(2).mean(dim=1).clamp_min(1e-24))
    linf = flat_delta.abs().max(dim=1).values
    below0 = (x_adv < 0).float()
    above1 = (x_adv > 1).float()
    oob = torch.clamp(-x_adv, min=0) + torch.clamp(x_adv - 1.0, min=0)
    out = {
        f"{prefix}_delta_l2_rms_mean": float(l2_rms.mean().detach().cpu()),
        f"{prefix}_delta_linf_mean": float(linf.mean().detach().cpu()),
        f"{prefix}_linf_over_l2rms_mean": float((linf / l2_rms.clamp_min(1e-12)).mean().detach().cpu()),
        f"{prefix}_x_min": float(flat_x.min().detach().cpu()),
        f"{prefix}_x_max": float(flat_x.max().detach().cpu()),
        f"{prefix}_below0_frac": float(below0.mean().detach().cpu()),
        f"{prefix}_above1_frac": float(above1.mean().detach().cpu()),
        f"{prefix}_oob_mean": float(oob.mean().detach().cpu()),
        f"{prefix}_oob_max": float(oob.max().detach().cpu()),
    }
    if y_adv is not None:
        flat_y = y_adv.detach().float().reshape(y_adv.shape[0], -1)
        out[f"{prefix}_y_min"] = float(flat_y.min().detach().cpu())
        out[f"{prefix}_y_max"] = float(flat_y.max().detach().cpu())
    return out


def lowpass_delta(delta: torch.Tensor, keep_modes: int, preserve_rms: bool) -> torch.Tensor:
    arr = delta.detach()
    if arr.ndim == 3 and arr.shape[-1] == 1:
        core = arr[..., 0]
        add_channel = True
    else:
        core = arr
        add_channel = False
    fft = torch.fft.rfft(core, dim=1)
    keep = min(max(1, int(keep_modes)), int(fft.shape[1]))
    filtered = torch.zeros_like(fft)
    filtered[:, :keep] = fft[:, :keep]
    out = torch.fft.irfft(filtered, n=core.shape[1], dim=1)
    if preserve_rms:
        old = torch.sqrt(core.pow(2).mean(dim=1, keepdim=True).clamp_min(1e-24))
        new = torch.sqrt(out.pow(2).mean(dim=1, keepdim=True).clamp_min(1e-24))
        out = out * (old / new)
    if add_channel:
        out = out.unsqueeze(-1)
    return out


def apply_transform(
    *,
    transform: str,
    x_clean: torch.Tensor,
    raw_x_adv: torch.Tensor,
    lowpass_keep_modes: int,
    lowpass_preserve_rms: bool,
) -> torch.Tensor:
    delta = raw_x_adv - x_clean
    if transform == "raw":
        x_used = raw_x_adv
    elif transform == "clip01":
        x_used = raw_x_adv.clamp(0.0, 1.0)
    elif transform == "lowpass":
        x_used = x_clean + lowpass_delta(delta, lowpass_keep_modes, lowpass_preserve_rms)
    elif transform == "lowpass_clip01":
        x_used = (x_clean + lowpass_delta(delta, lowpass_keep_modes, lowpass_preserve_rms)).clamp(0.0, 1.0)
    else:
        raise ValueError(f"unknown transform: {transform}")
    return x_used.detach()


def make_step_batches(n_train: int, batch_size: int, steps: int, seed: int) -> list[tuple[int, int, torch.Tensor]]:
    out: list[tuple[int, int, torch.Tensor]] = []
    epoch = 1
    while len(out) < steps:
        generator = torch.Generator().manual_seed(seed + epoch * 1009 + TASK_SEED_OFFSETS["burgers"])
        batches = make_indices(n_train, batch_size, generator)
        for local_batch_idx, idx in enumerate(batches, 1):
            out.append((epoch, local_batch_idx, idx))
            if len(out) >= steps:
                break
        epoch += 1
    return out


def cfg_for_variant(args, objective: str) -> dict[str, Any]:
    cfg = asdict(DEFAULTS["burgers"])
    cfg.update(
        {
            "batch_size": int(args.batch_size),
            "optimizer_batch_size": int(args.optimizer_batch_size),
            "eval_batch_size": int(args.eval_batch_size),
            "epochs": int(args.steps),
            "attack_method": "fast_replace_l2",
            "attack_steps": int(args.attack_steps),
            "epsilon_fraction": float(args.epsilon_fraction),
            "epsilon_abs": 0.0,
            "alpha_ratio": 1.0,
            "alpha_jitter_low": float(args.alpha_jitter_low),
            "alpha_jitter_high": float(args.alpha_jitter_high),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "random_start_fraction": float(args.random_start_fraction),
            "eps_jitter_low": float(args.eps_jitter_low),
            "eps_jitter_high": float(args.eps_jitter_high),
            "label_mode": "solver",
            "training_data_mode": "adv-only",
            "ns_attack_frames": "initial",
            "burgers_solver_remat": str(args.burgers_solver_remat),
            "burgers_solver_remat_chunk_steps": int(args.burgers_solver_remat_chunk_steps),
            "attack_loss_objective": objective,
        }
    )
    return cfg


def run_variant(
    *,
    args,
    variant_cfg: dict[str, str],
    device: torch.device,
    x_train_cpu: torch.Tensor,
    y_train_cpu: torch.Tensor,
    step_batches: list[tuple[int, int, torch.Tensor]],
    eval_sets: list[dict[str, Any]],
    ref_banks: dict[str, dict[str, Any]],
    out_dir: Path,
) -> None:
    variant = variant_cfg["variant"]
    objective = variant_cfg["attack_objective"]
    transform = variant_cfg["transform"]
    cfg = cfg_for_variant(args, objective)
    model = load_model("burgers", device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))

    eval_metrics_csv = out_dir / "eval_metrics.csv"
    eval_split_csv = out_dir / "eval_split_summary.csv"
    step_csv = out_dir / "step_attack_geometry.csv"
    sim_csv = out_dir / "x_similarity_by_split.csv"
    opt_csv = out_dir / "optimizer_microsteps.csv"

    rows, elapsed = evaluate_cached(model, eval_sets, device, int(args.eval_batch_size))
    for row in rows:
        append_row(eval_metrics_csv, {"variant": variant, "global_step": 0, "phase": "baseline_before_replay", **row})
    for row in split_summary(rows, variant=variant, step=0, phase="baseline_before_replay", elapsed=elapsed):
        append_row(eval_split_csv, row)

    for global_step, (source_epoch, local_batch_idx, idx) in enumerate(step_batches, 1):
        step_start = time.perf_counter()
        xb = x_train_cpu[idx].to(device, non_blocking=True)
        yb = y_train_cpu[idx].to(device, non_blocking=True)

        attack_start = time.perf_counter()
        attack_result = attack_batch(model, xb, yb, "burgers", cfg)
        attack_sec = time.perf_counter() - attack_start
        raw_x_adv = attack_result.x_train.detach()
        raw_y_adv = attack_result.y_train.detach()
        x_used = apply_transform(
            transform=transform,
            x_clean=xb,
            raw_x_adv=raw_x_adv,
            lowpass_keep_modes=int(args.lowpass_keep_modes),
            lowpass_preserve_rms=bool(args.lowpass_preserve_rms),
        )
        if transform == "raw":
            y_used = raw_y_adv
        else:
            y_used = solver_target_for_model_input("burgers", x_used, yb, cfg, allow_target_grad=False).detach()

        geom_row: dict[str, Any] = {
            "variant": variant,
            "attack_objective": objective,
            "transform": transform,
            "global_step": global_step,
            "source_epoch": source_epoch,
            "local_batch_idx": local_batch_idx,
            "batch_size": int(xb.shape[0]),
            "attack_wall_sec": attack_sec,
            "raw_equals_used": int(transform == "raw"),
        }
        geom_row.update({f"attack_{k}": v for k, v in attack_result.info.items() if isinstance(v, (int, float, str))})
        geom_row.update(geometry_stats(xb, raw_x_adv, raw_y_adv, "raw"))
        geom_row.update(geometry_stats(xb, x_used, y_used, "used"))
        append_row(step_csv, geom_row)

        for stage, x_stage in (("raw_attack_input", raw_x_adv), ("used_training_input", x_used)):
            for sim_row in stage_similarity_rows(
                variant=variant,
                step=global_step,
                input_stage=stage,
                x_stage=x_stage,
                ref_banks=ref_banks,
                query_samples=int(args.similarity_query_samples),
            ):
                append_row(sim_csv, sim_row)

        train_start = time.perf_counter()
        micro_losses: list[float] = []
        grad_norms: list[float] = []
        for micro_idx, micro_slice in enumerate(make_slices(int(x_used.shape[0]), int(args.optimizer_batch_size)), 1):
            x_micro = x_used[micro_slice]
            y_micro = y_used[micro_slice]
            optimizer.zero_grad(set_to_none=True)
            pred = model(x_micro)
            loss = finite_mse(pred, y_micro)
            loss.backward()
            grad_norm = safe_grad_norm(model)
            optimizer.step()
            loss_value = float(loss.detach().cpu())
            micro_losses.append(loss_value)
            grad_norms.append(grad_norm)
            append_row(
                opt_csv,
                {
                    "variant": variant,
                    "global_step": global_step,
                    "microbatch_idx": micro_idx,
                    "optimizer_batch_size": int(x_micro.shape[0]),
                    "train_loss_on_adv_microbatch": loss_value,
                    "grad_norm": grad_norm,
                },
            )
        train_sec = time.perf_counter() - train_start

        rows, elapsed = evaluate_cached(model, eval_sets, device, int(args.eval_batch_size))
        for row in rows:
            append_row(eval_metrics_csv, {"variant": variant, "global_step": global_step, "phase": "after_replay_step", **row})
        for row in split_summary(rows, variant=variant, step=global_step, phase="after_replay_step", elapsed=elapsed):
            row.update(
                {
                    "source_epoch": source_epoch,
                    "local_batch_idx": local_batch_idx,
                    "train_wall_sec": train_sec,
                    "step_wall_sec": time.perf_counter() - step_start,
                    "train_loss_on_adv_mean": float(np.mean(micro_losses)) if micro_losses else float("nan"),
                    "grad_norm_mean": float(np.mean(grad_norms)) if grad_norms else float("nan"),
                }
            )
            append_row(eval_split_csv, row)

        if global_step % int(args.progress_every) == 0 or global_step == 1 or global_step == int(args.steps):
            print(f"[{variant}] step {global_step}/{args.steps} done", flush=True)


def collect_final_tables(out_dir: Path) -> dict[str, list[dict[str, Any]]]:
    final_rows: list[dict[str, Any]] = []
    start_rows: list[dict[str, Any]] = []
    eval_split = out_dir / "eval_split_summary.csv"
    if eval_split.exists():
        with eval_split.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            if row.get("split") in {"train", "test", "generalization"}:
                if int(row.get("global_step", -1)) == 0:
                    start_rows.append(row)
                if int(row.get("global_step", -1)) == max(int(r.get("global_step", -1)) for r in rows if r.get("variant") == row.get("variant")):
                    final_rows.append(row)

    geom_final: list[dict[str, Any]] = []
    geom_path = out_dir / "step_attack_geometry.csv"
    if geom_path.exists():
        with geom_path.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        last_by_variant: dict[str, dict[str, Any]] = {}
        for row in rows:
            last_by_variant[row["variant"]] = row
        geom_final = list(last_by_variant.values())

    sim_final: list[dict[str, Any]] = []
    sim_path = out_dir / "x_similarity_by_split.csv"
    if sim_path.exists():
        with sim_path.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            if row.get("input_stage") == "used_training_input" and row.get("reference_split") == "generalization":
                sim_final.append(row)
    return {"start": start_rows, "final": final_rows, "geom_final": geom_final, "sim_final": sim_final}


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]], limit: int | None = None) -> list[str]:
    if limit is not None:
        rows = rows[:limit]
    lines = ["| " + " | ".join(label for label, _ in columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        vals: list[str] = []
        for _, key in columns:
            val = row.get(key, "")
            try:
                fval = float(val)
                if math.isfinite(fval):
                    vals.append(f"{fval:.6f}")
                else:
                    vals.append("")
            except Exception:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return lines


def write_readme(out_dir: Path, args) -> None:
    tables = collect_final_tables(out_dir)
    final_eval = [
        r for r in tables["final"]
        if r.get("split") in {"train", "test", "generalization"}
    ]
    final_eval.sort(key=lambda r: (r.get("variant", ""), {"train": 0, "test": 1, "generalization": 2}.get(r.get("split", ""), 9)))
    geom = tables["geom_final"]
    geom.sort(key=lambda r: r.get("variant", ""))
    sim = tables["sim_final"]
    sim.sort(key=lambda r: (r.get("variant", ""), int(r.get("global_step", 0))))
    last_step = int(args.steps)
    sim_last = [r for r in sim if int(r.get("global_step", -1)) == last_step]

    lines = [
        "# Burgers p2q2 loss1/loss2/loss3 50-step input-similarity replay",
        "",
        "This probe replays early adversarial training from the same retained Burgers FNO baseline.",
        f"It runs `{args.steps}` attack-batch training steps per variant. One step means one attacked batch plus all optimizer microbatches for that attacked batch.",
        "",
        "Variants:",
        "",
    ]
    for item in VARIANTS:
        lines.append(f"- `{item['variant']}`: attack objective `{item['attack_objective']}`, transform `{item['transform']}`")
    lines.extend(
        [
            "",
            "The raw attack input is always recorded.  For transformed loss3 variants, the actual used training input is also recorded after clipping/lowpass and after recomputing `solver(x_used)`.",
            "",
            "## Final prediction loss by split",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            final_eval,
            [
                ("variant", "variant"),
                ("step", "global_step"),
                ("split", "split"),
                ("RMSE", "rmse_dataset_mean"),
                ("relative L2", "relative_l2_dataset_mean"),
                ("score", "accuracy_score_dataset_mean"),
            ],
        )
    )
    lines.extend(["", "## Final attack/input geometry", ""])
    lines.extend(
        markdown_table(
            geom,
            [
                ("variant", "variant"),
                ("used delta RMS", "used_delta_l2_rms_mean"),
                ("used delta Linf", "used_delta_linf_mean"),
                ("used Linf/RMS", "used_linf_over_l2rms_mean"),
                ("used x min", "used_x_min"),
                ("used x max", "used_x_max"),
                ("used OOB mean", "used_oob_mean"),
                ("raw x min", "raw_x_min"),
                ("raw x max", "raw_x_max"),
            ],
        )
    )
    lines.extend(["", "## Final used-input similarity to generalization", ""])
    lines.extend(
        markdown_table(
            sim_last,
            [
                ("variant", "variant"),
                ("step", "global_step"),
                ("nearest RMS", "nearest_rms_mean"),
                ("max cosine", "max_cosine_mean"),
                ("angle deg", "max_cosine_angle_deg_mean"),
                ("spectrum cosine", "spectrum_cosine"),
                ("spectrum L1", "spectrum_l1_distance"),
                ("below ref min frac", "below_reference_min_frac"),
                ("above ref max frac", "above_reference_max_frac"),
                ("outside ref range mean", "outside_reference_range_mean"),
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Output files",
            "",
            "- `eval_metrics.csv`: per-dataset prediction metrics at baseline and after every step.",
            "- `eval_split_summary.csv`: train/test/generalization/ALL split averages at baseline and after every step.",
            "- `step_attack_geometry.csv`: raw and used attack geometry each step.",
            "- `x_similarity_by_split.csv`: raw/used input similarity to train/test/generalization input distributions.",
            "- `optimizer_microsteps.csv`: optimizer microbatch losses and gradient norms.",
            "- `config.json`: exact run configuration.",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "forensics/burgers_p2q2_loss123_50step_input_similarity_probe_20260604")
    parser.add_argument("--generalization-root", type=Path, default=PROJECT_ROOT / "generalization_datasets_rmse_1p5_3x_all_ns50")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=480)
    parser.add_argument("--optimizer-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--eval-max-samples", type=int, default=0)
    parser.add_argument("--max-generalization-eval", type=int, default=50)
    parser.add_argument("--attack-steps", type=int, default=5)
    parser.add_argument("--epsilon-fraction", type=float, default=0.06)
    parser.add_argument("--eps-jitter-low", type=float, default=0.75)
    parser.add_argument("--eps-jitter-high", type=float, default=1.25)
    parser.add_argument("--alpha-jitter-low", type=float, default=0.75)
    parser.add_argument("--alpha-jitter-high", type=float, default=1.25)
    parser.add_argument("--random-start-fraction", type=float, default=1e-6)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--burgers-solver-remat", default="none")
    parser.add_argument("--burgers-solver-remat-chunk-steps", type=int, default=20)
    parser.add_argument("--lowpass-keep-modes", type=int, default=16)
    parser.add_argument("--lowpass-preserve-rms", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--similarity-query-samples", type=int, default=64)
    parser.add_argument("--similarity-ref-samples", type=int, default=128)
    parser.add_argument("--progress-every", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "config.json").write_text(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2), encoding="utf-8")
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed) % (2**32 - 1))

    specs = [s for s in build_specs(args.generalization_root.resolve()) if s.task == "burgers"]
    train_spec = task_train_spec(specs, "burgers")
    x_train_cpu, y_train_cpu = load_train_xy(train_spec, "burgers", None)
    step_batches = make_step_batches(int(x_train_cpu.shape[0]), int(args.batch_size), int(args.steps), int(args.seed))
    eval_sets = load_eval_sets(
        specs,
        task="burgers",
        max_generalization_eval=int(args.max_generalization_eval),
        eval_max_samples=normalize_max_samples(args.eval_max_samples),
    )
    ref_banks = build_reference_banks(
        eval_sets,
        device=device,
        ref_samples=int(args.similarity_ref_samples),
        seed=int(args.seed) + 404,
    )

    for variant_cfg in VARIANTS:
        print(f"[start] {variant_cfg['variant']}", flush=True)
        run_variant(
            args=args,
            variant_cfg=variant_cfg,
            device=device,
            x_train_cpu=x_train_cpu,
            y_train_cpu=y_train_cpu,
            step_batches=step_batches,
            eval_sets=eval_sets,
            ref_banks=ref_banks,
            out_dir=args.out_dir,
        )
        print(f"[done] {variant_cfg['variant']}", flush=True)

    write_readme(args.out_dir, args)
    print(f"[done] wrote {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
