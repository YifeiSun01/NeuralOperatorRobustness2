#!/usr/bin/env python3
"""Time one NS2D loss3 self-training epoch on CUDA.

This is an epoch-level timing harness for the NS2D solver-label loss3 path. It
uses the same expensive pieces as tools/adversarial_training.py: differentiable
NS rollout from x0_adv, loss3 backward to the initial condition, regenerated
attacked solver labels, and an AdamW optimizer update for each attack batch.

When the real NS2D train tensor or retained checkpoint is missing, the script can
fall back to synthetic 256x256 initial vorticity fields and/or a random same-
architecture model. Those fallbacks are valid for runtime/memory capacity only,
not scientific metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_PATH = (
    PROJECT_ROOT
    / "2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/real_initial_laxmap_single/train/dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_ntimepoints21_all_frames.pt"
)
DEFAULT_OUT_DIR = PROJECT_ROOT / "forensics/ns2d_loss3_self_training_epoch_timing_20260608"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.adversarial_training import finite_mse, ns2d_solver_pair_from_initial  # noqa: E402
from tools.evaluate_generalization_models import tensor_xy, torch_load  # noqa: E402
from tools.probe_ns2d_loss3_adversarial_training_batch_20260608 import (  # noqa: E402
    DEFAULT_CKPT,
    expand_per_sample,
    gpu_preflight,
    jsonable,
    load_probe_model,
    make_synthetic_x0,
    per_sample_range,
)


def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S_UTC", time.gmtime())


def write_csv_row(path: Path, row: dict[str, Any]) -> None:
    exists = path.exists()
    fields = list(row.keys())
    if exists:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            try:
                fields = next(reader)
            except StopIteration:
                pass
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({key: jsonable(row.get(key, "")) for key in fields})


def load_x0_samples(args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, dict[str, Any]]:
    num_samples = int(args.num_samples)
    if str(args.data_source) in {"auto", "real"} and args.train_path.exists():
        data = torch_load(args.train_path)
        x_seq, _y_seq = tensor_xy(data, "ns2d")
        if x_seq.shape[0] < num_samples:
            raise ValueError(f"requested {num_samples} samples but real train tensor has {x_seq.shape[0]}")
        x0_cpu = x_seq[:num_samples, ..., 0].contiguous()
        return x0_cpu, {
            "data_source": "real_train_tensor",
            "train_path": str(args.train_path),
            "available_train_samples": int(x_seq.shape[0]),
        }
    if str(args.data_source) == "real":
        raise FileNotFoundError(f"real NS2D train path missing: {args.train_path}")
    x0 = make_synthetic_x0(num_samples, int(args.grid_size), device, int(args.seed)).detach().cpu().contiguous()
    return x0, {
        "data_source": "synthetic_vorticity_runtime_probe",
        "train_path": str(args.train_path),
        "train_path_missing": not args.train_path.exists(),
        "note": "Synthetic data is valid for timing/memory only, not scientific metrics.",
    }


def run_one_batch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    x0_cpu: torch.Tensor,
    batch_index: int,
    args: argparse.Namespace,
    device: torch.device,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    batch_start = batch_index * int(args.batch_size)
    batch_end = min(batch_start + int(args.batch_size), int(x0_cpu.shape[0]))
    x0 = x0_cpu[batch_start:batch_end].to(device, non_blocking=True).contiguous()
    batch_size_actual = int(x0.shape[0])
    eps = per_sample_range(x0) * float(args.epsilon_fraction)
    alpha = eps * float(args.alpha_ratio)
    eps_view = expand_per_sample(eps, x0)
    alpha_view = expand_per_sample(alpha, x0)
    delta = torch.zeros_like(x0)
    x0_adv = (x0 + delta).detach()

    torch.cuda.reset_peak_memory_stats(device)
    step_start = time.perf_counter()
    attack_start = step_start
    attack_losses: list[float] = []
    for _step in range(max(1, int(args.attack_steps))):
        x0_adv = x0_adv.detach().requires_grad_(True)
        x_seq_adv, y_seq_adv = ns2d_solver_pair_from_initial(x0_adv, cfg, for_attack=True)
        pred_adv = model(x_seq_adv)
        loss = finite_mse(pred_adv, y_seq_adv)
        attack_losses.append(float(loss.detach().cpu()))
        grad = torch.autograd.grad(loss, x0_adv, only_inputs=True)[0]
        grad = torch.nan_to_num(grad)
        if str(args.attack_method) == "fast_replace_linf":
            delta = eps_view * grad.sign()
        elif str(args.attack_method) == "fast_add_linf":
            delta = torch.clamp((x0_adv.detach() - x0) + alpha_view * grad.sign(), -eps_view, eps_view)
        else:
            raise ValueError(f"unsupported attack_method={args.attack_method!r}")
        x0_adv = (x0 + delta).detach()
        del x_seq_adv, y_seq_adv, pred_adv, loss, grad
    attack_seconds = time.perf_counter() - attack_start

    label_start = time.perf_counter()
    with torch.no_grad():
        x_train, y_train = ns2d_solver_pair_from_initial(x0_adv, cfg, for_attack=False)
    label_seconds = time.perf_counter() - label_start

    train_start = time.perf_counter()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    optimizer_batch = max(1, int(args.optimizer_batch_size))
    train_losses: list[float] = []
    for start_idx in range(0, batch_size_actual, optimizer_batch):
        xb = x_train[start_idx : start_idx + optimizer_batch]
        yb = y_train[start_idx : start_idx + optimizer_batch]
        pred = model(xb)
        train_loss = finite_mse(pred, yb)
        scaled = train_loss * (int(xb.shape[0]) / batch_size_actual)
        scaled.backward()
        train_losses.append(float(train_loss.detach().cpu()))
    optimizer.step()
    model.eval()
    train_seconds = time.perf_counter() - train_start
    step_seconds = time.perf_counter() - step_start

    delta_final = x0_adv.detach() - x0
    row = {
        "created_utc": now_stamp(),
        "batch_index": int(batch_index),
        "start": int(batch_start),
        "end": int(batch_end),
        "batch_size": batch_size_actual,
        "configured_batch_size": int(args.batch_size),
        "optimizer_batch_size": int(args.optimizer_batch_size),
        "attack_steps": int(args.attack_steps),
        "attack_method": str(args.attack_method),
        "epsilon_fraction": float(args.epsilon_fraction),
        "alpha_ratio": float(args.alpha_ratio),
        "solver_remat": str(args.solver_remat),
        "solver_remat_chunk_steps": int(args.solver_remat_chunk_steps),
        "attack_seconds": float(attack_seconds),
        "label_regen_seconds": float(label_seconds),
        "optimizer_train_seconds": float(train_seconds),
        "step_seconds": float(step_seconds),
        "samples_per_second": float(batch_size_actual / max(step_seconds, 1e-12)),
        "attack_loss_first": attack_losses[0] if attack_losses else float("nan"),
        "attack_loss_last": attack_losses[-1] if attack_losses else float("nan"),
        "train_loss_mean": float(np.mean(train_losses)) if train_losses else float("nan"),
        "boundary_ratio_mean": float((delta_final.abs() / eps_view.clamp_min(1e-12)).mean().cpu()),
        "delta_linf_mean": float(delta_final.abs().flatten(1).max(dim=1).values.mean().cpu()),
        "delta_l2_rms_mean": float(torch.sqrt(delta_final.pow(2).flatten(1).mean(dim=1)).mean().cpu()),
        "peak_cuda_allocated_gib": float(torch.cuda.max_memory_allocated(device) / (1024**3)),
        "peak_cuda_reserved_gib": float(torch.cuda.max_memory_reserved(device) / (1024**3)),
    }
    del x0, x0_adv, x_train, y_train, delta_final
    return row


def run_epoch(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    device = torch.device(args.device)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    preflight = gpu_preflight(device)
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed) % (2**32 - 1))

    model, model_info = load_probe_model(device, args.checkpoint.resolve(), bool(args.allow_random_model_for_memory_probe))
    model.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    x0_cpu, data_info = load_x0_samples(args, device)
    n_train = int(x0_cpu.shape[0])
    batches_per_epoch = int(math.ceil(n_train / max(1, int(args.batch_size))))
    cfg = {
        "label_mode": "solver",
        "ns2d_solver_remat": str(args.solver_remat),
        "ns2d_solver_remat_chunk_steps": int(args.solver_remat_chunk_steps),
    }

    batch_csv = out_dir / "epoch_train_batches.csv"
    if batch_csv.exists() and not args.append:
        batch_csv.unlink()

    status = "passed"
    exit_code = 0
    error_message = ""
    error_trace = ""
    epoch_start = time.perf_counter()
    rows: list[dict[str, Any]] = []
    completed_batches = 0
    try:
        for batch_index in range(batches_per_epoch):
            row = run_one_batch(model, optimizer, x0_cpu, batch_index, args, device, cfg)
            rows.append(row)
            completed_batches += 1
            write_csv_row(batch_csv, row)
            print(json.dumps({
                "event": "batch_done",
                "batch_index": row["batch_index"],
                "batch_size": row["batch_size"],
                "step_seconds": row["step_seconds"],
                "peak_cuda_allocated_gib": row["peak_cuda_allocated_gib"],
            }), flush=True)
    except RuntimeError as exc:
        error_message = str(exc)
        error_trace = traceback.format_exc()
        if "out of memory" in error_message.lower() or "alloc" in error_message.lower():
            status = "failed_oom"
            exit_code = 42
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        else:
            status = "failed_runtime_error"
            exit_code = 1
    except Exception as exc:
        error_message = str(exc)
        error_trace = traceback.format_exc()
        status = "failed_exception"
        exit_code = 1

    epoch_seconds = time.perf_counter() - epoch_start
    step_seconds = [float(r["step_seconds"]) for r in rows]
    mean_step = float(np.mean(step_seconds)) if step_seconds else float("nan")
    median_step = float(np.median(step_seconds)) if step_seconds else float("nan")
    full_1150_batches = int(math.ceil(1150 / max(1, int(args.batch_size))))
    summary = {
        "created_utc": now_stamp(),
        "status": status,
        "exit_code": int(exit_code),
        "num_samples": int(n_train),
        "batch_size": int(args.batch_size),
        "optimizer_batch_size": int(args.optimizer_batch_size),
        "batches_per_epoch": int(batches_per_epoch),
        "completed_batches": int(completed_batches),
        "attack_steps": int(args.attack_steps),
        "attack_method": str(args.attack_method),
        "epsilon_fraction": float(args.epsilon_fraction),
        "alpha_ratio": float(args.alpha_ratio),
        "solver_remat": str(args.solver_remat),
        "solver_remat_chunk_steps": int(args.solver_remat_chunk_steps),
        "measured_epoch_seconds": float(epoch_seconds),
        "measured_epoch_minutes": float(epoch_seconds / 60.0),
        "mean_step_seconds": mean_step,
        "median_step_seconds": median_step,
        "mean_attack_seconds": float(np.mean([float(r["attack_seconds"]) for r in rows])) if rows else float("nan"),
        "mean_label_regen_seconds": float(np.mean([float(r["label_regen_seconds"]) for r in rows])) if rows else float("nan"),
        "mean_optimizer_train_seconds": float(np.mean([float(r["optimizer_train_seconds"]) for r in rows])) if rows else float("nan"),
        "peak_cuda_allocated_gib_max": float(np.max([float(r["peak_cuda_allocated_gib"]) for r in rows])) if rows else float("nan"),
        "peak_cuda_reserved_gib_max": float(np.max([float(r["peak_cuda_reserved_gib"]) for r in rows])) if rows else float("nan"),
        "estimated_1150_samples_batches": full_1150_batches,
        "estimated_1150_samples_epoch_seconds_from_mean_step": float(mean_step * full_1150_batches) if math.isfinite(mean_step) else float("nan"),
        "estimated_1150_samples_epoch_hours_from_mean_step": float(mean_step * full_1150_batches / 3600.0) if math.isfinite(mean_step) else float("nan"),
        "batch_csv": str(batch_csv),
        "gpu_preflight": preflight,
        **model_info,
        **data_info,
    }
    if status != "passed":
        summary["error_message"] = error_message
        summary["traceback"] = error_trace
    (out_dir / "epoch_summary.json").write_text(json.dumps(jsonable(summary), indent=2), encoding="utf-8")
    return exit_code, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--optimizer-batch-size", type=int, default=1)
    parser.add_argument("--attack-steps", type=int, default=10)
    parser.add_argument("--attack-method", choices=["fast_add_linf", "fast_replace_linf"], default="fast_add_linf")
    parser.add_argument("--epsilon-fraction", type=float, default=0.035)
    parser.add_argument("--alpha-ratio", type=float, default=0.2)
    parser.add_argument("--solver-remat", choices=["none", "micro", "step", "chunk", "second"], default="chunk")
    parser.add_argument("--solver-remat-chunk-steps", type=int, default=20)
    parser.add_argument("--grid-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--data-source", choices=["auto", "real", "synthetic"], default="auto")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--allow-random-model-for-memory-probe", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exit_code, summary = run_epoch(args)
    print(json.dumps(jsonable({
        "status": summary.get("status"),
        "num_samples": summary.get("num_samples"),
        "batch_size": summary.get("batch_size"),
        "batches_per_epoch": summary.get("batches_per_epoch"),
        "measured_epoch_minutes": summary.get("measured_epoch_minutes"),
        "mean_step_seconds": summary.get("mean_step_seconds"),
        "estimated_1150_samples_epoch_hours_from_mean_step": summary.get("estimated_1150_samples_epoch_hours_from_mean_step"),
        "model_source": summary.get("model_source"),
        "data_source": summary.get("data_source"),
    }), indent=2), flush=True)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
