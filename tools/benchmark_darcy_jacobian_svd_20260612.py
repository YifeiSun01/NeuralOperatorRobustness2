#!/usr/bin/env python3
"""Benchmark explicit/full and block-projected Darcy Jacobian SVD."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.evaluate_generalization_models import tensor_xy, torch_load
import tools.adversarial_training as adv

RUN_TAG = "20260612_full50_timematched_1000c"
DATASET_ID = "darcy_binary_loss3targeted_20260611_00_matern_smooth_frac0p12_a3p65909_t2p05625"

CHECKPOINTS = {
    "loss1": PROJECT_ROOT / "adversarial_training_runs" / f"darcy_binary_loss3targeted_loss1_1000ep_full50_timematched_{RUN_TAG}" / "darcy" / "checkpoints" / "darcy_epoch1000_step001000.pt",
    "loss2": PROJECT_ROOT / "adversarial_training_runs" / f"darcy_binary_loss3targeted_loss2_1026ep_full50_timematched_{RUN_TAG}" / "darcy" / "checkpoints" / "darcy_epoch1026_step001026.pt",
    "loss3": PROJECT_ROOT / "adversarial_training_runs" / f"darcy_binary_loss3targeted_loss3_1011ep_full50_timematched_{RUN_TAG}" / "darcy" / "checkpoints" / "darcy_epoch1011_step001011.pt",
    "fixed": PROJECT_ROOT / "adversarial_training_runs" / f"darcy_binary_loss3targeted_physics_1040ep_full50_timematched_{RUN_TAG}" / "darcy" / "checkpoints" / "darcy_epoch1040_step001040.pt",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def mem_stats(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {}
    return {
        "cuda_allocated_gib": torch.cuda.memory_allocated(device) / 1024**3,
        "cuda_reserved_gib": torch.cuda.memory_reserved(device) / 1024**3,
        "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        "cuda_peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
    }


def load_one_sample(generalization_root: Path, dataset_id: str, sample_index: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    payload = torch_load(generalization_root / "darcy" / f"{dataset_id}.pt")
    x, y = tensor_xy(payload, "darcy")
    return x[sample_index : sample_index + 1].contiguous().to(device), y[sample_index : sample_index + 1].contiguous().to(device)


def explicit_jacobian_rows(func: Callable[[torch.Tensor], torch.Tensor], z0: torch.Tensor, row_chunk: int) -> tuple[torch.Tensor, float]:
    z = z0.detach().clone().requires_grad_(True)
    cuda_sync()
    t0 = time.perf_counter()
    y = func(z).reshape(-1)
    out_dim = int(y.numel())
    in_dim = int(z.numel())
    jac = torch.empty((out_dim, in_dim), device=z.device, dtype=z.dtype)
    eye = torch.eye(out_dim, device=z.device, dtype=z.dtype)
    for start in range(0, out_dim, row_chunk):
        end = min(start + row_chunk, out_dim)
        grad_outputs = eye[start:end]
        grad = torch.autograd.grad(
            y,
            z,
            grad_outputs=grad_outputs,
            retain_graph=True,
            create_graph=False,
            is_grads_batched=True,
        )[0]
        jac[start:end] = grad.reshape(end - start, in_dim)
        if (start // row_chunk) % max(1, 512 // row_chunk) == 0:
            print(f"[jacobian] rows {end}/{out_dim}", flush=True)
    cuda_sync()
    return jac.detach(), time.perf_counter() - t0


def make_full_func(model) -> tuple[Callable[[torch.Tensor], torch.Tensor], torch.Tensor]:
    def f(x: torch.Tensor) -> torch.Tensor:
        return model(x).reshape(-1)

    return f


def make_block_func(model, x0: torch.Tensor, factor: int) -> tuple[Callable[[torch.Tensor], torch.Tensor], torch.Tensor, int, int]:
    _, h, w, _ = x0.shape
    crop_h = (h // factor) * factor
    crop_w = (w // factor) * factor
    coarse_h = crop_h // factor
    coarse_w = crop_w // factor
    scale = math.sqrt(float(factor * factor))

    def lift(z: torch.Tensor) -> torch.Tensor:
        z_img = z.reshape(1, coarse_h, coarse_w, 1)
        fine = z_img.repeat_interleave(factor, dim=1).repeat_interleave(factor, dim=2) / scale
        x = x0.detach().clone()
        x[:, :crop_h, :crop_w, :] = x[:, :crop_h, :crop_w, :] + fine
        return x

    def project(y: torch.Tensor) -> torch.Tensor:
        yc = y[:, :crop_h, :crop_w, :].permute(0, 3, 1, 2)
        # Sum over each block divided by sqrt(block_area): orthonormal block projection.
        pooled = F.avg_pool2d(yc, kernel_size=factor, stride=factor) * scale
        return pooled.permute(0, 2, 3, 1).reshape(-1)

    def f(z: torch.Tensor) -> torch.Tensor:
        return project(model(lift(z)))

    z0 = torch.zeros((1, coarse_h, coarse_w, 1), device=x0.device, dtype=x0.dtype)
    return f, z0, crop_h, crop_w


def run(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    out_dir = (args.out_dir or PROJECT_ROOT / "analysis_outputs" / f"darcy_jacobian_svd_benchmark_{args.tag}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = CHECKPOINTS[args.model]
    model = adv.load_model("darcy", device, model_checkpoint_override=ckpt)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    x0, y0 = load_one_sample(args.generalization_root.resolve(), args.dataset_id, args.sample_index, device)

    if args.mode == "full":
        func = make_full_func(model)
        z0 = x0
        projection = "full_85x85"
        crop_h = crop_w = 85
    else:
        func, z0, crop_h, crop_w = make_block_func(model, x0, args.factor)
        projection = f"orthonormal_block_factor{args.factor}_crop{crop_h}x{crop_w}"

    result: dict[str, Any] = {
        "created_at": now_iso(),
        "model": args.model,
        "checkpoint": rel(ckpt),
        "dataset_id": args.dataset_id,
        "sample_index": args.sample_index,
        "mode": args.mode,
        "factor": args.factor if args.mode == "block" else 1,
        "projection": projection,
        "input_dim": int(z0.numel()),
        "output_dim": int(func(z0).numel()),
        "row_chunk": args.row_chunk,
        "device": str(device),
        "dtype": str(z0.dtype),
    }
    print(json.dumps(result, indent=2), flush=True)

    try:
        jac, jac_sec = explicit_jacobian_rows(func, z0, args.row_chunk)
        result["jacobian_seconds"] = jac_sec
        result["jacobian_shape"] = list(jac.shape)
        result["jacobian_numel"] = int(jac.numel())
        result["jacobian_storage_mib"] = float(jac.numel() * jac.element_size() / 1024**2)
        result.update({f"after_jacobian_{k}": v for k, v in mem_stats(device).items()})
        print(f"[timing] jacobian_seconds={jac_sec:.3f} storage_mib={result['jacobian_storage_mib']:.1f}", flush=True)

        cuda_sync()
        t0 = time.perf_counter()
        svals = torch.linalg.svdvals(jac)
        cuda_sync()
        svd_sec = time.perf_counter() - t0
        svals_cpu = svals.detach().cpu().numpy()
        result["svd_seconds"] = svd_sec
        result["total_seconds"] = jac_sec + svd_sec
        result["sigma_max"] = float(svals_cpu[0])
        result["sigma_min"] = float(svals_cpu[-1])
        result["sigma_mean"] = float(np.mean(svals_cpu))
        result["sigma_median"] = float(np.median(svals_cpu))
        result["top10_singular_values"] = [float(x) for x in svals_cpu[:10]]
        result["status"] = "ok"
        result.update({f"after_svd_{k}": v for k, v in mem_stats(device).items()})
        np.save(out_dir / f"singular_values_{args.model}_{args.mode}_factor{result['factor']}.npy", svals_cpu)
        if args.save_jacobian:
            torch.save(jac.detach().cpu(), out_dir / f"jacobian_{args.model}_{args.mode}_factor{result['factor']}.pt")
        print(f"[timing] svd_seconds={svd_sec:.3f} sigma_max={result['sigma_max']:.6g}", flush=True)
    except Exception as exc:
        result["status"] = "failed"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        result.update({f"failure_{k}": v for k, v in mem_stats(device).items()})
        print(f"[failed] {type(exc).__name__}: {exc}", flush=True)
        raise
    finally:
        (out_dir / f"result_{args.model}_{args.mode}_factor{result['factor']}.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        csv_path = out_dir / "summary.csv"
        exists = csv_path.exists()
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            keys = list(result.keys())
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            if not exists:
                writer.writeheader()
            writer.writerow(result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="20260612_loss3_one_matrix")
    parser.add_argument("--model", choices=sorted(CHECKPOINTS), default="loss3")
    parser.add_argument("--mode", choices=["full", "block"], default="full")
    parser.add_argument("--factor", type=int, choices=[2, 4], default=2)
    parser.add_argument("--row-chunk", type=int, default=16)
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--generalization-root", type=Path, default=PROJECT_ROOT / "generalization_datasets_darcy_binary_loss3targeted_20260611")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--save-jacobian", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
