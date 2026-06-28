#!/usr/bin/env python3
"""Time one Darcy/C-flow generalization dataset with 20-step tag attacks."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import tools.adversarial_training as adv


@dataclass(frozen=True)
class ModelSpec:
    name: str
    checkpoint: Path


def last_checkpoint_from_csv(path: Path) -> Path:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty checkpoint csv: {path}")
    return PROJECT_ROOT / rows[-1]["checkpoint_path"]


def model_specs() -> list[ModelSpec]:
    return [
        ModelSpec("baseline", PROJECT_ROOT / "2D_Darcy_FNO2d/saved_models/2D/darcy_screen_baseline_m64_w60_e50_20260607/best.pt"),
        ModelSpec("loss1", last_checkpoint_from_csv(PROJECT_ROOT / "adversarial_training_runs/darcy_lossdrop50_loss1_single_gpu_time_matched_loss3wall_20260608/darcy/checkpoints.csv")),
        ModelSpec("loss2", last_checkpoint_from_csv(PROJECT_ROOT / "adversarial_training_runs/darcy_lossdrop50_loss2_single_gpu_time_matched_loss3wall_20260608/darcy/checkpoints.csv")),
        ModelSpec("loss3", last_checkpoint_from_csv(PROJECT_ROOT / "adversarial_training_runs/darcy_lossdrop50_loss3_500ep_fromscreen_20260607/darcy/checkpoints.csv")),
        ModelSpec("physical_source", last_checkpoint_from_csv(PROJECT_ROOT / "adversarial_training_runs/darcy_lossdrop50_physics_time_matched_loss3wall_20260608/darcy/checkpoints.csv")),
    ]


def run_text_command(argv: list[str]) -> str:
    try:
        proc = subprocess.run(argv, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except FileNotFoundError as exc:
        return f"{argv[0]} not found: {exc}"
    return proc.stdout


def gpu_preflight(out_dir: Path) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; refusing CPU fallback.")
    x = torch.randn((256, 256), device="cuda")
    y = (x @ x.T).mean()
    torch.cuda.synchronize()
    info: dict[str, Any] = {
        "nvidia_smi": run_text_command(["nvidia-smi"]),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_device_name": torch.cuda.get_device_name(0),
        "torch_device_capability": torch.cuda.get_device_capability(0),
        "torch_cuda_arch_list": getattr(torch.cuda, "get_arch_list", lambda: [])(),
        "torch_gpu_matmul_mean": float(y.detach().cpu()),
        "xla_python_client_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
    }
    del x, y
    torch.cuda.empty_cache()

    import jax
    import jax.numpy as jnp

    j = jnp.ones((128, 128), dtype=jnp.float32)
    info.update(
        {
            "jax_version": jax.__version__,
            "jax_default_backend": jax.default_backend(),
            "jax_devices": [str(d) for d in jax.devices()],
            "jax_gpu_sum": float(jnp.sum(j @ j.T).block_until_ready()),
        }
    )
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"JAX backend is {jax.default_backend()}, refusing CPU fallback.")
    del j
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "gpu_preflight.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    (out_dir / "gpu_preflight_nvidia_smi.txt").write_text(str(info["nvidia_smi"]), encoding="utf-8")
    return info


def attack_cfg() -> dict[str, Any]:
    return {
        "task": "darcy",
        "label_mode": "solver",
        "attack_loss_objective": "loss3",
        "darcy_attack_loss_objective": "loss3",
        "darcy_physics_metric": "rel_l2",
        "darcy_physics_bc_weight": 1.0,
        "darcy_physics_forcing_value": 1.0,
        "darcy_loss1_random_start": True,
        "darcy_loss1_random_start_fraction": 1.0,
    }


def load_pair(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    x, y = adv.tensor_xy(payload, "darcy")
    return x.to("cuda"), y.to("cuda")


def run_attack(model, xb: torch.Tensor, yb: torch.Tensor, cfg: dict[str, Any], steps: int):
    return adv.binary_darcy_replace_attack(
        model,
        xb,
        yb,
        steps=steps,
        epsilon_fraction=0.025,
        jitter_low=1.0,
        jitter_high=1.0,
        random_pool_multiplier=1.0,
        random_score_noise=0.0,
        cfg=cfg,
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default=str(PROJECT_ROOT / "generalization_datasets_darcy_lossdrop50_selected_20260607/darcy"))
    parser.add_argument("--dataset-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "forensics/darcy_five_model_one_dataset_tag20_timing_20260609"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gpu_info = gpu_preflight(out_dir)

    dataset_paths = sorted(Path(args.dataset_dir).glob("*.pt"))
    if not dataset_paths:
        raise RuntimeError(f"No .pt files found in {args.dataset_dir}")
    dataset_path = dataset_paths[args.dataset_index]
    x, y = load_pair(dataset_path)
    batch_size = min(args.batch_size, int(x.shape[0]))
    xb, yb = x[:batch_size], y[:batch_size]
    cfg = attack_cfg()

    rows: list[dict[str, Any]] = []
    total_start = time.perf_counter()
    for spec in model_specs():
        if not spec.checkpoint.exists():
            raise FileNotFoundError(f"{spec.name} checkpoint is missing: {spec.checkpoint}")
        load_start = time.perf_counter()
        model = adv.load_model("darcy", torch.device("cuda"), model_checkpoint_override=spec.checkpoint)
        model.eval()
        torch.cuda.synchronize()
        load_sec = time.perf_counter() - load_start

        attack_start = time.perf_counter()
        result = run_attack(model, xb, yb, cfg, args.steps)
        torch.cuda.synchronize()
        attack_sec = time.perf_counter() - attack_start

        si = result.sample_info
        clean = si["clean_loss_before_attack"].detach().cpu().numpy().astype(np.float64)
        adv_after = si["adv_loss_after_attack"].detach().cpu().numpy().astype(np.float64)
        gain = si["attack_loss_gain"].detach().cpu().numpy().astype(np.float64)
        rel = si["attack_loss_gain_relative"].detach().cpu().numpy().astype(np.float64)
        rows.append(
            {
                "model": spec.name,
                "checkpoint": str(spec.checkpoint),
                "dataset": str(dataset_path),
                "batch_size": batch_size,
                "steps": args.steps,
                "load_wall_sec": load_sec,
                "attack_wall_sec": attack_sec,
                "clean_loss_mean": float(np.nanmean(clean)),
                "final_tag_loss_mean": float(np.nanmean(adv_after)),
                "loss_gain_mean": float(np.nanmean(gain)),
                "loss_gain_relative_mean": float(np.nanmean(rel)),
                "solver_mse_attack_gain_batch": float(result.info.get("solver_mse_attack_gain", np.nan)),
            }
        )
        write_csv(out_dir / "timing_by_model_partial.csv", rows)
        del model, result
        torch.cuda.empty_cache()
        gc.collect()

    total_sec = time.perf_counter() - total_start
    rows_sorted = sorted(rows, key=lambda row: row["loss_gain_mean"])
    write_csv(out_dir / "timing_by_model.csv", rows_sorted)
    result = {
        "dataset": str(dataset_path),
        "dataset_index": args.dataset_index,
        "batch_size": batch_size,
        "steps": args.steps,
        "model_count": len(rows),
        "total_wall_sec": total_sec,
        "mean_attack_wall_sec_per_model": float(np.mean([row["attack_wall_sec"] for row in rows])),
        "estimated_50_dataset_wall_sec": total_sec * 50.0,
        "estimated_50_dataset_wall_hours": total_sec * 50.0 / 3600.0,
        "least_burst_model_this_dataset": rows_sorted[0]["model"],
        "ranked_models": rows_sorted,
        "gpu_preflight": gpu_info,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
