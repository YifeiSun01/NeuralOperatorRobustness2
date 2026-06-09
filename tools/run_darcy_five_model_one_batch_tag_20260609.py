#!/usr/bin/env python3
"""One-batch Darcy/C-flow tag comparison for five formal models."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import subprocess
import sys
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
        ModelSpec(
            "baseline",
            PROJECT_ROOT / "2D_Darcy_FNO2d/saved_models/2D/darcy_screen_baseline_m64_w60_e50_20260607/best.pt",
        ),
        ModelSpec(
            "loss1",
            last_checkpoint_from_csv(
                PROJECT_ROOT
                / "adversarial_training_runs/darcy_lossdrop50_loss1_single_gpu_time_matched_loss3wall_20260608/darcy/checkpoints.csv"
            ),
        ),
        ModelSpec(
            "loss2",
            last_checkpoint_from_csv(
                PROJECT_ROOT
                / "adversarial_training_runs/darcy_lossdrop50_loss2_single_gpu_time_matched_loss3wall_20260608/darcy/checkpoints.csv"
            ),
        ),
        ModelSpec(
            "loss3",
            last_checkpoint_from_csv(
                PROJECT_ROOT
                / "adversarial_training_runs/darcy_lossdrop50_loss3_500ep_fromscreen_20260607/darcy/checkpoints.csv"
            ),
        ),
        ModelSpec(
            "physical_source",
            last_checkpoint_from_csv(
                PROJECT_ROOT
                / "adversarial_training_runs/darcy_lossdrop50_physics_time_matched_loss3wall_20260608/darcy/checkpoints.csv"
            ),
        ),
    ]


def run_text_command(argv: list[str]) -> str:
    try:
        proc = subprocess.run(argv, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except FileNotFoundError as exc:
        return f"{argv[0]} not found: {exc}"
    return proc.stdout


def gpu_preflight(out_dir: Path) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; refusing to run on CPU.")

    device = torch.device("cuda")
    x = torch.randn((256, 256), device=device)
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

    try:
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
    except Exception as exc:
        raise RuntimeError(f"JAX GPU preflight failed: {exc}") from exc

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


def load_pair(path: Path, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    x, y = adv.tensor_xy(payload, "darcy")
    return x.to(device), y.to(device)


def parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def run_attack_batch(model, xb: torch.Tensor, yb: torch.Tensor, cfg: dict[str, Any], steps: int):
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


def normalize_candidates(candidates: list[int], n: int) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for candidate in candidates:
        b = min(int(candidate), n)
        if b > 0 and b not in seen:
            out.append(b)
            seen.add(b)
    return out


def max_batch_for_model(spec: ModelSpec, x: torch.Tensor, y: torch.Tensor, cfg: dict[str, Any], candidates: list[int], steps: int) -> int:
    device = torch.device("cuda")
    model = adv.load_model("darcy", device, model_checkpoint_override=spec.checkpoint)
    model.eval()
    try:
        for b in candidates:
            try:
                run_attack_batch(model, x[:b], y[:b], cfg, steps)
                torch.cuda.empty_cache()
                return b
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                torch.cuda.empty_cache()
        return 1
    finally:
        del model
        torch.cuda.empty_cache()
        gc.collect()


def final_loss_for_model(spec: ModelSpec, x: torch.Tensor, y: torch.Tensor, cfg: dict[str, Any], batch_size: int, steps: int) -> dict[str, Any]:
    device = torch.device("cuda")
    model = adv.load_model("darcy", device, model_checkpoint_override=spec.checkpoint)
    model.eval()
    try:
        result = run_attack_batch(model, x[:batch_size], y[:batch_size], cfg, steps)
        si = result.sample_info
        clean = si["clean_loss_before_attack"].detach().cpu().numpy().astype(np.float64)
        adv_after = si["adv_loss_after_attack"].detach().cpu().numpy().astype(np.float64)
        gain = si["attack_loss_gain"].detach().cpu().numpy().astype(np.float64)
        rel = si["attack_loss_gain_relative"].detach().cpu().numpy().astype(np.float64)
        return {
            "model": spec.name,
            "checkpoint": str(spec.checkpoint),
            "batch_size": batch_size,
            "clean_loss_mean": float(np.nanmean(clean)),
            "final_tag_loss_mean": float(np.nanmean(adv_after)),
            "loss_gain_mean": float(np.nanmean(gain)),
            "loss_gain_relative_mean": float(np.nanmean(rel)),
            "clean_loss_last_sample": float(clean[-1]),
            "final_tag_loss_last_sample": float(adv_after[-1]),
            "loss_gain_last_sample": float(gain[-1]),
            "solver_mse_attack_gain_batch": float(result.info.get("solver_mse_attack_gain", np.nan)),
        }
    finally:
        del model
        torch.cuda.empty_cache()
        gc.collect()


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
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "forensics/darcy_five_model_one_batch_tag_20260609"))
    parser.add_argument("--batch-candidates", default="96,80,64,50,48,40,32,24,16,8")
    parser.add_argument("--steps", type=int, default=1)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gpu_info = gpu_preflight(out_dir)

    dataset_paths = sorted(Path(args.dataset_dir).glob("*.pt"))
    if not dataset_paths:
        raise RuntimeError(f"No .pt files found in {args.dataset_dir}")
    dataset_path = dataset_paths[0]

    specs = model_specs()
    missing = [str(spec.checkpoint) for spec in specs if not spec.checkpoint.exists()]
    if missing:
        raise FileNotFoundError("Missing checkpoints: " + "; ".join(missing))

    device = torch.device("cuda")
    x, y = load_pair(dataset_path, device)
    candidates = normalize_candidates(parse_int_list(args.batch_candidates), int(x.shape[0]))
    cfg = attack_cfg()

    probe_rows = []
    for spec in specs:
        b = max_batch_for_model(spec, x, y, cfg, candidates, args.steps)
        probe_rows.append({"model": spec.name, "max_batch": b, "checkpoint": str(spec.checkpoint)})

    common_batch = min(row["max_batch"] for row in probe_rows)
    result_rows = [final_loss_for_model(spec, x, y, cfg, common_batch, args.steps) for spec in specs]
    result_rows.sort(key=lambda row: row["loss_gain_mean"])

    write_csv(out_dir / "batch_probe_by_model.csv", probe_rows)
    write_csv(out_dir / "one_batch_summary_by_model.csv", result_rows)

    result = {
        "dataset": str(dataset_path),
        "dataset_sample_count": int(x.shape[0]),
        "candidate_batches": candidates,
        "common_batch_used": int(common_batch),
        "attack_protocol": {
            "shared_objective": "loss3",
            "attack_type": "binary_steepest_replace",
            "steps": args.steps,
            "epsilon_fraction": 0.025,
            "ranking_metric": "loss_gain_mean ascending",
        },
        "least_burst_model": result_rows[0]["model"],
        "least_burst_loss_gain_mean": result_rows[0]["loss_gain_mean"],
        "ranked_models": result_rows,
        "batch_probe_by_model": probe_rows,
        "gpu_preflight": gpu_info,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
