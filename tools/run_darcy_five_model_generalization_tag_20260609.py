#!/usr/bin/env python3
"""Run one full Darcy/C-flow generalization tag pass for five formal models.

Compares baseline, loss1, loss2, loss3, and physical-source models with one
shared attack objective so the average loss gain is directly comparable.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
                PROJECT_ROOT / "adversarial_training_runs/darcy_lossdrop50_loss1_single_gpu_time_matched_loss3wall_20260608/darcy/checkpoints.csv"
            ),
        ),
        ModelSpec(
            "loss2",
            last_checkpoint_from_csv(
                PROJECT_ROOT / "adversarial_training_runs/darcy_lossdrop50_loss2_single_gpu_time_matched_loss3wall_20260608/darcy/checkpoints.csv"
            ),
        ),
        ModelSpec(
            "loss3",
            last_checkpoint_from_csv(
                PROJECT_ROOT / "adversarial_training_runs/darcy_lossdrop50_loss3_500ep_fromscreen_20260607/darcy/checkpoints.csv"
            ),
        ),
        ModelSpec(
            "physical_source",
            last_checkpoint_from_csv(
                PROJECT_ROOT / "adversarial_training_runs/darcy_lossdrop50_physics_time_matched_loss3wall_20260608/darcy/checkpoints.csv"
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
        raise RuntimeError("CUDA is not available; refusing to run Darcy/C-flow tag experiment on CPU.")

    device = torch.device("cuda")
    x = torch.randn((512, 512), device=device)
    y = (x @ x.T).mean()
    torch.cuda.synchronize()

    info = {
        "nvidia_smi": run_text_command(["nvidia-smi"]),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_device_name": torch.cuda.get_device_name(0),
        "torch_device_capability": torch.cuda.get_device_capability(0),
        "torch_cuda_arch_list": getattr(torch.cuda, "get_arch_list", lambda: [])(),
        "torch_gpu_matmul_mean": float(y.detach().cpu()),
    }

    try:
        import jax
        import jax.numpy as jnp

        j = jnp.ones((1024, 1024), dtype=jnp.float32)
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
    except Exception as exc:
        raise RuntimeError(f"JAX GPU preflight failed: {exc}") from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "gpu_preflight.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    (out_dir / "gpu_preflight_nvidia_smi.txt").write_text(str(info["nvidia_smi"]), encoding="utf-8")
    return info


def load_pair(path: Path, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    x, y = adv.tensor_xy(payload, "darcy")
    return x.to(device), y.to(device)


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


def find_max_batch(model, x_probe: torch.Tensor, y_probe: torch.Tensor, cfg: dict[str, Any], candidates: list[int], steps: int) -> int:
    for b in candidates:
        b = min(int(b), int(x_probe.shape[0]))
        if b <= 0:
            continue
        try:
            run_attack_batch(model, x_probe[:b], y_probe[:b], cfg, steps)
            torch.cuda.empty_cache()
            return b
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            torch.cuda.empty_cache()
    return 1


def tensor_to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def evaluate_model(
    spec: ModelSpec,
    dataset_paths: list[Path],
    out_dir: Path,
    cfg: dict[str, Any],
    batch_candidates: list[int],
    steps: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    device = torch.device("cuda")
    model = adv.load_model("darcy", device, model_checkpoint_override=spec.checkpoint)
    model.eval()

    x_probe, y_probe = load_pair(dataset_paths[0], device)
    batch_size = find_max_batch(model, x_probe, y_probe, cfg, batch_candidates, steps)

    sample_rows: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []

    for ds_path in dataset_paths:
        x_all, y_all = load_pair(ds_path, device)
        dataset_gains: list[float] = []
        dataset_rel_gains: list[float] = []
        dataset_clean: list[float] = []
        dataset_adv: list[float] = []
        dataset_solver_gains: list[float] = []

        for start in range(0, int(x_all.shape[0]), batch_size):
            end = min(start + batch_size, int(x_all.shape[0]))
            result = run_attack_batch(model, x_all[start:end], y_all[start:end], cfg, steps)
            si = result.sample_info
            clean = tensor_to_numpy(si["clean_loss_before_attack"])
            adv_after = tensor_to_numpy(si["adv_loss_after_attack"])
            gain = tensor_to_numpy(si["attack_loss_gain"])
            rel_gain = tensor_to_numpy(si["attack_loss_gain_relative"])
            solver_gain = float(result.info.get("solver_mse_attack_gain", np.nan))

            dataset_gains.extend(float(v) for v in gain)
            dataset_rel_gains.extend(float(v) for v in rel_gain)
            dataset_clean.extend(float(v) for v in clean)
            dataset_adv.extend(float(v) for v in adv_after)
            dataset_solver_gains.append(solver_gain)

            for j in range(len(gain)):
                sample_rows.append(
                    {
                        "model": spec.name,
                        "dataset": ds_path.name,
                        "sample_index": start + j,
                        "attack_batch_size": batch_size,
                        "clean_loss_before_attack": float(clean[j]),
                        "adv_loss_after_attack": float(adv_after[j]),
                        "attack_loss_gain": float(gain[j]),
                        "attack_loss_gain_relative": float(rel_gain[j]),
                        "checkpoint": str(spec.checkpoint),
                    }
                )

        dataset_rows.append(
            {
                "model": spec.name,
                "dataset": ds_path.name,
                "attack_batch_size": batch_size,
                "mean_clean_loss": float(np.nanmean(dataset_clean)),
                "mean_adv_loss": float(np.nanmean(dataset_adv)),
                "mean_attack_loss_gain": float(np.nanmean(dataset_gains)),
                "mean_attack_loss_gain_relative": float(np.nanmean(dataset_rel_gains)),
                "mean_solver_mse_attack_gain_by_batch": float(np.nanmean(dataset_solver_gains)),
            }
        )

    all_gain = np.array([r["attack_loss_gain"] for r in sample_rows], dtype=np.float64)
    all_rel = np.array([r["attack_loss_gain_relative"] for r in sample_rows], dtype=np.float64)
    all_clean = np.array([r["clean_loss_before_attack"] for r in sample_rows], dtype=np.float64)
    all_adv = np.array([r["adv_loss_after_attack"] for r in sample_rows], dtype=np.float64)
    summary = {
        "model": spec.name,
        "checkpoint": str(spec.checkpoint),
        "attack_batch_size": batch_size,
        "sample_count": int(len(sample_rows)),
        "dataset_count": int(len(dataset_paths)),
        "mean_clean_loss": float(np.nanmean(all_clean)),
        "mean_adv_loss": float(np.nanmean(all_adv)),
        "mean_attack_loss_gain": float(np.nanmean(all_gain)),
        "mean_attack_loss_gain_relative": float(np.nanmean(all_rel)),
    }

    torch.cuda.empty_cache()
    return summary, dataset_rows, sample_rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default=str(PROJECT_ROOT / "generalization_datasets_darcy_lossdrop50_selected_20260607/darcy"))
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "forensics/darcy_five_model_generalization_tag_20260609"))
    parser.add_argument("--batch-candidates", default="96,80,64,50,48,40,32,24,16,8")
    parser.add_argument("--steps", type=int, default=1)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gpu_info = gpu_preflight(out_dir)

    dataset_paths = sorted(Path(args.dataset_dir).glob("*.pt"))
    if not dataset_paths:
        raise RuntimeError(f"No .pt files found in {args.dataset_dir}")

    specs = model_specs()
    for spec in specs:
        if not spec.checkpoint.exists():
            raise FileNotFoundError(f"{spec.name} checkpoint is missing: {spec.checkpoint}")

    cfg = attack_cfg()
    batch_candidates = parse_int_list(args.batch_candidates)
    all_model_rows: list[dict[str, Any]] = []
    all_dataset_rows: list[dict[str, Any]] = []
    all_sample_rows: list[dict[str, Any]] = []

    manifest = {
        "dataset_dir": str(Path(args.dataset_dir)),
        "dataset_count": len(dataset_paths),
        "models": [{"name": spec.name, "checkpoint": str(spec.checkpoint)} for spec in specs],
        "attack_protocol": {
            "shared_objective": "loss3",
            "attack_type": "binary_steepest_replace",
            "steps": args.steps,
            "epsilon_fraction": 0.025,
            "batch_candidates": batch_candidates,
            "ranking_metric": "mean_attack_loss_gain ascending",
        },
        "gpu_preflight": gpu_info,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for spec in specs:
        summary, dataset_rows, sample_rows = evaluate_model(spec, dataset_paths, out_dir, cfg, batch_candidates, args.steps)
        all_model_rows.append(summary)
        all_dataset_rows.extend(dataset_rows)
        all_sample_rows.extend(sample_rows)
        write_csv(
            out_dir / f"{spec.name}_samples.csv",
            sample_rows,
            [
                "model",
                "dataset",
                "sample_index",
                "attack_batch_size",
                "clean_loss_before_attack",
                "adv_loss_after_attack",
                "attack_loss_gain",
                "attack_loss_gain_relative",
                "checkpoint",
            ],
        )

    all_model_rows.sort(key=lambda row: row["mean_attack_loss_gain"])
    least_burst = all_model_rows[0]

    write_csv(
        out_dir / "summary_by_model.csv",
        all_model_rows,
        [
            "model",
            "checkpoint",
            "attack_batch_size",
            "sample_count",
            "dataset_count",
            "mean_clean_loss",
            "mean_adv_loss",
            "mean_attack_loss_gain",
            "mean_attack_loss_gain_relative",
        ],
    )
    write_csv(
        out_dir / "summary_by_dataset.csv",
        all_dataset_rows,
        [
            "model",
            "dataset",
            "attack_batch_size",
            "mean_clean_loss",
            "mean_adv_loss",
            "mean_attack_loss_gain",
            "mean_attack_loss_gain_relative",
            "mean_solver_mse_attack_gain_by_batch",
        ],
    )
    write_csv(
        out_dir / "all_samples.csv",
        all_sample_rows,
        [
            "model",
            "dataset",
            "sample_index",
            "attack_batch_size",
            "clean_loss_before_attack",
            "adv_loss_after_attack",
            "attack_loss_gain",
            "attack_loss_gain_relative",
            "checkpoint",
        ],
    )

    result = {
        "least_burst_model": least_burst["model"],
        "least_burst_mean_attack_loss_gain": least_burst["mean_attack_loss_gain"],
        "ranked_models": all_model_rows,
        "output_dir": str(out_dir),
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
