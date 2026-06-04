#!/usr/bin/env python3
"""Generate Burgers generalization datasets from raw loss3 adversarial samples.

This is a mechanism-probe dataset, not an unbiased benchmark: the data are
constructed to have the input geometry produced by the traditional loss3 attack.
Each output .pt file contains x_adv and y_adv=S(x_adv), compatible with the
existing generalization evaluation loaders.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.adversarial_training import attack_batch, load_model, load_train_xy, task_train_spec  # noqa: E402
from tools.evaluate_generalization_models import build_specs  # noqa: E402
from tools.generate_generalization_datasets import json_ready, write_manifest  # noqa: E402
from tools.probe_burgers_p2q2_loss123_50step_input_similarity import cfg_for_variant  # noqa: E402


def finite_stats(t: torch.Tensor) -> dict[str, float]:
    flat = t.detach().float().reshape(t.shape[0], -1)
    return {
        "mean": float(flat.mean()),
        "std": float(flat.std(unbiased=False)),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "rms": float(torch.sqrt(torch.mean(flat * flat))),
    }


def per_sample_delta_stats(x_clean: torch.Tensor, x_adv: torch.Tensor) -> dict[str, float]:
    delta = (x_adv - x_clean).detach().float().reshape(x_adv.shape[0], -1)
    rms = torch.sqrt(delta.pow(2).mean(dim=1).clamp_min(1e-24))
    linf = delta.abs().amax(dim=1)
    oob = torch.clamp(-x_adv.detach().float(), min=0) + torch.clamp(x_adv.detach().float() - 1.0, min=0)
    return {
        "delta_l2_rms_mean": float(rms.mean()),
        "delta_linf_mean": float(linf.mean()),
        "linf_over_l2rms_mean": float((linf / rms.clamp_min(1e-12)).mean()),
        "x_adv_min": float(x_adv.min()),
        "x_adv_max": float(x_adv.max()),
        "oob_mean": float(oob.mean()),
        "oob_max": float(oob.max()),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "generalization_datasets_burgers_loss3_aligned_search")
    parser.add_argument("--round-id", type=int, default=0)
    parser.add_argument("--datasets", type=int, default=50)
    parser.add_argument("--samples-per-dataset", type=int, default=50)
    parser.add_argument("--source-split", choices=["train", "test", "mixed"], default="train")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--steps", type=int, default=1, help="Dummy epoch count used only for shared attack config construction.")
    parser.add_argument("--batch-size", type=int, default=480)
    parser.add_argument("--optimizer-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=512)
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
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_root = args.output_root / f"round_{args.round_id:02d}"
    out_dir = out_root / "burgers"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    specs = [s for s in build_specs(PROJECT_ROOT / "generalization_datasets_rmse_1p5_3x_all_ns50") if s.task == "burgers"]
    train_spec = task_train_spec(specs, "burgers")
    x_train, y_train = load_train_xy(train_spec, "burgers", None)
    test_spec = next(s for s in specs if s.task == "burgers" and s.split == "test")
    x_test, y_test = load_train_xy(test_spec, "burgers", None)

    model = load_model("burgers", device)
    model.eval()
    cfg = cfg_for_variant(args, "loss3")

    records: list[dict[str, Any]] = []
    geom_rows: list[dict[str, Any]] = []
    gen = torch.Generator().manual_seed(int(args.seed) + int(args.round_id) * 1000)
    for dataset_idx in range(int(args.datasets)):
        dataset_id = f"burgers_loss3_aligned_r{args.round_id:02d}_d{dataset_idx:02d}"
        out_path = out_dir / f"{dataset_id}.pt"
        if out_path.exists() and not args.overwrite:
            records.append({"task": "burgers", "dataset_id": dataset_id, "tier": "far_range_pattern", "family": "loss3_aligned_adversarial", "n": int(args.samples_per_dataset), "seed": int(args.seed) + dataset_idx, "params": vars(args), "description": "Existing file reused.", "path": str(out_path), "skipped": True})
            continue

        if args.source_split == "test":
            base_x, base_y = x_test, y_test
            split_used = "test"
        elif args.source_split == "mixed" and dataset_idx % 2 == 1:
            base_x, base_y = x_test, y_test
            split_used = "test"
        else:
            base_x, base_y = x_train, y_train
            split_used = "train"
        n_base = int(base_x.shape[0])
        idx = torch.randperm(n_base, generator=gen)[: int(args.samples_per_dataset)]
        xb = base_x[idx].to(device, non_blocking=True)
        yb = base_y[idx].to(device, non_blocking=True)
        t0 = time.perf_counter()
        attack_result = attack_batch(model, xb, yb, "burgers", cfg)
        elapsed = time.perf_counter() - t0
        x_adv = attack_result.x_train.detach().cpu().float()
        y_adv = attack_result.y_train.detach().cpu().float()
        x_clean = xb.detach().cpu().float()
        stats = per_sample_delta_stats(x_clean, x_adv)
        stats.update({"dataset_id": dataset_id, "source_split": split_used, "seconds": elapsed})
        geom_rows.append(stats)

        x_save = x_adv.squeeze(-1).contiguous()
        y_save = y_adv.squeeze(-1).contiguous()
        metadata = {
            "dataset_id": dataset_id,
            "task": "burgers",
            "similarity_tier": "far_range_pattern",
            "family": "loss3_aligned_adversarial",
            "description": "Mechanism-probe generalization set built from raw loss3 adversarial inputs and solver labels.",
            "nsamples": int(args.samples_per_dataset),
            "seed": int(args.seed) + dataset_idx,
            "params": {
                "source_split": split_used,
                "attack_objective": "loss3",
                "attack_steps": int(args.attack_steps),
                "epsilon_fraction": float(args.epsilon_fraction),
                "eps_jitter_low": float(args.eps_jitter_low),
                "eps_jitter_high": float(args.eps_jitter_high),
                "alpha_jitter_low": float(args.alpha_jitter_low),
                "alpha_jitter_high": float(args.alpha_jitter_high),
                "random_start_fraction": float(args.random_start_fraction),
            },
            "training_reference": {
                "kernel": "gaussian",
                "correlation_length": 0.03,
                "bc": "periodic",
                "nu": 0.001,
                "t_final": 1.0,
                "nx": 1024,
                "seed": 45,
            },
            "solver": "training_pipeline_burgers_solver_target_for_model_input",
            "sample_metadata": [{"sample_index": i, "source_index": int(idx[i]), "source_split": split_used} for i in range(int(args.samples_per_dataset))],
            "attack_geometry": stats,
            "x_stats": finite_stats(x_save),
            "y_stats": finite_stats(y_save),
        }
        torch.save({"x": x_save, "y": y_save, "metadata": metadata}, out_path)
        records.append({"task": "burgers", "dataset_id": dataset_id, "tier": "far_range_pattern", "family": "loss3_aligned_adversarial", "n": int(args.samples_per_dataset), "seed": int(args.seed) + dataset_idx, "params": metadata["params"], "description": metadata["description"], "path": str(out_path), "bytes": out_path.stat().st_size})
        print(f"[{dataset_idx + 1:02d}/{int(args.datasets):02d}] {dataset_id} x=[{stats['x_adv_min']:.3f},{stats['x_adv_max']:.3f}] linf/rms={stats['linf_over_l2rms_mean']:.3f}", flush=True)

    write_manifest(out_root, records)
    write_json(out_root / "loss3_aligned_geometry.json", geom_rows)
    if geom_rows:
        summary = {key: float(np.mean([row[key] for row in geom_rows])) for key in ["delta_l2_rms_mean", "delta_linf_mean", "linf_over_l2rms_mean", "x_adv_min", "x_adv_max", "oob_mean", "oob_max"]}
        write_json(out_root / "loss3_aligned_geometry_summary.json", summary)
    print(f"[done] wrote {len(records)} loss3-aligned Burgers datasets to {out_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
