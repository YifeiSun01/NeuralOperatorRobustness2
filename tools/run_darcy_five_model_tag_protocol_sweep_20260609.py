#!/usr/bin/env python3
"""Sweep Darcy/C-flow tag protocols over steps and perturbation budgets."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
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
from tools.run_darcy_five_model_generalization_tag_20260609 import (
    gpu_preflight,
    load_pair,
    model_specs,
    write_csv,
)


@dataclass(frozen=True)
class Protocol:
    name: str
    steps: int
    epsilon_fraction: float


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


def default_protocols() -> list[Protocol]:
    base = 0.025
    quarter = base / 4.0
    return [
        Protocol("eps0p025_steps001", 1, base),
        Protocol("eps0p025_steps005", 5, base),
        Protocol("eps0p025_steps010", 10, base),
        Protocol("eps0p00625_steps001", 1, quarter),
        Protocol("eps0p00625_steps005", 5, quarter),
        Protocol("eps0p00625_steps010", 10, quarter),
        Protocol("eps0p00625_steps020", 20, quarter),
    ]


def tensor_to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def run_attack_batch(
    model,
    xb: torch.Tensor,
    yb: torch.Tensor,
    cfg: dict[str, Any],
    protocol: Protocol,
):
    return adv.binary_darcy_replace_attack(
        model,
        xb,
        yb,
        steps=protocol.steps,
        epsilon_fraction=protocol.epsilon_fraction,
        jitter_low=1.0,
        jitter_high=1.0,
        random_pool_multiplier=1.0,
        random_score_noise=0.0,
        cfg=cfg,
    )


def evaluate_model_protocol(
    spec,
    protocol: Protocol,
    dataset_paths: list[Path],
    cfg: dict[str, Any],
    batch_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    device = torch.device("cuda")
    model = adv.load_model("darcy", device, model_checkpoint_override=spec.checkpoint)
    model.eval()

    dataset_rows: list[dict[str, Any]] = []
    all_clean: list[float] = []
    all_adv: list[float] = []
    all_gain: list[float] = []
    all_rel: list[float] = []
    wall_start = time.perf_counter()

    try:
        for ds_path in dataset_paths:
            x_all, y_all = load_pair(ds_path, device)
            ds_clean: list[float] = []
            ds_adv: list[float] = []
            ds_gain: list[float] = []
            ds_rel: list[float] = []
            ds_boundary: list[float] = []
            ds_flip: list[float] = []
            ds_delta_l2: list[float] = []
            ds_delta_linf: list[float] = []

            for start in range(0, int(x_all.shape[0]), batch_size):
                end = min(start + batch_size, int(x_all.shape[0]))
                result = run_attack_batch(model, x_all[start:end], y_all[start:end], cfg, protocol)
                si = result.sample_info
                clean = tensor_to_numpy(si["clean_loss_before_attack"])
                adv_after = tensor_to_numpy(si["adv_loss_after_attack"])
                gain = tensor_to_numpy(si["attack_loss_gain"])
                rel = tensor_to_numpy(si["attack_loss_gain_relative"])

                ds_clean.extend(float(v) for v in clean)
                ds_adv.extend(float(v) for v in adv_after)
                ds_gain.extend(float(v) for v in gain)
                ds_rel.extend(float(v) for v in rel)
                ds_boundary.append(float(result.info.get("boundary_ratio_mean", np.nan)))
                ds_flip.append(float(result.info.get("darcy_flip_fraction", np.nan)))
                ds_delta_l2.append(float(result.info.get("delta_l2_rms_mean", np.nan)))
                ds_delta_linf.append(float(result.info.get("delta_linf_mean", np.nan)))

            all_clean.extend(ds_clean)
            all_adv.extend(ds_adv)
            all_gain.extend(ds_gain)
            all_rel.extend(ds_rel)
            dataset_rows.append(
                {
                    "protocol": protocol.name,
                    "steps": protocol.steps,
                    "epsilon_fraction": protocol.epsilon_fraction,
                    "model": spec.name,
                    "dataset": ds_path.name,
                    "batch_size": batch_size,
                    "mean_clean_loss": float(np.nanmean(ds_clean)),
                    "mean_adv_loss": float(np.nanmean(ds_adv)),
                    "mean_attack_loss_gain": float(np.nanmean(ds_gain)),
                    "mean_attack_loss_gain_relative": float(np.nanmean(ds_rel)),
                    "boundary_ratio_mean": float(np.nanmean(ds_boundary)),
                    "darcy_flip_fraction": float(np.nanmean(ds_flip)),
                    "delta_l2_rms_mean": float(np.nanmean(ds_delta_l2)),
                    "delta_linf_mean": float(np.nanmean(ds_delta_linf)),
                }
            )

        wall_sec = time.perf_counter() - wall_start
        summary = {
            "protocol": protocol.name,
            "steps": protocol.steps,
            "epsilon_fraction": protocol.epsilon_fraction,
            "model": spec.name,
            "checkpoint": str(spec.checkpoint),
            "batch_size": batch_size,
            "sample_count": len(all_gain),
            "dataset_count": len(dataset_paths),
            "wall_sec": wall_sec,
            "mean_clean_loss": float(np.nanmean(all_clean)),
            "mean_adv_loss": float(np.nanmean(all_adv)),
            "mean_attack_loss_gain": float(np.nanmean(all_gain)),
            "mean_attack_loss_gain_relative": float(np.nanmean(all_rel)),
            "boundary_ratio_mean": float(np.nanmean([r["boundary_ratio_mean"] for r in dataset_rows])),
            "darcy_flip_fraction": float(np.nanmean([r["darcy_flip_fraction"] for r in dataset_rows])),
            "delta_l2_rms_mean": float(np.nanmean([r["delta_l2_rms_mean"] for r in dataset_rows])),
            "delta_linf_mean": float(np.nanmean([r["delta_linf_mean"] for r in dataset_rows])),
        }
        return summary, dataset_rows
    finally:
        del model
        torch.cuda.empty_cache()
        gc.collect()


def protocol_winner_counts(dataset_rows: list[dict[str, Any]]) -> dict[str, int]:
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for row in dataset_rows:
        by_dataset.setdefault(row["dataset"], []).append(row)
    winners: dict[str, int] = {}
    for rows in by_dataset.values():
        winner = min(rows, key=lambda row: float(row["mean_attack_loss_gain"]))["model"]
        winners[winner] = winners.get(winner, 0) + 1
    return winners


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default=str(PROJECT_ROOT / "generalization_datasets_darcy_lossdrop50_selected_20260607/darcy"))
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "forensics/darcy_five_model_tag_protocol_sweep_20260609"))
    parser.add_argument("--batch-size", type=int, default=48)
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
    protocols = default_protocols()
    combined_summary: list[dict[str, Any]] = []
    combined_dataset_rows: list[dict[str, Any]] = []
    protocol_results: list[dict[str, Any]] = []
    sweep_start = time.perf_counter()

    manifest = {
        "dataset_dir": str(Path(args.dataset_dir)),
        "dataset_count": len(dataset_paths),
        "batch_size": args.batch_size,
        "models": [{"name": spec.name, "checkpoint": str(spec.checkpoint)} for spec in specs],
        "protocols": [protocol.__dict__ for protocol in protocols],
        "shared_objective": "loss3",
        "attack_type": "binary_steepest_replace",
        "gpu_preflight": gpu_info,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for protocol in protocols:
        protocol_dir = out_dir / protocol.name
        protocol_dir.mkdir(parents=True, exist_ok=True)
        protocol_summary: list[dict[str, Any]] = []
        protocol_dataset_rows: list[dict[str, Any]] = []

        for spec in specs:
            summary, dataset_rows = evaluate_model_protocol(spec, protocol, dataset_paths, cfg, args.batch_size)
            protocol_summary.append(summary)
            protocol_dataset_rows.extend(dataset_rows)
            combined_summary.append(summary)
            combined_dataset_rows.extend(dataset_rows)
            write_csv(protocol_dir / "summary_by_model_partial.csv", sorted(protocol_summary, key=lambda r: r["mean_attack_loss_gain"]), list(protocol_summary[0].keys()))

        protocol_summary = sorted(protocol_summary, key=lambda r: r["mean_attack_loss_gain"])
        winners = protocol_winner_counts(protocol_dataset_rows)
        protocol_result = {
            "protocol": protocol.name,
            "steps": protocol.steps,
            "epsilon_fraction": protocol.epsilon_fraction,
            "least_burst_model": protocol_summary[0]["model"],
            "least_burst_mean_attack_loss_gain": protocol_summary[0]["mean_attack_loss_gain"],
            "winner_counts": winners,
            "ranked_models": protocol_summary,
        }
        protocol_results.append(protocol_result)
        write_csv(protocol_dir / "summary_by_model.csv", protocol_summary, list(protocol_summary[0].keys()))
        write_csv(protocol_dir / "summary_by_dataset.csv", protocol_dataset_rows, list(protocol_dataset_rows[0].keys()))
        (protocol_dir / "result.json").write_text(json.dumps(protocol_result, indent=2), encoding="utf-8")
        write_csv(out_dir / "combined_summary_partial.csv", combined_summary, list(combined_summary[0].keys()))

    combined_summary_sorted = sorted(combined_summary, key=lambda r: (r["protocol"], r["mean_attack_loss_gain"]))
    write_csv(out_dir / "combined_summary_by_model.csv", combined_summary_sorted, list(combined_summary_sorted[0].keys()))
    write_csv(out_dir / "combined_summary_by_dataset.csv", combined_dataset_rows, list(combined_dataset_rows[0].keys()))
    result = {
        "total_wall_sec": time.perf_counter() - sweep_start,
        "protocol_results": protocol_results,
        "output_dir": str(out_dir),
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
