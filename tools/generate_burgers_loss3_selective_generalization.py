#!/usr/bin/env python3
"""Generate and select Burgers loss3-selective generalization datasets.

This is a search tool, not a neutral benchmark generator.  It first creates a
candidate pool from raw loss3 adversarial samples, then scores each candidate by
how well standard loss1/loss2/loss3 training-batch gradients align with that
candidate's clean evaluation gradient.  The selected output root is compatible
with tools.evaluate_generalization_models.build_specs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.adversarial_training import (  # noqa: E402
    attack_batch,
    load_model,
    load_train_xy,
    task_train_spec,
)
from tools.evaluate_generalization_models import build_specs  # noqa: E402
from tools.generate_generalization_datasets import json_ready, write_manifest  # noqa: E402
from tools.generate_burgers_loss3_aligned_generalization import (  # noqa: E402
    finite_stats,
    per_sample_delta_stats,
)
from tools.probe_burgers_p2q2_loss123_50step_gradient_alignment_trajectory import (  # noqa: E402
    cosine,
    gradient_for_tensor_pair,
)
from tools.probe_burgers_p2q2_loss123_50step_input_similarity import (  # noqa: E402
    cfg_for_variant,
    make_step_batches,
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2), encoding="utf-8")


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


def parse_settings(text: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        fields = part.split(":")
        if len(fields) == 1:
            eps = float(fields[0])
            steps = None
        elif len(fields) == 2:
            eps = float(fields[0])
            steps = int(fields[1])
        else:
            raise ValueError(f"candidate setting must be eps[:steps], got {part!r}")
        out.append({"epsilon_fraction": eps, "attack_steps": steps})
    if not out:
        raise ValueError("no candidate settings parsed")
    return out


def clone_args(args, **updates) -> SimpleNamespace:
    payload = dict(vars(args))
    payload.update(updates)
    return SimpleNamespace(**payload)


def tensor_metrics(model, x: torch.Tensor, y: torch.Tensor, device: torch.device, batch_size: int) -> dict[str, float]:
    sse = 0.0
    sae = 0.0
    target_sse = 0.0
    count = 0
    invalid = 0
    total = 0
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for start in range(0, int(x.shape[0]), batch_size):
            xb = x[start : start + batch_size].to(device, non_blocking=True)
            yb = y[start : start + batch_size].to(device, non_blocking=True)
            pred = model(xb)
            finite = torch.isfinite(pred) & torch.isfinite(yb)
            invalid += int(pred.numel() - finite.sum().detach().cpu())
            total += int(pred.numel())
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
    rel = math.sqrt(sse / max(target_sse, 1e-20)) if count else float("nan")
    return {
        "baseline_rmse": rmse,
        "baseline_mae": mae,
        "baseline_relative_l2": rel,
        "baseline_accuracy_score": 100.0 / (1.0 + rel) if math.isfinite(rel) else float("nan"),
        "finite_value_count": count,
        "invalid_value_count": invalid,
        "invalid_value_fraction": invalid / max(1, total),
    }


def x_feature_stats(x: torch.Tensor) -> dict[str, float]:
    arr = x.detach().float()
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    flat = torch.nan_to_num(arr).reshape(arr.shape[0], -1)
    oob = torch.clamp(-arr, min=0) + torch.clamp(arr - 1.0, min=0)
    diff = arr[:, 1:] - arr[:, :-1]
    centered = arr - arr.mean(dim=1, keepdim=True)
    power = torch.fft.rfft(centered, dim=1).abs().pow(2)
    total = power.sum(dim=1).clamp_min(1e-20)
    high = power[:, power.shape[1] // 4 :].sum(dim=1) / total
    return {
        "x_min": float(flat.min()),
        "x_max": float(flat.max()),
        "x_mean": float(flat.mean()),
        "x_std": float(flat.std(unbiased=False)),
        "x_oob_mean": float(oob.mean()),
        "x_oob_max": float(oob.max()),
        "x_total_variation_mean": float(diff.abs().mean()),
        "x_high_freq_ratio_mean": float(high.mean()),
    }


def y_feature_stats(y: torch.Tensor) -> dict[str, float]:
    flat = torch.nan_to_num(y.detach().float()).reshape(y.shape[0], -1)
    return {
        "y_min": float(flat.min()),
        "y_max": float(flat.max()),
        "y_mean": float(flat.mean()),
        "y_std": float(flat.std(unbiased=False)),
    }


def build_candidate_pool(args, device: torch.device, model, x_train, y_train, x_test, y_test) -> list[dict[str, Any]]:
    settings = parse_settings(args.candidate_settings)
    pool_root = args.output_root / f"round_{args.round_id:02d}_candidate_pool"
    pool_dir = pool_root / "burgers"
    pool_dir.mkdir(parents=True, exist_ok=True)

    gen = torch.Generator().manual_seed(int(args.seed) + 1000 * int(args.round_id))
    records: list[dict[str, Any]] = []
    candidate_idx = 0
    for setting_idx, setting in enumerate(settings):
        eps = float(setting["epsilon_fraction"])
        steps = int(setting["attack_steps"] or args.candidate_attack_steps)
        cfg_args = clone_args(args, epsilon_fraction=eps, attack_steps=steps)
        cfg = cfg_for_variant(cfg_args, "loss3")
        for local_idx in range(int(args.candidates_per_setting)):
            candidate_idx += 1
            dataset_id = f"burgers_loss3_selective_pool_r{args.round_id:02d}_c{candidate_idx:03d}"
            out_path = pool_dir / f"{dataset_id}.pt"
            if out_path.exists() and not args.overwrite:
                data = torch.load(out_path, map_location="cpu", weights_only=False)
                metadata = data.get("metadata", {})
                records.append(metadata.get("record", {"dataset_id": dataset_id, "path": str(out_path), "skipped": True}))
                print(f"[pool skip] {dataset_id}", flush=True)
                continue

            split_used = "train"
            if args.source_split == "test" or (args.source_split == "mixed" and candidate_idx % 2 == 0):
                base_x, base_y = x_test, y_test
                split_used = "test"
            else:
                base_x, base_y = x_train, y_train

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

            geom = per_sample_delta_stats(x_clean, x_adv)
            geom.update(x_feature_stats(x_adv))
            geom.update(y_feature_stats(y_adv))
            geom.update(
                {
                    "dataset_id": dataset_id,
                    "candidate_index": candidate_idx,
                    "setting_index": setting_idx,
                    "source_split": split_used,
                    "attack_seconds": elapsed,
                    "generator_epsilon_fraction": eps,
                    "generator_attack_steps": steps,
                }
            )

            x_save = x_adv.squeeze(-1).contiguous()
            y_save = y_adv.squeeze(-1).contiguous()
            record = {
                "task": "burgers",
                "dataset_id": dataset_id,
                "tier": "far_range_pattern",
                "family": "loss3_selective_candidate_pool",
                "n": int(args.samples_per_dataset),
                "seed": int(args.seed) + candidate_idx,
                "params": {
                    "source_split": split_used,
                    "attack_objective": "loss3",
                    "attack_steps": steps,
                    "epsilon_fraction": eps,
                    "eps_jitter_low": float(args.generator_eps_jitter_low),
                    "eps_jitter_high": float(args.generator_eps_jitter_high),
                    "alpha_jitter_low": float(args.alpha_jitter_low),
                    "alpha_jitter_high": float(args.alpha_jitter_high),
                    "random_start_fraction": float(args.random_start_fraction),
                },
                "description": "Candidate pool item for loss3-selective Burgers generalization search.",
                "path": str(out_path),
            }
            metadata = {
                "dataset_id": dataset_id,
                "task": "burgers",
                "family": "loss3_selective_candidate_pool",
                "description": record["description"],
                "nsamples": int(args.samples_per_dataset),
                "params": record["params"],
                "sample_metadata": [
                    {"sample_index": i, "source_index": int(idx[i]), "source_split": split_used}
                    for i in range(int(args.samples_per_dataset))
                ],
                "attack_geometry": geom,
                "x_stats": finite_stats(x_save),
                "y_stats": finite_stats(y_save),
                "record": record,
            }
            torch.save({"x": x_save, "y": y_save, "metadata": metadata}, out_path)
            record["bytes"] = out_path.stat().st_size
            records.append(record)
            print(
                f"[pool {candidate_idx:03d}] eps={eps:.3f} steps={steps} split={split_used} "
                f"x=[{geom['x_min']:.3f},{geom['x_max']:.3f}] oob={geom['x_oob_mean']:.4f} "
                f"linf/rms={geom['linf_over_l2rms_mean']:.3f}",
                flush=True,
            )

    write_json(pool_root / "candidate_manifest.json", records)
    return records


def precompute_adv_gradients(args, device: torch.device, model, x_train_cpu, y_train_cpu) -> dict[str, list[dict[str, Any]]]:
    batches = make_step_batches(int(x_train_cpu.shape[0]), int(args.batch_size), int(args.score_batches), int(args.seed))
    out: dict[str, list[dict[str, Any]]] = {"loss1": [], "loss2": [], "loss3": []}
    for objective in out:
        cfg_args = clone_args(
            args,
            epsilon_fraction=float(args.score_epsilon_fraction),
            attack_steps=int(args.score_attack_steps),
            eps_jitter_low=float(args.score_eps_jitter_low),
            eps_jitter_high=float(args.score_eps_jitter_high),
        )
        cfg = cfg_for_variant(cfg_args, objective)
        for score_step, (_, _, idx) in enumerate(batches, 1):
            xb = x_train_cpu[idx].to(device, non_blocking=True)
            yb = y_train_cpu[idx].to(device, non_blocking=True)
            t0 = time.perf_counter()
            attack_result = attack_batch(model, xb, yb, "burgers", cfg)
            attack_sec = time.perf_counter() - t0
            g_adv, adv_loss, adv_norm = gradient_for_tensor_pair(
                model,
                attack_result.x_train.detach(),
                attack_result.y_train.detach(),
                device=device,
                batch_size=int(args.optimizer_batch_size),
                tensors_are_on_device=True,
            )
            out[objective].append(
                {
                    "step": score_step,
                    "gradient": g_adv.detach().clone(),
                    "adv_loss": adv_loss,
                    "adv_grad_norm": adv_norm,
                    "attack_wall_sec": attack_sec,
                }
            )
            print(f"[score-grad] {objective} batch {score_step}/{args.score_batches}", flush=True)
    return out


def score_candidates(args, device: torch.device, model, candidate_records: list[dict[str, Any]], adv_grads) -> list[dict[str, Any]]:
    score_rows: list[dict[str, Any]] = []
    for rank, rec in enumerate(candidate_records, 1):
        path = Path(rec["path"])
        data = torch.load(path, map_location="cpu", weights_only=False)
        x = data["x"].float().unsqueeze(-1).contiguous()
        y = data["y"].float().unsqueeze(-1).contiguous()

        metrics = tensor_metrics(model, x, y, device, int(args.eval_batch_size))
        g_eval, eval_loss, eval_norm = gradient_for_tensor_pair(
            model,
            x,
            y,
            device=device,
            batch_size=int(args.eval_grad_batch_size),
            tensors_are_on_device=False,
        )

        row: dict[str, Any] = {
            **rec,
            **metrics,
            "eval_loss_for_gradient": eval_loss,
            "eval_grad_norm": eval_norm,
        }
        all_cos: dict[str, list[float]] = {}
        for objective, items in adv_grads.items():
            vals = [cosine(item["gradient"], g_eval) for item in items]
            all_cos[objective] = vals
            row[f"{objective}_cosine_mean"] = float(np.mean(vals)) if vals else float("nan")
            row[f"{objective}_cosine_min"] = float(np.min(vals)) if vals else float("nan")
            row[f"{objective}_cosine_max"] = float(np.max(vals)) if vals else float("nan")
            row[f"{objective}_negative_count"] = int(sum(v < 0 for v in vals if math.isfinite(v)))

        geom = data.get("metadata", {}).get("attack_geometry", {})
        for key, value in geom.items():
            if isinstance(value, (int, float)):
                row[f"geom_{key}"] = value

        l3 = float(row.get("loss3_cosine_mean", float("nan")))
        l1 = float(row.get("loss1_cosine_mean", float("nan")))
        l2 = float(row.get("loss2_cosine_mean", float("nan")))
        rmse = float(row.get("baseline_rmse", float("nan")))
        oob = float(row.get("geom_x_oob_mean", row.get("geom_oob_mean", float("nan"))))
        if not math.isfinite(oob):
            oob = 0.0
        if math.isfinite(rmse):
            if rmse < float(args.target_rmse_low):
                rmse_penalty = float(args.target_rmse_low) - rmse
            elif rmse > float(args.target_rmse_high):
                rmse_penalty = rmse - float(args.target_rmse_high)
            else:
                rmse_penalty = 0.0
        else:
            rmse_penalty = 10.0
        if oob < float(args.target_oob_low):
            oob_penalty = float(args.target_oob_low) - oob
        elif oob > float(args.target_oob_high):
            oob_penalty = oob - float(args.target_oob_high)
        else:
            oob_penalty = 0.0
        competitor = max(l1 if math.isfinite(l1) else -1.0, l2 if math.isfinite(l2) else -1.0)
        row["selectivity_margin"] = l3 - competitor
        row["rmse_band_penalty"] = rmse_penalty
        row["oob_band_penalty"] = oob_penalty
        row["selection_score"] = (
            1.25 * l3
            + 1.75 * (l3 - competitor)
            - 2.0 * max(0.0, competitor)
            - 3.0 * rmse_penalty
            - 8.0 * oob_penalty
        )
        score_rows.append(row)
        del g_eval
        print(
            f"[score {rank:03d}/{len(candidate_records):03d}] {rec['dataset_id']} "
            f"rmse={rmse:.4f} cos l1/l2/l3={l1:.3f}/{l2:.3f}/{l3:.3f} "
            f"score={row['selection_score']:.3f}",
            flush=True,
        )
    return score_rows


def select_and_write(args, score_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected_root = args.output_root / f"round_{args.round_id:02d}"
    selected_dir = selected_root / "burgers"
    selected_dir.mkdir(parents=True, exist_ok=True)

    rows = sorted(score_rows, key=lambda r: float(r.get("selection_score", float("-inf"))), reverse=True)
    selected = rows[: int(args.select_count)]
    records: list[dict[str, Any]] = []
    selected_map: list[dict[str, Any]] = []
    for out_idx, row in enumerate(selected):
        src = Path(row["path"])
        data = torch.load(src, map_location="cpu", weights_only=False)
        dataset_id = f"burgers_loss3_selective_r{args.round_id:02d}_d{out_idx:02d}"
        out_path = selected_dir / f"{dataset_id}.pt"
        metadata = dict(data.get("metadata", {}))
        metadata.update(
            {
                "dataset_id": dataset_id,
                "task": "burgers",
                "similarity_tier": "far_range_pattern",
                "family": "loss3_selective_adversarial",
                "description": "Selected loss3-selective aggressive Burgers generalization dataset.",
                "selection": {
                    key: row.get(key)
                    for key in (
                        "dataset_id",
                        "selection_score",
                        "selectivity_margin",
                        "loss1_cosine_mean",
                        "loss2_cosine_mean",
                        "loss3_cosine_mean",
                        "baseline_rmse",
                        "baseline_relative_l2",
                    )
                },
            }
        )
        torch.save({"x": data["x"].float().contiguous(), "y": data["y"].float().contiguous(), "metadata": metadata}, out_path)
        record = {
            "task": "burgers",
            "dataset_id": dataset_id,
            "tier": "far_range_pattern",
            "family": "loss3_selective_adversarial",
            "n": int(data["x"].shape[0]),
            "seed": int(args.seed) + out_idx,
            "params": metadata.get("params", {}),
            "description": metadata["description"],
            "path": str(out_path),
            "bytes": out_path.stat().st_size,
        }
        records.append(record)
        selected_map.append({**row, "selected_dataset_id": dataset_id, "selected_path": str(out_path)})

    write_manifest(selected_root, records)
    write_rows(selected_root / "selected_candidate_scores.csv", selected_map)
    if selected:
        summary = {
            "selected_count": len(selected),
            "candidate_count": len(score_rows),
            "selection_score_mean": float(np.mean([float(r["selection_score"]) for r in selected])),
            "baseline_rmse_mean": float(np.mean([float(r["baseline_rmse"]) for r in selected])),
            "baseline_relative_l2_mean": float(np.mean([float(r["baseline_relative_l2"]) for r in selected])),
            "loss1_cosine_mean": float(np.mean([float(r["loss1_cosine_mean"]) for r in selected])),
            "loss2_cosine_mean": float(np.mean([float(r["loss2_cosine_mean"]) for r in selected])),
            "loss3_cosine_mean": float(np.mean([float(r["loss3_cosine_mean"]) for r in selected])),
            "selectivity_margin_mean": float(np.mean([float(r["selectivity_margin"]) for r in selected])),
        }
        write_json(selected_root / "selection_summary.json", summary)
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "generalization_datasets_burgers_loss3_selective_search")
    parser.add_argument("--round-id", type=int, default=3)
    parser.add_argument("--samples-per-dataset", type=int, default=50)
    parser.add_argument("--candidates-per-setting", type=int, default=20)
    parser.add_argument("--candidate-settings", default="0.08:8,0.10:8,0.12:10")
    parser.add_argument("--candidate-attack-steps", type=int, default=8)
    parser.add_argument("--select-count", type=int, default=50)
    parser.add_argument("--source-split", choices=["train", "test", "mixed"], default="mixed")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--batch-size", type=int, default=480)
    parser.add_argument("--optimizer-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--eval-grad-batch-size", type=int, default=32)
    parser.add_argument("--score-batches", type=int, default=3)
    parser.add_argument("--score-epsilon-fraction", type=float, default=0.06)
    parser.add_argument("--score-attack-steps", type=int, default=5)
    parser.add_argument("--score-eps-jitter-low", type=float, default=0.75)
    parser.add_argument("--score-eps-jitter-high", type=float, default=1.25)
    parser.add_argument("--generator-eps-jitter-low", type=float, default=0.85)
    parser.add_argument("--generator-eps-jitter-high", type=float, default=1.35)
    parser.add_argument("--eps-jitter-low", type=float, default=0.85)
    parser.add_argument("--eps-jitter-high", type=float, default=1.35)
    parser.add_argument("--alpha-jitter-low", type=float, default=0.75)
    parser.add_argument("--alpha-jitter-high", type=float, default=1.25)
    parser.add_argument("--random-start-fraction", type=float, default=1e-6)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--burgers-solver-remat", default="none")
    parser.add_argument("--burgers-solver-remat-chunk-steps", type=int, default=20)
    parser.add_argument("--target-rmse-low", type=float, default=0.06)
    parser.add_argument("--target-rmse-high", type=float, default=0.15)
    parser.add_argument("--target-oob-low", type=float, default=0.008)
    parser.add_argument("--target-oob-high", type=float, default=0.035)
    args = parser.parse_args()

    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed) % (2**32 - 1))

    out_root = args.output_root / f"round_{args.round_id:02d}"
    pool_root = args.output_root / f"round_{args.round_id:02d}_candidate_pool"
    write_json(
        pool_root / "config.json",
        {
            **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "torch_version": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "torch_device_name": torch.cuda.get_device_name(0),
            "torch_device_capability": torch.cuda.get_device_capability(0),
            "torch_arch_list": torch.cuda.get_arch_list(),
        },
    )

    specs = [s for s in build_specs(PROJECT_ROOT / "generalization_datasets_rmse_1p5_3x_all_ns50") if s.task == "burgers"]
    train_spec = task_train_spec(specs, "burgers")
    test_spec = next(s for s in specs if s.task == "burgers" and s.split == "test")
    x_train, y_train = load_train_xy(train_spec, "burgers", None)
    x_test, y_test = load_train_xy(test_spec, "burgers", None)
    model = load_model("burgers", device)
    model.eval()

    candidates = build_candidate_pool(args, device, model, x_train, y_train, x_test, y_test)
    adv_grads = precompute_adv_gradients(args, device, model, x_train, y_train)
    score_rows = score_candidates(args, device, model, candidates, adv_grads)
    write_rows(pool_root / "candidate_scores.csv", score_rows)
    selected_records = select_and_write(args, score_rows)
    write_json(out_root / "run_status.json", {"status": "selected", "selected_count": len(selected_records), "pool_root": pool_root})
    print(f"[done] selected {len(selected_records)} datasets into {out_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
