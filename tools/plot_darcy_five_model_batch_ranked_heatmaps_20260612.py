#!/usr/bin/env python3
"""Batch 100-step Darcy attacks, then plot ranked five-model heatmaps."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.35")

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tools.adversarial_training as adv
import tools.plot_darcy_five_model_attack_heatmaps_20260612 as heat

DEFAULT_ATTACK20_TABLE = PROJECT_ROOT / "analysis_outputs" / "darcy_attack20_52datasets_50samples_20260612_attack20_1000c_52datasets_50samples" / "all_samples.csv"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "analysis_outputs"
DEFAULT_VIZ_ROOT = PROJECT_ROOT / "visualizations"
MODEL_ORDER = [m.name for m in heat.MODELS]
TRAINED_MODEL_NAMES = ["loss1", "loss2", "loss3", "physics loss"]
PHYSICS_ALIASES = {"physics", "physics loss"}

RANK_VARIANTS = [
    ("index0", "fixed source index 0"),
    ("loss3_best", "rank 1 / largest Loss3 advantage"),
    ("loss3_upper_quartile", "upper quartile by Loss3 advantage"),
    ("loss3_median", "median by Loss3 advantage"),
    ("loss3_lower_quartile", "lower quartile by Loss3 advantage"),
    ("loss3_worst", "worst by Loss3 advantage"),
]

ATTACK_RESULT_FIELDS = [
    "model",
    "dataset_id",
    "split",
    "sample_index",
    "attack_steps",
    "epsilon_fraction",
    "clean_loss_before_attack",
    "adv_loss_after_attack",
    "attack_loss_gain",
    "attack_loss_gain_relative",
    "elapsed_seconds",
]

SELECTION_FIELDS = [
    "variant",
    "sample_id",
    "dataset_rank",
    "rank_within_dataset_best_is_1",
    "dataset_id",
    "split",
    "sample_index",
    "loss3_adv_loss",
    "baseline_adv_loss",
    "loss1_adv_loss",
    "loss2_adv_loss",
    "physics_loss_adv_loss",
    "best_other_model",
    "best_other_adv_loss",
    "loss3_margin_vs_best_other",
    "loss3_margin_vs_mean_other",
]


def slug(value: str) -> str:
    return heat.slug(value)


def load_attack20_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def canonical_model_name(name: str) -> str:
    if name in PHYSICS_ALIASES:
        return "physics loss"
    return name


def choose_dataset_ids(path: Path, dataset_count: int) -> list[str]:
    rows = load_attack20_rows(path)
    grouped: dict[tuple[str, int], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        if row.get("split") != "generalization":
            continue
        key = (row["dataset_id"], int(float(row["source_sample_index"])))
        grouped[key][canonical_model_name(row["model"])] = row
    best_by_dataset: dict[str, float] = {}
    for (dataset_id, _idx), by_model in grouped.items():
        needed = {"loss1", "loss2", "loss3", "physics loss"}
        if not needed.issubset(by_model):
            continue
        loss3 = float(by_model["loss3"]["adv_loss_after_attack"])
        others = [float(by_model[m]["adv_loss_after_attack"]) for m in needed if m != "loss3"]
        score = min(others) - loss3
        if dataset_id not in best_by_dataset or score > best_by_dataset[dataset_id]:
            best_by_dataset[dataset_id] = score
    ordered = sorted(best_by_dataset, key=lambda d: best_by_dataset[d], reverse=True)
    return ordered[: int(dataset_count)]


def load_dataset_batch(dataset_paths: dict[str, Path], dataset_id: str, indices: list[int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    payload = heat.torch_load(dataset_paths[dataset_id])
    x, y = heat.tensor_xy(payload, "darcy")
    idx = torch.as_tensor(indices, dtype=torch.long)
    xb = x.index_select(0, idx).contiguous().to(device)
    yb = y.index_select(0, idx).contiguous().to(device)
    return xb, yb


def batch_darcy_attack_trace(
    model,
    xb: torch.Tensor,
    yb: torch.Tensor,
    *,
    steps: int,
    epsilon_fraction: float,
    keep_arrays: bool,
) -> list[dict[str, Any]]:
    cfg = heat.attack_cfg()
    was_training = model.training
    model.eval()
    original_requires_grad = [param.requires_grad for param in model.parameters()]
    for param in model.parameters():
        param.requires_grad_(False)

    x0 = xb.detach()
    x_adv = x0.detach()
    b = int(x0.shape[0])
    spatial_shape = x0.shape[1:-1] if x0.ndim == 4 else x0.shape[1:]
    n_pix = int(np.prod(spatial_shape))
    budget = max(1, min(n_pix, int(round(float(epsilon_fraction) * n_pix))))
    history: list[np.ndarray] = []
    try:
        with torch.no_grad():
            _, clean_loss_samples, _, _ = adv.darcy_attack_objective_loss(
                model,
                x0,
                x0,
                yb,
                cfg,
                allow_solver_backward=False,
            )
            history.append(clean_loss_samples.detach().float().cpu().numpy())

        for _step in range(max(1, int(steps))):
            x_score = x_adv.detach().clone().requires_grad_(True)
            loss, _, _, _ = adv.darcy_attack_objective_loss(
                model,
                x0,
                x_score,
                yb,
                cfg,
                allow_solver_backward=True,
            )
            grad = torch.autograd.grad(loss, x_score, only_inputs=True)[0].detach()
            grad = torch.nan_to_num(grad)
            flat_x = x_score.detach().reshape(b, -1)
            flat_grad = grad.reshape(b, -1)
            lo = x0.reshape(b, -1).min(dim=1).values.reshape(b, 1)
            hi = x0.reshape(b, -1).max(dim=1).values.reshape(b, 1)
            midpoint = (lo + hi) * 0.5
            other = torch.where(flat_x > midpoint, lo.expand_as(flat_x), hi.expand_as(flat_x))
            score = flat_grad * (other - flat_x)
            flat_adv = flat_x.clone()
            top_idx = torch.topk(score, k=budget, dim=1, largest=True).indices
            for i in range(b):
                flat_adv[i, top_idx[i]] = other[i, top_idx[i]]
            x_adv = flat_adv.reshape_as(x0).detach()
            with torch.no_grad():
                _, step_loss_samples, _, _ = adv.darcy_attack_objective_loss(
                    model,
                    x0,
                    x_adv,
                    yb,
                    cfg,
                    allow_solver_backward=False,
                )
                history.append(step_loss_samples.detach().float().cpu().numpy())

        hist = np.stack(history, axis=0).astype(np.float64, copy=False)
        final_loss = hist[-1]
        clean_loss = hist[0]
        gain = final_loss - clean_loss
        rel_gain = np.divide(gain, np.maximum(np.abs(clean_loss), 1e-20))

        arrays: dict[str, Any] = {}
        if keep_arrays:
            with torch.no_grad():
                y_train = adv.solver_target_for_model_input("darcy", x_adv, yb, cfg, allow_target_grad=False).detach()
                model_output = model(x_adv).detach()
            diff = model_output - y_train
            delta = x_adv - x0
            solver_flat = y_train.reshape(b, -1)
            diff_flat = diff.reshape(b, -1)
            rel_l2 = (
                torch.linalg.vector_norm(diff_flat, dim=1)
                / torch.linalg.vector_norm(solver_flat, dim=1).clamp_min(1e-20)
            ).detach().cpu().numpy()
            rmse = torch.sqrt(torch.mean(diff_flat * diff_flat, dim=1).clamp_min(0.0)).detach().cpu().numpy()
            changed = (delta.reshape(b, -1).abs() > 1e-12).float().mean(dim=1).detach().cpu().numpy()
            delta_l2 = torch.sqrt(torch.mean(delta.reshape(b, -1) ** 2, dim=1)).detach().cpu().numpy()
            arrays = {
                "x0": x0.detach().cpu(),
                "delta": delta.detach().cpu(),
                "x_adv": x_adv.detach().cpu(),
                "model_output": model_output.detach().cpu(),
                "solver_output": y_train.detach().cpu(),
                "model_minus_solver": diff.detach().cpu(),
                "adv_relative_l2": rel_l2,
                "adv_rmse": rmse,
                "delta_l0_fraction": changed,
                "delta_l2_rms": delta_l2,
            }
    finally:
        for param, flag in zip(model.parameters(), original_requires_grad):
            param.requires_grad_(flag)
        if was_training:
            model.train()

    out: list[dict[str, Any]] = []
    for i in range(b):
        row = {
            "clean_loss_before_attack": float(clean_loss[i]),
            "adv_loss_after_attack": float(final_loss[i]),
            "attack_loss_gain": float(gain[i]),
            "attack_loss_gain_relative": float(rel_gain[i]),
            "attack_loss_history": hist[:, i].copy(),
            "attack_loss_gain_history": (hist[:, i] - hist[0, i]).copy(),
        }
        if keep_arrays:
            for key in ("x0", "delta", "x_adv", "model_output", "solver_output", "model_minus_solver"):
                row[key] = heat.as2d(arrays[key][i])
            row["adv_relative_l2_model_vs_solver"] = float(arrays["adv_relative_l2"][i])
            row["adv_rmse_model_vs_solver"] = float(arrays["adv_rmse"][i])
            row["delta_l0_fraction"] = float(arrays["delta_l0_fraction"][i])
            row["delta_l2_rms"] = float(arrays["delta_l2_rms"][i])
        out.append(row)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_ranking_attacks(args: argparse.Namespace, dataset_ids: list[str], dataset_paths: dict[str, Path], device: torch.device, out_dir: Path) -> list[dict[str, Any]]:
    result_path = out_dir / f"all_{args.attack_steps}step_batch_attack_results.csv"
    if result_path.exists() and not args.force:
        with result_path.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        print(f"[reuse] ranking attacks {heat.rel(result_path)} rows={len(rows)}", flush=True)
        return rows

    all_rows: list[dict[str, Any]] = []
    sample_indices = list(range(int(args.samples_per_dataset)))
    for model_spec in heat.MODELS:
        model_start = time.perf_counter()
        print(f"[rank-model] {model_spec.name}", flush=True)
        model = heat.load_model(model_spec, device)
        for dataset_rank, dataset_id in enumerate(dataset_ids, 1):
            for start in range(0, len(sample_indices), int(args.attack_batch_size)):
                batch_indices = sample_indices[start : start + int(args.attack_batch_size)]
                xb, yb = load_dataset_batch(dataset_paths, dataset_id, batch_indices, device)
                t0 = time.perf_counter()
                batch_rows = batch_darcy_attack_trace(
                    model,
                    xb,
                    yb,
                    steps=int(args.attack_steps),
                    epsilon_fraction=float(args.epsilon_fraction),
                    keep_arrays=False,
                )
                elapsed = time.perf_counter() - t0
                for idx, row in zip(batch_indices, batch_rows):
                    all_rows.append(
                        {
                            "model": model_spec.name,
                            "dataset_id": dataset_id,
                            "split": "generalization",
                            "sample_index": int(idx),
                            "attack_steps": int(args.attack_steps),
                            "epsilon_fraction": float(args.epsilon_fraction),
                            "clean_loss_before_attack": row["clean_loss_before_attack"],
                            "adv_loss_after_attack": row["adv_loss_after_attack"],
                            "attack_loss_gain": row["attack_loss_gain"],
                            "attack_loss_gain_relative": row["attack_loss_gain_relative"],
                            "elapsed_seconds": elapsed / max(1, len(batch_indices)),
                        }
                    )
                print(
                    f"[rank-batch] {model_spec.name:12s} dataset={dataset_rank:02d}/{len(dataset_ids):02d} samples={batch_indices[0]:02d}-{batch_indices[-1]:02d} sec={elapsed:.2f}",
                    flush=True,
                )
                del xb, yb, batch_rows
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[rank-model-done] {model_spec.name} sec={time.perf_counter()-model_start:.1f}", flush=True)
    write_csv(result_path, all_rows, ATTACK_RESULT_FIELDS)
    return all_rows


def family_slug(dataset_id: str) -> str:
    for token, label in [
        ("matern_smooth", "maternsmooth"),
        ("matern_fine", "maternfine"),
        ("highpass", "highpass"),
        ("bandpass", "bandpass"),
        ("wave_mix", "wave"),
        ("blocky_tiles", "blocky"),
        ("rectangles", "rectangles"),
        ("cellular_blobs", "cellular"),
    ]:
        if token in dataset_id:
            return label
    return "gen"


def select_ranked_samples(rank_rows: list[dict[str, Any]], dataset_ids: list[str], out_dir: Path, samples_per_dataset: int) -> dict[str, list[heat.SampleSpec]]:
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rank_rows:
        grouped[(row["dataset_id"], int(float(row["sample_index"])))][row["model"]] = row

    selections: dict[str, list[heat.SampleSpec]] = {name: [] for name, _ in RANK_VARIANTS}
    score_rows: list[dict[str, Any]] = []
    for dataset_rank, dataset_id in enumerate(dataset_ids, 1):
        sample_scores = []
        for sample_index in range(int(samples_per_dataset)):
            by_model = grouped.get((dataset_id, sample_index), {})
            if not set(MODEL_ORDER).issubset(by_model):
                continue
            adv_loss = {m: float(by_model[m]["adv_loss_after_attack"]) for m in MODEL_ORDER}
            loss3 = adv_loss["loss3"]
            others = {m: v for m, v in adv_loss.items() if m != "loss3"}
            best_other_model, best_other = min(others.items(), key=lambda kv: kv[1])
            score = best_other - loss3
            sample_scores.append(
                {
                    "dataset_id": dataset_id,
                    "sample_index": sample_index,
                    "adv_loss": adv_loss,
                    "best_other_model": best_other_model,
                    "best_other_adv_loss": best_other,
                    "loss3_margin_vs_best_other": score,
                    "loss3_margin_vs_mean_other": float(np.mean(list(others.values())) - loss3),
                }
            )
        if len(sample_scores) != int(samples_per_dataset):
            raise RuntimeError(f"dataset {dataset_id} has {len(sample_scores)} complete samples, expected {int(samples_per_dataset)}")
        sample_scores.sort(key=lambda r: (r["loss3_margin_vs_best_other"], -r["adv_loss"]["loss3"]), reverse=True)
        pick_indices = {
            "index0": next(i for i, r in enumerate(sample_scores) if int(r["sample_index"]) == 0),
            "loss3_best": 0,
            "loss3_upper_quartile": int(round(0.25 * (len(sample_scores) - 1))),
            "loss3_median": int(round(0.50 * (len(sample_scores) - 1))),
            "loss3_lower_quartile": int(round(0.75 * (len(sample_scores) - 1))),
            "loss3_worst": len(sample_scores) - 1,
        }
        fam = family_slug(dataset_id)
        for variant, pick_pos in pick_indices.items():
            selected = sample_scores[pick_pos]
            rank = pick_pos + 1
            sample_id = f"{variant}_{dataset_rank:02d}_{fam}_idx{selected['sample_index']}"
            selections[variant].append(heat.SampleSpec(sample_id, "generalization", dataset_id, int(selected["sample_index"])))
            score_rows.append(
                {
                    "variant": variant,
                    "sample_id": sample_id,
                    "dataset_rank": dataset_rank,
                    "rank_within_dataset_best_is_1": rank,
                    "dataset_id": dataset_id,
                    "split": "generalization",
                    "sample_index": int(selected["sample_index"]),
                    "loss3_adv_loss": selected["adv_loss"]["loss3"],
                    "baseline_adv_loss": selected["adv_loss"]["baseline"],
                    "loss1_adv_loss": selected["adv_loss"]["loss1"],
                    "loss2_adv_loss": selected["adv_loss"]["loss2"],
                    "physics_loss_adv_loss": selected["adv_loss"]["physics loss"],
                    "best_other_model": selected["best_other_model"],
                    "best_other_adv_loss": selected["best_other_adv_loss"],
                    "loss3_margin_vs_best_other": selected["loss3_margin_vs_best_other"],
                    "loss3_margin_vs_mean_other": selected["loss3_margin_vs_mean_other"],
                }
            )
    write_csv(out_dir / "ranked_selection_scores.csv", score_rows, SELECTION_FIELDS)
    for variant, samples in selections.items():
        manifest = out_dir / f"{variant}_selected_samples.csv"
        with manifest.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["sample_id", "split", "dataset_id", "sample_index"])
            writer.writeheader()
            writer.writerows([asdict(s) for s in samples])
    return selections


def load_mixed_samples(dataset_paths: dict[str, Path], samples: list[heat.SampleSpec], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    for sample in samples:
        xb, yb = heat.load_sample(dataset_paths, sample, device)
        xs.append(xb.cpu())
        ys.append(yb.cpu())
    return torch.cat(xs, dim=0).to(device), torch.cat(ys, dim=0).to(device)


def write_variant_outputs(
    args: argparse.Namespace,
    variant: str,
    samples: list[heat.SampleSpec],
    dataset_paths: dict[str, Path],
    device: torch.device,
    out_root: Path,
    viz_root: Path,
) -> dict[str, Any]:
    tag = f"{args.tag}_{variant}"
    out_dir = out_root / f"darcy_five_model_attack_heatmaps_{tag}"
    viz_dir = viz_root / f"darcy_five_model_attack_heatmaps_{tag}"
    array_dir = out_dir / "arrays"
    out_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)
    array_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "selected_samples.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "split", "dataset_id", "sample_index"])
        writer.writeheader()
        writer.writerows([asdict(s) for s in samples])

    records: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    xb_all, yb_all = load_mixed_samples(dataset_paths, samples, device)
    for model_spec in heat.MODELS:
        print(f"[plot-model] variant={variant} model={model_spec.name}", flush=True)
        model = heat.load_model(model_spec, device)
        model_rows: list[dict[str, Any]] = []
        for start in range(0, len(samples), int(args.plot_batch_size)):
            end = min(len(samples), start + int(args.plot_batch_size))
            t0 = time.perf_counter()
            batch_rows = batch_darcy_attack_trace(
                model,
                xb_all[start:end],
                yb_all[start:end],
                steps=int(args.attack_steps),
                epsilon_fraction=float(args.epsilon_fraction),
                keep_arrays=True,
            )
            elapsed = time.perf_counter() - t0
            for local, row in enumerate(batch_rows, start):
                sample = samples[local]
                npz_path = array_dir / f"{sample.sample_id}_{slug(model_spec.name)}_attack_fields.npz"
                arrays = {
                    "x0": row["x0"],
                    "delta": row["delta"],
                    "x_adv": row["x_adv"],
                    "model_output": row["model_output"],
                    "solver_output": row["solver_output"],
                    "model_minus_solver": row["model_minus_solver"],
                    "attack_loss_history": row["attack_loss_history"].astype(np.float64),
                    "attack_loss_gain_history": row["attack_loss_gain_history"].astype(np.float64),
                }
                np.savez_compressed(npz_path, **arrays)
                rec = {
                    **arrays,
                    "sample_id": sample.sample_id,
                    "split": sample.split,
                    "dataset_id": sample.dataset_id,
                    "sample_index": sample.sample_index,
                    "model": model_spec.name,
                    "attack_loss_gain": row["attack_loss_gain"],
                    "adv_relative_l2_model_vs_solver": row["adv_relative_l2_model_vs_solver"],
                }
                records.append(rec)
                model_rows.append(
                    {
                        "sample_id": sample.sample_id,
                        "split": sample.split,
                        "dataset_id": sample.dataset_id,
                        "sample_index": sample.sample_index,
                        "model": model_spec.name,
                        "model_source": model_spec.source,
                        "checkpoint": heat.rel(model_spec.checkpoint),
                        "attack_steps": int(args.attack_steps),
                        "epsilon_fraction": float(args.epsilon_fraction),
                        "clean_loss_before_attack": row["clean_loss_before_attack"],
                        "adv_loss_after_attack": row["adv_loss_after_attack"],
                        "attack_loss_gain": row["attack_loss_gain"],
                        "attack_loss_gain_relative": row["attack_loss_gain_relative"],
                        "adv_relative_l2_model_vs_solver": row["adv_relative_l2_model_vs_solver"],
                        "adv_rmse_model_vs_solver": row["adv_rmse_model_vs_solver"],
                        "delta_l0_fraction": row["delta_l0_fraction"],
                        "delta_l2_rms": row["delta_l2_rms"],
                        "elapsed_seconds": elapsed / max(1, end - start),
                        "npz_path": heat.rel(npz_path),
                    }
                )
            print(f"[plot-batch] {variant} {model_spec.name:12s} samples={start:02d}-{end-1:02d} sec={elapsed:.2f}", flush=True)
        summary_rows.extend(model_rows)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    ranges = heat.shared_ranges(records)
    (out_dir / "shared_color_ranges.json").write_text(json.dumps(ranges, indent=2, sort_keys=True), encoding="utf-8")
    column_ranges = {
        "initial condition": ranges["coefficient"],
        "delta": ranges["delta"],
        "initial + delta": ranges["coefficient"],
        "model output": ranges["output_solver_shared"],
        "solver output": ranges["output_solver_shared"],
        "model - solver": ranges["model_minus_solver"],
        "note": "These vmin/vmax values are applied to every row/model in the corresponding column. Model output and solver output intentionally share the exact same range.",
    }
    (out_dir / "column_color_ranges_applied.json").write_text(json.dumps(column_ranges, indent=2, sort_keys=True), encoding="utf-8")
    heat.write_csv(out_dir / "summary.csv", summary_rows)
    fig_paths = [heat.plot_sample(sample, records, ranges, viz_dir) for sample in samples]
    report_args = argparse.Namespace(tag=tag, attack_steps=int(args.attack_steps), epsilon_fraction=float(args.epsilon_fraction))
    heat.write_report(out_dir, viz_dir, summary_rows, fig_paths, samples, report_args)
    return {"variant": variant, "out_dir": heat.rel(out_dir), "viz_dir": heat.rel(viz_dir), "figures": [heat.rel(p) for p in fig_paths]}


def run(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    out_root = args.out_root.resolve()
    viz_root = args.viz_root.resolve()
    run_dir = out_root / f"darcy_five_model_batch_ranked_heatmaps_{args.tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    dataset_paths = heat.load_dataset_map(args.generalization_root.resolve())
    dataset_ids = choose_dataset_ids(args.attack20_table.resolve(), int(args.dataset_count))
    (run_dir / "selected_dataset_ids.json").write_text(json.dumps(dataset_ids, indent=2), encoding="utf-8")
    print(f"[datasets] {len(dataset_ids)} selected", flush=True)
    rank_rows = run_ranking_attacks(args, dataset_ids, dataset_paths, device, run_dir)
    selections = select_ranked_samples(rank_rows, dataset_ids, run_dir, int(args.samples_per_dataset))
    outputs = []
    for variant, _desc in RANK_VARIANTS:
        if args.only_variant and variant not in set(args.only_variant.split(",")):
            continue
        outputs.append(write_variant_outputs(args, variant, selections[variant], dataset_paths, device, out_root, viz_root))
    manifest = {
        "tag": args.tag,
        "attack_steps": int(args.attack_steps),
        "epsilon_fraction": float(args.epsilon_fraction),
        "dataset_count": int(args.dataset_count),
        "samples_per_dataset": int(args.samples_per_dataset),
        "attack_batch_size": int(args.attack_batch_size),
        "plot_batch_size": int(args.plot_batch_size),
        "run_dir": heat.rel(run_dir),
        "outputs": outputs,
    }
    (run_dir / "batch_ranked_heatmap_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="20260612_loss3attack100_five_models_ranked10_polished")
    parser.add_argument("--generalization-root", type=Path, default=PROJECT_ROOT / "generalization_datasets_darcy_binary_loss3targeted_20260611")
    parser.add_argument("--attack20-table", type=Path, default=DEFAULT_ATTACK20_TABLE)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--viz-root", type=Path, default=DEFAULT_VIZ_ROOT)
    parser.add_argument("--dataset-count", type=int, default=10)
    parser.add_argument("--samples-per-dataset", type=int, default=50)
    parser.add_argument("--attack-steps", type=int, default=100)
    parser.add_argument("--epsilon-fraction", type=float, default=0.025)
    parser.add_argument("--attack-batch-size", type=int, default=25)
    parser.add_argument("--plot-batch-size", type=int, default=10)
    parser.add_argument("--only-variant", default=None, help="Comma-separated subset of variants to plot after ranking.")
    parser.add_argument("--force", action="store_true", help="Recompute ranking attacks even if all_<attack_steps>step_batch_attack_results.csv exists.")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
