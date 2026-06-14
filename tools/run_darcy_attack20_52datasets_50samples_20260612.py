#!/usr/bin/env python3
"""Attack the four first-stage Darcy adversarial models on 52 binary datasets.

For every Darcy dataset (train/test plus 50 generated generalization datasets),
select the same fixed set of samples, run a shared 20-step loss3 adversarial
attack against each trained model, and summarize average loss growth.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.40")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.evaluate_generalization_models import DatasetSpec, build_specs, tensor_xy, torch_load
import tools.adversarial_training as adv


DEFAULT_TAG = "20260612_attack20_1000c_52datasets_50samples"
RUN_TAG = "20260612_full50_timematched_1000c"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    checkpoint: Path
    trained_epochs: int
    training_objective: str


@dataclass
class DatasetBundle:
    spec: DatasetSpec
    indices: list[int]
    x: torch.Tensor
    y: torch.Tensor
    metadata: dict[str, Any]


MODEL_SPECS = [
    ModelSpec(
        "loss1",
        PROJECT_ROOT
        / "adversarial_training_runs"
        / f"darcy_binary_loss3targeted_loss1_1000ep_full50_timematched_{RUN_TAG}"
        / "darcy"
        / "checkpoints"
        / "darcy_epoch1000_step001000.pt",
        1000,
        "loss1",
    ),
    ModelSpec(
        "loss2",
        PROJECT_ROOT
        / "adversarial_training_runs"
        / f"darcy_binary_loss3targeted_loss2_1026ep_full50_timematched_{RUN_TAG}"
        / "darcy"
        / "checkpoints"
        / "darcy_epoch1026_step001026.pt",
        1026,
        "loss2",
    ),
    ModelSpec(
        "loss3",
        PROJECT_ROOT
        / "adversarial_training_runs"
        / f"darcy_binary_loss3targeted_loss3_1011ep_full50_timematched_{RUN_TAG}"
        / "darcy"
        / "checkpoints"
        / "darcy_epoch1011_step001011.pt",
        1011,
        "loss3",
    ),
    ModelSpec(
        "physics",
        PROJECT_ROOT
        / "adversarial_training_runs"
        / f"darcy_binary_loss3targeted_physics_1040ep_full50_timematched_{RUN_TAG}"
        / "darcy"
        / "checkpoints"
        / "darcy_epoch1040_step001040.pt",
        1040,
        "physics",
    ),
]


SAMPLE_FIELDS = [
    "model",
    "training_objective",
    "trained_epochs",
    "checkpoint",
    "dataset_id",
    "split",
    "source",
    "manual_tier",
    "manual_rank",
    "sample_ordinal",
    "source_sample_index",
    "attack_steps",
    "attack_batch_size",
    "epsilon_fraction",
    "clean_loss_before_attack",
    "adv_loss_after_attack",
    "attack_loss_gain",
    "attack_loss_gain_relative",
    "adv_to_clean_ratio",
    "solver_mse_attack_gain_batch",
    "delta_l2_rms",
    "delta_linf",
    "delta_flip_fraction",
    "darcy_budget_pixels_batch",
    "boundary_ratio_batch",
]

DATASET_FIELDS = [
    "model",
    "training_objective",
    "trained_epochs",
    "checkpoint",
    "dataset_id",
    "split",
    "source",
    "manual_tier",
    "manual_rank",
    "sample_count",
    "attack_steps",
    "attack_batch_size",
    "epsilon_fraction",
    "mean_clean_loss",
    "mean_adv_loss",
    "mean_attack_loss_gain",
    "mean_attack_loss_gain_relative",
    "relative_gain_from_means",
    "median_attack_loss_gain",
    "median_attack_loss_gain_relative",
    "mean_solver_mse_attack_gain_by_batch",
    "mean_delta_l2_rms",
    "mean_delta_linf",
    "mean_delta_flip_fraction",
    "mean_darcy_budget_pixels_by_batch",
    "mean_boundary_ratio_by_batch",
    "elapsed_seconds",
]

MODEL_FIELDS = [
    "model",
    "training_objective",
    "trained_epochs",
    "checkpoint",
    "split",
    "dataset_count",
    "sample_count",
    "attack_steps",
    "epsilon_fraction",
    "mean_clean_loss",
    "mean_adv_loss",
    "mean_attack_loss_gain",
    "mean_attack_loss_gain_relative",
    "relative_gain_from_means",
    "median_attack_loss_gain",
    "median_attack_loss_gain_relative",
    "mean_delta_l2_rms",
    "mean_delta_linf",
    "mean_delta_flip_fraction",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def rel_project(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(data), indent=2, sort_keys=True), encoding="utf-8")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return rel_project(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def reset_csv(path: Path, fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()


def append_csv_rows(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_models(value: str | None) -> list[ModelSpec]:
    if not value:
        return list(MODEL_SPECS)
    names = {p.strip() for p in value.split(",") if p.strip()}
    by_name = {m.name: m for m in MODEL_SPECS}
    missing = sorted(names - set(by_name))
    if missing:
        raise ValueError(f"unknown models {missing}; valid={sorted(by_name)}")
    return [m for m in MODEL_SPECS if m.name in names]


def dataset_sort_key(spec: DatasetSpec) -> tuple[int, float, str]:
    split_order = {"train": 0, "test": 1, "generalization": 2}.get(spec.split, 9)
    return (split_order, float(spec.manual_rank), spec.dataset_id)


def select_indices(n: int, count: int, seed: int, dataset_id: str, policy: str) -> list[int]:
    if n < count:
        raise ValueError(f"dataset {dataset_id} has only {n} samples; requested {count}")
    if policy == "first":
        return list(range(count))
    if policy != "random":
        raise ValueError(f"unknown sample policy: {policy}")
    digest = hashlib.sha256(dataset_id.encode("utf-8")).digest()
    offset = int.from_bytes(digest[:8], "little") % (2**32 - 1)
    rng = np.random.default_rng((int(seed) + offset) % (2**32 - 1))
    return sorted(int(i) for i in rng.choice(n, size=count, replace=False).tolist())


def load_dataset_bundles(
    specs: list[DatasetSpec],
    *,
    samples_per_dataset: int,
    seed: int,
    sample_policy: str,
    max_datasets: int | None,
    out_dir: Path,
) -> list[DatasetBundle]:
    selected = specs[: max_datasets or len(specs)]
    bundles: list[DatasetBundle] = []
    manifest_rows: list[dict[str, Any]] = []
    for spec in selected:
        payload = torch_load(spec.path)
        x, y = tensor_xy(payload, "darcy")
        indices = select_indices(int(x.shape[0]), samples_per_dataset, seed, spec.dataset_id, sample_policy)
        idx_tensor = torch.tensor(indices, dtype=torch.long)
        x_sel = x.index_select(0, idx_tensor).contiguous()
        y_sel = y.index_select(0, idx_tensor).contiguous()
        metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
        bundles.append(DatasetBundle(spec=spec, indices=indices, x=x_sel, y=y_sel, metadata=metadata))
        for ordinal, source_index in enumerate(indices):
            manifest_rows.append(
                {
                    "dataset_id": spec.dataset_id,
                    "split": spec.split,
                    "source": spec.source,
                    "manual_tier": spec.manual_tier,
                    "manual_rank": spec.manual_rank,
                    "path": rel_project(spec.path),
                    "sample_ordinal": ordinal,
                    "source_sample_index": source_index,
                    "sample_policy": sample_policy,
                    "seed": seed,
                    "total_samples_available": int(x.shape[0]),
                }
            )
        print(
            f"[select] {len(bundles):03d}/{len(selected):03d} {spec.split:14s} {spec.dataset_id} "
            f"samples={len(indices)} available={int(x.shape[0])}",
            flush=True,
        )
    fields = [
        "dataset_id",
        "split",
        "source",
        "manual_tier",
        "manual_rank",
        "path",
        "sample_ordinal",
        "source_sample_index",
        "sample_policy",
        "seed",
        "total_samples_available",
    ]
    reset_csv(out_dir / "selected_samples_manifest.csv", fields)
    append_csv_rows(out_dir / "selected_samples_manifest.csv", fields, manifest_rows)
    return bundles


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


def run_attack_batch(
    model,
    xb: torch.Tensor,
    yb: torch.Tensor,
    *,
    steps: int,
    epsilon_fraction: float,
    cfg: dict[str, Any],
):
    return adv.binary_darcy_replace_attack(
        model,
        xb,
        yb,
        steps=steps,
        epsilon_fraction=epsilon_fraction,
        jitter_low=1.0,
        jitter_high=1.0,
        random_pool_multiplier=1.0,
        random_score_noise=0.0,
        cfg=cfg,
    )


def try_attack_batch(
    model,
    xb: torch.Tensor,
    yb: torch.Tensor,
    *,
    steps: int,
    epsilon_fraction: float,
    cfg: dict[str, Any],
    requested_batch_size: int,
):
    current = int(xb.shape[0])
    while current > 0:
        try:
            return run_attack_batch(
                model,
                xb[:current],
                yb[:current],
                steps=steps,
                epsilon_fraction=epsilon_fraction,
                cfg=cfg,
            ), current
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or current == 1:
                raise
            torch.cuda.empty_cache()
            gc.collect()
            current = max(1, current // 2)
            print(
                f"[oom-fallback] requested_batch={requested_batch_size} retry_batch={current}",
                flush=True,
            )
    raise RuntimeError("unreachable batch fallback state")


def finite_array(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    vals = []
    for row in rows:
        try:
            val = float(row[key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(val):
            vals.append(val)
    return np.array(vals, dtype=np.float64)


def mean_or_nan(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def median_or_nan(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.median(arr))


def summarize_rows(rows: list[dict[str, Any]], *, model: ModelSpec, split: str, attack_steps: int, epsilon_fraction: float) -> dict[str, Any]:
    clean = finite_array(rows, "clean_loss_before_attack")
    adv_after = finite_array(rows, "adv_loss_after_attack")
    gain = finite_array(rows, "attack_loss_gain")
    rel_gain = finite_array(rows, "attack_loss_gain_relative")
    delta_l2 = finite_array(rows, "delta_l2_rms")
    delta_linf = finite_array(rows, "delta_linf")
    delta_flip = finite_array(rows, "delta_flip_fraction")
    clean_mean = mean_or_nan(clean)
    adv_mean = mean_or_nan(adv_after)
    rel_from_means = (adv_mean / clean_mean - 1.0) if math.isfinite(clean_mean) and abs(clean_mean) > 1e-20 else float("nan")
    datasets = {str(r.get("dataset_id", "")) for r in rows}
    return {
        "model": model.name,
        "training_objective": model.training_objective,
        "trained_epochs": model.trained_epochs,
        "checkpoint": rel_project(model.checkpoint),
        "split": split,
        "dataset_count": len(datasets),
        "sample_count": len(rows),
        "attack_steps": attack_steps,
        "epsilon_fraction": epsilon_fraction,
        "mean_clean_loss": clean_mean,
        "mean_adv_loss": adv_mean,
        "mean_attack_loss_gain": mean_or_nan(gain),
        "mean_attack_loss_gain_relative": mean_or_nan(rel_gain),
        "relative_gain_from_means": rel_from_means,
        "median_attack_loss_gain": median_or_nan(gain),
        "median_attack_loss_gain_relative": median_or_nan(rel_gain),
        "mean_delta_l2_rms": mean_or_nan(delta_l2),
        "mean_delta_linf": mean_or_nan(delta_linf),
        "mean_delta_flip_fraction": mean_or_nan(delta_flip),
    }


def load_model(spec: ModelSpec, device: torch.device):
    if not spec.checkpoint.exists():
        raise FileNotFoundError(f"missing checkpoint for {spec.name}: {spec.checkpoint}")
    model = adv.load_model("darcy", device, model_checkpoint_override=spec.checkpoint)
    model.eval()
    return model


def evaluate_model(
    spec: ModelSpec,
    bundles: list[DatasetBundle],
    *,
    out_dir: Path,
    batch_size: int,
    attack_steps: int,
    epsilon_fraction: float,
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    device = torch.device("cuda")
    print(f"[model] loading {spec.name} checkpoint={rel_project(spec.checkpoint)}", flush=True)
    model = load_model(spec, device)
    model_rows: list[dict[str, Any]] = []
    dataset_csv = out_dir / "summary_by_dataset_model.csv"
    sample_csv = out_dir / "all_samples.csv"

    for ds_i, bundle in enumerate(bundles, start=1):
        t0 = time.perf_counter()
        sample_rows: list[dict[str, Any]] = []
        dataset_batch_solver_gains: list[float] = []
        effective_batch_sizes: list[int] = []
        n = int(bundle.x.shape[0])
        start = 0
        while start < n:
            requested_end = min(start + batch_size, n)
            xb_cpu = bundle.x[start:requested_end]
            yb_cpu = bundle.y[start:requested_end]
            xb = xb_cpu.to(device, non_blocking=True)
            yb = yb_cpu.to(device, non_blocking=True)
            result, used_n = try_attack_batch(
                model,
                xb,
                yb,
                steps=attack_steps,
                epsilon_fraction=epsilon_fraction,
                cfg=cfg,
                requested_batch_size=batch_size,
            )
            info = result.info
            si = result.sample_info
            clean = si["clean_loss_before_attack"].detach().cpu().numpy().astype(float)
            adv_after = si["adv_loss_after_attack"].detach().cpu().numpy().astype(float)
            gain = si["attack_loss_gain"].detach().cpu().numpy().astype(float)
            rel_gain = si["attack_loss_gain_relative"].detach().cpu().numpy().astype(float)
            solver_gain = float(info.get("solver_mse_attack_gain", float("nan")))
            delta = (result.x_train.detach() - xb[:used_n].detach()).reshape(used_n, -1)
            delta_l2 = torch.sqrt(delta.pow(2).mean(dim=1)).detach().cpu().numpy().astype(float)
            delta_linf = delta.abs().max(dim=1).values.detach().cpu().numpy().astype(float)
            delta_flip = (delta.abs() > 1e-12).float().mean(dim=1).detach().cpu().numpy().astype(float)
            budget_pixels = float(info.get("darcy_budget_pixels_mean", float("nan")))
            boundary_ratio = float(info.get("boundary_ratio_mean", float("nan")))
            dataset_batch_solver_gains.append(solver_gain)
            effective_batch_sizes.append(int(used_n))
            for j in range(used_n):
                ordinal = start + j
                c = float(clean[j])
                a = float(adv_after[j])
                row = {
                    "model": spec.name,
                    "training_objective": spec.training_objective,
                    "trained_epochs": spec.trained_epochs,
                    "checkpoint": rel_project(spec.checkpoint),
                    "dataset_id": bundle.spec.dataset_id,
                    "split": bundle.spec.split,
                    "source": bundle.spec.source,
                    "manual_tier": bundle.spec.manual_tier,
                    "manual_rank": bundle.spec.manual_rank,
                    "sample_ordinal": ordinal,
                    "source_sample_index": bundle.indices[ordinal],
                    "attack_steps": attack_steps,
                    "attack_batch_size": used_n,
                    "epsilon_fraction": epsilon_fraction,
                    "clean_loss_before_attack": c,
                    "adv_loss_after_attack": a,
                    "attack_loss_gain": float(gain[j]),
                    "attack_loss_gain_relative": float(rel_gain[j]),
                    "adv_to_clean_ratio": (a / c) if math.isfinite(c) and abs(c) > 1e-20 else float("nan"),
                    "solver_mse_attack_gain_batch": solver_gain,
                    "delta_l2_rms": float(delta_l2[j]),
                    "delta_linf": float(delta_linf[j]),
                    "delta_flip_fraction": float(delta_flip[j]),
                    "darcy_budget_pixels_batch": budget_pixels,
                    "boundary_ratio_batch": boundary_ratio,
                }
                sample_rows.append(row)
            del xb, yb, result
            torch.cuda.empty_cache()
            start += used_n

        append_csv_rows(sample_csv, SAMPLE_FIELDS, sample_rows)
        model_rows.extend(sample_rows)
        summary = summarize_rows(
            sample_rows,
            model=spec,
            split=bundle.spec.split,
            attack_steps=attack_steps,
            epsilon_fraction=epsilon_fraction,
        )
        summary.update(
            {
                "dataset_id": bundle.spec.dataset_id,
                "source": bundle.spec.source,
                "manual_tier": bundle.spec.manual_tier,
                "manual_rank": bundle.spec.manual_rank,
                "sample_count": len(sample_rows),
                "attack_batch_size": int(max(effective_batch_sizes)) if effective_batch_sizes else 0,
                "mean_solver_mse_attack_gain_by_batch": mean_or_nan(dataset_batch_solver_gains),
                "mean_darcy_budget_pixels_by_batch": mean_or_nan([float(r["darcy_budget_pixels_batch"]) for r in sample_rows]),
                "mean_boundary_ratio_by_batch": mean_or_nan([float(r["boundary_ratio_batch"]) for r in sample_rows]),
                "elapsed_seconds": time.perf_counter() - t0,
            }
        )
        append_csv_rows(dataset_csv, DATASET_FIELDS, [summary])
        print(
            f"[attack] {spec.name:7s} {ds_i:03d}/{len(bundles):03d} {bundle.spec.split:14s} "
            f"{bundle.spec.dataset_id} clean={summary['mean_clean_loss']:.6g} "
            f"adv={summary['mean_adv_loss']:.6g} gain={summary['mean_attack_loss_gain']:.6g} "
            f"rel_from_means={summary['relative_gain_from_means']:.3f} elapsed={summary['elapsed_seconds']:.1f}s",
            flush=True,
        )

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return model_rows


def aggregate_model_summaries(all_rows: list[dict[str, Any]], models: list[ModelSpec], *, attack_steps: int, epsilon_fraction: float) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for model in models:
        rows_model = [r for r in all_rows if r["model"] == model.name]
        summaries.append(summarize_rows(rows_model, model=model, split="all", attack_steps=attack_steps, epsilon_fraction=epsilon_fraction))
        for split in ("train", "test", "generalization"):
            rows_split = [r for r in rows_model if r["split"] == split]
            if rows_split:
                summaries.append(summarize_rows(rows_split, model=model, split=split, attack_steps=attack_steps, epsilon_fraction=epsilon_fraction))
    return summaries


def rows_from_csv_numeric(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    numeric_keys = {
        "trained_epochs",
        "manual_rank",
        "sample_ordinal",
        "source_sample_index",
        "attack_steps",
        "attack_batch_size",
        "epsilon_fraction",
        "clean_loss_before_attack",
        "adv_loss_after_attack",
        "attack_loss_gain",
        "attack_loss_gain_relative",
        "adv_to_clean_ratio",
        "solver_mse_attack_gain_batch",
        "delta_l2_rms",
        "delta_linf",
        "delta_flip_fraction",
        "darcy_budget_pixels_batch",
        "boundary_ratio_batch",
        "dataset_count",
        "sample_count",
        "mean_clean_loss",
        "mean_adv_loss",
        "mean_attack_loss_gain",
        "mean_attack_loss_gain_relative",
        "relative_gain_from_means",
        "median_attack_loss_gain",
        "median_attack_loss_gain_relative",
        "mean_solver_mse_attack_gain_by_batch",
        "mean_delta_l2_rms",
        "mean_delta_linf",
        "mean_delta_flip_fraction",
        "mean_darcy_budget_pixels_by_batch",
        "mean_boundary_ratio_by_batch",
        "elapsed_seconds",
    }
    for row in read_csv_rows(path):
        out: dict[str, Any] = dict(row)
        for key in numeric_keys:
            if key in out and out[key] != "":
                try:
                    if key in {"trained_epochs", "sample_ordinal", "source_sample_index", "attack_steps", "attack_batch_size", "dataset_count", "sample_count"}:
                        out[key] = int(float(out[key]))
                    else:
                        out[key] = float(out[key])
                except ValueError:
                    pass
        rows.append(out)
    return rows


def plot_bar(summary_rows: list[dict[str, Any]], out_dir: Path) -> None:
    all_rows = [r for r in summary_rows if r["split"] == "all"]
    models = [r["model"] for r in all_rows]
    clean = [float(r["mean_clean_loss"]) for r in all_rows]
    adv_after = [float(r["mean_adv_loss"]) for r in all_rows]
    gain = [float(r["mean_attack_loss_gain"]) for r in all_rows]
    rel = [float(r["relative_gain_from_means"]) for r in all_rows]
    x = np.arange(len(models))

    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.36
    ax.bar(x - width / 2, clean, width, label="clean")
    ax.bar(x + width / 2, adv_after, width, label="after 20-step attack")
    ax.set_xticks(x, models)
    ax.set_ylabel("loss3 solver-consistent MSE")
    ax.set_title("Darcy attack20 mean loss before/after")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "attack20_mean_clean_vs_adv_by_model.png", dpi=220)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.bar(x, gain, color="#3b82f6", label="absolute gain")
    ax1.set_xticks(x, models)
    ax1.set_ylabel("mean absolute loss gain")
    ax1.grid(axis="y", alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(x, rel, color="#dc2626", marker="o", linewidth=2.0, label="relative gain from means")
    ax2.set_ylabel("adv / clean - 1")
    ax1.set_title("Darcy attack20 average loss growth by model")
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "attack20_mean_loss_growth_by_model.png", dpi=220)
    plt.close(fig)


def plot_split_grouped(summary_rows: list[dict[str, Any]], out_dir: Path, key: str, ylabel: str, filename: str) -> None:
    splits = ["train", "test", "generalization"]
    models = [r["model"] for r in summary_rows if r["split"] == "all"]
    lookup = {(r["model"], r["split"]): float(r[key]) for r in summary_rows if r["split"] in splits}
    x = np.arange(len(models))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, split in enumerate(splits):
        vals = [lookup.get((m, split), np.nan) for m in models]
        ax.bar(x + (i - 1) * width, vals, width, label=split)
    ax.set_xticks(x, models)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Darcy attack20 {ylabel} by split")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=220)
    plt.close(fig)


def plot_dataset_heatmap(dataset_rows: list[dict[str, Any]], out_dir: Path, key: str, filename: str) -> None:
    model_order = [m.name for m in MODEL_SPECS]
    models = [m for m in model_order if any(row["model"] == m for row in dataset_rows)]
    datasets = []
    seen = set()
    for row in dataset_rows:
        ds = row["dataset_id"]
        if ds not in seen:
            seen.add(ds)
            datasets.append(ds)
    matrix = np.full((len(datasets), len(models)), np.nan, dtype=np.float64)
    ds_index = {d: i for i, d in enumerate(datasets)}
    model_index = {m: i for i, m in enumerate(models)}
    for row in dataset_rows:
        m = row["model"]
        if m not in model_index:
            continue
        try:
            matrix[ds_index[row["dataset_id"]], model_index[m]] = float(row[key])
        except (ValueError, KeyError):
            pass
    labels = []
    for d in datasets:
        if d.startswith("darcy_binary_loss3targeted_20260611_"):
            labels.append(d.replace("darcy_binary_loss3targeted_20260611_", ""))
        else:
            labels.append(d.replace("_original_binary_grf_alpha2_tau3", ""))
    fig_h = max(10, len(datasets) * 0.26)
    fig, ax = plt.subplots(figsize=(8, fig_h))
    im = ax.imshow(matrix, aspect="auto", cmap="magma")
    ax.set_xticks(np.arange(len(models)), models)
    ax.set_yticks(np.arange(len(datasets)), labels, fontsize=7)
    ax.set_title(f"Darcy attack20 {key} per dataset/model")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(key)
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=220)
    plt.close(fig)


def write_report(summary_rows: list[dict[str, Any]], *, out_dir: Path, viz_dir: Path, args: argparse.Namespace) -> None:
    all_rows = [r for r in summary_rows if r["split"] == "all"]
    split_rows = [r for r in summary_rows if r["split"] != "all"]
    best_abs = min(all_rows, key=lambda r: float(r["mean_attack_loss_gain"])) if all_rows else None
    best_rel = min(all_rows, key=lambda r: float(r["relative_gain_from_means"])) if all_rows else None
    lines = [
        f"# Darcy 52-dataset 20-step attack benchmark ({args.tag})",
        "",
        f"- Created: {now_iso()}",
        f"- Attack objective: shared `loss3` solver-consistent MSE for all four models.",
        f"- Datasets: train + test + generated generalization, max datasets={args.max_datasets or 'all'}.",
        f"- Samples per dataset: {args.samples_per_dataset}, sample policy `{args.sample_policy}`, seed `{args.seed}`.",
        f"- Attack steps: {args.attack_steps}, epsilon fraction `{args.epsilon_fraction}`.",
        "",
    ]
    if best_abs:
        lines.append(
            f"Lowest mean absolute loss growth overall: `{best_abs['model']}` "
            f"({float(best_abs['mean_attack_loss_gain']):.6g})."
        )
    if best_rel:
        lines.append(
            f"Lowest relative growth from means overall: `{best_rel['model']}` "
            f"({float(best_rel['relative_gain_from_means']):.6g})."
        )
    lines += ["", "## Overall", "", "| model | clean | attacked | abs gain | rel gain from means | mean sample rel gain | delta RMS | flip frac | samples |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in all_rows:
        lines.append(
            f"| {r['model']} | {float(r['mean_clean_loss']):.6g} | {float(r['mean_adv_loss']):.6g} | "
            f"{float(r['mean_attack_loss_gain']):.6g} | {float(r['relative_gain_from_means']):.6g} | "
            f"{float(r['mean_attack_loss_gain_relative']):.6g} | {float(r.get('mean_delta_l2_rms', float('nan'))):.6g} | "
            f"{float(r.get('mean_delta_flip_fraction', float('nan'))):.6g} | {int(r['sample_count'])} |"
        )
    lines += ["", "## Split Summary", "", "| model | split | clean | attacked | abs gain | rel gain from means | samples |", "|---|---|---:|---:|---:|---:|---:|"]
    for r in split_rows:
        lines.append(
            f"| {r['model']} | {r['split']} | {float(r['mean_clean_loss']):.6g} | {float(r['mean_adv_loss']):.6g} | "
            f"{float(r['mean_attack_loss_gain']):.6g} | {float(r['relative_gain_from_means']):.6g} | {int(r['sample_count'])} |"
        )
    lines += [
        "",
        "## Outputs",
        "",
        f"- Samples CSV: `{rel_project(out_dir / 'all_samples.csv')}`",
        f"- Dataset summary CSV: `{rel_project(out_dir / 'summary_by_dataset_model.csv')}`",
        f"- Model summary CSV: `{rel_project(out_dir / 'summary_by_model_split.csv')}`",
        f"- Selected sample manifest: `{rel_project(out_dir / 'selected_samples_manifest.csv')}`",
        f"- Figures: `{rel_project(viz_dir)}`",
    ]
    report_path = out_dir / "README.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_all(out_dir: Path, viz_dir: Path) -> None:
    viz_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = rows_from_csv_numeric(out_dir / "summary_by_model_split.csv")
    dataset_rows = rows_from_csv_numeric(out_dir / "summary_by_dataset_model.csv")
    plot_bar(summary_rows, viz_dir)
    plot_split_grouped(
        summary_rows,
        viz_dir,
        "mean_attack_loss_gain",
        "mean absolute loss gain",
        "attack20_mean_loss_gain_by_split.png",
    )
    plot_split_grouped(
        summary_rows,
        viz_dir,
        "relative_gain_from_means",
        "relative gain from means",
        "attack20_relative_loss_growth_by_split.png",
    )
    plot_dataset_heatmap(
        dataset_rows,
        viz_dir,
        "mean_attack_loss_gain",
        "attack20_dataset_model_absolute_gain_heatmap.png",
    )
    plot_dataset_heatmap(
        dataset_rows,
        viz_dir,
        "relative_gain_from_means",
        "attack20_dataset_model_relative_gain_heatmap.png",
    )


def gpu_preflight(out_dir: Path) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    info = {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_name": torch.cuda.get_device_name(0),
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "created_at": now_iso(),
    }
    try:
        import jax
        import jax.numpy as jnp

        j = jnp.ones((32, 32), dtype=jnp.float32)
        info.update(
            {
                "jax_version": jax.__version__,
                "jax_backend": jax.default_backend(),
                "jax_devices": [str(d) for d in jax.devices()],
                "jax_probe_sum": float(jnp.sum(j @ j.T).block_until_ready()),
            }
        )
    except Exception as exc:
        info["jax_preflight_error"] = str(exc)
        raise
    write_json(out_dir / "gpu_preflight.json", info)
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--generalization-root", type=Path, default=PROJECT_ROOT / "generalization_datasets_darcy_binary_loss3targeted_20260611")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--viz-dir", type=Path, default=None)
    parser.add_argument("--samples-per-dataset", type=int, default=50)
    parser.add_argument("--sample-policy", choices=["random", "first"], default="random")
    parser.add_argument("--seed", type=int, default=20260612)
    parser.add_argument("--attack-steps", type=int, default=20)
    parser.add_argument("--epsilon-fraction", type=float, default=0.025)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--max-datasets", type=int, default=None)
    parser.add_argument("--models", default=None, help="comma-separated subset: loss1,loss2,loss3,physics")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir or (PROJECT_ROOT / "analysis_outputs" / f"darcy_attack20_52datasets_50samples_{args.tag}")
    viz_dir = args.viz_dir or (PROJECT_ROOT / "visualizations" / f"darcy_attack20_52datasets_50samples_{args.tag}")
    out_dir = out_dir.resolve()
    viz_dir = viz_dir.resolve()
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output dir already exists; pass --overwrite: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    models = parse_models(args.models)
    for spec in models:
        if not spec.checkpoint.exists():
            raise FileNotFoundError(f"checkpoint missing for {spec.name}: {spec.checkpoint}")

    specs = [s for s in build_specs(args.generalization_root.resolve()) if s.task == "darcy"]
    specs.sort(key=dataset_sort_key)
    if args.max_datasets is None and len(specs) != 52:
        raise RuntimeError(f"expected 52 Darcy datasets (train/test/50 gen), found {len(specs)}")

    reset_csv(out_dir / "all_samples.csv", SAMPLE_FIELDS)
    reset_csv(out_dir / "summary_by_dataset_model.csv", DATASET_FIELDS)
    reset_csv(out_dir / "summary_by_model_split.csv", MODEL_FIELDS)

    manifest = {
        "tag": args.tag,
        "created_at": now_iso(),
        "generalization_root": rel_project(args.generalization_root.resolve()),
        "dataset_count": len(specs[: args.max_datasets or len(specs)]),
        "samples_per_dataset": args.samples_per_dataset,
        "sample_policy": args.sample_policy,
        "seed": args.seed,
        "attack_steps": args.attack_steps,
        "epsilon_fraction": args.epsilon_fraction,
        "batch_size": args.batch_size,
        "models": [to_jsonable(m.__dict__) for m in models],
        "shared_attack_objective": "loss3",
        "notes": "All models are attacked with the same solver-consistent loss3 objective so loss growth is directly comparable.",
    }
    write_json(out_dir / "manifest.json", manifest)
    gpu_preflight(out_dir)

    bundles = load_dataset_bundles(
        specs,
        samples_per_dataset=args.samples_per_dataset,
        seed=args.seed,
        sample_policy=args.sample_policy,
        max_datasets=args.max_datasets,
        out_dir=out_dir,
    )

    cfg = attack_cfg()
    all_rows: list[dict[str, Any]] = []
    t_start = time.perf_counter()
    for model_spec in models:
        rows = evaluate_model(
            model_spec,
            bundles,
            out_dir=out_dir,
            batch_size=args.batch_size,
            attack_steps=args.attack_steps,
            epsilon_fraction=args.epsilon_fraction,
            cfg=cfg,
        )
        all_rows.extend(rows)
        summaries = aggregate_model_summaries(all_rows, models[: models.index(model_spec) + 1], attack_steps=args.attack_steps, epsilon_fraction=args.epsilon_fraction)
        reset_csv(out_dir / "summary_by_model_split.csv", MODEL_FIELDS)
        append_csv_rows(out_dir / "summary_by_model_split.csv", MODEL_FIELDS, summaries)
        plot_all(out_dir, viz_dir)
        write_report(
            rows_from_csv_numeric(out_dir / "summary_by_model_split.csv"),
            out_dir=out_dir,
            viz_dir=viz_dir,
            args=args,
        )

    final_rows = rows_from_csv_numeric(out_dir / "all_samples.csv")
    final_summaries = aggregate_model_summaries(final_rows, models, attack_steps=args.attack_steps, epsilon_fraction=args.epsilon_fraction)
    reset_csv(out_dir / "summary_by_model_split.csv", MODEL_FIELDS)
    append_csv_rows(out_dir / "summary_by_model_split.csv", MODEL_FIELDS, final_summaries)
    plot_all(out_dir, viz_dir)
    write_report(
        rows_from_csv_numeric(out_dir / "summary_by_model_split.csv"),
        out_dir=out_dir,
        viz_dir=viz_dir,
        args=args,
    )
    result = {
        "completed_at": now_iso(),
        "elapsed_seconds": time.perf_counter() - t_start,
        "summary_by_model_split": final_summaries,
        "out_dir": out_dir,
        "viz_dir": viz_dir,
    }
    write_json(out_dir / "result.json", result)
    print(f"[done] elapsed_seconds={result['elapsed_seconds']:.1f}", flush=True)
    print(f"[done] out_dir={out_dir}", flush=True)
    print(f"[done] viz_dir={viz_dir}", flush=True)


if __name__ == "__main__":
    main()
