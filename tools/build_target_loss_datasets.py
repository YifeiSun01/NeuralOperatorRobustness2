#!/usr/bin/env python3
"""Build a copied generalization root whose selected tasks land in a target RMSE band."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.evaluate_generalization_models import (  # noqa: E402
    DatasetSpec as EvalDatasetSpec,
    evaluate_dataset,
    load_burgers_model,
    load_darcy_model,
)
from tools.generate_generalization_datasets import (  # noqa: E402
    DatasetSpec as GenDatasetSpec,
    generate_burgers,
    generate_darcy,
    slug_float,
    write_manifest,
)


def torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=False)


def load_metrics(path: Path, task: str) -> tuple[float, dict[str, dict[str, Any]]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    test_rows = [r for r in rows if r["task"] == task and r["split"] == "test"]
    if len(test_rows) != 1:
        raise ValueError(f"expected one {task} test row in {path}, got {len(test_rows)}")
    by_id = {r["dataset_id"]: r for r in rows if r["task"] == task and r["split"] == "generalization"}
    return float(test_rows[0]["rmse"]), by_id


def params_key(params: dict[str, Any]) -> str:
    return json.dumps(params, sort_keys=True, separators=(",", ":"))


def existing_param_keys(task_dir: Path) -> set[str]:
    keys: set[str] = set()
    for path in sorted(task_dir.glob("*.pt")):
        meta = torch_load(path).get("metadata", {})
        params = meta.get("params")
        if isinstance(params, dict):
            keys.add(params_key(params))
    return keys


def copy_source_root(source_root: Path, target_root: Path, overwrite: bool) -> None:
    if target_root.exists():
        if not overwrite:
            raise FileExistsError(f"{target_root} already exists; pass --overwrite-target to replace it")
        shutil.rmtree(target_root)
    shutil.copytree(source_root, target_root)


def spec_from_existing(task: str, path: Path) -> GenDatasetSpec:
    data = torch_load(path)
    meta = data.get("metadata", {})
    params = meta.get("params") or {}
    n = int(meta.get("nsamples", data["y"].shape[0]))
    return GenDatasetSpec(
        task=task,
        dataset_id=str(meta.get("dataset_id", path.stem)),
        tier=str(meta.get("similarity_tier", "target_loss_kept")),
        family=str(meta.get("family", "kept_existing")),
        n=n,
        seed=int(meta.get("seed", 0)),
        params=dict(params),
        description=str(meta.get("description", "Kept from source dataset because its RMSE ratio is inside the target band.")),
    )


def evaluate_task_file(task: str, model, path: Path, batch_size: int, device: torch.device) -> dict[str, Any]:
    spec = EvalDatasetSpec(
        task=task,
        dataset_id=path.stem,
        split="generalization",
        path=path,
        source="generated",
        manual_tier="target_loss_candidate",
        manual_rank=3.0,
    )
    return evaluate_dataset(model, spec, device, batch_size=batch_size, max_samples=None)


def add_candidate(candidates: list[GenDatasetSpec], seen: set[str], spec: GenDatasetSpec) -> None:
    key = params_key(spec.params)
    if key in seen:
        return
    seen.add(key)
    candidates.append(spec)


def build_burgers_candidates(existing_keys: set[str], max_candidates: int) -> list[GenDatasetSpec]:
    candidates: list[GenDatasetSpec] = []
    seen = set(existing_keys)
    seed_base = 20280600
    idx = 1

    def add(dataset_id: str, tier: str, family: str, params: dict[str, Any], desc: str) -> None:
        nonlocal idx
        add_candidate(candidates, seen, GenDatasetSpec("burgers", dataset_id, tier, family, 200, seed_base + idx, params, desc))
        idx += 1

    # These correlation lengths bracket the empirical in-band region without
    # repeating the original train/test or old generated parameters.
    for corr in [
        0.065, 0.07, 0.075, 0.085, 0.09, 0.095, 0.105, 0.11, 0.115, 0.13,
        0.14, 0.15, 0.16, 0.17, 0.19, 0.2, 0.21, 0.22, 0.26, 0.28,
        0.32, 0.35, 0.45, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9,
    ]:
        add(
            f"burgers_target_gaussian_corr{slug_float(corr)}",
            "target_loss_param_shift",
            "gaussian_correlation_target_band",
            {"kernel": "gaussian", "correlation_length": corr, "transform": "identity"},
            "Gaussian GRF correlation length candidate accepted only if FNO RMSE is in the target band.",
        )

    for corr in [0.18, 0.2, 0.22, 0.26, 0.28, 0.3, 0.34, 0.38, 0.42, 0.6, 0.8, 0.9, 1.2]:
        for nu in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
            add(
                f"burgers_target_matern_corr{slug_float(corr)}_nu{slug_float(nu)}",
                "target_loss_kernel_shift",
                "matern_kernel_target_band",
                {"kernel": "matern", "correlation_length": corr, "matern_nu": nu, "transform": "identity"},
                "Matern GRF candidate accepted only if FNO RMSE is in the target band.",
            )

    for scale in [0.2, 0.22, 0.24, 0.26, 0.28, 0.32, 0.34, 0.36, 0.38, 0.4, 0.42, 0.45]:
        add(
            f"burgers_target_sawtooth_add_scale{slug_float(scale)}_shift0",
            "target_loss_pattern_shift",
            "sawtooth_target_band",
            {"kernel": "gaussian", "correlation_length": 0.03, "transform": "sawtooth_add", "scale": scale, "shift": 0.0},
            "Moderate sawtooth perturbation candidate accepted only if FNO RMSE is in the target band.",
        )

    for shift in [0.18, 0.22, 0.26, 0.3, 0.34, 0.36, 0.38, 0.42]:
        add(
            f"burgers_target_centered_scale_shift_scale1_shift{slug_float(shift)}",
            "target_loss_range_shift",
            "centered_positive_shift_target_band",
            {"kernel": "gaussian", "correlation_length": 0.03, "transform": "centered_scale_shift", "scale": 1.0, "shift": shift},
            "Moderate centered positive shift candidate accepted only if FNO RMSE is in the target band.",
        )

    return candidates[:max_candidates]


def build_darcy_candidates(existing_keys: set[str], max_candidates: int) -> list[GenDatasetSpec]:
    candidates: list[GenDatasetSpec] = []
    seen = set(existing_keys)
    seed_base = 20280700
    idx = 1

    def add(dataset_id: str, tier: str, family: str, params: dict[str, Any], desc: str) -> None:
        nonlocal idx
        add_candidate(candidates, seen, GenDatasetSpec("darcy", dataset_id, tier, family, 200, seed_base + idx, params, desc))
        idx += 1

    # Threshold bias around the old binary 3/12 setup gives a smooth knob for
    # moving the Darcy loss into the desired band while keeping coefficients binary.
    for bias in [
        0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28,
        0.30, 0.32, 0.34, 0.36, 0.38, 0.40, 0.42, 0.44, 0.46,
    ]:
        add(
            f"darcy_target_identity_bias{slug_float(bias)}",
            "target_loss_area_shift",
            "binary_area_target_band",
            {"alpha": 2.0, "tau": 3.0, "low": 3.0, "high": 12.0, "threshold_bias": bias, "latent_transform": "identity"},
            "Binary 3/12 coefficient with threshold area shift, accepted only if FNO RMSE is in the target band.",
        )

    for bias in [-0.06, -0.04, -0.02, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16]:
        add(
            f"darcy_target_square_centered_bias{slug_float(bias)}",
            "target_loss_latent_shift",
            "square_centered_target_band",
            {"alpha": 2.0, "tau": 3.0, "low": 3.0, "high": 12.0, "threshold_bias": bias, "latent_transform": "square_centered"},
            "Square-centered latent coefficient candidate accepted only if FNO RMSE is in the target band.",
        )

    for bias in [0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18]:
        add(
            f"darcy_target_log_abs_centered_bias{slug_float(bias)}",
            "target_loss_latent_shift",
            "log_abs_centered_target_band",
            {"alpha": 2.0, "tau": 3.0, "low": 3.0, "high": 12.0, "threshold_bias": bias, "latent_transform": "log_abs_centered"},
            "Log-abs centered latent coefficient candidate accepted only if FNO RMSE is in the target band.",
        )

    for alpha in [0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0, 1.1]:
        for tau in [1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0]:
            add(
                f"darcy_target_alpha{slug_float(alpha)}_tau{slug_float(tau)}_bin3_12",
                "target_loss_spectrum_shift",
                "binary_grf_spectrum_target_band",
                {"alpha": alpha, "tau": tau, "low": 3.0, "high": 12.0, "threshold_bias": 0.0, "latent_transform": "identity"},
                "Binary 3/12 GRF spectrum candidate accepted only if FNO RMSE is in the target band.",
            )

    for low, high in [(3.0, 14.0), (3.0, 16.0), (3.0, 18.0), (3.0, 20.0), (2.5, 12.0), (3.5, 12.0), (4.0, 12.0)]:
        for alpha, tau in [(0.5, 15.0), (0.6, 12.0), (0.7, 10.0), (0.8, 8.0), (1.0, 6.0), (1.2, 4.0), (1.5, 3.0)]:
            add(
                f"darcy_target_alpha{slug_float(alpha)}_tau{slug_float(tau)}_bin{slug_float(low)}_{slug_float(high)}",
                "target_loss_contrast_shift",
                "coefficient_contrast_target_band",
                {"alpha": alpha, "tau": tau, "low": low, "high": high, "threshold_bias": 0.0, "latent_transform": "identity"},
                "Coefficient contrast candidate accepted only if FNO RMSE is in the target band.",
            )

    return candidates[:max_candidates]


def write_selection_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "task", "status", "dataset_id", "rmse", "ratio_to_test", "relative_l2", "path", "seed",
        "kernel", "correlation_length", "matern_nu", "alpha", "tau", "low", "high",
        "threshold_bias", "latent_transform", "transform", "scale", "shift",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_selection_json(path: Path, task: str, test_rmse: float, low: float, high: float, kept: list[dict[str, Any]], accepted: list[dict[str, Any]], rejected: list[dict[str, Any]]) -> None:
    payload = {
        "task": task,
        "test_rmse": test_rmse,
        "target_ratio_low": low / test_rmse,
        "target_ratio_high": high / test_rmse,
        "target_rmse_low": low,
        "target_rmse_high": high,
        "kept_existing_count": len(kept),
        "accepted_new_count": len(accepted),
        "rejected_candidate_count": len(rejected),
        "kept_existing": kept,
        "accepted_new": accepted,
        "rejected_candidates": rejected,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def process_task(
    *,
    task: str,
    target_root: Path,
    metrics_csv: Path,
    target_low_ratio: float,
    target_high_ratio: float,
    target_count: int,
    max_candidates: int,
    chunk_size: int,
    generate_fn: Callable,
    build_candidates_fn: Callable[[set[str], int], list[GenDatasetSpec]],
    load_model_fn: Callable[[torch.device], Any],
    generation_batch_size: int,
    eval_batch_size: int,
    device: torch.device,
) -> None:
    task_dir = target_root / task
    test_rmse, metrics_by_id = load_metrics(metrics_csv, task)
    target_low = target_low_ratio * test_rmse
    target_high = target_high_ratio * test_rmse
    current_paths = sorted(task_dir.glob("*.pt"))
    kept_paths: list[Path] = []
    removed_paths: list[Path] = []
    kept_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []

    for path in current_paths:
        row = metrics_by_id.get(path.stem)
        if row is None:
            removed_paths.append(path)
            continue
        meta = torch_load(path).get("metadata", {})
        params = meta.get("params", {})
        rmse = float(row["rmse"])
        ratio = rmse / test_rmse
        record = {
            "task": task,
            "status": "kept_existing" if target_low_ratio <= ratio <= target_high_ratio else "removed_existing",
            "dataset_id": path.stem,
            "rmse": rmse,
            "ratio_to_test": ratio,
            "relative_l2": float(row["relative_l2"]),
            "path": str(path.relative_to(PROJECT_ROOT)),
            "seed": meta.get("seed"),
            **(params if isinstance(params, dict) else {}),
        }
        if target_low_ratio <= ratio <= target_high_ratio:
            kept_paths.append(path)
            kept_rows.append(record)
        else:
            removed_paths.append(path)
            rejected_rows.append(record)

    for path in removed_paths:
        path.unlink()

    print(
        f"[{task}] test_rmse={test_rmse:.8g}, target=[{target_low:.8g}, {target_high:.8g}], "
        f"kept={len(kept_paths)}, removed={len(removed_paths)}",
        flush=True,
    )

    existing_keys = existing_param_keys(task_dir)
    candidates = build_candidates_fn(existing_keys, max_candidates)
    if target_count <= len(kept_paths):
        candidates = []
    print(f"[{task}] need={target_count - len(kept_paths)}, queued={len(candidates)}", flush=True)

    model = load_model_fn(device)
    accepted_specs = [spec_from_existing(task, path) for path in kept_paths]
    accepted_rows: list[dict[str, Any]] = []
    tmp_root = target_root / f"_{task}_candidate_work"
    tmp_root.mkdir(parents=True, exist_ok=True)

    try:
        cursor = 0
        chunk_size = max(1, int(chunk_size))
        while cursor < len(candidates) and len(accepted_specs) < target_count:
            chunk = candidates[cursor : cursor + chunk_size]
            start_idx = cursor + 1
            cursor += len(chunk)
            print(f"[{task}] candidate chunk {start_idx}-{cursor} / {len(candidates)}", flush=True)
            shutil.rmtree(tmp_root / task, ignore_errors=True)
            generate_fn(chunk, tmp_root, overwrite=True, batch_size=generation_batch_size)
            for ci, spec in enumerate(chunk, start_idx):
                candidate_path = tmp_root / task / f"{spec.dataset_id}.pt"
                if len(accepted_specs) >= target_count:
                    candidate_path.unlink(missing_ok=True)
                    continue
                metrics = evaluate_task_file(task, model, candidate_path, batch_size=eval_batch_size, device=device)
                ratio = float(metrics["rmse"]) / test_rmse
                row = {
                    "task": task,
                    "status": "accepted_new" if target_low_ratio <= ratio <= target_high_ratio else "rejected_new",
                    "dataset_id": spec.dataset_id,
                    "rmse": float(metrics["rmse"]),
                    "ratio_to_test": ratio,
                    "relative_l2": float(metrics["relative_l2"]),
                    "path": str((task_dir / candidate_path.name).relative_to(PROJECT_ROOT)),
                    "seed": spec.seed,
                    **spec.params,
                }
                print(f"[{task} candidate {ci}/{len(candidates)}] rmse={row['rmse']:.8g}, ratio={ratio:.3f}, {row['status']}", flush=True)
                if target_low_ratio <= ratio <= target_high_ratio:
                    final_path = task_dir / candidate_path.name
                    shutil.move(str(candidate_path), final_path)
                    accepted_specs.append(spec)
                    accepted_rows.append(row)
                else:
                    rejected_rows.append(row)
                    candidate_path.unlink(missing_ok=True)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(accepted_specs) < target_count:
        raise RuntimeError(f"{task}: accepted {len(accepted_specs)}, target was {target_count}; increase candidates")

    rows = kept_rows + accepted_rows
    rows.sort(key=lambda r: (float(r["ratio_to_test"]), str(r["dataset_id"])))
    write_selection_csv(target_root / f"{task}_target_loss_selection.csv", rows)
    write_selection_json(
        target_root / f"{task}_target_loss_selection.json",
        task,
        test_rmse,
        target_low,
        target_high,
        kept_rows,
        accepted_rows,
        rejected_rows,
    )
    print(f"[{task}] done: {len(rows)} accepted/kept rows", flush=True)


def rebuild_manifest_from_files(root: Path) -> None:
    records = []
    for task in ("burgers", "darcy", "ns2d"):
        for path in sorted((root / task).glob("*.pt")):
            meta = torch_load(path).get("metadata", {})
            records.append(
                {
                    "task": task,
                    "dataset_id": meta.get("dataset_id", path.stem),
                    "tier": meta.get("similarity_tier", "copied"),
                    "family": meta.get("family", "copied"),
                    "n": meta.get("nsamples", 0),
                    "seed": meta.get("seed", 0),
                    "params": meta.get("params", {}),
                    "description": meta.get("description", "Copied from source dataset."),
                    "path": str(path),
                    "bytes": path.stat().st_size,
                }
            )
    write_manifest(root, records)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=PROJECT_ROOT / "generalization_datasets_ns2d_rmse_1p5_3x")
    parser.add_argument("--target-root", type=Path, default=PROJECT_ROOT / "generalization_datasets_rmse_1p5_3x_all")
    parser.add_argument("--metrics-csv", type=Path, default=PROJECT_ROOT / "generalization_eval_repaired_all" / "metrics.csv")
    parser.add_argument("--tasks", default="burgers,darcy")
    parser.add_argument("--target-low-ratio", type=float, default=1.5)
    parser.add_argument("--target-high-ratio", type=float, default=3.0)
    parser.add_argument("--burgers-target-count", type=int, default=50)
    parser.add_argument("--darcy-target-count", type=int, default=50)
    parser.add_argument("--max-burgers-candidates", type=int, default=140)
    parser.add_argument("--max-darcy-candidates", type=int, default=180)
    parser.add_argument("--candidate-generate-chunk-size", type=int, default=10)
    parser.add_argument("--burgers-generation-batch-size", type=int, default=200)
    parser.add_argument("--darcy-generation-batch-size", type=int, default=20)
    parser.add_argument("--burgers-eval-batch-size", type=int, default=128)
    parser.add_argument("--darcy-eval-batch-size", type=int, default=10)
    parser.add_argument("--overwrite-target", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    target_root = args.target_root.resolve()
    print(f"[copy] {source_root} -> {target_root}", flush=True)
    copy_source_root(source_root, target_root, args.overwrite_target)

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    device = torch.device(args.device)
    for task in tasks:
        if task == "burgers":
            process_task(
                task=task,
                target_root=target_root,
                metrics_csv=args.metrics_csv.resolve(),
                target_low_ratio=args.target_low_ratio,
                target_high_ratio=args.target_high_ratio,
                target_count=args.burgers_target_count,
                max_candidates=args.max_burgers_candidates,
                chunk_size=args.candidate_generate_chunk_size,
                generate_fn=generate_burgers,
                build_candidates_fn=build_burgers_candidates,
                load_model_fn=load_burgers_model,
                generation_batch_size=args.burgers_generation_batch_size,
                eval_batch_size=args.burgers_eval_batch_size,
                device=device,
            )
        elif task == "darcy":
            process_task(
                task=task,
                target_root=target_root,
                metrics_csv=args.metrics_csv.resolve(),
                target_low_ratio=args.target_low_ratio,
                target_high_ratio=args.target_high_ratio,
                target_count=args.darcy_target_count,
                max_candidates=args.max_darcy_candidates,
                chunk_size=args.candidate_generate_chunk_size,
                generate_fn=generate_darcy,
                build_candidates_fn=build_darcy_candidates,
                load_model_fn=load_darcy_model,
                generation_batch_size=args.darcy_generation_batch_size,
                eval_batch_size=args.darcy_eval_batch_size,
                device=device,
            )
        else:
            raise ValueError(f"unsupported task for this builder: {task}")

    rebuild_manifest_from_files(target_root)
    print(f"[done] wrote target-loss dataset root: {target_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
