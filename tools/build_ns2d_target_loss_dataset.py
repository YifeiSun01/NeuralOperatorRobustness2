#!/usr/bin/env python3
"""Build an NS2D generalization copy whose RMSE sits in a target test-loss band.

The source generalization dataset is left untouched.  The script copies the
source root, keeps NS2D files that already satisfy the requested RMSE/test RMSE
ratio, and replaces the rest with newly generated non-duplicate NS2D datasets.
Each candidate is accepted only after running the retained NS model on all 50
samples and checking the formal global RMSE used by the evaluator.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.evaluate_generalization_models import (  # noqa: E402
    DatasetSpec as EvalDatasetSpec,
    evaluate_dataset,
    load_ns2d_model,
)
from tools.generate_generalization_datasets import (  # noqa: E402
    DatasetSpec as GenDatasetSpec,
    generate_ns,
    slug_float,
    write_manifest,
)


def torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=False)


def load_metrics(path: Path) -> tuple[float, dict[str, dict[str, Any]]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    test_rows = [r for r in rows if r["task"] == "ns2d" and r["split"] == "test"]
    if len(test_rows) != 1:
        raise ValueError(f"expected exactly one ns2d test row in {path}, got {len(test_rows)}")
    test_rmse = float(test_rows[0]["rmse"])
    by_id = {
        r["dataset_id"]: r
        for r in rows
        if r["task"] == "ns2d" and r["split"] == "generalization"
    }
    return test_rmse, by_id


def params_key(params: dict[str, Any]) -> str:
    return json.dumps(params, sort_keys=True, separators=(",", ":"))


def existing_param_keys(ns_dir: Path) -> set[str]:
    keys: set[str] = set()
    for path in sorted(ns_dir.glob("*.pt")):
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


def spec_from_existing(path: Path) -> GenDatasetSpec:
    data = torch_load(path)
    meta = data.get("metadata", {})
    params = meta.get("params") or {}
    return GenDatasetSpec(
        task="ns2d",
        dataset_id=str(meta.get("dataset_id", path.stem)),
        tier=str(meta.get("similarity_tier", "target_loss_kept")),
        family=str(meta.get("family", "kept_existing")),
        n=int(meta.get("nsamples", data["y"].shape[0])),
        seed=int(meta.get("seed", 0)),
        params=dict(params),
        description=str(meta.get("description", "Kept from source dataset because its RMSE ratio is inside the target band.")),
    )


def evaluate_ns_file(model, path: Path, batch_size: int, device: torch.device) -> dict[str, Any]:
    spec = EvalDatasetSpec(
        task="ns2d",
        dataset_id=path.stem,
        split="generalization",
        path=path,
        source="generated",
        manual_tier="target_loss_candidate",
        manual_rank=3.0,
    )
    return evaluate_dataset(model, spec, device, batch_size=batch_size, max_samples=None)


def add_candidate(candidates: list[GenDatasetSpec], seen: set[str], candidate: GenDatasetSpec) -> None:
    key = params_key(candidate.params)
    if key in seen:
        return
    seen.add(key)
    candidates.append(candidate)


def build_candidate_specs(existing_keys: set[str], max_candidates: int) -> list[GenDatasetSpec]:
    """Create a deterministic but broad candidate queue around the empirical good band."""
    candidates: list[GenDatasetSpec] = []
    seen = set(existing_keys)
    seed_base = 20270600

    def add(dataset_id: str, tier: str, family: str, idx: int, params: dict[str, Any], description: str) -> None:
        add_candidate(
            candidates,
            seen,
            GenDatasetSpec("ns2d", dataset_id, tier, family, 50, seed_base + idx, params, description),
        )

    idx = 1

    # Identity GRF: alpha controls roughness, tau controls spatial scale.  Values
    # near alpha 1-2 were the useful region in the repaired metrics.
    identity_grid = [
        (1.05, 3.0), (1.05, 4.0), (1.05, 5.0), (1.05, 6.0), (1.05, 8.0), (1.05, 12.0),
        (1.10, 3.0), (1.10, 4.0), (1.10, 5.0), (1.10, 6.0), (1.10, 8.0), (1.10, 12.0),
        (1.15, 3.0), (1.15, 4.0), (1.15, 5.0), (1.15, 6.0), (1.15, 8.0), (1.15, 12.0),
        (1.20, 4.0), (1.20, 5.0), (1.20, 6.0), (1.20, 8.0), (1.20, 10.0), (1.20, 12.0),
        (1.25, 4.0), (1.25, 5.0), (1.25, 6.0), (1.25, 8.0), (1.25, 10.0), (1.25, 12.0),
        (1.30, 4.0), (1.30, 5.0), (1.30, 6.0), (1.30, 8.0), (1.30, 10.0), (1.30, 12.0),
        (1.35, 4.0), (1.35, 5.0), (1.35, 6.0), (1.35, 8.0), (1.35, 10.0), (1.35, 12.0),
        (1.40, 4.0), (1.40, 5.0), (1.40, 6.0), (1.40, 8.0), (1.40, 10.0), (1.40, 12.0),
        (1.45, 4.0), (1.45, 5.0), (1.45, 6.0), (1.45, 8.0), (1.45, 10.0), (1.45, 12.0),
        (1.50, 5.0), (1.50, 6.0), (1.50, 8.0), (1.50, 10.0), (1.50, 12.0),
        (1.60, 5.0), (1.60, 6.0), (1.60, 8.0), (1.60, 10.0), (1.60, 12.0),
        (1.70, 6.0), (1.70, 8.0), (1.70, 10.0), (1.70, 12.0),
        (1.80, 6.0), (1.80, 8.0), (1.80, 10.0), (1.80, 12.0),
    ]
    for alpha, tau in identity_grid:
        add(
            f"ns_target_grf_alpha{slug_float(alpha)}_tau{slug_float(tau)}",
            "target_loss_param_shift",
            "periodic_grf_target_band",
            idx,
            {"alpha": alpha, "tau": tau, "transform": "identity"},
            "Periodic GRF candidate tuned by real NS model RMSE into the target loss band.",
        )
        idx += 1

    # Moderate amplitude / offset perturbations.  The old 0.25 shift and scale
    # 2.0 cases were in-band or slightly above; this grid fills the gap without
    # creating duplicate parameter tuples.
    transform_grid = []
    for scale in [1.6, 1.7, 1.8, 1.9, 2.1, 2.2]:
        transform_grid.append(("scale", 2.5, 7.0, scale, 0.0))
    for shift in [0.18, 0.20, 0.22, 0.28, 0.30, 0.32, 0.35, 0.38, 0.40]:
        transform_grid.append(("positive_shift", 2.5, 7.0, 1.0, shift))
        transform_grid.append(("negative_shift", 2.5, 7.0, 1.0, shift))
    for scale in [1.25, 1.35, 1.45, 1.6, 1.7]:
        for shift in [-0.22, -0.18, 0.18, 0.22, 0.28]:
            transform_grid.append(("scale_shift", 2.5, 7.0, scale, shift))
    for scale in [0.35, 0.4, 0.45, 0.55, 0.6, 0.65, 0.75]:
        transform_grid.append(("sign", 2.5, 7.0, scale, 0.0))
    for scale in [0.55, 0.6, 0.7, 0.8, 0.9]:
        transform_grid.append(("sawtooth_add", 2.5, 7.0, scale, 0.0))
    for transform, alpha, tau, scale, shift in transform_grid:
        add(
            f"ns_target_{transform}_alpha{slug_float(alpha)}_tau{slug_float(tau)}_scale{slug_float(scale)}_shift{slug_float(shift)}",
            "target_loss_range_pattern",
            "range_or_pattern_target_band",
            idx,
            {"alpha": alpha, "tau": tau, "transform": transform, "scale": scale, "shift": shift},
            "Moderate range/pattern shift candidate accepted only if its NS model RMSE is in the target band.",
        )
        idx += 1

    return candidates[:max_candidates]


def write_selection_report(
    path: Path,
    *,
    test_rmse: float,
    target_low: float,
    target_high: float,
    kept: list[dict[str, Any]],
    accepted: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
) -> None:
    report = {
        "test_rmse": test_rmse,
        "target_ratio_low": target_low / test_rmse,
        "target_ratio_high": target_high / test_rmse,
        "target_rmse_low": target_low,
        "target_rmse_high": target_high,
        "kept_existing_count": len(kept),
        "accepted_new_count": len(accepted),
        "rejected_candidate_count": len(rejected),
        "kept_existing": kept,
        "accepted_new": accepted,
        "rejected_candidates": rejected,
    }
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def write_selection_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "status",
        "dataset_id",
        "rmse",
        "ratio_to_test",
        "relative_l2",
        "path",
        "alpha",
        "tau",
        "transform",
        "scale",
        "shift",
        "seed",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=PROJECT_ROOT / "generalization_datasets")
    parser.add_argument("--target-root", type=Path, default=PROJECT_ROOT / "generalization_datasets_ns2d_rmse_1p5_3x")
    parser.add_argument("--metrics-csv", type=Path, default=PROJECT_ROOT / "generalization_eval_repaired_all" / "metrics.csv")
    parser.add_argument("--target-low-ratio", type=float, default=1.5)
    parser.add_argument("--target-high-ratio", type=float, default=3.0)
    parser.add_argument("--target-count", type=int, default=49)
    parser.add_argument("--max-candidates", type=int, default=160)
    parser.add_argument("--overwrite-target", action="store_true")
    parser.add_argument("--ns-batch-size", type=int, default=10)
    parser.add_argument("--ns-solver-batch-size", type=int, default=5)
    parser.add_argument("--ns-solver-mode", choices=["vmap", "lax-map"], default="vmap")
    parser.add_argument("--ns-max-abs-threshold", type=float, default=5.0)
    parser.add_argument("--ns-max-step-halvings", type=int, default=6)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--candidate-generate-chunk-size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    target_root = args.target_root.resolve()
    target_ns_dir = target_root / "ns2d"
    test_rmse, metrics_by_id = load_metrics(args.metrics_csv.resolve())
    target_low = args.target_low_ratio * test_rmse
    target_high = args.target_high_ratio * test_rmse

    print(f"[copy] {source_root} -> {target_root}", flush=True)
    copy_source_root(source_root, target_root, overwrite=args.overwrite_target)

    current_paths = sorted(target_ns_dir.glob("*.pt"))
    kept_paths = []
    removed_paths = []
    kept_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []

    for path in current_paths:
        row = metrics_by_id.get(path.stem)
        if row is None:
            removed_paths.append(path)
            continue
        rmse = float(row["rmse"])
        ratio = rmse / test_rmse
        params = torch_load(path).get("metadata", {}).get("params", {})
        record = {
            "status": "kept_existing" if args.target_low_ratio <= ratio <= args.target_high_ratio else "removed_existing",
            "dataset_id": path.stem,
            "rmse": rmse,
            "ratio_to_test": ratio,
            "relative_l2": float(row["relative_l2"]),
            "path": str(path.relative_to(PROJECT_ROOT)),
            "seed": torch_load(path).get("metadata", {}).get("seed"),
            **(params if isinstance(params, dict) else {}),
        }
        if args.target_low_ratio <= ratio <= args.target_high_ratio:
            kept_paths.append(path)
            kept_rows.append(record)
        else:
            removed_paths.append(path)
            rejected_rows.append(record)

    for path in removed_paths:
        path.unlink()

    print(
        f"[select] test_rmse={test_rmse:.8g}, target=[{target_low:.8g}, {target_high:.8g}], "
        f"kept={len(kept_paths)}, removed={len(removed_paths)}",
        flush=True,
    )

    existing_keys = existing_param_keys(target_ns_dir)
    candidates = build_candidate_specs(existing_keys, args.max_candidates)
    need = args.target_count - len(kept_paths)
    if need <= 0:
        candidates = []
    print(f"[candidates] need={need}, queued={len(candidates)}", flush=True)

    device = torch.device(args.device)
    model = load_ns2d_model(device)
    accepted_specs = [spec_from_existing(path) for path in kept_paths]
    accepted_rows: list[dict[str, Any]] = []
    candidate_tmp_root = target_root / "_candidate_work"
    candidate_tmp_root.mkdir(parents=True, exist_ok=True)

    try:
        cursor = 0
        chunk_size = max(1, int(args.candidate_generate_chunk_size))
        while cursor < len(candidates) and len(accepted_specs) < args.target_count:
            chunk = candidates[cursor : cursor + chunk_size]
            start_idx = cursor + 1
            cursor += len(chunk)
            print(f"[candidate chunk] generate {start_idx}-{cursor} / {len(candidates)}", flush=True)
            generate_ns(
                chunk,
                candidate_tmp_root,
                overwrite=True,
                batch_size=args.ns_batch_size,
                solver_batch_size=args.ns_solver_batch_size,
                solver_mode=args.ns_solver_mode,
                max_abs_threshold=args.ns_max_abs_threshold,
                max_step_halvings=args.ns_max_step_halvings,
                allow_unstable=False,
            )
            for ci, spec in enumerate(chunk, start_idx):
                candidate_path = candidate_tmp_root / "ns2d" / f"{spec.dataset_id}.pt"
                if len(accepted_specs) >= args.target_count:
                    candidate_path.unlink(missing_ok=True)
                    continue
                metrics = evaluate_ns_file(model, candidate_path, batch_size=args.eval_batch_size, device=device)
                ratio = float(metrics["rmse"]) / test_rmse
                row = {
                    "status": "accepted_new" if args.target_low_ratio <= ratio <= args.target_high_ratio else "rejected_new",
                    "dataset_id": spec.dataset_id,
                    "rmse": float(metrics["rmse"]),
                    "ratio_to_test": ratio,
                    "relative_l2": float(metrics["relative_l2"]),
                    "path": str((target_ns_dir / candidate_path.name).relative_to(PROJECT_ROOT)),
                    "seed": spec.seed,
                    **spec.params,
                }
                print(
                    f"[candidate {ci}/{len(candidates)}] rmse={row['rmse']:.8g}, ratio={ratio:.3f}, {row['status']}",
                    flush=True,
                )
                if args.target_low_ratio <= ratio <= args.target_high_ratio:
                    final_path = target_ns_dir / candidate_path.name
                    shutil.move(str(candidate_path), final_path)
                    accepted_specs.append(spec)
                    accepted_rows.append(row)
                else:
                    rejected_rows.append(row)
                    candidate_path.unlink(missing_ok=True)
            tmp_ns_dir = candidate_tmp_root / "ns2d"
            if tmp_ns_dir.exists():
                for leftover in tmp_ns_dir.glob("*.pt"):
                    leftover.unlink(missing_ok=True)
    finally:
        shutil.rmtree(candidate_tmp_root, ignore_errors=True)

    if len(accepted_specs) < args.target_count:
        raise RuntimeError(
            f"accepted {len(accepted_specs)} NS2D datasets, target was {args.target_count}; "
            "increase --max-candidates or broaden candidate grid"
        )

    all_rows = kept_rows + accepted_rows
    all_rows.sort(key=lambda r: (float(r["ratio_to_test"]), str(r["dataset_id"])))
    write_selection_report(
        target_root / "ns2d_target_loss_selection.json",
        test_rmse=test_rmse,
        target_low=target_low,
        target_high=target_high,
        kept=kept_rows,
        accepted=accepted_rows,
        rejected=rejected_rows,
    )
    write_selection_csv(target_root / "ns2d_target_loss_selection.csv", all_rows)

    # Rebuild manifest for the copied root with old Burgers/Darcy plus selected NS2D.
    records = []
    for task in ("burgers", "darcy"):
        for path in sorted((target_root / task).glob("*.pt")):
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
    for spec in accepted_specs:
        final_path = target_ns_dir / f"{spec.dataset_id}.pt"
        records.append({**asdict(spec), "path": str(final_path), "bytes": final_path.stat().st_size})
    write_manifest(target_root, records)

    print(f"[done] wrote {len(accepted_specs)} NS2D datasets to {target_ns_dir}", flush=True)
    print(f"[done] selection CSV: {target_root / 'ns2d_target_loss_selection.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
