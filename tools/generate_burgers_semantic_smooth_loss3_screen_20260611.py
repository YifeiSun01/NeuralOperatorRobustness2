#!/usr/bin/env python3
"""Generate a smooth-dominant Burgers semantic generalization set and screen loss3.

This is the corrected version after the sawtooth-dominated loss3-favored search.
The candidate space is constrained to mostly Gaussian, Matern, sine-mixture, and
power-law Fourier fields, with only a tiny sawtooth/square quota and no spikes.
All target ranges stay near the original train/test [0, 1] scale.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.generate_generalization_datasets import json_ready, slug_float, write_manifest
from tools.generate_burgers_semantic_loss3_favored_generalization_20260611 import (
    PROJECT_ROOT,
    SearchSpec,
    add_spec,
    evaluate_candidates,
    generate_candidates,
    params_key,
    write_json,
    write_rows,
)

SMOOTH_QUOTAS = {
    "gaussian": 15,
    "matern": 14,
    "powerlaw_fourier": 10,
    "sine_mixture": 8,
    "sawtooth": 2,
    "square_wave": 1,
}
FAMILY_ORDER = ["gaussian", "matern", "powerlaw_fourier", "sine_mixture", "sawtooth", "square_wave"]
HARD_MIN = -0.7
HARD_MAX = 1.7


def build_smooth_specs(samples_per_dataset: int, max_candidates: int, seed_base: int) -> list[SearchSpec]:
    specs: list[SearchSpec] = []
    seen: set[str] = set()
    ranges = [
        (0.0, 1.0),
        (0.05, 1.05),
        (-0.05, 0.95),
        (0.10, 1.10),
        (-0.10, 1.10),
        (0.0, 1.20),
        (-0.20, 1.20),
        (0.15, 1.25),
        (-0.25, 1.25),
        (0.20, 1.40),
        (-0.30, 1.30),
        (0.0, 1.50),
        (-0.50, 1.50),
        (0.20, 1.60),
        (-0.60, 1.40),
    ]
    transforms = [
        ("range", {"transform": "range_affine"}, "direct smooth range remap"),
    ]
    base_configs: list[tuple[str, dict[str, Any], str]] = []
    for corr in [0.008, 0.012, 0.018, 0.025, 0.03, 0.045, 0.065, 0.09, 0.13, 0.20, 0.32, 0.50, 0.75]:
        base_configs.append((f"gauss_c{slug_float(corr)}", {"base_family": "gaussian", "correlation_length": corr}, f"Gaussian GRF corr={corr}"))
    for corr, nu in [
        (0.010, 0.5), (0.015, 0.7), (0.022, 1.0), (0.030, 1.5), (0.045, 2.5),
        (0.070, 3.5), (0.100, 5.0), (0.150, 7.0), (0.240, 2.5), (0.350, 5.0),
        (0.550, 8.0), (0.750, 8.0),
    ]:
        base_configs.append((f"matern_c{slug_float(corr)}_nu{slug_float(nu)}", {"base_family": "matern", "correlation_length": corr, "matern_nu": nu}, f"Matern GRF corr={corr}, nu={nu}"))
    for alpha, k0 in [
        (0.9, 3.0), (1.1, 4.0), (1.3, 6.0), (1.6, 8.0), (2.0, 10.0),
        (2.4, 14.0), (2.8, 18.0), (3.2, 22.0), (3.8, 28.0), (4.5, 34.0),
    ]:
        base_configs.append((f"powerlaw_a{slug_float(alpha)}_k{slug_float(k0)}", {"base_family": "powerlaw_fourier", "spectral_alpha": alpha, "k0": k0}, f"power-law Fourier alpha={alpha}, k0={k0}"))
    sine_configs = [
        ("sine_low", [1, 2, 3, 5], 0.70),
        ("sine_low_mid", [1, 3, 5, 8], 0.55),
        ("sine_mid", [2, 4, 7, 11, 17], 0.45),
        ("sine_mid_dense", [3, 5, 8, 13, 21], 0.55),
        ("sine_high_smooth", [5, 9, 15, 23, 35], 0.80),
        ("sine_mixed_decay", [2, 6, 10, 18, 31], 0.95),
        ("sine_dense_smooth", [1, 4, 9, 16, 25, 36], 1.10),
        ("sine_harmonic", [2, 4, 8, 16, 32], 0.85),
    ]
    for name, freqs, decay in sine_configs:
        base_configs.append((name, {"base_family": "sine_mixture", "frequencies": freqs, "decay": decay}, f"sine mixture freqs={freqs}, decay={decay}"))
    # Keep a tiny number of non-smooth comparison candidates. Selection quotas cap them.
    for freq in [2, 3, 5]:
        base_configs.append((f"saw_f{freq}", {"base_family": "sawtooth", "frequency": freq}, f"low-frequency sawtooth comparison freq={freq}"))
    for freq, duty in [(2, 0.45), (3, 0.40), (5, 0.35)]:
        base_configs.append((f"square_f{freq}_d{slug_float(duty)}", {"base_family": "square_wave", "frequency": freq, "duty": duty}, f"low-frequency square comparison freq={freq}, duty={duty}"))

    family_priority = {"gaussian": 0.00, "matern": 0.02, "powerlaw_fourier": 0.04, "sine_mixture": 0.06, "sawtooth": 0.50, "square_wave": 0.55}
    range_priority = {
        (0.0, 1.0): 0.00,
        (0.05, 1.05): 0.02,
        (-0.05, 0.95): 0.03,
        (0.10, 1.10): 0.04,
        (-0.10, 1.10): 0.05,
        (0.0, 1.20): 0.06,
        (-0.20, 1.20): 0.07,
        (0.15, 1.25): 0.08,
        (-0.25, 1.25): 0.09,
        (0.20, 1.40): 0.10,
        (-0.30, 1.30): 0.11,
        (0.0, 1.50): 0.12,
        (-0.50, 1.50): 0.16,
        (0.20, 1.60): 0.17,
        (-0.60, 1.40): 0.18,
    }
    combos: list[tuple[float, int, int, int, str, dict[str, Any], str, tuple[float, float], str, dict[str, Any], str]] = []
    for base_idx, (base_name, base_params, base_desc) in enumerate(base_configs):
        family = str(base_params["base_family"])
        for range_idx, (lo, hi) in enumerate(ranges):
            if lo < HARD_MIN or hi > HARD_MAX:
                continue
            for transform_idx, (transform_name, transform_params, transform_desc) in enumerate(transforms):
                tie = ((base_idx * 37 + range_idx * 11 + transform_idx * 5) % 997) * 1e-5
                priority = family_priority.get(family, 0.9) + range_priority.get((lo, hi), 1.0) + tie
                combos.append((priority, base_idx, range_idx, transform_idx, base_name, base_params, base_desc, (lo, hi), transform_name, transform_params, transform_desc))
    combos.sort(key=lambda item: item[0])
    for priority, base_idx, range_idx, transform_idx, base_name, base_params, base_desc, (lo, hi), transform_name, transform_params, transform_desc in combos:
        params = {
            **base_params,
            **transform_params,
            "target_min": lo,
            "target_max": hi,
            "base_config": base_name,
            "transform_variant": transform_name,
            "smooth_profile": True,
            "hard_value_min": HARD_MIN,
            "hard_value_max": HARD_MAX,
        }
        dataset_id = f"burgers_sem_smooth_c{len(specs):04d}_{base_name}_{transform_name}_range{slug_float(lo)}to{slug_float(hi)}"
        add_spec(
            specs,
            seen,
            dataset_id=dataset_id,
            family=f"semantic_smooth_{base_params['base_family']}_{transform_params['transform']}",
            seed=seed_base + len(specs) * 19 + base_idx * 1009 + range_idx * 101 + transform_idx,
            params=params,
            n=samples_per_dataset,
            description=f"Smooth-dominant semantic OOD Burgers dataset: {base_desc}; {transform_desc}; target range [{lo}, {hi}].",
        )
        if len(specs) >= max_candidates:
            return specs
    return specs


def parse_params(row: dict[str, Any]) -> dict[str, Any]:
    return json.loads(str(row["params"]))


def row_family(row: dict[str, Any]) -> str:
    return str(parse_params(row).get("base_family", ""))


def row_within_bounds(row: dict[str, Any]) -> bool:
    return float(row.get("x_min")) >= HARD_MIN - 1e-6 and float(row.get("x_max")) <= HARD_MAX + 1e-6


def row_loss3_strict(row: dict[str, Any]) -> bool:
    return bool(row["loss3_best_rmse"]) and bool(row["loss3_best_relative_l2"])


def copy_selected(rows: list[dict[str, Any]], selected_root: Path, select_count: int, require_strict: bool) -> list[dict[str, Any]]:
    if selected_root.exists():
        shutil.rmtree(selected_root)
    selected_dir = selected_root / "burgers"
    selected_dir.mkdir(parents=True, exist_ok=True)

    eligible = [r for r in rows if row_within_bounds(r)]
    # Rank strict loss3 winners first, then non-strict best ratios, while preserving family quotas.
    eligible.sort(key=lambda r: (not row_loss3_strict(r), float(r["loss3_rmse_ratio_vs_best_loss12"]), float(r["loss3_relative_l2_ratio_vs_best_loss12"])))
    selected: list[dict[str, Any]] = []
    used_keys: set[str] = set()
    family_counts = {k: 0 for k in FAMILY_ORDER}

    def take(row: dict[str, Any]) -> bool:
        family = row_family(row)
        if family not in SMOOTH_QUOTAS:
            return False
        if family_counts[family] >= SMOOTH_QUOTAS[family]:
            return False
        if row["params_key"] in used_keys:
            return False
        if require_strict and not row_loss3_strict(row):
            return False
        src = Path(row["path"])
        data = torch.load(src, map_location="cpu", weights_only=False, mmap=True)
        metadata = dict(data.get("metadata", {}))
        dataset_id = f"burgers_semantic_smooth_loss3screen_d{len(selected):02d}"
        metadata["dataset_id"] = dataset_id
        metadata["path"] = str(selected_dir / f"{dataset_id}.pt")
        metadata["selection"] = {k: row.get(k) for k in row if k not in {"params"}}
        metadata["smooth_selection_policy"] = {
            "family_quotas": SMOOTH_QUOTAS,
            "hard_value_min": HARD_MIN,
            "hard_value_max": HARD_MAX,
            "require_loss3_strict": require_strict,
        }
        out_path = selected_dir / f"{dataset_id}.pt"
        torch.save({"x": data["x"].float().contiguous(), "y": data["y"].float().contiguous(), "metadata": metadata}, out_path)
        selected.append({
            "task": "burgers",
            "tier": "smooth_semantic_loss3_screen",
            "dataset_id": dataset_id,
            "family": metadata.get("family", ""),
            "n": int(data["x"].shape[0]),
            "seed": metadata.get("seed", 0),
            "params": metadata.get("params", {}),
            "description": metadata.get("description", ""),
            "path": str(out_path),
            "bytes": out_path.stat().st_size,
            "source_candidate_id": row["dataset_id"],
            "source_family": family,
            "loss3_strict_best": row_loss3_strict(row),
            "loss3_rmse_ratio_vs_best_loss12": row["loss3_rmse_ratio_vs_best_loss12"],
            "loss3_relative_l2_ratio_vs_best_loss12": row["loss3_relative_l2_ratio_vs_best_loss12"],
            "x_min": row.get("x_min"),
            "x_max": row.get("x_max"),
        })
        used_keys.add(row["params_key"])
        family_counts[family] += 1
        return True

    for family in FAMILY_ORDER:
        for row in eligible:
            if len(selected) >= select_count:
                break
            if row_family(row) == family:
                take(row)
    # In case quotas sum differently, top off within bounds and allowed families.
    for row in eligible:
        if len(selected) >= select_count:
            break
        take(row)

    write_manifest(selected_root, selected)
    write_rows(selected_root / "selected_candidate_scores.csv", selected)
    param_keys = [params_key(item["params"]) for item in selected]
    duplicate_keys = sorted({key for key in param_keys if param_keys.count(key) > 1})
    strict_count = sum(1 for item in selected if item["loss3_strict_best"])
    write_json(selected_root / "uniqueness_report.json", {
        "selected_count": len(selected),
        "unique_param_count": len(set(param_keys)),
        "duplicate_param_count": len(duplicate_keys),
        "duplicate_param_keys": duplicate_keys,
        "family_counts": family_counts,
        "family_quotas": SMOOTH_QUOTAS,
        "hard_value_min": HARD_MIN,
        "hard_value_max": HARD_MAX,
        "loss3_strict_best_count": strict_count,
        "require_strict": require_strict,
    })
    return selected


def summarize_candidates(rows: list[dict[str, Any]], out_root: Path) -> None:
    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        family = row_family(row)
        counts.setdefault(family, {"total": 0, "within_bounds": 0, "strict_loss3": 0, "within_bounds_strict_loss3": 0})
        counts[family]["total"] += 1
        if row_within_bounds(row):
            counts[family]["within_bounds"] += 1
        if row_loss3_strict(row):
            counts[family]["strict_loss3"] += 1
        if row_within_bounds(row) and row_loss3_strict(row):
            counts[family]["within_bounds_strict_loss3"] += 1
    write_json(out_root / "candidate_family_summary.json", counts)


def read_score_rows(path: Path) -> list[dict[str, Any]]:
    bool_fields = {"loss3_best_rmse", "loss3_best_relative_l2"}
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for key, value in list(row.items()):
                if key in bool_fields:
                    row[key] = str(value).strip().lower() == "true"
                elif key not in {"dataset_id", "path", "params_key", "family", "tier", "params"}:
                    try:
                        row[key] = float(value)
                    except (TypeError, ValueError):
                        pass
            rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "generalization_datasets_burgers_semantic_smooth_loss3_screen_20260611")
    parser.add_argument("--round-id", type=int, default=0)
    parser.add_argument("--samples-per-dataset", type=int, default=200)
    parser.add_argument("--max-candidates", type=int, default=720)
    parser.add_argument("--select-count", type=int, default=50)
    parser.add_argument("--seed-base", type=int, default=2026061117)
    parser.add_argument("--generation-batch-size", type=int, default=200)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--require-strict", action="store_true", help="fail to fill quotas unless every selected dataset has loss3 as strict RMSE and relative-L2 winner")
    parser.add_argument("--overwrite-candidates", action="store_true")
    args = parser.parse_args()

    round_root = args.output_root / f"round_{args.round_id:02d}"
    candidate_root = args.output_root / f"round_{args.round_id:02d}_candidate_pool"
    specs = build_smooth_specs(int(args.samples_per_dataset), int(args.max_candidates), int(args.seed_base) + int(args.round_id) * 10000)
    if len({params_key(s.params) for s in specs}) != len(specs):
        raise RuntimeError("candidate spec parameter keys are not unique")
    write_json(candidate_root / "candidate_spec_manifest.json", [asdict(s) for s in specs])
    write_json(candidate_root / "config.json", {
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "candidate_count": len(specs),
        "policy": "smooth-dominant semantic OOD only; bounded ranges; tiny sawtooth/square quota; no spike train",
        "family_quotas": SMOOTH_QUOTAS,
        "hard_value_min": HARD_MIN,
        "hard_value_max": HARD_MAX,
    })
    generate_candidates(specs, candidate_root, int(args.generation_batch_size), bool(args.overwrite_candidates))
    candidate_paths = sorted((candidate_root / "burgers").glob("*.pt"))
    score_path = candidate_root / "candidate_model_scores.csv"
    if score_path.exists():
        print(f"[reuse scores] {score_path}", flush=True)
        rows = read_score_rows(score_path)
    else:
        rows = evaluate_candidates(candidate_paths, candidate_root, int(args.eval_batch_size))
    summarize_candidates(rows, candidate_root)
    selected = copy_selected(rows, round_root, int(args.select_count), bool(args.require_strict))
    status = {
        "status": "complete" if len(selected) >= int(args.select_count) else "insufficient_selected",
        "selected_count": len(selected),
        "select_count": int(args.select_count),
        "candidate_count": len(rows),
        "output_root": round_root,
        "candidate_root": candidate_root,
        "loss3_strict_best_count": sum(1 for item in selected if item["loss3_strict_best"]),
        "require_strict": bool(args.require_strict),
    }
    write_json(round_root / "run_status.json", status)
    print(json.dumps(json_ready(status), indent=2), flush=True)
    return 0 if len(selected) >= int(args.select_count) else 2


if __name__ == "__main__":
    raise SystemExit(main())
