#!/usr/bin/env python3
"""Build a wide-parameter Burgers dataset targeted to stronger loss3 advantage.

This keeps the user's wide visible-parameter requirements, avoids attack-derived
samples, and writes a new dataset root instead of modifying the previous one.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.generate_generalization_datasets import json_ready, slug_float, write_manifest
from tools.generate_burgers_semantic_loss3_favored_generalization_20260611 import (
    SearchSpec,
    evaluate_candidates,
    generate_candidates,
    params_key,
    write_json,
    write_rows,
)
from tools.generate_burgers_semantic_wideparam_visible_generalization_20260611 import (
    HARD_MAX,
    HARD_MIN,
    audit_dataset,
    descriptive_label,
    make_id,
)

OLD_DATA_ROOT = PROJECT_ROOT / "generalization_datasets_burgers_semantic_wideparam_visible_20260611/round_00"
OLD_METRICS = PROJECT_ROOT / "forensics/burgers_semantic_wideparam_visible_round00_clean_loss_final_models_20260611/per_dataset_clean_metrics.csv"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611"
DEFAULT_FORENSICS_ROOT = PROJECT_ROOT / "forensics/burgers_semantic_wideparam_visible_loss3targeted_round00_20260611"

TARGET_FAMILY_COUNTS = {
    "gaussian": 12,
    "matern": 10,
    "powerlaw_fourier": 14,
    "sine_mixture": 12,
    "sawtooth": 1,
    "square_wave": 1,
}
FAMILY_ORDER = ["gaussian", "matern", "powerlaw_fourier", "sine_mixture", "sawtooth", "square_wave"]


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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


def fval(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def canonical_params_key(params: dict[str, Any]) -> str:
    ignored = {
        "display_label",
        "descriptive_name",
        "base_config",
        "wide_parameter_visible_profile",
        "loss3_targeted_profile",
        "hard_value_min",
        "hard_value_max",
    }
    clean = {k: v for k, v in params.items() if k not in ignored}
    return json.dumps(json_ready(clean), sort_keys=True, separators=(",", ":"))


def range_slug(lo: float, hi: float) -> str:
    return f"range{slug_float(lo)}to{slug_float(hi)}"


def add_target_metadata(params: dict[str, Any], base_config: str) -> dict[str, Any]:
    out = {
        **params,
        "transform": "range_affine",
        "transform_variant": "range",
        "loss3_targeted_profile": True,
        "wide_parameter_visible_profile": True,
        "hard_value_min": HARD_MIN,
        "hard_value_max": HARD_MAX,
        "base_config": base_config,
    }
    out["display_label"] = descriptive_label(out)
    out["descriptive_name"] = out["display_label"]
    return out


def build_replacement_specs(samples_per_dataset: int, max_candidates: int, seed_base: int) -> list[SearchSpec]:
    ranges = [
        (-0.65, 1.35),
        (-0.50, 1.50),
        (-0.30, 1.30),
        (-0.20, 1.20),
        (-0.10, 1.10),
        (0.00, 1.50),
        (0.20, 1.60),
        (0.15, 1.25),
        (0.00, 1.20),
        (0.05, 1.05),
    ]
    bases: list[tuple[str, dict[str, Any], str]] = []
    for corr in [0.004, 0.006, 0.009, 0.012, 0.018, 0.025, 0.035, 0.05, 0.07, 0.10, 0.15, 0.22, 0.45, 0.65, 0.80]:
        bases.append((f"gauss_c{slug_float(corr)}", {"base_family": "gaussian", "correlation_length": corr}, "Gaussian GRF with visibly varied correlation length"))
    for corr, nu in [
        (0.020, 1.5), (0.030, 1.5), (0.040, 2.5), (0.055, 4.0),
        (0.060, 0.5), (0.080, 2.5), (0.100, 5.0), (0.120, 8.0),
        (0.180, 6.0), (0.250, 3.5), (0.350, 8.0), (0.550, 8.0),
        (0.700, 8.0), (0.900, 8.0),
    ]:
        bases.append((f"matern_c{slug_float(corr)}_nu{slug_float(nu)}", {"base_family": "matern", "correlation_length": corr, "matern_nu": nu}, "Matern GRF with broad c/nu but biased away from weak loss3 regions"))
    for alpha, k0 in [
        (0.45, 3.0), (0.60, 4.0), (0.80, 5.0), (1.00, 7.0),
        (1.20, 8.0), (1.50, 10.0), (1.80, 12.0), (2.20, 16.0),
        (2.50, 18.0), (3.00, 24.0), (3.50, 28.0), (4.00, 36.0),
        (4.80, 45.0), (5.50, 70.0), (6.20, 90.0),
    ]:
        bases.append((f"powerlaw_a{slug_float(alpha)}_k{slug_float(k0)}", {"base_family": "powerlaw_fourier", "spectral_alpha": alpha, "k0": k0}, "Power-law Fourier spectrum across slow-to-fast decay and wide radius"))
    sine_configs = [
        ([3, 5, 8, 13], 0.20), ([4, 7, 11], 0.35), ([5, 9, 15, 23], 0.25),
        ([7, 19, 43, 89], 0.15), ([8, 13, 21], 0.20), ([10, 20, 40, 80], 0.55),
        ([5, 10, 20, 40, 80], 0.90), ([16, 32, 64], 0.10), ([3, 9, 27, 81], 0.30),
        ([6, 12, 24, 48], 0.45), ([9, 27, 54, 108], 0.30), ([2, 6, 18, 54], 0.25),
        ([11, 17, 31, 47, 73], 0.15), ([1, 4, 9, 25, 64], 0.35),
    ]
    for freqs, decay in sine_configs:
        name = "sine_f" + "_".join(str(v) for v in freqs[:5]) + f"_decay{slug_float(decay)}"
        bases.append((name, {"base_family": "sine_mixture", "frequencies": freqs, "decay": decay}, "Sine-mixture field biased toward mid/high-frequency diverse patterns"))
    # Tiny sharp-wave quotas only.  These are generated but selection caps them at one each.
    for freq in [2, 3, 5]:
        bases.append((f"saw_f{freq}", {"base_family": "sawtooth", "frequency": freq}, "tiny sawtooth comparison quota"))
    for freq, duty in [(5, 0.35), (7, 0.35), (9, 0.40)]:
        bases.append((f"square_f{freq}_d{slug_float(duty)}", {"base_family": "square_wave", "frequency": freq, "duty": duty}, "tiny square-wave comparison quota"))

    specs: list[SearchSpec] = []
    seen: set[str] = set()
    for base_idx, (base_name, base_params, desc) in enumerate(bases):
        family = str(base_params["base_family"])
        family_ranges = ranges
        if family == "sine_mixture":
            family_ranges = [(-0.65, 1.35), (-0.50, 1.50), (-0.30, 1.30), (-0.20, 1.20), (0.20, 1.60), (0.00, 1.50)]
        elif family == "powerlaw_fourier":
            family_ranges = [(-0.65, 1.35), (-0.50, 1.50), (-0.20, 1.20), (-0.10, 1.10), (0.15, 1.25), (0.20, 1.60)]
        elif family == "matern":
            family_ranges = [(-0.65, 1.35), (-0.50, 1.50), (-0.30, 1.30), (0.00, 1.50), (0.15, 1.25), (0.20, 1.60)]
        for range_idx, (lo, hi) in enumerate(family_ranges):
            if lo < HARD_MIN or hi > HARD_MAX:
                continue
            params = add_target_metadata({**base_params, "target_min": lo, "target_max": hi}, base_name)
            key = canonical_params_key(params)
            if key in seen:
                continue
            seen.add(key)
            dataset_id = make_id("burgers_widevis_l3cand", params)
            dataset_id = f"{dataset_id}_r{range_idx:02d}_b{base_idx:03d}"
            specs.append(SearchSpec(
                dataset_id=dataset_id,
                tier="wide_parameter_visible_loss3targeted_candidate",
                family=f"wide_visible_loss3targeted_{family}",
                n=samples_per_dataset,
                seed=seed_base + base_idx * 1009 + range_idx * 37,
                params=params,
                description=f"{desc}; loss3-targeted replacement candidate; {params['display_label']}",
            ))
            if len(specs) >= max_candidates:
                return specs
    return specs


def old_pool_rows(metrics_path: Path, old_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in read_csv(metrics_path):
        path = Path(row["path"])
        if not path.exists():
            path = old_root / "burgers" / f"{row['dataset_id']}.pt"
        data = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        params = data.get("metadata", {}).get("params", {})
        family = str(params.get("base_family", row.get("family", "")))
        l1, l2, l3 = fval(row, "loss1_rmse_mean"), fval(row, "loss2_rmse_mean"), fval(row, "loss3_rmse_mean")
        r1, r2, r3 = fval(row, "loss1_relative_l2_mean"), fval(row, "loss2_relative_l2_mean"), fval(row, "loss3_relative_l2_mean")
        best_rmse = min(l1, l2)
        best_rel = min(r1, r2)
        strict = l3 < l1 and l3 < l2 and r3 < r1 and r3 < r2
        rows.append({
            "source_type": "old_wideparam",
            "dataset_id": row["dataset_id"],
            "path": str(path),
            "family": family,
            "params": json.dumps(json_ready(params), sort_keys=True),
            "params_key": canonical_params_key(params),
            "loss3_best_rmse": strict and l3 < l1 and l3 < l2,
            "loss3_best_relative_l2": strict,
            "loss3_rmse_ratio_vs_best_loss12": l3 / max(best_rmse, 1e-30),
            "loss3_relative_l2_ratio_vs_best_loss12": r3 / max(best_rel, 1e-30),
            "loss3_rmse_margin_vs_best_loss12": best_rmse - l3,
            "selection_score": 2.0 * (best_rmse - l3) + (best_rel - r3),
            "old_baseline_rmse": fval(row, "baseline_rmse_mean"),
            "old_loss1_rmse": l1,
            "old_loss2_rmse": l2,
            "old_loss3_rmse": l3,
        })
    return rows


def generated_pool_rows(score_path: Path) -> list[dict[str, Any]]:
    rows = read_score_rows(score_path)
    out: list[dict[str, Any]] = []
    for row in rows:
        params = json.loads(str(row["params"]))
        row["source_type"] = "generated_replacement"
        row["family"] = str(params.get("base_family", row.get("family", "")))
        row["params_key"] = canonical_params_key(params)
        out.append(row)
    return out


def row_strict(row: dict[str, Any]) -> bool:
    return bool(row.get("loss3_best_rmse")) and bool(row.get("loss3_best_relative_l2"))


def row_rank(row: dict[str, Any]) -> tuple[Any, ...]:
    ratio = fval(row, "loss3_rmse_ratio_vs_best_loss12")
    rel = fval(row, "loss3_relative_l2_ratio_vs_best_loss12")
    score = fval(row, "selection_score")
    source_bonus = 0 if row.get("source_type") == "generated_replacement" else 1
    return (not row_strict(row), ratio, rel, source_bonus, -score)


def copy_row_to_selected(row: dict[str, Any], out_path: Path, dataset_id: str) -> dict[str, Any]:
    src = Path(str(row["path"]))
    data = torch.load(src, map_location="cpu", weights_only=False, mmap=True)
    metadata = dict(data.get("metadata", {}))
    params = metadata.get("params", json.loads(str(row.get("params", "{}"))))
    metadata["dataset_id"] = dataset_id
    metadata["path"] = str(out_path)
    metadata["selection"] = {k: row.get(k) for k in row if k != "params"}
    metadata["selection_policy"] = {
        "name": "wide_parameter_visible_loss3targeted_replacement_20260611",
        "target_family_counts": TARGET_FAMILY_COUNTS,
        "hard_value_min": HARD_MIN,
        "hard_value_max": HARD_MAX,
        "non_attack_generated": True,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"x": data["x"].float().contiguous(), "y": data["y"].float().contiguous(), "metadata": metadata}, out_path)
    return {
        "task": "burgers",
        "dataset_id": dataset_id,
        "tier": "wide_parameter_visible_loss3targeted",
        "family": metadata.get("family", f"wide_visible_loss3targeted_{params.get('base_family', '')}"),
        "source_family": params.get("base_family", ""),
        "n": int(data["x"].shape[0]),
        "seed": metadata.get("seed", 0),
        "params": params,
        "description": metadata.get("description", params.get("display_label", "")),
        "path": str(out_path),
        "bytes": out_path.stat().st_size,
        "source_dataset_id": row.get("dataset_id"),
        "source_type": row.get("source_type"),
        "loss3_strict_best_screen": row_strict(row),
        "loss3_rmse_ratio_vs_best_loss12": row.get("loss3_rmse_ratio_vs_best_loss12"),
        "loss3_relative_l2_ratio_vs_best_loss12": row.get("loss3_relative_l2_ratio_vs_best_loss12"),
    }


def select_dataset(pool: list[dict[str, Any]], selected_root: Path, overwrite_selected: bool) -> list[dict[str, Any]]:
    selected_dir = selected_root / "burgers"
    if selected_dir.exists() and any(selected_dir.glob("*.pt")) and not overwrite_selected:
        raise RuntimeError(f"{selected_dir} already contains data; pass --overwrite-selected to replace these files")
    selected_dir.mkdir(parents=True, exist_ok=True)
    selected: list[dict[str, Any]] = []
    used_keys: set[str] = set()
    family_counts = {k: 0 for k in FAMILY_ORDER}
    sorted_pool = sorted(pool, key=row_rank)

    for family in FAMILY_ORDER:
        for row in sorted_pool:
            if family_counts[family] >= TARGET_FAMILY_COUNTS[family]:
                break
            if row.get("family") != family:
                continue
            if not row_strict(row):
                continue
            key = str(row["params_key"])
            if key in used_keys:
                continue
            dataset_id = f"burgers_widevis_l3target_d{len(selected):02d}"
            out_path = selected_dir / f"{dataset_id}.pt"
            selected.append(copy_row_to_selected(row, out_path, dataset_id))
            used_keys.add(key)
            family_counts[family] += 1

    if len(selected) < sum(TARGET_FAMILY_COUNTS.values()):
        for row in sorted_pool:
            family = str(row.get("family"))
            if family not in TARGET_FAMILY_COUNTS:
                continue
            if family_counts[family] >= TARGET_FAMILY_COUNTS[family]:
                continue
            key = str(row["params_key"])
            if key in used_keys:
                continue
            dataset_id = f"burgers_widevis_l3target_d{len(selected):02d}"
            out_path = selected_dir / f"{dataset_id}.pt"
            selected.append(copy_row_to_selected(row, out_path, dataset_id))
            used_keys.add(key)
            family_counts[family] += 1
            if len(selected) >= sum(TARGET_FAMILY_COUNTS.values()):
                break

    write_manifest(selected_root, selected)
    write_rows(selected_root / "selected_candidate_scores.csv", selected)
    write_json(selected_root / "uniqueness_report.json", {
        "selected_count": len(selected),
        "unique_param_count": len({canonical_params_key(s["params"]) for s in selected}),
        "family_counts": dict(Counter(s["source_family"] for s in selected)),
        "target_family_counts": TARGET_FAMILY_COUNTS,
        "loss3_strict_best_screen_count": sum(1 for s in selected if s["loss3_strict_best_screen"]),
        "hard_value_min": HARD_MIN,
        "hard_value_max": HARD_MAX,
    })
    return selected


def write_nonwin_analysis(pool: list[dict[str, Any]], out_root: Path) -> None:
    old = [r for r in pool if r.get("source_type") == "old_wideparam"]
    non = [r for r in old if not row_strict(r)]
    rows = []
    for r in sorted(non, key=lambda x: fval(x, "loss3_rmse_ratio_vs_best_loss12"), reverse=True):
        params = json.loads(str(r["params"]))
        rows.append({
            "dataset_id": r["dataset_id"],
            "family": r["family"],
            "loss3_rmse_ratio_vs_best_loss12": r["loss3_rmse_ratio_vs_best_loss12"],
            "loss3_rmse_margin_vs_best_loss12": r["loss3_rmse_margin_vs_best_loss12"],
            "old_loss1_rmse": r["old_loss1_rmse"],
            "old_loss2_rmse": r["old_loss2_rmse"],
            "old_loss3_rmse": r["old_loss3_rmse"],
            "target_min": params.get("target_min"),
            "target_max": params.get("target_max"),
            "correlation_length": params.get("correlation_length"),
            "matern_nu": params.get("matern_nu"),
            "spectral_alpha": params.get("spectral_alpha"),
            "k0": params.get("k0"),
            "frequencies": params.get("frequencies"),
            "decay": params.get("decay"),
            "display_label": params.get("display_label"),
        })
    write_rows(out_root / "old_nonwin_characteristics.csv", rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--round-id", type=int, default=0)
    ap.add_argument("--forensics-root", type=Path, default=DEFAULT_FORENSICS_ROOT)
    ap.add_argument("--samples-per-dataset", type=int, default=200)
    ap.add_argument("--max-candidates", type=int, default=360)
    ap.add_argument("--seed-base", type=int, default=2026061173)
    ap.add_argument("--generation-batch-size", type=int, default=200)
    ap.add_argument("--eval-batch-size", type=int, default=256)
    ap.add_argument("--overwrite-candidates", action="store_true")
    ap.add_argument("--overwrite-selected", action="store_true")
    args = ap.parse_args()

    round_root = args.output_root / f"round_{args.round_id:02d}"
    candidate_root = args.output_root / f"round_{args.round_id:02d}_candidate_pool"
    specs = build_replacement_specs(int(args.samples_per_dataset), int(args.max_candidates), int(args.seed_base) + int(args.round_id) * 10000)
    candidate_root.mkdir(parents=True, exist_ok=True)
    write_json(candidate_root / "candidate_spec_manifest.json", [asdict(s) for s in specs])
    write_json(candidate_root / "config.json", {
        "candidate_count": len(specs),
        "samples_per_dataset": int(args.samples_per_dataset),
        "policy": "wide visible-parameter semantic candidates; no attack-generated samples; target replacing old loss3 non-winners",
        "target_family_counts": TARGET_FAMILY_COUNTS,
        "hard_value_min": HARD_MIN,
        "hard_value_max": HARD_MAX,
    })
    generate_candidates(specs, candidate_root, int(args.generation_batch_size), bool(args.overwrite_candidates))
    score_path = candidate_root / "candidate_model_scores.csv"
    if score_path.exists() and not args.overwrite_candidates:
        rows_generated = generated_pool_rows(score_path)
    else:
        candidate_paths = sorted((candidate_root / "burgers").glob("*.pt"))
        evaluate_candidates(candidate_paths, candidate_root, int(args.eval_batch_size))
        rows_generated = generated_pool_rows(score_path)

    pool = old_pool_rows(OLD_METRICS, OLD_DATA_ROOT) + rows_generated
    write_nonwin_analysis(pool, round_root)
    selected = select_dataset(pool, round_root, bool(args.overwrite_selected))
    selected_specs = [
        SearchSpec(
            dataset_id=str(item["dataset_id"]),
            tier=str(item["tier"]),
            family=str(item["family"]),
            n=int(item["n"]),
            seed=int(item.get("seed") or 0),
            params=dict(item["params"]),
            description=str(item.get("description", "")),
        )
        for item in selected
    ]
    audit_dataset(round_root, args.forensics_root, selected_specs)
    status = {
        "status": "complete" if len(selected) == sum(TARGET_FAMILY_COUNTS.values()) else "incomplete",
        "selected_count": len(selected),
        "target_count": sum(TARGET_FAMILY_COUNTS.values()),
        "loss3_strict_best_screen_count": sum(1 for item in selected if item["loss3_strict_best_screen"]),
        "round_root": str(round_root),
        "candidate_root": str(candidate_root),
        "forensics_root": str(args.forensics_root),
    }
    write_json(round_root / "run_status.json", status)
    print(json.dumps(json_ready(status), indent=2), flush=True)
    return 0 if status["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
