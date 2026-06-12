#!/usr/bin/env python3
"""Build top-5% and top-10% visual-quality Darcy loss3 heatmap folders.

This reuses the existing 50-step ranked batch attack table, screens only the
per-dataset top 10% candidates, and chooses samples that balance two visual
requirements:

1. Loss3 model-minus-solver residual is visibly smaller than the other trained
   methods.
2. Loss3 delta has the same broad shape as the other trained methods, avoiding
   samples where the attack pattern is visually unrelated or residual sign flips.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tools.plot_darcy_five_model_attack_heatmaps_20260612 as heat
import tools.plot_darcy_five_model_batch_ranked_heatmaps_20260612 as batch

SOURCE_TAG = "20260612_loss3attack50_five_models_ranked5_batch_polished"
DEFAULT_RANKED_DIR = PROJECT_ROOT / "analysis_outputs" / f"darcy_five_model_batch_ranked_heatmaps_{SOURCE_TAG}"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "analysis_outputs"
DEFAULT_VIZ_ROOT = PROJECT_ROOT / "visualizations"
MODEL_ORDER = [m.name for m in heat.MODELS]
TRAINED_OTHERS = ["loss1", "loss2", "physics loss"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def ranked_samples(ranked_dir: Path, samples_per_dataset: int) -> tuple[list[str], list[dict[str, Any]]]:
    dataset_ids = json.loads((ranked_dir / "selected_dataset_ids.json").read_text(encoding="utf-8"))
    rows = read_csv(ranked_dir / "all_50step_batch_attack_results.csv")
    grouped: dict[tuple[str, int], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        grouped[(row["dataset_id"], int(float(row["sample_index"])) )][row["model"]] = row

    scored: list[dict[str, Any]] = []
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
            sample_scores.append(
                {
                    "dataset_rank": dataset_rank,
                    "dataset_id": dataset_id,
                    "sample_index": sample_index,
                    "loss3_adv_loss": loss3,
                    "best_other_model": best_other_model,
                    "best_other_adv_loss": best_other,
                    "loss3_margin_vs_best_other": best_other - loss3,
                    "loss3_margin_vs_mean_other": float(np.mean(list(others.values())) - loss3),
                    "baseline_adv_loss": adv_loss["baseline"],
                    "loss1_adv_loss": adv_loss["loss1"],
                    "loss2_adv_loss": adv_loss["loss2"],
                    "physics_loss_adv_loss": adv_loss["physics loss"],
                }
            )
        if len(sample_scores) != int(samples_per_dataset):
            raise RuntimeError(f"dataset {dataset_id} has {len(sample_scores)} complete samples, expected {samples_per_dataset}")
        sample_scores.sort(key=lambda r: (r["loss3_margin_vs_best_other"], -r["loss3_adv_loss"]), reverse=True)
        for rank, row in enumerate(sample_scores, 1):
            row["rank_within_dataset_best_is_1"] = rank
            scored.append(row)
    return dataset_ids, scored


def sample_id_for(row: dict[str, Any], prefix: str) -> str:
    fam = batch.family_slug(str(row["dataset_id"]))
    return f"{prefix}_{int(row['dataset_rank']):02d}_{fam}_idx{int(row['sample_index'])}_rank{int(row['rank_within_dataset_best_is_1']):02d}"


def make_samples(rows: list[dict[str, Any]], prefix: str) -> list[heat.SampleSpec]:
    return [
        heat.SampleSpec(
            sample_id_for(row, prefix),
            "generalization",
            str(row["dataset_id"]),
            int(row["sample_index"]),
        )
        for row in rows
    ]


def run_candidate_screen(args: argparse.Namespace, candidate_rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    variant = "loss3_top10pct_visualscreen_candidates"
    out_dir = args.out_root / f"darcy_five_model_attack_heatmaps_{args.tag}_{variant}"
    viz_dir = args.viz_root / f"darcy_five_model_attack_heatmaps_{args.tag}_{variant}"
    expected_arrays = len(candidate_rows) * len(MODEL_ORDER)
    existing_arrays = len(list((out_dir / "arrays").glob("*.npz"))) if (out_dir / "arrays").exists() else 0
    if out_dir.exists() and viz_dir.exists() and existing_arrays >= expected_arrays and not args.force:
        print(f"[reuse] candidate screen {heat.rel(out_dir)} arrays={existing_arrays}", flush=True)
        return out_dir, viz_dir

    if args.force:
        shutil.rmtree(out_dir, ignore_errors=True)
        shutil.rmtree(viz_dir, ignore_errors=True)
    samples = make_samples(candidate_rows, "candidate")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dataset_paths = heat.load_dataset_map(args.generalization_root.resolve())
    mini_args = argparse.Namespace(
        tag=args.tag,
        attack_steps=args.attack_steps,
        epsilon_fraction=args.epsilon_fraction,
        plot_batch_size=args.plot_batch_size,
    )
    print(f"[screen] running candidate attacks samples={len(samples)} models={len(MODEL_ORDER)}", flush=True)
    batch.write_variant_outputs(mini_args, variant, samples, dataset_paths, device, args.out_root, args.viz_root)
    return out_dir, viz_dir


def load_candidate_records(candidate_out: Path) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    summary_rows = read_csv(candidate_out / "summary.csv")
    records: list[dict[str, Any]] = []
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in summary_rows:
        npz_path = PROJECT_ROOT / row["npz_path"]
        arrays = np.load(npz_path)
        rec = {
            "sample_id": row["sample_id"],
            "split": row["split"],
            "dataset_id": row["dataset_id"],
            "sample_index": int(float(row["sample_index"])),
            "model": row["model"],
            "attack_loss_gain": float(row["attack_loss_gain"]),
            "adv_relative_l2_model_vs_solver": float(row["adv_relative_l2_model_vs_solver"]),
            "clean_loss_before_attack": float(row["clean_loss_before_attack"]),
            "adv_loss_after_attack": float(row["adv_loss_after_attack"]),
            "attack_loss_gain_relative": float(row["attack_loss_gain_relative"]),
            "adv_rmse_model_vs_solver": float(row["adv_rmse_model_vs_solver"]),
            "delta_l0_fraction": float(row["delta_l0_fraction"]),
            "delta_l2_rms": float(row["delta_l2_rms"]),
            "elapsed_seconds": float(row["elapsed_seconds"]),
            "model_source": row["model_source"],
            "checkpoint": row["checkpoint"],
        }
        for key in ["x0", "delta", "x_adv", "model_output", "solver_output", "model_minus_solver", "attack_loss_history", "attack_loss_gain_history"]:
            rec[key] = np.asarray(arrays[key])
        records.append(rec)
        by_key[(rec["sample_id"], rec["model"])] = rec
    return records, by_key


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    af = np.ravel(np.asarray(a, dtype=np.float64))
    bf = np.ravel(np.asarray(b, dtype=np.float64))
    denom = float(np.linalg.norm(af) * np.linalg.norm(bf))
    if denom <= 1e-20:
        return float("nan")
    return float(np.dot(af, bf) / denom)


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    am = np.abs(np.asarray(a)) > 1e-9
    bm = np.abs(np.asarray(b)) > 1e-9
    union = int(np.logical_or(am, bm).sum())
    if union == 0:
        return 1.0
    return float(np.logical_and(am, bm).sum() / union)


def score_candidate(row: dict[str, Any], by_key: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    sid = sample_id_for(row, "candidate")
    loss3 = by_key[(sid, "loss3")]
    d3 = loss3["delta"]
    e3 = loss3["model_minus_solver"]
    e3_norm = float(np.linalg.norm(np.ravel(e3)))
    e3_mean = float(np.nanmean(e3))
    cosines = []
    jaccards = []
    ratios = []
    sign_same = []
    all_ratios = []
    for model in TRAINED_OTHERS:
        rec = by_key[(sid, model)]
        cosines.append(cosine(d3, rec["delta"]))
        jaccards.append(jaccard(d3, rec["delta"]))
        enorm = float(np.linalg.norm(np.ravel(rec["model_minus_solver"])))
        ratios.append(enorm / max(e3_norm, 1e-20))
        sign_same.append(float(np.sign(e3_mean) == np.sign(float(np.nanmean(rec["model_minus_solver"])))) )
    for model in [m for m in MODEL_ORDER if m != "loss3"]:
        rec = by_key[(sid, model)]
        enorm = float(np.linalg.norm(np.ravel(rec["model_minus_solver"])))
        all_ratios.append(enorm / max(e3_norm, 1e-20))

    cos_mean = float(np.nanmean(cosines))
    cos_min = float(np.nanmin(cosines))
    jac_mean = float(np.nanmean(jaccards))
    ratio_mean = float(np.mean(ratios))
    ratio_min = float(np.min(ratios))
    sign_fraction = float(np.mean(sign_same))

    # Balance: prefer clear residual fade, but reject unrelated delta shapes and sign flips.
    contrast_score = min(1.0, max(0.0, (ratio_min - 1.05) / 1.20))
    shape_score = min(1.0, max(0.0, (cos_mean - 0.25) / 0.55))
    min_shape_score = min(1.0, max(0.0, (cos_min - 0.05) / 0.55))
    sign_score = sign_fraction
    margin_scale = min(1.0, max(0.0, float(row["loss3_margin_vs_best_other"]) / 4.8e-6))
    display_score = 0.32 * contrast_score + 0.28 * shape_score + 0.15 * min_shape_score + 0.15 * sign_score + 0.10 * margin_scale
    if cos_mean < 0.18:
        display_score -= 0.22
    if sign_fraction < 0.67:
        display_score -= 0.22
    if ratio_min < 1.12:
        display_score -= 0.10

    return {
        **row,
        "candidate_sample_id": sid,
        "loss3_residual_l2": e3_norm,
        "loss3_residual_mean": e3_mean,
        "delta_cos_mean_trained_others": cos_mean,
        "delta_cos_min_trained_others": cos_min,
        "delta_jaccard_mean_trained_others": jac_mean,
        "residual_ratio_mean_trained_others": ratio_mean,
        "residual_ratio_min_trained_others": ratio_min,
        "residual_ratio_mean_all_others": float(np.mean(all_ratios)),
        "residual_ratio_min_all_others": float(np.min(all_ratios)),
        "residual_sign_agreement_trained_others": sign_fraction,
        "visual_display_score": float(display_score),
    }


def choose_visual_samples(scored: list[dict[str, Any]], max_rank: int, *, exclude_keys: set[tuple[str, int]] | None = None) -> list[dict[str, Any]]:
    exclude_keys = exclude_keys or set()
    selected = []
    by_dataset: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        if int(row["rank_within_dataset_best_is_1"]) <= int(max_rank):
            by_dataset[int(row["dataset_rank"])].append(row)
    for dataset_rank in sorted(by_dataset):
        pool = by_dataset[dataset_rank]
        pool = sorted(pool, key=lambda r: (r["visual_display_score"], r["residual_ratio_min_trained_others"], r["delta_cos_mean_trained_others"]), reverse=True)
        filtered = [r for r in pool if (str(r["dataset_id"]), int(r["sample_index"])) not in exclude_keys]
        selected.append((filtered or pool)[0])
    return selected


def emit_variant(
    args: argparse.Namespace,
    variant: str,
    selected_rows: list[dict[str, Any]],
    by_key: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    out_dir = args.out_root / f"darcy_five_model_attack_heatmaps_{args.tag}_{variant}"
    viz_dir = args.viz_root / f"darcy_five_model_attack_heatmaps_{args.tag}_{variant}"
    if args.force:
        shutil.rmtree(out_dir, ignore_errors=True)
        shutil.rmtree(viz_dir, ignore_errors=True)
    array_dir = out_dir / "arrays"
    out_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)
    array_dir.mkdir(parents=True, exist_ok=True)

    samples: list[heat.SampleSpec] = []
    records: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    for row in selected_rows:
        old_sid = row["candidate_sample_id"]
        new_sid = sample_id_for(row, variant)
        sample = heat.SampleSpec(new_sid, "generalization", str(row["dataset_id"]), int(row["sample_index"]))
        samples.append(sample)
        score_rows.append({**row, "sample_id": new_sid})
        for model in MODEL_ORDER:
            src = by_key[(old_sid, model)]
            arrays = {key: np.asarray(src[key]) for key in ["x0", "delta", "x_adv", "model_output", "solver_output", "model_minus_solver", "attack_loss_history", "attack_loss_gain_history"]}
            npz_path = array_dir / f"{new_sid}_{heat.slug(model)}_attack_fields.npz"
            np.savez_compressed(npz_path, **arrays)
            rec = {
                **arrays,
                "sample_id": new_sid,
                "split": "generalization",
                "dataset_id": str(row["dataset_id"]),
                "sample_index": int(row["sample_index"]),
                "model": model,
                "attack_loss_gain": float(src["attack_loss_gain"]),
                "adv_relative_l2_model_vs_solver": float(src["adv_relative_l2_model_vs_solver"]),
            }
            records.append(rec)
            summary_rows.append(
                {
                    "sample_id": new_sid,
                    "split": "generalization",
                    "dataset_id": str(row["dataset_id"]),
                    "sample_index": int(row["sample_index"]),
                    "model": model,
                    "model_source": src["model_source"],
                    "checkpoint": src["checkpoint"],
                    "attack_steps": int(args.attack_steps),
                    "epsilon_fraction": float(args.epsilon_fraction),
                    "clean_loss_before_attack": float(src["clean_loss_before_attack"]),
                    "adv_loss_after_attack": float(src["adv_loss_after_attack"]),
                    "attack_loss_gain": float(src["attack_loss_gain"]),
                    "attack_loss_gain_relative": float(src["attack_loss_gain_relative"]),
                    "adv_relative_l2_model_vs_solver": float(src["adv_relative_l2_model_vs_solver"]),
                    "adv_rmse_model_vs_solver": float(src["adv_rmse_model_vs_solver"]),
                    "delta_l0_fraction": float(src["delta_l0_fraction"]),
                    "delta_l2_rms": float(src["delta_l2_rms"]),
                    "elapsed_seconds": float(src["elapsed_seconds"]),
                    "npz_path": heat.rel(npz_path),
                }
            )

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
    with (out_dir / "selected_samples.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "split", "dataset_id", "sample_index"])
        writer.writeheader()
        writer.writerows([asdict(s) for s in samples])
    heat.write_csv(out_dir / "summary.csv", summary_rows)
    score_fields = list(score_rows[0].keys()) if score_rows else []
    write_csv(out_dir / "visual_selection_scores.csv", score_rows, score_fields)
    fig_paths = [heat.plot_sample(sample, records, ranges, viz_dir) for sample in samples]
    report_args = argparse.Namespace(tag=f"{args.tag}_{variant}", attack_steps=int(args.attack_steps), epsilon_fraction=float(args.epsilon_fraction))
    heat.write_report(out_dir, viz_dir, summary_rows, fig_paths, samples, report_args)
    with (out_dir / "README.md").open("a", encoding="utf-8") as f:
        f.write("\n## Visual Selection Criteria\n\n")
        f.write("Selected from the per-dataset top-percentile Loss3-advantage candidates using a display score that rewards smaller Loss3 residuals, delta-shape similarity to loss1/loss2/physics, and residual sign agreement. See `visual_selection_scores.csv`.\n")
    return {"variant": variant, "out_dir": heat.rel(out_dir), "viz_dir": heat.rel(viz_dir), "figures": [heat.rel(p) for p in fig_paths]}


def run(args: argparse.Namespace) -> None:
    args.ranked_dir = args.ranked_dir.resolve()
    args.out_root = args.out_root.resolve()
    args.viz_root = args.viz_root.resolve()
    _dataset_ids, ranked = ranked_samples(args.ranked_dir, args.samples_per_dataset)
    top10_rank = max(1, int(math.ceil(args.samples_per_dataset * 0.10)))
    top5_rank = max(1, int(math.ceil(args.samples_per_dataset * 0.05)))
    candidates = [r for r in ranked if int(r["rank_within_dataset_best_is_1"]) <= top10_rank]
    write_csv(args.ranked_dir / "top10_candidate_rank_rows_for_visual_screen.csv", candidates)
    candidate_out, _candidate_viz = run_candidate_screen(args, candidates)
    _records, by_key = load_candidate_records(candidate_out)
    scored = [score_candidate(r, by_key) for r in candidates]
    fields = list(scored[0].keys()) if scored else []
    write_csv(args.ranked_dir / "top10_candidate_visual_scores.csv", scored, fields)

    top5 = choose_visual_samples(scored, top5_rank)
    exclude = {(str(r["dataset_id"]), int(r["sample_index"])) for r in top5}
    top10 = choose_visual_samples(scored, top10_rank, exclude_keys=exclude)
    outputs = [
        emit_variant(args, "loss3_top5pct_visual", top5, by_key),
        emit_variant(args, "loss3_top10pct_visual", top10, by_key),
    ]
    manifest = {
        "tag": args.tag,
        "source_ranked_dir": heat.rel(args.ranked_dir),
        "candidate_out_dir": heat.rel(candidate_out),
        "samples_per_dataset": int(args.samples_per_dataset),
        "top5_rank_cutoff": top5_rank,
        "top10_rank_cutoff": top10_rank,
        "selection_note": "top5 chooses the best display score among ranks <= top5 cutoff. top10 chooses the best display score among ranks <= top10 cutoff, excluding the top5 selected sample when another top10 candidate exists.",
        "outputs": outputs,
    }
    manifest_path = args.ranked_dir / "top_percentile_visual_heatmap_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=SOURCE_TAG)
    parser.add_argument("--ranked-dir", type=Path, default=DEFAULT_RANKED_DIR)
    parser.add_argument("--generalization-root", type=Path, default=PROJECT_ROOT / "generalization_datasets_darcy_binary_loss3targeted_20260611")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--viz-root", type=Path, default=DEFAULT_VIZ_ROOT)
    parser.add_argument("--samples-per-dataset", type=int, default=50)
    parser.add_argument("--attack-steps", type=int, default=50)
    parser.add_argument("--epsilon-fraction", type=float, default=0.025)
    parser.add_argument("--plot-batch-size", type=int, default=10)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
