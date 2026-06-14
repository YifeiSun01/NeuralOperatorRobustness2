#!/usr/bin/env python3
"""Find stronger Loss3 residual-fade Darcy heatmap examples.

The earlier ``*_visual_relaxed`` folders favored delta-shape similarity first.
This script searches a wider candidate pool from the existing 50-step batch
attack table and then re-runs only those candidates with arrays enabled.  The
final folders prioritize visually obvious ``model - solver`` fading while still
tracking residual sign agreement and delta-shape similarity.
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
import tools.build_darcy_loss3_visual_relaxed_heatmaps_20260612 as relaxed

SOURCE_TAG = "20260612_loss3attack50_five_models_ranked5_batch_polished"
RANKED_DIR = PROJECT_ROOT / "analysis_outputs" / f"darcy_five_model_batch_ranked_heatmaps_{SOURCE_TAG}"
OUT_ROOT = PROJECT_ROOT / "analysis_outputs"
VIZ_ROOT = PROJECT_ROOT / "visualizations"
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


def family_slug(dataset_id: str) -> str:
    return batch.family_slug(dataset_id)


def all_rank_rows(samples_per_dataset: int) -> list[dict[str, Any]]:
    dataset_ids = json.loads((RANKED_DIR / "selected_dataset_ids.json").read_text(encoding="utf-8"))
    rows = read_csv(RANKED_DIR / "all_50step_batch_attack_results.csv")
    grouped: dict[tuple[str, int], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        grouped[(row["dataset_id"], int(float(row["sample_index"])))][row["model"]] = row

    out: list[dict[str, Any]] = []
    for dataset_rank, dataset_id in enumerate(dataset_ids, 1):
        scored: list[dict[str, Any]] = []
        for sample_index in range(int(samples_per_dataset)):
            by_model = grouped.get((dataset_id, sample_index), {})
            if not set(MODEL_ORDER).issubset(by_model):
                continue
            adv_loss = {m: float(by_model[m]["adv_loss_after_attack"]) for m in MODEL_ORDER}
            loss3 = adv_loss["loss3"]
            trained_other_losses = [adv_loss[m] for m in TRAINED_OTHERS]
            best_other_model = min(TRAINED_OTHERS, key=lambda m: adv_loss[m])
            best_other = adv_loss[best_other_model]
            scored.append(
                {
                    "dataset_rank": dataset_rank,
                    "dataset_id": dataset_id,
                    "sample_index": sample_index,
                    "loss3_adv_loss": loss3,
                    "baseline_adv_loss": adv_loss["baseline"],
                    "loss1_adv_loss": adv_loss["loss1"],
                    "loss2_adv_loss": adv_loss["loss2"],
                    "physics_loss_adv_loss": adv_loss["physics loss"],
                    "best_other_model": best_other_model,
                    "best_other_adv_loss": best_other,
                    "loss3_margin_vs_best_other": best_other - loss3,
                    "loss3_margin_vs_mean_other": float(np.mean(trained_other_losses) - loss3),
                    "residual_ratio_min_est_from_loss": float(math.sqrt(max(best_other, 0.0) / max(loss3, 1e-30))),
                    "residual_ratio_mean_est_from_loss": float(math.sqrt(max(float(np.mean(trained_other_losses)), 0.0) / max(loss3, 1e-30))),
                }
            )
        scored.sort(key=lambda r: (r["loss3_margin_vs_best_other"], -r["loss3_adv_loss"]), reverse=True)
        for rank, row in enumerate(scored, 1):
            row["rank_within_dataset_best_is_1"] = rank
            out.append(row)
    return out


def choose_candidates(rank_rows: list[dict[str, Any]], per_dataset: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for dataset_rank in sorted({int(r["dataset_rank"]) for r in rank_rows}):
        pool = [
            r
            for r in rank_rows
            if int(r["dataset_rank"]) == dataset_rank
            and float(r["loss3_margin_vs_best_other"]) > 0.0
        ]
        # Use estimated residual fade to widen beyond the old strict top-5 only pool.
        pool = sorted(
            pool,
            key=lambda r: (
                float(r["residual_ratio_min_est_from_loss"]),
                float(r["loss3_margin_vs_best_other"]),
                -int(r["rank_within_dataset_best_is_1"]),
            ),
            reverse=True,
        )
        selected.extend(pool[: int(per_dataset)])
    return selected


def sample_id(row: dict[str, Any], prefix: str) -> str:
    fam = family_slug(str(row["dataset_id"]))
    return f"{prefix}_{int(row['dataset_rank']):02d}_{fam}_idx{int(row['sample_index'])}_rank{int(row['rank_within_dataset_best_is_1']):02d}"


def make_samples(rows: list[dict[str, Any]], prefix: str) -> list[heat.SampleSpec]:
    return [
        heat.SampleSpec(sample_id(row, prefix), "generalization", str(row["dataset_id"]), int(row["sample_index"]))
        for row in rows
    ]


def run_candidate_screen(args: argparse.Namespace, rows: list[dict[str, Any]]) -> Path:
    variant = "loss3_more_faded_candidate_screen"
    out_dir = args.out_root / f"darcy_five_model_attack_heatmaps_{args.tag}_{variant}"
    expected_arrays = len(rows) * len(MODEL_ORDER)
    existing_arrays = len(list((out_dir / "arrays").glob("*.npz"))) if (out_dir / "arrays").exists() else 0
    if out_dir.exists() and existing_arrays >= expected_arrays and not args.force:
        print(f"[reuse] {heat.rel(out_dir)} arrays={existing_arrays}", flush=True)
        return out_dir
    if args.force:
        shutil.rmtree(out_dir, ignore_errors=True)
        shutil.rmtree(args.viz_root / f"darcy_five_model_attack_heatmaps_{args.tag}_{variant}", ignore_errors=True)

    dataset_paths = heat.load_dataset_map(args.generalization_root.resolve())
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    mini_args = argparse.Namespace(
        tag=args.tag,
        attack_steps=int(args.attack_steps),
        epsilon_fraction=float(args.epsilon_fraction),
        plot_batch_size=int(args.plot_batch_size),
    )
    print(f"[candidate-screen] samples={len(rows)} models={len(MODEL_ORDER)}", flush=True)
    batch.write_variant_outputs(
        mini_args,
        variant,
        make_samples(rows, "morefadecandidate"),
        dataset_paths,
        device,
        args.out_root,
        args.viz_root,
    )
    return out_dir


def load_records(candidate_out: Path) -> dict[tuple[str, str], dict[str, Any]]:
    records: dict[tuple[str, str], dict[str, Any]] = {}
    for row in read_csv(candidate_out / "summary.csv"):
        npz_path = PROJECT_ROOT / row["npz_path"]
        z = np.load(npz_path)
        rec: dict[str, Any] = {
            "sample_id": row["sample_id"],
            "split": row["split"],
            "dataset_id": row["dataset_id"],
            "sample_index": int(float(row["sample_index"])),
            "model": row["model"],
            "model_source": row["model_source"],
            "checkpoint": row["checkpoint"],
            "attack_steps": int(float(row["attack_steps"])),
            "epsilon_fraction": float(row["epsilon_fraction"]),
            "clean_loss_before_attack": float(row["clean_loss_before_attack"]),
            "adv_loss_after_attack": float(row["adv_loss_after_attack"]),
            "attack_loss_gain": float(row["attack_loss_gain"]),
            "attack_loss_gain_relative": float(row["attack_loss_gain_relative"]),
            "adv_relative_l2_model_vs_solver": float(row["adv_relative_l2_model_vs_solver"]),
            "adv_rmse_model_vs_solver": float(row["adv_rmse_model_vs_solver"]),
            "delta_l0_fraction": float(row["delta_l0_fraction"]),
            "delta_l2_rms": float(row["delta_l2_rms"]),
            "elapsed_seconds": float(row["elapsed_seconds"]),
        }
        for key in ["x0", "delta", "x_adv", "model_output", "solver_output", "model_minus_solver", "attack_loss_history", "attack_loss_gain_history"]:
            rec[key] = np.asarray(z[key])
        records[(rec["sample_id"], rec["model"])] = rec
    return records


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    af = np.ravel(np.asarray(a, dtype=np.float64))
    bf = np.ravel(np.asarray(b, dtype=np.float64))
    denom = float(np.linalg.norm(af) * np.linalg.norm(bf))
    return float(np.dot(af, bf) / max(denom, 1e-20))


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    am = np.abs(np.asarray(a)) > 1e-9
    bm = np.abs(np.asarray(b)) > 1e-9
    return float(np.logical_and(am, bm).sum() / max(1, np.logical_or(am, bm).sum()))


def sign_agreement(a: np.ndarray, b: np.ndarray) -> float:
    af = np.ravel(np.asarray(a, dtype=np.float64))
    bf = np.ravel(np.asarray(b, dtype=np.float64))
    thresh_a = np.nanpercentile(np.abs(af), 25.0)
    thresh_b = np.nanpercentile(np.abs(bf), 25.0)
    mask = (np.abs(af) >= thresh_a) & (np.abs(bf) >= thresh_b)
    if not np.any(mask):
        return float(np.sign(np.nanmean(af)) == np.sign(np.nanmean(bf)))
    return float(np.mean(np.sign(af[mask]) == np.sign(bf[mask])))


def score_row(row: dict[str, Any], records: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    sid = str(row.get("candidate_sample_id") or sample_id(row, "morefadecandidate"))
    loss3 = records[(sid, "loss3")]
    d3 = loss3["delta"]
    e3 = loss3["model_minus_solver"]
    e3_norm = float(np.linalg.norm(np.ravel(e3)))
    e3_mean = float(np.nanmean(e3))
    delta_cosines: list[float] = []
    delta_jaccards: list[float] = []
    residual_ratios: list[float] = []
    residual_cosines: list[float] = []
    residual_mean_signs: list[float] = []
    residual_pixel_signs: list[float] = []
    for model in TRAINED_OTHERS:
        rec = records[(sid, model)]
        delta_cosines.append(cosine(d3, rec["delta"]))
        delta_jaccards.append(jaccard(d3, rec["delta"]))
        e = rec["model_minus_solver"]
        residual_ratios.append(float(np.linalg.norm(np.ravel(e)) / max(e3_norm, 1e-20)))
        residual_cosines.append(cosine(e3, e))
        residual_mean_signs.append(float(np.sign(e3_mean) == np.sign(float(np.nanmean(e)))))
        residual_pixel_signs.append(sign_agreement(e3, e))

    delta_cos_mean = float(np.nanmean(delta_cosines))
    delta_cos_min = float(np.nanmin(delta_cosines))
    residual_ratio_min = float(np.nanmin(residual_ratios))
    residual_ratio_mean = float(np.nanmean(residual_ratios))
    residual_cos_mean = float(np.nanmean(residual_cosines))
    residual_cos_min = float(np.nanmin(residual_cosines))
    mean_sign = float(np.mean(residual_mean_signs))
    pixel_sign = float(np.mean(residual_pixel_signs))

    fade_score = min(1.0, max(0.0, (residual_ratio_min - 1.20) / 2.20))
    residual_shape_score = min(1.0, max(0.0, (residual_cos_mean - 0.20) / 0.60))
    delta_shape_score = min(1.0, max(0.0, (delta_cos_mean - 0.10) / 0.45))
    sign_score = 0.70 * mean_sign + 0.30 * pixel_sign
    margin_score = min(1.0, max(0.0, float(row["loss3_margin_vs_best_other"]) / 4.8e-6))
    display_score = 0.46 * fade_score + 0.20 * residual_shape_score + 0.16 * delta_shape_score + 0.12 * sign_score + 0.06 * margin_score
    if mean_sign < 1.0:
        display_score -= 0.28
    if residual_cos_min < 0.05:
        display_score -= 0.12
    if delta_cos_mean < 0.12:
        display_score -= 0.08

    return {
        **row,
        "candidate_sample_id": sid,
        "loss3_residual_l2": e3_norm,
        "loss3_residual_mean": e3_mean,
        "delta_cos_mean_trained_others": delta_cos_mean,
        "delta_cos_min_trained_others": delta_cos_min,
        "delta_jaccard_mean_trained_others": float(np.nanmean(delta_jaccards)),
        "residual_ratio_min_trained_others": residual_ratio_min,
        "residual_ratio_mean_trained_others": residual_ratio_mean,
        "residual_cos_mean_trained_others": residual_cos_mean,
        "residual_cos_min_trained_others": residual_cos_min,
        "residual_mean_sign_agreement_trained_others": mean_sign,
        "residual_pixel_sign_agreement_trained_others": pixel_sign,
        "more_faded_display_score": float(display_score),
    }


def choose_selected(scored: list[dict[str, Any]], *, mode: str) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for dataset_rank in sorted({int(r["dataset_rank"]) for r in scored}):
        pool = [r for r in scored if int(r["dataset_rank"]) == dataset_rank]
        if mode == "balanced":
            filtered = [
                r
                for r in pool
                if float(r["residual_mean_sign_agreement_trained_others"]) >= 1.0
                and float(r["residual_ratio_min_trained_others"]) >= 1.45
                and float(r["residual_cos_mean_trained_others"]) >= 0.25
                and float(r["delta_cos_mean_trained_others"]) >= 0.14
            ]
            key = lambda r: (
                float(r["more_faded_display_score"]),
                float(r["residual_ratio_min_trained_others"]),
                float(r["residual_cos_mean_trained_others"]),
            )
        elif mode == "maximum_fade":
            filtered = [
                r
                for r in pool
                if float(r["residual_mean_sign_agreement_trained_others"]) >= 1.0
                and float(r["residual_ratio_min_trained_others"]) >= 2.0
                and float(r["delta_cos_mean_trained_others"]) >= 0.10
            ]
            key = lambda r: (
                float(r["residual_ratio_min_trained_others"]),
                float(r["residual_cos_mean_trained_others"]),
                float(r["delta_cos_mean_trained_others"]),
            )
        else:
            raise ValueError(mode)
        sign_same = [r for r in pool if float(r["residual_mean_sign_agreement_trained_others"]) >= 1.0]
        chosen_pool = filtered or sign_same or pool
        selected.append(sorted(chosen_pool, key=key, reverse=True)[0])
    return selected


def add_existing_relaxed_records(
    records: dict[tuple[str, str], dict[str, Any]],
    scored: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge old relaxed/source arrays so clean same-sign fallbacks are available."""
    existing_keys = {(str(r["dataset_id"]), int(r["sample_index"])) for r in scored}
    source_records = relaxed.load_source_records()
    ranks = relaxed.rank_lookup()
    added: list[dict[str, Any]] = []
    for dataset_id, sample_index, model in sorted(source_records):
        if model != "loss3":
            continue
        sample_key = (str(dataset_id), int(sample_index))
        if sample_key in existing_keys:
            continue
        if any((dataset_id, sample_index, m) not in source_records for m in MODEL_ORDER):
            continue
        rank_row = dict(ranks[(dataset_id, int(sample_index))])
        sid = f"morefadeexisting_{int(rank_row['dataset_rank']):02d}_{family_slug(str(dataset_id))}_idx{int(sample_index)}_rank{int(rank_row['rank_within_dataset_best_is_1']):02d}"
        rank_row["candidate_sample_id"] = sid
        for m in MODEL_ORDER:
            records[(sid, m)] = source_records[(dataset_id, int(sample_index), m)]
        added.append(score_row(rank_row, records))
        existing_keys.add(sample_key)
    return scored + added


def emit_variant(
    args: argparse.Namespace,
    variant: str,
    rows: list[dict[str, Any]],
    records: dict[tuple[str, str], dict[str, Any]],
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
    plot_records: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    for row in rows:
        old_sid = row["candidate_sample_id"]
        new_sid = sample_id(row, variant)
        sample = heat.SampleSpec(new_sid, "generalization", str(row["dataset_id"]), int(row["sample_index"]))
        samples.append(sample)
        score_rows.append({**row, "sample_id": new_sid})
        for model in MODEL_ORDER:
            src = records[(old_sid, model)]
            arrays = {key: np.asarray(src[key]) for key in ["x0", "delta", "x_adv", "model_output", "solver_output", "model_minus_solver", "attack_loss_history", "attack_loss_gain_history"]}
            npz_path = array_dir / f"{new_sid}_{heat.slug(model)}_attack_fields.npz"
            np.savez_compressed(npz_path, **arrays)
            plot_records.append(
                {
                    **arrays,
                    "sample_id": new_sid,
                    "split": "generalization",
                    "dataset_id": str(row["dataset_id"]),
                    "sample_index": int(row["sample_index"]),
                    "model": model,
                    "attack_loss_gain": float(src["attack_loss_gain"]),
                    "adv_relative_l2_model_vs_solver": float(src["adv_relative_l2_model_vs_solver"]),
                }
            )
            summary_rows.append(
                {
                    "sample_id": new_sid,
                    "split": "generalization",
                    "dataset_id": str(row["dataset_id"]),
                    "sample_index": int(row["sample_index"]),
                    "model": model,
                    "model_source": src["model_source"],
                    "checkpoint": src["checkpoint"],
                    "attack_steps": src["attack_steps"],
                    "epsilon_fraction": src["epsilon_fraction"],
                    "clean_loss_before_attack": src["clean_loss_before_attack"],
                    "adv_loss_after_attack": src["adv_loss_after_attack"],
                    "attack_loss_gain": src["attack_loss_gain"],
                    "attack_loss_gain_relative": src["attack_loss_gain_relative"],
                    "adv_relative_l2_model_vs_solver": src["adv_relative_l2_model_vs_solver"],
                    "adv_rmse_model_vs_solver": src["adv_rmse_model_vs_solver"],
                    "delta_l0_fraction": src["delta_l0_fraction"],
                    "delta_l2_rms": src["delta_l2_rms"],
                    "elapsed_seconds": src["elapsed_seconds"],
                    "npz_path": heat.rel(npz_path),
                }
            )

    ranges = heat.shared_ranges(plot_records)
    (out_dir / "shared_color_ranges.json").write_text(json.dumps(ranges, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "column_color_ranges_applied.json").write_text(
        json.dumps(
            {
                "initial condition": ranges["coefficient"],
                "delta": ranges["delta"],
                "initial + delta": ranges["coefficient"],
                "model output": ranges["output_solver_shared"],
                "solver output": ranges["output_solver_shared"],
                "model - solver": ranges["model_minus_solver"],
                "note": "Shared by column; model output and solver output use the same range.",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    with (out_dir / "selected_samples.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "split", "dataset_id", "sample_index"])
        writer.writeheader()
        writer.writerows([asdict(s) for s in samples])
    heat.write_csv(out_dir / "summary.csv", summary_rows)
    write_csv(out_dir / "visual_selection_scores.csv", score_rows)
    figs = [heat.plot_sample(sample, plot_records, ranges, viz_dir) for sample in samples]
    report_args = argparse.Namespace(tag=f"{args.tag}_{variant}", attack_steps=int(args.attack_steps), epsilon_fraction=float(args.epsilon_fraction))
    heat.write_report(out_dir, viz_dir, summary_rows, figs, samples, report_args)
    with (out_dir / "README.md").open("a", encoding="utf-8") as f:
        f.write("\n## More-Faded Selection Criteria\n\n")
        f.write("This folder was selected from a wider candidate pool than the previous relaxed folders. The score emphasizes a larger Loss3 residual fade in `model - solver`, with residual mean-sign agreement, residual cosine, and delta cosine recorded in `visual_selection_scores.csv`.\n")
    return {"variant": variant, "out_dir": heat.rel(out_dir), "viz_dir": heat.rel(viz_dir), "figures": [heat.rel(p) for p in figs]}


def run(args: argparse.Namespace) -> None:
    args.out_root = args.out_root.resolve()
    args.viz_root = args.viz_root.resolve()
    rank_rows = all_rank_rows(int(args.samples_per_dataset))
    candidates = choose_candidates(rank_rows, int(args.candidates_per_dataset))
    write_csv(RANKED_DIR / "more_faded_candidate_rank_rows.csv", candidates)
    candidate_out = run_candidate_screen(args, candidates)
    records = load_records(candidate_out)
    scored = [score_row(row, records) for row in candidates]
    scored = add_existing_relaxed_records(records, scored)
    scored = sorted(
        scored,
        key=lambda r: (
            int(r["dataset_rank"]),
            -float(r["more_faded_display_score"]),
            -float(r["residual_ratio_min_trained_others"]),
        ),
    )
    write_csv(RANKED_DIR / "more_faded_candidate_visual_scores.csv", scored)

    balanced = choose_selected(scored, mode="balanced")
    maximum = choose_selected(scored, mode="maximum_fade")
    outputs = [
        emit_variant(args, "loss3_more_faded_recommended", balanced, records),
        emit_variant(args, "loss3_more_faded_maxfade_signsame", maximum, records),
    ]
    manifest = {
        "tag": args.tag,
        "source_ranked_dir": heat.rel(RANKED_DIR),
        "candidate_out_dir": heat.rel(candidate_out),
        "samples_per_dataset": int(args.samples_per_dataset),
        "candidates_per_dataset": int(args.candidates_per_dataset),
        "selection_note": "balanced prioritizes visible fade while requiring residual sign agreement and mild shape similarity; maxfade prioritizes strongest residual fade with looser shape constraints.",
        "outputs": outputs,
    }
    (RANKED_DIR / "more_faded_heatmap_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=SOURCE_TAG)
    parser.add_argument("--generalization-root", type=Path, default=PROJECT_ROOT / "generalization_datasets_darcy_binary_loss3targeted_20260611")
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--viz-root", type=Path, default=VIZ_ROOT)
    parser.add_argument("--samples-per-dataset", type=int, default=50)
    parser.add_argument("--candidates-per-dataset", type=int, default=15)
    parser.add_argument("--attack-steps", type=int, default=50)
    parser.add_argument("--epsilon-fraction", type=float, default=0.025)
    parser.add_argument("--plot-batch-size", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
