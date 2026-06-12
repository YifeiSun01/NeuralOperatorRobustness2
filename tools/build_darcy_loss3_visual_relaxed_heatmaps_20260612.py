#!/usr/bin/env python3
"""Build visual-relaxed Darcy loss3 heatmap folders from existing attack arrays."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tools.plot_darcy_five_model_attack_heatmaps_20260612 as heat
import tools.plot_darcy_five_model_batch_ranked_heatmaps_20260612 as batch

TAG = "20260612_loss3attack50_five_models_ranked5_batch_polished"
RANKED_DIR = PROJECT_ROOT / "analysis_outputs" / f"darcy_five_model_batch_ranked_heatmaps_{TAG}"
OUT_ROOT = PROJECT_ROOT / "analysis_outputs"
VIZ_ROOT = PROJECT_ROOT / "visualizations"
MODEL_ORDER = [m.name for m in heat.MODELS]
TRAINED = ["loss1", "loss2", "loss3", "physics loss"]
TRAINED_OTHERS = ["loss1", "loss2", "physics loss"]
SOURCE_VARIANTS = [
    "loss3_top10pct_visualscreen_candidates",
    "index0",
    "loss3_best",
    "loss3_upper_quartile",
    "loss3_median",
    "loss3_lower_quartile",
    "loss3_worst",
]


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


def rank_lookup() -> dict[tuple[str, int], dict[str, Any]]:
    rows = read_csv(RANKED_DIR / "all_50step_batch_attack_results.csv")
    grouped: dict[tuple[str, int], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        grouped[(row["dataset_id"], int(float(row["sample_index"])) )][row["model"]] = row
    dataset_ids = json.loads((RANKED_DIR / "selected_dataset_ids.json").read_text(encoding="utf-8"))
    lookup = {}
    for dataset_rank, dataset_id in enumerate(dataset_ids, 1):
        scores = []
        for sample_index in range(50):
            by = grouped[(dataset_id, sample_index)]
            adv = {m: float(by[m]["adv_loss_after_attack"]) for m in MODEL_ORDER}
            loss3 = adv["loss3"]
            others = {m: v for m, v in adv.items() if m != "loss3"}
            best_model, best = min(others.items(), key=lambda kv: kv[1])
            scores.append({
                "dataset_rank": dataset_rank,
                "dataset_id": dataset_id,
                "sample_index": sample_index,
                "loss3_adv_loss": loss3,
                "best_other_model": best_model,
                "best_other_adv_loss": best,
                "loss3_margin_vs_best_other": best - loss3,
                "loss3_margin_vs_mean_other": float(np.mean(list(others.values())) - loss3),
                "baseline_adv_loss": adv["baseline"],
                "loss1_adv_loss": adv["loss1"],
                "loss2_adv_loss": adv["loss2"],
                "physics_loss_adv_loss": adv["physics loss"],
            })
        scores.sort(key=lambda r: (r["loss3_margin_vs_best_other"], -r["loss3_adv_loss"]), reverse=True)
        for rank, row in enumerate(scores, 1):
            row["rank_within_dataset_best_is_1"] = rank
            lookup[(dataset_id, int(row["sample_index"]))] = row
    return lookup


def load_source_records() -> dict[tuple[str, int, str], dict[str, Any]]:
    records = {}
    for variant in SOURCE_VARIANTS:
        out_dir = OUT_ROOT / f"darcy_five_model_attack_heatmaps_{TAG}_{variant}"
        if not (out_dir / "summary.csv").exists():
            continue
        for row in read_csv(out_dir / "summary.csv"):
            key = (row["dataset_id"], int(float(row["sample_index"])), row["model"])
            if key in records:
                continue
            npz_path = PROJECT_ROOT / row["npz_path"]
            if not npz_path.exists():
                continue
            z = np.load(npz_path)
            rec = {
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
            for akey in ["x0", "delta", "x_adv", "model_output", "solver_output", "model_minus_solver", "attack_loss_history", "attack_loss_gain_history"]:
                rec[akey] = np.asarray(z[akey])
            records[key] = rec
    return records


def cosine(a, b) -> float:
    af = np.ravel(np.asarray(a, dtype=np.float64)); bf = np.ravel(np.asarray(b, dtype=np.float64))
    return float(np.dot(af, bf) / max(np.linalg.norm(af) * np.linalg.norm(bf), 1e-20))


def jaccard(a, b) -> float:
    am = np.abs(a) > 1e-9; bm = np.abs(b) > 1e-9
    return float(np.logical_and(am,bm).sum() / max(1, np.logical_or(am,bm).sum()))


def score_dataset_sample(dataset_id: str, sample_index: int, records: dict[tuple[str, int, str], dict[str, Any]], ranks: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any] | None:
    if any((dataset_id, sample_index, m) not in records for m in MODEL_ORDER):
        return None
    loss3 = records[(dataset_id, sample_index, "loss3")]
    d3 = loss3["delta"]
    e3 = loss3["model_minus_solver"]
    e3_norm = float(np.linalg.norm(e3.ravel()))
    e3_mean = float(np.nanmean(e3))
    cosines=[]; jacs=[]; ratios=[]; signs=[]
    for model in TRAINED_OTHERS:
        rec=records[(dataset_id, sample_index, model)]
        cosines.append(cosine(d3, rec["delta"]))
        jacs.append(jaccard(d3, rec["delta"]))
        ratios.append(float(np.linalg.norm(rec["model_minus_solver"].ravel()) / max(e3_norm, 1e-20)))
        signs.append(float(np.sign(e3_mean) == np.sign(float(np.nanmean(rec["model_minus_solver"])))) )
    rank_row = ranks[(dataset_id, sample_index)]
    cos_mean=float(np.mean(cosines)); cos_min=float(np.min(cosines)); ratio_min=float(np.min(ratios)); ratio_mean=float(np.mean(ratios)); sign=float(np.mean(signs))
    pass_visual = bool(cos_mean >= 0.70 and cos_min >= 0.68 and ratio_min >= 1.16 and sign >= 1.0)
    score = (0.38 * min(1.0, (ratio_min - 1.0) / 0.45)
             + 0.28 * min(1.0, (cos_mean - 0.65) / 0.25)
             + 0.14 * min(1.0, (cos_min - 0.65) / 0.25)
             + 0.10 * min(1.0, float(rank_row["loss3_margin_vs_best_other"]) / 2.0e-6)
             + 0.10 * sign)
    return {
        **rank_row,
        "delta_cos_mean_trained_others": cos_mean,
        "delta_cos_min_trained_others": cos_min,
        "delta_jaccard_mean_trained_others": float(np.mean(jacs)),
        "residual_ratio_min_trained_others": ratio_min,
        "residual_ratio_mean_trained_others": ratio_mean,
        "residual_sign_agreement_trained_others": sign,
        "pass_visual_filter": pass_visual,
        "visual_display_score": float(score),
    }


def sample_id(row: dict[str, Any], variant: str) -> str:
    fam=batch.family_slug(str(row["dataset_id"]))
    return f"{variant}_{int(row['dataset_rank']):02d}_{fam}_idx{int(row['sample_index'])}_rank{int(row['rank_within_dataset_best_is_1']):02d}"


def emit(variant: str, rows: list[dict[str, Any]], records: dict[tuple[str,int,str],dict[str,Any]], force: bool) -> dict[str, Any]:
    out_dir = OUT_ROOT / f"darcy_five_model_attack_heatmaps_{TAG}_{variant}"
    viz_dir = VIZ_ROOT / f"darcy_five_model_attack_heatmaps_{TAG}_{variant}"
    if force:
        shutil.rmtree(out_dir, ignore_errors=True); shutil.rmtree(viz_dir, ignore_errors=True)
    array_dir=out_dir/'arrays'; array_dir.mkdir(parents=True, exist_ok=True); viz_dir.mkdir(parents=True, exist_ok=True)
    samples=[]; plot_records=[]; summary=[]; score_rows=[]
    for row in rows:
        sid=sample_id(row, variant)
        samples.append(heat.SampleSpec(sid, 'generalization', str(row['dataset_id']), int(row['sample_index'])))
        score_rows.append({**row, 'sample_id': sid})
        for model in MODEL_ORDER:
            src=records[(str(row['dataset_id']), int(row['sample_index']), model)]
            arrays={k:np.asarray(src[k]) for k in ['x0','delta','x_adv','model_output','solver_output','model_minus_solver','attack_loss_history','attack_loss_gain_history']}
            npz=array_dir/f'{sid}_{heat.slug(model)}_attack_fields.npz'
            np.savez_compressed(npz, **arrays)
            plot_records.append({**arrays,'sample_id':sid,'split':'generalization','dataset_id':str(row['dataset_id']),'sample_index':int(row['sample_index']),'model':model,'attack_loss_gain':float(src['attack_loss_gain']),'adv_relative_l2_model_vs_solver':float(src['adv_relative_l2_model_vs_solver'])})
            summary.append({'sample_id':sid,'split':'generalization','dataset_id':str(row['dataset_id']),'sample_index':int(row['sample_index']),'model':model,'model_source':src['model_source'],'checkpoint':src['checkpoint'],'attack_steps':src['attack_steps'],'epsilon_fraction':src['epsilon_fraction'],'clean_loss_before_attack':src['clean_loss_before_attack'],'adv_loss_after_attack':src['adv_loss_after_attack'],'attack_loss_gain':src['attack_loss_gain'],'attack_loss_gain_relative':src['attack_loss_gain_relative'],'adv_relative_l2_model_vs_solver':src['adv_relative_l2_model_vs_solver'],'adv_rmse_model_vs_solver':src['adv_rmse_model_vs_solver'],'delta_l0_fraction':src['delta_l0_fraction'],'delta_l2_rms':src['delta_l2_rms'],'elapsed_seconds':src['elapsed_seconds'],'npz_path':heat.rel(npz)})
    ranges=heat.shared_ranges(plot_records)
    (out_dir/'shared_color_ranges.json').write_text(json.dumps(ranges, indent=2, sort_keys=True), encoding='utf-8')
    (out_dir/'column_color_ranges_applied.json').write_text(json.dumps({'initial condition':ranges['coefficient'],'delta':ranges['delta'],'initial + delta':ranges['coefficient'],'model output':ranges['output_solver_shared'],'solver output':ranges['output_solver_shared'],'model - solver':ranges['model_minus_solver'],'note':'Shared by column; model output and solver output share range.'}, indent=2, sort_keys=True), encoding='utf-8')
    with (out_dir/'selected_samples.csv').open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=['sample_id','split','dataset_id','sample_index']); w.writeheader(); w.writerows([asdict(s) for s in samples])
    heat.write_csv(out_dir/'summary.csv', summary)
    write_csv(out_dir/'visual_selection_scores.csv', score_rows)
    figs=[heat.plot_sample(s, plot_records, ranges, viz_dir) for s in samples]
    args=argparse.Namespace(tag=f'{TAG}_{variant}', attack_steps=50, epsilon_fraction=0.025)
    heat.write_report(out_dir, viz_dir, summary, figs, samples, args)
    with (out_dir/'README.md').open('a', encoding='utf-8') as f:
        f.write('\n## Visual Relaxed Selection\n\n')
        f.write('These samples are selected by a hard visual filter: delta cosine mean >= 0.70, delta cosine min >= 0.68, residual ratio min >= 1.16, and residual sign agreement = 1 across loss1/loss2/physics. Within that filter, higher Loss3 margin and clearer residual fade are preferred.\n')
    return {'variant':variant,'out_dir':heat.rel(out_dir),'viz_dir':heat.rel(viz_dir),'figures':[heat.rel(p) for p in figs]}


def run(force: bool) -> None:
    ranks=rank_lookup(); records=load_source_records()
    scored=[]
    for (dataset_id, sample_index, model) in list(records):
        if model!='loss3': continue
        row=score_dataset_sample(dataset_id, sample_index, records, ranks)
        if row: scored.append(row)
    write_csv(RANKED_DIR/'visual_relaxed_candidate_scores_from_existing_arrays.csv', sorted(scored, key=lambda r:(r['dataset_rank'], -r['visual_display_score'])))
    outputs=[]
    used=set()
    for variant, pick_offset in [('loss3_top5pct_visual_relaxed',0), ('loss3_top10pct_visual_relaxed',1)]:
        rows=[]
        for dataset_rank in sorted({int(r['dataset_rank']) for r in scored}):
            pool=[r for r in scored if int(r['dataset_rank'])==dataset_rank and r['pass_visual_filter']]
            if not pool:
                pool=[r for r in scored if int(r['dataset_rank'])==dataset_rank]
            pool=sorted(pool, key=lambda r:(r['loss3_margin_vs_best_other'], r['visual_display_score'], -r['rank_within_dataset_best_is_1']), reverse=True)
            pool2=[r for r in pool if (r['dataset_id'], int(r['sample_index'])) not in used]
            chosen=(pool2 or pool)[min(pick_offset, len(pool2 or pool)-1)]
            rows.append(chosen); used.add((chosen['dataset_id'], int(chosen['sample_index'])))
        outputs.append(emit(variant, rows, records, force))
    manifest={'tag':TAG,'source_variants':SOURCE_VARIANTS,'selection':'visual relaxed hard filter, then highest available Loss3 margin','outputs':outputs}
    (RANKED_DIR/'visual_relaxed_heatmap_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))


def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--force', action='store_true')
    args=ap.parse_args(); run(args.force)

if __name__ == '__main__':
    main()
