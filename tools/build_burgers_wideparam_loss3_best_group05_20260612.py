#!/usr/bin/env python3
"""Build a loss3-best composite group from the five wideparam retrain P2Q2 groups."""
from __future__ import annotations

import csv
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
TRACE_ROOT = REPO / "forensics" / "burgers_wideparam_loss123_retrain_round00_p2q2_diverse_multi_sample_attack_visuals_batched_20260612"
NEW_GROUP = TRACE_ROOT / "group05_loss3_best"
VIS_ROOT = REPO / "visualizations" / "burgers_wideparam_loss123_retrain_round00_p2q2_comparison_dense_diverse_multi_sample_batched_20260612"
NEW_VIS_DIR = VIS_ROOT / "comparison_dense" / "group05_loss3_best"
BUNDLE_ROOT = REPO / "visualizations" / "burgers_wideparam_loss123_retrain_comparison_dense_image_only_bundle_20260612"
NEW_BUNDLE_DIR = BUNDLE_ROOT / "comparison_dense" / "group05_loss3_best"
COMBINED_PLOTTER = REPO / "tools" / "plot_burgers_round03_p2q2_combined_attack_panels.py"
SAMPLEWISE_PLOTTER = REPO / "tools" / "plot_burgers_wideparam_retrain_p2q2_samplewise_overlay_20260612.py"
LOSS_ORDER = ["loss1", "loss2", "loss3"]
MODEL_ORDER = ["baseline", "loss1", "loss2", "loss3"]
DISPLAY = {
    "baseline": "Baseline model",
    "loss1": "Wideparam retrain loss1 epoch8000",
    "loss2": "Wideparam retrain loss2 epoch2000",
    "loss3": "Wideparam retrain loss3 epoch1000",
}
OUTPUT_PREFIX = "wideparam_loss3targeted_round00_group05_loss3best_p2q2"


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def collect_final_rows() -> pd.DataFrame:
    rows = []
    for group_dir in sorted(TRACE_ROOT.glob("group[0-9][0-9]")):
        csv_path = group_dir / "attack_loss_curves_all_models.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        final_step = int(df["step"].max())
        final = df[df["step"] == final_step]
        for sample_id, sdf in final.groupby("sample_id", sort=True):
            losses = {r.model: float(r.loss_mse) for r in sdf.itertuples()}
            winner = min(losses.items(), key=lambda kv: kv[1])[0]
            others = {k: v for k, v in losses.items() if k != "loss3"}
            best_other_model, best_other = min(others.items(), key=lambda kv: kv[1])
            loss3 = losses["loss3"]
            row0 = sdf.iloc[0]
            rows.append(
                {
                    "group": group_dir.name,
                    "sample_id": sample_id,
                    "source_sample_index": int(str(sample_id).lstrip("S")) - 1,
                    "split": row0["split"],
                    "dataset_id": row0["dataset_id"],
                    "source_dataset_id": row0["source_dataset_id"],
                    "index": int(row0["index"]),
                    "winner_final_mse": winner,
                    "loss3_final_mse": loss3,
                    "best_other_model": best_other_model,
                    "best_other_final_mse": best_other,
                    "loss3_abs_margin_vs_best_other": best_other - loss3,
                    "loss3_rel_margin_vs_best_other": (best_other - loss3) / best_other if best_other else np.nan,
                    **{f"{k}_final_mse": v for k, v in losses.items()},
                }
            )
    return pd.DataFrame(rows)


def select_samples(final_rows: pd.DataFrame) -> pd.DataFrame:
    strict = final_rows[final_rows["winner_final_mse"] == "loss3"].copy()
    tests = strict[strict["split"] == "test"].sort_values("loss3_rel_margin_vs_best_other", ascending=False)
    gens = strict[strict["split"] == "generalization"].sort_values("loss3_rel_margin_vs_best_other", ascending=False)
    if len(tests) < 1 or len(gens) < 5:
        raise RuntimeError(f"not enough strict loss3 samples: tests={len(tests)}, gens={len(gens)}")
    selected = pd.concat([tests.head(1), gens.head(5)], ignore_index=True)
    selected["new_sample_id"] = [f"S{i}" for i in range(1, len(selected) + 1)]
    selected["new_position"] = np.arange(len(selected), dtype=int)
    return selected


def read_manifest(group: str) -> list[dict[str, object]]:
    return json.loads((TRACE_ROOT / group / "sample_manifest.json").read_text(encoding="utf-8"))


def select_npz_array(arr: np.ndarray, indices: list[int]) -> np.ndarray:
    if arr.ndim >= 2 and arr.shape[1] == 6:
        return arr[:, indices, ...]
    if arr.ndim >= 1 and arr.shape[0] == 6:
        return arr[indices, ...]
    return arr.copy()


def build_group(selected: pd.DataFrame, final_rows: pd.DataFrame) -> None:
    NEW_GROUP.mkdir(parents=True, exist_ok=True)
    selection_records = selected.to_dict(orient="records")
    manifests_by_group = {g: read_manifest(g) for g in selected["group"].unique()}
    new_manifest = []
    for rec in selection_records:
        source_item = dict(manifests_by_group[rec["group"]][int(rec["source_sample_index"])])
        source_item["sample_id"] = rec["new_sample_id"]
        source_item["selected_from_group"] = rec["group"]
        source_item["selected_from_sample_id"] = rec["sample_id"]
        source_item["selected_by"] = "loss3_final_mse_relative_margin_vs_best_non_loss3"
        for key in [
            "winner_final_mse",
            "loss3_final_mse",
            "best_other_model",
            "best_other_final_mse",
            "loss3_abs_margin_vs_best_other",
            "loss3_rel_margin_vs_best_other",
            "baseline_final_mse",
            "loss1_final_mse",
            "loss2_final_mse",
            "loss3_final_mse",
        ]:
            source_item[key] = rec[key]
        new_manifest.append(source_item)
    (NEW_GROUP / "sample_manifest.json").write_text(json.dumps(new_manifest, indent=2), encoding="utf-8")

    # Build selected NPZ traces for each loss directory.
    for loss in LOSS_ORDER:
        out_dir = NEW_GROUP / loss
        out_dir.mkdir(parents=True, exist_ok=True)
        arrays_by_key: dict[str, list[np.ndarray]] = {}
        step_arrays: dict[str, np.ndarray] = {}
        for rec in selection_records:
            source_npz = TRACE_ROOT / rec["group"] / loss / "attack_traces.npz"
            z = np.load(source_npz)
            source_idx = int(rec["source_sample_index"])
            for key in z.files:
                arr = z[key]
                if arr.ndim >= 2 and arr.shape[1] == 6:
                    arrays_by_key.setdefault(key, []).append(arr[:, source_idx : source_idx + 1, ...])
                elif arr.ndim >= 1 and arr.shape[0] == 6:
                    arrays_by_key.setdefault(key, []).append(arr[source_idx : source_idx + 1, ...])
                else:
                    step_arrays.setdefault(key, arr.copy())
        out_arrays = {}
        for key, arrs in arrays_by_key.items():
            axis = 1 if arrs[0].ndim >= 2 and arrs[0].shape[1] == 1 else 0
            out_arrays[key] = np.concatenate(arrs, axis=axis)
        out_arrays.update(step_arrays)
        np.savez_compressed(out_dir / "attack_traces.npz", **out_arrays)

        baseline_loss = out_arrays["baseline_loss"]
        target_loss = out_arrays["target_loss"]
        baseline_delta = out_arrays["baseline_delta_rms"]
        target_delta = out_arrays["target_delta_rms"]
        summary = {
            "trace_npz": str(out_dir / "attack_traces.npz"),
            "baseline_initial_loss_mean": float(np.mean(baseline_loss[0])),
            "baseline_final_loss_mean": float(np.mean(baseline_loss[-1])),
            "target_initial_loss_mean": float(np.mean(target_loss[0])),
            "target_final_loss_mean": float(np.mean(target_loss[-1])),
            "baseline_final_delta_rms_mean": float(np.mean(baseline_delta[-1])),
            "target_final_delta_rms_mean": float(np.mean(target_delta[-1])),
            "selected_group_sample_slice": [0, len(selected)],
            "source_groups": sorted(set(selected["group"])),
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Build combined attack loss CSV in the same shape as original groups.
    out_rows = []
    for rec in selection_records:
        source_csv = TRACE_ROOT / rec["group"] / "attack_loss_curves_all_models.csv"
        with source_csv.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["sample_id"] != rec["sample_id"]:
                    continue
                row = dict(row)
                row["sample_id"] = rec["new_sample_id"]
                out_rows.append(row)
    out_csv = NEW_GROUP / "attack_loss_curves_all_models.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "step", "sample_id", "split", "dataset_id", "source_dataset_id", "index", "loss_mse", "delta_rms"])
        writer.writeheader()
        writer.writerows(out_rows)

    # Per-group strict win report and selected sample report.
    group_report = []
    for group, gdf in final_rows.groupby("group", sort=True):
        wins = int((gdf["winner_final_mse"] == "loss3").sum())
        group_report.append({"group": group, "loss3_final_mse_wins": wins, "total_samples": int(len(gdf)), "strict_all_loss3": bool(wins == len(gdf))})
    summary = {
        "criteria": "Select one test sample and five generalization samples where loss3 has the largest relative final attacked MSE margin over the best non-loss3 model. All selected samples require loss3 to be the strict final-MSE winner.",
        "source_trace_root": str(TRACE_ROOT),
        "new_group_root": str(NEW_GROUP),
        "group_strict_win_report": group_report,
        "selected_samples": selection_records,
        "mean_final_mse_selected": {},
    }
    selected_final = []
    for rec in selection_records:
        row = final_rows[(final_rows["group"] == rec["group"]) & (final_rows["sample_id"] == rec["sample_id"])].iloc[0]
        selected_final.append(row)
    sf = pd.DataFrame(selected_final)
    for model in MODEL_ORDER:
        summary["mean_final_mse_selected"][model] = float(sf[f"{model}_final_mse"].mean())
    (NEW_GROUP / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    selected.to_csv(NEW_GROUP / "selected_samples.csv", index=False)
    final_rows.to_csv(NEW_GROUP / "source_group_final_mse_ranking.csv", index=False)


def render_group() -> list[Path]:
    NEW_VIS_DIR.mkdir(parents=True, exist_ok=True)
    mod = import_module(COMBINED_PLOTTER, "combined_group05_loss3_best")
    mod.TRACE_ROOT = NEW_GROUP
    mod.OUT_DIR = NEW_VIS_DIR
    mod.OUTPUT_PREFIX = OUTPUT_PREFIX
    mod.DISPLAY.update(DISPLAY)
    mod.TITLE_PREFIX = "Burgers wideparam retrain loss3-best composite group05"
    mod.FRAME_SUBTITLE = "Composite selected from five P2Q2 groups; one test plus five generalization samples where loss3 is the strict final-MSE winner."
    mod.OVERLAY_SUBTITLE = "Dashed curves are before perturbation; solid/darker curves are after 50 saved attack steps."
    clean, manifest, models, limits, loss_ylim, final_idx = mod.build_data()
    outputs = []
    specs = [
        (f"{OUTPUT_PREFIX}_baseline_loss1_loss2_loss3_step000_before_perturbation_four_column.png", mod.MODEL_ORDER, 0, mod.render_frame),
        (f"{OUTPUT_PREFIX}_baseline_loss1_loss2_loss3_step100_after_perturbation_four_column.png", mod.MODEL_ORDER, final_idx, mod.render_frame),
    ]
    for name, model_keys, frame_idx, fn in specs:
        path = NEW_VIS_DIR / name
        fn(path, model_keys, frame_idx, clean, manifest, models, limits, loss_ylim)
        outputs.append(path)
    path = NEW_VIS_DIR / f"{OUTPUT_PREFIX}_baseline_loss1_loss2_loss3_before_after_overlay_four_column.png"
    mod.render_overlay(path, mod.MODEL_ORDER, clean, manifest, models, limits, loss_ylim)
    outputs.append(path)
    for loss in mod.LOSS_ORDER:
        path = NEW_VIS_DIR / f"{OUTPUT_PREFIX}_baseline_vs_{loss}_before_after_overlay_two_column.png"
        mod.render_overlay(path, ["baseline", loss], clean, manifest, models, limits, loss_ylim)
        outputs.append(path)

    sample_mod = import_module(SAMPLEWISE_PLOTTER, "samplewise_group05_loss3_best")
    sample_mod.DISPLAY.update(DISPLAY)
    # Reuse the already imported combined plotter instance by passing its data to the renderer.
    path = NEW_VIS_DIR / f"{OUTPUT_PREFIX}_baseline_loss1_loss2_loss3_before_after_overlay_four_column_samplewise_loss.png"
    sample_mod.render_overlay_samplewise(mod, path, clean, manifest, models, limits)
    outputs.append(path)
    path = NEW_VIS_DIR / f"{OUTPUT_PREFIX}_baseline_loss1_loss2_loss3_before_after_overlay_four_column_samplewise_loss_one_row.png"
    sample_mod.render_overlay_samplewise(mod, path, clean, manifest, models, limits, bottom_layout="one_row")
    outputs.append(path)

    NEW_BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    for path in outputs:
        shutil.copy2(path, NEW_BUNDLE_DIR / path.name)
    return outputs


def main() -> int:
    final_rows = collect_final_rows()
    selected = select_samples(final_rows)
    build_group(selected, final_rows)
    outputs = render_group()
    print("Strict loss3 final-MSE wins by original group:")
    for group, gdf in final_rows.groupby("group", sort=True):
        wins = int((gdf["winner_final_mse"] == "loss3").sum())
        print(f"  {group}: {wins}/{len(gdf)} strict_all_loss3={wins == len(gdf)}")
    print("\nSelected composite group05 samples:")
    print(selected[["new_sample_id", "group", "sample_id", "split", "source_dataset_id", "index", "loss3_final_mse", "best_other_model", "best_other_final_mse", "loss3_rel_margin_vs_best_other"]].to_string(index=False))
    print("\nGenerated outputs:")
    for path in outputs:
        print(path)
    print("\nCopied outputs to:")
    print(NEW_BUNDLE_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
