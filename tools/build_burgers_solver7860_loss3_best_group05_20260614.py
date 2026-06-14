#!/usr/bin/env python3
"""Build group05 from existing solver7860 dense traces where loss3 wins most clearly."""
from __future__ import annotations

import csv
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[1]
PLOTTER_PATH = REPO / "tools" / "plot_burgers_wideparam_random_field_six_model_one_row_20260613.py"

TRACE_ROOT = REPO / "forensics" / "burgers_wideparam_loss123_randomsolver7860_clean8000_round00_p2q2_six_model_visuals_20260614"
VIS_ROOT = REPO / "visualizations" / "burgers_wideparam_loss123_randomsolver7860_clean8000_round00_p2q2_six_model_visuals_20260614"
BUNDLE_ROOT = REPO / "visualizations" / "burgers_wideparam_loss123_randomsolver7860_clean8000_comparison_dense_image_only_bundle_20260614"

GROUP_ID = 5
GROUP_NAME = "group05"
MODEL_ORDER = ["baseline", "loss1", "loss2", "loss3", "random_clean_y", "random_solver_y"]
TRACE_KEYS = ["step", "x_adv", "delta", "model", "solver", "diff", "loss", "delta_rms"]


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def rel(path: Path) -> str:
    return str(path.relative_to(REPO))


def read_manifest(group_dir: Path) -> list[dict[str, Any]]:
    return json.loads((group_dir / "sample_manifest.json").read_text(encoding="utf-8"))


def collect_candidates() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group_dir in sorted(TRACE_ROOT.glob("group[0-4][0-9]")):
        trace_path = group_dir / "six_model_attack_traces.npz"
        if not trace_path.exists():
            continue
        manifest = read_manifest(group_dir)
        with np.load(trace_path) as z:
            for sample_idx, item in enumerate(manifest):
                finals = {model: float(z[f"{model}_loss"][-1, sample_idx]) for model in MODEL_ORDER}
                initials = {model: float(z[f"{model}_loss"][0, sample_idx]) for model in MODEL_ORDER}
                increases = {model: finals[model] - initials[model] for model in MODEL_ORDER}
                other_models = [model for model in MODEL_ORDER if model != "loss3"]
                best_other_final_model = min(other_models, key=lambda model: finals[model])
                best_other_increase_model = min(other_models, key=lambda model: increases[model])
                best_other_final = finals[best_other_final_model]
                best_other_increase = increases[best_other_increase_model]
                final_margin = best_other_final - finals["loss3"]
                increase_margin = best_other_increase - increases["loss3"]
                row: dict[str, Any] = {
                    "group": group_dir.name,
                    "sample_id": item["sample_id"],
                    "source_sample_index": int(sample_idx),
                    "split": item.get("split", ""),
                    "dataset_id": item.get("dataset_id", ""),
                    "source_dataset_id": item.get("source_dataset_id", item.get("dataset_id", "")),
                    "index": int(item["index"]),
                    "final_winner": min(MODEL_ORDER, key=lambda model: finals[model]),
                    "increase_winner": min(MODEL_ORDER, key=lambda model: increases[model]),
                    "best_other_final_model": best_other_final_model,
                    "best_other_final": best_other_final,
                    "loss3_final_margin": final_margin,
                    "loss3_final_rel_margin": final_margin / best_other_final if best_other_final else float("nan"),
                    "best_other_increase_model": best_other_increase_model,
                    "best_other_increase": best_other_increase,
                    "loss3_increase_margin": increase_margin,
                    "loss3_increase_rel_margin": increase_margin / best_other_increase if best_other_increase else float("nan"),
                }
                for model in MODEL_ORDER:
                    row[f"{model}_initial_loss"] = initials[model]
                    row[f"{model}_final_loss"] = finals[model]
                    row[f"{model}_loss_increase"] = increases[model]
                row["loss3_advantage_score"] = float(row["loss3_final_rel_margin"] + row["loss3_increase_rel_margin"])
                rows.append(row)
    return rows


def select_group05(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    strict = [
        row
        for row in candidates
        if row["final_winner"] == "loss3"
        and row["increase_winner"] == "loss3"
        and row["loss3_final_margin"] > 0.0
        and row["loss3_increase_margin"] > 0.0
    ]
    tests = sorted(
        [row for row in strict if row["split"] == "test"],
        key=lambda row: row["loss3_advantage_score"],
        reverse=True,
    )
    gens = sorted(
        [row for row in strict if row["split"] == "generalization"],
        key=lambda row: row["loss3_advantage_score"],
        reverse=True,
    )
    if len(tests) < 1 or len(gens) < 5:
        raise RuntimeError(f"not enough strict loss3-win candidates: test={len(tests)} generalization={len(gens)}")
    selected = tests[:1] + gens[:5]
    for new_idx, row in enumerate(selected, start=1):
        row["new_sample_id"] = f"S{new_idx}"
        row["new_position"] = new_idx - 1
    return selected


def select_sample_from_array(arr: np.ndarray, sample_idx: int) -> np.ndarray:
    if arr.ndim >= 2 and arr.shape[1] > sample_idx:
        return arr[:, sample_idx : sample_idx + 1, ...]
    if arr.ndim >= 1 and arr.shape[0] > sample_idx:
        return arr[sample_idx : sample_idx + 1, ...]
    return arr.copy()


def build_trace_payload(selected: list[dict[str, Any]]) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    payload_parts: dict[str, list[np.ndarray]] = {"clean": []}
    scalar_or_step_values: dict[str, np.ndarray] = {}
    manifests_by_group = {row["group"]: read_manifest(TRACE_ROOT / row["group"]) for row in selected}
    new_manifest: list[dict[str, Any]] = []

    for row in selected:
        group_dir = TRACE_ROOT / row["group"]
        sample_idx = int(row["source_sample_index"])
        source_item = dict(manifests_by_group[row["group"]][sample_idx])
        source_item["sample_id"] = row["new_sample_id"]
        source_item["selected_from_group"] = row["group"]
        source_item["selected_from_sample_id"] = row["sample_id"]
        source_item["selected_by"] = "loss3_strict_final_and_increase_win_largest_relative_margin"
        for key in [
            "final_winner",
            "increase_winner",
            "best_other_final_model",
            "best_other_final",
            "loss3_final_margin",
            "loss3_final_rel_margin",
            "best_other_increase_model",
            "best_other_increase",
            "loss3_increase_margin",
            "loss3_increase_rel_margin",
            "loss3_advantage_score",
        ]:
            source_item[key] = row[key]
        for model in MODEL_ORDER:
            source_item[f"{model}_final_loss"] = row[f"{model}_final_loss"]
            source_item[f"{model}_loss_increase"] = row[f"{model}_loss_increase"]
        new_manifest.append(source_item)

        with np.load(group_dir / "six_model_attack_traces.npz") as z:
            payload_parts["clean"].append(z["clean"][sample_idx : sample_idx + 1, ...])
            for model in MODEL_ORDER:
                for key in TRACE_KEYS:
                    npz_key = f"{model}_{key}"
                    arr = z[npz_key]
                    if key == "step":
                        scalar_or_step_values[npz_key] = arr.copy()
                    else:
                        payload_parts.setdefault(npz_key, []).append(select_sample_from_array(arr, sample_idx))

    payload: dict[str, np.ndarray] = {}
    for key, pieces in payload_parts.items():
        if key == "clean":
            payload[key] = np.concatenate(pieces, axis=0)
        else:
            payload[key] = np.concatenate(pieces, axis=1)
    payload.update(scalar_or_step_values)
    return payload, new_manifest


def traces_from_payload(payload: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]]]:
    clean = payload["clean"]
    traces = {
        model: {key: payload[f"{model}_{key}"] for key in TRACE_KEYS}
        for model in MODEL_ORDER
    }
    return clean, traces


def write_loss_csv(path: Path, traces: dict[str, dict[str, np.ndarray]], manifest: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "step", "sample_id", "split", "dataset_id", "source_dataset_id", "index", "loss_mse", "delta_rms"])
        for model in MODEL_ORDER:
            tr = traces[model]
            for t, step in enumerate(tr["step"]):
                for sample_idx, item in enumerate(manifest):
                    writer.writerow([
                        model,
                        int(step),
                        item["sample_id"],
                        item.get("split", ""),
                        item.get("dataset_id", ""),
                        item.get("source_dataset_id", ""),
                        int(item["index"]),
                        float(tr["loss"][t, sample_idx]),
                        float(tr["delta_rms"][t, sample_idx]),
                    ])


def write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_group05(clean: np.ndarray, manifest: list[dict[str, Any]], traces: dict[str, dict[str, np.ndarray]]) -> dict[str, str]:
    plotter = import_module(PLOTTER_PATH, "burgers_solver7860_dense_group05_plotter")
    labels = plotter.load_module("combined_labels_for_solver7860_group05", plotter.COMBINED_PLOTTER)
    out_dir = VIS_ROOT / "comparison_dense" / GROUP_NAME
    bundle_dir = BUNDLE_ROOT / "comparison_dense" / GROUP_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, str] = {}
    for variant_key in ["all_models", "no_random_clean"]:
        model_order = plotter.MODEL_VARIANTS[variant_key]
        for loss_scale in ["log", "linear"]:
            out_name = plotter.dense_output_name(GROUP_ID, variant_key, loss_scale)
            out_path = out_dir / out_name
            plotter.render_one_row(
                labels,
                out_path,
                clean,
                manifest,
                traces,
                loss_scale=loss_scale,
                model_order=model_order,
                variant_label=plotter.variant_title(variant_key, model_order) + " / group05 loss3-best",
            )
            bundle_path = bundle_dir / out_name
            shutil.copy2(out_path, bundle_path)
            outputs[f"{variant_key}_{loss_scale}"] = rel(bundle_path)
    return outputs


def main() -> int:
    candidates = collect_candidates()
    selected = select_group05(candidates)
    payload, manifest = build_trace_payload(selected)
    clean, traces = traces_from_payload(payload)

    group_dir = TRACE_ROOT / GROUP_NAME
    group_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(group_dir / "six_model_attack_traces.npz", **payload)
    (group_dir / "sample_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_loss_csv(group_dir / "attack_loss_curves_all_six_models.csv", traces, manifest)
    write_table(group_dir / "source_group_loss3_advantage_ranking.csv", sorted(candidates, key=lambda row: row["loss3_advantage_score"], reverse=True))
    write_table(group_dir / "selected_samples.csv", selected)

    bundle_outputs = render_group05(clean, manifest, traces)
    vis_outputs = {
        key: value.replace(
            rel(BUNDLE_ROOT),
            rel(VIS_ROOT),
        )
        for key, value in bundle_outputs.items()
    }
    summary = {
        "group_id": GROUP_ID,
        "group_name": GROUP_NAME,
        "selection_criteria": (
            "Selected from existing group00-group04 dense traces. Candidates must have loss3 as the strict "
            "winner for both final attacked MSE and attack loss increase. Ranking score is the sum of loss3's "
            "relative final-loss margin and relative loss-increase margin versus the best non-loss3 model."
        ),
        "source_trace_root": rel(TRACE_ROOT),
        "trace_root": rel(group_dir),
        "visualization_dir": rel(VIS_ROOT / "comparison_dense" / GROUP_NAME),
        "bundle_dir": rel(BUNDLE_ROOT / "comparison_dense" / GROUP_NAME),
        "images": vis_outputs,
        "bundle_images": bundle_outputs,
        "models": MODEL_ORDER,
        "model_headers": {
            "baseline": "baseline e500",
            "loss1": "loss1 e8000",
            "loss2": "loss2 e2000",
            "loss3": "loss3 e1000",
            "random_clean_y": "random clean Y e8000",
            "random_solver_y": "random solver Y e7860",
        },
        "selected_samples": selected,
    }
    (group_dir / "six_model_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    top_summary_path = TRACE_ROOT / "six_model_summary.json"
    top_summary: dict[str, Any] = {}
    if top_summary_path.exists():
        top_summary = json.loads(top_summary_path.read_text(encoding="utf-8"))
    top_summary["group05_loss3_best"] = summary
    existing_outputs = list(top_summary.get("outputs", []))
    existing_bundle_outputs = list(top_summary.get("bundle_outputs", []))
    for value in vis_outputs.values():
        if value not in existing_outputs:
            existing_outputs.append(value)
    for value in bundle_outputs.values():
        if value not in existing_bundle_outputs:
            existing_bundle_outputs.append(value)
    top_summary["outputs"] = existing_outputs
    top_summary["bundle_outputs"] = existing_bundle_outputs
    top_summary_path.write_text(json.dumps(top_summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "group": GROUP_NAME,
        "selected": [
            {
                "new_sample_id": row["new_sample_id"],
                "source": f"{row['group']}/{row['sample_id']}",
                "split": row["split"],
                "source_dataset_id": row["source_dataset_id"],
                "loss3_final_loss": row["loss3_final_loss"],
                "best_other_final_model": row["best_other_final_model"],
                "best_other_final": row["best_other_final"],
                "loss3_final_rel_margin": row["loss3_final_rel_margin"],
                "loss3_loss_increase": row["loss3_loss_increase"],
                "best_other_increase_model": row["best_other_increase_model"],
                "best_other_increase": row["best_other_increase"],
                "loss3_increase_rel_margin": row["loss3_increase_rel_margin"],
            }
            for row in selected
        ],
        "bundle_pngs": list(bundle_outputs.values()),
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
