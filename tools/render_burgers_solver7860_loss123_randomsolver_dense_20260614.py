#!/usr/bin/env python3
"""Render loss1/loss2/loss3/random_solver dense Burgers panels from existing traces."""
from __future__ import annotations

import argparse
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
TRACE_KEYS = ["step", "x_adv", "delta", "model", "solver", "diff", "loss", "delta_rms"]
MODEL_ORDER = ["baseline", "loss1", "loss2", "loss3", "random_clean_y", "random_solver_y"]
VARIANT_KEY = "loss123_random_solver"


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


def load_group(group_dir: Path) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, dict[str, np.ndarray]]]:
    manifest_path = group_dir / "sample_manifest.json"
    trace_path = group_dir / "six_model_attack_traces.npz"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    if not trace_path.exists():
        raise FileNotFoundError(trace_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with np.load(trace_path) as z:
        clean = np.asarray(z["clean"])
        traces = {
            model_key: {
                key: np.asarray(z[f"{model_key}_{key}"])
                for key in TRACE_KEYS
            }
            for model_key in MODEL_ORDER
        }
    return clean, manifest, traces


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def append_unique(values: list[str], new_values: list[str]) -> list[str]:
    out = list(values)
    for value in new_values:
        if value not in out:
            out.append(value)
    return out


def render_group(plotter, labels, group_dir: Path, trace_root: Path, vis_root: Path, bundle_root: Path) -> dict[str, str]:
    group_name = group_dir.name
    group_id = int(group_name.replace("group", ""))
    clean, manifest, traces = load_group(group_dir)
    model_order = plotter.MODEL_VARIANTS[VARIANT_KEY]
    out_dir = vis_root / "comparison_dense" / group_name
    bundle_dir = bundle_root / "comparison_dense" / group_name
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    images: dict[str, str] = {}
    bundle_images: dict[str, str] = {}
    for loss_scale in ["log", "linear"]:
        out_name = plotter.dense_output_name(group_id, VARIANT_KEY, loss_scale)
        out_path = out_dir / out_name
        suffix = " / group05 loss3-best" if group_name == "group05" else ""
        plotter.render_one_row(
            labels,
            out_path,
            clean,
            manifest,
            traces,
            loss_scale=loss_scale,
            model_order=model_order,
            variant_label=plotter.variant_title(VARIANT_KEY, model_order) + suffix,
        )
        bundle_path = bundle_dir / out_name
        shutil.copy2(out_path, bundle_path)
        key = f"{VARIANT_KEY}_{loss_scale}"
        images[key] = rel(out_path)
        bundle_images[key] = rel(bundle_path)

    summary_path = group_dir / "six_model_summary.json"
    summary = load_json(summary_path)
    summary.setdefault("group_id", group_id)
    summary.setdefault("trace_root", rel(group_dir))
    summary.setdefault("visualization_dir", rel(out_dir))
    summary.setdefault("bundle_dir", rel(bundle_dir))
    summary.setdefault("models", MODEL_ORDER)
    summary.setdefault("images", {})
    summary.setdefault("bundle_images", {})
    summary["images"].update(images)
    summary["bundle_images"].update(bundle_images)
    summary.setdefault("model_variants", {})
    summary["model_variants"][VARIANT_KEY] = model_order
    summary.setdefault("model_headers", {model_key: plotter.model_header_label(model_key) for model_key in MODEL_ORDER})
    summary.setdefault("loss_scales", ["log", "linear"])
    summary["loss123_random_solver_note"] = "Rendered from existing traces with baseline and random_clean_y removed; random_solver_y is retained."
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    return {"images": images, "bundle_images": bundle_images}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-root", type=Path, default=TRACE_ROOT)
    parser.add_argument("--vis-root", type=Path, default=VIS_ROOT)
    parser.add_argument("--bundle-root", type=Path, default=BUNDLE_ROOT)
    parser.add_argument("--groups", nargs="*", default=None, help="Optional group names, e.g. group00 group05.")
    args = parser.parse_args()

    plotter = import_module(PLOTTER_PATH, "burgers_solver7860_loss123_randomsolver_plotter")
    labels = plotter.load_module("combined_labels_for_solver7860_loss123_randomsolver", plotter.COMBINED_PLOTTER)
    if VARIANT_KEY not in plotter.MODEL_VARIANTS:
        raise RuntimeError(f"{VARIANT_KEY} is missing from plotter.MODEL_VARIANTS")

    group_dirs = [p for p in sorted(args.trace_root.glob("group[0-9][0-9]")) if p.is_dir()]
    if args.groups:
        wanted = set(args.groups)
        group_dirs = [p for p in group_dirs if p.name in wanted]
    if not group_dirs:
        raise RuntimeError(f"No group directories found under {args.trace_root}")

    all_images: list[str] = []
    all_bundle_images: list[str] = []
    groups: dict[str, dict[str, dict[str, str]]] = {}
    for group_dir in group_dirs:
        rendered = render_group(plotter, labels, group_dir, args.trace_root, args.vis_root, args.bundle_root)
        all_images.extend(rendered["images"].values())
        all_bundle_images.extend(rendered["bundle_images"].values())
        groups[group_dir.name] = rendered

    top_summary_path = args.trace_root / "six_model_summary.json"
    top_summary = load_json(top_summary_path)
    top_summary.setdefault("trace_root", rel(args.trace_root))
    top_summary.setdefault("visualization_root", rel(args.vis_root))
    top_summary.setdefault("bundle_root", rel(args.bundle_root))
    top_summary.setdefault("models", MODEL_ORDER)
    top_summary.setdefault("model_variants", {})
    top_summary["model_variants"][VARIANT_KEY] = plotter.MODEL_VARIANTS[VARIANT_KEY]
    top_summary.setdefault("model_headers", {model_key: plotter.model_header_label(model_key) for model_key in MODEL_ORDER})
    top_summary.setdefault("loss_scales", ["log", "linear"])
    top_summary["outputs"] = append_unique(list(top_summary.get("outputs", [])), all_images)
    top_summary["bundle_outputs"] = append_unique(list(top_summary.get("bundle_outputs", [])), all_bundle_images)
    top_summary["loss123_random_solver_note"] = "Rendered from existing traces with baseline and random_clean_y removed; random_solver_y is retained."
    top_summary["loss123_random_solver_groups"] = groups
    top_summary_path.write_text(json.dumps(top_summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "variant": VARIANT_KEY,
        "groups": sorted(groups),
        "images": len(all_images),
        "bundle_images": len(all_bundle_images),
        "bundle_root": rel(args.bundle_root),
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
