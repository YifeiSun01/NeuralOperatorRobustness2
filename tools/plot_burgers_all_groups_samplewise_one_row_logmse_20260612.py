#!/usr/bin/env python3
"""Render one-row sample-wise Burgers overlays with log-MSE loss axes for all current groups."""
from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
COMBINED_PLOTTER = REPO / "tools" / "plot_burgers_round03_p2q2_combined_attack_panels.py"
SAMPLEWISE_PLOTTER = REPO / "tools" / "plot_burgers_wideparam_retrain_p2q2_samplewise_overlay_20260612.py"
TRACE_BASE = REPO / "forensics" / "burgers_wideparam_loss123_retrain_round00_p2q2_diverse_multi_sample_attack_visuals_batched_20260612"
VIS_BASE = REPO / "visualizations" / "burgers_wideparam_loss123_retrain_round00_p2q2_comparison_dense_diverse_multi_sample_batched_20260612" / "comparison_dense"
BUNDLE_BASE = REPO / "visualizations" / "burgers_wideparam_loss123_retrain_comparison_dense_image_only_bundle_20260612" / "comparison_dense"
DISPLAY = {
    "baseline": "Baseline model",
    "loss1": "Wideparam retrain loss1 epoch8000",
    "loss2": "Wideparam retrain loss2 epoch2000",
    "loss3": "Wideparam retrain loss3 epoch1000",
}

GROUP_SPECS = [
    (f"group{i:02d}", f"wideparam_loss3targeted_round00_group{i:02d}_p2q2") for i in range(5)
] + [
    ("group05_loss3_best", "wideparam_loss3targeted_round00_group05_loss3best_p2q2"),
]


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def render_group(group_name: str, output_prefix: str) -> tuple[Path, Path] | None:
    trace_root = TRACE_BASE / group_name
    if not trace_root.exists():
        return None
    vis_dir = VIS_BASE / group_name
    bundle_dir = BUNDLE_BASE / group_name
    vis_dir.mkdir(parents=True, exist_ok=True)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    combined = import_module(COMBINED_PLOTTER, f"combined_{group_name}_logmse")
    samplewise = import_module(SAMPLEWISE_PLOTTER, f"samplewise_{group_name}_logmse")
    combined.TRACE_ROOT = trace_root
    combined.OUT_DIR = vis_dir
    combined.OUTPUT_PREFIX = output_prefix
    combined.DISPLAY.update(DISPLAY)
    samplewise.DISPLAY.update(DISPLAY)

    clean, manifest, models, limits, _loss_ylim, _final_idx = combined.build_data()
    out_path = vis_dir / f"{output_prefix}_baseline_loss1_loss2_loss3_before_after_overlay_four_column_samplewise_loss_one_row_log_mse.png"
    samplewise.render_overlay_samplewise(
        combined,
        out_path,
        clean,
        manifest,
        models,
        limits,
        bottom_layout="one_row",
        loss_yscale="log",
    )
    bundle_path = bundle_dir / out_path.name
    shutil.copy2(out_path, bundle_path)
    return out_path, bundle_path


def main() -> int:
    outputs = []
    for group_name, output_prefix in GROUP_SPECS:
        result = render_group(group_name, output_prefix)
        if result is None:
            print(f"[skip missing] {group_name}")
            continue
        outputs.append(result)
    print("Generated log-MSE one-row overlays:")
    for out_path, bundle_path in outputs:
        print(out_path)
        print(bundle_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
