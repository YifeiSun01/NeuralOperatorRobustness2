#!/usr/bin/env python3
"""Render the group05 loss3-best one-row sample-wise overlay with log-MSE loss axes."""
from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
COMBINED_PLOTTER = REPO / "tools" / "plot_burgers_round03_p2q2_combined_attack_panels.py"
SAMPLEWISE_PLOTTER = REPO / "tools" / "plot_burgers_wideparam_retrain_p2q2_samplewise_overlay_20260612.py"
TRACE_ROOT = REPO / "forensics" / "burgers_wideparam_loss123_retrain_round00_p2q2_diverse_multi_sample_attack_visuals_batched_20260612" / "group05_loss3_best"
VIS_DIR = REPO / "visualizations" / "burgers_wideparam_loss123_retrain_round00_p2q2_comparison_dense_diverse_multi_sample_batched_20260612" / "comparison_dense" / "group05_loss3_best"
BUNDLE_DIR = REPO / "visualizations" / "burgers_wideparam_loss123_retrain_comparison_dense_image_only_bundle_20260612" / "comparison_dense" / "group05_loss3_best"
OUTPUT_PREFIX = "wideparam_loss3targeted_round00_group05_loss3best_p2q2"
DISPLAY = {
    "baseline": "Baseline model",
    "loss1": "Wideparam retrain loss1 epoch8000",
    "loss2": "Wideparam retrain loss2 epoch2000",
    "loss3": "Wideparam retrain loss3 epoch1000",
}


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    mod = import_module(COMBINED_PLOTTER, "combined_group05_logmse")
    sample_mod = import_module(SAMPLEWISE_PLOTTER, "samplewise_group05_logmse")
    mod.TRACE_ROOT = TRACE_ROOT
    mod.OUT_DIR = VIS_DIR
    mod.OUTPUT_PREFIX = OUTPUT_PREFIX
    mod.DISPLAY.update(DISPLAY)
    sample_mod.DISPLAY.update(DISPLAY)
    clean, manifest, models, limits, _loss_ylim, _final_idx = mod.build_data()
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = VIS_DIR / f"{OUTPUT_PREFIX}_baseline_loss1_loss2_loss3_before_after_overlay_four_column_samplewise_loss_one_row_log_mse.png"
    sample_mod.render_overlay_samplewise(
        mod,
        out_path,
        clean,
        manifest,
        models,
        limits,
        bottom_layout="one_row",
        loss_yscale="log",
    )
    bundle_path = BUNDLE_DIR / out_path.name
    shutil.copy2(out_path, bundle_path)
    print(out_path)
    print(bundle_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
