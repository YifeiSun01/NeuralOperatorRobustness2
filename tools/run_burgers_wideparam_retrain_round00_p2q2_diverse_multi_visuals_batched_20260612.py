#!/usr/bin/env python3
"""Render wide-parameter P2Q2 panels using the 20260611 retrained models."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASE_SCRIPT = REPO / "tools" / "run_burgers_wideparam_loss3targeted_round00_p2q2_diverse_multi_visuals_batched_20260611.py"


def load_base():
    spec = importlib.util.spec_from_file_location("wideparam_p2q2_base_20260611", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(BASE_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    mod = load_base()
    mod.TRACE_ROOT = REPO / "forensics" / "burgers_wideparam_loss123_retrain_round00_p2q2_diverse_multi_sample_attack_visuals_batched_20260612"
    mod.VIS_ROOT = REPO / "visualizations" / "burgers_wideparam_loss123_retrain_round00_p2q2_comparison_dense_diverse_multi_sample_batched_20260612"
    mod.MODEL_SPECS = {
        "baseline": {
            "label": "Baseline model",
            "checkpoint": mod.BASELINE_CKPT,
        },
        "loss1": {
            "label": "Wideparam retrain loss1 epoch8000",
            "checkpoint": REPO / "adversarial_training_runs/burgers_wideparam_loss1_8000ep_retrain_20260611/burgers/checkpoints/burgers_epoch8000_step008000.pt",
        },
        "loss2": {
            "label": "Wideparam retrain loss2 epoch2000",
            "checkpoint": REPO / "adversarial_training_runs/burgers_wideparam_loss2_2000ep_retrain_20260611/burgers/checkpoints/burgers_epoch2000_step002000.pt",
        },
        "loss3": {
            "label": "Wideparam retrain loss3 epoch1000",
            "checkpoint": REPO / "adversarial_training_runs/burgers_wideparam_loss3_1000ep_retrain_20260611/burgers/checkpoints/burgers_epoch1000_step001000.pt",
        },
    }
    return int(mod.main())


if __name__ == "__main__":
    raise SystemExit(main())
