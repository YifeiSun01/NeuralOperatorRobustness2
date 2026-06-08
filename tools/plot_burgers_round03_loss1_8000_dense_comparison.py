#!/usr/bin/env python3
"""Plot Burgers round03 comparison with loss1 extended to epoch8000."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_SCRIPT = PROJECT_ROOT / "tools" / "plot_burgers_round03_loss123_final_extension_dense_comparison.py"


def load_base_module():
    spec = importlib.util.spec_from_file_location("round03_final_extension_plot_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    mod = load_base_module()
    run_root = PROJECT_ROOT / "adversarial_training_runs"
    mod.RUN_SEGMENTS["loss1"] = [
        ("original", run_root / "burgers_loss3_selective_round03_loss1_1000ep_long_20260605", False),
        ("extended_1", run_root / "burgers_loss3_selective_round03_loss1_continue1000to3000_20260605", True),
        ("extended_2", run_root / "burgers_loss3_selective_round03_loss1_continue3000to5000_20260606", True),
        ("extended_3", run_root / "burgers_loss3_selective_round03_loss1_continue5000to8000_20260607", True),
    ]
    mod.DEFAULT_DENSE_OUT_DIR = PROJECT_ROOT / "visualizations/burgers_loss3_selective_round03_long_training_comparison_dense_20260605"
    mod.DEFAULT_COMPACT_OUT_DIR = PROJECT_ROOT / "visualizations/burgers_loss3_selective_round03_long_training_comparison_20260605"
    mod.DEFAULT_REPORT = PROJECT_ROOT / "docs/burgers_loss3_selective_round03_loss1_8000_plot_report_20260607.md"
    mod.__doc__ = __doc__
    mod.main()


if __name__ == "__main__":
    main()
