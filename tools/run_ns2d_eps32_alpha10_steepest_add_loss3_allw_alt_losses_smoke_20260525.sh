#!/usr/bin/env bash
set -euo pipefail

cd /workspace/NeuralOperatorRobustness2

export RUN_TAG="${RUN_TAG:-smoke_eps32_alpha10_steepest_add_loss3_allw_altmetrics_b10_s5_$(date -u +%Y%m%d_%H%M%S)_UTC}"
export BASE_OUT_ROOT="${BASE_OUT_ROOT:-2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/${RUN_TAG}}"
export TEST_PATH="${TEST_PATH:-${BASE_OUT_ROOT}/synthetic_ns2d_smoke_test.pt}"

# Keep the smoke test faithful to the real batch shape: one batch of 10 samples.
export INDICES="${INDICES:-0,1,2,3,4,5,6,7,8,9}"
export ATTACK_BATCH_SIZE="${ATTACK_BATCH_SIZE:-10}"
export STEPS="${STEPS:-5}"

# Smoke test defaults match the real run: only alternative loss3 metrics, no qnorm baseline.
export LOSS3_METRICS="${LOSS3_METRICS:-dists ms_ssim scattering2d affine_dists local_warp_dists}"
export LOSS3_MODE_SPEC="${LOSS3_MODE_SPEC:-all_w}"

# Disable common expensive recording/evaluation by default so the smoke test isolates loss overhead.
# Override these to 1 if you want to time the full logging path too.
export TRUE_LOSS_EVERY="${TRUE_LOSS_EVERY:-0}"
export RECORD_FINAL_STATE_OUTPUTS="${RECORD_FINAL_STATE_OUTPUTS:-0}"
export RECORD_STEP_SAMPLE_OUTPUTS="${RECORD_STEP_SAMPLE_OUTPUTS:-0}"
export RECORD_STEP_SAMPLE_GRADIENTS="${RECORD_STEP_SAMPLE_GRADIENTS:-0}"

mkdir -p "$(dirname "$TEST_PATH")"
if [[ ! -f "$TEST_PATH" ]]; then
  "${PYTHON_BIN:-adv_robust/bin/python}" - <<'PYDATA'
import os
from pathlib import Path
import torch
path = Path(os.environ["TEST_PATH"])
path.parent.mkdir(parents=True, exist_ok=True)
torch.manual_seed(12345)
# Small-amplitude synthetic initial states for smoke timing only.
x = 0.1 * torch.randn(10, 256, 256, dtype=torch.float32)
torch.save({"x": x}, path)
print(f"wrote_synthetic_test_path={path}")
PYDATA
fi

bash tools/run_ns2d_eps32_alpha10_steepest_add_loss3_allw_alt_losses_20260525.sh

"${PYTHON_BIN:-adv_robust/bin/python}" tools/summarize_ns2d_loss3_smoke_timing.py "$BASE_OUT_ROOT" --out "$BASE_OUT_ROOT/smoke_timing_summary.csv"
