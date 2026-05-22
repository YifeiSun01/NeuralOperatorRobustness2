#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON_BIN="${PYTHON_BIN:-adv_robust/bin/python}"
CHECKPOINT="${CHECKPOINT:-2D_NS_FNO2d_recurrent/saved_models/2D/modes64_modes64_width60_epochs500_Tin10_T10_recurrent_pytorch_20260521_090136_UTC/checkpoints/best.pt}"

"${PYTHON_BIN}" 2D_NS_FNO2d_recurrent/visualizations/visualize_recurrent_test_predictions.py \
  --checkpoint "${CHECKPOINT}" \
  --indices "${INDICES:-}" \
  --num-samples "${NUM_SAMPLES:-5}" \
  --seed "${SEED:-20260521}" \
  --out-dir "${OUT_DIR:-2D_NS_FNO2d_recurrent/visualizations/fno2d_recurrent_test_predictions_20260521}"
