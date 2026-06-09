#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/adv_robust/bin/python}"
RUN_NAME="${RUN_NAME:-ns2d_loss3_advtrain_chunk_bs6_steps10_20260608}"
GENERALIZATION_ROOT="${GENERALIZATION_ROOT:-${ROOT}/generalization_datasets_rmse_1p5_3x_all_ns50}"

# V100-SXM2-32GB production default from the 2026-05-30 NS2D adversarial
# training records: chunk remat batch 6 passed, batch 8/10 OOMed. Do not use
# A100/B200 larger-batch evidence for this machine without rerunning the probe.
"${PYTHON_BIN}" "${ROOT}/tools/adversarial_training.py" \
  --tasks ns2d \
  --generalization-root "${GENERALIZATION_ROOT}" \
  --output-root "${ROOT}/adversarial_training_runs" \
  --run-name "${RUN_NAME}" \
  --device cuda \
  --seed "${SEED:-20260608}" \
  --epochs "${EPOCHS:-500}" \
  --checkpoint-every-epochs "${CHECKPOINT_EVERY_EPOCHS:-50}" \
  --training-data-mode "${TRAINING_DATA_MODE:-adv-only}" \
  --label-mode solver \
  --epsilon-bucket-count "${EPSILON_BUCKET_COUNT:-5}" \
  --attack-probe-samples "${ATTACK_PROBE_SAMPLES:-5}" \
  --attack-probe-every-n-epochs "${ATTACK_PROBE_EVERY_N_EPOCHS:-1}" \
  --attack-probe-save-targets \
  --ns2d-attack-method fast_add_linf \
  --ns2d-attack-steps "${NS2D_ATTACK_STEPS:-10}" \
  --ns2d-batch-size "${NS2D_BATCH_SIZE:-6}" \
  --ns2d-optimizer-batch-size "${NS2D_OPTIMIZER_BATCH_SIZE:-1}" \
  --ns2d-epsilon-fraction "${NS2D_EPSILON_FRACTION:-0.035}" \
  --ns2d-alpha-ratio "${NS2D_ALPHA_RATIO:-0.2}" \
  --ns2d-solver-remat chunk \
  --ns2d-solver-remat-chunk-steps "${NS2D_SOLVER_REMAT_CHUNK_STEPS:-20}" \
  --eval-max-samples "${EVAL_MAX_SAMPLES:-0}" \
  --max-generalization-eval "${MAX_GENERALIZATION_EVAL:-50}"
