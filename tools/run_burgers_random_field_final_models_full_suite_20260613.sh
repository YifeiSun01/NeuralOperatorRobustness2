#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/workspace/NeuralOperatorRobustness2}"
PY="${PY:-/venv/adv_robust/bin/python}"
OUT_ROOT="${OUT_ROOT:-$ROOT/forensics/burgers_random_field_final_models_full_suite_20260613}"
LOG_ROOT="${LOG_ROOT:-$ROOT/run_logs/burgers_random_field_final_models_full_suite_20260613}"

mkdir -p "$OUT_ROOT" "$LOG_ROOT"
cd "$ROOT"

cmd=(
  "$PY" "$ROOT/tools/evaluate_burgers_random_field_final_models_20260613.py"
  --out-root "$OUT_ROOT"
  --clean-batch-size "${CLEAN_BATCH_SIZE:-256}"
  --attack-steps "${ATTACK_STEPS:-20}"
  --attack-batch-size "${ATTACK_BATCH_SIZE:-500}"
  --attack-train-count "${ATTACK_TRAIN_COUNT:-50}"
  --svd-max-samples "${SVD_MAX_SAMPLES:-25}"
  --svd-method "${SVD_METHOD:-topk}"
  --svd-top-k "${SVD_TOP_K:-20}"
)

if [[ -n "${STAGES:-}" ]]; then
  IFS=',' read -r -a stage_array <<< "$STAGES"
  for stage in "${stage_array[@]}"; do
    cmd+=(--stage "$stage")
  done
fi

if [[ -n "${CLEAN_MAX_SAMPLES:-}" ]]; then
  cmd+=(--clean-max-samples "$CLEAN_MAX_SAMPLES")
fi

if [[ -n "${ATTACK_MAX_SAMPLES:-}" ]]; then
  cmd+=(--attack-max-samples "$ATTACK_MAX_SAMPLES")
fi

{
  printf '[%s] command:' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf ' %q' "${cmd[@]}"
  printf '\n'
} | tee -a "$LOG_ROOT/driver.log"

"${cmd[@]}" > "$LOG_ROOT/pipeline.log" 2>&1

printf '[%s] pipeline finished\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG_ROOT/driver.log"
