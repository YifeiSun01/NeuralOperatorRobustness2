#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="$ROOT/adv_robust/bin/python"
GEN_ROOT="$ROOT/generalization_datasets_burgers_loss3_aligned_search/round_01"
OUT_ROOT="$ROOT/adversarial_training_runs"
LOG_ROOT="$ROOT/adversarial_training_runs/burgers_loss3_aligned_round01_long_training_20260604_logs"
mkdir -p "$LOG_ROOT"

COMMON=(
  "$PY" "$ROOT/tools/adversarial_training.py"
  --tasks burgers
  --generalization-root "$GEN_ROOT"
  --output-root "$OUT_ROOT"
  --device cuda
  --seed 20260601
  --checkpoint-every-epochs 100
  --training-data-mode adv-only
  --label-mode solver
  --epsilon-bucket-count 5
  --attack-probe-samples 5
  --attack-probe-every-n-epochs 1
  --attack-probe-save-targets
  --burgers-attack-method fast_replace_l2
  --burgers-require-p2q2
  --burgers-attack-steps 5
  --burgers-batch-size 480
  --burgers-optimizer-batch-size 32
  --burgers-epsilon-fraction 0.06
  --burgers-eps-jitter-low 0.75
  --burgers-eps-jitter-high 1.25
  --burgers-alpha-ratio 1.0
  --burgers-alpha-jitter-low 0.75
  --burgers-alpha-jitter-high 1.25
  --burgers-random-start-fraction 1e-6
  --eval-max-samples 0
  --max-generalization-eval 50
)

run_one() {
  local loss="$1"
  local epochs="$2"
  local run_name="burgers_loss3_aligned_round01_${loss}_${epochs}ep_20260604"
  local log="$LOG_ROOT/${run_name}.log"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] start ${run_name}" | tee -a "$LOG_ROOT/driver.log"
  "${COMMON[@]}" \
    --run-name "$run_name" \
    --epochs "$epochs" \
    --burgers-attack-loss-objective "$loss" \
    > "$log" 2>&1
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] done ${run_name}" | tee -a "$LOG_ROOT/driver.log"
}

run_one loss1 1000
run_one loss2 500
run_one loss3 500

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] all done" | tee -a "$LOG_ROOT/driver.log"
