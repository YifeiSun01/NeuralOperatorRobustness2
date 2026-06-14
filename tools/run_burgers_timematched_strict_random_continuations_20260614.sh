#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/workspace/NeuralOperatorRobustness2}"
PY="${PY:-/workspace/adv_robust/bin/python}"
MODE="${MODE:-formal}" # smoke or formal
LOG_ROOT="${LOG_ROOT:-$ROOT/adversarial_training_runs/burgers_timematched_strict_random_continuations_20260614_logs}"

CONTINUE_SCRIPT="$ROOT/tools/run_burgers_random_field_continue_to_8000_20260613.sh"
CLEAN_CKPT="$ROOT/adversarial_training_runs/burgers_wideparam_random_field_clean_y_8000ep_continue_20260613/burgers/checkpoints/burgers_epoch8000_step008000.pt"
SOLVER_CKPT="$ROOT/adversarial_training_runs/burgers_wideparam_random_field_solver_y_6000ep_continue_20260613/burgers/checkpoints/burgers_epoch6000_step006000.pt"

# Target: existing loss3 logged work-clock, 7.427461557 hours.
# Latest local timing estimate from train_steps.csv:
# - random_clean_y needs about 13,100 epochs beyond epoch 8000.
# - random_solver_y needs about 1,860 epochs beyond epoch 6000.
CLEAN_TARGET_EPOCH="${CLEAN_TARGET_EPOCH:-21100}"
SOLVER_TARGET_EPOCH="${SOLVER_TARGET_EPOCH:-7860}"

mkdir -p "$LOG_ROOT"
cd "$ROOT"

for required in "$PY" "$CONTINUE_SCRIPT" "$CLEAN_CKPT" "$SOLVER_CKPT"; do
  if [[ ! -e "$required" ]]; then
    echo "[missing] $required" >&2
    exit 1
  fi
done

run_clean() {
  local smoke_flag="$1"
  local target="$2"
  local stamp="$3"
  PY="$PY" \
  SMOKE="$smoke_flag" \
  SMOKE_STAMP="$stamp" \
  START_EPOCH=8000 \
  TARGET_EPOCHS="$target" \
  RUN_RANDOM_CLEAN_Y=1 \
  RUN_RANDOM_SOLVER_Y=0 \
  BASE_CLEAN_CKPT="$CLEAN_CKPT" \
  LOG_ROOT="$LOG_ROOT" \
  bash "$CONTINUE_SCRIPT"
}

run_solver() {
  local smoke_flag="$1"
  local target="$2"
  local stamp="$3"
  PY="$PY" \
  SMOKE="$smoke_flag" \
  SMOKE_STAMP="$stamp" \
  START_EPOCH=6000 \
  TARGET_EPOCHS="$target" \
  RUN_RANDOM_CLEAN_Y=0 \
  RUN_RANDOM_SOLVER_Y=1 \
  BASE_SOLVER_CKPT="$SOLVER_CKPT" \
  LOG_ROOT="$LOG_ROOT" \
  bash "$CONTINUE_SCRIPT"
}

case "$MODE" in
  smoke)
    run_clean 1 8001 "stricttime_smoke_clean_20260614"
    run_solver 1 6001 "stricttime_smoke_solver_20260614"
    ;;
  formal)
    run_solver 0 "$SOLVER_TARGET_EPOCH" "unused_formal_solver"
    run_clean 0 "$CLEAN_TARGET_EPOCH" "unused_formal_clean"
    ;;
  *)
    echo "[bad MODE] expected smoke or formal, got $MODE" >&2
    exit 1
    ;;
esac

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] strict random continuation driver finished MODE=$MODE" | tee -a "$LOG_ROOT/driver.log"
