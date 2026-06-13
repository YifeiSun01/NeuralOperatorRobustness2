#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/workspace/NeuralOperatorRobustness2}"
PY="${PY:-/workspace/adv_robust/bin/python}"
OUT_ROOT="${OUT_ROOT:-$ROOT/adversarial_training_runs}"
EVAL_OUT_ROOT="${EVAL_OUT_ROOT:-$ROOT/forensics/burgers_random_field_checkpoint_series_4000_6000_8000_20260613}"
LOG_ROOT="${LOG_ROOT:-$OUT_ROOT/burgers_random_field_continue_eval_series_20260613_logs}"

RUN_TRAINING="${RUN_TRAINING:-1}"
RUN_EVALUATION="${RUN_EVALUATION:-1}"
SMOKE="${SMOKE:-0}"

mkdir -p "$LOG_ROOT"
cd "$ROOT"

if [[ "$RUN_TRAINING" == "1" ]]; then
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] starting continuation training" | tee -a "$LOG_ROOT/driver.log"
  SMOKE="$SMOKE" "$ROOT/tools/run_burgers_random_field_continue_to_8000_20260613.sh" > "$LOG_ROOT/continuation.log" 2>&1
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] continuation training finished" | tee -a "$LOG_ROOT/driver.log"
fi

if [[ "$RUN_EVALUATION" != "1" ]]; then
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] evaluation disabled" | tee -a "$LOG_ROOT/driver.log"
  exit 0
fi

if [[ "$SMOKE" == "1" ]]; then
  START_EPOCH="${START_EPOCH:-2000}"
  TARGET="${SMOKE_TARGET_EPOCHS:-$((START_EPOCH + 1))}"
  STAMP="${SMOKE_STAMP:-}"
  if [[ -z "$STAMP" ]]; then
    CLEAN_CKPT="$(find "$OUT_ROOT" -path "*burgers_wideparam_random_field_clean_y_smoke_resume_${START_EPOCH}_to_${TARGET}_20260613_*/burgers/checkpoints/burgers_epoch${TARGET}_step$(printf "%06d" "$TARGET").pt" | sort | tail -n 1)"
    SOLVER_CKPT="$(find "$OUT_ROOT" -path "*burgers_wideparam_random_field_solver_y_smoke_resume_${START_EPOCH}_to_${TARGET}_20260613_*/burgers/checkpoints/burgers_epoch${TARGET}_step$(printf "%06d" "$TARGET").pt" | sort | tail -n 1)"
  else
    CLEAN_CKPT="$OUT_ROOT/burgers_wideparam_random_field_clean_y_smoke_resume_${START_EPOCH}_to_${TARGET}_20260613_${STAMP}/burgers/checkpoints/burgers_epoch${TARGET}_step$(printf "%06d" "$TARGET").pt"
    SOLVER_CKPT="$OUT_ROOT/burgers_wideparam_random_field_solver_y_smoke_resume_${START_EPOCH}_to_${TARGET}_20260613_${STAMP}/burgers/checkpoints/burgers_epoch${TARGET}_step$(printf "%06d" "$TARGET").pt"
  fi
  EVAL_OUT_ROOT="${SMOKE_EVAL_OUT_ROOT:-$ROOT/forensics/burgers_random_field_checkpoint_series_smoke_20260613}"
  CLEAN_MAX="${SMOKE_CLEAN_MAX_SAMPLES:-4}"
  ATTACK_MAX="${SMOKE_ATTACK_MAX_SAMPLES:-4}"
  ATTACK_STEPS="${SMOKE_ATTACK_STEPS:-1}"
  ATTACK_BATCH="${SMOKE_ATTACK_BATCH_SIZE:-2}"
  SVD_MAX="${SMOKE_SVD_MAX_SAMPLES:-1}"
  SVD_TOPK="${SMOKE_SVD_TOP_K:-2}"
else
  CLEAN_CKPT="$OUT_ROOT/burgers_wideparam_random_field_clean_y_4000ep_continue_20260613/burgers/checkpoints/burgers_epoch4000_step004000.pt"
  SOLVER_CKPT="$OUT_ROOT/burgers_wideparam_random_field_solver_y_4000ep_continue_20260613/burgers/checkpoints/burgers_epoch4000_step004000.pt"
  CLEAN_CKPT_6000="$OUT_ROOT/burgers_wideparam_random_field_clean_y_6000ep_continue_20260613/burgers/checkpoints/burgers_epoch6000_step006000.pt"
  SOLVER_CKPT_6000="$OUT_ROOT/burgers_wideparam_random_field_solver_y_6000ep_continue_20260613/burgers/checkpoints/burgers_epoch6000_step006000.pt"
  CLEAN_CKPT_8000="$OUT_ROOT/burgers_wideparam_random_field_clean_y_8000ep_continue_20260613/burgers/checkpoints/burgers_epoch8000_step008000.pt"
  SOLVER_CKPT_8000="$OUT_ROOT/burgers_wideparam_random_field_solver_y_8000ep_continue_20260613/burgers/checkpoints/burgers_epoch8000_step008000.pt"
  CLEAN_MAX="${CLEAN_MAX_SAMPLES:-0}"
  ATTACK_MAX="${ATTACK_MAX_SAMPLES:-0}"
  ATTACK_STEPS="${ATTACK_STEPS:-20}"
  ATTACK_BATCH="${ATTACK_BATCH_SIZE:-500}"
  SVD_MAX="${SVD_MAX_SAMPLES:-25}"
  SVD_TOPK="${SVD_TOP_K:-20}"
fi

for required in "$PY" "$ROOT/tools/evaluate_burgers_random_field_checkpoint_series_20260613.py" "$CLEAN_CKPT" "$SOLVER_CKPT"; do
  if [[ ! -e "$required" ]]; then
    echo "[missing] $required" >&2
    exit 1
  fi
done

MODEL_ARGS=(
  --model-spec "random_clean_y_4000=$CLEAN_CKPT"
  --model-spec "random_solver_y_4000=$SOLVER_CKPT"
)

if [[ "$SMOKE" != "1" ]]; then
  for required in "$CLEAN_CKPT_6000" "$SOLVER_CKPT_6000" "$CLEAN_CKPT_8000" "$SOLVER_CKPT_8000"; do
    if [[ ! -e "$required" ]]; then
      echo "[missing] $required" >&2
      exit 1
    fi
  done
  MODEL_ARGS+=(
    --model-spec "random_clean_y_6000=$CLEAN_CKPT_6000"
    --model-spec "random_solver_y_6000=$SOLVER_CKPT_6000"
    --model-spec "random_clean_y_8000=$CLEAN_CKPT_8000"
    --model-spec "random_solver_y_8000=$SOLVER_CKPT_8000"
  )
fi

STAGE_ARGS=()
if [[ -n "${EVAL_STAGES:-}" ]]; then
  IFS=',' read -r -a stages <<< "$EVAL_STAGES"
  for stage in "${stages[@]}"; do
    stage="${stage// /}"
    [[ -z "$stage" ]] && continue
    STAGE_ARGS+=(--stage "$stage")
  done
fi

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] starting evaluation out=$EVAL_OUT_ROOT" | tee -a "$LOG_ROOT/driver.log"
"$PY" "$ROOT/tools/evaluate_burgers_random_field_checkpoint_series_20260613.py" \
  --out-root "$EVAL_OUT_ROOT" \
  "${MODEL_ARGS[@]}" \
  "${STAGE_ARGS[@]}" \
  --clean-max-samples "$CLEAN_MAX" \
  --attack-max-samples "$ATTACK_MAX" \
  --attack-steps "$ATTACK_STEPS" \
  --attack-batch-size "$ATTACK_BATCH" \
  --svd-max-samples "$SVD_MAX" \
  --svd-method topk \
  --svd-top-k "$SVD_TOPK" \
  > "$LOG_ROOT/evaluation.log" 2>&1

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] full continuation/evaluation workflow finished" | tee -a "$LOG_ROOT/driver.log"
