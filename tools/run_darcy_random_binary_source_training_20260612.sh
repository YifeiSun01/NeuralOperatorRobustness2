#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-adv_robust/bin/python}"
TAG="${TAG:-20260612_random_binary_source_2000}"
GENERALIZATION_ROOT="${GENERALIZATION_ROOT:-generalization_datasets_darcy_binary_loss3targeted_20260611}"
OUT_ROOT="${OUT_ROOT:-adversarial_training_runs}"
TRAIN_MAX="${TRAIN_MAX:-64}"
DARCY_BATCH="${DARCY_BATCH:-64}"
OPT_BATCH="${OPT_BATCH:-32}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-10}"
MAX_GENERALIZATION_EVAL="${MAX_GENERALIZATION_EVAL:-50}"
ATTACK_PROBE_SAMPLES="${ATTACK_PROBE_SAMPLES:-5}"
ATTACK_PROBE_EVERY="${ATTACK_PROBE_EVERY:-1}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-200}"
EPOCHS_FIXED_Y="${EPOCHS_FIXED_Y:-2000}"
EPOCHS_SOLVER_Y="${EPOCHS_SOLVER_Y:-2000}"
RANDOM_KERNELS="${RANDOM_KERNELS:-gaussian,matern,highpass,bandpass,mixed}"
RANDOM_ALPHA_VALUES="${RANDOM_ALPHA_VALUES:-1.2,2.2,3.2,4.2,5.2}"
RANDOM_LENGTHSCALE_MIN="${RANDOM_LENGTHSCALE_MIN:-0.035}"
RANDOM_LENGTHSCALE_MAX="${RANDOM_LENGTHSCALE_MAX:-0.30}"
RANDOM_MIN_FLIP_FRACTION="${RANDOM_MIN_FLIP_FRACTION:-0.005}"
RANDOM_MAX_FLIP_FRACTION="${RANDOM_MAX_FLIP_FRACTION:-0.05}"
UPLOAD_TO_R2="${UPLOAD_TO_R2:-0}"
UPLOAD_RUN_DIRS="${UPLOAD_RUN_DIRS:-1}"
R2_BASE_PREFIX="${R2_PREFIX:-machine-sync/NeuralOperatorRobustness2-selected}"

MODES=(random-binary-fixed-y random-binary-solver-y)
declare -A EPOCHS_BY_MODE=(
  [random-binary-fixed-y]="$EPOCHS_FIXED_Y"
  [random-binary-solver-y]="$EPOCHS_SOLVER_Y"
)
declare -A RUN_DIRS=()

LOG_ROOT="$OUT_ROOT/darcy_random_binary_source_${TAG}_logs"
mkdir -p "$LOG_ROOT"
DRIVER_LOG="$LOG_ROOT/driver.log"

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$DRIVER_LOG"
}

run_dir_for() {
  local mode="$1"
  local epochs="${EPOCHS_BY_MODE[$mode]}"
  local safe_mode="${mode//-/_}"
  echo "$OUT_ROOT/darcy_binary_${safe_mode}_${epochs}ep_full50_${TAG}"
}

upload_path() {
  local path="$1"
  if [[ "$UPLOAD_TO_R2" != "1" ]]; then
    return 0
  fi
  if [[ ! -e "$path" ]]; then
    log "skip missing upload path: $path"
    return 0
  fi
  log "upload to R2: $path"
  R2_PREFIX="$R2_BASE_PREFIX/darcy_random_binary_source_${TAG}" \
  R2_UPLOAD_LOG_ROOT="$LOG_ROOT/r2_upload_logs" \
    "$ROOT/tools/upload_path_to_r2_20260525.sh" "$path"
}

log "Darcy random binary source launcher start: tag=$TAG"
log "Random source params: kernels=$RANDOM_KERNELS alpha=$RANDOM_ALPHA_VALUES lengthscale=[$RANDOM_LENGTHSCALE_MIN,$RANDOM_LENGTHSCALE_MAX] flip_fraction=[$RANDOM_MIN_FLIP_FRACTION,$RANDOM_MAX_FLIP_FRACTION]"

for mode in "${MODES[@]}"; do
  epochs="${EPOCHS_BY_MODE[$mode]}"
  run_dir="$(run_dir_for "$mode")"
  RUN_DIRS[$mode]="$run_dir"
  summary="$run_dir/darcy/summary.json"
  if [[ -f "$summary" ]]; then
    log "skip completed $mode run: $summary"
    continue
  fi
  if [[ -d "$run_dir" ]]; then
    log "ERROR: incomplete run directory exists for $mode: $run_dir"
    exit 6
  fi

  cmd=(
    "$PYTHON" tools/adversarial_training.py
    --tasks darcy
    --generalization-root "$GENERALIZATION_ROOT"
    --output-root "$OUT_ROOT"
    --run-name "$(basename "$run_dir")"
    --epochs "$epochs"
    --darcy-train-max "$TRAIN_MAX"
    --darcy-batch-size "$DARCY_BATCH"
    --darcy-optimizer-batch-size "$OPT_BATCH"
    --eval-max-samples "$EVAL_MAX_SAMPLES"
    --max-generalization-eval "$MAX_GENERALIZATION_EVAL"
    --training-data-mode "$mode"
    --label-mode solver
    --checkpoint-every-epochs "$CHECKPOINT_EVERY"
    --attack-probe-samples "$ATTACK_PROBE_SAMPLES"
    --attack-probe-every-n-epochs "$ATTACK_PROBE_EVERY"
    --attack-probe-save-targets
    --darcy-random-source-kernels "$RANDOM_KERNELS"
    --darcy-random-source-alpha-values "$RANDOM_ALPHA_VALUES"
    --darcy-random-source-lengthscale-min "$RANDOM_LENGTHSCALE_MIN"
    --darcy-random-source-lengthscale-max "$RANDOM_LENGTHSCALE_MAX"
    --darcy-random-source-min-flip-fraction "$RANDOM_MIN_FLIP_FRACTION"
    --darcy-random-source-max-flip-fraction "$RANDOM_MAX_FLIP_FRACTION"
  )
  log "start $mode epochs=$epochs run=$run_dir"
  printf '[command] %q ' "${cmd[@]}" | tee -a "$DRIVER_LOG"
  printf '\n' | tee -a "$DRIVER_LOG"
  "${cmd[@]}" 2>&1 | tee "$LOG_ROOT/$(basename "$run_dir").log"
  log "done $mode epochs=$epochs run=$run_dir"
done

if [[ "$UPLOAD_TO_R2" == "1" ]]; then
  if [[ "$UPLOAD_RUN_DIRS" == "1" ]]; then
    for mode in "${MODES[@]}"; do
      upload_path "${RUN_DIRS[$mode]}"
    done
  fi
  upload_path "$LOG_ROOT"
  upload_path "tools/run_darcy_random_binary_source_training_20260612.sh"
fi

log "Darcy random binary source launcher complete: tag=$TAG"
