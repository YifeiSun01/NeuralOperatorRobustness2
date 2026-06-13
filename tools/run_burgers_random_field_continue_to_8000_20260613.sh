#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/workspace/NeuralOperatorRobustness2}"
PY="${PY:-/workspace/adv_robust/bin/python}"
GEN_ROOT="${GEN_ROOT:-$ROOT/generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00}"
OUT_ROOT="${OUT_ROOT:-$ROOT/adversarial_training_runs}"
LOG_ROOT="${LOG_ROOT:-$OUT_ROOT/burgers_random_field_continue_to_8000_20260613_logs}"

BASE_CLEAN_CKPT="${BASE_CLEAN_CKPT:-$OUT_ROOT/burgers_wideparam_random_field_clean_y_2000ep_20260612/burgers/checkpoints/burgers_epoch2000_step002000.pt}"
BASE_SOLVER_CKPT="${BASE_SOLVER_CKPT:-$OUT_ROOT/burgers_wideparam_random_field_solver_y_2000ep_20260612/burgers/checkpoints/burgers_epoch2000_step002000.pt}"
START_EPOCH="${START_EPOCH:-2000}"
TARGET_EPOCHS="${TARGET_EPOCHS:-4000,6000,8000}"

RUN_RANDOM_CLEAN_Y="${RUN_RANDOM_CLEAN_Y:-1}"
RUN_RANDOM_SOLVER_Y="${RUN_RANDOM_SOLVER_Y:-1}"
SMOKE="${SMOKE:-0}"
DRY_RUN="${DRY_RUN:-0}"

SEED="${SEED:-20260612}"
RANDOM_BATCH="${RANDOM_BATCH:-1350}"
RANDOM_OPT_BATCH="${RANDOM_OPT_BATCH:-32}"
BURGERS_SOLVER_REMAT="${BURGERS_SOLVER_REMAT:-chunk}"
BURGERS_SOLVER_REMAT_CHUNK_STEPS="${BURGERS_SOLVER_REMAT_CHUNK_STEPS:-50}"
CHECKPOINT_EVERY_EPOCHS="${CHECKPOINT_EVERY_EPOCHS:-2000}"
CHECKPOINT_WALL_HOURS="${CHECKPOINT_WALL_HOURS:-0.5,1,1.5,2,2.5,3,4,5,6,7,8,9,10,12,14,16,18,20,24}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-0}"
MAX_GENERALIZATION_EVAL="${MAX_GENERALIZATION_EVAL:-50}"
ATTACK_PROBE_SAMPLES="${ATTACK_PROBE_SAMPLES:-5}"
ATTACK_PROBE_EVERY_N_EPOCHS="${ATTACK_PROBE_EVERY_N_EPOCHS:-1}"
EPSILON_BUCKET_COUNT="${EPSILON_BUCKET_COUNT:-5}"

RANDOM_FIELD_FAMILIES="${RANDOM_FIELD_FAMILIES:-gaussian,matern}"
RANDOM_FIELD_GAUSSIAN_CORR="${RANDOM_FIELD_GAUSSIAN_CORR:-0.015,0.03,0.06,0.12,0.24}"
RANDOM_FIELD_MATERN_CORR="${RANDOM_FIELD_MATERN_CORR:-0.015,0.03,0.06,0.12,0.24}"
RANDOM_FIELD_MATERN_NU="${RANDOM_FIELD_MATERN_NU:-1.2,2.2,3.2,4.2,5.2}"
RANDOM_FIELD_DOMAIN_EXTENT="${RANDOM_FIELD_DOMAIN_EXTENT:-2.0}"

mkdir -p "$LOG_ROOT"
cd "$ROOT"

for required in "$PY" "$GEN_ROOT/burgers" "$ROOT/tools/adversarial_training.py"; do
  if [[ ! -e "$required" ]]; then
    echo "[missing] $required" >&2
    exit 1
  fi
done

if [[ ! -f "$BASE_CLEAN_CKPT" ]]; then
  echo "[missing] BASE_CLEAN_CKPT=$BASE_CLEAN_CKPT" >&2
  exit 1
fi
if [[ ! -f "$BASE_SOLVER_CKPT" ]]; then
  echo "[missing] BASE_SOLVER_CKPT=$BASE_SOLVER_CKPT" >&2
  exit 1
fi

if [[ "$SMOKE" == "1" ]]; then
  TARGET_EPOCHS="${SMOKE_TARGET_EPOCHS:-$((START_EPOCH + 1))}"
  RANDOM_BATCH="${SMOKE_RANDOM_BATCH:-8}"
  RANDOM_OPT_BATCH="${SMOKE_RANDOM_OPT_BATCH:-4}"
  CHECKPOINT_EVERY_EPOCHS=1
  CHECKPOINT_WALL_HOURS="${SMOKE_CHECKPOINT_WALL_HOURS:-}"
  EVAL_MAX_SAMPLES="${SMOKE_EVAL_MAX_SAMPLES:-4}"
  MAX_GENERALIZATION_EVAL="${SMOKE_MAX_GENERALIZATION_EVAL:-1}"
  ATTACK_PROBE_SAMPLES="${SMOKE_ATTACK_PROBE_SAMPLES:-1}"
  ATTACK_PROBE_EVERY_N_EPOCHS=1
  EPSILON_BUCKET_COUNT="${SMOKE_EPSILON_BUCKET_COUNT:-1}"
fi

COMMON=(
  "$PY" "$ROOT/tools/adversarial_training.py"
  --tasks burgers
  --generalization-root "$GEN_ROOT"
  --output-root "$OUT_ROOT"
  --device cuda
  --seed "$SEED"
  --checkpoint-every-epochs "$CHECKPOINT_EVERY_EPOCHS"
  --training-data-mode adv-only
  --training-perturbation-mode random-field
  --label-mode solver
  --epsilon-bucket-count "$EPSILON_BUCKET_COUNT"
  --attack-probe-samples "$ATTACK_PROBE_SAMPLES"
  --attack-probe-every-n-epochs "$ATTACK_PROBE_EVERY_N_EPOCHS"
  --attack-probe-save-targets
  --burgers-attack-method fast_replace_l2
  --burgers-require-p2q2
  --burgers-attack-steps 0
  --burgers-epsilon-fraction "${BURGERS_EPSILON_FRACTION:-0.04}"
  --burgers-eps-jitter-low "${BURGERS_EPS_JITTER_LOW:-0.75}"
  --burgers-eps-jitter-high "${BURGERS_EPS_JITTER_HIGH:-1.25}"
  --eval-max-samples "$EVAL_MAX_SAMPLES"
  --max-generalization-eval "$MAX_GENERALIZATION_EVAL"
  --random-field-families "$RANDOM_FIELD_FAMILIES"
  --random-field-gaussian-correlation-choices "$RANDOM_FIELD_GAUSSIAN_CORR"
  --random-field-matern-correlation-choices "$RANDOM_FIELD_MATERN_CORR"
  --random-field-matern-nu-choices "$RANDOM_FIELD_MATERN_NU"
  --random-field-domain-extent "$RANDOM_FIELD_DOMAIN_EXTENT"
  --burgers-batch-size "$RANDOM_BATCH"
  --burgers-optimizer-batch-size "$RANDOM_OPT_BATCH"
  --burgers-solver-remat "$BURGERS_SOLVER_REMAT"
  --burgers-solver-remat-chunk-steps "$BURGERS_SOLVER_REMAT_CHUNK_STEPS"
)

if [[ -n "$CHECKPOINT_WALL_HOURS" ]]; then
  COMMON+=(--checkpoint-wall-hours "$CHECKPOINT_WALL_HOURS")
fi
if [[ -n "${RANDOM_FIELD_CLIP_X_MIN:-}" ]]; then
  COMMON+=(--random-field-clip-x-min "$RANDOM_FIELD_CLIP_X_MIN")
fi
if [[ -n "${RANDOM_FIELD_CLIP_X_MAX:-}" ]]; then
  COMMON+=(--random-field-clip-x-max "$RANDOM_FIELD_CLIP_X_MAX")
fi
if [[ -n "${MAX_WALL_SECONDS:-}" ]]; then
  COMMON+=(--max-wall-seconds "$MAX_WALL_SECONDS")
fi
if [[ "$SMOKE" == "1" ]]; then
  COMMON+=(--burgers-train-max "${SMOKE_TRAIN_MAX:-8}" --max-batches-per-epoch "${SMOKE_MAX_BATCHES_PER_EPOCH:-1}")
fi

epoch_pad() {
  printf "%06d" "$1"
}

run_segment() {
  local label="$1"
  local target_mode="$2"
  local prev_epoch="$3"
  local target_epoch="$4"
  local initial_ckpt="$5"
  local segment_epochs=$((target_epoch - prev_epoch))

  if (( segment_epochs <= 0 )); then
    echo "[bad-target] target_epoch=$target_epoch must be greater than prev_epoch=$prev_epoch" >&2
    exit 1
  fi

  local run_name
  if [[ "$SMOKE" == "1" ]]; then
    run_name="burgers_wideparam_random_field_${label}_smoke_resume_${prev_epoch}_to_${target_epoch}_20260613_${SMOKE_STAMP:-$(date -u +%H%M%S)}"
  else
    run_name="burgers_wideparam_random_field_${label}_${target_epoch}ep_continue_20260613"
  fi
  local run_dir="$OUT_ROOT/$run_name"
  local final_ckpt="$run_dir/burgers/checkpoints/burgers_epoch${target_epoch}_step$(epoch_pad "$target_epoch").pt"
  local log="$LOG_ROOT/${run_name}.log"

  if [[ -f "$run_dir/summary.json" && -f "$final_ckpt" ]]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] skip completed $run_name final=$final_ckpt" | tee -a "$LOG_ROOT/driver.log" >&2
    echo "$final_ckpt"
    return 0
  fi
  if [[ -d "$run_dir" ]]; then
    echo "[refuse] run directory exists without expected final checkpoint: $run_dir" >&2
    exit 1
  fi

  local -a cmd=(
    "${COMMON[@]}"
    --run-name "$run_name"
    --epochs "$segment_epochs"
    --random-field-target-mode "$target_mode"
    --burgers-initial-checkpoint "$initial_ckpt"
    --resume-epoch-offset "$prev_epoch"
    --resume-global-step-offset "$prev_epoch"
  )

  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] start $run_name label=$label target_mode=$target_mode resume=$prev_epoch target=$target_epoch segment_epochs=$segment_epochs" | tee -a "$LOG_ROOT/driver.log" >&2
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '[dry-run]' >&2
    printf ' %q' "${cmd[@]}" >&2
    printf '\n' >&2
    echo "$final_ckpt"
    return 0
  fi
  "${cmd[@]}" > "$log" 2>&1
  if [[ ! -f "$final_ckpt" ]]; then
    echo "[missing-final] expected $final_ckpt" >&2
    tail -80 "$log" >&2 || true
    exit 1
  fi
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] done $run_name final=$final_ckpt" | tee -a "$LOG_ROOT/driver.log" >&2
  echo "$final_ckpt"
}

continue_model() {
  local label="$1"
  local target_mode="$2"
  local ckpt="$3"
  local prev_epoch="$START_EPOCH"
  local target

  IFS=',' read -r -a targets <<< "$TARGET_EPOCHS"
  for target in "${targets[@]}"; do
    target="${target// /}"
    [[ -z "$target" ]] && continue
    ckpt="$(run_segment "$label" "$target_mode" "$prev_epoch" "$target" "$ckpt" | tail -n 1)"
    prev_epoch="$target"
  done
}

if [[ "$RUN_RANDOM_CLEAN_Y" == "1" ]]; then
  continue_model "clean_y" "clean-y" "$BASE_CLEAN_CKPT"
fi
if [[ "$RUN_RANDOM_SOLVER_Y" == "1" ]]; then
  continue_model "solver_y" "solver-y" "$BASE_SOLVER_CKPT"
fi

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] continuation workflow finished" | tee -a "$LOG_ROOT/driver.log"
