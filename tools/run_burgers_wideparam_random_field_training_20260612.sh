#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/workspace/NeuralOperatorRobustness2}"
PY="${PY:-/venv/adv_robust/bin/python}"
GEN_ROOT="${GEN_ROOT:-$ROOT/generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00}"
OUT_ROOT="${OUT_ROOT:-$ROOT/adversarial_training_runs}"
LOG_ROOT="${LOG_ROOT:-$OUT_ROOT/burgers_wideparam_random_field_training_20260612_logs}"
PREFLIGHT_DIR="${PREFLIGHT_DIR:-$ROOT/forensics/burgers_wideparam_random_field_training_preflight_20260612}"

CLEAN_Y_RUN="${CLEAN_Y_RUN:-burgers_wideparam_random_field_clean_y_2000ep_20260612}"
SOLVER_Y_RUN="${SOLVER_Y_RUN:-burgers_wideparam_random_field_solver_y_2000ep_20260612}"
RANDOM_EPOCHS="${RANDOM_EPOCHS:-2000}"
RANDOM_BATCH="${RANDOM_BATCH:-1350}"
RANDOM_OPT_BATCH="${RANDOM_OPT_BATCH:-32}"
BURGERS_SOLVER_REMAT="${BURGERS_SOLVER_REMAT:-chunk}"
BURGERS_SOLVER_REMAT_CHUNK_STEPS="${BURGERS_SOLVER_REMAT_CHUNK_STEPS:-50}"

RUN_RANDOM_CLEAN_Y="${RUN_RANDOM_CLEAN_Y:-1}"
RUN_RANDOM_SOLVER_Y="${RUN_RANDOM_SOLVER_Y:-1}"
RUN_RANDOM_PARALLEL="${RUN_RANDOM_PARALLEL:-0}"
RUN_PLOTS="${RUN_PLOTS:-1}"
DRY_RUN="${DRY_RUN:-0}"

RANDOM_FIELD_FAMILIES="${RANDOM_FIELD_FAMILIES:-gaussian,matern}"
RANDOM_FIELD_GAUSSIAN_CORR="${RANDOM_FIELD_GAUSSIAN_CORR:-0.015,0.03,0.06,0.12,0.24}"
RANDOM_FIELD_MATERN_CORR="${RANDOM_FIELD_MATERN_CORR:-0.015,0.03,0.06,0.12,0.24}"
RANDOM_FIELD_MATERN_NU="${RANDOM_FIELD_MATERN_NU:-1.2,2.2,3.2,4.2,5.2}"
RANDOM_FIELD_DOMAIN_EXTENT="${RANDOM_FIELD_DOMAIN_EXTENT:-2.0}"

mkdir -p "$LOG_ROOT" "$PREFLIGHT_DIR"
cd "$ROOT"

for required in "$PY" "$GEN_ROOT/burgers" "$ROOT/tools/adversarial_training.py" "$ROOT/tools/plot_burgers_wideparam_loss123_retrain_20260611.py"; do
  if [[ ! -e "$required" ]]; then
    echo "[missing] $required" >&2
    exit 1
  fi
done

"$PY" - <<'PYGPU' > "$PREFLIGHT_DIR/gpu_preflight.json"
import json
import sys
import time

import torch

payload = {
    "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "python": sys.executable,
    "torch_version": torch.__version__,
    "torch_cuda_version": torch.version.cuda,
    "torch_cuda_available": torch.cuda.is_available(),
}
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; refusing Burgers random-field training")
payload["torch_device_name"] = torch.cuda.get_device_name(0)
payload["torch_device_capability"] = torch.cuda.get_device_capability(0)
x = torch.randn(128, 128, device="cuda")
y = x @ x
torch.cuda.synchronize()
payload["torch_sanity_sum"] = float(y.sum().detach().cpu())
print(json.dumps(payload, indent=2))
PYGPU
nvidia-smi > "$PREFLIGHT_DIR/nvidia_smi_before.txt" || true

COMMON=(
  "$PY" "$ROOT/tools/adversarial_training.py"
  --tasks burgers
  --generalization-root "$GEN_ROOT"
  --output-root "$OUT_ROOT"
  --device cuda
  --seed "${SEED:-20260612}"
  --checkpoint-every-epochs "${CHECKPOINT_EVERY_EPOCHS:-100}"
  --checkpoint-wall-hours "${CHECKPOINT_WALL_HOURS:-0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,6,7,8,9,10,11,12,12.5,14,16,18,20}"
  --training-data-mode adv-only
  --training-perturbation-mode random-field
  --label-mode solver
  --epsilon-bucket-count "${EPSILON_BUCKET_COUNT:-5}"
  --attack-probe-samples "${ATTACK_PROBE_SAMPLES:-5}"
  --attack-probe-every-n-epochs "${ATTACK_PROBE_EVERY_N_EPOCHS:-1}"
  --attack-probe-save-targets
  --burgers-attack-method fast_replace_l2
  --burgers-require-p2q2
  --burgers-attack-steps 0
  --burgers-epsilon-fraction "${BURGERS_EPSILON_FRACTION:-0.04}"
  --burgers-eps-jitter-low "${BURGERS_EPS_JITTER_LOW:-0.75}"
  --burgers-eps-jitter-high "${BURGERS_EPS_JITTER_HIGH:-1.25}"
  --eval-max-samples "${EVAL_MAX_SAMPLES:-0}"
  --max-generalization-eval "${MAX_GENERALIZATION_EVAL:-50}"
  --random-field-families "$RANDOM_FIELD_FAMILIES"
  --random-field-gaussian-correlation-choices "$RANDOM_FIELD_GAUSSIAN_CORR"
  --random-field-matern-correlation-choices "$RANDOM_FIELD_MATERN_CORR"
  --random-field-matern-nu-choices "$RANDOM_FIELD_MATERN_NU"
  --random-field-domain-extent "$RANDOM_FIELD_DOMAIN_EXTENT"
)

if [[ -n "${RANDOM_FIELD_CLIP_X_MIN:-}" ]]; then
  COMMON+=(--random-field-clip-x-min "$RANDOM_FIELD_CLIP_X_MIN")
fi
if [[ -n "${RANDOM_FIELD_CLIP_X_MAX:-}" ]]; then
  COMMON+=(--random-field-clip-x-max "$RANDOM_FIELD_CLIP_X_MAX")
fi

run_one() {
  local target_mode="$1"
  local run_name="$2"
  local run_dir="$OUT_ROOT/$run_name"
  local log="$LOG_ROOT/${run_name}.log"

  if [[ -f "$run_dir/summary.json" ]]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] skip completed $run_name" | tee -a "$LOG_ROOT/driver.log"
    return 0
  fi
  if [[ -d "$run_dir" ]]; then
    echo "[refuse] run directory exists without summary.json: $run_dir" >&2
    exit 1
  fi

  local -a cmd=(
    "${COMMON[@]}"
    --run-name "$run_name"
    --epochs "$RANDOM_EPOCHS"
    --random-field-target-mode "$target_mode"
    --burgers-batch-size "$RANDOM_BATCH"
    --burgers-optimizer-batch-size "$RANDOM_OPT_BATCH"
    --burgers-solver-remat "$BURGERS_SOLVER_REMAT"
    --burgers-solver-remat-chunk-steps "$BURGERS_SOLVER_REMAT_CHUNK_STEPS"
  )
  if [[ -n "${MAX_WALL_SECONDS:-}" ]]; then
    cmd+=(--max-wall-seconds "$MAX_WALL_SECONDS")
  fi

  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] start $run_name target=$target_mode epochs=$RANDOM_EPOCHS batch=$RANDOM_BATCH opt_batch=$RANDOM_OPT_BATCH" | tee -a "$LOG_ROOT/driver.log"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '[dry-run]'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    return 0
  fi
  "${cmd[@]}" > "$log" 2>&1
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] done $run_name" | tee -a "$LOG_ROOT/driver.log"
}

run_group() {
  if [[ "$RUN_RANDOM_PARALLEL" == "1" ]]; then
    pids=()
    if [[ "$RUN_RANDOM_CLEAN_Y" == "1" ]]; then
      run_one clean-y "$CLEAN_Y_RUN" &
      pids+=("$!")
    fi
    if [[ "$RUN_RANDOM_SOLVER_Y" == "1" ]]; then
      run_one solver-y "$SOLVER_Y_RUN" &
      pids+=("$!")
    fi
    for pid in "${pids[@]}"; do
      wait "$pid"
    done
  else
    [[ "$RUN_RANDOM_CLEAN_Y" == "1" ]] && run_one clean-y "$CLEAN_Y_RUN"
    [[ "$RUN_RANDOM_SOLVER_Y" == "1" ]] && run_one solver-y "$SOLVER_Y_RUN"
  fi
}

run_group

nvidia-smi > "$PREFLIGHT_DIR/nvidia_smi_after.txt" || true

if [[ "$RUN_PLOTS" == "1" && "$DRY_RUN" != "1" ]]; then
  "$PY" "$ROOT/tools/plot_burgers_wideparam_loss123_retrain_20260611.py" \
    --run "random_clean_y=$OUT_ROOT/$CLEAN_Y_RUN" \
    --run "random_solver_y=$OUT_ROOT/$SOLVER_Y_RUN" \
    --out-dir "$ROOT/visualizations/burgers_wideparam_random_field_training_20260612" \
    --report-md "$ROOT/docs/burgers_wideparam_random_field_training_report_20260612.md" \
    --report-title "Burgers Wideparam Random-Field Noise Training Report - 2026-06-12" \
    --report-description "This report compares two non-adversarial random-delta training baselines: random_clean_y keeps the original clean target fixed, while random_solver_y recomputes the Burgers solver target at x+delta. Both sample fresh Gaussian/Matern random-field deltas every epoch." \
    > "$LOG_ROOT/plot_random_field_training.log" 2>&1
fi

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] workflow finished" | tee -a "$LOG_ROOT/driver.log"
