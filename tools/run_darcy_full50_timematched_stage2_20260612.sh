#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-adv_robust/bin/python}"
PREVIOUS_TAG="${PREVIOUS_TAG:-20260612_full50_timematched_1000c}"
STAGE2_TAG="${STAGE2_TAG:-20260612_stage2_2000_from_1000c}"
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
RUN_PLOT="${RUN_PLOT:-1}"
UPLOAD_TO_R2="${UPLOAD_TO_R2:-1}"
UPLOAD_RUN_DIRS="${UPLOAD_RUN_DIRS:-1}"
WAIT_FOR_PREVIOUS="${WAIT_FOR_PREVIOUS:-1}"
WAIT_POLL_SECONDS="${WAIT_POLL_SECONDS:-300}"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-0}"
REQUIRE_PREVIOUS_DRIVER_COMPLETE="${REQUIRE_PREVIOUS_DRIVER_COMPLETE:-1}"
R2_BASE_PREFIX="${R2_PREFIX:-machine-sync/NeuralOperatorRobustness2-selected}"

# Previous stage epoch counts for tag 20260612_full50_timematched_1000c.
declare -A PREVIOUS_EPOCHS_BY_OBJECTIVE=(
  [loss1]="${PREVIOUS_LOSS1_EPOCHS:-1000}"
  [loss2]="${PREVIOUS_LOSS2_EPOCHS:-1026}"
  [loss3]="${PREVIOUS_LOSS3_EPOCHS:-1011}"
  [physics]="${PREVIOUS_PHYSICS_EPOCHS:-1040}"
)

# Second stage is time-matched to loss1 2000 epochs. Because this run uses
# TRAIN_MAX == DARCY_BATCH by default, one epoch is one training step.
declare -A EPOCHS_BY_OBJECTIVE=(
  [loss1]="${LOSS1_STAGE2_EPOCHS:-2000}"
  [loss2]="${LOSS2_STAGE2_EPOCHS:-2053}"
  [loss3]="${LOSS3_STAGE2_EPOCHS:-2022}"
  [physics]="${PHYSICS_STAGE2_EPOCHS:-2081}"
)

OBJECTIVES=(loss1 loss2 loss3 physics)
LOG_ROOT="$OUT_ROOT/darcy_full50_timematched_stage2_${STAGE2_TAG}_logs"
mkdir -p "$LOG_ROOT"
DRIVER_LOG="$LOG_ROOT/driver.log"

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$DRIVER_LOG"
}

previous_run_dir_for() {
  local objective="$1"
  local epochs="${PREVIOUS_EPOCHS_BY_OBJECTIVE[$objective]}"
  echo "$OUT_ROOT/darcy_binary_loss3targeted_${objective}_${epochs}ep_full50_timematched_${PREVIOUS_TAG}"
}

stage2_run_dir_for() {
  local objective="$1"
  local stage2_epochs="${EPOCHS_BY_OBJECTIVE[$objective]}"
  local previous_epochs="${PREVIOUS_EPOCHS_BY_OBJECTIVE[$objective]}"
  echo "$OUT_ROOT/darcy_binary_loss3targeted_${objective}_continue${stage2_epochs}ep_from_${previous_epochs}ep_full50_timematched_${STAGE2_TAG}"
}

previous_driver_log() {
  echo "$OUT_ROOT/darcy_full50_timematched_long_${PREVIOUS_TAG}_logs/driver.log"
}

previous_complete_marker() {
  echo "Darcy full50 timematched launcher complete: mode=long tag=$PREVIOUS_TAG"
}

have_previous_summaries() {
  local objective
  for objective in "${OBJECTIVES[@]}"; do
    [[ -f "$(previous_run_dir_for "$objective")/darcy/summary.json" ]] || return 1
  done
  return 0
}

have_previous_driver_complete() {
  local driver
  driver="$(previous_driver_log)"
  [[ -f "$driver" ]] || return 1
  grep -Fq "$(previous_complete_marker)" "$driver"
}

wait_for_previous_if_needed() {
  if [[ "$WAIT_FOR_PREVIOUS" != "1" ]]; then
    log "WAIT_FOR_PREVIOUS=0; not waiting for previous stage"
    return 0
  fi

  log "waiting for previous full script to complete: tag=$PREVIOUS_TAG"
  local start now elapsed
  start="$(date +%s)"
  while true; do
    if have_previous_summaries; then
      if [[ "$REQUIRE_PREVIOUS_DRIVER_COMPLETE" != "1" ]] || have_previous_driver_complete; then
        log "previous stage complete: tag=$PREVIOUS_TAG"
        return 0
      fi
      log "previous summaries exist; waiting for driver completion marker after plot/upload"
    else
      log "previous summaries not ready yet"
    fi
    if [[ "$WAIT_TIMEOUT_SECONDS" != "0" ]]; then
      now="$(date +%s)"
      elapsed=$((now - start))
      if (( elapsed >= WAIT_TIMEOUT_SECONDS )); then
        log "ERROR: timed out waiting for previous stage after ${elapsed}s"
        exit 4
      fi
    fi
    sleep "$WAIT_POLL_SECONDS"
  done
}

resolve_previous_checkpoint() {
  local objective="$1"
  local summary
  summary="$(previous_run_dir_for "$objective")/darcy/summary.json"
  "$PYTHON" - "$summary" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
payload = json.loads(summary_path.read_text())
checkpoint = Path(payload["final_checkpoint"])
if not checkpoint.is_absolute():
    checkpoint = Path.cwd() / checkpoint
resume_epoch = int(payload.get("resume_epoch_offset", 0)) + int(payload.get("epochs", 0))
resume_step = int(payload.get("total_steps", resume_epoch))
print(checkpoint)
print(resume_epoch)
print(resume_step)
PY
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
  R2_PREFIX="$R2_BASE_PREFIX/darcy_full50_timematched_stage2_${STAGE2_TAG}" \
  R2_UPLOAD_LOG_ROOT="$LOG_ROOT/r2_upload_logs" \
    "$ROOT/tools/upload_path_to_r2_20260525.sh" "$path"
}

wait_for_previous_if_needed

declare -A RUN_DIRS=()
declare -A INITIAL_CHECKPOINTS=()
declare -A RESUME_EPOCH_OFFSETS=()
declare -A RESUME_GLOBAL_STEP_OFFSETS=()

for objective in "${OBJECTIVES[@]}"; do
  mapfile -t resolved < <(resolve_previous_checkpoint "$objective")
  INITIAL_CHECKPOINTS[$objective]="${resolved[0]}"
  RESUME_EPOCH_OFFSETS[$objective]="${resolved[1]}"
  RESUME_GLOBAL_STEP_OFFSETS[$objective]="${resolved[2]}"
  if [[ ! -f "${INITIAL_CHECKPOINTS[$objective]}" ]]; then
    log "ERROR: previous final checkpoint missing for $objective: ${INITIAL_CHECKPOINTS[$objective]}"
    exit 5
  fi
done

export LOSS1_EFFECTIVE_EPOCHS="${EPOCHS_BY_OBJECTIVE[loss1]}"
export LOSS2_EFFECTIVE_EPOCHS="${EPOCHS_BY_OBJECTIVE[loss2]}"
export LOSS3_EFFECTIVE_EPOCHS="${EPOCHS_BY_OBJECTIVE[loss3]}"
export PHYSICS_EFFECTIVE_EPOCHS="${EPOCHS_BY_OBJECTIVE[physics]}"

"$PYTHON" - "$STAGE2_TAG" "$PREVIOUS_TAG" "$GENERALIZATION_ROOT" "$EVAL_MAX_SAMPLES" "$MAX_GENERALIZATION_EVAL" "$CHECKPOINT_EVERY" "$LOG_ROOT/timematched_stage2_plan.json" <<'PYPLAN' | tee -a "$DRIVER_LOG"
import json
import os
import sys

stage2_tag, previous_tag, gen_root, eval_max, max_gen, ckpt_every, out = sys.argv[1:]
measured_sec_per_epoch = {
    "loss1": 4.680958273200085,
    "loss2": 4.562185353540117,
    "loss3": 4.632201948699949,
    "physics": 4.499857622220006,
}
epochs = {
    "loss1": int(os.environ["LOSS1_EFFECTIVE_EPOCHS"]),
    "loss2": int(os.environ["LOSS2_EFFECTIVE_EPOCHS"]),
    "loss3": int(os.environ["LOSS3_EFFECTIVE_EPOCHS"]),
    "physics": int(os.environ["PHYSICS_EFFECTIVE_EPOCHS"]),
}
objectives = {}
for name, epoch_count in epochs.items():
    projected = epoch_count * measured_sec_per_epoch[name]
    objectives[name] = {
        "stage2_epochs": epoch_count,
        "measured_sec_per_epoch_from_full50_100ep": measured_sec_per_epoch[name],
        "projected_seconds": projected,
        "projected_hours": projected / 3600.0,
    }
payload = {
    "stage2_tag": stage2_tag,
    "previous_tag": previous_tag,
    "generalization_root": gen_root,
    "eval_max_samples": int(eval_max),
    "max_generalization_eval": int(max_gen),
    "checkpoint_every_epochs": int(ckpt_every),
    "time_match_reference": "loss1 2000 epochs from measured full50 100-epoch run",
    "note": "With TRAIN_MAX == DARCY_BATCH, one epoch is one training step.",
    "objectives": objectives,
    "projected_total_seconds": sum(v["projected_seconds"] for v in objectives.values()),
    "projected_total_hours": sum(v["projected_seconds"] for v in objectives.values()) / 3600.0,
}
with open(out, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
print(json.dumps(payload, indent=2))
PYPLAN

log "Darcy full50 timematched stage2 launcher start: tag=$STAGE2_TAG previous=$PREVIOUS_TAG"

for objective in "${OBJECTIVES[@]}"; do
  epochs="${EPOCHS_BY_OBJECTIVE[$objective]}"
  run_dir="$(stage2_run_dir_for "$objective")"
  RUN_DIRS[$objective]="$run_dir"
  summary="$run_dir/darcy/summary.json"
  if [[ -f "$summary" ]]; then
    log "skip completed $objective stage2 run: $summary"
    continue
  fi
  if [[ -d "$run_dir" ]]; then
    log "ERROR: incomplete stage2 run directory exists for $objective: $run_dir"
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
    --training-data-mode adv-only
    --label-mode solver
    --checkpoint-every-epochs "$CHECKPOINT_EVERY"
    --darcy-attack-loss-objective "$objective"
    --attack-probe-samples "$ATTACK_PROBE_SAMPLES"
    --attack-probe-every-n-epochs "$ATTACK_PROBE_EVERY"
    --attack-probe-save-targets
    --darcy-initial-checkpoint "${INITIAL_CHECKPOINTS[$objective]}"
    --resume-epoch-offset "${RESUME_EPOCH_OFFSETS[$objective]}"
    --resume-global-step-offset "${RESUME_GLOBAL_STEP_OFFSETS[$objective]}"
  )
  log "start stage2 $objective epochs=$epochs resume_epoch=${RESUME_EPOCH_OFFSETS[$objective]} resume_step=${RESUME_GLOBAL_STEP_OFFSETS[$objective]} checkpoint=${INITIAL_CHECKPOINTS[$objective]} run=$run_dir"
  printf '[command] %q ' "${cmd[@]}" | tee -a "$DRIVER_LOG"
  printf '\n' | tee -a "$DRIVER_LOG"
  "${cmd[@]}" 2>&1 | tee "$LOG_ROOT/$(basename "$run_dir").log"
  log "done stage2 $objective epochs=$epochs run=$run_dir"
done

if [[ "$RUN_PLOT" == "1" ]]; then
  out_vis="visualizations/darcy_loss123physics_full50_timematched_stage2_${STAGE2_TAG}"
  report_md="docs/darcy_loss123physics_full50_timematched_stage2_${STAGE2_TAG}.md"
  plot_cmd=(
    "$PYTHON" tools/plot_darcy_loss123physics_polished_burgers_style_20260611.py
    --loss1-run-dir "${RUN_DIRS[loss1]}"
    --loss2-run-dir "${RUN_DIRS[loss2]}"
    --loss3-run-dir "${RUN_DIRS[loss3]}"
    --physics-run-dir "${RUN_DIRS[physics]}"
    --out-root "$out_vis"
    --report-md "$report_md"
  )
  log "start stage2 plotting: $out_vis"
  "${plot_cmd[@]}" 2>&1 | tee "$LOG_ROOT/plot.log"
  log "done stage2 plotting: $out_vis"

  combined_out="$out_vis/combined_four_method_burgers_style"
  combined_report_md="docs/darcy_loss123physics_full50_timematched_stage2_${STAGE2_TAG}_combined_four_method_burgers_style.md"
  combined_cmd=(
    "$PYTHON" tools/plot_darcy_loss123physics_adv_training_20260611.py
    --loss1-run-dir "${RUN_DIRS[loss1]}"
    --loss2-run-dir "${RUN_DIRS[loss2]}"
    --loss3-run-dir "${RUN_DIRS[loss3]}"
    --physics-run-dir "${RUN_DIRS[physics]}"
    --derive-full50-from-run-eval
    --out-dir "$combined_out"
    --report-md "$combined_report_md"
    --write-full-logging-launcher "/tmp/darcy_full_logging_launcher_stage2_${STAGE2_TAG}.sh"
  )
  log "start stage2 combined four-method Burgers-style plotting: $combined_out"
  "${combined_cmd[@]}" 2>&1 | tee "$LOG_ROOT/plot_combined_four_method.log"
  log "done stage2 combined four-method Burgers-style plotting: $combined_out"

  combined_linear_out="$out_vis/combined_four_method_burgers_style_linear"
  combined_linear_report_md="docs/darcy_loss123physics_full50_timematched_stage2_${STAGE2_TAG}_combined_four_method_burgers_style_linear.md"
  combined_linear_cmd=(
    "${combined_cmd[@]}"
    --out-dir "$combined_linear_out"
    --report-md "$combined_linear_report_md"
    --linear-scale
  )
  log "start stage2 combined four-method linear-scale plotting: $combined_linear_out"
  "${combined_linear_cmd[@]}" 2>&1 | tee "$LOG_ROOT/plot_combined_four_method_linear.log"
  log "done stage2 combined four-method linear-scale plotting: $combined_linear_out"
fi

if [[ "$UPLOAD_TO_R2" == "1" ]]; then
  if [[ "$UPLOAD_RUN_DIRS" == "1" ]]; then
    for objective in "${OBJECTIVES[@]}"; do
      upload_path "${RUN_DIRS[$objective]}"
    done
  else
    log "UPLOAD_RUN_DIRS=0; skipping stage2 run-dir checkpoint upload"
  fi
  upload_path "$LOG_ROOT"
  upload_path "tools/run_darcy_full50_timematched_stage2_20260612.sh"
  if [[ "$RUN_PLOT" == "1" ]]; then
    upload_path "visualizations/darcy_loss123physics_full50_timematched_stage2_${STAGE2_TAG}"
    upload_path "docs/darcy_loss123physics_full50_timematched_stage2_${STAGE2_TAG}.md"
    upload_path "docs/darcy_loss123physics_full50_timematched_stage2_${STAGE2_TAG}_combined_four_method_burgers_style.md"
    upload_path "docs/darcy_loss123physics_full50_timematched_stage2_${STAGE2_TAG}_combined_four_method_burgers_style_linear.md"
  fi
fi

log "Darcy full50 timematched stage2 launcher complete: tag=$STAGE2_TAG previous=$PREVIOUS_TAG"
