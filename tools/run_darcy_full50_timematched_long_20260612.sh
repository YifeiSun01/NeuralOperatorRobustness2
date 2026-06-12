#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-adv_robust/bin/python}"
MODE="${MODE:-long}"
TAG="${TAG:-20260612_full50_timematched_1000}"
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
R2_BASE_PREFIX="${R2_PREFIX:-machine-sync/NeuralOperatorRobustness2-selected}"

case "$MODE" in
  smoke)
    declare -A EPOCHS_BY_OBJECTIVE=(
      [loss1]="${SMOKE_EPOCHS:-1}"
      [loss2]="${SMOKE_EPOCHS:-1}"
      [loss3]="${SMOKE_EPOCHS:-1}"
      [physics]="${SMOKE_EPOCHS:-1}"
    )
    ;;
  long)
    # Time matched to the measured full50 100-epoch run on this V100:
    # loss1 4.681 sec/epoch is the slowest baseline. Faster objectives receive
    # enough extra epochs to match loss1's 1000-epoch wall-clock time.
    declare -A EPOCHS_BY_OBJECTIVE=(
      [loss1]="${LOSS1_EPOCHS:-1000}"
      [loss2]="${LOSS2_EPOCHS:-1026}"
      [loss3]="${LOSS3_EPOCHS:-1011}"
      [physics]="${PHYSICS_EPOCHS:-1040}"
    )
    ;;
  *)
    echo "ERROR: MODE must be smoke or long, got $MODE" >&2
    exit 2
    ;;
esac

OBJECTIVES=(loss1 loss2 loss3 physics)
LOG_ROOT="$OUT_ROOT/darcy_full50_timematched_${MODE}_${TAG}_logs"
mkdir -p "$LOG_ROOT"
DRIVER_LOG="$LOG_ROOT/driver.log"

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$DRIVER_LOG"
}

run_dir_for() {
  local objective="$1"
  local epochs="$2"
  echo "$OUT_ROOT/darcy_binary_loss3targeted_${objective}_${epochs}ep_full50_timematched_${TAG}"
}

write_plan() {
  "$PYTHON" - "$MODE" "$TAG" "$GENERALIZATION_ROOT" "$EVAL_MAX_SAMPLES" "$MAX_GENERALIZATION_EVAL" "$CHECKPOINT_EVERY" "$LOG_ROOT/timematched_plan.json" <<'PYPLAN'
import json
import sys
mode, tag, gen_root, eval_max, max_gen, ckpt_every, out = sys.argv[1:]
measured_sec_per_epoch = {
    "loss1": 4.680958273200085,
    "loss2": 4.562185353540117,
    "loss3": 4.632201948699949,
    "physics": 4.499857622220006,
}
epochs = {
    "loss1": int(__import__("os").environ["LOSS1_EFFECTIVE_EPOCHS"]),
    "loss2": int(__import__("os").environ["LOSS2_EFFECTIVE_EPOCHS"]),
    "loss3": int(__import__("os").environ["LOSS3_EFFECTIVE_EPOCHS"]),
    "physics": int(__import__("os").environ["PHYSICS_EFFECTIVE_EPOCHS"]),
}
rows = {}
for k, e in epochs.items():
    seconds = e * measured_sec_per_epoch[k]
    rows[k] = {
        "epochs": e,
        "measured_sec_per_epoch_from_full50_100ep": measured_sec_per_epoch[k],
        "projected_seconds": seconds,
        "projected_hours": seconds / 3600.0,
    }
payload = {
    "mode": mode,
    "tag": tag,
    "generalization_root": gen_root,
    "eval_max_samples": int(eval_max),
    "max_generalization_eval": int(max_gen),
    "checkpoint_every_epochs": int(ckpt_every),
    "time_match_reference": "loss1 1000 epochs from measured full50 100-epoch run",
    "objectives": rows,
    "projected_total_seconds": sum(v["projected_seconds"] for v in rows.values()),
    "projected_total_hours": sum(v["projected_seconds"] for v in rows.values()) / 3600.0,
}
with open(out, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
print(json.dumps(payload, indent=2))
PYPLAN
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
  R2_PREFIX="$R2_BASE_PREFIX/darcy_full50_timematched_${MODE}_${TAG}" \
  R2_UPLOAD_LOG_ROOT="$LOG_ROOT/r2_upload_logs" \
    "$ROOT/tools/upload_path_to_r2_20260525.sh" "$path"
}

export LOSS1_EFFECTIVE_EPOCHS="${EPOCHS_BY_OBJECTIVE[loss1]}"
export LOSS2_EFFECTIVE_EPOCHS="${EPOCHS_BY_OBJECTIVE[loss2]}"
export LOSS3_EFFECTIVE_EPOCHS="${EPOCHS_BY_OBJECTIVE[loss3]}"
export PHYSICS_EFFECTIVE_EPOCHS="${EPOCHS_BY_OBJECTIVE[physics]}"

log "Darcy full50 timematched launcher start: mode=$MODE tag=$TAG"
write_plan | tee -a "$DRIVER_LOG"

declare -A RUN_DIRS=()
for objective in "${OBJECTIVES[@]}"; do
  epochs="${EPOCHS_BY_OBJECTIVE[$objective]}"
  run_dir="$(run_dir_for "$objective" "$epochs")"
  RUN_DIRS[$objective]="$run_dir"
  summary="$run_dir/darcy/summary.json"
  if [[ -f "$summary" ]]; then
    log "skip completed $objective run: $summary"
    continue
  fi
  if [[ -d "$run_dir" ]]; then
    log "ERROR: incomplete run directory exists for $objective: $run_dir"
    exit 3
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
  )
  log "start $objective epochs=$epochs run=$run_dir"
  printf '[command] %q ' "${cmd[@]}" | tee -a "$DRIVER_LOG"
  printf '\n' | tee -a "$DRIVER_LOG"
  "${cmd[@]}" 2>&1 | tee "$LOG_ROOT/$(basename "$run_dir").log"
  log "done $objective epochs=$epochs run=$run_dir"
done

if [[ "$RUN_PLOT" == "1" ]]; then
  out_vis="visualizations/darcy_loss123physics_full50_timematched_${MODE}_${TAG}"
  report_md="docs/darcy_loss123physics_full50_timematched_${MODE}_${TAG}.md"
  plot_cmd=(
    "$PYTHON" tools/plot_darcy_loss123physics_polished_burgers_style_20260611.py
    --loss1-run-dir "${RUN_DIRS[loss1]}"
    --loss2-run-dir "${RUN_DIRS[loss2]}"
    --loss3-run-dir "${RUN_DIRS[loss3]}"
    --physics-run-dir "${RUN_DIRS[physics]}"
    --out-root "$out_vis"
    --report-md "$report_md"
  )
  log "start plotting: $out_vis"
  "${plot_cmd[@]}" 2>&1 | tee "$LOG_ROOT/plot.log"
  log "done plotting: $out_vis"

  combined_out="$out_vis/combined_four_method_burgers_style"
  combined_report_md="docs/darcy_loss123physics_full50_timematched_${MODE}_${TAG}_combined_four_method_burgers_style.md"
  combined_cmd=(
    "$PYTHON" tools/plot_darcy_loss123physics_adv_training_20260611.py
    --loss1-run-dir "${RUN_DIRS[loss1]}"
    --loss2-run-dir "${RUN_DIRS[loss2]}"
    --loss3-run-dir "${RUN_DIRS[loss3]}"
    --physics-run-dir "${RUN_DIRS[physics]}"
    --derive-full50-from-run-eval
    --out-dir "$combined_out"
    --report-md "$combined_report_md"
    --write-full-logging-launcher "/tmp/darcy_full_logging_launcher_${MODE}_${TAG}.sh"
  )
  log "start combined four-method Burgers-style plotting: $combined_out"
  "${combined_cmd[@]}" 2>&1 | tee "$LOG_ROOT/plot_combined_four_method.log"
  log "done combined four-method Burgers-style plotting: $combined_out"

  combined_linear_out="$out_vis/combined_four_method_burgers_style_linear"
  combined_linear_report_md="docs/darcy_loss123physics_full50_timematched_${MODE}_${TAG}_combined_four_method_burgers_style_linear.md"
  combined_linear_cmd=(
    "${combined_cmd[@]}"
    --out-dir "$combined_linear_out"
    --report-md "$combined_linear_report_md"
    --linear-scale
  )
  log "start combined four-method linear-scale plotting: $combined_linear_out"
  "${combined_linear_cmd[@]}" 2>&1 | tee "$LOG_ROOT/plot_combined_four_method_linear.log"
  log "done combined four-method linear-scale plotting: $combined_linear_out"
fi

if [[ "$UPLOAD_TO_R2" == "1" ]]; then
  if [[ "$UPLOAD_RUN_DIRS" == "1" ]]; then
    for objective in "${OBJECTIVES[@]}"; do
      upload_path "${RUN_DIRS[$objective]}"
    done
  else
    log "UPLOAD_RUN_DIRS=0; skipping run-dir checkpoint upload"
  fi
  upload_path "$LOG_ROOT"
  upload_path "tools/run_darcy_full50_timematched_long_20260612.sh"
  if [[ "$RUN_PLOT" == "1" ]]; then
    upload_path "visualizations/darcy_loss123physics_full50_timematched_${MODE}_${TAG}"
    upload_path "docs/darcy_loss123physics_full50_timematched_${MODE}_${TAG}.md"
    upload_path "docs/darcy_loss123physics_full50_timematched_${MODE}_${TAG}_combined_four_method_burgers_style.md"
    upload_path "docs/darcy_loss123physics_full50_timematched_${MODE}_${TAG}_combined_four_method_burgers_style_linear.md"
  fi
fi

log "Darcy full50 timematched launcher complete: mode=$MODE tag=$TAG"
