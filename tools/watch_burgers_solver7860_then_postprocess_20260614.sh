#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/workspace/NeuralOperatorRobustness2}"
PY="${PY:-/workspace/adv_robust/bin/python}"
LOG_ROOT="${LOG_ROOT:-$ROOT/run_logs/burgers_solver7860_postprocess_20260614}"
POLL_SECONDS="${POLL_SECONDS:-120}"

SOLVER_CKPT="$ROOT/adversarial_training_runs/burgers_wideparam_random_field_solver_y_7860ep_continue_20260613/burgers/checkpoints/burgers_epoch7860_step007860.pt"
CLEAN_CKPT="$ROOT/adversarial_training_runs/burgers_wideparam_random_field_clean_y_8000ep_continue_20260613/burgers/checkpoints/burgers_epoch8000_step008000.pt"
RANDOM_ROOT="$ROOT/forensics/burgers_random_solver7860_clean8000_full_suite_20260614"
SUMMARY_ROOT="$ROOT/forensics/burgers_six_model_solver7860_clean8000_summary_20260614"
AUDIT_OUT="$ROOT/outputs/burgers_timematched_solver7860_clean8000_audit_20260614"
R2_PREFIX="${R2_PREFIX:-neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected}"

mkdir -p "$LOG_ROOT"
cd "$ROOT"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG_ROOT/postprocess_driver.log"
}

log "waiting for solver checkpoint: $SOLVER_CKPT"
while [[ ! -f "$SOLVER_CKPT" ]]; do
  sleep "$POLL_SECONDS"
done
log "found solver checkpoint"

if [[ ! -f "$CLEAN_CKPT" ]]; then
  log "missing clean checkpoint: $CLEAN_CKPT"
  exit 1
fi

log "running clean+attack evaluation for random_clean_y=8000 and random_solver_y=7860"
"$PY" "$ROOT/tools/evaluate_burgers_random_field_final_models_20260613.py" \
  --out-root "$RANDOM_ROOT" \
  --stage clean \
  --stage attack \
  --clean-batch-size "${CLEAN_BATCH_SIZE:-256}" \
  --attack-steps "${ATTACK_STEPS:-20}" \
  --attack-batch-size "${ATTACK_BATCH_SIZE:-500}" \
  --attack-train-count "${ATTACK_TRAIN_COUNT:-50}" \
  --model-spec "random_clean_y=$CLEAN_CKPT" \
  --model-spec "random_solver_y=$SOLVER_CKPT" \
  > "$LOG_ROOT/evaluate_random_clean8000_solver7860.log" 2>&1

log "building selected summary"
"$PY" "$ROOT/tools/build_burgers_selected_worktime_six_model_summary_20260613.py" \
  --random-root "$RANDOM_ROOT" \
  --out-dir "$SUMMARY_ROOT" \
  --allow-incomplete \
  --random-clean-checkpoint "${CLEAN_CKPT#$ROOT/}" \
  --random-solver-checkpoint "${SOLVER_CKPT#$ROOT/}" \
  > "$LOG_ROOT/build_summary.log" 2>&1

log "building 8h audit/curve bundle"
"$PY" "$ROOT/tools/audit_burgers_timematched_full_20260614.py" \
  --out "$AUDIT_OUT" \
  --workclock-xmax 8.0 \
  > "$LOG_ROOT/audit_solver7860_clean8000.log" 2>&1

if [[ -n "${R2_ACCESS_KEY_ID:-}" && -n "${R2_SECRET_ACCESS_KEY:-}" && -n "${R2_ENDPOINT:-}" ]]; then
  log "uploading postprocess outputs to R2 prefix $R2_PREFIX"
  RCLONE_CONFIG_R2_TYPE=s3 \
  RCLONE_CONFIG_R2_PROVIDER=Cloudflare \
  RCLONE_CONFIG_R2_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" \
  RCLONE_CONFIG_R2_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" \
  RCLONE_CONFIG_R2_ENDPOINT="$R2_ENDPOINT" \
  rclone copy "$RANDOM_ROOT" "R2:$R2_PREFIX/forensics/${RANDOM_ROOT##*/}" --transfers 8 --checkers 16 --s3-no-check-bucket >> "$LOG_ROOT/r2_upload.log" 2>&1
  RCLONE_CONFIG_R2_TYPE=s3 \
  RCLONE_CONFIG_R2_PROVIDER=Cloudflare \
  RCLONE_CONFIG_R2_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" \
  RCLONE_CONFIG_R2_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" \
  RCLONE_CONFIG_R2_ENDPOINT="$R2_ENDPOINT" \
  rclone copy "$SUMMARY_ROOT" "R2:$R2_PREFIX/forensics/${SUMMARY_ROOT##*/}" --transfers 8 --checkers 16 --s3-no-check-bucket >> "$LOG_ROOT/r2_upload.log" 2>&1
  RCLONE_CONFIG_R2_TYPE=s3 \
  RCLONE_CONFIG_R2_PROVIDER=Cloudflare \
  RCLONE_CONFIG_R2_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" \
  RCLONE_CONFIG_R2_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" \
  RCLONE_CONFIG_R2_ENDPOINT="$R2_ENDPOINT" \
  rclone copy "$AUDIT_OUT" "R2:$R2_PREFIX/outputs/${AUDIT_OUT##*/}" --transfers 8 --checkers 16 --s3-no-check-bucket >> "$LOG_ROOT/r2_upload.log" 2>&1
else
  log "R2 env not present; skipping upload"
fi

log "postprocess complete"
