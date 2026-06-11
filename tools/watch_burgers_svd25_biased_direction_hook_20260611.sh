#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/workspace/NeuralOperatorRobustness2}"
PY="${PY:-/venv/adv_robust/bin/python}"
SVD_OUT_ROOT="${SVD_OUT_ROOT:-$ROOT/forensics/burgers_wideparam_loss3targeted_full1024_svd_attack25_reuse3_20260611}"
OUT_DIR="${OUT_DIR:-$ROOT/forensics/burgers_wideparam_loss3targeted_full1024_svd_attack25_biased_local_direction_20260611}"
REPORT_MD="${REPORT_MD:-$ROOT/docs/burgers_wideparam_loss3targeted_full1024_svd_attack25_biased_local_direction_20260611.md}"
STATE_DIR="${STATE_DIR:-$ROOT/forensics/burgers_svd25_then_retrain_upload_state_20260611}"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"
LOG="${LOG:-$LOG_DIR/burgers_svd25_biased_direction_hook_watcher_20260611.log}"
MASTER_PID_FILE="${MASTER_PID_FILE:-$STATE_DIR/pid.txt}"
DEVICE="${SVD_ANALYSIS_DEVICE:-cpu}"
POLL_SECONDS="${POLL_SECONDS:-60}"
GRACE_SECONDS="${GRACE_SECONDS:-180}"
WAIT_FOR_MASTER_EXIT="${WAIT_FOR_MASTER_EXIT:-0}"
RUN_UPLOAD="${RUN_UPLOAD:-1}"
FORCE="${FORCE:-0}"

mkdir -p "$STATE_DIR" "$LOG_DIR"
cd "$ROOT"
echo "$$" > "$STATE_DIR/biased_direction_hook_watcher.pid"

log() {
  echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*" | tee -a "$LOG"
}

repo_rel() {
  local path="$1"
  local abs
  abs="$(realpath -m "$path")"
  printf '%s\n' "${abs#$ROOT/}"
}

svd_ready() {
  [[ -s "$SVD_OUT_ROOT/sample_manifest.csv" ]] || return 1
  [[ -s "$SVD_OUT_ROOT/svd_summary.csv" ]] || return 1
  [[ -s "$SVD_OUT_ROOT/svd_attack_joined_metrics.csv" ]] || return 1
  [[ -s "$SVD_OUT_ROOT/svd_attack_correlations.csv" ]] || return 1
  return 0
}

hook_done() {
  [[ -s "$OUT_DIR/biased_direction_metrics.csv" ]] || return 1
  [[ -s "$OUT_DIR/biased_direction_correlations.csv" ]] || return 1
  [[ -s "$OUT_DIR/biased_direction_quantile_summary.csv" ]] || return 1
  [[ -s "$OUT_DIR/biased_direction_summary.json" ]] || return 1
  [[ -s "$REPORT_MD" ]] || return 1
  return 0
}

master_pid_alive() {
  [[ -f "$MASTER_PID_FILE" ]] || return 1
  local pid
  pid="$(tr -dc '0-9' < "$MASTER_PID_FILE" || true)"
  [[ -n "$pid" ]] || return 1
  kill -0 "$pid" 2>/dev/null
}

analysis_process_running() {
  pgrep -f "analyze_burgers_biased_local_attack_direction_20260611.py.*$(basename "$SVD_OUT_ROOT")" >/dev/null 2>&1
}

upload_path() {
  local path="$1"
  if [[ "$RUN_UPLOAD" != "1" ]]; then
    log "upload skipped for $path"
    return 0
  fi
  if [[ ! -e "$ROOT/$path" ]]; then
    log "upload skipped missing path $path"
    return 0
  fi
  log "upload start $path"
  R2_UPLOAD_LOG_ROOT="$STATE_DIR/r2_upload_logs" bash "$ROOT/tools/upload_path_to_r2_20260525.sh" "$path" 2>&1 | tee -a "$LOG"
  log "upload done $path"
}

LOCK_DIR="$STATE_DIR/biased_direction_hook.lock"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  log "another biased-direction hook watcher/runner holds $LOCK_DIR; exiting"
  exit 0
fi
trap 'rm -rf "$LOCK_DIR"' EXIT

log "watcher start"
log "svd_out=$SVD_OUT_ROOT"
log "out_dir=$OUT_DIR"
log "report=$REPORT_MD"
log "wait_for_master_exit=$WAIT_FOR_MASTER_EXIT poll_seconds=$POLL_SECONDS grace_seconds=$GRACE_SECONDS force=$FORCE"

if hook_done && [[ "$FORCE" != "1" ]]; then
  log "biased-direction hook already complete; exiting"
  exit 0
fi

while ! svd_ready; do
  if master_pid_alive; then
    log "waiting for SVD25 outputs; master process still alive"
  else
    log "waiting for SVD25 outputs; master process not detected"
  fi
  sleep "$POLL_SECONDS"
done

log "SVD25 outputs detected"

if [[ "$WAIT_FOR_MASTER_EXIT" == "1" ]]; then
  while master_pid_alive; do
    log "SVD ready; waiting for master process to exit before hook"
    sleep "$POLL_SECONDS"
  done
fi

if [[ "$GRACE_SECONDS" != "0" ]]; then
  log "grace wait ${GRACE_SECONDS}s to avoid racing the main driver hook"
  sleep "$GRACE_SECONDS"
fi

if hook_done && [[ "$FORCE" != "1" ]]; then
  log "biased-direction hook completed by another process during grace wait; exiting"
  exit 0
fi

while analysis_process_running; do
  log "biased-direction analysis process already running; waiting"
  sleep "$POLL_SECONDS"
  if hook_done && [[ "$FORCE" != "1" ]]; then
    log "biased-direction hook complete after external analysis process; exiting"
    exit 0
  fi
done

log "biased-direction analysis start"
"$PY" "$ROOT/tools/analyze_burgers_biased_local_attack_direction_20260611.py" \
  --svd-root "$SVD_OUT_ROOT" \
  --out-dir "$OUT_DIR" \
  --report-md "$REPORT_MD" \
  --device "$DEVICE" \
  2>&1 | tee -a "$LOG"
log "biased-direction analysis finished"

if hook_done; then
  touch "$OUT_DIR/.biased_direction_hook_done"
  upload_path "$(repo_rel "$OUT_DIR")"
  upload_path "$(repo_rel "$REPORT_MD")"
  upload_path "$(repo_rel "$LOG")"
  log "watcher complete"
else
  log "ERROR: biased-direction hook did not produce all expected outputs"
  exit 1
fi

