#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/workspace/NeuralOperatorRobustness2}"
WAIT_PID="${WAIT_PID:-142167}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MAX_POST_PID_SUMMARY_WAIT_SECONDS="${MAX_POST_PID_SUMMARY_WAIT_SECONDS:-1800}"
WAIT_GPU_FREE="${WAIT_GPU_FREE:-1}"
GPU_UTIL_MAX="${GPU_UTIL_MAX:-25}"
GPU_MEMORY_MAX_MIB="${GPU_MEMORY_MAX_MIB:-4096}"
GPU_FREE_CONSECUTIVE_POLLS="${GPU_FREE_CONSECUTIVE_POLLS:-2}"
LOG_DIR="${LOG_DIR:-$ROOT/adversarial_training_runs/burgers_wideparam_random_field_training_20260612_logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/watch_retrain_then_random_field_20260612.log}"
RANDOM_SCRIPT="${RANDOM_SCRIPT:-$ROOT/tools/run_burgers_wideparam_random_field_training_20260612.sh}"
REQUIRE_RETRAIN_SUMMARIES="${REQUIRE_RETRAIN_SUMMARIES:-1}"
START_RANDOM_FIELD="${START_RANDOM_FIELD:-1}"

mkdir -p "$LOG_DIR"
cd "$ROOT"

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG_FILE"
}

summary_paths=(
  "$ROOT/adversarial_training_runs/burgers_wideparam_loss1_8000ep_retrain_20260611/burgers/summary.json"
  "$ROOT/adversarial_training_runs/burgers_wideparam_loss2_2000ep_retrain_20260611/burgers/summary.json"
  "$ROOT/adversarial_training_runs/burgers_wideparam_loss3_1000ep_retrain_20260611/burgers/summary.json"
)

all_summaries_exist() {
  local path
  for path in "${summary_paths[@]}"; do
    [[ -f "$path" ]] || return 1
  done
  return 0
}

wait_for_pid_exit() {
  if [[ -z "$WAIT_PID" || "$WAIT_PID" == "0" || "$WAIT_PID" == "none" ]]; then
    log "skip pid wait WAIT_PID=$WAIT_PID"
    return 0
  fi
  if ! kill -0 "$WAIT_PID" 2>/dev/null; then
    log "pid $WAIT_PID is already absent"
    return 0
  fi
  log "waiting for upstream workflow pid=$WAIT_PID"
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep "$POLL_SECONDS"
  done
  log "upstream workflow pid=$WAIT_PID exited"
}

wait_for_summaries() {
  if [[ "$REQUIRE_RETRAIN_SUMMARIES" != "1" ]]; then
    log "skip retrain summary requirement"
    return 0
  fi
  local waited=0
  while ! all_summaries_exist; do
    if (( waited >= MAX_POST_PID_SUMMARY_WAIT_SECONDS )); then
      log "refuse random-field start: retrain summaries still missing after ${waited}s"
      for path in "${summary_paths[@]}"; do
        [[ -f "$path" ]] && log "summary ok: $path" || log "summary missing: $path"
      done
      exit 1
    fi
    for path in "${summary_paths[@]}"; do
      [[ -f "$path" ]] || log "waiting for missing summary: $path"
    done
    sleep "$POLL_SECONDS"
    waited=$((waited + POLL_SECONDS))
  done
  log "all retrain summaries present"
}

wait_for_gpu_free() {
  if [[ "$WAIT_GPU_FREE" != "1" ]]; then
    log "skip gpu-free wait"
    return 0
  fi
  local good=0
  local util mem
  log "waiting for GPU free: util<=${GPU_UTIL_MAX}%, memory<=${GPU_MEMORY_MAX_MIB}MiB, consecutive=${GPU_FREE_CONSECUTIVE_POLLS}"
  while true; do
    IFS=',' read -r util mem < <(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
    util="${util:-999}"
    mem="${mem:-999999}"
    log "gpu status util=${util}% memory=${mem}MiB"
    if (( util <= GPU_UTIL_MAX && mem <= GPU_MEMORY_MAX_MIB )); then
      good=$((good + 1))
    else
      good=0
    fi
    if (( good >= GPU_FREE_CONSECUTIVE_POLLS )); then
      log "gpu is free enough"
      return 0
    fi
    sleep "$POLL_SECONDS"
  done
}

start_random_field() {
  if [[ "$START_RANDOM_FIELD" != "1" ]]; then
    log "START_RANDOM_FIELD=$START_RANDOM_FIELD, not launching random-field script"
    return 0
  fi
  if [[ ! -x "$RANDOM_SCRIPT" ]]; then
    log "random-field script is not executable: $RANDOM_SCRIPT"
    exit 1
  fi
  log "starting random-field training script: $RANDOM_SCRIPT"
  bash "$RANDOM_SCRIPT" 2>&1 | tee -a "$LOG_FILE"
  log "random-field training script finished"
}

log "watcher start root=$ROOT wait_pid=$WAIT_PID random_script=$RANDOM_SCRIPT"
wait_for_pid_exit
wait_for_summaries
wait_for_gpu_free
start_random_field
log "watcher finished"
