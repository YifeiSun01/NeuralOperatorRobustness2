#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/workspace/NeuralOperatorRobustness2"
OUT="${ROOT}/analysis_outputs/optimizer_ablation_20260622"
LOGDIR="${OUT}/run_logs"
QUEUE_SCRIPT="${ROOT}/tools/run_optimizer_ablation_queue_20260622.sh"
QUEUE_PID_FILE="${LOGDIR}/optimizer_ablation_queue_20260622.pid"
WATCH_PID_FILE="${LOGDIR}/optimizer_ablation_watchdog_20260622.pid"
WATCH_LOG="${LOGDIR}/optimizer_ablation_watchdog_20260622.log"
DONE_FILE="${LOGDIR}/optimizer_ablation_queue_20260622.done"

mkdir -p "${LOGDIR}"

log() {
  printf '[optimizer-watchdog] %s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "${WATCH_LOG}"
}

pid_alive() {
  local pid="$1"
  [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null
}

queue_pid() {
  if [[ -f "${QUEUE_PID_FILE}" ]]; then
    tr -cd '0-9' < "${QUEUE_PID_FILE}"
  fi
}

optimizer_attack_running() {
  pgrep -af 'attack_ns2d_recurrent_core4.py|run_loss3_direction_proposal_ablation.py|attack_darcy_binary_loss_method_experiments.py' \
    | grep -q 'optimizer_ablation_20260622'
}

start_queue() {
  log "starting optimizer-ablation queue via setsid"
  (
    cd "${ROOT}"
    setsid bash "${QUEUE_SCRIPT}" > "${LOGDIR}/optimizer_ablation_queue_20260622.nohup.log" 2>&1 < /dev/null &
    echo $!
  ) >> "${WATCH_LOG}" 2>&1
}

main() {
  echo "$$" > "${WATCH_PID_FILE}"
  log "watchdog launched"
  while true; do
    if [[ -f "${DONE_FILE}" ]]; then
      log "done file exists; watchdog exiting"
      exit 0
    fi
    pid="$(queue_pid || true)"
    if pid_alive "${pid}"; then
      log "queue alive pid=${pid}"
    elif optimizer_attack_running; then
      log "queue pid not alive, but optimizer-ablation attack subprocess is still running; waiting"
    else
      log "queue not alive and no optimizer-ablation subprocess detected"
      start_queue
    fi
    sleep 3600
  done
}

main "$@"
