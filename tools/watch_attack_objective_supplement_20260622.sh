#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/workspace/NeuralOperatorRobustness2"
OUT="${ROOT}/analysis_outputs/attack_objective_true_loss3_comparison_20260622"
LOGDIR="${OUT}/run_logs"
QUEUE_SCRIPT="${ROOT}/tools/run_attack_objective_ns_cflow_supplement_20260622.sh"
QUEUE_PID_FILE="${LOGDIR}/supplement_ns_cflow_physics_20260622.pid"
WATCH_PID_FILE="${LOGDIR}/supplement_ns_cflow_physics_watchdog.pid"
WATCH_LOG="${LOGDIR}/supplement_ns_cflow_physics_watchdog.log"
DONE_FILE="${LOGDIR}/supplement_ns_cflow_physics_20260622.done"

mkdir -p "${LOGDIR}"

log() {
  printf '[watchdog] %s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "${WATCH_LOG}"
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

attack_running() {
  pgrep -f 'attack_ns2d_recurrent_core4.py --out-root .*ns2d_supplement|attack_darcy_binary_physics_loss4.py' >/dev/null 2>&1
}

start_queue() {
  log "starting supplement queue via setsid"
  (
    cd "${ROOT}"
    setsid bash "${QUEUE_SCRIPT}" > "${LOGDIR}/supplement_ns_cflow_physics_20260622.nohup.log" 2>&1 < /dev/null &
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
    elif attack_running; then
      log "queue pid not alive, but an attack subprocess is still running; waiting"
    else
      log "queue not alive and no attack subprocess detected"
      start_queue
    fi
    sleep 3600
  done
}

main "$@"
