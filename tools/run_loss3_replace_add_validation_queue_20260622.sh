#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/workspace/NeuralOperatorRobustness2}"
PY="${PY:-/venv/adv_robust/bin/python}"
OUT="${OUT:-${ROOT}/analysis_outputs/mechanism_20260622/replace_add_validation}"
LOGDIR="${OUT}/run_logs"
WAIT_PID="${WAIT_PID:-}"
NS_ROOT="${NS_ROOT:-${ROOT}/analysis_outputs/mechanism_20260622/ns2d/ns2d_eps32_alpha10_steps100_N3_trace}"

BURGERS_TRACE_ROOT="${OUT}/raw/burgers_eps8_alpha0p3_steps100_N5_trace"
BURGERS_LANDSCAPE_OUT="${OUT}/diagnostics/burgers_landscape_ridge_probe"
BURGERS_LANDSCAPE_DOC="${ROOT}/docs/loss3_burgers_replace_add_landscape_probe_20260622.md"
BURGERS_PRED_OUT="${OUT}/diagnostics/burgers_first_order_prediction"
NS_DIAG_OUT="${OUT}/diagnostics/ns2d_direction_stability"

LOG="${LOGDIR}/replace_add_validation_queue_20260622.log"
PID_FILE="${LOGDIR}/replace_add_validation_queue_20260622.pid"
DONE_FILE="${LOGDIR}/replace_add_validation_queue_20260622.done"

cd "${ROOT}"
mkdir -p "${LOGDIR}" "${OUT}/raw" "${OUT}/diagnostics"
exec >> "${LOG}" 2>&1
echo "$$" > "${PID_FILE}"
rm -f "${DONE_FILE}"

log() {
  printf '[replace-add-validation] %s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

gpu_snapshot() {
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits || true
}

trap 'rc=$?; log "ERROR rc=${rc} line=${LINENO}: ${BASH_COMMAND}"; exit "${rc}"' ERR

wait_for_pid() {
  if [[ -z "${WAIT_PID}" ]]; then
    log "no WAIT_PID specified; starting immediately"
    return 0
  fi
  if ! kill -0 "${WAIT_PID}" 2>/dev/null; then
    log "WAIT_PID=${WAIT_PID} is not alive; starting immediately"
    return 0
  fi
  log "waiting for existing NS mechanism pid=${WAIT_PID}"
  while kill -0 "${WAIT_PID}" 2>/dev/null; do
    gpu_snapshot
    sleep 120
  done
  log "WAIT_PID=${WAIT_PID} finished; continuing validation queue"
}

run_burgers_trace() {
  if [[ -f "${BURGERS_TRACE_ROOT}/manifest.json" ]] && rg -q '"status": "completed"' "${BURGERS_TRACE_ROOT}/manifest.json"; then
    log "skip Burgers trace: completed manifest exists at ${BURGERS_TRACE_ROOT}"
    return 0
  fi
  log "start Burgers N=5 core4 trajectory run"
  gpu_snapshot
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.55 \
  "${PY}" -u tools/run_loss3_direction_proposal_ablation.py \
    --out-root "${BURGERS_TRACE_ROOT}" \
    --methods raw_add raw_replace steepest_add steepest_replace \
    --batch-size 5 \
    --start-index 0 \
    --epsilon 8 \
    --alpha 0.3 \
    --steps 100 \
    --p 2 \
    --q 2 \
    --seed 0 \
    --device cuda \
    --trajectory-indices 0 1 2 3 4 \
    --selected-steps 0 1 2 5 10 20 50 100 \
    --save-delta-trajectory \
    --save-trajectory-final-conditions \
    --no-plots
  log "done Burgers N=5 core4 trajectory run"
}

run_burgers_first_order_prediction() {
  log "start Burgers first-order prediction probe"
  gpu_snapshot
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.55 \
  "${PY}" -u tools/probe_burgers_first_order_prediction_20260622.py \
    --run-root "${BURGERS_TRACE_ROOT}" \
    --out-dir "${BURGERS_PRED_OUT}" \
    --methods raw_add raw_replace steepest_add steepest_replace \
    --sample-ks 0 1 2 5 10 20 50 100 \
    --device cuda
  log "done Burgers first-order prediction probe"
}

run_burgers_landscape_probe() {
  log "start Burgers landscape/ridge probe"
  gpu_snapshot
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.55 \
  "${PY}" -u tools/run_loss3_core4_pq_landscape_probe.py \
    --setting-root "${BURGERS_TRACE_ROOT}" \
    --out-dir "${BURGERS_LANDSCAPE_OUT}" \
    --doc "${BURGERS_LANDSCAPE_DOC}" \
    --sample-indices 0 1 2 3 4 \
    --methods raw_add raw_replace steepest_add steepest_replace \
    --early-steps 1 5 10 20 \
    --num-radii 17 \
    --arc-points 17 \
    --slice-grid 5 \
    --device cuda
  log "done Burgers landscape/ridge probe"
}

run_ns_direction_diagnostics() {
  if [[ ! -d "${NS_ROOT}" ]]; then
    log "skip NS diagnostics: NS_ROOT missing: ${NS_ROOT}"
    return 0
  fi
  log "start NS direction-stability diagnostics"
  "${PY}" tools/analyze_step_sample_direction_stability_20260622.py \
    --root "${NS_ROOT}" \
    --out-dir "${NS_DIAG_OUT}"
  log "done NS direction-stability diagnostics"
}

main() {
  log "queue start"
  wait_for_pid
  run_burgers_trace
  run_burgers_first_order_prediction
  run_burgers_landscape_probe
  run_ns_direction_diagnostics
  printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${DONE_FILE}"
  log "queue complete"
}

main "$@"

