#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/workspace/NeuralOperatorRobustness2}"
PY="${PY:-/venv/adv_robust/bin/python}"
OUT="${OUT:-${ROOT}/analysis_outputs/mechanism_20260622/full_mechanism_validation}"
LOGDIR="${OUT}/run_logs"
WAIT_PID="${WAIT_PID:-213652}"

NS_EXISTING_ROOT="${NS_EXISTING_ROOT:-${ROOT}/analysis_outputs/mechanism_20260622/ns2d/ns2d_eps32_alpha10_steps100_N3_trace}"
NS_TOPUP_ROOT="${NS_TOPUP_ROOT:-${OUT}/ns2d/ns2d_eps32_alpha10_steps100_N2_topup_trace}"
NS_CKPT="${NS_CKPT:-${ROOT}/2D_NS_FNO2d_recurrent/saved_models/2D/modes64_modes64_width60_epochs500_Tin10_T10_recurrent_pytorch_20260521_090136_UTC/checkpoints/final.pt}"

BURGERS_TRACE_ROOT="${BURGERS_TRACE_ROOT:-${OUT}/raw/burgers_eps8_alpha0p3_steps100_N20_trace}"
BURGERS_PARSE_ROOT="${BURGERS_PARSE_ROOT:-${OUT}/raw/fno_nu0p001_eps8_alpha0p3_batch20_steps100_p2_q2}"
BURGERS_PRED_OUT="${BURGERS_PRED_OUT:-${OUT}/burgers_first_order_prediction_N20}"
BURGERS_LANDSCAPE_OUT="${BURGERS_LANDSCAPE_OUT:-${OUT}/burgers_landscape_ridge_probe_N20}"
BURGERS_LANDSCAPE_DOC="${BURGERS_LANDSCAPE_DOC:-${ROOT}/docs/loss3_burgers_replace_add_landscape_probe_N20_20260623.md}"

NS_DIR_OUT="${NS_DIR_OUT:-${OUT}/ns2d_direction_stability_N2_topup}"
NS_EXACT_TOPUP_OUT="${NS_EXACT_TOPUP_OUT:-${OUT}/ns2d_exact_mechanism_N2_topup}"
SUMMARY_OUT="${SUMMARY_OUT:-${OUT}/cross_system_summary}"
SUMMARY_DOC="${SUMMARY_DOC:-${ROOT}/docs/loss3_full_mechanism_validation_summary_20260623.md}"

LOG="${LOGDIR}/extended_mechanism_queue_20260623.log"
PID_FILE="${LOGDIR}/extended_mechanism_queue_20260623.pid"
DONE_FILE="${LOGDIR}/extended_mechanism_queue_20260623.done"

cd "${ROOT}"
mkdir -p "${LOGDIR}" "${OUT}/raw" "${OUT}/ns2d"
exec >> "${LOG}" 2>&1
echo "$$" > "${PID_FILE}"
rm -f "${DONE_FILE}"

log() {
  printf '[extended-mechanism] %s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

gpu_snapshot() {
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits || true
}

manifest_completed() {
  local path="$1"
  [[ -f "${path}" ]] && rg -q '"status"[[:space:]]*:[[:space:]]*"completed"' "${path}"
}

topup_trace_completed() {
  local root="$1"
  [[ -d "${root}" ]] || return 1
  local n
  n="$(find "${root}" -type f -name 'step_sample_trace.npz' 2>/dev/null | wc -l | tr -d ' ')"
  [[ "${n}" -ge 8 ]]
}

trap 'rc=$?; log "ERROR rc=${rc} line=${LINENO}: ${BASH_COMMAND}"; gpu_snapshot; exit "${rc}"' ERR

wait_for_pid() {
  if [[ -z "${WAIT_PID}" ]]; then
    log "no WAIT_PID specified; starting immediately"
    return 0
  fi
  if ! kill -0 "${WAIT_PID}" 2>/dev/null; then
    log "WAIT_PID=${WAIT_PID} is not alive; starting queued work immediately"
    return 0
  fi
  log "waiting for current NS exact mechanism pid=${WAIT_PID}"
  while kill -0 "${WAIT_PID}" 2>/dev/null; do
    gpu_snapshot
    sleep 120
  done
  log "WAIT_PID=${WAIT_PID} finished; continuing queued work"
}

run_burgers_trace_n20() {
  if manifest_completed "${BURGERS_TRACE_ROOT}/manifest.json"; then
    log "skip Burgers N=20 trace: completed manifest exists at ${BURGERS_TRACE_ROOT}"
    return 0
  fi
  log "start Burgers N=20 core4 trace"
  gpu_snapshot
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.55 \
  "${PY}" -u tools/run_loss3_direction_proposal_ablation.py \
    --out-root "${BURGERS_TRACE_ROOT}" \
    --methods raw_add raw_replace steepest_add steepest_replace \
    --batch-size 20 \
    --start-index 0 \
    --epsilon 8 \
    --alpha 0.3 \
    --steps 100 \
    --p 2 \
    --q 2 \
    --seed 0 \
    --device cuda \
    --trajectory-indices 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
    --selected-steps 0 1 2 5 10 20 50 100 \
    --save-delta-trajectory \
    --save-trajectory-final-conditions \
    --no-plots
  log "done Burgers N=20 core4 trace"
}

run_burgers_first_order_n20() {
  if manifest_completed "${BURGERS_PRED_OUT}/manifest.json"; then
    log "skip Burgers N=20 first-order probe: completed manifest exists at ${BURGERS_PRED_OUT}"
    return 0
  fi
  log "start Burgers N=20 first-order/full-budget prediction probe"
  gpu_snapshot
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.55 \
  "${PY}" -u tools/probe_burgers_first_order_prediction_20260622.py \
    --run-root "${BURGERS_TRACE_ROOT}" \
    --out-dir "${BURGERS_PRED_OUT}" \
    --methods raw_add raw_replace steepest_add steepest_replace \
    --sample-ks 0 1 2 5 10 20 50 100 \
    --device cuda
  log "done Burgers N=20 first-order/full-budget prediction probe"
}

run_burgers_landscape_n20() {
  if manifest_completed "${BURGERS_LANDSCAPE_OUT}/manifest.json"; then
    log "skip Burgers N=20 landscape probe: completed manifest exists at ${BURGERS_LANDSCAPE_OUT}"
    return 0
  fi
  if [[ ! -e "${BURGERS_PARSE_ROOT}" ]]; then
    ln -s "${BURGERS_TRACE_ROOT}" "${BURGERS_PARSE_ROOT}"
  fi
  log "start Burgers N=20 landscape/ridge probe"
  gpu_snapshot
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.55 \
  "${PY}" -u tools/run_loss3_core4_pq_landscape_probe.py \
    --setting-root "${BURGERS_PARSE_ROOT}" \
    --out-dir "${BURGERS_LANDSCAPE_OUT}" \
    --doc "${BURGERS_LANDSCAPE_DOC}" \
    --sample-indices 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
    --methods raw_add raw_replace steepest_add steepest_replace \
    --early-steps 1 5 10 20 \
    --num-radii 17 \
    --arc-points 17 \
    --slice-grid 5 \
    --device cuda
  log "done Burgers N=20 landscape/ridge probe"
}

run_ns_topup_trace_n2() {
  if topup_trace_completed "${NS_TOPUP_ROOT}"; then
    log "skip NS2D top-up trace: at least 8 step_sample_trace.npz files already exist under ${NS_TOPUP_ROOT}"
    return 0
  fi
  log "start NS2D N=2 top-up trace for dataset indices 3,4"
  gpu_snapshot
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 \
  "${PY}" -u 2D_NS_FNO2d_recurrent/perturbation_methods/attack_ns2d_recurrent_core4.py \
    --out-root "${NS_TOPUP_ROOT}" \
    --checkpoint "${NS_CKPT}" \
    --start-index 3 \
    --num-samples 2 \
    --attack-batch-size 1 \
    --loss-types loss3 \
    --methods raw_add raw_replace steepest_add steepest_replace \
    --mode-spec all_w \
    --epsilon 32 \
    --alpha 10 \
    --steps 100 \
    --p 2 \
    --q 2 \
    --true-loss-every 1 \
    --record-final-state-outputs \
    --record-step-sample-outputs \
    --record-step-sample-gradients \
    --record-step-sample-every 1 \
    --solver-remat chunk \
    --solver-remat-chunk-steps 20 \
    --dictionary-chunk-size 16 \
    --clear-jax-caches-after-batch
  log "done NS2D N=2 top-up trace"
}

run_ns_topup_direction() {
  if [[ -s "${NS_DIR_OUT}/step_sample_direction_summary.csv" ]]; then
    log "skip NS2D top-up direction diagnostics: summary CSV exists at ${NS_DIR_OUT}"
    return 0
  fi
  log "start NS2D N=2 top-up direction/path stability diagnostics"
  "${PY}" tools/analyze_step_sample_direction_stability_20260622.py \
    --root "${NS_TOPUP_ROOT}" \
    --out-dir "${NS_DIR_OUT}"
  log "done NS2D N=2 top-up direction/path stability diagnostics"
}

run_ns_topup_exact() {
  if manifest_completed "${NS_EXACT_TOPUP_OUT}/manifest.json"; then
    log "skip NS2D N=2 top-up exact probe: completed manifest exists at ${NS_EXACT_TOPUP_OUT}"
    return 0
  fi
  log "start NS2D N=2 top-up exact first-order/linearity/boundary probe"
  gpu_snapshot
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 \
  "${PY}" -u tools/probe_ns2d_loss3_exact_mechanism_20260623.py \
    --root "${NS_TOPUP_ROOT}" \
    --out-dir "${NS_EXACT_TOPUP_OUT}" \
    --methods raw_add raw_replace steepest_add steepest_replace \
    --eval-chunk 1 \
    --no-skip-if-complete
  log "done NS2D N=2 top-up exact first-order/linearity/boundary probe"
}

run_final_summary() {
  log "start final cross-system mechanism summary"
  "${PY}" tools/build_loss3_full_mechanism_validation_summary_20260623.py \
    --out-dir "${SUMMARY_OUT}" \
    --doc "${SUMMARY_DOC}"
  log "done final cross-system mechanism summary"
}

main() {
  log "queue start"
  log "existing NS root: ${NS_EXISTING_ROOT}"
  wait_for_pid
  run_burgers_trace_n20
  run_burgers_first_order_n20
  run_burgers_landscape_n20
  run_ns_topup_trace_n2
  run_ns_topup_direction
  run_ns_topup_exact
  run_final_summary
  printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${DONE_FILE}"
  log "queue complete"
}

main "$@"
