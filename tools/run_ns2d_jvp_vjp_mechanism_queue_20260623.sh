#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/workspace/NeuralOperatorRobustness2}"
PY="${PY:-/venv/adv_robust/bin/python}"
OUT="${OUT:-${ROOT}/analysis_outputs/mechanism_20260622/full_mechanism_validation}"
LOG_DIR="${LOG_DIR:-${OUT}/run_logs}"
EXTENDED_PID_FILE="${EXTENDED_PID_FILE:-${LOG_DIR}/extended_mechanism_queue_20260623.pid}"

mkdir -p "${LOG_DIR}"
cd "${ROOT}"

wait_for_pid() {
  local pid="$1"
  local label="$2"
  if [[ -z "${pid}" ]]; then
    return 0
  fi
  while kill -0 "${pid}" 2>/dev/null; do
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] waiting for ${label} PID ${pid}"
    sleep 60
  done
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] ${label} PID ${pid} is no longer running"
}

if [[ -n "${WAIT_PID:-}" ]]; then
  wait_for_pid "${WAIT_PID}" "upstream queue"
elif [[ -f "${EXTENDED_PID_FILE}" ]]; then
  wait_for_pid "$(tr -d '[:space:]' < "${EXTENDED_PID_FILE}")" "extended mechanism queue"
fi

run_probe() {
  local root_dir="$1"
  local out_dir="$2"
  local label="$3"
  if [[ ! -d "${root_dir}" ]]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] skip ${label}: missing root ${root_dir}"
    return 0
  fi
  if ! find "${root_dir}" -path '*step_sample_trace.npz' -print -quit | grep -q .; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] skip ${label}: no step_sample_trace.npz under ${root_dir}"
    return 0
  fi
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] start ${label}"
  "${PY}" -u tools/probe_ns2d_jvp_vjp_spectrum_20260623.py \
    --root "${root_dir}" \
    --out-dir "${out_dir}" \
    --methods steepest_add steepest_replace \
    --ks 0 1 10 50 100 \
    --power-iters 4 \
    --surrogate-directions top_singular grad saved_direction final_delta random \
    --radius-fractions 0.03 0.1 0.3 1.0 \
    --jvp-mode auto \
    --fd-step 0.01 \
    --project-candidates \
    --compute-grad-cosines \
    --eval-chunk 1
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] done ${label}"
}

run_probe \
  "${ROOT}/analysis_outputs/mechanism_20260622/ns2d/ns2d_eps32_alpha10_steps100_N3_trace" \
  "${OUT}/ns2d_jvp_vjp_spectrum_N3" \
  "NS2D N=3 JVP/VJP spectrum"

run_probe \
  "${OUT}/ns2d/ns2d_eps32_alpha10_steps100_N2_topup_trace" \
  "${OUT}/ns2d_jvp_vjp_spectrum_N2_topup" \
  "NS2D N=2 top-up JVP/VJP spectrum"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] NS2D JVP/VJP mechanism queue completed"
