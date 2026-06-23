#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/workspace/NeuralOperatorRobustness2}"
PY="${PY:-/venv/adv_robust/bin/python}"
OUT="${OUT:-${ROOT}/analysis_outputs/mechanism_20260622/full_mechanism_validation/boundary_volume_probe_20260623}"
LOG_DIR="${LOG_DIR:-${ROOT}/analysis_outputs/mechanism_20260622/full_mechanism_validation/run_logs}"

mkdir -p "${LOG_DIR}" "${OUT}"
cd "${ROOT}"

echo "[boundary-volume] $(date -u +%Y-%m-%dT%H:%M:%SZ) start"
"${PY}" -u tools/probe_loss3_boundary_volume_20260623.py \
  --systems burgers ns2d \
  --out-dir "${OUT}" \
  --burgers-num-samples "${BURGERS_NUM_SAMPLES:-5}" \
  --ns-num-samples "${NS_NUM_SAMPLES:-2}" \
  --global-samples "${GLOBAL_SAMPLES:-32}" \
  --cap-samples "${CAP_SAMPLES:-8}" \
  --angles ${ANGLES:-0 0.05 0.10 0.20 0.40 0.80} \
  --taus ${TAUS:-0.90 0.95 0.99} \
  --endpoint-methods steepest_add steepest_replace \
  --connectivity-pairs steepest_replace__steepest_add \
  --connectivity-points "${CONNECTIVITY_POINTS:-9}" \
  --seed "${SEED:-20260623}" \
  --device "${DEVICE:-cuda}"
echo "[boundary-volume] $(date -u +%Y-%m-%dT%H:%M:%SZ) complete"
