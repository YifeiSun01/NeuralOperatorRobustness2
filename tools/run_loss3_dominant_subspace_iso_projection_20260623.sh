#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY="${PY:-/venv/adv_robust/bin/python}"
OUT_DIR="${OUT_DIR:-analysis_outputs/mechanism_20260622/full_mechanism_validation/dominant_subspace_iso_projection_20260623}"

mkdir -p "${OUT_DIR}/logs"

exec "${PY}" -u tools/probe_loss3_dominant_subspace_iso_projection_20260623.py \
  --systems ns2d burgers \
  --burgers-num-samples "${BURGERS_NUM_SAMPLES:-5}" \
  --ns-num-samples "${NS_NUM_SAMPLES:-2}" \
  --random-draws "${RANDOM_DRAWS:-16}" \
  --ks ${KS:-1 2 4 8 16} \
  --betas ${BETAS:-0 0.25 0.5 1 2 4} \
  --out-dir "${OUT_DIR}" \
  --device "${DEVICE:-cuda}"
