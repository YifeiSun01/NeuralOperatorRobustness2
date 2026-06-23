#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY="${PY:-/venv/adv_robust/bin/python}"
OUT_DIR="${OUT_DIR:-analysis_outputs/mechanism_20260622/full_mechanism_validation/jacobian_mode_causal_ablation_20260623}"

mkdir -p "${OUT_DIR}/logs"

exec "${PY}" -u tools/probe_loss3_jacobian_mode_causal_ablation_20260623.py \
  --systems ns2d burgers \
  --burgers-num-samples "${BURGERS_NUM_SAMPLES:-5}" \
  --ns-num-samples "${NS_NUM_SAMPLES:-2}" \
  --ks ${KS:-1 2 4 8} \
  --power-iters "${POWER_ITERS:-3}" \
  --basis-anchors ${BASIS_ANCHORS:-clean endpoint} \
  --basin-steps "${BASIN_STEPS:-20}" \
  --out-dir "${OUT_DIR}" \
  --device "${DEVICE:-cuda}"
