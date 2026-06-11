#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.45}"

PYTHON_BIN="${PYTHON_BIN:-/venv/adv_robust/bin/python}"

"${PYTHON_BIN}" tools/generate_burgers_semantic_loss3_favored_generalization_20260611.py   --round-id "${ROUND_ID:-1}"   --max-candidates "${MAX_CANDIDATES:-720}"   --samples-per-dataset "${SAMPLES_PER_DATASET:-200}"   --generation-batch-size "${GENERATION_BATCH_SIZE:-200}"   --eval-batch-size "${EVAL_BATCH_SIZE:-256}"   --accept-ratio "${ACCEPT_RATIO:-1.0}"   "$@"
