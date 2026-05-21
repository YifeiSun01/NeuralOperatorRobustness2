#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON_BIN="${PYTHON_BIN:-adv_robust/bin/python}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to the trained FNO2d checkpoint path before running.}"

# Keep JAX memory lazy. Do not set XLA_PYTHON_CLIENT_ALLOCATOR=platform by
# default because it can reduce cached memory further but often slows runs.
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.40}"

EXTRA_ARGS=()
if [[ "${EMPTY_TORCH_CACHE_AFTER_BATCH:-1}" == "1" ]]; then
  EXTRA_ARGS+=(--empty-torch-cache-after-batch)
else
  EXTRA_ARGS+=(--no-empty-torch-cache-after-batch)
fi
if [[ "${EMPTY_TORCH_CACHE_AFTER_METHOD:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--empty-torch-cache-after-method)
fi
if [[ "${CLEAR_JAX_CACHES_AFTER_BATCH:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--clear-jax-caches-after-batch)
fi
if [[ -n "${SAVE_STEPS:-}" ]]; then
  read -r -a SAVE_STEP_ARGS <<< "${SAVE_STEPS}"
  EXTRA_ARGS+=(--save-steps "${SAVE_STEP_ARGS[@]}")
fi

"${PYTHON_BIN}" 2D_NS_FNO2d_recurrent/perturbation_methods/attack_ns2d_recurrent_core4.py \
  --checkpoint "${CHECKPOINT}" \
  --indices "${INDICES:-0,1,2,3,4}" \
  --attack-batch-size "${ATTACK_BATCH_SIZE:-1}" \
  --loss-types ${LOSS_TYPES:-loss1 loss2 loss3} \
  --methods ${METHODS:-raw_add raw_replace steepest_add steepest_replace} \
  --mode-spec "${MODE_SPEC:-all_w}" \
  --steps "${STEPS:-50}" \
  --epsilon "${EPSILON:-8.0}" \
  --alpha "${ALPHA:-0.3}" \
  --p "${P_ORDER:-2}" \
  --q "${Q_ORDER:-2}" \
  --solver-remat "${SOLVER_REMAT:-micro}" \
  --solver-remat-chunk-steps "${SOLVER_REMAT_CHUNK_STEPS:-20}" \
  --dictionary-chunk-size "${DICTIONARY_CHUNK_SIZE:-32}" \
  --clean-target-source "${CLEAN_TARGET_SOURCE:-dataset}" \
  "${EXTRA_ARGS[@]}"
