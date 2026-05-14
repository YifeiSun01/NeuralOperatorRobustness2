#!/usr/bin/env bash
set -euo pipefail

# Run the corrected batch-100 three-loss attack sweep.
#
# This script runs one setting at a time.  After each setting finishes, the
# Python wrapper automatically regenerates the mean/std loss curves and the
# dataset-index-0 loss curves.
#
# Important experiment rules:
#   - batch initial conditions are fixed to dataset indices 0..99
#   - loss1 uses one tiny random initial delta, shared by all methods/variants
#   - loss1 zero-initialized runs are intentionally not run
#   - loss2 uses fixed g(x), so the solver target is not recomputed per step
#   - loss3 uses full g(x+delta) with solver gradients, so the gradient contains J_f - J_g
#   - every run records per-step loss curves plus final-only diagnostics:
#       final delta p-norm
#       final 9 objective values
#       boundary-rescaled delta p-norm
#       boundary-rescaled 9 objective values

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-./adv_robust/bin/python}"
GPU_ID="${GPU_ID:-0}"

BATCH_SIZE="${BATCH_SIZE:-100}"
START_INDEX="${START_INDEX:-0}"
STEPS="${STEPS:-100}"
P_NORM="${P_NORM:-2}"
Q_NORM="${Q_NORM:-2}"
ETA="${ETA:-1e-6}"
REGULARIZATION_C="${REGULARIZATION_C:-1.0}"
SEED="${SEED:-0}"
LOSS1_RANDOM_START_SCALE="${LOSS1_RANDOM_START_SCALE:-1e-6}"

OUT_PREFIX="${OUT_PREFIX:-results/three_loss_batch100_full_loss3}"
LOG_DIR="${LOG_DIR:-logs/three_loss_batch100_full_loss3_sweep}"
mkdir -p "${LOG_DIR}"

# name epsilon alpha
CONFIGS=(
  "eps8_alpha0p3 8 0.3"
  "eps8_alpha1p5 8 1.5"
  "eps8_alpha3p0 8 3.0"
  "eps16_alpha0p3 16 0.3"
  "eps40_alpha0p3 40 0.3"
  "eps16_alpha1p5 16 1.5"
  "eps4_alpha0p3 4 0.3"
  "eps1p6_alpha0p3 1.6 0.3"
)

echo "[info] repo root: ${REPO_ROOT}"
echo "[info] python: ${PYTHON_BIN}"
echo "[info] GPU_ID: ${GPU_ID}"
echo "[info] batch indices: ${START_INDEX}..$((START_INDEX + BATCH_SIZE - 1))"
echo "[info] steps=${STEPS}, p=${P_NORM}, q=${Q_NORM}, eta=${ETA}, C=${REGULARIZATION_C}"
echo "[info] output prefix: ${OUT_PREFIX}"
echo "[info] logs: ${LOG_DIR}"

for cfg in "${CONFIGS[@]}"; do
  read -r name epsilon alpha <<< "${cfg}"
  out_root="${OUT_PREFIX}_${name}_final_boundary"
  log_path="${LOG_DIR}/${name}.log"

  echo
  echo "[run] ${name}: epsilon=${epsilon}, alpha=${alpha}"
  echo "[run] output: ${out_root}"
  echo "[run] log: ${log_path}"

  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" tools/run_and_plot_batch_three_loss_loss_only.py \
    --out-root "${out_root}" \
    --losses loss1 loss2 loss3 \
    --batch-size "${BATCH_SIZE}" \
    --start-index "${START_INDEX}" \
    --epsilon "${epsilon}" \
    --alpha "${alpha}" \
    --steps "${STEPS}" \
    --p "${P_NORM}" \
    --q "${Q_NORM}" \
    --eta "${ETA}" \
    --regularization-c "${REGULARIZATION_C}" \
    --seed "${SEED}" \
    --loss1-initial-delta random \
    --random-start-scale "${LOSS1_RANDOM_START_SCALE}" \
    --device cuda \
    2>&1 | tee "${log_path}"

  echo "[done] ${name}"
done

echo
echo "[all done] batch-100 full-loss3 sweep finished."
