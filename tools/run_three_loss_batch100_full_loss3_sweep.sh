#!/usr/bin/env bash
set -euo pipefail

# Run the corrected batch-100 three-loss attack sweep.
#
# This script runs one setting at a time.  After each setting finishes, the
# Python wrapper automatically regenerates the mean/std loss curves, the
# dataset-index-0 loss curves, and the final-delta comparison plots.
#
# Important experiment rules:
#   - batch initial conditions are fixed to dataset indices 0..99
#   - loss1 uses one tiny random initial delta, shared by all methods/variants
#   - loss1 zero-initialized runs are intentionally not run
#   - loss2 uses fixed g(x), so the solver target is not recomputed per step
#   - loss3 uses full g(x+delta) with solver gradients, so the gradient contains J_f - J_g
#   - every run records per-step loss curves plus final-only diagnostics:
#       final delta vectors
#       final delta p-norm
#       final 9 objective values
#       boundary-rescaled delta p-norm
#       boundary-rescaled 9 objective values
#   - every setting records pairwise final-delta cosine similarities between
#     the three attack methods for each loss/objective variant

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

# model_key model_kind model_label burgers_nu test_path checkpoint stats_path
MODEL_CONFIGS=(
  "fno fno FNO 0.001 1D_Burgers/datasets/1D/Burgers/batched_exponax_splits/dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45/dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45_test.pt 1D_Burgers/trained_models/attack_ready/burgers_nu0.001_fno1d_500/checkpoints/pytorch_fno1d_500.pt none"
  "deeponet deeponet DeepONet/default-net 0.01 1D_Burgers/datasets/1D/Burgers/batched_exponax_splits/dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.01_t1.0_seed45/dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.01_t1.0_seed45_test.pt deeponet_training_runs/burgers_nu0p01_deeponet_lu_ref_50k/checkpoints/deeponet_burgers_nu0p01.pt deeponet_training_runs/burgers_nu0p01_deeponet_lu_ref_50k/training_logs/output_transform_stats.npz"
)

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

attack_complete() {
  local root="$1"
  [[ -f "${root}/loss3_regularized_generalized_power/loss_stats.csv" ]] &&
    [[ -f "${root}/loss3_regularized_generalized_power/final_delta_summary.json" ]] &&
    [[ -f "${root}/loss3_regularized_generalized_power/final_delta.npz" ]]
}

plots_complete() {
  local root="$1"
  [[ -f "${root}/figures/loss_curves/png/loss1_init_random_original_increment_ratio_regularized_method_curves_mean_std.png" ]] &&
    [[ -f "${root}/figures/loss_curves/png/loss2_original_increment_ratio_regularized_method_curves_mean_std.png" ]] &&
    [[ -f "${root}/figures/loss_curves/png/loss3_original_increment_ratio_regularized_method_curves_mean_std.png" ]] &&
    [[ -f "${root}/figures/loss_curves/index_png/loss1_init_random_index0_original_increment_ratio_regularized_method_curves.png" ]] &&
    [[ -f "${root}/figures/loss_curves/index_png/loss2_index0_original_increment_ratio_regularized_method_curves.png" ]] &&
    [[ -f "${root}/figures/loss_curves/index_png/loss3_index0_original_increment_ratio_regularized_method_curves.png" ]] &&
    [[ -f "${root}/figures/final_delta_comparison/png/loss1_init_random_final_delta_comparison_index0.png" ]] &&
    [[ -f "${root}/figures/final_delta_comparison/png/loss2_final_delta_comparison_index0.png" ]] &&
    [[ -f "${root}/figures/final_delta_comparison/png/loss3_final_delta_comparison_index0.png" ]] &&
    [[ -f "${root}/final_delta_comparisons/cosine_similarity_summary.csv" ]]
}

echo "[info] repo root: ${REPO_ROOT}"
echo "[info] python: ${PYTHON_BIN}"
echo "[info] GPU_ID: ${GPU_ID}"
echo "[info] batch indices: ${START_INDEX}..$((START_INDEX + BATCH_SIZE - 1))"
echo "[info] steps=${STEPS}, p=${P_NORM}, q=${Q_NORM}, eta=${ETA}, C=${REGULARIZATION_C}"
echo "[info] output prefix: ${OUT_PREFIX}"
echo "[info] logs: ${LOG_DIR}"

for model_cfg in "${MODEL_CONFIGS[@]}"; do
  read -r model_key model_kind model_label burgers_nu test_path checkpoint stats_path <<< "${model_cfg}"
  for cfg in "${CONFIGS[@]}"; do
    read -r name epsilon alpha <<< "${cfg}"
    out_root="${OUT_PREFIX}_${model_key}_${name}_final_boundary"
    log_path="${LOG_DIR}/${model_key}_${name}.log"

    echo
    echo "[run] model=${model_label}, ${name}: nu=${burgers_nu}, epsilon=${epsilon}, alpha=${alpha}"
    echo "[run] output: ${out_root}"
    echo "[run] log: ${log_path}"

    if [[ "${FORCE_PLOTS:-0}" != "1" ]] && attack_complete "${out_root}" && plots_complete "${out_root}"; then
      echo "[skip] complete: ${out_root}"
      continue
    fi

    cmd=(
      "${PYTHON_BIN}" tools/run_and_plot_batch_three_loss_loss_only.py
      --out-root "${out_root}"
      --losses loss1 loss2 loss3
      --batch-size "${BATCH_SIZE}"
      --start-index "${START_INDEX}"
      --epsilon "${epsilon}"
      --alpha "${alpha}"
      --steps "${STEPS}"
      --p "${P_NORM}"
      --q "${Q_NORM}"
      --eta "${ETA}"
      --regularization-c "${REGULARIZATION_C}"
      --seed "${SEED}"
      --loss1-initial-delta random
      --random-start-scale "${LOSS1_RANDOM_START_SCALE}"
      --device cuda
      --model-kind "${model_kind}"
      --model-label "${model_label}"
      --burgers-test-path "${test_path}"
      --burgers-nu "${burgers_nu}"
    )
    if attack_complete "${out_root}"; then
      echo "[resume] attack complete; regenerating plots only for ${out_root}"
      cmd+=(--skip-attack)
    fi
    if [[ "${model_kind}" == "fno" ]]; then
      cmd+=(--burgers-torch-checkpoint "${checkpoint}")
    else
      cmd+=(--deeponet-checkpoint "${checkpoint}" --deeponet-output-transform-stats "${stats_path}")
    fi

    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${cmd[@]}" 2>&1 | tee "${log_path}"
    echo "[done] model=${model_label}, ${name}"
  done
done

echo
echo "[all done] batch-100 full-loss3 FNO + DeepONet/default-net sweep finished."
