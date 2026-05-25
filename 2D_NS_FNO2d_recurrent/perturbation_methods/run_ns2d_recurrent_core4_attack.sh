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

if [[ -n "${EPSILON_ALPHA_PAIRS:-}" ]]; then
  read -r -a EPSILON_ALPHA_PAIR_ARGS <<< "${EPSILON_ALPHA_PAIRS}"
  EXTRA_ARGS+=(--epsilon-alpha-pairs "${EPSILON_ALPHA_PAIR_ARGS[@]}")
else
  if [[ -n "${EPSILONS:-}" ]]; then
    read -r -a EPSILON_ARGS <<< "${EPSILONS}"
    EXTRA_ARGS+=(--epsilons "${EPSILON_ARGS[@]}")
  else
    EXTRA_ARGS+=(--epsilon "${EPSILON:-8.0}")
  fi
  if [[ -n "${ALPHAS:-}" ]]; then
    read -r -a ALPHA_ARGS <<< "${ALPHAS}"
    EXTRA_ARGS+=(--alphas "${ALPHA_ARGS[@]}")
  else
    EXTRA_ARGS+=(--alpha "${ALPHA:-0.3}")
  fi
fi
if [[ "${LOSS1_RANDOM_START:-1}" == "1" ]]; then
  EXTRA_ARGS+=(--loss1-random-start)
else
  EXTRA_ARGS+=(--no-loss1-random-start)
fi
EXTRA_ARGS+=(--loss1-random-start-fraction "${LOSS1_RANDOM_START_FRACTION:-0.001}")
EXTRA_ARGS+=(--loss1-random-start-seed "${LOSS1_RANDOM_START_SEED:-12345}")

if [[ "${RECORD_FINAL_STATE_OUTPUTS:-1}" == "1" ]]; then
  EXTRA_ARGS+=(--record-final-state-outputs)
else
  EXTRA_ARGS+=(--no-record-final-state-outputs)
fi
if [[ "${RECORD_STEP_SAMPLE_OUTPUTS:-1}" == "1" ]]; then
  EXTRA_ARGS+=(--record-step-sample-outputs)
else
  EXTRA_ARGS+=(--no-record-step-sample-outputs)
fi
EXTRA_ARGS+=(--record-step-sample-position "${RECORD_STEP_SAMPLE_POSITION:-0}")
EXTRA_ARGS+=(--record-step-sample-every "${RECORD_STEP_SAMPLE_EVERY:-1}")
if [[ "${RECORD_STEP_SAMPLE_GRADIENTS:-1}" == "1" ]]; then
  EXTRA_ARGS+=(--record-step-sample-gradients)
else
  EXTRA_ARGS+=(--no-record-step-sample-gradients)
fi

"${PYTHON_BIN}" 2D_NS_FNO2d_recurrent/perturbation_methods/attack_ns2d_recurrent_core4.py \
  --checkpoint "${CHECKPOINT}" \
  --test-path "${TEST_PATH:-2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/real_initial_laxmap_single/test/dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_test_ntimepoints21_all_frames.pt}" \
  --dictionary-path "${DICTIONARY_PATH:-2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dictionary/dim2d_nx256_N2000_solver=exponax_nu0.000_t20.0_dict_ntimepoints21_batch0_all_frames.pt}" \
  --out-root "${OUT_ROOT:-2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack}" \
  --indices "${INDICES:-0,1,2,3,4}" \
  --attack-batch-size "${ATTACK_BATCH_SIZE:-1}" \
  --loss-types ${LOSS_TYPES:-loss1 loss2 loss3} \
  --loss3-metric "${LOSS3_METRIC:-qnorm}" \
  --loss3-image-normalization "${LOSS3_IMAGE_NORMALIZATION:-pair_minmax_detached}" \
  --loss3-metric-eps "${LOSS3_METRIC_EPS:-1e-6}" \
  --loss3-scattering-j "${LOSS3_SCATTERING_J:-3}" \
  --loss3-align-objective "${LOSS3_ALIGN_OBJECTIVE:-l2}" \
  --loss3-affine-inner-steps "${LOSS3_AFFINE_INNER_STEPS:-8}" \
  --loss3-affine-lr "${LOSS3_AFFINE_LR:-0.05}" \
  --loss3-affine-max-shift-ratio "${LOSS3_AFFINE_MAX_SHIFT_RATIO:-0.05}" \
  --loss3-affine-max-angle-deg "${LOSS3_AFFINE_MAX_ANGLE_DEG:-10.0}" \
  --loss3-affine-max-log-scale "${LOSS3_AFFINE_MAX_LOG_SCALE:-0.09531017980432493}" \
  --loss3-affine-reg-weight "${LOSS3_AFFINE_REG_WEIGHT:-0.01}" \
  --loss3-local-grid-size "${LOSS3_LOCAL_GRID_SIZE:-8}" \
  --loss3-local-inner-steps "${LOSS3_LOCAL_INNER_STEPS:-8}" \
  --loss3-local-lr "${LOSS3_LOCAL_LR:-0.05}" \
  --loss3-local-max-disp-ratio "${LOSS3_LOCAL_MAX_DISP_RATIO:-0.03}" \
  --loss3-local-mag-weight "${LOSS3_LOCAL_MAG_WEIGHT:-0.01}" \
  --loss3-local-smooth-weight "${LOSS3_LOCAL_SMOOTH_WEIGHT:-0.05}" \
  --loss3-homography-inner-steps "${LOSS3_HOMOGRAPHY_INNER_STEPS:-8}" \
  --loss3-homography-lr "${LOSS3_HOMOGRAPHY_LR:-0.05}" \
  --loss3-homography-max-corner-ratio "${LOSS3_HOMOGRAPHY_MAX_CORNER_RATIO:-0.05}" \
  --loss3-homography-reg-weight "${LOSS3_HOMOGRAPHY_REG_WEIGHT:-0.01}" \
  --loss3-tps-grid-size "${LOSS3_TPS_GRID_SIZE:-4}" \
  --loss3-tps-inner-steps "${LOSS3_TPS_INNER_STEPS:-8}" \
  --loss3-tps-lr "${LOSS3_TPS_LR:-0.05}" \
  --loss3-tps-max-disp-ratio "${LOSS3_TPS_MAX_DISP_RATIO:-0.05}" \
  --loss3-tps-offset-weight "${LOSS3_TPS_OFFSET_WEIGHT:-0.01}" \
  --loss3-tps-smooth-weight "${LOSS3_TPS_SMOOTH_WEIGHT:-0.05}" \
  --loss3-elastic-grid-size "${LOSS3_ELASTIC_GRID_SIZE:-16}" \
  --loss3-elastic-inner-steps "${LOSS3_ELASTIC_INNER_STEPS:-8}" \
  --loss3-elastic-lr "${LOSS3_ELASTIC_LR:-0.05}" \
  --loss3-elastic-max-disp-ratio "${LOSS3_ELASTIC_MAX_DISP_RATIO:-0.05}" \
  --loss3-elastic-smooth-kernel "${LOSS3_ELASTIC_SMOOTH_KERNEL:-9}" \
  --loss3-elastic-smooth-passes "${LOSS3_ELASTIC_SMOOTH_PASSES:-2}" \
  --loss3-elastic-mag-weight "${LOSS3_ELASTIC_MAG_WEIGHT:-0.01}" \
  --loss3-elastic-smooth-weight "${LOSS3_ELASTIC_SMOOTH_WEIGHT:-0.03}" \
  --loss3-svf-grid-size "${LOSS3_SVF_GRID_SIZE:-8}" \
  --loss3-svf-inner-steps "${LOSS3_SVF_INNER_STEPS:-8}" \
  --loss3-svf-lr "${LOSS3_SVF_LR:-0.05}" \
  --loss3-svf-max-vel-ratio "${LOSS3_SVF_MAX_VEL_RATIO:-0.04}" \
  --loss3-svf-int-steps "${LOSS3_SVF_INT_STEPS:-5}" \
  --loss3-svf-mag-weight "${LOSS3_SVF_MAG_WEIGHT:-0.01}" \
  --loss3-svf-smooth-weight "${LOSS3_SVF_SMOOTH_WEIGHT:-0.05}" \
  --methods ${METHODS:-raw_add raw_replace steepest_add steepest_replace} \
  --mode-spec "${MODE_SPEC:-all_w}" \
  --steps "${STEPS:-50}" \
  --p "${P_ORDER:-2}" \
  --q "${Q_ORDER:-2}" \
  --true-loss-every "${TRUE_LOSS_EVERY:-1}" \
  --fixed-step "${FIXED_STEP:-0.005}" \
  --solver-remat "${SOLVER_REMAT:-micro}" \
  --solver-remat-chunk-steps "${SOLVER_REMAT_CHUNK_STEPS:-20}" \
  --dictionary-chunk-size "${DICTIONARY_CHUNK_SIZE:-32}" \
  --clean-target-source "${CLEAN_TARGET_SOURCE:-dataset}" \
  "${EXTRA_ARGS[@]}"
