#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "adv_robust/bin/python" ]]; then
    PYTHON_BIN="adv_robust/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

TRAIN_PATH_DEFAULT="2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/real_initial_laxmap_single/train/dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_ntimepoints21_all_frames.pt"
TEST_PATH_DEFAULT="2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/real_initial_laxmap_single/test/dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_test_ntimepoints21_all_frames.pt"
R2_ENDPOINT_DEFAULT="https://606bf6c862a4e8f63dabb6243dce2df7.r2.cloudflarestorage.com"
R2_BUCKET_PREFIX_DEFAULT="neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected/2D_NS_FNO2d_recurrent/saved_models/2D"

R2_ARGS=()
if [[ "${R2_UPLOAD:-1}" == "1" ]]; then
  R2_ARGS+=(
    --r2-upload
    --r2-endpoint "${R2_ENDPOINT:-$R2_ENDPOINT_DEFAULT}"
    --r2-bucket-prefix "${R2_BUCKET_PREFIX:-$R2_BUCKET_PREFIX_DEFAULT}"
    --r2-sync-every-epochs "${R2_SYNC_EVERY_EPOCHS:-0}"
  )
fi

EXTRA_ARGS=()
if [[ "${AMP:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--amp)
fi
if [[ -n "${NUM_WORKERS:-}" ]]; then
  EXTRA_ARGS+=(--num-workers "${NUM_WORKERS}")
fi

exec "$PYTHON_BIN" -u 2D_NS_FNO2d_recurrent/training_models/train_fno2d_recurrent_cli.py \
  --train-path "${TRAIN_PATH:-$TRAIN_PATH_DEFAULT}" \
  --test-path "${TEST_PATH:-$TEST_PATH_DEFAULT}" \
  --modes1 64 \
  --modes2 64 \
  --width 60 \
  --epochs "${EPOCHS:-500}" \
  --batch-size "${BATCH_SIZE:-16}" \
  --eval-batch-size "${EVAL_BATCH_SIZE:-16}" \
  --ntrain "${NTRAIN:-1150}" \
  --ntest "${NTEST:-50}" \
  --learning-rate "${LR:-0.001}" \
  --weight-decay "${WEIGHT_DECAY:-0.0001}" \
  --eval-every "${EVAL_EVERY:-10}" \
  --progress-every "${PROGRESS_EVERY:-10}" \
  --data-residency "${DATA_RESIDENCY:-gpu}" \
  --save-every "${SAVE_EVERY:-25}" \
  "${R2_ARGS[@]}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
