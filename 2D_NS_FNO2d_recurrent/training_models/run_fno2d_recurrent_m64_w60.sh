#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "adv_robust/bin/python" ]]; then
    PYTHON_BIN="adv_robust/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

exec "$PYTHON_BIN" -u 2D_NS_FNO2d_recurrent/training_models/train_fno2d_recurrent_cli.py \
  --modes1 64 \
  --modes2 64 \
  --width 60 \
  --epochs "${EPOCHS:-500}" \
  --batch-size "${BATCH_SIZE:-4}" \
  --eval-batch-size "${EVAL_BATCH_SIZE:-4}" \
  --ntrain "${NTRAIN:-1000}" \
  --ntest "${NTEST:-100}" \
  --learning-rate "${LR:-0.001}" \
  --weight-decay "${WEIGHT_DECAY:-0.0001}" \
  "$@"
