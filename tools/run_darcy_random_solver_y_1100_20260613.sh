#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/NeuralOperatorRobustness2"
cd "$ROOT"

RUN="darcy_binary_random_binary_solver_y_1100ep_full50_20260613_random_binary_source_1100"

exec adv_robust/bin/python tools/adversarial_training.py \
  --tasks darcy \
  --generalization-root generalization_datasets_darcy_binary_loss3targeted_20260611 \
  --output-root adversarial_training_runs \
  --run-name "$RUN" \
  --epochs 1100 \
  --darcy-train-max 64 \
  --darcy-batch-size 64 \
  --darcy-optimizer-batch-size 32 \
  --eval-max-samples 10 \
  --max-generalization-eval 50 \
  --training-data-mode random-binary-solver-y \
  --label-mode solver \
  --checkpoint-every-epochs 200 \
  --attack-probe-samples 5 \
  --attack-probe-every-n-epochs 1 \
  --attack-probe-save-targets \
  --darcy-random-source-kernels gaussian,matern,highpass,bandpass,mixed \
  --darcy-random-source-alpha-values 1.2,2.2,3.2,4.2,5.2 \
  --darcy-random-source-lengthscale-min 0.035 \
  --darcy-random-source-lengthscale-max 0.30 \
  --darcy-random-source-min-flip-fraction 0.005 \
  --darcy-random-source-max-flip-fraction 0.05
