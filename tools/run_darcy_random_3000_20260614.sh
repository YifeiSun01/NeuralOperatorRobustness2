#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/NeuralOperatorRobustness2"
cd "$ROOT"

method="${1:?usage: $0 random_clean|random_solver}"

case "$method" in
  random_clean)
    run_name="darcy_binary_random_binary_fixed_y_3000ep_full50_20260614_random_binary_source_3000_supervised"
    training_mode="random-binary-fixed-y"
    ;;
  random_solver)
    run_name="darcy_binary_random_binary_solver_y_3000ep_full50_20260614_random_binary_source_3000_supervised"
    training_mode="random-binary-solver-y"
    ;;
  *)
    echo "unknown method: $method" >&2
    exit 2
    ;;
esac

exec adv_robust/bin/python tools/adversarial_training.py \
  --tasks darcy \
  --generalization-root generalization_datasets_darcy_binary_loss3targeted_20260611 \
  --output-root adversarial_training_runs \
  --run-name "$run_name" \
  --epochs 3000 \
  --darcy-train-max 64 \
  --darcy-batch-size 64 \
  --darcy-optimizer-batch-size 32 \
  --eval-max-samples 10 \
  --max-generalization-eval 50 \
  --training-data-mode "$training_mode" \
  --label-mode solver \
  --checkpoint-every-epochs 500 \
  --attack-probe-samples 0 \
  --darcy-random-source-kernels gaussian,matern,highpass,bandpass,mixed \
  --darcy-random-source-alpha-values 1.2,2.2,3.2,4.2,5.2 \
  --darcy-random-source-lengthscale-min 0.035 \
  --darcy-random-source-lengthscale-max 0.30 \
  --darcy-random-source-min-flip-fraction 0.005 \
  --darcy-random-source-max-flip-fraction 0.05
