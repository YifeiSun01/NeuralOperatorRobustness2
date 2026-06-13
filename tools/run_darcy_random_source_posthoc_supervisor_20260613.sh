#!/usr/bin/env bash
set -euo pipefail
cd /workspace/NeuralOperatorRobustness2
exec adv_robust/bin/python tools/run_darcy_random_source_posthoc_20260613.py \
  --tag 20260613_random_binary_source_1100_posthoc \
  --generalization-root generalization_datasets_darcy_binary_loss3targeted_20260611 \
  --out-root analysis_outputs/darcy_random_binary_source_20260613_random_binary_source_1100_posthoc \
  --viz-root visualizations/darcy_random_binary_source_20260613_random_binary_source_1100_posthoc \
  --clean-max-samples 10 \
  --wait \
  --timeout-hours 24 \
  --poll-seconds 120
