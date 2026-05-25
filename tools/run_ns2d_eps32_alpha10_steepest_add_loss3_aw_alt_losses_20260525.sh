#!/usr/bin/env bash
set -euo pipefail
cd /workspace/NeuralOperatorRobustness2
exec bash tools/run_ns2d_eps32_alpha10_steepest_add_loss3_allw_alt_losses_20260525.sh "$@"
