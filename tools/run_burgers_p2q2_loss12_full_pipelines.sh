#!/usr/bin/env bash
set -euo pipefail
cd /workspace/NeuralOperatorRobustness2
PYTHON_BIN="${PYTHON_BIN:-adv_robust/bin/python}"
"$PYTHON_BIN" tools/run_burgers_p2q2_full_pipeline.py --attack-loss loss1 "$@"
"$PYTHON_BIN" tools/run_burgers_p2q2_full_pipeline.py --attack-loss loss2 "$@"
