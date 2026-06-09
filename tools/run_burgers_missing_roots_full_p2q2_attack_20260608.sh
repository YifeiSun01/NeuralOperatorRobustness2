#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/NeuralOperatorRobustness2"
PY="${ROOT}/adv_robust/bin/python"
LOG_ROOT="${ROOT}/run_logs/burgers_missing_roots_full_p2q2_20260608"
mkdir -p "${LOG_ROOT}"
cd "${ROOT}"

"${PY}" tools/run_burgers_round03_full_p2q2_finalonly_attack.py   --gen-root generalization_datasets/burgers   --run-name burgers_first_master_full_p2q2_52datasets_4models_finalonly_20step_20260608   --steps 20   --batch-size 500   --train-count 50   > "${LOG_ROOT}/first_master_full_p2q2_20step.log" 2>&1

"${PY}" tools/run_burgers_round03_full_p2q2_finalonly_attack.py   --gen-root generalization_datasets_burgers_loss3_selective_search/round_03/burgers   --run-name burgers_round03_selective_full_p2q2_52datasets_4models_finalonly_20step_20260608   --steps 20   --batch-size 500   --train-count 50   > "${LOG_ROOT}/round03_selective_full_p2q2_20step.log" 2>&1
