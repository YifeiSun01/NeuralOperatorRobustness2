#!/usr/bin/env bash
set -euo pipefail
cd /workspace/NeuralOperatorRobustness2
mkdir -p adversarial_training_runs/burgers_round03_full_p2q2_20step_pilot_20260607_logs
LOG="adversarial_training_runs/burgers_round03_full_p2q2_20step_pilot_20260607_logs/run_$(date -u +%Y%m%d_%H%M%S)_UTC.log"
{
  echo "[launch] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  nvidia-smi
  adv_robust/bin/python tools/run_burgers_round03_full_p2q2_finalonly_attack.py \
    --steps 20 \
    --batch-size 500 \
    --train-count 50 \
    --run-name burgers_round03_full_p2q2_52datasets_4models_finalonly_20step_20260607
  echo "[done] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} 2>&1 | tee -a "$LOG"
