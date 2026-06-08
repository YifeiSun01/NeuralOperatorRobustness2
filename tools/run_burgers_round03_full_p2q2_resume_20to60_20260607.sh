#!/usr/bin/env bash
set -euo pipefail
cd /workspace/NeuralOperatorRobustness2
RESUME_ROOT="forensics/burgers_round03_full_p2q2_52datasets_4models_finalonly_20step_20260607"
if [[ ! -f "$RESUME_ROOT/summary.json" ]]; then
  echo "[error] Cannot resume to 60: $RESUME_ROOT/summary.json is missing; wait for the 20-step run to finish." >&2
  exit 2
fi
for model in baseline loss1_epoch8000 loss2_epoch2000 loss3_epoch1500; do
  if [[ ! -f "$RESUME_ROOT/$model/final_delta_by_sample.npz" ]]; then
    echo "[error] Cannot resume to 60: missing $RESUME_ROOT/$model/final_delta_by_sample.npz" >&2
    exit 2
  fi
  if [[ ! -f "$RESUME_ROOT/$model/losses_and_delta_rms_by_sample.npz" ]]; then
    echo "[error] Cannot resume to 60: missing $RESUME_ROOT/$model/losses_and_delta_rms_by_sample.npz" >&2
    exit 2
  fi
done
mkdir -p adversarial_training_runs/burgers_round03_full_p2q2_resume_20to60_20260607_logs
LOG="adversarial_training_runs/burgers_round03_full_p2q2_resume_20to60_20260607_logs/run_$(date -u +%Y%m%d_%H%M%S)_UTC.log"
{
  echo "[launch] $(date -u +%Y-%m-%dT%H:%M:%SZ) resume 20->60"
  nvidia-smi
  adv_robust/bin/python tools/run_burgers_round03_full_p2q2_finalonly_attack.py     --resume-from-root "$RESUME_ROOT"     --target-total-steps 60     --batch-size 500     --train-count 50     --run-name burgers_round03_full_p2q2_52datasets_4models_finalonly_60step_from20_20260607
  echo "[done] $(date -u +%Y-%m-%dT%H:%M:%SZ) resume 20->60"
} 2>&1 | tee -a "$LOG"
