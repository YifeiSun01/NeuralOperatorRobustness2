#!/usr/bin/env bash
set -euo pipefail

cd /workspace/NeuralOperatorRobustness2

PYTHON=${PYTHON:-/workspace/adv_robust/bin/python}

exec "$PYTHON" tools/evaluate_burgers_random_field_final_models_20260613.py \
  --stage attack \
  --out-root forensics/burgers_latest_old4_widevis_full52_p2q2_20step_20260614 \
  --gen-root generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00/burgers \
  --attack-steps 20 \
  --attack-batch-size 500 \
  --attack-train-count 50 \
  --model-spec baseline=1D_Burgers/trained_models/attack_ready/burgers_nu0.001_fno1d_500/checkpoints/pytorch_fno1d_500.pt \
  --model-spec loss1=adversarial_training_runs/burgers_wideparam_loss1_8000ep_retrain_20260611/burgers/checkpoints/burgers_epoch8000_step008000.pt \
  --model-spec loss2=adversarial_training_runs/burgers_wideparam_loss2_2000ep_retrain_20260611/burgers/checkpoints/burgers_epoch2000_step002000.pt \
  --model-spec loss3=adversarial_training_runs/burgers_wideparam_loss3_1000ep_retrain_20260611/burgers/checkpoints/burgers_epoch1000_step001000.pt
