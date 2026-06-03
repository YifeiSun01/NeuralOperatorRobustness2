#!/usr/bin/env bash
set -euo pipefail

cd /workspace/NeuralOperatorRobustness2

PYTHON_BIN="${PYTHON_BIN:-adv_robust/bin/python}"
LOG_DIR="adversarial_training_runs/burgers_p2q2_loss12_time_matched_loss3_20260602_logs"
mkdir -p "$LOG_DIR"

# Loss3 reference times, recorded for the wall-clock rationale.
LOSS3_EPOCH200_SECONDS="10686.52828713262"  # 1/5 of 1000 epochs, about 2.97 h
LOSS3_EPOCH400_SECONDS="21172.02130720811"  # 2/5 of 1000 epochs, about 5.88 h

COMMON_ARGS=(
  --mode full
  --skip-jacobian
  --skip-svd-plots
  --skip-r2-upload
  --skip-git-push
  --seed 20260601
  --batch-size 480
  --optimizer-batch-size 32
  --attack-steps 5
  --epsilon-fraction 0.06
  --eps-jitter-low 0.75
  --eps-jitter-high 1.25
  --alpha-ratio 1.0
  --alpha-jitter-low 0.75
  --alpha-jitter-high 1.25
  --attack-probe-samples 5
)

run_loss() {
  local loss_name="$1"
  local epochs="$2"
  local checkpoint_every="$3"
  local random_start="$4"
  local run_name="$5"
  local target_note="$6"
  local log_path="$LOG_DIR/${run_name}.log"

  printf '[%s] starting %s: epochs=%s checkpoint_every=%s random_start=%s target_note=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$loss_name" "$epochs" "$checkpoint_every" "$random_start" "$target_note" >> "$log_path"

  "$PYTHON_BIN" tools/run_burgers_p2q2_full_pipeline.py \
    --attack-loss "$loss_name" \
    --run-name "$run_name" \
    --training-epochs "$epochs" \
    --checkpoint-every-epochs "$checkpoint_every" \
    --random-start-fraction "$random_start" \
    --viz-root "visualizations/${run_name}" \
    --forensics-root "forensics/${run_name}_checkpoint_series_jacobian_svd" \
    --attack-gif-root "forensics/${run_name}_baseline_vs_epoch1000_attack_visualization" \
    --selected-top "/workspace/polished_selected_download_${run_name}" \
    --selected-zip "/workspace/polished_selected_download_${run_name}.zip" \
    "${COMMON_ARGS[@]}" >> "$log_path" 2>&1

  printf '[%s] finished %s: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$loss_name" "$run_name" >> "$log_path"
}

printf '[%s] runner started\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG_DIR/runner.log"
printf 'loss3_1over5_seconds=%s\nloss3_2over5_seconds=%s\nloss1_save_epochs=1000,2000\nloss2_save_epochs=900,1000\n' \
  "$LOSS3_EPOCH200_SECONDS" "$LOSS3_EPOCH400_SECONDS" >> "$LOG_DIR/runner.log"

# User-corrected checkpoint policy:
#   loss1: save only epoch 1000 and epoch 2000.
#   loss2: save only epoch 900 and epoch 1000.
run_loss loss1 2000 1000 1e-6 burgers_p2q2_loss1_save1000_2000_v5_20260602 "loss3 1/5 ~= epoch 2000"
run_loss loss2 1000 900 0.0 burgers_p2q2_loss2_save900_1000_v5_20260602 "loss3 2/5 ~= epoch 900"

printf '[%s] runner finished\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG_DIR/runner.log"
