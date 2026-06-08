#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-/workspace/NeuralOperatorRobustness2/adv_robust/bin/python}"
ATTACK_SCRIPT="tools/run_burgers_svd20_p2q2_attack_correlation_20260608.py"
BASELINE="/workspace/NeuralOperatorRobustness2/1D_Burgers/trained_models/attack_ready/burgers_nu0.001_fno1d_500/checkpoints/pytorch_fno1d_500.pt"
LOG_DIR="run_logs/burgers_existing_svd_attack_correlations_20260608"
mkdir -p "$LOG_DIR"

run_join() {
  local name="$1"
  shift
  echo "[join-start] ${name} $(date -Is)" | tee "$LOG_DIR/${name}.join.status"
  "$PY" "$ATTACK_SCRIPT" --correlate-only "$@" 2>&1 | tee "$LOG_DIR/${name}.join.log"
  echo "[join-done] ${name} $(date -Is)" | tee -a "$LOG_DIR/${name}.join.status"
}

run_join run2_ns50_loss3_checkpoint_series \
  --svd-manifest forensics/burgers_p2q2_checkpoint_series_jacobian_svd_20260601/representative20_sample_manifest.csv \
  --svd-summary forensics/burgers_p2q2_checkpoint_series_jacobian_svd_20260601/checkpoint_series_jacobian_svd_summary.csv \
  --out-dir forensics/burgers_run2_ns50_p2q2_loss3_checkpoint_series_svd20_attack_correlation_20260608 \
  --steps 20 --batch-size 20 \
  --model-spec "baseline=${BASELINE}" \
  --model-spec "epoch0200=/workspace/NeuralOperatorRobustness2/adversarial_training_runs/burgers_p2q2_advonly_random_jitter_1000ep_bs480_steps5_eps5bucket_20260601/burgers/checkpoints/burgers_epoch200_step000600.pt" \
  --model-spec "epoch0400=/workspace/NeuralOperatorRobustness2/adversarial_training_runs/burgers_p2q2_advonly_random_jitter_1000ep_bs480_steps5_eps5bucket_20260601/burgers/checkpoints/burgers_epoch400_step001200.pt" \
  --model-spec "epoch0600=/workspace/NeuralOperatorRobustness2/adversarial_training_runs/burgers_p2q2_advonly_random_jitter_1000ep_bs480_steps5_eps5bucket_20260601/burgers/checkpoints/burgers_epoch600_step001800.pt" \
  --model-spec "epoch0800=/workspace/NeuralOperatorRobustness2/adversarial_training_runs/burgers_p2q2_advonly_random_jitter_1000ep_bs480_steps5_eps5bucket_20260601/burgers/checkpoints/burgers_epoch800_step002400.pt" \
  --model-spec "epoch1000=/workspace/NeuralOperatorRobustness2/adversarial_training_runs/burgers_p2q2_advonly_random_jitter_1000ep_bs480_steps5_eps5bucket_20260601/burgers/checkpoints/burgers_epoch1000_step003000.pt"

run_join run2_ns50_loss1_epoch2000 \
  --svd-manifest forensics/burgers_p2q2_loss1_epoch2000_jacobian_svd_rep20_top100_20260603/representative20_sample_manifest.csv \
  --svd-summary forensics/burgers_p2q2_loss1_epoch2000_jacobian_svd_rep20_top100_20260603/checkpoint_series_jacobian_svd_summary.csv \
  --out-dir forensics/burgers_run2_ns50_p2q2_loss1_epoch2000_svd20_attack_correlation_20260608 \
  --steps 20 --batch-size 20 \
  --model-spec "baseline=${BASELINE}" \
  --model-spec "epoch2000=/workspace/NeuralOperatorRobustness2/adversarial_training_runs/burgers_p2q2_loss1_save1000_2000_v5_20260602/burgers/checkpoints/burgers_epoch2000_step006000.pt"

run_join run2_ns50_loss2_epoch0900 \
  --svd-manifest forensics/burgers_p2q2_loss2_epoch900_jacobian_svd_rep20_top100_20260603/representative20_sample_manifest.csv \
  --svd-summary forensics/burgers_p2q2_loss2_epoch900_jacobian_svd_rep20_top100_20260603/checkpoint_series_jacobian_svd_summary.csv \
  --out-dir forensics/burgers_run2_ns50_p2q2_loss2_epoch900_svd20_attack_correlation_20260608 \
  --steps 20 --batch-size 20 \
  --model-spec "baseline=${BASELINE}" \
  --model-spec "epoch0900=/workspace/NeuralOperatorRobustness2/adversarial_training_runs/burgers_p2q2_loss2_save900_1000_v5_20260602/burgers/checkpoints/burgers_epoch900_step002700.pt"

run_join round01_aligned_final_loss123 \
  --svd-manifest forensics/burgers_loss3_aligned_round01_final_jacobian_svd_rep20_top100_20260604/round01_sample_manifest.csv \
  --svd-summary forensics/burgers_loss3_aligned_round01_final_jacobian_svd_rep20_top100_20260604/round01_jacobian_svd_summary.csv \
  --out-dir forensics/burgers_round01_aligned_final_loss123_svd20_p2q2_attack_correlation_20260608 \
  --steps 20 --batch-size 20 \
  --model-spec "baseline=${BASELINE}" \
  --model-spec "loss1_epoch1000=/workspace/NeuralOperatorRobustness2/adversarial_training_runs/burgers_loss3_aligned_round01_loss1_1000ep_20260604/burgers/checkpoints/burgers_epoch1000_step003000.pt" \
  --model-spec "loss2_epoch500=/workspace/NeuralOperatorRobustness2/adversarial_training_runs/burgers_loss3_aligned_round01_loss2_500ep_20260604/burgers/checkpoints/burgers_epoch500_step001500.pt" \
  --model-spec "loss3_epoch500=/workspace/NeuralOperatorRobustness2/adversarial_training_runs/burgers_loss3_aligned_round01_loss3_500ep_20260604/burgers/checkpoints/burgers_epoch500_step001500.pt"

run_join round03_long_final_loss123 \
  --svd-manifest forensics/burgers_loss3_selective_round03_long_final_jacobian_svd_rep20_top100_20260605/round03_long_final_sample_manifest.csv \
  --svd-summary forensics/burgers_loss3_selective_round03_long_final_jacobian_svd_rep20_top100_20260605/round03_long_final_jacobian_svd_summary.csv \
  --out-dir forensics/burgers_round03_long_final_loss123_svd20_p2q2_attack_correlation_20260608 \
  --steps 20 --batch-size 20 \
  --model-spec "baseline=${BASELINE}" \
  --model-spec "loss1_epoch1000=/workspace/NeuralOperatorRobustness2/adversarial_training_runs/burgers_loss3_selective_round03_loss1_1000ep_long_20260605/burgers/checkpoints/burgers_epoch1000_step003000.pt" \
  --model-spec "loss2_epoch0500=/workspace/NeuralOperatorRobustness2/adversarial_training_runs/burgers_loss3_selective_round03_loss2_500ep_long_20260605/burgers/checkpoints/burgers_epoch500_step001500.pt" \
  --model-spec "loss3_epoch0500=/workspace/NeuralOperatorRobustness2/adversarial_training_runs/burgers_loss3_selective_round03_loss3_500ep_long_20260605/burgers/checkpoints/burgers_epoch500_step001500.pt"
