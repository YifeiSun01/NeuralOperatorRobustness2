#!/usr/bin/env bash
set -euo pipefail

cd /workspace/NeuralOperatorRobustness2

export PYTHON_BIN="${PYTHON_BIN:-adv_robust/bin/python}"
export INDICES="${INDICES:-0,1,2,3,4,5,6,7,8,9}"
export ATTACK_BATCH_SIZE="${ATTACK_BATCH_SIZE:-10}"
export LOSS3_MODE_SPEC="${LOSS3_MODE_SPEC:-all_w}"
export LOSS3_METRICS="${LOSS3_METRICS:-dists ms_ssim scattering2d affine_dists local_warp_dists homography_dists tps_dists elastic_dists svf_dists}"
export BUDGET_PRESETS="${BUDGET_PRESETS:-very_strong strong medium weak}"
export AUTO_PLOT_DATASET0="${AUTO_PLOT_DATASET0:-1}"
export AUTO_UPLOAD_R2="${AUTO_UPLOAD_R2:-1}"

MASTER_TAG="${MASTER_TAG:-eps32_alpha10_steepest_add_loss3_allw_alt9_budget_sweep_steps25_then100_b10_$(date -u +%Y%m%d_%H%M%S)_UTC}"
MASTER_ROOT="${MASTER_ROOT:-2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/${MASTER_TAG}}"
mkdir -p "$MASTER_ROOT"
MASTER_LOG="$MASTER_ROOT/nohup_${MASTER_TAG}.log"

run_steps() {
  local steps="$1"
  export STEPS="$steps"
  export BASE_SWEEP_TAG="${MASTER_TAG}_steps${steps}"
  export BASE_SWEEP_ROOT="${MASTER_ROOT}/steps${steps}"
  mkdir -p "$BASE_SWEEP_ROOT"

  echo
  echo "===== steps${steps} sweep started $(date -u '+%Y-%m-%dT%H:%M:%SZ') ====="
  echo "base_sweep_tag=$BASE_SWEEP_TAG"
  echo "base_sweep_root=$BASE_SWEEP_ROOT"
  echo "auto_plot_dataset0=$AUTO_PLOT_DATASET0"
  echo "auto_upload_r2=$AUTO_UPLOAD_R2"
  bash tools/run_ns2d_eps32_alpha10_steepest_add_loss3_allw_alt9_budget_sweep_steps25_20260525.sh
  echo "===== steps${steps} sweep finished $(date -u '+%Y-%m-%dT%H:%M:%SZ') ====="
}

{
  echo "master_tag=$MASTER_TAG"
  echo "master_root=$MASTER_ROOT"
  echo "budget_order=$BUDGET_PRESETS"
  echo "loss3_metrics=$LOSS3_METRICS"
  echo "indices=$INDICES"
  echo "attack_batch_size=$ATTACK_BATCH_SIZE"
  echo "r2_bucket=${R2_BUCKET:-neural-operator-robustness}"
  echo "r2_prefix=${R2_PREFIX:-machine-sync/NeuralOperatorRobustness2-selected}"
  echo "start_time_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"

  run_steps 25
  run_steps 100

  echo "finish_time_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "===== steps25_then100 completed ====="
} 2>&1 | tee -a "$MASTER_LOG"
