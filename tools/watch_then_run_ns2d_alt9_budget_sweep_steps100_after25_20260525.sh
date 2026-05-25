#!/usr/bin/env bash
set -euo pipefail

cd /workspace/NeuralOperatorRobustness2

PYTHON_BIN="${PYTHON_BIN:-adv_robust/bin/python}"
CURRENT_25_ROOT="${1:-${CURRENT_25_ROOT:-}}"
if [[ -z "$CURRENT_25_ROOT" ]]; then
  echo "ERROR: pass current 25-step BASE_SWEEP_ROOT as arg 1 or CURRENT_25_ROOT" >&2
  exit 2
fi

POLL_SECONDS="${POLL_SECONDS:-120}"
CURRENT_25_LOG="${CURRENT_25_LOG:-}"
if [[ -z "$CURRENT_25_LOG" ]]; then
  CURRENT_25_LOG="$(find "$CURRENT_25_ROOT" -maxdepth 1 -name 'nohup_*.log' -type f | sort | tail -1)"
fi

if [[ -z "$CURRENT_25_LOG" || ! -f "$CURRENT_25_LOG" ]]; then
  echo "ERROR: cannot find current 25-step sweep log under $CURRENT_25_ROOT" >&2
  exit 2
fi

echo "watch_100_start_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "current_25_root=$CURRENT_25_ROOT"
echo "current_25_log=$CURRENT_25_LOG"
echo "poll_seconds=$POLL_SECONDS"

while ! grep -q '===== budget sweep completed =====' "$CURRENT_25_LOG"; do
  sleep "$POLL_SECONDS"
done

echo "current_25_completed_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"

export PYTHON_BIN
export STEPS=100
export INDICES="${INDICES:-0,1,2,3,4,5,6,7,8,9}"
export ATTACK_BATCH_SIZE="${ATTACK_BATCH_SIZE:-10}"
export LOSS3_MODE_SPEC="${LOSS3_MODE_SPEC:-all_w}"
export LOSS3_METRICS="${LOSS3_METRICS:-dists ms_ssim scattering2d affine_dists local_warp_dists homography_dists tps_dists elastic_dists svf_dists}"
export BUDGET_PRESETS="${BUDGET_PRESETS:-very_very_strong very_strong strong medium weak}"
export AUTO_PLOT_DATASET0="${AUTO_PLOT_DATASET0:-1}"
export BASE_SWEEP_TAG="${BASE_SWEEP_TAG:-eps32_alpha10_steepest_add_loss3_allw_alt9_budget_sweep_steps100_b10_$(date -u +%Y%m%d_%H%M%S)_UTC}"

echo "launch_steps100_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "steps100_base_sweep_tag=$BASE_SWEEP_TAG"
echo "steps100_budget_order=$BUDGET_PRESETS"
echo "steps100_metrics=$LOSS3_METRICS"

bash tools/run_ns2d_eps32_alpha10_steepest_add_loss3_allw_alt9_budget_sweep_steps25_20260525.sh

echo "watch_100_finish_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
