#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-adv_robust/bin/python}"
TAG="${TAG:-20260612_loss3attack50_five_models_ranked5_batch_polished}"
DATASET_COUNT="${DATASET_COUNT:-5}"
SAMPLES_PER_DATASET="${SAMPLES_PER_DATASET:-50}"
ATTACK_STEPS="${ATTACK_STEPS:-50}"
ATTACK_BATCH_SIZE="${ATTACK_BATCH_SIZE:-25}"
PLOT_BATCH_SIZE="${PLOT_BATCH_SIZE:-10}"
EPSILON_FRACTION="${EPSILON_FRACTION:-0.025}"
GENERALIZATION_ROOT="${GENERALIZATION_ROOT:-generalization_datasets_darcy_binary_loss3targeted_20260611}"
ATTACK20_TABLE="${ATTACK20_TABLE:-analysis_outputs/darcy_attack20_52datasets_50samples_20260612_attack20_1000c_52datasets_50samples/all_samples.csv}"
OUT_ROOT="${OUT_ROOT:-analysis_outputs}"
VIZ_ROOT="${VIZ_ROOT:-visualizations}"
FORCE="${FORCE:-0}"
ONLY_VARIANT="${ONLY_VARIANT:-}"

LOG_ROOT="$OUT_ROOT/darcy_five_model_batch_ranked_heatmaps_${TAG}_logs"
mkdir -p "$LOG_ROOT"
DRIVER_LOG="$LOG_ROOT/driver.log"

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$DRIVER_LOG"
}

cmd=(
  "$PYTHON" tools/plot_darcy_five_model_batch_ranked_heatmaps_20260612.py
  --tag "$TAG"
  --generalization-root "$GENERALIZATION_ROOT"
  --attack20-table "$ATTACK20_TABLE"
  --out-root "$OUT_ROOT"
  --viz-root "$VIZ_ROOT"
  --dataset-count "$DATASET_COUNT"
  --samples-per-dataset "$SAMPLES_PER_DATASET"
  --attack-steps "$ATTACK_STEPS"
  --epsilon-fraction "$EPSILON_FRACTION"
  --attack-batch-size "$ATTACK_BATCH_SIZE"
  --plot-batch-size "$PLOT_BATCH_SIZE"
)
if [[ "$FORCE" == "1" ]]; then
  cmd+=(--force)
fi
if [[ -n "$ONLY_VARIANT" ]]; then
  cmd+=(--only-variant "$ONLY_VARIANT")
fi

log "start Darcy batch ranked heatmaps tag=$TAG attack_steps=$ATTACK_STEPS dataset_count=$DATASET_COUNT samples_per_dataset=$SAMPLES_PER_DATASET attack_batch_size=$ATTACK_BATCH_SIZE plot_batch_size=$PLOT_BATCH_SIZE"
printf '[command] %q ' "${cmd[@]}" | tee -a "$DRIVER_LOG"
printf '\n' | tee -a "$DRIVER_LOG"
"${cmd[@]}" 2>&1 | tee -a "$LOG_ROOT/full_run.log"
log "complete Darcy batch ranked heatmaps tag=$TAG"
