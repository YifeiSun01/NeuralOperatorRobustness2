#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-adv_robust/bin/python}"
TAG_PREFIX="${TAG_PREFIX:-20260613_loss3attack50_seven_models_random_inclusive}"
ATTACK_STEPS="${ATTACK_STEPS:-50}"
EPSILON_FRACTION="${EPSILON_FRACTION:-0.025}"

variants=(
  index0
  loss3_best
  loss3_upper_quartile
  loss3_median
  loss3_lower_quartile
  loss3_worst
  loss3_more_faded_recommended
  loss3_more_faded_maxfade_signsame
  loss3_top5pct_visual_relaxed
  loss3_top10pct_visual_relaxed
)

for variant in "${variants[@]}"; do
  manifest="analysis_outputs/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_${variant}/selected_samples.csv"
  if [[ ! -f "$manifest" ]]; then
    echo "[skip] missing $manifest"
    continue
  fi
  tag="${TAG_PREFIX}_${variant}"
  out_dir="analysis_outputs/darcy_seven_model_attack_heatmaps_${tag}"
  viz_dir="visualizations/darcy_seven_model_attack_heatmaps_${tag}"
  if [[ -f "$out_dir/summary.csv" && "${FORCE:-0}" != "1" ]]; then
    echo "[skip] existing $out_dir/summary.csv"
    continue
  fi
  echo "[seven-heatmap] $variant -> $viz_dir"
  "$PYTHON_BIN" tools/plot_darcy_seven_model_attack_heatmaps_20260613.py \
    --tag "$tag" \
    --sample-manifest "$manifest" \
    --out-dir "$out_dir" \
    --viz-dir "$viz_dir" \
    --attack-steps "$ATTACK_STEPS" \
    --epsilon-fraction "$EPSILON_FRACTION"
done
