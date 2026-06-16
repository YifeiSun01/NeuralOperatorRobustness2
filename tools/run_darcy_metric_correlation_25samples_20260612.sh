#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-adv_robust/bin/python}"
TAG="${TAG:-20260612_25samples_loss3attack20_corr_chunk64}"
MAX_SAMPLES="${MAX_SAMPLES:-25}"
ATTACK_STEPS="${ATTACK_STEPS:-20}"
BLOCK_ROW_CHUNK="${BLOCK_ROW_CHUNK:-64}"
GENERALIZATION_ROOT="${GENERALIZATION_ROOT:-generalization_datasets_darcy_binary_loss3targeted_20260611}"
OUT_DIR="${OUT_DIR:-analysis_outputs/darcy_metric_correlation_25samples_${TAG}}"
VIZ_DIR="${VIZ_DIR:-visualizations/darcy_metric_correlation_25samples_${TAG}}"
LOG_DIR="${LOG_DIR:-analysis_outputs/darcy_metric_correlation_25samples_${TAG}_logs}"
mkdir -p "$LOG_DIR"

"$PYTHON" tools/run_darcy_metric_correlation_25samples_20260612.py   --tag "$TAG"   --generalization-root "$GENERALIZATION_ROOT"   --out-dir "$OUT_DIR"   --viz-dir "$VIZ_DIR"   --max-samples "$MAX_SAMPLES"   --attack-steps "$ATTACK_STEPS"   --block-row-chunk "$BLOCK_ROW_CHUNK"   2>&1 | tee -a "$LOG_DIR/full_run.log"
