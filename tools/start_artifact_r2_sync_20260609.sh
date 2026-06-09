#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 R2:bucket/prefix" >&2
  exit 2
fi

DEST="${1%/}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$PROJECT_ROOT/forensics/artifact_r2_sync_20260609"
LOG="$RUN_DIR/rclone_artifact_sync.log"
MANIFEST="$RUN_DIR/artifact_sync_manifest.txt"

mkdir -p "$RUN_DIR"

{
  printf 'Started: %s\n' "$(date -Iseconds)"
  printf 'Project root: %s\n' "$PROJECT_ROOT"
  printf 'Destination: %s\n' "$DEST"
  printf 'Mode: rclone copy, size-only incremental artifact upload\n'
  printf 'Intent: upload model checkpoints, datasets, generated numeric results, logs, images, visualizations, and archives; keep source code and Markdown in GitHub.\n'
  printf '\nIncluded artifact directories and patterns are encoded in the rclone filter flags in %s.\n' "$0"
} > "$MANIFEST"

cd "$PROJECT_ROOT"

printf '[%s] Artifact copy -> %s\n' "$(date -Iseconds)" "$DEST" | tee -a "$LOG"

rclone copy "." "$DEST" \
  --s3-no-check-bucket \
  --s3-disable-checksum \
  --retries 5 \
  --low-level-retries 5 \
  --transfers 8 \
  --checkers 24 \
  --fast-list \
  --size-only \
  --stats 60s \
  --stats-one-line \
  --exclude '.git/**' \
  --exclude 'adv_robust/**' \
  --exclude '__pycache__/**' \
  --exclude '*.pyc' \
  --exclude '.DS_Store' \
  --include 'adversarial_training_runs/**' \
  --include 'forensics/**' \
  --include 'visualizations/**' \
  --include 'visualization_outputs/**' \
  --include 'analysis_outputs/**' \
  --include 'figures/**' \
  --include 'results/**' \
  --include 'run_logs/**' \
  --include 'data/**' \
  --include 'Model/**' \
  --include 'generalization_datasets*/**' \
  --include 'generalization_eval*/**' \
  --include '**/saved_models/**' \
  --include '**/saved_models_expanded/**' \
  --include '**/saved_models_kernel_abstracts/**' \
  --include '**/trained_models/**' \
  --include '**/datasets/**' \
  --include '**/test_train_datasets/**' \
  --include '**/GRFs/**' \
  --include '**/perturbation_results/**' \
  --include '**/perturbed_results_as_inputs/**' \
  --include '**/attack_result_viz/**' \
  --include '**/viz_results/**' \
  --include '**/visualizations/**' \
  --include '**/*.pt' \
  --include '**/*.pth' \
  --include '**/*.ckpt' \
  --include '**/*.safetensors' \
  --include '**/*.npz' \
  --include '**/*.npy' \
  --include '**/*.mat' \
  --include '**/*.h5' \
  --include '**/*.hdf5' \
  --include '**/*.nc' \
  --include '**/*.csv' \
  --include '**/*.json' \
  --include '**/*.jsonl' \
  --include '**/*.yaml' \
  --include '**/*.yml' \
  --include '**/*.txt' \
  --include '**/*.log' \
  --include '**/*.png' \
  --include '**/*.jpg' \
  --include '**/*.jpeg' \
  --include '**/*.pdf' \
  --include '**/*.svg' \
  --include '**/*.gif' \
  --include '**/*.mp4' \
  --include '**/*.webm' \
  --include '**/*.zip' \
  --include '**/*.tar' \
  --include '**/*.tar.gz' \
  --include '**/*.tgz' \
  --exclude '*' \
  2>&1 | tee -a "$LOG"

printf '\nCompleted: %s\n' "$(date -Iseconds)" | tee -a "$LOG"
