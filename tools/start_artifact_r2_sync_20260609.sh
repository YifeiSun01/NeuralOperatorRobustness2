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
  --filter '- .git/' \
  --filter '- .git/**' \
  --filter '- adv_robust/' \
  --filter '- adv_robust/**' \
  --filter '- __pycache__/' \
  --filter '- __pycache__/**' \
  --filter '- **/__pycache__/**' \
  --filter '- **/*.pyc' \
  --filter '- **/.DS_Store' \
  --filter '- **/*.py' \
  --filter '- **/*.sh' \
  --filter '- **/*.md' \
  --filter '+ */' \
  --filter '+ adversarial_training_runs/**' \
  --filter '+ forensics/**' \
  --filter '+ visualizations/**' \
  --filter '+ visualization_outputs/**' \
  --filter '+ analysis_outputs/**' \
  --filter '+ figures/**' \
  --filter '+ results/**' \
  --filter '+ run_logs/**' \
  --filter '+ data/**' \
  --filter '+ Model/**' \
  --filter '+ generalization_datasets*/**' \
  --filter '+ generalization_eval*/**' \
  --filter '+ **/saved_models/**' \
  --filter '+ **/saved_models_expanded/**' \
  --filter '+ **/saved_models_kernel_abstracts/**' \
  --filter '+ **/trained_models/**' \
  --filter '+ **/datasets/**' \
  --filter '+ **/test_train_datasets/**' \
  --filter '+ **/GRFs/**' \
  --filter '+ **/perturbation_results/**' \
  --filter '+ **/perturbed_results_as_inputs/**' \
  --filter '+ **/attack_result_viz/**' \
  --filter '+ **/viz_results/**' \
  --filter '+ **/visualizations/**' \
  --filter '+ **/*.pt' \
  --filter '+ **/*.pth' \
  --filter '+ **/*.ckpt' \
  --filter '+ **/*.safetensors' \
  --filter '+ **/*.npz' \
  --filter '+ **/*.npy' \
  --filter '+ **/*.mat' \
  --filter '+ **/*.h5' \
  --filter '+ **/*.hdf5' \
  --filter '+ **/*.nc' \
  --filter '+ **/*.csv' \
  --filter '+ **/*.json' \
  --filter '+ **/*.jsonl' \
  --filter '+ **/*.yaml' \
  --filter '+ **/*.yml' \
  --filter '+ **/*.txt' \
  --filter '+ **/*.log' \
  --filter '+ **/*.png' \
  --filter '+ **/*.jpg' \
  --filter '+ **/*.jpeg' \
  --filter '+ **/*.pdf' \
  --filter '+ **/*.svg' \
  --filter '+ **/*.gif' \
  --filter '+ **/*.mp4' \
  --filter '+ **/*.webm' \
  --filter '+ **/*.zip' \
  --filter '+ **/*.tar' \
  --filter '+ **/*.tar.gz' \
  --filter '+ **/*.tgz' \
  --filter '- *' \
  2>&1 | tee -a "$LOG"

printf '\nCompleted: %s\n' "$(date -Iseconds)" | tee -a "$LOG"
