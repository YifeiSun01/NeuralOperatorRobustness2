#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 R2:bucket/prefix" >&2
  exit 2
fi

DEST="${1%/}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$PROJECT_ROOT/forensics/full_r2_sync_20260609"
LOG="$RUN_DIR/rclone_full_sync.log"
MANIFEST="$RUN_DIR/full_sync_manifest.txt"

mkdir -p "$RUN_DIR"

roots=()
for path in \
  1D_Burgers \
  1D_Burgers_FNO_generalization \
  1D_Burgers_deeponet \
  2D_Darcy_FNO2d \
  2D_NS_FNO2d_recurrent \
  adversarial_training_runs \
  forensics \
  visualizations \
  run_logs; do
  [[ -e "$PROJECT_ROOT/$path" ]] && roots+=("$path")
done

shopt -s nullglob
for path in generalization_datasets* generalization_eval*; do
  [[ -e "$PROJECT_ROOT/$path" ]] && roots+=("$path")
done
shopt -u nullglob

{
  printf 'Started: %s\n' "$(date -Iseconds)"
  printf 'Project root: %s\n' "$PROJECT_ROOT"
  printf 'Destination: %s\n' "$DEST"
  printf 'Roots:\n'
  printf '  %s\n' "${roots[@]}"
} > "$MANIFEST"

cd "$PROJECT_ROOT"

for root in "${roots[@]}"; do
  printf '\n[%s] COPY %s -> %s/%s\n' "$(date -Iseconds)" "$root" "$DEST" "$root" | tee -a "$LOG"
  rclone copy "$root" "$DEST/$root" \
    --s3-no-check-bucket \
    --s3-disable-checksum \
    --retries 3 \
    --low-level-retries 2 \
    --transfers 8 \
    --checkers 16 \
    --fast-list \
    --stats 60s \
    --stats-one-line \
    --exclude '.git/**' \
    --exclude 'adv_robust/**' \
    --exclude '__pycache__/**' \
    --exclude '*.pyc' \
    --exclude '.DS_Store' \
    2>&1 | tee -a "$LOG"
done

printf '\nCompleted: %s\n' "$(date -Iseconds)" | tee -a "$LOG"
