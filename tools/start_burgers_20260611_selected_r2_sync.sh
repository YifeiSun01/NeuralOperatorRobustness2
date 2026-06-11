#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 R2:bucket/prefix" >&2
  exit 2
fi

DEST="${1%/}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$PROJECT_ROOT/forensics/burgers_20260611_selected_r2_sync"
LOG="$RUN_DIR/rclone_selected_sync.log"
MANIFEST="$RUN_DIR/selected_sync_manifest.txt"
mkdir -p "$RUN_DIR"

paths=(
  "generalization_datasets_burgers_semantic_loss3fav_search_20260611"
  "generalization_datasets_burgers_semantic_smooth_loss3_screen_20260611"
  "generalization_datasets_burgers_semantic_wideparam_visible_20260611"
  "generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611"
  "forensics/burgers_semantic_loss3fav_round01_dataset_range_audit_20260611"
  "forensics/burgers_semantic_loss3fav_round01_p2q2_multi_sample_attack_visuals_20260611"
  "forensics/burgers_semantic_smooth_loss3_screen_round00_20260611"
  "forensics/burgers_semantic_smooth_loss3_screen_round00_clean_loss_significance_20260611"
  "forensics/burgers_semantic_smooth_loss3_screen_round00_p2q2_multi_sample_attack_visuals_20260611"
  "forensics/burgers_semantic_smooth_loss3_screen_round00_p2q2_multi_sample_attack_visuals_batched_20260611"
  "forensics/burgers_semantic_wideparam_visible_round00_20260611"
  "forensics/burgers_semantic_wideparam_visible_round00_clean_loss_final_models_20260611"
  "forensics/burgers_semantic_wideparam_visible_loss3targeted_round00_20260611"
  "forensics/burgers_semantic_wideparam_visible_loss3targeted_round00_clean_loss_final_models_20260611"
  "forensics/burgers_semantic_wideparam_visible_loss3targeted_round00_52dataset_clean_metrics_20260611"
  "forensics/burgers_semantic_wideparam_visible_loss3targeted_round00_p2q2_diverse_multi_sample_attack_visuals_batched_20260611"
  "forensics/burgers_wideparam_loss3targeted_full1024_svd_attack3_20260611"
  "visualizations/burgers_semantic_loss3fav_round01_p2q2_comparison_dense_multi_sample_bundle_20260611"
  "visualizations/burgers_semantic_smooth_loss3_screen_round00_p2q2_comparison_dense_multi_sample_bundle_20260611"
  "visualizations/burgers_semantic_smooth_loss3_screen_round00_p2q2_comparison_dense_multi_sample_batched_bundle_20260611"
  "visualizations/burgers_semantic_wideparam_visible_loss3targeted_round00_p2q2_comparison_dense_diverse_multi_sample_batched_wrapped_labels_20260611"
)

{
  printf 'Started: %s
' "$(date -Iseconds)"
  printf 'Project root: %s
' "$PROJECT_ROOT"
  printf 'Destination: %s
' "$DEST"
  printf 'Mode: selected 2026-06-11 Burgers artifact upload; rclone copy; size-only incremental; no delete.
'
  printf '
Paths:
'
  printf '%s
' "${paths[@]}"
} > "$MANIFEST"

cd "$PROJECT_ROOT"
: > "$LOG"

for path in "${paths[@]}"; do
  if [[ ! -e "$path" ]]; then
    printf '[%s] SKIP missing %s
' "$(date -Iseconds)" "$path" | tee -a "$LOG"
    continue
  fi
  printf '[%s] COPY %s -> %s/%s
' "$(date -Iseconds)" "$path" "$DEST" "$path" | tee -a "$LOG"
  rclone copy "$path" "$DEST/$path"     --s3-no-check-bucket     --s3-disable-checksum     --retries 5     --low-level-retries 5     --transfers 8     --checkers 24     --fast-list     --size-only     --stats 60s     --stats-one-line 2>&1 | tee -a "$LOG"
done

printf '
Completed: %s
' "$(date -Iseconds)" | tee -a "$LOG"
