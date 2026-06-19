#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

REMOTE_PREFIX="${REMOTE_PREFIX:-R2:neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected}"
TRANSFERS="${TRANSFERS:-8}"
CHECKERS="${CHECKERS:-16}"

FINAL_CHECKPOINTS=(
  "adversarial_training_runs/burgers_wideparam_loss1_8000ep_retrain_20260611/burgers/checkpoints/burgers_epoch8000_step008000.pt"
  "adversarial_training_runs/burgers_wideparam_loss2_2000ep_retrain_20260611/burgers/checkpoints/burgers_epoch2000_step002000.pt"
  "adversarial_training_runs/burgers_wideparam_loss3_1000ep_retrain_20260611/burgers/checkpoints/burgers_epoch1000_step001000.pt"
  "adversarial_training_runs/burgers_wideparam_random_field_clean_y_8000ep_continue_20260613/burgers/checkpoints/burgers_epoch8000_step008000.pt"
  "adversarial_training_runs/burgers_wideparam_random_field_solver_y_7860ep_continue_20260613/burgers/checkpoints/burgers_epoch7860_step007860.pt"
)

METADATA_FILES=(
  "README.md"
  "run_config.json"
  "summary.json"
  "burgers/summary.json"
  "burgers/eval_split_summary.csv"
  "burgers/attack_epoch_summary.csv"
  "burgers/attack_epsilon_bucket_summary.csv"
)

remote_name="${REMOTE_PREFIX%%:*}:"
if ! rclone listremotes | grep -Fxq "$remote_name"; then
  echo "Missing rclone remote '$remote_name'. Configure R2 or set REMOTE_PREFIX to an accessible remote prefix." >&2
  exit 2
fi

for checkpoint in "${FINAL_CHECKPOINTS[@]}"; do
  echo "Restoring $checkpoint"
  mkdir -p "$(dirname "$checkpoint")"
  rclone copyto "${REMOTE_PREFIX}/${checkpoint}" "$checkpoint" \
    --transfers "$TRANSFERS" \
    --checkers "$CHECKERS" \
    --progress
done

for checkpoint in "${FINAL_CHECKPOINTS[@]}"; do
  run_dir="${checkpoint%%/burgers/checkpoints/*}"
  for rel_meta in "${METADATA_FILES[@]}"; do
    src="${REMOTE_PREFIX}/${run_dir}/${rel_meta}"
    dst="${run_dir}/${rel_meta}"
    mkdir -p "$(dirname "$dst")"
    rclone copyto "$src" "$dst" \
      --transfers 1 \
      --checkers 4 \
      --ignore-existing \
      --error-on-no-transfer=false >/dev/null 2>&1 || true
  done
done

missing=0
for checkpoint in "${FINAL_CHECKPOINTS[@]}"; do
  if [[ -f "$checkpoint" ]]; then
    bytes="$(stat -c '%s' "$checkpoint")"
    echo "FOUND $checkpoint ($bytes bytes)"
  else
    echo "MISSING $checkpoint" >&2
    missing=1
  fi
done

exit "$missing"
