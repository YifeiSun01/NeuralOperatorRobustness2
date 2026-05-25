#!/usr/bin/env bash
set -u

cd /workspace/NeuralOperatorRobustness2

PYTHON_BIN="${PYTHON_BIN:-adv_robust/bin/python}"
BASE_SWEEP_ROOT="${1:-${BASE_SWEEP_ROOT:-}}"
if [[ -z "$BASE_SWEEP_ROOT" ]]; then
  echo "ERROR: pass BASE_SWEEP_ROOT as arg 1 or env var" >&2
  exit 2
fi

METRICS="${LOSS3_METRICS:-dists ms_ssim scattering2d affine_dists local_warp_dists homography_dists tps_dists elastic_dists svf_dists}"
PRESET_DIRS="${BUDGET_PRESET_DIRS:-1_very_strong 2_strong 3_medium 4_weak}"
POLL_SECONDS="${POLL_SECONDS:-60}"

metric_ready() {
  local preset_root="$1"
  local metric="$2"
  local count
  count=$(find "$preset_root/eps32_alpha10/$metric" -path '*/mode_*/batch_0000_0009/loss3/steepest_add/final_state_outputs.npz' -type f 2>/dev/null | wc -l)
  [[ "$count" == "1" ]]
}

preset_ready() {
  local preset_root="$1"
  local metric
  for metric in $METRICS; do
    if ! metric_ready "$preset_root" "$metric"; then
      return 1
    fi
  done
  return 0
}

upload_to_r2_if_enabled() {
  local src="$1"
  local label="$2"
  if [[ "${AUTO_UPLOAD_R2:-0}" != "1" ]]; then
    echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') r2_upload_${label}=disabled"
    return 0
  fi
  echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') r2_upload_started_${label}=$src"
  if tools/upload_path_to_r2_20260525.sh "$src"; then
    echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') r2_upload_finished_${label}=$src"
  else
    echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') r2_upload_failed_${label}=$src" >&2
  fi
}

plot_preset() {
  local preset_root="$1"
  local marker="$preset_root/figures_dataset0/.auto_plot_dataset0_done"
  local alt_root="$preset_root/eps32_alpha10"
  local out_dir="$preset_root/figures_dataset0"
  mkdir -p "$out_dir"

  if [[ -f "$marker" ]]; then
    echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') already_plotted=$preset_root"
    return 0
  fi

  echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') plotting_started=$preset_root"
  if "$PYTHON_BIN" tools/plot_ns2d_eps32_alpha10_altloss_heatmaps_spectrum_loss_curves_cleanstyle_20260525.py \
      --alt-root "$alt_root" \
      --out-dir "$out_dir"; then
    echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') cleanstyle_ok=$preset_root"
  else
    echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') cleanstyle_failed=$preset_root" >&2
    return 1
  fi

  if "$PYTHON_BIN" tools/plot_ns2d_alignment_before_after_diff_20260525.py \
      --alt-root "$alt_root" \
      --out-dir "$out_dir"; then
    echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') before_after_ok=$preset_root"
  else
    echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') before_after_failed=$preset_root" >&2
    return 1
  fi

  date -u '+%Y-%m-%dT%H:%M:%SZ' > "$marker"
  echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') plotting_finished=$preset_root"
  upload_to_r2_if_enabled "$preset_root" "preset_with_figures"
}

echo "watcher_start_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "base_sweep_root=$BASE_SWEEP_ROOT"
echo "metrics=$METRICS"
echo "preset_dirs=$PRESET_DIRS"
echo "poll_seconds=$POLL_SECONDS"

for preset_dir in $PRESET_DIRS; do
  preset_root="$BASE_SWEEP_ROOT/$preset_dir"
  echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') waiting_for=$preset_root"
  while ! preset_ready "$preset_root"; do
    sleep "$POLL_SECONDS"
  done
  plot_preset "$preset_root"
done

echo "watcher_finish_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
