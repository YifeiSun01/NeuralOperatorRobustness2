#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-adv_robust/bin/python}"
ROOT="${ROOT:-results/three_loss_objective_round1}"
OUT_SUBDIR="${OUT_SUBDIR:-gif_frames}"
FPS="${FPS:-8}"
STRIDE="${STRIDE:-2}"
CLEANUP_FRAMES="${CLEANUP_FRAMES:-1}"
FINAL_PNG_ONLY="${FINAL_PNG_ONLY:-0}"

while IFS= read -r summary_path; do
  run_dir="$(dirname "${summary_path}")"
  if [[ ! -f "${run_dir}/trajectory.npz" ]]; then
    echo "[skip] no trajectory.npz: ${run_dir}"
    continue
  fi
  args=(
    --run_dir "${run_dir}"
    --outdir "${run_dir}/${OUT_SUBDIR}"
    --fps "${FPS}"
    --stride "${STRIDE}"
  )
  if [[ "${CLEANUP_FRAMES}" == "1" ]]; then
    args+=(--cleanup_frames)
  fi
  if [[ "${FINAL_PNG_ONLY}" == "1" ]]; then
    args+=(--final_png_only)
  fi
  "${PYTHON_BIN}" plot_three_loss_objective_gif.py "${args[@]}"
done < <(find "${ROOT}" -name summary.json | sort)
