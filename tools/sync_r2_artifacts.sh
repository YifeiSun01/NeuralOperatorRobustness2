#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  R2_REMOTE=r2:BUCKET/NeuralOperatorRobustness2 tools/sync_r2_artifacts.sh upload PATH [PATH ...]
  R2_REMOTE=r2:BUCKET/NeuralOperatorRobustness2 tools/sync_r2_artifacts.sh check PATH [PATH ...]
  R2_REMOTE=r2:BUCKET/NeuralOperatorRobustness2 tools/sync_r2_artifacts.sh download PATH [PATH ...]

Examples:
  R2_REMOTE=r2:my-bucket/NeuralOperatorRobustness2 \
    tools/sync_r2_artifacts.sh upload results/three_loss_batch100_full_loss3_delta_rerun_20260514

  R2_REMOTE=r2:my-bucket/NeuralOperatorRobustness2 \
    tools/sync_r2_artifacts.sh check 1D_Burgers/datasets 1D_Burgers/trained_models

This script never syncs .git. Excludes are read from .r2exclude.
USAGE
}

if [[ $# -lt 2 ]]; then
  usage
  exit 2
fi

mode="$1"
shift

if [[ -z "${R2_REMOTE:-}" ]]; then
  echo "error: set R2_REMOTE, for example r2:YOUR_BUCKET/NeuralOperatorRobustness2" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

exclude_file="${R2_EXCLUDE_FILE:-.r2exclude}"
if [[ ! -f "${exclude_file}" ]]; then
  echo "error: exclude file not found: ${exclude_file}" >&2
  exit 2
fi

remote_root="${R2_REMOTE%/}"

for path in "$@"; do
  clean_path="${path#./}"
  local_path="${clean_path%/}"
  remote_path="${remote_root}/${local_path}"

  case "${mode}" in
    upload)
      if [[ ! -e "${local_path}" ]]; then
        echo "error: local path not found: ${local_path}" >&2
        exit 1
      fi
      echo "[r2 upload] ${local_path} -> ${remote_path}"
      rclone sync "${local_path}" "${remote_path}" --checksum --exclude-from "${exclude_file}"
      ;;
    check)
      if [[ ! -e "${local_path}" ]]; then
        echo "error: local path not found: ${local_path}" >&2
        exit 1
      fi
      echo "[r2 check] ${local_path} <-> ${remote_path}"
      rclone check "${local_path}" "${remote_path}" --checksum --exclude-from "${exclude_file}"
      ;;
    download)
      echo "[r2 download] ${remote_path} -> ${local_path}"
      mkdir -p "$(dirname "${local_path}")"
      rclone sync "${remote_path}" "${local_path}" --checksum --exclude-from "${exclude_file}"
      ;;
    *)
      usage
      exit 2
      ;;
  esac
done
