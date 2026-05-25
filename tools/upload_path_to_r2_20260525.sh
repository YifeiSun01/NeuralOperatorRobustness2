#!/usr/bin/env bash
set -euo pipefail

cd /workspace/NeuralOperatorRobustness2

SRC="${1:-}"
if [[ -z "$SRC" ]]; then
  echo "ERROR: usage: $0 <repo-local-path>" >&2
  exit 2
fi
if [[ ! -e "$SRC" ]]; then
  echo "ERROR: source path does not exist: $SRC" >&2
  exit 2
fi

: "${R2_ACCESS_KEY_ID:?missing R2_ACCESS_KEY_ID}"
: "${R2_SECRET_ACCESS_KEY:?missing R2_SECRET_ACCESS_KEY}"

R2_ENDPOINT="${R2_ENDPOINT:-https://606bf6c862a4e8f63dabb6243dce2df7.r2.cloudflarestorage.com}"
R2_BUCKET="${R2_BUCKET:-neural-operator-robustness}"
R2_PREFIX="${R2_PREFIX:-machine-sync/NeuralOperatorRobustness2-selected}"
R2_REMOTE_NAME="${R2_REMOTE_NAME:-r2auto}"
REMOTE_UPPER="${R2_REMOTE_NAME^^}"
REPO_ROOT="/workspace/NeuralOperatorRobustness2"
SRC_ABS="$(realpath "$SRC")"
REL_SRC="${SRC_ABS#${REPO_ROOT}/}"
DEST="${R2_REMOTE_NAME}:${R2_BUCKET}/${R2_PREFIX}/${REL_SRC}"
LOG_ROOT="${R2_UPLOAD_LOG_ROOT:-${REPO_ROOT}/2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/r2_upload_logs}"
mkdir -p "$LOG_ROOT"
SAFE_NAME="$(echo "$REL_SRC" | tr '/ ' '__')"
LOG="$LOG_ROOT/${SAFE_NAME}_$(date -u +%Y%m%d_%H%M%S)_UTC.log"

export "RCLONE_CONFIG_${REMOTE_UPPER}_TYPE=s3"
export "RCLONE_CONFIG_${REMOTE_UPPER}_PROVIDER=Cloudflare"
export "RCLONE_CONFIG_${REMOTE_UPPER}_ACCESS_KEY_ID=${R2_ACCESS_KEY_ID}"
export "RCLONE_CONFIG_${REMOTE_UPPER}_SECRET_ACCESS_KEY=${R2_SECRET_ACCESS_KEY}"
export "RCLONE_CONFIG_${REMOTE_UPPER}_ENDPOINT=${R2_ENDPOINT}"
export "RCLONE_CONFIG_${REMOTE_UPPER}_ACL=private"

{
  echo "upload_start_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "source=$SRC_ABS"
  echo "destination=$DEST"
  if [[ -d "$SRC_ABS" ]]; then
    rclone copy "$SRC_ABS" "$DEST" \
      --transfers "${R2_TRANSFERS:-4}" \
      --checkers "${R2_CHECKERS:-8}" \
      --fast-list \
      --s3-no-check-bucket \
      --exclude '.r2_upload_done' \
      --stats 30s \
      --stats-one-line
  else
    rclone copyto "$SRC_ABS" "$DEST" \
      --s3-no-check-bucket \
      --stats 30s \
      --stats-one-line
  fi
  echo "upload_finish_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
} 2>&1 | tee "$LOG"

echo "r2_upload_log=$LOG"
