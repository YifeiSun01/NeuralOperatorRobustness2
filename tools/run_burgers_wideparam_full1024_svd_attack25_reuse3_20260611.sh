#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/workspace/NeuralOperatorRobustness2}"
PY="${PY:-/venv/adv_robust/bin/python}"
OUT_ROOT="${OUT_ROOT:-$ROOT/forensics/burgers_wideparam_loss3targeted_full1024_svd_attack25_reuse3_20260611}"
REPORT_MD="${REPORT_MD:-$ROOT/docs/burgers_wideparam_loss3targeted_full1024_svd_attack25_reuse3_20260611.md}"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/burgers_wideparam_full1024_svd_attack25_reuse3_20260611.log}"

mkdir -p "$OUT_ROOT" "$LOG_DIR"

{
  echo "[start] $(date -Is)"
  echo "[root] $ROOT"
  echo "[python] $PY"
  echo "[out] $OUT_ROOT"
  echo "[report] $REPORT_MD"
  echo "[log] $LOG_FILE"
} | tee "$LOG_FILE"

cd "$ROOT"
stdbuf -oL -eL "$PY" "$ROOT/tools/run_burgers_wideparam_full1024_svd_attack25_reuse3_20260611.py" \
  --out-root "$OUT_ROOT" \
  --report-md "$REPORT_MD" \
  "$@" 2>&1 | tee -a "$LOG_FILE"

echo "[done] $(date -Is)" | tee -a "$LOG_FILE"
