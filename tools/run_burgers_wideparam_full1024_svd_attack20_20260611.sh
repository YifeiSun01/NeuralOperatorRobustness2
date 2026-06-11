#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/workspace/NeuralOperatorRobustness2}"
PY="${PY:-/venv/adv_robust/bin/python}"
OUT_ROOT="${OUT_ROOT:-$ROOT/forensics/burgers_wideparam_loss3targeted_full1024_svd_attack20_20260611}"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/burgers_wideparam_full1024_svd_attack20_20260611.log}"

mkdir -p "$OUT_ROOT" "$LOG_DIR"

echo "[start] $(date -Is)"
echo "[root] $ROOT"
echo "[python] $PY"
echo "[out] $OUT_ROOT"
echo "[log] $LOG_FILE"

cd "$ROOT"
stdbuf -oL -eL "$PY" "$ROOT/tools/run_burgers_wideparam_full1024_svd_attack20_20260611.py" \
  --out-root "$OUT_ROOT" \
  "$@" 2>&1 | tee "$LOG_FILE"

echo "[done] $(date -Is)"
