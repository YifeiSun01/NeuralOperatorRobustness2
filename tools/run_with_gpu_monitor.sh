#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: bash tools/run_with_gpu_monitor.sh <command> [args...]"
  echo "Example: bash tools/run_with_gpu_monitor.sh python -u path/to/train.py"
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/adv_robust}"
RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/run_logs}"
LOG_DIR="${LOG_DIR:-$LOG_ROOT/$RUN_NAME}"
GPU_MONITOR_INTERVAL="${GPU_MONITOR_INTERVAL:-5}"

mkdir -p "$LOG_DIR"

if [ -f "$VENV_DIR/bin/activate" ]; then
  # shellcheck source=/dev/null
  source "$VENV_DIR/bin/activate"
fi

export PROJECT_ROOT
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp}"

GPU_CSV="$LOG_DIR/gpu_samples.csv"
PROC_CSV="$LOG_DIR/gpu_processes.csv"
RUN_LOG="$LOG_DIR/run.log"
META_TXT="$LOG_DIR/run_metadata.txt"

{
  echo "project_root=$PROJECT_ROOT"
  echo "venv_dir=$VENV_DIR"
  echo "run_name=$RUN_NAME"
  echo "log_dir=$LOG_DIR"
  echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  echo "gpu_monitor_interval=$GPU_MONITOR_INTERVAL"
  echo "command=$*"
  echo "started_at=$(date --iso-8601=seconds)"
  echo
  nvidia-smi -L || true
} > "$META_TXT"

echo "timestamp,gpu_index,gpu_name,utilization_gpu_percent,memory_used_mib,memory_total_mib,power_draw_w,temperature_c" > "$GPU_CSV"
echo "timestamp,pid,process_name,used_gpu_memory_mib" > "$PROC_CSV"

monitor_gpu() {
  while true; do
    ts="$(date --iso-8601=seconds)"
    nvidia-smi \
      --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
      --format=csv,noheader,nounits 2>/dev/null \
      | awk -v ts="$ts" 'BEGIN { FS=", "; OFS="," } NF >= 7 { print ts,$1,$2,$3,$4,$5,$6,$7 }' \
      >> "$GPU_CSV" || true

    nvidia-smi \
      --query-compute-apps=pid,process_name,used_memory \
      --format=csv,noheader,nounits 2>/dev/null \
      | awk -v ts="$ts" 'BEGIN { FS=", "; OFS="," } NF >= 3 { print ts,$1,$2,$3 }' \
      >> "$PROC_CSV" || true

    sleep "$GPU_MONITOR_INTERVAL"
  done
}

monitor_gpu &
MONITOR_PID=$!

cleanup() {
  kill "$MONITOR_PID" 2>/dev/null || true
  wait "$MONITOR_PID" 2>/dev/null || true
  {
    echo
    echo "finished_at=$(date --iso-8601=seconds)"
    echo "exit_code=${RUN_EXIT_CODE:-unknown}"
  } >> "$META_TXT"
}
trap cleanup EXIT

echo "[monitor] log_dir=$LOG_DIR"
echo "[monitor] command=$*"

set +e
"$@" 2>&1 | tee "$RUN_LOG"
RUN_EXIT_CODE=${PIPESTATUS[0]}
set -e

exit "$RUN_EXIT_CODE"
