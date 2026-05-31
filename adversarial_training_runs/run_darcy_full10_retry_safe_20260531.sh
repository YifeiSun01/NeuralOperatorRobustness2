#!/usr/bin/env bash
set +e
cd /workspace/NeuralOperatorRobustness2

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

QUEUE_ROOT="adversarial_training_runs/darcy_full10_retry_safe_20260531"
LOG_ROOT="$QUEUE_ROOT/logs"
STATUS_CSV="$QUEUE_ROOT/queue_status.csv"
mkdir -p "$LOG_ROOT"

echo "case_id,task,training_data_mode,status,start_utc,end_utc,exit_code,log_path" > "$STATUS_CSV"

run_case() {
  local case_id="$1"
  local task="$2"
  local mode="$3"
  shift 3
  local log_path="$LOG_ROOT/${case_id}.log"
  local start_utc
  start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  echo "[$start_utc] START $case_id task=$task mode=$mode" | tee -a "$QUEUE_ROOT/queue.log"
  echo "$case_id,$task,$mode,running,$start_utc,,,$log_path" >> "$STATUS_CSV"

  stdbuf -oL -eL "$@" > "$log_path" 2>&1
  local exit_code=$?
  local end_utc
  end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  local status="done"
  if [ "$exit_code" -ne 0 ]; then
    status="failed"
  fi
  echo "[$end_utc] END $case_id status=$status exit=$exit_code" | tee -a "$QUEUE_ROOT/queue.log"
  echo "$case_id,$task,$mode,$status,$start_utc,$end_utc,$exit_code,$log_path" >> "$STATUS_CSV"
}

PYTHON="adv_robust/bin/python"
COMMON=(tools/adversarial_training.py --label-mode solver --eval-every-fraction 0.2 --eval-max-samples 50 --max-generalization-eval 50)
DARCY_SAFE=(--darcy-attack-steps 10 --darcy-batch-size 416 --darcy-optimizer-batch-size 128)

run_case darcy_adv_only_retry_safe darcy adv-only \
  "$PYTHON" "${COMMON[@]}" --tasks darcy --run-name full10_darcy_adv_only_retry_safe_20260531 \
  --training-data-mode adv-only "${DARCY_SAFE[@]}"

run_case darcy_clean_plus_adv_retry_safe darcy clean-plus-adv \
  "$PYTHON" "${COMMON[@]}" --tasks darcy --run-name full10_darcy_clean_plus_adv_retry_safe_20260531 \
  --training-data-mode clean-plus-adv "${DARCY_SAFE[@]}"

finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
echo "[$finished_utc] DARCY RETRY QUEUE FINISHED" | tee -a "$QUEUE_ROOT/queue.log"
