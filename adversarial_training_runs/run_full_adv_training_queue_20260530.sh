#!/usr/bin/env bash
set +e
cd /workspace/NeuralOperatorRobustness2

QUEUE_ROOT="adversarial_training_runs/full_10step_two_modes_queue_20260530"
LOG_ROOT="$QUEUE_ROOT/logs"
STATUS_CSV="$QUEUE_ROOT/queue_status.csv"
mkdir -p "$LOG_ROOT"

if [ ! -f "$STATUS_CSV" ]; then
  echo "case_id,task,training_data_mode,status,start_utc,end_utc,exit_code,log_path" > "$STATUS_CSV"
fi

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

run_case burgers_adv_only burgers adv-only \
  "$PYTHON" "${COMMON[@]}" --tasks burgers --run-name full10_burgers_adv_only_20260530 --training-data-mode adv-only \
  --burgers-attack-steps 10 --burgers-batch-size 1350 --burgers-optimizer-batch-size 32 \
  --burgers-solver-remat chunk --burgers-solver-remat-chunk-steps 50

run_case burgers_clean_plus_adv burgers clean-plus-adv \
  "$PYTHON" "${COMMON[@]}" --tasks burgers --run-name full10_burgers_clean_plus_adv_20260530 --training-data-mode clean-plus-adv \
  --burgers-attack-steps 10 --burgers-batch-size 1350 --burgers-optimizer-batch-size 32 \
  --burgers-solver-remat chunk --burgers-solver-remat-chunk-steps 50

run_case darcy_adv_only darcy adv-only \
  "$PYTHON" "${COMMON[@]}" --tasks darcy --run-name full10_darcy_adv_only_20260530 --training-data-mode adv-only \
  --darcy-attack-steps 10 --darcy-batch-size 416 --darcy-optimizer-batch-size 128

run_case darcy_clean_plus_adv darcy clean-plus-adv \
  "$PYTHON" "${COMMON[@]}" --tasks darcy --run-name full10_darcy_clean_plus_adv_20260530 --training-data-mode clean-plus-adv \
  --darcy-attack-steps 10 --darcy-batch-size 416 --darcy-optimizer-batch-size 128

run_case ns2d_adv_only ns2d adv-only \
  "$PYTHON" "${COMMON[@]}" --tasks ns2d --run-name full10_ns2d_adv_only_20260530 --training-data-mode adv-only \
  --ns2d-attack-steps 10 --ns2d-alpha-ratio 0.2 --ns2d-batch-size 6 --ns2d-optimizer-batch-size 1 \
  --ns2d-solver-remat chunk --ns2d-solver-remat-chunk-steps 20

run_case ns2d_clean_plus_adv ns2d clean-plus-adv \
  "$PYTHON" "${COMMON[@]}" --tasks ns2d --run-name full10_ns2d_clean_plus_adv_20260530 --training-data-mode clean-plus-adv \
  --ns2d-attack-steps 10 --ns2d-alpha-ratio 0.2 --ns2d-batch-size 6 --ns2d-optimizer-batch-size 1 \
  --ns2d-solver-remat chunk --ns2d-solver-remat-chunk-steps 20

finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
echo "[$finished_utc] QUEUE FINISHED" | tee -a "$QUEUE_ROOT/queue.log"
