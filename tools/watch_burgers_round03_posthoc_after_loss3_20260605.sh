#!/usr/bin/env bash
# Wait for round03 loss3 long-training completion, then run basic posthoc stages through final SVD.

ROOT=/workspace/NeuralOperatorRobustness2
cd "$ROOT" || exit 1
PY="$ROOT/adv_robust/bin/python"
SUMMARY="$ROOT/adversarial_training_runs/burgers_loss3_selective_round03_loss3_500ep_long_20260605/summary.json"
EVAL="$ROOT/adversarial_training_runs/burgers_loss3_selective_round03_loss3_500ep_long_20260605/burgers/eval_split_summary.csv"
STAMP_FILE="$ROOT/adversarial_training_runs/burgers_loss3_selective_round03_full_pipeline_20260605_logs/posthoc_after_loss3_watcher_v2.status"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] watcher v2 started; waiting for $SUMMARY"
echo "watcher_v2_started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STAMP_FILE"

while [ ! -f "$SUMMARY" ]; do
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] loss3 still running; latest eval:"
  if [ -f "$EVAL" ]; then
    tail -4 "$EVAL"
  else
    echo "eval file not found yet: $EVAL"
  fi
  sleep 60
done

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] loss3 summary found; starting posthoc stages"
echo "posthoc_started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STAMP_FILE"
"$PY" "$ROOT/tools/run_burgers_loss3_selective_round03_full_pipeline_20260605.py" \
  --stages summarize,plots,gradient,svd-final \
  --skip-existing
rc=$?
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] posthoc stages exited rc=$rc"
echo "posthoc_exit_rc=$rc" >> "$STAMP_FILE"
echo "posthoc_finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STAMP_FILE"
exit "$rc"
