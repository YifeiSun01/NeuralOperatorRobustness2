#!/usr/bin/env bash
set -euo pipefail

BATCH_SIZE="${BATCH_SIZE:-100}"
SEED="${SEED:-20260511}"
STEPS="${STEPS:-100}"
GPU_IDS="${GPU_IDS:-0}"
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"

source adv_robust/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export PYTHONUNBUFFERED=1

indices="$(python3 -c "import random; random.seed(int(\"$SEED\")); print(\" \".join(map(str, sorted(random.sample(range(150), int(\"$BATCH_SIZE\"))))))")"

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
MAX_PARALLEL=$(( ${#GPU_ARRAY[@]} * JOBS_PER_GPU ))

FUTURE_OUT="results/burgers_future_loss3_bad_search_batch100_random100_sequential"
SETTING18_OUT="results/burgers_loss3_18setting_batch100_random100_losses_parallel6"

# Exact DeepONet result directories deleted after discovering the missing output-transform bug.
# Format: out_base|nu|norm|epsilon|alpha
settings=(
  "${FUTURE_OUT}|0.01|inf|0.025|0.001"
  "${FUTURE_OUT}|0.01|inf|0.05|0.002"
  "${FUTURE_OUT}|0.01|inf|0.075|0.003"
  "${FUTURE_OUT}|0.01|inf|0.75|0.03"
  "${FUTURE_OUT}|0.01|inf|1.0|0.04"
  "${FUTURE_OUT}|0.01|2|0.5|0.009"
  "${FUTURE_OUT}|0.01|2|1.0|0.01875"
  "${FUTURE_OUT}|0.01|2|2.0|0.0375"
  "${FUTURE_OUT}|0.01|2|12.0|0.225"
  "${SETTING18_OUT}|0.01|inf|0.05|0.002"
  "${SETTING18_OUT}|0.01|inf|0.75|0.03"
  "${SETTING18_OUT}|0.01|2|1.0|0.01875"
)

tag_float() { echo "$1" | sed "s/\./p/g"; }
norm_tag_for() {
  if [ "$1" = "inf" ]; then echo "linf"; else echo "l${1}"; fi
}

run_case() {
  local out_base="$1" nu="$2" norm="$3" epsilon="$4" alpha="$5" gpu="$6"
  local model="deeponet"
  local runner="run_burgers_deeponet_corrected_oldstyle_5loss.py"

  local nu_tag eps_tag alpha_tag norm_tag root tag batch_tag log status_file
  nu_tag="$(tag_float "$nu")"
  eps_tag="$(tag_float "$epsilon")"
  alpha_tag="$(tag_float "$alpha")"
  norm_tag="$(norm_tag_for "$norm")"

  mkdir -p "$out_base"
  status_file="${out_base}/deeponet_fixed_rerun_status.csv"
  if [ ! -s "$status_file" ]; then
    echo "timestamp_utc,model,nu,norm,epsilon,alpha,status,elapsed_seconds,output_root,log" > "$status_file"
  fi

  root="${out_base}/${model}_nu${nu_tag}_${norm_tag}_eps${eps_tag}_alpha${alpha_tag}_batch${BATCH_SIZE}_seed${SEED}"
  tag="${model}_nu${nu_tag}_domain2_${norm_tag}_eps${eps_tag}_alpha${alpha_tag}_steps${STEPS}_index{index}_batch${BATCH_SIZE}_losses_rs"
  batch_tag="${model}_nu${nu_tag}_domain2_${norm_tag}_eps${eps_tag}_alpha${alpha_tag}_steps${STEPS}_batch${BATCH_SIZE}_losses_seed${SEED}_rs"
  log="${out_base}/${model}_nu${nu_tag}_${norm_tag}_eps${eps_tag}_alpha${alpha_tag}_fixed_rerun.log"

  if [ -s "$root/true_loss_summary_aggregate.csv" ] && [ -s "$root/ratio_summary/burgers_attack_ratio_by_parameter.csv" ]; then
    echo "[skip] complete: ${root}"
    echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ'),${model},${nu},${norm},${epsilon},${alpha},skipped_complete,0,${root},${log}" >> "$status_file"
    return 0
  fi

  local start_ts end_ts elapsed status
  start_ts="$(date +%s)"
  status="complete"
  echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ'),${model},${nu},${norm},${epsilon},${alpha},started,0,${root},${log}" >> "$status_file"

  {
    echo "============================================================"
    echo "[attack] fixed DeepONet output transform"
    echo "[attack] nu=${nu} norm=${norm} eps=${epsilon} alpha=${alpha} batch=${BATCH_SIZE} gpu=${gpu}"
    echo "[root]   ${root}"
    echo "============================================================"

    CUDA_VISIBLE_DEVICES="$gpu" python -u "$runner" \
      --nu "$nu" \
      --domain 2.0 \
      --nx 1024 \
      --t_final 1.0 \
      --dt 0.001 \
      --norm "$norm" \
      --epsilon "$epsilon" \
      --alpha "$alpha" \
      --steps "$STEPS" \
      --indices $indices \
      --output_root "$root" \
      --tag "$tag" \
      --batch_tag "$batch_tag" \
      --record_steps losses

    python -u summarize_burgers_corrected_oldstyle_true_losses.py \
      --result_root "$root" \
      --outdir "$root"

    python -u summarize_burgers_attack_ratios.py \
      --result_root "$root" \
      --outdir "$root/ratio_summary"
  } > "$log" 2>&1 || status="failed"

  end_ts="$(date +%s)"
  elapsed="$((end_ts - start_ts))"
  echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ'),${model},${nu},${norm},${epsilon},${alpha},${status},${elapsed},${root},${log}" >> "$status_file"

  if [ "$status" != "complete" ]; then
    echo "[failed] ${root}; see ${log}" >&2
    return 1
  fi
}

echo "[deeponet-fixed-rerun] settings=${#settings[@]} GPU_IDS=${GPU_IDS} JOBS_PER_GPU=${JOBS_PER_GPU} MAX_PARALLEL=${MAX_PARALLEL}"
echo "[deeponet-fixed-rerun] BATCH_SIZE=${BATCH_SIZE} SEED=${SEED} STEPS=${STEPS}"

active=0
launched=0
for setting in "${settings[@]}"; do
  IFS='|' read -r out_base nu norm epsilon alpha <<< "$setting"
  gpu="${GPU_ARRAY[$(( launched % ${#GPU_ARRAY[@]} ))]}"

  run_case "$out_base" "$nu" "$norm" "$epsilon" "$alpha" "$gpu" &
  pid=$!
  echo "[launch] pid=${pid} gpu=${gpu} ${setting}"

  launched=$((launched + 1))
  active=$((active + 1))

  if [ "$active" -ge "$MAX_PARALLEL" ]; then
    wait -n
    active=$((active - 1))
  fi
done

while [ "$active" -gt 0 ]; do
  wait -n
  active=$((active - 1))
done

python -u summarize_burgers_attack_ratios.py \
  --result_root "$FUTURE_OUT" \
  --outdir "$FUTURE_OUT/global_ratio_summary"

python -u summarize_burgers_attack_ratios.py \
  --result_root "$SETTING18_OUT" \
  --outdir "$SETTING18_OUT/global_ratio_summary"

python -u tools/rebuild_burgers_clean_summary.py \
  --root results \
  --outdir results/clean_recomputed_summary

cp results/clean_recomputed_summary/README.md results/burgers_loss3_clean_recomputed_summary.md

echo
echo "[done] fixed DeepONet 12-directory rerun finished"
echo "[done] future output: ${FUTURE_OUT}"
echo "[done] 18setting output: ${SETTING18_OUT}"
echo "[done] clean summary: results/burgers_loss3_clean_recomputed_summary.md"
