#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/workspace/NeuralOperatorRobustness2}"
PY="${PY:-/venv/adv_robust/bin/python}"
GEN_ROOT="${GEN_ROOT:-$ROOT/generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00}"
OUT_ROOT="${OUT_ROOT:-$ROOT/adversarial_training_runs}"
LOG_ROOT="${LOG_ROOT:-$OUT_ROOT/burgers_wideparam_loss123_retrain_20260611_logs}"
PREFLIGHT_DIR="${PREFLIGHT_DIR:-$ROOT/forensics/burgers_wideparam_loss123_retrain_preflight_20260611}"

LOSS1_RUN="${LOSS1_RUN:-burgers_wideparam_loss1_8000ep_retrain_20260611}"
LOSS2_RUN="${LOSS2_RUN:-burgers_wideparam_loss2_2000ep_retrain_20260611}"
LOSS3_RUN="${LOSS3_RUN:-burgers_wideparam_loss3_1000ep_retrain_20260611}"

LOSS1_EPOCHS="${LOSS1_EPOCHS:-8000}"
LOSS2_EPOCHS="${LOSS2_EPOCHS:-2000}"
LOSS3_EPOCHS="${LOSS3_EPOCHS:-1000}"

LOSS1_BATCH="${LOSS1_BATCH:-1350}"
LOSS2_BATCH="${LOSS2_BATCH:-1350}"
LOSS3_BATCH="${LOSS3_BATCH:-1350}"
LOSS1_OPT_BATCH="${LOSS1_OPT_BATCH:-32}"
LOSS2_OPT_BATCH="${LOSS2_OPT_BATCH:-32}"
LOSS3_OPT_BATCH="${LOSS3_OPT_BATCH:-32}"

LOSS12_BURGERS_SOLVER_REMAT="${LOSS12_BURGERS_SOLVER_REMAT:-chunk}"
LOSS3_BURGERS_SOLVER_REMAT="${LOSS3_BURGERS_SOLVER_REMAT:-chunk}"
BURGERS_SOLVER_REMAT_CHUNK_STEPS="${BURGERS_SOLVER_REMAT_CHUNK_STEPS:-50}"

RUN_LOSS12_PARALLEL="${RUN_LOSS12_PARALLEL:-1}"
RETRAIN_ORDER="${RETRAIN_ORDER:-loss3_then_loss12}"
RUN_LOSS1="${RUN_LOSS1:-1}"
RUN_LOSS2="${RUN_LOSS2:-1}"
RUN_LOSS3="${RUN_LOSS3:-1}"
RUN_PLOTS="${RUN_PLOTS:-1}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "$LOG_ROOT" "$PREFLIGHT_DIR"
cd "$ROOT"

for required in "$PY" "$GEN_ROOT/burgers" "$ROOT/tools/adversarial_training.py" "$ROOT/tools/plot_burgers_wideparam_loss123_retrain_20260611.py"; do
  if [[ ! -e "$required" ]]; then
    echo "[missing] $required" >&2
    exit 1
  fi
done

"$PY" - <<'PYGPU' > "$PREFLIGHT_DIR/gpu_preflight.json"
import json
import sys
import time

import jax
import jax.numpy as jnp
import torch

payload = {
    "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "python": sys.executable,
    "torch_version": torch.__version__,
    "torch_cuda_version": torch.version.cuda,
    "torch_cuda_available": torch.cuda.is_available(),
    "jax_version": jax.__version__,
    "jax_backend": jax.default_backend(),
    "jax_devices": [str(d) for d in jax.devices()],
}
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; refusing Burgers retrain")
payload["torch_device_name"] = torch.cuda.get_device_name(0)
payload["torch_device_capability"] = torch.cuda.get_device_capability(0)
payload["torch_arch_list"] = torch.cuda.get_arch_list()
if jax.default_backend() != "gpu":
    raise SystemExit(f"JAX backend is not gpu: {jax.default_backend()}")
x = torch.randn(128, 128, device="cuda")
y = x @ x
torch.cuda.synchronize()
z = (jnp.ones((128, 128)) @ jnp.ones((128, 128))).block_until_ready()
payload["torch_sanity_sum"] = float(y.sum().detach().cpu())
payload["jax_sanity_sum"] = float(jnp.sum(z))
print(json.dumps(payload, indent=2))
PYGPU
nvidia-smi > "$PREFLIGHT_DIR/nvidia_smi_before.txt"

COMMON=(
  "$PY" "$ROOT/tools/adversarial_training.py"
  --tasks burgers
  --generalization-root "$GEN_ROOT"
  --output-root "$OUT_ROOT"
  --device cuda
  --seed 20260611
  --checkpoint-every-epochs "${CHECKPOINT_EVERY_EPOCHS:-100}"
  --checkpoint-wall-hours "${CHECKPOINT_WALL_HOURS:-0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,7,8,9,10,11,12,12.5,14,16,18,20}"
  --training-data-mode adv-only
  --label-mode solver
  --epsilon-bucket-count "${EPSILON_BUCKET_COUNT:-5}"
  --attack-probe-samples "${ATTACK_PROBE_SAMPLES:-5}"
  --attack-probe-every-n-epochs "${ATTACK_PROBE_EVERY_N_EPOCHS:-1}"
  --attack-probe-save-targets
  --burgers-attack-method fast_replace_l2
  --burgers-require-p2q2
  --burgers-attack-steps "${BURGERS_ATTACK_STEPS:-5}"
  --burgers-epsilon-fraction "${BURGERS_EPSILON_FRACTION:-0.06}"
  --burgers-eps-jitter-low "${BURGERS_EPS_JITTER_LOW:-0.75}"
  --burgers-eps-jitter-high "${BURGERS_EPS_JITTER_HIGH:-1.25}"
  --burgers-alpha-ratio "${BURGERS_ALPHA_RATIO:-1.0}"
  --burgers-alpha-jitter-low "${BURGERS_ALPHA_JITTER_LOW:-0.75}"
  --burgers-alpha-jitter-high "${BURGERS_ALPHA_JITTER_HIGH:-1.25}"
  --burgers-random-start-fraction "${BURGERS_RANDOM_START_FRACTION:-1e-6}"
  --eval-max-samples "${EVAL_MAX_SAMPLES:-0}"
  --max-generalization-eval "${MAX_GENERALIZATION_EVAL:-50}"
)

run_one() {
  local loss="$1"
  local run_name="$2"
  local epochs="$3"
  local batch="$4"
  local opt_batch="$5"
  local remat="$6"
  local run_dir="$OUT_ROOT/$run_name"
  local log="$LOG_ROOT/${run_name}.log"

  if [[ -f "$run_dir/summary.json" ]]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] skip completed $run_name" | tee -a "$LOG_ROOT/driver.log"
    return 0
  fi
  if [[ -d "$run_dir" ]]; then
    echo "[refuse] run directory exists without summary.json: $run_dir" >&2
    exit 1
  fi

  local -a cmd=(
    "${COMMON[@]}"
    --run-name "$run_name"
    --epochs "$epochs"
    --burgers-attack-loss-objective "$loss"
    --burgers-batch-size "$batch"
    --burgers-optimizer-batch-size "$opt_batch"
    --burgers-solver-remat "$remat"
    --burgers-solver-remat-chunk-steps "$BURGERS_SOLVER_REMAT_CHUNK_STEPS"
  )
  if [[ -n "${MAX_WALL_SECONDS:-}" ]]; then
    cmd+=(--max-wall-seconds "$MAX_WALL_SECONDS")
  fi

  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] start $run_name loss=$loss epochs=$epochs batch=$batch opt_batch=$opt_batch remat=$remat" | tee -a "$LOG_ROOT/driver.log"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '[dry-run]'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    return 0
  fi
  "${cmd[@]}" > "$log" 2>&1
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] done $run_name" | tee -a "$LOG_ROOT/driver.log"
}

run_loss12_group() {
  if [[ "$RUN_LOSS12_PARALLEL" == "1" ]]; then
    pids=()
    if [[ "$RUN_LOSS1" == "1" ]]; then
      run_one loss1 "$LOSS1_RUN" "$LOSS1_EPOCHS" "$LOSS1_BATCH" "$LOSS1_OPT_BATCH" "$LOSS12_BURGERS_SOLVER_REMAT" &
      pids+=("$!")
    fi
    if [[ "$RUN_LOSS2" == "1" ]]; then
      run_one loss2 "$LOSS2_RUN" "$LOSS2_EPOCHS" "$LOSS2_BATCH" "$LOSS2_OPT_BATCH" "$LOSS12_BURGERS_SOLVER_REMAT" &
      pids+=("$!")
    fi
    for pid in "${pids[@]}"; do
      wait "$pid"
    done
  else
    [[ "$RUN_LOSS1" == "1" ]] && run_one loss1 "$LOSS1_RUN" "$LOSS1_EPOCHS" "$LOSS1_BATCH" "$LOSS1_OPT_BATCH" "$LOSS12_BURGERS_SOLVER_REMAT"
    [[ "$RUN_LOSS2" == "1" ]] && run_one loss2 "$LOSS2_RUN" "$LOSS2_EPOCHS" "$LOSS2_BATCH" "$LOSS2_OPT_BATCH" "$LOSS12_BURGERS_SOLVER_REMAT"
  fi
}

case "$RETRAIN_ORDER" in
  loss3_then_loss12)
    [[ "$RUN_LOSS3" == "1" ]] && run_one loss3 "$LOSS3_RUN" "$LOSS3_EPOCHS" "$LOSS3_BATCH" "$LOSS3_OPT_BATCH" "$LOSS3_BURGERS_SOLVER_REMAT"
    run_loss12_group
    ;;
  loss12_then_loss3)
    run_loss12_group
    [[ "$RUN_LOSS3" == "1" ]] && run_one loss3 "$LOSS3_RUN" "$LOSS3_EPOCHS" "$LOSS3_BATCH" "$LOSS3_OPT_BATCH" "$LOSS3_BURGERS_SOLVER_REMAT"
    ;;
  *)
    echo "[refuse] unknown RETRAIN_ORDER=$RETRAIN_ORDER; use loss3_then_loss12 or loss12_then_loss3" >&2
    exit 1
    ;;
esac

nvidia-smi > "$PREFLIGHT_DIR/nvidia_smi_after.txt" || true

if [[ "$RUN_PLOTS" == "1" && "$DRY_RUN" != "1" ]]; then
  "$PY" "$ROOT/tools/plot_burgers_wideparam_loss123_retrain_20260611.py" \
    --loss1-run-dir "$OUT_ROOT/$LOSS1_RUN" \
    --loss2-run-dir "$OUT_ROOT/$LOSS2_RUN" \
    --loss3-run-dir "$OUT_ROOT/$LOSS3_RUN" \
    --out-dir "$ROOT/visualizations/burgers_wideparam_loss123_retrain_20260611" \
    --report-md "$ROOT/docs/burgers_wideparam_loss123_retrain_report_20260611.md" \
    > "$LOG_ROOT/plot_wideparam_loss123_retrain.log" 2>&1
fi

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] workflow finished" | tee -a "$LOG_ROOT/driver.log"
