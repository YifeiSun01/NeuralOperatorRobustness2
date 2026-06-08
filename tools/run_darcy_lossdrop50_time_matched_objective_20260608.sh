#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OBJECTIVE="${1:-loss1}"
MODE="${2:-full}"
case "$OBJECTIVE" in
  loss1|loss2|loss3|physics|loss4|loss4_physics) ;;
  *) echo "[error] objective must be loss1/loss2/loss3/physics/loss4/loss4_physics, got $OBJECTIVE" >&2; exit 2 ;;
esac
case "$MODE" in
  timing|full) ;;
  *) echo "[error] mode must be timing or full, got $MODE" >&2; exit 2 ;;
esac

PY="$ROOT/adv_robust/bin/python"
GEN_ROOT="$ROOT/generalization_datasets_darcy_lossdrop50_selected_20260607"
OUT_ROOT="$ROOT/adversarial_training_runs"
LOSS3_SUMMARY="$OUT_ROOT/darcy_lossdrop50_loss3_500ep_fromscreen_20260607/summary.json"
TASK_LOSS3_SUMMARY="$OUT_ROOT/darcy_lossdrop50_loss3_500ep_fromscreen_20260607/darcy/summary.json"
MODEL_CKPT="$ROOT/2D_Darcy_FNO2d/saved_models/2D/darcy_screen_baseline_m64_w60_e50_20260607/best.pt"
TRAIN_PATH="$ROOT/2D_Darcy_FNO2d/datasets/grf_darcy_screen_20260607/train/dim2d_darcy_nx85_N384_solver=jaxcg_solve421_alpha2_tau3_binary3-12_f1_seed45_train.pt"
TEST_PATH="$ROOT/2D_Darcy_FNO2d/datasets/grf_darcy_screen_20260607/test/dim2d_darcy_nx85_N96_solver=jaxcg_solve421_alpha2_tau3_binary3-12_f1_seed45_test.pt"

TARGET_SECONDS="${DARCY_TIME_MATCH_SECONDS:-}"
if [[ -z "$TARGET_SECONDS" ]]; then
  TARGET_SECONDS="$($PY - "$LOSS3_SUMMARY" "$TASK_LOSS3_SUMMARY" <<'PYSEC'
import json, sys
run_summary = json.load(open(sys.argv[1])) if __import__('pathlib').Path(sys.argv[1]).exists() else {}
task_summary = json.load(open(sys.argv[2])) if __import__('pathlib').Path(sys.argv[2]).exists() else {}
value = task_summary.get('elapsed_seconds') or run_summary.get('total_wall_seconds') or 4975.1246
print(float(value))
PYSEC
)"
fi

if [[ "$MODE" == "timing" ]]; then
  RUN_NAME="darcy_lossdrop50_${OBJECTIVE}_timing${DARCY_TIMING_EPOCHS:-5}ep_20260608"
  EPOCHS="${DARCY_TIMING_EPOCHS:-5}"
  MAX_WALL_ARGS=()
else
  RUN_NAME="darcy_lossdrop50_${OBJECTIVE}_time_matched_loss3wall_20260608"
  EPOCHS="${DARCY_TIME_MATCH_MAX_EPOCHS:-100000}"
  MAX_WALL_ARGS=(--max-wall-seconds "$TARGET_SECONDS")
fi
RUN_DIR="$OUT_ROOT/$RUN_NAME"
LOG_ROOT="$OUT_ROOT/${RUN_NAME}_logs"
PREFLIGHT_DIR="$ROOT/forensics/${RUN_NAME}_gpu_preflight"
DOC_PATH="$ROOT/docs/${RUN_NAME}.md"

export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.35}"

mkdir -p "$LOG_ROOT" "$PREFLIGHT_DIR"
for required in "$PY" "$GEN_ROOT/candidate_manifest.csv" "$GEN_ROOT/darcy" "$MODEL_CKPT" "$TRAIN_PATH" "$TEST_PATH"; do
  if [[ ! -e "$required" ]]; then
    echo "[missing] $required" >&2
    exit 1
  fi
done
if [[ -d "$RUN_DIR" && ! -f "$RUN_DIR/summary.json" ]]; then
  echo "[refuse] run directory exists without summary.json: $RUN_DIR" >&2
  exit 1
fi

"$PY" - <<'PYGPU' > "$PREFLIGHT_DIR/gpu_preflight.json"
import json
import os
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
    "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
    "XLA_PYTHON_CLIENT_PREALLOCATE": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
    "XLA_PYTHON_CLIENT_MEM_FRACTION": os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION"),
}
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; refusing Darcy time-matched training")
payload["torch_device_name"] = torch.cuda.get_device_name(0)
payload["torch_device_capability"] = torch.cuda.get_device_capability(0)
payload["torch_arch_list"] = torch.cuda.get_arch_list()
if "sm_70" not in torch.cuda.get_arch_list():
    raise SystemExit(f"PyTorch arch list does not include sm_70: {torch.cuda.get_arch_list()}")
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
nvidia-smi > "$PREFLIGHT_DIR/nvidia_smi.txt"

if [[ -f "$RUN_DIR/summary.json" ]]; then
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] skip completed $RUN_NAME" | tee -a "$LOG_ROOT/driver.log"
else
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] start $RUN_NAME objective=$OBJECTIVE mode=$MODE target_seconds=$TARGET_SECONDS epochs=$EPOCHS" | tee -a "$LOG_ROOT/driver.log"
  "$PY" "$ROOT/tools/adversarial_training.py" \
    --tasks darcy \
    --generalization-root "$GEN_ROOT" \
    --output-root "$OUT_ROOT" \
    --run-name "$RUN_NAME" \
    --device cuda \
    --seed "${DARCY_TIME_MATCH_SEED:-20260608}" \
    --epochs "$EPOCHS" \
    "${MAX_WALL_ARGS[@]}" \
    --checkpoint-every-epochs "${DARCY_CHECKPOINT_EVERY_EPOCHS:-50}" \
    --checkpoint-wall-hours "${DARCY_CHECKPOINT_WALL_HOURS:-0.5,1.0,2.0,3.0,4.0,6.0,8.0,12.0,16.0,20.0,24.0}" \
    --training-data-mode adv-only \
    --label-mode solver \
    --epsilon-bucket-count 5 \
    --attack-probe-samples "${DARCY_ATTACK_PROBE_SAMPLES:-5}" \
    --attack-probe-every-n-epochs "${DARCY_ATTACK_PROBE_EVERY_N_EPOCHS:-1}" \
    --attack-probe-save-targets \
    --darcy-model-checkpoint "$MODEL_CKPT" \
    --darcy-train-path "$TRAIN_PATH" \
    --darcy-test-path "$TEST_PATH" \
    --darcy-attack-method binary_steepest_replace \
    --darcy-attack-loss-objective "$OBJECTIVE" \
    --darcy-attack-steps "${DARCY_ATTACK_STEPS:-1}" \
    --darcy-batch-size "${DARCY_BATCH:-96}" \
    --darcy-optimizer-batch-size "${DARCY_OPT_BATCH:-24}" \
    --darcy-epsilon-fraction "${DARCY_EPSILON_FRACTION:-0.025}" \
    --darcy-eps-jitter-low "${DARCY_EPS_JITTER_LOW:-1.0}" \
    --darcy-eps-jitter-high "${DARCY_EPS_JITTER_HIGH:-1.0}" \
    --darcy-alpha-ratio 1.0 \
    --darcy-physics-metric "${DARCY_PHYSICS_METRIC:-rel_l2}" \
    --darcy-physics-bc-weight "${DARCY_PHYSICS_BC_WEIGHT:-1.0}" \
    --darcy-physics-forcing-value "${DARCY_PHYSICS_FORCING_VALUE:-1.0}" \
    --darcy-loss1-random-start-fraction "${DARCY_LOSS1_RANDOM_START_FRACTION:-1.0}" \
    --eval-max-samples 0 \
    --max-generalization-eval 50 \
    > "$LOG_ROOT/$RUN_NAME.log" 2>&1
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] done $RUN_NAME" | tee -a "$LOG_ROOT/driver.log"
fi

SUMMARY_OUT="$RUN_DIR/darcy_time_matched_summary"
"$PY" "$ROOT/tools/summarize_darcy_time_matched_runs_20260608.py" \
  --run-dir "$RUN_DIR" \
  --output-dir "$SUMMARY_OUT" \
  --output-md "$DOC_PATH" \
  > "$LOG_ROOT/postprocess_summary.json"

FIG_PATH="$SUMMARY_OUT/eval_train_loss_progress.png"
"$PY" "$ROOT/tools/plot_eval_train_loss_progress.py" \
  --run-dir "$RUN_DIR" \
  --task darcy \
  --out-path "$FIG_PATH" \
  --title "Darcy $OBJECTIVE time-matched self-training" \
  >> "$LOG_ROOT/postprocess_summary.json" 2>&1 || true

if [[ "${AUTO_UPLOAD_R2:-0}" == "1" ]]; then
  export R2_UPLOAD_LOG_ROOT="${R2_UPLOAD_LOG_ROOT:-$LOG_ROOT/r2_upload_logs}"
  export R2_PREFIX="${R2_PREFIX:-machine-sync/NeuralOperatorRobustness2-selected}/darcy_time_matched_${OBJECTIVE}_${MODE}_20260608"
  "$ROOT/tools/upload_path_to_r2_20260525.sh" "$RUN_DIR"
  "$ROOT/tools/upload_path_to_r2_20260525.sh" "$DOC_PATH"
fi

if [[ "${AUTO_GIT_PUSH:-0}" == "1" ]]; then
  git add \
    tools/adversarial_training.py \
    tools/run_darcy_lossdrop50_time_matched_objective_20260608.sh \
    tools/summarize_darcy_time_matched_runs_20260608.py \
    "$DOC_PATH" \
    EXPERIMENT_LEDGER.md
  git commit -m "Add Darcy time-matched self-training objective pipeline" || true
  git push origin "${GIT_BRANCH:-vast-ai}"
fi

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] workflow complete run=$RUN_DIR doc=$DOC_PATH" | tee -a "$LOG_ROOT/driver.log"
