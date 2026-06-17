#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${MODE:-smoke}" # smoke or full
TAG="${TAG:-$(date -u +%Y%m%d_%H%M%S_UTC)}"
BUNDLE="${BUNDLE:-$ROOT/outputs/darcy_sir20_timematched_full_${TAG}}"
PYTHON="${PYTHON:-$ROOT/adv_robust/bin/python}"

CALIBRATION_EPOCHS="${CALIBRATION_EPOCHS:-20}"
CALIBRATION_WARMUP_EPOCHS="${CALIBRATION_WARMUP_EPOCHS:-2}"
LOSS3_REFERENCE_EPOCHS="${LOSS3_REFERENCE_EPOCHS:-3000}"
DARCY_TRAIN_MAX="${DARCY_TRAIN_MAX:-64}"
DARCY_BATCH="${DARCY_BATCH:-64}"
OPT_BATCH="${OPT_BATCH:-32}"
CHECKPOINT_EVERY_EPOCHS="${CHECKPOINT_EVERY_EPOCHS:-100}"
EPOCH_MULTIPLIER="${EPOCH_MULTIPLIER:-2.0}"
DARCY_EPS_JITTER_LOW="${DARCY_EPS_JITTER_LOW:-0.25}"
DARCY_EPS_JITTER_HIGH="${DARCY_EPS_JITTER_HIGH:-1.75}"
DARCY_ATTACK_PROBE_SAMPLES="${DARCY_ATTACK_PROBE_SAMPLES:-8}"
DARCY_ATTACK_PROBE_EVERY_N_EPOCHS="${DARCY_ATTACK_PROBE_EVERY_N_EPOCHS:-1}"
CHECKPOINT_ORIGINAL_AND_FINAL="${CHECKPOINT_ORIGINAL_AND_FINAL:-1}"
UPLOAD_TO_R2="${UPLOAD_TO_R2:-1}"
AUTO_GIT_PUSH="${AUTO_GIT_PUSH:-1}"
R2_PREFIX="${R2_PREFIX:-machine-sync/NeuralOperatorRobustness2-selected}"
GIT_BRANCH="${GIT_BRANCH:-vast-ai-darcy-flow}"

export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.35}"

mkdir -p "$BUNDLE"/{figures,data,checkpoints_manifest,reports,logs}
DRIVER_LOG="$BUNDLE/logs/driver_${MODE}.log"

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$DRIVER_LOG"
}

run_logged() {
  local name="$1"
  shift
  log "start ${name}"
  {
    echo "[command] $*"
    "$@"
  } 2>&1 | tee "$BUNDLE/logs/${name}.log"
  log "done ${name}"
}

upload_path() {
  local path="$1"
  [[ "$UPLOAD_TO_R2" == "1" ]] || return 0
  if [[ ! -e "$path" ]]; then
    log "skip missing upload path: $path"
    return 0
  fi
  if [[ -z "${R2_ACCESS_KEY_ID:-}" || -z "${R2_SECRET_ACCESS_KEY:-}" ]]; then
    if [[ -z "${R2_RCLONE_CONFIG_FILE:-}" && -f /tmp/r2-darcy-long-20260612_full50_timematched_1000b.conf ]]; then
      export R2_RCLONE_CONFIG_FILE=/tmp/r2-darcy-long-20260612_full50_timematched_1000b.conf
    fi
  fi
  log "upload to R2: $path"
  R2_UPLOAD_LOG_ROOT="$BUNDLE/logs/r2_upload" \
  R2_PREFIX="$R2_PREFIX" \
    "$ROOT/tools/upload_path_to_r2_20260525.sh" "$path"
}

git_push_code() {
  [[ "$AUTO_GIT_PUSH" == "1" ]] || return 0
  log "git commit/push source and Markdown to ${GIT_BRANCH}"
  git config user.name "${GIT_COMMITTER_NAME:-vast-ai-runner}"
  git config user.email "${GIT_COMMITTER_EMAIL:-vast-ai-runner@users.noreply.github.com}"
  git add \
    tools/adversarial_training.py \
    tools/evaluate_generalization_models.py \
    tools/darcy_sir20_common.py \
    tools/darcy_sir20_calibrate.py \
    tools/darcy_sir20_train_launcher.py \
    tools/darcy_sir20_evaluate.py \
    tools/darcy_sir20_robustness.py \
    tools/darcy_sir20_visualize.py \
    tools/run_darcy_sir20_timematched_full_20260614.sh \
    docs/darcy_sir20_serial_double_budget_20260617.md \
    docs/darcy_sir20_timematched_full_20260614.md \
    EXPERIMENT_LEDGER.md
  git commit -m "Update Darcy SIR20 serial double-budget pipeline" || true
  git push origin "HEAD:${GIT_BRANCH}"
}

log "Darcy/SIR20 ${MODE} pipeline start bundle=$BUNDLE"
log "Serial-training epsilon jitter factor per batch: uniform[$DARCY_EPS_JITTER_LOW,$DARCY_EPS_JITTER_HIGH]"
log "Epoch multiplier: $EPOCH_MULTIPLIER; checkpoint original/final=$CHECKPOINT_ORIGINAL_AND_FINAL"
log "Darcy training samples per epoch: $DARCY_TRAIN_MAX; batch=$DARCY_BATCH; opt_batch=$OPT_BATCH"
log "Darcy attack probes: samples=$DARCY_ATTACK_PROBE_SAMPLES; every_n_epochs=$DARCY_ATTACK_PROBE_EVERY_N_EPOCHS"

PREFLIGHT_PY="$BUNDLE/data/preflight_check.py"
cat > "$PREFLIGHT_PY" <<'PY'
import json, os, sys, time
import jax, jax.numpy as jnp
import torch
payload = {
    "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "python": sys.executable,
    "torch_version": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "torch_cuda_available": torch.cuda.is_available(),
    "jax_version": jax.__version__,
    "jax_backend": jax.default_backend(),
    "jax_devices": [str(d) for d in jax.devices()],
    "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
}
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable")
payload["device_name"] = torch.cuda.get_device_name(0)
payload["device_capability"] = torch.cuda.get_device_capability(0)
payload["torch_arch_list"] = torch.cuda.get_arch_list()
if "sm_70" not in torch.cuda.get_arch_list():
    raise SystemExit(f"sm_70 missing from torch arch list: {torch.cuda.get_arch_list()}")
if jax.default_backend() != "gpu":
    raise SystemExit(f"JAX backend is not gpu: {jax.default_backend()}")
x = torch.randn(64, 64, device="cuda")
payload["torch_probe_sum"] = float((x @ x).sum().detach().cpu())
payload["jax_probe_sum"] = float(jnp.sum(jnp.ones((64, 64)) @ jnp.ones((64, 64))).block_until_ready())
print(json.dumps(payload, indent=2))
PY
run_logged preflight "$PYTHON" "$PREFLIGHT_PY"
nvidia-smi > "$BUNDLE/logs/nvidia_smi.txt" || true

if [[ "$MODE" == "smoke" ]]; then
  run_logged smoke_train "$PYTHON" tools/darcy_sir20_train_launcher.py \
    --bundle "$BUNDLE" \
    --mode smoke \
    --smoke-epochs "${SMOKE_EPOCHS:-1}" \
    --smoke-eval-max-samples "${SMOKE_EVAL_MAX_SAMPLES:-1}" \
    --batch-size "$DARCY_BATCH" \
    --optimizer-batch-size "$OPT_BATCH" \
    --darcy-train-max "$DARCY_TRAIN_MAX" \
    --eps-jitter-low "$DARCY_EPS_JITTER_LOW" \
    --eps-jitter-high "$DARCY_EPS_JITTER_HIGH" \
    --attack-probe-samples "$DARCY_ATTACK_PROBE_SAMPLES" \
    --attack-probe-every-n-epochs "$DARCY_ATTACK_PROBE_EVERY_N_EPOCHS"
  MANIFEST="$BUNDLE/checkpoints_manifest/training_checkpoints_smoke.json"
  run_logged smoke_eval "$PYTHON" tools/darcy_sir20_evaluate.py \
    --bundle "$BUNDLE" \
    --checkpoint-manifest "$MANIFEST" \
    --eval-max-samples "${SMOKE_EVAL_MAX_SAMPLES:-1}" \
    --batch-size 256
  run_logged smoke_robustness "$PYTHON" tools/darcy_sir20_robustness.py \
    --bundle "$BUNDLE" \
    --checkpoint-manifest "$MANIFEST" \
    --samples-per-dataset "${SMOKE_ATTACK_SAMPLES_PER_DATASET:-2}" \
    --attack-steps "${SMOKE_ATTACK_STEPS:-1}" \
    --svd-max-samples "${SMOKE_SVD_MAX_SAMPLES:-3}" \
    --block-row-chunk "${SMOKE_BLOCK_ROW_CHUNK:-128}"
  run_logged smoke_visualize "$PYTHON" tools/darcy_sir20_visualize.py \
    --bundle "$BUNDLE" \
    --checkpoint-manifest "$MANIFEST"
else
  run_logged calibration "$PYTHON" tools/darcy_sir20_calibrate.py \
    --bundle "$BUNDLE" \
    --tag "$TAG" \
    --epochs "$CALIBRATION_EPOCHS" \
    --warmup-epochs "$CALIBRATION_WARMUP_EPOCHS" \
    --loss3-reference-epochs "$LOSS3_REFERENCE_EPOCHS" \
    --batch-size "$DARCY_BATCH" \
    --optimizer-batch-size "$OPT_BATCH" \
    --darcy-train-max "$DARCY_TRAIN_MAX" \
    --reuse
  PLAN="$BUNDLE/data/timing_calibration.json"
  run_logged full_train "$PYTHON" tools/darcy_sir20_train_launcher.py \
    --bundle "$BUNDLE" \
    --mode full \
    --plan-json "$PLAN" \
    --loss3-reference-epochs "$LOSS3_REFERENCE_EPOCHS" \
    --batch-size "$DARCY_BATCH" \
    --optimizer-batch-size "$OPT_BATCH" \
    --darcy-train-max "$DARCY_TRAIN_MAX" \
    --checkpoint-every-epochs "$CHECKPOINT_EVERY_EPOCHS" \
    --epoch-multiplier "$EPOCH_MULTIPLIER" \
    --eps-jitter-low "$DARCY_EPS_JITTER_LOW" \
    --eps-jitter-high "$DARCY_EPS_JITTER_HIGH" \
    --attack-probe-samples "$DARCY_ATTACK_PROBE_SAMPLES" \
    --attack-probe-every-n-epochs "$DARCY_ATTACK_PROBE_EVERY_N_EPOCHS" \
    $(if [[ "$CHECKPOINT_ORIGINAL_AND_FINAL" == "1" ]]; then printf '%s' '--checkpoint-at-original-and-final'; else printf '%s' '--no-checkpoint-at-original-and-final'; fi) \
    --reuse
  FINAL_MANIFEST="$BUNDLE/checkpoints_manifest/training_checkpoints_full.json"
  ANALYSIS_MANIFEST="$BUNDLE/checkpoints_manifest/training_checkpoints_full_all_saved.json"
  if [[ ! -f "$ANALYSIS_MANIFEST" ]]; then
    ANALYSIS_MANIFEST="$FINAL_MANIFEST"
  fi
  run_logged final_eval "$PYTHON" tools/darcy_sir20_evaluate.py \
    --bundle "$BUNDLE" \
    --checkpoint-manifest "$ANALYSIS_MANIFEST" \
    --eval-max-samples 0 \
    --batch-size 256
  run_logged advanced_serial_attack_jacobian_svd "$PYTHON" tools/darcy_sir20_robustness.py \
    --bundle "$BUNDLE" \
    --checkpoint-manifest "$ANALYSIS_MANIFEST" \
    --samples-per-dataset 50 \
    --attack-steps "${ROBUSTNESS_ATTACK_STEPS:-20}" \
    --svd-max-samples 25 \
    --block-row-chunk "${BLOCK_ROW_CHUNK:-128}"
  run_logged visualize "$PYTHON" tools/darcy_sir20_visualize.py \
    --bundle "$BUNDLE" \
    --checkpoint-manifest "$FINAL_MANIFEST" \
    --final-eval-csv "$BUNDLE/data/final_eval_metrics.csv"
fi

upload_path "$BUNDLE"
git_push_code

log "Darcy/SIR20 ${MODE} pipeline complete bundle=$BUNDLE"
