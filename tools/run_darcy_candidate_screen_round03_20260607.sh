#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="$ROOT/adv_robust/bin/python"
LOG_ROOT="$ROOT/adversarial_training_runs/darcy_candidate_screen_round03_20260607_logs"
PREFLIGHT_DIR="$ROOT/forensics/darcy_candidate_screen_round03_gpu_preflight_20260607"
SCREEN_DATA_DIR="$ROOT/2D_Darcy_FNO2d/datasets/grf_darcy_screen_20260607"
SCREEN_MODEL_DIR="$ROOT/2D_Darcy_FNO2d/saved_models/2D"
SCREEN_RUN_NAME="darcy_screen_baseline_m64_w60_e50_20260607"
SCREEN_MODEL="$SCREEN_MODEL_DIR/$SCREEN_RUN_NAME/best.pt"
TRAIN_PATH="$SCREEN_DATA_DIR/train/dim2d_darcy_nx85_N384_solver=jaxcg_solve421_alpha2_tau3_binary3-12_f1_seed45_train.pt"
TEST_PATH="$SCREEN_DATA_DIR/test/dim2d_darcy_nx85_N96_solver=jaxcg_solve421_alpha2_tau3_binary3-12_f1_seed45_test.pt"
GEN_ROOT="$ROOT/generalization_datasets_darcy_candidate_screen_round03_20260607"
FORENSICS_DIR="$ROOT/forensics/darcy_candidate_gradient_alignment_screen_round03_20260607"

mkdir -p "$LOG_ROOT" "$PREFLIGHT_DIR"

if [[ ! -x "$PY" ]]; then
  echo "[missing] $PY" >&2
  exit 1
fi
if [[ ! -f "$SCREEN_MODEL" ]]; then
  echo "[missing] screening baseline model: $SCREEN_MODEL" >&2
  echo "Run tools/run_darcy_candidate_screen_20260607.sh first." >&2
  exit 1
fi
if [[ ! -f "$TRAIN_PATH" || ! -f "$TEST_PATH" ]]; then
  echo "[missing] screening train/test data under $SCREEN_DATA_DIR" >&2
  echo "Run tools/run_darcy_candidate_screen_20260607.sh first." >&2
  exit 1
fi

"$PY" - <<PYGPU > "$PREFLIGHT_DIR/gpu_preflight.json"
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
    raise SystemExit("CUDA is unavailable; refusing Darcy Flow round03 screen")
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

if [[ ! -f "$GEN_ROOT/candidate_manifest.csv" ]]; then
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] generate Darcy round03 candidate generalization datasets" | tee -a "$LOG_ROOT/driver.log"
  "$PY" "$ROOT/tools/generate_darcy_generalization_candidates_round03.py" \
    --output-root "$GEN_ROOT" \
    --samples-per-candidate "${DARCY_R03_CANDIDATE_SAMPLES:-48}" \
    --batch-size "${DARCY_R03_CANDIDATE_BATCH:-8}" \
    > "$LOG_ROOT/candidate_generation.log" 2>&1
else
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] reuse existing Darcy round03 candidate manifest" | tee -a "$LOG_ROOT/driver.log"
fi

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] run Darcy round03 candidate gradient-alignment screen" | tee -a "$LOG_ROOT/driver.log"
"$PY" "$ROOT/tools/probe_darcy_gradient_alignment_screen.py" \
  --checkpoint "$SCREEN_MODEL" \
  --train-path "$TRAIN_PATH" \
  --test-path "$TEST_PATH" \
  --generalization-root "$GEN_ROOT" \
  --out-dir "$FORENSICS_DIR" \
  --device cuda \
  --steps "${DARCY_R03_SCREEN_STEPS:-50}" \
  --batch-size "${DARCY_R03_SCREEN_ATTACK_BATCH:-96}" \
  --optimizer-batch-size "${DARCY_R03_SCREEN_OPT_BATCH:-24}" \
  --eval-grad-batch-size "${DARCY_R03_SCREEN_EVAL_BATCH:-16}" \
  > "$LOG_ROOT/gradient_alignment_screen.log" 2>&1

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Darcy round03 candidate screen done" | tee -a "$LOG_ROOT/driver.log"
