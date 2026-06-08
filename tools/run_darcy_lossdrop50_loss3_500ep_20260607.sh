#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="$ROOT/adv_robust/bin/python"
GEN_ROOT="$ROOT/generalization_datasets_darcy_lossdrop50_selected_20260607"
OUT_ROOT="$ROOT/adversarial_training_runs"
RUN_NAME="darcy_lossdrop50_loss3_500ep_fromscreen_20260607"
RUN_DIR="$OUT_ROOT/$RUN_NAME"
LOG_ROOT="$OUT_ROOT/${RUN_NAME}_logs"
PREFLIGHT_DIR="$ROOT/forensics/darcy_lossdrop50_loss3_500ep_gpu_preflight_20260607"
MODEL_CKPT="$ROOT/2D_Darcy_FNO2d/saved_models/2D/darcy_screen_baseline_m64_w60_e50_20260607/best.pt"
TRAIN_PATH="$ROOT/2D_Darcy_FNO2d/datasets/grf_darcy_screen_20260607/train/dim2d_darcy_nx85_N384_solver=jaxcg_solve421_alpha2_tau3_binary3-12_f1_seed45_train.pt"
TEST_PATH="$ROOT/2D_Darcy_FNO2d/datasets/grf_darcy_screen_20260607/test/dim2d_darcy_nx85_N96_solver=jaxcg_solve421_alpha2_tau3_binary3-12_f1_seed45_test.pt"

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
    "xla_preallocate": None,
    "xla_mem_fraction": None,
}
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; refusing Darcy Loss3 500ep training")
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
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] start $RUN_NAME" | tee -a "$LOG_ROOT/driver.log"
  "$PY" "$ROOT/tools/adversarial_training.py"     --tasks darcy     --generalization-root "$GEN_ROOT"     --output-root "$OUT_ROOT"     --run-name "$RUN_NAME"     --device cuda     --seed 20260607     --epochs 500     --checkpoint-every-epochs 50     --checkpoint-wall-hours 0.5,1.0,2.0,3.0,4.0,6.0,8.0,12.0,16.0,20.0,24.0     --training-data-mode adv-only     --label-mode solver     --epsilon-bucket-count 5     --attack-probe-samples 5     --attack-probe-every-n-epochs 1     --attack-probe-save-targets     --darcy-model-checkpoint "$MODEL_CKPT"     --darcy-train-path "$TRAIN_PATH"     --darcy-test-path "$TEST_PATH"     --darcy-attack-method binary_steepest_replace     --darcy-attack-steps 1     --darcy-batch-size "${DARCY_LOSS3_BATCH:-96}"     --darcy-optimizer-batch-size "${DARCY_LOSS3_OPT_BATCH:-24}"     --darcy-epsilon-fraction "${DARCY_LOSS3_EPSILON_FRACTION:-0.025}"     --darcy-eps-jitter-low "${DARCY_LOSS3_EPS_JITTER_LOW:-1.0}"     --darcy-eps-jitter-high "${DARCY_LOSS3_EPS_JITTER_HIGH:-1.0}"     --darcy-alpha-ratio 1.0     --eval-max-samples 0     --max-generalization-eval 50     > "$LOG_ROOT/$RUN_NAME.log" 2>&1
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] done $RUN_NAME" | tee -a "$LOG_ROOT/driver.log"
fi

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Darcy Loss3 500ep workflow done" | tee -a "$LOG_ROOT/driver.log"
