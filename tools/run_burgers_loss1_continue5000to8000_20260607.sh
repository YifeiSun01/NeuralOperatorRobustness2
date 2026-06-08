#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="$ROOT/adv_robust/bin/python"
GEN_ROOT="$ROOT/generalization_datasets_burgers_loss3_selective_search/round_03"
OUT_ROOT="$ROOT/adversarial_training_runs"
LOG_ROOT="$OUT_ROOT/burgers_loss3_selective_round03_loss1_continue5000to8000_20260607_logs"
PREFLIGHT_DIR="$ROOT/forensics/burgers_loss1_5000to8000_gpu_preflight_20260607"

START_RUN="$OUT_ROOT/burgers_loss3_selective_round03_loss1_continue3000to5000_20260606"
START_CKPT="$START_RUN/burgers/checkpoints/burgers_epoch5000_step015000.pt"
RUN_NAME="burgers_loss3_selective_round03_loss1_continue5000to8000_20260607"
RUN_DIR="$OUT_ROOT/$RUN_NAME"
FINAL_CKPT="$RUN_DIR/burgers/checkpoints/burgers_epoch8000_step024000.pt"
PLOT_LOG="$LOG_ROOT/plot_loss1_8000.log"

mkdir -p "$LOG_ROOT" "$PREFLIGHT_DIR"

for required in "$PY" "$GEN_ROOT/burgers" "$START_CKPT"; do
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
}
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; refusing Burgers Loss1 continuation")
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

COMMON=(
  "$PY" "$ROOT/tools/adversarial_training.py"
  --tasks burgers
  --generalization-root "$GEN_ROOT"
  --output-root "$OUT_ROOT"
  --device cuda
  --seed 20260601
  --checkpoint-every-epochs 100
  --checkpoint-wall-hours 0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0
  --training-data-mode adv-only
  --label-mode solver
  --epsilon-bucket-count 5
  --attack-probe-samples 5
  --attack-probe-every-n-epochs 1
  --attack-probe-save-targets
  --burgers-attack-method fast_replace_l2
  --burgers-require-p2q2
  --burgers-attack-steps 5
  --burgers-batch-size 480
  --burgers-optimizer-batch-size 32
  --burgers-epsilon-fraction 0.06
  --burgers-eps-jitter-low 0.75
  --burgers-eps-jitter-high 1.25
  --burgers-alpha-ratio 1.0
  --burgers-alpha-jitter-low 0.75
  --burgers-alpha-jitter-high 1.25
  --burgers-random-start-fraction 1e-6
  --eval-max-samples 0
  --max-generalization-eval 50
)

if [[ -f "$RUN_DIR/summary.json" ]]; then
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] skip completed $RUN_NAME" | tee -a "$LOG_ROOT/driver.log"
else
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] start $RUN_NAME from $START_CKPT" | tee -a "$LOG_ROOT/driver.log"
  "${COMMON[@]}" \
    --run-name "$RUN_NAME" \
    --epochs 3000 \
    --resume-epoch-offset 5000 \
    --resume-global-step-offset 15000 \
    --burgers-initial-checkpoint "$START_CKPT" \
    --burgers-attack-loss-objective loss1 \
    > "$LOG_ROOT/$RUN_NAME.log" 2>&1
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] done $RUN_NAME" | tee -a "$LOG_ROOT/driver.log"
fi

if [[ ! -f "$FINAL_CKPT" ]]; then
  echo "[missing] expected final checkpoint after training: $FINAL_CKPT" >&2
  exit 1
fi

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] refresh loss1 epoch8000 plots" | tee -a "$LOG_ROOT/driver.log"
{
  "$PY" "$ROOT/tools/plot_burgers_round03_stitched_single_run_visualizations.py" \
    --base-run-dir "$ROOT/adversarial_training_runs/burgers_loss3_selective_round03_loss1_1000ep_long_20260605" \
    --extension-run-dir "$ROOT/adversarial_training_runs/burgers_loss3_selective_round03_loss1_continue1000to3000_20260605" \
    --extension-run-dir "$ROOT/adversarial_training_runs/burgers_loss3_selective_round03_loss1_continue3000to5000_20260606" \
    --extension-run-dir "$RUN_DIR" \
    --out-dir "$ROOT/visualizations/burgers_loss3_selective_round03_loss1_8000ep_long_20260605_plots" \
    --suffix round03_long
  "$PY" "$ROOT/tools/plot_burgers_round03_loss1_8000_dense_comparison.py"
} > "$PLOT_LOG" 2>&1
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] loss1 epoch8000 workflow done" | tee -a "$LOG_ROOT/driver.log"
