#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="$ROOT/adv_robust/bin/python"
GEN_ROOT="$ROOT/generalization_datasets_burgers_loss3_selective_search/round_03"
OUT_ROOT="$ROOT/adversarial_training_runs"
LOG_ROOT="$OUT_ROOT/burgers_loss3_selective_round03_loss123_continuation_20260605_logs"
PREFLIGHT_DIR="$ROOT/forensics/burgers_loss3_selective_round03_loss123_continuation_gpu_preflight_20260605"
SVD_FINAL_DIR="$ROOT/forensics/burgers_loss3_selective_round03_long_final_jacobian_svd_rep20_top100_20260605"
SVD_FINAL_SUMMARY="$SVD_FINAL_DIR/round03_long_final_jacobian_svd_summary.csv"
mkdir -p "$LOG_ROOT" "$PREFLIGHT_DIR"

if [[ "${ALLOW_BEFORE_SVD_COMPLETE:-0}" != "1" ]]; then
  if [[ ! -f "$SVD_FINAL_SUMMARY" ]]; then
    echo "[guard] final Jacobian/SVD summary is not complete yet: $SVD_FINAL_SUMMARY" >&2
    echo "[guard] per user request, do not start continuation training before the active SVD job finishes." >&2
    echo "[guard] set ALLOW_BEFORE_SVD_COMPLETE=1 only if you intentionally want to override this guard." >&2
    exit 2
  fi
fi

LOSS1_BASE_RUN="$OUT_ROOT/burgers_loss3_selective_round03_loss1_1000ep_long_20260605"
LOSS2_BASE_RUN="$OUT_ROOT/burgers_loss3_selective_round03_loss2_500ep_long_20260605"
LOSS3_BASE_RUN="$OUT_ROOT/burgers_loss3_selective_round03_loss3_500ep_long_20260605"
LOSS1_CKPT="$LOSS1_BASE_RUN/burgers/checkpoints/burgers_epoch1000_step003000.pt"
LOSS2_CKPT="$LOSS2_BASE_RUN/burgers/checkpoints/burgers_epoch500_step001500.pt"
LOSS3_CKPT="$LOSS3_BASE_RUN/burgers/checkpoints/burgers_epoch500_step001500.pt"
LOSS1_CONT_RUN="burgers_loss3_selective_round03_loss1_continue1000to3000_20260605"
LOSS2_CONT_RUN="burgers_loss3_selective_round03_loss2_continue500to1000_20260605"
LOSS3_CONT_RUN="burgers_loss3_selective_round03_loss3_continue500to1000_20260605"

for required in "$PY" "$GEN_ROOT/burgers" "$LOSS1_CKPT" "$LOSS2_CKPT" "$LOSS3_CKPT"; do
  if [[ ! -e "$required" ]]; then
    echo "[missing] $required" >&2
    exit 1
  fi
done

for run_name in "$LOSS1_CONT_RUN" "$LOSS2_CONT_RUN" "$LOSS3_CONT_RUN"; do
  run_dir="$OUT_ROOT/$run_name"
  if [[ -f "$run_dir/summary.json" ]]; then
    echo "[skip] completed continuation already exists: $run_dir"
    continue
  fi
  if [[ -d "$run_dir" ]]; then
    echo "[refuse] run directory exists without summary.json: $run_dir" >&2
    echo "[refuse] inspect or move it before starting continuation training." >&2
    exit 1
  fi
done

"$PY" - <<'PYGPU' > "$PREFLIGHT_DIR/gpu_preflight.json"
import json
import sys
import time

import torch

payload = {
    "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "python": sys.executable,
    "torch_version": torch.__version__,
    "torch_cuda_version": torch.version.cuda,
    "torch_cuda_available": torch.cuda.is_available(),
}
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; aborting per GPU-only rule")
payload.update(
    {
        "torch_device_name": torch.cuda.get_device_name(0),
        "torch_device_capability": torch.cuda.get_device_capability(0),
        "torch_arch_list": torch.cuda.get_arch_list(),
    }
)
if "sm_70" not in torch.cuda.get_arch_list():
    raise SystemExit(f"PyTorch arch list does not include sm_70: {torch.cuda.get_arch_list()}")
try:
    import jax

    payload.update(
        {
            "jax_version": jax.__version__,
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(d) for d in jax.devices()],
        }
    )
    if jax.default_backend() != "gpu":
        raise SystemExit(f"JAX backend is not gpu: {jax.default_backend()}")
except Exception as exc:
    payload["jax_error"] = repr(exc)
    raise
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
  --checkpoint-wall-hours 0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0
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

run_continue() {
  local loss="$1"
  local run_name="$2"
  local initial_checkpoint="$3"
  local local_epochs="$4"
  local epoch_offset="$5"
  local global_step_offset="$6"
  local log="$LOG_ROOT/${run_name}.log"
  local run_dir="$OUT_ROOT/$run_name"

  if [[ -f "$run_dir/summary.json" ]]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] skip completed ${run_name}" | tee -a "$LOG_ROOT/driver.log"
    return 0
  fi

  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] start ${run_name} from ${initial_checkpoint}" | tee -a "$LOG_ROOT/driver.log"
  "${COMMON[@]}" \
    --run-name "$run_name" \
    --epochs "$local_epochs" \
    --resume-epoch-offset "$epoch_offset" \
    --resume-global-step-offset "$global_step_offset" \
    --burgers-initial-checkpoint "$initial_checkpoint" \
    --burgers-attack-loss-objective "$loss" \
    > "$log" 2>&1
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] done ${run_name}" | tee -a "$LOG_ROOT/driver.log"
}

# Continuation targets:
# - loss1: epoch1000 -> epoch3000, so train 2000 more local epochs.
# - loss2: epoch500 -> epoch1000, so train 500 more local epochs.
# - loss3: epoch500 -> epoch1000, so train 500 more local epochs.
# Existing checkpoints contain model weights/config, not AdamW optimizer state;
# continuation resumes weights and epoch/global-step numbering while AdamW restarts.
run_continue loss1 "$LOSS1_CONT_RUN" "$LOSS1_CKPT" 2000 1000 3000
run_continue loss2 "$LOSS2_CONT_RUN" "$LOSS2_CKPT" 500 500 1500
run_continue loss3 "$LOSS3_CONT_RUN" "$LOSS3_CKPT" 500 500 1500

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] continuation all done" | tee -a "$LOG_ROOT/driver.log"
