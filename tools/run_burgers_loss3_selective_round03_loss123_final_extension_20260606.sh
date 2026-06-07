#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="$ROOT/adv_robust/bin/python"
GEN_ROOT="$ROOT/generalization_datasets_burgers_loss3_selective_search/round_03"
OUT_ROOT="$ROOT/adversarial_training_runs"
LOG_ROOT="$OUT_ROOT/burgers_loss3_selective_round03_loss123_final_extension_20260606_logs"
PREFLIGHT_DIR="$ROOT/forensics/burgers_loss3_selective_round03_loss123_final_extension_gpu_preflight_20260606"
SAMPLE_MANIFEST="$ROOT/forensics/burgers_loss3_selective_round03_long_final_jacobian_svd_rep20_top100_20260605/round03_long_final_sample_manifest.csv"
SVD_OUT_DIR="$ROOT/forensics/burgers_loss3_selective_round03_loss123_final_extension_jacobian_svd_rep20_top100_20260606"
SVD_PREFIX="round03_loss123_final_extension"
SVD_SUMMARY="$SVD_OUT_DIR/${SVD_PREFIX}_jacobian_svd_summary.csv"
SVD_LOG="$LOG_ROOT/svd_final_5000_2000_1500_rep20_top100.log"
PLOT_LOG="$LOG_ROOT/plot_final_extension_5000_2000_1500.log"
mkdir -p "$LOG_ROOT" "$PREFLIGHT_DIR"

LOSS1_START_RUN="$OUT_ROOT/burgers_loss3_selective_round03_loss1_continue1000to3000_20260605"
LOSS2_START_RUN="$OUT_ROOT/burgers_loss3_selective_round03_loss2_continue500to1000_20260605"
LOSS3_START_RUN="$OUT_ROOT/burgers_loss3_selective_round03_loss3_continue500to1000_20260605"
LOSS1_START_CKPT="$LOSS1_START_RUN/burgers/checkpoints/burgers_epoch3000_step009000.pt"
LOSS2_START_CKPT="$LOSS2_START_RUN/burgers/checkpoints/burgers_epoch1000_step003000.pt"
LOSS3_START_CKPT="$LOSS3_START_RUN/burgers/checkpoints/burgers_epoch1000_step003000.pt"

LOSS1_RUN="burgers_loss3_selective_round03_loss1_continue3000to5000_20260606"
LOSS2_RUN="burgers_loss3_selective_round03_loss2_continue1000to2000_20260606"
LOSS3_RUN="burgers_loss3_selective_round03_loss3_continue1000to1500_20260606"
LOSS1_FINAL_CKPT="$OUT_ROOT/$LOSS1_RUN/burgers/checkpoints/burgers_epoch5000_step015000.pt"
LOSS2_FINAL_CKPT="$OUT_ROOT/$LOSS2_RUN/burgers/checkpoints/burgers_epoch2000_step006000.pt"
LOSS3_FINAL_CKPT="$OUT_ROOT/$LOSS3_RUN/burgers/checkpoints/burgers_epoch1500_step004500.pt"

for required in "$PY" "$GEN_ROOT/burgers" "$SAMPLE_MANIFEST" "$LOSS1_START_CKPT" "$LOSS2_START_CKPT" "$LOSS3_START_CKPT"; do
  if [[ ! -e "$required" ]]; then
    echo "[missing] $required" >&2
    exit 1
  fi
done

for run_name in "$LOSS1_RUN" "$LOSS2_RUN" "$LOSS3_RUN"; do
  run_dir="$OUT_ROOT/$run_name"
  if [[ -f "$run_dir/summary.json" ]]; then
    echo "[skip] completed final-extension run already exists: $run_dir"
    continue
  fi
  if [[ -d "$run_dir" ]]; then
    echo "[refuse] run directory exists without summary.json: $run_dir" >&2
    echo "[refuse] inspect or move it before starting final-extension training." >&2
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
  --checkpoint-wall-hours 0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,8.0,9.0,10.0
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

run_final_svd() {
  if [[ "${RUN_FINAL_EXTENSION_SVD:-1}" != "1" ]]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] skip final-extension SVD because RUN_FINAL_EXTENSION_SVD=${RUN_FINAL_EXTENSION_SVD:-0}" | tee -a "$LOG_ROOT/driver.log"
    return 0
  fi
  if [[ -f "$SVD_SUMMARY" ]]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] skip completed final-extension SVD: $SVD_SUMMARY" | tee -a "$LOG_ROOT/driver.log"
    return 0
  fi
  for required in "$LOSS1_FINAL_CKPT" "$LOSS2_FINAL_CKPT" "$LOSS3_FINAL_CKPT" "$SAMPLE_MANIFEST"; do
    if [[ ! -f "$required" ]]; then
      echo "[missing] final-extension SVD prerequisite missing: $required" >&2
      exit 1
    fi
  done

  mkdir -p "$SVD_OUT_DIR"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] start final-extension Jacobian/SVD -> $SVD_OUT_DIR" | tee -a "$LOG_ROOT/driver.log"
  {
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] START final-extension Jacobian/SVD"
    echo "out_root=$SVD_OUT_DIR"
    echo "sample_manifest=$SAMPLE_MANIFEST"
    echo "loss1_checkpoint=$LOSS1_FINAL_CKPT"
    echo "loss2_checkpoint=$LOSS2_FINAL_CKPT"
    echo "loss3_checkpoint=$LOSS3_FINAL_CKPT"
  } >> "$SVD_LOG"

  "$PY" - <<'PYGPU' >> "$SVD_LOG" 2>&1
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
    raise SystemExit("CUDA is unavailable; refusing final-extension SVD per GPU-only rule")
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
  nvidia-smi >> "$SVD_LOG" 2>&1

  local -a svd_cmd=(
    "$PY" "$ROOT/tools/compare_burgers_round01_final_jacobian_svd.py"
    --out-root "$SVD_OUT_DIR"
    --generalization-root "$GEN_ROOT"
    --sample-manifest "$SAMPLE_MANIFEST"
    --output-prefix "$SVD_PREFIX"
    --report-title "Burgers Loss3-Selective Round03 Final-Extension Final-Model Jacobian/SVD"
    --report-note "Observed on round03 final-extension checkpoints with the same fixed sample manifest used by the pre-continuation long final SVD: loss1 epoch5000, loss2 epoch2000, loss3 epoch1500."
    --loss1-checkpoint "$LOSS1_FINAL_CKPT"
    --loss2-checkpoint "$LOSS2_FINAL_CKPT"
    --loss3-checkpoint "$LOSS3_FINAL_CKPT"
    --loss1-label loss1_epoch5000
    --loss2-label loss2_epoch2000
    --loss3-label loss3_epoch1500
    --loss1-epoch 5000
    --loss2-epoch 2000
    --loss3-epoch 1500
    --train-samples "${FINAL_SVD_TRAIN_SAMPLES:-5}"
    --test-samples "${FINAL_SVD_TEST_SAMPLES:-5}"
    --generalization-samples "${FINAL_SVD_GENERALIZATION_SAMPLES:-10}"
    --seed 20260606
    --device cuda
    --top-k "${FINAL_SVD_TOP_K:-100}"
    --svd-method "${FINAL_SVD_METHOD:-topk}"
    --svd-solver "${FINAL_SVD_SOLVER:-propack}"
  )
  {
    printf '[cmd]'
    printf ' %q' "${svd_cmd[@]}"
    printf '%s\n' ''
  } >> "$SVD_LOG"
  "${svd_cmd[@]}" >> "$SVD_LOG" 2>&1
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] done final-extension Jacobian/SVD" | tee -a "$LOG_ROOT/driver.log"
}

prepare_single_run_plot_dir() {
  local stale_dir="$1"
  local final_dir="$2"
  if [[ -d "$stale_dir" && ! -e "$final_dir" ]]; then
    mv "$stale_dir" "$final_dir"
  fi
  mkdir -p "$final_dir"
}

run_plots() {
  if [[ "${RUN_FINAL_EXTENSION_PLOTS:-1}" != "1" ]]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] skip final-extension plots because RUN_FINAL_EXTENSION_PLOTS=${RUN_FINAL_EXTENSION_PLOTS:-0}" | tee -a "$LOG_ROOT/driver.log"
    return 0
  fi
  for required in "$OUT_ROOT/$LOSS1_RUN/summary.json" "$OUT_ROOT/$LOSS2_RUN/summary.json" "$OUT_ROOT/$LOSS3_RUN/summary.json"; do
    if [[ ! -f "$required" ]]; then
      echo "[missing] final-extension plot prerequisite missing: $required" >&2
      exit 1
    fi
  done

  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] start final-extension plots" | tee -a "$LOG_ROOT/driver.log"
  prepare_single_run_plot_dir \
    "$ROOT/visualizations/burgers_loss3_selective_round03_loss1_3000ep_long_20260605_plots" \
    "$ROOT/visualizations/burgers_loss3_selective_round03_loss1_5000ep_long_20260605_plots"
  prepare_single_run_plot_dir \
    "$ROOT/visualizations/burgers_loss3_selective_round03_loss2_1000ep_long_20260605_plots" \
    "$ROOT/visualizations/burgers_loss3_selective_round03_loss2_2000ep_long_20260605_plots"
  prepare_single_run_plot_dir \
    "$ROOT/visualizations/burgers_loss3_selective_round03_loss3_1000ep_long_20260605_plots" \
    "$ROOT/visualizations/burgers_loss3_selective_round03_loss3_1500ep_long_20260605_plots"

  {
    "$PY" "$ROOT/tools/plot_burgers_round03_stitched_single_run_visualizations.py" \
      --base-run-dir "$ROOT/adversarial_training_runs/burgers_loss3_selective_round03_loss1_1000ep_long_20260605" \
      --extension-run-dir "$LOSS1_START_RUN" \
      --extension-run-dir "$OUT_ROOT/$LOSS1_RUN" \
      --out-dir "$ROOT/visualizations/burgers_loss3_selective_round03_loss1_5000ep_long_20260605_plots" \
      --suffix round03_long
    "$PY" "$ROOT/tools/plot_burgers_round03_stitched_single_run_visualizations.py" \
      --base-run-dir "$ROOT/adversarial_training_runs/burgers_loss3_selective_round03_loss2_500ep_long_20260605" \
      --extension-run-dir "$LOSS2_START_RUN" \
      --extension-run-dir "$OUT_ROOT/$LOSS2_RUN" \
      --out-dir "$ROOT/visualizations/burgers_loss3_selective_round03_loss2_2000ep_long_20260605_plots" \
      --suffix round03_long
    "$PY" "$ROOT/tools/plot_burgers_round03_stitched_single_run_visualizations.py" \
      --base-run-dir "$ROOT/adversarial_training_runs/burgers_loss3_selective_round03_loss3_500ep_long_20260605" \
      --extension-run-dir "$LOSS3_START_RUN" \
      --extension-run-dir "$OUT_ROOT/$LOSS3_RUN" \
      --out-dir "$ROOT/visualizations/burgers_loss3_selective_round03_loss3_1500ep_long_20260605_plots" \
      --suffix round03_long
    "$PY" "$ROOT/tools/plot_burgers_round03_loss123_final_extension_dense_comparison.py"
  } > "$PLOT_LOG" 2>&1
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] done final-extension plots" | tee -a "$LOG_ROOT/driver.log"
}

# Final-extension targets:
# - loss1: epoch3000 -> epoch5000, so train 2000 more local epochs.
# - loss2: epoch1000 -> epoch2000, so train 1000 more local epochs.
# - loss3: epoch1000 -> epoch1500, so train 500 more local epochs.
# Existing checkpoints contain model weights/config, not AdamW optimizer state;
# continuation resumes weights and epoch/global-step numbering while AdamW restarts.
run_continue loss1 "$LOSS1_RUN" "$LOSS1_START_CKPT" 2000 3000 9000
run_continue loss2 "$LOSS2_RUN" "$LOSS2_START_CKPT" 1000 1000 3000
run_continue loss3 "$LOSS3_RUN" "$LOSS3_START_CKPT" 500 1000 3000
run_final_svd
run_plots

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] final-extension workflow done" | tee -a "$LOG_ROOT/driver.log"
