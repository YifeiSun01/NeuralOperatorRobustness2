#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="$ROOT/adv_robust/bin/python"
GEN_ROOT="$ROOT/generalization_datasets_darcy_lossdrop50_selected_20260607"
OUT_ROOT="$ROOT/adversarial_training_runs"
LOSS3_TASK_SUMMARY="$OUT_ROOT/darcy_lossdrop50_loss3_500ep_fromscreen_20260607/darcy/summary.json"
MODEL_CKPT="$ROOT/2D_Darcy_FNO2d/saved_models/2D/darcy_screen_baseline_m64_w60_e50_20260607/best.pt"
TRAIN_PATH="$ROOT/2D_Darcy_FNO2d/datasets/grf_darcy_screen_20260607/train/dim2d_darcy_nx85_N384_solver=jaxcg_solve421_alpha2_tau3_binary3-12_f1_seed45_train.pt"
TEST_PATH="$ROOT/2D_Darcy_FNO2d/datasets/grf_darcy_screen_20260607/test/dim2d_darcy_nx85_N96_solver=jaxcg_solve421_alpha2_tau3_binary3-12_f1_seed45_test.pt"
LOG_ROOT="$OUT_ROOT/darcy_loss12_single_gpu_time_matched_20260608_logs"
PREFLIGHT_DIR="$ROOT/forensics/darcy_loss12_single_gpu_time_matched_20260608_gpu_preflight"
TARGET_SECONDS="${DARCY_TIME_MATCH_SECONDS:-$($PY - "$LOSS3_TASK_SUMMARY" <<'PYSEC'
import json, sys
print(float(json.load(open(sys.argv[1]))['elapsed_seconds']))
PYSEC
)}"

export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.35}"

mkdir -p "$LOG_ROOT" "$PREFLIGHT_DIR"

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
    raise SystemExit("CUDA is unavailable; refusing Darcy training")
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
nvidia-smi > "$PREFLIGHT_DIR/nvidia_smi_start.txt"

run_objective() {
  local objective="$1"
  local run_name="darcy_lossdrop50_${objective}_single_gpu_time_matched_loss3wall_20260608"
  local run_dir="$OUT_ROOT/$run_name"
  local doc_path="$ROOT/docs/${run_name}.md"
  local summary_out="$run_dir/darcy_time_matched_summary"
  local fig_path="$summary_out/eval_train_loss_progress.png"

  if [[ -f "$run_dir/summary.json" ]]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] skip completed $run_name" | tee -a "$LOG_ROOT/driver.log"
  else
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] start $run_name objective=$objective target_seconds=$TARGET_SECONDS" | tee -a "$LOG_ROOT/driver.log"
    "$PY" "$ROOT/tools/adversarial_training.py" \
      --tasks darcy \
      --generalization-root "$GEN_ROOT" \
      --output-root "$OUT_ROOT" \
      --run-name "$run_name" \
      --device cuda \
      --seed "${DARCY_TIME_MATCH_SEED:-20260608}" \
      --epochs "${DARCY_TIME_MATCH_MAX_EPOCHS:-100000}" \
      --max-wall-seconds "$TARGET_SECONDS" \
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
      --darcy-attack-loss-objective "$objective" \
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
      > "$LOG_ROOT/$run_name.log" 2>&1
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] done $run_name" | tee -a "$LOG_ROOT/driver.log"
  fi

  "$PY" "$ROOT/tools/summarize_darcy_time_matched_runs_20260608.py" \
    --run-dir "$run_dir" \
    --output-dir "$summary_out" \
    --output-md "$doc_path" \
    > "$LOG_ROOT/${run_name}_postprocess_summary.json"

  "$PY" "$ROOT/tools/plot_eval_train_loss_progress.py" \
    --run-dir "$run_dir" \
    --task darcy \
    --out-path "$fig_path" \
    --title "Darcy $objective single-GPU time-matched self-training" \
    >> "$LOG_ROOT/${run_name}_postprocess_summary.json" 2>&1 || true

  nvidia-smi > "$PREFLIGHT_DIR/nvidia_smi_after_${objective}.txt"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] workflow complete run=$run_dir doc=$doc_path" | tee -a "$LOG_ROOT/driver.log"
}

run_objective loss1
run_objective loss2

nvidia-smi > "$PREFLIGHT_DIR/nvidia_smi_end.txt"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] sequential loss1/loss2 single-GPU time-matched workflow complete" | tee -a "$LOG_ROOT/driver.log"
