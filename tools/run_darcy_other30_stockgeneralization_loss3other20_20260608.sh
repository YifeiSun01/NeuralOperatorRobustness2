#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON_BIN:-$ROOT/adv_robust/bin/python}"
OUT_ROOT="$ROOT/adversarial_training_runs"
LOG_ROOT="$OUT_ROOT/darcy_other30_stockgeneralization_loss3other20_20260608_logs"
PREFLIGHT_DIR="$ROOT/forensics/darcy_other30_stockgeneralization_loss3other20_20260608_gpu_preflight"
SCREEN_DATA_DIR="$ROOT/2D_Darcy_FNO2d/datasets/grf_darcy_screen_20260607"
TRAIN_PATH="$SCREEN_DATA_DIR/train/dim2d_darcy_nx85_N384_solver=jaxcg_solve421_alpha2_tau3_binary3-12_f1_seed45_train.pt"
TEST_PATH="$SCREEN_DATA_DIR/test/dim2d_darcy_nx85_N96_solver=jaxcg_solve421_alpha2_tau3_binary3-12_f1_seed45_test.pt"
SAVE_ROOT="$ROOT/2D_Darcy_FNO2d/saved_models/2D"
OTHER30_RUN_NAME="${OTHER30_RUN_NAME:-darcy_flow_other30training_20260608}"
OTHER30_DIR="$SAVE_ROOT/$OTHER30_RUN_NAME"
POOL_ROOT="${POOL_ROOT:-$ROOT/generalization_datasets_darcy_stockgeneralization_pool_20260608}"
SCREEN_DIR="${SCREEN_DIR:-$ROOT/forensics/darcy_stockgeneralization_gradient_screen_20260608}"
STOCK_ROOT="${STOCK_ROOT:-$ROOT/generalization_datasets_darcy_stockgeneralization_20260608}"
LOSS3_RUN_NAME="${LOSS3_RUN_NAME:-darcy_stockloss3other20training_20260608}"
LOSS3_RUN_DIR="$OUT_ROOT/$LOSS3_RUN_NAME"
LOSS3_DOC="$ROOT/docs/darcy_stockloss3other20training_20260608.md"
PIPELINE_DOC="$ROOT/docs/darcy_other30_stockgeneralization_loss3other20_pipeline_20260608.md"
WAIT_FOR_IDLE="${WAIT_FOR_IDLE:-1}"
MIN_FREE_MIB="${MIN_FREE_MIB:-30000}"
SLEEP_SECONDS="${SLEEP_SECONDS:-60}"
ACTIVE_PATTERN="${ACTIVE_PATTERN:-tools/adversarial_training.py|run_darcy_loss12_single_gpu_time_matched_20260608.sh|run_darcy_loss12_continue_to650_20260608.sh|run_ns2d_loss3|run_burgers_round03}"

export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.35}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$LOG_ROOT" "$PREFLIGHT_DIR"

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG_ROOT/driver.log"
}

wait_for_idle_gpu() {
  if [[ "$WAIT_FOR_IDLE" != "1" ]]; then
    return 0
  fi
  while true; do
    active_training="$(pgrep -fa "$ACTIVE_PATTERN" || true)"
    used_mib="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')"
    total_mib="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')"
    free_mib=$(( total_mib - used_mib ))
    if [[ -z "$active_training" && "$free_mib" -ge "$MIN_FREE_MIB" ]]; then
      log "GPU idle enough: free_mib=$free_mib used_mib=$used_mib threshold=$MIN_FREE_MIB"
      break
    fi
    {
      date -u
      printf '[wait] active_training=%s\n' "${active_training:-none}"
      printf '[wait] gpu_free_mib=%s gpu_used_mib=%s threshold=%s\n' "$free_mib" "$used_mib" "$MIN_FREE_MIB"
    } | tee -a "$LOG_ROOT/wait_for_idle.log"
    sleep "$SLEEP_SECONDS"
  done
}

record_gpu_preflight() {
  nvidia-smi > "$PREFLIGHT_DIR/nvidia_smi_start.txt"
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
    raise SystemExit("CUDA unavailable; refusing Darcy pipeline")
payload["torch_device_name"] = torch.cuda.get_device_name(0)
payload["torch_compute_capability"] = "sm_%d%d" % torch.cuda.get_device_capability(0)
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
}

require_inputs() {
  for required in "$PY" "$TRAIN_PATH" "$TEST_PATH"; do
    if [[ ! -e "$required" ]]; then
      echo "[missing] $required" >&2
      exit 1
    fi
  done
}

run_other30_training() {
  if [[ -f "$OTHER30_DIR/final.pt" && -f "$OTHER30_DIR/best.pt" ]]; then
    log "reuse completed other30 training: $OTHER30_DIR"
    return 0
  fi
  if [[ -d "$OTHER30_DIR" && ! -f "$OTHER30_DIR/final.pt" ]]; then
    echo "[refuse] incomplete other30 directory exists: $OTHER30_DIR" >&2
    exit 2
  fi
  log "start Darcy Flow other30training: $OTHER30_RUN_NAME"
  "$PY" "$ROOT/2D_Darcy_FNO2d/training_models/train_darcy_fno2d.py" \
    --train-path "$TRAIN_PATH" \
    --test-path "$TEST_PATH" \
    --epochs "${OTHER30_EPOCHS:-30}" \
    --batch-size "${OTHER30_BATCH_SIZE:-16}" \
    --learning-rate "${OTHER30_LR:-0.001}" \
    --weight-decay "${OTHER30_WEIGHT_DECAY:-0.0001}" \
    --modes "${OTHER30_MODES:-64}" \
    --width "${OTHER30_WIDTH:-60}" \
    --num-layers "${OTHER30_NUM_LAYERS:-4}" \
    --padding "${OTHER30_PADDING:-0}" \
    --require-cuda \
    --save-dir "$SAVE_ROOT" \
    --run-name "$OTHER30_RUN_NAME" \
    --save-every "${OTHER30_SAVE_EVERY:-10}" \
    > "$LOG_ROOT/other30training.log" 2>&1
  log "done Darcy Flow other30training: $OTHER30_DIR"
}

run_stockgeneralization() {
  if [[ ! -f "$POOL_ROOT/candidate_manifest.csv" ]]; then
    log "generate stockgeneralization pool: $POOL_ROOT"
    "$PY" "$ROOT/tools/generate_darcy_lossdrop50_pool_20260607.py" \
      --output-root "$POOL_ROOT" \
      --samples-per-candidate "${STOCK_GEN_SAMPLES_PER_CANDIDATE:-48}" \
      --batch-size "${STOCK_GEN_BATCH_SIZE:-8}" \
      --seed "${STOCK_GEN_SEED:-20260608}" \
      > "$LOG_ROOT/stockgeneralization_pool_generation.log" 2>&1
  else
    log "reuse stockgeneralization pool manifest: $POOL_ROOT/candidate_manifest.csv"
  fi

  if [[ ! -f "$SCREEN_DIR/candidate_screen_summary.csv" ]]; then
    log "screen stockgeneralization pool using other30 checkpoint"
    "$PY" "$ROOT/tools/probe_darcy_gradient_alignment_screen.py" \
      --checkpoint "$OTHER30_DIR/best.pt" \
      --train-path "$TRAIN_PATH" \
      --test-path "$TEST_PATH" \
      --generalization-root "$POOL_ROOT" \
      --out-dir "$SCREEN_DIR" \
      --device cuda \
      --steps "${STOCK_SCREEN_STEPS:-50}" \
      --batch-size "${STOCK_SCREEN_ATTACK_BATCH:-96}" \
      --optimizer-batch-size "${STOCK_SCREEN_OPT_BATCH:-24}" \
      --eval-grad-batch-size "${STOCK_SCREEN_EVAL_BATCH:-16}" \
      --eval-gen-samples-per-dataset "${STOCK_SCREEN_EVAL_GEN_SAMPLES:-48}" \
      > "$LOG_ROOT/stockgeneralization_gradient_screen.log" 2>&1
  else
    log "reuse stockgeneralization screen summary: $SCREEN_DIR/candidate_screen_summary.csv"
  fi

  if [[ ! -f "$STOCK_ROOT/candidate_manifest.csv" ]]; then
    log "select 50 stockgeneralization datasets: $STOCK_ROOT"
    "$PY" "$ROOT/tools/select_darcy_lossdrop50_20260607.py" \
      --pool-root "$POOL_ROOT" \
      --screen-dir "$SCREEN_DIR" \
      --selected-root "$STOCK_ROOT" \
      --count "${STOCK_SELECTED_COUNT:-50}" \
      > "$LOG_ROOT/stockgeneralization_selection.log" 2>&1
  else
    log "reuse selected stockgeneralization manifest: $STOCK_ROOT/candidate_manifest.csv"
  fi
}

run_stockloss3other20_training() {
  if [[ -f "$LOSS3_RUN_DIR/summary.json" ]]; then
    log "reuse completed stockloss3other20training: $LOSS3_RUN_DIR"
    return 0
  fi
  if [[ -d "$LOSS3_RUN_DIR" && ! -f "$LOSS3_RUN_DIR/summary.json" ]]; then
    echo "[refuse] incomplete loss3 other20 directory exists: $LOSS3_RUN_DIR" >&2
    exit 2
  fi
  log "start stockloss3other20training: $LOSS3_RUN_NAME"
  "$PY" "$ROOT/tools/adversarial_training.py" \
    --tasks darcy \
    --generalization-root "$STOCK_ROOT" \
    --output-root "$OUT_ROOT" \
    --run-name "$LOSS3_RUN_NAME" \
    --device cuda \
    --seed "${LOSS3_OTHER20_SEED:-20260608}" \
    --epochs "${LOSS3_OTHER20_EPOCHS:-20}" \
    --checkpoint-every-epochs "${LOSS3_OTHER20_CHECKPOINT_EVERY:-10}" \
    --checkpoint-wall-hours "${LOSS3_OTHER20_CHECKPOINT_WALL_HOURS:-0.5,1.0,2.0,3.0,4.0,6.0}" \
    --training-data-mode adv-only \
    --label-mode solver \
    --epsilon-bucket-count 5 \
    --attack-probe-samples "${LOSS3_OTHER20_ATTACK_PROBE_SAMPLES:-5}" \
    --attack-probe-every-n-epochs "${LOSS3_OTHER20_ATTACK_PROBE_EVERY:-1}" \
    --attack-probe-save-targets \
    --darcy-model-checkpoint "$OTHER30_DIR/best.pt" \
    --darcy-train-path "$TRAIN_PATH" \
    --darcy-test-path "$TEST_PATH" \
    --darcy-attack-method binary_steepest_replace \
    --darcy-attack-loss-objective loss3 \
    --darcy-attack-steps "${LOSS3_OTHER20_ATTACK_STEPS:-1}" \
    --darcy-batch-size "${LOSS3_OTHER20_BATCH:-96}" \
    --darcy-optimizer-batch-size "${LOSS3_OTHER20_OPT_BATCH:-24}" \
    --darcy-epsilon-fraction "${LOSS3_OTHER20_EPSILON_FRACTION:-0.025}" \
    --darcy-eps-jitter-low "${LOSS3_OTHER20_EPS_JITTER_LOW:-1.0}" \
    --darcy-eps-jitter-high "${LOSS3_OTHER20_EPS_JITTER_HIGH:-1.0}" \
    --darcy-alpha-ratio 1.0 \
    --eval-max-samples 0 \
    --max-generalization-eval 50 \
    > "$LOG_ROOT/stockloss3other20training.log" 2>&1
  log "done stockloss3other20training: $LOSS3_RUN_DIR"
}

postprocess_loss3() {
  if [[ -f "$LOSS3_RUN_DIR/summary.json" ]]; then
    SUMMARY_OUT="$LOSS3_RUN_DIR/darcy_time_matched_summary"
    "$PY" "$ROOT/tools/summarize_darcy_time_matched_runs_20260608.py" \
      --run-dir "$LOSS3_RUN_DIR" \
      --output-dir "$SUMMARY_OUT" \
      --output-md "$LOSS3_DOC" \
      > "$LOG_ROOT/stockloss3other20_summary.json" || true
    "$PY" "$ROOT/tools/plot_eval_train_loss_progress.py" \
      --run-dir "$LOSS3_RUN_DIR" \
      --task darcy \
      --out-path "$SUMMARY_OUT/eval_train_loss_progress.png" \
      --title "Darcy stock loss3 other20 training" \
      >> "$LOG_ROOT/stockloss3other20_summary.json" 2>&1 || true
  fi
}

write_pipeline_status() {
  "$PY" - "$PIPELINE_DOC" "$OTHER30_DIR" "$POOL_ROOT" "$SCREEN_DIR" "$STOCK_ROOT" "$LOSS3_RUN_DIR" <<'PYSTATUS'
import json
import time
from pathlib import Path
import sys
path = Path(sys.argv[1])
other_dir = Path(sys.argv[2])
pool_root = Path(sys.argv[3])
screen_dir = Path(sys.argv[4])
stock_root = Path(sys.argv[5])
loss3_dir = Path(sys.argv[6])
lines = [
    "# Darcy other30 + stockgeneralization + stockloss3other20 pipeline",
    "",
    f"Updated UTC: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
    "",
    "Observed evidence and paths:",
    "",
    f"- other30 training directory: `{other_dir}`; final exists: `{(other_dir / 'final.pt').exists()}`; best exists: `{(other_dir / 'best.pt').exists()}`.",
    f"- stockgeneralization pool: `{pool_root}`; manifest exists: `{(pool_root / 'candidate_manifest.csv').exists()}`.",
    f"- stockgeneralization screen: `{screen_dir}`; summary exists: `{(screen_dir / 'candidate_screen_summary.csv').exists()}`.",
    f"- selected stockgeneralization root: `{stock_root}`; manifest exists: `{(stock_root / 'candidate_manifest.csv').exists()}`.",
    f"- stockloss3other20 run: `{loss3_dir}`; summary exists: `{(loss3_dir / 'summary.json').exists()}`.",
    "",
    "Inference:",
    "",
    "- This pipeline treats `other30training` as a 30-epoch Darcy FNO baseline trained on the local Darcy screen train/test tensors.",
    "- It treats `stockgeneralization` as a newly generated Darcy loss-drop-style candidate pool screened with the other30 checkpoint and selected to 50 datasets.",
    "- It treats `stockloss3other20training` as 20 epochs of Darcy solver-label loss3 adversarial self-training initialized from the other30 checkpoint.",
]
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps({"pipeline_doc": str(path), "loss3_summary_exists": (loss3_dir / 'summary.json').exists()}, indent=2))
PYSTATUS
}

main() {
  require_inputs
  log "pipeline requested: other30training -> stockgeneralization -> stockloss3other20training"
  wait_for_idle_gpu
  record_gpu_preflight
  run_other30_training
  run_stockgeneralization
  run_stockloss3other20_training
  postprocess_loss3
  write_pipeline_status | tee -a "$LOG_ROOT/driver.log"
  log "pipeline complete"
}

main "$@"
