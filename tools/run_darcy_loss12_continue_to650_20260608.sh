#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="$ROOT/adv_robust/bin/python"
OUT_ROOT="$ROOT/adversarial_training_runs"
GEN_ROOT="$ROOT/generalization_datasets_darcy_lossdrop50_selected_20260607"
MODEL_CKPT="$ROOT/2D_Darcy_FNO2d/saved_models/2D/darcy_screen_baseline_m64_w60_e50_20260607/best.pt"
TRAIN_PATH="$ROOT/2D_Darcy_FNO2d/datasets/grf_darcy_screen_20260607/train/dim2d_darcy_nx85_N384_solver=jaxcg_solve421_alpha2_tau3_binary3-12_f1_seed45_train.pt"
TEST_PATH="$ROOT/2D_Darcy_FNO2d/datasets/grf_darcy_screen_20260607/test/dim2d_darcy_nx85_N96_solver=jaxcg_solve421_alpha2_tau3_binary3-12_f1_seed45_test.pt"
TARGET_EPOCH="${DARCY_TARGET_EPOCH:-650}"
WAIT_FOR_SOURCE="${WAIT_FOR_SOURCE:-1}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-60}"
UPLOAD_TO_R2="${UPLOAD_TO_R2:-1}"
R2_BASE_PREFIX="${R2_PREFIX:-machine-sync/NeuralOperatorRobustness2-selected}"
RUN_TAG="${RUN_TAG:-20260608}"
LOG_ROOT="$OUT_ROOT/darcy_loss12_continue_to650_${RUN_TAG}_logs"
PREFLIGHT_DIR="$ROOT/forensics/darcy_loss12_continue_to650_${RUN_TAG}_gpu_preflight"
COMBINED_SUMMARY_DIR="$OUT_ROOT/darcy_lossdrop50_loss12_to650_loss3_physics_summary_${RUN_TAG}"
COMBINED_DOC="$ROOT/docs/darcy_lossdrop50_loss12_to650_loss3_physics_summary_${RUN_TAG}.md"
LEDGER="$ROOT/EXPERIMENT_LEDGER.md"

LOSS1_SOURCE_RUN="${LOSS1_SOURCE_RUN:-$OUT_ROOT/darcy_lossdrop50_loss1_single_gpu_time_matched_loss3wall_20260608}"
LOSS2_SOURCE_RUN="${LOSS2_SOURCE_RUN:-$OUT_ROOT/darcy_lossdrop50_loss2_single_gpu_time_matched_loss3wall_20260608}"
LOSS3_RUN="${LOSS3_RUN:-$OUT_ROOT/darcy_lossdrop50_loss3_500ep_fromscreen_20260607}"
PHYSICS_RUN="${PHYSICS_RUN:-$OUT_ROOT/darcy_lossdrop50_physics_time_matched_loss3wall_20260608}"

export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.35}"

mkdir -p "$LOG_ROOT" "$PREFLIGHT_DIR"

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG_ROOT/driver.log"
}

wait_for_summary() {
  local objective="$1"
  local run_dir="$2"
  local summary="$run_dir/darcy/summary.json"
  while [[ ! -f "$summary" ]]; do
    if [[ "$WAIT_FOR_SOURCE" != "1" ]]; then
      echo "[error] missing source summary for $objective: $summary" >&2
      exit 2
    fi
    log "waiting for $objective source summary: $summary"
    sleep "$CHECK_INTERVAL_SECONDS"
  done
}

wait_no_active_training() {
  local objective="$1"
  while pgrep -af "tools/adversarial_training.py" >/tmp/darcy_active_training_pids.txt; do
    if [[ ! -s /tmp/darcy_active_training_pids.txt ]]; then
      break
    fi
    log "waiting before $objective continuation; active adversarial_training.py process exists: $(tr '\n' '; ' </tmp/darcy_active_training_pids.txt)"
    sleep "$CHECK_INTERVAL_SECONDS"
  done
}

record_gpu_preflight() {
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
    raise SystemExit("CUDA is unavailable; refusing Darcy continuation")
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
}

source_info_shell() {
  local run_dir="$1"
  "$PY" - "$ROOT" "$run_dir" "$TARGET_EPOCH" <<'PYINFO'
import json
import shlex
import sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
run_dir = Path(sys.argv[2]).resolve()
target_epoch = int(sys.argv[3])
task_summary_path = run_dir / "darcy" / "summary.json"
summary = json.loads(task_summary_path.read_text(encoding="utf-8"))
start_epoch = int(summary["epochs"])
start_step = int(summary["total_steps"])
final_checkpoint = Path(summary["final_checkpoint"])
if not final_checkpoint.is_absolute():
    final_checkpoint = root / final_checkpoint
if start_epoch > target_epoch:
    raise SystemExit(f"source epoch {start_epoch} is already beyond target {target_epoch}: {task_summary_path}")
if start_epoch < target_epoch and not final_checkpoint.exists():
    raise SystemExit(f"missing source final checkpoint: {final_checkpoint}")
local_epochs = max(0, target_epoch - start_epoch)
print(f"START_EPOCH={start_epoch}")
print(f"START_STEP={start_step}")
print(f"LOCAL_EPOCHS={local_epochs}")
print(f"START_CKPT={shlex.quote(str(final_checkpoint))}")
PYINFO
}

run_continue_to650() {
  local objective="$1"
  local source_run="$2"
  wait_for_summary "$objective" "$source_run"
  eval "$(source_info_shell "$source_run")"
  local cont_run_name="darcy_lossdrop50_${objective}_continue_to${TARGET_EPOCH}_from_epoch${START_EPOCH}_${RUN_TAG}"
  local cont_run_dir="$OUT_ROOT/$cont_run_name"
  local stitched_run_dir="$OUT_ROOT/darcy_lossdrop50_${objective}_to${TARGET_EPOCH}_stitched_${RUN_TAG}"
  local cont_doc="$ROOT/docs/${cont_run_name}.md"
  local stitched_doc="$ROOT/docs/darcy_lossdrop50_${objective}_to${TARGET_EPOCH}_stitched_${RUN_TAG}.md"
  local cont_summary_out="$cont_run_dir/darcy_time_matched_summary"
  local stitched_summary_out="$stitched_run_dir/darcy_time_matched_summary"

  if [[ "$LOCAL_EPOCHS" -eq 0 ]]; then
    log "$objective source is already at target epoch $TARGET_EPOCH; using source run as final"
    cont_run_dir="$source_run"
  elif [[ -f "$cont_run_dir/summary.json" ]]; then
    log "skip completed continuation $cont_run_name"
  else
    wait_no_active_training "$objective"
    log "start $cont_run_name from epoch=$START_EPOCH step=$START_STEP ckpt=$START_CKPT local_epochs=$LOCAL_EPOCHS target_epoch=$TARGET_EPOCH"
    nvidia-smi > "$PREFLIGHT_DIR/nvidia_smi_before_${objective}_continue.txt"
    "$PY" "$ROOT/tools/adversarial_training.py" \
      --tasks darcy \
      --generalization-root "$GEN_ROOT" \
      --output-root "$OUT_ROOT" \
      --run-name "$cont_run_name" \
      --device cuda \
      --seed "${DARCY_TIME_MATCH_SEED:-20260608}" \
      --epochs "$LOCAL_EPOCHS" \
      --resume-epoch-offset "$START_EPOCH" \
      --resume-global-step-offset "$START_STEP" \
      --checkpoint-every-epochs "${DARCY_CHECKPOINT_EVERY_EPOCHS:-50}" \
      --checkpoint-wall-hours "${DARCY_CHECKPOINT_WALL_HOURS:-0.5,1.0,2.0,3.0,4.0,6.0,8.0,12.0,16.0,20.0,24.0}" \
      --training-data-mode adv-only \
      --label-mode solver \
      --epsilon-bucket-count 5 \
      --attack-probe-samples "${DARCY_ATTACK_PROBE_SAMPLES:-5}" \
      --attack-probe-every-n-epochs "${DARCY_ATTACK_PROBE_EVERY_N_EPOCHS:-1}" \
      --attack-probe-save-targets \
      --darcy-model-checkpoint "$MODEL_CKPT" \
      --darcy-initial-checkpoint "$START_CKPT" \
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
      > "$LOG_ROOT/$cont_run_name.log" 2>&1
    log "done $cont_run_name"
    nvidia-smi > "$PREFLIGHT_DIR/nvidia_smi_after_${objective}_continue.txt"
  fi

  if [[ "$LOCAL_EPOCHS" -eq 0 ]]; then
    mkdir -p "$stitched_run_dir"
    rsync -a --exclude 'checkpoints' --exclude 'attack_probe_samples' "$source_run/" "$stitched_run_dir/"
  else
    "$PY" "$ROOT/tools/stitch_darcy_continuation_run_20260608.py" \
      --source-run "$source_run" \
      --continuation-run "$cont_run_dir" \
      --output-run "$stitched_run_dir" \
      --objective "$objective" \
      --target-epoch "$TARGET_EPOCH" \
      > "$LOG_ROOT/${cont_run_name}_stitch.json"
  fi

  "$PY" "$ROOT/tools/summarize_darcy_time_matched_runs_20260608.py" \
    --run-dir "$cont_run_dir" \
    --output-dir "$cont_summary_out" \
    --output-md "$cont_doc" \
    > "$LOG_ROOT/${cont_run_name}_postprocess_summary.json"

  "$PY" "$ROOT/tools/summarize_darcy_time_matched_runs_20260608.py" \
    --run-dir "$stitched_run_dir" \
    --output-dir "$stitched_summary_out" \
    --output-md "$stitched_doc" \
    > "$LOG_ROOT/$(basename "$stitched_run_dir")_postprocess_summary.json"

  "$PY" "$ROOT/tools/plot_eval_train_loss_progress.py" \
    --run-dir "$stitched_run_dir" \
    --task darcy \
    --out-path "$stitched_summary_out/eval_train_loss_progress.png" \
    --title "Darcy $objective stitched self-training to epoch $TARGET_EPOCH" \
    >> "$LOG_ROOT/$(basename "$stitched_run_dir")_postprocess_summary.json" 2>&1 || true

  echo "$stitched_run_dir"
}

upload_path() {
  local path="$1"
  if [[ "$UPLOAD_TO_R2" != "1" ]]; then
    return 0
  fi
  if [[ ! -e "$path" ]]; then
    log "skip missing upload path: $path"
    return 0
  fi
  log "upload to R2: $path"
  R2_PREFIX="$R2_BASE_PREFIX/darcy_loss12_to${TARGET_EPOCH}_${RUN_TAG}" \
  R2_UPLOAD_LOG_ROOT="$LOG_ROOT/r2_upload_logs" \
    "$ROOT/tools/upload_path_to_r2_20260525.sh" "$path"
}

record_final_ledger() {
  "$PY" - "$LEDGER" "$COMBINED_SUMMARY_DIR/darcy_time_matched_summary.csv" "$COMBINED_DOC" "$TARGET_EPOCH" <<'PYLEDGER'
import sys
from pathlib import Path
import pandas as pd
ledger = Path(sys.argv[1])
csv_path = Path(sys.argv[2])
doc_path = Path(sys.argv[3])
target_epoch = int(sys.argv[4])
df = pd.read_csv(csv_path)
rows = []
for _, row in df.iterrows():
    rows.append(
        f"- {row['objective']}: epochs `{int(row['epochs_completed'])}`, "
        f"generalization RMSE `{row['generalization_rmse_baseline']:.9g} -> {row['generalization_rmse_final']:.9g}` "
        f"(`{row['generalization_rmse_final_drop_pct']:.2f}%` drop), "
        f"relative L2 `{row['generalization_relative_l2_baseline']:.9g} -> {row['generalization_relative_l2_final']:.9g}` "
        f"(`{row['generalization_relative_l2_final_drop_pct']:.2f}%` drop)."
    )
entry = "\n".join([
    "",
    f"## 2026-06-08 - Darcy loss1/loss2 continuation to epoch {target_epoch} completion",
    "",
    "Status: completed continuation workflow and generated stitched summaries/plots for Darcy loss1/loss2 to the requested target epoch.",
    "",
    "Observed evidence:",
    f"- Combined summary CSV: `{csv_path}`.",
    f"- Combined result doc: `{doc_path}`.",
    "- Stitched loss1/loss2 run directories contain full baseline-to-target CSV records; large checkpoints remain in the source and continuation run directories.",
    "",
    "Key metrics:",
    *rows,
    "",
    "Inference:",
    "- These loss1/loss2 rows supersede the concurrency-contaminated time-matched loss1/loss2 rows for target-epoch comparison.",
    "",
    "Remaining work:",
    "- Inspect uploaded R2 logs if any upload reports warnings or retry failures.",
    "",
])
text = ledger.read_text(encoding='utf-8') if ledger.exists() else ""
if f"## 2026-06-08 - Darcy loss1/loss2 continuation to epoch {target_epoch} completion" not in text:
    ledger.write_text(text.rstrip() + "\n" + entry, encoding='utf-8')
PYLEDGER
}

record_gpu_preflight
log "continuation-to-$TARGET_EPOCH workflow configured: loss1_source=$LOSS1_SOURCE_RUN loss2_source=$LOSS2_SOURCE_RUN upload_to_r2=$UPLOAD_TO_R2"

LOSS1_STITCHED_RUN="$(run_continue_to650 loss1 "$LOSS1_SOURCE_RUN" | tail -n 1)"
LOSS2_STITCHED_RUN="$(run_continue_to650 loss2 "$LOSS2_SOURCE_RUN" | tail -n 1)"

"$PY" "$ROOT/tools/summarize_darcy_time_matched_runs_20260608.py" \
  --run-dir "$LOSS1_STITCHED_RUN" \
  --run-dir "$LOSS2_STITCHED_RUN" \
  --run-dir "$LOSS3_RUN" \
  --run-dir "$PHYSICS_RUN" \
  --output-dir "$COMBINED_SUMMARY_DIR" \
  --output-md "$COMBINED_DOC" \
  > "$LOG_ROOT/combined_loss12_to${TARGET_EPOCH}_postprocess_summary.json"

record_final_ledger

upload_path "$LOSS1_SOURCE_RUN"
upload_path "$LOSS2_SOURCE_RUN"
for path in "$OUT_ROOT"/darcy_lossdrop50_loss1_continue_to"$TARGET_EPOCH"_from_epoch*_"$RUN_TAG" "$OUT_ROOT"/darcy_lossdrop50_loss2_continue_to"$TARGET_EPOCH"_from_epoch*_"$RUN_TAG"; do
  if [[ -e "$path" ]]; then
    upload_path "$path"
  fi
done
upload_path "$LOSS1_STITCHED_RUN"
upload_path "$LOSS2_STITCHED_RUN"
upload_path "$COMBINED_SUMMARY_DIR"
upload_path "$COMBINED_DOC"
upload_path "$ROOT/docs/darcy_lossdrop50_loss1_to${TARGET_EPOCH}_stitched_${RUN_TAG}.md"
upload_path "$ROOT/docs/darcy_lossdrop50_loss2_to${TARGET_EPOCH}_stitched_${RUN_TAG}.md"
upload_path "$ROOT/tools/run_darcy_loss12_continue_to650_20260608.sh"
upload_path "$ROOT/tools/stitch_darcy_continuation_run_20260608.py"
upload_path "$PREFLIGHT_DIR"
upload_path "$LOG_ROOT"

nvidia-smi > "$PREFLIGHT_DIR/nvidia_smi_end.txt"
log "continuation-to-$TARGET_EPOCH workflow complete"
