#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/workspace/NeuralOperatorRobustness2}"
PY="${PY:-$ROOT/adv_robust/bin/python}"
DATE_TAG="${DATE_TAG:-20260617}"
GEN_ROOT="${GEN_ROOT:-$ROOT/generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00}"
GEN_DATA_ROOT="$GEN_ROOT/burgers"
OUT_ROOT="${OUT_ROOT:-$ROOT/adversarial_training_runs}"
LOG_ROOT="${LOG_ROOT:-$ROOT/run_logs/burgers_fixed_eps1_full_${DATE_TAG}}"
PREFLIGHT_DIR="${PREFLIGHT_DIR:-$ROOT/forensics/burgers_fixed_eps1_preflight_${DATE_TAG}}"
TRAIN_VIZ_DIR="${TRAIN_VIZ_DIR:-$ROOT/visualizations/burgers_fixed_eps1_six_method_training_curves_${DATE_TAG}}"
FULL_SUITE_ROOT="${FULL_SUITE_ROOT:-$ROOT/forensics/burgers_fixed_eps1_baseline_plus_six_full_suite_${DATE_TAG}}"
REPORT_MD="${REPORT_MD:-$ROOT/docs/burgers_fixed_eps1_full_pipeline_${DATE_TAG}.md}"
GIT_BRANCH="${GIT_BRANCH:-vast-ai-darcy-flow}"
UPLOAD_TO_R2="${UPLOAD_TO_R2:-1}"
AUTO_GIT_PUSH="${AUTO_GIT_PUSH:-1}"
DRY_RUN="${DRY_RUN:-0}"

TRAIN_TEST_ROOT="$ROOT/1D_Burgers/datasets/1D/Burgers/batched_exponax_splits/dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45"
TRAIN_PATH="$TRAIN_TEST_ROOT/dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45_train.pt"
TEST_PATH="$TRAIN_TEST_ROOT/dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45_test.pt"
BASELINE_CKPT="$ROOT/1D_Burgers/trained_models/attack_ready/burgers_nu0.001_fno1d_500/checkpoints/pytorch_fno1d_500.pt"
SVD_MANIFEST="${SVD_MANIFEST:-$ROOT/forensics/burgers_wideparam_loss3targeted_full1024_svd_attack25_reuse3_20260611/sample_manifest.csv}"

RUN_LOSS1="${RUN_LOSS1:-burgers_fixed_eps1_loss1_8h_${DATE_TAG}}"
RUN_LOSS2="${RUN_LOSS2:-burgers_fixed_eps1_loss2_8h_${DATE_TAG}}"
RUN_LOSS3="${RUN_LOSS3:-burgers_fixed_eps1_loss3_8h_${DATE_TAG}}"
RUN_CLEAN="${RUN_CLEAN:-burgers_fixed_eps1_clean_8h_${DATE_TAG}}"
RUN_RANDOM_CLEAN="${RUN_RANDOM_CLEAN:-burgers_fixed_eps1_random_clean_y_8h_${DATE_TAG}}"
RUN_RANDOM_SOLVER="${RUN_RANDOM_SOLVER:-burgers_fixed_eps1_random_solver_y_8h_${DATE_TAG}}"

MAX_WORK_SECONDS="${MAX_WORK_SECONDS:-28800}"
CHECKPOINT_EVERY_EPOCHS="${CHECKPOINT_EVERY_EPOCHS:-100}"
CHECKPOINT_WALL_HOURS="${CHECKPOINT_WALL_HOURS:-0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,9,10,12,16,20,24}"
ATTACK_PROBE_SAMPLES="${ATTACK_PROBE_SAMPLES:-5}"
EPSILON_BUCKET_COUNT="${EPSILON_BUCKET_COUNT:-1}"
BURGERS_BATCH="${BURGERS_BATCH:-1350}"
BURGERS_OPT_BATCH="${BURGERS_OPT_BATCH:-32}"
BURGERS_SOLVER_REMAT="${BURGERS_SOLVER_REMAT:-chunk}"
BURGERS_SOLVER_REMAT_CHUNK_STEPS="${BURGERS_SOLVER_REMAT_CHUNK_STEPS:-50}"
ADV_ATTACK_STEPS="${ADV_ATTACK_STEPS:-5}"
ADV_EPSILON_FRACTION="${ADV_EPSILON_FRACTION:-0.06}"
RANDOM_EPSILON_FRACTION="${RANDOM_EPSILON_FRACTION:-0.04}"
CLEAN_EPOCH_CAP="${CLEAN_EPOCH_CAP:-30000}"
LOSS1_EPOCH_CAP="${LOSS1_EPOCH_CAP:-12000}"
LOSS2_EPOCH_CAP="${LOSS2_EPOCH_CAP:-4000}"
LOSS3_EPOCH_CAP="${LOSS3_EPOCH_CAP:-2500}"
RANDOM_EPOCH_CAP="${RANDOM_EPOCH_CAP:-30000}"

mkdir -p "$LOG_ROOT" "$PREFLIGHT_DIR" "$(dirname "$REPORT_MD")"
cd "$ROOT"

if [[ -f /root/.darcy_sir20_secrets.env ]]; then
  # shellcheck disable=SC1091
  source /root/.darcy_sir20_secrets.env
fi
export GIT_ASKPASS="${GIT_ASKPASS:-/root/.git_askpass_github_token.sh}"
export GIT_TERMINAL_PROMPT="${GIT_TERMINAL_PROMPT:-0}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.35}"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG_ROOT/driver.log"
}

write_report() {
  local status="$1"
  cat > "$REPORT_MD" <<EOF
# Burgers Fixed-Epsilon Full Pipeline, ${DATE_TAG}

Status: ${status}

This reruns Burgers 1024 adversarial/noise/clean training on the latest 52-dataset wide-parameter loss3-targeted set.

- Dataset root: \`generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00\`
- Train/test split: \`1D_Burgers/datasets/1D/Burgers/batched_exponax_splits/...nu0.001...seed45\`
- Baseline: \`1D_Burgers/trained_models/attack_ready/burgers_nu0.001_fno1d_500/checkpoints/pytorch_fno1d_500.pt\`
- Methods: \`loss1\`, \`loss2\`, \`loss3\`, \`clean\`, \`random_clean_y\`, \`random_solver_y\`
- Epsilon jitter: fixed \`low=1.0\`, \`high=1.0\` for every Burgers run
- Work-clock target: about ${MAX_WORK_SECONDS} seconds per training method
- Training curves: \`${TRAIN_VIZ_DIR#$ROOT/}\`
- Full clean/attack/SVD suite: \`${FULL_SUITE_ROOT#$ROOT/}\`
- Preflight: \`${PREFLIGHT_DIR#$ROOT/}\`
EOF
}

require_path() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    log "missing required path: $path"
    exit 1
  fi
}

configure_rclone_env() {
  export RCLONE_CONFIG_R2RUN_TYPE=s3
  export RCLONE_CONFIG_R2RUN_PROVIDER=Cloudflare
  export RCLONE_CONFIG_R2RUN_ACCESS_KEY_ID="${R2_ACCESS_KEY_ID:-}"
  export RCLONE_CONFIG_R2RUN_SECRET_ACCESS_KEY="${R2_SECRET_ACCESS_KEY:-}"
  export RCLONE_CONFIG_R2RUN_ENDPOINT="${R2_ENDPOINT:-}"
}

download_if_missing() {
  if [[ -d "$GEN_DATA_ROOT" && -f "$TRAIN_PATH" && -f "$TEST_PATH" && -f "$BASELINE_CKPT" && -f "$SVD_MANIFEST" ]]; then
    return 0
  fi
  if [[ -z "${R2_BUCKET:-}" || -z "${R2_PREFIX:-}" || -z "${R2_ACCESS_KEY_ID:-}" || -z "${R2_SECRET_ACCESS_KEY:-}" || -z "${R2_ENDPOINT:-}" ]]; then
    log "R2 credentials not available and local Burgers data are incomplete"
    exit 1
  fi
  configure_rclone_env
  local remote="R2RUN:${R2_BUCKET}/${R2_PREFIX}"
  log "downloading missing Burgers data from R2"
  rclone copy "$remote/generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611" \
    "$ROOT/generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611" \
    --create-empty-src-dirs --transfers 16 --checkers 32 --fast-list >> "$LOG_ROOT/r2_download.log" 2>&1
  rclone copy "$remote/1D_Burgers/datasets/1D/Burgers/batched_exponax_splits/dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45" \
    "$TRAIN_TEST_ROOT" --transfers 8 --checkers 16 >> "$LOG_ROOT/r2_download.log" 2>&1
  rclone copy "$remote/1D_Burgers/trained_models/attack_ready/burgers_nu0.001_fno1d_500/checkpoints" \
    "$(dirname "$BASELINE_CKPT")" --transfers 8 --checkers 16 >> "$LOG_ROOT/r2_download.log" 2>&1
  rclone copy "$remote/forensics/burgers_wideparam_loss3targeted_full1024_svd_attack25_reuse3_20260611" \
    "$ROOT/forensics/burgers_wideparam_loss3targeted_full1024_svd_attack25_reuse3_20260611" \
    --include 'sample_manifest.csv' --include 'config.json' --transfers 4 --checkers 8 >> "$LOG_ROOT/r2_download.log" 2>&1
}

preflight() {
  download_if_missing
  require_path "$PY"
  require_path "$GEN_DATA_ROOT"
  require_path "$TRAIN_PATH"
  require_path "$TEST_PATH"
  require_path "$BASELINE_CKPT"
  require_path "$SVD_MANIFEST"
  "$PY" - <<'PY' > "$PREFLIGHT_DIR/dataset_52_nx1024_validation.json"
import json, time
from pathlib import Path
import torch

root = Path.cwd()
train = root / "1D_Burgers/datasets/1D/Burgers/batched_exponax_splits/dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45/dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45_train.pt"
test = train.with_name("dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45_test.pt")
gen_root = root / "generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00/burgers"
baseline = root / "1D_Burgers/trained_models/attack_ready/burgers_nu0.001_fno1d_500/checkpoints/pytorch_fno1d_500.pt"
rows = []
for dataset_id, path in [("train_original_gaussian_corr0p03", train), ("test_original_gaussian_corr0p03", test)] + [(p.stem, p) for p in sorted(gen_root.glob("*.pt"))]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    x = data["x"].float()
    y = data["y"].float()
    if x.shape[-1] != 1024 or y.shape[-1] != 1024:
        raise ValueError(f"{path} is not nx1024: x={tuple(x.shape)} y={tuple(y.shape)}")
    rows.append({
        "dataset_id": dataset_id,
        "path": str(path.relative_to(root)),
        "samples": int(x.shape[0]),
        "x_shape": list(x.shape),
        "y_shape": list(y.shape),
        "x_min": float(x.min()),
        "x_max": float(x.max()),
        "y_min": float(y.min()),
        "y_max": float(y.max()),
    })
summary = {
    "checked_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "dataset_count": len(rows),
    "generalization_count": max(0, len(rows) - 2),
    "all_nx1024": all(r["x_shape"][-1] == 1024 and r["y_shape"][-1] == 1024 for r in rows),
    "baseline_checkpoint_exists": baseline.exists(),
    "sample_count_summary": {
        "train": rows[0]["samples"],
        "test": rows[1]["samples"],
        "generalization_unique": sorted({r["samples"] for r in rows[2:]}),
    },
    "rows": rows,
}
if summary["dataset_count"] != 52 or summary["generalization_count"] != 50 or not summary["all_nx1024"] or not summary["baseline_checkpoint_exists"]:
    raise SystemExit(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY
  "$PY" - <<'PYGPU' > "$PREFLIGHT_DIR/gpu_preflight.json"
import json, sys, time, torch
payload = {
    "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "python": sys.executable,
    "torch_version": torch.__version__,
    "torch_cuda_version": torch.version.cuda,
    "torch_cuda_available": torch.cuda.is_available(),
}
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable")
payload["device_name"] = torch.cuda.get_device_name(0)
payload["device_capability"] = list(torch.cuda.get_device_capability(0))
x = torch.randn(128, 128, device="cuda")
y = x @ x
torch.cuda.synchronize()
payload["torch_sanity_sum"] = float(y.sum().detach().cpu())
print(json.dumps(payload, indent=2))
PYGPU
}

common_train_args() {
  printf '%s\n' \
    "$PY" "$ROOT/tools/adversarial_training.py" \
    --tasks burgers \
    --generalization-root "$GEN_ROOT" \
    --output-root "$OUT_ROOT" \
    --device cuda \
    --seed 20260617 \
    --checkpoint-every-epochs "$CHECKPOINT_EVERY_EPOCHS" \
    --checkpoint-wall-hours "$CHECKPOINT_WALL_HOURS" \
    --label-mode solver \
    --epsilon-bucket-count "$EPSILON_BUCKET_COUNT" \
    --attack-probe-samples "$ATTACK_PROBE_SAMPLES" \
    --attack-probe-every-n-epochs 1 \
    --attack-probe-save-targets \
    --burgers-attack-method fast_replace_l2 \
    --burgers-require-p2q2 \
    --burgers-batch-size "$BURGERS_BATCH" \
    --burgers-optimizer-batch-size "$BURGERS_OPT_BATCH" \
    --burgers-solver-remat "$BURGERS_SOLVER_REMAT" \
    --burgers-solver-remat-chunk-steps "$BURGERS_SOLVER_REMAT_CHUNK_STEPS" \
    --burgers-eps-jitter-low 1.0 \
    --burgers-eps-jitter-high 1.0 \
    --burgers-alpha-ratio 1.0 \
    --burgers-alpha-jitter-low 0.75 \
    --burgers-alpha-jitter-high 1.25 \
    --eval-max-samples 0 \
    --max-generalization-eval 50 \
    --max-work-seconds "$MAX_WORK_SECONDS"
}

run_cmd_logged() {
  local log_path="$1"
  shift
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '[dry-run]' | tee -a "$log_path"
    printf ' %q' "$@" | tee -a "$log_path"
    printf '\n' | tee -a "$log_path"
    return 0
  fi
  "$@" > "$log_path" 2>&1
}

run_adv_method() {
  local label="$1"
  local run_name="$2"
  local loss="$3"
  local epoch_cap="$4"
  local run_dir="$OUT_ROOT/$run_name"
  local log_path="$LOG_ROOT/${run_name}.log"
  if [[ -f "$run_dir/burgers/summary.json" ]]; then
    log "skip completed $label: $run_name"
    return 0
  fi
  if [[ -d "$run_dir" ]]; then
    log "refusing existing incomplete run dir: $run_dir"
    exit 1
  fi
  mapfile -t cmd < <(common_train_args)
  cmd+=(--run-name "$run_name" --epochs "$epoch_cap" --training-data-mode adv-only)
  cmd+=(--burgers-attack-loss-objective "$loss" --burgers-attack-steps "$ADV_ATTACK_STEPS")
  cmd+=(--burgers-epsilon-fraction "$ADV_EPSILON_FRACTION" --burgers-random-start-fraction 1e-6)
  log "start $label fixed-eps1 run=$run_name cap_epochs=$epoch_cap max_work_seconds=$MAX_WORK_SECONDS"
  run_cmd_logged "$log_path" "${cmd[@]}"
  log "done $label fixed-eps1 run=$run_name"
}

run_clean_method() {
  local run_name="$RUN_CLEAN"
  local run_dir="$OUT_ROOT/$run_name"
  local log_path="$LOG_ROOT/${run_name}.log"
  if [[ -f "$run_dir/burgers/summary.json" ]]; then
    log "skip completed clean: $run_name"
    return 0
  fi
  if [[ -d "$run_dir" ]]; then
    log "refusing existing incomplete run dir: $run_dir"
    exit 1
  fi
  mapfile -t cmd < <(common_train_args)
  cmd+=(--run-name "$run_name" --epochs "$CLEAN_EPOCH_CAP" --training-data-mode clean-only)
  cmd+=(--burgers-attack-loss-objective loss3 --burgers-attack-steps 0)
  cmd+=(--burgers-epsilon-fraction "$ADV_EPSILON_FRACTION" --burgers-random-start-fraction 0.0)
  log "start clean-only fixed-eps1 run=$run_name cap_epochs=$CLEAN_EPOCH_CAP max_work_seconds=$MAX_WORK_SECONDS"
  run_cmd_logged "$log_path" "${cmd[@]}"
  log "done clean-only fixed-eps1 run=$run_name"
}

run_random_method() {
  local label="$1"
  local run_name="$2"
  local target_mode="$3"
  local run_dir="$OUT_ROOT/$run_name"
  local log_path="$LOG_ROOT/${run_name}.log"
  if [[ -f "$run_dir/burgers/summary.json" ]]; then
    log "skip completed $label: $run_name"
    return 0
  fi
  if [[ -d "$run_dir" ]]; then
    log "refusing existing incomplete run dir: $run_dir"
    exit 1
  fi
  mapfile -t cmd < <(common_train_args)
  cmd+=(--run-name "$run_name" --epochs "$RANDOM_EPOCH_CAP" --training-data-mode adv-only)
  cmd+=(--training-perturbation-mode random-field --random-field-target-mode "$target_mode")
  cmd+=(--burgers-attack-loss-objective loss3 --burgers-attack-steps 0)
  cmd+=(--burgers-epsilon-fraction "$RANDOM_EPSILON_FRACTION" --burgers-random-start-fraction 0.0)
  log "start $label fixed-eps1 run=$run_name target=$target_mode cap_epochs=$RANDOM_EPOCH_CAP max_work_seconds=$MAX_WORK_SECONDS"
  run_cmd_logged "$log_path" "${cmd[@]}"
  log "done $label fixed-eps1 run=$run_name"
}

checkpoint_from_summary() {
  local run_name="$1"
  "$PY" - "$OUT_ROOT/$run_name/burgers/summary.json" <<'PY'
import json, sys
from pathlib import Path
summary = Path(sys.argv[1])
payload = json.loads(summary.read_text(encoding="utf-8"))
path = Path(payload["final_checkpoint"])
print(str(path if path.is_absolute() else Path.cwd() / path))
PY
}

plot_training_curves() {
  log "plotting six-method training curves"
  "$PY" "$ROOT/tools/plot_burgers_wideparam_loss123_retrain_20260611.py" \
    --run "loss1=$OUT_ROOT/$RUN_LOSS1" \
    --run "loss2=$OUT_ROOT/$RUN_LOSS2" \
    --run "loss3=$OUT_ROOT/$RUN_LOSS3" \
    --run "clean=$OUT_ROOT/$RUN_CLEAN" \
    --run "random_clean_y=$OUT_ROOT/$RUN_RANDOM_CLEAN" \
    --run "random_solver_y=$OUT_ROOT/$RUN_RANDOM_SOLVER" \
    --out-dir "$TRAIN_VIZ_DIR" \
    --report-md "$ROOT/docs/burgers_fixed_eps1_training_curves_${DATE_TAG}.md" \
    --report-title "Burgers Fixed-Epsilon Six-Method Training Curves - ${DATE_TAG}" \
    --report-description "Six Burgers 1024 training methods rerun with epsilon jitter fixed to 1.0/1.0 on the latest 52-dataset widevis loss3-targeted set." \
    --wall-clock-xmax-hours 8 \
    > "$LOG_ROOT/plot_training_curves.log" 2>&1
}

run_full_suite() {
  if [[ -f "$FULL_SUITE_ROOT/done.json" ]]; then
    log "skip completed full suite: $FULL_SUITE_ROOT"
    return 0
  fi
  local loss1_ckpt loss2_ckpt loss3_ckpt clean_ckpt random_clean_ckpt random_solver_ckpt
  loss1_ckpt="$(checkpoint_from_summary "$RUN_LOSS1")"
  loss2_ckpt="$(checkpoint_from_summary "$RUN_LOSS2")"
  loss3_ckpt="$(checkpoint_from_summary "$RUN_LOSS3")"
  clean_ckpt="$(checkpoint_from_summary "$RUN_CLEAN")"
  random_clean_ckpt="$(checkpoint_from_summary "$RUN_RANDOM_CLEAN")"
  random_solver_ckpt="$(checkpoint_from_summary "$RUN_RANDOM_SOLVER")"
  log "running baseline-plus-six clean/attack/SVD suite"
  "$PY" "$ROOT/tools/evaluate_burgers_random_field_checkpoint_series_20260613.py" \
    --out-root "$FULL_SUITE_ROOT" \
    --stage clean \
    --stage attack \
    --stage svd \
    --stage postprocess \
    --clean-batch-size "${CLEAN_BATCH_SIZE:-256}" \
    --attack-steps "${FULL_SUITE_ATTACK_STEPS:-20}" \
    --attack-batch-size "${FULL_SUITE_ATTACK_BATCH_SIZE:-500}" \
    --attack-train-count "${FULL_SUITE_ATTACK_TRAIN_COUNT:-50}" \
    --svd-manifest "$SVD_MANIFEST" \
    --svd-max-samples "${FULL_SUITE_SVD_MAX_SAMPLES:-25}" \
    --svd-method "${FULL_SUITE_SVD_METHOD:-topk}" \
    --svd-top-k "${FULL_SUITE_SVD_TOP_K:-20}" \
    --model-spec "baseline=$BASELINE_CKPT" \
    --model-spec "loss1=$loss1_ckpt" \
    --model-spec "loss2=$loss2_ckpt" \
    --model-spec "loss3=$loss3_ckpt" \
    --model-spec "clean=$clean_ckpt" \
    --model-spec "random_clean_y=$random_clean_ckpt" \
    --model-spec "random_solver_y=$random_solver_ckpt" \
    > "$LOG_ROOT/full_suite.log" 2>&1
  log "done baseline-plus-six full suite"
}

upload_path() {
  local path="$1"
  if [[ "$UPLOAD_TO_R2" != "1" ]]; then
    log "R2 upload disabled for $path"
    return 0
  fi
  if [[ ! -e "$path" ]]; then
    log "skip missing upload path: $path"
    return 0
  fi
  if [[ -z "${R2_BUCKET:-}" || -z "${R2_PREFIX:-}" || -z "${R2_ACCESS_KEY_ID:-}" || -z "${R2_SECRET_ACCESS_KEY:-}" || -z "${R2_ENDPOINT:-}" ]]; then
    log "R2 credentials missing; skip upload for $path"
    return 0
  fi
  configure_rclone_env
  local rel="${path#$ROOT/}"
  log "uploading to R2: $rel"
  if [[ -f "$path" ]]; then
    rclone copyto "$path" "R2RUN:${R2_BUCKET}/${R2_PREFIX}/${rel}" --s3-no-check-bucket >> "$LOG_ROOT/r2_upload.log" 2>&1
  else
    rclone copy "$path" "R2RUN:${R2_BUCKET}/${R2_PREFIX}/${rel}" --transfers 8 --checkers 16 --s3-no-check-bucket >> "$LOG_ROOT/r2_upload.log" 2>&1
  fi
}

git_commit_push() {
  if [[ "$AUTO_GIT_PUSH" != "1" ]]; then
    log "git push disabled"
    return 0
  fi
  local candidates=(
    "$ROOT/tools/adversarial_training.py"
    "$ROOT/tools/plot_burgers_wideparam_loss123_retrain_20260611.py"
    "$ROOT/tools/run_burgers_fixed_eps1_full_${DATE_TAG}.sh"
    "$REPORT_MD"
    "$ROOT/docs/burgers_fixed_eps1_training_curves_${DATE_TAG}.md"
    "$PREFLIGHT_DIR/dataset_52_nx1024_validation.json"
    "$PREFLIGHT_DIR/gpu_preflight.json"
  )
  local existing=()
  local p
  for p in "${candidates[@]}"; do
    [[ -e "$p" ]] && existing+=("$p")
  done
  if [[ "${#existing[@]}" -eq 0 ]]; then
    log "git no existing paths"
    return 0
  fi
  git add "${existing[@]}"
  if git diff --cached --quiet; then
    log "git no staged changes"
  else
    git commit -m "Add Burgers fixed-epsilon rerun pipeline" | tee -a "$LOG_ROOT/git.log"
  fi
  GIT_TERMINAL_PROMPT=0 git push origin "$GIT_BRANCH" | tee -a "$LOG_ROOT/git.log"
}

main() {
  write_report "prepared/running"
  log "Burgers fixed-eps1 full pipeline start"
  preflight
  run_adv_method loss1 "$RUN_LOSS1" loss1 "$LOSS1_EPOCH_CAP"
  run_adv_method loss2 "$RUN_LOSS2" loss2 "$LOSS2_EPOCH_CAP"
  run_adv_method loss3 "$RUN_LOSS3" loss3 "$LOSS3_EPOCH_CAP"
  run_clean_method
  run_random_method random_clean_y "$RUN_RANDOM_CLEAN" clean-y
  run_random_method random_solver_y "$RUN_RANDOM_SOLVER" solver-y
  if [[ "$DRY_RUN" == "1" ]]; then
    log "dry-run complete after command construction"
    return 0
  fi
  plot_training_curves
  run_full_suite
  write_report "complete"
  upload_path "$OUT_ROOT/$RUN_LOSS1"
  upload_path "$OUT_ROOT/$RUN_LOSS2"
  upload_path "$OUT_ROOT/$RUN_LOSS3"
  upload_path "$OUT_ROOT/$RUN_CLEAN"
  upload_path "$OUT_ROOT/$RUN_RANDOM_CLEAN"
  upload_path "$OUT_ROOT/$RUN_RANDOM_SOLVER"
  upload_path "$TRAIN_VIZ_DIR"
  upload_path "$FULL_SUITE_ROOT"
  upload_path "$PREFLIGHT_DIR"
  upload_path "$REPORT_MD"
  upload_path "$LOG_ROOT"
  git_commit_push
  log "Burgers fixed-eps1 full pipeline complete"
}

main "$@"
