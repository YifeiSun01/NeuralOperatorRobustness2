#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/workspace/NeuralOperatorRobustness2}"
PY="${PY:-/venv/adv_robust/bin/python}"
BRANCH="${BRANCH:-vast-ai}"
SVD_OUT_ROOT="${SVD_OUT_ROOT:-$ROOT/forensics/burgers_wideparam_loss3targeted_full1024_svd_attack25_reuse3_20260611}"
SVD_REPORT="${SVD_REPORT:-$ROOT/docs/burgers_wideparam_loss3targeted_full1024_svd_attack25_reuse3_20260611.md}"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"
MASTER_LOG="${MASTER_LOG:-$LOG_DIR/burgers_svd25_then_retrain_upload_20260611.log}"
STATE_DIR="${STATE_DIR:-$ROOT/forensics/burgers_svd25_then_retrain_upload_state_20260611}"
SVD_BIASED_DIR_OUT="${SVD_BIASED_DIR_OUT:-$ROOT/forensics/burgers_wideparam_loss3targeted_full1024_svd_attack25_biased_local_direction_20260611}"
SVD_BIASED_DIR_REPORT="${SVD_BIASED_DIR_REPORT:-$ROOT/docs/burgers_wideparam_loss3targeted_full1024_svd_attack25_biased_local_direction_20260611.md}"
RUN_SVD_ANALYSIS="${RUN_SVD_ANALYSIS:-1}"
RUN_SVD="${RUN_SVD:-1}"
RUN_RETRAIN="${RUN_RETRAIN:-1}"
RUN_UPLOAD="${RUN_UPLOAD:-1}"
RUN_GIT="${RUN_GIT:-1}"
SMOKE_ONLY="${SMOKE_ONLY:-0}"

mkdir -p "$LOG_DIR" "$STATE_DIR"
cd "$ROOT"

log() {
  echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*" | tee -a "$MASTER_LOG"
}

upload_path() {
  local rel="$1"
  if [[ "$RUN_UPLOAD" != "1" ]]; then
    log "upload skipped for $rel"
    return 0
  fi
  if [[ ! -e "$rel" ]]; then
    log "upload skipped missing path $rel"
    return 0
  fi
  log "upload start $rel"
  R2_UPLOAD_LOG_ROOT="$STATE_DIR/r2_upload_logs" bash "$ROOT/tools/upload_path_to_r2_20260525.sh" "$rel" | tee -a "$MASTER_LOG"
  log "upload done $rel"
}

git_commit_push() {
  local message="$1"
  shift
  if [[ "$RUN_GIT" != "1" ]]; then
    log "git skipped: $message"
    return 0
  fi
  local existing=()
  local candidate
  for candidate in "$@"; do
    if [[ -e "$candidate" ]]; then
      existing+=("$candidate")
    else
      log "git skip missing path: $candidate"
    fi
  done
  if [[ "${#existing[@]}" -eq 0 ]]; then
    log "git no existing paths: $message"
    return 0
  fi
  git add "${existing[@]}"
  if git diff --cached --quiet; then
    log "git no staged changes: $message"
  else
    git commit -m "$message" | tee -a "$MASTER_LOG"
  fi
  if [[ -n "${GITHUB_TOKEN_FILE:-}" && -f "$GITHUB_TOKEN_FILE" ]]; then
    export GITHUB_TOKEN="$(<"$GITHUB_TOKEN_FILE")"
  fi
  if [[ -n "${GH_TOKEN:-}" && -z "${GITHUB_TOKEN:-}" ]]; then
    export GITHUB_TOKEN="$GH_TOKEN"
  fi
  if [[ -n "${GITHUB_TOKEN:-}" ]]; then
    local askpass
    askpass="$(mktemp)"
    cat > "$askpass" <<'ASKPASS'
#!/usr/bin/env bash
case "$1" in
  *Username*) printf '%s\n' "${GIT_PUSH_USER:-YifeiSun01}" ;;
  *Password*) printf '%s\n' "${GITHUB_TOKEN}" ;;
  *) printf '\n' ;;
esac
ASKPASS
    chmod 700 "$askpass"
    GIT_ASKPASS="$askpass" GIT_TERMINAL_PROMPT=0 git push origin "$BRANCH" | tee -a "$MASTER_LOG"
    rm -f "$askpass"
  else
    GIT_TERMINAL_PROMPT=0 git push origin "$BRANCH" | tee -a "$MASTER_LOG"
  fi
}

log "workflow start root=$ROOT branch=$BRANCH smoke=$SMOKE_ONLY"
log "svd_out=$SVD_OUT_ROOT"
log "svd_report=$SVD_REPORT"

if [[ "$SMOKE_ONLY" == "1" ]]; then
  log "smoke: SVD manifest/report estimate only"
  bash "$ROOT/tools/run_burgers_wideparam_full1024_svd_attack25_reuse3_20260611.sh" --estimate-only | tee -a "$MASTER_LOG"
  log "smoke: SVD biased-direction analysis would run after completed SVD root"
  log "smoke: retrain dry-run"
  DRY_RUN=1 bash "$ROOT/tools/run_burgers_wideparam_loss123_retrain_20260611.sh" | tee -a "$MASTER_LOG"
  log "smoke complete"
  exit 0
fi

if [[ "$RUN_SVD" == "1" ]]; then
  log "SVD25 full run start"
  bash "$ROOT/tools/run_burgers_wideparam_full1024_svd_attack25_reuse3_20260611.sh" | tee -a "$MASTER_LOG"
  log "SVD25 full run finished"
  if [[ "$RUN_SVD_ANALYSIS" == "1" ]]; then
    log "SVD25 biased local direction/correlation analysis start"
    "$PY" "$ROOT/tools/analyze_burgers_biased_local_attack_direction_20260611.py" \
      --svd-root "$SVD_OUT_ROOT" \
      --out-dir "$SVD_BIASED_DIR_OUT" \
      --report-md "$SVD_BIASED_DIR_REPORT" \
      --device "${SVD_ANALYSIS_DEVICE:-cpu}" \
      > "$LOG_DIR/burgers_wideparam_full1024_svd_attack25_biased_direction_20260611.log" 2>&1
    log "SVD25 biased local direction/correlation analysis finished"
    upload_path "${SVD_BIASED_DIR_OUT#$ROOT/}"
    upload_path "${SVD_BIASED_DIR_REPORT#$ROOT/}"
    upload_path "logs/burgers_wideparam_full1024_svd_attack25_biased_direction_20260611.log"
  fi
  upload_path "${SVD_OUT_ROOT#$ROOT/}"
  upload_path "${SVD_REPORT#$ROOT/}"
  upload_path "logs/burgers_wideparam_full1024_svd_attack25_reuse3_20260611.log"
  git_commit_push "Add Burgers full1024 SVD25 reuse3 workflow" \
    EXPERIMENT_LEDGER.md \
    docs/burgers_wideparam_full1024_svd_attack20_plan_20260611.md \
    docs/burgers_wideparam_full1024_svd_attack25_reuse3_workflow_20260611.md \
    docs/burgers_wideparam_loss3targeted_full1024_svd_attack20_20260611.md \
    docs/burgers_wideparam_loss3targeted_full1024_svd_attack25_reuse3_20260611.md \
    docs/burgers_wideparam_loss3targeted_full1024_svd_attack25_biased_local_direction_20260611.md \
    docs/burgers_wideparam_loss123_retrain_plan_20260611.md \
    docs/burgers_wideparam_loss123_retrain_report_20260611.md \
    forensics/burgers_wideparam_loss3targeted_full1024_svd_attack25_biased_local_direction_20260611 \
    tools/analyze_burgers_biased_local_attack_direction_20260611.py \
    tools/run_burgers_wideparam_full1024_svd_attack20_20260611.py \
    tools/run_burgers_wideparam_full1024_svd_attack20_20260611.sh \
    tools/run_burgers_wideparam_full1024_svd_attack25_reuse3_20260611.py \
    tools/run_burgers_wideparam_full1024_svd_attack25_reuse3_20260611.sh \
    tools/run_burgers_wideparam_svd25_then_retrain_upload_20260611.sh \
    tools/run_burgers_wideparam_loss123_retrain_20260611.sh \
    tools/probe_burgers_wideparam_loss123_vram_20260611.sh \
    tools/plot_burgers_wideparam_loss123_retrain_20260611.py
fi

if [[ "$RUN_RETRAIN" == "1" ]]; then
  log "retrain loss3 first, then loss1/loss2 start"
  RETRAIN_ORDER="${RETRAIN_ORDER:-loss3_then_loss12}" bash "$ROOT/tools/run_burgers_wideparam_loss123_retrain_20260611.sh" | tee -a "$MASTER_LOG"
  log "retrain finished"
  upload_path "adversarial_training_runs/burgers_wideparam_loss1_8000ep_retrain_20260611"
  upload_path "adversarial_training_runs/burgers_wideparam_loss2_2000ep_retrain_20260611"
  upload_path "adversarial_training_runs/burgers_wideparam_loss3_1000ep_retrain_20260611"
  upload_path "adversarial_training_runs/burgers_wideparam_loss123_retrain_20260611_logs"
  upload_path "visualizations/burgers_wideparam_loss123_retrain_20260611"
  upload_path "docs/burgers_wideparam_loss123_retrain_report_20260611.md"
  git_commit_push "Add Burgers wideparam adversarial retrain results" \
    EXPERIMENT_LEDGER.md \
    docs/burgers_wideparam_loss123_retrain_plan_20260611.md \
    docs/burgers_wideparam_loss123_retrain_report_20260611.md \
    tools/run_burgers_wideparam_loss123_retrain_20260611.sh \
    tools/probe_burgers_wideparam_loss123_vram_20260611.sh \
    tools/plot_burgers_wideparam_loss123_retrain_20260611.py \
    tools/analyze_burgers_biased_local_attack_direction_20260611.py
fi

upload_path "${STATE_DIR#$ROOT/}"
log "workflow finished"
