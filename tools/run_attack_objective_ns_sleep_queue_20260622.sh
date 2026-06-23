#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/NeuralOperatorRobustness2"
PY="/venv/adv_robust/bin/python"
OUT="${ROOT}/analysis_outputs/attack_objective_true_loss3_comparison_20260622"
LOGDIR="${OUT}/run_logs"
QUEUE_LOG="${LOGDIR}/ns2d_sleep_queue_20260622.log"

PRIMARY_WAIT_PIDS="${PRIMARY_WAIT_PIDS:-14967 60305}"
R2_DEST="${R2_DEST:-r2:neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected/analysis_outputs/attack_objective_true_loss3_comparison_20260622}"
RCLONE_CONFIG_FILE="${RCLONE_CONFIG_FILE:-}"
GIT_ASKPASS_FILE="${GIT_ASKPASS_FILE:-}"
RUN_ID="$(date -u +%Y%m%d_%H%M%S_UTC)"

cd "${ROOT}"
mkdir -p "${LOGDIR}" "${OUT}/raw_runs/ns2d" "${OUT}/archives"

log() {
  printf '[sleep-queue] %s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

wait_for_pid() {
  local pid="$1"
  local label="$2"
  if [[ -z "${pid}" ]]; then
    return 0
  fi
  if ! kill -0 "${pid}" 2>/dev/null; then
    log "${label} pid ${pid} is not running; continuing"
    return 0
  fi
  log "waiting for ${label} pid ${pid}"
  while kill -0 "${pid}" 2>/dev/null; do
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader,nounits || true
    sleep 60
  done
  log "${label} pid ${pid} finished"
}

summarize() {
  log "summarizing current raw outputs"
  "${PY}" tools/summarize_attack_objective_true_loss3.py --analysis-root "${OUT}"
}

ns_base_args() {
  local num_samples="$1"
  local batch_size="$2"
  printf '%s\n' \
    --checkpoint \
    2D_NS_FNO2d_recurrent/saved_models/2D/modes64_modes64_width60_epochs500_Tin10_T10_recurrent_pytorch_20260521_090136_UTC/checkpoints/final.pt \
    --test-path \
    '2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/real_initial_laxmap_single/test/dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_test_ntimepoints21_all_frames.pt' \
    --dictionary-path \
    '2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dictionary/dim2d_nx256_N2000_solver=exponax_nu0.000_t20.0_dict_ntimepoints21_batch0_all_frames.pt' \
    --start-index 0 \
    --num-samples "${num_samples}" \
    --attack-batch-size "${batch_size}" \
    --methods steepest_add \
    --steps 50 \
    --p 2 \
    --q 2 \
    --true-loss-every 10 \
    --solver-remat chunk \
    --solver-remat-chunk-steps 20 \
    --dictionary-chunk-size 16 \
    --record-final-state-outputs \
    --no-record-step-sample-outputs \
    --no-record-step-sample-gradients \
    --no-empty-torch-cache-after-method \
    --clear-jax-caches-after-batch
}

run_ns_one() {
  local tag="$1"
  local num_samples="$2"
  local batch_size="$3"
  local eps="$4"
  local alpha="$5"
  local losses="$6"
  local mode_spec="$7"
  local log_path="${LOGDIR}/${tag}.log"

  log "start ${tag}: N=${num_samples}, batch=${batch_size}, eps=${eps}, alpha=${alpha}, losses=${losses}, mode=${mode_spec}"
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 \
  "${PY}" -u 2D_NS_FNO2d_recurrent/perturbation_methods/attack_ns2d_recurrent_core4.py \
    --out-root "${OUT}/raw_runs/ns2d/${tag}" \
    $(ns_base_args "${num_samples}" "${batch_size}") \
    --loss-types ${losses} \
    --mode-spec "${mode_spec}" \
    --epsilon "${eps}" \
    --alpha "${alpha}" \
    2>&1 | tee "${log_path}"
  log "done ${tag}"
}

run_ns_one_with_fallback() {
  local tag="$1"
  local num_samples="$2"
  local batch_size="$3"
  local eps="$4"
  local alpha="$5"
  local losses="$6"
  local mode_spec="$7"

  set +e
  run_ns_one "${tag}" "${num_samples}" "${batch_size}" "${eps}" "${alpha}" "${losses}" "${mode_spec}"
  local rc=$?
  set -e
  if [[ "${rc}" -eq 0 ]]; then
    return 0
  fi
  local fallback_batch=2
  local fallback_tag="${tag}_fallback_batch${fallback_batch}"
  log "${tag} failed with rc=${rc}; retrying as ${fallback_tag}"
  run_ns_one "${fallback_tag}" "${num_samples}" "${fallback_batch}" "${eps}" "${alpha}" "${losses}" "${mode_spec}"
}

run_ns_budget_pair() {
  local eps="$1"
  local alpha="$2"
  local eps_tag="${eps//./p}"
  local alpha_tag="${alpha//./p}"
  local num_samples=12
  local batch_size=4

  run_ns_one_with_fallback \
    "ns2d_sleep_budget_allw_loss1_loss3_eps${eps_tag}_alpha${alpha_tag}_N${num_samples}_steps50_batch${batch_size}_chunkremat" \
    "${num_samples}" "${batch_size}" "${eps}" "${alpha}" "loss1 loss3" "all_w"
  summarize

  run_ns_one_with_fallback \
    "ns2d_sleep_budget_loss2_all_a_target_w_eps${eps_tag}_alpha${alpha_tag}_N${num_samples}_steps50_batch${batch_size}_chunkremat" \
    "${num_samples}" "${batch_size}" "${eps}" "${alpha}" "loss2" "all_a_target_w"
  summarize
}

make_archive() {
  local archive="${OUT}/archives/ns2d_sleep_queue_results_${RUN_ID}.tar.gz"
  log "creating archive ${archive}"
  tar -C "${OUT}" -czf "${archive}" \
    tables figures reports raw_runs/ns2d run_logs \
    --exclude='run_logs/*.pid' \
    --exclude='run_logs/*.runner.pid'
  log "archive ready ${archive}"
}

upload_r2() {
  if [[ -z "${RCLONE_CONFIG_FILE}" || ! -f "${RCLONE_CONFIG_FILE}" ]]; then
    log "R2 upload skipped: RCLONE_CONFIG_FILE is missing"
    return 0
  fi
  log "uploading analysis root to R2 destination ${R2_DEST}"
  rclone copy "${OUT}" "${R2_DEST}" \
    --config "${RCLONE_CONFIG_FILE}" \
    --fast-list \
    --transfers 16 \
    --checkers 32 \
    --s3-no-check-bucket \
    --stats 60s
  log "R2 upload complete"
}

git_backup() {
  log "preparing GitHub backup commit"
  git add \
    tools/summarize_attack_objective_true_loss3.py \
    tools/run_attack_objective_ns_sleep_queue_20260622.sh \
    tools/run_attack_objective_ns_sweeps_resume_20260622.sh \
    tools/run_attack_objective_extra_sweeps_20260622.sh \
    "${OUT}/tables" \
    "${OUT}/figures" \
    "${OUT}/reports"

  if git diff --cached --quiet; then
    log "no staged Git changes to commit"
  else
    git commit -m "Add NS2D attack objective sleep queue results"
  fi

  if [[ -n "${GIT_ASKPASS_FILE}" && -x "${GIT_ASKPASS_FILE}" ]]; then
    log "pushing GitHub backup"
    GIT_ASKPASS="${GIT_ASKPASS_FILE}" GIT_TERMINAL_PROMPT=0 git push origin "$(git branch --show-current)"
    log "GitHub push complete"
  else
    log "GitHub push skipped: GIT_ASKPASS_FILE is missing or not executable"
  fi
}

main() {
  exec > >(tee -a "${QUEUE_LOG}") 2>&1
  log "queue launched with PRIMARY_WAIT_PIDS=${PRIMARY_WAIT_PIDS}"
  log "phase 0: wait for current full NS jobs"
  for pid in ${PRIMARY_WAIT_PIDS}; do
    wait_for_pid "${pid}" "current NS2D job"
  done

  log "phase 1: final summary after full NS jobs"
  summarize

  log "phase 2: sleep budget sweeps, N=12, steps=50, eps/alpha={8/0.25,16/0.5,32/1,64/2}"
  run_ns_budget_pair 8 0.25
  run_ns_budget_pair 16 0.5
  run_ns_budget_pair 32 1
  run_ns_budget_pair 64 2

  log "phase 3: final summary, archive, R2 upload, GitHub backup"
  summarize
  make_archive
  upload_r2
  git_backup
  log "queue complete"
}

main "$@"
