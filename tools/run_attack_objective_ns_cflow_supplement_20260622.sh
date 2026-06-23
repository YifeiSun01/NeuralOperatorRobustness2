#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/workspace/NeuralOperatorRobustness2"
PY="/venv/adv_robust/bin/python"
OUT="${ROOT}/analysis_outputs/attack_objective_true_loss3_comparison_20260622"
LOGDIR="${OUT}/run_logs"
ABORTED_DIR="${OUT}/aborted_runs"
QUEUE_LOG="${LOGDIR}/supplement_ns_cflow_physics_20260622.log"
DONE_FILE="${LOGDIR}/supplement_ns_cflow_physics_20260622.done"
PID_FILE="${LOGDIR}/supplement_ns_cflow_physics_20260622.pid"
RCLONE_CONFIG_FILE="${RCLONE_CONFIG_FILE:-/tmp/rclone-r2-nor.conf}"
R2_DEST="${R2_DEST:-r2:neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected/analysis_outputs/attack_objective_true_loss3_comparison_20260622}"
RUN_ID="$(date -u +%Y%m%d_%H%M%S_UTC)"

NS_CHECKPOINT="2D_NS_FNO2d_recurrent/saved_models/2D/modes64_modes64_width60_epochs500_Tin10_T10_recurrent_pytorch_20260521_090136_UTC/checkpoints/final.pt"
NS_TEST_PATH="2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/real_initial_laxmap_single/test/dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_test_ntimepoints21_all_frames.pt"
NS_DICTIONARY_PATH="2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dictionary/dim2d_nx256_N2000_solver=exponax_nu0.000_t20.0_dict_ntimepoints21_batch0_all_frames.pt"

DARCY_DATASET="${ROOT}/2D_Darcy_FNO2d/datasets/grf_darcy_20260528_N1500/test/dim2d_darcy_nx211_N300_solver=jaxcg_solve421_alpha2_tau3_binary3-12_f1_seed45_test.pt"
DARCY_CHECKPOINT="${ROOT}/2D_Darcy_FNO2d/saved_models/2D/darcy_N1500_nx85_m64_w60_e500_20260528/best.pt"

cd "${ROOT}"
mkdir -p "${LOGDIR}" "${ABORTED_DIR}" "${OUT}/raw_runs/ns2d" "${OUT}/raw_runs/darcy_physics" "${OUT}/archives"

log() {
  printf '[supplement] %s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

trap 'rc=$?; log "ERROR rc=${rc} line=${LINENO}: ${BASH_COMMAND}"; exit "${rc}"' ERR

gpu_snapshot() {
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits || true
}

summarize() {
  log "refreshing objective tables, figures, and report"
  if ! "${PY}" tools/summarize_attack_objective_true_loss3.py --analysis-root "${OUT}"; then
    log "WARN: summary refresh failed; continuing GPU queue and watchdog will retry later"
  fi
}

summarize_strict() {
  log "refreshing final objective tables, figures, and report"
  "${PY}" tools/summarize_attack_objective_true_loss3.py --analysis-root "${OUT}"
}

quarantine_incomplete_dir() {
  local path="$1"
  local label="$2"
  if [[ ! -d "${path}" ]]; then
    return 0
  fi
  local dest="${ABORTED_DIR}/${label}_incomplete_${RUN_ID}_$(date -u +%Y%m%d_%H%M%S_UTC)"
  log "quarantining incomplete previous output ${path} -> ${dest}"
  mv "${path}" "${dest}"
}

ns_tag_complete() {
  local tag="$1"
  local start_index="$2"
  local num_samples="$3"
  shift 3
  "${PY}" - "$OUT/raw_runs/ns2d/${tag}" "${start_index}" "${num_samples}" "$@" <<'PY'
from pathlib import Path
import sys
import pandas as pd

root = Path(sys.argv[1])
start = int(sys.argv[2])
num = int(sys.argv[3])
losses = sys.argv[4:]
expected = set(range(start, start + num))
if not root.exists():
    raise SystemExit(1)
for loss in losses:
    seen = set()
    for csv_path in root.rglob(f"{loss}/steepest_add/final_state_metrics.csv"):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if "dataset_index" in df:
            seen.update(int(v) for v in df["dataset_index"].dropna().tolist())
    if not expected.issubset(seen):
        raise SystemExit(1)
raise SystemExit(0)
PY
}

run_ns_one() {
  local tag="$1"
  local start_index="$2"
  local num_samples="$3"
  local batch_size="$4"
  local eps="$5"
  local alpha="$6"
  local losses="$7"
  local mode_spec="$8"
  local log_path="${LOGDIR}/${tag}.log"
  local -a loss_args
  read -r -a loss_args <<< "${losses}"

  if ns_tag_complete "${tag}" "${start_index}" "${num_samples}" "${loss_args[@]}"; then
    log "skip ${tag}: already has samples ${start_index}..$((start_index + num_samples - 1)) for ${losses}"
    return 0
  fi
  quarantine_incomplete_dir "${OUT}/raw_runs/ns2d/${tag}" "${tag}"

  log "start ${tag}: start=${start_index}, N=${num_samples}, batch=${batch_size}, eps=${eps}, alpha=${alpha}, losses=${losses}, mode=${mode_spec}"
  gpu_snapshot
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 \
  "${PY}" -u 2D_NS_FNO2d_recurrent/perturbation_methods/attack_ns2d_recurrent_core4.py \
    --out-root "${OUT}/raw_runs/ns2d/${tag}" \
    --checkpoint "${NS_CHECKPOINT}" \
    --test-path "${NS_TEST_PATH}" \
    --dictionary-path "${NS_DICTIONARY_PATH}" \
    --start-index "${start_index}" \
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
    --clear-jax-caches-after-batch \
    --loss-types "${loss_args[@]}" \
    --mode-spec "${mode_spec}" \
    --epsilon "${eps}" \
    --alpha "${alpha}" \
    2>&1 | tee "${log_path}"
  log "done ${tag}"
  gpu_snapshot
}

run_ns_one_with_fallback() {
  local tag="$1"
  local start_index="$2"
  local num_samples="$3"
  local batch_size="$4"
  local eps="$5"
  local alpha="$6"
  local losses="$7"
  local mode_spec="$8"

  set +e
  run_ns_one "${tag}" "${start_index}" "${num_samples}" "${batch_size}" "${eps}" "${alpha}" "${losses}" "${mode_spec}"
  local rc=$?
  set -e
  if [[ "${rc}" -eq 0 ]]; then
    return 0
  fi

  local fallback_batch=2
  local fallback_tag="${tag}_fallback_batch${fallback_batch}_${RUN_ID}"
  log "${tag} failed with rc=${rc}; retrying as ${fallback_tag}"
  run_ns_one "${fallback_tag}" "${start_index}" "${num_samples}" "${fallback_batch}" "${eps}" "${alpha}" "${losses}" "${mode_spec}"
}

run_ns_budget_pair() {
  local eps="$1"
  local alpha="$2"
  local eps_tag="${eps//./p}"
  local alpha_tag="${alpha//./p}"
  local start_index=12
  local num_samples=8
  local batch_size=4

  run_ns_one_with_fallback \
    "ns2d_supplement_allw_loss1_loss3_eps${eps_tag}_alpha${alpha_tag}_start${start_index}_N${num_samples}_to_N20_steps50_batch${batch_size}_chunkremat" \
    "${start_index}" "${num_samples}" "${batch_size}" "${eps}" "${alpha}" "loss1 loss3" "all_w"
  summarize

  run_ns_one_with_fallback \
    "ns2d_supplement_loss2_all_a_target_w_eps${eps_tag}_alpha${alpha_tag}_start${start_index}_N${num_samples}_to_N20_steps50_batch${batch_size}_chunkremat" \
    "${start_index}" "${num_samples}" "${batch_size}" "${eps}" "${alpha}" "loss2" "all_a_target_w"
  summarize
}

darcy_physics_complete() {
  local run_name="$1"
  "${PY}" - "$OUT/raw_runs/darcy_physics/${run_name}/final_per_sample.csv" <<'PY'
from pathlib import Path
import sys
import pandas as pd

path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit(1)
df = pd.read_csv(path)
if "dataset_index" not in df:
    raise SystemExit(1)
seen = set(int(v) for v in df["dataset_index"].dropna().tolist())
raise SystemExit(0 if set(range(20)).issubset(seen) else 1)
PY
}

run_darcy_physics_one() {
  local flips="$1"
  local run_name="darcy_cflow_loss4_physics_epsflips${flips}_alphaflips5_steps100_N20_steepest_replace"
  local log_path="${LOGDIR}/${run_name}.log"

  if darcy_physics_complete "${run_name}"; then
    log "skip ${run_name}: already has N=20 physics final_per_sample rows"
    return 0
  fi
  quarantine_incomplete_dir "${OUT}/raw_runs/darcy_physics/${run_name}" "${run_name}"

  log "start ${run_name}: physics loss4, flips=${flips}, alpha_flips=5, N=20"
  gpu_snapshot
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 \
  "${PY}" -u 2D_Darcy_FNO2d/perturbation_methods/attack_darcy_binary_physics_loss4.py \
    --dataset "${DARCY_DATASET}" \
    --checkpoint "${DARCY_CHECKPOINT}" \
    --start 0 \
    --num-samples 20 \
    --steps 100 \
    --method steepest_replace \
    --metric rel_l2 \
    --physics-metric rel_l2 \
    --bc-weight 1.0 \
    --epsilon-fraction 0.01 \
    --epsilon-flips "${flips}" \
    --alpha-flips 5 \
    --output-root "${OUT}/raw_runs/darcy_physics" \
    --run-name "${run_name}" \
    2>&1 | tee "${log_path}"
  log "done ${run_name}"
  gpu_snapshot
  summarize
}

make_archive() {
  local archive="${OUT}/archives/ns_cflow_supplement_${RUN_ID}.tar.gz"
  log "creating compact archive ${archive}"
  tar -C "${OUT}" \
    --exclude='archives/*.tar.gz' \
    --exclude='run_logs/*.pid' \
    -czf "${archive}" \
    tables figures reports raw_runs/ns2d raw_runs/darcy_physics run_logs
  log "archive ready ${archive}"
}

wait_for_existing_rclone() {
  local current_pid
  while current_pid="$(pgrep -f "rclone copy .*attack_objective_true_loss3_comparison_20260622" | tr '\n' ' ')" && [[ -n "${current_pid// }" ]]; do
    log "waiting for existing R2 upload pid(s): ${current_pid}"
    sleep 60
  done
}

upload_r2() {
  if [[ ! -f "${RCLONE_CONFIG_FILE}" ]]; then
    log "R2 upload skipped: config file not found at ${RCLONE_CONFIG_FILE}"
    return 0
  fi
  wait_for_existing_rclone
  log "syncing updated analysis outputs to R2: ${R2_DEST}"
  rclone copy "${OUT}" "${R2_DEST}" \
    --config "${RCLONE_CONFIG_FILE}" \
    --fast-list \
    --transfers 8 \
    --checkers 16 \
    --s3-no-check-bucket \
    --s3-no-head \
    --s3-no-system-metadata \
    --s3-disable-checksum \
    --exclude 'archives/*.tar.gz' \
    --stats 60s \
    --stats-one-line
  log "R2 sync complete"
}

main() {
  exec >> "${QUEUE_LOG}" 2>&1
  echo "$$" > "${PID_FILE}"
  rm -f "${DONE_FILE}"

  log "supplement queue launched"
  log "phase 1: NS2D supplement samples 12..19, four budgets, loss1/loss2/loss3 objectives"
  run_ns_budget_pair 8 0.25
  run_ns_budget_pair 16 0.5
  run_ns_budget_pair 32 1
  run_ns_budget_pair 64 2

  log "phase 2: Darcy/CFlow auxiliary physics loss4 for matching flip budgets"
  run_darcy_physics_one 100
  run_darcy_physics_one 250
  run_darcy_physics_one 437
  run_darcy_physics_one 875

  log "phase 3: final summary, archive, and R2 resync"
  summarize_strict
  make_archive
  upload_r2
  printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${DONE_FILE}"
  log "supplement queue complete"
}

main "$@"
