#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/workspace/NeuralOperatorRobustness2"
PY="/venv/adv_robust/bin/python"
OUT="${ROOT}/analysis_outputs/optimizer_ablation_20260622"
LOGDIR="${OUT}/run_logs"
QUEUE_LOG="${LOGDIR}/optimizer_ablation_queue_20260622.log"
PID_FILE="${LOGDIR}/optimizer_ablation_queue_20260622.pid"
DONE_FILE="${LOGDIR}/optimizer_ablation_queue_20260622.done"
CURRENT_DONE="${ROOT}/analysis_outputs/attack_objective_true_loss3_comparison_20260622/run_logs/supplement_ns_cflow_physics_20260622.done"
RCLONE_CONFIG_FILE="${RCLONE_CONFIG_FILE:-/tmp/rclone-r2-nor.conf}"
R2_DEST="${R2_DEST:-r2:neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected/analysis_outputs/optimizer_ablation_20260622}"

NS_N="${OPT_ABLATION_NS_N:-8}"
NS_BATCH="${OPT_ABLATION_NS_BATCH:-4}"
BURGERS_N="${OPT_ABLATION_BURGERS_N:-32}"
DARCY_N="${OPT_ABLATION_DARCY_N:-8}"

NS_CHECKPOINT="2D_NS_FNO2d_recurrent/saved_models/2D/modes64_modes64_width60_epochs500_Tin10_T10_recurrent_pytorch_20260521_090136_UTC/checkpoints/final.pt"
NS_TEST_PATH="2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/real_initial_laxmap_single/test/dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_test_ntimepoints21_all_frames.pt"
NS_DICTIONARY_PATH="2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dictionary/dim2d_nx256_N2000_solver=exponax_nu0.000_t20.0_dict_ntimepoints21_batch0_all_frames.pt"

DARCY_DATASET="${ROOT}/2D_Darcy_FNO2d/datasets/grf_darcy_20260528_N1500/test/dim2d_darcy_nx211_N300_solver=jaxcg_solve421_alpha2_tau3_binary3-12_f1_seed45_test.pt"
DARCY_CHECKPOINT="${ROOT}/2D_Darcy_FNO2d/saved_models/2D/darcy_N1500_nx85_m64_w60_e500_20260528/best.pt"

cd "${ROOT}"
mkdir -p "${LOGDIR}" "${OUT}/raw_runs/ns2d" "${OUT}/raw_runs/burgers" "${OUT}/raw_runs/darcy" "${OUT}/archives"

log() {
  printf '[optimizer-ablation] %s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

trap 'rc=$?; log "ERROR rc=${rc} line=${LINENO}: ${BASH_COMMAND}"; exit "${rc}"' ERR

summarize() {
  log "refreshing optimizer-ablation tables and figures"
  "${PY}" tools/summarize_optimizer_ablation_20260622.py --analysis-root "${OUT}"
}

gpu_snapshot() {
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits || true
}

wait_for_current_queue() {
  log "waiting for current loss-objective supplement queue to finish: ${CURRENT_DONE}"
  while [[ ! -f "${CURRENT_DONE}" ]]; do
    gpu_snapshot
    pgrep -af 'run_attack_objective_ns_cflow_supplement|attack_ns2d|attack_darcy_binary_physics' || true
    sleep 300
  done
  log "current supplement queue done; starting optimizer ablation"
}

ns_complete() {
  local tag="$1"
  "${PY}" - "$OUT/raw_runs/ns2d/${tag}" "$NS_N" <<'PY'
from pathlib import Path
import sys
import pandas as pd

root = Path(sys.argv[1])
n = int(sys.argv[2])
expected = set(range(n))
methods = {"raw_add", "raw_replace", "steepest_add", "steepest_replace"}
if not root.exists():
    raise SystemExit(1)
seen_methods = set()
for method in methods:
    seen = set()
    for path in root.rglob(f"loss3/{method}/final_state_metrics.csv"):
        df = pd.read_csv(path)
        seen.update(int(v) for v in df.get("dataset_index", []))
    if expected.issubset(seen):
        seen_methods.add(method)
raise SystemExit(0 if seen_methods == methods else 1)
PY
}

run_ns_budget() {
  local eps="$1"
  local alpha="$2"
  local eps_tag="${eps//./p}"
  local alpha_tag="${alpha//./p}"
  local tag="ns2d_loss3_core4_eps${eps_tag}_alpha${alpha_tag}_steps100_N${NS_N}_batch${NS_BATCH}_chunkremat"
  local log_path="${LOGDIR}/${tag}.log"
  if ns_complete "${tag}"; then
    log "skip ${tag}: complete"
    return 0
  fi
  log "start ${tag}"
  gpu_snapshot
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 \
  "${PY}" -u 2D_NS_FNO2d_recurrent/perturbation_methods/attack_ns2d_recurrent_core4.py \
    --out-root "${OUT}/raw_runs/ns2d/${tag}" \
    --checkpoint "${NS_CHECKPOINT}" \
    --test-path "${NS_TEST_PATH}" \
    --dictionary-path "${NS_DICTIONARY_PATH}" \
    --start-index 0 \
    --num-samples "${NS_N}" \
    --attack-batch-size "${NS_BATCH}" \
    --methods raw_add raw_replace steepest_add steepest_replace \
    --loss-types loss3 \
    --mode-spec all_w \
    --steps 100 \
    --epsilon "${eps}" \
    --alpha "${alpha}" \
    --p 2 \
    --q 2 \
    --true-loss-every 1 \
    --solver-remat chunk \
    --solver-remat-chunk-steps 20 \
    --dictionary-chunk-size 16 \
    --record-final-state-outputs \
    --no-record-step-sample-outputs \
    --no-record-step-sample-gradients \
    --no-empty-torch-cache-after-method \
    --clear-jax-caches-after-batch \
    2>&1 | tee "${log_path}"
  log "done ${tag}"
  summarize
}

burgers_complete() {
  local root="$OUT/raw_runs/burgers/burgers_loss3_core4_eps4_alpha0p4_steps100_N${BURGERS_N}"
  "${PY}" - "$root/per_sample_step_metrics.csv" "$BURGERS_N" <<'PY'
from pathlib import Path
import sys
import pandas as pd

path = Path(sys.argv[1])
n = int(sys.argv[2])
if not path.exists():
    raise SystemExit(1)
df = pd.read_csv(path)
methods = {"raw_add", "raw_replace", "steepest_add", "steepest_replace"}
if set(df.get("method", [])) >= methods and df.get("dataset_index", pd.Series(dtype=int)).nunique() >= n:
    raise SystemExit(0)
raise SystemExit(1)
PY
}

run_burgers_if_needed() {
  local root="$OUT/raw_runs/burgers/burgers_loss3_core4_eps4_alpha0p4_steps100_N${BURGERS_N}"
  if burgers_complete; then
    log "skip Burgers core4: complete"
    return 0
  fi
  log "start Burgers core4 small rerun: eps=4 alpha=0.4 steps=100 N=${BURGERS_N}"
  gpu_snapshot
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 \
  "${PY}" -u tools/run_loss3_direction_proposal_ablation.py \
    --out-root "${root}" \
    --methods raw_add raw_replace steepest_add steepest_replace \
    --batch-size "${BURGERS_N}" \
    --start-index 0 \
    --epsilon 4 \
    --alpha 0.4 \
    --steps 100 \
    --p 2 \
    --q 2 \
    --model-kind fno \
    --burgers-nu 0.001 \
    --no-save-delta-trajectory \
    --no-plots \
    2>&1 | tee "${LOGDIR}/burgers_loss3_core4_eps4_alpha0p4_steps100_N${BURGERS_N}.log"
  log "done Burgers core4"
  summarize
}

darcy_complete() {
  local root="$OUT/raw_runs/darcy/darcy_cflow_binary_loss3_methods_epsflips437_alphaflips5_steps100_N${DARCY_N}_core4"
  "${PY}" - "$root" "$DARCY_N" <<'PY'
from pathlib import Path
import sys
import pandas as pd

root = Path(sys.argv[1])
n = int(sys.argv[2])
methods = {"raw_add", "raw_replace", "steepest_add", "steepest_replace"}
if not root.exists():
    raise SystemExit(1)
complete = set()
for method in methods:
    path = root / "loss3_methods" / "loss3" / method / "final_per_sample.csv"
    if not path.exists():
        continue
    df = pd.read_csv(path)
    if df.get("dataset_index", pd.Series(dtype=int)).nunique() >= n:
        complete.add(method)
raise SystemExit(0 if complete == methods else 1)
PY
}

run_darcy_if_needed() {
  local run_name="darcy_cflow_binary_loss3_methods_epsflips437_alphaflips5_steps100_N${DARCY_N}_core4"
  if darcy_complete; then
    log "skip Darcy/CFlow binary core4: complete"
    return 0
  fi
  log "start Darcy/CFlow binary core4 small rerun: K=437 alpha_flips=5 steps=100 N=${DARCY_N}"
  gpu_snapshot
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 \
  "${PY}" -u 2D_Darcy_FNO2d/perturbation_methods/attack_darcy_binary_loss_method_experiments.py \
    --dataset "${DARCY_DATASET}" \
    --checkpoint "${DARCY_CHECKPOINT}" \
    --output-root "${OUT}/raw_runs/darcy" \
    --run-name "${run_name}" \
    --experiment loss3_methods \
    --losses loss3 \
    --methods raw_add raw_replace steepest_add steepest_replace \
    --start 0 \
    --num-samples "${DARCY_N}" \
    --steps 100 \
    --metric rel_l2 \
    --epsilon-fraction 0.01 \
    --epsilon-flips 437 \
    --alpha-flips 5 \
    --trace-true-loss3-every 1 \
    --trace-sample-index -1 \
    2>&1 | tee "${LOGDIR}/${run_name}.log"
  log "done Darcy/CFlow binary core4"
  summarize
}

upload_r2() {
  if [[ ! -f "${RCLONE_CONFIG_FILE}" ]]; then
    log "R2 sync skipped: ${RCLONE_CONFIG_FILE} not found"
    return 0
  fi
  log "syncing optimizer ablation outputs to R2"
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
}

main() {
  exec >> "${QUEUE_LOG}" 2>&1
  echo "$$" > "${PID_FILE}"
  rm -f "${DONE_FILE}"
  log "optimizer ablation queue launched"
  summarize
  wait_for_current_queue

  run_ns_budget 1 0.3125
  run_ns_budget 8 2.5
  run_ns_budget 32 10
  run_burgers_if_needed
  run_darcy_if_needed

  summarize
  upload_r2
  printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${DONE_FILE}"
  log "optimizer ablation queue complete"
}

main "$@"
