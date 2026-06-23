#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/workspace/NeuralOperatorRobustness2}"
PY="${PY:-/venv/adv_robust/bin/python}"
OUT="${OUT:-${ROOT}/analysis_outputs/optimizer_ablation_20260622}"
LOGDIR="${OUT}/run_logs"
TARGET_N="${TARGET_N:-20}"
STEPS="${STEPS:-100}"
WAIT_FOR_EXISTING_QUEUE="${WAIT_FOR_EXISTING_QUEUE:-1}"

METHODS=(raw_add raw_replace steepest_add steepest_replace)

NS_EPS="${NS_EPS:-32}"
NS_ALPHA="${NS_ALPHA:-10}"
NS_BATCH="${NS_BATCH:-4}"
NS_CHECKPOINT="${NS_CHECKPOINT:-2D_NS_FNO2d_recurrent/saved_models/2D/modes64_modes64_width60_epochs500_Tin10_T10_recurrent_pytorch_20260521_090136_UTC/checkpoints/final.pt}"
NS_TEST_PATH="${NS_TEST_PATH:-2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/real_initial_laxmap_single/test/dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_test_ntimepoints21_all_frames.pt}"
NS_DICTIONARY_PATH="${NS_DICTIONARY_PATH:-2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dictionary/dim2d_nx256_N2000_solver=exponax_nu0.000_t20.0_dict_ntimepoints21_batch0_all_frames.pt}"

BURGERS_EPS="${BURGERS_EPS:-8}"
BURGERS_ALPHA="${BURGERS_ALPHA:-0.3}"

DARCY_EPS_FLIPS="${DARCY_EPS_FLIPS:-437}"
DARCY_ALPHA_FLIPS="${DARCY_ALPHA_FLIPS:-5}"
DARCY_DATASET="${DARCY_DATASET:-${ROOT}/2D_Darcy_FNO2d/datasets/grf_darcy_20260528_N1500/test/dim2d_darcy_nx211_N300_solver=jaxcg_solve421_alpha2_tau3_binary3-12_f1_seed45_test.pt}"
DARCY_CHECKPOINT="${DARCY_CHECKPOINT:-${ROOT}/2D_Darcy_FNO2d/saved_models/2D/darcy_N1500_nx85_m64_w60_e500_20260528/best.pt}"

QUEUE_LOG="${LOGDIR}/loss3_three_system_optimizer_curves_n20_20260622.log"
PID_FILE="${LOGDIR}/loss3_three_system_optimizer_curves_n20_20260622.pid"
DONE_FILE="${LOGDIR}/loss3_three_system_optimizer_curves_n20_20260622.done"
OLD_QUEUE_PID_FILE="${LOGDIR}/optimizer_ablation_queue_20260622.pid"
OLD_QUEUE_DONE_FILE="${LOGDIR}/optimizer_ablation_queue_20260622.done"

cd "${ROOT}"
mkdir -p "${LOGDIR}" "${OUT}/raw_runs/ns2d" "${OUT}/raw_runs/burgers" "${OUT}/raw_runs/darcy"

exec >> "${QUEUE_LOG}" 2>&1
echo "$$" > "${PID_FILE}"
rm -f "${DONE_FILE}"

log() {
  printf '[loss3-n20] %s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

trap 'rc=$?; log "ERROR rc=${rc} line=${LINENO}: ${BASH_COMMAND}"; exit "${rc}"' ERR

float_tag() {
  local value="$1"
  printf '%s' "${value//./p}"
}

gpu_snapshot() {
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits || true
}

refresh_outputs() {
  log "refreshing summary tables"
  "${PY}" tools/summarize_optimizer_ablation_20260622.py --analysis-root "${OUT}"
  log "refreshing three-panel mean-curve figure"
  "${PY}" tools/plot_loss3_three_system_optimizer_curves_20260622.py --analysis-root "${OUT}" --require-preferred
}

wait_for_old_queue_if_needed() {
  if [[ "${WAIT_FOR_EXISTING_QUEUE}" != "1" ]]; then
    log "not waiting for existing optimizer queue because WAIT_FOR_EXISTING_QUEUE=${WAIT_FOR_EXISTING_QUEUE}"
    return 0
  fi
  if [[ -f "${OLD_QUEUE_DONE_FILE}" ]]; then
    log "existing optimizer queue already marked done"
    return 0
  fi
  if [[ ! -f "${OLD_QUEUE_PID_FILE}" ]]; then
    log "no existing optimizer queue pid file found"
    return 0
  fi
  local old_pid
  old_pid="$(tr -dc '0-9' < "${OLD_QUEUE_PID_FILE}" || true)"
  if [[ -z "${old_pid}" ]] || ! kill -0 "${old_pid}" 2>/dev/null; then
    log "existing optimizer queue pid is not alive"
    return 0
  fi
  log "waiting for existing broad optimizer queue pid=${old_pid}; set WAIT_FOR_EXISTING_QUEUE=0 to run immediately"
  while kill -0 "${old_pid}" 2>/dev/null && [[ ! -f "${OLD_QUEUE_DONE_FILE}" ]]; do
    gpu_snapshot
    sleep 300
  done
  log "existing broad optimizer queue is no longer active; continuing with N=${TARGET_N} top-up"
}

missing_range() {
  local problem="$1"
  local eps="$2"
  local alpha="$3"
  local target="$4"
  local steps="$5"
  "${PY}" - "${OUT}/tables/raw_per_step.csv" "${problem}" "${eps}" "${alpha}" "${target}" "${steps}" "${METHODS[@]}" <<'PY'
from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

table = Path(sys.argv[1])
problem = sys.argv[2]
eps = float(sys.argv[3])
alpha = float(sys.argv[4])
target = int(sys.argv[5])
steps = int(sys.argv[6])
methods = sys.argv[7:]

if not table.exists() or table.stat().st_size == 0:
    print(f"0 {target}")
    raise SystemExit(0)

df = pd.read_csv(table)
if df.empty:
    print(f"0 {target}")
    raise SystemExit(0)

for col in ("epsilon", "alpha", "step"):
    df[col] = pd.to_numeric(df[col], errors="coerce")

sub = df[
    (df["problem"] == problem)
    & (df["epsilon"].sub(eps).abs() < 1e-9)
    & (df["alpha"].sub(alpha).abs() < 1e-9)
    & (df["step"] == steps)
    & (df["optimizer"].isin(methods))
].copy()

complete_ids: set[int] | None = None
for method in methods:
    ids = set(int(v) for v in pd.to_numeric(sub.loc[sub["optimizer"] == method, "sample_id"], errors="coerce").dropna())
    complete_ids = ids if complete_ids is None else complete_ids.intersection(ids)

complete_ids = complete_ids or set()
missing = [i for i in range(target) if i not in complete_ids]
if not missing:
    print(f"{target} 0")
    raise SystemExit(0)

start = min(missing)
count = target - start
print(f"{start} {count}")
PY
}

run_ns_if_needed() {
  refresh_outputs
  local start count
  read -r start count < <(missing_range "ns2d_recurrent" "${NS_EPS}" "${NS_ALPHA}" "${TARGET_N}" "${STEPS}")
  if (( count <= 0 )); then
    log "skip NS2D: already have N=${TARGET_N} complete samples for eps=${NS_EPS}, alpha=${NS_ALPHA}"
    return 0
  fi

  local eps_tag alpha_tag tag
  eps_tag="$(float_tag "${NS_EPS}")"
  alpha_tag="$(float_tag "${NS_ALPHA}")"
  tag="ns2d_loss3_core4_eps${eps_tag}_alpha${alpha_tag}_steps${STEPS}_N${TARGET_N}_start${start}_count${count}_batch${NS_BATCH}_chunkremat"
  log "start NS2D top-up: eps=${NS_EPS} alpha=${NS_ALPHA} start=${start} count=${count}"
  gpu_snapshot
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 \
  "${PY}" -u 2D_NS_FNO2d_recurrent/perturbation_methods/attack_ns2d_recurrent_core4.py \
    --out-root "${OUT}/raw_runs/ns2d/${tag}" \
    --checkpoint "${NS_CHECKPOINT}" \
    --test-path "${NS_TEST_PATH}" \
    --dictionary-path "${NS_DICTIONARY_PATH}" \
    --start-index "${start}" \
    --num-samples "${count}" \
    --attack-batch-size "${NS_BATCH}" \
    --methods "${METHODS[@]}" \
    --loss-types loss3 \
    --mode-spec all_w \
    --steps "${STEPS}" \
    --epsilon "${NS_EPS}" \
    --alpha "${NS_ALPHA}" \
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
    --clear-jax-caches-after-batch
  log "done NS2D top-up"
}

run_burgers_if_needed() {
  refresh_outputs
  local start count
  read -r start count < <(missing_range "burgers1d" "${BURGERS_EPS}" "${BURGERS_ALPHA}" "${TARGET_N}" "${STEPS}")
  if (( count <= 0 )); then
    log "skip Burgers: already have N=${TARGET_N} complete samples for eps=${BURGERS_EPS}, alpha=${BURGERS_ALPHA}"
    return 0
  fi

  local eps_tag alpha_tag tag
  eps_tag="$(float_tag "${BURGERS_EPS}")"
  alpha_tag="$(float_tag "${BURGERS_ALPHA}")"
  tag="burgers_loss3_core4_eps${eps_tag}_alpha${alpha_tag}_steps${STEPS}_N${TARGET_N}_start${start}_count${count}"
  log "start Burgers top-up: eps=${BURGERS_EPS} alpha=${BURGERS_ALPHA} start=${start} count=${count}"
  gpu_snapshot
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 \
  "${PY}" -u tools/run_loss3_direction_proposal_ablation.py \
    --out-root "${OUT}/raw_runs/burgers/${tag}" \
    --methods "${METHODS[@]}" \
    --batch-size "${count}" \
    --start-index "${start}" \
    --epsilon "${BURGERS_EPS}" \
    --alpha "${BURGERS_ALPHA}" \
    --steps "${STEPS}" \
    --p 2 \
    --q 2 \
    --model-kind fno \
    --burgers-nu 0.001 \
    --no-save-delta-trajectory \
    --no-plots
  log "done Burgers top-up"
}

run_darcy_if_needed() {
  refresh_outputs
  local start count
  read -r start count < <(missing_range "darcy_cflow_binary" "${DARCY_EPS_FLIPS}" "${DARCY_ALPHA_FLIPS}" "${TARGET_N}" "${STEPS}")
  if (( count <= 0 )); then
    log "skip Darcy/CFlow: already have N=${TARGET_N} complete samples for eps_flips=${DARCY_EPS_FLIPS}, alpha_flips=${DARCY_ALPHA_FLIPS}"
    return 0
  fi

  local run_name
  run_name="darcy_cflow_binary_loss3_methods_epsflips${DARCY_EPS_FLIPS}_alphaflips${DARCY_ALPHA_FLIPS}_steps${STEPS}_N${TARGET_N}_start${start}_count${count}_core4"
  log "start Darcy/CFlow top-up: eps_flips=${DARCY_EPS_FLIPS} alpha_flips=${DARCY_ALPHA_FLIPS} start=${start} count=${count}"
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
    --methods "${METHODS[@]}" \
    --start "${start}" \
    --num-samples "${count}" \
    --steps "${STEPS}" \
    --metric rel_l2 \
    --epsilon-fraction 0.01 \
    --epsilon-flips "${DARCY_EPS_FLIPS}" \
    --alpha-flips "${DARCY_ALPHA_FLIPS}" \
    --trace-true-loss3-every 1 \
    --trace-sample-index -1
  log "done Darcy/CFlow top-up"
}

main() {
  log "launched N=${TARGET_N} loss3 three-system optimizer top-up"
  wait_for_old_queue_if_needed
  run_ns_if_needed
  run_burgers_if_needed
  run_darcy_if_needed
  refresh_outputs
  printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${DONE_FILE}"
  log "complete"
}

main "$@"
