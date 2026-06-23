#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/NeuralOperatorRobustness2"
PY="/venv/adv_robust/bin/python"
OUT="${ROOT}/analysis_outputs/attack_objective_true_loss3_comparison_20260622"
LOGDIR="${OUT}/run_logs"

cd "${ROOT}"
mkdir -p "${LOGDIR}" "${OUT}/raw_runs/darcy" "${OUT}/raw_runs/ns2d"

BURGERS_WAIT_PID="${BURGERS_WAIT_PID:-19781}"
NS_WAIT_PID="${NS_WAIT_PID:-14967}"

wait_for_pid() {
  local pid="$1"
  local label="$2"
  if [[ -z "${pid}" ]]; then
    return 0
  fi
  if ! kill -0 "${pid}" 2>/dev/null; then
    echo "[queue] ${label} pid ${pid} is not running; continuing $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    return 0
  fi
  echo "[queue] waiting for ${label} pid ${pid} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  while kill -0 "${pid}" 2>/dev/null; do
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits || true
    sleep 30
  done
  echo "[queue] ${label} pid ${pid} finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

summarize() {
  "${PY}" tools/summarize_attack_objective_true_loss3.py --analysis-root "${OUT}" \
    | tee -a "${LOGDIR}/extra_sweeps_summary.log"
}

run_darcy_budget() {
  local flips="$1"
  local tag="darcy_cflow_epsflips${flips}_alphaflips5_steps100_N20_loss123_steepest_replace"
  echo "[darcy sweep] start ${tag} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.20 \
  "${PY}" -u 2D_Darcy_FNO2d/perturbation_methods/attack_darcy_binary_loss_method_experiments.py \
    --num-samples 20 \
    --steps 100 \
    --experiment loss_objectives \
    --losses loss1 loss2 loss3 \
    --loss-objective-method steepest_replace \
    --epsilon-flips "${flips}" \
    --alpha-flips 5 \
    --trace-true-loss3-every 10 \
    --trace-sample-index -1 \
    --plot-samples 0 \
    --output-root "${OUT}/raw_runs/darcy" \
    --run-name "${tag}" \
    2>&1 | tee "${LOGDIR}/${tag}.log"
  echo "[darcy sweep] done ${tag} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

ns_base_args() {
  printf '%s\n' \
    --checkpoint \
    2D_NS_FNO2d_recurrent/saved_models/2D/modes64_modes64_width60_epochs500_Tin10_T10_recurrent_pytorch_20260521_090136_UTC/checkpoints/final.pt \
    --test-path \
    '2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/real_initial_laxmap_single/test/dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_test_ntimepoints21_all_frames.pt' \
    --dictionary-path \
    '2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dictionary/dim2d_nx256_N2000_solver=exponax_nu0.000_t20.0_dict_ntimepoints21_batch0_all_frames.pt' \
    --start-index 0 \
    --num-samples 4 \
    --attack-batch-size 4 \
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
    --no-empty-torch-cache-after-method
}

run_ns_budget() {
  local eps="$1"
  local alpha="$2"
  local eps_tag="${eps//./p}"
  local alpha_tag="${alpha//./p}"

  local tag_allw="ns2d_probe_allw_loss1_loss3_eps${eps_tag}_alpha${alpha_tag}_N4_steps50_batch4_chunkremat"
  echo "[ns sweep] start ${tag_allw} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 \
  "${PY}" -u 2D_NS_FNO2d_recurrent/perturbation_methods/attack_ns2d_recurrent_core4.py \
    --out-root "${OUT}/raw_runs/ns2d/${tag_allw}" \
    $(ns_base_args) \
    --loss-types loss1 loss3 \
    --mode-spec all_w \
    --epsilon "${eps}" \
    --alpha "${alpha}" \
    2>&1 | tee "${LOGDIR}/${tag_allw}.log"
  echo "[ns sweep] done ${tag_allw} $(date -u +%Y-%m-%dT%H:%M:%SZ)"

  local tag_loss2="ns2d_probe_loss2_all_a_target_w_eps${eps_tag}_alpha${alpha_tag}_N4_steps50_batch4_chunkremat"
  echo "[ns sweep] start ${tag_loss2} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 \
  "${PY}" -u 2D_NS_FNO2d_recurrent/perturbation_methods/attack_ns2d_recurrent_core4.py \
    --out-root "${OUT}/raw_runs/ns2d/${tag_loss2}" \
    $(ns_base_args) \
    --loss-types loss2 \
    --mode-spec all_a_target_w \
    --epsilon "${eps}" \
    --alpha "${alpha}" \
    2>&1 | tee "${LOGDIR}/${tag_loss2}.log"
  echo "[ns sweep] done ${tag_loss2} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

echo "[queue] extra sweeps launched $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[queue] Darcy/CFlow: N=20, steps=100, flips={100,250,437,875}, alpha_flips=5, objectives=loss1/loss2/loss3"
echo "[queue] NS2D: N=4, steps=50, pairs={8:0.25,16:0.5,32:1,64:2}, batch=4, solver_remat=chunk"

wait_for_pid "${BURGERS_WAIT_PID}" "current Burgers sweep"

for flips in 100 250 437 875; do
  run_darcy_budget "${flips}"
  summarize
done

wait_for_pid "${NS_WAIT_PID}" "current NS2D run"

run_ns_budget 8 0.25
summarize
run_ns_budget 16 0.5
summarize
run_ns_budget 32 1
summarize
run_ns_budget 64 2
summarize

echo "[queue] extra sweeps complete $(date -u +%Y-%m-%dT%H:%M:%SZ)"
