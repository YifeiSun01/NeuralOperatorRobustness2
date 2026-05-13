#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-adv_robust/bin/python}"
CASE="${CASE:-burgers}"
NORM="${NORM:-2}"
INPUT_P="${INPUT_P:-$NORM}"
OUTPUT_Q="${OUTPUT_Q:-2}"
EPSILON="${EPSILON:-0.01}"
ALPHA="${ALPHA:-0.005}"
STEPS="${STEPS:-100}"
INDEX="${INDEX:-0}"
SEED="${SEED:-0}"
RATIO_DENOMINATOR_EPSILON="${RATIO_DENOMINATOR_EPSILON:-1e-6}"
REGULARIZATION_C="${REGULARIZATION_C:-1.0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/three_loss_objective_round1}"
POWER_RADIUS_MODE="${POWER_RADIUS_MODE:-alpha_schedule}"
SAVE_TRAJECTORY="${SAVE_TRAJECTORY:-0}"

losses=(loss1 loss2 loss3)
objectives=(original increment_ratio regularized)
methods=(pgd lp_steepest_pgd power_iteration)

for loss_type in "${losses[@]}"; do
  for objective in "${objectives[@]}"; do
    for method in "${methods[@]}"; do
      out_dir="${OUTPUT_ROOT}/${CASE}_${loss_type}_${objective}_${method}_p${INPUT_P}_q${OUTPUT_Q}_eta${RATIO_DENOMINATOR_EPSILON}_C${REGULARIZATION_C}_eps${EPSILON}_alpha${ALPHA}_steps${STEPS}_idx${INDEX}_seed${SEED}"
      extra=()
      if [[ "${objective}" == "increment_ratio" && ( "${method}" == "pgd" || "${method}" == "lp_steepest_pgd" ) ]]; then
        extra+=(--random_start)
      fi
      if [[ "${objective}" == "regularized" && ( "${method}" == "pgd" || "${method}" == "lp_steepest_pgd" ) ]]; then
        extra+=(--random_start)
      fi
      if [[ "${method}" == "power_iteration" ]]; then
        extra+=(--power_variant objective_gradient)
        if [[ "${objective}" != "original" ]]; then
          extra+=(--power_min_radius "${ALPHA}")
        fi
      fi
      if [[ "${SAVE_TRAJECTORY}" == "1" ]]; then
        extra+=(--save_trajectory)
      fi
      "${PYTHON_BIN}" run_three_loss_objective_attack.py \
        --case "${CASE}" \
        --solver_backend jax \
        --model_backend torch \
        --loss_type "${loss_type}" \
        --objective_variant "${objective}" \
        --attack_method "${method}" \
        --norm "${NORM}" \
        --input_p "${INPUT_P}" \
        --output_q "${OUTPUT_Q}" \
        --epsilon "${EPSILON}" \
        --alpha "${ALPHA}" \
        --steps "${STEPS}" \
        --index "${INDEX}" \
        --seed "${SEED}" \
        --ratio_denominator_epsilon "${RATIO_DENOMINATOR_EPSILON}" \
        --regularization_c "${REGULARIZATION_C}" \
        --power_radius_mode "${POWER_RADIUS_MODE}" \
        --output_dir "${out_dir}" \
        --no-progress \
        "${extra[@]}"
    done
  done
done
