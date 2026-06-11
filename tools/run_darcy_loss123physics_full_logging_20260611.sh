#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-adv_robust/bin/python}
EPOCHS=${EPOCHS:-100}
TAG=${TAG:-20260611}
GENERALIZATION_ROOT=${GENERALIZATION_ROOT:-generalization_datasets_darcy_binary_loss3targeted_20260611}
OUT_ROOT=${OUT_ROOT:-adversarial_training_runs}
TRAIN_MAX=${TRAIN_MAX:-64}
DARCY_BATCH=${DARCY_BATCH:-64}
OPT_BATCH=${OPT_BATCH:-32}
EVAL_MAX_SAMPLES=${EVAL_MAX_SAMPLES:-0}
MAX_GENERALIZATION_EVAL=${MAX_GENERALIZATION_EVAL:-50}
ATTACK_PROBE_SAMPLES=${ATTACK_PROBE_SAMPLES:-5}
ATTACK_PROBE_EVERY=${ATTACK_PROBE_EVERY:-1}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-200}
RUN_PLOT=${RUN_PLOT:-1}
DRY_RUN=${DRY_RUN:-0}

objectives=(loss1 loss2 loss3 physics)

for objective in "${objectives[@]}"; do
  run_name="darcy_binary_loss3targeted_${objective}_${EPOCHS}ep_full_logging_${TAG}"
  cmd=(
    "$PYTHON" tools/adversarial_training.py
    --tasks darcy
    --generalization-root "$GENERALIZATION_ROOT"
    --output-root "$OUT_ROOT"
    --run-name "$run_name"
    --epochs "$EPOCHS"
    --darcy-train-max "$TRAIN_MAX"
    --darcy-batch-size "$DARCY_BATCH"
    --darcy-optimizer-batch-size "$OPT_BATCH"
    --eval-max-samples "$EVAL_MAX_SAMPLES"
    --max-generalization-eval "$MAX_GENERALIZATION_EVAL"
    --training-data-mode adv-only
    --label-mode solver
    --checkpoint-every-epochs "$CHECKPOINT_EVERY"
    --darcy-attack-loss-objective "$objective"
    --attack-probe-samples "$ATTACK_PROBE_SAMPLES"
    --attack-probe-every-n-epochs "$ATTACK_PROBE_EVERY"
    --attack-probe-save-targets
  )
  printf '\n[darcy-full-logging] %s\n' "${cmd[*]}"
  if [[ "$DRY_RUN" != "1" ]]; then
    "${cmd[@]}"
  fi
done

if [[ "$RUN_PLOT" == "1" ]]; then
  plot_cmd=(
    "$PYTHON" tools/plot_darcy_loss123physics_adv_training_20260611.py
    --loss1-run-dir "$OUT_ROOT/darcy_binary_loss3targeted_loss1_${EPOCHS}ep_full_logging_${TAG}"
    --loss2-run-dir "$OUT_ROOT/darcy_binary_loss3targeted_loss2_${EPOCHS}ep_full_logging_${TAG}"
    --loss3-run-dir "$OUT_ROOT/darcy_binary_loss3targeted_loss3_${EPOCHS}ep_full_logging_${TAG}"
    --physics-run-dir "$OUT_ROOT/darcy_binary_loss3targeted_physics_${EPOCHS}ep_full_logging_${TAG}"
    --out-dir "visualizations/darcy_loss123physics_full_logging_${EPOCHS}ep_${TAG}"
    --report-md "docs/darcy_loss123physics_full_logging_${EPOCHS}ep_${TAG}.md"
    --derive-full50-from-run-eval
  )
  printf '\n[darcy-full-logging-plot] %s\n' "${plot_cmd[*]}"
  if [[ "$DRY_RUN" != "1" ]]; then
    "${plot_cmd[@]}"
  fi
fi
