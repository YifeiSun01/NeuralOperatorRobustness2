# NS2D Core4 Attack Loss/Mode Mapping Correction

Date: 2026-05-21

This note corrects the earlier overly broad Cartesian-product command. No GPU run was launched for this note.

## Key Correction

`loss_type` and `MODE_SPEC` should not be treated as two fully independent axes.

Conceptual mapping:

- `loss1`: model-vs-model objective. It does not need the ADW target-mode sweep. Run it once with the canonical solver/model input mode, normally `all_w`.
- `loss2`: fixed/approximate target objective. The pure dictionary/no-solver style mode belongs here, e.g. `all_a_target_w` under the current preset naming.
- `loss3`: perturbed-target objective. W/D/A mixed modes belong here because they control how the perturbed target/solver path participates in forward and backward.

The earlier full Cartesian product:

```text
6 mode presets * 3 losses * 4 methods = 72 combinations
```

is conceptually too broad because it repeats `loss1` under modes that do not define separate loss1 experiments.

## Recommended Grouping

Run these groups instead:

```text
loss1 group:
  MODE_SPEC=all_w
  LOSS_TYPES=loss1

loss2 group:
  MODE_SPEC=all_a_target_w
  LOSS_TYPES=loss2

loss3 groups:
  MODE_SPEC=all_w
  MODE_SPEC=all_d_target_w
  MODE_SPEC=w1_5_d6_9_target_w
  MODE_SPEC=d1_5_w6_9_target_w
  MODE_SPEC=a1_5_d6_9_target_w
  LOSS_TYPES=loss3
```

With four core methods, this is:

```text
(1 loss1 group + 1 loss2 group + 5 loss3 groups) * 4 methods = 28 combinations
```

not 72 combinations.

## Corrected Full Command

This command uses the high-throughput setting `SOLVER_REMAT=chunk`, `SOLVER_REMAT_CHUNK_STEPS=20`, `ATTACK_BATCH_SIZE=17`.

Modes containing `a` require the dictionary file. If the dictionary is missing, this command exits before running.

```bash
cd /workspace/NeuralOperatorRobustness2

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.40

CHECKPOINT=2D_NS_FNO2d_recurrent/saved_models/2D/modes64_modes64_width60_epochs500_Tin10_T10_recurrent_pytorch_20260521_090136_UTC/checkpoints/final.pt
DICT=2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dictionary/dim2d_nx256_N2000_solver=exponax_nu0.000_t20.0_dict_ntimepoints21_batch0_all_frames.pt

if [ ! -f "$DICT" ]; then
  echo "Missing dictionary for A-mode groups: $DICT" >&2
  exit 1
fi

run_attack_group () {
  local mode="$1"
  local loss="$2"
  echo "=== Running LOSS_TYPES=$loss MODE_SPEC=$mode ==="
  adv_robust/bin/python 2D_NS_FNO2d_recurrent/perturbation_methods/attack_ns2d_recurrent_core4.py \
    --checkpoint "$CHECKPOINT" \
    --dictionary-path "$DICT" \
    --indices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 \
    --attack-batch-size 17 \
    --loss-types "$loss" \
    --methods raw_add raw_replace steepest_add steepest_replace \
    --mode-spec "$mode" \
    --steps 100 \
    --epsilon 32 \
    --alpha 1 \
    --p 2 \
    --q 2 \
    --solver-remat chunk \
    --solver-remat-chunk-steps 20 \
    --dictionary-chunk-size 32 \
    --empty-torch-cache-after-batch
}

# loss1: no ADW mode sweep
run_attack_group all_w loss1

# loss2: fixed/approximate-target style
run_attack_group all_a_target_w loss2

# loss3: W/D/A solver-target variants
for MODE in \
  all_w \
  all_d_target_w \
  w1_5_d6_9_target_w \
  d1_5_w6_9_target_w \
  a1_5_d6_9_target_w
do
  run_attack_group "$MODE" loss3
done
```

## W/D-Only Fallback If Dictionary Is Missing

If the dictionary is not available yet, do not run A-mode groups. This runs only `loss1` and W/D `loss3` variants:

```bash
cd /workspace/NeuralOperatorRobustness2

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.40

CHECKPOINT=2D_NS_FNO2d_recurrent/saved_models/2D/modes64_modes64_width60_epochs500_Tin10_T10_recurrent_pytorch_20260521_090136_UTC/checkpoints/final.pt

run_attack_group () {
  local mode="$1"
  local loss="$2"
  echo "=== Running LOSS_TYPES=$loss MODE_SPEC=$mode ==="
  adv_robust/bin/python 2D_NS_FNO2d_recurrent/perturbation_methods/attack_ns2d_recurrent_core4.py \
    --checkpoint "$CHECKPOINT" \
    --indices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 \
    --attack-batch-size 17 \
    --loss-types "$loss" \
    --methods raw_add raw_replace steepest_add steepest_replace \
    --mode-spec "$mode" \
    --steps 100 \
    --epsilon 32 \
    --alpha 1 \
    --p 2 \
    --q 2 \
    --solver-remat chunk \
    --solver-remat-chunk-steps 20 \
    --empty-torch-cache-after-batch
}

run_attack_group all_w loss1

for MODE in \
  all_w \
  all_d_target_w \
  w1_5_d6_9_target_w \
  d1_5_w6_9_target_w
do
  run_attack_group "$MODE" loss3
done
```

## Estimated Scale

Corrected full grouping:

```text
28 combinations * ~32.4 minutes per expensive all-W-like combination ~= 15.1 hours upper rough estimate
```

This is much smaller than the incorrect 72-combination estimate. Actual timing may differ because A and D variants can change solver/dictionary/backward cost.


Note: current attack logging records loss/delta curves in CSV and saves final delta automatically. Do not pass `--save-steps` unless intermediate trajectory arrays are explicitly needed.
