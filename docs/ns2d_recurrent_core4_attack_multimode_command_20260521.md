# NS2D Core4 Attack Multi-Mode Command

Date: 2026-05-21

This note records the recommended multi-mode attack command. No GPU run was launched for this note.

Recommended high-throughput settings from existing benchmark records:

```text
SOLVER_REMAT=chunk
SOLVER_REMAT_CHUNK_STEPS=20
ATTACK_BATCH_SIZE=17
```

The CLI currently accepts one `--mode-spec` per run, so multiple ADW presets should be run with a shell loop. The full preset set requested here is:

```text
all_w                    -> wwwwwwwwww
all_d_target_w            -> dddddddddw
all_a_target_w            -> aaaaaaaaaw
w1_5_d6_9_target_w        -> wwwwwddddw
d1_5_w6_9_target_w        -> dddddwwwww
a1_5_d6_9_target_w        -> aaaaaddddw
```

Modes containing `a` require the default dictionary file:

```text
2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dictionary/dim2d_nx256_N2000_solver=exponax_nu0.000_t20.0_dict_ntimepoints21_batch0_all_frames.pt
```

At the time this command was written, that dictionary file was missing locally, so the full ADW command includes a guard that fails early if it is not present.

## Full ADW Command

```bash
cd /workspace/NeuralOperatorRobustness2

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.40

CHECKPOINT=2D_NS_FNO2d_recurrent/saved_models/2D/modes64_modes64_width60_epochs500_Tin10_T10_recurrent_pytorch_20260521_090136_UTC/checkpoints/final.pt
DICT=2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dictionary/dim2d_nx256_N2000_solver=exponax_nu0.000_t20.0_dict_ntimepoints21_batch0_all_frames.pt

if [ ! -f "$DICT" ]; then
  echo "Missing dictionary for A modes: $DICT" >&2
  exit 1
fi

for MODE in \
  all_w \
  all_d_target_w \
  all_a_target_w \
  w1_5_d6_9_target_w \
  d1_5_w6_9_target_w \
  a1_5_d6_9_target_w
do
  echo "=== Running MODE_SPEC=$MODE ==="
  adv_robust/bin/python 2D_NS_FNO2d_recurrent/perturbation_methods/attack_ns2d_recurrent_core4.py \
    --checkpoint "$CHECKPOINT" \
    --dictionary-path "$DICT" \
    --indices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 \
    --attack-batch-size 17 \
    --loss-types loss1 loss2 loss3 \
    --methods raw_add raw_replace steepest_add steepest_replace \
    --mode-spec "$MODE" \
    --steps 100 \
    --epsilon 32 \
    --alpha 1 \
    --p 2 \
    --q 2 \
    --true-loss-every 1 \
    --solver-remat chunk \
    --solver-remat-chunk-steps 20 \
    --dictionary-chunk-size 32 \
    --empty-torch-cache-after-batch
done
```

## W/D-Only Command If Dictionary Is Still Missing

```bash
cd /workspace/NeuralOperatorRobustness2

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.40

CHECKPOINT=2D_NS_FNO2d_recurrent/saved_models/2D/modes64_modes64_width60_epochs500_Tin10_T10_recurrent_pytorch_20260521_090136_UTC/checkpoints/final.pt

for MODE in \
  all_w \
  all_d_target_w \
  w1_5_d6_9_target_w \
  d1_5_w6_9_target_w
do
  echo "=== Running MODE_SPEC=$MODE ==="
  adv_robust/bin/python 2D_NS_FNO2d_recurrent/perturbation_methods/attack_ns2d_recurrent_core4.py \
    --checkpoint "$CHECKPOINT" \
    --indices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 \
    --attack-batch-size 17 \
    --loss-types loss1 loss2 loss3 \
    --methods raw_add raw_replace steepest_add steepest_replace \
    --mode-spec "$MODE" \
    --steps 100 \
    --epsilon 32 \
    --alpha 1 \
    --p 2 \
    --q 2 \
    --true-loss-every 1 \
    --solver-remat chunk \
    --solver-remat-chunk-steps 20 \
    --empty-torch-cache-after-batch
done
```

## Expected Scale

Full ADW command runs:

```text
6 mode presets * 3 losses * 4 methods = 72 sequential combinations
```

Using the measured all-W expensive benchmark as a rough upper estimate:

```text
one combination: about 32.4 minutes for 100 steps and 17 samples
72 combinations: about 38.9 hours if all combinations cost the same
```

Actual A/D combinations may differ because A uses dictionary approximation and D detaches solver gradients for selected frames.

## Correction

The full Cartesian product command above is broader than the intended experiment design. `loss1` should not be swept across ADW modes. Use `docs/ns2d_recurrent_core4_attack_loss_mode_mapping_20260521.md` for the corrected grouped command: loss1 once, loss2 for the pure A/fixed-target style group, and loss3 for W/D/A target-path variants.


Note: current attack logging records loss/delta curves in CSV and saves final delta automatically. Do not pass `--save-steps` unless intermediate trajectory arrays are explicitly needed.


Note: `--true-loss-every 1` records the full all-W solver true-loss curve at every attack step, alongside the surrogate loss being optimized.
