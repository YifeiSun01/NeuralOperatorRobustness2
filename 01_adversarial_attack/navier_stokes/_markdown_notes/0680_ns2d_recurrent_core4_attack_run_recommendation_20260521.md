# NS2D Core4 Attack Run Recommendation

Date: 2026-05-21

This recommendation is based on existing benchmark records only. No new GPU run was launched for this note.

## Objective

Maximize useful attack sample-updates per unit time while avoiding OOM during a long run.

The relevant metric is:

```text
throughput = attack_batch_size / warm_update_seconds
```

Do not choose only by seconds per update. `none` is fastest per update but can only run batch 2, so its useful throughput is poor.

## Observed Throughput Ranking

| Rank | Mode | Batch | Warm update time | Throughput | Peak reserved | Risk |
|---:|---|---:|---:|---:|---:|---|
| 1 | `chunk=20` | `17` | `19.41s/update` | `0.876 sample-updates/s` | `73.34 GiB / 78.74 GB` | Highest throughput, tight memory headroom. |
| 2 | `second` | `15` | `18.68s/update` | `0.803 sample-updates/s` | `64.89 GiB / 69.67 GB` | Good middle-ground. |
| 3 | `micro/original` | `12` | `15.01s/update` | `0.799 sample-updates/s` | `52.02 GiB / 55.86 GB` at batch 12 | Safest recommendation. |
| 4 | `none` | `2` | `7.76s/update` | `0.258 sample-updates/s` | `10.42 GiB / 11.19 GB` | Not useful for full sweeps. |

`chunk=100` has memory probes at batch 12 and 15 but no 5-step steady timing probe, so it is not recommended yet for maximum-throughput runs.

## Recommended Parameters

### Maximum Throughput Observed

Use this when the GPU is otherwise free and you are willing to monitor memory:

```text
SOLVER_REMAT=chunk
SOLVER_REMAT_CHUNK_STEPS=20
ATTACK_BATCH_SIZE=17
```

This is the best observed sample throughput. It is close to the A100 80GB memory ceiling, so avoid running other GPU work at the same time.

### Safer Long Sweep

Use this when you care more about not losing a long sweep:

```text
SOLVER_REMAT=micro
ATTACK_BATCH_SIZE=12
```

This has slightly lower throughput than `chunk=20 batch17`, but much more memory headroom.

### Middle Ground

Use this if you want larger batch than micro but less memory pressure than chunk20 batch17:

```text
SOLVER_REMAT=second
ATTACK_BATCH_SIZE=15
```

Its throughput is roughly tied with `micro batch12`, but with larger batch and less headroom than micro.

## How Many Samples To Run Per Group

If using `ATTACK_BATCH_SIZE=17`, choose sample counts that are multiples of 17 when possible. If the experiment only needs around 5 samples, run 17 only if you are okay with extra samples; otherwise use a smaller explicit `INDICES` list and accept under-utilization.

Recommended practical sample counts:

```text
quick sanity: 17 samples, 5-10 steps
small comparison: 17 samples per loss/method/mode, 50-100 steps
stronger comparison: 34 samples per loss/method/mode, 50-100 steps
```

For comparing multiple losses and methods, run combinations sequentially rather than simultaneously. The attack script already runs loss/method combinations sequentially, so `ATTACK_BATCH_SIZE=17` means 17 initial conditions for one active combination at a time, not all combinations in memory at once.

## Command Template: Highest Throughput Observed

```bash
cd /workspace/NeuralOperatorRobustness2

PYTHON_BIN=adv_robust/bin/python CHECKPOINT=2D_NS_FNO2d_recurrent/saved_models/2D/modes64_modes64_width60_epochs500_Tin10_T10_recurrent_pytorch_20260521_090136_UTC/checkpoints/final.pt INDICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 ATTACK_BATCH_SIZE=17 LOSS_TYPES=loss1,loss2,loss3 METHODS=raw_add,raw_replace,steepest_add,steepest_replace MODE_SPEC=all_w STEPS=100 EPSILON=32 ALPHA=1 P_ORDER=2 Q_ORDER=2 SAVE_STEPS=0,1,5,10,25,50,100 SOLVER_REMAT=chunk SOLVER_REMAT_CHUNK_STEPS=20 EMPTY_TORCH_CACHE_AFTER_BATCH=1 CLEAR_JAX_CACHES_AFTER_BATCH=0 ./2D_NS_FNO2d_recurrent/perturbation_methods/run_ns2d_recurrent_core4_attack.sh
```

## Command Template: Safer Long Sweep

```bash
cd /workspace/NeuralOperatorRobustness2

PYTHON_BIN=adv_robust/bin/python CHECKPOINT=2D_NS_FNO2d_recurrent/saved_models/2D/modes64_modes64_width60_epochs500_Tin10_T10_recurrent_pytorch_20260521_090136_UTC/checkpoints/final.pt INDICES=0,1,2,3,4,5,6,7,8,9,10,11 ATTACK_BATCH_SIZE=12 LOSS_TYPES=loss1,loss2,loss3 METHODS=raw_add,raw_replace,steepest_add,steepest_replace MODE_SPEC=all_w STEPS=100 EPSILON=32 ALPHA=1 P_ORDER=2 Q_ORDER=2 SAVE_STEPS=0,1,5,10,25,50,100 SOLVER_REMAT=micro EMPTY_TORCH_CACHE_AFTER_BATCH=1 CLEAR_JAX_CACHES_AFTER_BATCH=0 ./2D_NS_FNO2d_recurrent/perturbation_methods/run_ns2d_recurrent_core4_attack.sh
```

## Time Estimate

For the highest-throughput observed setting:

```text
chunk=20, batch=17, 100 steps, one loss/method/mode combination
~= 100 * 19.41s = 1941s = 32.4 minutes
```

If running 3 losses * 4 methods * 1 mode preset sequentially:

```text
12 combinations * 32.4 min ~= 6.5 hours for 17 samples
```

This estimate excludes extra save/compression overhead and assumes behavior stays close to the measured `loss3/raw_add/all_w` benchmark. Other loss/method/mode combinations may differ.

## Final Recommendation

Use `chunk=20`, `ATTACK_BATCH_SIZE=17` for the maximum observed unit-time sample throughput. Use `micro`, `ATTACK_BATCH_SIZE=12` if the run is very long and you want more memory safety.


Note: current attack logging records loss/delta curves in CSV and saves final delta automatically. Do not pass `--save-steps` unless intermediate trajectory arrays are explicitly needed.
