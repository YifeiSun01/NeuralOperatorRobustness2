# Runtime Blow-Up Diagnosis For Adversarial Training

Date: 2026-05-30 UTC

This note answers why the runtime appeared to jump from “tens of minutes” to roughly 7 hours for Burgers/Darcy and much longer for NS2D.

## Short Answer

The 7-hour number is not one epoch and not one batch. It is a 500-epoch full-training estimate.

For the corrected 10-step full-solver attack policy, the measured per-epoch cost is only about one minute for Burgers and Darcy/C-flow:

| task | corrected setting | full train samples | batch | batches/epoch | measured step sec/batch | sec/epoch | 20 epochs | 500 epochs |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Burgers | 10-step, solver label, chunk remat | 1350 | 1350 | 1 | 54.498 | 54.498 s | 18.17 min | 7.57 h |
| Darcy/C-flow | 10-step, solver label | 1200 | 448 | 3 | 16.594 | 49.781 s | 16.59 min | 6.91 h |
| NS2D | 10-step, solver label, chunk remat | 50 | 6 | 9 | 434.494 | 3910.445 s | 21.72 h | 543.12 h |

So the reason Burgers/Darcy can look like “十几分钟” in a short run is simple: 20 epochs is about 16-18 minutes. The 7-hour estimate is the same per-epoch speed multiplied to 500 epochs.

## Code Path Checked

The current defaults in `tools/adversarial_training.py` set 500 epochs unless `--epochs` overrides it:

- Burgers default: `epochs=500`, default `attack_steps=3`.
- Darcy/C-flow default: `epochs=500`, default `attack_steps=1`.
- NS2D default: `epochs=500`, default `attack_steps=5`.

Code reference: `tools/adversarial_training.py:72-120`.

The smoke path is a completely different runtime regime:

- `--smoke` forces `epochs=1`.
- It uses tiny `train_max_samples`.
- It sets `max_batches_per_epoch=1`.
- It clips attack steps to at most 2.

Code reference: `tools/adversarial_training.py:1308-1318`.

The formal runtime multiplier is computed here:

```text
batches_per_epoch = ceil(n_train / batch_size), optionally capped by max_batches_per_epoch
total_steps = epochs * batches_per_epoch
```

Code reference: `tools/adversarial_training.py:1361-1374`.

Each training step runs attack first, then optimizer microbatches:

```text
attack_result = attack_batch(...)
combine_training_pairs(...)
for each optimizer microbatch:
    pred = model(x_micro)
    loss.backward()
    optimizer.step()
```

Code reference: `tools/adversarial_training.py:1419-1508`.

## What Changed Versus The Short Runs

The earlier large-batch run `full_adversarial_training_20260530_v3_large_batches` was not comparable to the corrected production estimate:

| task | old run samples | old batch | old epochs | old total attack batches | old attack steps | old elapsed |
|---|---:|---:|---:|---:|---:|---:|
| Burgers | 200 | 100 | 5 | 10 | 3 | 2.76 s |
| Darcy/C-flow | 200 | 50 | 5 | 20 | 1 | 28.28 s |
| NS2D | 50 | 5 | 5 | 50 | 5 | recorded separately in that run |

The corrected production estimate is:

| task | corrected full samples | corrected batch | epochs | total attack batches | attack steps |
|---|---:|---:|---:|---:|---:|
| Burgers | 1350 | 1350 | 500 | 500 | 10 |
| Darcy/C-flow | 1200 | 448 | 500 | 1500 | 10 |
| NS2D | 50 | 6 | 500 | 4500 | 10 |

Just the number of attack batches increased by:

| task | old attack batches | corrected attack batches | multiplier |
|---|---:|---:|---:|
| Burgers | 10 | 500 | 50.0x |
| Darcy/C-flow | 20 | 1500 | 75.0x |

Then the attack itself also changed. It is now the corrected full-solver 10-step attack instead of the earlier short/cheap settings.

Fairer same-batch comparisons:

| task | comparison | old attack sec | new attack sec | multiplier |
|---|---|---:|---:|---:|
| Burgers | batch 768, 3-step -> 10-step | 17.941 | 50.119 | 2.79x |
| Darcy/C-flow | batch 384, 1-step -> 10-step | 4.174 | 12.773 | 3.06x |

The multiplier is not exactly 10x because fixed overheads, batching, solver/JAX behavior, and memory reuse amortize part of the extra steps. But it is still several times slower per attack batch.

## Why The Old Smoke/Planned Estimate Was Misleading

Some old planned estimates were built from an old smoke summary. Example: the Burgers `remat_sweep_burgers_chunk_bs1350_steps10` plan estimated about 10 seconds total from an old smoke reference, but the real measured one-batch run took about 54.5 seconds.

That old smoke reference was stale because it did not represent the corrected large-batch, 10-step, full-solver, checkpointed setting. For runtime planning, use the corrected measured summaries, not the earlier smoke-based estimate.

Corrected measured summaries:

- Burgers: `adversarial_training_runs/remat_sweep_burgers_chunk_bs1350_steps10/burgers/summary.json`
- Darcy/C-flow: `adversarial_training_runs/remat_sweep_darcy_bs448_steps10/darcy/summary.json`
- NS2D: `adversarial_training_runs/remat_sweep_ns2d_chunk_bs6_steps10_alpha0p2/ns2d/summary.json`

## Is Eval The Cause?

No. The 7-hour Burgers/Darcy estimate is dominated by training attack steps, not eval.

The corrected runtime document assumed 6 eval passes total: baseline plus 5 scheduled evaluations. Eval is clean model evaluation over train/test/generalization datasets. It does not regenerate 10-step adversarial examples. Its estimated contribution is tiny compared with attack training.

Main multiplier:

```text
formal runtime ~= measured corrected step time * epochs * batches_per_epoch
```

For Burgers:

```text
54.498 sec/batch * 1 batch/epoch * 500 epochs = 27,249 sec = 7.57 h
```

For Darcy/C-flow:

```text
16.594 sec/batch * 3 batches/epoch * 500 epochs = 24,890 sec = 6.91 h
```

For NS2D:

```text
434.494 sec/batch * 9 batches/epoch * 500 epochs = 1,955,222 sec = 543.12 h
```

## Conclusion

The apparent jump is explained by a change in experimental口径, not by a mysterious slowdown:

1. Earlier runs were small: often 1-5 epochs, capped samples, capped batches, or smoke/probe settings.
2. Current estimate is 500 epochs.
3. Current estimate uses full train coverage per epoch.
4. Current attack policy is corrected to 10 steps.
5. Current labels are full-solver labels, so attack and training regenerate solver-consistent targets.
6. Checkpoint/rematerialization keeps memory under control but adds recomputation cost.
7. Old smoke-based estimates are not reliable for the corrected large-batch 10-step setting.

For Burgers and Darcy/C-flow, a 10-20 epoch pilot should indeed finish around the “十几分钟” range. The 7-hour number only appears when scaling the same per-epoch cost to 500 epochs.
