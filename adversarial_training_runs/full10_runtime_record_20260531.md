# Full10 adversarial training runtime record, 2026-05-31

This note records the observed and estimated runtime for the two adversarial
training modes:

- `adv-only`
- `clean-plus-adv`

## Burgers, completed runs

| Task | Run | Mode | Status | Epochs | Steps | Batch | Optimizer batch | Elapsed |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Burgers | `full10_burgers_adv_only_20260530` | `adv-only` | completed | 500 | 500 | 1350 | 32 | 27019.6 s = 450.3 min = 7.51 h |
| Burgers | `full10_burgers_clean_plus_adv_20260530` | `clean-plus-adv` | completed | 500 | 500 | 1350 | 32 | 27378.0 s = 456.3 min = 7.61 h |

Observed mean step time:

- Burgers `adv-only`: 54.03 s / training step.
- Burgers `clean-plus-adv`: 54.75 s / training step.

The clean evaluation over 52 datasets is cheap for Burgers, about 0.24 to
0.26 seconds per scheduled evaluation.

## Darcy, original failed runs

| Task | Run | Mode | Status | Config | Observed progress |
|---|---|---:|---:|---|---|
| Darcy | `full10_darcy_adv_only_20260530` | `adv-only` | failed OOM | batch 448, optimizer batch 32 | only 1 training row, about 16.68 s before failure |
| Darcy | `full10_darcy_clean_plus_adv_20260530` | `clean-plus-adv` | failed OOM | batch 448, optimizer batch 32 | only 1 training row, about 18.66 s before failure |

These two did not produce formal checkpoints, `checkpoints.csv`, or
`summary.json`.

## Darcy, safe retry estimate

Current safe retry configuration:

- batch size: 416
- optimizer batch size: 128
- attack steps: 10
- epochs: 500
- total training steps: 1500

Current run status at the time this note was written:

- `full10_darcy_adv_only_retry_safe_20260531`
- status: running
- current progress: `global_step=175/1500`, epoch 59, about 11.7%
- observed mean step time so far: 11.82 s / step
- observed elapsed training-step sum: 2067.8 s = 34.5 min

Estimated total time for Darcy `adv-only` safe retry from the current observed
speed:

- 1500 steps * 11.82 s/step = 17724 s = 295.4 min = 4.92 h
- remaining from step 175: about 4.35 h

Estimated total time for Darcy `clean-plus-adv` safe retry, using the batch
sweep measurement for batch 416 / optimizer batch 128:

- observed sweep step time: 16.35 s / step
- 1500 steps * 16.35 s/step = 24518 s = 408.6 min = 6.81 h

## Short practical summary

- Burgers `adv-only`: finished in about 7.5 h.
- Burgers `clean-plus-adv`: finished in about 7.6 h.
- Darcy old formal runs: failed immediately from CUDA OOM.
- Darcy safe retry `adv-only`: estimated about 4.9 h total, currently running.
- Darcy safe retry `clean-plus-adv`: estimated about 6.8 h total, not started yet at the time of this note.
