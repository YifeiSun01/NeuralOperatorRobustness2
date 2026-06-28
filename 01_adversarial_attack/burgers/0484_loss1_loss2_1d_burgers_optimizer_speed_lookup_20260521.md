# Loss1/Loss2 1D Burgers Optimizer Speed Lookup - 2026-05-21

Status: completed lookup from existing local result files. No attack was rerun.

## Question

Check whether 1D Burgers has existing evidence for how fast different optimizers
improve `loss1` and `loss2`.

## Source Data

Observed from:

- `results/three_loss_batch100_full_loss3_delta_rerun_20260514_fno_eps8_alpha0p3_final_boundary/`
- Per-run files: `loss_stats.csv` and `summary.json` under each
  `loss1_*_*` and `loss2_*_*` attack tag.
- Existing documentation:
  `docs/unified_eval_metric_loss_curve_plots_20260516.md` and
  `THREE_LOSS_BATCH100_FULL_LOSS3_SWEEP.md`.

Key setting:

- FNO / 1D Burgers `nu=0.001`
- batch size `100`, dataset indices `0..99`
- `epsilon=8`, `alpha=0.3`, `steps=100`
- `p=2`, `q=2`
- methods present for `loss1` and `loss2`: `pgd`, `lp_steepest_pgd`,
  `generalized_power`

No local result directory or documentation evidence was found for applying
`raw_replace`, `steepest_replace`, or `power_replace` directly to `loss1` or
`loss2`. Those replacement-method runs are locally evidenced for later
`loss3`-focused experiments.

## Observed Original-Loss Speed

The table reports batch-mean optimized objective values. `k95` is the first step
at which the mean reaches at least 95% of the final step-100 value.

| optimized objective | method | final mean | peak mean | peak step | k95 | runtime seconds |
|---|---:|---:|---:|---:|---:|---:|
| `loss1_original` | `pgd` | 11.1718 | 11.1718 | 100 | 26 | 137.665 |
| `loss1_original` | `lp_steepest_pgd` | 11.1493 | 11.1493 | 100 | 27 | 137.466 |
| `loss1_original` | `generalized_power` | 11.0143 | 11.0254 | 47 | 3 | 137.489 |
| `loss2_original` | `pgd` | 11.3734 | 11.3734 | 100 | 24 | 137.374 |
| `loss2_original` | `lp_steepest_pgd` | 11.3812 | 11.3812 | 100 | 27 | 137.806 |
| `loss2_original` | `generalized_power` | 11.2360 | 11.2534 | 43 | 2 | 137.555 |

Selected mean curve values:

| objective | method | k1 | k2 | k5 | k10 | k25 | k50 | k100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `loss1_original` | `pgd` | 0.2921 | 2.7601 | 4.8222 | 6.8273 | 10.6131 | 11.0415 | 11.1718 |
| `loss1_original` | `lp_steepest_pgd` | 0.8787 | 1.6884 | 3.4114 | 5.4495 | 10.0439 | 11.0398 | 11.1493 |
| `loss1_original` | `generalized_power` | 8.4488 | 10.3223 | 10.9192 | 10.9837 | 11.0000 | 11.0099 | 11.0143 |
| `loss2_original` | `pgd` | 2.2798 | 3.4917 | 5.3146 | 7.2486 | 10.9560 | 11.2618 | 11.3734 |
| `loss2_original` | `lp_steepest_pgd` | 1.2339 | 2.0247 | 3.7050 | 5.6965 | 10.3191 | 11.2523 | 11.3812 |
| `loss2_original` | `generalized_power` | 9.8090 | 10.8204 | 11.1750 | 11.1897 | 11.2523 | 11.2356 | 11.2360 |

## Interpretation

Observed from the existing CSVs: in optimization-step terms,
`generalized_power` is much faster for both `loss1_original` and
`loss2_original`, reaching near-final values within 2-3 steps. Ordinary `pgd`
and `lp_steepest_pgd` take roughly 24-27 steps to reach 95% of their final
means.

Observed from `summary.json`: wall-clock runtime is essentially the same across
these three methods for this batch run, about 137-139 seconds per tag. The speed
difference is therefore a convergence-per-step difference, not a meaningful
per-run wall-time difference in the saved data.

Inference from the available `loss1`/`loss2` evidence: for the original
objectives, `generalized_power` is the fast-to-plateau method, while `pgd` and
`lp_steepest_pgd` are slower but end at slightly larger mean objective values
for this `epsilon=8, alpha=0.3, p=q=2` setting.

Remaining work:

- If the intended comparison includes `raw_replace`, `steepest_replace`, or
  `power_replace` on `loss1`/`loss2`, those runs are not currently evidenced
  locally and would need to be run or restored from external storage.
