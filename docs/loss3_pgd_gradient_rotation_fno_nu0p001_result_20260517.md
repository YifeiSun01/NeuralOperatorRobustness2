# Loss3 PGD Gradient Rotation Result

Date: 2026-05-17 UTC

## Scope

This is only Experiment 1 from `docs/simple_path_linearity_validation_plan_20260517.md`.

Run setting:

- model/task: FNO, 1D Burgers, `nu=0.001`
- objective: `loss3_original_pgd`
- samples: `0, 7, 40, 47, 115`
- PGD: `epsilon=8.0`, `alpha=0.3`, `steps=50`
- saved trajectory points: `k=0,5,10,15,20,25,30,35,40,45,50`
- random start: enabled, `random_start_scale=1e-6`, `seed=0`
- no Hessian, no dense Jacobian, no SVD

Trajectory root:

`forensics/loss3_simple_path_linearity_20260517/fno_nu0p001/loss3_original_pgd_trajectories/`

Analysis root:

`forensics/loss3_simple_path_linearity_20260517/fno_nu0p001/experiment1_pgd_gradient_rotation/`

## Definitions

At each saved PGD point,

`z_k = x0 + delta_k`.

The Loss3 objective is

`L3(z) = ||f(z) - j(z)||_2`.

The recomputed true gradient is

`g_k = grad_z L3(z_k)`.

Adjacent saved-step gradient rotation:

`theta_prev(k) = angle(g_k, g_{k-5})`.

Accumulated rotation relative to the first saved point:

`theta_k0(k) = angle(g_k, g_0)`.

Rotation relative to the exact clean input gradient:

`theta_clean(k) = angle(g_k, grad_z L3(x0))`.

Step-alignment diagnostic:

`theta_step(k) = angle(g_k, delta_k - delta_{k-5})`.

The main test variable is `theta_prev(k)`.  If late-path local geometry is more stable, this angle should tend to be smaller later along the trajectory.

## Aggregate Results

Rows are means over the five samples.

| k | budget ratio | Loss3 | grad norm | theta_prev | theta_k0 | theta_clean | theta_step |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.000001 | 0.2815 | 0.1665 | nan | 0.0000 | 0.0002 | nan |
| 5 | 0.0359 | 0.3438 | 0.2455 | 17.87 | 17.87 | 17.87 | 10.39 |
| 10 | 0.0924 | 0.5438 | 0.4023 | 16.07 | 32.84 | 32.84 | 8.70 |
| 15 | 0.1713 | 0.9348 | 0.5046 | 14.60 | 43.37 | 43.37 | 8.65 |
| 20 | 0.2687 | 1.5692 | 0.5504 | 11.30 | 49.23 | 49.23 | 5.72 |
| 25 | 0.3649 | 2.1124 | 0.5318 | 8.63 | 52.47 | 52.47 | 5.12 |
| 30 | 0.4595 | 2.6134 | 0.5334 | 9.38 | 56.03 | 56.03 | 5.89 |
| 35 | 0.5543 | 3.1326 | 0.5766 | 12.66 | 61.50 | 61.50 | 7.61 |
| 40 | 0.6319 | 3.6236 | 0.6818 | 12.40 | 67.22 | 67.22 | 12.73 |
| 45 | 0.7077 | 4.1997 | 0.6981 | 6.38 | 68.91 | 68.91 | 18.77 |
| 50 | 0.7636 | 4.6169 | 0.6962 | 4.49 | 69.69 | 69.69 | 32.82 |

## Early/Middle/Late Summary

| region | k range | mean theta_prev | mean theta_clean | mean budget ratio | mean Loss3 |
| --- | --- | ---: | ---: | ---: | ---: |
| early | 5,10,15 | 16.18 | 31.36 | 0.0998 | 0.6075 |
| middle | 20,25,30,35 | 10.49 | 54.81 | 0.4118 | 2.3569 |
| late | 40,45,50 | 7.75 | 68.61 | 0.7011 | 4.1467 |

## Interpretation

The adjacent saved-step gradient angle is not perfectly monotone.  It drops from `17.87 deg` at `k=5` to `8.63 deg` at `k=25`, then has a middle bump at `k=35` and `k=40`, and finally drops to `4.49 deg` at `k=50`.

The coarse early/middle/late averages show the intended pattern:

`16.18 deg -> 10.49 deg -> 7.75 deg`.

So this run supports the weaker and more accurate statement:

`Along the true Loss3 PGD path, the local optimizer gradient direction is more unstable early and more stable late on average, but not strictly monotonically at every saved step.`

At the same time, the accumulated angle to clean grows:

`17.87 deg -> 32.84 deg -> ... -> 69.69 deg`.

That is important.  Late stability does not mean the geometry returns to the clean-point geometry.  It means the path appears to enter a different, more stable direction regime.

The `theta_step` column is only a secondary diagnostic.  It grows at late steps because the cumulative PGD path and the immediate saved displacement are not identical objects, especially as projection and finite step effects matter more.  The main evidence for this experiment is `theta_prev`.

## Output Files

- `pgd_gradient_rotation_by_sample_k.csv`
- `pgd_gradient_rotation_aggregate_by_k.csv`
- `pgd_gradient_rotation_summary_by_sample.csv`
- `summary.md`
- `manifest.json`

Analysis script:

`tools/analyze_loss3_pgd_gradient_rotation.py`
