# Loss3 PGD Gradient Rotation Result, 100 Steps

Date: 2026-05-17 UTC

## Scope

This reruns Experiment 1 with `steps=100` instead of `steps=50`.

Run setting:

- model/task: FNO, 1D Burgers, `nu=0.001`
- objective: `loss3_original_pgd`
- samples: `0, 7, 40, 47, 115`
- PGD: `epsilon=8.0`, `alpha=0.3`, `steps=100`
- saved trajectory points: every 5 steps, `k=0,5,10,...,100`
- random start: enabled, `random_start_scale=1e-6`, `seed=0`
- no Hessian, no dense Jacobian, no SVD

Trajectory root:

`forensics/loss3_simple_path_linearity_20260517/fno_nu0p001/loss3_original_pgd_trajectories_steps100/`

Analysis root:

`forensics/loss3_simple_path_linearity_20260517/fno_nu0p001/experiment1_pgd_gradient_rotation_steps100/`

## What Is Averaged

Each row below is the mean over five samples:

`0, 7, 40, 47, 115`.

The standard deviation is also over those same five samples.

The main angle is:

`theta_prev(k) = angle(g_k, g_{k-5})`,

where

`g_k = grad_z L3(z_k)`.

So this is not a cross-loss angle and not a Jacobian angle.  It is the true Loss3 PGD gradient direction at the current saved point compared with the true Loss3 PGD gradient direction at the previous saved point.

## Aggregate Results With Standard Deviation

| k | budget mean | Loss3 mean | theta_prev mean | theta_prev std | theta_prev min | theta_prev max | theta_clean mean | theta_clean std |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.0000 | 0.2815 | nan | nan | nan | nan | 0.0002 | 0.0000 |
| 5 | 0.0359 | 0.3438 | 17.87 | 5.77 | 13.20 | 28.84 | 17.87 | 5.77 |
| 10 | 0.0924 | 0.5438 | 16.07 | 3.84 | 11.87 | 22.50 | 32.84 | 8.65 |
| 15 | 0.1713 | 0.9348 | 14.60 | 5.38 | 8.68 | 23.71 | 43.37 | 7.42 |
| 20 | 0.2687 | 1.5692 | 11.30 | 2.84 | 7.50 | 15.41 | 49.23 | 7.29 |
| 25 | 0.3649 | 2.1124 | 8.63 | 1.61 | 7.07 | 11.19 | 52.47 | 6.22 |
| 30 | 0.4595 | 2.6134 | 9.38 | 3.40 | 6.86 | 15.99 | 56.03 | 6.03 |
| 35 | 0.5543 | 3.1326 | 12.66 | 8.69 | 5.48 | 25.87 | 61.50 | 9.07 |
| 40 | 0.6319 | 3.6236 | 12.40 | 12.04 | 2.99 | 35.37 | 67.22 | 11.89 |
| 45 | 0.7077 | 4.1997 | 6.38 | 4.14 | 1.77 | 13.62 | 68.91 | 12.55 |
| 50 | 0.7636 | 4.6169 | 4.49 | 3.15 | 1.25 | 8.76 | 69.69 | 12.79 |
| 55 | 0.8228 | 5.0604 | 11.07 | 14.36 | 1.15 | 38.80 | 68.64 | 11.20 |
| 60 | 0.8769 | 5.5226 | 6.08 | 5.53 | 0.93 | 13.97 | 69.27 | 11.64 |
| 65 | 0.9156 | 5.8692 | 4.84 | 4.98 | 0.77 | 14.34 | 69.37 | 11.52 |
| 70 | 0.9312 | 6.0512 | 2.91 | 1.83 | 0.66 | 5.00 | 69.42 | 11.37 |
| 75 | 0.9409 | 6.1696 | 2.86 | 2.17 | 0.57 | 6.52 | 69.55 | 11.28 |
| 80 | 0.9506 | 6.2756 | 3.94 | 3.60 | 0.51 | 9.77 | 69.88 | 11.14 |
| 85 | 0.9606 | 6.3779 | 3.77 | 4.66 | 0.45 | 12.93 | 70.51 | 10.74 |
| 90 | 0.9711 | 6.4733 | 3.74 | 4.52 | 0.41 | 12.52 | 71.39 | 10.29 |
| 95 | 0.9825 | 6.5710 | 2.87 | 3.40 | 0.38 | 9.52 | 72.20 | 10.07 |
| 100 | 0.9948 | 6.6760 | 2.21 | 2.60 | 0.35 | 7.35 | 72.88 | 10.03 |

## Region Summary

| region | k range | mean theta_prev | mean theta_prev std | mean theta_clean | mean budget | mean Loss3 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| early | 5,10,15 | 16.18 | 5.00 | 31.36 | 0.0998 | 0.6075 |
| middle | 20,25,30,35,40,45,50 | 9.32 | 5.12 | 60.72 | 0.5358 | 3.1240 |
| late | 55,60,65,70,75,80,85,90,95,100 | 4.43 | 4.77 | 70.31 | 0.9347 | 6.1047 |
| very late | 75,80,85,90,95,100 | 3.23 | 3.49 | 71.07 | 0.9668 | 6.4239 |

## Interpretation

The 100-step run strengthens the same qualitative conclusion.

The adjacent saved-step gradient rotation decreases by region:

`16.18 deg -> 9.32 deg -> 4.43 deg`.

The very-late part is even smaller:

`3.23 deg` over `k=75,80,85,90,95,100`.

There is still a visible outlier-like bump at `k=55`:

`theta_prev mean = 11.07 deg`, `std = 14.36 deg`, `max = 38.80 deg`.

So the result should not be described as step-by-step monotone.  The accurate claim is:

`Along the true Loss3 PGD path, the adjacent gradient direction becomes much more stable later on average, but individual samples can still have transition events after k=50.`

The accumulated clean-reference angle stays large and slowly grows:

`theta_clean mean = 69.69 deg` at `k=50`,

`theta_clean mean = 72.88 deg` at `k=100`.

So the late stability is not a return to clean geometry.  It is a stable high-radius direction regime.

## Evidence Strength And Limitation

This PGD-gradient-rotation experiment is useful, but it should not be treated as the strongest proof of the local-twist / far-stability claim.

The reason is that the PGD path is endogenous:

`z_{k+1}` is produced by the same Loss3 gradient field that we are measuring.

So when `angle(g_k, g_{k-5})` becomes smaller later, two explanations are mixed together:

1. the residual landscape itself may become locally more stable farther away from `x0`;
2. PGD may have steered itself into a more stable optimizer corridor because of the loss gradient, projection, finite step size, or boundary effects.

This means the causal interpretation is weaker than the straight-line Jacobian/subspace experiment.

The straight-line experiment fixes the path externally:

`z(t) = x0 + t delta*`.

Because `delta*` is fixed, the experiment asks a cleaner question:

`Along the same outward chord, does the local residual-Jacobian geometry rotate sharply near x0 and then stabilize farther out?`

That is closer to the target claim about the geometry of the space itself.

A good evidence hierarchy is:

1. straight-line residual-Jacobian / top-k subspace rotation: strongest current evidence, because the path is fixed and the measured object is local geometry;
2. PGD-path Jacobian/subspace rotation: useful realistic auxiliary evidence, but the path is chosen by the optimizer;
3. PGD-gradient rotation: weaker auxiliary evidence, because it measures optimizer direction rather than the Jacobian geometry directly;
4. endpoint-vs-movement gradient angle: mostly explains clean residual bias becoming less important, not direct evidence that nonlinearity decreases.

Therefore the most accurate interpretation of this 100-step PGD-gradient experiment is:

`The real Loss3 optimizer path shows a similar late-stabilization pattern, so it is consistent with the straight-line geometry story.  But by itself it is not the main proof, because the optimizer-generated path is confounded with the gradient field being measured.`

## Output Files

- `pgd_gradient_rotation_by_sample_k.csv`
- `pgd_gradient_rotation_aggregate_by_k.csv`
- `pgd_gradient_rotation_summary_by_sample.csv`
- `summary.md`
- `manifest.json`

Analysis script:

`tools/analyze_loss3_pgd_gradient_rotation.py`
