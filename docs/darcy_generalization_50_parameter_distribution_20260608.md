# Darcy Flow Selected-50 Generalization Parameter Distribution - 2026-06-08

## Scope

This note explains how the Darcy Flow selected 50 generalization datasets were generated, what parameters differ across the 50 datasets, and what distributional differences those parameters imply.

Two selected-50 roots exist locally:

- `generalization_datasets_darcy_lossdrop50_selected_20260607`
- `generalization_datasets_darcy_stockgeneralization_20260608`

They use the same pool-generation family design but different seeds/screening checkpoints, so the selected independent draws are not identical.

## Generation Method

Observed from `tools/generate_darcy_generalization_candidates.py` and `2D_Darcy_FNO2d/solvers/darcy_jax_solver.py`:

1. Sample a latent 2D Gaussian random field using `make_grf_batch_fn`.
2. Convert the latent field into a Darcy coefficient field `a(x)`.
3. Solve `-div(a grad u) = 1` with zero boundary using `make_darcy_solve_fn`, a JAX matrix-free CG finite-difference solver.
4. Save `x=a`, `y=u`, `latent`, and `metadata` in each `.pt` file.

All selected datasets here use soft coefficients:

`a = low + (high - low) * sigmoid(beta * latent)`

So `low/high` set the coefficient range and contrast, while `soft_beta` controls how close the coefficient field is to a hard two-phase threshold.

Common settings across selected datasets:

- `alpha = 2.0`
- `tau = 3.0`
- `soft_coefficients = True`
- `samples = 48` per dataset
- `solve_resolution = 421`
- saved/downsampled `resolution = 85`
- solver: `jax_matrix_free_cg_second_order_fd`

Because `alpha` and `tau` are fixed, the latent-field spectrum/correlation family is mostly fixed. The selected datasets differ mainly by coefficient range, softness/sharpness, and independent random draw index.

## Lossdrop50 Selected - 2026-06-07

Observed from `generalization_datasets_darcy_lossdrop50_selected_20260607/candidate_manifest.csv`.

Parameter-family counts:

| family | low | high | beta | count |
|---|---:|---:|---:|---:|
| `low4_high10_beta8` | 4.0 | 10.0 | 8.0 | 18 |
| `low4_high10_beta10` | 4.0 | 10.0 | 10.0 | 16 |
| `low4_high10_beta12` | 4.0 | 10.0 | 12.0 | 9 |
| `low4_high10_beta14` | 4.0 | 10.0 | 14.0 | 4 |
| `low4_high10p5_beta12` | 4.0 | 10.5 | 12.0 | 3 |

Overall stats across the 50 selected datasets:

- `x_mean`: min `6.94479`, mean `7.01535`, max `7.25341`
- `x_std`: min `1.88902`, mean `2.13180`, max `2.45557`
- `y_mean`: min `0.00498218`, mean `0.00513066`, max `0.00520640`
- `y_std`: min `0.00327111`, mean `0.00334251`, max `0.00340757`
- selected `eval_loss_delta`: min `-2.31767e-07`, mean `-2.01330e-07`, max `-1.80238e-07`
- `cosine_mean`: min `0.759493`, mean `0.760843`, max `0.772317`

Group-level pattern:

- `beta=8` is the largest group and has lower `x_std` on average, around `1.95`.
- Higher beta values make the soft coefficient field sharper and closer to a binary low/high medium; `x_std` rises to about `2.36` for `beta=14`.
- `high=10.5` raises the coefficient mean to about `7.224` and lowers the solution mean to about `0.005006`, because larger coefficients reduce the Darcy solution amplitude under the same forcing.

## Stockgeneralization Selected - 2026-06-08

Observed from `generalization_datasets_darcy_stockgeneralization_20260608/candidate_manifest.csv`.

Parameter-family counts:

| family | low | high | beta | count |
|---|---:|---:|---:|---:|
| `low3_high12_beta6` | 3.0 | 12.0 | 6.0 | 2 |
| `low4_high10_beta8` | 4.0 | 10.0 | 8.0 | 18 |
| `low4_high10_beta10` | 4.0 | 10.0 | 10.0 | 15 |
| `low4_high10_beta12` | 4.0 | 10.0 | 12.0 | 6 |
| `low4_high10_beta14` | 4.0 | 10.0 | 14.0 | 3 |
| `low4_high10p5_beta12` | 4.0 | 10.5 | 12.0 | 6 |

Overall stats across the 50 selected datasets:

- `x_mean`: min `6.93874`, mean `7.05271`, max `7.52155`
- `x_std`: min `1.90966`, mean `2.16765`, max `2.57390`
- `y_mean`: min `0.00485010`, mean `0.00510589`, max `0.00519780`
- `y_std`: min `0.00318530`, mean `0.00332903`, max `0.00340902`
- selected `eval_loss_delta`: min `-2.51991e-07`, mean `-2.09082e-07`, max `-1.76161e-07`
- `cosine_mean`: min `0.798467`, mean `0.803976`, max `0.866127`

Group-level pattern:

- The stock set is still dominated by `low=4, high=10` soft families.
- It includes two wider-contrast `low=3, high=12, beta=6` datasets. These have much higher coefficient mean/std and lower solution mean than the low4/high10 family.
- Its cosine scores are higher than the 2026-06-07 lossdrop50 selected root because the screen used the other30 checkpoint and a different generation seed.

## Interpretation

Observed evidence:

- These selected 50 roots are not 50 unrelated PDE distributions. They are mostly independent draws from a small set of soft Darcy coefficient families.
- The major controlled differences are coefficient range (`low/high`) and soft-threshold sharpness (`soft_beta`).
- The latent GRF spectrum settings `alpha=2.0` and `tau=3.0` are fixed across selected datasets.
- Dataset IDs ending in `_01`, `_02`, etc. are independent random draws within the same parameter family, not different parameter settings.

Inference:

- This is a narrow, screening-biased loss-drop suite centered on soft coefficient fields, especially `low=4, high=10, beta=8/10/12/14`.
- It is useful for the loss-drop/self-training experiment it was designed for.
- It should not be presented as 50 broad, uniformly random, independent OOD distributions. It is better described as 50 selected independent draws from a few soft-coefficient distribution families.

## Derived Tables

Detailed per-dataset parameter rows and group statistics were written to:

- `forensics/darcy_generalization_50_parameter_distribution_20260608/lossdrop50_selected_20260607_selected50_detailed_manifest.csv`
- `forensics/darcy_generalization_50_parameter_distribution_20260608/lossdrop50_selected_20260607_param_family_counts.csv`
- `forensics/darcy_generalization_50_parameter_distribution_20260608/lossdrop50_selected_20260607_param_family_stats.csv`
- `forensics/darcy_generalization_50_parameter_distribution_20260608/stockgeneralization_20260608_selected50_detailed_manifest.csv`
- `forensics/darcy_generalization_50_parameter_distribution_20260608/stockgeneralization_20260608_param_family_counts.csv`
- `forensics/darcy_generalization_50_parameter_distribution_20260608/stockgeneralization_20260608_param_family_stats.csv`
