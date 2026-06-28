# Darcy Random-Inclusive Training Figures - 2026-06-13

These figures update the Darcy loss/metric plots so the two random-source training methods appear alongside loss1/loss2/loss3/physics loss.

- Analysis: `analysis_outputs/darcy_random_inclusive_training_figures_20260613`
- Figures: `visualizations/darcy_random_inclusive_training_figures_20260613/comparison_dense`
- Random methods: `random clean y`, `random solver y`.
- Training/loss/Delta curves are capped at epochs `0..1000`.
- Old adversarial methods are first-stage 1000-ish runs; random methods were trained to 1100, but the curve figures do not plot epochs after 1000.
- Wall-clock plots use training-only time from `train_steps.step_wall_sec`: delta/random-source generation + x/y training-pair construction + optimizer forward/backward/update. Baseline and per-epoch evaluation are excluded.
- Wall-clock components CSV: `analysis_outputs/darcy_random_inclusive_training_figures_20260613/six_method_wall_clock_components.csv`

## High-Contrast Line Colors

| method | color |
|---|---|
| `loss1` | `#2563eb` |
| `loss2` | `#f97316` |
| `loss3` | `#dc2626` |
| `physics loss` | `#7c3aed` |
| `random clean y` | `#059669` |
| `random solver y` | `#0891b2` |

## Wall-Clock Correction

Earlier random-inclusive wall-clock figures included every-epoch evaluation, which made the random-source runs look artificially slow because those two runs were executed concurrently and their evaluation passes were resource-contended. The regenerated figures exclude evaluation and use training-only `step_wall_sec`.

| method | training-only min | attack/random min | optimizer min | eval min excluded | observed elapsed min |
|---|---:|---:|---:|---:|---:|
| `loss1` | 13.640 | 8.287 | 3.834 | 62.947 | 77.007 |
| `loss2` | 12.721 | 7.092 | 3.919 | 63.907 | 77.058 |
| `loss3` | 13.323 | 7.786 | 3.863 | 63.637 | 77.392 |
| `physics loss` | 11.446 | 6.145 | 3.985 | 64.804 | 76.689 |
| `random clean y` | 18.815 | 5.871 | 9.283 | 283.652 | 303.769 |
| `random solver y` | 22.710 | 8.544 | 10.008 | 282.927 | 306.838 |

## Figures

- `visualizations/darcy_random_inclusive_training_figures_20260613/comparison_dense/darcy_six_method_epoch_relative_l2_train_test_generalization.png`
- `visualizations/darcy_random_inclusive_training_figures_20260613/comparison_dense/darcy_six_method_epoch_rmse_train_test_generalization.png`
- `visualizations/darcy_random_inclusive_training_figures_20260613/comparison_dense/darcy_six_method_wall_clock_relative_l2_train_test_generalization.png`
- `visualizations/darcy_random_inclusive_training_figures_20260613/comparison_dense/darcy_six_method_wall_clock_rmse_train_test_generalization.png`
- `visualizations/darcy_random_inclusive_training_figures_20260613/comparison_dense/darcy_six_method_optimizer_train_loss.png`
- `visualizations/darcy_random_inclusive_training_figures_20260613/comparison_dense/darcy_six_method_grad_norm.png`
- `visualizations/darcy_random_inclusive_training_figures_20260613/comparison_dense/darcy_six_method_attack_loss_gain_mean.png`
- `visualizations/darcy_random_inclusive_training_figures_20260613/comparison_dense/darcy_six_method_delta_l2_rms_mean.png`
- `visualizations/darcy_random_inclusive_training_figures_20260613/comparison_dense/darcy_six_method_final_clean_and_attack_summary.png`
- `visualizations/darcy_random_inclusive_training_figures_20260613/comparison_dense/darcy_six_method_metric_jacobian_summary.png`
