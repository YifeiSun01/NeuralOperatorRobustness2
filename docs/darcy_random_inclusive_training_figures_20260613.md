# Darcy Random-Inclusive Training Figures - 2026-06-13

These figures update the Darcy loss/metric plots so the two random-source training methods appear alongside loss1/loss2/loss3/physics loss.

- Analysis: `analysis_outputs/darcy_random_inclusive_training_figures_20260613`
- Figures: `visualizations/darcy_random_inclusive_training_figures_20260613/comparison_dense`
- Random methods: `random clean y`, `random solver y`.
- Old adversarial methods are first-stage 1000-ish runs; random methods are 1100 epoch runs.
- Wall-clock plots use true cumulative time: baseline eval + attack/random-source generation + optimizer + eval, calibrated to each run summary elapsed time.
- Wall-clock components CSV: `analysis_outputs/darcy_random_inclusive_training_figures_20260613/six_method_wall_clock_components.csv`

## Wall-Clock Correction

Earlier generated wall-clock figures incorrectly used only `optimizer_wall_sec`, so the x-axis showed optimizer-update minutes rather than real elapsed wall-clock time. The regenerated figures use baseline evaluation + attack/random-source generation + optimizer updates + evaluation, calibrated to each run summary elapsed time.

| method | optimizer min | attack/random min | eval min | true elapsed min |
|---|---:|---:|---:|---:|
| `loss1` | 3.834 | 8.287 | 62.947 | 77.007 |
| `loss2` | 3.919 | 7.092 | 63.907 | 77.058 |
| `loss3` | 3.863 | 7.786 | 63.637 | 77.392 |
| `physics loss` | 3.985 | 6.145 | 64.804 | 76.689 |
| `random clean y` | 9.283 | 5.871 | 283.652 | 303.769 |
| `random solver y` | 10.008 | 8.544 | 282.927 | 306.838 |

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
