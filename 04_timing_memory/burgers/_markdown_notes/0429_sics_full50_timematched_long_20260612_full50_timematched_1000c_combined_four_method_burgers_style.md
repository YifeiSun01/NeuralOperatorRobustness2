# Darcy Flow Loss1/Loss2/Loss3/Physics Burgers-Style Plot Report - 2026-06-11

This report applies the Burgers-style plotting vocabulary to Darcy Flow runs. The plots are Darcy plots, not Burgers plots; they reuse the same style and diagnostic categories: epoch curves, wall-clock curves, attack-gain curves, runtime/memory traces, final generated-dataset reductions, and fixed attack-probe delta FFT curves when probe data exists.

## Run Summary

| method | exists | epochs_completed | elapsed_minutes | logged_gen_dataset_count | logged_final_gen_relative_l2 | full50_improved_count | full50_mean_delta | peak_allocated_gib |
|---|---|---|---|---|---|---|---|---|
| loss1 | True | 1000 | 77.0066 | 50 | 0.101024 | 0 | 0.00968697 | 7.9736 |
| loss2 | True | 1026 | 77.0585 | 50 | 0.0864258 | 44 | -0.00491106 | 7.45051 |
| loss3 | True | 1011 | 77.3917 | 50 | 0.0728309 | 50 | -0.018506 | 7.45263 |
| physics | True | 1040 | 76.6887 | 50 | 0.0921814 | 18 | 0.000844458 | 7.45259 |

## Evaluation Coverage In Existing Runs

| method | train datasets | test datasets | logged generalization datasets |
|---|---:|---:|---:|
| loss1 | 1 | 1 | 50 |
| loss2 | 1 | 1 | 50 |
| loss3 | 1 | 1 | 50 |
| physics | 1 | 1 | 50 |

Saved run configuration confirms the existing 100-epoch probes used `eval_max_samples=10`, `max_generalization_eval=8`, and `attack_probe_samples=0`. Therefore the current runs support polished aggregate curves and final full-50 comparison, but they do not contain every-epoch full-50 curves or fixed-probe delta FFT histories.

## Outputs

- `visualizations/darcy_loss123physics_full50_timematched_long_20260612_full50_timematched_1000c/combined_four_method_burgers_style/darcy_adv_training_epoch_relative_l2_train_test_generalization.png`
- `visualizations/darcy_loss123physics_full50_timematched_long_20260612_full50_timematched_1000c/combined_four_method_burgers_style/darcy_adv_training_wall_clock_relative_l2_train_test_generalization.png`
- `visualizations/darcy_loss123physics_full50_timematched_long_20260612_full50_timematched_1000c/combined_four_method_burgers_style/darcy_adv_training_epoch_rmse_train_test_generalization.png`
- `visualizations/darcy_loss123physics_full50_timematched_long_20260612_full50_timematched_1000c/combined_four_method_burgers_style/darcy_adv_training_wall_clock_rmse_train_test_generalization.png`
- `visualizations/darcy_loss123physics_full50_timematched_long_20260612_full50_timematched_1000c/combined_four_method_burgers_style/darcy_adv_training_attack_loss_gain_mean.png`
- `visualizations/darcy_loss123physics_full50_timematched_long_20260612_full50_timematched_1000c/combined_four_method_burgers_style/darcy_adv_training_attack_loss_gain_relative_mean.png`
- `visualizations/darcy_loss123physics_full50_timematched_long_20260612_full50_timematched_1000c/combined_four_method_burgers_style/darcy_adv_training_train_loss_used_for_optimizer_updates_mean.png`
- `visualizations/darcy_loss123physics_full50_timematched_long_20260612_full50_timematched_1000c/combined_four_method_burgers_style/darcy_adv_training_delta_l2_rms_mean.png`
- `visualizations/darcy_loss123physics_full50_timematched_long_20260612_full50_timematched_1000c/combined_four_method_burgers_style/darcy_adv_training_attack_samples_per_sec.png`
- `visualizations/darcy_loss123physics_full50_timematched_long_20260612_full50_timematched_1000c/combined_four_method_burgers_style/darcy_adv_training_train_loss_on_adv.png`
- `visualizations/darcy_loss123physics_full50_timematched_long_20260612_full50_timematched_1000c/combined_four_method_burgers_style/darcy_adv_training_runtime_components.png`
- `visualizations/darcy_loss123physics_full50_timematched_long_20260612_full50_timematched_1000c/combined_four_method_burgers_style/darcy_adv_training_cuda_peak_allocated_gib.png`
- `visualizations/darcy_loss123physics_full50_timematched_long_20260612_full50_timematched_1000c/combined_four_method_burgers_style/darcy_adv_training_probe_delta_fft_high_freq_ratio.png`
- `visualizations/darcy_loss123physics_full50_timematched_long_20260612_full50_timematched_1000c/combined_four_method_burgers_style/darcy_adv_training_probe_delta_fft_spectral_centroid.png`
- `visualizations/darcy_loss123physics_full50_timematched_long_20260612_full50_timematched_1000c/combined_four_method_burgers_style/darcy_adv_training_probe_delta_total_variation.png`
- `visualizations/darcy_loss123physics_full50_timematched_long_20260612_full50_timematched_1000c/combined_four_method_burgers_style/darcy_adv_training_probe_attack_loss_gain_sample.png`
- `visualizations/darcy_loss123physics_full50_timematched_long_20260612_full50_timematched_1000c/combined_four_method_burgers_style/darcy_full50_mean_relative_l2_train_test_generalization.png`
- `visualizations/darcy_loss123physics_full50_timematched_long_20260612_full50_timematched_1000c/combined_four_method_burgers_style/darcy_full50_delta_relative_l2_heatmap.png`
- `visualizations/darcy_loss123physics_full50_timematched_long_20260612_full50_timematched_1000c/combined_four_method_burgers_style/darcy_full50_improvement_summary.png`

## Merged Tables

- `visualizations/darcy_loss123physics_full50_timematched_long_20260612_full50_timematched_1000c/combined_four_method_burgers_style/darcy_loss123physics_summary.csv`
- `visualizations/darcy_loss123physics_full50_timematched_long_20260612_full50_timematched_1000c/combined_four_method_burgers_style/darcy_eval_split_summary_merged.csv`
- `visualizations/darcy_loss123physics_full50_timematched_long_20260612_full50_timematched_1000c/combined_four_method_burgers_style/darcy_train_steps_merged.csv`
- `visualizations/darcy_loss123physics_full50_timematched_long_20260612_full50_timematched_1000c/combined_four_method_burgers_style/darcy_attack_epoch_summary_merged.csv`
- `visualizations/darcy_loss123physics_full50_timematched_long_20260612_full50_timematched_1000c/combined_four_method_burgers_style/darcy_attack_epsilon_bucket_summary_merged.csv`
- `visualizations/darcy_loss123physics_full50_timematched_long_20260612_full50_timematched_1000c/combined_four_method_burgers_style/darcy_memory_merged.csv`
- `visualizations/darcy_loss123physics_full50_timematched_long_20260612_full50_timematched_1000c/combined_four_method_burgers_style/darcy_full50_eval_merged.csv`

When `--derive-full50-from-run-eval` is used, this table is built from the final epoch of each run's `eval_metrics.csv`; otherwise it uses the supplied `--full50-eval-csv` posthoc evaluation table.

## Full Logging Launcher

- `/tmp/darcy_full_logging_launcher_unused_20260612.sh`

The launcher runs Darcy loss1/loss2/loss3/physics with `--eval-max-samples 0`, `--max-generalization-eval 50`, and fixed attack probes enabled. That is the setting needed to produce every-epoch full-50 generalization curves and delta FFT histories like the Burgers polished reports.
