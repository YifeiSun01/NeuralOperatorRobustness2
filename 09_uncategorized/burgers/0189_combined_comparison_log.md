# Darcy Flow Loss1/Loss2/Loss3/Physics Burgers-Style Plot Report - 2026-06-11

This report applies the Burgers-style plotting vocabulary to Darcy Flow runs. The plots are Darcy plots, not Burgers plots; they reuse the same style and diagnostic categories: epoch curves, wall-clock curves, attack-gain curves, runtime/memory traces, final generated-dataset reductions, and fixed attack-probe delta FFT curves when probe data exists.

## Run Summary

| method | exists | epochs_completed | elapsed_minutes | logged_gen_dataset_count | logged_final_gen_relative_l2 | full50_improved_count | full50_mean_delta | peak_allocated_gib |
|---|---|---|---|---|---|---|---|---|
| loss1 | True | 3000 | 243.974 | 50 | 0.0879285 | 42 | -0.00340838 | 7.9736 |
| loss2 | True | 3000 | 265.862 | 50 | 0.072153 | 50 | -0.0191839 | 7.45111 |
| loss3 | True | 3000 | 231.59 | 50 | 0.0614064 | 49 | -0.0299305 | 7.45284 |
| physics | True | 3000 | 230.764 | 50 | 0.0878201 | 41 | -0.00351684 | 7.45284 |

## Evaluation Coverage In Existing Runs

| method | train datasets | test datasets | logged generalization datasets |
|---|---:|---:|---:|
| loss1 | 1 | 1 | 50 |
| loss2 | 1 | 1 | 50 |
| loss3 | 1 | 1 | 50 |
| physics | 1 | 1 | 50 |

Saved run configuration confirms the existing 100-epoch probes used `eval_max_samples=10`, `max_generalization_eval=8`, and `attack_probe_samples=0`. Therefore the current runs support polished aggregate curves and final full-50 comparison, but they do not contain every-epoch full-50 curves or fixed-probe delta FFT histories.

## Plot Scale

- Loss/error y-axis scale: log

## Outputs

- `analysis_outputs/darcy_full3000_burgers_image_only_bundle_20260612_work/work_visualizations/comparison_dense/darcy_adv_training_epoch_relative_l2_train_test_generalization.png`
- `analysis_outputs/darcy_full3000_burgers_image_only_bundle_20260612_work/work_visualizations/comparison_dense/darcy_adv_training_wall_clock_relative_l2_train_test_generalization.png`
- `analysis_outputs/darcy_full3000_burgers_image_only_bundle_20260612_work/work_visualizations/comparison_dense/darcy_adv_training_epoch_rmse_train_test_generalization.png`
- `analysis_outputs/darcy_full3000_burgers_image_only_bundle_20260612_work/work_visualizations/comparison_dense/darcy_adv_training_wall_clock_rmse_train_test_generalization.png`
- `analysis_outputs/darcy_full3000_burgers_image_only_bundle_20260612_work/work_visualizations/comparison_dense/darcy_adv_training_attack_loss_gain_mean.png`
- `analysis_outputs/darcy_full3000_burgers_image_only_bundle_20260612_work/work_visualizations/comparison_dense/darcy_adv_training_attack_loss_gain_relative_mean.png`
- `analysis_outputs/darcy_full3000_burgers_image_only_bundle_20260612_work/work_visualizations/comparison_dense/darcy_adv_training_train_loss_used_for_optimizer_updates_mean.png`
- `analysis_outputs/darcy_full3000_burgers_image_only_bundle_20260612_work/work_visualizations/comparison_dense/darcy_adv_training_delta_l2_rms_mean.png`
- `analysis_outputs/darcy_full3000_burgers_image_only_bundle_20260612_work/work_visualizations/comparison_dense/darcy_adv_training_attack_samples_per_sec.png`
- `analysis_outputs/darcy_full3000_burgers_image_only_bundle_20260612_work/work_visualizations/comparison_dense/darcy_adv_training_train_loss_on_adv.png`
- `analysis_outputs/darcy_full3000_burgers_image_only_bundle_20260612_work/work_visualizations/comparison_dense/darcy_adv_training_runtime_components.png`
- `analysis_outputs/darcy_full3000_burgers_image_only_bundle_20260612_work/work_visualizations/comparison_dense/darcy_adv_training_cuda_peak_allocated_gib.png`
- `analysis_outputs/darcy_full3000_burgers_image_only_bundle_20260612_work/work_visualizations/comparison_dense/darcy_adv_training_probe_delta_fft_high_freq_ratio.png`
- `analysis_outputs/darcy_full3000_burgers_image_only_bundle_20260612_work/work_visualizations/comparison_dense/darcy_adv_training_probe_delta_fft_spectral_centroid.png`
- `analysis_outputs/darcy_full3000_burgers_image_only_bundle_20260612_work/work_visualizations/comparison_dense/darcy_adv_training_probe_delta_total_variation.png`
- `analysis_outputs/darcy_full3000_burgers_image_only_bundle_20260612_work/work_visualizations/comparison_dense/darcy_adv_training_probe_attack_loss_gain_sample.png`
- `analysis_outputs/darcy_full3000_burgers_image_only_bundle_20260612_work/work_visualizations/comparison_dense/darcy_full50_mean_relative_l2_train_test_generalization.png`
- `analysis_outputs/darcy_full3000_burgers_image_only_bundle_20260612_work/work_visualizations/comparison_dense/darcy_full50_delta_relative_l2_heatmap.png`
- `analysis_outputs/darcy_full3000_burgers_image_only_bundle_20260612_work/work_visualizations/comparison_dense/darcy_full50_improvement_summary.png`

## Merged Tables

- `analysis_outputs/darcy_full3000_burgers_image_only_bundle_20260612_work/work_visualizations/comparison_dense/darcy_loss123physics_summary.csv`
- `analysis_outputs/darcy_full3000_burgers_image_only_bundle_20260612_work/work_visualizations/comparison_dense/darcy_eval_split_summary_merged.csv`
- `analysis_outputs/darcy_full3000_burgers_image_only_bundle_20260612_work/work_visualizations/comparison_dense/darcy_train_steps_merged.csv`
- `analysis_outputs/darcy_full3000_burgers_image_only_bundle_20260612_work/work_visualizations/comparison_dense/darcy_attack_epoch_summary_merged.csv`
- `analysis_outputs/darcy_full3000_burgers_image_only_bundle_20260612_work/work_visualizations/comparison_dense/darcy_attack_epsilon_bucket_summary_merged.csv`
- `analysis_outputs/darcy_full3000_burgers_image_only_bundle_20260612_work/work_visualizations/comparison_dense/darcy_memory_merged.csv`
- `analysis_outputs/darcy_full3000_burgers_image_only_bundle_20260612_work/work_visualizations/comparison_dense/darcy_full50_eval_merged.csv`

When `--derive-full50-from-run-eval` is used, this table is built from the final epoch of each run's `eval_metrics.csv`; otherwise it uses the supplied `--full50-eval-csv` posthoc evaluation table.

## Full Logging Launcher

- `tools/run_darcy_loss123physics_full_logging_20260611.sh`

The launcher runs Darcy loss1/loss2/loss3/physics with `--eval-max-samples 0`, `--max-generalization-eval 50`, and fixed attack probes enabled. That is the setting needed to produce every-epoch full-50 generalization curves and delta FFT histories like the Burgers polished reports.
