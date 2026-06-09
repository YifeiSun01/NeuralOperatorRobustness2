# Darcy/C-flow visualization artifact locations - 2026-06-08

Status: inspected local Darcy/C-flow artifacts. The existing PNG visualizations are saved under each run's `darcy_time_matched_summary/` directory. Attack perturbation/frequency information is present in `attack_probe_samples.csv` and `attack_probe_samples/*.npz`, but no standalone Darcy perturbation-frequency PNGs were found locally.

## Clean single-GPU wall-clock runs

### loss1

- Plot directory: `adversarial_training_runs/darcy_lossdrop50_loss1_single_gpu_time_matched_loss3wall_20260608/darcy_time_matched_summary/`
- Split metric plots:
  - `adversarial_training_runs/darcy_lossdrop50_loss1_single_gpu_time_matched_loss3wall_20260608/darcy_time_matched_summary/darcy_lossdrop50_loss1_single_gpu_time_matched_loss3wall_20260608/rmse_split_curves.png`
  - `adversarial_training_runs/darcy_lossdrop50_loss1_single_gpu_time_matched_loss3wall_20260608/darcy_time_matched_summary/darcy_lossdrop50_loss1_single_gpu_time_matched_loss3wall_20260608/relative_l2_split_curves.png`
- Attack/loss plot:
  - `adversarial_training_runs/darcy_lossdrop50_loss1_single_gpu_time_matched_loss3wall_20260608/darcy_time_matched_summary/darcy_lossdrop50_loss1_single_gpu_time_matched_loss3wall_20260608/attack_objective_and_solver_mse_curves.png`
- Eval/train loss progress:
  - `adversarial_training_runs/darcy_lossdrop50_loss1_single_gpu_time_matched_loss3wall_20260608/darcy_time_matched_summary/eval_train_loss_progress.png`
- Attack perturbation raw data:
  - `adversarial_training_runs/darcy_lossdrop50_loss1_single_gpu_time_matched_loss3wall_20260608/darcy/attack_probe_samples.csv`
  - `adversarial_training_runs/darcy_lossdrop50_loss1_single_gpu_time_matched_loss3wall_20260608/darcy/attack_probe_samples/`

### loss2

- Plot directory: `adversarial_training_runs/darcy_lossdrop50_loss2_single_gpu_time_matched_loss3wall_20260608/darcy_time_matched_summary/`
- Split metric plots:
  - `adversarial_training_runs/darcy_lossdrop50_loss2_single_gpu_time_matched_loss3wall_20260608/darcy_time_matched_summary/darcy_lossdrop50_loss2_single_gpu_time_matched_loss3wall_20260608/rmse_split_curves.png`
  - `adversarial_training_runs/darcy_lossdrop50_loss2_single_gpu_time_matched_loss3wall_20260608/darcy_time_matched_summary/darcy_lossdrop50_loss2_single_gpu_time_matched_loss3wall_20260608/relative_l2_split_curves.png`
- Attack/loss plot:
  - `adversarial_training_runs/darcy_lossdrop50_loss2_single_gpu_time_matched_loss3wall_20260608/darcy_time_matched_summary/darcy_lossdrop50_loss2_single_gpu_time_matched_loss3wall_20260608/attack_objective_and_solver_mse_curves.png`
- Eval/train loss progress:
  - `adversarial_training_runs/darcy_lossdrop50_loss2_single_gpu_time_matched_loss3wall_20260608/darcy_time_matched_summary/eval_train_loss_progress.png`
- Attack perturbation raw data:
  - `adversarial_training_runs/darcy_lossdrop50_loss2_single_gpu_time_matched_loss3wall_20260608/darcy/attack_probe_samples.csv`
  - `adversarial_training_runs/darcy_lossdrop50_loss2_single_gpu_time_matched_loss3wall_20260608/darcy/attack_probe_samples/`

### loss3

- Plot directory: `adversarial_training_runs/darcy_lossdrop50_loss3_500ep_fromscreen_20260607/darcy_time_matched_summary/`
- Split metric plots:
  - `adversarial_training_runs/darcy_lossdrop50_loss3_500ep_fromscreen_20260607/darcy_time_matched_summary/darcy_lossdrop50_loss3_500ep_fromscreen_20260607/rmse_split_curves.png`
  - `adversarial_training_runs/darcy_lossdrop50_loss3_500ep_fromscreen_20260607/darcy_time_matched_summary/darcy_lossdrop50_loss3_500ep_fromscreen_20260607/relative_l2_split_curves.png`
- Attack/loss plot:
  - `adversarial_training_runs/darcy_lossdrop50_loss3_500ep_fromscreen_20260607/darcy_time_matched_summary/darcy_lossdrop50_loss3_500ep_fromscreen_20260607/attack_objective_and_solver_mse_curves.png`
- Attack perturbation raw data:
  - `adversarial_training_runs/darcy_lossdrop50_loss3_500ep_fromscreen_20260607/darcy/attack_probe_samples.csv`
  - `adversarial_training_runs/darcy_lossdrop50_loss3_500ep_fromscreen_20260607/darcy/attack_probe_samples/`

### physics

- Plot directory: `adversarial_training_runs/darcy_lossdrop50_physics_time_matched_loss3wall_20260608/darcy_time_matched_summary/`
- Split metric plots:
  - `adversarial_training_runs/darcy_lossdrop50_physics_time_matched_loss3wall_20260608/darcy_time_matched_summary/darcy_lossdrop50_physics_time_matched_loss3wall_20260608/rmse_split_curves.png`
  - `adversarial_training_runs/darcy_lossdrop50_physics_time_matched_loss3wall_20260608/darcy_time_matched_summary/darcy_lossdrop50_physics_time_matched_loss3wall_20260608/relative_l2_split_curves.png`
- Attack/loss plot:
  - `adversarial_training_runs/darcy_lossdrop50_physics_time_matched_loss3wall_20260608/darcy_time_matched_summary/darcy_lossdrop50_physics_time_matched_loss3wall_20260608/attack_objective_and_solver_mse_curves.png`
- Eval/train loss progress:
  - `adversarial_training_runs/darcy_lossdrop50_physics_time_matched_loss3wall_20260608/darcy_time_matched_summary/eval_train_loss_progress.png`
- Attack perturbation raw data:
  - `adversarial_training_runs/darcy_lossdrop50_physics_time_matched_loss3wall_20260608/darcy/attack_probe_samples.csv`
  - `adversarial_training_runs/darcy_lossdrop50_physics_time_matched_loss3wall_20260608/darcy/attack_probe_samples/`

## Combined older summary copy

The earlier combined summary directory contains copied PNGs for loss1/loss2/loss3/physics:

- `adversarial_training_runs/darcy_lossdrop50_time_matched_loss123_physics_summary_20260608/`

Note: that combined directory contains the older concurrently-run loss1/loss2 copy, not the corrected single-GPU wall-clock loss1/loss2 outputs.

## Current stockloss3other20 run

- Plot directory: `adversarial_training_runs/darcy_stockloss3other20training_20260608/darcy_time_matched_summary/`
- Attack perturbation raw data: `adversarial_training_runs/darcy_stockloss3other20training_20260608/darcy/attack_probe_samples.csv` and `adversarial_training_runs/darcy_stockloss3other20training_20260608/darcy/attack_probe_samples/`

## Missing visualization class

No Darcy/C-flow standalone PNGs were found locally for perturbation-frequency/spectral maps like the Burgers-specific deeper attack visualizations. The ingredients exist in raw form: `attack_probe_samples.csv` includes `delta_fft_high_freq_ratio`, `delta_fft_spectral_centroid`, `delta_total_variation`, and `delta_sign_change_fraction`, and the `.npz` probe files contain the attacked/clean fields and deltas.

## Image-only comparison bundle update - 2026-06-08

The current flat image-only bundle is at:

- `visualizations/darcy_cflow_loss123_physics_baseline_image_only_20260608/`

Observed current contents:

- `7` polished PNG files.
- No non-PNG files in the bundle directory.
- Exactly four formal objective methods in method curves: `loss1`, `loss2`, `loss3`, and `physics loss`.
- `baseline model` is shown as a reference in clean-evaluation figures and as a category in the generalization summary bar chart.
- `stockloss3other20` remains excluded because it is a separate 20-epoch stock/generalization pipeline run using loss3, not a formal fifth objective.
- Perturbation examples and perturbation-frequency analyses are restored in the bundle.

Current PNGs:

- `comparison_attack_objective_normalized_all_methods.png`
- `darcy_cflow_clean_rmse_loss_methods.png`
- `darcy_cflow_clean_relative_l2_loss_methods.png`
- `darcy_cflow_generalization_final_best_loss_methods.png`
- `darcy_cflow_final_perturbation_examples.png`
- `darcy_cflow_final_delta_radial_spectrum_loss_methods.png`
- `darcy_cflow_perturbation_frequency_metrics_loss_methods.png`

Note: attack-objective plots are attack-generation diagnostics, not clean evaluation or final robustness scores. Perturbation/frequency plots are generated from existing `attack_probe_samples.csv` and final stored `attack_probe_samples/*.npz` artifacts.

