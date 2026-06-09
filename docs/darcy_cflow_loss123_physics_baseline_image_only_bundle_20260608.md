# Darcy/C-flow polished loss and perturbation image bundle - 2026-06-08

Status: regenerated the Darcy/C-flow visualization bundle with baseline model reference and restored perturbation/frequency plots. No new training, attack generation, model-forward evaluation, or solver-forward evaluation was run; figures were regenerated from existing local CSV/NPZ artifacts.

## Current Output

Observed output directory:

- `visualizations/darcy_cflow_loss123_physics_baseline_image_only_20260608/`

Observed verification:

- `7` PNG files are present.
- `find visualizations/darcy_cflow_loss123_physics_baseline_image_only_20260608 -maxdepth 1 -type f ! -name '*.png' -print` returned no files, so the directory is image-only.
- The regenerated PNGs passed a PIL nonblank check by image dimensions and nonzero RGB standard deviation.
- Syntax check passed: `adv_robust/bin/python -m py_compile tools/build_darcy_cflow_image_only_bundle_20260608.py`.

Current PNGs:

- `comparison_attack_objective_normalized_all_methods.png`
- `darcy_cflow_clean_rmse_loss_methods.png`
- `darcy_cflow_clean_relative_l2_loss_methods.png`
- `darcy_cflow_generalization_final_best_loss_methods.png`
- `darcy_cflow_final_perturbation_examples.png`
- `darcy_cflow_final_delta_radial_spectrum_loss_methods.png`
- `darcy_cflow_perturbation_frequency_metrics_loss_methods.png`

## Figure Groups

Clean evaluation figures:

- `darcy_cflow_clean_rmse_loss_methods.png`
- `darcy_cflow_clean_relative_l2_loss_methods.png`
- `darcy_cflow_generalization_final_best_loss_methods.png`

Observed meaning:

- Clean RMSE and clean Relative L2 are both clean prediction-error metrics on train/test/generalization inputs.
- RMSE is an absolute error scale.
- Relative L2 is normalized by the target solution norm and is therefore scale-normalized.
- The clean RMSE and clean Relative L2 plots have the same structure because they are two metrics computed from the same clean evaluation passes; they answer the same comparison question under different error normalizations.
- The summary bar plot includes `baseline model` explicitly as the first reference category.

Attack-generation objective figure:

- `comparison_attack_objective_normalized_all_methods.png`

Observed meaning:

- `Perturbed-input objective value` is the attack objective evaluated after the attack perturbation has been applied.
- `Attack-induced objective increase` is the attack objective after perturbation minus the corresponding objective before perturbation.
- These are adversarial-training attack-generation diagnostics, not clean prediction-error metrics and not final robustness scores.
- Because loss1/loss2/loss3/physics have different objective definitions and units, the plotted values are normalized within each method to that method's first finite nonzero value. The figure should be used to read within-method trends, not to compare absolute y-values across methods.

Perturbation analysis figures:

- `darcy_cflow_final_perturbation_examples.png`
- `darcy_cflow_final_delta_radial_spectrum_loss_methods.png`
- `darcy_cflow_perturbation_frequency_metrics_loss_methods.png`

Observed meaning:

- The perturbation example figure shows one stored final attack probe per objective: clean coefficient, attacked coefficient, and the delta perturbation.
- The radial spectrum figure summarizes final stored perturbation power by normalized radial frequency.
- The frequency-metric figure tracks stored attack-probe metrics over training: high-frequency energy share, spectral centroid, total variation, and sign-change fraction.

## Methods Plotted

Exactly four formal objective methods are plotted as method curves:

- `loss1`
- `loss2`
- `loss3`
- `physics loss`

Baseline handling:

- `baseline model` is included as a reference in clean-evaluation plots and explicitly as a category in the summary bar chart.
- `baseline model` is not a fifth training objective.

Excluded from method comparison:

- `stockloss3other20` is not plotted. It is a separate 20-epoch stock/generalization pipeline run using loss3, not one of the four formal Darcy/C-flow loss methods.

## Source Artifacts

Observed source runs:

- loss1: `adversarial_training_runs/darcy_lossdrop50_loss1_single_gpu_time_matched_loss3wall_20260608/`
- loss2: `adversarial_training_runs/darcy_lossdrop50_loss2_single_gpu_time_matched_loss3wall_20260608/`
- loss3: `adversarial_training_runs/darcy_lossdrop50_loss3_500ep_fromscreen_20260607/`
- physics/loss4: `adversarial_training_runs/darcy_lossdrop50_physics_time_matched_loss3wall_20260608/`

Exact observed numeric source paths:

- loss1 eval: `adversarial_training_runs/darcy_lossdrop50_loss1_single_gpu_time_matched_loss3wall_20260608/darcy/eval_split_summary.csv`
- loss1 attack: `adversarial_training_runs/darcy_lossdrop50_loss1_single_gpu_time_matched_loss3wall_20260608/darcy/attack_epoch_summary.csv`
- loss1 probes: `adversarial_training_runs/darcy_lossdrop50_loss1_single_gpu_time_matched_loss3wall_20260608/darcy/attack_probe_samples.csv` and `adversarial_training_runs/darcy_lossdrop50_loss1_single_gpu_time_matched_loss3wall_20260608/darcy/attack_probe_samples/`
- loss1 timing: `adversarial_training_runs/darcy_lossdrop50_loss1_single_gpu_time_matched_loss3wall_20260608/darcy/train_steps.csv` and `adversarial_training_runs/darcy_lossdrop50_loss1_single_gpu_time_matched_loss3wall_20260608/darcy/summary.json`
- loss2 eval: `adversarial_training_runs/darcy_lossdrop50_loss2_single_gpu_time_matched_loss3wall_20260608/darcy/eval_split_summary.csv`
- loss2 attack: `adversarial_training_runs/darcy_lossdrop50_loss2_single_gpu_time_matched_loss3wall_20260608/darcy/attack_epoch_summary.csv`
- loss2 probes: `adversarial_training_runs/darcy_lossdrop50_loss2_single_gpu_time_matched_loss3wall_20260608/darcy/attack_probe_samples.csv` and `adversarial_training_runs/darcy_lossdrop50_loss2_single_gpu_time_matched_loss3wall_20260608/darcy/attack_probe_samples/`
- loss2 timing: `adversarial_training_runs/darcy_lossdrop50_loss2_single_gpu_time_matched_loss3wall_20260608/darcy/train_steps.csv` and `adversarial_training_runs/darcy_lossdrop50_loss2_single_gpu_time_matched_loss3wall_20260608/darcy/summary.json`
- loss3 eval: `adversarial_training_runs/darcy_lossdrop50_loss3_500ep_fromscreen_20260607/darcy/eval_split_summary.csv`
- loss3 attack: `adversarial_training_runs/darcy_lossdrop50_loss3_500ep_fromscreen_20260607/darcy/attack_epoch_summary.csv`
- loss3 probes: `adversarial_training_runs/darcy_lossdrop50_loss3_500ep_fromscreen_20260607/darcy/attack_probe_samples.csv` and `adversarial_training_runs/darcy_lossdrop50_loss3_500ep_fromscreen_20260607/darcy/attack_probe_samples/`
- loss3 timing: `adversarial_training_runs/darcy_lossdrop50_loss3_500ep_fromscreen_20260607/darcy/train_steps.csv` and `adversarial_training_runs/darcy_lossdrop50_loss3_500ep_fromscreen_20260607/darcy/summary.json`
- physics eval: `adversarial_training_runs/darcy_lossdrop50_physics_time_matched_loss3wall_20260608/darcy/eval_split_summary.csv`
- physics attack: `adversarial_training_runs/darcy_lossdrop50_physics_time_matched_loss3wall_20260608/darcy/attack_epoch_summary.csv`
- physics probes: `adversarial_training_runs/darcy_lossdrop50_physics_time_matched_loss3wall_20260608/darcy/attack_probe_samples.csv` and `adversarial_training_runs/darcy_lossdrop50_physics_time_matched_loss3wall_20260608/darcy/attack_probe_samples/`
- physics timing: `adversarial_training_runs/darcy_lossdrop50_physics_time_matched_loss3wall_20260608/darcy/train_steps.csv` and `adversarial_training_runs/darcy_lossdrop50_physics_time_matched_loss3wall_20260608/darcy/summary.json`

## Notes

Observed from the CSVs: `eval_split_summary.csv` records epoch/global-step split metrics but not cumulative wall time. The comparison plots therefore use wall minutes reconstructed from cumulative `train_steps.csv:step_wall_sec`, scaled to the run's recorded `summary.json:elapsed_seconds` when available.

Inference from the run design: this scaled wall-minute x-axis is a visualization aid for the time-matched runs, not exact per-evaluation timestamp evidence.

## Tool

Observed local tool used to build the current bundle:

- `tools/build_darcy_cflow_image_only_bundle_20260608.py`

Command used:

```bash
adv_robust/bin/python tools/build_darcy_cflow_image_only_bundle_20260608.py
```

## Remaining Work

- If exact wall-clock x-axis values at evaluation time are required, future runs should log cumulative wall timestamps directly into `eval_split_summary.csv` or an adjacent evaluation timing CSV.
- If a paper-ready attack-robustness metric figure is needed, generate it separately from these attack-generation objective diagnostics.

## 2026-06-09 Layout Correction

Observed correction:

- Regenerated `darcy_cflow_final_perturbation_examples.png` after the title/subtitle overlapped the first row of image panels.
- The plotting script now reserves a fixed top title/subtitle band and fixed right-side colorbar bands for this figure.
- The regenerated bundle still contains `7` PNG files and no non-PNG files.
- PIL verification for `darcy_cflow_final_perturbation_examples.png`: image size `2470 x 2605`, nonzero RGB standard deviations `[66.79, 86.64, 81.73]`.

