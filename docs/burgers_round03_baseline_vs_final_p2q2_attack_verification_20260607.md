# Burgers Round03 Baseline vs Final P2Q2 Attack Verification - 2026-06-07

## Status

Generated per-model before/after p=2,q=2 RMS-L2 attack verification PNGs for round03 `loss1`, `loss2`, and `loss3` final self-training models at `2026-06-07T20:44:37Z`.

## Observed Evidence

- Summary JSON: `forensics/burgers_round03_baseline_vs_final_p2q2_attack_verification_20260607/summary.json`.
- Attack: 100 PGD-style RMS-L2 steps, epsilon RMS `0.12`, alpha RMS `0.012`, same six samples as the earlier p2q2 before/after visualization.
- Device: `Tesla V100-SXM2-32GB` with PyTorch `2.8.0+cu126` and CUDA `12.6`.
- Baseline checkpoint: `/workspace/NeuralOperatorRobustness2/1D_Burgers/trained_models/attack_ready/burgers_nu0.001_fno1d_500/checkpoints/pytorch_fno1d_500.pt`.

| model | final model label | baseline final MSE | target final MSE | target / baseline | before PNG | after PNG |
| --- | --- | ---: | ---: | ---: | --- | --- |
| loss1 | Round03 loss1 self-training final model, epoch 8000 | 0.026507497 | 0.0062939017 | 0.2374 | `/workspace/NeuralOperatorRobustness2/visualizations/burgers_loss3_selective_round03_longtraining_comparison_dense_image_only_bundle_20260607/loss1/polished_report/round03_loss1_baseline_vs_epoch8000_p2q2_attack_step000_before_perturbation.png` | `/workspace/NeuralOperatorRobustness2/visualizations/burgers_loss3_selective_round03_longtraining_comparison_dense_image_only_bundle_20260607/loss1/polished_report/round03_loss1_baseline_vs_epoch8000_p2q2_attack_step100_after_perturbation.png` |
| loss2 | Round03 loss2 self-training final model, epoch 2000 | 0.026507497 | 0.0092682792 | 0.3496 | `/workspace/NeuralOperatorRobustness2/visualizations/burgers_loss3_selective_round03_longtraining_comparison_dense_image_only_bundle_20260607/loss2/polished_report/round03_loss2_baseline_vs_epoch2000_p2q2_attack_step000_before_perturbation.png` | `/workspace/NeuralOperatorRobustness2/visualizations/burgers_loss3_selective_round03_longtraining_comparison_dense_image_only_bundle_20260607/loss2/polished_report/round03_loss2_baseline_vs_epoch2000_p2q2_attack_step100_after_perturbation.png` |
| loss3 | Round03 loss3 self-training final model, epoch 1500 | 0.026507497 | 0.0029296223 | 0.1105 | `/workspace/NeuralOperatorRobustness2/visualizations/burgers_loss3_selective_round03_longtraining_comparison_dense_image_only_bundle_20260607/loss3/polished_report/round03_loss3_baseline_vs_epoch1500_p2q2_attack_step000_before_perturbation.png` | `/workspace/NeuralOperatorRobustness2/visualizations/burgers_loss3_selective_round03_longtraining_comparison_dense_image_only_bundle_20260607/loss3/polished_report/round03_loss3_baseline_vs_epoch1500_p2q2_attack_step100_after_perturbation.png` |

## Inference

For the same attack budget and same six input conditions, all three round03 self-training final models have lower final attack MSE than the baseline. This is consistent with the intended visual read: after self-training, the p2q2 attack produces a smaller model-vs-solver discrepancy than it does on the baseline model. In this six-sample visualization, `loss3` is the hardest among the three to drive away from the solver, followed by `loss1`, then `loss2`.

## Initial-Condition Source Check - 2026-06-07T20:48:36Z

Observed from `tools/plot_burgers_round03_baseline_vs_final_attack_panels.py`: the round03 p2q2 before/after figure calls `base_mod.load_samples()`, reusing the sample list from `tools/plot_burgers_p2q2_baseline_vs_epoch1000_attack_gif.py`.

Observed from `forensics/burgers_round03_baseline_vs_final_p2q2_attack_verification_20260607/sample_manifest.json`: the six initial conditions are one `test` sample and five `generalization` samples. No `train` sample is used.

Observed sample sources:

- `test_original_gaussian_corr0p03`, index `23`, from the original Burgers `seed45_test.pt` test split.
- `burgers_near_gaussian_corr0p1`, index `17`, from `generalization_datasets_rmse_1p5_3x_all_ns50/burgers`.
- `burgers_target_gaussian_corr0p2`, index `42`, from `generalization_datasets_rmse_1p5_3x_all_ns50/burgers`.
- `burgers_target_matern_corr0p6_nu3`, index `73`, from `generalization_datasets_rmse_1p5_3x_all_ns50/burgers`.
- `burgers_mid_matern_corr1_nu4`, index `29`, from `generalization_datasets_rmse_1p5_3x_all_ns50/burgers`.
- `burgers_far_sawtooth_add_scale0p3_shift0`, index `11`, from `generalization_datasets_rmse_1p5_3x_all_ns50/burgers`.

Inference: this visualization is a mixed test/generalization probe: `1/6` test and `5/6` generalization. It is not a train-set visualization.



## Loss3 After-Attack Visual Interpretation - 2026-06-07T21:13:26Z

Observed source image: `visualizations/burgers_loss3_selective_round03_longtraining_comparison_dense_image_only_bundle_20260607/loss3/polished_report/round03_loss3_baseline_vs_epoch1500_p2q2_attack_step100_after_perturbation.png`.

Observed numeric source: `forensics/burgers_round03_baseline_vs_final_p2q2_attack_verification_20260607/loss3/summary.json`, `attack_loss_curves.csv`, and `attack_traces.npz`.

Observed from the plot code in `tools/plot_burgers_round03_baseline_vs_final_attack_panels.py`: baseline and target panels use shared per-row y-limits for delta, input, output, and diff panels, so the visual comparison is not caused by separate left/right autoscaling.

Observed at attack step 100: both baseline and loss3 final model use the same final perturbation RMS budget, approximately `0.12`. Mean final MSE is `0.0265074968` for baseline and `0.0029296223` for the loss3 epoch1500 model. Mean final RMS of the model-solver difference is `0.1535285` for baseline and `0.0525339` for loss3. Global max absolute model-solver difference is `1.1547195` for baseline and `0.8558748` for loss3.

Observed per-sample final MSE target/baseline ratios: S1 `0.8445`, S2 `0.1003`, S3 `0.0805`, S4 `0.0622`, S5 `0.0920`, S6 `0.1905`.

Inference: the loss3 epoch1500 model is not receiving a smaller input attack; rather, under the same input perturbation budget, its attacked output stays much closer to the solver. The visual reduction should be especially clear in the output-fill and diff panels for the five generalization samples, while S1 is a weaker visual case.


## Combined Baseline/Loss1/Loss2/Loss3 Panels - 2026-06-07T21:21:56Z

Observed action: generated combined visualization PNGs from existing saved traces only; no new training and no new attack optimization was run.

Source script: `tools/plot_burgers_round03_p2q2_combined_attack_panels.py`.

Numeric sources: `forensics/burgers_round03_baseline_vs_final_p2q2_attack_verification_20260607/loss1/attack_traces.npz`, `loss2/attack_traces.npz`, `loss3/attack_traces.npz`, and `sample_manifest.json`.

Observed trace consistency: the script asserts that the clean samples and baseline traces are bytewise/numerically consistent across the three loss trace files before drawing the shared-baseline panels.

Observed output files:
  - `visualizations/burgers_loss3_selective_round03_longtraining_comparison_dense_image_only_bundle_20260607/comparison_dense/round03_p2q2_baseline_loss1_loss2_loss3_step000_before_perturbation_four_column.png`
  - `visualizations/burgers_loss3_selective_round03_longtraining_comparison_dense_image_only_bundle_20260607/comparison_dense/round03_p2q2_baseline_loss1_loss2_loss3_step100_after_perturbation_four_column.png`
  - `visualizations/burgers_loss3_selective_round03_longtraining_comparison_dense_image_only_bundle_20260607/comparison_dense/round03_p2q2_baseline_loss1_loss2_loss3_before_after_overlay_four_column.png`
  - `visualizations/burgers_loss3_selective_round03_longtraining_comparison_dense_image_only_bundle_20260607/comparison_dense/round03_p2q2_baseline_vs_loss1_before_after_overlay_two_column.png`
  - `visualizations/burgers_loss3_selective_round03_longtraining_comparison_dense_image_only_bundle_20260607/comparison_dense/round03_p2q2_baseline_vs_loss2_before_after_overlay_two_column.png`
  - `visualizations/burgers_loss3_selective_round03_longtraining_comparison_dense_image_only_bundle_20260607/comparison_dense/round03_p2q2_baseline_vs_loss3_before_after_overlay_two_column.png`

Observed image dimensions: four-column images are `7200x2528`; two-column overlay images are `3600x2528`. Pixel standard-deviation checks were nonzero for all six PNGs.

Inference: the four-column before/after images remove the repeated baseline half-panel and put `baseline`, `loss1`, `loss2`, and `loss3` on the same row-wise y-limits for direct comparison. The overlay images use faint dashed curves for before perturbation and darker solid curves for after 100 attack steps.


## All-Model P2Q2 Comparison Correction - 2026-06-07T21:24:43Z

Observed correction: the interpretation must compare `loss3` against `baseline`, `loss1`, and `loss2`, not only against `baseline`.

Observed numeric sources: `forensics/burgers_round03_baseline_vs_final_p2q2_attack_verification_20260607/loss1/attack_traces.npz`, `loss2/attack_traces.npz`, `loss3/attack_traces.npz`, and `summary.json`.

Observed before perturbation at step 0, mean MSE: baseline `0.0002251479`, loss1 `0.0000075847`, loss2 `0.0000088161`, loss3 `0.0000356045`. Before the attack, loss1/loss2 have lower clean model-solver MSE than loss3 on this six-sample set.

Observed after perturbation at step 100, all with final `delta_rms` approximately `0.12`: baseline mean MSE `0.0265074968`, loss1 `0.0062939017`, loss2 `0.0092682792`, loss3 `0.0029296223`. Mean model-solver diff RMS: baseline `0.1535285`, loss1 `0.0706373`, loss2 `0.0884283`, loss3 `0.0525339`.

Observed ratios at step100: loss3/baseline mean MSE `0.1105`, loss3/loss1 `0.4655`, loss3/loss2 `0.3161`. Thus loss3 has about `2.15x` lower final mean MSE than loss1 and about `3.16x` lower final mean MSE than loss2 on this p2q2 attack set.

Observed per-sample final MSE rankings: S1 loss1 < loss2 < loss3 < baseline; S2 loss1 < loss3 < loss2 < baseline; S3 loss3 < loss1 < loss2 < baseline; S4 loss3 < loss1 < loss2 < baseline; S5 loss3 < loss1 < loss2 < baseline; S6 loss3 < loss1 < loss2 < baseline.

Inference: loss3 is not the best clean-fit model before perturbation, but under the same p2q2 step100 attack it has the smallest mean output deviation and wins on four of six samples. The visual claim should therefore be phrased as an all-model robustness comparison: baseline is easiest to drive away from the solver, loss2 is worse than loss1 on average here, loss1 is strong on S1/S2, and loss3 has the best overall attacked-output stability.


## Overlay Before-Curve Alpha Correction - 2026-06-07T21:30:51Z

Observed issue: the first overlay version rendered before-perturbation main curves with alpha values `0.26`, `0.28`, and `0.30`, while after-perturbation curves used approximately `0.92..0.94`. The before curves were therefore too faint for visual inspection.

Correction: updated `tools/plot_burgers_round03_p2q2_combined_attack_panels.py` so all before-perturbation dashed main curves in overlay panels use `alpha=0.80`: delta, attacked input, solver before, model before, and model-solver diff before. Low-alpha fill bands remain light to avoid covering the after-step100 solid curves.

Regenerated overlay outputs:
  - `visualizations/burgers_loss3_selective_round03_longtraining_comparison_dense_image_only_bundle_20260607/comparison_dense/round03_p2q2_baseline_loss1_loss2_loss3_before_after_overlay_four_column.png`
  - `visualizations/burgers_loss3_selective_round03_longtraining_comparison_dense_image_only_bundle_20260607/comparison_dense/round03_p2q2_baseline_vs_loss1_before_after_overlay_two_column.png`
  - `visualizations/burgers_loss3_selective_round03_longtraining_comparison_dense_image_only_bundle_20260607/comparison_dense/round03_p2q2_baseline_vs_loss2_before_after_overlay_two_column.png`
  - `visualizations/burgers_loss3_selective_round03_longtraining_comparison_dense_image_only_bundle_20260607/comparison_dense/round03_p2q2_baseline_vs_loss3_before_after_overlay_two_column.png`

Observed verification: overlay PNG dimensions remain `7200x2528` for the four-column overlay and `3600x2528` for the two-column overlays. The image-only bundle still contains `59` PNG files and `0` non-PNG files.

Inference: the overlay figures now keep the before/after distinction via dashed vs solid line style while making the before curves visible enough for direct comparison.
