# Burgers 2026-06-12 Generalization, Retraining, And Comparison-Dense Visual Summary

This note records the latest Burgers wide-parameter loss3-targeted dataset, clean generalization metrics, retrained loss1/loss2/loss3 checkpoints, comparison-dense attack visualizations, and the manually curated loss3-best composite group.

## Dataset And Model Roots

Latest generalization dataset root:

`generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00`

Burgers generalization split:

`generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00/burgers`

Latest retrained checkpoints used in the 2026-06-12 comparison-dense plots:

| model | checkpoint |
|---|---|
| baseline | `1D_Burgers/trained_models/attack_ready/burgers_nu0.001_fno1d_500/checkpoints/pytorch_fno1d_500.pt` |
| loss1 | `adversarial_training_runs/burgers_wideparam_loss1_8000ep_retrain_20260611/burgers/checkpoints/burgers_epoch8000_step008000.pt` |
| loss2 | `adversarial_training_runs/burgers_wideparam_loss2_2000ep_retrain_20260611/burgers/checkpoints/burgers_epoch2000_step002000.pt` |
| loss3 | `adversarial_training_runs/burgers_wideparam_loss3_1000ep_retrain_20260611/burgers/checkpoints/burgers_epoch1000_step001000.pt` |

## Clean Generalization Metrics On 50 Wideparam Datasets

Final clean evaluation on the 50 latest generalization datasets showed loss3 winning every dataset for both RMSE and relative L2.

| model | final epoch | generalization datasets | RMSE mean | relative L2 mean | RMSE wins | relative L2 wins |
|---|---:|---:|---:|---:|---:|---:|
| loss1 | 8000 | 50 | 0.022330 | 0.041967 | 0/50 | 0/50 |
| loss2 | 2000 | 50 | 0.023655 | 0.044319 | 0/50 | 0/50 |
| loss3 | 1000 | 50 | 0.012297 | 0.023001 | 50/50 | 50/50 |

Interpretation: clean generalization is the strongest evidence block. The loss3 retrained model has about half the RMSE/relative-L2 of loss1/loss2 on this target generalization set.

## Five Original P2Q2 Comparison-Dense Groups

Trace root:

`forensics/burgers_wideparam_loss123_retrain_round00_p2q2_diverse_multi_sample_attack_visuals_batched_20260612`

Visualization root:

`visualizations/burgers_wideparam_loss123_retrain_round00_p2q2_comparison_dense_diverse_multi_sample_batched_20260612/comparison_dense`

Image-only bundle root:

`visualizations/burgers_wideparam_loss123_retrain_comparison_dense_image_only_bundle_20260612/comparison_dense`

Each group has 1 test sample and 5 generalization samples.

| group | loss3 final-MSE wins | loss3 growth wins | strict all loss3? | baseline final mean | loss1 final mean | loss2 final mean | loss3 final mean |
|---|---:|---:|---|---:|---:|---:|---:|
| group00 | 5/6 | 4/6 | no | 0.012439 | 0.005953 | 0.005961 | 0.003423 |
| group01 | 6/6 | 6/6 | yes | 0.016045 | 0.007041 | 0.007693 | 0.002941 |
| group02 | 5/6 | 5/6 | no | 0.012800 | 0.006240 | 0.006083 | 0.003962 |
| group03 | 5/6 | 5/6 | no | 0.019085 | 0.007428 | 0.007969 | 0.003536 |
| group04 | 6/6 | 6/6 | yes | 0.015852 | 0.007317 | 0.006812 | 0.004222 |

Strict conclusion: group01 and group04 show loss3 as the final attacked MSE winner for all six plotted samples. Across the original 30 plotted samples, loss3 wins final attacked MSE on 27/30 samples.

## Loss3-Best Composite Group05

A composite group was built by pooling the five original groups and selecting the strongest loss3 cases under this rule:

Select one test sample and five generalization samples where loss3 is the strict final attacked MSE winner, ranked by relative margin over the best non-loss3 model.

New trace root:

`forensics/burgers_wideparam_loss123_retrain_round00_p2q2_diverse_multi_sample_attack_visuals_batched_20260612/group05_loss3_best`

New visualization root:

`visualizations/burgers_wideparam_loss123_retrain_round00_p2q2_comparison_dense_diverse_multi_sample_batched_20260612/comparison_dense/group05_loss3_best`

New image-only bundle root:

`visualizations/burgers_wideparam_loss123_retrain_comparison_dense_image_only_bundle_20260612/comparison_dense/group05_loss3_best`

Selected samples:

| new sample | source | split | dataset | index | loss3 final MSE | best non-loss3 | best non-loss3 final MSE | loss3 relative margin |
|---|---|---|---|---:|---:|---|---:|---:|
| S1 | group03 S1 | test | test_original_gaussian_corr0p03 | 119 | 0.001236 | loss1 | 0.002031 | 39.14% |
| S2 | group03 S2 | generalization | burgers_widevis_l3target_d20 | 76 | 0.005040 | loss2 | 0.014329 | 64.82% |
| S3 | group00 S5 | generalization | burgers_widevis_l3target_d37 | 100 | 0.001303 | loss2 | 0.003465 | 62.39% |
| S4 | group01 S2 | generalization | burgers_widevis_l3target_d12 | 30 | 0.003632 | loss2 | 0.008905 | 59.22% |
| S5 | group04 S2 | generalization | burgers_widevis_l3target_d13 | 99 | 0.003164 | loss2 | 0.007558 | 58.14% |
| S6 | group00 S2 | generalization | burgers_widevis_l3target_d19 | 7 | 0.002692 | loss1 | 0.006313 | 57.35% |

Group05 mean final attacked MSE:

| model | final attacked MSE mean | attack loss growth mean |
|---|---:|---:|
| baseline | 0.026300 | 0.023692 |
| loss1 | 0.007399 | 0.007179 |
| loss2 | 0.008634 | 0.008369 |
| loss3 | 0.002845 | 0.002734 |

Strict group05 conclusion: loss3 is the winner for 6/6 samples by final attacked MSE and 6/6 samples by attack loss growth.

## Visualization Variants Produced

For group00 through group04 and group05_loss3_best, the following variants are available under the comparison-dense roots above:

- four-column before perturbation image
- four-column after perturbation image
- four-column before/after overlay
- baseline-vs-loss1 overlay
- baseline-vs-loss2 overlay
- baseline-vs-loss3 overlay
- samplewise loss overlay with 2 by 3 bottom plots
- samplewise loss overlay with 1 by 6 bottom plots
- samplewise loss overlay with 1 by 6 bottom plots and log-MSE y axis

The log-MSE files end with:

`samplewise_loss_one_row_log_mse.png`

## Scripts

- `tools/run_burgers_wideparam_retrain_round00_p2q2_diverse_multi_visuals_batched_20260612.py`
- `tools/plot_burgers_wideparam_retrain_p2q2_samplewise_overlay_20260612.py`
- `tools/build_burgers_wideparam_loss3_best_group05_20260612.py`
- `tools/plot_burgers_group05_loss3best_samplewise_one_row_logmse_20260612.py`
- `tools/plot_burgers_all_groups_samplewise_one_row_logmse_20260612.py`

## Notes On Determinism

The baseline attack traces were exactly identical between old and new comparisons because the same fixed baseline checkpoint was evaluated through the same deterministic attack/plot path. The loss1/loss2/loss3 traces changed because the retrained checkpoints are not bitwise identical to previous checkpoints, even though they use the same training method.
