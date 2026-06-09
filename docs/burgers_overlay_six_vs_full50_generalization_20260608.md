# Burgers overlay six samples vs full 50 generalization - 2026-06-08

Status: clarified that the P2Q2 before/after overlay uses six selected visualization samples, while the full clean-inference generalization suite contains 50 datasets.

Observed evidence:
- Overlay sample manifest: `forensics/burgers_round03_baseline_vs_final_p2q2_attack_verification_20260607/sample_manifest.json`.
- The overlay figure uses six rows: S1 is one original test sample; S2-S6 are five selected samples from `generalization_datasets_rmse_1p5_3x_all_ns50/burgers`.
- Full root checked on 2026-06-08: `generalization_datasets_rmse_1p5_3x_all_ns50/burgers` contains exactly 50 `.pt` files.
- Full clean-inference source CSV: `forensics/burgers_neutral_generalization_final_models_20260608/per_dataset_metrics.csv`.
- Full split summary CSV: `forensics/burgers_neutral_generalization_final_models_20260608/split_summary.csv`.
- Full percent-change CSV: `forensics/burgers_neutral_generalization_final_models_20260608/relative_to_baseline_split_summary.csv`.

Exact overlay sample sources:
| row | split | dataset | index | source root |
|---|---|---|---:|---|
| S1 | test | `test_original_gaussian_corr0p03` | 23 | original Burgers seed45 test split |
| S2 | generalization | `burgers_near_gaussian_corr0p1` | 17 | `generalization_datasets_rmse_1p5_3x_all_ns50/burgers` |
| S3 | generalization | `burgers_target_gaussian_corr0p2` | 42 | `generalization_datasets_rmse_1p5_3x_all_ns50/burgers` |
| S4 | generalization | `burgers_target_matern_corr0p6_nu3` | 73 | `generalization_datasets_rmse_1p5_3x_all_ns50/burgers` |
| S5 | generalization | `burgers_mid_matern_corr1_nu4` | 29 | `generalization_datasets_rmse_1p5_3x_all_ns50/burgers` |
| S6 | generalization | `burgers_far_sawtooth_add_scale0p3_shift0` | 11 | `generalization_datasets_rmse_1p5_3x_all_ns50/burgers` |

Full 50 clean-inference generalization split metrics:
| model | dataset count | RMSE mean | relative L2 mean | MAE mean |
|---|---:|---:|---:|---:|
| baseline | 50 | 0.017932 | 0.031019 | 0.006034 |
| loss1_epoch8000 | 50 | 0.001770 | 0.003065 | 0.000909 |
| loss2_epoch2000 | 50 | 0.002916 | 0.005033 | 0.001184 |
| loss3_epoch1500 | 50 | 0.006730 | 0.011615 | 0.001877 |

Full 50 clean-inference percent change vs baseline:
| model | RMSE % | relative L2 % | MAE % |
|---|---:|---:|---:|
| loss1_epoch8000 | -90.130 | -90.121 | -84.932 |
| loss2_epoch2000 | -83.741 | -83.774 | -80.376 |
| loss3_epoch1500 | -62.472 | -62.554 | -68.897 |

Full 50 clean-inference lowest-loss counts:
| metric | winner counts |
|---|---|
| RMSE | loss1_epoch8000: 50 |
| relative L2 | loss1_epoch8000: 50 |
| MAE | loss1_epoch8000: 50 |

Selection mechanism evidence:
- Observed from `tools/plot_burgers_p2q2_baseline_vs_epoch1000_attack_gif.py`: S1-S6 are hard-coded in the `SAMPLES` list, with `GEN_ROOT = generalization_datasets_rmse_1p5_3x_all_ns50/burgers`.
- Observed from `tools/plot_burgers_round03_baseline_vs_final_attack_panels.py`: the round03 baseline/loss1/loss2/loss3 overlay imports and reuses that earlier sample loader.
- Inference: these six rows were not freshly selected from the 50 datasets by an automatic round03 criterion such as maximum loss3 advantage. They are selected visualization samples reused for the P2Q2 before/after figure.

Inference:
- The six-row overlay figure is a selected P2Q2 attack/robustness visualization, not the full generalization suite.
- The complete suite behind S2-S6 is 50 datasets under `generalization_datasets_rmse_1p5_3x_all_ns50/burgers`.
- On clean inference over the full 50 datasets, loss1 is best on all 50 by RMSE, relative L2, and MAE.
- The visual loss3 advantage in the overlay is an attacked-output stability result under the P2Q2 perturbation, not a clean generalization-loss result over the full 50 datasets.
