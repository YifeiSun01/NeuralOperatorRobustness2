# Burgers final-model evaluation on master_semantic_run1 - 2026-06-08

Status: completed GPU inference evaluation on train/test plus the requested Burgers generalization root.

Observed source data:
- Generalization root: `generalization_datasets/burgers`.
- This root contains exactly 50 Burgers `.pt` files.
- Provenance check found no `attack_objective`, `attack_steps`, or `epsilon_fraction` fields in the Burgers dataset metadata.
- Tier counts from metadata: `{"far_range_pattern": 20, "mid_kernel_spectrum": 13, "near_param_shift": 17}`.
- Caveat: this root is the master semantic generalization suite recorded in `GENERALIZATION_DATASETS.md`; it is deterministic/curated, not an attack-generated root.

Models evaluated:
- `baseline`: epoch `0`, checkpoint `1D_Burgers/trained_models/attack_ready/burgers_nu0.001_fno1d_500/checkpoints/pytorch_fno1d_500.pt`
- `loss1_epoch8000`: epoch `8000`, checkpoint `adversarial_training_runs/burgers_loss3_selective_round03_loss1_continue5000to8000_20260607/burgers/checkpoints/burgers_epoch8000_step024000.pt`
- `loss2_epoch2000`: epoch `2000`, checkpoint `adversarial_training_runs/burgers_loss3_selective_round03_loss2_continue1000to2000_20260606/burgers/checkpoints/burgers_epoch2000_step006000.pt`
- `loss3_epoch1500`: epoch `1500`, checkpoint `adversarial_training_runs/burgers_loss3_selective_round03_loss3_continue1000to1500_20260606/burgers/checkpoints/burgers_epoch1500_step004500.pt`

Output files:
- Per-dataset metrics: `forensics/burgers_master_semantic_final_models_20260608/per_dataset_metrics.csv`
- Split summary: `forensics/burgers_master_semantic_final_models_20260608/split_summary.csv`
- Relative-to-baseline by dataset: `forensics/burgers_master_semantic_final_models_20260608/relative_to_baseline_by_dataset.csv`
- Relative-to-baseline split summary: `forensics/burgers_master_semantic_final_models_20260608/relative_to_baseline_split_summary.csv`
- Dataset provenance audit: `forensics/burgers_master_semantic_final_models_20260608/neutral_dataset_provenance.csv`
- GPU preflight: `forensics/burgers_master_semantic_final_models_20260608/gpu_preflight.json`
- Manifest: `forensics/burgers_master_semantic_final_models_20260608/manifest.json`

Evaluation wall time: `3.31` seconds.

Split Mean Raw Metrics:
| model | split | dataset_count | rmse_dataset_mean | relative_l2_dataset_mean | mae_dataset_mean |
|---|---|---|---|---|---|
| baseline | generalization | 50 | 0.091671 | 0.248715 | 0.0609927 |
| baseline | test | 1 | 0.0095436 | 0.0177549 | 0.00640021 |
| baseline | train | 1 | 0.00898408 | 0.0168978 | 0.00619658 |
| loss1_epoch8000 | generalization | 50 | 0.085163 | 0.212877 | 0.059952 |
| loss1_epoch8000 | test | 1 | 0.000881722 | 0.00164036 | 0.000514324 |
| loss1_epoch8000 | train | 1 | 0.000773952 | 0.00145569 | 0.000471492 |
| loss2_epoch2000 | generalization | 50 | 0.0733419 | 0.188096 | 0.0502736 |
| loss2_epoch2000 | test | 1 | 0.00157905 | 0.00293767 | 0.00083273 |
| loss2_epoch2000 | train | 1 | 0.00146332 | 0.0027523 | 0.000786357 |
| loss3_epoch1500 | generalization | 50 | 0.0740975 | 0.176689 | 0.051457 |
| loss3_epoch1500 | test | 1 | 0.00452757 | 0.00842311 | 0.00167394 |
| loss3_epoch1500 | train | 1 | 0.0044022 | 0.00827993 | 0.00161252 |

Percent Change vs Baseline by Split (negative means lower loss/error):
| model | split | pct_change_rmse_dataset_mean | pct_change_relative_l2_dataset_mean | pct_change_mae_dataset_mean |
|---|---|---|---|---|
| loss1_epoch8000 | generalization | -7.09925 | -14.4095 | -1.7064 |
| loss1_epoch8000 | test | -90.7611 | -90.7611 | -91.964 |
| loss1_epoch8000 | train | -91.3853 | -91.3853 | -92.3911 |
| loss2_epoch2000 | generalization | -19.9944 | -24.3731 | -17.5745 |
| loss2_epoch2000 | test | -83.4543 | -83.4543 | -86.989 |
| loss2_epoch2000 | train | -83.7121 | -83.7121 | -87.3098 |
| loss3_epoch1500 | generalization | -19.1702 | -28.9592 | -15.6343 |
| loss3_epoch1500 | test | -52.5591 | -52.5591 | -73.8456 |
| loss3_epoch1500 | train | -51 | -51 | -73.9772 |

Per-generalization best-model counts across the 50 datasets:
| metric | lowest-loss model counts |
|---|---|
| rmse | `baseline`: 3, `loss1_epoch8000`: 26, `loss2_epoch2000`: 2, `loss3_epoch1500`: 19 |
| relative_l2 | `baseline`: 3, `loss1_epoch8000`: 26, `loss2_epoch2000`: 2, `loss3_epoch1500`: 19 |
| mae | `baseline`: 3, `loss1_epoch8000`: 26, `loss2_epoch2000`: 3, `loss3_epoch1500`: 18 |

Inference:
- This evaluation does not use the round03 loss3-selective attack-generated stress root.
- Use the relative-to-baseline CSVs for per-dataset increase/decrease percentages and the split table above for aggregate train/test/generalization changes.
