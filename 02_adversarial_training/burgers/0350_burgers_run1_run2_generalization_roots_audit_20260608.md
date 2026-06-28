# Burgers run1/run2 generalization roots audit - 2026-06-08

Status: corrected the earlier incomplete response. The earlier evaluation only covered the locally present `generalization_datasets_rmse_1p5_3x_all_ns50/burgers` root. This audit adds the master semantic root and records the current R2 access limitation.

Update later on 2026-06-08: the R2 access limitation in this note was superseded after the user provided direct S3/R2 credentials. The later R2 audit is recorded in `docs/r2_burgers_three_dataset_conclusion_audit_20260608.md` and `docs/r2_burgers_generalization_data_lookup_20260608.md`.

Observed local/R2 access evidence:
- Local master semantic root `generalization_datasets/burgers` was missing before this turn, but `GENERALIZATION_DATASETS.md` and `GENERALIZATION_MASTER_RECORD.md` record it as a 50-dataset neutral semantic suite.
- R2 sync record `docs/r2_and_github_sync_20260608.md` says `generalization_datasets_*` and `generalization_eval*` were included in the 2026-06-08 R2 snapshot.
- Current shell cannot list R2 because `rclone` has no configured `R2` remote: observed error `Config file "/root/.config/rclone/rclone.conf" not found` and `didn't find section in config file`.
- The local directory `generalization_datasets_rmse_1p5_3x_all` is not present; only documentation directories such as `generalization_eval_rmse_1p5_3x_all` remain locally, and those currently contain only `GENERALIZATION_EVALUATION.md`.

Roots evaluated now/available locally:
- `master_semantic_run1`: `generalization_datasets/burgers`. Regenerated locally with `tools/generate_generalization_datasets.py --tasks burgers --output-root generalization_datasets --burgers-batch-size 200` because the root was documented but absent locally. It contains exactly 50 Burgers `.pt` files, tier split `near_param_shift=17`, `mid_kernel_spectrum=13`, `far_range_pattern=20`, and no attack metadata fields.
- `target_band_ns50_current`: `generalization_datasets_rmse_1p5_3x_all_ns50/burgers`. This was the root evaluated in `docs/burgers_neutral_generalization_final_models_20260608.md`. It contains exactly 50 Burgers `.pt` files, tier split `near_param_shift=8`, `mid_kernel_spectrum=2`, `far_range_pattern=2`, `target_loss_param_shift=23`, `target_loss_kernel_shift=15`, and no attack metadata fields.

GPU and generation evidence:
- Preflight before regenerating master semantic Burgers: `forensics/burgers_master_generalization_regen_20260608/gpu_preflight.json`.
- Observed GPU path: Tesla V100-SXM2-32GB, PyTorch `2.8.0+cu126`, CUDA `12.6`, compute capability `(7, 0)`, PyTorch arch list including `sm_70`, and JAX backend `gpu` with device `cuda:0`.
- Master root generation completed 50/50 Burgers datasets in about 25 seconds.

Model checkpoints evaluated on both roots:
- baseline: `1D_Burgers/trained_models/attack_ready/burgers_nu0.001_fno1d_500/checkpoints/pytorch_fno1d_500.pt`
- loss1: `adversarial_training_runs/burgers_loss3_selective_round03_loss1_continue5000to8000_20260607/burgers/checkpoints/burgers_epoch8000_step024000.pt`
- loss2: `adversarial_training_runs/burgers_loss3_selective_round03_loss2_continue1000to2000_20260606/burgers/checkpoints/burgers_epoch2000_step006000.pt`
- loss3: `adversarial_training_runs/burgers_loss3_selective_round03_loss3_continue1000to1500_20260606/burgers/checkpoints/burgers_epoch1500_step004500.pt`

Output files:
- Master/run1 report: `docs/burgers_master_semantic_final_models_20260608.md`
- Master/run1 metrics: `forensics/burgers_master_semantic_final_models_20260608/per_dataset_metrics.csv`
- Master/run1 split summary: `forensics/burgers_master_semantic_final_models_20260608/split_summary.csv`
- Master/run1 percent changes: `forensics/burgers_master_semantic_final_models_20260608/relative_to_baseline_split_summary.csv`
- Target-band/ns50 report: `docs/burgers_neutral_generalization_final_models_20260608.md`
- Target-band/ns50 metrics: `forensics/burgers_neutral_generalization_final_models_20260608/per_dataset_metrics.csv`
- Flexible evaluator added: `tools/evaluate_burgers_final_models_generalization_root_20260608.py`

Generalization split mean raw metrics:
| root | model | RMSE | relative L2 | MAE |
|---|---|---:|---:|---:|
| master_semantic_run1 | baseline | 0.091671 | 0.248715 | 0.060993 |
| master_semantic_run1 | loss1_epoch8000 | 0.085163 | 0.212877 | 0.059952 |
| master_semantic_run1 | loss2_epoch2000 | 0.073342 | 0.188096 | 0.050274 |
| master_semantic_run1 | loss3_epoch1500 | 0.074097 | 0.176689 | 0.051457 |
| target_band_ns50_current | baseline | 0.017932 | 0.031019 | 0.006034 |
| target_band_ns50_current | loss1_epoch8000 | 0.001770 | 0.003065 | 0.000909 |
| target_band_ns50_current | loss2_epoch2000 | 0.002916 | 0.005033 | 0.001184 |
| target_band_ns50_current | loss3_epoch1500 | 0.006730 | 0.011615 | 0.001877 |

Generalization percent change vs baseline:
| root | model | RMSE % | relative L2 % | MAE % |
|---|---|---:|---:|---:|
| master_semantic_run1 | loss1_epoch8000 | -7.099 | -14.409 | -1.706 |
| master_semantic_run1 | loss2_epoch2000 | -19.994 | -24.373 | -17.574 |
| master_semantic_run1 | loss3_epoch1500 | -19.170 | -28.959 | -15.634 |
| target_band_ns50_current | loss1_epoch8000 | -90.130 | -90.121 | -84.932 |
| target_band_ns50_current | loss2_epoch2000 | -83.741 | -83.774 | -80.376 |
| target_band_ns50_current | loss3_epoch1500 | -62.472 | -62.554 | -68.897 |

Per-generalization lowest-loss counts:
| root | metric | counts |
|---|---|---|
| master_semantic_run1 | RMSE | baseline 3, loss1 26, loss2 2, loss3 19 |
| master_semantic_run1 | relative L2 | baseline 3, loss1 26, loss2 2, loss3 19 |
| master_semantic_run1 | MAE | baseline 3, loss1 26, loss2 3, loss3 18 |
| target_band_ns50_current | RMSE | loss1 50 |
| target_band_ns50_current | relative L2 | loss1 50 |
| target_band_ns50_current | MAE | loss1 50 |

Inference:
- The earlier answer was incomplete because it evaluated only `target_band_ns50_current` and did not cover the master semantic root.
- On `master_semantic_run1`, loss3 has the best aggregate relative L2, loss2 has the best aggregate RMSE/MAE, and loss1 wins the most individual generalization datasets.
- On `target_band_ns50_current`, loss1 is clearly best by aggregate metrics and wins all 50 individual generalization datasets.
- Neither evaluated root is the round03 loss3-selective attack-generated stress root.
- A possible third uploaded root, likely related to `generalization_datasets_rmse_1p5_3x_all`, is not currently present locally and cannot be listed from R2 until the R2/rclone configuration is restored on this machine.

Remaining work:
- Restore R2 access or provide the missing local root if the exact third upload/root should also be evaluated.
- After restoring that root, run `tools/evaluate_burgers_final_models_generalization_root_20260608.py` with a fresh output/report path.
