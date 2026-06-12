# Darcy Cflow R2 Sync - 2026-06-12

R2 upload target used for the Darcy Cflow result bundle:

```text
bucket: neural-operator-robustness
prefix: machine-sync/NeuralOperatorRobustness2-selected/darcy_cflow_20260612
```

Uploaded scope:

```text
analysis_outputs/darcy_metric_correlation_25samples_20260612_25samples_loss3attack20_corr_chunk64
visualizations/darcy_metric_correlation_25samples_20260612_25samples_loss3attack20_corr_chunk64
analysis_outputs/darcy_metric_correlation_10samples_20260612_10samples_loss3attack20_corr_chunk64
visualizations/darcy_metric_correlation_10samples_20260612_10samples_loss3attack20_corr_chunk64
analysis_outputs/darcy_five_model_batch_ranked_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished
analysis_outputs/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_*
visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_*
generalization_datasets_darcy_binary_loss3targeted_20260611
docs/darcy_cflow_results_20260612.md
selected Darcy Cflow py/sh scripts under tools/
```

Verification after upload:

```text
Total objects: 367
Total size: 243.923 MiB
```

Stage2 continuation upload, added after the 3000-ish epoch runs finished:

```text
visualizations/darcy_loss123physics_full50_timematched_stage2_20260612_stage2_2000_from_1000c
docs/darcy_loss123physics_full50_timematched_stage2_20260612_stage2_2000_from_1000c*.md
adversarial_training_runs/darcy_binary_loss3targeted_loss*_continue*stage2_2000_from_1000c metadata only
adversarial_training_runs/darcy_binary_loss3targeted_physics_continue*stage2_2000_from_1000c metadata only
```

Stage2 visualization verification:

```text
Total objects: 131
Total size: 228.017 MiB
```

Notes:

```text
The temporary rclone config was removed after upload.
Large adversarial_training_runs checkpoints were not included in this R2 bundle; the stage2 training directories alone are about 42 GiB and should be archived separately only if explicitly needed.
GitHub commit intentionally keeps large binary scientific artifacts out of Git according to .gitignore.
```
