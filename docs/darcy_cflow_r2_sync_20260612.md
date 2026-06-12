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


Full training artifact upload, added after the explicit full checkpoint/archive request:

```text
adversarial_training_runs/darcy_binary_loss3targeted_loss1_1000ep_full50_timematched_20260612_full50_timematched_1000c
adversarial_training_runs/darcy_binary_loss3targeted_loss2_1026ep_full50_timematched_20260612_full50_timematched_1000c
adversarial_training_runs/darcy_binary_loss3targeted_loss3_1011ep_full50_timematched_20260612_full50_timematched_1000c
adversarial_training_runs/darcy_binary_loss3targeted_physics_1040ep_full50_timematched_20260612_full50_timematched_1000c
adversarial_training_runs/darcy_binary_loss3targeted_loss1_continue2000ep_from_1000ep_full50_timematched_20260612_stage2_2000_from_1000c
adversarial_training_runs/darcy_binary_loss3targeted_loss2_continue2053ep_from_1026ep_full50_timematched_20260612_stage2_2000_from_1000c
adversarial_training_runs/darcy_binary_loss3targeted_loss3_continue2022ep_from_1011ep_full50_timematched_20260612_stage2_2000_from_1000c
adversarial_training_runs/darcy_binary_loss3targeted_physics_continue2081ep_from_1040ep_full50_timematched_20260612_stage2_2000_from_1000c
```

This full upload includes the large `.pt` checkpoints and `.npz` attack probe/sample files from the formal first-stage and stage2 Darcy Cflow runs.

Full artifact verification after upload:

```text
Total objects: 12,467
Total size: 61.318 GiB
```

Full 0..3000 Burgers-style Darcy image-only bundle upload, added after the
Darcy figures were regenerated as a single stitched 0..3000 trajectory:

```text
visualizations/darcy_loss123physics_full3000_burgers_image_only_bundle_20260612
analysis_outputs/darcy_full3000_burgers_image_only_bundle_20260612_work
docs/darcy_loss123physics_full3000_burgers_image_only_bundle_20260612.md
tools/build_darcy_full3000_burgers_image_only_bundle_20260612.py
```

Image-only bundle verification after upload:

```text
Total objects: 116
Total size: 105.931 MiB
```

Stitched work-directory verification after upload, excluding the already-uploaded
per-epoch `.npz` attack-probe symlinks:

```text
Total objects: 202
Total size: 655.146 MiB
```

Notes:

```text
The temporary rclone config was removed after upload.
Large adversarial_training_runs checkpoints are now included in the R2 bundle under the same Darcy Cflow prefix.
GitHub commit intentionally keeps large binary scientific artifacts out of Git according to .gitignore.
```
