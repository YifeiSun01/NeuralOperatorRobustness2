# Darcy Optimizer-Artifact Corrected Curves

This bundle is a derived diagnostic visualization. Raw files were not overwritten.
The corrected columns bridge the known optimizer-state restart artifact near the stage1/stage2 boundary.

Files:
- Combined CSV: `/workspace/NeuralOperatorRobustness2/outputs/darcy_optimizer_artifact_corrected_20260614/data/all_methods_artifact_corrected_train_steps.csv`
- Manifest CSV: `/workspace/NeuralOperatorRobustness2/outputs/darcy_optimizer_artifact_corrected_20260614/data/artifact_correction_manifest.csv`
- Figures: `/workspace/NeuralOperatorRobustness2/outputs/darcy_optimizer_artifact_corrected_20260614/figures`
- Previous-style corrected loss figure: `/workspace/NeuralOperatorRobustness2/outputs/darcy_optimizer_artifact_corrected_20260614/figures/previous_style_corrected/darcy_adv_training_train_loss_on_adv.png`

Correction method: log10-domain smoothstep bridge with residual scale interpolated between pre-boundary and post-boundary robust statistics.
Rows changed are flagged with `*_is_imputed = 1`.
