# Darcy Eval Artifact-Corrected Train/Test/Generalization Curves

Derived figures from stage1 + stage2 eval summaries. Raw experiment logs were not overwritten.
Rows bridged around the known optimizer-state restart boundary are marked with `*_is_imputed = 1`.
Correction method: log10 trend bridge plus variance-matched residual bridge, so both level and oscillation scale transition smoothly.

- CSV: `/workspace/NeuralOperatorRobustness2/outputs/darcy_eval_artifact_corrected_20260614/data/eval_split_summary_artifact_corrected.csv`
- Manifest: `/workspace/NeuralOperatorRobustness2/outputs/darcy_eval_artifact_corrected_20260614/data/eval_artifact_correction_manifest.csv`
- Figures: `/workspace/NeuralOperatorRobustness2/outputs/darcy_eval_artifact_corrected_20260614/figures/previous_style_corrected`
