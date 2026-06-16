# Darcy CFlow Organized Release Contents

This folder is restricted to the formal Darcy CFlow final-model clean-evaluation
release.

Kept in the formal release:

- `data/clean_52dataset_metric_long_ranked.csv`
- `data/cflow_clean_52dataset_metric_long_ranked.csv`
- `data/source_tables/source_eval_metrics_artifact_corrected.csv`
- `data/source_tables/source_eval_split_summary_artifact_corrected.csv`
- `data/source_tables/six_method_common_range_eval_metrics.csv`
- `data/source_tables/six_method_common_range_eval_split_summary.csv`
- `figures/main_curves/`
- `figures/polished_report/`
- Supplemental copied attack heatmaps:
  `figures/attack_heatmaps_loss3_advantage_extra15/`
- Supplemental copied attack heatmap data:
  `data/attack_heatmaps_loss3_advantage_extra15/`

Not included in the formal release folder:

- Non-final attack sample tables and NPZ arrays.
- Non-final SVD/Jacobian sample tables and vectors.
- Non-final attack heatmaps should not be used as final robustness rankings.
  The requested `loss3_advantage_extra15` 50-step heatmaps are now copied into
  this release as supplemental visual figures, but their manifest marks them as
  earlier 1000-1100 checkpoint heatmaps rather than final 3000-3500 checkpoint
  heatmaps.
- Derived rankings and reports that mixed non-final diagnostics with final clean
  evaluation metrics.
- Polished report dashboards and panels generated from non-final diagnostics.

For final clean/generalization conclusions, use the clean 52-dataset metric
tables and the clean RMSE/Relative L2 figures in this folder.

Final-model attack/SVD rankings should use the completed final robustness bundle.
Any final-model attack heatmaps should be explicitly labeled as final 3000-3500
checkpoint heatmaps before being used for conclusions.
