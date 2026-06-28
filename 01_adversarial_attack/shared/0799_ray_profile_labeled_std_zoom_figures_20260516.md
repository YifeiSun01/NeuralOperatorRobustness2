# Ray Profile Labeled Std Zoom Figures - 2026-05-16

Data source: corrected fixed-sign PGD100 Ray Profile output.

Dense zoom CSV: `/workspace/NeuralOperatorRobustness2/forensics/loss3_ray_profile_pgd_20260516/fno_nu0p001_gpu_v100_batch100_steps100_zero_fixedsign/ray_profile_dense_zoom_0to1.csv`
Dense aggregate CSV: `/workspace/NeuralOperatorRobustness2/forensics/loss3_ray_profile_pgd_20260516/fno_nu0p001_gpu_v100_batch100_steps100_zero_fixedsign/ray_profile_dense_zoom_0to1_aggregate_curves.csv`
Crossover summary CSV: `/workspace/NeuralOperatorRobustness2/forensics/loss3_ray_profile_pgd_20260516/fno_nu0p001_gpu_v100_batch100_steps100_zero_fixedsign/ray_profile_loss3_crossover_summary_dense_0to1.csv`

Generated figures:
- `normal_batch100_all_metrics_by_direction_full_r_grid_labeled_std.png`
- `normal_batch100_all_metrics_by_direction_zoom_0to0p1_labeled_std.png`
- `normal_batch100_all_metrics_by_direction_zoom_0to0p5_labeled_std.png`
- `normal_batch100_all_metrics_by_direction_zoom_0to1_labeled_std.png`
- `normal_batch100_loss3_crossover_zoom_0to0p5_labeled_std.png`
- `normal_batch100_loss3_crossover_zoom_0to1p0_labeled_std.png`

The zoom figures use a dense forward-only ray sweep over `r in [0, 1]` with saved directions; no attack directions were re-optimized.
