# Darcy CFlow Polished Report

This folder contains polished clean-evaluation figures for the current Darcy
binary loss3-targeted release.

Included figure types:

- Cross-method split-mean RMSE and Relative L2 panels.
- Per-method 52-dataset RMSE and Relative L2 heatmaps.
- Per-method 11-checkpoint RMSE and Relative L2 heatmaps.
- Final attack50 Delta FFT heatmaps and radial-spectrum summaries from the
  binary 20260611 robustness bundle.
Excluded from this formal release folder:

- Non-final attack diagnostics.
- Non-final SVD/Jacobian diagnostics.
- Attack-delta FFT addendum figures derived from the obsolete Darcy root.
- Dashboards that mixed clean metrics with those diagnostics.

## Final Delta FFT Addendum
Final attack-delta FFT figures were added from the completed attack50 delta NPZ files.

These are final-checkpoint FFT diagnostics, not epoch-wise attack-probe histories.

Generated files:
- `outputs/darcy_cflow_timematched_organized_release_20260614/figures/polished_report/baseline/polished_final_delta_fft_log_magnitude_heatmap_baseline.png`
- `outputs/darcy_cflow_timematched_organized_release_20260614/figures/polished_report/baseline/polished_final_delta_fft_dataset_radial_heatmap_baseline.png`
- `outputs/darcy_cflow_timematched_organized_release_20260614/figures/polished_report/loss1/polished_final_delta_fft_log_magnitude_heatmap_loss1.png`
- `outputs/darcy_cflow_timematched_organized_release_20260614/figures/polished_report/loss1/polished_final_delta_fft_dataset_radial_heatmap_loss1.png`
- `outputs/darcy_cflow_timematched_organized_release_20260614/figures/polished_report/loss2/polished_final_delta_fft_log_magnitude_heatmap_loss2.png`
- `outputs/darcy_cflow_timematched_organized_release_20260614/figures/polished_report/loss2/polished_final_delta_fft_dataset_radial_heatmap_loss2.png`
- `outputs/darcy_cflow_timematched_organized_release_20260614/figures/polished_report/loss3/polished_final_delta_fft_log_magnitude_heatmap_loss3.png`
- `outputs/darcy_cflow_timematched_organized_release_20260614/figures/polished_report/loss3/polished_final_delta_fft_dataset_radial_heatmap_loss3.png`
- `outputs/darcy_cflow_timematched_organized_release_20260614/figures/polished_report/physics_loss/polished_final_delta_fft_log_magnitude_heatmap_physics_loss.png`
- `outputs/darcy_cflow_timematched_organized_release_20260614/figures/polished_report/physics_loss/polished_final_delta_fft_dataset_radial_heatmap_physics_loss.png`
- `outputs/darcy_cflow_timematched_organized_release_20260614/figures/polished_report/random_clean/polished_final_delta_fft_log_magnitude_heatmap_random_clean.png`
- `outputs/darcy_cflow_timematched_organized_release_20260614/figures/polished_report/random_clean/polished_final_delta_fft_dataset_radial_heatmap_random_clean.png`
- `outputs/darcy_cflow_timematched_organized_release_20260614/figures/polished_report/random_solver/polished_final_delta_fft_log_magnitude_heatmap_random_solver.png`
- `outputs/darcy_cflow_timematched_organized_release_20260614/figures/polished_report/random_solver/polished_final_delta_fft_dataset_radial_heatmap_random_solver.png`
- `outputs/darcy_cflow_timematched_organized_release_20260614/figures/polished_report/final_delta_fft/polished_final_delta_fft_log_magnitude_grid_all7.png`
- `outputs/darcy_cflow_timematched_organized_release_20260614/figures/polished_report/final_delta_fft/polished_final_delta_radial_spectrum_all7.png`
- `outputs/darcy_cflow_timematched_organized_release_20260614/data/polished_report/final_delta_fft/final_delta_fft_dataset_metrics.csv`
- `outputs/darcy_cflow_timematched_organized_release_20260614/data/polished_report/final_delta_fft/final_delta_fft_summary_by_model.csv`
- `outputs/darcy_cflow_timematched_organized_release_20260614/manifests/final_delta_fft_polished_report_20260615.json`
