# Burgers Solver7860 Polished Reports - 2026-06-14

Status: generated the missing polished-report visualization set for the final
Burgers solver7860/clean8000 audit package.

Local artifacts:

- Full report bundle:
  `visualizations/burgers_solver7860_clean8000_polished_reports_20260614/`
- PNG-only bundle:
  `visualizations/burgers_solver7860_clean8000_polished_reports_image_only_bundle_20260614/`
- Also copied into the final audit output under:
  `outputs/burgers_timematched_solver7860_clean8000_audit_20260614/figures/polished_reports/`

Models covered:

- `loss1`, epoch 8000
- `loss2`, epoch 2000
- `loss3`, epoch 1000
- `random_clean_y`, epoch 8000
- `random_solver_y`, epoch 7860

What was generated for each model:

- Attack-loss polished report with clean loss, adversarial loss, attack gain,
  and epsilon-bucket curves.
- Raw RMSE heatmap plus group mean lineplot.
- Raw Relative-L2 heatmap plus group mean lineplot.
- Checkpoint-style RMSE heatmap plus lineplot.
- Checkpoint-style Relative-L2 heatmap plus lineplot.
- Delta FFT raw power heatmap plus selected smoothed spectra.
- High-transparency max-five RMSE line plots, raw and MA25.
- High-transparency max-five Relative-L2 line plots, raw and MA25.

Counts:

- Per model: 10 PNGs, 6 CSVs, 1 JSON manifest.
- Full report bundle: 50 PNGs, 30 CSVs, 5 JSON manifests.
- PNG-only bundle: 50 PNGs and no non-PNG files.
- Final audit output now contains 98 PNGs total, including 50 polished-report
  PNGs under `figures/polished_reports/`.

Data handling:

- No training was rerun.
- Existing local training/evaluation CSVs were used.
- Loss1/loss2/loss3 raw attack-probe NPZ files were synced back from R2 so the
  Delta FFT heatmaps could be drawn from raw deltas instead of CSV-only summary
  metrics.
- Random-clean and random-solver probe NPZ files were already present locally.
- The plotting script was updated so continuation runs use their actual logged
  epoch span instead of drawing a large blank x-axis region before the resumed
  epoch range.

R2 verification:

- `visualizations/burgers_solver7860_clean8000_polished_reports_image_only_bundle_20260614`:
  50 PNGs, 0 non-PNG files.
- `visualizations/burgers_solver7860_clean8000_polished_reports_20260614`:
  50 PNGs, 30 CSVs, 5 JSON manifests.
- `outputs/burgers_timematched_solver7860_clean8000_audit_20260614`:
  98 PNGs total, 50 under `figures/polished_reports/`.
