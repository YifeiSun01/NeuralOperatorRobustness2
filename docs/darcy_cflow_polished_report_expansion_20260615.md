# Darcy CFlow Polished Report Expansion - 2026-06-15

## Scope

Expanded the Darcy CFlow organized release with Burgers-style per-method
polished-report figures.

Target payload:

`outputs/darcy_cflow_timematched_organized_release_20260614/`

## What Changed

- Added `tools/build_darcy_cflow_polished_report_20260615.py`.
- Rebuilt `figures/polished_report/<method>/` for:
  - `loss1`
  - `loss2`
  - `loss3`
  - `physics`
  - `random_clean`
  - `random_solver`
- Added 7 per-method PNG figures for each method:
  - full RMSE heatmap with dataset-family line plot
  - 11-checkpoint RMSE heatmap with dataset-family line plot
  - full Relative L2 heatmap with dataset-family line plot
  - 11-checkpoint Relative L2 heatmap with dataset-family line plot
  - final attack clean/adv/gain diagnostics
  - final delta and SVD/Jacobian diagnostics
  - six-panel polished diagnostic dashboard
- Added 6 per-method CSV tables for each method under
  `data/polished_report/<method>/`.
- Added manifest:
  `outputs/darcy_cflow_timematched_organized_release_20260614/manifests/darcy_cflow_polished_report_20260615.json`.

## Counts

- Newly generated per-method polished PNG files: `42`.
- Total polished-report PNG files including the previous overview panels: `48`.
- Newly generated polished-report CSV files: `36`.
- Main-curve `work_hours` PNG count remains `0`.
- Organized-release logical size after expansion:
  `1,688,634,476` bytes.

## Notes

- The final attack plots use the organized-release final robustness sample table.
  That table is not an epoch-wise attack-batch training log, so those figures are
  intentionally labeled as final attack samples rather than epoch trajectories.
- Random clean and random solver do not have `wall_seconds` in the source eval
  tables. The per-method polished report therefore uses epoch axes and does not
  generate any `work_hours` plots.
