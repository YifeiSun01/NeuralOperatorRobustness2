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
  - smoke attack clean/adv/gain diagnostics
  - smoke delta and SVD/Jacobian diagnostics
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

- The attack and SVD/Jacobian polished panels use the organized-release smoke
  robustness/SVD diagnostic tables, not final time-matched robustness runs. Those
  source rows use `darcy_sir20_smoke_*_1ep` checkpoints and should not be
  interpreted as final rankings.
- Random clean and random solver do not have `wall_seconds` in the source eval
  tables. The per-method polished report therefore uses epoch axes and does not
  generate any `work_hours` plots.
