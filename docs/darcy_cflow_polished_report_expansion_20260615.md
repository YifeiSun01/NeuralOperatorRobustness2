# Darcy CFlow Polished Report Expansion - 2026-06-15

## Scope

This note records the current formal Darcy CFlow polished-report contents after
the release cleanup on 2026-06-15.

Target folder:

`outputs/darcy_cflow_timematched_organized_release_20260614/`

## Current Polished Report Contents

The current `figures/polished_report/` folder contains clean-evaluation figures
only:

- Cross-method split-mean RMSE and Relative L2 panels.
- Per-method 52-dataset RMSE and Relative L2 heatmaps.
- Per-method 11-checkpoint RMSE and Relative L2 heatmaps.

The current per-method folders are:

- `loss1`
- `loss2`
- `loss3`
- `physics`
- `random_clean`
- `random_solver`

The corresponding clean CSV tables remain under
`data/polished_report/<method>/`.

## Cleanup

The formal organized release no longer contains non-final attack or
SVD/Jacobian diagnostic panels, dashboards, source tables, vectors, or derived
rankings.

## Evidence

Observed from local file scan after cleanup:

- `figures/polished_report/` has no attack/SVD/dashboard diagnostic PNGs.
- `data/polished_report/` has no attack/SVD diagnostic CSVs.
- The formal release file names and Markdown/JSON/TXT files have no residual
  non-final diagnostic labels.

## Interpretation

Use this polished-report folder for clean RMSE and Relative L2 visualization
only. Use the real attack20 audit document for attack-loss-increase evidence:

`docs/darcy_cflow_attack20_attack50_loss_increase_audit_20260615.md`
