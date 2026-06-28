# NS2D R2 And GitHub Sync - 2026-05-22

Updated: 2026-05-22 22:43:44 UTC

Status: R2 upload completed; GitHub source/record commit prepared in this turn.

## R2 Uploads

Remote base:

`r2_auto:neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected`

Uploaded with `rclone copy`, not destructive `sync`, so existing remote-only objects were not deleted.

Credentials were supplied only through a temporary rclone config under `/tmp`; no R2 secret was written to repository files.

### Visualization Artifacts

Local source:

`2D_NS_FNO2d_recurrent/visualizations`

Remote target:

`r2_auto:neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected/2D_NS_FNO2d_recurrent/visualizations`

Verification size for the consolidated report package:

`r2_auto:neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected/2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_visual_fft_report_package_20260522`

Observed from `rclone size` after upload:

- total objects: `137`
- total size: `348.500 MiB` (`365429044` bytes)

### Attack Result Artifacts

Local source:

`2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/full_adw_b10_pair_outer_baseline_first_20260522`

Remote target:

`r2_auto:neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected/2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/full_adw_b10_pair_outer_baseline_first_20260522`

Observed from `rclone size` after upload:

- total objects: `411`
- total size: `7.679 GiB` (`8245142677` bytes)

## GitHub Files Prepared

The GitHub commit intentionally stages lightweight source and record files only:

- `EXPERIMENT_LEDGER.md`
- NS2D optimizer/FFT/visualization Markdown records under `docs/`
- EPS32/alpha10 plotting scripts under `2D_NS_FNO2d_recurrent/visualizations/`
- README files for generated visualization packages

Large local generated artifacts remain untracked locally and are stored in R2 instead of GitHub:

- `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/`
- `2D_NS_FNO2d_recurrent/saved_models/2D/`
- generated image folders under `2D_NS_FNO2d_recurrent/visualizations/`

## Notes

- `git diff --cached --check` passed before commit preparation.
- `rclone copy` completed without reported errors.
