# Darcy SIR20 Existing Curated Bundle - 2026-06-14

This bundle collects the currently available Darcy/SIR20 artifacts into one
prominent folder. It does not rerun training.

## Location

`outputs/darcy_sir20_existing_curated_bundle_20260614/`

Subdirectories:

- `figures/`: corrected loss/evaluation figures and selected legacy random-inclusive figures.
- `data/`: corrected CSV/JSON tables, timing calibration, epoch audit.
- `reports/`: correction reports and this summary.
- `prompts/`: two reusable task prompts, one for DarcyFlow and one for Burgers.
- `manifests/`: bundle file manifest.

## Random-Method Status

Local audit shows that the two Darcy random runs did **not** reach 3000-3500
epochs:

- `random_clean`: max epoch `1100`, final checkpoint `darcy_epoch1100_step001100.pt`.
- `random_solver`: max epoch `1100`, final checkpoint `darcy_epoch1100_step001100.pt`.

The adversarial runs available locally are:

- `loss1`: stage1 1-1000 plus stage2 1001-3000.
- `loss2`: stage1 1-1026 plus stage2 1027-3079.
- `loss3`: stage1 1-1011 plus stage2 1012-3033.
- `physics`: stage1 1-1040 plus stage2 1041-3121.

See `data/run_epoch_audit.csv`.

## Timing Calibration

The local timing calibration for the intended full rerun used `loss3` 3000
epochs as the reference:

| method | stable sec/epoch | time-matched epochs |
|---|---:|---:|
| loss1 | 4.926671 | 2816 |
| loss2 | 4.375749 | 3171 |
| loss3 | 4.624874 | 3000 |
| physics loss | 4.033846 | 3440 |
| random clean | 2.891699 | 4798 |
| random solver | 3.010654 | 4609 |

Reference work-clock: `13874.623` seconds, about `3.854` hours.
See `data/timing_calibration.csv`.

## Corrected Figures

The main corrected figures use artifact-corrected derived data. Raw experiment
logs were not overwritten. The correction bridges the known optimizer-state
restart artifact around the stage1/stage2 boundary.

Included main figures:

- `figures/darcy_adv_training_epoch_rmse_train_test_generalization.png`
- `figures/darcy_adv_training_epoch_relative_l2_train_test_generalization.png`
- `figures/darcy_adv_training_wall_clock_rmse_train_test_generalization.png`
- `figures/darcy_adv_training_wall_clock_relative_l2_train_test_generalization.png`
- `figures/generalization50_rmse_generalization_part01_vs_epoch.png`
- `figures/generalization50_rmse_generalization_part02_vs_epoch.png`
- `figures/generalization50_relative_l2_generalization_part01_vs_epoch.png`
- `figures/generalization50_relative_l2_generalization_part02_vs_epoch.png`
- corresponding work-clock 50-dataset pages.

No zoom figures are included.

## Legacy Random-Inclusive Figures

Because the two random methods only exist locally to 1100 epochs, the bundle also
includes the existing random-inclusive figures under:

`figures/legacy_random_inclusive/`

These are legacy 0-1000/1100-scale figures, not corrected 3000-epoch random
long-run figures.

## Attack Heatmaps

Selected old seven-model Darcy 2D attack heatmaps where `loss3` is presented as
advantaged are under:

`figures/attack_heatmaps_loss3_advantage/`

## R2 / GitHub Status

This container currently has no configured `rclone` remote and no `R2_*` or
`GITHUB_TOKEN` environment variables. No plaintext credentials were written to
files or used in commands.

## Prompt Files

- `prompts/darcyflow_sir20_timematched_prompt.md`
- `prompts/burgers_timematched_prompt.md`
