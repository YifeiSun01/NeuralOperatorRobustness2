# Darcy CFlow Final Robustness/Jacobian/SVD Status - 2026-06-15

## Bottom line

The requested final-model robustness/Jacobian/SVD quantities have **not** been
computed for the seven final Darcy CFlow models.

The current formal organized release contains final clean 52-dataset evaluation
curves and tables only. It does not contain the final seven-model robustness
tables, final attack deltas, final SVD/Jacobian tables, or final attack heatmaps.

## What exists and is valid

Formal release folder:

`outputs/darcy_cflow_timematched_organized_release_20260614/`

Valid final clean-evaluation table:

`outputs/darcy_cflow_timematched_organized_release_20260614/data/cflow_clean_52dataset_metric_long_ranked.csv`

That table contains seven final models on the 52 datasets:

| method | final epoch in clean table | dataset coverage |
|---|---:|---:|
| baseline | 0 | train + test + 50 generalization |
| loss1 | 3000 | train + test + 50 generalization |
| loss2 | 3079 | train + test + 50 generalization |
| loss3 | 3033 | train + test + 50 generalization |
| physics | 3121 | train + test + 50 generalization |
| random_clean | 3500 | train + test + 50 generalization |
| random_solver | 3500 | train + test + 50 generalization |

## What is missing

For the final seven checkpoints, these required outputs are missing:

- 52 datasets x 50 samples per dataset surrogate/adversarial attack.
- Attack with the intended final attack configuration, not one-step smoke.
- `clean_loss`, `adv_loss`, `loss_increase`, `relative_increase`.
- Per-sample `delta` fields from the final attack.
- Fixed 25-sample SVD/Jacobian manifest shared across all seven models.
- Error function values/vectors on the fixed samples.
- Jacobian matrices or Jacobian action outputs needed for the diagnostics.
- Singular values and right singular vectors.
- Operator norm / top singular value summaries.
- `J error` / `J^T error` vectors and their norms.
- Cosine similarity, angle, and pairwise correlation among singular vector,
  `J^T error`, and attack `delta`.
- Same-sample correlations among singular value, `J^T error` norm, and
  attack loss increase.

## Invalid or insufficient artifacts found locally

The local 50-step attack heatmap summaries that still exist under
`analysis_outputs/` are not final-model results. They use the earlier checkpoint
family:

- loss1: epoch 1000
- loss2: epoch 1026
- loss3: epoch 1011
- physics: epoch 1040
- random_clean: epoch 1100
- random_solver: epoch 1100

Those files can be useful only as old diagnostics. They must not be used as final
3000-3500 epoch robustness rankings.

Earlier smoke/partial robustness files were also invalid for the final claim:
they used one-epoch or small-sample checks and have been removed or excluded from
the formal release.

## Correct final checkpoint family to use for regeneration

The final robustness stage should be regenerated from the final checkpoint family
only:

- baseline:
  `2D_Darcy_FNO2d/saved_models/2D/darcy_screen_baseline_m64_w60_e50_20260607/best.pt`
- loss1:
  `adversarial_training_runs/darcy_binary_loss3targeted_loss1_continue2000ep_from_1000ep_full50_timematched_20260612_stage2_2000_from_1000c/darcy/checkpoints/darcy_epoch3000_step003000.pt`
- loss2:
  `adversarial_training_runs/darcy_binary_loss3targeted_loss2_continue2053ep_from_1026ep_full50_timematched_20260612_stage2_2000_from_1000c/darcy/checkpoints/darcy_epoch3079_step003079.pt`
- loss3:
  `adversarial_training_runs/darcy_binary_loss3targeted_loss3_continue2022ep_from_1011ep_full50_timematched_20260612_stage2_2000_from_1000c/darcy/checkpoints/darcy_epoch3033_step003033.pt`
- physics:
  `adversarial_training_runs/darcy_binary_loss3targeted_physics_continue2081ep_from_1040ep_full50_timematched_20260612_stage2_2000_from_1000c/darcy/checkpoints/darcy_epoch3121_step003121.pt`
- random_clean:
  `adversarial_training_runs/darcy_binary_random_binary_fixed_y_continue_to3500_from_3000_20260614_supervised/darcy/checkpoints/darcy_epoch3500_step003500.pt`
- random_solver:
  `adversarial_training_runs/darcy_binary_random_binary_solver_y_continue_to3500_from_3000_20260614_supervised/darcy/checkpoints/darcy_epoch3500_step003500.pt`

## Audit conclusion

The answer to whether the requested final robustness/Jacobian/SVD quantities were
computed is: **no**.

Only final clean evaluation is currently valid in the formal Darcy CFlow release.
All final robustness claims must wait until the attack and SVD/Jacobian stage is
rerun from the seven final checkpoints listed above.
