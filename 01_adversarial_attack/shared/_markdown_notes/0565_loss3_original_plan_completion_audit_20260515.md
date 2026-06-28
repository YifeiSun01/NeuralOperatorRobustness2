# Loss3 Original Plan Completion Audit - 2026-05-15

## Status

Completed audit only. No numerical experiment was run.

> Remote-status update: this file was a current-local-working-tree audit. For the R2-backed status, especially the completed Boundary-Rescaled Comparison, see `docs/loss3_original_plan_r2_completion_audit_20260515.md`.

## Source Files Inspected

- `docs/loss3_original_theory_experiment_plan.md`
- `docs/main_objective_mechanism_experiment1_result_20260514.md`
- `docs/fno_solver_jacobian_similarity_result_20260514.md`
- `docs/deeponet_solver_jacobian_similarity_result_20260515.md`
- `docs/local_jacobian_svd_experiment_purpose_20260515.md`
- `docs/local_jacobian_svd_direction_taxonomy_20260515.md`
- `docs/nine_row_fno001_fno01_deeponet01_svd_interpretation_20260515.md`
- `docs/fno_nu0p001_loss_gradient_path_result_20260515.md`
- `THREE_LOSS_BATCH100_FULL_LOSS3_SWEEP.md`
- `results/burgers_loss3_clean_recomputed_summary.md`
- `forensics/fno_solver_jacobian_similarity_20260514/summary.md`
- `results/fno_nu0p001_loss_gradient_path_steps50_save5_gpu_nocudnn_20260515_200631/gradient_direction_analysis/summary.md`

## Output Files

- `docs/loss3_original_plan_completion_audit_20260515.md`
- `EXPERIMENT_LEDGER.md`

## Working-Tree Caveat

Observed from `git status --short`: the working tree currently contains many
tracked experiment artifacts marked deleted under paths including
`benchmark_results/`, `fno_training_runs/`, `gradient_audit/`, `path_audit/`,
and `results/`. Those deleted artifacts were not interpreted as current local
results.

## Completion Summary

| Plan item | Status | Evidence | Remaining work |
|---|---|---|---|
| Experiment 1: Main Objective Comparison | Done as documented result; source result directory is not currently local | `docs/main_objective_mechanism_experiment1_result_20260514.md` records FNO/Burgers `nu=0.001`, batch 100, `epsilon=8`, `alpha=0.3`, `steps=100`, and compares `loss1_original`, `loss2_original`, `loss3_original` across PGD, LP-steepest PGD, and generalized power. It records `||Delta f||`, `||Delta j||`, `||Delta f-Delta j||`, `||e(x+delta)||`, `cos(Delta f,Delta j)`, `D_f`, and `D_sym`. | If strict reproducibility is needed, restore or fetch the source directory `results/three_loss_batch100_full_loss3_delta_rerun_20260514_fno_eps8_alpha0p3_final_boundary/` and CSVs such as `mechanism_diagnostics/mechanism_summary.csv`; they were named in the note but were not found in the current local tree during this audit. |
| Experiment 2: Local Response Decomposition Table | Mostly done for SVD/Jacobian directions; not complete for the outward-growth direction | Current local evidence includes `forensics/fno_solver_jacobian_similarity_20260514/aggregate_direction_comovement_summary.csv`, `forensics/fno_solver_jacobian_similarity_20260514/summary.md`, `forensics/fno_solver_jacobian_similarity_20260514_raw_recomputed/`, `forensics/deeponet_solver_jacobian_similarity_20260515/`, and related docs. These cover top `J_f`, top `J_j`, top `J_e`, and random directions, with response gains, mismatch gains, cosines, and `D_f`. | Add the explicit `v_growth* = argmax <e/||e||, (J_f-J_j)v>` / `A^T b` outward-growth direction table if the plan requires the exact Experiment 2 specification. |
| Experiment 3: Small-Epsilon Sweep | Not done as specified | `results/burgers_loss3_clean_recomputed_summary.md` contains older small-epsilon attack rows, for example FNO `nu=0.001`, Linf `eps=0.01` and `eps=0.1`, but this is not the planned local sweep over `{1e-4, 1e-3, 1e-2, 1e-1}` estimating `L_f(eps)`, `L_j(eps)`, `L_e(eps)`, `G_e(eps)`, and direction cosines across epsilons. No current local source directory matching this planned sweep was found. | Run the planned small-epsilon local ratio/residual-ratio sweep and record value stability plus `cos(v*(eps_i), v*(eps_j))`. |
| Experiment 4: Ray Profile / Local-to-Global Profile | Not done; only located in the plan | `docs/ray_profile_markdown_lookup_20260515.md` records that the plan section exists at `docs/loss3_original_theory_experiment_plan.md:741`, but it was a lookup only. No Ray-profile curve outputs were found. | For directions from `loss3_original`, `loss3_increment_ratio`, `loss3_residual_increment_ratio`, `loss3_regularized`, and random, plot `r -> ||e(x+rv)||`, norm increment ratio, and residual increment ratio for `r in [0, epsilon]`. |
| Experiment 5: Boundary-Rescaled Comparison | Partially implemented/documented, but no current local numeric result evidence | `THREE_LOSS_BATCH100_FULL_LOSS3_SWEEP.md` documents final-only boundary diagnostics and the formulas for `delta_bdry = epsilon * delta_final / ||delta_final||`, including `final_<loss/objective>` and `boundary_<loss/objective>`. The scripts `tools/run_three_loss_batch100_full_loss3_sweep.sh`, `tools/run_and_plot_batch_three_loss_loss_only.py`, and `tools/run_batch_three_loss_loss_only.py` exist. However, no `results/three_loss_batch100_full_loss3*` directories or `final_delta_summary.json` files were found locally during this audit. | Restore/fetch old sweep outputs or rerun the boundary-rescaled diagnostics, then summarize final vs boundary `loss3_original` and `loss3_increment_ratio` for ratio/regularized directions. |
| Experiment 6: Direction Rotation Along Path | Related path-gradient experiment exists, but the planned `v_e*(x_t)` rotation experiment is not done | `docs/fno_nu0p001_loss_gradient_path_result_20260515.md` and `results/fno_nu0p001_loss_gradient_path_steps50_save5_gpu_nocudnn_20260515_200631/gradient_direction_analysis/summary.md` analyze exact gradients `g1`, `g2`, `g3` along attack paths. They show, for example, mean `g1`/`g3` angle decreasing from about `74.24 deg` at `k=5` to `46.81 deg` at `k=50`. This is useful path evidence, but it does not re-estimate `v_e*(x_t)=argmax ||J_e(x_t)v||` at each path point. | Run the actual direction-rotation experiment: along `x_t=x+t delta*`, recompute/estimate local `J_e(x_t)` top directions and record `cos(v_e*(x_t), v_e*(x_0))` and adjacent cosines. |

## Minimal Experiment Set From Section 8

Observed from the current plan, the recommended minimal set has seven items.
Current status is:

1. `loss1_original` vs `loss2_original` vs `loss3_original`: done as documented
   in `docs/main_objective_mechanism_experiment1_result_20260514.md`, with the
   caveat that its named source result directory is not current local evidence.
2. `loss3_original` vs `loss3_increment_ratio` vs
   `loss3_residual_increment_ratio` vs `loss3_regularized`: partially covered
   by three-loss sweep documentation and existing scripts, but no current local
   numeric result directory was found for the full comparison.
3. Local response decomposition table: mostly done for SVD/Jacobian directions;
   missing explicit outward-growth `A^T b` direction.
4. Small-epsilon sweep: not done as specified.
5. Ray profile: not done.
6. Boundary-rescaled comparison: partially implemented/documented; current
   numeric outputs not found locally.
7. Direction rotation: related gradient-path work exists, but the planned
   `v_e*(x_t)` recomputation is not done.

## Inference

The strongest completed evidence for the paper narrative is already available
for Experiment 1 and the local Jacobian/SVD mechanism behind Experiment 2. The
largest remaining gaps are the explicitly local-to-global diagnostics:
small-epsilon stability, ray profiles, boundary-rescaled endpoint checks, and
path rotation of the local residual Jacobian direction.

## Remaining Work Priority

1. Restore or rerun the boundary diagnostics from the three-loss sweep, because
   the scripts already exist and this would directly support Experiment 5.
2. Run the Ray Profile experiment next, since it directly visualizes the local
   ratio versus finite-radius endpoint distinction.
3. Run the Small-Epsilon Sweep to validate the local meaning of the ratio and
   residual-ratio objectives.
4. Add the missing outward-growth `A^T b` direction to the local response table.
5. Run the full `v_e*(x_t)` direction-rotation experiment.
