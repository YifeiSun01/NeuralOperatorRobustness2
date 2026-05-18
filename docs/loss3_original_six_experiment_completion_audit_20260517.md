# Loss3 Original Six-Experiment Completion Audit - 2026-05-17

Status: completed audit of the six-experiment plan in
`docs/loss3_original_theory_experiment_plan.md`. No numerical experiment was
rerun for this audit.

## Scope

The completion claim is for the current intended scope:

- Model/task: FNO / 1D Burgers.
- Viscosity: `nu=0.001`.
- Main finite-radius setting where applicable: `epsilon=8`, `alpha=0.3`,
  batch size `100`.

Broader claims for other viscosities, DeepONet, or alternate architectures would
require additional runs and are not counted as part of this minimal six-part
plan.

## Summary

Observed from local files and experiment records:

| Experiment | Status | Main local evidence |
|---|---|---|
| 1. Main Objective Comparison | Done | `docs/main_objective_mechanism_experiment1_result_20260514.md`; `results/three_loss_batch100_full_loss3_delta_rerun_20260514_fno_eps8_alpha0p3_final_boundary/mechanism_diagnostics/mechanism_summary.csv` |
| 2. Local Response Decomposition Table | Done | `docs/fno_solver_jacobian_similarity_result_20260514.md`; `docs/outward_growth_direction_result_20260515.md`; `forensics/fno_solver_jacobian_similarity_20260514/`; `forensics/outward_growth_direction_20260515/fno_nu0p001/` |
| 3. Small-Epsilon Sweep | Done | `docs/loss3_small_epsilon_sweep_fno_nu0p001_gpu_result_20260516.md`; `docs/loss3_gradient_direction_optimization_fno_nu0p001_gpu_steps12_result_20260516.md`; corresponding `forensics/loss3_*_20260516/...` directories |
| 4. Ray Profile / Local-to-Global Profile | Done | `docs/loss3_ray_profile_pgd_fno_nu0p001_gpu_batch100_steps100_fixedsign_result_20260516.md`; `forensics/loss3_ray_profile_pgd_20260516/fno_nu0p001_gpu_v100_batch100_steps100_zero_fixedsign/`; `results/three_loss_batch100_full_loss3_delta_rerun_20260516_fno_eps8_alpha0p3_final_boundary_local_repro/` |
| 5. Boundary-Rescaled Comparison | Done | `results/three_loss_batch100_full_loss3_delta_rerun_20260514_fno_eps8_alpha0p3_final_boundary/`; `docs/loss3_original_plan_r2_completion_audit_20260515.md`; 2026-05-16 local cross-check directory above |
| 6. Direction Rotation Along Path | Done | `docs/loss3_direction_rotation_path_fno_nu0p001_result_20260516.md`; `docs/loss3_jacobian_subspace_rotation_path_fno_nu0p001_result_20260516.md`; corresponding `forensics/loss3_*rotation_path_20260516/...` directories |

Inference from the observed files:

- The minimal six-experiment plan is complete for FNO / Burgers `nu=0.001`.
- The old status table in `docs/loss3_original_theory_experiment_plan.md` was
  stale because it still marked Experiment 6 as partial. This audit updates that
  table to `Done: 6`, `Partially done: 0`, `Not yet done: 0`.

## Per-Experiment Evidence

### Experiment 1: Main Objective Comparison

Observed from:

- Result doc: `docs/main_objective_mechanism_experiment1_result_20260514.md`.
- Source result directory:
  `results/three_loss_batch100_full_loss3_delta_rerun_20260514_fno_eps8_alpha0p3_final_boundary/`.
- Core numeric table:
  `results/three_loss_batch100_full_loss3_delta_rerun_20260514_fno_eps8_alpha0p3_final_boundary/mechanism_diagnostics/mechanism_summary.csv`.

Observed local checks:

- The source result directory exists and has `27` loss/objective/method run
  directories.
- `mechanism_summary.csv` has `55` lines.

R2 recovery update:

- The older derived folder `results/main_objective_mechanism_summary_20260514/`
  has now been restored locally from R2. It contains `combined_mechanism_summary.csv`,
  `focused_original_objectives.csv`, and three focused-objective PNG plots.
- Local line counts after restore: `combined_mechanism_summary.csv` has `325`
  lines and `focused_original_objectives.csv` has `55` lines.

### Experiment 2: Local Response Decomposition Table

Observed from:

- `docs/fno_solver_jacobian_similarity_result_20260514.md`.
- `docs/outward_growth_direction_result_20260515.md`.
- `forensics/fno_solver_jacobian_similarity_20260514/`.
- `forensics/outward_growth_direction_20260515/fno_nu0p001/`.

Observed local files include aggregate direction-comovement tables, singular
value/vector summaries, per-index tables, outward-growth direction response
files, finite-difference growth tables, and manifests.

Inference:

- The originally missing `v_growth* = normalize(A^T b)` row is present, so the
  Experiment 2 table is complete for the current `nu=0.001` scope.

### Experiment 3: Small-Epsilon Sweep

Observed from:

- `docs/loss3_small_epsilon_sweep_fno_nu0p001_gpu_result_20260516.md`.
- `forensics/loss3_small_epsilon_sweep_20260516/fno_nu0p001_gpu_v100/`.
- `docs/loss3_gradient_direction_optimization_fno_nu0p001_gpu_steps12_result_20260516.md`.
- `forensics/loss3_gradient_direction_optimization_20260516/fno_nu0p001_gpu_v100_steps12/`.

Observed local files include `candidate_metrics.csv`,
`best_by_objective_epsilon_index.csv`,
`aggregate_best_by_objective_epsilon.csv`, `direction_stability.csv`,
`direction_stability_summary.csv`, local-reference tables, gradient-optimization
summary tables, figures, and GPU manifests.

Inference:

- The planned sweep over `epsilon in {1e-4, 1e-3, 1e-2, 1e-1}` is complete for
  the current scope.

### Experiment 4: Ray Profile / Local-to-Global Profile

Observed from:

- `docs/loss3_ray_profile_pgd_fno_nu0p001_gpu_batch100_steps100_fixedsign_result_20260516.md`.
- `forensics/loss3_ray_profile_pgd_20260516/fno_nu0p001_gpu_v100_batch100_steps100_zero_fixedsign/`.
- Historical-script cross-check:
  `results/three_loss_batch100_full_loss3_delta_rerun_20260516_fno_eps8_alpha0p3_final_boundary_local_repro/`.

Observed local checks from the previous 2026-05-17 environment/artifact audit:

- `ray_profile.csv` has `31501` lines, matching `100 samples * 7 directions * 45 radii + header`.
- `ray_winner_summary.csv` has `101` lines.
- `attack_final_by_sample.csv` has `401` lines.
- The 2026-05-16 local cross-check directory has all `9` expected `loss3_*`
  run directories and each contains the expected summary, loss, and final-delta
  files.

Inference:

- Experiment 4 is complete. The fixed-sign PGD100 result supersedes the earlier
  non-fixed-sign wrapper output.

### Experiment 5: Boundary-Rescaled Comparison

Observed from:

- `results/three_loss_batch100_full_loss3_delta_rerun_20260514_fno_eps8_alpha0p3_final_boundary/`.
- `docs/loss3_original_plan_r2_completion_audit_20260515.md`.
- `results/three_loss_batch100_full_loss3_delta_rerun_20260516_fno_eps8_alpha0p3_final_boundary_local_repro/`.
- `docs/loss3_ray_profile_pgd_fno_nu0p001_gpu_batch100_steps100_fixedsign_result_20260516.md`.

Observed local checks:

- The 2026-05-14 final-boundary source directory has `27` loss/objective/method
  run directories and many `final_delta_summary.json` files.
- The 2026-05-16 local cross-check has `9` `loss3_*` run directories with
  expected final-delta diagnostics.

Inference:

- The boundary-rescaled comparison is complete as an experiment. A more compact
  paper-ready table would be presentation work, not missing experiment work.

### Experiment 6: Direction Rotation Along Path

Observed from:

- `docs/loss3_direction_rotation_path_fno_nu0p001_result_20260516.md`.
- `forensics/loss3_direction_rotation_path_20260516/fno_nu0p001_pilot/`.
- `docs/loss3_jacobian_subspace_rotation_path_fno_nu0p001_result_20260516.md`.
- `forensics/loss3_jacobian_subspace_rotation_path_20260516/fno_nu0p001/`.

Observed local checks:

- The expanded Experiment 6 manifest records `status: completed`, GPU runtime on
  `Tesla V100-SXM2-32GB`, `sm_70`, PyTorch `2.8.0+cu126`, and JAX backend `gpu`.
- `jacobian_subspace_rotation_aggregate_by_t.csv` has `12` lines.
- `jacobian_subspace_rotation_by_sample_t.csv` has `56` lines.
- `jacobian_subspace_rotation_summary_by_sample.csv` has `6` lines.

Inference:

- Experiment 6 is no longer partial. The planned re-estimation of local
  residual-Jacobian directions along `x_t = x + t delta*` has been done, and
  the expanded run adds top-k subspace, spectral-norm, and response-sketch
  evidence.

## Remaining Caveats

Observed from current local checks:

- `results/main_objective_mechanism_summary_20260514/` was initially missing locally but has now been restored from R2.
- Some experiment artifact directories are untracked in git.
- This audit did not perform a live R2 remote listing. It relies on current
  local files and previously recorded R2 sync/audit docs.
- No deleted tracked experiment artifacts were shown in the final `git status --short` check for this audit.

## Conclusion

Observed evidence supports marking all six planned experiments complete for the
current FNO / Burgers `nu=0.001` story. The plan file has been updated so the
completion table no longer says Experiment 6 is partial, and the missing
Experiment 1 derived summary directory has been restored from R2.

## Detailed Six-Experiment Results And Conclusions

This section records what each of the six experiments actually showed for the
current FNO / Burgers `nu=0.001` story.  Each item separates observed evidence
from the interpretation we should use in the writeup.

### Experiment 1: Main Objective Comparison

Data sources:

- Result note: `docs/main_objective_mechanism_experiment1_result_20260514.md`.
- Main source directory:
  `results/three_loss_batch100_full_loss3_delta_rerun_20260514_fno_eps8_alpha0p3_final_boundary/`.
- Core numeric table:
  `results/three_loss_batch100_full_loss3_delta_rerun_20260514_fno_eps8_alpha0p3_final_boundary/mechanism_diagnostics/mechanism_summary.csv`.
- Restored derived summaries:
  `results/main_objective_mechanism_summary_20260514/combined_mechanism_summary.csv`
  and `results/main_objective_mechanism_summary_20260514/focused_original_objectives.csv`.

Observed results:

| PGD optimized objective | model movement `||Delta f||` | solver movement `||Delta j||` | mismatch `||Delta f-Delta j||` | true error `||e(x+delta)||` | response cosine |
|---|---:|---:|---:|---:|---:|
| `loss1_original` | `11.172 +/- 0.918` | `10.630 +/- 0.993` | `4.351 +/- 1.837` | `4.340 +/- 1.830` | `0.910 +/- 0.068` |
| `loss2_original` | `11.295 +/- 0.989` | `10.700 +/- 0.854` | `4.377 +/- 1.908` | `4.372 +/- 1.917` | `0.912 +/- 0.063` |
| `loss3_original` | `5.866 +/- 3.237` | `5.839 +/- 2.582` | `5.374 +/- 2.773` | `5.395 +/- 2.748` | `0.565 +/- 0.219` |

Additional observed diagnostics:

| PGD optimized objective | error growth | `D_f` | `D_sym` | mismatch per perturbation norm |
|---|---:|---:|---:|---:|
| `loss1_original` | `4.042 +/- 1.822` | `0.385 +/- 0.150` | `0.199 +/- 0.084` | `0.544 +/- 0.230` |
| `loss2_original` | `4.074 +/- 1.913` | `0.382 +/- 0.152` | `0.197 +/- 0.082` | `0.547 +/- 0.239` |
| `loss3_original` | `5.097 +/- 2.718` | `1.012 +/- 0.376` | `0.469 +/- 0.113` | `0.682 +/- 0.336` |

Conclusion / role:

- Observed evidence shows that `loss1_original` and `loss2_original` produce
  large model movement, but the solver also moves strongly in almost the same
  direction.  Their response cosine is about `0.91`, so much of the model
  movement is co-movement with the oracle rather than regression failure.
- Observed evidence shows that `loss3_original` produces smaller raw model
  movement but larger model-solver mismatch and larger true endpoint error.
- Inference: `loss1` and `loss2` are not reliable substitutes for true
  oracle-relative error.  For finite-radius robustness, `loss3_original` is the
  objective that directly measures the failure we care about.

### Experiment 2: Local Response Decomposition Table

Data sources:

- Result notes: `docs/fno_solver_jacobian_similarity_result_20260514.md` and
  `docs/outward_growth_direction_result_20260515.md`.
- SVD/Jacobian outputs: `forensics/fno_solver_jacobian_similarity_20260514/`.
- Outward-growth outputs:
  `forensics/outward_growth_direction_20260515/fno_nu0p001/`.

Observed results:

- The completed outward-growth run covered samples `0, 7, 40, 47, 115`.
- `outward_growth` outward component mean: `0.166514`.
- Control means for the same outward component:
  `error_top = 0.0140571`, `fno_top = 0.00218766`,
  `solver_top = -0.00819025`, and random-best `0.0114951`.
- `error_top` has larger residual-movement / mismatch-gain mean, `0.412169`,
  than `outward_growth`, `0.368053`.
- Finite-difference checks match the linear prediction for `outward_growth`:
  predicted `0.166514`, actual `0.167286` at `rho=1e-4`, and actual `0.166668`
  at `rho=1e-3`.

Conclusion / role:

- Observed evidence separates two local objects:
  `v_e*` maximizes residual-field movement `||A v||`, while
  `v_growth* = normalize(A^T b)` maximizes first-order outward growth of the
  current clean residual norm.
- The two directions are not the same diagnostic: `error_top` moves the residual
  field more, but `outward_growth` pushes the current residual norm outward much
  more strongly.
- Inference: local Lipschitz / residual-movement diagnostics help explain
  geometry, but they are not identical to the finite-radius endpoint objective.
  This supplies the missing local-decomposition row needed by the plan.

### Experiment 3: Small-Epsilon Sweep

Data sources:

- Result note:
  `docs/loss3_small_epsilon_sweep_fno_nu0p001_gpu_result_20260516.md`.
- Direction-optimization confirmation:
  `docs/loss3_gradient_direction_optimization_fno_nu0p001_gpu_steps12_result_20260516.md`.
- Output directories:
  `forensics/loss3_small_epsilon_sweep_20260516/fno_nu0p001_gpu_v100/` and
  `forensics/loss3_gradient_direction_optimization_20260516/fno_nu0p001_gpu_v100_steps12/`.

Observed results:

- Samples: `0, 7, 40, 47, 115`.
- Radii: `epsilon = 1e-4, 1e-3, 1e-2, 1e-1`.
- Candidate evaluations: `5 samples x 4 epsilons x 178 directions = 3560`.
- Best source counts across all samples and epsilons:

| objective | winning direction source | count |
|---|---|---:|
| `L_f` | FNO `J_f` top direction | `20 / 20` |
| `L_j` | solver `J_j` top direction | `20 / 20` |
| `L_e` | residual/error `J_e` top direction | `20 / 20` |
| `G_e` | clean residual outward-growth direction | `20 / 20` |

Clean local reference means:

| objective | clean local reference mean |
|---|---:|
| `L_f` | `3.895` |
| `L_j` | `4.095` |
| `L_e` | `0.8682` |
| `G_e` | `0.1665` |

Best/local ratios by radius:

| objective | `1e-4` | `1e-3` | `1e-2` | `1e-1` |
|---|---:|---:|---:|---:|
| `L_f` | `1.0005` | `1.0000` | `1.0001` | `0.9901` |
| `L_j` | `0.9997` | `1.0000` | `0.9999` | `0.9854` |
| `L_e` | `1.0004` | `1.0002` | `1.0010` | `0.9608` |
| `G_e` | `0.9965` | `1.0005` | `1.0061` | `1.0576` |

Conclusion / role:

- Observed evidence shows an epsilon-refinement / local-convergence phenomenon:
  for `epsilon <= 1e-2`, finite-difference ratios match the clean local Jacobian
  references and selected direction subspaces are stable.
- Observed evidence also shows finite-radius drift beginning at `epsilon=0.1`,
  especially for `L_e` and `G_e`.
- Inference: ratio objectives are meaningful as local structure diagnostics when
  epsilon is sufficiently small, but they should not be overinterpreted as the
  full finite-radius attack objective.

### Experiment 4: Ray Profile / Local-to-Global Profile

Data sources:

- Corrected result note:
  `docs/loss3_ray_profile_pgd_fno_nu0p001_gpu_batch100_steps100_fixedsign_result_20260516.md`.
- Output directory:
  `forensics/loss3_ray_profile_pgd_20260516/fno_nu0p001_gpu_v100_batch100_steps100_zero_fixedsign/`.
- Historical-script cross-check:
  `results/three_loss_batch100_full_loss3_delta_rerun_20260516_fno_eps8_alpha0p3_final_boundary_local_repro/`.

Observed results:

| direction | endpoint mean `loss3` at `r=8` | endpoint wins | small clean-growth wins | small residual-increment wins |
|---|---:|---:|---:|---:|
| `loss3_original_final` | `5.447` | `58` | `0` | `0` |
| `loss3_increment_ratio_final` | `4.221` | `10` | `0` | `0` |
| `local_residual_movement` | `4.139` | `31` | `0` | `100` |
| `local_outward_growth` | `2.720` | `0` | `100` | `0` |
| `loss3_regularized_final` | `2.068` | `1` | `0` | `0` |
| `random` | `0.5268` | `0` | `0` | `0` |

Additional observed local-to-global gap metrics:

| quantity | mean | std | min | max |
|---|---:|---:|---:|---:|
| endpoint over small-growth endpoint `loss3` | `3.124` | `1.960` | `1.130` | `12.15` |
| endpoint over small-residual endpoint `loss3` | `2.716` | `4.144` | `1.000` | `25.01` |

Historical-script cross-check under the same normal PGD protocol:

| PGD objective | final delta norm mean | boundary `loss3_original` mean |
|---|---:|---:|
| `loss3_original_pgd` | `7.7510` | `5.4469` |
| `loss3_increment_ratio_pgd` | `7.8505` | `4.2215` |
| `loss3_regularized_pgd` | `0.3043` final, boundary-rescaled to `8.0` | `2.0275` |

Conclusion / role:

- Observed evidence shows that the clean local winners are not the finite-radius
  endpoint winners.  `local_outward_growth` wins the tiny-radius clean-growth
  diagnostic in `100/100` samples but has weak endpoint `loss3`; direct
  `loss3_original` is strongest at the endpoint.
- Inference: local optimality and finite-radius endpoint optimality are
  different.  This is the main local-to-global nonlinear gap result.
- The corrected fixed-sign PGD run supersedes the earlier non-fixed-sign wrapper
  output that had a manual-PGD sign error.

### Experiment 5: Boundary-Rescaled Comparison

Data sources:

- R2/local completion audit:
  `docs/loss3_original_plan_r2_completion_audit_20260515.md`.
- Final-boundary source directory:
  `results/three_loss_batch100_full_loss3_delta_rerun_20260514_fno_eps8_alpha0p3_final_boundary/`.
- 2026-05-16 local cross-check:
  `results/three_loss_batch100_full_loss3_delta_rerun_20260516_fno_eps8_alpha0p3_final_boundary_local_repro/`.

Observed results from the 2026-05-14 final-boundary summaries:

| tag | final `||delta||_2` mean | boundary `||delta||_2` mean | final `loss3_original` mean | boundary `loss3_original` mean |
|---|---:|---:|---:|---:|
| `loss3_original_pgd` | `7.7524` | `8.0000` | `5.3949` | `5.4525` |
| `loss3_increment_ratio_pgd` | `7.8499` | `8.0000` | `4.1478` | `4.2225` |
| `loss3_regularized_pgd` | `0.3018` | `8.0000` | `0.4928` | `2.0995` |
| `loss2_increment_ratio_pgd` | `6.3068` | `8.0000` | `2.0591` | `3.1936` |
| `loss2_regularized_pgd` | `5.1939` | `8.0000` | `1.9873` | `3.6531` |

Conclusion / role:

- Observed evidence shows that regularized objectives can produce perturbations
  far inside the epsilon boundary.  For example, `loss3_regularized_pgd` has
  mean final norm only `0.3018` under `epsilon=8`.
- Boundary rescaling can increase endpoint `loss3_original`, but it still does
  not make ratio / regularized directions beat direct `loss3_original`.  For
  example, `loss3_regularized_pgd` rises from `0.4928` to `2.0995` after
  rescaling, still far below `loss3_original_pgd` at `5.4525`.
- Inference: local efficiency or cost-aware objectives do not automatically
  become good finite-radius endpoint attacks just because we rescale their
  directions to the full perturbation budget.

### Experiment 6: Direction Rotation Along Path

Data sources:

- Direction-rotation result:
  `docs/loss3_direction_rotation_path_fno_nu0p001_result_20260516.md`.
- Expanded subspace result:
  `docs/loss3_jacobian_subspace_rotation_path_fno_nu0p001_result_20260516.md`.
- Output directories:
  `forensics/loss3_direction_rotation_path_20260516/fno_nu0p001_pilot/` and
  `forensics/loss3_jacobian_subspace_rotation_path_20260516/fno_nu0p001/`.

Observed results from the direction-rotation run:

| path fraction `t` | mean angle to clean direction | mean adjacent angle | mean local gain ratio over clean direction |
|---|---:|---:|---:|
| `0` | `0` | `NA` | `1.000` |
| `0.25` | `50.2151 deg` | `50.2151 deg` | `2.6753` |
| `0.5` | `50.7279 deg` | `18.7846 deg` | `3.2327` |
| `0.75` | `55.4477 deg` | `12.7040 deg` | `3.1452` |
| `1` | `57.0735 deg` | `4.7341 deg` | `2.6840` |

Observed results from the expanded Jacobian-subspace run:

- Mean top-1 angle to the clean residual-Jacobian direction is already
  `40.55 deg` by `t=0.1` and reaches `57.64 deg` at `t=1`.
- The endpoint top-4 max principal angle to the clean subspace is `81.13 deg`;
  the endpoint top-8 max principal angle is `86.79 deg`.
- Mean spectral norm `sigma1(J_e(x_t))` grows from `0.868` at `t=0` to `8.495`
  at `t=1`.
- The random-probe response-sketch relative difference from clean reaches
  `6.767` at `t=1`.

Conclusion / role:

- Observed evidence shows that the local residual-Jacobian geometry changes
  rapidly after leaving the clean input.  The clean top direction is no longer a
  good representative of the endpoint local geometry.
- Observed evidence also shows that the residual Jacobian not only rotates but
  becomes much steeper along the path.
- Inference: a single clean-point linearization is only a starting snapshot.  It
  cannot fully explain the finite-radius `loss3_original` attack path, because
  the dangerous local directions and local amplification evolve along the path.

### Overall Story Supported By The Six Experiments

Taken together, the six experiments support the following chain:

1. `loss1` and `loss2` can find large model/solver co-movement, so they are not
   reliable proxies for true oracle-relative error.
2. Local Jacobian decomposition explains why movement, residual movement, and
   clean residual outward growth are different objects.
3. Small-epsilon sweeps show when ratio diagnostics really behave like local
   Jacobian quantities.
4. Ray profiles show that local winners do not necessarily remain endpoint
   winners at radius `epsilon=8`.
5. Boundary rescaling shows that simply forcing a local or regularized direction
   to use the full budget does not make it optimal for endpoint `loss3`.
6. Path-rotation diagnostics explain why: the residual-Jacobian geometry rotates
   and steepens along the finite-radius attack path.

The combined conclusion is that local diagnostics are useful for mechanism and
interpretability, but the finite-radius robustness objective must be evaluated
and optimized as its own endpoint problem.

