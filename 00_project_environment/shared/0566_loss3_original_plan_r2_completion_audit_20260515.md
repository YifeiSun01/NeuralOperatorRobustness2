# Loss3 Original Plan R2 Completion Audit - 2026-05-15

## Status

Completed remote R2 audit only. No numerical experiment was run.

Credentials were used only for read-only R2 listing / small JSON reads during
this audit and are not recorded here.

## Remote Root Inspected

`neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected`

## Key Observed R2 Evidence

Observed from R2 top-level listings:

- `docs/main_objective_mechanism_experiment1_result_20260514.md`
- `docs/fno_solver_jacobian_similarity_result_20260514.md`
- `forensics/fno_solver_jacobian_similarity_20260514/`
- `forensics/fno_nu0p01_solver_jacobian_similarity_20260515/`
- `forensics/fno_deeponet_nu0p01_comprehensive_svd_diagnostics_20260515_no_std/`
- `forensics/fno_nu0p001_nu0p01_deeponet_nu0p01_ninerow_svd_diagnostics_20260515_no_std/`
- `results/main_objective_mechanism_summary_20260514/`
- `results/three_loss_objective_round1_l2_eps8_alpha0p3/`
- `results/three_loss_batch100_full_loss3_delta_rerun_20260514_fno_eps8_alpha0p3_final_boundary/`
- `results/three_loss_batch100_full_loss3_delta_rerun_20260514_fno_eps8_alpha1p5_final_boundary/`
- `results/three_loss_batch100_full_loss3_delta_rerun_20260514_fno_eps8_alpha3p0_final_boundary/`
- `results/three_loss_batch100_full_loss3_delta_rerun_20260514_deeponet_eps8_alpha0p3_final_boundary/`
- `results/three_loss_batch100_full_loss3_delta_rerun_20260514_deeponet_eps8_alpha1p5_final_boundary/`
- `results/three_loss_batch100_full_loss3_delta_rerun_20260514_deeponet_eps8_alpha3p0_final_boundary/`
- `results/burgers_corrected_oldstyle_5loss_nu0p001_small_eps_only/`
- `results/burgers_loss3_fno_nu0p01_eps_sweep_plus_deeponet_batch100_random100_losses/`

## Completion Summary Using R2 Evidence

| Plan item | R2 status | Observed evidence | Remaining work |
|---|---|---|---|
| Experiment 1: Main Objective Comparison | Done | R2 contains `docs/main_objective_mechanism_experiment1_result_20260514.md`, `results/main_objective_mechanism_summary_20260514/`, and the source final-boundary run directory `results/three_loss_batch100_full_loss3_delta_rerun_20260514_fno_eps8_alpha0p3_final_boundary/`. | None for the documented FNO `nu=0.001`, `eps=8`, `alpha=0.3` comparison. |
| Experiment 2: Local Response Decomposition Table | Mostly done | R2 contains local Jacobian/SVD forensics for FNO and FNO/DeepONet comparisons. These cover model, solver, error, and random direction response tables. | Still add the explicit clean-residual outward-growth direction `v_growth*` / `A^T b` if the exact plan table is required. |
| Experiment 3: Small-Epsilon Sweep | Related old runs exist, but not done as specified | R2 contains `results/burgers_corrected_oldstyle_5loss_nu0p001_small_eps_only/` and other epsilon-sweep attack summaries. These are attack-ratio / loss3 candidate evidence, not the planned local sweep over `{1e-4,1e-3,1e-2,1e-1}` with `L_f`, `L_j`, `L_e`, `G_e`, and direction-cosine stability. | Run the planned local small-epsilon sweep if that exact claim is needed. |
| Experiment 4: Ray Profile / Local-to-Global Profile | Not observed on R2 | No R2 `docs/`, `results/`, or `forensics/` top-level evidence for a ray-profile result was observed in this audit. | Run ray-profile curves for final directions from `loss3_original`, `loss3_increment_ratio`, `loss3_residual_increment_ratio`, `loss3_regularized`, and random. |
| Experiment 5: Boundary-Rescaled Comparison | Done | R2 contains six `three_loss_batch100_full_loss3_delta_rerun_20260514_*_final_boundary/` directories and many `final_delta_summary.json`, `final_delta_diagnostics.csv`, and `final_delta_diagnostics.npz` files. | None for the already completed boundary-rescaled comparison. Optional: write a polished paper table from the JSON summaries. |
| Experiment 6: Direction Rotation Along Path | Not observed as specified on R2 | R2 contains `gradient_audit/` directories, but this audit did not observe a result that recomputes `v_e*(x_t)=argmax ||J_e(x_t)v||` along `x_t=x+t delta*`. | Run the planned path direction-rotation experiment if needed. |

## Boundary-Rescaled Evidence Details

Observed from R2 JSON summaries under:

`results/three_loss_batch100_full_loss3_delta_rerun_20260514_fno_eps8_alpha0p3_final_boundary/`

Selected examples:

| tag | final `||delta||_2` mean | boundary `||delta||_2` mean | final `loss3_original` mean | boundary `loss3_original` mean |
|---|---:|---:|---:|---:|
| `loss3_original_pgd` | 7.7524 | 8.0000 | 5.3949 | 5.4525 |
| `loss3_increment_ratio_pgd` | 7.8499 | 8.0000 | 4.1478 | 4.2225 |
| `loss3_regularized_pgd` | 0.3018 | 8.0000 | 0.4928 | 2.0995 |
| `loss2_increment_ratio_pgd` | 6.3068 | 8.0000 | 2.0591 | 3.1936 |
| `loss2_regularized_pgd` | 5.1939 | 8.0000 | 1.9873 | 3.6531 |

Observed from these examples:

- Regularized objectives can strongly penalize the perturbation norm, producing
  final deltas well inside the boundary. The clearest example is
  `loss3_regularized_pgd`, with mean final `||delta||_2 = 0.3018` under
  `epsilon = 8`.
- Rescaling such directions to the boundary can increase endpoint
  `loss3_original`, but the rescaled endpoint is still worse than directly
  optimizing `loss3_original`. For example, `loss3_regularized_pgd` rises from
  `0.4928` to `2.0995` after boundary rescaling, still far below
  `loss3_original_pgd` at `5.4525` on the boundary.
- Increment-ratio directions also do not become best endpoint directions merely
  by rescaling. For example, `loss3_increment_ratio_pgd` has boundary
  `loss3_original = 4.2225`, below the direct `loss3_original_pgd` boundary
  value `5.4525`; `loss2_increment_ratio_pgd` is lower still at `3.1936`.

Inference from the R2 evidence:

- The boundary-rescaled comparison supports the user's stated conclusion:
  local efficiency / cost-aware objectives can find directions that look good
  locally or internally, but scaling those directions to the full budget does
  not necessarily make them the best finite-radius endpoint attack.
- This is evidence for nonlinear local-to-global mismatch: the finite-radius
  endpoint problem should be optimized directly when the goal is largest
  `loss3_original` at radius `epsilon`.

## Other R2 Notes

- R2 contains `results/three_loss_objective_round1_l2_eps8_alpha0p3/`, with the
  27-run grid for three losses x three objective variants x three optimizers at
  `p=2`, `q=2`, `epsilon=8`, `alpha=0.3`, `steps=100`, `seed=0`, index `0`.
  The directory listing showed per-run `metrics.csv`, `summary.json`, and
  per-step arrays including all nine objective evaluations.
- This R2 evidence is stronger than the local-only audit for Experiment 5,
  because the current local working tree does not contain the final-boundary
  result directories, while R2 does.

## Remaining Work Priority After R2 Audit

1. Ray Profile / Local-to-Global Profile.
2. Planned Small-Epsilon Sweep with local operator/risk-growth estimators.
3. Exact Direction Rotation Along Path using `v_e*(x_t)`.
4. Optional: add explicit `A^T b` outward-growth direction to the local response table.
5. Optional: create a compact paper-ready boundary-rescaled table from the R2 JSON summaries.
