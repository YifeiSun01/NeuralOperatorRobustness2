# Burgers Metric Role Interpretation

Generated: 2026-06-14

This short note fixes the interpretation boundary for the Burgers solver7860/clean8000 audit tables. The large ranked tables keep all numeric fields, but not every numeric field is evidence that one model is closer to the solver.

## Main Rule

Use `metric_best_summary_six_model_evidence_ranked.csv` for six-model claims.

Do not use the all-metric table as a scoreboard. It intentionally contains evidence metrics, partial-scope metrics, and diagnostic/process quantities.

## Counts That Matter

- Six-model evidence metrics: 168 best-model rows.
- Evidence metrics with partial coverage only: 90 best-model rows.
- Diagnostic/process metrics: 63 best-model rows.

Six-model evidence best counts:

| best model | metric rows |
| --- | ---: |
| loss3 | 113 |
| random_solver_y | 51 |
| loss1 | 4 |

Partial-scope evidence rows are recorded but do not support six-model claims. For example, random-only residual metrics compare `random_solver_y` against `random_clean_y`; they do not compare either random model against `loss3`.

## Metrics Not To Count As Model Quality Evidence

`model_spectral_norm` is the largest singular value of the model Jacobian alone. It says how large the model's local input-output gain is, but it does not say whether the model is close to the solver. It is tagged as `operator_scale_not_error`.

`solver_spectral_norm` is the same kind of quantity for the solver. It is a reference scale, not a model ranking metric.

`final_delta_rms_mean`, `delta_l2`, and related delta-size fields are attack budget/process quantities. They are constrained by the attack protocol, so smaller or larger values should not be used as evidence that a model is better.

`attack_delta_svd_abs_cos`, `attack_delta_outward_abs_cos`, `attack_delta_affine_eps_abs_cos`, `delta_top_error_sv_abs_cos`, and angle fields measure direction alignment. They can help explain attack mechanisms, but they do not directly measure model-solver closeness.

`j_error_delta_l2` and `j_error_delta_rms` measure the response of the error Jacobian to the already chosen attack perturbation: `||J_error delta||`. This is a directional-response diagnostic, not a standalone local-closeness score.

`error_effective_rank` describes spectrum shape. It is not the same as error magnitude.

Gain ratios such as `affine_over_svd_gain_ratio` compare diagnostic directions. They are mechanism diagnostics, not quality evidence.

## Metrics That Can Count As Evidence

Clean accuracy evidence:

- RMSE
- Relative L2
- MSE

Attack outcome evidence, with the documented mixed-source caveat for the recovered full-52 table:

- attack initial loss/MSE
- attack final loss/MSE
- attack loss increase

Clean residual evidence:

- `clean_residual_mse_recomputed`
- `clean_residual_norm_l2`
- old/random-prefixed residual variants, only within their coverage class

Local error-gradient evidence:

- `bias_gradient_norm`
- `bias_gradient_rms`
- `j_error_transpose_error_l2`
- `j_error_transpose_error_rms`
- `atb_norm`

These are based on `J_error^T e`, where `J_error = J_model - J_solver` and `e = model(x) - solver(x)`.

Error-operator evidence:

- error singular values
- top-k error singular spectrum
- `error_spectral_norm`
- `error_fro_norm`
- comparable/top100 SVD supplement metrics

Local similarity evidence:

- model-solver top-k left/right subspace cosine metrics

Local response evidence:

- finite-epsilon local gain MSE metrics, when reported as absolute MSE/linear/quadratic MSE rather than as ratios.

## Current Careful Interpretation

Loss3 is strongly supported on clean/generalization accuracy and on the error-operator SVD spectrum.

Loss3 is also usually strongest on the 25-sample six-model robustness/Jacobian/SVD evidence rows, but not every evidence family is a blanket loss3 win. Some six-model evidence rows favor `random_solver_y`, especially recovered full-52 attack outcome rows and some model-solver top50/top100 subspace similarity rows.

Direction cosines, attack delta sizes, model-only spectral norm, and random-only residual metrics must not be used to argue against or for loss3 in the six-model claim.
