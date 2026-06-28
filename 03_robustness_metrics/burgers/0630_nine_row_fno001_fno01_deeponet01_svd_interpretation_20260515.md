# Nine-Row FNO/DeepONet Local SVD Interpretation

Date: 2026-05-15

This note summarizes the already-computed local Jacobian SVD diagnostics for five initial-condition indices `0, 7, 40, 47, 115`.  No new Jacobians are computed here; this document aggregates existing SVD, direction-response, frequency, subspace-angle, and orthogonality outputs.

## Compared Operators

| row group | model Jacobian | solver Jacobian | error Jacobian |
|---|---|---|---|
| FNO nu=0.001 | `J_f` | `J_j(nu=0.001)` | `J_f - J_j` |
| FNO nu=0.01 | `J_f` | `J_j(nu=0.01)` | `J_f - J_j` |
| DeepONet nu=0.01 | `J_d` | `J_j(nu=0.01)` | `J_d - J_j` |

The most important qualitative result is: FNO is locally solver-like, while DeepONet is locally dominated by high-frequency model sensitivity.  For FNO, the error Jacobian is a small residual left after two similar maps subtract.  For DeepONet, the error Jacobian is almost the DeepONet Jacobian itself because the solver response is much smaller and nearly orthogonal to the DeepONet dominant directions.

## Singular-Value Scale

| metric | FNO nu=0.001 | FNO nu=0.01 | DeepONet nu=0.01 |
|---|---:|---:|---:|
| model spectral norm | 3.895 | 1.379 | 7.202 |
| solver spectral norm | 4.095 | 1.373 | 1.373 |
| error spectral norm | 0.8682 | 0.0373 | 7.1 |
| model Frobenius norm | 6.047 | 2.347 | 11.68 |
| solver Frobenius norm | 6.349 | 2.262 | 2.262 |
| error Frobenius norm | 1.493 | 0.5883 | 11.48 |
| model effective rank | 4.963 | 8.358 | 7.043 |
| solver effective rank | 4.809 | 4.747 | 4.747 |
| error effective rank | 15.08 | 782.02 | 7.077 |

Interpretation:

- FNO `nu=0.001`: model and solver have almost the same spectral scale (`3.895` vs `4.095`), and the error spectral norm is much smaller (`0.868`).
- FNO `nu=0.01`: model and solver are even closer (`1.379` vs `1.373`), and the error spectral norm collapses to `0.0373`, but its effective rank is very high (`782`), meaning the residual is tiny in amplitude but spread across many weak directions.
- DeepONet `nu=0.01`: model spectral norm is huge compared with the solver (`7.202` vs `1.373`), and the error spectral norm is almost identical to the DeepONet model (`7.100`).

## Model Top-8 Direction Response

These values answer: if the perturbation direction is chosen from the model top-8 right singular vectors, how much does the model move, how much does the solver move, and how much model-solver mismatch remains?

| metric along model top-8 directions | FNO nu=0.001 | FNO nu=0.01 | DeepONet nu=0.01 |
|---|---:|---:|---:|
| model response gain | 1.703 | 0.6752 | 3.664 |
| solver response gain | 1.773 | 0.6717 | 0.177 |
| mismatch gain | 0.3378 | 0.01552 | 3.529 |
| model/solver response cosine | 0.981 | 0.9993 | 0.718 |
| relative mismatch D | 0.1918 | 0.0326 | 0.9642 |

Interpretation:

- FNO `nu=0.001`: model top-8 directions are also solver-active directions.  The solver response `1.773` is comparable to the FNO response `1.703`; cosine is `0.981`; mismatch is only `0.338`.
- FNO `nu=0.01`: the match is stronger.  Model response `0.675`, solver response `0.672`, cosine `0.9993`, mismatch `0.0155`.
- DeepONet `nu=0.01`: DeepONet top-8 directions do not activate the solver much.  DeepONet response `3.664`, solver response only `0.177`, mismatch `3.529`, cosine `0.718`.

## Frequency and Roughness of Right Singular Vectors

`hi128` is the Fourier energy fraction above the first 128 modes; zero crossings count oscillations in the right singular vector.  Larger values mean more high-frequency / jagged structure.

| metric | FNO nu=0.001 model | FNO nu=0.001 error | FNO nu=0.01 model | FNO nu=0.01 error | DeepONet nu=0.01 model | DeepONet nu=0.01 error |
|---|---:|---:|---:|---:|---:|---:|
| top-1 hi128 | 2.231e-07 | 4.559e-05 | 2.820e-09 | 0.1957 | 0.7614 | 0.8033 |
| top-8 mean hi128 | 4.341e-07 | 2.791e-04 | 2.519e-08 | 0.734 | 0.7445 | 0.7745 |
| top-1 zero crossings | 13.60 | 16.80 | 0.4 | 14.40 | 507.20 | 516.00 |
| top-8 mean zero crossings | 12.47 | 23.00 | 4 | 23.07 | 503.27 | 517.17 |
| top-1 rough1 | 0.01307 | 0.02458 | 0.005991 | 0.3858 | 1.412 | 1.433 |
| top-1 rough2 | 0.001296 | 0.00778 | 1.174e-04 | 0.6415 | 2.424 | 2.459 |

Interpretation:

- FNO `nu=0.001` model is very low-frequency: model top-1 `hi128 = 2.23e-7`, zero crossings `13.6`.  Its error direction is still mostly low-frequency at top-1, but slightly less smooth.
- FNO `nu=0.01` model is almost purely low-frequency at top-1: `hi128 = 2.82e-9`, zero crossings `0.4`.  Its error top-1 has `hi128 = 0.1957` but zero crossings only `14.4`; this means the error residual contains more high-frequency energy than the model, while still much smaller in amplitude.
- DeepONet is the outlier: model top-1 `hi128 = 0.761`, zero crossings `507.2`; error top-1 `hi128 = 0.803`, zero crossings `516`.  The DeepONet error direction is basically the same high-frequency object as the DeepONet model direction.

## Within-Family Subspace Alignment

| pair / metric | FNO nu=0.001 | FNO nu=0.01 | DeepONet nu=0.01 |
|---|---:|---:|---:|
| model vs solver top-1 cosine | 0.9961 | 0.999 | 0.1247 |
| model vs solver top-8 mean cosine | 0.9812 | 0.9987 | 0.1697 |
| model vs solver top-8 worst angle deg | 24.85 | 6.944 | 87.97 |
| model vs error top-1 cosine | 0.2182 | 0.4272 | 0.9848 |
| model vs error top-8 mean cosine | 0.6019 | 0.08777 | 0.9807 |
| model vs error top-8 worst angle deg | 88.56 | 90.00 | 18.45 |
| solver vs error top-1 cosine | 0.2065 | 0.4379 | 0.009707 |
| solver vs error top-8 mean cosine | 0.6088 | 0.08768 | 0.02409 |
| solver vs error top-8 worst angle deg | 88.43 | 90.00 | 89.93 |

Interpretation:

- FNO `nu=0.001`: model and solver subspaces are highly aligned overall: top-1 cosine `0.996`, top-8 mean cosine `0.981`.  Error is not aligned with the model: model-vs-error top-1 angle is about `74.3 deg`.
- FNO `nu=0.01`: model and solver are almost the same local linear map: top-1 cosine `0.999`, top-8 mean cosine `0.9987`, top-8 worst angle only `6.94 deg`.  The error subspace is strongly different once you look beyond rank 1.
- DeepONet `nu=0.01`: model and solver are almost orthogonal in dominant directions: top-1 cosine `0.125`, top-8 mean cosine `0.170`, worst top-8 angle about `88 deg`.  But DeepONet model and DeepONet error are very aligned: top-1 cosine `0.985`, top-8 mean cosine `0.981`.

## Model-vs-Model Subspace Comparison

This is newly aggregated here from existing right singular vectors.  It compares only model Jacobian right-singular subspaces, not model-vs-solver pairs.

| model pair | top-1 cosine | top-8 mean cosine | top-8 worst angle deg |
|---|---:|---:|---:|
| FNO nu=0.001 model vs DeepONet nu=0.01 model | 0.02456 | 0.07638 | 89.85 |
| FNO nu=0.001 model vs FNO nu=0.01 model | 0.004423 | 0.03379 | 90.00 |
| FNO nu=0.01 model vs DeepONet nu=0.01 model | 0.02419 | 0.07315 | 89.64 |

Interpretation:

- All cross-model dominant right-singular subspaces are weakly aligned in this direct comparison.  This is not the same question as model-vs-solver alignment inside one family.
- The clean architecture comparison is FNO `nu=0.01` vs DeepONet `nu=0.01`: top-1 cosine `0.024`, top-8 mean cosine `0.073`, so their dominant model directions are almost orthogonal.  This matches the visual result that FNO is low-frequency/solver-like while DeepONet is high-frequency.
- FNO `nu=0.001` vs FNO `nu=0.01` also has very low direct top-subspace cosine.  Because this changes viscosity, trained checkpoint, and solver dynamics, treat it as a descriptive cross-run fact, not as evidence that FNO is not solver-like within each run.

## Singular-Vector Orthogonality

| operator | max off-diagonal abs in top-8 Gram | Frobenius off-diagonal | max diagonal error |
|---|---:|---:|---:|
| FNO nu=0.001 model | 2.208e-09 | 4.739e-09 | 7.148e-09 |
| FNO nu=0.001 solver | 1.937e-09 | 3.893e-09 | 1.524e-08 |
| FNO nu=0.001 error | 3.327e-09 | 7.868e-09 | 6.617e-09 |
| FNO nu=0.01 model | 2.513e-09 | 7.380e-09 | 4.594e-09 |
| FNO nu=0.01 solver | 2.300e-09 | 7.069e-09 | 5.281e-09 |
| FNO nu=0.01 error | 2.563e-08 | 6.682e-08 | 5.070e-08 |
| DeepONet nu=0.01 model | 2.593e-09 | 8.272e-09 | 4.684e-09 |
| DeepONet nu=0.01 solver | 2.300e-09 | 7.069e-09 | 5.281e-09 |
| DeepONet nu=0.01 error | 2.437e-09 | 8.304e-09 | 3.920e-09 |

Interpretation:

Within a single SVD, right singular vectors are orthonormal by construction.  The observed off-diagonal values are around `1e-9` to `1e-8`, i.e. numerical noise.  This confirms the SVD computation is internally well-conditioned for the top-8 vector basis; the important angles are across different Jacobians, not within the same Jacobian.

## Main Conclusion

The nine-row comparison supports the mechanism:

1. FNO `nu=0.001` and especially FNO `nu=0.01` locally behave like the solver.  Their dominant singular-vector shapes and response directions are aligned with the solver, so `J_model - J_solver` is a smaller, different residual object.
2. DeepONet `nu=0.01` locally does not behave like the solver in its dominant directions.  Its leading directions are high-frequency, solver response along them is tiny, and `J_deeponet - J_solver` remains almost the same as `J_deeponet`.
3. This explains why loss3/error-Jacobian directions can differ strongly from loss1/model-Jacobian directions for FNO: the model and solver co-move, so subtracting them changes the dangerous direction.  For DeepONet, loss3 and model-sensitive directions can look much more similar because the solver contributes little along DeepONet dominant directions.
4. The apparent tension with earlier attack results is therefore not a contradiction by itself.  The local Jacobian result predicts a larger conceptual separation between loss1 and loss3 for FNO than for DeepONet, but global attack performance also depends on constraints, step size, nonlinear rollout, optimization difficulty, and whether the attack can actually find the local error-Jacobian directions.

## Source Files

- `forensics/fno_nu0p001_nu0p01_deeponet_nu0p01_ninerow_svd_diagnostics_20260515_no_std/summary.md`
- `forensics/fno_solver_jacobian_similarity_20260514_raw_recomputed/summary.md`
- `forensics/fno_nu0p01_solver_jacobian_similarity_20260515/summary.md`
- `forensics/deeponet_solver_jacobian_similarity_20260515/summary.md`
- `forensics/fno_nu0p001_nu0p01_deeponet_nu0p01_ninerow_svd_diagnostics_20260515_no_std/aggregate_model_vs_model_subspace_angles.csv`
