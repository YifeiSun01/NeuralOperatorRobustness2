# FNO vs Solver Local Jacobian Similarity Result

Date: 2026-05-14

## Question

This experiment checks the local-linear mechanism behind the Experiment 1
finite-radius result:

> Are the FNO Jacobian `J_f` and the solver/oracle Jacobian `J_j` locally
> similar, so that many directions with large `Delta f` also produce aligned
> `Delta j`?

If `J_f` and `J_j` are similar, then optimizing only model movement can find
directions where model and solver co-move.  The true local mismatch object is:

```text
J_e = J_f - J_j
```

## Setting

```text
model: FNO
solver/oracle: JAX/Exponax Burgers solver
nu: 0.001
nx: 1024
t_final: 1.0
dt: 0.001
domain: 2.0
sample indices: 0, 7, 40, 47, 115
Jacobian shape: 1024 x 1024
device: cuda
FNO Jacobian: reused from existing explicit-Jacobian outputs
solver Jacobian: newly computed by row-wise VJP/autograd
```

Script:

```text
tools/analyze_fno_solver_jacobian_similarity.py
```

Output directory:

```text
forensics/fno_solver_jacobian_similarity_20260514/
```

Main generated files:

```text
summary.md
aggregate_direction_comovement_summary.csv
aggregate_singular_value_summary.csv
aggregate_principal_angles_summary.csv
all_direction_comovement_table.csv
all_singular_vector_overlap.csv
all_fourier_gain_comparison.csv
index_*/...
```

## Aggregate Singular Values

Values are mean +/- std across the five sampled indices.

| Jacobian | spectral norm | Frobenius norm | effective rank | top-8 energy | top-32 energy |
|---|---:|---:|---:|---:|---:|
| `J_f` FNO | 3.895 +/- 0.389 | 6.047 +/- 0.530 | 4.963 +/- 0.758 | 0.972 +/- 0.013 | 0.996 +/- 0.001 |
| `J_j` solver | 4.095 +/- 0.457 | 6.349 +/- 0.510 | 4.809 +/- 0.758 | 0.971 +/- 0.013 | 1.000 +/- 0.000 |
| `J_e = J_f - J_j` | 0.868 +/- 0.202 | 1.493 +/- 0.244 | 15.079 +/- 4.966 | 0.811 +/- 0.053 | 0.926 +/- 0.023 |

Interpretation:

- `J_f` and `J_j` have very similar leading scale:
  spectral norms around `3.9--4.1`.
- The residual Jacobian `J_e` is much smaller:
  spectral norm `0.868`, about `22%` of the FNO spectral norm.
- This supports the co-movement story: the model and solver have similar local
  dominant response, and the mismatch Jacobian is a smaller residual object.

## Direction Co-Movement

For directions `v` taken from top right singular vectors of each Jacobian, this
table records:

```text
||J_f v||
||J_j v||
||J_f v - J_j v||
cos(J_f v, J_j v)
D_f = ||J_f v - J_j v|| / (||J_f v|| + eta)
```

Values are aggregated over top-8 directions from each source and over five
sample indices.

| direction source | `||J_f v||` | `||J_j v||` | `||J_f v - J_j v||` | `cos(J_f v,J_j v)` | `D_f` |
|---|---:|---:|---:|---:|---:|
| top `J_f` directions | 1.703 +/- 1.260 | 1.773 +/- 1.330 | 0.338 +/- 0.301 | 0.981 +/- 0.017 | 0.192 +/- 0.083 |
| top `J_j` directions | 1.699 +/- 1.256 | 1.777 +/- 1.334 | 0.340 +/- 0.303 | 0.981 +/- 0.015 | 0.192 +/- 0.080 |
| top `J_e` directions | 1.628 +/- 1.200 | 1.708 +/- 1.278 | 0.412 +/- 0.259 | 0.945 +/- 0.070 | 0.329 +/- 0.190 |
| random directions | 0.164 +/- 0.066 | 0.173 +/- 0.067 | 0.042 +/- 0.015 | 0.966 +/- 0.018 | 0.270 +/- 0.072 |

Interpretation:

- Along FNO top directions, `J_f v` and `J_j v` are extremely aligned:
  cosine `0.981 +/- 0.017`.
- Solver top directions show the same pattern:
  cosine `0.981 +/- 0.015`.
- Top residual directions have larger mismatch and larger `D_f`, but their
  responses are still often positively aligned.  The residual is not usually an
  opposite-direction effect; it is often a smaller amplitude/shape difference
  between two broadly co-moving responses.

This directly supports Experiment 1:

```text
large model movement can be large model-solver co-movement
```

Therefore, `loss1_original` can find high `||Delta f||` directions without
finding the largest regression error.

## Principal-Angle / Subspace Similarity

Top-k right-singular-vector subspace comparison between `J_f` and `J_j`:

| k | mean principal cosine | mean angle |
|---:|---:|---:|
| 1 | 0.996 | 4.86 deg |
| 2 | 0.997 | 4.10 deg |
| 4 | 0.947 | 8.69 deg |
| 8 | 0.981 | 7.01 deg |
| 16 | 0.925 | 14.71 deg |
| 32 | 0.577 | 42.87 deg |

Interpretation:

- The leading local input-response subspaces of FNO and solver are highly
  similar for small-to-moderate `k`, especially `k <= 8`.
- The top-32 comparison drops because lower-energy tail directions are less
  stable and less aligned.  This does not weaken the main conclusion, because
  top-8 already contains about `97%` of the singular energy for both `J_f` and
  `J_j`.

Subspace similarity between `J_f` and `J_e` is much weaker in the leading
direction:

```text
k=1 mean principal cosine for J_f vs J_e: 0.218
k=1 mean principal cosine for J_j vs J_e: 0.206
```

So the top residual/mismatch direction is generally not the same as the top
model-sensitivity or top solver-sensitivity direction.

## Fourier Gain

Selected mean Fourier gains across five indices:

| frequency k | `||J_f phi_k||` | `||J_j phi_k||` | `||J_e phi_k||` | `||J_e phi_k|| / ||J_f phi_k||` |
|---:|---:|---:|---:|---:|
| 1 | 2.517 | 2.644 | 0.521 | 0.207 |
| 2 | 1.970 | 2.038 | 0.416 | 0.211 |
| 4 | 0.842 | 0.869 | 0.185 | 0.220 |
| 8 | 0.380 | 0.388 | 0.093 | 0.245 |
| 16 | 0.0145 | 0.1525 | 0.1547 | 10.646 |
| 32 | 0.0138 | 0.0230 | 0.0262 | 1.899 |
| 64 | 0.0138 | 0.000203 | 0.0138 | 1.000 |
| 128 | 0.0138 | 1.21e-7 | 0.0138 | 1.000 |
| 256 | 0.0138 | 1.41e-8 | 0.0138 | 1.000 |
| 512 | 0.0392 | 0.0335 | 0.0160 | 0.409 |

Interpretation:

- At low frequencies, FNO and solver gains are close, and the residual gain is
  only about `20%--25%` of the FNO gain.
- At high frequencies, the physical solver is strongly damped, while FNO has a
  small but nonzero residual response.  The relative residual can therefore look
  large, but the absolute high-frequency gains are still much smaller than the
  leading low-frequency gains.

## Main Conclusion

This local-Jacobian experiment strongly supports the mechanism proposed after
Experiment 1:

```text
J_f and J_j are locally very similar in their dominant directions.
```

In particular:

- `J_f` and `J_j` have similar leading singular values.
- Their top input-response directions are highly aligned.
- Along top `J_f` directions, `J_f v` and `J_j v` have cosine about `0.981`.
- The residual Jacobian `J_e = J_f - J_j` is much smaller in spectral norm.

Therefore:

```text
high ||J_f v|| does not imply high ||(J_f - J_j)v||
```

because often:

```text
J_f v and J_j v point in nearly the same direction.
```

This is the local-linear explanation for why `loss1_original` can create large
model movement without maximizing true regression error.  It also supports the
argument that `loss3_original` is the right finite-radius regression attack
objective, because it targets the model-oracle mismatch rather than model
movement alone.

---

# Comprehensive Synthesis For The Loss3 Argument

This section collects the full result in one place: the finite-radius objective
comparison, the local Jacobian/SVD mechanism, the perturbation-theory
interpretation, and the role of these findings in the paper.

## A. Central Claim

The central claim is:

```text
For neural-operator regression, adversarial risk should be measured by
oracle-relative disagreement, not by model-output movement alone.
```

Let:

```text
f = learned neural operator
j = numerical solver / oracle
e = f - j
```

At input `x` with perturbation `delta`:

```text
Delta f = f(x + delta) - f(x)
Delta j = j(x + delta) - j(x)
b       = f(x) - j(x)
```

Then:

```text
e(x + delta) = f(x + delta) - j(x + delta)
             = b + Delta f - Delta j
```

So the key dangerous quantity is not only:

```text
Delta f
```

but:

```text
Delta f - Delta j
```

This is the model-solver response mismatch. If `Delta f` and `Delta j` are
large but aligned, then the model can move a lot without creating the largest
regression error.

## B. Objective Interpretation

| objective | quantity | interpretation |
|---|---|---|
| `loss1_original` | `||Delta f||` | model output movement |
| `loss2_original` | `||b + Delta f||` | fixed clean-target error |
| `loss3_original` | `||b + Delta f - Delta j||` | perturbed-input oracle-relative error |

Thus:

```text
loss1_original ignores the solver response.
loss2_original freezes the oracle target at the clean input.
loss3_original compares the model to the oracle at the perturbed input.
```

This is why `loss1_original` and `loss2_original` are useful baselines, but
not reliable substitutes for `loss3_original`.

## C. Experiment 1: Finite-Radius Evidence

Setting:

```text
model: FNO
dataset/task: 1D Burgers
nu: 0.001
nx: 1024
t_final: 1.0
dt: 0.001
domain: 2.0
epsilon: 8.0
alpha: 0.3
steps: 100
batch size: 100
dataset indices: 0..99
delta kind: final delta
```

Source data:

```text
results/main_objective_mechanism_summary_20260514/focused_original_objectives.csv
results/main_objective_mechanism_summary_20260514/combined_mechanism_summary.csv
```

All values below are mean +/- std over 100 samples.

### C.1 PGD

| optimized objective | `||Delta f||` | `||Delta j||` | `||Delta f - Delta j||` | `||e(x + delta)||` | `cos(Delta f, Delta j)` |
|---|---:|---:|---:|---:|---:|
| `loss1_original` | 11.172 +/- 0.918 | 10.630 +/- 0.993 | 4.351 +/- 1.837 | 4.340 +/- 1.830 | 0.910 +/- 0.068 |
| `loss2_original` | 11.295 +/- 0.989 | 10.700 +/- 0.854 | 4.377 +/- 1.908 | 4.372 +/- 1.917 | 0.912 +/- 0.063 |
| `loss3_original` | 5.866 +/- 3.237 | 5.839 +/- 2.582 | 5.374 +/- 2.773 | 5.395 +/- 2.748 | 0.565 +/- 0.219 |

Additional diagnostics:

| optimized objective | error growth | `D_f` | `D_sym` | `||Delta f - Delta j|| / ||delta||` |
|---|---:|---:|---:|---:|
| `loss1_original` | 4.042 +/- 1.822 | 0.385 +/- 0.150 | 0.199 +/- 0.084 | 0.544 +/- 0.230 |
| `loss2_original` | 4.074 +/- 1.913 | 0.382 +/- 0.152 | 0.197 +/- 0.082 | 0.547 +/- 0.239 |
| `loss3_original` | 5.097 +/- 2.718 | 1.012 +/- 0.376 | 0.469 +/- 0.113 | 0.682 +/- 0.336 |

### C.2 LP-Steepest PGD

| optimized objective | `||Delta f||` | `||Delta j||` | `||Delta f - Delta j||` | `||e(x + delta)||` | `cos(Delta f, Delta j)` |
|---|---:|---:|---:|---:|---:|
| `loss1_original` | 11.149 +/- 0.928 | 10.621 +/- 1.014 | 4.331 +/- 1.767 | 4.319 +/- 1.762 | 0.912 +/- 0.064 |
| `loss2_original` | 11.303 +/- 0.979 | 10.719 +/- 0.868 | 4.355 +/- 1.868 | 4.349 +/- 1.876 | 0.913 +/- 0.060 |
| `loss3_original` | 7.154 +/- 2.853 | 6.718 +/- 2.229 | 6.372 +/- 2.300 | 6.378 +/- 2.288 | 0.561 +/- 0.187 |

Additional diagnostics:

| optimized objective | error growth | `D_f` | `D_sym` | `||Delta f - Delta j|| / ||delta||` |
|---|---:|---:|---:|---:|
| `loss1_original` | 4.022 +/- 1.756 | 0.384 +/- 0.143 | 0.198 +/- 0.080 | 0.541 +/- 0.221 |
| `loss2_original` | 4.051 +/- 1.872 | 0.380 +/- 0.148 | 0.196 +/- 0.080 | 0.544 +/- 0.233 |
| `loss3_original` | 6.081 +/- 2.262 | 0.955 +/- 0.272 | 0.471 +/- 0.099 | 0.796 +/- 0.288 |

### C.3 Generalized Power

| optimized objective | `||Delta f||` | `||Delta j||` | `||Delta f - Delta j||` | `||e(x + delta)||` | `cos(Delta f, Delta j)` |
|---|---:|---:|---:|---:|---:|
| `loss1_original` | 11.015 +/- 0.881 | 10.720 +/- 0.895 | 4.094 +/- 1.848 | 4.083 +/- 1.844 | 0.916 +/- 0.068 |
| `loss2_original` | 11.158 +/- 0.932 | 10.670 +/- 0.811 | 4.251 +/- 1.935 | 4.243 +/- 1.944 | 0.913 +/- 0.062 |
| `loss3_original` | 8.526 +/- 1.507 | 7.878 +/- 1.633 | 6.862 +/- 1.424 | 6.857 +/- 1.429 | 0.615 +/- 0.201 |

Additional diagnostics:

| optimized objective | error growth | `D_f` | `D_sym` | `||Delta f - Delta j|| / ||delta||` |
|---|---:|---:|---:|---:|
| `loss1_original` | 3.786 +/- 1.838 | 0.369 +/- 0.159 | 0.188 +/- 0.086 | 0.512 +/- 0.231 |
| `loss2_original` | 3.945 +/- 1.941 | 0.376 +/- 0.158 | 0.194 +/- 0.085 | 0.531 +/- 0.242 |
| `loss3_original` | 6.560 +/- 1.414 | 0.826 +/- 0.217 | 0.430 +/- 0.114 | 0.858 +/- 0.178 |

### C.4 Main Finite-Radius Finding

The cleanest example is the generalized-power result:

```text
loss1_original:
  ||Delta f||              = 11.015 +/- 0.881
  ||Delta f - Delta j||    = 4.094 +/- 1.848
  ||e(x + delta)||         = 4.083 +/- 1.844
  cos(Delta f, Delta j)    = 0.916 +/- 0.068

loss3_original:
  ||Delta f||              = 8.526 +/- 1.507
  ||Delta f - Delta j||    = 6.862 +/- 1.424
  ||e(x + delta)||         = 6.857 +/- 1.429
  cos(Delta f, Delta j)    = 0.615 +/- 0.201
```

So `loss1_original` makes the model move more, but `loss3_original` makes the
model disagree with the solver more.

This supports:

```text
high model sensitivity != high regression error
```

## D. Local Jacobian/SVD Evidence

The local-linear objects are:

```text
J_f = d f(x) / d x
J_j = d j(x) / d x
J_e = J_f - J_j
```

Local linearization gives:

```text
f(x + delta) - f(x) ~= J_f delta
j(x + delta) - j(x) ~= J_j delta
```

Therefore:

```text
(f - j)(x + delta) - (f - j)(x)
~= (J_f - J_j) delta
= J_e delta
```

Setting:

```text
model: FNO
solver/oracle: JAX/Exponax Burgers solver
sample indices: 0, 7, 40, 47, 115
Jacobian shape: 1024 x 1024
device: cuda
```

Source data:

```text
forensics/fno_solver_jacobian_similarity_20260514/
```

### D.1 Singular Values

Values are mean +/- std across five sample indices.

| Jacobian | spectral norm | Frobenius norm | effective rank | top-8 energy | top-32 energy |
|---|---:|---:|---:|---:|---:|
| `J_f` FNO | 3.895 +/- 0.389 | 6.047 +/- 0.530 | 4.963 +/- 0.758 | 0.972 +/- 0.013 | 0.996 +/- 0.001 |
| `J_j` solver | 4.095 +/- 0.457 | 6.349 +/- 0.510 | 4.809 +/- 0.758 | 0.971 +/- 0.013 | 1.000 +/- 0.000 |
| `J_e = J_f - J_j` | 0.868 +/- 0.202 | 1.493 +/- 0.244 | 15.079 +/- 4.966 | 0.811 +/- 0.053 | 0.926 +/- 0.023 |

Interpretation:

```text
J_f and J_j have similar dominant local sensitivity.
J_e is much smaller than either J_f or J_j.
```

This suggests that the model and solver share dominant response directions,
and their difference is a smaller residual operator.

### D.2 Pairwise Singular Values And Singular Vectors

| rank | `sigma_f` | `sigma_j` | abs diff | rel diff vs solver | right cos | right angle | left cos | left angle |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3.895 | 4.095 | 0.200 | 0.047 | 0.996 | 4.86 deg | 0.985 | 9.48 deg |
| 2 | 3.125 | 3.277 | 0.152 | 0.046 | 0.996 | 5.18 deg | 0.967 | 14.55 deg |
| 3 | 2.114 | 2.207 | 0.093 | 0.045 | 0.794 | 23.71 deg | 0.781 | 27.56 deg |
| 4 | 1.784 | 1.854 | 0.129 | 0.073 | 0.600 | 39.07 deg | 0.596 | 40.66 deg |
| 5 | 1.048 | 1.084 | 0.046 | 0.040 | 0.452 | 53.83 deg | 0.453 | 54.20 deg |
| 6 | 0.645 | 0.676 | 0.032 | 0.041 | 0.635 | 41.17 deg | 0.630 | 42.49 deg |
| 7 | 0.542 | 0.538 | 0.016 | 0.029 | 0.786 | 27.04 deg | 0.784 | 27.34 deg |
| 8 | 0.470 | 0.483 | 0.020 | 0.042 | 0.687 | 40.44 deg | 0.681 | 41.17 deg |
| 9 | 0.420 | 0.430 | 0.011 | 0.027 | 0.669 | 43.58 deg | 0.672 | 43.15 deg |
| 10 | 0.393 | 0.410 | 0.016 | 0.040 | 0.578 | 50.84 deg | 0.590 | 49.47 deg |

Top-K aggregate summaries:

| top K ranks | mean abs singular diff | mean relative diff | right vector mean cos | right vector mean angle | left vector mean cos | left vector mean angle |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.124 +/- 0.098 | 0.050 +/- 0.033 | 0.768 +/- 0.406 | 25.33 +/- 34.86 deg | 0.756 +/- 0.400 | 29.29 +/- 32.75 deg |
| 8 | 0.086 +/- 0.093 | 0.046 +/- 0.033 | 0.743 +/- 0.398 | 29.41 +/- 34.00 deg | 0.735 +/- 0.392 | 32.18 +/- 32.28 deg |
| 10 | 0.072 +/- 0.088 | 0.043 +/- 0.032 | 0.719 +/- 0.389 | 32.97 +/- 33.11 deg | 0.714 +/- 0.384 | 35.01 +/- 31.50 deg |
| 20 | 0.069 +/- 0.065 | 0.179 +/- 0.179 | 0.575 +/- 0.383 | 47.66 +/- 31.54 deg | 0.619 +/- 0.382 | 44.88 +/- 30.27 deg |

Interpretation:

- Rank 1 and rank 2 right singular vectors are extremely close:
  `4.86 deg` and `5.18 deg`.
- Later rank-by-rank angles become larger. This is expected because individual
  singular vectors are less stable when singular values are close.
- The top-10 singular values themselves remain close: mean relative difference
  is about `4.3%`.

### D.3 Direction Co-Movement Table

For a direction `v`, the table records:

```text
||J_f v||
||J_j v||
||J_f v - J_j v||
cos(J_f v, J_j v)
D_f = ||J_f v - J_j v|| / (||J_f v|| + eta)
```

Values are aggregated over top-8 directions from each source and five indices.

| direction source | `||J_f v||` | `||J_j v||` | `||J_f v - J_j v||` | `cos(J_f v,J_j v)` | `D_f` |
|---|---:|---:|---:|---:|---:|
| top `J_f` directions | 1.703 +/- 1.260 | 1.773 +/- 1.330 | 0.338 +/- 0.301 | 0.981 +/- 0.017 | 0.192 +/- 0.083 |
| top `J_j` directions | 1.699 +/- 1.256 | 1.777 +/- 1.334 | 0.340 +/- 0.303 | 0.981 +/- 0.015 | 0.192 +/- 0.080 |
| top `J_e` directions | 1.628 +/- 1.200 | 1.708 +/- 1.278 | 0.412 +/- 0.259 | 0.945 +/- 0.070 | 0.329 +/- 0.190 |
| random directions | 0.164 +/- 0.066 | 0.173 +/- 0.067 | 0.042 +/- 0.015 | 0.966 +/- 0.018 | 0.270 +/- 0.072 |

This is one of the most important tables:

```text
Along the most sensitive directions of J_f and J_j, the FNO response and solver
response are almost co-linear.
```

Numerically:

```text
cos(J_f v, J_j v) ~= 0.981
```

So these are cancellation directions, not reinforcement directions.

### D.4 Top-K Subspace Angles

For:

```text
J = U S V^T
```

the right singular vectors in `V` are input perturbation directions.

For the top-k subspaces:

```text
Q_f = [v_f,1, ..., v_f,k]
Q_j = [v_j,1, ..., v_j,k]
```

compute:

```text
Q_f^T Q_j = U Sigma V^T
```

The singular values of `Q_f^T Q_j` are principal cosines:

```text
cos(theta_1), ..., cos(theta_k)
```

The angles are:

```text
theta_i = arccos(sigma_i)
```

Important:

```text
For k > 1, there are k principal angles, not one angle.
The table reports summary statistics over those k angles.
```

Results for `J_f` vs `J_j`:

| k | mean principal cosine | min principal cosine | mean principal angle | max principal angle |
|---:|---:|---:|---:|---:|
| 1 | 0.996 | 0.996 | 4.86 deg | 4.86 deg |
| 2 | 0.997 | 0.997 | 4.10 deg | 4.38 deg |
| 4 | 0.947 | 0.798 | 8.69 deg | 22.53 deg |
| 8 | 0.981 | 0.876 | 7.01 deg | 24.85 deg |
| 16 | 0.925 | 0.550 | 14.71 deg | 55.50 deg |
| 32 | 0.577 | 0.000 | 42.87 deg | 90.00 deg |

Interpretation:

```text
The leading sensitive input subspaces of J_f and J_j are close, especially up
to k = 8.
```

This is stronger and more stable than only comparing singular vectors one by
one, because top-k subspace comparison allows rotation or mixing inside the
leading subspace.

The residual/error subspace is different:

```text
k=1 mean principal cosine for J_f vs J_e: 0.218
k=1 mean principal cosine for J_j vs J_e: 0.206
```

So the top mismatch direction is not just the top model-sensitivity direction.

### D.5 Fourier Gain

| frequency k | `||J_f phi_k||` | `||J_j phi_k||` | `||J_e phi_k||` | `||J_e phi_k|| / ||J_f phi_k||` |
|---:|---:|---:|---:|---:|
| 1 | 2.517 | 2.644 | 0.521 | 0.207 |
| 2 | 1.970 | 2.038 | 0.416 | 0.211 |
| 4 | 0.842 | 0.869 | 0.185 | 0.220 |
| 8 | 0.380 | 0.388 | 0.093 | 0.245 |
| 16 | 0.0145 | 0.1525 | 0.1547 | 10.646 |
| 32 | 0.0138 | 0.0230 | 0.0262 | 1.899 |
| 64 | 0.0138 | 0.000203 | 0.0138 | 1.000 |
| 128 | 0.0138 | 1.21e-7 | 0.0138 | 1.000 |
| 256 | 0.0138 | 1.41e-8 | 0.0138 | 1.000 |
| 512 | 0.0392 | 0.0335 | 0.0160 | 0.409 |

Interpretation:

- Low-frequency FNO and solver gains are close.
- Low-frequency residual gains are about `20%--25%` of FNO gains.
- High-frequency relative residuals can look large because the solver is
  strongly damped, but absolute gains are still much smaller than dominant
  low-frequency gains.

## E. Mathematical Mechanism

For any fixed direction `v`, define:

```text
a = J_f v
b = J_j v
```

Then:

```text
||J_f v - J_j v||^2 = ||a - b||^2
```

By the cosine rule:

```text
||a - b||^2
= ||a||^2 + ||b||^2 - 2 <a, b>
```

and:

```text
<a, b> = ||a|| ||b|| cos(theta_v)
```

Therefore:

```text
||J_f v - J_j v||^2
= ||J_f v||^2
 + ||J_j v||^2
 - 2 ||J_f v|| ||J_j v|| cos(theta_v)
```

where:

```text
theta_v = angle between J_f v and J_j v
```

This directly explains cancellation and reinforcement.

If:

```text
cos(theta_v) ~= 1
```

then the two responses are aligned and the difference is small. This is
cancellation.

If:

```text
cos(theta_v) ~= -1
```

then the two responses are opposed and the difference is large. This is
reinforcement.

Our data:

```text
top J_f directions: cos(J_f v, J_j v) = 0.981 +/- 0.017
top J_j directions: cos(J_f v, J_j v) = 0.981 +/- 0.015
```

Therefore the dominant directions are cancellation directions.

This is the local-linear explanation of the finite-radius result:

```text
loss1 finds large J_f v directions.
But those directions often also have large, aligned J_j v.
So J_f v - J_j v is not maximized.
```

`loss3_original` instead targets the residual:

```text
J_e v = (J_f - J_j) v
```

and therefore is closer to the true regression failure.

## F. Weyl-Type Singular-Value Perturbation

A basic singular-value perturbation inequality says:

```text
|sigma_i(A + E) - sigma_i(A)| <= ||E||_2
```

In our setting:

```text
J_f = J_j + J_e
```

so:

```text
|sigma_i(J_f) - sigma_i(J_j)| <= ||J_e||_2
```

For the largest singular value:

```text
sigma_max(J_f)       ~= 3.895
sigma_max(J_j)       ~= 4.095
sigma_max(J_f - J_j) ~= 0.868
```

Thus:

```text
|3.895 - 4.095| = 0.200 <= 0.868
```

This explains why the singular spectra of `J_f` and `J_j` can be close when
their residual `J_e` is small.

A related bound is:

```text
|sigma_max(J_f) - sigma_max(J_j)|
<= sigma_max(J_f - J_j)
<= sigma_max(J_f) + sigma_max(J_j)
```

Numerically:

```text
lower bound = 0.200
actual      = 0.868
upper bound = 7.990
```

The actual value is far from the upper bound and much closer to the lower
bound. This is consistent with cancellation rather than reinforcement.

Important nuance:

```text
Weyl explains why J_f and J_j have close singular values if J_e is small.
It does not by itself explain why J_e is small.
The fixed-direction cosine formula explains why J_e is small along dominant
directions: J_f v and J_j v are highly aligned.
```

## G. Wedin-Type Sin-Theta Interpretation

Wedin-type sin-theta theory concerns singular-vector and singular-subspace
perturbation.

For:

```text
J_f = J_j + J_e
```

a schematic form is:

```text
||sin Theta|| <= perturbation_size / singular_value_gap
```

where:

```text
Theta = principal angles between singular subspaces
perturbation_size = size of the residual / perturbation
singular_value_gap = separation between the singular values of interest and
                     neighboring singular values
```

The practical meaning is:

```text
If the residual operator is small relative to the relevant spectral gap, the
top singular subspace is stable.
```

This is useful for our experiment because it explains why top-k subspace
comparison is the right diagnostic:

- individual singular vectors can be unstable when singular values are close;
- top-k subspaces can remain stable even if vectors rotate inside the subspace;
- therefore principal angles between top-k subspaces are more reliable than
  strict rank-by-rank singular-vector matching.

This theory does not directly prove that `loss3_original` is globally optimal.
Its role is more precise:

```text
It justifies comparing the leading sensitive subspaces of J_f and J_j, and it
explains why subspace-level similarity is the right geometric evidence for
local co-movement.
```

## H. What This Proves For The Paper

The evidence supports several concrete claims.

### H.1 Solver Response Is Not Optional

In PDE regression, the true solution changes when the input changes. Therefore:

```text
large model output change
```

does not automatically mean:

```text
large model error
```

The solver response must be included.

### H.2 Loss1 Can Produce False Alerts

`loss1_original` optimizes:

```text
||Delta f||
```

But our data show:

```text
loss1_original:
  ||Delta f|| large
  ||Delta j|| also large
  cos(Delta f, Delta j) high
  mismatch smaller than loss3
```

So `loss1_original` can identify directions where the PDE solution itself is
sensitive, not necessarily directions where the model fails.

### H.3 Loss2 Has A Fixed-Target Error

`loss2_original` freezes the oracle at the clean input. It ignores:

```text
j(x + delta) - j(x)
```

When the oracle response is non-negligible, this objective can also confuse
normal solution movement with error.

### H.4 Loss3 Targets The Right Object

`loss3_original` directly optimizes:

```text
||f(x + delta) - j(x + delta)||
```

This is the perturbed-input true regression error. It is aligned with:

```text
Delta f - Delta j
```

and locally with:

```text
(J_f - J_j) delta
```

Therefore `loss3_original` targets the difference operator / error operator,
not the learned model operator alone.

### H.5 Singular Values Alone Are Insufficient

Even if `J_f` and `J_j` both have large singular values, the error can be small
if their responses are aligned.

To understand regression failure, one must examine:

- singular values;
- right singular directions;
- left/output response directions;
- top-k subspace angles;
- response cosine `cos(J_f v, J_j v)`;
- the residual operator `J_f - J_j`.

### H.6 The Residual Operator Is The Correct Local Object

The local dangerous directions are not necessarily the top directions of
`J_f`. They are the top directions of:

```text
J_e = J_f - J_j
```

This explains why the top residual direction is not the same as the top
model-sensitivity direction:

```text
k=1 mean principal cosine for J_f vs J_e: 0.218
k=1 mean principal cosine for J_j vs J_e: 0.206
```

## I. Role In The Overall Paper

This experiment should be positioned as a mechanism section, not merely an
appendix visualization.

It provides the bridge:

```text
finite-radius loss comparison
        ->
local Jacobian response geometry
        ->
theoretical perturbation interpretation
        ->
justification for loss3_original
```

It says:

```text
loss3_original is not better because of optimizer luck.
It is better because it optimizes the correct geometric object:
the oracle-relative difference operator.
```

Future experiments should use this as the diagnostic lens:

1. If an attack increases `||Delta f||` but keeps
   `cos(Delta f, Delta j)` high, it may be a co-movement direction.
2. If an attack increases `||Delta f - Delta j||` and lowers the response
   cosine, it is closer to a true regression failure.
3. If local SVD shows `J_f` and `J_j` have aligned dominant subspaces, then
   model-only sensitivity objectives are expected to be biased.
4. If `J_e` has its own dominant directions, those are the local directions
   that expose model-oracle disagreement.

## J. Paper-Ready Summary

The finite-radius diagnostics show that model-output movement alone is a
misleading surrogate for regression failure. For FNO at `epsilon=8` and
`alpha=0.3`, `loss1_original` produces much larger model movement than
`loss3_original`, but the solver response is also large and highly aligned
with the model response. Consequently, the resulting mismatch is smaller than
that obtained by `loss3_original`.

The local Jacobian analysis explains this phenomenon. The dominant singular
values and leading right singular subspaces of the FNO Jacobian and the solver
Jacobian are close, and along their top response directions the output
responses have cosine similarity approximately `0.981`. Thus large
model-response directions often correspond to co-moving solver-response
directions, causing cancellation in `J_f - J_j`.

The difference operator isolates the residual directions where this co-movement
breaks down. This explains why `loss3_original` is more aligned with true
regression error than objectives that maximize model movement alone.

In one sentence:

```text
loss1_original optimizes model movement, whereas loss3_original optimizes
oracle-relative model failure.
```

## K. Careful Wording And Limits

The safe claim is:

```text
The experiments provide strong mechanistic evidence that loss3_original is
better aligned with true regression failure because it optimizes the
oracle-relative difference operator, while loss1/loss2 can be dominated by
co-moving model and solver responses.
```

Limitations:

- The Jacobian/SVD experiment uses five sample indices:
  `0, 7, 40, 47, 115`.
- It is local-linear evidence around clean inputs.
- The finite-radius objective comparison is broader because it evaluates
  actual nonlinear perturbed samples over 100 inputs.
- Weyl/Wedin theory provides interpretation and diagnostic justification; it
  is not a global theorem proving finite-radius optimality.

## L. References

References used for the perturbation-theory interpretation:

- UCLA SVD notes, singular-value inequalities:
  https://math.ucla.edu/~njhu/notes/nla/eig/svd/
- Springer, "Singular Value Perturbation and Deep Network Optimization":
  https://link.springer.com/article/10.1007/s00365-022-09601-5
- UW-Madison spectral algorithms notes, Davis-Kahan/Wedin-style subspace
  perturbation:
  https://pages.cs.wisc.edu/~yudongchen/cs839_sp22/6_spectral_algorithms1.pdf
- Cai and Zhang, "Rate-Optimal Perturbation Bounds for Singular Subspaces":
  https://arxiv.org/abs/1605.00353
