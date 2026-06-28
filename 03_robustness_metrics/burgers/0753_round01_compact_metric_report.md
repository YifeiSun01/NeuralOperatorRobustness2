# Burgers Round01 Final Jacobian/SVD Compact Report

This report compares the new loss3-aligned round01 final checkpoints on the new round01 sample points:

- `baseline` epoch 0
- `loss1_epoch1000`
- `loss2_epoch500`
- `loss3_epoch500`

Important detail: solver and baseline Jacobians were recomputed on the new round01 `X`, not reused from the older p2q2 representative set.

## Main Result

For the local error Jacobian,

```text
J_error(x) = J_model(x) - J_solver(x)
```

`loss1_epoch1000` is still the best overall and best on generalization by mean model-minus-solver spectral norm. `loss3_epoch500` improves a lot over baseline and is closer on the new generalization split than `loss2_epoch500`, but it is still worse than `loss1_epoch1000` on this round01 rep20/top100 diagnostic.

## Model-Minus-Solver Error Spectral Norm

| model | split | n | error mean | baseline error mean | ratio mean | ratio median | smaller than baseline |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline | ALL | 20 | 2.890990 | 2.890990 | 1.000000 | 1.000000 | 20 |
| baseline | train | 5 | 1.337703 | 1.337703 | 1.000000 | 1.000000 | 5 |
| baseline | test | 5 | 1.131711 | 1.131711 | 1.000000 | 1.000000 | 5 |
| baseline | generalization | 10 | 4.547272 | 4.547272 | 1.000000 | 1.000000 | 10 |
| loss1_epoch1000 | ALL | 20 | 0.858957 | 2.890990 | 0.339883 | 0.319523 | 20 |
| loss1_epoch1000 | train | 5 | 0.317488 | 1.337703 | 0.308900 | 0.235256 | 5 |
| loss1_epoch1000 | test | 5 | 0.403351 | 1.131711 | 0.372048 | 0.388484 | 5 |
| loss1_epoch1000 | generalization | 10 | 1.357495 | 4.547272 | 0.339293 | 0.301217 | 10 |
| loss2_epoch500 | ALL | 20 | 1.072888 | 2.890990 | 0.379300 | 0.355006 | 20 |
| loss2_epoch500 | train | 5 | 0.356644 | 1.337703 | 0.343998 | 0.236877 | 5 |
| loss2_epoch500 | test | 5 | 0.365690 | 1.131711 | 0.357092 | 0.312406 | 5 |
| loss2_epoch500 | generalization | 10 | 1.784608 | 4.547272 | 0.408055 | 0.422370 | 10 |
| loss3_epoch500 | ALL | 20 | 1.221072 | 2.890990 | 0.584777 | 0.519203 | 17 |
| loss3_epoch500 | train | 5 | 0.961461 | 1.337703 | 0.891682 | 0.591177 | 3 |
| loss3_epoch500 | test | 5 | 0.683756 | 1.131711 | 0.739958 | 0.733302 | 4 |
| loss3_epoch500 | generalization | 10 | 1.619536 | 4.547272 | 0.353735 | 0.306315 | 10 |

## Model Jacobian Spectral Norm

| model | split | spectral mean | spectral median | spectral max |
|---|---|---:|---:|---:|
| solver | ALL | 4.72115 | 4.58903 | 6.72744 |
| baseline | ALL | 4.26463 | 4.18122 | 5.96134 |
| loss1_epoch1000 | ALL | 4.58930 | 4.38625 | 6.26329 |
| loss2_epoch500 | ALL | 4.54534 | 4.26304 | 6.15242 |
| loss3_epoch500 | ALL | 4.51333 | 4.29300 | 6.08519 |
| solver | generalization | 4.98565 | 5.17000 | 6.72744 |
| baseline | generalization | 4.38145 | 4.39305 | 5.13272 |
| loss1_epoch1000 | generalization | 4.71121 | 4.59843 | 6.25118 |
| loss2_epoch500 | generalization | 4.64845 | 4.56053 | 6.13012 |
| loss3_epoch500 | generalization | 4.65351 | 4.62462 | 6.08519 |

## Rank-1 Model Direction Similarity To Solver

Values are absolute cosine similarities of the rank-1 singular vectors. Angle is `acos(absdot)` in degrees.

| split | model | model sv mean | solver sv mean | right absdot | right angle | left absdot | left angle |
|---|---|---:|---:|---:|---:|---:|---:|
| ALL | baseline | 4.264634 | 4.721146 | 0.862994 | 19.31 | 0.653091 | 42.23 |
| ALL | loss1_epoch1000 | 4.589300 | 4.721146 | 0.948262 | 7.97 | 0.932874 | 12.75 |
| ALL | loss2_epoch500 | 4.545342 | 4.721146 | 0.947978 | 8.13 | 0.921396 | 14.27 |
| ALL | loss3_epoch500 | 4.513331 | 4.721146 | 0.901483 | 11.49 | 0.871306 | 20.08 |
| generalization | baseline | 4.381449 | 4.985648 | 0.821541 | 24.06 | 0.434740 | 61.49 |
| generalization | loss1_epoch1000 | 4.711207 | 4.985648 | 0.897855 | 13.08 | 0.868428 | 21.79 |
| generalization | loss2_epoch500 | 4.648449 | 4.985648 | 0.897317 | 13.35 | 0.844978 | 25.08 |
| generalization | loss3_epoch500 | 4.653508 | 4.985648 | 0.900103 | 11.67 | 0.854701 | 22.04 |

## Top-10 Model Subspace Similarity To Solver

| split | model | right mean cosine | right min cosine | left mean cosine | left min cosine |
|---|---|---:|---:|---:|---:|
| ALL | baseline | 0.949119 | 0.640299 | 0.918823 | 0.565630 |
| ALL | loss1_epoch1000 | 0.985885 | 0.919078 | 0.986311 | 0.928836 |
| ALL | loss2_epoch500 | 0.983195 | 0.894443 | 0.980559 | 0.884843 |
| ALL | loss3_epoch500 | 0.988239 | 0.907934 | 0.979986 | 0.883492 |
| generalization | baseline | 0.937319 | 0.580337 | 0.877637 | 0.412110 |
| generalization | loss1_epoch1000 | 0.981113 | 0.901153 | 0.978017 | 0.900220 |
| generalization | loss2_epoch500 | 0.980669 | 0.897979 | 0.971827 | 0.860271 |
| generalization | loss3_epoch500 | 0.990655 | 0.937370 | 0.978535 | 0.894284 |

## Interpretation

The new aggressive/loss3-aligned generalization set does make `loss3_epoch500` look more competitive in the model Jacobian direction space. On generalization, `loss3_epoch500` has the best rank-1 right-vector similarity and the best top-10 right-subspace similarity to the solver.

However, the actual model-minus-solver error spectral norm still ranks:

```text
generalization: loss1_epoch1000 < loss3_epoch500 < loss2_epoch500 << baseline
ALL:            loss1_epoch1000 < loss2_epoch500 < loss3_epoch500 << baseline
```

So the current evidence is nuanced:

- The new round01 generalization points are more favorable to loss3 than the previous benchmark.
- `loss3_epoch500` aligns well with solver right-singular subspaces on generalization.
- But `loss1_epoch1000` still has the smallest model-minus-solver Jacobian error norm on train, test, generalization, and ALL.

## Generated Files

- `round01_jacobian_svd_summary.md`
- `round01_error_spectral_norm_aggregate.csv`
- `round01_jacobian_spectral_norm_aggregate.csv`
- `round01_solver_similarity_rankwise.csv`
- `round01_solver_similarity_subspaces.csv`
- `round01_selected_rank_singular_values_and_vector_angles.csv`
- `round01_topk_model_solver_subspace_similarity_summary.csv`
- `round01_top_singular_values_long.csv`
- `sample_000` through `sample_019`
