# Burgers Round03 Final Extension SVD Split Tables - Corrected - 2026-06-07

This note records the corrected compact tables for the final-extension models after the user review. The goal is to keep the train/test/generalization distinction explicit and avoid treating the baseline denominator as a trained-model row.

Source directory:

- `forensics/burgers_loss3_selective_round03_loss123_final_extension_jacobian_svd_rep20_top100_20260606`

Primary source files:

- `forensics/burgers_loss3_selective_round03_loss123_final_extension_jacobian_svd_rep20_top100_20260606/round03_loss123_final_extension_error_spectral_norm_aggregate.csv`
- `forensics/burgers_loss3_selective_round03_loss123_final_extension_jacobian_svd_rep20_top100_20260606/round03_loss123_final_extension_jacobian_spectral_norm_aggregate.csv`
- `forensics/burgers_loss3_selective_round03_loss123_final_extension_jacobian_svd_rep20_top100_20260606/round03_loss123_final_extension_solver_similarity_rankwise.csv`
- `forensics/burgers_loss3_selective_round03_loss123_final_extension_jacobian_svd_rep20_top100_20260606/round03_loss123_final_extension_solver_similarity_subspaces.csv`
- `forensics/burgers_loss3_selective_round03_loss123_final_extension_jacobian_svd_rep20_top100_20260606/round03_loss123_final_extension_jacobian_svd_summary.csv`
- `forensics/burgers_loss3_selective_round03_loss123_final_extension_jacobian_svd_rep20_top100_20260606/round03_loss123_final_extension_runtime.csv`
- `forensics/burgers_loss3_selective_round03_loss123_final_extension_jacobian_svd_rep20_top100_20260606/round03_loss123_final_extension_sample_manifest.csv`

Final generated/generalization metric source files:
- `adversarial_training_runs/burgers_loss3_selective_round03_loss1_continue3000to5000_20260606/burgers/eval_split_summary.csv`
- `adversarial_training_runs/burgers_loss3_selective_round03_loss2_continue1000to2000_20260606/burgers/eval_split_summary.csv`
- `adversarial_training_runs/burgers_loss3_selective_round03_loss3_continue1000to1500_20260606/burgers/eval_split_summary.csv`

Definitions and cautions:

- `baseline denom` is the original baseline model-minus-solver error spectral norm mean on the same split. It is the denominator for `trained/base`, not a separate trained result.
- The baseline row is intentionally omitted from the trained-model comparison tables because baseline-vs-baseline is tautological.
- `smaller/n` counts how many per-sample trained model-minus-solver error spectral norms are smaller than the corresponding baseline error spectral norm on the same sample.
- The SVD manifest uses `20` samples total: `5` train, `5` test, and `10` generated/generalization samples.
- `spectral norm` means the largest singular value. Rank-wise `sv ratio` is the mean of per-sample model/solver singular-value ratios, not the ratio of the displayed means.
- Rank-wise and top-k similarity tables below use `jacobian_kind=model`, i.e. model Jacobian singular vectors/subspaces compared with solver Jacobian singular vectors/subspaces. The separate `jacobian_kind=error` rows are not mixed into these solver-similarity summaries.
- Top-k subspace values are mean principal cosines to the solver subspace. Top50/top100 minimum cosines can be near zero because lower directions are numerically weak, so the mean cosines are the cleaner summary.

## Final Generated/Generalization Prediction Metrics

| model | epoch | RMSE | relative L2 | source |
| --- | ---: | ---: | ---: | ---: |
| loss1 5000 | 5000 | 0.032454 | 0.058183 | adversarial_training_runs/burgers_loss3_selective_round03_loss1_continue3000to5000_20260606/burgers/eval_split_summary.csv |
| loss2 2000 | 2000 | 0.035740 | 0.064081 | adversarial_training_runs/burgers_loss3_selective_round03_loss2_continue1000to2000_20260606/burgers/eval_split_summary.csv |
| loss3 1500 | 1500 | 0.021638 | 0.038774 | adversarial_training_runs/burgers_loss3_selective_round03_loss3_continue1000to1500_20260606/burgers/eval_split_summary.csv |

## Model-Minus-Solver Error Compression

### Train

| model | n | trained error mean | baseline denom | trained/base | drop vs baseline % | smaller/n |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| loss1 5000 | 5 | 0.261678 | 1.084617 | 0.284415 | 71.56 | 5/5 |
| loss2 2000 | 5 | 0.280222 | 1.084617 | 0.301074 | 69.89 | 5/5 |
| loss3 1500 | 5 | 0.922264 | 1.084617 | 1.008787 | -0.88 | 3/5 |

### Test

| model | n | trained error mean | baseline denom | trained/base | drop vs baseline % | smaller/n |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| loss1 5000 | 5 | 0.268445 | 1.237877 | 0.294030 | 70.60 | 5/5 |
| loss2 2000 | 5 | 0.299860 | 1.237877 | 0.317824 | 68.22 | 5/5 |
| loss3 1500 | 5 | 0.772003 | 1.237877 | 0.669956 | 33.00 | 5/5 |

### Generalization

| model | n | trained error mean | baseline denom | trained/base | drop vs baseline % | smaller/n |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| loss1 5000 | 10 | 2.753176 | 5.774618 | 0.494448 | 50.56 | 9/10 |
| loss2 2000 | 10 | 3.322257 | 5.774618 | 0.578548 | 42.15 | 9/10 |
| loss3 1500 | 10 | 2.499599 | 5.774618 | 0.411547 | 58.85 | 9/10 |

### All 20 Samples

| model | n | trained error mean | baseline denom | trained/base | drop vs baseline % | smaller/n |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| loss1 5000 | 20 | 1.509119 | 3.467933 | 0.391835 | 60.82 | 19/20 |
| loss2 2000 | 20 | 1.806149 | 3.467933 | 0.443998 | 55.60 | 19/20 |
| loss3 1500 | 20 | 1.673366 | 3.467933 | 0.625459 | 37.45 | 17/20 |

## Rank-Wise Singular Values And Singular-Vector Similarity

### Train

| model | rank | model sv mean | solver sv mean | sv ratio | right vec cos | left vec cos |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| loss1 5000 | 1 | 4.427619 | 4.433845 | 0.998410 | 0.999343 | 0.999666 |
| loss1 5000 | 5 | 1.161395 | 1.167877 | 0.994917 | 0.998165 | 0.999195 |
| loss1 5000 | 20 | 0.064878 | 0.107532 | 0.599258 | 0.267838 | 0.394326 |
| loss2 2000 | 1 | 4.432995 | 4.433845 | 1.000141 | 0.999284 | 0.998819 |
| loss2 2000 | 5 | 1.165993 | 1.167877 | 1.001154 | 0.996817 | 0.997583 |
| loss2 2000 | 20 | 0.058690 | 0.107532 | 0.529926 | 0.259840 | 0.431803 |
| loss3 1500 | 1 | 4.401293 | 4.433845 | 0.992770 | 0.999552 | 0.980093 |
| loss3 1500 | 5 | 1.171975 | 1.167877 | 1.003486 | 0.801653 | 0.801586 |
| loss3 1500 | 20 | 0.095144 | 0.107532 | 0.875063 | 0.808957 | 0.845350 |

### Test

| model | rank | model sv mean | solver sv mean | sv ratio | right vec cos | left vec cos |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| loss1 5000 | 1 | 4.500837 | 4.506583 | 0.998416 | 0.823568 | 0.824174 |
| loss1 5000 | 5 | 0.884414 | 0.899135 | 0.982889 | 0.997940 | 0.999463 |
| loss1 5000 | 20 | 0.052806 | 0.101510 | 0.521791 | 0.235783 | 0.395218 |
| loss2 2000 | 1 | 4.499924 | 4.506583 | 0.997775 | 0.801614 | 0.801649 |
| loss2 2000 | 5 | 0.884563 | 0.899135 | 0.985355 | 0.996496 | 0.998051 |
| loss2 2000 | 20 | 0.046986 | 0.101510 | 0.461226 | 0.202237 | 0.377145 |
| loss3 1500 | 1 | 4.465220 | 4.506583 | 0.989344 | 0.812299 | 0.803453 |
| loss3 1500 | 5 | 0.893187 | 0.899135 | 0.991407 | 0.999273 | 0.998074 |
| loss3 1500 | 20 | 0.089242 | 0.101510 | 0.876298 | 0.921768 | 0.952160 |

### Generalization

| model | rank | model sv mean | solver sv mean | sv ratio | right vec cos | left vec cos |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| loss1 5000 | 1 | 4.954742 | 5.457302 | 0.920052 | 0.887093 | 0.798991 |
| loss1 5000 | 5 | 1.073361 | 1.012471 | 1.088706 | 0.629963 | 0.631109 |
| loss1 5000 | 20 | 0.072715 | 0.072045 | 1.067973 | 0.075187 | 0.108301 |
| loss2 2000 | 1 | 4.860458 | 5.457302 | 0.905033 | 0.811990 | 0.684895 |
| loss2 2000 | 5 | 0.999420 | 1.012471 | 0.989607 | 0.620768 | 0.618543 |
| loss2 2000 | 20 | 0.065508 | 0.072045 | 0.939718 | 0.091225 | 0.090583 |
| loss3 1500 | 1 | 4.845658 | 5.457302 | 0.902356 | 0.757473 | 0.688929 |
| loss3 1500 | 5 | 1.087621 | 1.012471 | 1.097058 | 0.848092 | 0.841338 |
| loss3 1500 | 20 | 0.083689 | 0.072045 | 1.250181 | 0.287197 | 0.341686 |

## Solver Top-k Subspace Similarity

### Train

| model | top k | right mean cos | left mean cos | right min cos mean | left min cos mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| loss1 5000 | 5 | 0.998588 | 0.999762 | 0.996894 | 0.999525 |
| loss1 5000 | 10 | 0.994771 | 0.998993 | 0.976314 | 0.994051 |
| loss1 5000 | 20 | 0.861196 | 0.938001 | 0.213914 | 0.350255 |
| loss1 5000 | 50 | 0.431891 | 0.491005 | 0.000000 | 0.000000 |
| loss1 5000 | 100 | 0.295235 | 0.577535 | 0.000000 | 0.000000 |
| loss2 2000 | 5 | 0.998428 | 0.999401 | 0.996561 | 0.998569 |
| loss2 2000 | 10 | 0.994091 | 0.998488 | 0.972341 | 0.991140 |
| loss2 2000 | 20 | 0.861359 | 0.951877 | 0.213048 | 0.370823 |
| loss2 2000 | 50 | 0.443158 | 0.508369 | 0.000000 | 0.000000 |
| loss2 2000 | 100 | 0.305939 | 0.586679 | 0.000000 | 0.000000 |
| loss3 1500 | 5 | 0.999772 | 0.993890 | 0.999325 | 0.977765 |
| loss3 1500 | 10 | 0.998779 | 0.996220 | 0.993198 | 0.978620 |
| loss3 1500 | 20 | 0.988913 | 0.993492 | 0.879952 | 0.929610 |
| loss3 1500 | 50 | 0.619651 | 0.687186 | 0.000000 | 0.000003 |
| loss3 1500 | 100 | 0.448765 | 0.578994 | 0.000002 | 0.000003 |

### Test

| model | top k | right mean cos | left mean cos | right min cos mean | left min cos mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| loss1 5000 | 5 | 0.998501 | 0.999727 | 0.997003 | 0.999330 |
| loss1 5000 | 10 | 0.992738 | 0.996981 | 0.952772 | 0.973197 |
| loss1 5000 | 20 | 0.847930 | 0.939690 | 0.178028 | 0.370723 |
| loss1 5000 | 50 | 0.424010 | 0.486883 | 0.000000 | 0.000000 |
| loss1 5000 | 100 | 0.310888 | 0.584199 | 0.000000 | 0.000000 |
| loss2 2000 | 5 | 0.998126 | 0.998960 | 0.996025 | 0.997251 |
| loss2 2000 | 10 | 0.983336 | 0.987421 | 0.861687 | 0.881387 |
| loss2 2000 | 20 | 0.847490 | 0.949509 | 0.182687 | 0.400499 |
| loss2 2000 | 50 | 0.435965 | 0.505797 | 0.000000 | 0.000000 |
| loss2 2000 | 100 | 0.323266 | 0.590530 | 0.000000 | 0.000000 |
| loss3 1500 | 5 | 0.999699 | 0.995037 | 0.999122 | 0.984391 |
| loss3 1500 | 10 | 0.998709 | 0.996472 | 0.992335 | 0.984096 |
| loss3 1500 | 20 | 0.989350 | 0.994005 | 0.905725 | 0.951407 |
| loss3 1500 | 50 | 0.635956 | 0.708764 | 0.000001 | 0.000003 |
| loss3 1500 | 100 | 0.447616 | 0.602991 | 0.000000 | 0.000001 |

### Generalization

| model | top k | right mean cos | left mean cos | right min cos mean | left min cos mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| loss1 5000 | 5 | 0.933509 | 0.891366 | 0.681417 | 0.621953 |
| loss1 5000 | 10 | 0.923182 | 0.911801 | 0.470626 | 0.493536 |
| loss1 5000 | 20 | 0.770393 | 0.840468 | 0.011183 | 0.032078 |
| loss1 5000 | 50 | 0.402757 | 0.473060 | 0.000000 | 0.000000 |
| loss1 5000 | 100 | 0.303563 | 0.538255 | 0.000000 | 0.000000 |
| loss2 2000 | 5 | 0.947542 | 0.890202 | 0.752726 | 0.637310 |
| loss2 2000 | 10 | 0.927097 | 0.902027 | 0.491694 | 0.470800 |
| loss2 2000 | 20 | 0.776335 | 0.850286 | 0.010371 | 0.029071 |
| loss2 2000 | 50 | 0.409674 | 0.486186 | 0.000000 | 0.000000 |
| loss2 2000 | 100 | 0.315150 | 0.570673 | 0.000000 | 0.000000 |
| loss3 1500 | 5 | 0.954183 | 0.908376 | 0.779743 | 0.705635 |
| loss3 1500 | 10 | 0.940608 | 0.918709 | 0.688619 | 0.683906 |
| loss3 1500 | 20 | 0.909340 | 0.910852 | 0.206751 | 0.232327 |
| loss3 1500 | 50 | 0.590300 | 0.665892 | 0.000002 | 0.000010 |
| loss3 1500 | 100 | 0.425009 | 0.530209 | 0.000001 | 0.000001 |

## Interpretation

Observed from the final-extension prediction summaries: loss3 epoch1500 is best on generated/generalization prediction metrics, with RMSE `0.021638` and relative L2 `0.038774`; loss1 epoch5000 is second and loss2 epoch2000 is third.

Observed from `round03_loss123_final_extension_error_spectral_norm_aggregate.csv`:

- On generated/generalization model-minus-solver error spectral norm, loss3 epoch1500 has the strongest baseline-relative shrinkage: loss3 `58.85%`, loss1 `50.56%`, loss2 `42.15%`.
- On train and test model-minus-solver error spectral norm, loss1/loss2 are stronger than loss3. Train drop is loss1 `71.56%`, loss2 `69.89%`, loss3 `-0.88%`; test drop is loss1 `70.60%`, loss2 `68.22%`, loss3 `33.00%`.

Observed from `round03_loss123_final_extension_solver_similarity_rankwise.csv` and `round03_loss123_final_extension_solver_similarity_subspaces.csv`, filtering to `jacobian_kind=model`:

- Rank-1 singular-vector similarity does not show a loss3 advantage; loss1/loss2 are often as strong or stronger at rank 1.
- The clearest loss3 geometric advantage is in top-k solver subspace similarity, especially top20/top50 mean cosines. This appears on generalization and also remains visible on train/test top20/top50, even though train/test error compression favors loss1/loss2.
- Generalization top20 right/left mean cosines are loss1 `0.770393/0.840468`, loss2 `0.776335/0.850286`, and loss3 `0.909340/0.910852`, so loss3 has the strongest generated/generalization subspace alignment among the final models.

Inference from the observed evidence: the user suspicion is supported for error compression: loss3 has the cleanest final-extension SVD advantage on generated/generalization, while train/test error compression does not favor loss3. A separate subspace-alignment advantage for loss3 is visible at broader top-k subspaces across all splits, especially top20/top50.
