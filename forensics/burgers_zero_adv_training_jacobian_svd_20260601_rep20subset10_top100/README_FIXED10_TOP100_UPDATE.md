# Burgers Fixed-10 Jacobian/SVD Top100 Update

## What Was Found

The previous complete same-point Jacobian/SVD run is:

`forensics/burgers_adv_training_jacobian_svd_20260531_representative20_same_points`

It already contains 20 fixed samples with seven Jacobian objects per sample:

- `solver`
- `baseline`
- `baseline_error = baseline - solver`
- `adv_only` from the older 20260530 ADV-only checkpoint
- `adv_only_error` from that older checkpoint
- `clean_plus_adv`
- `clean_plus_adv_error`

The code used for this diagnostic is:

`tools/compare_burgers_adversarial_jacobian_svd.py`

## Fixed 10 Samples Reused From The Old 20

Selected source sample IDs: `0, 1, 6, 7, 10, 11, 12, 15, 18, 19`.

That is `2 train + 2 test + 6 generalization`. The generalization points cover near Gaussian, shifted/target Gaussian, Matern, far sawtooth, and far offset/shift cases.

## What Was Reused vs Recomputed

Reused from the old full SVD:

- `J_solver`
- baseline `J_model`
- baseline `J_error`
- old clean+ADV `J_model`
- old clean+ADV `J_error`

Recomputed now:

- new `adv_only` `J_model` for `burgers_epoch1000_step003000.pt`
- new `adv_only_error = J_adv1000 - J_solver`

All stored SVDs now keep top 100 singular values and top 100 left/right singular vectors.

Important caveat: this new 1000-epoch checkpoint is from the run that used `fast_replace_linf`; interpret it as the current L-infinity/sign-replace run, not the intended future `p=2, q=2` run.

## Error Jacobian Spectral Norm Result

| split | n | baseline mean | adv1000 mean | clean+ADV mean | adv1000 smaller | clean+ADV smaller |
|---|---:|---:|---:|---:|---:|---:|
| ALL | 10 | 2.34177 | 1.09426 | 1.16228 | 10/10 | 9/10 |
| generalization | 6 | 3.1704 | 1.53598 | 1.54703 | 6/6 | 5/6 |
| test | 2 | 0.847638 | 0.334061 | 0.361379 | 2/2 | 2/2 |
| train | 2 | 1.35 | 0.529305 | 0.808953 | 2/2 | 2/2 |

For these 10 fixed reused points, the new 1000-epoch ADV-only error Jacobian is smaller than baseline on every point in this subset, with ALL mean ratio:

`adv1000 / baseline = 0.4852`

## Top-k Model-vs-Solver Similarity Files

The detailed direction and magnitude comparisons are saved as:

- `top100_singular_values_long_all7.csv`: top 100 singular values for all seven Jacobian objects
- `top100_rankwise_left_right_vector_absdot_vs_solver.csv`: rank-by-rank `abs(dot)` for left and right singular vectors against solver
- `topk_model_vs_solver_similarity.csv`: top-k singular-value and subspace similarity for k = 1, 5, 10, 20, 50, 100
- `topk_model_vs_solver_similarity_split_summary.csv`: split-level means/stds for the same similarity metrics
- `same_point_error_spectral_norm_comparison_top100.csv`: per-sample error Jacobian norm comparison
- `same_point_error_spectral_norm_split_summary_top100.csv`: split summary for error Jacobian norms

## Quick Top-20 Direction Summary Against Solver

| model | sv rel L2 lower is better | left subspace cosine higher is better | right subspace cosine higher is better |
|---|---:|---:|---:|
| baseline | 0.116681 | 0.872649 | 0.802887 |
| adv_only | 0.0378733 | 0.985867 | 0.967583 |
| clean_plus_adv | 0.0299595 | 0.982401 | 0.955581 |

This table is for model Jacobians versus solver Jacobians. Error Jacobian shrinkage is a separate question and is summarized above.
