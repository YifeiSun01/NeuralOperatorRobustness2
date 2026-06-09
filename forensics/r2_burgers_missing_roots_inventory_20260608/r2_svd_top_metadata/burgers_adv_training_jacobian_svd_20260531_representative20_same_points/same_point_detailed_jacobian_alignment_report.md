# Same-Point Burgers Jacobian Detailed Interpretation

This note expands the same-point 20-sample analysis. The input point `x` is fixed first; then all Jacobians below are evaluated at exactly that same `x`.

## What Was Computed

For every fixed sample point:

```text
J_solver(x)                    = d solver(x) / d x
J_model_baseline(x)            = d baseline_model(x) / d x
J_model_adv_only(x)            = d ADV_only_trained_model(x) / d x
J_model_clean_plus_adv(x)      = d clean_plus_ADV_trained_model(x) / d x
J_error_baseline(x)            = J_model_baseline(x)       - J_solver(x)
J_error_adv_only(x)            = J_model_adv_only(x)       - J_solver(x)
J_error_clean_plus_adv(x)      = J_model_clean_plus_adv(x) - J_solver(x)
```

So yes: for one input sample there are seven Jacobian objects saved: one solver Jacobian, three model Jacobians, and three model-minus-solver error Jacobians.

## Plot Terms

- `singular values`: the SVD spectrum of one local Jacobian. Rank 1 is the largest local gain: the output change size produced by the worst unit input perturbation.
- `top right vectors`: the right singular vectors `v_k`. These live in input space. For Burgers, they are perturbation shapes in the initial condition `x`; `v_1` is the input direction that gives the largest local output change.
- `Fourier gain`: not an SVD quantity. It feeds normalized sine/cosine waves of frequency `k` into the Jacobian and records `||J sin_k||` and `||J cos_k||`. It is a frequency-response diagnostic: which input frequencies the local map amplifies.
- `baseline_error`, `adv_only_error`, `clean_plus_adv_error`: these are not separate models. They are `J_model - J_solver` for the corresponding model checkpoint.

## 1. Error Jacobian Spectral Norm

| split | n | baseline mean/std | ADV-only mean/std | clean+ADV mean/std | ADV-only lower | clean+ADV lower |
|---|---:|---:|---:|---:|---:|---:|
| train | 6 | 1.5080 +/- 1.4020 | 0.4898 +/- 0.1353 | 0.7549 +/- 0.1831 | 6/6 | 5/6 |
| test | 4 | 0.7690 +/- 0.1918 | 0.4383 +/- 0.0616 | 0.7058 +/- 0.5148 | 4/4 | 2/4 |
| generalization | 10 | 3.5431 +/- 1.7412 | 1.4922 +/- 0.7180 | 1.7005 +/- 0.8382 | 10/10 | 9/10 |
| ALL | 20 | 2.3778 +/- 1.8595 | 0.9807 +/- 0.7248 | 1.2179 +/- 0.7931 | 20/20 | 16/20 |

Reading: ADV-only reduces `||J_model - J_solver||_2` on all 20/20 same points. clean+ADV reduces it on 16/20 points.

## 2. Model Jacobian Spectral Norm Itself

| split | n | solver mean/std | baseline model mean/std | ADV-only model mean/std | clean+ADV model mean/std | ADV-only model lower than baseline | clean+ADV model lower than baseline |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 6 | 5.1784 +/- 1.0697 | 4.9188 +/- 0.9226 | 5.1531 +/- 1.0830 | 5.1484 +/- 1.0874 | 0/6 | 0/6 |
| test | 4 | 4.3837 +/- 0.9837 | 4.2242 +/- 0.9247 | 4.3215 +/- 1.0643 | 4.3606 +/- 1.0044 | 1/4 | 1/4 |
| generalization | 10 | 6.9464 +/- 1.8650 | 5.9363 +/- 1.3024 | 6.5188 +/- 1.5383 | 6.6519 +/- 1.6733 | 0/10 | 0/10 |
| ALL | 20 | 5.9035 +/- 1.8238 | 5.2886 +/- 1.2904 | 5.6696 +/- 1.5668 | 5.7426 +/- 1.6570 | 1/20 | 1/20 |

Important: the model Jacobian spectral norm did not generally shrink. Across all 20 points, ADV-only model norm is larger than baseline on most points. The main effect is not "the model became globally flatter"; the main effect is "the model Jacobian moved closer to the solver Jacobian".

## 3. Is the Model Jacobian Closer to the Solver?

| split | n | spectral-gap-to-solver improved ADV-only | spectral-gap-to-solver improved clean+ADV | top-right-vector alignment improved ADV-only | top-right-vector alignment improved clean+ADV | Fourier-gain curve improved ADV-only | Fourier-gain curve improved clean+ADV |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 6 | 6/6 | 6/6 | 6/6 | 5/6 | 6/6 | 6/6 |
| test | 4 | 3/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 |
| generalization | 10 | 10/10 | 10/10 | 9/10 | 9/10 | 10/10 | 10/10 |
| ALL | 20 | 19/20 | 20/20 | 19/20 | 18/20 | 20/20 | 20/20 |

## 4. Direction and Frequency Summary

| split | model | top-1 right-vector abs dot vs solver mean/std | top-8 right-subspace mean cos mean/std | Fourier gain relative RMSE vs solver mean/std |
|---|---|---:|---:|---:|
| train | baseline | 0.9952 +/- 0.0056 | 0.9640 +/- 0.0410 | 0.1306 +/- 0.0064 |
| train | adv_only | 0.9994 +/- 0.0002 | 0.9943 +/- 0.0080 | 0.0424 +/- 0.0037 |
| train | clean_plus_adv | 0.9984 +/- 0.0032 | 0.9967 +/- 0.0049 | 0.0283 +/- 0.0037 |
| test | baseline | 0.9955 +/- 0.0039 | 0.9621 +/- 0.0588 | 0.1259 +/- 0.0071 |
| test | adv_only | 0.9994 +/- 0.0002 | 0.9977 +/- 0.0008 | 0.0422 +/- 0.0030 |
| test | clean_plus_adv | 0.9996 +/- 0.0001 | 0.9988 +/- 0.0005 | 0.0264 +/- 0.0012 |
| generalization | baseline | 0.8140 +/- 0.3477 | 0.9531 +/- 0.0291 | 0.1853 +/- 0.0487 |
| generalization | adv_only | 0.9273 +/- 0.2224 | 0.9853 +/- 0.0196 | 0.0729 +/- 0.0239 |
| generalization | clean_plus_adv | 0.9875 +/- 0.0373 | 0.9818 +/- 0.0179 | 0.0519 +/- 0.0146 |
| ALL | baseline | 0.9047 +/- 0.2568 | 0.9582 +/- 0.0376 | 0.1570 +/- 0.0446 |
| ALL | adv_only | 0.9633 +/- 0.1575 | 0.9905 +/- 0.0151 | 0.0576 +/- 0.0228 |
| ALL | clean_plus_adv | 0.9932 +/- 0.0264 | 0.9897 +/- 0.0150 | 0.0397 +/- 0.0162 |

Interpretation: top-1 singular-vector alignment is a fragile diagnostic because the leading singular values can be close and the sign of singular vectors is arbitrary, so the table uses absolute dot products. The top-8 subspace and Fourier gain distance are more stable ways to ask whether the local linear behavior is solver-like.

## 5. Per-Sample Details

| id | split | dataset | local index | solver ||J||2 | baseline model ||J||2 | ADV model ||J||2 | clean+ADV model ||J||2 | baseline error ||J||2 | ADV error ||J||2 | clean+ADV error ||J||2 | ADV error ratio | clean+ADV error ratio |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | train | train_original_gaussian_corr0p03 | 816 | 6.3212 | 5.8410 | 6.3214 | 6.2992 | 1.7319 | 0.7002 | 0.7883 | 0.4043 | 0.4552 |
| 1 | train | train_original_gaussian_corr0p03 | 978 | 5.4668 | 5.2094 | 5.5075 | 5.4809 | 0.9681 | 0.2915 | 0.8296 | 0.3011 | 0.8569 |
| 2 | train | train_original_gaussian_corr0p03 | 1284 | 6.2762 | 5.8627 | 6.2528 | 6.2526 | 4.2472 | 0.5157 | 1.0267 | 0.1214 | 0.2417 |
| 3 | train | train_original_gaussian_corr0p03 | 377 | 4.7530 | 4.6467 | 4.6675 | 4.7478 | 0.8121 | 0.5140 | 0.7507 | 0.6329 | 0.9243 |
| 4 | train | train_original_gaussian_corr0p03 | 1062 | 4.7319 | 4.5147 | 4.6530 | 4.6495 | 0.5834 | 0.4098 | 0.6577 | 0.7025 | 1.1273 |
| 5 | train | train_original_gaussian_corr0p03 | 168 | 3.5211 | 3.4380 | 3.5165 | 3.4605 | 0.7052 | 0.5074 | 0.4763 | 0.7195 | 0.6754 |
| 6 | test | test_original_gaussian_corr0p03 | 41 | 3.6108 | 3.3521 | 3.4368 | 3.5590 | 0.8923 | 0.4722 | 0.3781 | 0.5292 | 0.4237 |
| 7 | test | test_original_gaussian_corr0p03 | 57 | 3.5690 | 3.6255 | 3.5121 | 3.5507 | 0.8030 | 0.4676 | 0.3447 | 0.5823 | 0.4293 |
| 8 | test | test_original_gaussian_corr0p03 | 14 | 5.6143 | 5.3866 | 5.6810 | 5.6298 | 0.8924 | 0.4674 | 1.4502 | 0.5238 | 1.6250 |
| 9 | test | test_original_gaussian_corr0p03 | 80 | 4.7406 | 4.5326 | 4.6559 | 4.7030 | 0.4884 | 0.3460 | 0.6504 | 0.7084 | 1.3316 |
| 10 | generalization | burgers_near_gaussian_corr0p08 | 156 | 5.9649 | 5.7315 | 5.9691 | 5.9364 | 0.7865 | 0.5500 | 1.0956 | 0.6993 | 1.3930 |
| 11 | generalization | burgers_near_gaussian_corr0p3 | 92 | 8.8564 | 7.0407 | 7.8497 | 8.3172 | 5.6322 | 2.8880 | 2.9739 | 0.5128 | 0.5280 |
| 12 | generalization | burgers_target_gaussian_corr0p105 | 186 | 7.9552 | 6.2724 | 7.4296 | 7.4282 | 3.3956 | 1.1392 | 1.9171 | 0.3355 | 0.5646 |
| 13 | generalization | burgers_target_gaussian_corr0p21 | 9 | 8.7193 | 7.1512 | 7.8832 | 8.3188 | 6.6107 | 1.8568 | 2.5110 | 0.2809 | 0.3798 |
| 14 | generalization | burgers_target_gaussian_corr0p55 | 17 | 8.8164 | 6.9890 | 7.8788 | 8.2757 | 4.4325 | 2.2366 | 2.7502 | 0.5046 | 0.6205 |
| 15 | generalization | burgers_mid_matern_corr0p75_nu2p5 | 62 | 7.7749 | 6.5035 | 7.2700 | 7.2760 | 3.1234 | 1.0622 | 1.6562 | 0.3401 | 0.5303 |
| 16 | generalization | burgers_target_matern_corr0p42_nu4 | 93 | 6.0376 | 5.4810 | 5.9849 | 5.9784 | 3.8945 | 0.9246 | 1.1677 | 0.2374 | 0.2998 |
| 17 | generalization | burgers_target_matern_corr0p8_nu2 | 101 | 6.8682 | 6.4295 | 6.5399 | 6.6073 | 1.4711 | 1.1662 | 1.2938 | 0.7927 | 0.8795 |
| 18 | generalization | burgers_far_sawtooth_add_scale0p3_shift0 | 79 | 5.4227 | 4.9064 | 5.4156 | 5.4025 | 3.0963 | 1.9804 | 1.3330 | 0.6396 | 0.4305 |
| 19 | generalization | burgers_far_centered_scale_shift_scale1_shift0p4 | 168 | 3.0484 | 2.8581 | 2.9670 | 2.9790 | 2.9884 | 1.1176 | 0.3063 | 0.3740 | 0.1025 |

## Bottom Line

The present evidence supports the stronger claim for `J_error`: adversarial training, especially ADV-only, makes the model-minus-solver Jacobian much smaller at these 20 fixed points. It does not support the simpler claim that the model Jacobian norm itself simply becomes much smaller. In fact, the model spectral norm often increases slightly, usually toward the solver spectral norm. The cleaner statement is: the trained model local derivative becomes more solver-like, and the residual derivative `J_model - J_solver` contracts.

## New Files

- `same_point_per_sample_jacobian_norm_alignment_detailed.csv`
- `same_point_compact_split_summary_with_std.csv`
- `same_point_detailed_metric_split_stats.csv`
- `same_point_detailed_jacobian_alignment_report.md`
