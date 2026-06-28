# Burgers Same-Point Jacobian/SVD Full Similarity Summary

This report summarizes the representative 20 same-input-point Burgers Jacobian/SVD comparison. The key rule is: each row compares different Jacobian objects at the same fixed input `x`.

## 0. Sample Scope

- 20 fixed input points total.
- 6 train points from the original Burgers train distribution.
- 4 test points from the original Burgers test distribution.
- 10 generalization points from generated out-of-distribution or shifted distributions.

| sample_id | split | dataset_id | local_index |
|---:|---|---|---:|
| 0 | train | train_original_gaussian_corr0p03 | 816 |
| 1 | train | train_original_gaussian_corr0p03 | 978 |
| 2 | train | train_original_gaussian_corr0p03 | 1284 |
| 3 | train | train_original_gaussian_corr0p03 | 377 |
| 4 | train | train_original_gaussian_corr0p03 | 1062 |
| 5 | train | train_original_gaussian_corr0p03 | 168 |
| 6 | test | test_original_gaussian_corr0p03 | 41 |
| 7 | test | test_original_gaussian_corr0p03 | 57 |
| 8 | test | test_original_gaussian_corr0p03 | 14 |
| 9 | test | test_original_gaussian_corr0p03 | 80 |
| 10 | generalization | burgers_near_gaussian_corr0p08 | 156 |
| 11 | generalization | burgers_near_gaussian_corr0p3 | 92 |
| 12 | generalization | burgers_target_gaussian_corr0p105 | 186 |
| 13 | generalization | burgers_target_gaussian_corr0p21 | 9 |
| 14 | generalization | burgers_target_gaussian_corr0p55 | 17 |
| 15 | generalization | burgers_mid_matern_corr0p75_nu2p5 | 62 |
| 16 | generalization | burgers_target_matern_corr0p42_nu4 | 93 |
| 17 | generalization | burgers_target_matern_corr0p8_nu2 | 101 |
| 18 | generalization | burgers_far_sawtooth_add_scale0p3_shift0 | 79 |
| 19 | generalization | burgers_far_centered_scale_shift_scale1_shift0p4 | 168 |

## 1. The Seven Jacobian Objects

For each fixed input point `x`, the run saved seven local Jacobian/SVD objects:

```text
1. solver
   J_solver(x)

2. baseline
   J_baseline_model(x)

3. baseline_error
   J_baseline_model(x) - J_solver(x)

4. adv_only
   J_adv_only_model(x)

5. adv_only_error
   J_adv_only_model(x) - J_solver(x)

6. clean_plus_adv
   J_clean_plus_adv_model(x)

7. clean_plus_adv_error
   J_clean_plus_adv_model(x) - J_solver(x)
```

Names without `_error` are model or solver Jacobians themselves. Names with `_error` are model-minus-solver Jacobians. `solver` is the PDE solver and has no before/after training version.

## 2. Diagnostic Meanings

- `singular values`: local amplification factors of a Jacobian. The first singular value is the spectral norm.
- `right singular vectors`: input-space perturbation directions. For Burgers these are perturbation shapes in the initial condition. The top right vector is the worst local input direction for that Jacobian.
- `left singular vectors`: output-space response directions. They describe the shape in the predicted/solver final state where the perturbation appears.
- `top-k subspace similarity`: compares the subspace spanned by the first `k` singular vectors. This is more stable than comparing only `v_1` when top singular values are close.
- `Fourier gain`: fixed-frequency response. For each Fourier mode `k`, it measures `||J sin_k||` and `||J cos_k||`. It asks whether a model amplifies low/mid/high frequency input waves like the solver does.

## 3. Main Result: Error Jacobian Contracts

| split | n | baseline error ||J||2 | ADV-only error ||J||2 | clean+ADV error ||J||2 | ADV-only lower | clean+ADV lower |
|---|---:|---:|---:|---:|---:|---:|
| train | 6 | 1.5080 +/- 1.4020 | 0.4898 +/- 0.1353 | 0.7549 +/- 0.1831 | 6/6 | 5/6 |
| test | 4 | 0.7690 +/- 0.1918 | 0.4383 +/- 0.0616 | 0.7058 +/- 0.5148 | 4/4 | 2/4 |
| generalization | 10 | 3.5431 +/- 1.7412 | 1.4922 +/- 0.7180 | 1.7005 +/- 0.8382 | 10/10 | 9/10 |
| ALL | 20 | 2.3778 +/- 1.8595 | 0.9807 +/- 0.7248 | 1.2179 +/- 0.7931 | 20/20 | 16/20 |

Interpretation: `ADV-only` reduces the spectral norm of `J_model - J_solver` on all 20/20 fixed points. `clean+ADV` reduces it on 16/20 fixed points. This is the strongest evidence that adversarial training makes the local residual derivative smaller.

## 4. Model Jacobian Norm Does Not Simply Shrink

| split | n | solver ||J||2 | baseline model ||J||2 | ADV-only model ||J||2 | clean+ADV model ||J||2 |
|---|---:|---:|---:|---:|---:|
| train | 6 | 5.1784 +/- 1.0697 | 4.9188 +/- 0.9226 | 5.1531 +/- 1.0830 | 5.1484 +/- 1.0874 |
| test | 4 | 4.3837 +/- 0.9837 | 4.2242 +/- 0.9247 | 4.3215 +/- 1.0643 | 4.3606 +/- 1.0044 |
| generalization | 10 | 6.9464 +/- 1.8650 | 5.9363 +/- 1.3024 | 6.5188 +/- 1.5383 | 6.6519 +/- 1.6733 |
| ALL | 20 | 5.9035 +/- 1.8238 | 5.2886 +/- 1.2904 | 5.6696 +/- 1.5668 | 5.7426 +/- 1.6570 |

Interpretation: the model Jacobian itself is not simply flattened. The baseline model tends to have a smaller spectral norm than the solver; after adversarial training the model norm usually moves closer to the solver norm.

## 5. Spectral-Norm Closeness to Solver

| split | n | ADV-only spectral gap to solver improved | clean+ADV spectral gap to solver improved |
|---|---:|---:|---:|
| train | 6 | 6/6 | 6/6 |
| test | 4 | 3/4 | 4/4 |
| generalization | 10 | 10/10 | 10/10 |
| ALL | 20 | 19/20 | 20/20 |

Across all 20 points, ADV-only improves the model-solver spectral-norm gap on 19/20 points, and clean+ADV improves it on 20/20 points.

## 6. Top-k Singular Value Similarity

Metric: relative RMSE between the first `k` singular values of `J_model` and `J_solver`. Smaller is more solver-like.

| split | top-k | baseline | ADV-only | clean+ADV |
|---|---:|---:|---:|---:|
| train | 1 | 0.0468 +/- 0.0217 | 0.0079 +/- 0.0078 | 0.0076 +/- 0.0076 |
| train | 5 | 0.0624 +/- 0.0090 | 0.0165 +/- 0.0049 | 0.0106 +/- 0.0045 |
| train | 10 | 0.0626 +/- 0.0088 | 0.0171 +/- 0.0047 | 0.0110 +/- 0.0042 |
| train | 20 | 0.0688 +/- 0.0085 | 0.0197 +/- 0.0043 | 0.0136 +/- 0.0031 |
| test | 1 | 0.0430 +/- 0.0228 | 0.0235 +/- 0.0167 | 0.0075 +/- 0.0050 |
| test | 5 | 0.0491 +/- 0.0077 | 0.0214 +/- 0.0102 | 0.0104 +/- 0.0036 |
| test | 10 | 0.0493 +/- 0.0078 | 0.0228 +/- 0.0112 | 0.0113 +/- 0.0046 |
| test | 20 | 0.0579 +/- 0.0077 | 0.0254 +/- 0.0098 | 0.0140 +/- 0.0032 |
| generalization | 1 | 0.1320 +/- 0.0680 | 0.0532 +/- 0.0432 | 0.0378 +/- 0.0256 |
| generalization | 5 | 0.1446 +/- 0.0522 | 0.0590 +/- 0.0364 | 0.0388 +/- 0.0230 |
| generalization | 10 | 0.1432 +/- 0.0516 | 0.0589 +/- 0.0355 | 0.0386 +/- 0.0226 |
| generalization | 20 | 0.1469 +/- 0.0492 | 0.0604 +/- 0.0338 | 0.0397 +/- 0.0213 |
| ALL | 1 | 0.0886 +/- 0.0662 | 0.0337 +/- 0.0371 | 0.0227 +/- 0.0239 |
| ALL | 5 | 0.1008 +/- 0.0579 | 0.0387 +/- 0.0330 | 0.0247 +/- 0.0217 |
| ALL | 10 | 0.1003 +/- 0.0571 | 0.0391 +/- 0.0322 | 0.0249 +/- 0.0212 |
| ALL | 20 | 0.1057 +/- 0.0546 | 0.0412 +/- 0.0309 | 0.0267 +/- 0.0199 |

Interpretation: top-k singular values generally become closer to solver after adversarial training. This means the local amplification magnitudes are more solver-like, not only the top spectral norm.

## 7. Right Singular Vector Similarity

Right singular vectors are input perturbation directions. Higher cosine is more similar to the solver input-sensitive directions.

| split | top-k right subspace mean cos | baseline | ADV-only | clean+ADV |
|---|---:|---:|---:|---:|
| train | 1 | 0.9952 +/- 0.0056 | 0.9994 +/- 0.0002 | 0.9984 +/- 0.0032 |
| train | 4 | 0.9954 +/- 0.0014 | 0.9989 +/- 0.0007 | 0.9995 +/- 0.0002 |
| train | 8 | 0.9640 +/- 0.0410 | 0.9943 +/- 0.0080 | 0.9967 +/- 0.0049 |
| train | 20 | 0.7627 +/- 0.0923 | 0.9260 +/- 0.0322 | 0.9472 +/- 0.0252 |
| test | 1 | 0.9955 +/- 0.0039 | 0.9994 +/- 0.0002 | 0.9996 +/- 0.0001 |
| test | 4 | 0.9722 +/- 0.0480 | 0.9428 +/- 0.1125 | 0.9978 +/- 0.0036 |
| test | 8 | 0.9621 +/- 0.0588 | 0.9977 +/- 0.0008 | 0.9988 +/- 0.0005 |
| test | 20 | 0.7759 +/- 0.0651 | 0.9243 +/- 0.0384 | 0.9480 +/- 0.0308 |
| generalization | 1 | 0.8140 +/- 0.3477 | 0.9273 +/- 0.2224 | 0.9875 +/- 0.0373 |
| generalization | 4 | 0.9193 +/- 0.0744 | 0.9545 +/- 0.0843 | 0.9765 +/- 0.0336 |
| generalization | 8 | 0.9531 +/- 0.0291 | 0.9853 +/- 0.0196 | 0.9818 +/- 0.0179 |
| generalization | 20 | 0.8270 +/- 0.0693 | 0.9439 +/- 0.0447 | 0.9582 +/- 0.0359 |
| ALL | 1 | 0.9047 +/- 0.2568 | 0.9633 +/- 0.1575 | 0.9932 +/- 0.0264 |
| ALL | 4 | 0.9527 +/- 0.0650 | 0.9655 +/- 0.0767 | 0.9877 +/- 0.0259 |
| ALL | 8 | 0.9582 +/- 0.0376 | 0.9905 +/- 0.0151 | 0.9897 +/- 0.0150 |
| ALL | 20 | 0.7975 +/- 0.0782 | 0.9346 +/- 0.0393 | 0.9529 +/- 0.0309 |

Across all samples: top-1 right-vector similarity improves from baseline `0.9047 +/- 0.2568` to ADV-only `0.9633 +/- 0.1575` and clean+ADV `0.9932 +/- 0.0264`. Top-8 right-subspace mean cosine improves from `0.9582 +/- 0.0376` to about `0.990` for both trained models.

## 8. Left Singular Vector Similarity

Left singular vectors are output response directions. Higher cosine means the output shape excited by the local sensitive directions is more solver-like.

| split | top-k left subspace mean cos | baseline | ADV-only | clean+ADV |
|---|---:|---:|---:|---:|
| train | 1 | 0.9490 +/- 0.0943 | 0.9968 +/- 0.0022 | 0.9895 +/- 0.0049 |
| train | 4 | 0.9729 +/- 0.0249 | 0.9952 +/- 0.0023 | 0.9956 +/- 0.0012 |
| train | 8 | 0.9576 +/- 0.0411 | 0.9932 +/- 0.0066 | 0.9950 +/- 0.0054 |
| train | 20 | 0.8237 +/- 0.0892 | 0.9738 +/- 0.0193 | 0.9765 +/- 0.0221 |
| test | 1 | 0.9921 +/- 0.0065 | 0.9959 +/- 0.0031 | 0.9870 +/- 0.0140 |
| test | 4 | 0.9619 +/- 0.0463 | 0.9388 +/- 0.1113 | 0.9933 +/- 0.0059 |
| test | 8 | 0.9556 +/- 0.0547 | 0.9960 +/- 0.0007 | 0.9967 +/- 0.0009 |
| test | 20 | 0.8518 +/- 0.0614 | 0.9670 +/- 0.0210 | 0.9753 +/- 0.0144 |
| generalization | 1 | 0.7223 +/- 0.3137 | 0.9076 +/- 0.2169 | 0.9618 +/- 0.0323 |
| generalization | 4 | 0.8798 +/- 0.0724 | 0.9456 +/- 0.0844 | 0.9665 +/- 0.0365 |
| generalization | 8 | 0.9386 +/- 0.0355 | 0.9819 +/- 0.0194 | 0.9766 +/- 0.0182 |
| generalization | 20 | 0.8989 +/- 0.0570 | 0.9737 +/- 0.0252 | 0.9821 +/- 0.0147 |
| ALL | 1 | 0.8443 +/- 0.2547 | 0.9520 +/- 0.1561 | 0.9752 +/- 0.0268 |
| ALL | 4 | 0.9241 +/- 0.0712 | 0.9591 +/- 0.0770 | 0.9806 +/- 0.0291 |
| ALL | 8 | 0.9477 +/- 0.0400 | 0.9881 +/- 0.0152 | 0.9861 +/- 0.0161 |
| ALL | 20 | 0.8669 +/- 0.0735 | 0.9724 +/- 0.0218 | 0.9790 +/- 0.0165 |

Interpretation: the output response directions also become more solver-like. This matters because matching only input directions would not be enough; the model also needs to move output perturbations in the same solution-space directions as the solver.

## 9. Fourier Gain / Frequency Response

Fourier gain is not SVD. It measures the response to fixed sine/cosine input waves. Lower relative RMSE against solver means the whole frequency-response curve is more solver-like.

| split | band | baseline rel RMSE | ADV-only rel RMSE | clean+ADV rel RMSE |
|---|---|---:|---:|---:|
| train | k001_008_low | 0.0543 +/- 0.0040 | 0.0120 +/- 0.0057 | 0.0083 +/- 0.0026 |
| train | k009_032_low_mid | 0.4965 +/- 0.0272 | 0.1991 +/- 0.0160 | 0.1275 +/- 0.0135 |
| train | k033_128_mid_high | 2.3941 +/- 0.4720 | 0.8709 +/- 0.0346 | 0.7053 +/- 0.0978 |
| train | k129_512_high | 12.3421 +/- 3.8585 | 1.3566 +/- 0.3749 | 1.2630 +/- 0.3063 |
| test | k001_008_low | 0.0336 +/- 0.0076 | 0.0130 +/- 0.0024 | 0.0071 +/- 0.0026 |
| test | k009_032_low_mid | 0.5040 +/- 0.0236 | 0.1910 +/- 0.0142 | 0.1156 +/- 0.0161 |
| test | k033_128_mid_high | 1.9062 +/- 0.3520 | 0.8526 +/- 0.0353 | 0.6415 +/- 0.0876 |
| test | k129_512_high | 10.0798 +/- 0.9353 | 1.1294 +/- 0.0731 | 1.0224 +/- 0.1425 |
| generalization | k001_008_low | 0.1458 +/- 0.0672 | 0.0546 +/- 0.0312 | 0.0352 +/- 0.0194 |
| generalization | k009_032_low_mid | 0.4928 +/- 0.0152 | 0.2431 +/- 0.0329 | 0.1896 +/- 0.0347 |
| generalization | k033_128_mid_high | 9.9908 +/- 5.5272 | 2.0688 +/- 1.2323 | 2.0560 +/- 1.0794 |
| generalization | k129_512_high | 11.3037 +/- 5.0233 | 1.8844 +/- 0.6731 | 2.1452 +/- 0.8643 |
| ALL | k001_008_low | 0.0959 +/- 0.0695 | 0.0335 +/- 0.0306 | 0.0215 +/- 0.0195 |
| ALL | k009_032_low_mid | 0.4962 +/- 0.0203 | 0.2195 +/- 0.0348 | 0.1562 +/- 0.0431 |
| ALL | k033_128_mid_high | 6.0948 +/- 5.5278 | 1.4662 +/- 1.0498 | 1.3679 +/- 1.0269 |
| ALL | k129_512_high | 11.3704 +/- 4.0816 | 1.5750 +/- 0.5998 | 1.6560 +/- 0.8006 |

Interpretation: Fourier gain improves on all 20/20 points for both trained models in the full-curve metric. The trained models reproduce the solver frequency response much better, especially compared with baseline on generalization samples.

## 10. Error Jacobian Selected Singular Values

This table shows selected ranks of the `J_error = J_model - J_solver` singular spectrum. Smaller is better.

| split | rank | baseline error SV | ADV-only error SV | clean+ADV error SV |
|---|---:|---:|---:|---:|
| train | 1 | 1.5080 +/- 1.4020 | 0.4898 +/- 0.1353 | 0.7549 +/- 0.1831 |
| train | 2 | 0.6709 +/- 0.2810 | 0.3124 +/- 0.0455 | 0.2671 +/- 0.0989 |
| train | 3 | 0.5787 +/- 0.2788 | 0.2266 +/- 0.0256 | 0.1577 +/- 0.0325 |
| train | 5 | 0.2703 +/- 0.0398 | 0.1426 +/- 0.0176 | 0.1073 +/- 0.0135 |
| train | 10 | 0.1626 +/- 0.0151 | 0.0887 +/- 0.0058 | 0.0728 +/- 0.0067 |
| train | 20 | 0.0790 +/- 0.0084 | 0.0367 +/- 0.0052 | 0.0286 +/- 0.0037 |
| test | 1 | 0.7690 +/- 0.1918 | 0.4383 +/- 0.0616 | 0.7058 +/- 0.5148 |
| test | 2 | 0.5797 +/- 0.1188 | 0.3292 +/- 0.0721 | 0.3014 +/- 0.0241 |
| test | 3 | 0.4052 +/- 0.1084 | 0.2459 +/- 0.0609 | 0.1779 +/- 0.0360 |
| test | 5 | 0.2829 +/- 0.0059 | 0.1493 +/- 0.0099 | 0.1158 +/- 0.0136 |
| test | 10 | 0.1726 +/- 0.0013 | 0.0927 +/- 0.0039 | 0.0711 +/- 0.0047 |
| test | 20 | 0.0711 +/- 0.0006 | 0.0400 +/- 0.0056 | 0.0295 +/- 0.0031 |
| generalization | 1 | 3.5431 +/- 1.7412 | 1.4922 +/- 0.7180 | 1.7005 +/- 0.8382 |
| generalization | 2 | 1.4258 +/- 0.6407 | 0.4683 +/- 0.1636 | 0.3431 +/- 0.1805 |
| generalization | 3 | 0.5630 +/- 0.2412 | 0.2187 +/- 0.0655 | 0.1967 +/- 0.0518 |
| generalization | 5 | 0.3385 +/- 0.0937 | 0.1651 +/- 0.0433 | 0.1309 +/- 0.0122 |
| generalization | 10 | 0.1914 +/- 0.0272 | 0.1039 +/- 0.0158 | 0.0851 +/- 0.0057 |
| generalization | 20 | 0.0774 +/- 0.0102 | 0.0452 +/- 0.0066 | 0.0379 +/- 0.0050 |
| ALL | 1 | 2.3778 +/- 1.8595 | 0.9807 +/- 0.7248 | 1.2179 +/- 0.7931 |
| ALL | 2 | 1.0301 +/- 0.6191 | 0.3937 +/- 0.1412 | 0.3119 +/- 0.1388 |
| ALL | 3 | 0.5361 +/- 0.2333 | 0.2265 +/- 0.0539 | 0.1813 +/- 0.0453 |
| ALL | 5 | 0.3069 +/- 0.0752 | 0.1552 +/- 0.0330 | 0.1208 +/- 0.0162 |
| ALL | 10 | 0.1790 +/- 0.0242 | 0.0971 +/- 0.0134 | 0.0786 +/- 0.0087 |
| ALL | 20 | 0.0766 +/- 0.0088 | 0.0416 +/- 0.0069 | 0.0334 +/- 0.0061 |

Interpretation: the reduction is not only rank-1. The leading part of the residual-Jacobian spectrum contracts after adversarial training, with ADV-only being the most consistent in top spectral norm.

## 11. Per-Sample Main Table

| id | split | dataset | solver ||J||2 | baseline model | ADV model | clean+ADV model | baseline error | ADV error | clean+ADV error | ADV ratio | clean+ADV ratio |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | train | train_original_gaussian_corr0p03 | 6.3212 | 5.8410 | 6.3214 | 6.2992 | 1.7319 | 0.7002 | 0.7883 | 0.4043 | 0.4552 |
| 1 | train | train_original_gaussian_corr0p03 | 5.4668 | 5.2094 | 5.5075 | 5.4809 | 0.9681 | 0.2915 | 0.8296 | 0.3011 | 0.8569 |
| 2 | train | train_original_gaussian_corr0p03 | 6.2762 | 5.8627 | 6.2528 | 6.2526 | 4.2472 | 0.5157 | 1.0267 | 0.1214 | 0.2417 |
| 3 | train | train_original_gaussian_corr0p03 | 4.7530 | 4.6467 | 4.6675 | 4.7478 | 0.8121 | 0.5140 | 0.7507 | 0.6329 | 0.9243 |
| 4 | train | train_original_gaussian_corr0p03 | 4.7319 | 4.5147 | 4.6530 | 4.6495 | 0.5834 | 0.4098 | 0.6577 | 0.7025 | 1.1273 |
| 5 | train | train_original_gaussian_corr0p03 | 3.5211 | 3.4380 | 3.5165 | 3.4605 | 0.7052 | 0.5074 | 0.4763 | 0.7195 | 0.6754 |
| 6 | test | test_original_gaussian_corr0p03 | 3.6108 | 3.3521 | 3.4368 | 3.5590 | 0.8923 | 0.4722 | 0.3781 | 0.5292 | 0.4237 |
| 7 | test | test_original_gaussian_corr0p03 | 3.5690 | 3.6255 | 3.5121 | 3.5507 | 0.8030 | 0.4676 | 0.3447 | 0.5823 | 0.4293 |
| 8 | test | test_original_gaussian_corr0p03 | 5.6143 | 5.3866 | 5.6810 | 5.6298 | 0.8924 | 0.4674 | 1.4502 | 0.5238 | 1.6250 |
| 9 | test | test_original_gaussian_corr0p03 | 4.7406 | 4.5326 | 4.6559 | 4.7030 | 0.4884 | 0.3460 | 0.6504 | 0.7084 | 1.3316 |
| 10 | generalization | burgers_near_gaussian_corr0p08 | 5.9649 | 5.7315 | 5.9691 | 5.9364 | 0.7865 | 0.5500 | 1.0956 | 0.6993 | 1.3930 |
| 11 | generalization | burgers_near_gaussian_corr0p3 | 8.8564 | 7.0407 | 7.8497 | 8.3172 | 5.6322 | 2.8880 | 2.9739 | 0.5128 | 0.5280 |
| 12 | generalization | burgers_target_gaussian_corr0p105 | 7.9552 | 6.2724 | 7.4296 | 7.4282 | 3.3956 | 1.1392 | 1.9171 | 0.3355 | 0.5646 |
| 13 | generalization | burgers_target_gaussian_corr0p21 | 8.7193 | 7.1512 | 7.8832 | 8.3188 | 6.6107 | 1.8568 | 2.5110 | 0.2809 | 0.3798 |
| 14 | generalization | burgers_target_gaussian_corr0p55 | 8.8164 | 6.9890 | 7.8788 | 8.2757 | 4.4325 | 2.2366 | 2.7502 | 0.5046 | 0.6205 |
| 15 | generalization | burgers_mid_matern_corr0p75_nu2p5 | 7.7749 | 6.5035 | 7.2700 | 7.2760 | 3.1234 | 1.0622 | 1.6562 | 0.3401 | 0.5303 |
| 16 | generalization | burgers_target_matern_corr0p42_nu4 | 6.0376 | 5.4810 | 5.9849 | 5.9784 | 3.8945 | 0.9246 | 1.1677 | 0.2374 | 0.2998 |
| 17 | generalization | burgers_target_matern_corr0p8_nu2 | 6.8682 | 6.4295 | 6.5399 | 6.6073 | 1.4711 | 1.1662 | 1.2938 | 0.7927 | 0.8795 |
| 18 | generalization | burgers_far_sawtooth_add_scale0p3_shift0 | 5.4227 | 4.9064 | 5.4156 | 5.4025 | 3.0963 | 1.9804 | 1.3330 | 0.6396 | 0.4305 |
| 19 | generalization | burgers_far_centered_scale_shift_scale1_shift0p4 | 3.0484 | 2.8581 | 2.9670 | 2.9790 | 2.9884 | 1.1176 | 0.3063 | 0.3740 | 0.1025 |

## 12. Final Interpretation

The data support the following statement: adversarial training does not merely reduce the model Jacobian norm. Instead, it makes the local model derivative more solver-like. This appears in several independent diagnostics:

- `J_error = J_model - J_solver` spectral norm contracts strongly, especially for ADV-only.
- model spectral norms move closer to solver spectral norms rather than simply shrinking.
- top-k singular values become closer to solver singular values.
- right singular vector subspaces, i.e. sensitive input perturbation directions, become more solver-like.
- left singular vector subspaces, i.e. output response directions, also become more solver-like.
- Fourier gain curves become closer to solver frequency response across all 20 points.

Between the two trained models, ADV-only is more stable if the target metric is `||J_model - J_solver||_2`: it improves 20/20 points. clean+ADV has very strong direction and Fourier-gain similarity, but its error spectral norm is less stable: 16/20 points improve and 4/20 get worse relative to baseline.

In short: the strongest claim supported by this run is not that the trained model becomes flatter. The strongest claim is that the trained model becomes locally more solver-like, and therefore the residual Jacobian `J_model - J_solver` becomes much smaller.

## 13. Generated Data Files

- `same_point_topk_singular_value_similarity_vs_solver.csv`
- `same_point_topk_singular_value_similarity_split_summary.csv`
- `same_point_topk_left_right_vector_similarity_vs_solver.csv`
- `same_point_topk_left_right_vector_similarity_split_summary.csv`
- `same_point_fourier_gain_band_similarity_vs_solver.csv`
- `same_point_fourier_gain_band_similarity_split_summary.csv`
- `same_point_error_singular_values_selected_ranks.csv`
- `same_point_error_singular_values_selected_ranks_split_summary.csv`
