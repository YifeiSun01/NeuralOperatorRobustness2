# Darcy CFlow Singular Vector, Attack Delta, And J^T Error Angle Summary - 2026-06-15

Observed from:

- Source table: `outputs/darcy_cflow_final_robustness_20260615/data/svd_jacobian_metrics.csv`
- Generated summary table:
  `outputs/darcy_cflow_final_robustness_20260615/data/final_metric_mean_std_20260615/svd_vector_cosine_angle_summary_20260615.csv`

This uses the final seven-model Darcy CFlow robustness/SVD run on
`generalization_datasets_darcy_binary_loss3targeted_20260611`, with 25 fixed
SVD/Jacobian samples per model. It does not use the older
`lossdrop50_selected_20260607` data.

## Notes On Angle Convention

For right singular vectors, the sign is arbitrary: `v` and `-v` represent the
same singular direction. Therefore:

- `signed_cos_mean` and `signed_angle_deg_mean` are reported for traceability.
- `abs_cos_mean` and `acute_angle_deg_mean` are the more stable sign-invariant
  interpretation for pairs involving a singular vector.
- For `J^T error` vs attack `delta`, the sign is not arbitrary, so signed angle
  still has direct directional meaning.

## Overall Means

| scope | pair | n | signed cos mean | abs cos mean | signed angle mean | acute angle mean | corr mean |
|---|---|---:|---:|---:|---:|---:|---:|
| all 7 models | singular vector vs J^T error | 175 | 0.546203 | 0.877174 | 48.5674 | 24.7984 | 0.521904 |
| all 7 models | singular vector vs attack delta | 175 | 0.0274889 | 0.127962 | 88.3778 | 82.5348 | 0.152320 |
| all 7 models | J^T error vs attack delta | 175 | -0.0295667 | 0.108833 | 91.7419 | 83.6980 | 0.192911 |
| all 7 models | top-k subspace vs J^T error | 175 | 0.973333 | 0.973333 | 12.6770 | 12.6770 | n/a |
| all 7 models | top-k subspace vs attack delta | 175 | 0.324149 | 0.324149 | 70.9043 | 70.9043 | n/a |
| generalization only | singular vector vs J^T error | 147 | 0.613726 | 0.912920 | 43.4275 | 21.3827 | 0.585791 |
| generalization only | singular vector vs attack delta | 147 | 0.0103850 | 0.0886419 | 89.3945 | 84.8911 | 0.155427 |
| generalization only | J^T error vs attack delta | 147 | -0.0116168 | 0.0872512 | 90.6812 | 84.9743 | 0.235012 |
| generalization only | top-k subspace vs J^T error | 147 | 0.977640 | 0.977640 | 11.8117 | 11.8117 | n/a |
| generalization only | top-k subspace vs attack delta | 147 | 0.299977 | 0.299977 | 72.4135 | 72.4135 | n/a |

## Per-Model Vector Pair Means

| model | pair | n | signed cos mean | abs cos mean | signed angle mean | acute angle mean | corr mean |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline | singular vector vs J^T error | 25 | 0.537798 | 0.879358 | 49.1469 | 25.4173 | 0.527885 |
| baseline | singular vector vs attack delta | 25 | -0.0490608 | 0.172738 | 92.7663 | 79.9564 | 0.0203680 |
| baseline | J^T error vs attack delta | 25 | -0.111108 | 0.122527 | 96.4064 | 82.9389 | 0.0264108 |
| loss1 | singular vector vs J^T error | 25 | 0.580041 | 0.925152 | 45.3803 | 20.9394 | 0.560538 |
| loss1 | singular vector vs attack delta | 25 | 0.0684748 | 0.118159 | 85.9021 | 83.0530 | 0.235118 |
| loss1 | J^T error vs attack delta | 25 | -0.0637762 | 0.111056 | 93.7638 | 83.5243 | 0.198419 |
| loss2 | singular vector vs J^T error | 25 | 0.644131 | 0.924588 | 41.2175 | 20.2021 | 0.607797 |
| loss2 | singular vector vs attack delta | 25 | -0.0368961 | 0.103498 | 92.1112 | 83.9715 | 0.138622 |
| loss2 | J^T error vs attack delta | 25 | -0.0625831 | 0.0910704 | 93.6368 | 84.7303 | 0.211348 |
| loss3 | singular vector vs J^T error | 25 | 0.412331 | 0.769640 | 59.7763 | 35.8440 | 0.352755 |
| loss3 | singular vector vs attack delta | 25 | 0.165079 | 0.191555 | 80.4053 | 78.8813 | 0.196427 |
| loss3 | J^T error vs attack delta | 25 | 0.0681231 | 0.178908 | 86.1309 | 79.6301 | 0.186082 |
| Physics Loss | singular vector vs J^T error | 25 | 0.561830 | 0.875385 | 47.4383 | 23.8929 | 0.542949 |
| Physics Loss | singular vector vs attack delta | 25 | 0.0279062 | 0.105883 | 88.4403 | 83.8003 | 0.174802 |
| Physics Loss | J^T error vs attack delta | 25 | 0.0118650 | 0.0862114 | 89.3321 | 85.0225 | 0.260637 |
| random clean | singular vector vs J^T error | 25 | 0.453378 | 0.881005 | 55.4717 | 24.6521 | 0.457475 |
| random clean | singular vector vs attack delta | 25 | -0.0193458 | 0.110557 | 91.1724 | 83.5292 | 0.0898295 |
| random clean | J^T error vs attack delta | 25 | -0.0634839 | 0.105138 | 93.7278 | 83.8835 | 0.184993 |
| random solver | singular vector vs J^T error | 25 | 0.633914 | 0.885087 | 41.5412 | 22.6407 | 0.603928 |
| random solver | singular vector vs attack delta | 25 | 0.0362643 | 0.0933457 | 87.8470 | 84.5518 | 0.211072 |
| random solver | J^T error vs attack delta | 25 | 0.0139958 | 0.0669205 | 89.1958 | 86.1564 | 0.282484 |

## Interpretation

Observed:

- The top singular direction is strongly aligned with `J^T error` when sign is
  ignored: all-model `abs_cos_mean = 0.877174`, acute angle mean `24.7984`
  degrees; generalization-only acute angle mean `21.3827` degrees.
- The actual attack delta is nearly orthogonal to the top singular vector:
  all-model acute angle mean `82.5348` degrees; generalization-only acute angle
  mean `84.8911` degrees.
- The actual attack delta is also nearly orthogonal to `J^T error` in signed
  direction: all-model signed angle mean `91.7419` degrees; generalization-only
  signed angle mean `90.6812` degrees.
- The top-10 lifted singular subspace contains most of the `J^T error`
  direction: all-model subspace angle mean `12.6770` degrees.
- The top-10 lifted singular subspace is still far from the attack delta:
  all-model subspace angle mean `70.9043` degrees.

Inference:

- The local SVD/Jacobian diagnostics and the discrete binary attack delta are
  not pointing in the same one-dimensional direction.
- `J^T error` mostly lies inside the top-K sensitive subspace, but the final
  binary attack delta is constrained by the discrete coefficient replacement
  attack and is much less aligned with either the top singular vector or the
  `J^T error` vector.

