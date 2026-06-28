# Burgers Adversarial Training Jacobian/SVD Comparison

This diagnostic compares three Burgers FNO checkpoints:

- `baseline`: original pre-adversarial-training checkpoint
- `adv_only`: adversarial-training final checkpoint
- `clean_plus_adv`: clean+adversarial-training final checkpoint

For each sampled input, it computes:

```text
J_model = d model(x) / d x
J_solver = d solver(x) / d x
J_error = J_model - J_solver
```

The main robustness-locality object is `J_error`, not `J_model` alone.

## Aggregate Spectral Norms

| jacobian | n | spectral norm mean | spectral norm std | fro norm mean | effective rank mean |
|---|---:|---:|---:|---:|---:|
| adv_only_error | 10 | 1.09426 | 0.899164 | 1.25752 | 4.08034 |
| adv_only_model | 10 | 5.61956 | 1.70338 | 7.17091 | 3.46685 |
| baseline_error | 10 | 2.34177 | 1.50608 | 2.85343 | 5.46841 |
| baseline_model | 10 | 5.13406 | 1.35469 | 6.56241 | 3.48326 |
| clean_plus_adv_error | 10 | 1.16228 | 0.79975 | 1.31841 | 4.50008 |
| clean_plus_adv_model | 10 | 5.6229 | 1.7152 | 7.1688 | 3.43365 |
| solver | 10 | 5.79905 | 1.89156 | 7.35455 | 3.43174 |

## Files

- `sample_manifest.csv`: fixed sampled train/test/generalization inputs
- `jacobian_svd_summary.csv`: per-sample spectral norms and top-20 singular values
- `top_singular_values_long.csv`: long-format top singular values
- `aggregate_jacobian_svd_summary.csv`: aggregate spectral-norm table
- `sample_*/`: per-sample SVD NPZ files for solver, model, and error Jacobians

Config seed: `20260531`
