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
| adv_only_error | 20 | 0.980677 | 0.70645 | 1.20588 | 5.85438 |
| adv_only_model | 20 | 5.66961 | 1.52709 | 7.12989 | 3.30886 |
| baseline_error | 20 | 2.37776 | 1.81241 | 2.90702 | 9.41433 |
| baseline_model | 20 | 5.28864 | 1.2577 | 6.671 | 3.44172 |
| clean_plus_adv_error | 20 | 1.21787 | 0.772981 | 1.34641 | 3.6877 |
| clean_plus_adv_model | 20 | 5.74261 | 1.61501 | 7.22334 | 3.32557 |
| solver | 20 | 5.90345 | 1.77762 | 7.38804 | 3.31898 |

## Files

- `sample_manifest.csv`: fixed sampled train/test/generalization inputs
- `jacobian_svd_summary.csv`: per-sample spectral norms and top-20 singular values
- `top_singular_values_long.csv`: long-format top singular values
- `aggregate_jacobian_svd_summary.csv`: aggregate spectral-norm table
- `sample_*/`: per-sample SVD NPZ files for solver, model, and error Jacobians

Config seed: `20260531`
