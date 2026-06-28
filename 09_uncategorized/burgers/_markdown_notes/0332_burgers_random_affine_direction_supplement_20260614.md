# Burgers Random Affine Direction Supplement, 20260614

This supplement rebuilds the old4-style affine/local-gain diagnostics for
`random_clean_y` and `random_solver_y`.

It uses existing checkpoint/data artifacts and stored Jacobian matrices. It does
not train, generate attacks, or generate Jacobians. The clean residual vector is
recomputed by model forward pass because the completed random suite stored
residual norms/MSE but not the full residual vector.

One fixed sample (`sample_id=4`, train local index 128) is outside the saved
52-dataset attack manifest's train-first-50 subset, so attack-delta cosine
columns are blank for that sample while local affine/SVD/outward gains remain
available.

## Summary

```json
{
  "device": "cuda",
  "rows": 50,
  "samples": 25,
  "models": [
    "random_clean_y",
    "random_solver_y"
  ],
  "attack_delta_missing_rows": 2,
  "attack_delta_available_rows": 48,
  "residual_source": "model forward from existing checkpoint",
  "jacobian_source": "stored random_solver7860_clean8000 Jacobian NPZ"
}
```

## Headline Means

| model | metric | n | mean | std | median | min | max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| random_clean_y | affine_local_gain_eps_mse | 25 | 0.279901 | 0.133179 | 0.258168 | 0.0876071 | 0.705975 |
| random_clean_y | attack_delta_affine_eps_abs_cos | 24 | 0.418702 | 0.247725 | 0.475993 | 0.00729298 | 0.846794 |
| random_clean_y | outward_local_gain_eps_mse | 25 | 0.20425 | 0.107477 | 0.158143 | 0.0675845 | 0.461702 |
| random_clean_y | svd_local_gain_eps_mse | 25 | 0.277981 | 0.133168 | 0.254633 | 0.0859071 | 0.701731 |
| random_solver_y | affine_local_gain_eps_mse | 25 | 0.0560134 | 0.072 | 0.0257276 | 9.78477e-05 | 0.296282 |
| random_solver_y | attack_delta_affine_eps_abs_cos | 24 | 0.157086 | 0.140664 | 0.109665 | 0.0100775 | 0.487687 |
| random_solver_y | outward_local_gain_eps_mse | 25 | 0.0148004 | 0.0271062 | 0.00448877 | 6.04629e-05 | 0.132744 |
| random_solver_y | svd_local_gain_eps_mse | 25 | 0.0559697 | 0.0719726 | 0.0257224 | 9.75311e-05 | 0.296184 |

## Output Files

- `data/random_affine_direction_supplement_20260614/random_affine_direction_metrics.csv`
- `data/random_affine_direction_supplement_20260614/random_affine_direction_model_summary.csv`
- `data/random_affine_direction_supplement_20260614/random_affine_direction_correlations.csv`
- `data/random_affine_direction_supplement_20260614/random_affine_direction_summary.json`
