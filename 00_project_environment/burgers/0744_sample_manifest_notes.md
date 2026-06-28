# Representative Burgers Jacobian/SVD Sample Manifest

Seed: `20260531`

Purpose: compare the same fixed input points `x` across three model checkpoints:

- `baseline`: before adversarial training
- `adv_only`: after ADV-only adversarial training
- `clean_plus_adv`: after clean+ADV adversarial training

For every row/sample point:

```text
J_error_baseline(x)      = J_model_baseline(x) - J_solver(x)
J_error_adv_only(x)      = J_model_adv_only(x) - J_solver(x)
J_error_clean_plus_adv(x)= J_model_clean_plus_adv(x) - J_solver(x)
```

The labels `adv_only` and `clean_plus_adv` refer to model checkpoints, not to the input sample source.

Sample mix: 6 clean train, 4 clean test, 10 clean generalization points covering near/target/far and Gaussian/Matern/sawtooth/offset cases.

Manifest: `representative20_sample_manifest.csv`
