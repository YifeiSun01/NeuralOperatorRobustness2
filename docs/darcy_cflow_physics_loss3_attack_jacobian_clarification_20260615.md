# Darcy cflow physics-vs-loss3 attack and Jacobian clarification

Date: 2026-06-15.

## Attack Metrics: What They Mean

Observed source:

- `outputs/darcy_cflow_timematched_organized_release_20260614/data/cflow_attack_metric_long_ranked.csv`
- `outputs/darcy_cflow_timematched_organized_release_20260614/data/source_tables/robustness_attack_52datasets_samples.csv`

Definitions used in the robustness table:

- `clean_loss`: the attack objective evaluated at the original clean input
  `x0`, before adding the adversarial perturbation.
- `adv_loss`: the same attack objective evaluated after the adversarially
  perturbed input `x_adv = x0 + delta`.
- `loss_increase`: `adv_loss - clean_loss`.
- `relative_increase`: `(adv_loss - clean_loss) / clean_loss`.
- `delta`: the actual attack perturbation added to the input coefficient field.

The available organized-release attack table is partial smoke coverage:
`2` samples per dataset and `attack_steps=1` in the source table. It should be
read as evidence from the current release, not as the requested full
50-sample-per-dataset attack sweep.

## Physics vs Loss3 on Generalization Attack Samples

On the 100 generalization attack samples in this release:

| metric | lower is better | physics mean | loss3 mean | physics better count | loss3 better count |
|---|---:|---:|---:|---:|---:|
| clean_loss | yes | 1.358e-07 | 1.890e-07 | 93/100 | 7/100 |
| adv_loss | yes | 1.722e-07 | 2.293e-07 | 91/100 | 9/100 |
| loss_increase | yes | 3.644e-08 | 4.031e-08 | 72/100 | 28/100 |
| relative_increase | yes | 0.3345 | 0.2268 | 28/100 | 72/100 |

Interpretation:

- No, physics is not better than loss3 on every dataset/sample.
- For absolute `adv_loss`, physics is lower on 91/100 generalization samples.
- For absolute `loss_increase`, physics is lower on 72/100 generalization
  samples.
- For `relative_increase`, loss3 is better: it is lower on 72/100 samples and
  has the lower mean.

So the precise statement is: in this partial smoke attack table, physics has
smaller absolute post-attack loss and absolute loss increase on average, while
loss3 has smaller relative loss increase.

This can happen because `relative_increase` divides by each model's own
`clean_loss`. In the generalization attack table, physics usually starts from a
smaller clean loss:

| metric mean | physics | loss3 | physics / loss3 |
|---|---:|---:|---:|
| clean_loss | 1.358e-07 | 1.890e-07 | 0.718 |
| adv_loss | 1.722e-07 | 2.293e-07 | 0.751 |
| loss_increase | 3.644e-08 | 4.031e-08 | 0.904 |
| relative_increase | 0.3345 | 0.2268 | 1.475 |

The absolute increase is only about 10% smaller for physics, but its clean-loss
denominator is about 28% smaller. That smaller denominator can make the
percentage increase larger. Direct paired counts confirm the overlap: in `45`
of the 100 generalization samples, physics has a smaller absolute
`loss_increase` while loss3 has a smaller `relative_increase`.

## Jacobian / SVD Quantities: What They Mean

Observed source:

- `outputs/darcy_cflow_timematched_organized_release_20260614/data/cflow_svd_jacobian_metric_long_ranked.csv`
- `outputs/darcy_cflow_timematched_organized_release_20260614/data/source_tables/svd_jacobian_metrics.csv`

The visible SVD/Jacobian table in this release is very small:
`partial_smoke_3samples_per_model`, with 2 train samples and 1 test sample.
It should be treated as a mechanism diagnostic, not as a complete
generalization conclusion.

Definitions:

- `error`: model prediction minus target, `model(x) - y`.
- `error_l2_norm`: L2 norm of that prediction error.
- `J`: local Jacobian of the model output with respect to the input coefficient
  field at the sample `x`.
- `sigma_input_right`: an estimate of the strongest input-direction singular
  value, i.e. how much the model output can change under the most amplified
  local input direction.
- `input_right_singular_vector`: the input-space direction that locally produces
  the largest model-output change.
- `jt_error`: `J^T error`, computed as the gradient of `0.5 * ||model(x)-y||^2`
  with respect to the input. This is the local steepest-error direction in input
  space.
- `jt_error_l2_norm`: magnitude of `J^T error`. Larger values mean the local
  clean error is more sensitive to input perturbation in the gradient direction.
- `attack_delta`: the actual adversarial perturbation produced by the attack.
- `cos_*`, `angle_*_deg`, and `corr_*`: similarity between two input-space
  vectors. Angles near `0` mean the vectors point similarly; angles near `90`
  mean near-orthogonal; angles above `90` mean opposite-ish. Pearson
  correlations are sign/shape correlations after centering.

The three vector comparisons are:

- `singular_jt_error`: top singular input vector vs `J^T error`.
- `singular_attack_delta`: top singular input vector vs actual attack `delta`.
- `jt_error_attack_delta`: `J^T error` vs actual attack `delta`.

## Physics vs Loss3 on Visible SVD/Jacobian Samples

Across the 3 visible train/test SVD samples:

| metric | lower/higher better | physics mean | loss3 mean | physics better count | loss3 better count |
|---|---|---:|---:|---:|---:|
| attack_loss_increase | lower | 8.569e-07 | 6.099e-07 | 0/3 | 3/3 |
| attack_relative_increase | lower | 3.258 | 1.512 | 1/3 | 2/3 |
| clean_loss | lower | 3.331e-07 | 3.142e-07 | 1/3 | 2/3 |
| error_l2_norm | lower | 0.04599 | 0.04571 | 1/3 | 2/3 |
| jt_error_l2_norm | lower | 3.679e-05 | 3.769e-05 | 2/3 | 1/3 |
| sigma_input_right | lower sensitivity | 0.001101 | 0.001176 | 3/3 | 0/3 |
| angle singular vs delta | lower angle | 93.13 deg | 90.31 deg | 1/3 | 2/3 |
| angle J^T error vs delta | lower angle | 93.82 deg | 90.08 deg | 1/3 | 2/3 |
| corr singular vs delta | higher | 0.0881 | 0.0324 | 3/3 | 0/3 |
| corr J^T error vs delta | higher | 0.1145 | 0.0442 | 3/3 | 0/3 |

Interpretation:

- On these 3 visible SVD/Jacobian samples, loss3 has smaller attack loss
  increase than physics on all 3.
- Physics has a smaller estimated strongest singular sensitivity
  (`sigma_input_right`) on all 3.
- Physics has higher Pearson correlations between the diagnostic directions and
  the attack delta, but the cosine angles are still near 90 degrees for both
  methods. That means these vectors are mostly not strongly aligned with the
  actual attack delta in this tiny sample.
- Because this is only 3 samples, these Jacobian/SVD numbers should not be used
  as a broad claim about all generalization datasets.

## Bottom Line

For the current release tables:

- Clean 52-dataset metrics: loss3 has the broadest and strongest advantage.
- Partial smoke attack metrics: physics is better than loss3 on absolute
  `adv_loss` and `loss_increase` averages, but not on every sample, and loss3 is
  better on `relative_increase`.
- Visible SVD/Jacobian diagnostics: loss3 has smaller attack loss increase on
  the 3 visible samples, while physics has smaller local singular sensitivity.
  The vector-angle diagnostics are mostly near-orthogonal to attack delta, so
  they should be treated as qualitative mechanism diagnostics rather than a
  simple win/loss table.
