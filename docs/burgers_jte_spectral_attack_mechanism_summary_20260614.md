# Burgers JTe, Spectral Norm, And Attack Mechanism Summary, 20260614

This note records the concrete answers from the Burgers mechanism audit:
whether `loss3` is best on attack loss, error-Jacobian spectral/Frobenius norms,
`J_error^T error`, and direction/correlation diagnostics.

## Source Tables

Observed from:

- `outputs/burgers_timematched_solver7860_clean8000_audit_20260614/data/ranked_metric_tables_20260614/attack_52dataset_metric_long_ranked.csv`
- `outputs/burgers_timematched_solver7860_clean8000_audit_20260614/data/per_dataset_loss3_significance_20260614/per_dataset_loss3_vs_best_other_significance.csv`
- `outputs/burgers_timematched_solver7860_clean8000_audit_20260614/data/ranked_metric_tables_20260614/robustness_25sample_metric_long_ranked.csv`
- `outputs/burgers_timematched_solver7860_clean8000_audit_20260614/data/ranked_metric_tables_20260614/metric_model_summary_ranked.csv`
- `outputs/burgers_timematched_solver7860_clean8000_audit_20260614/data/ranked_metric_tables_20260614/correlations_with_attack_sorted.csv`
- `outputs/burgers_timematched_solver7860_clean8000_audit_20260614/data/metric_coverage_inventory_20260614/direction_angle_compact.csv`
- `forensics/burgers_wideparam_loss3targeted_full1024_svd_attack25_biased_local_direction_20260611/direction_pairwise_angle_summary_20260612.csv`
- `forensics/burgers_random_solver7860_clean8000_full_suite_20260614/postprocess/svd_attack_bias_gradient_by_sample.csv`

No training, attack generation, Jacobian generation, or SVD computation was rerun
for this note.

## Full 52-Dataset Attack Outcome

For the strict latest full-52 attack table, lower is better.

| scope | metric | loss3 mean | runner-up | runner-up mean | dataset rows where loss3 is best | significant |
| --- | --- | ---: | --- | ---: | ---: | --- |
| all 52 datasets | attack loss increase | 0.00382075 | loss1 | 0.00658905 | 52/52 | yes |
| all 52 datasets | attack final loss | 0.00400196 | loss1 | 0.00715168 | 52/52 | yes |
| generalization 50 datasets | attack loss increase | 0.00394899 | loss1 | 0.00679898 | 50/50 | yes |
| generalization 50 datasets | attack final loss | 0.004137 | loss1 | 0.007384 | 50/50 | yes |

Per-dataset paired tests also support the same result: for attack loss increase
and attack final loss, `loss3` is rank 1 and significantly better on every one
of the 52 dataset rows.

## 25-Sample Local Robustness And Error-Operator Metrics

The 25-sample set contains train 2, test 2, and generalization 21 samples. Lower
is better for all metrics in this table.

| metric | scope | loss3 mean | runner-up | runner-up mean | q best vs runner-up | loss3 first by sample |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| attack loss increase | all 25 | 0.003063 | loss2 | 0.005623 | 3.79e-06 | 22/25 |
| attack final MSE | all 25 | 0.003307 | loss2 | 0.006479 | 1.01e-06 | 23/25 |
| error spectral norm, top singular value | all 25 | 1.271530 | random_solver_y | 1.552516 | 5.43e-02 | 15/25 |
| error Frobenius norm | all 25 | 1.898529 | random_solver_y | 2.512214 | 4.08e-03 | 16/25 |
| `||J_error^T error||` | all 25 | 0.111950 | loss1 | 0.315854 | 1.24e-04 | 16/25 |
| RMS(`J_error^T error`) | all 25 | 0.003498 | loss1 | 0.009870 | 1.24e-04 | 16/25 |
| attack loss increase | generalization 21 | 0.003520 | loss2 | 0.006387 | 8.03e-06 | 19/21 |
| attack final MSE | generalization 21 | 0.003806 | loss2 | 0.007405 | 1.28e-06 | 20/21 |
| error spectral norm, top singular value | generalization 21 | 1.352116 | random_solver_y | 1.827109 | 3.62e-03 | 15/21 |
| error Frobenius norm | generalization 21 | 2.077731 | random_solver_y | 2.944790 | 1.71e-04 | 16/21 |
| `||J_error^T error||` | generalization 21 | 0.131375 | loss1 | 0.375505 | 6.72e-05 | 16/21 |
| RMS(`J_error^T error`) | generalization 21 | 0.004105 | loss1 | 0.011735 | 6.72e-05 | 16/21 |

Important boundary:

- `loss3` is not first on every local 25-sample row.
- The full 52-dataset attack outcome is a 52/52 `loss3` sweep.
- The local Jacobian/SVD diagnostics are mean-best and usually significant for
  `loss3`, but their per-sample first-place counts are not 25/25.
- Top singular value / spectral norm is the weakest of the listed mechanism
  metrics: `loss3` is mean-best on all-25, but the best-vs-runner-up q-value is
  0.0543, which does not pass a strict 0.05 threshold. It is significant on the
  generalization-21 subset.

## Correlation Sample Counts

Correlation tables use model-sample points, not just raw samples.

| correlation scope | definition | n |
| --- | --- | ---: |
| all_six_150 | 25 samples times 6 models | 150 |
| generalization_all_six_126 | 21 generalization samples times 6 models | 126 |
| old_four_100 | 25 samples times baseline/loss1/loss2/loss3 | 100 |
| random_two_50 | 25 samples times random_clean_y/random_solver_y | 50 |

Partial fields such as `j_error_delta_l2` have smaller n because the field is
not available for every model/sample row: 48 for all/random and 42 for
generalization.

## JTe Versus Spectral Norm Correlation With Attack

This table tests whether the loss-gradient direction norm `||J_error^T error||`
tracks attack loss increase more strongly than the largest error singular value
`spectral_norm(J_error)`.

| scope | metric | n | Pearson with attack loss increase | Spearman with attack loss increase |
| --- | --- | ---: | ---: | ---: |
| all_six_150 | `||J_error^T error||` | 150 | 0.8997 | 0.8976 |
| all_six_150 | `spectral_norm(J_error)` | 150 | 0.6938 | 0.8312 |
| generalization_all_six_126 | `||J_error^T error||` | 126 | 0.9254 | 0.8758 |
| generalization_all_six_126 | `spectral_norm(J_error)` | 126 | 0.6377 | 0.7863 |
| random_two_50 | `||J_error^T error||` | 50 | 0.8844 | 0.9114 |
| random_two_50 | `spectral_norm(J_error)` | 50 | 0.7698 | 0.8259 |
| old_four_100 | `||J_error^T error||` | 100 | 0.7453 | 0.8507 |
| old_four_100 | `spectral_norm(J_error)` | 100 | 0.6372 | 0.7778 |

Observed conclusion: in all four scopes, `||J_error^T error||` has higher
Pearson and higher Spearman correlation with attack loss increase than
`spectral_norm(J_error)`.

## Direction Similarity And Angle

All six-model direction summaries use 25 samples per model. Cosines are absolute
cosines because singular vector signs are arbitrary.

| model | delta vs top-SV cos | delta vs top-SV angle | delta vs `J_error^T error` cos | delta vs `J_error^T error` angle | top-SV vs `J_error^T error` cos | top-SV vs `J_error^T error` angle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.3049 | 70.7 deg | 0.7935 | 36.2 deg | 0.3041 | 71.4 deg |
| loss1 | 0.2091 | 77.2 deg | 0.8031 | 35.9 deg | 0.2161 | 76.6 deg |
| loss2 | 0.1965 | 77.8 deg | 0.7987 | 35.9 deg | 0.2542 | 73.7 deg |
| loss3 | 0.0915 | 84.7 deg | 0.7265 | 42.7 deg | 0.1856 | 79.1 deg |
| random_clean_y | 0.3992 | 65.7 deg | 0.7098 | 43.1 deg | 0.5250 | 56.3 deg |
| random_solver_y | 0.2771 | 73.1 deg | 0.7304 | 42.3 deg | 0.2447 | 75.4 deg |

Old4 combined direction summary:

| group | delta vs top-SV mean angle | delta vs `J_error^T error` mean angle | top-SV vs `J_error^T error` mean angle |
| --- | ---: | ---: | ---: |
| baseline/loss1/loss2/loss3, 100 model-sample rows | 77.6 deg | 37.7 deg | 75.2 deg |

Observed conclusion: the attack delta is much closer to the local loss-gradient
direction `J_error^T error` than to the top error singular vector. This supports
the mechanism interpretation that the top singular direction alone is not the
best explanation of attack loss growth.

## Final Interpretation

Observed evidence supports these claims:

- `loss3` is the strongest model on strict latest full-52 attack loss increase
  and attack final loss: 52/52 dataset rows, significant per dataset.
- On local 25-sample Jacobian/SVD diagnostics, `loss3` is not first on every
  sample, but it is mean-best and significant for attack loss increase, attack
  final MSE, Frobenius norm, and `||J_error^T error||`.
- Top singular value / spectral norm is a weaker mechanism metric than
  `||J_error^T error||`: it is not a 25/25 sample sweep, and all-25 significance
  is borderline rather than strict.
- The correlation and direction-angle evidence both support the same mechanism:
  attack loss growth is better explained by the local gradient direction
  `J_error^T error` than by the largest error singular direction alone.
