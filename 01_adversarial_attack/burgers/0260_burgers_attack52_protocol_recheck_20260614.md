# Burgers Attack-52 Protocol Recheck, 2026-06-14

This note records a protocol audit after the visual dense attack panels appeared
to contradict the ranked 52-dataset attack table.

## Question

The ranked appendix reported that `random_solver_y` had lower mean
`attack_loss_increase_mean` than `loss3` on the recovered 52-dataset attack
table. The dense comparison panels, however, visually show `loss3` with the
smallest attack-loss curves on most displayed samples.

## Observed Sources

- Ranked attack table:
  `outputs/burgers_timematched_solver7860_clean8000_audit_20260614/data/ranked_metric_tables_20260614/attack_52dataset_metric_long_ranked.csv`
- Recovered attack builder:
  `tools/recover_burgers_prior_metrics_into_solver7860_audit_20260614.py`
- Original old-four full-52 source:
  `forensics/burgers_first_master_full_p2q2_52datasets_4models_finalonly_20step_20260608/summary_by_model_dataset.csv`
- Current random-model full-52 source:
  `forensics/burgers_random_solver7860_clean8000_full_suite_20260614/p2q2_attack/summary_by_model_dataset.csv`
- Dense trace source:
  `forensics/burgers_wideparam_loss123_randomsolver7860_clean8000_round00_p2q2_six_model_visuals_20260614/group*/attack_loss_curves_all_six_models.csv`
- Dense summary metadata:
  `forensics/burgers_wideparam_loss123_randomsolver7860_clean8000_round00_p2q2_six_model_visuals_20260614/group00/six_model_summary.json`

## Findings

1. The recovered 52-dataset ranked attack table is not a single fresh six-model
   rerun under one identical latest-checkpoint protocol. It combines:
   - old-four `baseline/loss1/loss2/loss3` rows from the old full-52
     `20step` source;
   - current `random_clean_y/random_solver_y` rows from the solver7860/clean8000
     random full suite.

2. The old-four source maps `loss3` from `loss3_epoch1500`, while the latest
   dense comparison panels label and use `loss3 e1000`.

3. On the mixed recovered 52-dataset table, the mean attack increase is lower
   for `random_solver_y` than for the old `loss3_epoch1500` row:
   - all 52 datasets: `random_solver_y = 0.008167`,
     old `loss3_epoch1500 = 0.040146`;
   - 50 generalization datasets: `random_solver_y = 0.008381`,
     old `loss3_epoch1500 = 0.041718`.
   However, this should be described as a mixed historical/recovered-table
   result, not as a strict latest `loss3 e1000` full-52 attack result.

4. Even inside that mixed recovered table, `loss3` wins by dataset count but
   loses by mean because several Gaussian/Matern datasets are large outliers:
   - `attack_loss_increase_mean`, all 52: `loss3` lower than `random_solver_y`
     on 35/52 datasets; `random_solver_y` lower on 17/52 datasets.
   - `attack_loss_increase_mean`, generalization 50: `loss3` lower on 33/50
     datasets; `random_solver_y` lower on 17/50 datasets.

5. On the latest dense visual trace set, which uses `loss3 e1000`,
   `random_clean_y e8000`, and `random_solver_y e7860`, `loss3` is best on the
   displayed samples:
   - pooled dense visual samples, 36 row occurrences:
     `loss3` mean attack increase `0.003685`,
     `random_solver_y` mean attack increase `0.007188`;
   - dense winner count by attack increase:
     `loss3` wins 31/36, `baseline` 3/36, `loss1` 1/36, `loss2` 1/36,
     `random_solver_y` 0/36, `random_clean_y` 0/36.
   This is still a selected/dense visualization set, not a complete full-52
   latest-checkpoint attack suite.

6. The current local audit did not find a complete full-52, 50-sample-per-dataset
   attack table for the latest `loss3 e1000` checkpoint against
   `random_solver_y e7860`. The strict statement "random_solver_y is
   significantly better than latest loss3 e1000 on full 52 attack" is therefore
   not currently evidenced by a same-protocol full-52 artifact.

## Correct Interpretation

- It is valid to say that the mixed recovered historical 52-dataset table has a
  lower mean attack increase for `random_solver_y` than for old
  `loss3_epoch1500`, mainly because several old-loss3 dataset outliers dominate
  the mean.
- It is not valid to present that mixed table as proof that `random_solver_y`
  beats the latest `loss3 e1000` checkpoint on a strict full-52 attack protocol.
- For latest `loss3 e1000`, the direct available attack evidence points the
  other way on the dense visualization set and on the 25-sample robustness/SVD
  set: `loss3` has lower attack increase than `random_solver_y`.

## Recommended Report Wording

Use:

> Clean generalization and the latest dense/25-sample robustness evidence favor
> `loss3 e1000`. The recovered full-52 attack appendix is a mixed historical
> table: its old-four rows use an older `loss3_epoch1500` full-52 attack source,
> while the random rows use the current solver7860/clean8000 suite. That mixed
> table should be treated as historical coverage evidence, not as a strict
> same-protocol latest six-model attack comparison.

Avoid:

> `random_solver_y` is significantly more robust than latest `loss3 e1000` on
> the full 52-dataset attack suite.

