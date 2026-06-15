# Darcy CFlow Ten-Pass Consistency Audit

Date: 2026-06-15.

## Correction

The previous contradiction came from reading smoke attack/SVD diagnostics as if
they were final time-matched robustness results. That was wrong.

Observed from
`outputs/darcy_cflow_timematched_organized_release_20260614/data/source_tables/robustness_attack_52datasets_samples.csv`:

- all attack rows have `attack_steps = 1`;
- all attack delta paths contain `smoke_initial`;
- all non-baseline attack checkpoints are `darcy_sir20_smoke_*_1ep`.

Observed from
`outputs/darcy_cflow_timematched_organized_release_20260614/data/source_tables/svd_jacobian_metrics.csv`:

- only `21` SVD/Jacobian rows exist, exactly `3` samples per model;
- rows cover train/test only, not the requested fixed 25-sample set.

Therefore attack `clean_loss`, `adv_loss`, `loss_increase`,
`relative_increase`, and SVD/Jacobian rankings in the current organized release
are smoke diagnostics. They must not be used to contradict final clean
generalization curves.

## Ten Checks

| check | result | evidence |
|---|---:|---|
| 1. Attack source steps | PASS | unique `attack_steps` is `[1]` |
| 2. Attack delta provenance | PASS | `728/728` delta paths contain `smoke_initial` |
| 3. Attack checkpoint provenance | PASS | `624/624` non-baseline rows use smoke checkpoints |
| 4. Attack ranked-table coverage label | PASS | all `4368` rows are `partial_smoke_2sample_per_dataset` |
| 5. SVD vector provenance | PASS | `21/21` vector paths contain `smoke_initial` |
| 6. SVD sample count | PASS | every model has exactly `3` rows |
| 7. SVD ranked-table coverage label | PASS | all `315` rows are `partial_smoke_3samples_per_model` |
| 8. Final clean Relative L2: loss3 vs physics | PASS | loss3 better on `50/50`; means `0.0561538980191` vs `0.0882931316905` |
| 9. Final clean RMSE/data MSE/accuracy: loss3 vs physics | PASS | loss3 better on `50/50` for each metric |
| 10. Final clean best-by-mean main metrics | PASS | loss3 is best by mean for Relative L2, RMSE, data MSE, and accuracy |

## Correct Reading

Final clean generalization:

- `loss3` beats `physics` on all `50/50` generalization datasets for Relative
  L2, RMSE, data MSE, and accuracy.
- `loss3` is the best mean model on the main final clean generalization metrics.

Smoke robustness diagnostics:

- `physics` appears smaller on smoke attack `clean_loss`, `adv_loss`, and
  absolute `loss_increase` in parts of the current smoke table.
- That evidence is not final-model robustness evidence because the source run is
  a 1-step smoke diagnostic from 1-epoch smoke checkpoints.

## Files Updated

- `docs/darcy_cflow_smoke_attack_metric_correction_20260615.md`
- `docs/darcy_cflow_best_model_by_metric_20260615.md`
- `docs/darcy_cflow_loss3_random_clean_solver_summary_20260615.md`
- `docs/darcy_cflow_physics_loss3_attack_jacobian_clarification_20260615.md`
- `docs/darcy_cflow_loss3_advantage_extra15_20260615.md`
- `docs/darcy_cflow_polished_report_expansion_20260615.md`
- organized-release Markdown reports under
  `outputs/darcy_cflow_timematched_organized_release_20260614/reports/`
- organized-release polished-report README under
  `outputs/darcy_cflow_timematched_organized_release_20260614/figures/polished_report/README.md`

## Remaining Work

A real final robustness comparison still requires recomputing attacks and
SVD/Jacobian diagnostics from the final time-matched checkpoints. The current
organized release does not contain that complete final robustness artifact.
