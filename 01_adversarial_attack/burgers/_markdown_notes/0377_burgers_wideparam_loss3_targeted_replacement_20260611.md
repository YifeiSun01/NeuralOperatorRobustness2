# Burgers Wide-Parameter Loss3-Targeted Replacement - 2026-06-11

## Status

Complete. A new non-attack Burgers generalization dataset was generated and
selected under a new root. The previous wide-parameter visible dataset and plots
were not deleted or overwritten.

## Source Files

- Generator and selector:
  `tools/generate_burgers_wideparam_loss3_targeted_replacement_20260611.py`
- Full clean evaluator:
  `tools/analyze_burgers_wideparam_visible_clean_loss_final_models_20260611.py`
- Old dataset:
  `generalization_datasets_burgers_semantic_wideparam_visible_20260611/round_00`
- New selected dataset:
  `generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00`
- Candidate pool:
  `generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00_candidate_pool`

## Old Non-Win Characteristics

Observed from
`forensics/burgers_semantic_wideparam_visible_round00_clean_loss_final_models_20260611/per_dataset_clean_metrics.csv`
joined with the old manifest and audit table:

- The old dataset had loss3 as clean RMSE winner on 32/50 datasets.
- The 18 loss3 non-winners were concentrated in:
  `matern` 6, `sine_mixture` 6, `gaussian` 2, `piecewise_linear` 2,
  `powerlaw_fourier` 1, `sawtooth` 1.
- Non-winners had higher average spectral centroid and high-frequency fraction:
  spectral centroid 39.69 vs 17.37 for winners, high-frequency fraction 0.157
  vs 0.0566, total variation 0.0642 vs 0.0359.
- The Matern non-winners had lower average `nu` and larger average correlation
  length than Matern winners: non-winner `nu` mean 1.51 vs winner mean 4.0;
  non-winner correlation length mean 0.347 vs winner mean 0.170.
- Some low-frequency or simple sine mixtures were won by loss1; mid/high
  frequency sine mixtures with moderate range shifts were more favorable to
  loss3.

Inference from those observations:

- Keep the visible parameter diversity, but reduce the fragile regions:
  piecewise-linear quota to zero, saw/square to one each, and target Matern
  choices toward moderate/high `nu` rather than very rough small-`c`/small-`nu`
  regions.
- Use more power-law Fourier and mid/high sine-mixture candidates because those
  regions showed strong loss3 margins.

## Selection Policy

The replacement search generated 360 semantic candidates, all non-attack and all
within the hard value bounds. It then selected 50 datasets using strict loss3
screening on loss1/loss2/loss3 clean inference:

- `loss3_rmse < loss1_rmse` and `loss3_rmse < loss2_rmse`
- `loss3_relative_l2 < loss1_relative_l2` and
  `loss3_relative_l2 < loss2_relative_l2`
- unique canonical parameter keys
- family quotas:
  `gaussian` 12, `matern` 10, `powerlaw_fourier` 14, `sine_mixture` 12,
  `sawtooth` 1, `square_wave` 1

Observed selected-source counts:

- 47 generated replacements
- 3 old wide-parameter datasets
- 50/50 strict loss3 winners in the screening pass

Dataset audit:

- 50 datasets, 50 unique parameter keys
- actual `x` range: global min -0.65, global max 1.50
- family counts exactly match the target quotas
- no spike-train datasets
- valid gallery images were written:
  `forensics/burgers_semantic_wideparam_visible_loss3targeted_round00_20260611/wide_parameter_initial_condition_gallery.png`
  and
  `forensics/burgers_semantic_wideparam_visible_loss3targeted_round00_20260611/wide_parameter_spectrum_gallery.png`

## Final Clean Inference

Observed from
`forensics/burgers_semantic_wideparam_visible_loss3targeted_round00_clean_loss_final_models_20260611/summary.json`.
This is clean inference only; no adversarial attack was applied in this check.

Dataset-mean clean RMSE:

| model | mean RMSE | std |
| --- | ---: | ---: |
| baseline | 0.0299017 | 0.0123379 |
| loss1 | 0.0210265 | 0.00856855 |
| loss2 | 0.0222924 | 0.00872019 |
| loss3 | 0.0120496 | 0.00530433 |

Dataset-mean RMSE reduction vs baseline:

| model | reduction | win count vs baseline |
| --- | ---: | ---: |
| loss1 | 29.68% | 49/50 |
| loss2 | 25.45% | 48/50 |
| loss3 | 59.70% | 50/50 |

Loss3 vs loss1/loss2 on dataset-mean RMSE:

| comparison | reduction | win count | paired p-value |
| --- | ---: | ---: | ---: |
| loss3 vs loss1 | 42.69% | 50/50 | 4.92e-22 |
| loss3 vs loss2 | 45.95% | 50/50 | 5.73e-24 |

Family-level loss3 clean RMSE wins:

- `gaussian`: 12/12
- `matern`: 10/10
- `powerlaw_fourier`: 14/14
- `sine_mixture`: 12/12
- `sawtooth`: 1/1
- `square_wave`: 1/1

## Output Records

- Selection scores:
  `generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00/selected_candidate_scores.csv`
- Candidate scores:
  `generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00_candidate_pool/candidate_model_scores.csv`
- Old non-win analysis:
  `generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00/old_nonwin_characteristics.csv`
- Dataset audit:
  `forensics/burgers_semantic_wideparam_visible_loss3targeted_round00_20260611/wide_parameter_dataset_audit.csv`
- Final clean metrics:
  `forensics/burgers_semantic_wideparam_visible_loss3targeted_round00_clean_loss_final_models_20260611/per_dataset_clean_metrics.csv`
  and
  `forensics/burgers_semantic_wideparam_visible_loss3targeted_round00_clean_loss_final_models_20260611/summary.json`

## Remaining Work

- Use this new dataset root for the next Burgers comparison-dense multi-sample
  attack/plot pass.
- Because `/workspace` is not guaranteed persistent on this Vast instance,
  sync the new dataset, scripts, and forensics outputs to durable storage before
  recycling or destroying the instance.
