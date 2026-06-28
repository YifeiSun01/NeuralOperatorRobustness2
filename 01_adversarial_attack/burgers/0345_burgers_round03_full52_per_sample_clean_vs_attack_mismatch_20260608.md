# Burgers Round03 Full-52 Per-Sample Clean vs Attack Mismatch - 2026-06-08

## Scope

This note audits the full per-sample tagged/attacked Burgers round03 P2Q2 artifact, not the six-sample visualization subset.

Question: does loss3 often have worse clean/generalization prediction loss while showing smaller fixed-budget adversarial-tag/attack loss growth?

## Source Evidence

Observed source run:

- `forensics/burgers_round03_full_p2q2_52datasets_4models_finalonly_20step_20260607/`

Observed source files:

- `manifest.json`: 10200 rows, 52 datasets.
- `baseline/losses_and_delta_rms_by_sample.npz`
- `loss1_epoch8000/losses_and_delta_rms_by_sample.npz`
- `loss2_epoch2000/losses_and_delta_rms_by_sample.npz`
- `loss3_epoch1500/losses_and_delta_rms_by_sample.npz`

Each model NPZ contains `initial_loss`, `initial_diff_rms`, `final_loss`, `final_diff_rms`, and `final_delta_rms`, each with shape `(10200,)`.

Observed attack setting from the source run: Burgers P2Q2 RMS-L2, epsilon RMS `0.12`, alpha RMS `0.012`, `20` steps, zero-start attack. `initial_loss` is clean/prediction MSE; `final_loss` is final attacked MSE; attack growth is `final_loss - initial_loss`.

Derived outputs written on 2026-06-08:

- `forensics/burgers_round03_full52_per_sample_clean_vs_attack_mismatch_20260608/per_sample_clean_attack_wide.csv`
- `forensics/burgers_round03_full52_per_sample_clean_vs_attack_mismatch_20260608/per_sample_mismatch_summary_by_split.csv`
- `forensics/burgers_round03_full52_per_sample_clean_vs_attack_mismatch_20260608/direct_loss3_mismatch_cross_counts.csv`
- `forensics/burgers_round03_full52_per_sample_clean_vs_attack_mismatch_20260608/per_sample_clean_attack_correlations.csv`
- `forensics/burgers_round03_full52_per_sample_clean_vs_attack_mismatch_20260608/per_dataset_sample_level_mismatch_summary.csv`
- `forensics/burgers_round03_full52_per_sample_clean_vs_attack_mismatch_20260608/metadata.json`

## Direct Cross Counts

Observed from `direct_loss3_mismatch_cross_counts.csv`:

| scope | samples | loss3 clean winner | loss3 final attack winner | loss3 attack-increase winner | loss3 not clean winner and final attack winner | loss3 not clean winner and attack-increase winner |
|---|---:|---:|---:|---:|---:|---:|
| all 52 | 10200 | 35 / 10200 = 0.34% | 9090 / 10200 = 89.12% | 9166 / 10200 = 89.86% | 9063 / 10200 = 88.85% | 9139 / 10200 = 89.60% |
| generalization 50 | 10000 | 35 / 10000 = 0.35% | 8916 / 10000 = 89.16% | 8992 / 10000 = 89.92% | 8889 / 10000 = 88.89% | 8965 / 10000 = 89.65% |
| train | 50 | 0 / 50 = 0.00% | 45 / 50 = 90.00% | 45 / 50 = 90.00% | 45 / 50 = 90.00% | 45 / 50 = 90.00% |
| test | 150 | 0 / 150 = 0.00% | 129 / 150 = 86.00% | 129 / 150 = 86.00% | 129 / 150 = 86.00% | 129 / 150 = 86.00% |

Even stricter condition, observed from the same file:

| scope | loss3 clean worse than both loss1/loss2 and final better than both | loss3 clean worse than both loss1/loss2 and increase better than both |
|---|---:|---:|
| all 52 | 8991 / 10200 = 88.15% | 9067 / 10200 = 88.89% |
| generalization 50 | 8817 / 10000 = 88.17% | 8893 / 10000 = 88.93% |
| train | 45 / 50 = 90.00% | 45 / 50 = 90.00% |
| test | 129 / 150 = 86.00% | 129 / 150 = 86.00% |

## Winner Counts By Metric

Observed from `per_sample_mismatch_summary_by_split.csv`:

| scope | samples | clean wins loss1/loss2/loss3 | final attack wins loss1/loss2/loss3 | attack increase wins loss1/loss2/loss3 |
|---|---:|---|---|---|
| all 52 | 10200 | 9392 / 773 / 35 | 753 / 357 / 9090 | 707 / 327 / 9166 |
| generalization 50 | 10000 | 9195 / 770 / 35 | 729 / 355 / 8916 | 683 / 325 / 8992 |
| train | 50 | 48 / 2 / 0 | 5 / 0 / 45 | 5 / 0 / 45 |
| test | 150 | 149 / 1 / 0 | 19 / 2 / 129 | 19 / 2 / 129 |

## Mean Losses

Observed sample means:

| scope | model | clean MSE mean | final attack MSE mean | attack increase MSE mean |
|---|---|---:|---:|---:|
| all 52 | loss1_epoch8000 | 0.000003371238103 | 0.003606513523 | 0.003603142285 |
| all 52 | loss2_epoch2000 | 0.000008680059300 | 0.004632377415 | 0.004623697355 |
| all 52 | loss3_epoch1500 | 0.00004514041717 | 0.001694242117 | 0.001649101700 |
| generalization 50 | loss1_epoch8000 | 0.000003424036669 | 0.003650282849 | 0.003646858812 |
| generalization 50 | loss2_epoch2000 | 0.000008806193927 | 0.004692557371 | 0.004683751177 |
| generalization 50 | loss3_epoch1500 | 0.00004564248681 | 0.001710434531 | 0.001664792044 |

Median loss3 ratios on generalization samples:

- loss3 clean MSE / loss1 clean MSE: `16.3252`
- loss3 final attack MSE / loss1 final attack MSE: `0.558101`
- loss3 attack increase / loss1 attack increase: `0.541635`
- loss3 clean MSE / loss2 clean MSE: `5.96581`
- loss3 final attack MSE / loss2 final attack MSE: `0.436635`
- loss3 attack increase / loss2 attack increase: `0.424699`

## Correlation

Observed from `per_sample_clean_attack_correlations.csv`:

- Across all trained model/sample rows on the 50 generalization datasets: Pearson clean vs final attacked loss is `-0.268017`; Pearson clean vs attack increase is `-0.276073`.
- Within each fixed model on the 50 generalization datasets, clean loss and attacked loss are positively related: loss1 Pearson clean vs final `0.190063`, loss2 `0.295740`, loss3 `0.333517`.

Interpretation of this correlation split:

- Within one model, harder clean samples often remain harder after attack.
- Across the trained methods, the model ranking reverses: loss3 is usually worse clean but usually better under the fixed-budget attack.

## Conclusion

Observed evidence supports the user's concern. On the existing full-52 per-sample tagged/attacked Burgers round03 P2Q2 artifact, loss3 generally has worse clean/generalization prediction loss, but substantially smaller final adversarial-tag/attack loss and smaller attack growth.

This is not a six-sample artifact. On the 10000 generalization samples, `88.89%` of samples have `loss3` not being the clean winner while being the final-attack winner, and `89.65%` have `loss3` not being the clean winner while being the attack-increase winner.

Caveat: this is the local full-52 `20`-step P2Q2 artifact. The six-sample overlay used a separate `100`-step visualization run. If a full `100`-step all-sample artifact exists only on R2 or another machine, it should be audited separately.

## Sample Count Clarification

Observed from `manifest.json` on 2026-06-08:

- Total samples in this local full-tag artifact: `10200`, not `12000`.
- Split counts: `train=50`, `test=150`, `generalization=10000`.
- Dataset count: `52`.
- Generalization dataset count: `50`, with `200` samples each, so `50 * 200 = 10000`.
- The train split in this attack/tag artifact is deliberately `train_original_gaussian_corr0p03_first50`, so it contributes only `50` samples, not the full training set.
- The test split contributes `150` samples.

Therefore the full artifact total is:

`50 train + 150 test + 50 generalization datasets * 200 samples = 10200 samples`.

Winner-count rows are mutually exclusive over the corresponding scope. For the `generalization` scope, examples are:

- Clean winners: `9195 + 770 + 35 = 10000`.
- Final attack winners: `729 + 355 + 8916 = 10000`.
- Attack-increase winners: `683 + 325 + 8992 = 10000`.

For the `all_10200` scope, the corresponding winner rows sum to `10200`.

## Generalization Source Provenance Clarification

Observed on 2026-06-08: the `10000` generalization samples in this full-tag artifact all point to `generalization_datasets_rmse_1p5_3x_all_ns50/burgers`, also recorded locally as `target_band_ns50_current`. A full metadata audit of the 50 `.pt` files found no attack-like fields such as `attack_geometry`, `source_split`, `epsilon`, `steps`, `selection`, or `record`.

Therefore, the per-sample mismatch result is based on the non-attack semantic/target-band generalization root, not the problematic round03 attack-generated root. Caveat: this root is curated target-band semantic data, not an unbiased random external benchmark. Dedicated provenance note: `docs/burgers_fulltag_generalization_source_provenance_20260608.md`.
