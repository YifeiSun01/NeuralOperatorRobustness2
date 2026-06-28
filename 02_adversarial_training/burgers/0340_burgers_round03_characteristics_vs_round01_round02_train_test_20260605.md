# Burgers Round03 Dataset Characteristics Vs Round01/Round02/Train/Test

Date: 2026-06-05

## Status

This note summarizes the input/output and attack-geometry characteristics of round03 against the original train/test splits, the official round01 loss3-aligned set, and the local partial round02 files. Round02 is not official: it has 10 `.pt` files and no manifest/summary.

## Sources

- Detail CSV: `forensics/burgers_round03_characteristics_vs_round01_round02_train_test_20260605/dataset_characteristics_detail.csv`
- Group summary CSV: `forensics/burgers_round03_characteristics_vs_round01_round02_train_test_20260605/group_characteristics_summary.csv`
- Baseline difficulty source CSV: `forensics/burgers_round03_characteristics_vs_round01_round02_train_test_20260605/baseline_difficulty_sources.csv`
- Round03 vs round01 ratios JSON: `forensics/burgers_round03_characteristics_vs_round01_round02_train_test_20260605/round3_vs_round1_ratios.json`

## Group Summary

| group | datasets | samples_total | official | x_min_mean | x_max_mean | x_oob_mean_mean | x_oob_frac_mean | delta_l2_rms_mean_mean | delta_linf_mean_mean | linf_over_l2rms_mean_mean | baseline_rmse_eval_mean | baseline_relative_l2_eval_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train_original | 1 | 1350 | True | 0.000000 | 1.000000 | 0.000000 | 0.000000 |  |  |  | 0.008984 | 0.016898 |
| test_original | 1 | 150 | True | 0.000000 | 1.000000 | 0.000000 | 0.000000 |  |  |  | 0.009544 | 0.017755 |
| round01_loss3_aligned | 50 | 2500 | True | -0.287345 | 1.307812 | 0.006013 | 0.060861 | 0.060071 | 0.280736 | 4.680996 | 0.042543 | 0.078831 |
| round02_partial_unofficial | 10 | 500 | False | -0.508235 | 1.610399 | 0.017986 | 0.095082 | 0.110147 | 0.467989 | 4.258522 |  |  |
| round03_loss3_selective | 50 | 2500 | True | -0.542318 | 1.579254 | 0.019239 | 0.096704 | 0.114373 | 0.476434 | 4.208629 | 0.088998 | 0.159756 |

## Source And Selection Geometry

| group | source_split_counts | epsilon_fraction_counts | attack_steps_counts | loss1_cosine_mean_mean | loss2_cosine_mean_mean | loss3_cosine_mean_mean | selectivity_margin_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train_original | {} | {} | {} |  |  |  |  |
| test_original | {} | {} | {} |  |  |  |  |
| round01_loss3_aligned | {"train": 50} | {"0.06": 50} | {"5": 50} |  |  |  |  |
| round02_partial_unofficial | {"test": 5, "train": 5} | {"0.1": 10} | {"8": 10} |  |  |  |  |
| round03_loss3_selective | {"test": 27, "train": 23} | {"0.08": 10, "0.1": 20, "0.12": 20} | {"10": 20, "8": 30} | 0.364237 | 0.152738 | 0.740184 | 0.375216 |

## Observed Differences

- Train/test inputs are clean `[0, 1]` data: `x_min=0`, `x_max=1`, and zero out-of-bound amount/fraction in this audit.
- Round01 is a mild loss3-aligned adversarial shift. Its mean baseline generated RMSE is `0.042543`, mean `x_oob_mean` is `0.006013`, mean delta RMS is `0.060071`, and mean delta L_inf is `0.280736`.
- Round03 is the deliberate stronger/selective shift. Its mean baseline generated RMSE is `0.088998`, about `2.09x` round01. Its mean `x_oob_mean` is `0.019239`, about `3.20x` round01. Its delta RMS and L_inf are about `1.90x` and `1.70x` round01.
- Round03 is not a random broad distribution shift like the older aggressive round00 style. It was selected by loss3-selective gradient geometry: selection cosine means are loss1/loss2/loss3 `0.364237 / 0.152738 / 0.740184` from `selection_summary.json`.
- Round03 uses mixed train/test sources and stronger attack settings: selected files include epsilon fractions `0.08`, `0.10`, and `0.12`, with 8 or 10 attack steps. Round01 used train-source loss3 adversarial samples with epsilon fraction `0.06` and 5 steps.
- Partial round02 sits between round01 and round03 in many raw range/perturbation statistics, but it is incomplete and should not be used as a formal comparison target. It has only 10 local `.pt` files and no manifest/summary/evaluation record.

## Interpretation

- Compared with train/test, round03 deliberately leaves the clean data manifold: inputs frequently go below 0 or above 1, and solver outputs also extend below 0 and above 1. This is why clean train/test performance is not the claim.
- Compared with round01, round03 is about twice as hard at baseline and roughly 2x stronger in perturbation size, but it keeps loss3 directional selectivity. That is the key design difference.
- Compared with partial round02, round03 is official, fully selected, and backed by screening/50-epoch/SVD evidence. Round02 is just a stopped partial artifact.
