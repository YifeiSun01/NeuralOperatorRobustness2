# Burgers Round03 Loss3-Selective Generalization Complete Report

Date: 2026-06-05

## Executive Summary

Round03 is the strongest current Burgers 1D generalization dataset for demonstrating a generated-OOD advantage of loss3 adversarial training. It is not an all-splits win: loss2 remains better on the original clean train/test splits. The defensible claim is that round03 creates a harder, loss3-selective generated generalization distribution where loss3 wins consistently.

Observed headline results:

- Round03 baseline generated RMSE mean is `0.088998`, about `2.09x` round01.
- Round03 selected-set gradient scoring cosine means are loss1/loss2/loss3 = `0.364237 / 0.152738 / 0.740184`.
- In 10 epoch screening, epoch10 generated RMSE is loss1/loss2/loss3 = `0.068384 / 0.065954 / 0.050512`.
- In 50 epoch confirmation, epoch50 generated RMSE is loss1/loss2/loss3 = `0.055974 / 0.056011 / 0.045295`.
- Per-dataset audit: loss3 wins `50/50` generated datasets at epoch10 and `50/50` generated datasets at epoch50; it wins `50/52` if original train/test are included.
- Small rep5/top20 Jacobian/SVD: mean `||J_model - J_solver||_2` is baseline/loss1/loss2/loss3 = `6.639156 / 5.407318 / 5.620033 / 4.979735`.

## Terminology

- `train_original` and `test_original` are the original clean Burgers data splits. They are one dataset each, so together they contribute 2 evaluated datasets.
- `generated generalization` means one generated `.pt` dataset file. Round03 has 50 such files, `burgers_loss3_selective_r03_d00.pt` through `burgers_loss3_selective_r03_d49.pt`; each file contains 50 Burgers samples.
- `52 datasets` means 1 train + 1 test + 50 generated generalization datasets.
- `d00-d19` means the first 20 selected generated datasets after candidate ranking/selection, not the top 20 samples inside a dataset.
- `loss3 advantage pct` is computed against the better of loss1/loss2 on the same dataset: `(best(loss1, loss2) - loss3) / best(loss1, loss2) * 100`.

## Source Artifacts

- Round03 dataset root: `generalization_datasets_burgers_loss3_selective_search/round_03`
- Round03 candidate pool: `generalization_datasets_burgers_loss3_selective_search/round_03_candidate_pool`
- Round01 dataset root: `generalization_datasets_burgers_loss3_aligned_search/round_01`
- Local partial round02 root: `generalization_datasets_burgers_loss3_aligned_search/round_02`
- Round03 selection summary: `generalization_datasets_burgers_loss3_selective_search/round_03/selection_summary.json`
- Gradient trajectory summary: `forensics/burgers_loss3_selective_round03_loss123_gradient_alignment_10step_20260605/gradient_alignment_mean_by_variant.csv`
- Per-dataset audit detail: `forensics/burgers_loss3_selective_round03_per_dataset_loss3_advantage_20260605/per_dataset_loss3_vs_loss12.csv`
- Per-dataset audit summary: `forensics/burgers_loss3_selective_round03_per_dataset_loss3_advantage_20260605/per_dataset_advantage_summary.csv`
- Small SVD diagnostic root: `forensics/burgers_loss3_selective_round03_final_jacobian_svd_gen5_top20_20260605`

Large generated datasets, checkpoints, NPZ files, and CSV outputs are intentionally left local and are not meant to be committed to Git.

## Prior Rounds Context

| item | value | source |
| --- | --- | --- |
| round00 baseline generated RMSE mean | 0.374300 | docs/burgers_generalization_round00_round01_direction_analysis_20260605.md |
| round00 gradient cosine loss1/loss2/loss3 | -0.0155 / 0.0374 / 0.0342 | docs/burgers_generalization_round00_round01_direction_analysis_20260605.md |
| round01 baseline generated RMSE mean | 0.042540 | generalization_eval_burgers_loss3_aligned_search_round01/metrics_sorted_by_similarity.csv |
| round01 50-step gradient cosine loss1/loss2/loss3 | 0.2357 / 0.1667 / 0.7354 | forensics/burgers_loss3_aligned_round01_loss123_gradient_alignment_50step_20260604 |
| round01 best gen RMSE loss3 epoch463 | 0.007302 | round01 training report |
| round01 final gen RMSE loss1 epoch1000 / loss2 epoch500 / loss3 epoch500 | 0.009127 / 0.012053 / 0.009939 | round01 eval_split_summary.csv |
| round01 gen error spectral norm mean loss1 / loss2 / loss3 | 1.3575 / 1.7846 / 1.6195 | forensics/burgers_loss3_aligned_round01_final_jacobian_svd_rep20_top100_20260604 |

Inference from prior rounds: round00 was hard but not loss3-selective; round01 was loss3-selective but not hard enough to make loss3 clearly beat loss1/loss2 at final checkpoints. Round03 was designed to be more aggressive than round01 while keeping loss3-selective gradient geometry.

## Dataset Characteristics

Observed from `.pt` tensors and existing eval/selection files:

| group | datasets | samples_total | official | x_min_mean | x_max_mean | x_oob_mean_mean | x_oob_frac_mean | delta_l2_rms_mean_mean | delta_linf_mean_mean | linf_over_l2rms_mean_mean | baseline_rmse_eval_mean | baseline_relative_l2_eval_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train_original | 1 | 1350 | true | 0.000000 | 1.000000 | 0.000000 | 0.000000 |  |  |  | 0.008984 | 0.016898 |
| test_original | 1 | 150 | true | 0.000000 | 1.000000 | 0.000000 | 0.000000 |  |  |  | 0.009544 | 0.017755 |
| round01_loss3_aligned | 50 | 2500 | true | -0.287345 | 1.307812 | 0.006013 | 0.060861 | 0.060071 | 0.280736 | 4.680996 | 0.042543 | 0.078831 |
| round02_partial_unofficial | 10 | 500 | false | -0.508235 | 1.610399 | 0.017986 | 0.095082 | 0.110147 | 0.467989 | 4.258522 |  |  |
| round03_loss3_selective | 50 | 2500 | true | -0.542318 | 1.579254 | 0.019239 | 0.096704 | 0.114373 | 0.476434 | 4.208629 | 0.088998 | 0.159756 |

Round03 vs round01 ratios:

| ratio | value |
| --- | --- |
| baseline_rmse_round3_over_round1 | 2.091955 |
| x_oob_mean_round3_over_round1 | 3.199424 |
| delta_l2_rms_round3_over_round1 | 1.903950 |
| delta_linf_round3_over_round1 | 1.697090 |

Source and attack settings:

| group | source_split_counts | epsilon_fraction_counts | attack_steps_counts | loss1_cosine_mean_mean | loss2_cosine_mean_mean | loss3_cosine_mean_mean | selectivity_margin_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train_original | {} | {} | {} |  |  |  |  |
| test_original | {} | {} | {} |  |  |  |  |
| round01_loss3_aligned | {"train": 50} | {"0.06": 50} | {"5": 50} |  |  |  |  |
| round02_partial_unofficial | {"test": 5, "train": 5} | {"0.1": 10} | {"8": 10} |  |  |  |  |
| round03_loss3_selective | {"test": 27, "train": 23} | {"0.08": 10, "0.1": 20, "0.12": 20} | {"10": 20, "8": 30} | 0.364237 | 0.152738 | 0.740184 | 0.375216 |

Interpretation: train/test are clean `[0, 1]` data. Round01 is a mild loss3-aligned adversarial shift. Round03 is more out-of-bound, stronger in perturbation size, uses mixed train/test sources, and is selected by loss3 direction rather than raw difficulty alone. Round02 is local partial only: 10 `.pt` files, no manifest, no summary, no evaluation record.

## Round03 Selection And Gradient Alignment

Selection summary observed from `selection_summary.json`:

| selected_count | candidate_count | selection_score_mean | baseline_rmse_mean | baseline_relative_l2_mean | loss1_cosine_mean | loss2_cosine_mean | loss3_cosine_mean | selectivity_margin_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 50 | 60 | 0.851922 | 0.088998 | 0.159756 | 0.364237 | 0.152738 | 0.740184 | 0.375216 |

10-step gradient alignment trajectory observed from `gradient_alignment_mean_by_variant.csv`:

| variant | clean_train | clean_test | generalization_mixed50 |
| --- | --- | --- | --- |
| loss1_raw | 0.937148 | 0.913948 | 0.093088 |
| loss2_raw | 0.952633 | 0.953660 | 0.091060 |
| loss3_raw | 0.961161 | 0.950632 | 0.231641 |

The trajectory cosine is weaker than the original candidate scoring cosine, but loss3 still has about 2.5x the generated-generalization cosine of loss1/loss2.

## 10 Epoch Screening

| objective | epoch | gen_rmse | gen_rmse_improve_pct | gen_relative_l2 | gen_relative_l2_improve_pct | gen_accuracy_score |
| --- | --- | --- | --- | --- | --- | --- |
| loss1 | 5 | 0.071092 | 20.118903 | 0.127589 | 20.135321 | 88.726488 |
| loss1 | 10 | 0.068384 | 23.162416 | 0.122723 | 23.180926 | 89.110173 |
| loss2 | 5 | 0.069942 | 21.411096 | 0.125526 | 21.426641 | 88.888696 |
| loss2 | 10 | 0.065954 | 25.892190 | 0.118358 | 25.912996 | 89.458598 |
| loss3 | 5 | 0.053529 | 39.853576 | 0.096069 | 39.864934 | 91.267289 |
| loss3 | 10 | 0.050512 | 43.243750 | 0.090660 | 43.251031 | 91.718221 |

## 50 Epoch Confirmation

Final epoch 50 generated-generalization improvement:

| objective | epoch | gen_rmse | gen_rmse_improve_pct | gen_relative_l2 | gen_relative_l2_improve_pct | gen_accuracy_score |
| --- | --- | --- | --- | --- | --- | --- |
| loss1 | 50 | 0.055974 | 37.105807 | 0.100435 | 37.132450 | 90.913048 |
| loss2 | 50 | 0.056011 | 37.064953 | 0.100501 | 37.090659 | 90.907065 |
| loss3 | 50 | 0.045295 | 49.105775 | 0.081251 | 49.140740 | 92.508632 |

Final epoch 50 train/test/generalization metrics:

| objective | split | rmse_dataset_mean | relative_l2_dataset_mean | accuracy_score_dataset_mean |
| --- | --- | --- | --- | --- |
| loss1 | generalization | 0.055974 | 0.100435 | 90.913048 |
| loss1 | test | 0.005391 | 0.010030 | 99.006994 |
| loss1 | train | 0.005179 | 0.009742 | 99.035209 |
| loss2 | generalization | 0.056011 | 0.100501 | 90.907065 |
| loss2 | test | 0.004358 | 0.008107 | 99.195776 |
| loss2 | train | 0.003996 | 0.007517 | 99.253941 |
| loss3 | generalization | 0.045295 | 0.081251 | 92.508632 |
| loss3 | test | 0.013491 | 0.025099 | 97.551526 |
| loss3 | train | 0.013342 | 0.025094 | 97.552006 |

Best generated-generalization epoch in the 50 epoch runs:

| objective | epoch | rmse_dataset_mean | rmse_improve_pct | relative_l2_dataset_mean | accuracy_score_dataset_mean |
| --- | --- | --- | --- | --- | --- |
| loss1 | 49 | 0.055525 | 37.611147 | 0.099623 | 90.979039 |
| loss2 | 49 | 0.054623 | 38.623986 | 0.098006 | 91.113345 |
| loss3 | 42 | 0.041589 | 53.269696 | 0.074630 | 93.082388 |

## Per-Dataset Generated-Generalization Audit

RMSE win-count summary:

| case | group | n | loss3_wins_vs_both_count | loss1_winner_count | loss2_winner_count | loss3_winner_count | advantage_pct_mean | advantage_pct_median | advantage_pct_min | advantage_pct_gt_10_count | advantage_pct_lt_0_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| epoch10_screen | all52 | 52 | 50 | 0 | 2 | 50 | 14.118169 | 23.258338 | -239.130 | 50 | 2 |
| epoch10_screen | gen50 | 50 | 50 | 0 | 0 | 50 | 23.626023 | 23.399377 | 19.536203 | 50 | 0 |
| epoch10_screen | gen_bottom30_d20_d49 | 30 | 30 | 0 | 0 | 30 | 24.456681 | 24.845561 | 19.766960 | 30 | 0 |
| epoch10_screen | gen_top20_d00_d19 | 20 | 20 | 0 | 0 | 20 | 22.380037 | 22.450409 | 19.536203 | 20 | 0 |
| epoch10_screen | train_test2 | 2 | 0 | 0 | 2 | 0 | -223.578 | -223.578 | -239.130 | 0 | 2 |
| epoch50_confirm | all52 | 52 | 50 | 0 | 2 | 50 | 9.313432 | 19.196214 | -233.848 | 49 | 2 |
| epoch50_confirm | gen50 | 50 | 50 | 0 | 0 | 50 | 18.554593 | 19.236268 | 2.915085 | 49 | 0 |
| epoch50_confirm | gen_bottom30_d20_d49 | 30 | 30 | 0 | 0 | 30 | 17.584070 | 17.737709 | 2.915085 | 29 | 0 |
| epoch50_confirm | gen_top20_d00_d19 | 20 | 20 | 0 | 0 | 20 | 20.010378 | 19.691701 | 15.027234 | 20 | 0 |
| epoch50_confirm | train_test2 | 2 | 0 | 0 | 2 | 0 | -221.716 | -221.716 | -233.848 | 0 | 2 |

Original train/test rows, showing why loss3 is not a 52/52 winner:

| case | dataset_id | split | rmse_loss1 | rmse_loss2 | rmse_loss3 | rmse_winner | rmse_loss3_advantage_pct_vs_best12 | relative_l2_winner | relative_l2_loss3_advantage_pct_vs_best12 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| epoch10_screen | test_original_gaussian_corr0p03 | test | 0.006257 | 0.005577 | 0.017178 | loss2 | -208.026 | loss2 | -208.026 |
| epoch10_screen | train_original_gaussian_corr0p03 | train | 0.005711 | 0.005011 | 0.016993 | loss2 | -239.130 | loss2 | -239.130 |
| epoch50_confirm | test_original_gaussian_corr0p03 | test | 0.005391 | 0.004358 | 0.013491 | loss2 | -209.583 | loss2 | -209.583 |
| epoch50_confirm | train_original_gaussian_corr0p03 | train | 0.005179 | 0.003996 | 0.013342 | loss2 | -233.848 | loss2 | -233.848 |

### Epoch10: All 50 Generated Datasets

| dataset_id | rmse_loss1 | rmse_loss2 | rmse_loss3 | rmse_best_loss12_objective | rmse_loss3_advantage_pct_vs_best12 | relative_l2_loss1 | relative_l2_loss2 | relative_l2_loss3 | relative_l2_loss3_advantage_pct_vs_best12 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| burgers_loss3_selective_r03_d00 | 0.095129 | 0.092775 | 0.072485 | loss2 | 21.870291 | 0.167217 | 0.163079 | 0.127413 | 21.870291 |
| burgers_loss3_selective_r03_d01 | 0.091113 | 0.088631 | 0.069788 | loss2 | 21.259708 | 0.165170 | 0.160670 | 0.126512 | 21.259708 |
| burgers_loss3_selective_r03_d02 | 0.084413 | 0.082038 | 0.062666 | loss2 | 23.612879 | 0.152231 | 0.147948 | 0.113013 | 23.612879 |
| burgers_loss3_selective_r03_d03 | 0.088112 | 0.085685 | 0.067170 | loss2 | 21.608132 | 0.156049 | 0.151749 | 0.118959 | 21.608132 |
| burgers_loss3_selective_r03_d04 | 0.080688 | 0.078025 | 0.059153 | loss2 | 24.186262 | 0.143431 | 0.138697 | 0.105151 | 24.186262 |
| burgers_loss3_selective_r03_d05 | 0.076778 | 0.074886 | 0.058898 | loss2 | 21.348899 | 0.144956 | 0.141383 | 0.111199 | 21.348899 |
| burgers_loss3_selective_r03_d06 | 0.087977 | 0.085581 | 0.065549 | loss2 | 23.406841 | 0.152919 | 0.148755 | 0.113936 | 23.406841 |
| burgers_loss3_selective_r03_d07 | 0.082583 | 0.079683 | 0.061377 | loss2 | 22.973626 | 0.145292 | 0.140190 | 0.107983 | 22.973626 |
| burgers_loss3_selective_r03_d08 | 0.081386 | 0.078915 | 0.062123 | loss2 | 21.278836 | 0.146501 | 0.142054 | 0.111826 | 21.278836 |
| burgers_loss3_selective_r03_d09 | 0.080347 | 0.077960 | 0.062730 | loss2 | 19.536203 | 0.147345 | 0.142968 | 0.115037 | 19.536203 |
| burgers_loss3_selective_r03_d10 | 0.079915 | 0.077545 | 0.061136 | loss2 | 21.160102 | 0.149519 | 0.145085 | 0.114385 | 21.160102 |
| burgers_loss3_selective_r03_d11 | 0.081519 | 0.078990 | 0.060513 | loss2 | 23.391913 | 0.143706 | 0.139247 | 0.106675 | 23.391913 |
| burgers_loss3_selective_r03_d12 | 0.085319 | 0.082985 | 0.064781 | loss2 | 21.936084 | 0.150457 | 0.146340 | 0.114239 | 21.936084 |
| burgers_loss3_selective_r03_d13 | 0.085392 | 0.083090 | 0.064623 | loss2 | 22.225781 | 0.146693 | 0.142739 | 0.111014 | 22.225781 |
| burgers_loss3_selective_r03_d14 | 0.079847 | 0.077551 | 0.059896 | loss2 | 22.765734 | 0.139348 | 0.135342 | 0.104530 | 22.765734 |
| burgers_loss3_selective_r03_d15 | 0.078925 | 0.076357 | 0.058902 | loss2 | 22.859763 | 0.140480 | 0.135911 | 0.104842 | 22.859763 |
| burgers_loss3_selective_r03_d16 | 0.076527 | 0.074297 | 0.057967 | loss2 | 21.979798 | 0.141160 | 0.137047 | 0.106925 | 21.979798 |
| burgers_loss3_selective_r03_d17 | 0.084584 | 0.082323 | 0.062236 | loss2 | 24.400094 | 0.146630 | 0.142710 | 0.107889 | 24.400094 |
| burgers_loss3_selective_r03_d18 | 0.074245 | 0.071885 | 0.055262 | loss2 | 23.124762 | 0.129903 | 0.125773 | 0.096689 | 23.124762 |
| burgers_loss3_selective_r03_d19 | 0.067347 | 0.064877 | 0.050166 | loss2 | 22.675038 | 0.123511 | 0.118981 | 0.092002 | 22.675038 |
| burgers_loss3_selective_r03_d20 | 0.063878 | 0.061644 | 0.047786 | loss2 | 22.481124 | 0.118106 | 0.113976 | 0.088353 | 22.481124 |
| burgers_loss3_selective_r03_d21 | 0.068548 | 0.066082 | 0.049214 | loss2 | 25.526544 | 0.123315 | 0.118878 | 0.088533 | 25.526544 |
| burgers_loss3_selective_r03_d22 | 0.060632 | 0.058330 | 0.044586 | loss2 | 23.562698 | 0.111702 | 0.107461 | 0.082140 | 23.562698 |
| burgers_loss3_selective_r03_d23 | 0.085664 | 0.083459 | 0.066961 | loss2 | 19.766960 | 0.159743 | 0.155631 | 0.124867 | 19.766960 |
| burgers_loss3_selective_r03_d24 | 0.061035 | 0.058315 | 0.043778 | loss2 | 24.928445 | 0.110575 | 0.105647 | 0.079311 | 24.928445 |
| burgers_loss3_selective_r03_d25 | 0.061720 | 0.059187 | 0.043759 | loss2 | 26.067271 | 0.112278 | 0.107670 | 0.079604 | 26.067271 |
| burgers_loss3_selective_r03_d26 | 0.064990 | 0.062636 | 0.046635 | loss2 | 25.546981 | 0.116638 | 0.112414 | 0.083695 | 25.546981 |
| burgers_loss3_selective_r03_d27 | 0.064173 | 0.061867 | 0.048304 | loss2 | 21.923948 | 0.116524 | 0.112336 | 0.087708 | 21.923948 |
| burgers_loss3_selective_r03_d28 | 0.066945 | 0.064331 | 0.047935 | loss2 | 25.486925 | 0.119940 | 0.115256 | 0.085881 | 25.486925 |
| burgers_loss3_selective_r03_d29 | 0.060497 | 0.058479 | 0.045793 | loss2 | 21.692903 | 0.110813 | 0.107117 | 0.083880 | 21.692903 |
| burgers_loss3_selective_r03_d30 | 0.065098 | 0.062631 | 0.047317 | loss2 | 24.451377 | 0.115228 | 0.110861 | 0.083754 | 24.451377 |
| burgers_loss3_selective_r03_d31 | 0.067005 | 0.064549 | 0.047647 | loss2 | 26.184713 | 0.116925 | 0.112640 | 0.083146 | 26.184713 |
| burgers_loss3_selective_r03_d32 | 0.058886 | 0.056445 | 0.042426 | loss2 | 24.837221 | 0.110688 | 0.106100 | 0.079747 | 24.837221 |
| burgers_loss3_selective_r03_d33 | 0.073235 | 0.070891 | 0.053259 | loss2 | 24.871765 | 0.124987 | 0.120986 | 0.090895 | 24.871765 |
| burgers_loss3_selective_r03_d34 | 0.061878 | 0.059139 | 0.043510 | loss2 | 26.427000 | 0.108597 | 0.103789 | 0.076361 | 26.427000 |
| burgers_loss3_selective_r03_d35 | 0.065033 | 0.062723 | 0.047479 | loss2 | 24.302598 | 0.114254 | 0.110194 | 0.083414 | 24.302598 |
| burgers_loss3_selective_r03_d36 | 0.061236 | 0.058908 | 0.045557 | loss2 | 22.663918 | 0.109592 | 0.105426 | 0.081533 | 22.663918 |
| burgers_loss3_selective_r03_d37 | 0.063904 | 0.061236 | 0.044866 | loss2 | 26.732415 | 0.113688 | 0.108941 | 0.079818 | 26.732415 |
| burgers_loss3_selective_r03_d38 | 0.059074 | 0.056589 | 0.043946 | loss2 | 22.342629 | 0.104773 | 0.100366 | 0.077942 | 22.342629 |
| burgers_loss3_selective_r03_d39 | 0.051353 | 0.048541 | 0.036365 | loss2 | 25.084084 | 0.094267 | 0.089104 | 0.066753 | 25.084084 |
| burgers_loss3_selective_r03_d40 | 0.052777 | 0.050339 | 0.036516 | loss2 | 27.459914 | 0.096035 | 0.091600 | 0.066447 | 27.459914 |
| burgers_loss3_selective_r03_d41 | 0.057480 | 0.055223 | 0.041997 | loss2 | 23.950423 | 0.103635 | 0.099565 | 0.075719 | 23.950423 |
| burgers_loss3_selective_r03_d42 | 0.046523 | 0.043913 | 0.033935 | loss2 | 22.723494 | 0.085197 | 0.080418 | 0.062144 | 22.723494 |
| burgers_loss3_selective_r03_d43 | 0.052642 | 0.050202 | 0.037725 | loss2 | 24.853901 | 0.093750 | 0.089405 | 0.067185 | 24.853901 |
| burgers_loss3_selective_r03_d44 | 0.050101 | 0.047558 | 0.034865 | loss2 | 26.689967 | 0.090851 | 0.086240 | 0.063222 | 26.689967 |
| burgers_loss3_selective_r03_d45 | 0.045625 | 0.043405 | 0.033799 | loss2 | 22.130742 | 0.082480 | 0.078467 | 0.061102 | 22.130742 |
| burgers_loss3_selective_r03_d46 | 0.047349 | 0.044837 | 0.033172 | loss2 | 26.017309 | 0.087187 | 0.082562 | 0.061082 | 26.017309 |
| burgers_loss3_selective_r03_d47 | 0.048183 | 0.045572 | 0.034255 | loss2 | 24.832349 | 0.086721 | 0.082021 | 0.061653 | 24.832349 |
| burgers_loss3_selective_r03_d48 | 0.043447 | 0.041068 | 0.031699 | loss2 | 22.814427 | 0.079732 | 0.075366 | 0.058172 | 22.814427 |
| burgers_loss3_selective_r03_d49 | 0.048129 | 0.045534 | 0.033082 | loss2 | 27.346374 | 0.085418 | 0.080812 | 0.058713 | 27.346374 |

### Epoch50: All 50 Generated Datasets

| dataset_id | rmse_loss1 | rmse_loss2 | rmse_loss3 | rmse_best_loss12_objective | rmse_loss3_advantage_pct_vs_best12 | relative_l2_loss1 | relative_l2_loss2 | relative_l2_loss3 | relative_l2_loss3_advantage_pct_vs_best12 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| burgers_loss3_selective_r03_d00 | 0.081367 | 0.081196 | 0.064648 | loss2 | 20.380696 | 0.143026 | 0.142726 | 0.113637 | 20.380696 |
| burgers_loss3_selective_r03_d01 | 0.077535 | 0.077345 | 0.059980 | loss2 | 22.450859 | 0.140555 | 0.140211 | 0.108732 | 22.450859 |
| burgers_loss3_selective_r03_d02 | 0.070458 | 0.070477 | 0.056486 | loss1 | 19.830407 | 0.127065 | 0.127099 | 0.101867 | 19.830407 |
| burgers_loss3_selective_r03_d03 | 0.074885 | 0.074768 | 0.060356 | loss2 | 19.276211 | 0.132623 | 0.132416 | 0.106891 | 19.276211 |
| burgers_loss3_selective_r03_d04 | 0.066814 | 0.067018 | 0.053791 | loss1 | 19.491860 | 0.118770 | 0.119132 | 0.095619 | 19.491860 |
| burgers_loss3_selective_r03_d05 | 0.064236 | 0.064664 | 0.051503 | loss1 | 19.821606 | 0.121276 | 0.122085 | 0.097237 | 19.821606 |
| burgers_loss3_selective_r03_d06 | 0.075254 | 0.075087 | 0.060327 | loss2 | 19.657836 | 0.130804 | 0.130515 | 0.104858 | 19.657836 |
| burgers_loss3_selective_r03_d07 | 0.068123 | 0.067988 | 0.055228 | loss2 | 18.768441 | 0.119852 | 0.119614 | 0.097164 | 18.768441 |
| burgers_loss3_selective_r03_d08 | 0.068537 | 0.068477 | 0.056252 | loss2 | 17.853391 | 0.123372 | 0.123264 | 0.101257 | 17.853391 |
| burgers_loss3_selective_r03_d09 | 0.067366 | 0.066999 | 0.051879 | loss2 | 22.567593 | 0.123540 | 0.122867 | 0.095139 | 22.567593 |
| burgers_loss3_selective_r03_d10 | 0.066677 | 0.066565 | 0.050436 | loss2 | 24.230980 | 0.124752 | 0.124543 | 0.094365 | 24.230980 |
| burgers_loss3_selective_r03_d11 | 0.068204 | 0.068466 | 0.055068 | loss1 | 19.260285 | 0.120232 | 0.120694 | 0.097075 | 19.260285 |
| burgers_loss3_selective_r03_d12 | 0.072402 | 0.072380 | 0.058103 | loss2 | 19.725565 | 0.127678 | 0.127639 | 0.102462 | 19.725565 |
| burgers_loss3_selective_r03_d13 | 0.072857 | 0.072816 | 0.057120 | loss2 | 21.555750 | 0.125160 | 0.125089 | 0.098125 | 21.555750 |
| burgers_loss3_selective_r03_d14 | 0.067223 | 0.067162 | 0.054311 | loss2 | 19.134930 | 0.117316 | 0.117210 | 0.094782 | 19.134930 |
| burgers_loss3_selective_r03_d15 | 0.065809 | 0.065739 | 0.053151 | loss2 | 19.148961 | 0.117136 | 0.117011 | 0.094605 | 19.148961 |
| burgers_loss3_selective_r03_d16 | 0.064660 | 0.064593 | 0.049139 | loss2 | 23.924879 | 0.119270 | 0.119147 | 0.090641 | 23.924879 |
| burgers_loss3_selective_r03_d17 | 0.070975 | 0.071338 | 0.060017 | loss1 | 15.438652 | 0.123039 | 0.123667 | 0.104043 | 15.438652 |
| burgers_loss3_selective_r03_d18 | 0.061552 | 0.061350 | 0.052131 | loss2 | 15.027234 | 0.107695 | 0.107341 | 0.091211 | 15.027234 |
| burgers_loss3_selective_r03_d19 | 0.055525 | 0.055607 | 0.042942 | loss1 | 22.661417 | 0.101829 | 0.101979 | 0.078753 | 22.661417 |
| burgers_loss3_selective_r03_d20 | 0.052198 | 0.052171 | 0.040310 | loss2 | 22.735797 | 0.096511 | 0.096461 | 0.074530 | 22.735797 |
| burgers_loss3_selective_r03_d21 | 0.056317 | 0.056396 | 0.044731 | loss1 | 20.572082 | 0.101312 | 0.101454 | 0.080470 | 20.572082 |
| burgers_loss3_selective_r03_d22 | 0.047915 | 0.048128 | 0.039491 | loss1 | 17.581212 | 0.088272 | 0.088666 | 0.072753 | 17.581212 |
| burgers_loss3_selective_r03_d23 | 0.073417 | 0.073332 | 0.055726 | loss2 | 24.008568 | 0.136905 | 0.136748 | 0.103917 | 24.008568 |
| burgers_loss3_selective_r03_d24 | 0.047800 | 0.047794 | 0.040232 | loss2 | 15.822268 | 0.086598 | 0.086587 | 0.072887 | 15.822268 |
| burgers_loss3_selective_r03_d25 | 0.049179 | 0.049282 | 0.039269 | loss1 | 20.149911 | 0.089464 | 0.089652 | 0.071437 | 20.149911 |
| burgers_loss3_selective_r03_d26 | 0.052583 | 0.052601 | 0.041965 | loss1 | 20.193825 | 0.094372 | 0.094404 | 0.075314 | 20.193825 |
| burgers_loss3_selective_r03_d27 | 0.052718 | 0.052822 | 0.041765 | loss1 | 20.776116 | 0.095723 | 0.095912 | 0.075835 | 20.776116 |
| burgers_loss3_selective_r03_d28 | 0.054153 | 0.054182 | 0.043139 | loss1 | 20.339151 | 0.097022 | 0.097075 | 0.077289 | 20.339151 |
| burgers_loss3_selective_r03_d29 | 0.050256 | 0.050052 | 0.041939 | loss2 | 16.209628 | 0.092055 | 0.091682 | 0.076821 | 16.209628 |
| burgers_loss3_selective_r03_d30 | 0.052398 | 0.052649 | 0.042304 | loss1 | 19.264015 | 0.092748 | 0.093192 | 0.074881 | 19.264015 |
| burgers_loss3_selective_r03_d31 | 0.054376 | 0.054353 | 0.043910 | loss2 | 19.212250 | 0.094887 | 0.094847 | 0.076625 | 19.212250 |
| burgers_loss3_selective_r03_d32 | 0.046787 | 0.046987 | 0.036682 | loss1 | 21.599035 | 0.087946 | 0.088322 | 0.068950 | 21.599035 |
| burgers_loss3_selective_r03_d33 | 0.060645 | 0.060438 | 0.048846 | loss2 | 19.180178 | 0.103499 | 0.103146 | 0.083363 | 19.180178 |
| burgers_loss3_selective_r03_d34 | 0.048067 | 0.048303 | 0.039466 | loss1 | 17.894207 | 0.084358 | 0.084772 | 0.069263 | 17.894207 |
| burgers_loss3_selective_r03_d35 | 0.052361 | 0.052516 | 0.044196 | loss1 | 15.593059 | 0.091991 | 0.092264 | 0.077646 | 15.593059 |
| burgers_loss3_selective_r03_d36 | 0.049737 | 0.049645 | 0.042129 | loss2 | 15.139409 | 0.089013 | 0.088848 | 0.075397 | 15.139409 |
| burgers_loss3_selective_r03_d37 | 0.051290 | 0.051508 | 0.043425 | loss1 | 15.333480 | 0.091247 | 0.091636 | 0.077256 | 15.333480 |
| burgers_loss3_selective_r03_d38 | 0.046872 | 0.046740 | 0.039969 | loss2 | 14.485723 | 0.083132 | 0.082897 | 0.070889 | 14.485723 |
| burgers_loss3_selective_r03_d39 | 0.039439 | 0.039497 | 0.029299 | loss1 | 25.709332 | 0.072396 | 0.072502 | 0.053783 | 25.709332 |
| burgers_loss3_selective_r03_d40 | 0.041537 | 0.041500 | 0.032859 | loss2 | 20.822290 | 0.075583 | 0.075516 | 0.059792 | 20.822290 |
| burgers_loss3_selective_r03_d41 | 0.047053 | 0.047053 | 0.040296 | loss1 | 14.360179 | 0.084835 | 0.084837 | 0.072653 | 14.360179 |
| burgers_loss3_selective_r03_d42 | 0.034971 | 0.035245 | 0.030364 | loss1 | 13.171764 | 0.064041 | 0.064544 | 0.055606 | 13.171764 |
| burgers_loss3_selective_r03_d43 | 0.041163 | 0.040959 | 0.034472 | loss2 | 15.838080 | 0.073308 | 0.072944 | 0.061391 | 15.838080 |
| burgers_loss3_selective_r03_d44 | 0.038853 | 0.039027 | 0.031559 | loss1 | 18.773316 | 0.070454 | 0.070770 | 0.057228 | 18.773316 |
| burgers_loss3_selective_r03_d45 | 0.035095 | 0.035020 | 0.029784 | loss2 | 14.951531 | 0.063444 | 0.063310 | 0.053844 | 14.951531 |
| burgers_loss3_selective_r03_d46 | 0.036278 | 0.036637 | 0.030539 | loss1 | 15.819540 | 0.066801 | 0.067463 | 0.056234 | 15.819540 |
| burgers_loss3_selective_r03_d47 | 0.037016 | 0.037044 | 0.030551 | loss1 | 17.464451 | 0.066622 | 0.066673 | 0.054987 | 17.464451 |
| burgers_loss3_selective_r03_d48 | 0.031412 | 0.031810 | 0.030497 | loss1 | 2.915085 | 0.057647 | 0.058377 | 0.055966 | 2.915085 |
| burgers_loss3_selective_r03_d49 | 0.036375 | 0.036807 | 0.032153 | loss1 | 11.606631 | 0.064557 | 0.065324 | 0.057064 | 11.606631 |

Observed from the per-dataset audit: loss3 wins all 50 generated datasets at epoch10 and epoch50. At epoch50, 49/50 generated datasets exceed 10% RMSE advantage; `burgers_loss3_selective_r03_d48` is positive but weak at 2.915%.

## Small Jacobian/SVD Diagnostic

This was a lightweight rep5/top20 diagnostic on round03 generated points. Solver Jacobians were recomputed on the round03 X values; old round01 solver/baseline SVDs were not reused.

Model-minus-solver error spectral norm:

| checkpoint_label | n | error_spectral_norm_mean | error_spectral_norm_median | ratio_to_baseline_error_mean | ratio_to_baseline_error_median | count_error_smaller_than_baseline |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 5 | 6.639156 | 6.594715 | 1.000000 | 1.000000 | 5 |
| loss1_epoch50 | 5 | 5.407318 | 3.983636 | 0.775478 | 0.773610 | 5 |
| loss2_epoch50 | 5 | 5.620033 | 4.295862 | 0.813720 | 0.794559 | 5 |
| loss3_epoch50 | 5 | 4.979735 | 4.966424 | 0.784002 | 0.757993 | 5 |

Model/solver subspace mean principal cosine:

| checkpoint_label | top_k | right_subspace_mean_principal_cosine | left_subspace_mean_principal_cosine |
| --- | --- | --- | --- |
| baseline | 5 | 0.949460 | 0.728854 |
| baseline | 10 | 0.905455 | 0.809408 |
| baseline | 20 | 0.708978 | 0.750351 |
| loss1_epoch50 | 5 | 0.987760 | 0.852786 |
| loss1_epoch50 | 10 | 0.924214 | 0.867358 |
| loss1_epoch50 | 20 | 0.754245 | 0.821771 |
| loss2_epoch50 | 5 | 0.987126 | 0.847488 |
| loss2_epoch50 | 10 | 0.925862 | 0.867251 |
| loss2_epoch50 | 20 | 0.751182 | 0.818680 |
| loss3_epoch50 | 5 | 0.981046 | 0.826461 |
| loss3_epoch50 | 10 | 0.936632 | 0.869525 |
| loss3_epoch50 | 20 | 0.755611 | 0.825277 |

Inference from SVD: loss3 has the lowest mean error spectral norm and best top10/top20 right-subspace means in this small diagnostic, but top5 subspace alignment is mixed. A publication-grade Jacobian/subspace claim should use rep20/top100 and should consider loss3 best epoch42 as well as epoch50.

## Final Claim Wording

Safe wording:

> Round03 is a stronger, loss3-selective Burgers 1D generated-generalization dataset. On the 50 generated OOD datasets, loss3 adversarial training clearly outperforms loss1/loss2 at both 10 and 50 epochs. This is not a clean train/test win: loss2 remains better on original train/test, so the claim should be framed as generated-generalization advantage.

## Remaining Work

- For official Jacobian/subspace evidence, run round03 rep20/top100 SVD with solver Jacobians recomputed on round03 X.
- Compare loss3 epoch42 against loss1/loss2 best-generalization checkpoints around epoch49.
- Keep `.pt`, `.npz`, checkpoints, and large CSV/PNG artifacts local or upload to R2/S3 rather than Git.
