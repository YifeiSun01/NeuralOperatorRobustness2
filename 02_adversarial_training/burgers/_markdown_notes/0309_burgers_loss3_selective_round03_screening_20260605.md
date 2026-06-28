# Burgers Round03 Loss3-Selective Generalization Search

Date: 2026-06-05

## Status

Round03 is a strong loss3-selective Burgers 1D generalization dataset candidate. The prediction-loss result is clearly favorable to loss3 on the new generalization split at 10 epochs and remains favorable at 50 epochs. A small rep5/top20 Jacobian/SVD diagnostic also supports loss3 on model-minus-solver error spectral norm and top10/top20 subspace alignment, while top5 subspace alignment is mixed.

## GPU Path

Observed from pre-run checks in this session:

- `nvidia-smi`: Tesla V100-SXM2-32GB, 32768 MiB, no active process before runs.
- PyTorch: `2.8.0+cu126`; CUDA runtime `12.6`; CUDA available; device Tesla V100-SXM2-32GB; compute capability `(7, 0)`; arch list includes `sm_70`; CUDA matmul succeeded.
- JAX: `0.10.0`; backend `gpu`; devices `[CudaDevice(id=0)]`; GPU matmul succeeded.

## Source And Output Paths

- Dataset root: `generalization_datasets_burgers_loss3_selective_search/round_03`
- Candidate pool: `generalization_datasets_burgers_loss3_selective_search/round_03_candidate_pool`
- Generator/scorer tool: `tools/generate_burgers_loss3_selective_generalization.py`
- Selection summary JSON: `generalization_datasets_burgers_loss3_selective_search/round_03/selection_summary.json`
- Gradient trajectory CSV: `forensics/burgers_loss3_selective_round03_loss123_gradient_alignment_10step_20260605/gradient_alignment_mean_by_variant.csv`
- 10 epoch run CSVs: `adversarial_training_runs/burgers_loss3_selective_round03_loss1_10ep_20260605/burgers/eval_split_summary.csv`, `adversarial_training_runs/burgers_loss3_selective_round03_loss2_10ep_20260605/burgers/eval_split_summary.csv`, `adversarial_training_runs/burgers_loss3_selective_round03_loss3_10ep_20260605/burgers/eval_split_summary.csv`
- 50 epoch run CSVs: `adversarial_training_runs/burgers_loss3_selective_round03_loss1_50ep_20260605/burgers/eval_split_summary.csv`, `adversarial_training_runs/burgers_loss3_selective_round03_loss2_50ep_20260605/burgers/eval_split_summary.csv`, `adversarial_training_runs/burgers_loss3_selective_round03_loss3_50ep_20260605/burgers/eval_split_summary.csv`
- SVD diagnostic root: `forensics/burgers_loss3_selective_round03_final_jacobian_svd_gen5_top20_20260605`

Note: the SVD diagnostic reuses `tools/compare_burgers_round01_final_jacobian_svd.py`, so several output filenames retain a `round01_` prefix. The `config.json` and sample manifest in the SVD output point to the round03 dataset and loss*_epoch50 checkpoints.

## Observed Evidence

### Dataset Selection

Observed from `generalization_datasets_burgers_loss3_selective_search/round_03/selection_summary.json`:

- selected_count: `50` from `60` candidates
- baseline RMSE mean: `0.088998`
- baseline relative L2 mean: `0.159756`
- candidate scoring cosine means: loss1 `0.364237`, loss2 `0.152738`, loss3 `0.740184`
- selectivity margin mean: `0.375216`

### 10-Step Gradient Alignment Trajectory

Observed from `forensics/burgers_loss3_selective_round03_loss123_gradient_alignment_10step_20260605/gradient_alignment_mean_by_variant.csv`:

| variant | clean_train | clean_test | generalization_mixed50 |
| --- | --- | --- | --- |
| loss1_raw | 0.937148 | 0.913948 | 0.093088 |
| loss2_raw | 0.952633 | 0.953660 | 0.091060 |
| loss3_raw | 0.961161 | 0.950632 | 0.231641 |

The trajectory cosine is weaker than the candidate scoring cosine, but loss3 remains about 2.5x the loss1/loss2 generalization cosine in this trajectory probe.

### 10 Epoch Screening

Observed from the three 10 epoch `eval_split_summary.csv` files:

| objective | epoch | gen_rmse | gen_rmse_improve_pct | gen_rel_l2 | gen_rel_l2_improve_pct |
| --- | --- | --- | --- | --- | --- |
| loss1 | 5 | 0.071092 | 20.118903 | 0.127589 | 20.135321 |
| loss1 | 10 | 0.068384 | 23.162416 | 0.122723 | 23.180926 |
| loss2 | 5 | 0.069942 | 21.411096 | 0.125526 | 21.426641 |
| loss2 | 10 | 0.065954 | 25.892190 | 0.118358 | 25.912996 |
| loss3 | 5 | 0.053529 | 39.853576 | 0.096069 | 39.864934 |
| loss3 | 10 | 0.050512 | 43.243750 | 0.090660 | 43.251031 |

### 50 Epoch Confirmation

Final epoch 50 generalization metrics observed from the three 50 epoch `eval_split_summary.csv` files:

| objective | epoch | rmse_dataset_mean | rmse_improve_pct | relative_l2_dataset_mean | rel_l2_improve_pct | accuracy_score_dataset_mean |
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

Best observed generalization epoch in the 50 epoch runs:

| objective | epoch | rmse_dataset_mean | rmse_improve_pct | relative_l2_dataset_mean | accuracy_score_dataset_mean |
| --- | --- | --- | --- | --- | --- |
| loss1 | 49 | 0.055525 | 37.611147 | 0.099623 | 90.979039 |
| loss2 | 49 | 0.054623 | 38.623986 | 0.098006 | 91.113345 |
| loss3 | 42 | 0.041589 | 53.269696 | 0.074630 | 93.082388 |

### Small Jacobian/SVD Diagnostic

Observed from `forensics/burgers_loss3_selective_round03_final_jacobian_svd_gen5_top20_20260605` on 5 round03 generalization samples with top_k=20. Solver Jacobians were recomputed on the new round03 X values.

Model-minus-solver error spectral norm aggregate:

| checkpoint_label | n | error_spectral_norm_mean | error_spectral_norm_median | ratio_to_baseline_error_mean | ratio_to_baseline_error_median | count_error_smaller_than_baseline |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 5 | 6.639156 | 6.594715 | 1.000000 | 1.000000 | 5 |
| loss1_epoch50 | 5 | 5.407318 | 3.983636 | 0.775478 | 0.773610 | 5 |
| loss2_epoch50 | 5 | 5.620033 | 4.295862 | 0.813720 | 0.794559 | 5 |
| loss3_epoch50 | 5 | 4.979735 | 4.966424 | 0.784002 | 0.757993 | 5 |

Model/solver subspace mean principal cosine, de-duplicated by sample/checkpoint/top_k:

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

## Inference

- Round03 is materially better than round01 for demonstrating loss3 generalization improvement. It is harder than round01 at baseline and is much more loss3-selective in both candidate scoring and actual 10/50 epoch generalization improvement.
- The result should be described as a generalization-selective loss3 advantage, not an all-splits advantage. At epoch 50, loss3 has much better round03 generalization RMSE/relative L2 than loss1/loss2, but worse clean train/test RMSE/relative L2.
- The small SVD result is encouraging but not definitive. Loss3 has the lowest mean `||J_model - J_solver||_2` on the rep5 set and the best top10/top20 right subspace means, while top5 subspace means still favor loss1/loss2. If this becomes an official claim, run a larger rep20/top100 diagnostic and compare loss3 best epoch 42 as well as epoch 50.

## Remaining Work

- For a publication-grade Jacobian/subspace claim: run rep20/top100 SVD on round03, recomputing solver Jacobians on round03 X.
- Consider evaluating loss3 epoch42 checkpoint against loss1/loss2 epoch49 best checkpoints, because best generalization RMSE occurred at epoch42 for loss3.
- Keep `.pt`, `.npz`, large CSV/PNG artifacts local or upload to R2/S3; do not add them to Git unless explicitly requested.

## Per-Dataset Loss3 Advantage Audit

Observed from per-dataset `eval_metrics.csv` rows in the 10 epoch screening runs and 50 epoch confirmation runs. Output audit files:

- Detail CSV: `forensics/burgers_loss3_selective_round03_per_dataset_loss3_advantage_20260605/per_dataset_loss3_vs_loss12.csv`
- Summary CSV: `forensics/burgers_loss3_selective_round03_per_dataset_loss3_advantage_20260605/per_dataset_advantage_summary.csv`
- Generated-dataset RMSE exceptions CSV: `forensics/burgers_loss3_selective_round03_per_dataset_loss3_advantage_20260605/generalization_rmse_loss3_not_best.csv`

Definition: `loss3_wins_vs_both` means the loss3 RMSE is lower than both loss1 and loss2 on the same dataset and epoch. `advantage_pct` is computed against the better of loss1/loss2: `(best(loss1, loss2) - loss3) / best(loss1, loss2) * 100`.

### RMSE Win Counts

| case | group | n | loss3_wins_vs_both_count | loss1_winner_count | loss2_winner_count | loss3_winner_count | advantage_pct_mean | advantage_pct_median | advantage_pct_min | advantage_pct_gt_10_count | advantage_pct_lt_0_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| epoch10_screen | all52 | 52 | 50 | 0 | 2 | 50 | 14.118 | 23.258 | -239.130 | 50 | 2 |
| epoch10_screen | gen50 | 50 | 50 | 0 | 0 | 50 | 23.626 | 23.399 | 19.536 | 50 | 0 |
| epoch10_screen | gen_bottom30_d20_d49 | 30 | 30 | 0 | 0 | 30 | 24.457 | 24.846 | 19.767 | 30 | 0 |
| epoch10_screen | gen_top20_d00_d19 | 20 | 20 | 0 | 0 | 20 | 22.380 | 22.450 | 19.536 | 20 | 0 |
| epoch10_screen | train_test2 | 2 | 0 | 0 | 2 | 0 | -223.578 | -223.578 | -239.130 | 0 | 2 |
| epoch50_confirm | all52 | 52 | 50 | 0 | 2 | 50 | 9.313 | 19.196 | -233.848 | 49 | 2 |
| epoch50_confirm | gen50 | 50 | 50 | 0 | 0 | 50 | 18.555 | 19.236 | 2.915 | 49 | 0 |
| epoch50_confirm | gen_bottom30_d20_d49 | 30 | 30 | 0 | 0 | 30 | 17.584 | 17.738 | 2.915 | 29 | 0 |
| epoch50_confirm | gen_top20_d00_d19 | 20 | 20 | 0 | 0 | 20 | 20.010 | 19.692 | 15.027 | 20 | 0 |
| epoch50_confirm | train_test2 | 2 | 0 | 0 | 2 | 0 | -221.716 | -221.716 | -233.848 | 0 | 2 |

### Epoch50 Weakest Generated-Dataset Loss3 Advantages

| dataset_id | rmse_loss1 | rmse_loss2 | rmse_loss3 | rmse_best_loss12_objective | rmse_loss3_advantage_pct_vs_best12 |
| --- | --- | --- | --- | --- | --- |
| burgers_loss3_selective_r03_d48 | 0.031 | 0.032 | 0.030 | loss1 | 2.915 |
| burgers_loss3_selective_r03_d49 | 0.036 | 0.037 | 0.032 | loss1 | 11.607 |
| burgers_loss3_selective_r03_d42 | 0.035 | 0.035 | 0.030 | loss1 | 13.172 |
| burgers_loss3_selective_r03_d41 | 0.047 | 0.047 | 0.040 | loss1 | 14.360 |
| burgers_loss3_selective_r03_d38 | 0.047 | 0.047 | 0.040 | loss2 | 14.486 |

### Train/Test Are Not Loss3 Wins

| case | dataset_id | split | rmse_loss1 | rmse_loss2 | rmse_loss3 | rmse_winner | rmse_loss3_advantage_pct_vs_best12 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| epoch10_screen | test_original_gaussian_corr0p03 | test | 0.006 | 0.006 | 0.017 | loss2 | -208.026 |
| epoch10_screen | train_original_gaussian_corr0p03 | train | 0.006 | 0.005 | 0.017 | loss2 | -239.130 |
| epoch50_confirm | test_original_gaussian_corr0p03 | test | 0.005 | 0.004 | 0.013 | loss2 | -209.583 |
| epoch50_confirm | train_original_gaussian_corr0p03 | train | 0.005 | 0.004 | 0.013 | loss2 | -233.848 |

Inference from the audit:

- For the 50 generated round03 generalization datasets, loss3 wins 50/50 at epoch10 and 50/50 at epoch50 on both RMSE and relative L2.
- For the selected top20 generated datasets (`d00`-`d19`), loss3 wins 20/20 at epoch10 and 20/20 at epoch50; epoch50 top20 minimum RMSE advantage is above 15%.
- For all 52 evaluated datasets, loss3 does not win 52/52 because the original train and test datasets are both better under loss2. The correct claim is therefore: round03 gives a strong generated-generalization advantage for loss3, not a universal train/test/generalization advantage.
- At epoch50, generated dataset `burgers_loss3_selective_r03_d48` is a weak-but-positive case: loss3 still wins, but only by 2.915% over loss1. The other 49 generated datasets exceed 10% RMSE advantage.
