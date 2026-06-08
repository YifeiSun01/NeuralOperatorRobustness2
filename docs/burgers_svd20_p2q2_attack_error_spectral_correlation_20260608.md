# Burgers SVD20 P2Q2 Attack / Error-Spectral Correlation - 2026-06-08

Status: completed GPU attack run on all 20 existing Burgers SVD sample points and correlated attack loss with error-Jacobian spectral norm. No Jacobian/SVD recomputation was performed; this run used the existing SVD artifact.

## Source Artifacts

SVD source:

- `forensics/burgers_loss3_selective_round03_loss123_final_extension_jacobian_svd_rep20_top100_20260606/round03_loss123_final_extension_sample_manifest.csv`
- `forensics/burgers_loss3_selective_round03_loss123_final_extension_jacobian_svd_rep20_top100_20260606/round03_loss123_final_extension_jacobian_svd_summary.csv`

Attack output:

- `forensics/burgers_svd20_p2q2_attack_correlation_20260608/`

Attack settings:

- P2Q2 RMS-L2 attack
- `epsilon_rms=0.12`
- `alpha_rms=0.012`
- `20` steps
- Batch size `20`

Models attacked, matching the SVD checkpoints:

- `baseline`
- `loss1_epoch5000`
- `loss2_epoch2000`
- `loss3_epoch1500`

GPU evidence:

- `forensics/burgers_svd20_p2q2_attack_correlation_20260608/nvidia_smi.txt`
- `forensics/burgers_svd20_p2q2_attack_correlation_20260608/config.json`
- Observed in config: PyTorch `2.8.0+cu126`, CUDA `12.6`, Tesla V100-SXM2-32GB, compute capability `[7, 0]`, arch list includes `sm_70`, CUDA matmul sanity value `128.0`.

## Sample Scope

Observed SVD manifest scope:

- `20` total SVD samples
- `5` train samples
- `5` test samples
- `10` round03 stress-root generalization samples

Important caveat:

- This answers the user's requested question for the **20 SVD samples**.
- It is not a full second-root ns50 generalization correlation, because the SVD generalization rows are from `generalization_datasets_burgers_loss3_selective_search/round_03/burgers`, not `generalization_datasets_rmse_1p5_3x_all_ns50/burgers`.

## Attack Mean Results

Observed from `summary.json` and `attack_loss_by_model_sample.csv`:

| model | initial loss mean | final attack loss mean | attack increase mean |
|---|---:|---:|---:|
| baseline | 0.003500117 | 0.022570767 | 0.019070650 |
| loss1_epoch5000 | 0.000392569 | 0.010061328 | 0.009668759 |
| loss2_epoch2000 | 0.000499535 | 0.009710704 | 0.009211169 |
| loss3_epoch1500 | 0.000213873 | 0.007146387 | 0.006932515 |

Inference from these means:

- On these 20 SVD samples under the same 20-step P2Q2 budget, loss3 has the lowest mean final attack loss and the smallest mean attack increase among loss1/loss2/loss3.

## Error-Spectral Correlation

Observed from `error_svd_attack_correlation_summary.csv`.

Main correlation between error-Jacobian spectral norm `||J_model - J_solver||_2` and MSE attack increase:

| scope | rows | Pearson | Spearman | reading |
|---|---:|---:|---:|---|
| all 20 samples, all 4 models | 80 | 0.832552 | 0.807712 | high positive |
| all 20 samples, trained 3 models only | 60 | 0.832868 | 0.726702 | high positive |
| round03 generalization 10 samples, all 4 models | 40 | 0.768775 | 0.781051 | high positive |
| round03 generalization 10 samples, trained 3 only | 30 | 0.764073 | 0.727697 | high positive |

Per-model correlations across the same 20 samples:

| model | rows | Pearson vs attack increase | Spearman vs attack increase |
|---|---:|---:|---:|
| baseline | 20 | 0.801739 | 0.793985 |
| loss1_epoch5000 | 20 | 0.828419 | 0.890226 |
| loss2_epoch2000 | 20 | 0.845560 | 0.921805 |
| loss3_epoch1500 | 20 | 0.871745 | 0.771429 |

Correlation with final attack loss, not just increase:

| scope | rows | Pearson | Spearman |
|---|---:|---:|---:|
| all 20 samples, all 4 models | 80 | 0.848999 | 0.815331 |
| all 20 samples, trained 3 models only | 60 | 0.833907 | 0.732981 |
| round03 generalization 10 samples, trained 3 only | 30 | 0.769792 | 0.731257 |

## Conclusion

Observed evidence on the 20 SVD sample points:

- Yes, for the error Jacobian `||J_model - J_solver||_2`, the correlation with attack loss growth is high and positive on the full 20-sample SVD probe.
- This is much stronger than the earlier 5-test-sample overlap probe.
- The high relationship is specifically for the **error Jacobian**. The previous "raw model Jacobian spectral norm" question remains a different quantity and was not the user's target here.

Inference:

- On these SVD-selected samples, a larger local model-solver error operator norm is a strong predictor of larger fixed-budget P2Q2 attack damage.
- This supports the mechanism interpretation that robustness is tied to the local error operator, not merely to clean prediction loss or the raw model Jacobian size.

Remaining work:

- To make the same claim for the second ns50 root as a full benchmark, run Jacobian/SVD on representative second-root samples and join them to the existing full attack artifact.

