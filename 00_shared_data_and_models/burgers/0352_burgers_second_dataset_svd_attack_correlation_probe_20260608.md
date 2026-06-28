# Burgers Second-Dataset SVD/Attack Correlation Probe - 2026-06-08

Status: computed the correlation that can be computed from existing local artifacts. No new training, attack run, or Jacobian/SVD computation was launched.

## Question

Is Jacobian/SVD spectral norm highly correlated with attack-after loss growth?

## Source Artifacts

Attack/tag artifact:

- `forensics/burgers_round03_full_p2q2_52datasets_4models_finalonly_20step_20260607`
- Attack setting: P2Q2 RMS-L2, `epsilon_rms=0.12`, `alpha_rms=0.012`, `20` steps.
- Its `10000` generalization samples are from the second dataset root, `generalization_datasets_rmse_1p5_3x_all_ns50/burgers`.

Jacobian/SVD artifact:

- `forensics/burgers_loss3_selective_round03_loss123_final_extension_jacobian_svd_rep20_top100_20260606/round03_loss123_final_extension_jacobian_svd_summary.csv`

Important matching caveat:

- The SVD artifact's generalization samples are from the third root, `generalization_datasets_burgers_loss3_selective_search/round_03/burgers`, not the second root.
- Therefore, the only strict sample overlap with the second-root full attack artifact is the shared test split: `5` test samples.
- Recheck detail: the SVD manifest has `20` samples total: `5` train, `5` test, and `10` third-root generated/generalization samples. The second-root full attack artifact has only the first `50` train samples (`source_index=0..49`), while the SVD train indices are `210`, `1247`, `809`, `503`, and `1186`, so none of the SVD train samples overlap. The `5` SVD test indices `20`, `146`, `109`, `50`, and `56` overlap because the attack artifact includes all `150` test samples. The `10` SVD generalization samples do not overlap because they are round03 stress-root files such as `burgers_loss3_selective_r03_d24.pt`.
- Strict checkpoint overlap exists for `baseline`, `loss2_epoch2000`, and `loss3_epoch1500`.
- `loss1` is not strict: SVD has `loss1_epoch5000`, while the full attack artifact uses `loss1_epoch8000`.

## Outputs

Derived outputs written:

- `forensics/burgers_second_dataset_svd_attack_correlation_probe_20260608/matched_svd_attack_rows.csv`
- `forensics/burgers_second_dataset_svd_attack_correlation_probe_20260608/correlation_summary.csv`

## Main Result

For strict checkpoint-matched rows only, using `baseline`, `loss2_epoch2000`, and `loss3_epoch1500` on the `5` shared test samples:

| x | y | rows | Pearson | Spearman |
|---|---|---:|---:|---:|
| model Jacobian spectral norm | final attack loss | 15 | `0.1340` | `0.0714` |
| model Jacobian spectral norm | attack increase | 15 | `0.1329` | `0.0893` |
| error Jacobian spectral norm `||J_model-J_solver||_2` | final attack loss | 15 | `0.5433` | `0.4036` |
| error Jacobian spectral norm `||J_model-J_solver||_2` | attack increase | 15 | `0.5330` | `0.3929` |

Including the non-strict loss1 row, where SVD uses `loss1_epoch5000` but attack uses `loss1_epoch8000`:

| x | y | rows | Pearson | Spearman |
|---|---|---:|---:|---:|
| model Jacobian spectral norm | attack increase | 20 | `0.1184` | `0.1023` |
| error Jacobian spectral norm `||J_model-J_solver||_2` | attack increase | 20 | `0.5439` | `0.2331` |

Per-model error-Jacobian correlations are very small-sample checks only (`n=5` each):

| model | y | Pearson | Spearman |
|---|---|---:|---:|
| baseline | attack increase | `0.3714` | `0.8000` |
| loss2_epoch2000 | attack increase | `0.7438` | `0.9000` |
| loss3_epoch1500 | attack increase | `0.3458` | `0.4000` |
| loss1_epoch5000 SVD vs loss1_epoch8000 attack | attack increase | `0.4756` | `0.2000` |

## Conclusion

Observed evidence from the currently matchable rows:

- Plain model Jacobian spectral norm is basically not correlated with attack loss growth in the matched subset (`Pearson about 0.13`).
- Error-Jacobian spectral norm, `||J_model-J_solver||_2`, has a moderate positive correlation with attack loss growth (`Pearson about 0.53` on strict rows).
- This is not strong enough, and the sample count is too small, to call it a high-correlation result.

Inference:

- The existing artifacts suggest that the **error operator** spectral norm is the more relevant quantity than the raw model Jacobian spectral norm.
- But for the second generalization dataset specifically, a real answer still needs a matched SVD/Jacobian run on representative second-root generalization samples. The existing SVD generalization rows are from the third stress root, so this probe cannot settle the full second-root question.

