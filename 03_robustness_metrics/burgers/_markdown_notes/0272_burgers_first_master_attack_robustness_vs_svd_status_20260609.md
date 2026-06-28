# Burgers First/Master Attack Robustness Versus SVD Status 20260609

## Purpose

Record the completed first/master Burgers P2Q2 self-attack/tag result and the current status of the requested SVD spectral-norm correlation.

The user's requested main metric is:

- `||J_model - J_solver||_2` versus attack loss / attack loss increase.

This document therefore separates:

- completed first/master attack/robustness evidence;
- auxiliary clean-loss versus attack-damage correlations;
- SVD availability and why the first/master SVD correlation is not currently computable.

## Completed Attack Run

Observed completed run:

- Attack root: `forensics/burgers_first_master_full_p2q2_52datasets_4models_finalonly_20step_20260608`
- Source summary table: `forensics/burgers_first_master_full_p2q2_52datasets_4models_finalonly_20step_20260608/summary_by_model_dataset.csv`
- Derived clean/attack analysis: `forensics/burgers_first_master_full_p2q2_clean_attack_correlation_20260609`
- First/master data root: `generalization_datasets/burgers`
- Dataset count: `52` = train first50 + test + 50 generalization datasets
- Sample count: `10200`
- Attack: P2Q2, `epsilon_rms=0.12`, `alpha_rms=0.012`, `20` steps, `batch_size=500`
- Models: `baseline`, `loss1_epoch8000`, `loss2_epoch2000`, `loss3_epoch1500`

The queued second `round03_selective` follow-up was stopped after the first/master root finished. Its partial output should not be used as a formal result.

## Model-Level Attack Means

### Test Set

| model | clean initial loss | attack final loss | attack increase |
|---|---:|---:|---:|
| baseline | `0.0000910810` | `0.00765143` | `0.00756035` |
| loss1_epoch8000 | `0.000000777436` | `0.00147068` | `0.00146990` |
| loss2_epoch2000 | `0.00000249355` | `0.00165785` | `0.00165536` |
| loss3_epoch1500 | `0.0000204998` | `0.000897277` | `0.000876777` |

Observed conclusion on test: `loss3_epoch1500` has the lowest attack final loss and the smallest attack increase.

### Generalization, Sample-Weighted Over 10000 Samples

| model | clean initial loss | attack final loss | attack increase |
|---|---:|---:|---:|
| baseline | `0.0293379` | `0.0729095` | `0.0435716` |
| loss1_epoch8000 | `0.0383823` | `0.0887901` | `0.0504078` |
| loss2_epoch2000 | `0.0256413` | `0.0638325` | `0.0381912` |
| loss3_epoch1500 | `0.0278850` | `0.0696031` | `0.0417181` |

Observed conclusion on sample-weighted generalization means: `loss2_epoch2000` is slightly lower than `loss3_epoch1500` on both attack final loss and attack increase. `loss3_epoch1500` is clearly better than `loss1_epoch8000`.

## Dataset-Winner Counts On 50 Generalization Datasets

| metric | baseline | loss1_epoch8000 | loss2_epoch2000 | loss3_epoch1500 |
|---|---:|---:|---:|---:|
| clean initial loss lowest | `3` | `26` | `2` | `19` |
| attack final loss lowest | `5` | `0` | `7` | `38` |
| attack increase smallest | `8` | `0` | `5` | `37` |

Observed conclusion on dataset-winner robustness: `loss3_epoch1500` is the clear winner by count. It wins attack final loss on `38/50` generalization datasets and attack increase on `37/50`. `loss1_epoch8000` wins `0/50` robustness datasets by both attack final loss and attack increase.

## Robustness Interpretation

The evidence supports a nuanced statement:

- By per-dataset winner count, `loss3_epoch1500` is much more broadly robust than `loss1_epoch8000` and `loss2_epoch2000`.
- By sample-weighted mean over all `10000` generalization samples, `loss2_epoch2000` is slightly better than `loss3_epoch1500`.
- Therefore, `loss3_epoch1500` has the stronger "wins on most generalization datasets" robustness profile, while `loss2_epoch2000` has the slightly lower aggregate mean damage on this first/master root.

This means the result supports `loss3` as more robust under a majority-of-datasets criterion, but not as a strict winner under every aggregation criterion.

## Auxiliary Clean-Loss Versus Attack-Damage Correlations

These are not the requested SVD spectral-norm correlations. They are recorded only as auxiliary evidence from the completed attack table.

| scope | n | x | y | Pearson | Spearman |
|---|---:|---|---|---:|---:|
| all rows / all models | `208` | clean initial loss | attack final loss | `0.982369` | `0.820054` |
| all rows / all models | `208` | clean initial loss | attack increase | `0.924343` | `0.813025` |
| generalization rows / all models | `200` | clean initial loss | attack final loss | `0.982409` | `0.820932` |
| generalization rows / all models | `200` | clean initial loss | attack increase | `0.924292` | `0.814273` |

Observed auxiliary conclusion: on the first/master attack table, clean loss and attack damage are strongly positively correlated. This does not answer the SVD question by itself.

## Requested SVD Correlation Status

Requested target:

- `||J_model - J_solver||_2` versus attack final loss;
- `||J_model - J_solver||_2` versus `attack_increase = final_loss - initial_loss`.

Current observed status:

- Local first/master SVD directory exists: `forensics/burgers_first_master_finalmodels_jacobian_svd_rep20_top100_20260608`
- Local visible files include manifest/preflight/partial sample evidence only.
- No completed local `jacobian_svd_summary.csv` was visible.
- No completed local error spectral norm CSV was visible.

Live R2 lookup evidence:

- R2 lookup record: `forensics/burgers_first_master_svd_r2_live_lookup_20260609`
- R2 result doc: `docs/burgers_first_master_svd_r2_live_lookup_20260609.md`
- Selected R2 `forensics/` top-level directory count: `127`
- SVD/Jacobian-related top-level directories found: `27`
- First/master-related top-level directories found: `0`
- Direct lookup of `forensics/burgers_first_master_finalmodels_jacobian_svd_rep20_top100_20260608/` returned no visible files.

Observed R2 conclusion: the selected R2 prefix contains completed SVD/Jacobian evidence for second/ns50, round01/aligned, and round03/selective families, but not for the true first/master semantic root `generalization_datasets/burgers`.

## Important Caveat

The existing `round01/aligned` SVD result cannot be silently substituted for first/master semantic root SVD.

Round01/aligned existing result:

- SVD artifact: `forensics/burgers_loss3_aligned_round01_final_jacobian_svd_rep20_top100_20260604`
- Attack/SVD correlation artifact: `forensics/burgers_round01_aligned_final_loss123_svd20_p2q2_attack_correlation_20260608`
- `||J_model - J_solver||_2` versus attack increase, all 20 samples x 4 models: Pearson `0.821269`, Spearman `0.764088`
- Generalization rows only: Pearson `0.805611`, Spearman `0.737148`

This supports the SVD/attack relationship on `round01/aligned`, but it is not the true first/master root.

## Final Status

Observed evidence supports:

- first/master full P2Q2 attack/tag is complete;
- the queued second run was stopped after first/master completion;
- `loss3_epoch1500` is the broad per-dataset robustness winner on first/master generalization;
- `loss2_epoch2000` is slightly better by sample-weighted mean attack damage;
- clean loss and attack damage are strongly positively correlated on this completed attack table;
- first/master SVD spectral-norm correlation cannot currently be computed because the required first/master SVD spectral-norm table is not available locally or in the selected R2 prefix.

Required next step for the originally requested SVD correlation:

- locate a first/master SVD artifact outside the selected R2 prefix, or
- explicitly run the expensive first/master SVD audit and then join it with the completed first/master attack table.
