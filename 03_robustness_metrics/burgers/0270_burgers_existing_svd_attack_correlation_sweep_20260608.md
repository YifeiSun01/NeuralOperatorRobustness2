# Burgers Existing SVD Attack Correlation Sweep 20260608

## Question

Check whether existing Burgers Jacobian/SVD spectral norms correlate with fixed-budget P2Q2 adversarial attack damage, without recomputing any SVD/Jacobian artifacts.

## Scope And Definitions

- Observed SVD source: existing `*_jacobian_svd_summary.csv` and sample manifests under `forensics/`.
- Attack setting: P2Q2, `epsilon_rms=0.12`, `alpha_rms=0.012`, `20` steps, `batch_size=20`.
- Main reported response variable: `attack_increase = final_loss - initial_loss`.
- Also recorded: raw `final_loss`, `attack_increase_rms`, and `final_diff_rms`.
- Main predictor: `error_spectral_norm = ||J_model - J_solver||_2`.
- No new SVD/Jacobian computation was run for this sweep.

## Important Provenance

- The earlier completed SVD20 result `forensics/burgers_svd20_p2q2_attack_correlation_20260608` uses the third round03 stress root, not the first master root.
- Existing local and selected R2 SVD summaries were found for the second ns50 root, round01 aligned generated root, and third round03 stress root.
- A true first master root SVD summary for `generalization_datasets/burgers` was not found locally or in the selected R2 prefix. The local `forensics/burgers_first_master_finalmodels_jacobian_svd_rep20_top100_20260608` directory is an incomplete diagnostic manifest/preflight only and must not be interpreted as completed SVD evidence.

## Key Correlations

| job | dataset/root class | models | scope | n | Pearson attack increase | Spearman attack increase | Pearson final loss | Spearman final loss |
|---|---|---|---|---:|---:|---:|---:|---:|
| `run2_ns50_loss3_checkpoint_series` | second ns50 | loss3 checkpoint series, baseline + epoch0200/0400/0600/0800/1000 | all | 120 | 0.515865 | 0.654920 | 0.531880 | 0.670755 |
| `run2_ns50_loss3_checkpoint_series` | second ns50 | loss3 checkpoint series, baseline + epoch0200/0400/0600/0800/1000 | generalization | 60 | 0.405998 | 0.479911 | 0.424426 | 0.507919 |
| `run2_ns50_loss1_epoch2000` | second ns50 | loss1 epoch2000 + baseline | all | 40 | 0.843321 | 0.834522 | 0.846915 | 0.842026 |
| `run2_ns50_loss1_epoch2000` | second ns50 | loss1 epoch2000 + baseline | generalization | 20 | 0.827078 | 0.783459 | 0.831095 | 0.800000 |
| `run2_ns50_loss2_epoch0900` | second ns50 | loss2 epoch0900 + baseline | all | 40 | 0.833547 | 0.774859 | 0.837543 | 0.781238 |
| `run2_ns50_loss2_epoch0900` | second ns50 | loss2 epoch0900 + baseline | generalization | 20 | 0.816659 | 0.780451 | 0.821358 | 0.795489 |
| `round01_aligned_final_loss123` | round01 aligned generated | baseline + loss1/loss2/loss3 round01 final | all | 80 | 0.821269 | 0.764088 | 0.843871 | 0.771730 |
| `round01_aligned_final_loss123` | round01 aligned generated | baseline + loss1/loss2/loss3 round01 final | generalization | 40 | 0.805611 | 0.737148 | 0.834263 | 0.741839 |
| `round03_long_final_loss123` | third round03 stress | baseline + round03 long-final loss1/loss2/loss3 | all | 80 | 0.850377 | 0.824473 | 0.849561 | 0.834646 |
| `round03_long_final_loss123` | third round03 stress | baseline + round03 long-final loss1/loss2/loss3 | generalization | 40 | 0.806629 | 0.802627 | 0.804663 | 0.827580 |
| `round03_final_extension_loss123` | third round03 stress | already-completed final-extension loss1/loss2/loss3 | all | 80 | 0.832552 | 0.807712 | 0.848999 | 0.815331 |
| `round03_final_extension_loss123` | third round03 stress | already-completed final-extension loss1/loss2/loss3 | generalization | 40 | 0.768775 | 0.781051 | 0.798171 | 0.804503 |

## Observed Evidence

- Key summary CSV: `forensics/burgers_existing_svd_attack_correlation_key_summary_20260608.csv`.
- Per-job outputs contain `attack_loss_by_model_sample.csv`, `error_svd_attack_joined_rows.csv`, `error_svd_attack_correlation_summary.csv`, `config.json`, and `manifest.json`.
- Attack-only launcher: `tools/run_burgers_existing_svd_attack_correlations_20260608.sh`.
- Join refresh launcher: `tools/refresh_burgers_existing_svd_attack_correlation_joins_20260608.sh`.
- Parameterized attack/join script: `tools/run_burgers_svd20_p2q2_attack_correlation_20260608.py`.

## Inference

The existing-SVD evidence supports a positive relationship between local model-minus-solver Jacobian spectral norm and finite-radius P2Q2 attack damage. The relationship is strongest for the loss1/loss2 ns50 SVD slices and for round01/round03 loss123 final-model SVDs. The second-root loss3 checkpoint-series sweep is still positive, but weaker when many epochs are pooled together.

## Caveats

- These are 20-point SVD sample audits, not full-dataset SVD over all samples.
- The dataset/root class matters: second ns50, round01 aligned, and third round03 stress are not interchangeable.
- The first master semantic root was not evaluated here because completed first-root SVD summaries were not evidenced locally or in the selected R2 prefix.
