# Burgers Round01 Aligned SVD And Attack Correlation 20260609

## Question

The question is whether the existing `round01/aligned` Jacobian/SVD result can be paired with adversarial attack loss growth, and whether the attack damage is correlated with the error-Jacobian spectral norm.

## Scope

This record concerns the `round01/aligned` generated Burgers dataset family, using the existing SVD artifact:

- SVD artifact: `forensics/burgers_loss3_aligned_round01_final_jacobian_svd_rep20_top100_20260604`
- SVD summary: `forensics/burgers_loss3_aligned_round01_final_jacobian_svd_rep20_top100_20260604/round01_jacobian_svd_summary.csv`
- SVD sample manifest: `forensics/burgers_loss3_aligned_round01_final_jacobian_svd_rep20_top100_20260604/round01_sample_manifest.csv`

Important distinction: this is `round01/aligned`. It is not currently evidenced as the same thing as the master semantic root `generalization_datasets/burgers`.

## Existing Attack Result

The corresponding P2Q2 adversarial attack/correlation artifact already exists locally:

- Attack/correlation directory: `forensics/burgers_round01_aligned_final_loss123_svd20_p2q2_attack_correlation_20260608`
- Attack rows: `forensics/burgers_round01_aligned_final_loss123_svd20_p2q2_attack_correlation_20260608/attack_loss_by_model_sample.csv`
- Joined SVD/attack rows: `forensics/burgers_round01_aligned_final_loss123_svd20_p2q2_attack_correlation_20260608/error_svd_attack_joined_rows.csv`
- Correlation summary: `forensics/burgers_round01_aligned_final_loss123_svd20_p2q2_attack_correlation_20260608/error_svd_attack_correlation_summary.csv`
- Config: `forensics/burgers_round01_aligned_final_loss123_svd20_p2q2_attack_correlation_20260608/config.json`
- Summary: `forensics/burgers_round01_aligned_final_loss123_svd20_p2q2_attack_correlation_20260608/summary.json`

Attack setting observed from the existing summary/config:

- Method: P2Q2 adversarial attack
- Steps: `20`
- `epsilon_rms`: `0.12`
- `alpha_rms`: `0.012`
- Batch size: `20`
- Sample count: `20` SVD sample points
- Models: `baseline`, `loss1_epoch1000`, `loss2_epoch500`, `loss3_epoch500`

GPU evidence recorded in the existing summary:

- PyTorch: `2.8.0+cu126`
- CUDA: `12.6`
- Device: Tesla V100-SXM2-32GB
- Compute capability: `[7, 0]`
- PyTorch arch list includes `sm_70`

## Metrics

Predictor:

- `error_spectral_norm = ||J_model - J_solver||_2`

Main response:

- `attack_increase = final_loss - initial_loss`

Also recorded:

- `final_loss`
- `final_diff_rms`
- `attack_increase_rms`

## Observed Correlations

From `forensics/burgers_round01_aligned_final_loss123_svd20_p2q2_attack_correlation_20260608/error_svd_attack_correlation_summary.csv`:

| scope | n | x | y | Pearson | Spearman |
|---|---:|---|---|---:|---:|
| all 20 samples x all 4 models | 80 | `error_spectral_norm` | `attack_increase` | `0.821269` | `0.764088` |
| all 20 samples x all 4 models | 80 | `error_spectral_norm` | `final_loss` | `0.843871` | `0.771730` |
| all 20 samples x trained 3 models | 60 | `error_spectral_norm` | `attack_increase` | `0.669044` | `0.699972` |
| generalization rows | 40 | `error_spectral_norm` | `attack_increase` | `0.805611` | `0.737148` |
| generalization rows | 40 | `error_spectral_norm` | `final_loss` | `0.834263` | `0.741839` |

Observed mean attack losses from `summary.json`:

| model | initial loss mean | final loss mean | attack increase mean |
|---|---:|---:|---:|
| baseline | `0.001070643` | `0.013917414` | `0.012846773` |
| loss1_epoch1000 | `0.000036162` | `0.004194695` | `0.004158533` |
| loss2_epoch500 | `0.000068821` | `0.004770340` | `0.004701519` |
| loss3_epoch500 | `0.000078280` | `0.003168333` | `0.003090054` |

## Conclusion

Observed evidence supports a strong positive relationship on `round01/aligned`: larger error-Jacobian spectral norm is associated with larger finite-budget P2Q2 attack damage.

The cleanest statement for paper notes is:

> On the `round01/aligned` Burgers SVD audit, the local error-Jacobian spectral norm `||J_model - J_solver||_2` is strongly positively correlated with P2Q2 endpoint attack growth `final_loss - initial_loss` across the 20 sampled points and four model variants.

## Caveats

- This is a 20-sample SVD audit, not a full-dataset SVD.
- This is `round01/aligned`, not proven to be the master semantic root `generalization_datasets/burgers`.
- The metric `attack_increase = final_loss - initial_loss` is endpoint attack damage. It is closely related to, but not identical to, the theorem-side residual movement quantity `||E(x + delta) - E(x)||`.
- The true master root `generalization_datasets/burgers` still needs a completed SVD artifact before making the same SVD-vs-attack correlation claim for that root.

## Remaining Work

- If the user wants the true master semantic first-root comparison, locate or complete a finished SVD summary for `generalization_datasets/burgers`.
- If no such SVD artifact exists, reuse the attack script only after the SVD sample manifest is available; do not recompute SVD unless explicitly requested because SVD is expensive.
