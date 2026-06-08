# Burgers Jacobian Norm Versus PGD Loss Correlation

Date: 2026-06-08

## Question

Check whether saved Burgers model-solver error Jacobian norms correlate with saved
PGD attack loss growth.

## Source Data

Observed from:

- Jacobian/SVD source:
  `forensics/burgers_loss3_selective_round03_loss123_final_extension_jacobian_svd_rep20_top100_20260606/round03_loss123_final_extension_jacobian_svd_summary.csv`
- PGD source:
  `forensics/burgers_round03_full_p2q2_52datasets_4models_finalonly_20step_20260607/summary_by_model_dataset.csv`
- Analysis script:
  `tools/analyze_burgers_jacobian_pgd_correlation.py`

Generated outputs:

- `analysis_outputs/burgers_jacobian_pgd_correlation_20260608/model_split_join.csv`
- `analysis_outputs/burgers_jacobian_pgd_correlation_20260608/correlations.csv`
- `analysis_outputs/burgers_jacobian_pgd_correlation_20260608/model_split_sigma_vs_attack_mse.svg`
- `analysis_outputs/burgers_jacobian_pgd_correlation_20260608/model_split_sigma_ratio_vs_attack_mse_ratio.svg`

## Alignment Limits

Observed:

- The SVD artifact has model-solver error spectral norms for `baseline`,
  `loss1_epoch5000`, `loss2_epoch2000`, and `loss3_epoch1500` on 20 fixed
  samples: 5 train, 5 test, and 10 round03 selective generated samples.
- The 52-dataset PGD table uses `baseline`, `loss1_epoch8000`,
  `loss2_epoch2000`, and `loss3_epoch1500`.
- No local `loss1_epoch8000` Jacobian/SVD artifact is available; this is also
  noted in `docs/20260608_burgers_downsample_svd_1024_512_256_summary.md`.
- The SVD generalization samples are `burgers_loss3_selective_r03_dXX`, while
  the 52-dataset PGD table uses the RMSE/generalization sweep dataset names.
  Therefore there is no strict per-dataset or per-sample join for those
  generalization rows.

Inference:

- The cleanest available correlation from existing files is a `model x split`
  aggregate join, not a strict pointwise \(a\)-by-\(a\) correlation.
- Exact-checkpoint correlations exclude loss1. A separate "with loss1 proxy"
  version includes `loss1_epoch5000` SVD against `loss1_epoch8000` PGD and is
  explicitly marked as a proxy.

## Correlation Results

Observed from `analysis_outputs/burgers_jacobian_pgd_correlation_20260608/correlations.csv`:

| analysis set | x | y | n | Pearson | Spearman | partial Pearson controlling clean MSE |
|---|---|---|---:|---:|---:|---:|
| with loss1 proxy | mean error-Jacobian spectral norm | final PGD MSE | 12 | 0.8113 | 0.7203 | 0.5003 |
| with loss1 proxy | mean error-Jacobian spectral norm | final PGD RMS | 12 | 0.8229 | 0.7203 | 0.5013 |
| exact checkpoints only | mean error-Jacobian spectral norm | final PGD MSE | 9 | 0.8392 | 0.7167 | 0.4332 |
| exact checkpoints only | mean error-Jacobian spectral norm | final PGD RMS | 9 | 0.8317 | 0.7167 | 0.3600 |

Baseline-normalized ratios:

| analysis set | x | y | n | Pearson | Spearman |
|---|---|---|---:|---:|---:|
| with loss1 proxy | spectral-norm ratio to baseline | final PGD MSE ratio to baseline | 12 | 0.7059 | 0.1690 |
| with loss1 proxy | spectral-norm ratio to baseline | final PGD RMS ratio to baseline | 12 | 0.6482 | 0.2183 |
| exact checkpoints only | spectral-norm ratio to baseline | final PGD MSE ratio to baseline | 9 | 0.6895 | 0.2543 |
| exact checkpoints only | spectral-norm ratio to baseline | final PGD RMS ratio to baseline | 9 | 0.6493 | 0.3390 |

## Interpretation

Observed evidence supports a positive aggregate relationship: larger saved
model-solver error spectral norms tend to align with larger saved finite-radius
PGD final loss at the `model x split` level.

Inference:

- The raw aggregate relation is fairly strong: Pearson about `0.81-0.84`.
- After controlling for clean/initial MSE, the relation remains positive but is
  weaker: partial Pearson about `0.36-0.50`.
- Baseline-normalized Pearson correlations are moderate positive, about
  `0.65-0.71`, but rank correlations are weak. This means the norm is a useful
  robustness proxy, but it does not fully determine finite-radius PGD ordering.
- This is consistent with the theory: \( \|D\mathcal E(a)\| \) controls
  infinitesimal error-field amplification, while the saved PGD metric is a
  finite-radius endpoint loss affected by residual terms, nonlinear path
  rotation, dataset difficulty, and PGD optimization details.

## Bottom Line

From the currently visible saved Burgers artifacts, the answer is:

**Yes, there is a clear positive aggregate correlation between local
error-Jacobian norm and PGD loss growth, but the available files do not prove a
strict pointwise correlation. The relation becomes less clean after baseline
normalization or clean-loss control, which is exactly the caveat predicted by
the local-theory versus finite-radius-PGD distinction.**

## Remaining Work

To answer the pointwise version cleanly, run one of:

- compute `loss1_epoch8000` SVD on the same 20-sample SVD manifest, then rerun
  this join;
- run the same 20-step PGD attack on the exact 20 SVD manifest samples and save
  per-sample initial/final loss;
- compute representative Jacobian/SVD probes for the same 52 datasets used by
  the full p2q2 final-only attack table.

No new GPU experiment was run for this analysis.
