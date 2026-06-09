# Burgers Second Dataset Attack/SVD Correlation Clarification - 2026-06-08

Status: inspected local records and the R2 selected prefix for the question of whether the second Burgers generalization dataset already has an observed high correlation between Jacobian/SVD spectral norm and adversarial attack loss growth. No new experiment was run.

## Question

Does the second dataset, `generalization_datasets_rmse_1p5_3x_all_ns50/burgers`, already have both:

- a full adversarial attack/tag evaluation, and
- a Jacobian/SVD spectral-norm analysis,

with a clear conclusion that larger spectral norm generally implies larger attack-after loss growth?

## Observed Evidence

Observed full attack/tag evidence for the second dataset:

- Source artifact: `forensics/burgers_round03_full_p2q2_52datasets_4models_finalonly_20step_20260607`
- Source/derived docs:
  - `docs/burgers_round03_full52_per_sample_clean_vs_attack_mismatch_20260608.md`
  - `docs/burgers_round03_full52_clean_generalization_vs_attack_robustness_mismatch_20260608.md`
- The `10000` generalization samples in that artifact use `generalization_datasets_rmse_1p5_3x_all_ns50/burgers`.
- On those samples, clean winners loss1/loss2/loss3 are `9195 / 770 / 35`; final-attack winners are `729 / 355 / 8916`; attack-increase winners are `683 / 325 / 8992`.

Observed correlation already computed for that full attack/tag artifact:

- File: `forensics/burgers_round03_full52_clean_vs_attack_mismatch_20260608/clean_attack_correlation_summary.csv`
- Trained generalization-50 dataset rows: Pearson clean-vs-final-attack `-0.549044`; Pearson clean-vs-attack-increase `-0.556730`.
- This is a clean-loss versus attack-loss/growth correlation, not a Jacobian/SVD spectral-norm correlation.

Observed Jacobian/SVD evidence:

- Round03 final-extension SVD source: `forensics/burgers_loss3_selective_round03_loss123_final_extension_jacobian_svd_rep20_top100_20260606`
- Earlier round03 final SVD source: `forensics/burgers_loss3_selective_round03_long_final_jacobian_svd_rep20_top100_20260605`
- Their sample manifests use third-root generalization paths such as `generalization_datasets_burgers_loss3_selective_search/round_03/burgers/burgers_loss3_selective_r03_d24.pt`.
- Therefore, those Jacobian/SVD generalization rows are from the third loss3-selective stress root, not the second ns50 root.

Observed R2 search:

- R2 file-name search under `forensics` found `burgers_p2q2_loss123_50step_input_similarity_probe_20260604/correlation_summary.csv` and `burgers_round03_loss123_svd_vector_frequency_probe_20260608/frequency_proxy_correlations.csv`.
- The first is perturbation geometry versus generalization RMSE, not Jacobian spectral norm versus attack growth.
- The second is SVD vector frequency/projection residual versus downsample sigma proxy error, not attack growth.
- No file name matching a second-root Jacobian/SVD spectral-norm versus attack-loss-growth correlation table was found in the inspected selected R2 prefix.

## Conclusion

Observed evidence does **not** support saying that, on the second dataset, we already clearly observed a high correlation where larger Jacobian/SVD spectral norm implies larger attack-after loss growth.

What is supported:

- The second dataset has a full P2Q2 attack/tag result.
- The second dataset shows the clean/robustness mismatch very strongly: loss3 is usually worse clean but usually best after attack.
- Separate Jacobian/SVD results exist, but their generalization rows are for the third round03 stress root, not the second ns50 root.
- Existing correlation files measure different relationships: clean-vs-attack correlation, perturbation-geometry-vs-RMSE correlation, or SVD-frequency-proxy-vs-downsample-error correlation.

Inference:

- A direct claim linking second-root attack growth to Jacobian/SVD spectral norm still requires either locating a missing artifact outside the inspected paths or running a joined analysis on matched samples.

