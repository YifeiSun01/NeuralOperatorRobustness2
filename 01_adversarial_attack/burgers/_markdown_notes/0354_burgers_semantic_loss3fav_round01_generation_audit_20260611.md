# Burgers Semantic Loss3-Favored Generalization Round 01 Audit (2026-06-11)

## Purpose

Generate a new Burgers generalization set that keeps the first/second dataset philosophy: semantic OOD initial-condition families, true Burgers solver labels, no attack-generated samples.  The selected set should favor the final `loss3_epoch1500` adversarial-training model on clean inference compared with `loss1_epoch8000` and `loss2_epoch2000`.

## Prior Dataset Audit

I checked both the local GitHub repository and the R2 mirror for the three earlier Burgers generalization roots.

- First root: `generalization_datasets/burgers`
  - R2 had 50 Burgers `.pt` files.
  - The names and generator path show semantic OOD construction: Gaussian/Matern variants, correlation-length shifts, range shifts, sawtooth-like transforms, sign/centered transforms.
  - This is a real generalization dataset root.

- Second root: `generalization_datasets_rmse_1p5_3x_all_ns50/burgers`
  - R2 had 50 Burgers `.pt` files.
  - This root was built from semantic candidates and target-band filtering, not from adversarial attack outputs.
  - This is also a real generalization dataset root.

- Third root: `generalization_datasets_burgers_loss3_selective_search/round_03/burgers`
  - R2 had 50 selected files plus `selected_candidate_scores.csv`.
  - GitHub records and `tools/generate_burgers_loss3_selective_generalization.py` show the candidate pool was made by sampling train/test examples and running a loss3 attack, then treating attacked samples as generalization samples.
  - The selected root collapsed to a small number of repeated source/attack configurations.  This matches the concern that it was not a true 50-way semantic generalization set.

Related local records include:

- `docs/burgers_round03_attack_generated_generalization_validity_caveat_20260608.md`
- `docs/burgers_round03_generalization_5x5_wallclock_panels_20260608.md`
- `docs/r2_burgers_three_dataset_conclusion_audit_20260608.md`
- `docs/r2_burgers_generalization_data_lookup_20260608.md`

## New Generator

Script:

```bash
/venv/adv_robust/bin/python tools/generate_burgers_semantic_loss3_favored_generalization_20260611.py   --round-id 1   --max-candidates 720   --accept-ratio 1.0   --overwrite-candidates
```

Selected root:

```text
generalization_datasets_burgers_semantic_loss3fav_search_20260611/round_01/burgers
```

Candidate pool:

```text
generalization_datasets_burgers_semantic_loss3fav_search_20260611/round_01_candidate_pool/burgers
```

The generator constructs semantic initial conditions, then labels them with the true Burgers solver through `make_burgers_rollout_fn`.  It does not use adversarial attacks to create samples.  The model scores are used only after generation, to select candidates whose clean inference metrics favor loss3.

Semantic candidate families included:

- Gaussian GRF with multiple correlation lengths.
- Matern GRF with multiple correlation lengths and `nu` values.
- Power-law Fourier fields with different spectral decay `alpha` and radius `k0`.
- Low/mid/high/ultra sine mixtures.
- Piecewise-linear fields with different knot counts and roughness.
- Sawtooth waves with different frequencies.
- Square waves with different frequencies and duty cycles.
- Spike-train fields with different spike counts and widths.

Range and transform variations included `[-1,1]`, `[-3,3]`, `[0,2]`, `[-2,2]`, `[-1,0]`, `[0,3]`, direct range remaps, triangular additions, sign quantization, high-frequency sine additions, sawtooth additions, and spike emphasis.

## Round 01 Result

- Candidate datasets generated: 720.
- Candidate pool size: about 1.2 GB.
- Selected datasets: 50.
- Selected root size: about 79 MB.
- Samples per selected dataset: 200.
- Parameter uniqueness: 50 unique parameter keys out of 50 selected datasets.
- Duplicate parameter keys: 0.
- Loss3 clean-inference wins: 50/50 selected datasets.

For every selected dataset, `loss3_epoch1500` has lower RMSE and lower relative L2 than both `loss1_epoch8000` and `loss2_epoch2000`.

Loss3 RMSE ratio against the better of loss1/loss2:

- Minimum ratio: 0.2487.
- Median ratio: 0.6854.
- Maximum ratio: 0.7719.
- Mean ratio: 0.6379.

Equivalently, loss3 improves over the better of loss1/loss2 by:

- Best case: about 75.1% lower RMSE.
- Median case: about 31.5% lower RMSE.
- Worst selected case: about 22.8% lower RMSE.

Selected family counts:

- Sawtooth: 27.
- Square wave: 6.
- Spike train: 6.
- Power-law Fourier: 6.
- Matern: 4.
- Sine mixture: 1.

Selected transform counts:

- Triangle addition: 27.
- Direct range affine: 10.
- Sign quantization: 9.
- High-frequency sine addition: 4.

Selected range counts:

- `[-1, 1]`: 18.
- `[0, 2]`: 15.
- `[-3, 3]`: 11.
- `[-2, 2]`: 6.

## Output Files

- `generalization_datasets_burgers_semantic_loss3fav_search_20260611/round_01/manifest.json`
- `generalization_datasets_burgers_semantic_loss3fav_search_20260611/round_01/selected_candidate_scores.csv`
- `generalization_datasets_burgers_semantic_loss3fav_search_20260611/round_01/summary.json`
- `generalization_datasets_burgers_semantic_loss3fav_search_20260611/round_01/uniqueness_report.json`
- `generalization_datasets_burgers_semantic_loss3fav_search_20260611/round_01_candidate_pool/candidate_model_scores.csv`
- `generalization_datasets_burgers_semantic_loss3fav_search_20260611/round_01_candidate_pool/candidate_spec_manifest.json`

## Notes

This round satisfies the first requirement: build a non-attack, true semantic Burgers generalization root where the selected 50 datasets have unique parameters and loss3 is clean-inference-best on all selected datasets.  Robustness/P2Q2 growth and Jacobian/SVD spectral-norm checks have not been run yet for this new root.
