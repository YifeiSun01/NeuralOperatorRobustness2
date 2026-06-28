# Burgers Full-Tag Generalization Source Provenance - 2026-06-08

## Question

Audit whether the Burgers full-tag/per-sample clean-vs-attack mismatch result used a real semantic generalization dataset root, or whether it used train/test samples that had been tagged/attacked and then treated as generalization.

## Short Answer

Observed evidence says the full-tag result used the local semantic/non-attack generalization root:

`generalization_datasets_rmse_1p5_3x_all_ns50/burgers`

The local audit/docs call this root `target_band_ns50_current`. If the user's "Run2" means this root, then yes: the full-tag result used that root. The local records do not literally name this root `Run2`, so if `Run2` refers to a different R2-only root, that separate root still needs to be recovered and audited.

This root is not the problematic round03 attack-generated root:

`generalization_datasets_burgers_loss3_selective_search/round_03/burgers`

## Source Evidence

Observed from the full-tag manifest:

- Source manifest: `forensics/burgers_round03_full_p2q2_52datasets_4models_finalonly_20step_20260607/manifest.json`
- Generalization samples in that manifest: `10000`
- Unique generalization datasets: `50`
- Every generalization `source_path` points under `generalization_datasets_rmse_1p5_3x_all_ns50/burgers`.

Derived provenance audit written on 2026-06-08:

- `forensics/burgers_fulltag_generalization_source_provenance_20260608/summary.json`
- `forensics/burgers_fulltag_generalization_source_provenance_20260608/target_band_ns50_metadata_audit.csv`

Observed from `summary.json`:

- `full_tag_generalization_sample_count`: `10000`
- `full_tag_generalization_unique_dataset_count`: `50`
- `root_file_count`: `50`
- `all_generalization_paths_under_target_band_ns50`: `true`
- `attack_like_metadata_key_hits`: `[]`

Observed tier counts across the 50 `.pt` files:

- `near_param_shift`: `8`
- `mid_kernel_spectrum`: `2`
- `far_range_pattern`: `2`
- `target_loss_param_shift`: `23`
- `target_loss_kernel_shift`: `15`

## Metadata Spot Check

Observed from selected files under `generalization_datasets_rmse_1p5_3x_all_ns50/burgers`:

- Payload keys: `x`, `y`, `metadata`
- Metadata keys include semantic/generation fields such as `dataset_id`, `family`, `params`, `sample_metadata`, `seed`, `similarity_tier`, `solver`, `training_reference`, `nu`, `nx`, and `t_final`.
- Tensor shapes are `x: (200, 1024)` and `y: (200, 1024)`.
- Example dataset IDs include `burgers_far_sawtooth_add_scale0p3_shift0`, `burgers_target_gaussian_corr0p2`, and `burgers_target_matern_corr0p6_nu3`.

Observed contrast with the problematic round03 stress root:

- A sample from `generalization_datasets_burgers_loss3_selective_search/round_03/burgers` contains metadata keys such as `attack_geometry`, `selection`, and `record`.
- The semantic/non-attack root audited here does not contain those attack-generated metadata fields.

## Interpretation

Observed evidence:

- The per-sample full-tag result's 10000 generalization samples are sourced from `generalization_datasets_rmse_1p5_3x_all_ns50/burgers`.
- The 50 dataset payloads in that root have semantic generation metadata and no attack-like metadata fields.
- This is distinct from the round03 loss3-selective attack-generated stress root.

Inference:

- The clean-vs-adversarial-tag mismatch result reported from the local full-tag artifact is based on a non-attack semantic generalization root, not on train/test samples that were attacked and relabeled as generalization.
- Caveat: `target_band_ns50_current` is still curated/target-band semantic data, not an unbiased random sample from all possible Burgers distributions. It is valid to call it non-attack semantic generalization, but not uniformly random external generalization.

Remaining work:

- If the user's `Run2` refers to a different uploaded root, restore/list R2 and repeat this exact provenance audit on that root before claiming equivalence.

## Round 02 Clarification

Observed on 2026-06-08:

- The full-tag script `tools/run_burgers_round03_full_p2q2_finalonly_attack.py` sets `GEN_ROOT = generalization_datasets_rmse_1p5_3x_all_ns50/burgers`.
- The resulting full-tag manifest `forensics/burgers_round03_full_p2q2_52datasets_4models_finalonly_20step_20260607/manifest.json` has all `10000` generalization samples under that same root.
- The local directory `generalization_datasets_burgers_loss3_aligned_search/round_02/burgers` exists, but it was not used in the `10200`-sample full-tag/per-sample mismatch audit.
- The local directory `generalization_datasets_burgers_loss3_selective_search` contains `round_03` locally, not a `round_02` subdirectory.

Therefore, if "round 2" means `generalization_datasets_burgers_loss3_aligned_search/round_02/burgers`, then the answer is no: that was not the dataset root used for the reported `9195/770/35`, `729/355/8916`, and `683/325/8992` generalization-sample winner counts. Those counts came from the neutral/semantic target-band root `generalization_datasets_rmse_1p5_3x_all_ns50/burgers`, evaluated with the round03-trained checkpoints.
