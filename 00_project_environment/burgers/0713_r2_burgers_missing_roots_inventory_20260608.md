# R2 Burgers Missing-Roots Inventory (2026-06-08)

## Scope

The user asked to search the R2 bucket/prefix for missing Burgers robustness/SVD/generalization artifacts, especially the first and third generalization-root robustness results.

Remote prefix inspected:

- `neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected`

No credentials are recorded in this document.

## Listing Method

Observed inventory files written locally:

- `forensics/r2_burgers_missing_roots_inventory_20260608/r2_forensics_topdirs.txt`
- `forensics/r2_burgers_missing_roots_inventory_20260608/r2_adversarial_training_runs_topdirs.txt`
- `forensics/r2_burgers_missing_roots_inventory_20260608/r2_docs_topdirs.txt`
- `forensics/r2_burgers_missing_roots_inventory_20260608/r2_run_logs_topdirs.txt`
- `forensics/r2_burgers_missing_roots_inventory_20260608/r2_generalization_datasets_lsf_pst.tsv`
- `forensics/r2_burgers_missing_roots_inventory_20260608/r2_generalization_datasets_rmse_1p5_3x_all_ns50_lsf_pst.tsv`
- `forensics/r2_burgers_missing_roots_inventory_20260608/r2_generalization_datasets_burgers_loss3_selective_search_lsf_pst.tsv`
- `forensics/r2_burgers_missing_roots_inventory_20260608/r2_docs_lsf_pst.tsv`
- `forensics/r2_burgers_missing_roots_inventory_20260608/r2_run_logs_lsf_pst.tsv`
- `forensics/r2_burgers_missing_roots_inventory_20260608/r2_burgers_relevant_all_matches.txt`

Important caveat:

- Full recursive listing of the whole selected prefix was stopped because it entered `.git/objects`, which is not useful for experiment artifact discovery.
- Recursive listing of `forensics/` and `adversarial_training_runs/` was also stopped after enough evidence was obtained, because those directories contain large SVD/training artifact trees. Their top-level directory listings are complete and are the main evidence for whether a full-tag artifact directory exists.
- Recursive listing of the three generalization roots, `docs/`, and `run_logs/` completed.

## Generalization Dataset Roots On R2

Observed counts from the completed R2 listings:

| root | Burgers `.pt` count | evidence file |
|---|---:|---|
| first: `generalization_datasets/burgers` | 50 | `r2_generalization_datasets_lsf_pst.tsv` |
| second: `generalization_datasets_rmse_1p5_3x_all_ns50/burgers` | 50 | `r2_generalization_datasets_rmse_1p5_3x_all_ns50_lsf_pst.tsv` |
| third: `generalization_datasets_burgers_loss3_selective_search/round_03/burgers` | 50 | `r2_generalization_datasets_burgers_loss3_selective_search_lsf_pst.tsv` |

The third root also has a candidate pool:

- `round_03_candidate_pool/burgers`: 60 `.pt` files.

Observed first-root file names include semantic names such as:

- `burgers_far_centered_scale_shift_scale0p5_shift0.pt`
- `burgers_far_sawtooth_add_scale0p3_shift0.pt`
- `burgers_mid_matern_corr0p02_nu0p8.pt`
- `burgers_near_gaussian_corr0p015.pt`

This confirms that the first root itself is present on R2 and has the expected 50 Burgers datasets.

## Full Robustness / Tag Artifacts Found On R2

Observed R2 top-level `forensics/` directories include:

- `burgers_round03_full_p2q2_52datasets_4models_finalonly_20step_20260607/`
- `burgers_round03_full_p2q2_clean_loss_per_dataset_20260608/`
- `burgers_round03_full_p2q2_debug_10samples_1step_20260607/`
- `burgers_round03_baseline_vs_final_p2q2_attack_verification_20260607/`

Interpretation:

- The full 52-dataset/4-model/final-only/20-step artifact exists on R2, but this artifact is the previously identified second-root full-tag run, matching the local artifact `forensics/burgers_round03_full_p2q2_52datasets_4models_finalonly_20step_20260607`.
- No R2 top-level artifact was found with a first-root full-tag name such as `first_master`, `master`, or a first-root-specific `full_p2q2` name.
- No R2 top-level artifact was found with a third-root-specific full-tag name such as `round03_selective_full_p2q2`, `loss3_selective_round03_full_p2q2`, or a separate third-root `52datasets_4models_finalonly` name.

## SVD / Representative Attack Evidence Found On R2

Observed R2 top-level `forensics/` directories include:

- `burgers_loss3_aligned_round01_final_jacobian_svd_rep20_top100_20260604/`
- `burgers_loss3_selective_round03_final_jacobian_svd_gen5_top20_20260605/`
- `burgers_loss3_selective_round03_long_final_jacobian_svd_rep20_top100_20260605/`
- `burgers_loss3_selective_round03_loss123_final_extension_jacobian_svd_rep20_top100_20260606/`
- `burgers_loss3_selective_round03_per_dataset_loss3_advantage_20260605/`
- `burgers_p2q2_loss1_epoch2000_jacobian_svd_rep20_top100_20260603/`
- `burgers_p2q2_loss2_epoch900_jacobian_svd_rep20_top100_20260603/`

Interpretation:

- R2 does contain the third-root/round03 SVD and representative robustness-related evidence.
- This is not the same thing as a full 10200-sample attack/tag table over all 50 third-root datasets.

## Current Conclusion

Observed evidence from R2 supports this corrected status:

1. The three Burgers generalization dataset roots themselves are present on R2, each with 50 Burgers `.pt` datasets.
2. R2 contains the completed second-root full P2Q2 final-only attack/tag artifact.
3. R2 contains third-root SVD/representative attack/verification artifacts.
4. R2 does not show a separate first-root full P2Q2 final-only artifact in the inspected top-level artifact directories.
5. R2 does not show a separate third-root full P2Q2 final-only artifact in the inspected top-level artifact directories.

Therefore, the local补跑 launched in `tools/run_burgers_missing_roots_full_p2q2_attack_20260608.sh` is still needed to produce same-strength first-root and third-root robustness/tag evidence.


## Correction: R2 SVD Metadata Provenance Audit

After the initial top-level inventory, all R2 `forensics/` top-level directories containing `svd` or `jacobian` were enumerated. There are 27 such SVD/Jacobian-related top-level artifact directories in the selected R2 prefix.

To avoid downloading large `.npz` SVD matrices, only top-level `.json`, `.csv`, and `.md` metadata files were copied locally under:

- `forensics/r2_burgers_missing_roots_inventory_20260608/r2_svd_top_metadata/`

Derived provenance summaries:

- `forensics/r2_burgers_missing_roots_inventory_20260608/r2_svd_manifest_provenance_summary.csv`
- `forensics/r2_burgers_missing_roots_inventory_20260608/r2_svd_manifest_provenance_summary.md`

Observed from the metadata:

- R2 does contain multiple Burgers Jacobian/SVD artifact families, not just one.
- Second-root ns50 SVD evidence is present. Example artifacts include:
  - `burgers_adv_training_jacobian_svd_20260531_representative20_same_points`
  - `burgers_p2q2_checkpoint_series_jacobian_svd_20260601`
  - `burgers_p2q2_loss1_epoch2000_jacobian_svd_rep20_top100_20260603`
  - `burgers_p2q2_loss2_epoch900_jacobian_svd_rep20_top100_20260603`
- Those second-root manifests include rows categorized as `second_ns50` plus original train/test samples, commonly `20` representative rows with about `10` generalization samples and `10` train/test samples.
- Round01/aligned SVD evidence is present. Example artifact:
  - `burgers_loss3_aligned_round01_final_jacobian_svd_rep20_top100_20260604`
- Third-root/round03 selective SVD evidence is present. Example artifacts include:
  - `burgers_loss3_selective_round03_final_jacobian_svd_gen5_top20_20260605`
  - `burgers_loss3_selective_round03_long_final_jacobian_svd_rep20_top100_20260605`
  - `burgers_loss3_selective_round03_loss123_final_extension_jacobian_svd_rep20_top100_20260606`
- The third-root manifests explicitly point to `generalization_datasets_burgers_loss3_selective_search/round_03` and include representative generalization rows.

Observed absence from the downloaded R2 SVD metadata:

- Exact string search did not find `generalization_datasets/burgers` inside the downloaded R2 SVD metadata files.
- Therefore this audit still does not evidence a completed first-root/master-semantic SVD artifact on R2, unless it exists under a different prefix not covered by the selected R2 path or a manifest that does not record the root path.

Corrected interpretation:

- It is wrong to say that the earlier SVD work was not done for the other roots. R2 clearly contains completed SVD metadata for the second ns50 root, round01/aligned root, and third round03 selective root.
- The only still-unevidenced SVD item in this R2-selected-prefix audit is the first/master semantic root `generalization_datasets/burgers`.
- This SVD question is separate from the missing full attack/tag tables: the currently running background job is still full P2Q2 attack/tag, not SVD.
