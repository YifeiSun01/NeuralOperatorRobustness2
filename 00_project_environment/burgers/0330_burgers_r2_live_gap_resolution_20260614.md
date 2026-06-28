# Burgers R2 Live Gap Resolution, 20260614

Generated: 2026-06-14T11:14:54+00:00

This report records the live R2 selected-prefix search used to resolve the Burgers metric coverage questions. Credentials were used only as process environment variables for the listing commands; no token or secret value is written in this repository artifact.

## R2 Scope

- Selected prefix: `neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected`
- Local listing cache: `outputs/burgers_timematched_solver7860_clean8000_audit_20260614/data/r2_live_gap_search_20260614/`
- High-priority recursive listing manifest: `r2_high_priority_listing_manifest.tsv`

## High-Priority Recursive Listings

| r2_dir | object_count | bytes | mb |
| --- | --- | --- | --- |
| outputs/burgers_timematched_solver7860_clean8000_audit_20260614/data/r2_live_gap_search_20260614/forensics_burgers_first_master_finalmodels_jacobian_svd_rep20_top100_20260608_recursive_lsf_pst.tsv | 6 | 11827 | 0.012 |
| outputs/burgers_timematched_solver7860_clean8000_audit_20260614/data/r2_live_gap_search_20260614/forensics_burgers_first_master_full_p2q2_52datasets_4models_finalonly_20step_20260608_recursive_lsf_pst.tsv | 21 | 160566008 | 160.566 |
| outputs/burgers_timematched_solver7860_clean8000_audit_20260614/data/r2_live_gap_search_20260614/forensics_burgers_random_field_selected_worktime_full_suite_20260613_recursive_lsf_pst.tsv | 313 | 594670773 | 594.671 |
| outputs/burgers_timematched_solver7860_clean8000_audit_20260614/data/r2_live_gap_search_20260614/forensics_burgers_random_solver7860_clean8000_full_suite_20260614_recursive_lsf_pst.tsv | 313 | 594738874 | 594.739 |
| outputs/burgers_timematched_solver7860_clean8000_audit_20260614/data/r2_live_gap_search_20260614/forensics_burgers_round03_selective_full_p2q2_52datasets_4models_finalonly_20step_20260608_recursive_lsf_pst.tsv | 7 | 11558698 | 11.559 |
| outputs/burgers_timematched_solver7860_clean8000_audit_20260614/data/r2_live_gap_search_20260614/forensics_burgers_six_model_latest_wideparam_summary_20260613_recursive_lsf_pst.tsv | 42 | 1299145 | 1.299 |
| outputs/burgers_timematched_solver7860_clean8000_audit_20260614/data/r2_live_gap_search_20260614/forensics_burgers_six_model_selected_worktime_summary_20260613_recursive_lsf_pst.tsv | 18 | 1295344 | 1.295 |
| outputs/burgers_timematched_solver7860_clean8000_audit_20260614/data/r2_live_gap_search_20260614/forensics_burgers_six_model_solver7860_clean8000_summary_20260614_recursive_lsf_pst.tsv | 18 | 1294598 | 1.295 |
| outputs/burgers_timematched_solver7860_clean8000_audit_20260614/data/r2_live_gap_search_20260614/forensics_burgers_wideparam_loss123_randomsolver7860_clean8000_round00_p2q2_six_model_visuals_20260614_recursive_lsf_pst.tsv | 27 | 203197102 | 203.197 |
| outputs/burgers_timematched_solver7860_clean8000_audit_20260614/data/r2_live_gap_search_20260614/forensics_burgers_wideparam_loss3targeted_biased_local_direction_20260611_recursive_lsf_pst.tsv | 4 | 46751 | 0.047 |
| outputs/burgers_timematched_solver7860_clean8000_audit_20260614/data/r2_live_gap_search_20260614/forensics_burgers_wideparam_loss3targeted_full1024_svd_attack25_biased_local_direction_20260611_recursive_lsf_pst.tsv | 6 | 305297 | 0.305 |
| outputs/burgers_timematched_solver7860_clean8000_audit_20260614/data/r2_live_gap_search_20260614/forensics_burgers_wideparam_loss3targeted_full1024_svd_attack25_reuse3_20260611_recursive_lsf_pst.tsv | 501 | 2670190545 | 2670.191 |
| outputs/burgers_timematched_solver7860_clean8000_audit_20260614/data/r2_live_gap_search_20260614/run_logs_burgers_missing_roots_full_p2q2_20260608_recursive_lsf_pst.tsv | 3 | 53166 | 0.053 |
| outputs/burgers_timematched_solver7860_clean8000_audit_20260614/data/r2_live_gap_search_20260614/visualizations_burgers_wideparam_loss123_randomsolver7860_clean8000_comparison_dense_image_only_bundle_20260614_recursive_lsf_pst.tsv | 48 | 116384295 | 116.384 |
| outputs/burgers_timematched_solver7860_clean8000_audit_20260614/data/r2_live_gap_search_20260614/visualizations_burgers_wideparam_loss123_randomsolver7860_clean8000_round00_p2q2_six_model_visuals_20260614_recursive_lsf_pst.tsv | 48 | 116384295 | 116.384 |

## Resolved Gaps

- Random clean/solver top50/top100 SVD: no already-exported random top50/top100 table was found, but the completed random suite stored full `1024 x 1024` Jacobian matrices. The supplement in `data/random_top100_svd_supplement_20260614/` derives top100 SVD from those existing matrices. No training, attack generation, or Jacobian generation was rerun.
- Random clean/solver affine/local-gain: no direct old4-style random affine sweep export was found. The supplement in `data/random_affine_direction_supplement_20260614/` uses existing checkpoints and stored Jacobians. The clean residual vector is recomputed by model forward because the completed suite stored residual norms/MSE but not the full vector.

## Remaining Caveats

- `sample_id=4` has no attack-delta cosine for random affine metrics because that train sample is outside the saved attack trace manifest. Local affine/SVD/outward gains are still present for that sample.
- The recovered six-model 52-dataset attack table is mixed-source: old4 rows come from historical full-52 attack artifacts, while random rows come from solver7860/clean8000. The dense latest visual traces remain the latest same-panel direct attack evidence.

## Main Output Tables

- `data/ranked_metric_tables_20260614/svd_error_top100_supplement_ranked_long.csv`
- `data/ranked_metric_tables_20260614/svd_error_topk_top100_supplement_ranked_long.csv`
- `data/ranked_metric_tables_20260614/model_solver_subspace_top100_supplement_ranked_long.csv`
- `data/ranked_metric_tables_20260614/random_affine_direction_supplement_metric_long_ranked.csv`
- `data/missing_metric_coverage_audit.csv`
- `data/missing_metric_coverage_audit.json`
