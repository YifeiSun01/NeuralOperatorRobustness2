# Burgers Solver7860 Clean8000 Organized Release - 2026-06-14

Status: local organized release rebuilt with ranked metric tables, verified, and synced to R2.

Observed from `outputs/burgers_solver7860_clean8000_organized_release_20260614/MANIFEST.json`:

- Organized output root: `outputs/burgers_solver7860_clean8000_organized_release_20260614/`
- Files: `1995`
- Exact byte size: see `outputs/burgers_solver7860_clean8000_organized_release_20260614/MANIFEST.json` after each rebuild. This release includes its own docs and ledger, so the byte count changes slightly whenever those records are refreshed.
- Missing expected inputs: `0`
- Figure files: `218` PNG
- Table/data files: `266` CSV, `770` JSON, `640` NPZ, `13` JSONL
- Logs and code references: `40` log files, `11` Python scripts, `3` shell scripts

Layered layout:

- `00_start_here/`: final audit report, audit manifest, summary markdown, and current result notes.
- `01_summary_tables/`: clean 52-dataset tables, recovered six-model 52-dataset attack tables, 25-sample robustness/SVD tables, correlations, rankings, paired loss3-vs-other tests, model-version/runtime JSON, first-master top100 metadata, and recovered summary roots.
- `02_figures/`: linear training curves, log-y training curves, no-random-clean variants, polished report figures, dense six-model attack panels, and summary plots.
- `03_dense_six_model_attack_data/`: six dense six-model attack groups with sample manifests, attack-loss curves, summaries, and NPZ traces, including `group05` selected for strongest `loss3` advantage.
- `01_summary_tables/03_robustness_25sample/historical_svd_attack25_reuse3/`: full old4 SVD25 reuse3 raw NPZ payloads plus top100 singular values, subspace tables, attack joins, and correlations recovered from R2.
- `01_summary_tables/04_correlations_and_rankings/biased_local_direction/`: biased-local-direction metrics, correlations, quantile summaries, and angle summaries.
- `01_summary_tables/08_recovered_prior_full_artifacts/`: old4 52-dataset P2Q2 attack raw payloads, six-model latest/selected/solver7860 summaries, and first-master top100 metadata.
- `04_random_model_full_suite/`: current random-clean/random-solver clean-loss, P2Q2 attack, Jacobian/SVD, and postprocess outputs.
- `04_random_model_full_suite/historical_selected_worktime_20260613/`: completed selected-worktime random full suite recovered from existing local/R2 results.
- `04_random_model_full_suite/historical_final_models_20260613/`: completed final-model random full suite recovered from existing local results.
- `05_polished_report_data/`: CSV/JSON/log data backing the per-model polished reports.
- `06_logs/`: postprocess and upload logs.
- `07_source_code_and_references/`: relevant scripts and experiment notes.
- `08_dense_image_only_bundle_full_copy/`: complete dense image-only bundle as its own subfolder, with 48 PNGs under `comparison_dense/group00..group05`, including the corrected `loss1/loss2/loss3/random_solver_y` variant with baseline and random clean Y removed.
- `01_summary_tables/09_ranked_metric_tables_20260614/`: complete ranked CSV tables for clean 52-dataset metrics, 52-dataset attack metrics, 25-sample robustness/Jacobian/SVD metrics, top20/top-k error singular values, model-level scalar summaries, correlations, best-vs-other significance tests, loss3-vs-other significance tests, and explicit random-model partial-coverage notes.
- `00_start_here/burgers_all_metric_ranked_tables_20260614.md`: Markdown appendix with best models bolded, mean/std/n, runner-up gap, paired t-test/Wilcoxon p-values, and BH-FDR q-values.

Recovered source roots now included:

- `outputs/burgers_timematched_solver7860_clean8000_audit_20260614`
- `forensics/burgers_six_model_solver7860_clean8000_summary_20260614`
- `forensics/burgers_six_model_selected_worktime_summary_20260613`
- `forensics/burgers_six_model_latest_wideparam_summary_20260613`
- `forensics/burgers_random_solver7860_clean8000_full_suite_20260614`
- `forensics/burgers_random_field_selected_worktime_full_suite_20260613`
- `forensics/burgers_random_field_final_models_full_suite_20260613`
- `forensics/burgers_first_master_full_p2q2_52datasets_4models_finalonly_20step_20260608`
- `forensics/burgers_first_master_finalmodels_jacobian_svd_rep20_top100_20260608`
- `forensics/burgers_wideparam_loss3targeted_full1024_svd_attack25_reuse3_20260611`
- `forensics/burgers_wideparam_loss3targeted_full1024_svd_attack25_biased_local_direction_20260611`
- `forensics/burgers_wideparam_loss123_randomsolver7860_clean8000_round00_p2q2_six_model_visuals_20260614`
- `visualizations/burgers_wideparam_loss123_randomsolver7860_clean8000_comparison_dense_image_only_bundle_20260614`
- `visualizations/burgers_solver7860_clean8000_polished_reports_20260614`
- `run_logs/burgers_solver7860_postprocess_20260614`

R2 target prefix:

- `neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected/outputs/burgers_solver7860_clean8000_organized_release_20260614`

R2 verification after the extra recovered-history placement:

- `{"count":1949,"bytes":5496318865,"sizeless":0}`

Latest R2 verification after adding ranked metric tables:

- Final audit output R2: `{"count":1732,"bytes":5203523912,"sizeless":0}`
- Organized release R2: 1969 objects; exact bytes should be read from the current `MANIFEST.json` and the final `rclone size --json` verification after the last docs/ledger refresh.
- Ranked tables under organized release R2: `16` CSV objects

Coverage note:

- The recovered random suites contain full saved Jacobian matrices and top20 SVD values/vectors, but no already-exported random top50/top100 SVD table was found in the local/R2 selected-prefix audit.
- Old4 has top100 SVD values/subspaces in `historical_svd_attack25_reuse3/`.
- Random clean/solver have corrected J^T-error, SVD/outward, attack-delta direction summaries, subspace similarity, and full10200 delta/loss correlation tables in the six-model summary roots, but no completed random old4-style affine local-gain sweep was found under the checked sources.
