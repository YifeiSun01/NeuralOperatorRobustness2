# Burgers Solver7860 Clean8000 Organized Release - 2026-06-14

Status: local organized release created and verified.

Observed from `outputs/burgers_solver7860_clean8000_organized_release_20260614/MANIFEST.json`:

- Organized output root: `outputs/burgers_solver7860_clean8000_organized_release_20260614/`
- Files: `1230`
- Size recorded in manifest: `4301766869` bytes
- Disk usage: about `4.1G`
- Missing expected inputs: `0`
- Figure files: `178` PNG
- Table/data files: `155` CSV, `446` JSON, `378` NPZ, `5` JSONL
- Logs and code references: `37` log files, `9` Python scripts, `3` shell scripts

Layered layout:

- `00_start_here/`: final audit report, audit manifest, summary markdown, and current result notes.
- `01_summary_tables/`: clean 52-dataset tables, recovered six-model 52-dataset attack tables, 25-sample robustness/SVD tables, correlations, rankings, paired loss3-vs-other tests, model-version and runtime JSON.
- `02_figures/`: linear training curves, log-y training curves, no-random-clean variants, polished report figures, dense six-model attack panels, and summary plots.
- `03_dense_six_model_attack_data/`: six dense six-model attack groups with sample manifests, attack-loss curves, summaries, and NPZ traces, including `group05` selected for strongest `loss3` advantage.
- `01_summary_tables/03_robustness_25sample/historical_svd_attack25_reuse3/`: full old4 SVD25 reuse3 raw NPZ payloads plus top100 singular values, subspace tables, attack joins, and correlations recovered from R2.
- `01_summary_tables/04_correlations_and_rankings/biased_local_direction/`: biased-local-direction metrics, correlations, quantile summaries, and angle summaries.
- `01_summary_tables/08_recovered_prior_full_artifacts/`: old4 52-dataset P2Q2 attack raw payloads and six-model latest/corrected metric-correlation summaries recovered from previous completed runs.
- `04_random_model_full_suite/`: random-clean/random-solver clean-loss, P2Q2 attack, Jacobian/SVD, and postprocess outputs.
- `05_polished_report_data/`: CSV/JSON/log data backing the per-model polished reports.
- `06_logs/`: postprocess and upload logs.
- `07_source_code_and_references/`: relevant scripts and experiment notes.
- `08_dense_image_only_bundle_full_copy/`: complete dense image-only bundle as
  its own subfolder, with 36 PNGs under `comparison_dense/group00..group05`.

Observed source roots:

- `outputs/burgers_timematched_solver7860_clean8000_audit_20260614`
- `forensics/burgers_six_model_solver7860_clean8000_summary_20260614`
- `forensics/burgers_random_solver7860_clean8000_full_suite_20260614`
- `forensics/burgers_wideparam_loss123_randomsolver7860_clean8000_round00_p2q2_six_model_visuals_20260614`
- `visualizations/burgers_wideparam_loss123_randomsolver7860_clean8000_comparison_dense_image_only_bundle_20260614`
- `visualizations/burgers_solver7860_clean8000_polished_reports_20260614`
- `run_logs/burgers_solver7860_postprocess_20260614`
- `forensics/burgers_wideparam_loss3targeted_full1024_svd_attack25_reuse3_20260611`
- `forensics/burgers_wideparam_loss3targeted_full1024_svd_attack25_biased_local_direction_20260611`
- `forensics/burgers_six_model_latest_wideparam_summary_20260613`
- `forensics/burgers_first_master_full_p2q2_52datasets_4models_finalonly_20step_20260608`

Inference from the manifest: this organized folder is now the cleanest entry point for inspecting the final solver7860/clean8000 Burgers audit. It does not replace or delete the original result folders; it is a categorized copy built for review and R2 synchronization.

Generation script:

- `tools/organize_burgers_solver7860_clean8000_release_20260614.py`

R2 target prefix:

- `neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected/outputs/burgers_solver7860_clean8000_organized_release_20260614`

R2 verification:

- Verified after recovering prior 52-dataset attack, full old4 SVD25 raw
  payloads, six-model latest/corrected metric summaries, and paired test tables:
  `{"count":1230,"bytes":4301766869,"sizeless":0}`.

Remaining known gap:

- Old4 has top100 SVD values/subspaces, but the recovered random
  `random_clean_y`/`random_solver_y` SVD source contains top20 only. Therefore
  the common six-model recovered singular-value table is top20.
