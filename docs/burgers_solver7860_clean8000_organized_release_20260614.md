# Burgers Solver7860 Clean8000 Organized Release - 2026-06-14

Status: local organized release created and verified.

Observed from `outputs/burgers_solver7860_clean8000_organized_release_20260614/MANIFEST.json`:

- Organized output root: `outputs/burgers_solver7860_clean8000_organized_release_20260614/`
- Files: `661`
- Size recorded in manifest: `1469211584` bytes
- Disk usage: about `1.4G`
- Missing expected inputs: `0`
- Figure files: `178` PNG
- Table/data files: `94` CSV, `187` JSON, `137` NPZ, `4` JSONL
- Logs and code references: `37` log files, `8` Python scripts, `3` shell scripts

Layered layout:

- `00_start_here/`: final audit report, audit manifest, summary markdown, and current result notes.
- `01_summary_tables/`: clean 52-dataset tables, 52-dataset attack tables, 25-sample robustness/SVD tables, correlations, rankings, model-version and runtime JSON.
- `02_figures/`: linear training curves, log-y training curves, no-random-clean variants, polished report figures, dense six-model attack panels, and summary plots.
- `03_dense_six_model_attack_data/`: six dense six-model attack groups with sample manifests, attack-loss curves, summaries, and NPZ traces, including `group05` selected for strongest `loss3` advantage.
- `01_summary_tables/03_robustness_25sample/historical_svd_attack25_reuse3/`: compact SVD25 reuse3 attack/subspace tables.
- `01_summary_tables/04_correlations_and_rankings/biased_local_direction/`: biased-local-direction metrics, correlations, quantile summaries, and angle summaries.
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

Inference from the manifest: this organized folder is now the cleanest entry point for inspecting the final solver7860/clean8000 Burgers audit. It does not replace or delete the original result folders; it is a categorized copy built for review and R2 synchronization.

Generation script:

- `tools/organize_burgers_solver7860_clean8000_release_20260614.py`

R2 target prefix:

- `neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected/outputs/burgers_solver7860_clean8000_organized_release_20260614`

R2 verification:

- Verified after adding compact SVD25 reuse3 and biased-local-direction
  tables: `{"count":661,"bytes":1469211584,"sizeless":0}`.
