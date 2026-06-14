# Burgers Solver7860 Clean8000 Organized Release - 2026-06-14

Status: local organized release created and verified.

Observed from `outputs/burgers_solver7860_clean8000_organized_release_20260614/MANIFEST.json`:

- Organized output root: `outputs/burgers_solver7860_clean8000_organized_release_20260614/`
- Files: `600`
- Size recorded in manifest: `1363789188` bytes
- Disk usage: about `1.3G`
- Missing expected inputs: `0`
- Figure files: `146` PNG
- Table/data files: `82` CSV, `181` JSON, `136` NPZ, `4` JSONL
- Logs and code references: `29` log files, `6` Python scripts, `3` shell scripts

Layered layout:

- `00_start_here/`: final audit report, audit manifest, summary markdown, and current result notes.
- `01_summary_tables/`: clean 52-dataset tables, 52-dataset attack tables, 25-sample robustness/SVD tables, correlations, rankings, model-version and runtime JSON.
- `02_figures/`: linear training curves, log-y training curves, no-random-clean variants, polished report figures, dense six-model attack panels, and summary plots.
- `03_dense_six_model_attack_data/`: five dense six-model attack groups with sample manifests, attack-loss curves, summaries, and NPZ traces.
- `04_random_model_full_suite/`: random-clean/random-solver clean-loss, P2Q2 attack, Jacobian/SVD, and postprocess outputs.
- `05_polished_report_data/`: CSV/JSON/log data backing the per-model polished reports.
- `06_logs/`: postprocess and upload logs.
- `07_source_code_and_references/`: relevant scripts and experiment notes.
- `08_dense_image_only_bundle_full_copy/`: complete dense image-only bundle as
  its own subfolder, with 20 PNGs under `comparison_dense/group00..group04`.

Observed source roots:

- `outputs/burgers_timematched_solver7860_clean8000_audit_20260614`
- `forensics/burgers_six_model_solver7860_clean8000_summary_20260614`
- `forensics/burgers_random_solver7860_clean8000_full_suite_20260614`
- `forensics/burgers_wideparam_loss123_randomsolver7860_clean8000_round00_p2q2_six_model_visuals_20260614`
- `visualizations/burgers_wideparam_loss123_randomsolver7860_clean8000_comparison_dense_image_only_bundle_20260614`
- `visualizations/burgers_solver7860_clean8000_polished_reports_20260614`
- `run_logs/burgers_solver7860_postprocess_20260614`

Inference from the manifest: this organized folder is now the cleanest entry point for inspecting the final solver7860/clean8000 Burgers audit. It does not replace or delete the original result folders; it is a categorized copy built for review and R2 synchronization.

Generation script:

- `tools/organize_burgers_solver7860_clean8000_release_20260614.py`

R2 target prefix:

- `neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected/outputs/burgers_solver7860_clean8000_organized_release_20260614`

R2 verification:

- `rclone size --json` returned `{"count":600,"bytes":1363789188,"sizeless":0}`.
- The sync log showed transient Cloudflare R2 `501 NotImplemented` retries, but the final retry succeeded and the verified remote object count/byte size matches local `MANIFEST.json`.
