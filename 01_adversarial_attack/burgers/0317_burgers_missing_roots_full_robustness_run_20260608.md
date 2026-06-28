# Burgers Three Generalization Roots: Robustness/SVD Inventory And Missing Full-Tag Run (2026-06-08)

## Scope

This note records the current evidence status for the three Burgers generalization roots under the user's three-run naming:

1. First root: `generalization_datasets/burgers`
2. Second root: `generalization_datasets_rmse_1p5_3x_all_ns50/burgers`
3. Third root: `generalization_datasets_burgers_loss3_selective_search/round_03/burgers`

The target robustness table is the same full final-only P2Q2 attack/tag format as the completed second-root run: train first 50, full test, 50 generalization datasets, 4 models, 20 PGD steps, `epsilon_rms=0.12`, `alpha_rms=0.012`, batch size 500.

## Observed Evidence: Local And Prior R2 Audit

Observed locally:

- Second-root full attack/tag is complete at `forensics/burgers_round03_full_p2q2_52datasets_4models_finalonly_20step_20260607`.
- The second-root run has `10200` samples and `52` datasets, with per-model arrays and `summary_by_model_dataset.csv`.
- Derived second-root per-sample clean-vs-attack audit is at `forensics/burgers_round03_full52_per_sample_clean_vs_attack_mismatch_20260608/per_sample_clean_attack_wide.csv`.
- First-root clean/generalization evidence exists in `docs/burgers_master_semantic_final_models_20260608.md`.
- A completed first-root full P2Q2 attack/tag table analogous to the second-root `10200`-sample artifact was not found locally.
- A first-root SVD directory exists at `forensics/burgers_first_master_finalmodels_jacobian_svd_rep20_top100_20260608`, but observed files are preflight/config/manifest/sample-start files only; no completed `jacobian_svd_summary.csv` was found there in this audit.
- Third-root stress-set clean/generated evidence exists in `docs/burgers_loss3_selective_round03_complete_report_20260605.md` and `forensics/burgers_loss3_selective_round03_per_dataset_loss3_advantage_20260605/per_dataset_advantage_summary.csv`.
- Third-root SVD20/attack-correlation evidence exists, for example `forensics/burgers_round03_long_final_loss123_svd20_p2q2_attack_correlation_20260608` and `forensics/burgers_svd20_p2q2_attack_correlation_20260608`.
- A separate third-root full `10200`-sample P2Q2 final-only attack/tag table was not found locally before this run.

Observed current R2 access state:

- `rclone listremotes` reported no configured remote and no `/root/.config/rclone/rclone.conf` in the current shell.
- Therefore this turn did not independently enumerate R2. The prior local R2 audit note remains the available R2 evidence: `docs/r2_burgers_three_dataset_conclusion_audit_20260608.md`.

Inference from the current file evidence:

- The second root is the only root that was already fully complete under the `10200`-sample full-tag robustness protocol.
- First-root and third-root full robustness/tag results need to be generated or found remotely. Since current R2 remote is not configured, they are being generated locally now with the same protocol.

## Code Change For Reproducibility

Updated source script:

- `tools/run_burgers_round03_full_p2q2_finalonly_attack.py`

Change:

- Added `--gen-root` so the same full-tag protocol can be run on any of the three generalization roots.
- The default remains the second-root path, preserving the old behavior.
- The chosen `gen_root` is now recorded in each run's `config.json`.

New launcher:

- `tools/run_burgers_missing_roots_full_p2q2_attack_20260608.sh`

It runs sequentially, not concurrently:

1. First root output: `forensics/burgers_first_master_full_p2q2_52datasets_4models_finalonly_20step_20260608`
2. Third root output: `forensics/burgers_round03_selective_full_p2q2_52datasets_4models_finalonly_20step_20260608`

Logs:

- GPU preflight: `run_logs/burgers_missing_roots_full_p2q2_20260608/gpu_preflight.txt`
- First-root run log: `run_logs/burgers_missing_roots_full_p2q2_20260608/first_master_full_p2q2_20step.log`
- Third-root run log: `run_logs/burgers_missing_roots_full_p2q2_20260608/round03_selective_full_p2q2_20step.log`

## GPU Preflight

Observed from `run_logs/burgers_missing_roots_full_p2q2_20260608/gpu_preflight.txt`:

- GPU: Tesla V100-SXM2-32GB
- Driver: 580.76.05
- PyTorch: `2.8.0+cu126`
- PyTorch CUDA: `12.6`
- CUDA available: `true`
- Compute capability: `[7, 0]`
- PyTorch arch list includes `sm_70`
- CUDA matmul sanity check passed with value `128.0`

## Current Run Status

Observed launch state:

- tmux session: `burgers_missing_roots_fulltag_20260608`
- First-root run started at `forensics/burgers_first_master_full_p2q2_52datasets_4models_finalonly_20step_20260608`.
- First-root start log reports `sample_count=10200`, `dataset_count=52`, `additional_steps=20`, `batch_size=500`.
- `nvidia-smi` after launch showed about `30084 MiB / 32768 MiB` in use, consistent with the previous completed full-tag run's high V100 utilization.
- Third-root full run is pending behind the first-root run in the same launcher.

## Status Conclusion

The correct current status is not "all three roots already had full robustness/tag complete." The complete existing full-tag evidence was the second root. The missing same-strength full robustness/tag evidence for the first and third roots is now being generated with the same script and protocol.


## Status Check: 2026-06-08 23:54 UTC

Observed process:

- tmux session: `burgers_missing_roots_fulltag_20260608`
- command: `tools/run_burgers_round03_full_p2q2_finalonly_attack.py --gen-root generalization_datasets/burgers --run-name burgers_first_master_full_p2q2_52datasets_4models_finalonly_20step_20260608 --steps 20 --batch-size 500 --train-count 50`

Observed conclusion:

- This background job is a full P2Q2 attack/tag run, not a Jacobian/SVD run.
- No active `jacobian_svd`/SVD process was observed in the matching process list.
- Current first-root run has completed baseline and is running `loss1_epoch8000`; latest observed completed batch was `loss1_epoch8000` samples `6000:6500`.
- GPU use remains high at about `30084 MiB / 32768 MiB` on Tesla V100-SXM2-32GB.

Inference:

- Runtime should be on the order of the existing full-tag attack run, not the much longer dense SVD runtime. It still needs time because it runs 20-step attacks over `10200` samples for 4 models per root, then repeats for the third root, but it is not recomputing SVD.
