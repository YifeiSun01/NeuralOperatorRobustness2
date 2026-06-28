# NS2D GitHub and R2 Sync Record - 2026-05-23

Status: active NS2D attack was paused before synchronization. No experiment process was killed. This record summarizes the GitHub and R2 update performed after the pause.

## Paused Experiment

Observed from `ps` after the pause:

- Parent launcher: `/tmp/run_ns2d_pair_outer_attack.sh`
- Active block: `eps16_alpha5 / loss3 / all_d_target_w`
- Python PID: `408673`
- Wrapper PID: `408671`
- Process state after pause: `Tl`
- Resume command: `kill -CONT 408673`

Observed from `nvidia-smi` after the pause:

- GPU: `NVIDIA A100-SXM4-80GB`
- GPU utilization: `0%`
- GPU memory remained allocated by the stopped CUDA process, which is expected for `SIGSTOP`.

## GitHub Update

Observed evidence:

- Staged source, script, Markdown, and ledger files only.
- Large generated arrays, checkpoints, perturbation outputs, and images were not added to Git.
- Main pushed experiment-record/tooling commit: `c2cc0ea` on branch `vast-ai`.

Files included in the main pushed commit:

- `EXPERIMENT_LEDGER.md`
- `docs/burgers_epsilon_dependence_vs_ns2d_check_20260523.md`
- `docs/burgers_vs_ns2d_unified_experiment_coverage_20260523.md`
- `docs/burgers_vs_ns2d_unified_optimizer_mechanism_experiment_plan_20260523.md`
- `docs/ns2d_optimizer_hypothesis_validation_runtime_estimate_20260522.md`
- `docs/ns2d_optimizer_validation_next_experiments_20260523.md`
- `docs/ns2d_optimizer_validation_offline_diagnostics_20260523.md`
- `docs/ns2d_recurrent_core4_attack_runtime_status_20260522.md`
- `docs/ns2d_recurrent_core4_attack_runtime_status_20260523.md`
- `docs/ns2d_recurrent_eps32_alpha10_step_trace_gif_20260523.md`
- `docs/ns2d_recurrent_eps8_alpha2p5_visualization_summary_20260523.md`
- `docs/ns2d_vs_1d_burgers_optimizer_curve_location_check_20260523.md`
- `tools/analyze_ns2d_optimizer_validation_offline.py`
- `tools/plot_ns2d_pair_attack_overview.py`
- `tools/plot_ns2d_step_trace_gif.py`
- `tools/render_ns2d_step_trace_gifs_batch.py`

## R2 Update

R2 prefix:

- `neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected`

Observed upload method:

- Used non-destructive `rclone copy`, not destructive sync.
- Temporary R2 config was removed after use.

Observed uploaded local paths:

- `EXPERIMENT_LEDGER.md`
- `docs/`
- `tools/`
- `2D_NS_FNO2d_recurrent/visualizations/`
- `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/`
- `2D_NS_FNO2d_recurrent/saved_models/2D/`
- `2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dictionary/`

Observed R2 verification:

- Root contains `EXPERIMENT_LEDGER.md`.
- `docs/` contains `ns2d_recurrent_eps8_alpha2p5_visualization_summary_20260523.md`.
- `docs/` contains `ns2d_vs_1d_burgers_optimizer_curve_location_check_20260523.md`.
- `tools/` contains the new NS2D analysis/plotting scripts.
- `2D_NS_FNO2d_recurrent/visualizations/eps8_alpha2p5_image_gif_download_package_20260523`: `39` objects, `553.377 MiB`.
- `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/full_adw_b10_pair_outer_baseline_first_20260522`: `682` objects, `12.727 GiB`.
- `2D_NS_FNO2d_recurrent/saved_models/2D/modes64_modes64_width60_epochs500_Tin10_T10_recurrent_pytorch_20260521_090136_UTC`: `15` objects, `8.795 GiB`.
- `2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dictionary`: `18` objects, `10.743 GiB`.

## Remaining Work

Observed:

- The paused process can be resumed with `kill -CONT 408673`.
- The working tree still contains untracked generated artifact directories. They are intentionally kept out of Git and are stored in R2.

Inference:

- The important experiment records, scripts, generated visualizations, perturbation results, trained model outputs, and dictionary data have been backed up to R2.
- The GitHub repository records the code and Markdown state needed to understand and reproduce the analysis without committing large binary artifacts.
