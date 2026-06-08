# Burgers Loss1 5000-to-8000 Launch - 2026-06-07

## Status

Observed from local tmux/log checks: the Burgers round03 Loss1 continuation from epoch5000 to epoch8000 was launched in tmux session `burgers_loss1_8000_20260607` at `2026-06-07T14:37:26Z`.

## Source And Output Paths

- Launcher: `tools/run_burgers_loss1_continue5000to8000_20260607.sh`
- Plot wrapper: `tools/plot_burgers_round03_loss1_8000_dense_comparison.py`
- Start checkpoint: `adversarial_training_runs/burgers_loss3_selective_round03_loss1_continue3000to5000_20260606/burgers/checkpoints/burgers_epoch5000_step015000.pt`
- Active run directory: `adversarial_training_runs/burgers_loss3_selective_round03_loss1_continue5000to8000_20260607`
- Driver log: `adversarial_training_runs/burgers_loss3_selective_round03_loss1_continue5000to8000_20260607_logs/driver.log`
- Training log: `adversarial_training_runs/burgers_loss3_selective_round03_loss1_continue5000to8000_20260607_logs/burgers_loss3_selective_round03_loss1_continue5000to8000_20260607.log`
- GPU preflight: `forensics/burgers_loss1_5000to8000_gpu_preflight_20260607/gpu_preflight.json`
- Expected final checkpoint: `adversarial_training_runs/burgers_loss3_selective_round03_loss1_continue5000to8000_20260607/burgers/checkpoints/burgers_epoch8000_step024000.pt`
- Expected single-run plot directory after completion: `visualizations/burgers_loss3_selective_round03_loss1_8000ep_long_20260605_plots`
- Expected dense/compact comparison plot directories after completion: `visualizations/burgers_loss3_selective_round03_loss1_8000_dense_comparison_20260607` and `visualizations/burgers_loss3_selective_round03_loss1_8000_comparison_20260607`

## Key Settings

Observed from launcher/config:

- Device: `cuda`
- Epoch continuation: `--epochs 3000`, `--resume-epoch-offset 5000`, `--resume-global-step-offset 15000`
- Attack objective: `loss1`
- Attack method: `fast_replace_l2`
- Burgers p=2/q=2 guard: `--burgers-require-p2q2`
- Attack steps: `5`
- Attack batch size: `480`
- Optimizer microbatch size: `32`
- Epsilon fraction: `0.06`
- Epsilon/alpha jitter: `0.75..1.25`
- Random-start fraction: `1e-6`
- Training data mode: `adv-only`
- Evaluation: all 50 round03 generated/generalization datasets, plus train/test
- Checkpoints: every 100 epochs plus wall-clock checkpoints and final

## Ten-Minute Monitor Evidence

Observed at `2026-06-07T14:47:41Z`:

- tmux session `burgers_loss1_8000_20260607` was still present.
- `nvidia-smi` showed Tesla V100-SXM2-32GB on GPU, `16190 MiB / 32768 MiB` used, `100%` utilization while Burgers and Darcy screening were both active.
- Burgers output files existed and were growing.
- `adversarial_training_runs/burgers_loss3_selective_round03_loss1_continue5000to8000_20260607/burgers/train_steps.csv` had `209` total lines.
- `adversarial_training_runs/burgers_loss3_selective_round03_loss1_continue5000to8000_20260607/burgers/eval_split_summary.csv` had `281` total lines.

## Current Interpretation

Observed evidence supports that the continuation launched correctly and survived the requested first 10 minutes. The epoch8000 checkpoint and refreshed plots are not yet evidenced; they should only be claimed after the final checkpoint and plot outputs exist.

## Remaining Work

- Let the tmux session continue to epoch8000.
- After completion, verify `summary.json`, `burgers_epoch8000_step024000.pt`, and refreshed plot directories.
- Record final train/test/generalization metrics and plot paths in `EXPERIMENT_LEDGER.md`.

## Progress Check - 2026-06-07T15:45Z

Observed from local files after the requested 10-minute monitor window:

- tmux session `burgers_loss1_8000_20260607` is still running.
- Latest `train_steps.csv` rows show epoch `5730`, step `17190`.
- Latest `eval_split_summary.csv` rows show epoch `5730`, step `17190`, progress fraction `0.71625`.
- Checkpoints currently observed in `adversarial_training_runs/burgers_loss3_selective_round03_loss1_continue5000to8000_20260607/burgers/checkpoints` include regular checkpoints through `burgers_epoch5700_step017100.pt` plus wall-time checkpoints through `burgers_epoch5625_step016875_wall_0003600s.pt`.

Inference:

- The Burgers Loss1 5000-to-8000 continuation is still healthy and writing outputs, but the requested final epoch8000 checkpoint and refreshed final plots are not yet evidenced locally.
## Progress Check - 2026-06-07T18:15Z

Observed from local files and tmux:

- tmux session `burgers_loss1_8000_20260607` is still running.
- Current UTC check time: `2026-06-07T18:15:12Z`.
- Latest `eval_split_summary.csv` rows show epoch `7320`, step `21960`, progress fraction `0.915` of the total epoch0-to-8000/global-step scale.
- For this continuation segment, that means `2320 / 3000` epochs completed from epoch5000 to epoch8000, with `680` epochs remaining.
- Latest regular checkpoint observed: `burgers_epoch7300_step021900.pt`.
- Wall-time checkpoints observed through `burgers_epoch7230_step021690_wall_0012600s.pt`.

Runtime estimate:

- Observed elapsed time from tmux creation at `2026-06-07T14:37:26Z` to `2026-06-07T18:15:12Z`: `13066` seconds.
- Observed continuation speed: about `5.63` seconds per epoch.
- Estimated training completion for epoch8000: about `2026-06-07T19:19Z`.
- The launcher refreshes stitched and comparison plots after the final checkpoint, so full workflow completion is expected a few minutes after training, approximately `2026-06-07T19:25Z` if the current speed holds.

Inference:

- The run is healthy and well past the midpoint, but the final epoch8000 checkpoint and refreshed plots are not yet evidenced locally.
## Completion And Plot Fix - 2026-06-07T19:39Z

Observed from local files/logs:

- Training completed at `2026-06-07T19:35:51Z` according to `adversarial_training_runs/burgers_loss3_selective_round03_loss1_continue5000to8000_20260607_logs/driver.log`.
- Final checkpoint exists: `adversarial_training_runs/burgers_loss3_selective_round03_loss1_continue5000to8000_20260607/burgers/checkpoints/burgers_epoch8000_step024000.pt`.
- Run summary exists: `adversarial_training_runs/burgers_loss3_selective_round03_loss1_continue5000to8000_20260607/summary.json`.
- Stitched single-run plots were generated under `visualizations/burgers_loss3_selective_round03_loss1_8000ep_long_20260605_plots`.
- The first automatic dense-comparison plot call failed with a Python dataclass import-wrapper error, not a GPU/OOM failure.
- Fixed `tools/plot_burgers_round03_loss1_8000_dense_comparison.py` by registering the dynamically loaded base module in `sys.modules`, then reran the plot script successfully.
- Dense/compact plot manifests now exist: `visualizations/burgers_loss3_selective_round03_loss1_8000_dense_comparison_20260607/round03_dense_plot_manifest.txt` and `visualizations/burgers_loss3_selective_round03_loss1_8000_comparison_20260607/round03_long_training_comparison_plot_manifest.txt`.
- Plot report exists: `docs/burgers_loss3_selective_round03_loss1_8000_plot_report_20260607.md`.

Inference:

- Burgers Loss1 5000-to-8000 is complete. The only interruption was a post-training plot wrapper bug, which has been fixed and rerun.
