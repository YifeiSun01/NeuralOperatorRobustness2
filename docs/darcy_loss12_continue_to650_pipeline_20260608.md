# Darcy Loss1/Loss2 Continue-to-650 Pipeline - 2026-06-08

Status: code prepared. This document describes the continuation pipeline; it does not claim the 650-epoch continuation has completed.

## Purpose

Continue Darcy loss1 and loss2 self-training from already trained checkpoints to an explicit target epoch `650`, then run the same postprocessing family as the previous Darcy time-matched workflow: summaries, RMSE/relative-L2 split curves, attack diagnostic curves, eval-train-loss progress plots, stitched full-history records, ledger update, and R2 upload.

## Code

- Runner: `tools/run_darcy_loss12_continue_to650_20260608.sh`
- Stitching helper: `tools/stitch_darcy_continuation_run_20260608.py`
- Training CLI support: `tools/adversarial_training.py` now includes `--darcy-initial-checkpoint` so Darcy continuations can load a checkpoint while preserving `--resume-epoch-offset` and `--resume-global-step-offset`.

## Default Source Runs

The runner defaults to these source runs and waits for their `darcy/summary.json` files if needed:

- loss1 source: `adversarial_training_runs/darcy_lossdrop50_loss1_single_gpu_time_matched_loss3wall_20260608`
- loss2 source: `adversarial_training_runs/darcy_lossdrop50_loss2_single_gpu_time_matched_loss3wall_20260608`

If a source run finishes below epoch 650, the continuation local epoch count is computed as `650 - source_epoch`. Example: source epoch `632` continues for `18` local epochs and writes a final epoch `650` checkpoint.

## Main Outputs

- Continuation runs: `adversarial_training_runs/darcy_lossdrop50_loss{1,2}_continue_to650_from_epoch*_20260608/`
- Stitched full-history records: `adversarial_training_runs/darcy_lossdrop50_loss{1,2}_to650_stitched_20260608/`
- Combined comparison: `adversarial_training_runs/darcy_lossdrop50_loss12_to650_loss3_physics_summary_20260608/`
- Combined doc: `docs/darcy_lossdrop50_loss12_to650_loss3_physics_summary_20260608.md`
- Logs: `adversarial_training_runs/darcy_loss12_continue_to650_20260608_logs/`
- GPU preflight: `forensics/darcy_loss12_continue_to650_20260608_gpu_preflight/`

## Run Command

```bash
tmux new-session -d -s darcy_loss12_continue_to650_20260608   'cd /workspace/NeuralOperatorRobustness2 && tools/run_darcy_loss12_continue_to650_20260608.sh'
```

By default `UPLOAD_TO_R2=1`. Set `UPLOAD_TO_R2=0` only for a dry local run. Set `WAIT_FOR_SOURCE=0` to fail immediately if source summaries are not present.

## Notes

- The runner is sequential: loss1 continuation runs first, then loss2. It waits for active `tools/adversarial_training.py` processes before starting a continuation to avoid concurrent GPU contamination.
- Existing checkpoints contain model weights/config, not Adam optimizer state; continuation resumes weights and epoch/global-step numbering while AdamW restarts, matching the existing Burgers continuation convention.
