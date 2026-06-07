# Burgers Round03 Final Extension Automation - 2026-06-06

Status: implemented a new automatic workflow for the requested next continuation: loss1 epoch3000 to epoch5000, loss2 epoch1000 to epoch2000, and loss3 epoch1000 to epoch1500.

## Driver

New driver:

- `tools/run_burgers_loss3_selective_round03_loss123_final_extension_20260606.sh`

The driver performs, in order:

1. GPU preflight and dataset/checkpoint checks.
2. Continuation training for loss1, loss2, and loss3.
3. Automatic final-checkpoint Jacobian/SVD on the newly trained final models.
4. Automatic plot refresh so user-facing curves extend to the new final epochs.

## Training Targets

New run names:

- `burgers_loss3_selective_round03_loss1_continue3000to5000_20260606`
- `burgers_loss3_selective_round03_loss2_continue1000to2000_20260606`
- `burgers_loss3_selective_round03_loss3_continue1000to1500_20260606`

Starting checkpoints:

- loss1 epoch3000: `adversarial_training_runs/burgers_loss3_selective_round03_loss1_continue1000to3000_20260605/burgers/checkpoints/burgers_epoch3000_step009000.pt`
- loss2 epoch1000: `adversarial_training_runs/burgers_loss3_selective_round03_loss2_continue500to1000_20260605/burgers/checkpoints/burgers_epoch1000_step003000.pt`
- loss3 epoch1000: `adversarial_training_runs/burgers_loss3_selective_round03_loss3_continue500to1000_20260605/burgers/checkpoints/burgers_epoch1000_step003000.pt`

Expected final checkpoints:

- loss1 epoch5000: `adversarial_training_runs/burgers_loss3_selective_round03_loss1_continue3000to5000_20260606/burgers/checkpoints/burgers_epoch5000_step015000.pt`
- loss2 epoch2000: `adversarial_training_runs/burgers_loss3_selective_round03_loss2_continue1000to2000_20260606/burgers/checkpoints/burgers_epoch2000_step006000.pt`
- loss3 epoch1500: `adversarial_training_runs/burgers_loss3_selective_round03_loss3_continue1000to1500_20260606/burgers/checkpoints/burgers_epoch1500_step004500.pt`

As with the prior continuation, existing checkpoints contain model weights/config but not AdamW optimizer state; continuation resumes model weights and epoch/global-step numbering while AdamW moments restart.

## Automatic Final SVD

Final SVD output directory:

- `forensics/burgers_loss3_selective_round03_loss123_final_extension_jacobian_svd_rep20_top100_20260606`

Output prefix:

- `round03_loss123_final_extension`

The SVD stage uses the same fixed sample manifest as the pre-continuation long final SVD:

- `forensics/burgers_loss3_selective_round03_long_final_jacobian_svd_rep20_top100_20260605/round03_long_final_sample_manifest.csv`

This computes solver/model/error Jacobians, top singular values, spectral/Frobenius/effective-rank summaries, rankwise singular-vector similarity to solver, top-k singular-subspace similarity to solver, and model-minus-solver error spectral norm shrinkage.

Same-wall/time-matched SVD remains cancelled; this automation is final-checkpoint SVD only.

## Automatic Plot Refresh

Updated plotting support:

- `tools/plot_burgers_round03_stitched_single_run_visualizations.py` now accepts multiple `--extension-run-dir` arguments, so base + first continuation + final extension can be stitched as one curve.
- `tools/plot_burgers_round03_loss123_final_extension_dense_comparison.py` generates dense and compact comparison figures through the new final epochs.

User-facing single-run plot directories after completion:

- `visualizations/burgers_loss3_selective_round03_loss1_5000ep_long_20260605_plots`
- `visualizations/burgers_loss3_selective_round03_loss2_2000ep_long_20260605_plots`
- `visualizations/burgers_loss3_selective_round03_loss3_1500ep_long_20260605_plots`

Comparison plot directories refreshed in place:

- `visualizations/burgers_loss3_selective_round03_long_training_comparison_dense_20260605`
- `visualizations/burgers_loss3_selective_round03_long_training_comparison_20260605`

The driver renames the old single-run plot directories to the new final-epoch names when the new directory does not already exist, then overwrites the plots with stitched curves.

## Verification Before Launch

Observed checks before launch:

- `bash -n tools/run_burgers_loss3_selective_round03_loss123_final_extension_20260606.sh` passed.
- `python3 -m py_compile tools/plot_burgers_round03_loss123_final_extension_dense_comparison.py tools/plot_burgers_round03_stitched_single_run_visualizations.py` passed.
- Starting checkpoints and the fixed SVD sample manifest exist locally.
- Target final-extension run directories were not present before launch.


## Launch Update

Launched in tmux session:

- `round03_loss123_final_extension`

Observed launch evidence:

- Session created at `2026-06-06T14:48:05Z`.
- Driver log recorded start of `burgers_loss3_selective_round03_loss1_continue3000to5000_20260606` at `2026-06-06T14:48:08Z`.
- Observed process command used `--device cuda`, `--epochs 2000`, `--resume-epoch-offset 3000`, `--resume-global-step-offset 9000`, and the loss1 epoch3000 checkpoint as `--burgers-initial-checkpoint`.
- GPU preflight recorded PyTorch `2.8.0+cu126`, Torch CUDA `12.6`, Tesla V100-SXM2-32GB, capability `[7, 0]`, arch list including `sm_70`, JAX `0.10.0`, JAX backend `gpu`, and JAX device `cuda:0`.
- `nvidia-smi` during launch showed GPU memory in use and nonzero utilization.
- Loss1 run log passed dataset-count preflight and wrote `planned_workload_estimate.json`.

Current active phase:

- loss1 final-extension training, epoch3000 to epoch5000.

After loss1 completes, the same driver will run loss2 epoch1000 to epoch2000, loss3 epoch1000 to epoch1500, final-checkpoint SVD, then plot refresh.


## Ten-Minute Monitor Update

A short monitor was run after launch and then stopped without interrupting the training job.

Observed from the monitor window:

- Monitoring started shortly after launch and ended at `2026-06-06T15:02:03Z`.
- tmux session `round03_loss123_final_extension` remained present throughout.
- Active phase remained `loss1_3000_to_5000_running`.
- The process command remained the expected loss1 final-extension training command, not a retrain of old runs and not SVD.
- GPU sampling remained active with about `4896 MiB / 32768 MiB` memory used and nonzero utilization.
- `train_steps.csv` advanced from about epoch `3052` to epoch `3160` during the monitor window.
- The loss1 log continued to show the run after successful dataset preflight.

Current state after monitor:

- Tracking stopped.
- Training was not stopped.
- Background tmux training continues.

Next expected stages after loss1 finishes:

1. loss2 epoch1000 to epoch2000.
2. loss3 epoch1000 to epoch1500.
3. final-checkpoint Jacobian/SVD.
4. stitched plot refresh to 0..5000, 0..2000, and 0..1500.

## Health Check Near Loss1 Completion

Observed at `2026-06-06T17:35:09Z`:

- tmux session `round03_loss123_final_extension` remained active.
- Active process was still the expected loss1 continuation `burgers_loss3_selective_round03_loss1_continue3000to5000_20260606`, using `--device cuda` and the epoch3000 checkpoint.
- `nvidia-smi` showed Tesla V100-SXM2-32GB using `4896 MiB / 32768 MiB` with nonzero utilization; no CPU fallback evidence was observed.
- `train_steps.csv` had advanced to epoch `4987`, global step `14961`, progress `0.9974`.
- `eval_split_summary.csv` had advanced to epoch `4990`, global step `14970`, progress `0.998`; generated/generalization RMSE was `0.03246346465091287` and relative L2 was `0.058198833879218884`.
- `checkpoints.csv` showed periodic checkpoints through epoch `4900`.

Inference: the workflow is running normally and is close to finishing loss1 epoch5000 before handing off to loss2.

## Loss1 Completion And Loss2 Handoff

Observed at `2026-06-06T17:36:53Z`:

- Driver log recorded `done burgers_loss3_selective_round03_loss1_continue3000to5000_20260606` at `2026-06-06T17:36:15Z`.
- The same driver immediately started `burgers_loss3_selective_round03_loss2_continue1000to2000_20260606` from the loss2 epoch1000 checkpoint.
- Active process is now the expected loss2 continuation command with `--device cuda`, `--epochs 1000`, `--resume-epoch-offset 1000`, `--resume-global-step-offset 3000`, and objective `loss2`.
- Loss1 final evaluation reached epoch `5000`, global step `15000`; generated/generalization RMSE was `0.03245434070606764` and relative L2 was `0.058182682528028995`.
- Loss2 has already written training rows through epoch `1002`, global step `3004`, and evaluation rows through epoch `1001`, global step `3003`.
- `nvidia-smi` showed Tesla V100-SXM2-32GB using `4896 MiB / 32768 MiB` with `54%` GPU utilization.

Inference: the first final-extension segment completed normally, and the automated handoff to loss2 is working.

## Loss2 Late-Stage Health Check

Observed at `2026-06-06T22:28:29Z`:

- tmux session `round03_loss123_final_extension` remained active.
- Active process was `burgers_loss3_selective_round03_loss2_continue1000to2000_20260606`, using `--device cuda`, `--epochs 1000`, `--resume-epoch-offset 1000`, `--resume-global-step-offset 3000`, and objective `loss2`.
- `nvidia-smi` showed Tesla V100-SXM2-32GB using `4896 MiB / 32768 MiB` with nonzero GPU utilization.
- Loss2 evaluation had advanced through epoch `1774`, global step `5322`, progress `0.887`; generated/generalization RMSE was `0.03540885411475918` and relative L2 was `0.06349008010938685`.
- Loss2 training rows had advanced through epoch `1776`, global step `5327`, progress `0.8878333333333334`.
- Loss2 checkpoints were present through periodic epoch `1700` and wall-clock epoch `1717` at wall target `16200` seconds.
- The inspected loss2 log tail showed no error.

ETA inference from observed wall-clock data: loss2 has roughly `1.4` hours left, loss3 should take on the order of `6.4` hours based on the prior 500-epoch loss3 continuation, and the final SVD/plot stage likely adds hours rather than minutes. A reasonable total remaining estimate from this check is roughly `9.5` to `10.5` hours.

## Loss3 Active Health Check

Observed at `2026-06-07T01:08:11Z`:

- tmux session `round03_loss123_final_extension` remained active.
- Driver log recorded loss2 completion at `2026-06-06T23:56:56Z` and immediate loss3 start at `2026-06-06T23:56:56Z`.
- Active process was `burgers_loss3_selective_round03_loss3_continue1000to1500_20260606`, using `--device cuda`, `--epochs 500`, `--resume-epoch-offset 1000`, `--resume-global-step-offset 3000`, and objective `loss3`.
- `nvidia-smi` showed Tesla V100-SXM2-32GB using `30732 MiB / 32768 MiB` with `44%` GPU utilization.
- Loss2 final generated/generalization metrics at epoch `2000`: RMSE `0.03573958119889953`, relative L2 `0.06408098797752262`; total wall seconds `22837.659613220952`.
- Loss3 training rows had advanced to epoch `1095`, global step `3284`; evaluation rows had advanced to epoch `1093`, global step `3279`.
- Latest visible loss3 generated/generalization metrics: RMSE `0.019878102911421624`, relative L2 `0.03562912872951677`.
- Loss3 checkpoint rows showed wall-clock checkpoints at epoch `1040` and `1080`; epoch `1080` elapsed seconds was `3614.595871511847`.
- Final SVD and final plot refresh had not started yet.

ETA inference: loss3 has about `405` epochs remaining from epoch `1095`; using the observed checkpoint rate, remaining loss3 training is about `5.1` hours. Final SVD and plot refresh will follow automatically after loss3 finishes.

## Loss3 Mid-Late Health Check

Observed at `2026-06-07T04:18:34Z`:

- tmux session `round03_loss123_final_extension` remained active.
- Active process was `burgers_loss3_selective_round03_loss3_continue1000to1500_20260606`, using `--device cuda`, `--epochs 500`, `--resume-epoch-offset 1000`, `--resume-global-step-offset 3000`, and objective `loss3`.
- `nvidia-smi` showed Tesla V100-SXM2-32GB using `30732 MiB / 32768 MiB` with `30%` GPU utilization.
- Loss3 training rows had advanced to epoch `1345`, global step `4035`; evaluation rows had also advanced to epoch `1345`, global step `4035`.
- Latest visible loss3 generated/generalization metrics: RMSE `0.021111329793382722`, relative L2 `0.037838767998016394`.
- Loss3 checkpoints were present through wall-clock epoch `1317` and periodic epoch `1300`.
- Final SVD and final plot refresh had not started yet.

ETA inference: loss3 has about `155` epochs remaining from epoch `1345`; using the observed checkpoint rate, remaining loss3 training is about `2.0` hours. Final SVD and plot refresh will follow automatically after loss3 finishes.

## Completion Update

Observed at final inspection on `2026-06-07`:

- The final-extension workflow completed end-to-end at `2026-06-07T09:29:28Z` according to `adversarial_training_runs/burgers_loss3_selective_round03_loss123_final_extension_20260606_logs/driver.log`.
- loss1 epoch5000, loss2 epoch2000, and loss3 epoch1500 summaries and checkpoints exist.
- Final-checkpoint Jacobian/SVD completed under `forensics/burgers_loss3_selective_round03_loss123_final_extension_jacobian_svd_rep20_top100_20260606`.
- Stitched single-run plots were refreshed under the loss1 5000ep, loss2 2000ep, and loss3 1500ep visualization directories.
- Dense and compact comparison plots were refreshed under `visualizations/burgers_loss3_selective_round03_long_training_comparison_dense_20260605` and `visualizations/burgers_loss3_selective_round03_long_training_comparison_20260605`.
- Dedicated completion report: `docs/burgers_loss3_selective_round03_final_extension_completion_20260607.md`.
