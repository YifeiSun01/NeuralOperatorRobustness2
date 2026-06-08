# Darcy Flow Loss3 500-Epoch Training Launch - 2026-06-07

## Status

Observed from local tmux/log checks: launched the requested Darcy Flow Loss3 adversarial training run for 500 epochs using the selected 50 loss-drop generalization datasets, plus screening train/test datasets.

This is a screening-baseline Loss3 run, not an official full Darcy benchmark, because the official local Darcy train/test/checkpoint artifacts were missing. The run explicitly records the screening overrides used.

## Run Identity

- tmux session: `darcy_lossdrop50_loss3_500ep_20260607`
- Run name: `darcy_lossdrop50_loss3_500ep_fromscreen_20260607`
- Run directory: `adversarial_training_runs/darcy_lossdrop50_loss3_500ep_fromscreen_20260607`
- Logs: `adversarial_training_runs/darcy_lossdrop50_loss3_500ep_fromscreen_20260607_logs`
- Runner: `tools/run_darcy_lossdrop50_loss3_500ep_20260607.sh`
- Training script: `tools/adversarial_training.py`

## Source Data

- Selected 50 generalization root: `generalization_datasets_darcy_lossdrop50_selected_20260607`
- Selected manifest: `generalization_datasets_darcy_lossdrop50_selected_20260607/candidate_manifest.csv`
- Train override: `2D_Darcy_FNO2d/datasets/grf_darcy_screen_20260607/train/dim2d_darcy_nx85_N384_solver=jaxcg_solve421_alpha2_tau3_binary3-12_f1_seed45_train.pt`
- Test override: `2D_Darcy_FNO2d/datasets/grf_darcy_screen_20260607/test/dim2d_darcy_nx85_N96_solver=jaxcg_solve421_alpha2_tau3_binary3-12_f1_seed45_test.pt`
- Starting model checkpoint override: `2D_Darcy_FNO2d/saved_models/2D/darcy_screen_baseline_m64_w60_e50_20260607/best.pt`

## Key Settings

Observed from `darcy/config.json`:

- Task: `darcy`
- Epochs: `500`
- Attack objective: Darcy non-Burgers path uses Loss3, `MSE(model(x_adv), solver(x_adv))`.
- Attack method: `binary_steepest_replace`
- Attack steps: `1`
- Batch size: `96`
- Optimizer microbatch size: `24`
- Epsilon fraction: `0.025`
- Epsilon jitter: `1.0..1.0`
- Training data mode: `adv-only`
- Label mode: `solver`
- Evaluation: train/test plus `50` selected generalization datasets
- Checkpoints: every `50` epochs plus wall-clock checkpoints
- Attack probes: `5` fixed train samples, saved every epoch with targets

## Output Files For Analysis

The run writes the same style of analysis artifacts as the Burgers adversarial-training runs:

- `darcy/train_steps.csv`
- `darcy/attack_batches.csv`
- `darcy/attack_epoch_summary.csv`
- `darcy/attack_epsilon_bucket_summary.csv`
- `darcy/optimizer_steps.csv`
- `darcy/eval_metrics.csv`
- `darcy/eval_split_summary.csv`
- `darcy/evaluation_passes.csv`
- `darcy/memory.csv`
- `darcy/attack_probe_epochs.csv`
- `darcy/attack_probe_samples.csv`
- `darcy/attack_probe_samples/*.npz`
- `darcy/checkpoints/*.pt` once checkpoint epochs are reached
- `darcy/config.json`, `run_config.json`, and final `summary.json` after completion

## Startup Evidence

Observed after launch:

- Driver log recorded start at `2026-06-07T18:40:28Z`.
- Dataset preflight passed: train=1, test=1, generalization=50.
- Output CSVs and attack probe NPZ files were created.
- At `2026-06-07T18:44:10Z`, latest observed progress was epoch `14/500`, step `56`, progress fraction `0.028`.
- CSV row counts at that check: `train_steps.csv` 57 lines, `attack_batches.csv` 57 lines, `eval_split_summary.csv` 57 lines, `attack_probe_epochs.csv` 15 lines.
- GPU monitoring showed Tesla V100-SXM2-32GB execution, approximately `16726 MiB / 32768 MiB` used and `100%` utilization while Burgers and Darcy were both active.

## Runtime Estimate

Observed early speed at epoch 14 was roughly `15-16` seconds per epoch, including per-epoch evaluation on train/test/50 generalization datasets. A rough ETA for epoch500 is around two hours after launch; speed may improve after the concurrent Burgers run finishes.

## Remaining Work

- Let tmux session `darcy_lossdrop50_loss3_500ep_20260607` continue.
- After completion, verify final `summary.json`, epoch500 checkpoint, CSV completeness, and attack probe files.
- Build visualization scripts or analysis notebooks from the recorded CSV/NPZ outputs.
## Progress Check - 2026-06-07T19:39Z

Observed from local files/logs:

- tmux session `darcy_lossdrop50_loss3_500ep_20260607` is still running.
- Current UTC check time: `2026-06-07T19:39:02Z`.
- Latest `eval_split_summary.csv` rows show epoch `265/500`, step `1060`, progress fraction `0.53`.
- Latest checkpoint observed: `darcy_epoch250_step001000.pt`; wall-time checkpoint observed: `darcy_epoch128_step000512_wall_0001800s.pt`.
- `nvidia-smi` showed Tesla V100-SXM2-32GB with `11834 MiB / 32768 MiB` used and about `95%` GPU utilization.
- Searching Darcy run logs for `Traceback`, `CUDA out of memory`, `out of memory`, `Killed`, `ERROR`, and `RuntimeError` returned no matches.

Runtime estimate:

- Based on launch at `2026-06-07T18:40:28Z` and epoch `265` at `2026-06-07T19:39:02Z`, observed speed is about `13.26` seconds per epoch.
- Remaining epochs: `235`.
- Estimated completion: about `2026-06-07T20:31Z`, roughly `52` minutes from the check time.

Inference:

- Darcy Flow Loss3 500-epoch run is healthy and about halfway done. No OOM or crash is evidenced.


## Status Check - 2026-06-07T22:19:52Z

Observed GPU state: `nvidia-smi` reported `30084 MiB / 32768 MiB` used, GPU utilization about `43%`, temperature `51C`, and no OOM/ECC error. The NVIDIA process table still did not list the Python PID, but `pgrep` showed the active Burgers runner PID `865719`.

Observed active sessions: `tmux list-sessions` showed `burgers_p2q2_20step_20260607` still running. No Darcy Flow training tmux session was active.

Observed Burgers progress: `progress.jsonl` had `27` completed batches out of the expected `84` model batches. Baseline completed all `10200` samples; loss1 epoch8000 had completed through sample range `2500:3000`. `summary.json` was not yet written, so the Burgers pilot was still running.

Observed latest Burgers batch: loss1 epoch8000 `2500:3000`, `54.2218s`, final loss mean `0.0028975329`, final delta RMS mean `0.1199999973`, peak allocated `28.5961 GiB`.

Inference: Burgers p2q2 pilot has not blown GPU memory and is progressing normally. At the current batch rate, expected completion is roughly `55-65` minutes after this check, plus small final FFT/write overhead.

Observed Darcy Flow status: `adversarial_training_runs/darcy_lossdrop50_loss3_500ep_fromscreen_20260607/summary.json` exists and reports finished UTC `20260607_200325_UTC`, `epochs=500`, final checkpoint `adversarial_training_runs/darcy_lossdrop50_loss3_500ep_fromscreen_20260607/darcy/checkpoints/darcy_epoch500_step002000.pt`, total wall `4975.1246s` (`82.9187 min`). Darcy peak allocated CUDA memory was `9335.814 MB`, current allocated at finish `3626.451 MB`, reserved `11288 MB`.

Inference: Darcy Flow completed normally before this Burgers status check and did not remain active while the current Burgers p2q2 pilot was using ~30 GiB.

## Generalization Loss Result - 2026-06-07T22:41Z

Observed source files:

- `adversarial_training_runs/darcy_lossdrop50_loss3_500ep_fromscreen_20260607/darcy/eval_split_summary.csv`
- `adversarial_training_runs/darcy_lossdrop50_loss3_500ep_fromscreen_20260607/darcy/eval_metrics.csv`
- `adversarial_training_runs/darcy_lossdrop50_loss3_500ep_fromscreen_20260607/darcy/summary.json`

Observed scope: this is clean evaluation on train/test/50 selected generalization datasets during the Darcy Flow loss3 self-training run. It is not an adversarial attack evaluation.

Observed final epoch500 generalization result versus baseline:

| Metric | Baseline epoch0 | Final epoch500 | Change |
|---|---:|---:|---:|
| generalization RMSE dataset mean | `0.0005717215820` | `0.0003075836784` | `46.20%` lower |
| generalization MAE dataset mean | `0.0004752948281` | `0.0002511947812` | `47.15%` lower |
| generalization relative L2 dataset mean | `0.09337999846` | `0.05023221956` | `46.21%` lower |
| generalization accuracy score dataset mean | `91.46002810` | `95.21719090` | `+3.7572` points |

Observed per-dataset result at epoch500: all `50/50` generalization datasets improved relative to baseline for RMSE, MAE, relative L2, and accuracy score. Final RMSE ratios across datasets ranged from `0.4933` to `0.5798` of baseline; final relative-L2 ratios matched the same range.

Observed best clean generalization point: epoch `479`, global step `1916`, not epoch500. At epoch479 the generalization RMSE dataset mean was `0.0001979864422`, relative L2 dataset mean was `0.03234040494`, and MAE dataset mean was `0.0001593592566`. Epoch500 remained much better than baseline but had partially regressed from this best point.

Observed train/test clean metrics at epoch500 moved in the opposite direction:

| Split | Metric | Baseline epoch0 | Final epoch500 | Direction |
|---|---|---:|---:|---|
| train | RMSE mean | `0.0002389861588` | `0.0004251050391` | `77.88%` higher |
| test | RMSE mean | `0.0003051046005` | `0.0004394615336` | `44.04%` higher |
| train | MAE mean | `0.0001717641692` | `0.0003802275410` | `121.37%` higher |
| test | MAE mean | `0.0002191964648` | `0.0003843123580` | `75.33%` higher |
| train | relative L2 mean | `0.03555337338` | `0.06324181391` | `77.88%` higher |
| test | relative L2 mean | `0.04471923445` | `0.06441195355` | `44.04%` higher |
| train | accuracy score mean | `96.56672710` | `94.05198205` | `-2.5147` points |
| test | accuracy score mean | `95.71949736` | `93.94858792` | `-1.7709` points |

Inference: the selected Darcy Flow lossdrop50 generalization set was directly helped by loss3 self-training, and the final model has substantially lower clean generalization loss than the baseline. The effect is not monotonic and the epoch500 checkpoint is not the best clean generalization checkpoint; epoch479 is better by the recorded eval metrics. The clean train/test degradation suggests this run specialized toward the selected generalization distribution rather than uniformly improving all splits.

## Loss1/Loss2 Comparison And Jacobian-SVD Feasibility Note - 2026-06-07T22:50Z

Observed comparison status:

- Local run directories show the completed current Darcy Flow lossdrop50 self-training run `adversarial_training_runs/darcy_lossdrop50_loss3_500ep_fromscreen_20260607`.
- Search of local source/result paths did not show matching Darcy Flow lossdrop50 self-training runs for loss1 or loss2 under the same 50 selected generalization datasets.
- The old Darcy Flow loss1/loss2/loss3 artifacts under `2D_Darcy_FNO2d/perturbation_results/...` are attack/evaluation visualizations, not directly comparable 500-epoch adversarial self-training runs on the current lossdrop50 selected dataset.

Observed code behavior:

- `tools/adversarial_training.py` exposes `--burgers-attack-loss-objective loss1/loss2/loss3` only for Burgers.
- For non-Burgers tasks, `attack_objective_target(...)` returns attacked solver output and labels it as `loss3` style: `MSE(model(x_adv), solver(x_adv))`.
- The current Darcy launcher `tools/run_darcy_lossdrop50_loss3_500ep_20260607.sh` uses `--tasks darcy`, `--training-data-mode adv-only`, `--label-mode solver`, `--darcy-attack-method binary_steepest_replace`, and no Darcy-specific loss1/loss2 switch.

Inference: the current Darcy Flow loss3 self-training has not yet been compared against matching Darcy Flow loss1/loss2 self-training runs. To make that comparison real, the code would need either new Darcy loss1/loss2 attack-objective support or separate launchers that implement Darcy-style analogues of loss1/loss2, followed by matched training/evaluation on the same 50 selected datasets.

Observed data shape for this run:

- Train dataset: `x (384, 85, 85)`, `y (384, 85, 85)`.
- Single-sample flattened input dimension: `7225`.
- Single-sample flattened output dimension: `7225`.
- A dense local Jacobian for one sample is therefore `7225 x 7225`, or `52,200,625` entries.
- Stored as float32, one dense Jacobian is about `199 MiB`; model, solver, and error Jacobians together are about `597 MiB` before SVD work arrays. Stored/processed as float64, the memory roughly doubles.

Comparison with the Burgers Jacobian/SVD path:

- The existing Burgers dense-Jacobian helper computes a row-wise explicit Jacobian by one backward pass per output coordinate.
- Burgers `1024 x 1024` needs `1024` row-wise VJP/backward passes per operator per sample.
- Current Darcy `85 x 85` would need `7225` row-wise passes per operator per sample, about `7.1x` more rows than Burgers, with a much heavier 2D model/solver path.
- If using a true `400 x 400` grid, the flattened dimension would be `160000`, and a dense Jacobian would be `160000 x 160000`, about `25.6 billion` entries, roughly `102 GB` for one float32 Jacobian alone. That is not practical for this workflow.

Inference: full dense Jacobian plus SVD is not a sensible default diagnostic for Darcy Flow. For the current `85 x 85` run, one or a few dense sample Jacobians might be possible but expensive and not suitable for all datasets/checkpoints. For larger 2D grids it is effectively infeasible. A better diagnostic is matrix-free top-k probing with JVP/VJP or finite-difference JVP plus VJP, or smaller directional probes that estimate top singular values/vectors and gradient alignment without materializing the full matrix.


## Burgers Attack Plus Darcy Flow Self-Training Memory Assessment - 2026-06-07T22:53Z

Observed from `nvidia-smi` at `2026-06-07T22:52:47Z`: Tesla V100-SXM2-32GB reported `30084 MiB / 32768 MiB` used, about `42%` GPU utilization, and no OOM/ECC error shown. The NVIDIA process table did not list the Python PID, but `pgrep` showed the active Burgers attack runner PID `865719` under tmux session `burgers_p2q2_20step_20260607`.

Observed from `forensics/burgers_round03_full_p2q2_52datasets_4models_finalonly_20step_20260607/progress.jsonl`: the Burgers p2q2 final-only 20-step attack had `64/84` completed batch rows. The latest completed batch was `loss3_epoch1500` samples `0:500`, with `peak_allocated_gib=28.59606409072876`. Recent batches took roughly `54-59` seconds.

Observed from `adversarial_training_runs/darcy_lossdrop50_loss3_500ep_fromscreen_20260607/summary.json`: the completed Darcy Flow loss3 self-training run used `batch_size=96`, `attack_batch_size=96`, `optimizer_batch_size=24`, and reported CUDA memory `cuda_peak_allocated_mb=9335.81396484375`, `cuda_reserved_mb=11288.0`, and final `cuda_allocated_mb=3626.45068359375`.

Inference: starting a same-settings Darcy Flow self-training run while the current Burgers attack is active is not safe on this 32GB V100. The card has only about `2684 MiB` free by `nvidia-smi`, while the prior Darcy loss3 run alone needed about `9.1 GiB` peak allocated and reserved about `11.0 GiB`. A same-batch Darcy loss1/loss2 run would therefore be expected to OOM or force allocator failure if launched concurrently with the current Burgers attack.

Inference: after the Burgers attack finishes and releases memory, one Darcy Flow self-training run at the previous loss3 settings should fit based on the observed `~11.3GB` reserved footprint. Running Darcy loss1 and Darcy loss2 at the same time without Burgers might fit in raw memory if their profiles match loss3, but it is still risky because two runs would reserve roughly `22GB` before transient evaluation/attack overhead and allocator fragmentation. Sequential loss1 then loss2 is the safer default. If concurrency is required, reduce Darcy `attack_batch_size`, `batch_size`, and `optimizer_batch_size` first and run a short GPU-memory smoke test before a full run.

Observed implementation status: the current completed Darcy run is loss3-style. The existing `--burgers-attack-loss-objective loss1/loss2/loss3` switch is Burgers-specific, so true Darcy Flow loss1/loss2 self-training needs a Darcy-specific objective implementation or launcher adjustment before official loss1/loss2 runs are comparable.


## Accuracy Score Definition Audit - 2026-06-07T22:58Z

Observed from `tools/adversarial_training.py:35-40`: the adversarial-training evaluator imports `evaluate_dataset` from `tools.evaluate_generalization_models`.

Observed from `tools/evaluate_generalization_models.py:152-155`: per dataset, `rmse = sqrt(sse / count)`, `mae = sae / count`, `relative_l2 = sqrt(sse / max(target_sse, 1e-20))`, and `accuracy_score = 100.0 / (1.0 + relative_l2)`.

Observed from `tools/evaluate_generalization_models.py:407`: the code documents this as an accuracy-like score for regression PDE tasks, not classification accuracy. Higher is better because lower relative L2 maps to a larger score.

Observed from `tools/adversarial_training.py:2133-2156`: `eval_split_summary.csv` averages per-dataset `accuracy_score` values across a split. Therefore `accuracy_score_dataset_mean` is `mean(100/(1+relative_l2_dataset))`, not necessarily exactly `100/(1+relative_l2_dataset_mean)` for multi-dataset splits such as the 50 Darcy generalization datasets.

Inference: in the Darcy Flow loss3 self-training records, `accuracy_score` should be interpreted only as a monotone rescaling of relative L2 for plots/tables. It is not classification accuracy and should not be used as an independent metric from relative L2.
