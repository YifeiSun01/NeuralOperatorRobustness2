# Darcy Flow Self-Training And Generalization Summary - 2026-06-08

## Source Evidence

Observed source reports:

- `docs/darcy_lossdrop50_loss3_500ep_launch_20260607.md`
- `docs/darcy_candidate_generalization_gradient_screen_20260607.md`
- `docs/darcy_lossdrop50_selected_generalization_suite_20260607.md`
- `docs/darcy_flow_naming_audit_20260607.md`

Observed run/source paths:

- Run directory: `adversarial_training_runs/darcy_lossdrop50_loss3_500ep_fromscreen_20260607`
- Generalization root: `generalization_datasets_darcy_lossdrop50_selected_20260607`
- Runner: `tools/run_darcy_lossdrop50_loss3_500ep_20260607.sh`
- Dataset generation/screening tools: `tools/generate_darcy_lossdrop50_pool_20260607.py`, `tools/select_darcy_lossdrop50_20260607.py`, `tools/probe_darcy_gradient_alignment_screen.py`

## Naming Audit

The correct dataset/task name is **Darcy Flow**, not Daisy Flow or DiverseFlow. Markdown, plot titles, and generated filenames should use `Darcy Flow` / `darcy` consistently.

## Loss3 500-Epoch Run

Observed run identity:

- Run name: `darcy_lossdrop50_loss3_500ep_fromscreen_20260607`.
- Epochs: `500`.
- Attack objective: non-Burgers/Darcy path uses loss3-style target, `MSE(model(x_adv), solver(x_adv))`.
- Attack method: `binary_steepest_replace`.
- Attack steps: `1`.
- Batch size: `96`.
- Optimizer microbatch size: `24`.
- Evaluation: train/test plus 50 selected generalization datasets.

Observed completion:

- Finished UTC: `20260607_200325_UTC`.
- Final checkpoint: `adversarial_training_runs/darcy_lossdrop50_loss3_500ep_fromscreen_20260607/darcy/checkpoints/darcy_epoch500_step002000.pt`.
- Total wall time: `4975.1246s`, about `82.9187 min`.
- CUDA peak allocated: `9335.814 MB`.
- CUDA reserved at finish: `11288 MB`.

## Clean Generalization Result

Observed from `eval_split_summary.csv` and `eval_metrics.csv`: clean evaluation, not adversarial attack evaluation.

Generalization epoch0 to epoch500:

| Metric | Epoch0 | Epoch500 | Change |
|---|---:|---:|---:|
| RMSE dataset mean | `0.0005717215820` | `0.0003075836784` | `46.20%` lower |
| MAE dataset mean | `0.0004752948281` | `0.0002511947812` | `47.15%` lower |
| relative L2 dataset mean | `0.09337999846` | `0.05023221956` | `46.21%` lower |
| accuracy score dataset mean | `91.46002810` | `95.21719090` | `+3.7572` points |

Observed per-dataset result: all `50/50` selected generalization datasets improved at epoch500 for RMSE, MAE, relative L2, and accuracy score.

Best observed clean generalization checkpoint:

- Epoch `479`, global step `1916`.
- RMSE dataset mean: `0.0001979864422`.
- Relative L2 dataset mean: `0.03234040494`.
- MAE dataset mean: `0.0001593592566`.

Inference: epoch500 remains much better than baseline but has regressed from the best clean generalization point.

## Train/Test Degradation

Observed epoch0 to epoch500 train/test clean metrics:

| Split | Metric | Epoch0 | Epoch500 | Direction |
|---|---|---:|---:|---|
| train | RMSE | `0.0002389861588` | `0.0004251050391` | `77.88%` higher |
| test | RMSE | `0.0003051046005` | `0.0004394615336` | `44.04%` higher |
| train | MAE | `0.0001717641692` | `0.0003802275410` | `121.37%` higher |
| test | MAE | `0.0002191964648` | `0.0003843123580` | `75.33%` higher |
| train | relative L2 | `0.03555337338` | `0.06324181391` | `77.88%` higher |
| test | relative L2 | `0.04471923445` | `0.06441195355` | `44.04%` higher |
| train | accuracy score | `96.56672710` | `94.05198205` | `-2.5147` points |
| test | accuracy score | `95.71949736` | `93.94858792` | `-1.7709` points |

Inference: loss3 self-training specialized toward the selected 50 generalization datasets rather than uniformly improving train/test/generalization.

## Accuracy Score Definition

Observed code definition:

\[
\mathrm{accuracy\_score}=\frac{100}{1+\mathrm{relative\_L2}}.
\]

This is an accuracy-like regression score, not classification accuracy. It is a monotone rescaling of relative L2 for tables/plots.

For split means over many datasets, the code averages per-dataset accuracy scores:

\[
\mathrm{mean}\left(\frac{100}{1+\mathrm{relative\_L2}_{d}}\right),
\]

which is not necessarily exactly

\[
\frac{100}{1+\mathrm{mean}(\mathrm{relative\_L2}_d)}.
\]

## Loss1/Loss2 Comparison Status

Observed local status: no matching Darcy Flow loss1/loss2 500-epoch self-training runs on the same selected 50 lossdrop datasets were found.

Observed code behavior: `--burgers-attack-loss-objective loss1/loss2/loss3` is Burgers-specific. Non-Burgers/Darcy attack target currently follows loss3-style `MSE(model(x_adv), solver(x_adv))`.

Inference: a true Darcy loss1/loss2 comparison requires Darcy-specific objective implementation or launchers before results are comparable.

## Dense Jacobian/SVD Feasibility

Observed current Darcy data shape:

- Train `x`: `(384, 85, 85)`.
- Train `y`: `(384, 85, 85)`.
- Flattened dimension: `7225`.
- Dense single-sample Jacobian: `7225 x 7225`, or `52,200,625` entries.

Memory estimate:

- One float32 dense Jacobian: about `199 MiB`.
- Model, solver, and error Jacobians together: about `597 MiB` before SVD work arrays.

Comparison:

- Burgers `1024 x 1024` row-wise dense Jacobian needs 1024 backward/VJP rows.
- Darcy `85 x 85` would need 7225 rows per operator per sample, about `7.1x` more rows, with heavier 2D model/solver work.
- A true `400 x 400` grid would imply dimension `160000`, dense Jacobian `160000 x 160000`, about `25.6 billion` entries or roughly `102 GB` for one float32 matrix alone.

Inference: full dense Darcy Jacobian/SVD is not a sensible default. Matrix-free JVP/VJP top-k methods or smaller directional probes are the better path.

## Concurrent Memory Note

Observed while Burgers p2q2 pilot was active: GPU memory around `30084 MiB / 32768 MiB`; prior Darcy loss3 run alone needed about `9.1 GiB` peak allocated and `11.0 GiB` reserved.

Inference: do not launch same-settings Darcy self-training concurrently with the active Burgers batch-500 p2q2 attack. After Burgers releases memory, one Darcy run should fit; two simultaneous Darcy runs may be risky without batch-size reduction and a short memory smoke test.
