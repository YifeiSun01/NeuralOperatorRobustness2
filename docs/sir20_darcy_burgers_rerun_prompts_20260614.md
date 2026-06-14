# SIR20 Darcy/Burgers Rerun Prompts - 2026-06-14

This file contains two cleaned execution prompts derived from the current local
repository state and the latest experiment request. Do not paste cloud or GitHub
secrets into source, docs, logs, command history, or reports. Use environment
variables for upload credentials.

## Prompt 1 - Darcy Flow SIR20 Work-Clock Matched 7-Model Rerun

You are working in `/workspace/NeuralOperatorRobustness2` on the
`vast-ai-darcy-flow` branch. The task is Darcy Flow, not Burgers. Use the
existing Darcy artifacts as context, especially:

- `visualizations/darcy_random_inclusive_burgers_style_bundle_20260613/`
- `analysis_outputs/darcy_complete_7model_statistics_20260613/`
- `docs/darcy_random_inclusive_burgers_style_bundle_20260613.md`
- `docs/darcy_complete_7model_statistics_20260613.md`
- existing tools under `tools/` with names containing `darcy`, `timematched`,
  `random_source`, `metric_correlation`, `attack20`, and `burgers_style_bundle`.

First audit local and R2 artifacts. If a required result already exists and has
the required coverage, reuse it and record the evidence. If anything is missing,
run only the missing part. Do not invent results.

Preflight:

1. Activate `adv_robust` and verify GPU-only execution. Record `nvidia-smi`,
   PyTorch version/CUDA/device, and JAX backend/devices if JAX is used.
2. Confirm the working branch is `vast-ai-darcy-flow`.
3. Confirm no real credentials are written into tracked files. Upload credentials
   must come from environment variables such as `R2_ENDPOINT_URL`,
   `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `R2_BUCKET`, `R2_PREFIX`, and
   a Git credential already configured outside tracked files.

Models/methods:

- Baseline: evaluation-only reference model.
- Six trained methods: `loss1`, `loss2`, `loss3`, `physics_loss`,
  `random_clean_y`, and `random_solver_y`.
- Darcy includes physics loss. Do not add Burgers-only assumptions.

Write modular code, then a top-level orchestrator. Keep modules small and
composable:

- artifact discovery and coverage audit;
- timing probe and epoch-plan calculation;
- smoke run;
- training/resume;
- per-epoch 52-dataset evaluation logging;
- final clean 52-dataset evaluation;
- 52-dataset x 50-sample adversarial attack;
- fixed-sample Jacobian/SVD/`J^T error` robustness analysis;
- correlations and vector-angle/similarity analysis;
- plotting;
- upload/backup;
- Markdown/CSV/JSON report generation.

Work-clock matching:

- Work-clock means only the time spent generating the training perturbation or
  random source data, solver calls needed to obtain training targets, attack
  optimization, and model optimizer/backpropagation updates.
- Evaluation, plotting, checkpoint serialization, upload, and bookkeeping must
  be logged separately and must not be counted in training work-clock.
- Run a 10-30 epoch timing probe for every trained method. Ignore warm-up epochs
  when estimating stable per-epoch work-clock.
- Set `loss3` as the reference at 3000 epochs. Compute
  `target_work_seconds = stable_loss3_work_seconds_per_epoch * 3000`.
- For every other trained method, compute
  `target_epochs = round(target_work_seconds / stable_method_work_seconds_per_epoch)`.
- The expected Darcy range is roughly 3000-3600 epochs, but use measured timings.
- Runs may be parallelized only when GPU memory and throughput make it safe.

Optimizer continuity:

- The full 1-to-final curve must not have an artificial jump at epoch 1000 or
  any continuation boundary.
- If resuming, load and save model weights, Adam optimizer state, scheduler/scaler
  state if present, RNG state, epoch number, work-clock totals, and metric logs.
- Never resume by loading only model weights and reinitializing Adam unless the
  report explicitly labels the curve as a new training run.

Smoke test:

- Before the long run, run a tiny end-to-end smoke for each method: timing,
  one or a few training epochs, evaluation on a small subset, checkpoint save and
  reload, a tiny attack, and a tiny robustness sample.
- Only start the long run after all smokes pass.
- Start the long run as a managed/nohup/supervisor-style background process,
  monitor for about 10 minutes, then stop monitoring while leaving the job
  running.

Training/evaluation logging:

- At every epoch for each trained method, evaluate and record all 52 datasets:
  original train, original test, and all 50 Darcy generalization datasets.
- Record at least: method, epoch, workclock_seconds, true_elapsed_seconds,
  eval_wall_seconds, split, dataset_id/name, sample count, RMSE, relative L2,
  and regression accuracy score. If there is no native accuracy, use and record
  `accuracy_score = 100 / (1 + relative_l2)`.
- Keep final 52-dataset RMSE/relative-L2/accuracy CSVs for all 7 models,
  including baseline.

Final adversarial attack:

- Run post-training adversarial attack only on final saved checkpoints.
- For each of the 52 datasets, choose 50 samples and run one batched attack per
  dataset when possible.
- Record clean loss before attack, attacked loss after attack, absolute and
  relative loss increase, delta arrays or delta summaries, delta L2/RMS,
  flip fraction for binary Darcy, and sample identifiers.
- The 52 x 50 attack sample manifest must include every fixed robustness sample
  listed below so correlations compare identical initial conditions.

Fixed robustness sample manifest:

- Use exactly the same fixed sample manifest for all 7 models.
- Select 25 samples total: 2 train samples, 2 test samples, and 21 samples from
  21 fixed generalization datasets, one sample per selected generalization set.
- Save the manifest as CSV and JSON with split, dataset name, dataset index,
  sample index, and any seed.
- Ensure these exact 25 sample identities are included in the 52 x 50 attack
  sample set.

Robustness metrics on fixed samples:

- For each fixed sample and each model, compute the error-function Jacobian
  diagnostics:
  `J`, top singular value/vector from SVD or a documented top-SVD approximation,
  `J^T error`, `||J^T error||`, `||J^T error||^2`, first-order binary gain proxy,
  and any `sigma * ||error||` style proxy already used in the repo.
- Compare vectors for the same model and same initial condition:
  top singular vector, `J^T error` direction, and final attack delta. Save cosine
  similarities and angles.
- Save correlations between loss increase, top singular value/proxy, and
  `||J^T error||`/first-order proxies. Include global, per-model, per-split, and
  within-sample-centered correlations where possible.

Visualization requirements:

- Produce only the requested curve families plus 2D attack heatmaps. Do not add
  extra grad-norm, optimizer-loss, bar-chart, or metric-summary figures unless
  they are data tables only.
- For RMSE and relative L2, make:
  - epoch x-axis, train/test/generalization-mean curves;
  - work-clock x-axis, train/test/generalization-mean curves;
  - epoch x-axis, 50 generalization datasets split into two 25-dataset panels;
  - work-clock x-axis, 50 generalization datasets split into two 25-dataset panels.
- Every RMSE and relative-L2 plot must include a horizontal baseline reference
  line for the original baseline model metric on that same split/dataset.
- For work-clock plots, clip the x-axis to the largest common work-clock shared
  by all compared methods, so all curves reach the right edge together.
- Rebuild Darcy 2D attack heatmaps using the existing style. Select examples
  where `loss3` looks robust or has smaller loss growth when possible, and record
  the selection rule rather than hand-waving.

Output organization:

- Create one clearly named root, for example
  `analysis_outputs/sir20_darcy_workclock_matched_YYYYMMDD/`.
- Under it, use clear subfolders such as `data/`, `figures/`, `reports/`,
  `logs/`, and `manifests/`. Put generated checkpoints or large arrays under a
  path that is uploaded to R2, not Git.
- Save CSV, JSON, and Markdown summaries for generalization, robustness,
  correlations, vector similarities, epoch plans, timing probes, and coverage.
- Save figures in a matching
  `visualizations/sir20_darcy_workclock_matched_YYYYMMDD/` tree if the existing
  visualization convention expects `visualizations/`.

Backup and records:

- Update `EXPERIMENT_LEDGER.md` and a dedicated `docs/` report with observed
  evidence, not guesses.
- Commit/push only code, shell scripts, and Markdown to Git on
  `vast-ai-darcy-flow`. Do not Git-add checkpoints, large arrays, or generated
  bulk images unless explicitly requested.
- Upload bulky outputs, model checkpoints, CSV/JSON result tables, and figures
  to R2 using environment-provided credentials. Include a manifest of uploaded
  paths. Do not print or store secrets.

## Prompt 2 - Burgers SIR20 Work-Clock Matched Audit/Reuse/Rerun

You are working in `/workspace/NeuralOperatorRobustness2` on the `vast-ai`
branch. The task is 1D Burgers. Use existing Burgers artifacts first; do not
retrain if local or R2 data already covers the requested analysis. Important
local references include:

- `docs/burgers_loss3_selective_round03_complete_report_20260605.md`
- `docs/burgers_loss3_selective_round03_wall_clock_analysis_20260605.md`
- `docs/burgers_round03_generalization_5x5_wallclock_panels_20260608.md`
- `docs/burgers_loss3targeted_52dataset_clean_metrics_20260611.md`
- `docs/burgers_wideparam_loss123_retrain_report_20260611.md`
- `docs/burgers_wideparam_full1024_svd_attack25_reuse3_workflow_20260611.md`
- existing tools under `tools/` with names containing `burgers`, `round03`,
  `loss123`, `svd_attack`, `correlation`, and `wallclock`.

First audit local and R2 artifacts. If complete Burgers training curves,
final checkpoints, 52-dataset metrics, SVD/Jacobian samples, attack samples,
and visualizations already exist with matching sample identities and acceptable
work-clock coverage, reuse them and only rebuild the missing summaries/figures.
If coverage is missing, run only the missing pieces.

Preflight:

1. Activate `adv_robust` and verify GPU-only execution. Record `nvidia-smi`,
   PyTorch version/CUDA/device, and JAX backend/devices if JAX is used.
2. Confirm the working branch is `vast-ai`.
3. Use environment variables for R2/Git credentials. Do not write any secret
   values into tracked files, logs, reports, or command history.

Models/methods:

- Baseline: evaluation-only reference model.
- Burgers trained methods: `loss1`, `loss2`, `loss3`, `random_clean_y`,
  and `random_solver_y` if the existing Burgers protocol has both random
  methods.
- Burgers does not use Darcy `physics_loss`; do not add a physics-loss Burgers
  row unless an existing, documented Burgers implementation actually supports it.

Code structure:

- Reuse existing Burgers scripts where possible, but add a top-level orchestrator
  that performs artifact audit, R2 lookup, timing/epoch planning, optional smoke,
  optional training, final evaluation, attack, fixed-sample robustness analysis,
  plotting, and upload.
- Keep the workflow modular in the same way as the Darcy prompt.

Work-clock matching:

- Work-clock means only perturbation/random generation, solver calls needed for
  training targets, attack optimization, and optimizer/backpropagation updates.
- Evaluation, plotting, checkpointing, upload, and bookkeeping are not training
  work-clock and must be logged separately.
- Burgers reference target: `loss3` for about 8 hours of work-clock, or the
  closest existing documented loss3 run if it is already within acceptable
  coverage. The user expects this may correspond to roughly 1000 loss3 epochs,
  but compute it from measured stable timing rather than assuming.
- Run or reuse a 10-30 epoch timing probe for each method. Ignore warm-up epochs.
- Compute `target_work_seconds` from the measured/reused loss3 reference and
  calculate other methods' target epochs as
  `round(target_work_seconds / stable_method_work_seconds_per_epoch)`.
- For final model analysis and heatmaps, exact epoch equality is not required if
  work-clock is within about +/-25%; for training curves, use the measured
  work-clock axis and clip shared plots to the common work-clock endpoint.

Optimizer continuity:

- If Burgers existing curves already resume cleanly with continuous Adam state,
  do not rerun training just to recreate them.
- If any continuation is needed, resume with model weights, Adam optimizer state,
  scheduler/scaler state if present, RNG state, epoch, logs, and work-clock
  totals. Do not reset Adam at a continuation boundary.

Smoke/reuse decision:

- If retraining is needed, smoke each method end-to-end before the long run.
- If no retraining is needed, smoke only the plotting/statistics rebuild against
  existing artifacts.
- Record explicitly which outputs were reused, which were regenerated, and which
  were absent from local/R2.

Training/evaluation logging:

- At every epoch for each trained method, record RMSE, relative L2, and
  regression accuracy score on train, test, and every available Burgers
  generalization dataset in the selected 52-dataset protocol.
- If there is no native accuracy, use and record
  `accuracy_score = 100 / (1 + relative_l2)`.
- Save final 52-dataset RMSE/relative-L2/accuracy CSVs for all models including
  baseline.

Final adversarial attack:

- Run final-model attack analysis only after final checkpoints are fixed, or
  reuse existing attack outputs if they have matching sample identities.
- For each dataset, select 50 samples where feasible and record clean loss,
  attacked loss, loss increase, relative increase, delta fields/summaries, and
  sample identifiers.
- The attack sample set must include all fixed robustness samples below.

Fixed robustness sample manifest:

- Use the same fixed sample manifest for all models: 2 train, 2 test, and 21
  generalization samples, one sample from each selected generalization dataset.
- Save manifest as CSV/JSON and use it consistently in SVD/Jacobian, `J^T error`,
  and attack-delta comparisons.

Robustness and similarity:

- For each fixed sample/model, compute or reuse: top singular value/vector of the
  error Jacobian, `J^T error`, `||J^T error||`, related first-order gain proxies,
  final attack loss increase, and final attack delta.
- Save correlations between loss increase, singular-value/proxy metrics, and
  `J^T error` metrics. Include global, per-model, per-split, and
  within-sample-centered summaries where possible.
- Save vector cosine similarities and angles among the top singular direction,
  `J^T error` direction, and the actual attack delta for the same model and
  same initial condition.

Visualization requirements:

- Draw only RMSE and relative-L2 training/generalization curves and the requested
  attack heatmaps. Do not add extra grad-norm, optimizer-loss, miscellaneous
  metric-summary, or bar-chart figures unless they are saved only as tables.
- For both RMSE and relative L2, produce:
  - epoch x-axis train/test/generalization-mean curves;
  - work-clock x-axis train/test/generalization-mean curves;
  - epoch x-axis 50-generalization-dataset curves split into 25+25 panels;
  - work-clock x-axis 50-generalization-dataset curves split into 25+25 panels.
- Every curve plot must include a horizontal baseline reference line showing the
  original baseline model RMSE or relative L2 for that same split/dataset.
- For work-clock plots, clip to the shared maximum common work-clock so all
  method curves reach the right edge.
- If existing Burgers curves are already smooth and optimizer-continuous, reuse
  them and regenerate only cleaner figures/reports.

Output organization:

- Create a clear standalone output root such as
  `analysis_outputs/sir20_burgers_workclock_matched_YYYYMMDD/`.
- Use `data/`, `figures/`, `reports/`, `logs/`, and `manifests/` subfolders, or
  mirror the repo's existing Burgers visualization convention under
  `visualizations/sir20_burgers_workclock_matched_YYYYMMDD/`.
- Save CSV, JSON, and Markdown summaries for artifact coverage, epoch plans,
  final 52-dataset metrics, attacks, robustness metrics, correlations, vector
  angles, and plot manifests.

Backup and records:

- Update `EXPERIMENT_LEDGER.md` and a dedicated `docs/` report.
- Commit/push code, shell scripts, and Markdown to Git on `vast-ai`.
- Upload bulky outputs, checkpoints, arrays, result tables, and figures to R2
  using environment variables. Save an upload manifest. Do not expose secrets.
