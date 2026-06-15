# Burgers Time-Matched Experiment Prompt

Work in `/workspace/NeuralOperatorRobustness2` using virtualenv `adv_robust`.
Read `AGENTS.md` and every `/etc/vast_agents/*.md` first.

Goal: assemble or run the Burgers time-matched adversarial-training study.
Before rerunning anything, search local files and the configured R2 backup for
complete existing artifacts. If complete training, final evaluation, robustness,
and figures already exist, reuse them instead of retraining.

Do not store secrets in code, logs, markdown, or git history. Use environment
variables only: `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_ENDPOINT`,
`GITHUB_TOKEN`.

## Methods

Burgers does not use Darcy `physics loss`. Use:

- baseline;
- `loss1`;
- `loss2`;
- `loss3`;
- `random clean`;
- `random solver`, if the Burgers workflow has an equivalent random-solver target.

If an existing Burgers workflow lacks either random method, document the missing
method explicitly instead of silently substituting another method.

## Timing

Use `loss3` as the reference training method. For Burgers, target the existing
long-run reference budget, approximately 8 hours of work-clock for `loss3`
unless the repository/R2 artifacts show a more authoritative reference.

First run or recover timing calibration for every method. Ignore the first 1-2
warmup epochs and estimate stable work-clock seconds per epoch.

Work-clock must include only:

- adversarial delta generation;
- random delta generation;
- solver target generation;
- model forward/backward;
- optimizer step.

Work-clock must exclude:

- train/test/generalization evaluation;
- plotting;
- checkpoint upload;
- R2/GitHub upload;
- reporting.

Compute time-matched epoch counts for all non-baseline methods and save timing
CSV, JSON, and markdown.

## Training

If existing artifacts already cover the needed time-matched run within roughly
25 percent of the reference work-clock budget, do not rerun training. Reuse and
document the selected checkpoints.

If rerunning is necessary:

- smoke test every method first;
- train each method to the time-matched epoch count or max-work-seconds;
- record every epoch evaluation on train/test/generalization datasets;
- save full checkpoints with model and optimizer state;
- resume must restore Adam/AdamW optimizer state, not just model weights.

## Final Evaluation

Evaluate baseline plus trained final models on train/test/generalization datasets.
Record RMSE, Relative L2, MAE, accuracy/available finite metric as CSV/JSON.

## Robustness

For every final model and every evaluation dataset:

- deterministically select 50 samples;
- include the fixed SVD/Jacobian samples inside this 50-sample manifest;
- run one-batch adversarial attack;
- record clean loss, adversarial loss, loss increase, relative increase;
- save attack deltas.

For SVD/Jacobian, use a fixed small manifest analogous to Darcy:

- train 2 samples;
- test 2 samples;
- 21 generalization datasets x 1 sample when enough generalization datasets exist.

All models must use the exact same dataset id, sample index, and initial condition.
Compute singular value/vector, `J^T error`, `||J^T error||`, attack delta,
pairwise vector cosine similarity/angle/correlation, and sample-wise correlations
between singular value, `||J^T error||`, and loss increase.

## Visualization

Only draw the required figures:

- RMSE and Relative L2 vs epoch:
  train/test/generalization mean in one figure;
- RMSE and Relative L2 vs work-clock:
  train/test/generalization mean in one figure;
- RMSE and Relative L2 vs epoch:
  generalization datasets split into 25-panel pages;
- RMSE and Relative L2 vs work-clock:
  generalization datasets split into 25-panel pages;
- attack visualization appropriate for Burgers, selecting a sample where `loss3`
  has visibly smaller loss increase / stronger robustness.

Every loss curve must include the baseline model as a horizontal line.

For work-clock plots, truncate the x-axis to the largest common work-clock time
available to all methods. Do not draw grad norm, optimizer loss, bar charts,
extra summaries, or zoom figures unless explicitly requested.

## Output Bundle

Create one prominent folder:

`outputs/burgers_timematched_full_<date>/`

with subdirectories:

- `figures/`
- `data/`
- `checkpoints_manifest/`
- `reports/`
- `logs/`

Archive every CSV, JSON, markdown, PNG, manifest, and upload log there.

## Upload

Commit only code, shell scripts, and markdown to GitHub branch:

`vast-ai`

Upload large outputs and figures to Cloudflare R2 prefix:

`neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected`

Use environment variables for all credentials. Never commit tokens or secrets.
