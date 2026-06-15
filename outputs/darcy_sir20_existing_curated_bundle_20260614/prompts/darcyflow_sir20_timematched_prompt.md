# DarcyFlow / SIR20 Time-Matched Experiment Prompt

Work in `/workspace/NeuralOperatorRobustness2` using virtualenv `adv_robust`.
Read `AGENTS.md` and every `/etc/vast_agents/*.md` first.

Goal: run or assemble the DarcyFlow/SIR20 seven-model study:
baseline plus six trained models: `loss1`, `loss2`, `loss3`, `physics loss`,
`random clean`, `random solver`.

Do not store secrets in code, logs, markdown, or git history. Use environment
variables only: `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_ENDPOINT`,
`GITHUB_TOKEN`.

## Timing

Use `loss3` at 3000 epochs as the reference budget. First run timing calibration
for every training method for 10-30 epochs. Ignore the first 1-2 warmup epochs
and estimate stable work-clock seconds per epoch.

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

Compute:

- stable work-clock seconds per epoch for each method;
- `loss3` 3000-epoch total work-clock;
- time-matched epoch counts for `loss1`, `loss2`, `physics loss`,
  `random clean`, `random solver`.

Save timing outputs as CSV, JSON, and markdown.

## Training

Before the full run, smoke test every method with tiny epochs and verify:

- training starts and finishes;
- all 52 datasets evaluate;
- checkpoints save;
- final analysis can read checkpoints;
- figures render;
- upload scripts do not crash in dry-run or safe mode.

Then launch the full run. Track logs for 10 minutes, then stop active tracking
while the process continues.

For every trained method:

- train to the time-matched epoch count or equivalent max-work-seconds;
- record every epoch evaluation on 52 datasets:
  train, test, and 50 generalization datasets;
- record RMSE, Relative L2, MAE, accuracy/available finite metric;
- save complete checkpoints containing model state and optimizer state.

Checkpoint resume must restore Adam/AdamW optimizer state. A `1000+2000` resume
must not reset optimizer moments and must not create a loss-curve jump.

## Final Evaluation

After training, evaluate baseline plus six final models on all 52 datasets.
Record per-dataset RMSE and Relative L2 as CSV/JSON.

## Robustness

For each final trained model and each of the 52 datasets:

- deterministically select 50 samples;
- include the 25 SVD/Jacobian samples inside this 50-sample manifest;
- run one-batch adversarial attack;
- record clean loss, adversarial loss, loss increase, relative increase;
- save attack delta arrays.

For SVD/Jacobian:

- select exactly 25 fixed samples:
  train 2, test 2, generalization 21 datasets x 1 sample;
- every model must use exactly the same dataset id, sample index, and initial condition;
- compute top singular value/vector for the error Jacobian;
- compute `J^T error`, its norm, and vector;
- record singular vector, `J^T error` vector, attack delta vector;
- compute cosine similarity, angle, and pairwise vector correlations;
- compute sample-wise correlations between singular value, `||J^T error||`,
  and adversarial loss increase.

## Visualization

Only draw the required figures:

- RMSE and Relative L2 vs epoch:
  train/test/generalization mean in one figure;
- RMSE and Relative L2 vs work-clock:
  train/test/generalization mean in one figure;
- RMSE and Relative L2 vs epoch:
  50 generalization datasets split into two 25-panel pages;
- RMSE and Relative L2 vs work-clock:
  50 generalization datasets split into two 25-panel pages;
- Darcy 2D attack heatmap, selecting a sample where `loss3` has visibly smaller
  loss increase / stronger robustness.

Every loss curve must include the baseline model as a horizontal line.

For work-clock plots, truncate the x-axis to the largest common work-clock time
available to all methods, so no curve extends farther than the others.

Do not draw grad norm, optimizer loss, bar charts, extra summaries, or zoom
figures unless explicitly requested.

## Output Bundle

Create one prominent folder:

`outputs/darcy_sir20_timematched_full_<date>/`

with subdirectories:

- `figures/`
- `data/`
- `checkpoints_manifest/`
- `reports/`
- `logs/`

Archive every CSV, JSON, markdown, PNG, manifest, and upload log there.

## Upload

Commit only code, shell scripts, and markdown to GitHub branch:

`vast-ai-darcy-flow`

Upload large outputs and figures to Cloudflare R2 prefix:

`neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected`

Use environment variables for all credentials. Never commit tokens or secrets.
