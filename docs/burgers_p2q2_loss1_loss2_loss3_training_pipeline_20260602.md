# Burgers P2Q2 Loss1/Loss2/Loss3 Adversarial Training Pipeline

Date: 2026-06-02

This note records the three Burgers attack objectives and the corrected one-command pipeline that can run the same 1D Burgers adversarial training workflow with loss1, loss2, or loss3.

## Core Distinction

The difference among loss1, loss2, and loss3 is in the adversarial attack objective, not in the downstream bookkeeping pipeline.

The shared pipeline still does the same downstream work:

- Train for 1000 epochs by default.
- Use Burgers P=2, Q=2 RMS-L2 attack geometry through `fast_replace_l2`.
- Save model checkpoints every 200 epochs and at the final epoch.
- Save fixed-index attack probe samples every epoch.
- Evaluate train/test/50 generalization datasets every epoch.
- Write loss curves, epsilon bucket statistics, delta spectrum statistics, and runtime CSV/JSON files.
- Generate the polished visualizations.
- Run checkpoint-series Jacobian/SVD analysis.
- Render the baseline-vs-epoch1000 attack GIF/PNG diagnostics.
- Upload configured artifacts to R2 and push lightweight code/Markdown/figure summaries to GitHub when credentials are available.

## Three Attack Objectives

Let:

```text
x       = clean initial condition
x_adv   = x + delta
f(.)    = neural operator / FNO model
solver(.) = PDE solver final state
```

### Loss1

```text
loss1 = MSE(f(x_adv), f(x).detach())
```

Meaning:

- Attack uses only the model.
- The solver is not called in the attack step.
- There is no solver forward and no solver backward during attack objective construction.
- This is the fastest objective in principle.

Important caveat:

- If `delta = 0` and there is no random start, loss1 has zero objective and zero gradient at the first step because `f(x_adv) = f(x)`.
- The current default keeps `random_start_fraction = 0.0` so the parameters match loss3 exactly.
- A dedicated loss1 experiment may need a small `--random-start-fraction`, but that would no longer be strictly identical to the loss3 parameter setting.

### Loss2

```text
loss2 = MSE(f(x_adv), solver(x).detach())
```

Meaning:

- Attack uses the solver forward once at the clean input `x`.
- The solver target is fixed during the attack.
- The solver output is detached, so there is no solver backward through the attack objective.
- This should be much faster than loss3 because it avoids differentiating through the solver rollout.

### Loss3

```text
loss3 = MSE(f(x_adv), solver(x_adv))
```

Meaning:

- Attack uses solver forward at the attacked input.
- Attack also differentiates through the solver, so solver backward is part of the attack gradient.
- This is the current expensive full solver-gradient adversarial training path.

## Training Target After Attack

The new code changes the attack generation objective only.

After `x_adv` is generated, the optimizer training pair remains:

```text
model(x_adv) -> solver(x_adv).detach()
```

This is intentional because the requested distinction was mainly in the attack step. Therefore:

- loss1 chooses adversarial points using model-only sensitivity.
- loss2 chooses adversarial points using a fixed clean solver target.
- loss3 chooses adversarial points using the full attacked solver target and solver gradient.
- All three still train the model on attacked solver-consistent pairs unless `--training-data-mode` is changed later.

## New Code Switches

Main training script:

```bash
adv_robust/bin/python tools/adversarial_training.py   --tasks burgers   --burgers-attack-loss-objective loss1
```

Allowed values:

```text
loss1
loss2
loss3
```

The same option is exposed in the full pipeline as:

```bash
--attack-loss loss1
--attack-loss loss2
--attack-loss loss3
```

## One-Command Pipelines

Run loss1 full pipeline:

```bash
tools/run_burgers_p2q2_loss1_full_pipeline.sh
```

Run loss2 full pipeline:

```bash
tools/run_burgers_p2q2_loss2_full_pipeline.sh
```

Run loss1 then loss2 sequentially:

```bash
tools/run_burgers_p2q2_loss12_full_pipelines.sh
```

Equivalent explicit Python commands:

```bash
adv_robust/bin/python tools/run_burgers_p2q2_full_pipeline.py --attack-loss loss1
adv_robust/bin/python tools/run_burgers_p2q2_full_pipeline.py --attack-loss loss2
```

Smoke test commands:

```bash
adv_robust/bin/python tools/run_burgers_p2q2_full_pipeline.py --mode smoke --attack-loss loss1 --skip-r2-upload --skip-git-push
adv_robust/bin/python tools/run_burgers_p2q2_full_pipeline.py --mode smoke --attack-loss loss2 --skip-r2-upload --skip-git-push
```

Dry-run postprocess command:

```bash
adv_robust/bin/python tools/run_burgers_p2q2_full_pipeline.py --mode postprocess --attack-loss loss1 --dry-run --skip-r2-upload --skip-git-push
```

## Default Output Locations

For loss1:

```text
adversarial_training_runs/burgers_p2q2_loss1_advonly_random_jitter_1000ep_bs480_steps5_eps5bucket_20260602
visualizations/burgers_p2q2_loss1_adv_training_20260602
forensics/burgers_p2q2_loss1_checkpoint_series_jacobian_svd_20260602
forensics/burgers_p2q2_loss1_baseline_vs_epoch1000_attack_visualization_20260602
/workspace/polished_selected_download_burgers_p2q2_loss1_20260602
```

For loss2:

```text
adversarial_training_runs/burgers_p2q2_loss2_advonly_random_jitter_1000ep_bs480_steps5_eps5bucket_20260602
visualizations/burgers_p2q2_loss2_adv_training_20260602
forensics/burgers_p2q2_loss2_checkpoint_series_jacobian_svd_20260602
forensics/burgers_p2q2_loss2_baseline_vs_epoch1000_attack_visualization_20260602
/workspace/polished_selected_download_burgers_p2q2_loss2_20260602
```

R2 default prefixes:

```text
machine-sync/NeuralOperatorRobustness2-selected/20260602_burgers_p2q2_loss1_adv_training_full_pipeline
machine-sync/NeuralOperatorRobustness2-selected/20260602_burgers_p2q2_loss2_adv_training_full_pipeline
```

## Timing Records

Timing is recorded in multiple places.

Per step:

```text
burgers/train_steps.csv
  attack_wall_sec
  train_wall_sec
  step_wall_sec
  attack_sec_per_sample
  train_sec_per_sample
  step_samples_per_sec
```

Per optimizer microbatch:

```text
burgers/optimizer_steps.csv
  optimizer_wall_sec
```

Per epoch attack summary:

```text
burgers/attack_epoch_summary.csv
  attack_wall_sec_total
  attack_samples_per_sec
```

Task summary:

```text
burgers/summary.json
  elapsed_seconds
  elapsed_minutes
```

Whole pipeline summary:

```text
pipeline_logs/pipeline_done.json
  elapsed_sec
```

Whole run summary:

```text
summary.json
  total_wall_seconds
  total_wall_minutes
```

These are the numbers to compare loss1, loss2, and loss3 runtime. The important expected ordering is:

```text
loss1 fastest: no solver in attack
loss2 middle: solver forward only in attack
loss3 slowest: solver forward + solver backward in attack
```

## What To Compare Later

For loss1/loss2/loss3, compare:

- Total wall time for 1000 epochs.
- Average `attack_wall_sec` per training batch.
- Average `train_wall_sec` per training batch.
- Evaluation loss decrease across 52 datasets.
- Attack loss gain and epsilon bucket summaries.
- Delta FFT spectrum and high-frequency metrics.
- Checkpoint Jacobian/SVD error norm and singular subspace similarity to solver.

A later fixed-wall-clock comparison can run loss1/loss2 for enough epochs to match the loss3 wall time, for example about 14 hours, but the first default comparison should keep all three at 1000 epochs.
