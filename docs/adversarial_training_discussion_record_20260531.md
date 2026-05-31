# Adversarial Training Discussion Record, 2026-05-31

This note records the main decisions, explanations, results, and file paths from
the recent Burgers / Darcy / NS2D adversarial-training discussion.

Sensitive access tokens, API keys, and secrets are intentionally not repeated in
this document.

## 1. Repository And Environment

- Repository: `/workspace/NeuralOperatorRobustness2`
- Branch used in this workspace: `vast-ai`
- GitHub repository: `https://github.com/YifeiSun01/NeuralOperatorRobustness2`
- Virtual environment: `adv_robust`
- Main training script: `tools/adversarial_training.py`
- Main queue scripts:
  - `adversarial_training_runs/run_full_adv_training_queue_20260530.sh`
  - `adversarial_training_runs/run_darcy_full10_retry_safe_20260531.sh`
- Main output root: `adversarial_training_runs/`

## 1.1 Backup Targets And Credentials Policy

The requested backup scope is:

```text
Model code for 1D Burgers, 2D Darcy Flow, and 2D Navier Stokes
Data
Visualization outputs
Python files
Shell files
Markdown notes
Selected run outputs and figures
```

Primary local repository folder:

```text
/workspace/NeuralOperatorRobustness2
```

Remote GitHub target:

```text
owner: YifeiSun01
repo: NeuralOperatorRobustness2
branch: vast-ai
remote: origin
```

Cloudflare R2 / S3-compatible backup target was provided in chat.  The intended
object prefix is:

```text
neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected
```

For security, the raw Cloudflare API token, S3 access key, S3 secret key, and
GitHub personal access token are not written here and should never be committed.
Use environment variables or a credential helper instead:

```text
CLOUDFLARE_API_TOKEN
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_ENDPOINT_URL_S3
GITHUB_TOKEN
```

The exact tokens pasted in chat should be treated as exposed credentials after
appearing in the conversation.  They should be rotated before being used for any
long-term workflow.

## 2. Saved Burgers Models

The Burgers adversarial-training checkpoints were saved.

Baseline, before adversarial training:

```text
1D_Burgers/trained_models/attack_ready/burgers_nu0.001_fno1d_500/checkpoints/pytorch_fno1d_500.pt
```

ADV-only, after adversarial training:

```text
adversarial_training_runs/full10_burgers_adv_only_20260530/burgers/checkpoints/burgers_epoch500_step000500.pt
```

clean+ADV, after adversarial training:

```text
adversarial_training_runs/full10_burgers_clean_plus_adv_20260530/burgers/checkpoints/burgers_epoch500_step000500.pt
```

Intermediate checkpoints also exist at epoch 100, 200, 300, and 400 for both
Burgers modes.

## 3. Completed Burgers Runtime

| Run | Mode | Epochs | Steps | Batch | Optimizer batch | Actual elapsed |
|---|---|---:|---:|---:|---:|---:|
| `full10_burgers_adv_only_20260530` | `adv-only` | 500 | 500 | 1350 | 32 | 27019.6 s = 450.3 min = 7.51 h |
| `full10_burgers_clean_plus_adv_20260530` | `clean-plus-adv` | 500 | 500 | 1350 | 32 | 27378.0 s = 456.3 min = 7.61 h |

Per-epoch timing:

| Run | Attack time / epoch | Optimizer train time / epoch | Total step time / epoch |
|---|---:|---:|---:|
| Burgers ADV-only | about 52.91 s | about 1.12 s | about 54.03 s |
| Burgers clean+ADV | about 52.56 s | about 2.19 s | about 54.75 s |

## 4. Darcy Flow Status And Runtime

The original Darcy full10 formal runs failed from CUDA OOM:

| Run | Mode | Status | Observed progress |
|---|---|---|---|
| `full10_darcy_adv_only_20260530` | `adv-only` | failed OOM | only 1 training row, about 16.68 s before failure |
| `full10_darcy_clean_plus_adv_20260530` | `clean-plus-adv` | failed OOM | only 1 training row, about 18.66 s before failure |

The OOM happened with:

```text
--darcy-batch-size 448
--darcy-optimizer-batch-size 32
--darcy-attack-steps 10
```

Safe retry configuration:

```text
--darcy-batch-size 416
--darcy-optimizer-batch-size 128
--darcy-attack-steps 10
```

At an earlier check in the discussion, Darcy ADV-only retry was running:

```text
run: full10_darcy_adv_only_retry_safe_20260531
progress: 386 / 1500 steps
epoch: 129 / 500
progress fraction: about 25.73%
mean step time: about 11.79 s
estimated total ADV-only time: about 4.91 h
estimated remaining ADV-only time then: about 3.65 h
```

If the queue continues into Darcy clean+ADV after ADV-only, the clean+ADV part
was estimated from the batch sweep at about 6.8 h.

At the later live check on 2026-05-31 13:42 UTC:

```text
run: full10_darcy_adv_only_retry_safe_20260531
status: running
epoch: 184 / 500
global_step: 552 / 1500
progress fraction: 36.8%
elapsed runtime: about 1 h 48 min 49 s
estimated remaining ADV-only time: about 3.0 to 3.5 h
next checkpoint/evaluation: step 600, about 9 to 10 min after that check
```

## 5. Why Darcy Can Be Faster Than Burgers

The surprising result was that 2D Darcy can run faster than 1D Burgers.  The
reason is not that Darcy has fewer epochs.

Both formal settings use:

```text
epochs = 500
attack steps = 10
full train coverage per epoch
```

But the work per epoch differs:

| Task/run | Batch schedule | Samples / epoch | Optimizer microbatches / epoch | Mean total time / epoch |
|---|---:|---:|---:|---:|
| Burgers ADV-only | one batch of 1350 | 1350 | 43 | about 54.03 s |
| Darcy ADV-only retry | 416 + 416 + 368 | 1200 | about 11 | about 35.29 s |

The main cost is attack generation, not optimizer training:

| Task/run | Attack time / epoch | Optimizer train time / epoch |
|---|---:|---:|
| Burgers ADV-only | about 52.9 s | about 1.1 s |
| Darcy ADV-only retry | about 32.0 s | about 3.3 s |

The key reason:

- Burgers uses a continuous solver-gradient attack through a time-dependent PDE
  solver rollout.
- Burgers uses `dt=0.001`, `t_final=1.0`, so the solver target involves about
  1000 time steps.
- Darcy uses a binary coefficient replacement attack, `binary_steepest_replace`.
- Darcy is 2D, but it is not a 1000-step time rollout; it is an elliptic solve
  plus binary pixel flips.

Therefore 2D Darcy is not automatically slower.  The attack/solver structure is
the decisive cost.

## 6. Evaluation Schedule And New Delta Probe Logging

The training script was modified so that new runs evaluate every epoch.

Old behavior:

```text
evaluation at progress fractions such as 0, 0.2, 0.4, 0.6, 0.8, 1.0
```

New behavior for newly started runs:

```text
evaluation at epoch 0, 1, 2, ..., 500
```

Checkpoint saving stays on the fraction schedule by default, so it does not
save 500 checkpoints unless explicitly configured.

New output files:

```text
eval_metrics.csv
evaluation_passes.csv
attack_probe_config.json
attack_probe_epochs.csv
attack_probe_samples.csv
attack_probe_samples/*.npz
```

The new attack probe saves fixed train-set source indices every probe epoch.
For each captured sample it saves:

```text
x_clean
x_adv
delta = x_adv - x_clean
source_index
probe_rank
epoch
attack_global_step
```

CSV summary metrics include:

```text
delta_linf
delta_l2_rms
delta_abs_mean
delta_total_variation
delta_sign_change_fraction
delta_fft_high_freq_ratio
delta_fft_spectral_centroid
attack_loss_gain
clean_loss_before_attack
adv_loss_after_attack
```

The purpose is to check whether the attack perturbation `delta` becomes more
high-frequency, more jagged, or otherwise more structured during training.

Important note: already-running Python training processes do not automatically
pick up this script change.  Only newly started runs use it.

## 7. NS2D Alpha Jitter

The NS2D attack step size was changed so that alpha is not a fixed multiple of
epsilon.

New default for NS2D:

```text
alpha = epsilon * alpha_ratio * Uniform(0.75, 1.25)
```

Example:

```text
if alpha_ratio = 0.2,
then alpha / epsilon is sampled around 0.15 to 0.25
```

Burgers and Darcy keep fixed alpha jitter by default:

```text
alpha_jitter_low = 1.0
alpha_jitter_high = 1.0
```

New CLI options:

```text
--ns2d-alpha-jitter-low
--ns2d-alpha-jitter-high
--burgers-alpha-jitter-low
--burgers-alpha-jitter-high
--darcy-alpha-jitter-low
--darcy-alpha-jitter-high
```

## 8. Evaluation Loss Is Not Monotonic

Burgers evaluation loss is not strictly monotonic.  This is real behavior, not
just a plotting artifact.

Example, Burgers clean+ADV generalization Relative L2:

```text
epoch 0:   0.03106
epoch 100: 0.01086
epoch 200: 0.00897
epoch 300: 0.01333
epoch 400: 0.02525
epoch 500: 0.01270
```

Interpretation:

- The model improves strongly early on.
- Later checkpoints can get worse.
- Epoch 400 for clean+ADV is a bad checkpoint.
- Final checkpoint is not necessarily the best checkpoint.

Recommendation:

```text
select best checkpoint by validation/generalization Relative L2,
not automatically the final checkpoint
```

Reasons for non-monotonic evaluation:

- adversarial examples are regenerated from the current model;
- the training objective is not the same as the fixed evaluation loss;
- epsilon jitter and random starts add noise;
- clean+ADV mixes clean and adversarial objectives;
- evaluation was previously sparse, every 100 epochs.

## 9. Training/Attack Loss Terms

The main logged quantities mean different things.

```text
clean_loss_before_attack
= MSE on the current clean training batch before attack

adv_loss_after_attack
= MSE on the current attacked batch immediately after attack,
   before optimizer updates

attack_loss_gain
= adv_loss_after_attack - clean_loss_before_attack

train_loss_on_adv
= the average loss used during optimizer updates
```

Important clarification:

`train_loss_on_adv` is a misleading name, especially for clean+ADV.  It is
better thought of as:

```text
microbatch_loss_during_optimizer_updates
```

Why it can be much smaller than `adv_loss_after_attack`:

- `adv_loss_after_attack` is computed once with the old model on the full
  attacked batch.
- The optimizer loss is computed microbatch by microbatch.
- After each microbatch, `optimizer.step()` updates the model.
- Therefore later microbatch losses are computed with an already-updated model.
- In clean+ADV, the optimizer loss also mixes clean samples and adversarial
  samples, and clean samples are usually easier.

So:

```text
adv_loss_after_attack
= how bad the old model is immediately after attack

optimizer update loss
= what the model sees while it is being updated online over microbatches
```

To judge attack strength, use:

```text
clean_loss_before_attack
adv_loss_after_attack
attack_loss_gain
```

Do not directly compare `optimizer update MSE` with `adv_loss_after_attack` as
if they were the same physical quantity.

## 10. Attack Gain And Robustness

Attack gain decreasing is a positive robustness signal under the same attack
configuration.

Observed Burgers trend:

```text
ADV-only attack gain:
epoch 1-100:   about 1.75e-4
epoch 401-500: about 5.79e-5

clean+ADV attack gain:
epoch 1-100:   about 1.80e-4
epoch 401-500: about 3.47e-5
```

Interpretation:

- The attack radius did not shrink.
- The perturbation norm did not shrink.
- The same-size attack causes less extra loss later in training.

So the model becomes less sensitive to this attack, but this is not a complete
proof of global robustness.  Stronger attacks, held-out seeds, and
generalization evaluation are still needed.

## 11. What Was Saved For Attack Deltas

The completed old Burgers full10 runs did not save full per-sample `delta` or
`x_adv` arrays.  They saved only summary statistics such as:

```text
epsilon_mean/min/max
alpha_mean/min/max
delta_linf_mean
delta_l2_rms_mean
clean_loss_before_attack
adv_loss_after_attack
attack_loss_gain
```

Therefore the old completed runs can show that attack gain decreased, but they
cannot show whether the perturbation became more high-frequency or more
structured.

The new attack-probe logging was added to solve this for future runs.

## 12. Plotting And Labels

The plotted labels were clarified:

- use `evaluation epoch`, not `eval step`;
- use `Relative L2 loss`;
- rename training-batch attack diagnostics so they are not confused with
  evaluation metrics.

Generated plot categories included:

- 2x3 common-y RMSE grids for evaluation epochs 0 through 500;
- 2x3 common-y Relative L2 grids;
- separate ADV-only and clean+ADV versions;
- clearer training/attack diagnostic plots.

Important conceptual distinction:

```text
evaluation plots
= fixed train/test/generalization datasets

training/attack diagnostic plots
= current training batch during adversarial training
```

## 13. Jacobian/SVD Definition

The previous Jacobian documents defined the key local objects as:

```text
f = model
j = solver / oracle
J_f = model Jacobian at x
J_j = solver Jacobian at x
J_e = J_f - J_j
```

For local perturbations:

```text
Delta f ~= J_f delta
Delta j ~= J_j delta
Delta e ~= (J_f - J_j) delta
```

The main robustness-locality object is:

```text
J_error = J_model - J_solver
```

not `J_model` alone.

The old result for baseline FNO showed:

```text
J_f and J_j had similar dominant directions;
J_e = J_f - J_j was much smaller than J_f or J_j.
```

This supports the idea that large model movement can be harmless if the solver
moves with it.

Relevant old docs:

```text
docs/fno_solver_jacobian_similarity_result_20260514.md
docs/local_jacobian_svd_experiment_purpose_20260515.md
docs/local_jacobian_svd_direction_taxonomy_20260515.md
```

## 14. New Burgers Three-Checkpoint Jacobian Script

A new script was added:

```text
tools/compare_burgers_adversarial_jacobian_svd.py
```

It compares:

```text
baseline
ADV-only epoch500
clean+ADV epoch500
```

For each sampled Burgers input, it computes:

```text
J_model
J_solver
J_error = J_model - J_solver
```

It saves:

```text
sample_manifest.csv
jacobian_svd_summary.csv
top_singular_values_long.csv
aggregate_jacobian_svd_summary.csv
runtime.csv
sample_*/...jacobian_svd.npz
plots/mean_top20_model_singular_values.png
plots/mean_top20_error_singular_values.png
summary.md
```

A dry-run sample manifest was generated here:

```text
forensics/burgers_adv_training_jacobian_svd_20260531_dryrun/sample_manifest.csv
```

The full Jacobian/SVD run was not started during the discussion because the GPU
was fully occupied by Darcy training.  The old solver-Jacobian timing was about
280 seconds per sample, so a 20-sample full comparison is a heavy job.

## 15. Backup And Sync Notes

The user asked to back up code, data, plots, Python files, shell files, and
Markdown files to the repository / remote storage locations.

This document does not record any access token, S3 key, Cloudflare API token, or
GitHub personal access token.  Those secrets should not be committed to GitHub
or written into Markdown records.

## 16. Practical Next Steps

1. Let the current Darcy ADV-only retry finish.
2. Decide whether to let the queue continue into Darcy clean+ADV.
3. For future runs, use the updated every-epoch evaluation and attack-probe
   delta logging.
4. After GPU is free, run the Burgers three-checkpoint Jacobian/SVD comparison.
5. For model selection, use best generalization Relative L2 rather than blindly
   using the final checkpoint.
6. For robustness claims, report both evaluation metrics and attack-gain /
   `J_error` diagnostics.

