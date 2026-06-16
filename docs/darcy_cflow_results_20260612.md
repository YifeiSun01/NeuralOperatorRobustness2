# Darcy Cflow Binary Robustness Results - 2026-06-12

This note records the Darcy-flow binary-coefficient adversarial experiments, metric studies, visualization scripts, and current result locations prepared on the Vast AI instance.

## Branch

Git branch for this record:

```text
VastAI.Cflow
```

## Core Setup

```text
Task: 2D Darcy flow
Input coefficient: binary only, values 3 and 12
Generalization data: binary coefficient fields with diverse pattern families
Models compared: baseline, loss1, loss2, loss3, physics loss
First-stage trained checkpoints: time-matched around 1000 epochs
Attack geometry: binary feasible flips only, 3 <-> 12
Primary attack objective: Darcy loss3 binary attack
```

## First-Stage Time-Matched Training

```text
loss1:   1000 epochs, 77.01 min
loss2:   1026 epochs, 77.06 min
loss3:   1011 epochs, 77.39 min
physics: 1040 epochs, 76.69 min
```

The first-stage 1000-ish run completed successfully. A longer stage2 continuation is still running separately and is not required for the first-stage analysis summarized here.

## Main Robustness Findings

Loss3 is not best on every possible metric. The useful distinction is:

```text
Loss3 improves actual binary adversarial robustness and error-aligned sensitivity.
Loss3 does not minimize the pure Jacobian spectral norm sigma_max.
```

Key 10-sample model means:

```text
20-step attack gain:
loss3 2.64e-6 < loss2 3.84e-6 < physics 4.21e-6 < loss1 4.88e-6

||J^T e||:
loss3 9.29e-5 < loss2 1.02e-4 < physics 1.18e-4 < loss1 1.26e-4

binary first-order gain:
loss3 1.20e-6 < loss2 1.26e-6 < physics 1.52e-6 < loss1 1.57e-6

sigma_max:
loss2 0.001721 < loss1 0.001790 < physics 0.001853 < loss3 0.002067
```

Interpretation:

```text
The top singular value measures worst-case input-output sensitivity:
sigma_max(J) = max_v ||J v||.

The input gradient of the squared error is error-aligned sensitivity:
d loss / d a = J^T (F(a) - u(a)).

For binary Darcy attacks, feasible first-order flip scores are:
score_i = (J^T e)_i * (other(a_i) - a_i), where other(a_i) is 3 or 12.
```

The current evidence says Loss3 reduces the error-aligned and binary-feasible attack direction more than it shrinks the whole Jacobian operator norm.

## Metric Correlation Pilot

10-sample correlation run:

```text
Samples: 10 generated Darcy samples
Models: loss1, loss2, loss3, physics
Rows: 40 model-sample pairs
Attack: 20-step binary loss3 attack
Sigma estimate: block/2 top singular vector lifted to full grid, then one full-space power refinement step
```

Within-sample Spearman correlations with attack gain:

```text
|<e,u1>|:              0.680
||J^T e||:             0.620
||J^T e||^2:           0.529
binary first-order:    0.523
sigma_max * ||e||:     0.514
clean loss:            0.477
sigma_max:            -0.530
```

This is the most relevant view because it compares models on the same initial condition. It supports the mechanism that attack gain tracks error-aligned sensitivity more than pure worst-case sensitivity.

25-sample correlation run completed:

```text
Samples: 25 generated Darcy samples
Models: loss1, loss2, loss3, physics
Rows: 100 model-sample pairs
Attack: 20-step binary loss3 attack
```

25-sample model means:

```text
loss1    attack_gain=4.788547e-06  jt_error=1.391878e-04  sigma=1.799438e-03  binary_first_order=1.704681e-06
loss2    attack_gain=4.030971e-06  jt_error=1.166779e-04  sigma=1.722550e-03  binary_first_order=1.423322e-06
loss3    attack_gain=2.756509e-06  jt_error=1.057695e-04  sigma=2.072769e-03  binary_first_order=1.321766e-06
physics  attack_gain=4.139830e-06  jt_error=1.321585e-04  sigma=1.860728e-03  binary_first_order=1.683077e-06
```

25-sample within-sample residual Spearman correlations with attack gain:

```text
clean loss:             0.604
||J^T e||:              0.638
||J^T e||^2:            0.552
binary first-order:     0.522
sigma_max:             -0.599
sigma_max * ||e||:      0.544
|<e,u1>|:               0.629
```

The 25-sample run strengthens the same qualitative mechanism: Loss3 has the smallest attack gain, smallest `J^T e`, and smallest binary first-order score, while still having the largest sigma_max. Within the same coefficient field, attack gain tracks error-aligned and binary-feasible quantities more directly than pure worst-case spectral sensitivity.

Main 25-sample outputs:

```text
analysis_outputs/darcy_metric_correlation_25samples_20260612_25samples_loss3attack20_corr_chunk64
visualizations/darcy_metric_correlation_25samples_20260612_25samples_loss3attack20_corr_chunk64
```

## Block SVD / Power Refinement Finding

For estimating sigma_max:

```text
block/2 sigma vs full sigma:       about 2.89% mean error
lifted ||Jv|| vs full sigma:       about 2.83% mean error
one-step full power refinement:    about 0.03% mean error
```

The block/2 right singular direction is already close. A single full-space refinement step:

```text
u0 = normalize(J v0)
v1 = normalize(J^T u0)
sigma_est = ||J v1||
```

nearly recovers the full-space top singular value without computing a full explicit SVD every time.

## Five-Model Attack Heatmaps

The heatmap plotting code was updated to match the requested Burgers-style layout more closely:

```text
Models: baseline, loss1, loss2, loss3, physics loss
Rows per model: initial condition, delta, perturbed input, model output, solver output, model-solver error
Bottom panel: attack loss growth curves for all models
Label fix: fixed -> physics loss
Color ranges: model output and solver output share the same range; ranges are recorded in column_color_ranges_applied.json
```

A shorter batch-ranked heatmap run completed:

```text
Datasets: 5
Samples per dataset: 50
Models: 5
Attack steps: 50
Ranking variants: index0, loss3_best, loss3_upper_quartile, loss3_median, loss3_lower_quartile, loss3_worst
Figures: 6 variants x 5 datasets = 30 PNG heatmaps
```

Output directories use this tag:

```text
20260612_loss3attack50_five_models_ranked5_batch_polished
```

The ranking CSV has a corrected 50-step filename:

```text
analysis_outputs/darcy_five_model_batch_ranked_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished/all_50step_batch_attack_results.csv
```

The original `all_100step_batch_attack_results.csv` file is retained only for backward compatibility with the first run naming.

## Reproducibility Scripts

```text
tools/plot_darcy_five_model_attack_heatmaps_20260612.py
tools/plot_darcy_five_model_batch_ranked_heatmaps_20260612.py
tools/run_darcy_batch_ranked_heatmaps_20260612.sh
tools/run_darcy_metric_correlation_10samples_20260612.py
tools/run_darcy_metric_correlation_25samples_20260612.py
tools/run_darcy_metric_correlation_25samples_20260612.sh
tools/run_darcy_random_binary_source_training_20260612.sh
```

## Random Binary Source Training Modes

Two non-adversarial random-noise training modes were added in `tools/adversarial_training.py`:

```text
random_binary_fixed_y:
  Add random binary-feasible perturbations/noise to the input coefficient but keep the original y target.

random_binary_solver_y:
  Add random binary-feasible perturbations/noise to the input coefficient and recompute y using the Darcy solver.
```

The random field/noise parameters are resampled each epoch rather than fixed globally, and inputs remain binary coefficient fields.

## Git / R2 Policy Used Here

```text
GitHub: code, shell scripts, Markdown reports, CSV/JSON summaries, and reasonably small visualizations.
R2: full visualization/result sync, including larger generated images and analysis-output directories.
Large model checkpoints are kept out of Git because many .pt files exceed GitHub's normal file-size limits.
```
