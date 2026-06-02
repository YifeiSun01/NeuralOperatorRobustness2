# Burgers p2q2 adversarial training, visualization, and Jacobian/SVD record

Date: 2026-06-02

This note records the corrected Burgers adversarial-training run and the related visualization/Jacobian discussions.  It intentionally does not contain any GitHub, Cloudflare, R2, or S3 credentials.

## 1. Corrected training run

The corrected Burgers run is:

```text
adversarial_training_runs/burgers_p2q2_advonly_random_jitter_1000ep_bs480_steps5_eps5bucket_20260601
```

The important correction is that the attack is now p=2, q=2 / RMS-L2 style, not the accidental coordinatewise Linf sign attack.

| item | value |
|---|---|
| task | 1D Burgers |
| training mode | ADV-only |
| label mode | solver labels |
| attack method | `fast_replace_l2` |
| attack type | `continuous_l2_rms` |
| p, q | `p=2`, `q=2` |
| epochs | 1000 |
| global steps | 3000 |
| train samples | 1350 |
| attack batch size | 480 |
| optimizer batch size | 32 |
| optimizer steps | 43000 |
| attack steps | 5 |
| checkpoint schedule | every 200 epochs |
| evaluation schedule | every epoch |
| eval datasets per epoch | 52 = train + test + 50 generalization |
| fixed attack probes | 5 fixed source indices saved every epoch |

Saved model checkpoints:

```text
burgers_epoch200_step000600.pt
burgers_epoch400_step001200.pt
burgers_epoch600_step001800.pt
burgers_epoch800_step002400.pt
burgers_epoch1000_step003000.pt
```

The completed p2q2 run took:

```text
51959.51 sec = 865.99 min = 14.43 h
```

For comparison, the earlier ordinary non-adversarial 500-epoch Burgers training took about:

```text
82.04 sec = 1.37 min
```

The adversarial run is much slower because each attack batch first solves a local worst-case perturbation problem before doing optimizer updates.

## 2. Epsilon and perturbation scale

The completed p2q2 run used `epsilon_fraction = 0.06` with per-sample RMS-L2 semantics:

```text
epsilon means RMS(delta) per sample, not coordinatewise max(delta).
```

Observed in `attack_batches.csv`:

| statistic | value |
|---|---:|
| epsilon mean | about 0.060 |
| epsilon min | about 0.045 |
| epsilon max | about 0.075 |
| delta L2 RMS mean | about 0.060 |
| delta L2 total mean | about 1.90 |

Because the Burgers input range is approximately `[0, 1]`, this means the average pointwise RMS perturbation is about `4.5%` to `7.5%` of the input range, with mean around `6%`.

Note: the code defaults were later adjusted for future runs to allow a wider epsilon jitter range.  The completed run recorded here used the narrower observed `0.045` to `0.075` RMS range.

## 3. Delta geometry check

The geometry validator output is:

```text
adversarial_training_runs/burgers_p2q2_advonly_random_jitter_1000ep_bs480_steps5_eps5bucket_20260601/burgers/l2_delta_geometry_check.json
```

Key validation results:

| check | result |
|---|---|
| metadata attack method | `fast_replace_l2` |
| metadata attack type | `continuous_l2_rms` |
| median rounded unique delta values | about 1020 |
| top absolute-value fraction | 0.0009765625 |
| passed non-rectangular check | true |

Interpretation:

```text
The saved perturbation delta is not a two-level +epsilon/-epsilon rectangle.
It has many amplitudes across the 1024 grid points, which is the expected L2-like behavior.
```

This fixes the previous accidental `fast_replace_linf` behavior, where `delta = epsilon * sign(grad)` forced nearly every coordinate onto `+epsilon` or `-epsilon`.

## 4. Attack-loss interpretation

The main attack-loss quantities are:

| quantity | meaning |
|---|---|
| clean loss before attack | model loss on the current clean training batch before perturbation |
| adversarial loss after attack | old/current model loss on the attacked batch immediately after attack generation |
| attack loss gain | `adversarial loss after attack - clean loss before attack` |
| relative attack gain | `attack loss gain / clean loss before attack` |
| optimizer update loss | microbatch loss actually used for optimizer updates; not the same object as full attacked-batch loss |

Important conclusion:

```text
The absolute attack gain tends to decrease during training.
The relative attack gain is much less decisive, because the clean loss itself also decreases.
```

So the safest claim is:

```text
The attack produces smaller absolute loss increases later in training, but this alone does not prove robustness unless it is separated from the fact that clean loss also became smaller.
```

The epsilon-bucket plots are useful because random epsilon mixes small, medium, and large perturbation radii.  The bucketed curves estimate:

```text
E[attack loss | epsilon bucket]
E[attack gain | epsilon bucket]
E[relative attack gain | epsilon bucket]
```

## 5. Delta spectrum conclusion

The perturbation spectrum figures show a robust trend:

```text
As adversarial training progresses, the saved fixed-probe delta becomes more high-frequency.
```

The supporting diagnostics are:

| diagnostic | observed trend |
|---|---|
| FFT heat map over Fourier modes and epochs | energy moves toward higher modes |
| high-frequency energy share | increases |
| spectral centroid | increases |
| total variation | increases |
| sign-change fraction | increases |

The raw FFT heat map is the most faithful visualization.  Moving average can make the trend easier to see in line plots, but the selected report version should avoid misleading title language such as "no moving average" and should describe the scientific content instead.

## 6. Selected visualization files

Selected figures are collected in:

```text
visualizations/burgers_p2q2_adv_training_20260601/polished_selected_download_burgers_p2q2_20260601
```

The selected PNGs are:

```text
corrected_attack_loss_three_lines_plus_buckets.png
corrected_relative_l2_full_heatmap_raw_group_line.png
corrected_rmse_full_heatmap_raw_group_line.png
fixed_source_index000_delta_trajectory_epochs000_100_200_300_400_500_600_700_800_900_1000.png
polished_checkpoint_style_delta_fft_raw_heatmap_raw_lines_no_moving_average.png
relative_l2_grouped_shared_y_distinct_datasets_max5_high_transparency_raw.png
relative_l2_grouped_shared_y_distinct_datasets_max5_high_transparency_ma25.png
rmse_grouped_shared_y_distinct_datasets_max5_high_transparency_raw.png
rmse_grouped_shared_y_distinct_datasets_max5_high_transparency_ma25.png
```

Visualization design rules settled in the discussion:

| case | preferred visualization |
|---|---|
| many datasets and many epochs | heat map, because color carries the loss value |
| few curves, such as attack buckets | line plot with mean/std shading |
| dense dataset line plots | max 5 lines per subplot, distinct colors, high transparency |
| loss heat maps | raw values in heat map, group color strip on the left, group text outside the color strip |
| group line plots below heat map | use the same colors as the group strip |
| delta FFT spectrum | raw heat map over epoch and Fourier mode; line plots may be smoothed only when explicitly useful |

## 7. What the Jacobian/SVD analysis computes

For a fixed input point `x`, the analysis compares:

```text
J_solver(x) = d solver(x) / d x
J_model(x) = d model(x) / d x
J_error(x) = J_model(x) - J_solver(x)
```

The sample points are fixed across all models and checkpoints.  This matters because Jacobian comparisons only make sense at the same local input point.

The p2q2 checkpoint-series analysis uses the previous representative20 sample manifest:

```text
6 train + 4 test + 10 generalization = 20 fixed points
```

For each checkpoint, it saves top-100 SVD data for:

```text
J_model(x)
J_error(x)
```

The solver and baseline SVD files were reused from the earlier representative20 run when available.

Main output directory:

```text
forensics/burgers_p2q2_checkpoint_series_jacobian_svd_20260601
```

Main files:

```text
checkpoint_series_summary.md
checkpoint_series_jacobian_svd_summary.csv
checkpoint_series_top_singular_values_long.csv
checkpoint_series_solver_similarity_rankwise.csv
checkpoint_series_solver_similarity_subspaces.csv
checkpoint_series_error_aggregate.csv
sample_000/ ... sample_019/
```

## 8. SVD physical meaning

For a local Jacobian:

```text
J = U Sigma V^T
J v_i = sigma_i u_i
```

Meaning:

| object | meaning |
|---|---|
| right singular vector `v_i` | input perturbation direction |
| left singular vector `u_i` | output response direction |
| singular value `sigma_i` | amplification factor |

So:

```text
v_1 = most dangerous input perturbation direction
u_1 = main output response caused by that perturbation
sigma_1 = spectral norm / maximum local amplification
```

When comparing a model to the solver, there are three layers of evidence:

```text
singular values closer to solver
right singular vectors closer to solver
left singular vectors closer to solver
```

If all three improve, the model is not merely becoming flatter; it is becoming locally more solver-like.

Important caveats:

```text
v and -v are the same singular-vector direction, so vector similarity uses abs(dot).
When singular values are close, individual vectors can rotate, so top-k subspace similarity is also recorded.
```

## 9. Model-Jacobian norm summary

This table summarizes `||J_model||_2` on the same 20 points.  The solver is fixed.

| object | spectral norm mean | spectral norm std | ratio to baseline model |
|---|---:|---:|---:|
| solver | 5.903 | 1.824 | n/a |
| baseline model | 5.289 | 1.290 | 1.000 |
| epoch 200 model | 5.266 | 1.356 | 0.996 |
| epoch 400 model | 5.656 | 1.550 | 1.069 |
| epoch 600 model | 5.698 | 1.641 | 1.077 |
| epoch 800 model | 5.715 | 1.653 | 1.081 |
| epoch 1000 model | 5.743 | 1.648 | 1.086 |

Interpretation:

```text
The trained model Jacobian norm did not collapse or simply become smaller.
It slightly increased toward the solver scale.
```

This is important because the desirable result is not `J_model -> 0`; the desirable result is:

```text
J_model -> J_solver
```

## 10. Error-Jacobian spectral norm summary

This is the key robustness/local-solver-match result for:

```text
J_error = J_model - J_solver
```

| object | all-point mean | all-point std | ratio to baseline error | change vs baseline | smaller than baseline |
|---|---:|---:|---:|---:|---:|
| baseline error | 2.378 | 1.859 | 1.000 | 0.0% | n/a |
| epoch 200 error | 2.699 | 2.022 | 1.388 | -13.5% | 4/20 |
| epoch 400 error | 2.268 | 1.989 | 1.039 | 4.6% | 12/20 |
| epoch 600 error | 2.084 | 1.650 | 0.989 | 12.3% | 14/20 |
| epoch 800 error | 0.755 | 0.440 | 0.464 | 68.2% | 19/20 |
| epoch 1000 error | 0.915 | 0.404 | 0.591 | 61.5% | 17/20 |

Split-level means:

| split | baseline | epoch 200 | epoch 400 | epoch 600 | epoch 800 | epoch 1000 |
|---|---:|---:|---:|---:|---:|---:|
| train | 1.508 | 1.391 | 0.975 | 1.033 | 0.533 | 0.783 |
| test | 0.769 | 1.514 | 1.117 | 0.880 | 0.458 | 0.647 |
| generalization | 3.543 | 3.957 | 3.504 | 3.196 | 1.008 | 1.101 |

Interpretation:

```text
Early checkpoints are noisy and sometimes worse.
By epoch 800 and epoch 1000, the error Jacobian is much smaller than baseline, especially on generalization samples.
Epoch 800 is the best checkpoint by mean error-Jacobian norm; epoch 1000 is slightly worse but still much better than baseline.
```

## 11. Singular-value shape

Top singular values of `J_model` show that the trained model becomes closer to the solver scale, not uniformly smaller.

| object | sigma1 | sigma2 | sigma3 | sigma5 | sigma10 |
|---|---:|---:|---:|---:|---:|
| solver | 5.903 | 2.603 | 1.908 | 0.822 | 0.411 |
| baseline model | 5.289 | 2.400 | 1.784 | 0.802 | 0.387 |
| epoch 1000 model | 5.743 | 2.571 | 1.886 | 0.812 | 0.406 |

Top singular values of `J_error` shrink strongly:

| object | sigma1 | sigma2 | sigma3 | sigma5 | sigma10 |
|---|---:|---:|---:|---:|---:|
| baseline error | 2.378 | 1.030 | 0.536 | 0.307 | 0.179 |
| epoch 200 error | 2.699 | 0.843 | 0.560 | 0.250 | 0.149 |
| epoch 400 error | 2.268 | 0.637 | 0.350 | 0.189 | 0.108 |
| epoch 600 error | 2.084 | 0.490 | 0.291 | 0.147 | 0.087 |
| epoch 800 error | 0.755 | 0.279 | 0.184 | 0.113 | 0.071 |
| epoch 1000 error | 0.915 | 0.303 | 0.198 | 0.113 | 0.067 |

Interpretation:

```text
The main reduction is in the mismatch Jacobian, not in the raw model Jacobian.
That supports the claim that the trained model's local derivative is becoming more like the solver derivative.
```

## 12. Singular-vector and subspace similarity to solver

Rankwise similarity to solver:

| checkpoint | right v top1 | left u top1 | right v top20 | left u top20 | singular-value ratio top20 |
|---|---:|---:|---:|---:|---:|
| epoch 200 | 0.951 +/- 0.147 | 0.853 +/- 0.174 | 0.662 +/- 0.312 | 0.713 +/- 0.268 | 0.852 +/- 0.168 |
| epoch 400 | 0.923 +/- 0.234 | 0.849 +/- 0.233 | 0.755 +/- 0.284 | 0.776 +/- 0.267 | 0.892 +/- 0.131 |
| epoch 600 | 0.997 +/- 0.005 | 0.936 +/- 0.066 | 0.813 +/- 0.254 | 0.830 +/- 0.236 | 0.922 +/- 0.106 |
| epoch 800 | 0.999 +/- 0.001 | 0.993 +/- 0.006 | 0.877 +/- 0.201 | 0.891 +/- 0.185 | 0.948 +/- 0.082 |
| epoch 1000 | 0.999 +/- 0.001 | 0.987 +/- 0.009 | 0.879 +/- 0.194 | 0.893 +/- 0.184 | 0.953 +/- 0.071 |

Top-20 subspace similarity to solver:

| checkpoint | right subspace mean principal cosine | left subspace mean principal cosine |
|---|---:|---:|
| epoch 200 | 0.877 +/- 0.058 | 0.941 +/- 0.024 |
| epoch 400 | 0.930 +/- 0.041 | 0.960 +/- 0.022 |
| epoch 600 | 0.949 +/- 0.034 | 0.970 +/- 0.019 |
| epoch 800 | 0.962 +/- 0.028 | 0.979 +/- 0.017 |
| epoch 1000 | 0.971 +/- 0.023 | 0.984 +/- 0.014 |

Interpretation:

```text
The trained checkpoints become increasingly solver-like in singular-vector direction and top-k subspace.
The strongest evidence appears from epoch 600 onward, and remains strong at epoch 800/1000.
```

## 13. Main scientific conclusion

The p2q2 corrected run supports this statement:

```text
Adversarial training does not simply reduce the model Jacobian norm.
Instead, it makes the model's local Jacobian closer to the solver's local Jacobian.
This is visible in the shrinking J_error spectral norm, shrinking J_error singular values, and increasing singular-vector/subspace similarity to the solver.
```

A concise version:

```text
J_model itself stays solver-scale.
J_error = J_model - J_solver becomes much smaller.
The singular values, right singular directions, left singular directions, and top-k subspaces all move closer to the solver.
```

## 14. R2 upload status and 501 explanation

The automated pipeline reached the R2 upload stage but the earlier upload log reported:

```text
NotImplemented: Not Implemented
```

This is a Cloudflare R2/S3 compatibility issue, not a training failure.  R2 does not implement every AWS S3 feature.  A common trigger is sending an unsupported ACL-related S3 option/header.  The upload should be retried with R2-compatible options and without ACL headers.

Manual retry on 2026-06-02 succeeded after removing ACL use and letting rclone retry the first-round `501` object-copy failures.  The first GitHub commit containing this document and the related code changes was:

```text
b53afde Document Burgers p2q2 adversarial analysis
```

R2 backup target prefix:

```text
machine-sync/NeuralOperatorRobustness2-selected/20260601_burgers_p2q2_adv_training_full_pipeline/
```

The intended R2 payloads are:

```text
adversarial_training_runs/burgers_p2q2_advonly_random_jitter_1000ep_bs480_steps5_eps5bucket_20260601
visualizations/burgers_p2q2_adv_training_20260601
forensics/burgers_p2q2_checkpoint_series_jacobian_svd_20260601
```

These three R2 prefixes were verified with `rclone lsf` after upload.

GitHub should receive lightweight reproducibility files:

```text
docs/*.md
tools/*.py
tools/*.sh
selected manifests and small CSV/Markdown summaries
```

Large checkpoints, NPZ files, and full figure/data payloads should stay on R2 rather than GitHub.

