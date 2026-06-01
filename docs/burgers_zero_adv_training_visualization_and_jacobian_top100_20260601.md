# Burgers Zero Adversarial Training Visualization And Jacobian/SVD Record - 2026-06-01

This note consolidates the latest Burgers adversarial-training discussion,
visualization decisions, and fixed-point Jacobian/SVD results. It intentionally
does not contain API tokens, access keys, or GitHub credentials.

## 1. Scope

The main objects discussed here are:

- Burgers zero/adversarial self-training, ADV-only, 1000 epochs.
- The existing visualization set for loss curves, attack loss, epsilon buckets,
  and perturbation spectra.
- The new fixed-10 Jacobian/SVD top100 comparison using input points reused from
  the older representative20 same-point study.
- The distinction between the accidental `fast_replace_linf` run and the
  intended future `p=2, q=2` run.

Important caveat:

```text
The currently analyzed 1000-epoch checkpoint comes from fast_replace_linf.
Treat it as an L-infinity/sign-replace ablation, not as the final intended
p=2, q=2 experiment.
```

## 2. Current Artifact Locations

Training run:

```text
adversarial_training_runs/burgers_zero_advonly_random_jitter_1000ep_bs480_steps5_eps5bucket_20260601/
```

Final checkpoint analyzed:

```text
adversarial_training_runs/burgers_zero_advonly_random_jitter_1000ep_bs480_steps5_eps5bucket_20260601/burgers/checkpoints/burgers_epoch1000_step003000.pt
```

Fixed-10 Jacobian/SVD top100 result directory:

```text
forensics/burgers_zero_adv_training_jacobian_svd_20260601_rep20subset10_top100/
```

Selected visualization download folder:

```text
/workspace/polished_selected_download_20260601/
```

Selected visualization zip:

```text
/workspace/polished_selected_download_20260601.zip
```

## 3. Visualization Strategy

The visualization problem has too many axes:

- training epoch
- dataset identity, including 52 train/test/generalization datasets
- loss value or spectrum value
- split/group identity
- attack epsilon bucket
- perturbation Fourier mode

Two visualization forms are useful:

```text
Line plot:
  good when the number of lines is small, or when group averages are shown.

Heat map:
  good when there are many datasets, many epochs, or many Fourier modes.
  The value is encoded as color, saving one visual axis.
```

The main plotted quantities are:

- evaluation loss on train/test/generalization datasets
- attack clean loss, adversarial loss, attack gain, and relative attack gain
- epsilon-bucket attack statistics
- delta spectrum / normalized FFT power
- high-frequency energy share and related delta roughness metrics

## 4. Visualization Rules To Preserve

For multi-dataset line plots:

- If a subplot contains two or more lines, give each line transparency.
- For dense dataset plots, keep at most five dataset lines per subplot.
- Use distinct colors; do not allow all dataset lines to share one color.
- Shared y-axis ranges are required for comparable RMSE/relative-L2 panels.
- Moving averages are useful for line plots only when raw lines are too noisy.
- Do not hide raw data entirely; keep raw versions where possible.

For loss heat maps:

- Use raw values in the heat map unless a smoothed heat map is explicitly needed.
- Put group labels outside the colored group bar so text does not overlap color.
- Keep group separators, because the gray/white separators make dataset groups
  readable.
- For polished loss heat maps, use a color direction where larger loss is more
  visually intense/bright and smaller loss is visually darker, if that is the
  chosen convention for the final figure set.

For line plots below heat maps:

- If the line is already a group average over many datasets, epoch moving average
  is usually unnecessary.
- If the plot is a raw per-dataset line plot with many overlapping lines, use
  transparency and optionally a moving-average companion figure.
- Keep the line panel short and wide below the heat map.

For delta spectrum plots:

- Raw heat maps are valuable because they show the true noisy spectrum.
- The line plot below the heat map can use smoothing, because raw spectral lines
  are often too noisy to read.
- If smoothing is used for spectrum lines, clearly distinguish whether smoothing
  is over epochs or Fourier modes.
- For the final spectrum heat map, the raw no-moving-average version may be the
  cleanest honest representation.

For high-frequency share plots:

- The y-axis should match the observed scale. If values live below about 0.05,
  do not force a 0 to 1 axis.
- Plot multiple high-frequency thresholds, such as top 1%, 2%, 5%, 10%, 20%,
  and 50% Fourier-mode energy shares, when useful.

## 5. Selected Final Visualization Files

The current selected figure folder contains only PNG files:

```text
/workspace/polished_selected_download_20260601/corrected_attack_loss_three_lines_plus_buckets.png
/workspace/polished_selected_download_20260601/corrected_relative_l2_full_heatmap_raw_group_line.png
/workspace/polished_selected_download_20260601/corrected_rmse_full_heatmap_raw_group_line.png
/workspace/polished_selected_download_20260601/polished_checkpoint_style_delta_fft_raw_heatmap_25epoch_smoothed_lines.png
/workspace/polished_selected_download_20260601/relative_l2_grouped_shared_y_distinct_datasets_max5_high_transparency_raw.png
/workspace/polished_selected_download_20260601/relative_l2_grouped_shared_y_distinct_datasets_max5_high_transparency_ma25.png
/workspace/polished_selected_download_20260601/rmse_grouped_shared_y_distinct_datasets_max5_high_transparency_raw.png
/workspace/polished_selected_download_20260601/rmse_grouped_shared_y_distinct_datasets_max5_high_transparency_ma25.png
```

The selected set is meant to cover:

- attack loss and epsilon-bucket behavior
- relative-L2 heat map over 52 datasets and all evaluation epochs
- RMSE heat map over 52 datasets and all evaluation epochs
- delta FFT spectrum heat map and smoothed spectrum lines
- max-five-line dataset panels for relative-L2 and RMSE, both raw and MA25

## 6. Attack-Loss Interpretation

Definitions:

```text
clean_loss_before_attack:
  MSE(model_t(x_clean), solver(x_clean))

adv_loss_after_attack:
  MSE(model_t(x_adv), solver(x_adv))

attack_loss_gain:
  adv_loss_after_attack - clean_loss_before_attack

relative_attack_gain:
  attack_loss_gain / clean_loss_before_attack
```

The important interpretation from the plots:

- Absolute attack gain decreases over training.
- Clean loss also decreases, so a lower adversarial loss after attack does not
  by itself prove that the attack space is shrinking.
- Relative attack gain is necessary to separate "attack became harder" from
  "clean loss got smaller."
- In the current run, absolute gain decreases more clearly than relative gain.
  This is evidence of improvement, but it is not by itself a complete proof of
  robustification.

## 7. Delta Spectrum Interpretation

The current `fast_replace_linf` run produces rectangular/sign-like deltas
because the attack uses an L-infinity sign replacement step:

```text
delta_i = +epsilon if grad_i > 0
delta_i = -epsilon if grad_i < 0
```

That is why the saved delta fields can look like two-level square-wave
perturbations.

This is not the intended `p=2, q=2` geometry. Under `p=2, q=2`, the perturbation
should generally be a normalized continuous gradient-like direction, not a pure
sign pattern.

Observed spectrum behavior for this L-infinity run:

- delta total variation increases
- sign-change fraction increases
- spectral centroid increases
- high-frequency energy share increases
- the FFT heat map shows energy moving toward higher Fourier modes over epochs

So within this run, the perturbation becomes more high-frequency over training.
The caveat is that this behavior is tied to the sign-replace L-infinity attack
and must be checked again for the intended `p=2, q=2` run.

## 8. Jacobian/SVD Object Definitions

For a fixed input point `x`, the seven Jacobian objects are:

```text
1. J_solver(x)
2. J_baseline_model(x)
3. J_adv_model(x)
4. J_clean_plus_adv_model(x)
5. J_baseline_error(x)       = J_baseline_model(x) - J_solver(x)
6. J_adv_error(x)            = J_adv_model(x) - J_solver(x)
7. J_clean_plus_adv_error(x) = J_clean_plus_adv_model(x) - J_solver(x)
```

The key object for "model became more solver-like" is:

```text
J_error = J_model - J_solver
```

The desired result is not simply:

```text
J_model becomes small
```

The desired result is:

```text
J_model becomes closer to J_solver
```

That can happen even if `||J_model||_2` does not decrease.

## 9. Fixed-10 Top100 Jacobian/SVD Study

The 10 fixed points were selected from the older completed representative20
study to avoid recomputing expensive solver and baseline Jacobians.

Selected source sample IDs:

```text
0, 1, 6, 7, 10, 11, 12, 15, 18, 19
```

Split:

```text
2 train
2 test
6 generalization
```

Generalization coverage:

- near Gaussian lower correlation
- near Gaussian higher correlation
- target shifted Gaussian
- Matern
- far sawtooth additive pattern
- far offset/shift pattern

Reused from the old full SVD:

- `J_solver`
- baseline `J_model`
- baseline `J_error`
- old clean+ADV `J_model`
- old clean+ADV `J_error`

Newly computed in this run:

- new 1000-epoch ADV-only `J_model`
- new 1000-epoch ADV-only `J_error = J_adv1000 - J_solver`

All stored SVD files keep:

```text
top 100 singular values
top 100 left singular vectors
top 100 right singular vectors
```

## 10. Error Jacobian Spectral Norms

Mean spectral norms over the 10 fixed points:

| object | mean | std | min | max |
|---|---:|---:|---:|---:|
| baseline error | 2.34177 | 1.50608 | 0.786506 | 5.63225 |
| old ADV-only500 error | 1.06689 | 0.766451 | 0.291521 | 2.88800 |
| new ADV1000 error | 1.09426 | 0.899164 | 0.320148 | 3.40401 |
| clean+ADV500 error | 1.16228 | 0.799750 | 0.306270 | 2.97388 |

Relative to baseline:

| model | mean ratio to baseline | mean decrease | smaller than baseline |
|---|---:|---:|---:|
| old ADV-only500 | 0.4718 | 52.82% | 10/10 |
| new ADV1000 | 0.4852 | 51.48% | 10/10 |
| clean+ADV500 | 0.5714 | 42.86% | 9/10 |

Interpretation:

- All adversarially trained models strongly reduce `J_error`.
- The new 1000-epoch ADV-only model is much better than baseline.
- The old ADV-only500 has a slightly lower top-1 error spectral norm mean than
  the new ADV1000.
- However, the new ADV1000 is better than old ADV-only500 on 7/10 individual
  points; a few larger cases raise its mean.

## 11. Error Spectrum Contraction By Rank

Mean singular values and decrease relative to baseline:

| rank | baseline | old ADV500 | old decrease | new ADV1000 | new decrease | clean+ADV500 | clean decrease |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.34177 | 1.06689 | 54.44% | 1.09426 | 53.27% | 1.16228 | 50.37% |
| 2 | 1.06797 | 0.39042 | 63.44% | 0.37221 | 65.15% | 0.33454 | 68.67% |
| 3 | 0.48041 | 0.22928 | 52.28% | 0.19625 | 59.15% | 0.18455 | 61.58% |
| 4 | 0.35477 | 0.20840 | 41.26% | 0.15094 | 57.45% | 0.14421 | 59.35% |
| 5 | 0.31377 | 0.16444 | 47.59% | 0.12067 | 61.54% | 0.12623 | 59.77% |
| 10 | 0.18346 | 0.10206 | 44.37% | 0.07507 | 59.08% | 0.08127 | 55.70% |
| 20 | 0.07870 | 0.04455 | 43.39% | 0.03404 | 56.74% | 0.03601 | 54.24% |
| 50 | 0.02654 | 0.00648 | 75.60% | 0.00464 | 82.52% | 0.00557 | 79.01% |
| 100 | 0.01898 | 0.00329 | 82.69% | 0.00260 | 86.30% | 0.00356 | 81.25% |

The key nuance:

```text
top-1 error norm:
  old ADV500 is slightly better.

top5/top20/top100 error spectrum:
  new ADV1000 contracts the broader error spectrum more strongly.
```

## 12. Top-k Mean Error Spectrum

Top-k mean singular-value decrease:

| top-k | old ADV500 decrease | new ADV1000 decrease | clean+ADV500 decrease |
|---:|---:|---:|---:|
| 1 | 52.82% | 51.48% | 42.86% |
| 5 | 53.05% | 58.37% | 55.28% |
| 10 | 52.01% | 58.93% | 55.97% |
| 20 | 50.74% | 58.87% | 56.12% |
| 50 | 52.11% | 59.96% | 57.61% |
| 100 | 55.78% | 63.10% | 60.50% |

This is the strongest evidence that the new 1000-epoch model reduces the
overall error spectrum, even though its largest single error singular value is
not the best.

## 13. Model Jacobian Spectral Norms

The model Jacobian itself does not simply shrink:

| object | spectral norm mean |
|---|---:|
| solver | 5.799 |
| baseline model | 5.134 |
| old ADV-only500 model | 5.568 |
| new ADV1000 model | 5.620 |
| clean+ADV500 model | 5.623 |

Interpretation:

```text
Adversarial training does not just flatten J_model.
Instead, J_model moves toward the solver's scale while J_error shrinks.
```

This matters because a small `J_model` alone would not prove solver-like local
behavior.

## 14. Model-vs-Solver Singular-Value Similarity

Top-k relative L2 difference between model singular values and solver singular
values:

| top-k | baseline | old ADV500 | new ADV1000 | clean+ADV500 |
|---:|---:|---:|---:|---:|
| 1 | 0.09874 | 0.03450 | 0.03714 | 0.02481 |
| 5 | 0.11232 | 0.04055 | 0.03738 | 0.02768 |
| 10 | 0.11163 | 0.04148 | 0.03759 | 0.02817 |
| 20 | 0.11668 | 0.04358 | 0.03787 | 0.02996 |
| 50 | 0.11969 | 0.04543 | 0.03916 | 0.03231 |
| 100 | 0.12211 | 0.04565 | 0.03929 | 0.03266 |

Interpretation:

- All trained models have singular-value spectra closer to solver than baseline.
- New ADV1000 is closer than old ADV500 for top5 through top100.
- Clean+ADV500 is best on singular-value magnitude similarity in this fixed-10
  subset, although it is not best on `J_error` spectral norm.

## 15. Singular Vector / Subspace Similarity

Top20 model-vs-solver subspace similarity:

| model | singular-value rel diff | left subspace cosine | right subspace cosine |
|---|---:|---:|---:|
| baseline | 0.11668 | 0.87265 | 0.80289 |
| old ADV500 | 0.04358 | 0.97331 | 0.93699 |
| new ADV1000 | 0.03787 | 0.98587 | 0.96758 |
| clean+ADV500 | 0.02996 | 0.98240 | 0.95558 |

Meaning:

- Right singular vectors correspond to input perturbation directions.
- Left singular vectors correspond to output response directions.
- New ADV1000 is more solver-like than old ADV500 in top20 left and right
  subspaces.
- The right-subspace improvement is especially important because it means the
  model's dangerous input directions are more aligned with the solver's
  dangerous input directions.

## 16. Overall Interpretation

The full picture is:

```text
1. J_model itself does not shrink.
2. J_model moves closer to the solver's scale and structure.
3. J_error shrinks strongly after adversarial training.
4. New ADV1000 does not beat old ADV500 on the single largest error singular
   value mean.
5. New ADV1000 does beat old ADV500 on broader top5/top20/top100 error-spectrum
   contraction.
6. New ADV1000 also has more solver-like top20/top50 singular-vector subspaces.
```

So the correct conclusion is not:

```text
new ADV1000 is uniformly better on every metric
```

The correct conclusion is:

```text
new ADV1000 is clearly better than baseline and has a more solver-like local
Jacobian shape than old ADV500, but its maximum error Jacobian spectral norm is
not uniformly better than old ADV500.
```

## 17. Files Generated For This Analysis

Important fixed-10/top100 files:

```text
forensics/burgers_zero_adv_training_jacobian_svd_20260601_rep20subset10_top100/README_FIXED10_TOP100_UPDATE.md
forensics/burgers_zero_adv_training_jacobian_svd_20260601_rep20subset10_top100/README_REUSE_PLAN.md
forensics/burgers_zero_adv_training_jacobian_svd_20260601_rep20subset10_top100/fixed10_from_representative20_manifest.csv
forensics/burgers_zero_adv_training_jacobian_svd_20260601_rep20subset10_top100/runtime.csv
forensics/burgers_zero_adv_training_jacobian_svd_20260601_rep20subset10_top100/jacobian_svd_summary.csv
forensics/burgers_zero_adv_training_jacobian_svd_20260601_rep20subset10_top100/aggregate_jacobian_svd_summary.csv
forensics/burgers_zero_adv_training_jacobian_svd_20260601_rep20subset10_top100/top100_singular_values_long_all7.csv
forensics/burgers_zero_adv_training_jacobian_svd_20260601_rep20subset10_top100/top100_rankwise_left_right_vector_absdot_vs_solver.csv
forensics/burgers_zero_adv_training_jacobian_svd_20260601_rep20subset10_top100/topk_model_vs_solver_similarity.csv
forensics/burgers_zero_adv_training_jacobian_svd_20260601_rep20subset10_top100/topk_model_vs_solver_similarity_split_summary.csv
forensics/burgers_zero_adv_training_jacobian_svd_20260601_rep20subset10_top100/same_point_error_spectral_norm_comparison_top100.csv
forensics/burgers_zero_adv_training_jacobian_svd_20260601_rep20subset10_top100/same_point_error_spectral_norm_split_summary_top100.csv
```

The NPZ files under `sample_*/` contain the dense Jacobian and top100 SVD
triplets for each fixed point/object.

## 18. Code Changes Relevant To This Analysis

The key script is:

```text
tools/compare_burgers_adversarial_jacobian_svd.py
```

It now supports top-k SVD through:

```text
--svd-method topk
--svd-solver propack
--top-k 100
```

This avoids computing a full 1024-by-1024 SVD when only the leading singular
triplets are needed. The stored output still includes the full dense Jacobian,
but only the requested top-k singular vectors/values.

## 19. Backup Plan

GitHub should contain:

- this Markdown record
- relevant source scripts
- lightweight summaries/manifests

R2 should contain:

- the full fixed-10/top100 result directory, including NPZ data
- selected visualization PNGs
- selected visualization zip
- the Markdown records

Credentials are intentionally omitted from this document and should not be
committed.
