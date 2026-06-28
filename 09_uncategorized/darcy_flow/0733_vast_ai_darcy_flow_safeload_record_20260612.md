# Vast AI Darcy Flow SafeLoad Record - 2026-06-12

This note records the Darcy Flow binary-coefficient adversarial-training results,
visualization outputs, attack benchmarks, Jacobian diagnostics, SVD timing tests,
and the current mechanistic interpretation developed on the Vast AI instance.
No credentials or tokens are recorded here.

## Branch And Scope

- Working branch for this record: `vastai.safeload`.
- Prior working branch: `vast-ai-darcy-flow`.
- Task: 2D Darcy Flow FNO with binary coefficient fields.
- Binary coefficient values: `3` and `12` only.
- Main generated dataset root: `generalization_datasets_darcy_binary_loss3targeted_20260611/darcy`.
- Main first-stage time-matched tag: `20260612_full50_timematched_1000c`.
- Baseline model checkpoint: `2D_Darcy_FNO2d/saved_models/2D/darcy_N1500_nx85_m64_w60_e500_20260528/best.pt`.

## Binary Generalization Dataset Audit

Audit file: `analysis_outputs/darcy_loss3targeted_full50_dataset_uniqueness_audit_20260612.json`.

The generated Darcy generalization set satisfies the binary-data requirement:

| item | value |
|---|---:|
| dataset count | 50 |
| unique hashes | 50 |
| duplicate hash count | 0 |
| all values binary `3/12` | true |
| tensor shape set | `(50, 85, 85)` |
| high-value fraction range | `0.1180` to `0.2417` |
| edge-density range | `0.00619` to `0.34890` |
| pairwise Hamming min / median / max | `0.2074 / 0.2992 / 0.3797` |
| pairwise Hamming p05 / p95 | `0.2145 / 0.3589` |

This supports the intended diversity: smooth, fine, high-pass, band-pass, wave,
blocky, rectangular, and cellular/blob-like patterns are all represented, while
all coefficient fields remain binary.

## Time-Matched Adversarial Training Results

Combined linear-scale report:
`docs/darcy_loss123physics_full50_timematched_long_20260612_full50_timematched_1000c_combined_four_method_burgers_style_linear.md`.

All four first-stage runs logged `52` datasets per epoch: original train,
original test, and all `50` generated binary generalization datasets.

| method | epochs completed | elapsed minutes | logged gen datasets | final gen relative L2 | full50 improved count | full50 mean delta | peak allocated GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| loss1 | 1000 | 77.0066 | 50 | 0.101024 | 0 | 0.00968697 | 7.97360 |
| loss2 | 1026 | 77.0585 | 50 | 0.0864258 | 44 | -0.00491106 | 7.45051 |
| loss3 | 1011 | 77.3917 | 50 | 0.0728309 | 50 | -0.0185060 | 7.45263 |
| physics | 1040 | 76.6887 | 50 | 0.0921814 | 18 | 0.000844458 | 7.45259 |

Primary training conclusion: under approximately equal wall-clock time, `loss3`
has the lowest final generated-dataset relative L2 and improves all 50 generated
binary datasets.

Burgers-style Darcy plots were generated under:

- `visualizations/darcy_loss123physics_full50_timematched_long_20260612_full50_timematched_1000c/combined_four_method_burgers_style_linear`
- Per-method polished outputs under `visualizations/darcy_loss123physics_full50_timematched_long_20260612_full50_timematched_1000c/{loss1,loss2,loss3,physics}_polished_variable_epoch`

These include epoch curves, wall-clock curves, attack-gain curves, runtime/memory
traces, final generated-dataset reduction plots, heatmap-plus-line plots, and
delta FFT diagnostics where probe data exists.

## 20-Step Adversarial Attack Benchmark

Benchmark report:
`analysis_outputs/darcy_attack20_52datasets_50samples_20260612_attack20_1000c_52datasets_50samples/README.md`.

Setup:

- Objective: shared solver-consistent `loss3` MSE attack for all models.
- Datasets: train + test + 50 generated generalization datasets.
- Samples per dataset: 50.
- Total samples per model: 2600.
- Attack steps: 20.
- Epsilon fraction: `0.025`.

Overall mean absolute loss growth:

| model | clean | attacked | absolute gain | samples |
|---|---:|---:|---:|---:|
| loss1 | 9.38239e-07 | 5.71414e-06 | 4.77590e-06 | 2600 |
| loss2 | 6.89173e-07 | 4.62150e-06 | 3.93232e-06 | 2600 |
| loss3 | 4.61900e-07 | 3.15314e-06 | 2.69124e-06 | 2600 |
| physics | 7.61504e-07 | 4.85337e-06 | 4.09187e-06 | 2600 |

Split-level mean absolute loss growth:

| model | train gain | test gain | generalization gain |
|---|---:|---:|---:|
| loss1 | 2.74234e-06 | 3.00025e-06 | 4.85208e-06 |
| loss2 | 3.70394e-07 | 6.70834e-07 | 4.06879e-06 |
| loss3 | 1.67747e-06 | 1.73059e-06 | 2.73073e-06 |
| physics | 2.81314e-06 | 2.70985e-06 | 4.14508e-06 |

Attack conclusion:

- By absolute gain, `loss3` is best overall.
- On the 50 generated generalization datasets, `loss3` is best on all `50/50` datasets.
- On the original train/test splits, `loss2` has the smallest absolute gain.

## Shared-Range Attack Heatmaps

Report:
`analysis_outputs/darcy_five_model_attack_heatmaps_20260612_loss3attack20_five_models_sharedrange/README.md`.

Figures:
`visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack20_five_models_sharedrange`.

Setup:

- Models: `baseline`, `loss1`, `loss2`, `loss3`, `fixed`/physics.
- Samples: 1 original test sample plus 4 generated generalization samples.
- Attack: 20-step binary solver-consistent `loss3` attack with epsilon fraction `0.025`.
- Each figure uses rows for models and columns for `initial condition`, `delta`,
  `initial + delta`, `model output`, `solver output`, and `model - solver`.
- Color ranges are globally shared across samples/models by semantic panel type.
- Verified arrays: 25 `.npz` files, all `85 x 85`; initial and attacked coefficients
  contain only `3/12`; deltas contain only `-9/0/9`.

Mean attack gain on these 5 selected samples:

| model | mean attack gain | mean attacked rel L2 |
|---|---:|---:|
| baseline | 3.63662e-06 | 0.156661 |
| loss1 | 3.69891e-06 | 0.157255 |
| loss2 | 3.29655e-06 | 0.160397 |
| loss3 | 2.55610e-06 | 0.150055 |
| fixed | 3.20832e-06 | 0.150615 |

## Jacobian Probe: Error-Aligned Sensitivity

Report:
`analysis_outputs/darcy_jacobian_probe_5gen_20260612_jacobian_5gen_1000c/README.md`.

Definitions:

```text
e = F(a) - u(a)
L(a) = 1/2 ||F(a) - u(a)||_2^2
g = dL/da = J^T e
```

Here `J` is the model Jacobian with respect to the coefficient field. The probe
uses 5 generated generalization samples and four trained models.

| model | mean rel L2 | mean `||J^T e||` | mean `||J^T e||^2` | mean `||J e||` | mean top sigma |
|---|---:|---:|---:|---:|---:|
| fixed | 0.0982516 | 1.53299e-04 | 2.75102e-08 | 6.23465e-05 | 0.00188965 |
| loss1 | 0.1067340 | 1.60822e-04 | 3.01177e-08 | 6.57173e-05 | 0.00182829 |
| loss2 | 0.0943854 | 1.34626e-04 | 2.24368e-08 | 5.52805e-05 | 0.00175943 |
| loss3 | 0.0785856 | 1.26450e-04 | 1.85496e-08 | 5.04957e-05 | 0.00210832 |

Key observation:

- `loss3` has the smallest mean rel L2, smallest `||J^T e||`, and smallest
  `||J^T e||^2`.
- `loss3` does not have the smallest top singular value; it has the largest mean
  top sigma among these four models.
- `loss2` has the smallest mean top sigma.

This supports the hypothesis that `loss3` reduces error-aligned sensitivity rather
than uniformly shrinking the worst-case model Jacobian.

## Full Jacobian SVD And Block Projection

Full one-matrix benchmark:
`analysis_outputs/darcy_jacobian_svd_benchmark_full_loss3_one_matrix/README.md`.

Clarification: full dimension `7225` is `85 x 85`, not `84 x 84`. Block/2 uses an
orthonormal `2 x 2` block basis on the top-left `84 x 84` crop, giving dimension
`42 x 42 = 1764`.

For one smooth sample with the `loss3` model:

| mode | dimension | Jacobian seconds | SVD seconds | total seconds | sigma max |
|---|---:|---:|---:|---:|---:|
| full | 7225 x 7225 | 42.684 | 26.918 | 69.602 | 0.00231427 |
| block/2 | 1764 x 1764 | 13.064 | 0.554 | 13.618 | 0.00227061 |
| block/4 | 441 x 441 | 5.561 | 0.121 | 5.682 | 0.00225469 |

Full-vs-block/2 top-20 comparison report:
`analysis_outputs/darcy_block2_full_svd_vector_compare_20260612_loss3_3samples_top20/README.md`.

For 3 samples with `loss3`:

| sample | full total sec | block/2 total sec | full sigma1 | block/2 sigma1 | rel error |
|---|---:|---:|---:|---:|---:|
| smooth_idx0 | 76.45 | 11.60 | 0.00231427 | 0.00227061 | -1.89% |
| highpass_idx0 | 75.70 | 11.34 | 0.00194708 | 0.00185930 | -4.51% |
| wave_idx0 | 74.31 | 11.16 | 0.00231545 | 0.00225953 | -2.42% |

Mean top-k singular value relative error:

| k | mean abs relative error |
|---:|---:|
| 5 | 3.82% |
| 10 | 4.28% |
| 20 | 4.92% |

Same-rank vector alignment after projecting full vectors to block/2:

| k | mean right abs dot | mean left abs dot | mean right projection energy | mean left projection energy |
|---:|---:|---:|---:|---:|
| 5 | 0.999 | 0.999 | 0.930 | 0.995 |
| 10 | 0.991 | 0.990 | 0.924 | 0.992 |
| 20 | 0.972 | 0.971 | 0.917 | 0.985 |

Subspace alignment:

| k | mean right principal cosine | mean left principal cosine |
|---:|---:|---:|
| 5 | 1.000 | 0.999 |
| 10 | 0.999 | 0.998 |
| 20 | 0.997 | 0.996 |

Interpretation: block/2 gives a much faster approximation to the leading singular
structure. It slightly underestimates singular values, especially in high-frequency
samples, but the leading directions and subspaces are highly aligned.

## Sigma1 From Block Vector Plus Power Refinement

Report:
`analysis_outputs/darcy_sigma1_from_block_vector_20260612_loss3_6samples_blockvec_jvp/README.md`.

Question tested: if block/2 gives a good top right singular vector, can it be
lifted back to full space and used to estimate the full `sigma_max(J)` cheaply?

Three estimates were compared:

```text
sigma_block = sigma_max(P J L)
v0 = L v_block
sigma_lift = ||J v0||
u0 = normalize(J v0)
v1 = normalize(J^T u0)
sigma_power = ||J v1||
```

Mean absolute relative error versus explicit full sigma1 over 6 generated samples:

| estimate | mean abs rel error |
|---|---:|
| block/2 sigma1 | 2.89% |
| lifted `||Jv||` | 2.83% |
| one power step | 0.03% |

Per-sample results:

| sample | full sigma1 | block/2 sigma1 | block err | lifted Jv | lifted err | one-power | one-power err |
|---|---:|---:|---:|---:|---:|---:|---:|
| smooth_idx0 | 0.00231427 | 0.00227061 | -1.89% | 0.00227231 | -1.81% | 0.00231346 | -0.04% |
| highpass_idx0 | 0.00194708 | 0.00185930 | -4.51% | 0.00186088 | -4.43% | 0.00194628 | -0.04% |
| wave_idx0 | 0.00231545 | 0.00225953 | -2.42% | 0.00226099 | -2.35% | 0.00231462 | -0.04% |
| bandpass_idx0 | 0.00229337 | 0.00223824 | -2.40% | 0.00223937 | -2.35% | 0.00229295 | -0.02% |
| blocky_idx0 | 0.00195464 | 0.00191609 | -1.97% | 0.00191710 | -1.92% | 0.00195377 | -0.04% |
| rectangles_idx0 | 0.00196784 | 0.00188590 | -4.16% | 0.00188737 | -4.09% | 0.00196777 | -0.00% |

Timing summary for the new samples:

- Full reference mean: approximately `68.8` seconds per sample.
- Block/2 reference mean: approximately `11.2` seconds per sample.
- Lifted JVP plus one-power refinement: sub-second to around one second scale in
  these runs.

Conclusion: block/2 top vector is an excellent initialization for top singular
value estimation. Direct `||Jv||` after lifting is only a small improvement over
block/2 sigma. One full-space power-refinement step nearly matches explicit full
SVD for `sigma_max`. This method estimates the largest singular value/direction;
it does not directly recover the full spectrum or arbitrary lower singular values.

## Mechanistic Interpretation

The central mechanism hypothesis is:

```text
The top singular value measures worst-case input-output sensitivity, but
adversarial loss growth is controlled more directly by the error-aligned input
gradient J^T(F(a)-u(a)). Loss3 improves robustness by reducing error-aligned
sensitivity, not necessarily by uniformly shrinking the Jacobian operator norm.
```

Why this matters:

```text
sigma_max(J) = max_v ||J v||
```

This measures the strongest possible model-output response direction, independent
of the current prediction error.

For the squared error loss,

```text
L(a) = 1/2 ||F(a) - u(a)||_2^2
dL/da = J^T e, where e = F(a) - u(a)
```

The first-order loss increase under a small perturbation is

```text
L(a + delta) - L(a) ~= delta^T J^T e
```

Therefore attack growth should correlate more directly with `||J^T e||` or a
binary-feasible first-order gain than with `sigma_max(J)` alone.

Current evidence:

- 20-step attack gain: `loss3` is best overall and best on `50/50` generated
  generalization datasets.
- Jacobian probe: `loss3` has the smallest `||J^T e||` and `||J^T e||^2`.
- Top singular value: `loss3` has the largest mean top sigma in the 5-sample
  Jacobian probe, while `loss2` has the smallest.

This separation is useful: it shows that robustness improvement can arise from
smaller error-aligned sensitivity even when worst-case operator sensitivity is not
smallest.

## Proposed Next Evidence Package

The next strongest experiment should make the mechanism quantitative, not just
qualitative. Use a common sample set across attack and Jacobian diagnostics, then
plot and tabulate:

1. `attack20 absolute gain` vs `||J^T e||`.
2. `attack20 absolute gain` vs `||J^T e||^2`.
3. `attack20 absolute gain` vs `sigma_max(J)`.
4. `attack20 absolute gain` vs `sigma_max(J) * ||e||`.
5. `attack20 absolute gain` vs binary-feasible first-order gain.

Also compute correlations:

- Pearson correlation.
- Spearman correlation.
- Linear-regression `R^2`.
- Partial correlation or regression controlling for clean loss / clean error.

The most important singular-vector diagnostic is the decomposition

```text
||J^T e||^2 = sum_i sigma_i^2 <e, u_i>^2
```

For representative samples, compare across models:

| model | sigma1 | `|<e/||e||, u1>|` | `||J^T e||` | attack20 gain |
|---|---:|---:|---:|---:|
| loss1 | TBD | TBD | TBD | TBD |
| loss2 | TBD | TBD | TBD | TBD |
| loss3 | TBD | TBD | TBD | TBD |
| physics | TBD | TBD | TBD | TBD |

A strong paper-quality claim would be:

```text
For Darcy Flow, adversarial loss growth is better predicted by error-aligned
input sensitivity and binary-feasible first-order gain than by Jacobian spectral
norm. Loss3 reduces the former while not necessarily reducing the latter.
```


## Ten-Sample Metric-Attack Correlation Run

Correlation report:
`analysis_outputs/darcy_metric_correlation_10samples_20260612_10samples_loss3attack20_corr_chunk64/README.md`.

Fixed-effect supplement:
`analysis_outputs/darcy_metric_correlation_10samples_20260612_10samples_loss3attack20_corr_chunk64/FIXED_EFFECT_INTERPRETATION.md`.

Figures:
`visualizations/darcy_metric_correlation_10samples_20260612_10samples_loss3attack20_corr_chunk64`.

Script:
`tools/run_darcy_metric_correlation_10samples_20260612.py`.

Setup:

- Samples: 10 diverse generated Darcy generalization samples.
- Models: `loss1`, `loss2`, `loss3`, and `physics`.
- Total model-sample rows: 40.
- Attack: shared 20-step binary `loss3` attack, epsilon fraction `0.025`.
- Sigma metric: block/2 top singular vector lifted to the full grid, followed by
  one full-space singular-vector power-refinement step.
- Additional metrics: clean loss, error norm, `||J^T e||`, `||J^T e||^2`,
  binary-feasible first-order gain, `sigma_max * ||e||`, top-left singular-vector
  error alignment, and top-1 weighted error energy.

The first full attempt used `block-row-chunk=256` and hit CUDA OOM because it
needed an extra 18.90 GiB while the stage2 training process was also resident on
the GPU. The stable completed run used `block-row-chunk=64`.

Output verification:

| file | rows |
|---|---:|
| `metrics_by_model_sample.csv` | 40 data rows + header |
| `correlations.csv` | 85 data rows + header |
| `selected_samples.csv` | 10 data rows + header |

Mean metrics by model in the 10-sample run:

| model | mean attack gain | mean `||J^T e||` | mean `sigma_max` | mean binary first-order gain |
|---|---:|---:|---:|---:|
| loss1 | 4.87872e-06 | 1.25913e-04 | 0.00178995 | 1.56671e-06 |
| loss2 | 3.84134e-06 | 1.02028e-04 | 0.00172110 | 1.25888e-06 |
| loss3 | 2.64413e-06 | 9.28814e-05 | 0.00206670 | 1.20221e-06 |
| physics | 4.20670e-06 | 1.17896e-04 | 0.00185268 | 1.51572e-06 |

The raw overall correlation is not the right final evidence by itself, because it
mixes model-level effects with sample-level effects. In raw overall correlation,
`sigma_max` has a strong negative Spearman correlation with attack gain because
`loss3` has lower attack gain while also having relatively high `sigma_max`.

Key Spearman correlations under different fixed-effect views:

| scope | clean loss | `||J^T e||` | `||J^T e||^2` | binary first-order | `sigma_max` | `sigma*||e||` | `|<e,u1>|` |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw overall | -0.323 | -0.221 | -0.221 | -0.267 | -0.592 | -0.304 | 0.258 |
| within model | -0.716 | -0.545 | -0.582 | -0.599 | -0.377 | -0.556 | -0.006 |
| within sample | 0.477 | 0.620 | 0.529 | 0.523 | -0.530 | 0.514 | 0.680 |
| model+sample residual | -0.537 | -0.418 | -0.522 | -0.346 | 0.224 | -0.453 | 0.421 |

Interpretation:

- The model means strengthen the main observation: `loss3` has the smallest mean
  attack gain, smallest mean `||J^T e||`, and smallest mean binary first-order
  gain, while its mean `sigma_max` is the largest.
- For comparing models on the same initial condition, the `within sample` view is
  the most relevant. In that view, `||J^T e||`, `||J^T e||^2`, binary first-order
  gain, and top-left error alignment are positively correlated with attack gain,
  while `sigma_max` alone is negatively correlated.
- The result supports the more careful claim that Darcy adversarial loss growth is
  better explained by error-aligned sensitivity and feasible first-order attack
  geometry than by worst-case Jacobian spectral norm alone.
- The 10-sample result is a strong case study, not yet the final statistical proof;
  the next scaling step should run the same fixed-effect analysis on more samples.

## Stage2 Continuation Status At Time Of This Record

The stage2 continuation launcher was active when this record was written:

- Launcher: `tools/run_darcy_full50_timematched_stage2_20260612.sh`.
- Current active run at the time checked: `loss1` continuation.
- Run name: `darcy_binary_loss3targeted_loss1_continue2000ep_from_1000ep_full50_timematched_20260612_stage2_2000_from_1000c`.
- Initial checkpoint: first-stage `loss1` epoch-1000 checkpoint.
- Resume offsets: epoch `1000`, global step `1000`.
- Planned stage2 rough total from launcher: about `10.405` hours across all four methods.

This status is only a snapshot; final stage2 results should be documented after
that launcher completes.

## Key Scripts Added Or Used

- `tools/run_darcy_full50_timematched_long_20260612.sh`
- `tools/run_darcy_full50_timematched_stage2_20260612.sh`
- `tools/run_darcy_attack20_52datasets_50samples_20260612.py`
- `tools/run_darcy_jacobian_probe_5gen_20260612.py`
- `tools/benchmark_darcy_jacobian_svd_20260612.py`
- `tools/compare_darcy_block2_full_svd_vectors_20260612.py`
- `tools/estimate_darcy_sigma_from_block_vector_20260612.py`
- `tools/plot_darcy_five_model_attack_heatmaps_20260612.py`
- `tools/plot_darcy_loss123physics_adv_training_20260611.py`

## Key Output Directories

- `analysis_outputs/darcy_attack20_52datasets_50samples_20260612_attack20_1000c_52datasets_50samples`
- `analysis_outputs/darcy_jacobian_probe_5gen_20260612_jacobian_5gen_1000c`
- `analysis_outputs/darcy_block2_full_svd_vector_compare_20260612_loss3_3samples_top20`
- `analysis_outputs/darcy_sigma1_from_block_vector_20260612_loss3_6samples_blockvec_jvp`
- `analysis_outputs/darcy_five_model_attack_heatmaps_20260612_loss3attack20_five_models_sharedrange`
- `visualizations/darcy_loss123physics_full50_timematched_long_20260612_full50_timematched_1000c`
- `visualizations/darcy_attack20_52datasets_50samples_20260612_attack20_1000c_52datasets_50samples`
- `visualizations/darcy_jacobian_probe_5gen_20260612_jacobian_5gen_1000c`
- `visualizations/darcy_block2_full_svd_vector_compare_20260612_loss3_3samples_top20`
- `visualizations/darcy_sigma1_from_block_vector_20260612_loss3_6samples_blockvec_jvp`
- `visualizations/darcy_five_model_attack_heatmaps_20260612_loss3attack20_five_models_sharedrange`
