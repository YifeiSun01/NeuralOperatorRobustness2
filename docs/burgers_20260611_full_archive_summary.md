# Burgers 2026-06-11 Full Archive Summary

Date: 2026-06-11 UTC

Status: archive and handoff record for the 2026-06-11 Burgers generalization / robustness work. This file summarizes the discussion, generated datasets, clean inference, P2Q2 visuals, full 1024 SVD smoke test, runtime estimates, and future experiment notes.

No Cloudflare, GitHub, or S3 credential values are recorded in this document.

## 1. Storage And Workflow

Project root:

```text
/workspace/NeuralOperatorRobustness2
```

Branch:

```text
vast-ai
```

Python environment:

```text
/venv/adv_robust/bin/python
```

Workflow rule used in this archive:

- GitHub stores source code, shell scripts, Markdown records, and small reproducible instructions.
- R2 stores generated datasets, model artifacts, visualizations, forensics outputs, traces, full SVD NPZ files, and large result folders.
- `/workspace` on this Vast instance should not be treated as durable unless separately verified; important artifacts should be pushed to GitHub or copied to R2.

## 2. Workspace Instruction Files

The `/workspace` instruction files are instance-provided symlinks:

```text
/workspace/AGENTS.md -> /etc/vast_agents/base.md
/workspace/CLAUDE.md -> /etc/vast_agents/base.md
```

The arrow shown by `ls -l` means a symbolic link. These files are not experiment outputs and should not be interpreted as generated Burgers data.

The repository also has its own tracked instruction file:

```text
/workspace/NeuralOperatorRobustness2/AGENTS.md
```

## 3. Earlier Burgers Generalization Audit

Three earlier Burgers generalization roots were discussed and audited:

| root | status | interpretation |
| --- | --- | --- |
| `generalization_datasets/burgers` | earlier root | semantic OOD construction; real generalization dataset |
| `generalization_datasets_rmse_1p5_3x_all_ns50/burgers` | earlier root | semantic candidates and target-band filtering; real generalization dataset |
| `generalization_datasets_burgers_loss3_selective_search/round_03/burgers` | third root | problematic: built from train/test samples attacked by loss3 and then treated as generalization |

Main conclusion:

- The first two roots follow the intended semantic generalization idea.
- The third root is not a clean semantic generalization root because it reused attacked train/test-derived samples and had repeated source/attack parameter configurations.
- The replacement work below was designed to keep the semantic OOD idea while producing a dataset where loss3 is clean-generalization-best.

Primary audit record:

```text
docs/burgers_semantic_loss3fav_round01_generation_audit_20260611.md
```

## 4. Rejected Sawtooth-Heavy Round

A first semantic replacement attempt selected 50 strict loss3 winners, but it was visually and distributionally unsatisfactory.

Root:

```text
generalization_datasets_burgers_semantic_loss3fav_search_20260611/round_01
```

Key facts:

| item | value |
| --- | ---: |
| generated candidates | 720 |
| selected datasets | 50 |
| unique parameter signatures | 50/50 |
| loss3 clean winners | 50/50 |
| selected root size | about 79 MB |
| candidate pool size | about 1.2 GB |

Selected family counts:

| family | count |
| --- | ---: |
| sawtooth | 27 |
| square_wave | 6 |
| spike_train | 6 |
| powerlaw_fourier | 6 |
| matern | 4 |
| sine_mixture | 1 |

Why it was rejected:

- Too many sawtooth/square/spike-train style curves.
- Some value ranges were too extreme, including `[-3, 3]` and `[-2, 2]`.
- The initial conditions looked too jagged and too far from the expected clean Burgers train/test style.
- Before attack, predictions were already visually poor in many panels, making the attack visualization less useful.

## 5. Smooth-Dominant Semantic Screen

A smoother replacement dataset was generated after the sawtooth-heavy round was rejected.

Root:

```text
generalization_datasets_burgers_semantic_smooth_loss3_screen_20260611/round_00
```

Policy:

- No attack-generated samples.
- Smooth families dominate: Gaussian, Matern, power-law Fourier, sine mixture.
- Sawtooth and square wave are capped to tiny diagnostic counts.
- Spike train is removed.
- Hard range bound: `[-0.7, 1.7]`.
- Actual selected range stayed inside about `[-0.600, 1.500]`.
- Every selected dataset has a unique parameter signature.

Selected family counts:

| family | count |
| --- | ---: |
| gaussian | 15 |
| matern | 14 |
| powerlaw_fourier | 10 |
| sine_mixture | 8 |
| sawtooth | 2 |
| square_wave | 1 |

Clean screen result:

| metric | value |
| --- | ---: |
| selected datasets | 50 |
| duplicate parameter signatures | 0 |
| loss3 strict best count | 50/50 |
| loss3/best(loss1,loss2) RMSE ratio mean | 0.5797 |
| loss3/best(loss1,loss2) RMSE ratio median | 0.5964 |

Record:

```text
docs/burgers_semantic_smooth_loss3_screen_round00_20260611.md
```

## 6. Wide-Parameter Visible Dataset Before Targeted Replacement

The next dataset version emphasized visibly different parameter settings: correlation length, Matern `nu`, spectral decay, frequency, range, and total variation were widened so the plotted curves would not look almost identical.

Root:

```text
generalization_datasets_burgers_semantic_wideparam_visible_20260611/round_00
```

Observed issue:

- Loss3 was clean RMSE winner on only 32/50 datasets.
- The 18 non-winners were concentrated in Matern and sine-mixture cases.
- Non-winners tended to be higher frequency: spectral centroid about `39.69` versus `17.37` for winners; high-frequency fraction about `0.157` versus `0.0566`.
- Certain low `nu` Matern, overly smooth/simple sine, piecewise-linear, and fragile sharp cases were less favorable to loss3.

This analysis drove the final targeted replacement policy.

## 7. Final Loss3-Targeted Wide-Parameter Dataset

Final selected root:

```text
generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00
```

Burgers `.pt` dataset directory:

```text
generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00/burgers
```

Candidate pool:

```text
generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00_candidate_pool
```

Generator:

```text
tools/generate_burgers_wideparam_loss3_targeted_replacement_20260611.py
```

Selection policy:

- Semantic OOD generation only; no attack-generated generalization samples.
- Unique canonical parameter keys.
- Strict clean inference screen: loss3 must beat loss1 and loss2 in RMSE and relative L2.
- Keep wide, visible parameter variation.
- Avoid overusing sawtooth/square; remove spike train.

Final family counts:

| family | count |
| --- | ---: |
| gaussian | 12 |
| matern | 10 |
| powerlaw_fourier | 14 |
| sine_mixture | 12 |
| sawtooth | 1 |
| square_wave | 1 |
| spike_train | 0 |

Audit numbers:

| item | value |
| --- | ---: |
| datasets | 50 |
| unique parameter keys | 50 |
| actual global `x_min` | -0.65 |
| actual global `x_max` | 1.50 |
| hard lower bound | -0.70 |
| hard upper bound | 1.70 |
| spectral centroid min | 5.9669 |
| spectral centroid max | 32.1312 |
| total variation min | 0.00429 |
| total variation max | 0.08584 |

Record:

```text
docs/burgers_wideparam_loss3_targeted_replacement_20260611.md
```

## 8. Clean Inference On Final 50 Generalization Datasets

Clean inference means no adversarial attack was applied. Models used:

| model | checkpoint |
| --- | --- |
| baseline | `1D_Burgers/trained_models/attack_ready/burgers_nu0.001_fno1d_500/checkpoints/pytorch_fno1d_500.pt` |
| loss1 | `adversarial_training_runs/burgers_loss3_selective_round03_loss1_continue5000to8000_20260607/burgers/checkpoints/burgers_epoch8000_step024000.pt` |
| loss2 | `adversarial_training_runs/burgers_loss3_selective_round03_loss2_continue1000to2000_20260606/burgers/checkpoints/burgers_epoch2000_step006000.pt` |
| loss3 | `adversarial_training_runs/burgers_loss3_selective_round03_loss3_continue1000to1500_20260606/burgers/checkpoints/burgers_epoch1500_step004500.pt` |

Final 50-dataset generalization aggregate:

| model | clean RMSE mean | clean relative L2 mean |
| --- | ---: | ---: |
| baseline | 0.0299017 | 0.0577373 |
| loss1 | 0.0210265 | 0.0411709 |
| loss2 | 0.0222924 | 0.0435424 |
| loss3 | 0.0120496 | 0.0236437 |

RMSE reduction:

| comparison | reduction | win count |
| --- | ---: | ---: |
| loss1 vs baseline | 29.68% | 49/50 |
| loss2 vs baseline | 25.45% | 48/50 |
| loss3 vs baseline | 59.70% | 50/50 |
| loss3 vs loss1 | 42.69% | 50/50 |
| loss3 vs loss2 | 45.95% | 50/50 |

Statistical evidence:

| comparison | paired p-value |
| --- | ---: |
| loss3 vs loss1 RMSE | 4.92e-22 |
| loss3 vs loss2 RMSE | 5.73e-24 |

Records:

```text
forensics/burgers_semantic_wideparam_visible_loss3targeted_round00_clean_loss_final_models_20260611/summary.json
forensics/burgers_semantic_wideparam_visible_loss3targeted_round00_clean_loss_final_models_20260611/per_dataset_clean_metrics.csv
forensics/burgers_semantic_wideparam_visible_loss3targeted_round00_clean_loss_final_models_20260611/per_sample_clean_metrics.csv
```

## 9. Full 52-Dataset Clean Metrics

A 52-dataset report was generated by adding original train and original test splits to the 50 final generalization datasets.

Scope:

| group | samples |
| --- | ---: |
| original train | 1,350 |
| original test | 150 |
| 50 generalization datasets | 10,000 |
| total | 11,500 |

Train split clean RMSE:

| model | RMSE mean | RMSE reduction vs baseline |
| --- | ---: | ---: |
| baseline | 0.00875634 | - |
| loss1 | 0.000746925 | 91.47% |
| loss2 | 0.00140180 | 83.99% |
| loss3 | 0.00425008 | 51.46% |

Test split clean RMSE:

| model | RMSE mean | RMSE reduction vs baseline |
| --- | ---: | ---: |
| baseline | 0.00923196 | - |
| loss1 | 0.000836263 | 90.94% |
| loss2 | 0.00149737 | 83.78% |
| loss3 | 0.00437389 | 52.62% |

Interpretation:

- On in-distribution train/test, loss1 and loss2 have much lower clean loss than loss3.
- On the 50 final wide-parameter generalization datasets, loss3 is clean RMSE best on 50/50 datasets.
- This is a generalization-targeted result, not a claim that loss3 is best on original train/test.

Records:

```text
docs/burgers_loss3targeted_52dataset_clean_metrics_20260611.md
docs/burgers_loss3targeted_52dataset_clean_metrics_appendix_20260611.md
forensics/burgers_semantic_wideparam_visible_loss3targeted_round00_52dataset_clean_metrics_20260611/per_dataset_52_clean_metrics.csv
forensics/burgers_semantic_wideparam_visible_loss3targeted_round00_52dataset_clean_metrics_20260611/per_dataset_52_compact_rmse_relative_l2_reductions.csv
forensics/burgers_semantic_wideparam_visible_loss3targeted_round00_52dataset_clean_metrics_20260611/per_sample_52_clean_metrics.csv
```

## 10. Spectrum Discussion

In the dataset discussion, `spectrum` refers to the Fourier-domain energy/magnitude distribution of an initial condition as a function of frequency. It is not a separate physical field.

Useful interpretation:

- Smooth Gaussian/Matern fields have most energy at low frequencies and decay quickly.
- Short correlation length or rougher parameters move more energy into higher frequencies.
- Power-law Fourier fields directly control spectral decay through parameters like `alpha` and `k0`.
- Sine mixtures show explicit frequency peaks at chosen sinusoidal frequencies.
- Sawtooth/square waves have sharper discontinuities or corners, so their Fourier spectra contain stronger high-frequency harmonics.

This is why spectrum galleries were used together with initial-condition galleries: the curve shape shows visible behavior, while the spectrum shows how energy is distributed across low/mid/high frequencies.

## 11. P2Q2 Comparison-Dense Multi-Sample Attack Visuals

Final visual run used the final loss3-targeted wide-parameter dataset.

Runner:

```text
tools/run_burgers_wideparam_loss3targeted_round00_p2q2_diverse_multi_visuals_batched_20260611.py
```

Plotter updated for wrapped labels:

```text
tools/plot_burgers_round03_p2q2_combined_attack_panels.py
```

Trace root:

```text
forensics/burgers_semantic_wideparam_visible_loss3targeted_round00_p2q2_diverse_multi_sample_attack_visuals_batched_20260611
```

Final wrapped-label visualization root:

```text
visualizations/burgers_semantic_wideparam_visible_loss3targeted_round00_p2q2_comparison_dense_diverse_multi_sample_batched_wrapped_labels_20260611/comparison_dense
```

Attack settings:

| setting | value |
| --- | ---: |
| p | 2 |
| q | 2 |
| attack steps | 100 |
| record every | 2 steps |
| RMS epsilon | 0.12 |
| alpha RMS | 0.012 |
| groups | 5 |
| samples per group | 1 test + 5 generalization |
| total batched samples | 30 |
| PNG outputs | 30 |

Final attacked MSE means by group:

| group | baseline | loss1 | loss2 | loss3 |
| --- | ---: | ---: | ---: | ---: |
| group00 | 1.2439e-02 | 6.0062e-03 | 6.0574e-03 | 3.2049e-03 |
| group01 | 1.6045e-02 | 6.9050e-03 | 7.7222e-03 | 4.0798e-03 |
| group02 | 1.2800e-02 | 6.5433e-03 | 5.9747e-03 | 3.9501e-03 |
| group03 | 1.9085e-02 | 7.8953e-03 | 6.4595e-03 | 5.1924e-03 |
| group04 | 1.5852e-02 | 7.3700e-03 | 6.8700e-03 | 4.1244e-03 |

Interpretation:

- Loss3 has the lowest final attacked MSE mean in all 5 displayed groups.
- The visual groups use 25 distinct generalization datasets with no reuse.
- The group selection intentionally mixes smooth kernels, power-law spectra, sine mixtures, and the tiny sharp-case quota.

## 12. Label Wrapping Fix For Dense Figures

The row labels in the comparison-dense panels were changed from long one-line names to multi-line parameter blocks.

Example label fields:

```text
S2 generalization
Matern
C=0.08
Nu=2.5
Range=[-0.5,1.5]
Centroid=5.97
TV=0.0165
idx=7
```

Reason:

- Long semantic names were covering the delta / initial-condition / output panels.
- Multi-line labels keep the important kernel/range/spectral metadata while staying inside the left label gutter.

Validation:

| layout | max label right edge | first axes-left | margin |
| --- | ---: | ---: | ---: |
| four-column | 0.0306 | 0.045 | 0.0144 |
| two-column | 0.0522 | 0.082 | 0.0298 |

Record:

```text
docs/burgers_wideparam_loss3targeted_p2q2_diverse_visuals_20260611.md
```

## 13. Full 1024 x 1024 SVD Plus 15-Step Attack Smoke Test

User requirement:

- No block projection.
- No coarse SVD.
- Use full dense `1024 x 1024` Jacobians and full SVD.

Runner:

```text
tools/run_burgers_full1024_svd_attack_correlation_20260611.py
```

Output root:

```text
forensics/burgers_wideparam_loss3targeted_full1024_svd_attack3_20260611
```

Smoke-test sample count:

```text
3 random generalization samples
```

Samples:

| sample | dataset | index | family |
| --- | --- | ---: | --- |
| 0 | `burgers_widevis_l3target_d31` | 10 | powerlaw_fourier |
| 1 | `burgers_widevis_l3target_d17` | 177 | matern |
| 2 | `burgers_widevis_l3target_d05` | 45 | gaussian |

Runtime:

| component | time |
| --- | ---: |
| total wall time | 2160.4 sec = 36.0 min |
| attack time | 185.5 sec |
| Jacobian + SVD time | 1974.6 sec |
| full SVD NPZ files | 27 |

Attack settings:

| setting | value |
| --- | ---: |
| attack steps | 15 |
| RMS epsilon | 0.12 |
| alpha RMS | 0.012 |
| final delta RMS | 0.12 for all 12 model-sample attacks |

Model means over 3 samples:

| model | mean error spectral norm | initial MSE | final attacked MSE | absolute growth |
| --- | ---: | ---: | ---: | ---: |
| baseline | 2.01134 | 7.57474e-04 | 1.62244e-02 | 1.54669e-02 |
| loss1 | 2.08314 | 7.47913e-04 | 7.78605e-03 | 7.03814e-03 |
| loss2 | 2.09002 | 7.64422e-04 | 6.35045e-03 | 5.58603e-03 |
| loss3 | 1.06348 | 1.65457e-04 | 4.08682e-03 | 3.92137e-03 |

Correlation across 12 aligned model-sample pairs:

| x | y | Pearson | Spearman |
| --- | --- | ---: | ---: |
| error spectral norm | attack absolute loss growth | 0.4767 | 0.8462 |
| error spectral norm | attack final MSE | 0.5127 | 0.8811 |
| error spectral norm | attack growth ratio | -0.5515 | -0.3636 |

Interpretation:

- In this small 3-sample smoke test, full error-Jacobian spectral norm tracks absolute/final adversarial damage better than multiplicative growth ratio.
- Growth ratio can be misleading when clean initial loss is tiny, especially for loss3.
- The sample size is too small for a final statistical claim; it is an implementation and sanity smoke test.

Record:

```text
docs/burgers_wideparam_loss3targeted_full1024_svd_attack3_20260611.md
```

## 14. 12.5-Hour Epoch Estimate

Question: if each Burgers adversarial-training objective is allowed about 12.5 wall-clock hours, approximately how many epochs does that correspond to for loss1, loss2, and loss3?

Source record:

```text
docs/burgers_loss3_selective_round03_loss1_8000_plot_report_20260607.md
```

Observed final-extension wall-clock anchors:

| loss | observed final epoch | observed final wall h | avg sec / epoch |
| --- | ---: | ---: | ---: |
| loss1 | 8000 | 11.9919 | 5.396 |
| loss2 | 2000 | 12.4230 | 22.361 |
| loss3 | 1500 | 19.3331 | 46.399 |

Linear wall-clock estimate for 12.5 hours:

| loss | estimated epoch at 12.5 h | interpretation |
| --- | ---: | --- |
| loss1 | about 8339 epochs | slightly beyond the observed 8000 checkpoint; if continued, 12.5 h is roughly 8.3k epochs |
| loss2 | about 2012 epochs | essentially the same as the observed 2000 checkpoint |
| loss3 | about 970 epochs by overall average; about 960 by base+first-continuation segment interpolation | roughly 0.96k to 0.97k epochs |

Practical summary:

```text
12.5 h ~= loss1 8.3k epochs, loss2 2.0k epochs, loss3 0.96-0.97k epochs.
```

Caveat:

- Loss1 is much cheaper per epoch.
- Loss2 is about 4x slower than loss1.
- Loss3 is about 9x slower than loss1 because the attack uses solver forward and backward paths.
- Therefore same-wall-clock and same-epoch comparisons are very different.

## 15. Future Baseline Experiment Note

A future comparison baseline was recorded:

```text
docs/future_label_preserving_input_perturbation_augmentation_baseline_20260611.md
```

Idea:

```text
original:  (x, y)
augmented: (x + delta, y)
```

where `delta` is a small random/spectral/noise perturbation and the label `y` is intentionally kept unchanged.

Important interpretation:

- This is not true adversarial training in the loss1/loss2/loss3 sense.
- It is a flawed but useful control baseline.
- It may improve local train/test robustness.
- It is physically inconsistent because changing `x` should usually change the PDE output `y`.
- It probably cannot explore far OOD `x` distributions and is expected to be weaker than real adversarial training on broad generalization datasets.

## 16. Main Files Created Or Updated For GitHub

Markdown records:

```text
docs/burgers_semantic_loss3fav_round01_generation_audit_20260611.md
docs/burgers_semantic_smooth_loss3_screen_round00_20260611.md
docs/burgers_wideparam_loss3_targeted_replacement_20260611.md
docs/burgers_loss3targeted_52dataset_clean_metrics_20260611.md
docs/burgers_loss3targeted_52dataset_clean_metrics_appendix_20260611.md
docs/burgers_wideparam_loss3targeted_p2q2_diverse_visuals_20260611.md
docs/burgers_wideparam_loss3targeted_full1024_svd_attack3_20260611.md
docs/future_label_preserving_input_perturbation_augmentation_baseline_20260611.md
docs/burgers_20260611_full_archive_summary.md
```

Python and shell tools:

```text
tools/generate_burgers_semantic_loss3_favored_generalization_20260611.py
tools/generate_burgers_semantic_smooth_loss3_screen_20260611.py
tools/generate_burgers_semantic_wideparam_visible_generalization_20260611.py
tools/generate_burgers_wideparam_loss3_targeted_replacement_20260611.py
tools/analyze_burgers_smooth_round00_clean_loss_significance_20260611.py
tools/analyze_burgers_wideparam_visible_clean_loss_final_models_20260611.py
tools/report_burgers_loss3targeted_52_clean_metrics_20260611.py
tools/run_burgers_semantic_loss3fav_round01_20260611.sh
tools/run_burgers_semantic_loss3fav_round01_p2q2_multi_visuals_20260611.py
tools/run_burgers_semantic_smooth_loss3_screen_round00_p2q2_multi_visuals_20260611.py
tools/run_burgers_semantic_smooth_loss3_screen_round00_p2q2_multi_visuals_batched_20260611.py
tools/run_burgers_semantic_smooth_loss3_screen_round00_p2q2_multi_visuals_descriptive_mixed_20260611.py
tools/run_burgers_wideparam_loss3targeted_round00_p2q2_diverse_multi_visuals_batched_20260611.py
tools/run_burgers_full1024_svd_attack_correlation_20260611.py
tools/start_burgers_20260611_selected_r2_sync.sh
tools/plot_burgers_round03_p2q2_combined_attack_panels.py
```

Ledger:

```text
EXPERIMENT_LEDGER.md
```

## 17. Main Artifact Roots For R2

Dataset roots:

```text
generalization_datasets_burgers_semantic_loss3fav_search_20260611
generalization_datasets_burgers_semantic_smooth_loss3_screen_20260611
generalization_datasets_burgers_semantic_wideparam_visible_20260611
generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611
```

Forensics roots:

```text
forensics/burgers_semantic_loss3fav_round01_dataset_range_audit_20260611
forensics/burgers_semantic_loss3fav_round01_p2q2_multi_sample_attack_visuals_20260611
forensics/burgers_semantic_smooth_loss3_screen_round00_20260611
forensics/burgers_semantic_smooth_loss3_screen_round00_clean_loss_significance_20260611
forensics/burgers_semantic_smooth_loss3_screen_round00_p2q2_multi_sample_attack_visuals_20260611
forensics/burgers_semantic_smooth_loss3_screen_round00_p2q2_multi_sample_attack_visuals_batched_20260611
forensics/burgers_semantic_wideparam_visible_round00_20260611
forensics/burgers_semantic_wideparam_visible_round00_clean_loss_final_models_20260611
forensics/burgers_semantic_wideparam_visible_loss3targeted_round00_20260611
forensics/burgers_semantic_wideparam_visible_loss3targeted_round00_clean_loss_final_models_20260611
forensics/burgers_semantic_wideparam_visible_loss3targeted_round00_52dataset_clean_metrics_20260611
forensics/burgers_semantic_wideparam_visible_loss3targeted_round00_p2q2_diverse_multi_sample_attack_visuals_batched_20260611
forensics/burgers_wideparam_loss3targeted_full1024_svd_attack3_20260611
```

Visualization roots:

```text
visualizations/burgers_semantic_loss3fav_round01_p2q2_comparison_dense_multi_sample_bundle_20260611
visualizations/burgers_semantic_smooth_loss3_screen_round00_p2q2_comparison_dense_multi_sample_bundle_20260611
visualizations/burgers_semantic_smooth_loss3_screen_round00_p2q2_comparison_dense_multi_sample_batched_bundle_20260611
visualizations/burgers_semantic_wideparam_visible_loss3targeted_round00_p2q2_comparison_dense_diverse_multi_sample_batched_wrapped_labels_20260611
```

Approximate selected artifact sizes observed locally:

| artifact group | size |
| --- | ---: |
| `generalization_datasets_burgers_semantic_loss3fav_search_20260611` | 1.5G |
| `generalization_datasets_burgers_semantic_smooth_loss3_screen_20260611` | 1.2G |
| `generalization_datasets_burgers_semantic_wideparam_visible_20260611` | 79M |
| `generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611` | 646M |
| final loss3-targeted P2Q2 forensics | 163M |
| full 1024 SVD forensics | 305M |
| final wrapped-label visualizations | 47M |

## 18. Sync Targets

GitHub target:

```text
https://github.com/YifeiSun01/NeuralOperatorRobustness2/tree/vast-ai
```

R2 artifact target prefix:

```text
neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected
```

R2 selected-upload script:

```text
tools/start_burgers_20260611_selected_r2_sync.sh
```

This archive document should be updated after GitHub push and R2 copy are verified.
