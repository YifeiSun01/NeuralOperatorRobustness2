# Burgers Round03 Training, Visualization, And P2Q2 Attack Summary - 2026-06-08

## Source Evidence

Observed source reports:

- `docs/burgers_loss1_5000to8000_launch_20260607.md`
- `docs/burgers_loss3_selective_round03_loss1_8000_plot_report_20260607.md`
- `docs/burgers_loss3_selective_round03_image_only_bundle_20260607.md`
- `docs/burgers_round03_baseline_vs_final_p2q2_attack_verification_20260607.md`
- `docs/burgers_round03_p2q2_metrics_record_20260607.md`
- `docs/burgers_round03_full_p2q2_20step_pilot_launch_20260607.md`

Observed source code/launchers:

- `tools/run_burgers_loss1_continue5000to8000_20260607.sh`
- `tools/plot_burgers_round03_loss1_8000_dense_comparison.py`
- `tools/plot_burgers_round03_baseline_vs_final_attack_panels.py`
- `tools/plot_burgers_round03_p2q2_combined_attack_panels.py`
- `tools/run_burgers_round03_full_p2q2_finalonly_attack.py`
- `tools/run_burgers_round03_full_p2q2_20step_pilot_20260607.sh`
- `tools/run_burgers_round03_full_p2q2_resume_20to40_20260607.sh`
- `tools/run_burgers_round03_full_p2q2_resume_20to60_20260607.sh`

## Long Training / Plot Hygiene

Observed task: continue Burgers round03 loss1 from epoch5000 to epoch8000 and update plots in the existing folder structure rather than a separate one-off folder.

Important naming/plotting constraints recorded during the work:

- Dense comparison plots should show every epoch where every-epoch CSVs exist, not every-5/large checkpoint-only traces.
- Checkpoint-style sparse heatmap/line plots with only every hundreds or thousands of epochs were removed from the polished image-only bundle when the user requested every-epoch plots only.
- Loss1 plots should extend to epoch8000 where that data exists; loss2 remains epoch2000 and loss3 remains epoch1500.
- The image-only bundle should contain PNGs only, split by loss1/loss2/loss3/comparison dense folders.

## Six-Sample P2Q2 Attack Verification

Observed source: `docs/burgers_round03_p2q2_metrics_record_20260607.md`.

Attack settings:

- Device: CUDA on Tesla V100-SXM2-32GB.
- `epsilon_rms=0.12`.
- `alpha_rms=0.012`.
- `attack_steps=100`.
- Models: baseline, loss1 epoch8000, loss2 epoch2000, loss3 epoch1500.
- Six samples: one test sample plus five generalization samples.

Step0 before perturbation:

| Model | Mean MSE | Mean model-solver RMS | Global max abs diff |
|---|---:|---:|---:|
| baseline | `0.0002251479` | `0.0134999231` | `0.3224229813` |
| loss1 | `7.5847369772e-06` | `0.0022291869` | `0.0617683977` |
| loss2 | `8.8161114036e-06` | `0.0026218866` | `0.0741456747` |
| loss3 | `3.5604502045e-05` | `0.0055780425` | `0.1446411014` |

Step100 after perturbation:

| Model | Mean MSE | Mean model-solver RMS | Global max abs diff | Mean delta RMS | Ratio vs baseline MSE |
|---|---:|---:|---:|---:|---:|
| baseline | `0.0265074968` | `0.1535285264` | `1.1547194719` | `0.1199999973` | `1` |
| loss1 | `0.0062939017` | `0.0706373230` | `1.1724832058` | `0.1199999973` | `0.2374385536` |
| loss2 | `0.0092682792` | `0.0884283185` | `1.3292436600` | `0.1200000048` | `0.3496474694` |
| loss3 | `0.0029296223` | `0.0525338911` | `0.8558747768` | `0.1200000048` | `0.1105205190` |

Observed interpretation: after 100 p2q2 attack steps under the same perturbation budget, loss3 epoch1500 has the best mean attacked-output stability; loss1 is second, loss2 third, baseline worst. The visual conclusion should compare all four models, not only loss3 versus baseline.

Observed note: “step100 mean MSE” and “mean model-solver RMS” are different statistics. MSE is `mean(e^2)` and emphasizes spikes; RMS is `sqrt(mean(e^2))` per sample and has output units. `mean_i(sqrt(MSE_i))` is not equal to `sqrt(mean_i(MSE_i))`.

## Before/After Overlay Plots

Observed plot bundle:

- `visualizations/burgers_loss3_selective_round03_longtraining_comparison_dense_image_only_bundle_20260607/comparison_dense/round03_p2q2_baseline_loss1_loss2_loss3_step000_before_perturbation_four_column.png`
- `.../round03_p2q2_baseline_loss1_loss2_loss3_step100_after_perturbation_four_column.png`
- `.../round03_p2q2_baseline_loss1_loss2_loss3_before_after_overlay_four_column.png`

Overlay alpha correction:

- Earlier before-perturbation curves were too faint.
- Corrected before-curve alpha is `0.80`; after curves remain high alpha.

## All-Dataset 20-Step P2Q2 Pilot

Observed source: `docs/burgers_round03_full_p2q2_20step_pilot_launch_20260607.md`.

Scope:

- Models: baseline, loss1 epoch8000, loss2 epoch2000, loss3 epoch1500.
- Data: first 50 train samples, all 150 test samples, and all 50 generalization datasets with 200 samples each.
- Per model: 10200 samples.
- All models: 40800 model-sample attacks.
- Attack settings: `steps=20`, `epsilon_rms=0.12`, `alpha_rms=0.012`, `batch_size=500`.

Observed runtime/memory during monitoring:

- GPU use around `30084 MiB / 32768 MiB`.
- Batch time around `50-59s`.
- Peak allocated about `28.5961 GiB`.
- No OOM/ECC error observed during checks.

Purpose: resolve whether the six-sample visual impression that loss1/loss2/loss3 perturbations look high-frequency is representative across all datasets. Final outputs include per-model `fft_summary_by_dataset.csv` and final deltas.

Resume support:

- `tools/run_burgers_round03_full_p2q2_resume_20to40_20260607.sh`
- `tools/run_burgers_round03_full_p2q2_resume_20to60_20260607.sh`

These launchers continue from saved 20-step deltas rather than restarting from zero.
