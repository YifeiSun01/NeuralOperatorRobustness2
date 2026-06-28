# Darcy Flow Vast.ai full experiment record - 2026-06-11

This is the consolidated record for the Darcy Flow work performed on the Vast.ai GPU instance. It intentionally records the experiment results, code paths, artifact locations, and conclusions without including any cloud or GitHub credentials.

## Repository and environment

- Repository: `YifeiSun01/NeuralOperatorRobustness2`.
- Working directory on the instance: `/workspace/NeuralOperatorRobustness2`.
- Starting branch: `vast-ai`.
- Python environment: `adv_robust` created from project requirements.
- GPU stack verified: PyTorch `2.8.0+cu126`, CUDA `12.6`, JAX `0.10.0`, JAX backend `gpu`.
- GPU observed for solver benchmark: Tesla V100-SXM2-32GB.

## Loaded Darcy Flow checkpoint

Previously trained Darcy Flow FNO checkpoint restored from object storage:

- Local checkpoint directory: `2D_Darcy_FNO2d/saved_models/2D/darcy_N1500_nx85_m64_w60_e500_20260528/`.
- Checkpoint file: `best.pt`.
- Config file: `config.json`.
- Train log: `train_log.csv`.
- Architecture: `FNO2d(modes1=64, modes2=64, width=60, num_layers=4, in_channels=1, out_channels=1, padding=0)`.
- Resolution: `85 x 85`.
- Parameter count: `235,988,641`.
- Training data size from config: `1200` train, `300` test.
- Darcy coefficient values: binary `3.0` and `12.0`.
- GPU load smoke test: strict load succeeded with `0` missing keys and `0` unexpected keys; random input `(1,85,85,1)` produced finite output `(1,85,85,1)`.

Supporting doc: `docs/vast_ai_darcy_m64_w60_checkpoint_load_20260611.md`.

## Binary-coefficient correction

The important correction was that Darcy input coefficients must be hard binary. The coefficient field can be generated from many kinds of latent random fields, but the final model/solver input `x` must contain exactly the two values `[3.0, 12.0]`.

Code changes:

- Added `tools/generate_darcy_binary_diverse_generalization_20260611.py`.
- Added a Darcy binary guard to `tools/evaluate_generalization_models.py` so Darcy evaluation/training data must be finite and exactly binary `[3.0, 12.0]`.
- Added `tools/visualize_darcy_binary_attack_heatmaps_20260611.py`.
- Added `tools/visualize_darcy_binary_dataset_contact_sheets_20260611.py`.
- Added `tools/benchmark_darcy_vs_burgers_solver_runtime_20260611.py`.

The generator supports diverse latent families and transforms before hard thresholding, including smooth/fine Matern GRF, high-pass and band-pass GRF, wave mixtures, blocky tiles, rectangles, cellular blobs, and skewing transforms such as exponential/log/square/cubic/tanh variants.

## Generated binary Darcy generalization datasets

Three generated roots were produced locally. The large `.pt` files are intentionally not committed to GitHub; their manifests and summaries are small text artifacts.

Preview root:

- `generalization_datasets_darcy_binary_diverse_preview_20260611/darcy`
- 8 datasets, 4 samples each.
- All coefficient fields verified as exactly `[3.0, 12.0]`.

Broad diverse root:

- `generalization_datasets_darcy_binary_diverse_20260611/darcy`
- 50 datasets, 50 samples each.
- Solve resolution: `421`; model/output resolution: `85`.
- High-phase fraction range: `0.1595` to `0.8408`.
- Edge density range: about `0.007` to `0.4558`.
- Manifest: `candidate_manifest.csv`.
- Summary: `generation_summary.json`.

Loss3-targeted root:

- `generalization_datasets_darcy_binary_loss3targeted_20260611/darcy`
- 50 datasets, 50 samples each.
- Target high fractions: `0.12, 0.16, 0.20, 0.24`.
- Observed high-phase fraction range: `0.118004` to `0.241722`.
- Edge density range: `0.006195` to `0.348902`.
- Verification: 50/50 files contained exactly `[3.0, 12.0]`; `unique_bad=[]`.
- Manifest: `candidate_manifest.csv`.
- Summary: `generation_summary.json`.

## 50-epoch broad diverse probe

Setup:

- Base model: retained 500-epoch Darcy FNO checkpoint above.
- Task: Darcy adversarial training.
- Training mode: `adv-only`.
- Original train subset: 64 samples.
- Darcy batch size: 64.
- Optimizer microbatch size: 32.
- Objectives: `loss1`, `loss2`, `loss3`.
- Final evaluation: all 50 broad generated datasets.

Run directories:

- `adversarial_training_runs/darcy_binary_diverse_loss1_50ep_probe_20260611/`
- `adversarial_training_runs/darcy_binary_diverse_loss2_50ep_probe_20260611/`
- `adversarial_training_runs/darcy_binary_diverse_loss3_50ep_probe_20260611/`

Full-50 broad evaluation:

| objective | improved sets | mean delta vs baseline | best delta | worst delta |
| --- | ---: | ---: | ---: | ---: |
| loss1 | 6/50 | `+0.00355663` | `-0.0036233` | `+0.00921768` |
| loss2 | 20/50 | `-0.00111427` | `-0.0191767` | `+0.0103350` |
| loss3 | 21/50 | `-0.00153609` | `-0.0276510` | `+0.0199218` |

Interpretation: `loss3` was best on average but only modestly better than `loss2` on the broad distribution. Its strongest improvements concentrated on sparse high-phase cases, motivating the loss3-targeted dataset.

## 50-epoch loss3-targeted probe

Run directories:

- `adversarial_training_runs/darcy_binary_loss3targeted_loss1_50ep_probe_20260611/`
- `adversarial_training_runs/darcy_binary_loss3targeted_loss2_50ep_probe_20260611/`
- `adversarial_training_runs/darcy_binary_loss3targeted_loss3_50ep_probe_20260611/`

Full-50 targeted evaluation:

| objective | improved sets | mean baseline rel L2 | mean final rel L2 | mean delta | best delta | worst delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| loss1 | 0/50 | `0.0914835` | `0.0993488` | `+0.00786537` | `+0.00651591` | `+0.00952041` |
| loss2 | 0/50 | `0.0914835` | `0.0981791` | `+0.00669565` | `+0.00424793` | `+0.01052998` |
| loss3 | 50/50 | `0.0914835` | `0.0669620` | `-0.02452147` | `-0.03314255` | `-0.00730017` |

Conclusion: the targeted dataset clearly exposes the `loss3` advantage. `loss3` improves all 50 generated datasets, while `loss1` and `loss2` degrade all 50 under the same 50-epoch probe.

Strongest `loss3` improvements:

| dataset index | family | high fraction | edge density | baseline rel L2 | loss3 rel L2 | delta |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 22 | rectangles | `0.118004` | `0.054041` | `0.139344` | `0.106202` | `-0.033143` |
| 30 | rectangles | `0.158630` | `0.082454` | `0.122097` | `0.090105` | `-0.031992` |
| 38 | rectangles | `0.197849` | `0.079616` | `0.115082` | `0.083283` | `-0.031799` |

Original train/test tradeoff at 50 epochs:

| model | original train rel L2 | original test rel L2 |
| --- | ---: | ---: |
| baseline | `0.0198393` | `0.0227883` |
| loss1 | `0.0232676` | `0.0265920` |
| loss2 | `0.0209703` | `0.0243789` |
| loss3 | `0.0310504` | `0.0312917` |

This short targeted `adv-only` probe demonstrates a generated-distribution effect, not a final production replacement model.

## 100-epoch loss1/loss2/loss3 follow-up

Run directories:

- `adversarial_training_runs/darcy_binary_loss3targeted_loss1_100ep_probe_20260611/`
- `adversarial_training_runs/darcy_binary_loss3targeted_loss2_100ep_probe_20260611/`
- `adversarial_training_runs/darcy_binary_loss3targeted_loss3_100ep_probe_20260611/`

Full-50 targeted evaluation at 100 epochs:

| objective | improved sets | mean baseline rel L2 | mean final rel L2 | mean delta | best delta | worst delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| loss1 | 0/50 | `0.0914835` | `0.0949038` | `+0.00342029` | `+0.00233175` | `+0.00511462` |
| loss2 | 8/50 | `0.0914835` | `0.0928236` | `+0.00134011` | `-0.00136263` | `+0.00550652` |
| loss3 | 50/50 | `0.0914835` | `0.0818090` | `-0.00967451` | `-0.01805176` | `-0.00287292` |

Conclusion: after 100 epochs, `loss3` remains clearly best and still improves all 50 generated sets. However, the 50-epoch `loss3` checkpoint is stronger on the targeted generated distribution than the 100-epoch checkpoint.

## 100-epoch physics/loss4 follow-up

A fourth Darcy adversarial objective, `physics` / `loss4`, was also run for 100 epochs. It uses Darcy PDE residual plus boundary penalty for attack generation while still training against the solver-generated `solver(a_adv)` target.

Run directory:

- `adversarial_training_runs/darcy_binary_loss3targeted_physics_100ep_probe_20260611/`

Four-objective 100-epoch comparison:

| objective | improved sets | mean baseline rel L2 | mean final rel L2 | mean delta | best delta | worst delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| loss1 | 0/50 | `0.0914835` | `0.0949038` | `+0.00342029` | `+0.00233175` | `+0.00511462` |
| loss2 | 8/50 | `0.0914835` | `0.0928236` | `+0.00134011` | `-0.00136263` | `+0.00550652` |
| loss3 | 50/50 | `0.0914835` | `0.0818090` | `-0.00967451` | `-0.01805176` | `-0.00287292` |
| physics/loss4 | 9/50 | `0.0914835` | `0.0925279` | `+0.00104446` | `-0.00221515` | `+0.00478751` |

Physics/loss4 training loss decreased from `1.15183e-06` at epoch 1 to `1.41416e-08` at epoch 100, so the objective was active. It still did not produce a positive mean generated-set improvement. `loss3` remains the strongest objective.

## Runtime estimates for 500 epochs

Measured per-epoch wall times from completed Darcy runs:

| method | measured basis | sec/epoch | 500-epoch estimate | epochs in loss3-500 wall time |
| --- | ---: | ---: | ---: | ---: |
| loss1 attack training | 100 epochs | `1.8050` | `15.04 min` | `502.7` |
| loss2 attack training | 100 epochs | `1.7689` | `14.74 min` | `513.0` |
| loss3 attack training | 100 epochs | `1.8148` | `15.12 min` | `500.0` |
| physics/loss4 attack training | 100 epochs | `1.6937` | `14.11 min` | `535.6` |
| standard clean, same eval load | 50 epochs | `1.2060` | `10.05 min` | `752.4` |
| standard clean, train-only/no eval | 50 epochs | `0.2118` | `1.77 min` | `4283.9` |

Important clarification: `loss1`, `loss2`, `loss3`, and `physics/loss4` runs all start from the already trained 500-epoch Darcy checkpoint. A phrase like `loss3 500` means 500 epochs of adversarial fine-tuning from the retained baseline, not training a brand-new baseline model for 500 epochs.

Why the Darcy objective timings are so similar: Darcy is a steady elliptic solve, not a long time rollout. In this setup, eval/IO overhead is a large fraction of epoch time, all objectives still compute solver labels for attacked coefficients, and JAX CG/VJP is fast after compilation.

## Attack heat maps

Attack visualization script:

- `tools/visualize_darcy_binary_attack_heatmaps_20260611.py`.

Selected indices:

- `22`, `30`, `38`, and `10`, sample index `0`.
- Indices `22/30/38` are rectangle-pattern cases with strong `loss3` improvements.
- Index `10` is a high-frequency highpass control/counterexample.

Each heat-map figure has six columns:

1. clean binary coefficient / initial condition `a`,
2. attack delta `a_adv - a`,
3. attacked coefficient `a_adv`,
4. `model(a_adv)`,
5. `solver(a_adv)`,
6. `model(a_adv) - solver(a_adv)`.

Attack setting:

- Binary steepest-replace Darcy attack.
- `loss3` objective.
- One attack step.
- `epsilon_fraction=0.025`.
- On an `85 x 85` grid this flips 181 coefficient pixels.
- Since coefficients are hard binary, `delta` contains only `-9`, `0`, and/or `+9`.

Artifacts:

- Baseline heat map: `visualizations/darcy_binary_attack_heatmaps_20260611/baseline_m64w60/darcy_baseline_m64w60_loss3_indices_22-30-38-10_samples_0-0-0-0_heatmaps.png`.
- Loss3 epoch-50 heat map: `visualizations/darcy_binary_attack_heatmaps_20260611/loss3targeted_epoch50/darcy_loss3targeted_epoch50_loss3_indices_22-30-38-10_samples_0-0-0-0_heatmaps.png`.

Baseline attacked rel L2:

| dataset index | family | flips | clean objective | attacked objective | attacked rel L2 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 22 | rectangles | 181 | `2.10167e-06` | `3.37166e-06` | `0.147734` |
| 30 | rectangles | 181 | `1.55676e-06` | `3.01129e-06` | `0.149122` |
| 38 | rectangles | 181 | `1.23923e-06` | `2.14363e-06` | `0.134387` |
| 10 | highpass_grf | 181 | `5.00683e-08` | `1.06950e-07` | `0.039511` |

Loss3 epoch-50 attacked rel L2:

| dataset index | family | flips | clean objective | attacked objective | attacked rel L2 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 22 | rectangles | 181 | `1.12016e-06` | `1.99358e-06` | `0.113630` |
| 30 | rectangles | 181 | `8.18360e-07` | `1.77639e-06` | `0.114749` |
| 38 | rectangles | 181 | `5.42579e-07` | `1.10416e-06` | `0.097067` |
| 10 | highpass_grf | 181 | `1.78040e-07` | `4.27888e-07` | `0.081059` |

## Generated-vs-training contact sheets

Contact-sheet script:

- `tools/visualize_darcy_binary_dataset_contact_sheets_20260611.py`.

Selection:

- Generated: sample index `0` from all 50 loss3-targeted generated datasets.
- Training comparison: 50 samples selected evenly across the original 1200-sample binary Darcy train split.
- Layout: 25 samples per page, 5 by 5 grid.
- Coefficient scale: `vmin=3.0`, `vmax=12.0`.

Artifacts:

- `visualizations/darcy_binary_dataset_contact_sheets_20260611/loss3targeted_vs_train_even50/darcy_binary_generated_contact_sheet_page01_00_24.png`
- `visualizations/darcy_binary_dataset_contact_sheets_20260611/loss3targeted_vs_train_even50/darcy_binary_generated_contact_sheet_page02_25_49.png`
- `visualizations/darcy_binary_dataset_contact_sheets_20260611/loss3targeted_vs_train_even50/darcy_binary_train_contact_sheet_page01_00_24.png`
- `visualizations/darcy_binary_dataset_contact_sheets_20260611/loss3targeted_vs_train_even50/darcy_binary_train_contact_sheet_page02_25_49.png`
- `visualizations/darcy_binary_dataset_contact_sheets_20260611/loss3targeted_vs_train_even50/darcy_binary_generated_vs_train_feature_comparison.png`

Feature contrast:

| group | high-fraction min | high-fraction max | high-fraction mean | edge-density min | edge-density max | edge-density mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| generated selected 50 | `0.115709` | `0.245121` | `0.180529` | `0.005532` | `0.350980` | `0.099057` |
| training selected 50 | `0.368997` | `0.630311` | `0.510624` | `0.013866` | `0.050280` | `0.026877` |
| training full 1200 | `0.302145` | `0.700208` | `0.501751` | `0.008263` | `0.058613` | `0.025060` |

Interpretation: the generated loss3-targeted data is deliberately much sparser in the high coefficient phase than the original training distribution and has a much wider edge-density range.

## Darcy solver vs Burgers solver benchmark

Benchmark script:

- `tools/benchmark_darcy_vs_burgers_solver_runtime_20260611.py`.

Main output:

- `analysis_outputs/darcy_vs_burgers_solver_forward_backward_benchmark_20260611/solver_runtime_summary.json`.
- Larger-batch check: `analysis_outputs/darcy_vs_burgers_solver_forward_backward_benchmark_20260611_large_burgers_batches/solver_runtime_summary.json`.
- Dedicated report: `docs/darcy_vs_burgers_solver_forward_backward_benchmark_20260611.md`.

Setup:

- Darcy uses the actual binary Darcy train coefficients at `85 x 85`, values `[3.0, 12.0]`, JAX CG solve with VJP/implicit differentiation.
- Burgers uses the project `Burgers1DETDRK4` solver at `N=1024`, `dt=0.001`, `t_final=1.0`, `1000` ETDRK4 steps, with smooth deterministic periodic inputs for timing.
- The benchmark separates `forward_no_grad`, `forward_with_grad`, and `backward_only`.
- JAX first-call compilation is recorded separately; comparisons use warmed-up means.

Same-batch timing:

| batch | Darcy forward_with_grad | Darcy backward_only | Darcy backward/forward | Burgers forward_with_grad | Burgers backward_only | Burgers backward/forward | Burgers/Darcy forward | Burgers/Darcy backward |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `0.00948s` | `0.01978s` | `2.09x` | `0.85522s` | `1.37897s` | `1.61x` | `90.17x` | `69.73x` |
| 4 | `0.01253s` | `0.02399s` | `1.92x` | `0.85916s` | `1.36196s` | `1.59x` | `68.60x` | `56.78x` |
| 8 | `0.01361s` | `0.02587s` | `1.90x` | `0.86495s` | `1.37320s` | `1.59x` | `63.54x` | `53.08x` |

Larger-batch check:

| solver | batch | forward_with_grad | backward_only | backward/forward |
| --- | ---: | ---: | ---: | ---: |
| Darcy | 64 | `0.02317s` | `0.04253s` | `1.84x` |
| Burgers | 16 | `0.84689s` | `1.35751s` | `1.60x` |
| Burgers | 32 | `0.91488s` | `1.43250s` | `1.57x` |

Conclusion: the current Darcy solver is indeed much faster than the Burgers solver in this code path. On equal batches `1/4/8`, Burgers forward-with-grad is about `64x-90x` slower and Burgers backward-only is about `53x-70x` slower. The memory that Darcy backward is roughly twice forward is correct: Darcy `backward_only / forward_with_grad` was `1.84x-2.09x` after warm-up.

The reason is structural: Darcy is a steady elliptic CG solve, while Burgers is a time-dynamic rollout of 1000 ETDRK4/FFT steps.

## Committed small artifacts

The intended Git branch should include:

- This consolidated Markdown record.
- Supporting Darcy docs.
- Darcy generation, visualization, benchmark scripts.
- Small CSV/JSON result summaries under `analysis_outputs/`.
- Small visualization PNG/CSV/JSON artifacts under `visualizations/darcy_binary_*`.
- Dataset manifests/summaries, but not the large `.pt` generated datasets.
- Checkpoint config/train log, but not the large `best.pt` checkpoint.

Large local artifacts intentionally excluded from GitHub:

- `2D_Darcy_FNO2d/saved_models/.../best.pt` (~944 MB).
- Original Darcy train/test `.pt` datasets (~130 MB total).
- Generated Darcy `.pt` datasets, especially `generalization_datasets_darcy_binary_loss3targeted_20260611/` (~211 MB).
- Adversarial training run checkpoint directories.

## Primary conclusion

For the hard-binary Darcy Flow setting, the final coefficient input must always be exactly `[3.0, 12.0]`. Under that corrected data assumption, the loss3-targeted generated distribution shows a clear and repeatable `loss3` adversarial-training advantage over `loss1`, `loss2`, and `physics/loss4`. The effect is strongest at the 50-epoch targeted checkpoint and remains present at 100 epochs. The current Darcy solver is also orders of magnitude faster than the Burgers time-rollout solver, and Darcy solver backward is empirically close to twice its forward time after warm-up.
