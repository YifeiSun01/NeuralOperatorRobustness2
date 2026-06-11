# Darcy binary diverse generalization and loss1/loss2/loss3 adversarial probe - 2026-06-11

## Objective

The Darcy coefficient field is required to be hard binary everywhere: each input
`x` must contain only values `3.0` and `12.0`. The generated generalization data
therefore must not store a continuous GRF coefficient. A latent field can be GRF,
wave, block, rectangle, high-pass, band-pass, or transformed/skewed, but the final
coefficient passed to the model and solver is thresholded to the two Darcy values.

## Code changes

- Added `tools/generate_darcy_binary_diverse_generalization_20260611.py`.
  - Generates binary Darcy coefficient fields and solves `-div(A grad U)=1` with
    the existing JAX matrix-free CG workflow.
  - Supports multiple latent families: `matern_smooth`, `matern_fine`,
    `highpass_grf`, `bandpass_grf`, `wave_mix`, `blocky_tiles`, `rectangles`, and
    `cellular_blobs`.
  - Supports latent transforms such as `identity`, `exp_high_bias`, `log_signed`,
    `square_extremes`, `cubic_skew`, and `tanh_plateau` before binary thresholding.
  - Records target and observed high-phase fractions, edge density, solver
    residuals, and variant metadata.
  - Refuses CPU fallback for this run path.
- Updated `tools/evaluate_generalization_models.py` with a Darcy binary guard.
  - Any Darcy dataset evaluated through `tensor_xy` must have shape `(N,H,W)` or
    `(N,H,W,1)`, finite coefficients, and exactly the unique values `[3.0, 12.0]`.
  - `tools/adversarial_training.py` imports `tensor_xy`, so the same guard is also
    active for the adversarial training data path.

## Data generated

Preview batch:

- Root: `generalization_datasets_darcy_binary_diverse_preview_20260611/darcy`
- Files: 8 `.pt` datasets, 4 samples each.
- Verification: every preview `x` contained only `[3.0, 12.0]`.
- Individual previews: `generalization_datasets_darcy_binary_diverse_preview_20260611/darcy/previews/`.

Broad diverse batch:

- Root: `generalization_datasets_darcy_binary_diverse_20260611/darcy`
- Files: 50 `.pt` datasets, 50 samples each.
- Output resolution: 85; solve resolution: 421.
- Verification: `all_binary_verified=true`; no file contained values outside
  `[3.0, 12.0]`.
- Observed high-phase fraction range: `0.1595` to `0.8408`.
- Edge density range: about `0.007` to `0.4558`.
- Family counts: 7 `matern_smooth`, 7 `matern_fine`, and 6 each for
  `highpass_grf`, `bandpass_grf`, `wave_mix`, `blocky_tiles`, `rectangles`, and
  `cellular_blobs`.
- Manifest: `generalization_datasets_darcy_binary_diverse_20260611/darcy/candidate_manifest.csv`.
- Summary: `generalization_datasets_darcy_binary_diverse_20260611/darcy/generation_summary.json`.

Loss3-targeted batch:

- Root: `generalization_datasets_darcy_binary_loss3targeted_20260611/darcy`
- Files: 50 `.pt` datasets, 50 samples each.
- Output resolution: 85; solve resolution: 421.
- High-phase fractions were targeted to `0.12,0.16,0.20,0.24` because the first
  broad probe showed loss3 was strongest on sparse high-phase datasets.
- Verification: 50/50 files contained exactly `[3.0, 12.0]`; `unique_bad=[]`.
- Observed high-phase fraction range: `0.118004` to `0.241722`.
- Edge density range from generator summary: `0.006195` to `0.348902`.
- Manifest: `generalization_datasets_darcy_binary_loss3targeted_20260611/darcy/candidate_manifest.csv`.
- Summary: `generalization_datasets_darcy_binary_loss3targeted_20260611/darcy/generation_summary.json`.
- Individual previews: `generalization_datasets_darcy_binary_loss3targeted_20260611/darcy/previews/`.
- Contact sheet: `generalization_datasets_darcy_binary_loss3targeted_20260611/darcy/previews/contact_sheet_first24_coefficients.png`.

## Training setup

Base model:

- Checkpoint: `2D_Darcy_FNO2d/saved_models/2D/darcy_N1500_nx85_m64_w60_e500_20260528/best.pt`
- Architecture: FNO2d, modes `64 x 64`, width `60`, 4 layers, padding `0`.

Adversarial training probe settings:

- Script: `tools/adversarial_training.py`
- Task: Darcy only.
- Epochs: 50.
- Training mode: `adv-only`.
- Original train subset: 64 samples.
- Darcy batch size: 64.
- Optimizer batch size: 32.
- Attack objectives compared: `loss1`, `loss2`, `loss3`.
- Checkpointing: final checkpoint at epoch 50.
- Training-time generalization eval: first 8 generated datasets only.
- Final full evaluation: all 50 generated datasets.

## Broad batch result

Run directories:

- `adversarial_training_runs/darcy_binary_diverse_loss1_50ep_probe_20260611/`
- `adversarial_training_runs/darcy_binary_diverse_loss2_50ep_probe_20260611/`
- `adversarial_training_runs/darcy_binary_diverse_loss3_50ep_probe_20260611/`

Full-50 generated evaluation artifact:

- CSV: `analysis_outputs/darcy_binary_diverse_loss123_50ep_probe_full50_eval_20260611.csv`
- Summary JSON: `analysis_outputs/darcy_binary_diverse_loss123_50ep_probe_full50_eval_20260611_summary.json`

Observed final relative-L2 deltas versus the baseline model on the broad 50-set
batch:

| objective | improved sets | mean delta vs baseline | best delta | worst delta |
| --- | ---: | ---: | ---: | ---: |
| loss1 | 6/50 | `+0.00355663` | `-0.0036233` | `+0.00921768` |
| loss2 | 20/50 | `-0.00111427` | `-0.0191767` | `+0.0103350` |
| loss3 | 21/50 | `-0.00153609` | `-0.0276510` | `+0.0199218` |

Interpretation: loss3 was best on average and had the strongest best cases, but
on the broad distribution the advantage over loss2 was modest. The strongest
loss3 improvements concentrated on sparse high-phase cases around high fraction
`0.16`, so a second targeted batch was generated.

## Loss3-targeted batch result

Run directories:

- `adversarial_training_runs/darcy_binary_loss3targeted_loss1_50ep_probe_20260611/`
- `adversarial_training_runs/darcy_binary_loss3targeted_loss2_50ep_probe_20260611/`
- `adversarial_training_runs/darcy_binary_loss3targeted_loss3_50ep_probe_20260611/`

Full-50 generated evaluation artifact:

- CSV: `analysis_outputs/darcy_binary_loss3targeted_loss123_50ep_probe_full50_eval_20260611.csv`
- Summary JSON: `analysis_outputs/darcy_binary_loss3targeted_loss123_50ep_probe_full50_eval_20260611_summary.json`

Final full-50 generated relative-L2 deltas versus the baseline model:

| objective | improved sets | mean baseline rel L2 | mean final rel L2 | mean delta | best delta | worst delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| loss1 | 0/50 | `0.0914835` | `0.0993488` | `+0.00786537` | `+0.00651591` | `+0.00952041` |
| loss2 | 0/50 | `0.0914835` | `0.0981791` | `+0.00669565` | `+0.00424793` | `+0.01052998` |
| loss3 | 50/50 | `0.0914835` | `0.0669620` | `-0.02452147` | `-0.03314255` | `-0.00730017` |

The targeted result satisfies the intended diagnostic: loss3 adversarial training
has a clear generalization advantage on this binary Darcy distribution. Loss1 and
loss2 both degrade every generated set under the same 50-epoch, 64-sample probe.

Best loss3 generated improvements:

| dataset | family | high fraction | edge density | baseline rel L2 | loss3 rel L2 | delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `darcy_binary_loss3targeted_20260611_22_rectangles_frac0p12_a1p43234_t10p2242` | rectangles | `0.118004` | `0.054041` | `0.139344` | `0.106202` | `-0.033143` |
| `darcy_binary_loss3targeted_20260611_30_rectangles_frac0p16_a1p24225_t5p53206` | rectangles | `0.158630` | `0.082454` | `0.122097` | `0.090105` | `-0.031992` |
| `darcy_binary_loss3targeted_20260611_38_rectangles_frac0p2_a1p49751_t8p36934` | rectangles | `0.197849` | `0.079616` | `0.115082` | `0.083283` | `-0.031799` |

Weakest loss3 generated improvements, still improved versus baseline:

| dataset | family | high fraction | edge density | baseline rel L2 | loss3 rel L2 | delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `darcy_binary_loss3targeted_20260611_10_highpass_grf_frac0p24_a1p11037_t11p7271` | highpass_grf | `0.239762` | `0.348902` | `0.036661` | `0.029361` | `-0.007300` |
| `darcy_binary_loss3targeted_20260611_42_highpass_grf_frac0p24_a1p18205_t11p6872` | highpass_grf | `0.240025` | `0.340763` | `0.039964` | `0.030438` | `-0.009527` |
| `darcy_binary_loss3targeted_20260611_01_matern_fine_frac0p24_a1p77634_t11p8641` | matern_fine | `0.241279` | `0.077466` | `0.049660` | `0.039995` | `-0.009665` |

## Training-loss check

All three 50-epoch probes reduced `train_loss_on_adv` on the adversarial training
batch:

| objective | epoch 1 train_loss_on_adv | epoch 50 train_loss_on_adv |
| --- | ---: | ---: |
| loss1 | `9.34485e-07` | `2.06951e-08` |
| loss2 | `7.37117e-07` | `1.83066e-08` |
| loss3 | `1.11358e-06` | `4.57787e-08` |

Training-time evaluation on the first 8 targeted generated datasets also showed
loss3 moving in the desired direction by epoch 50:

| objective | baseline gen rel L2 mean | epoch-50 gen rel L2 mean |
| --- | ---: | ---: |
| loss1 | `0.0799387` | `0.0883200` |
| loss2 | `0.0799387` | `0.0877426` |
| loss3 | `0.0799387` | `0.0585123` |

## Tradeoff on original train/test

The loss3-targeted checkpoint improves the targeted generated distribution, but it
also degrades the original train/test distribution in this short probe:

| model | original train rel L2 | original test rel L2 |
| --- | ---: | ---: |
| baseline | `0.0198393` | `0.0227883` |
| loss1 | `0.0232676` | `0.0265920` |
| loss2 | `0.0209703` | `0.0243789` |
| loss3 | `0.0310504` | `0.0312917` |

This is expected for a 64-sample, adv-only, targeted probe. It demonstrates the
loss3 generalization effect on the selected binary distribution, not a final
production replacement model.

## Verification notes

- `python -m py_compile` passed for:
  - `tools/generate_darcy_binary_diverse_generalization_20260611.py`
  - `tools/evaluate_generalization_models.py`
  - `tools/adversarial_training.py`
- Full generated dataset validation loaded every targeted `.pt` file and found
  `unique_bad=[]` for coefficient field `x`.
- CUDA was used for training and evaluation.
- `/workspace` is not persistent on this Vast instance; important outputs should
  be synced to Git/R2 before recycle or destroy.

## 100-epoch follow-up on the loss3-targeted binary dataset

A 100-epoch follow-up was run on the same loss3-targeted 50-set binary Darcy
generalization dataset. The setup was kept matched to the 50-epoch probe:
Darcy only, `adv-only`, 64 original train samples, binary steepest-replace attack,
and separate loss1/loss2/loss3 attack objectives.

Run directories:

- `adversarial_training_runs/darcy_binary_loss3targeted_loss1_100ep_probe_20260611/`
- `adversarial_training_runs/darcy_binary_loss3targeted_loss2_100ep_probe_20260611/`
- `adversarial_training_runs/darcy_binary_loss3targeted_loss3_100ep_probe_20260611/`

Full-50 generated evaluation artifacts:

- CSV: `analysis_outputs/darcy_binary_loss3targeted_loss123_100ep_probe_full50_eval_20260611.csv`
- Summary JSON: `analysis_outputs/darcy_binary_loss3targeted_loss123_100ep_probe_full50_eval_20260611_summary.json`

Final full-50 generated relative-L2 deltas versus the retained baseline model:

| objective | improved sets | mean baseline rel L2 | mean final rel L2 | mean delta | best delta | worst delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| loss1 | 0/50 | `0.0914835` | `0.0949038` | `+0.00342029` | `+0.00233175` | `+0.00511462` |
| loss2 | 8/50 | `0.0914835` | `0.0928236` | `+0.00134011` | `-0.00136263` | `+0.00550652` |
| loss3 | 50/50 | `0.0914835` | `0.0818090` | `-0.00967451` | `-0.01805176` | `-0.00287292` |

The 100-epoch result still shows a clear loss3 advantage: loss3 improves all 50
of 50 generated datasets, while loss1 improves none and loss2 improves only 8/50
with a worse mean relative L2 than baseline. However, the 100-epoch loss3 result
is weaker than the 50-epoch loss3 checkpoint on this targeted generated
distribution.

50-epoch versus 100-epoch full-50 comparison:

| objective | 50ep improved | 50ep mean delta | 50ep mean rel L2 | 100ep improved | 100ep mean delta | 100ep mean rel L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| loss1 | 0/50 | `+0.00786537` | `0.0993488` | 0/50 | `+0.00342029` | `0.0949038` |
| loss2 | 0/50 | `+0.00669565` | `0.0981791` | 8/50 | `+0.00134011` | `0.0928236` |
| loss3 | 50/50 | `-0.02452147` | `0.0669620` | 50/50 | `-0.00967451` | `0.0818090` |

Training-loss check for the 100-epoch probes:

| objective | epoch 1 train_loss_on_adv | epoch 100 train_loss_on_adv | min train_loss_on_adv |
| --- | ---: | ---: | ---: |
| loss1 | `9.34485e-07` | `1.91481e-08` | `1.79545e-08` |
| loss2 | `7.37117e-07` | `1.74630e-08` | `1.63025e-08` |
| loss3 | `1.11358e-06` | `1.15016e-07` | `3.74738e-08` |

The loss3 adversarial training loss reaches its minimum before epoch 100 and is
higher at epoch 100 than its best observed value, matching the full-50 observation
that 100 epochs is not the best checkpoint for targeted generalization here.

Original train/test tradeoff from the 100-epoch full evaluation:

| model | original train rel L2 | original test rel L2 |
| --- | ---: | ---: |
| baseline | `0.0198393` | `0.0227883` |
| loss1-100 | `0.0230732` | `0.0260815` |
| loss2-100 | `0.0214772` | `0.0244608` |
| loss3-100 | `0.0307564` | `0.0332032` |

Conclusion for this follow-up: 100 epochs still makes the loss3 advantage obvious
relative to loss1/loss2 on the targeted binary generalization set, but the 50-epoch
loss3 checkpoint remains the stronger targeted-generalization checkpoint.

## 100-epoch physics/loss4 follow-up

A fourth Darcy adversarial-training objective was added to the targeted binary
comparison: `physics` / `loss4`. This objective uses the Darcy PDE residual plus
boundary penalty as the attack-generation loss; the optimizer training target
remains the solver-generated `solver(a_adv)` label, consistent with the other
Darcy adversarial-training probes.

Run directory:

- `adversarial_training_runs/darcy_binary_loss3targeted_physics_100ep_probe_20260611/`

Full-50 evaluation artifacts:

- CSV: `analysis_outputs/darcy_binary_loss3targeted_loss123physics_100ep_probe_full50_eval_20260611.csv`
- Summary JSON: `analysis_outputs/darcy_binary_loss3targeted_loss123physics_100ep_probe_full50_eval_20260611_summary.json`

Four-objective 100-epoch generated-set comparison:

| objective | improved sets | mean baseline rel L2 | mean final rel L2 | mean delta | best delta | worst delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| loss1 | 0/50 | `0.0914835` | `0.0949038` | `+0.00342029` | `+0.00233175` | `+0.00511462` |
| loss2 | 8/50 | `0.0914835` | `0.0928236` | `+0.00134011` | `-0.00136263` | `+0.00550652` |
| loss3 | 50/50 | `0.0914835` | `0.0818090` | `-0.00967451` | `-0.01805176` | `-0.00287292` |
| physics/loss4 | 9/50 | `0.0914835` | `0.0925279` | `+0.00104446` | `-0.00221515` | `+0.00478751` |

The physics/loss4 training loss on the attacked training batch decreased from
`1.15183e-06` at epoch 1 to `1.41416e-08` at epoch 100. The attack objective
itself also increased clean-to-attacked (`3.08406 -> 4.86512` at epoch 1 and
`3.18645 -> 5.06627` at epoch 100), so the physics attack is active. However,
its final checkpoint does not produce a generated-set generalization advantage:
only 9/50 generated datasets improve, and the mean delta remains positive.

Original train/test relative L2 from the four-objective full evaluation:

| model | original train rel L2 | original test rel L2 |
| --- | ---: | ---: |
| baseline | `0.0198393` | `0.0227883` |
| loss1-100 | `0.0230732` | `0.0260815` |
| loss2-100 | `0.0214772` | `0.0244608` |
| loss3-100 | `0.0307564` | `0.0332032` |
| physics-100 | `0.0216113` | `0.0251844` |

Conclusion: after adding the fourth Darcy objective, loss3 remains the clearly
strongest adversarial-training objective on the loss3-targeted binary generated
set. Physics/loss4 is slightly better than loss2 by mean generated delta, but it
is still not a positive mean improvement over the retained baseline and is far
behind loss3.

## 500-epoch runtime estimate

Runtime estimates were computed from the actual targeted Darcy 100-epoch probes
for loss1/loss2/loss3 and from a 50-epoch standard-clean timing benchmark. The
standard-clean benchmark used the same 64 original training samples, optimizer
microbatch size 32, and the same per-epoch train/test plus first-8-generated eval
load, but no adversarial attack, no solver relabeling, and no checkpoint writes.

Artifacts:

- CSV: `analysis_outputs/darcy_binary_loss3targeted_runtime_estimate_500ep_20260611.csv`
- JSON: `analysis_outputs/darcy_binary_loss3targeted_runtime_estimate_500ep_20260611.json`
- Standard-clean timing root: `analysis_outputs/darcy_binary_loss3targeted_standard_clean_timing_50ep_20260611/`

Estimated wall-clock time:

| method | measured basis | sec/epoch | 500-epoch estimate | epochs in loss3-500 time |
| --- | ---: | ---: | ---: | ---: |
| loss1 attack training | 100 epochs | `1.8050` | `15.04 min` | `502.7` |
| loss2 attack training | 100 epochs | `1.7689` | `14.74 min` | `513.0` |
| loss3 attack training | 100 epochs | `1.8148` | `15.12 min` | `500.0` |
| standard clean, same eval load | 50 epochs | `1.2060` | `10.05 min` | `752.4` |
| standard clean, train-only/no eval | 50 epochs | `0.2118` | `1.77 min` | `4283.9` |

If running loss3 attack training for 500 epochs plus standard clean training for
500 epochs under the same-eval benchmark, the combined estimate is about
`25.17 min`. The same wall time as loss3-500 would allow roughly `503` loss1
epochs or `513` loss2 epochs, because loss1/loss2 are only slightly faster than
loss3 in this binary Darcy setup.

## Generated-vs-training coefficient contact sheets

After the 100-epoch follow-up confirmed that loss3 still has a clear advantage
on the targeted binary Darcy distribution, contact-sheet heat maps were generated
to inspect what the generated datasets look like relative to the original training
set.

Selection:

- Generated: all 50 loss3-targeted generated datasets, sample index `0` from each
  dataset.
- Training comparison: 50 samples selected evenly across the original 1200-sample
  binary Darcy training split.
- Layout: 25 samples per figure, 5 by 5 grid, same coefficient color scale
  `vmin=3.0`, `vmax=12.0` for generated and training sheets.

Artifacts:

- Generated datasets, page 1: `visualizations/darcy_binary_dataset_contact_sheets_20260611/loss3targeted_vs_train_even50/darcy_binary_generated_contact_sheet_page01_00_24.png`
- Generated datasets, page 2: `visualizations/darcy_binary_dataset_contact_sheets_20260611/loss3targeted_vs_train_even50/darcy_binary_generated_contact_sheet_page02_25_49.png`
- Training set, page 1: `visualizations/darcy_binary_dataset_contact_sheets_20260611/loss3targeted_vs_train_even50/darcy_binary_train_contact_sheet_page01_00_24.png`
- Training set, page 2: `visualizations/darcy_binary_dataset_contact_sheets_20260611/loss3targeted_vs_train_even50/darcy_binary_train_contact_sheet_page02_25_49.png`
- Feature comparison figure: `visualizations/darcy_binary_dataset_contact_sheets_20260611/loss3targeted_vs_train_even50/darcy_binary_generated_vs_train_feature_comparison.png`
- Sample metadata CSV: `visualizations/darcy_binary_dataset_contact_sheets_20260611/loss3targeted_vs_train_even50/darcy_binary_generated50_train50_contact_sheet_samples.csv`
- Summary JSON: `visualizations/darcy_binary_dataset_contact_sheets_20260611/loss3targeted_vs_train_even50/darcy_binary_generated50_train50_contact_sheet_summary.json`

Observed coefficient-feature contrast:

| group | high-fraction min | high-fraction max | high-fraction mean | edge-density min | edge-density max | edge-density mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| generated selected 50 | `0.115709` | `0.245121` | `0.180529` | `0.005532` | `0.350980` | `0.099057` |
| training selected 50 | `0.368997` | `0.630311` | `0.510624` | `0.013866` | `0.050280` | `0.026877` |
| training full 1200 | `0.302145` | `0.700208` | `0.501751` | `0.008263` | `0.058613` | `0.025060` |

Interpretation: the loss3-targeted generated data is deliberately much sparser in
the high coefficient phase than the training distribution, and it contains a much
wider edge-density range because it mixes smooth blobs, highpass fields,
rectangles, bandpass fields, wave patterns, and blocky/cellular patterns. The
original training set is binary but is much closer to a balanced GRF thresholded
around 50/50, with lower and narrower edge density.

## Attack heatmap visualization

Added `tools/visualize_darcy_binary_attack_heatmaps_20260611.py` to visualize
selected binary Darcy attack samples. Each selected sample is plotted with six
heat-map columns:

1. clean binary coefficient / initial condition `a`,
2. attack delta `a_adv - a`,
3. attacked coefficient `a_adv`,
4. `model(a_adv)`,
5. `solver(a_adv)`,
6. `model(a_adv) - solver(a_adv)`.

The default visualization uses the same binary steepest-replace Darcy attack as
training: loss3 objective, one attack step, and `epsilon_fraction=0.025`. On the
85 by 85 grid, this flips 181 coefficient pixels. Because the coefficient field
is hard binary, the delta heat map contains only `-9`, `0`, and/or `+9`.

Selected targeted dataset indices: `22`, `30`, `38`, and `10`, all sample index
`0`. Indices `22`, `30`, and `38` are rectangle-pattern cases where full-50
evaluation showed strong loss3 improvements; index `10` is a high-frequency
highpass counterexample/control.

Baseline retained checkpoint visualization:

- Heat map: `visualizations/darcy_binary_attack_heatmaps_20260611/baseline_m64w60/darcy_baseline_m64w60_loss3_indices_22-30-38-10_samples_0-0-0-0_heatmaps.png`
- Summary CSV: `visualizations/darcy_binary_attack_heatmaps_20260611/baseline_m64w60/darcy_baseline_m64w60_loss3_indices_22-30-38-10_samples_0-0-0-0_summary.csv`
- Arrays NPZ: `visualizations/darcy_binary_attack_heatmaps_20260611/baseline_m64w60/darcy_baseline_m64w60_loss3_indices_22-30-38-10_samples_0-0-0-0_arrays.npz`

| dataset index | family | flips | clean objective | attacked objective | attacked rel L2 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 22 | rectangles | 181 | `2.10167e-06` | `3.37166e-06` | `0.147734` |
| 30 | rectangles | 181 | `1.55676e-06` | `3.01129e-06` | `0.149122` |
| 38 | rectangles | 181 | `1.23923e-06` | `2.14363e-06` | `0.134387` |
| 10 | highpass_grf | 181 | `5.00683e-08` | `1.06950e-07` | `0.039511` |

Loss3-targeted epoch-50 checkpoint visualization:

- Heat map: `visualizations/darcy_binary_attack_heatmaps_20260611/loss3targeted_epoch50/darcy_loss3targeted_epoch50_loss3_indices_22-30-38-10_samples_0-0-0-0_heatmaps.png`
- Summary CSV: `visualizations/darcy_binary_attack_heatmaps_20260611/loss3targeted_epoch50/darcy_loss3targeted_epoch50_loss3_indices_22-30-38-10_samples_0-0-0-0_summary.csv`
- Arrays NPZ: `visualizations/darcy_binary_attack_heatmaps_20260611/loss3targeted_epoch50/darcy_loss3targeted_epoch50_loss3_indices_22-30-38-10_samples_0-0-0-0_arrays.npz`

| dataset index | family | flips | clean objective | attacked objective | attacked rel L2 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 22 | rectangles | 181 | `1.12016e-06` | `1.99358e-06` | `0.113630` |
| 30 | rectangles | 181 | `8.18360e-07` | `1.77639e-06` | `0.114749` |
| 38 | rectangles | 181 | `5.42579e-07` | `1.10416e-06` | `0.097067` |
| 10 | highpass_grf | 181 | `1.78040e-07` | `4.27888e-07` | `0.081059` |

Both PNGs were verified as readable `3240 x 1944` RGBA images.
