# Darcy Flow Candidate Generalization Gradient Screen - 2026-06-07

## Status

Observed from local logs and output CSVs: three Darcy Flow candidate-generalization screens completed on GPU on 2026-06-07.

This is a screening study, not an official full Darcy adversarial-training result. The official local Darcy train/test/checkpoint artifacts were missing, so the workflow generated a small screening train/test split and trained a 50-epoch screening baseline before probing candidate datasets.

## Method

The workflow implements the requested dataset-first screening idea:

- Generate candidate Darcy generalization distributions.
- Run short adversarial-training probes for 50 attack-batch steps.
- Before each optimizer update, compute `g_adv`, the model-parameter gradient of the adversarial training batch loss.
- Also compute `g_eval`, the model-parameter gradient that would reduce each clean train/test/candidate-generalization dataset loss.
- Record `cosine(g_adv, g_eval)`, candidate eval loss, and adversarial training loss.

Observed/inference separation:

- Observed values live in the `candidate_screen_summary.csv` and `gradient_alignment_by_step.csv` files listed below.
- Inference from those values: candidates with high positive cosine are first-order compatible with the adversarial training objective, but the user-requested acceptance criterion also requires that candidate eval loss not rise during the 50-step probe.

## Source And Output Paths

Source files:

- Round01 candidate generator: `tools/generate_darcy_generalization_candidates.py`
- Round02 candidate generator: `tools/generate_darcy_generalization_candidates_round02.py`
- Round03 candidate generator: `tools/generate_darcy_generalization_candidates_round03.py`
- Gradient screen: `tools/probe_darcy_gradient_alignment_screen.py`
- Round01 launcher: `tools/run_darcy_candidate_screen_20260607.sh`
- Round02 launcher: `tools/run_darcy_candidate_screen_round02_20260607.sh`
- Round03 launcher: `tools/run_darcy_candidate_screen_round03_20260607.sh`

Screening data/model:

- Screening train/test data: `2D_Darcy_FNO2d/datasets/grf_darcy_screen_20260607`
- Screening baseline checkpoint: `2D_Darcy_FNO2d/saved_models/2D/darcy_screen_baseline_m64_w60_e50_20260607/best.pt`

Candidate roots:

- Round01: `generalization_datasets_darcy_candidate_screen_20260607`
- Round02: `generalization_datasets_darcy_candidate_screen_round02_20260607`
- Round03: `generalization_datasets_darcy_candidate_screen_round03_20260607`

Gradient-screen outputs:

- Round01: `forensics/darcy_candidate_gradient_alignment_screen_20260607`
- Round02: `forensics/darcy_candidate_gradient_alignment_screen_round02_20260607`
- Round03: `forensics/darcy_candidate_gradient_alignment_screen_round03_20260607`

GPU preflight records:

- Round01: `forensics/darcy_candidate_screen_gpu_preflight_20260607/gpu_preflight.json`
- Round02: `forensics/darcy_candidate_screen_round02_gpu_preflight_20260607/gpu_preflight.json`
- Round03: `forensics/darcy_candidate_screen_round03_gpu_preflight_20260607/gpu_preflight.json`

## Completion Evidence

Observed from driver logs:

- Round01 completed at `2026-06-07T14:48:27Z` and wrote `forensics/darcy_candidate_gradient_alignment_screen_20260607`.
- Round02 completed at `2026-06-07T15:33:12Z` and wrote `forensics/darcy_candidate_gradient_alignment_screen_round02_20260607`.
- Round03 completed at `2026-06-07T15:43:37Z` and wrote `forensics/darcy_candidate_gradient_alignment_screen_round03_20260607`.

Observed GPU path:

- GPU preflights required PyTorch CUDA and JAX GPU backend.
- `nvidia-smi` monitoring during concurrent Burgers/Darcy screening showed Tesla V100-SXM2-32GB execution with approximately `16190 MiB / 32768 MiB` used and near-100% utilization during probes.

## Main Metrics

Acceptance criterion used here: candidate eval loss should decrease over the 50-step probe and `cosine(g_adv, g_eval)` should be high with few negative-cosine steps.

Best loss-decreasing candidates observed across all three rounds:

| round | candidate | cosine mean | negative cosine steps | eval loss delta |
|---|---:|---:|---:|---:|
| round01 | `darcy_screen_soft_beta12` | 0.879517 | 3/50 | -2.356520e-08 |
| round03 | `darcy_r03_contrast_low4_high11p25` | 0.874474 | 3/50 | -4.724270e-08 |
| round02 | `darcy_r02_contrast_low4_high11` | 0.873782 | 3/50 | -3.474140e-08 |
| round03 | `darcy_r03_contrast_low3p75_high10p75` | 0.855212 | 4/50 | -7.073240e-08 |
| round03 | `darcy_r03_contrast_low4_high10p75` | 0.849130 | 4/50 | -5.766581e-08 |
| round02 | `darcy_r02_soft_beta10` | 0.846222 | 4/50 | -7.597235e-08 |

High-cosine candidates that failed the loss-decrease criterion:

| round | candidate | cosine mean | negative cosine steps | eval loss delta |
|---|---:|---:|---:|---:|
| round01 | `darcy_screen_mid_tau5` | 0.999412 | 0/50 | +1.875534e-07 |
| round01 | `darcy_screen_mid_alpha2p5_tau3` | 0.988405 | 0/50 | +1.742665e-07 |
| round01 | `darcy_screen_mid_alpha1p5_tau3` | 0.960054 | 1/50 | +2.088597e-07 |
| round01 | `darcy_screen_near_tau3p5` | 0.957481 | 1/50 | +2.165926e-07 |
| round01 | `darcy_screen_near_tau2p5` | 0.932928 | 2/50 | +2.256141e-07 |

A clearly poor candidate:

- `darcy_screen_contrast_low2_high14`: cosine mean `-0.102057`, negative cosine steps `28/50`, eval loss delta `+8.911378e-07`.

## Dataset-Statistic Notes

Observed coefficient statistics for screening train/test and representative candidates:

| dataset | x mean | x std | x min | x max | high frac | low frac | edge density |
|---|---:|---:|---:|---:|---:|---:|---:|
| `screen_train` | 7.531862 | 4.499887 | 3.0 | 12.0 | 0.503540 | 0.496460 | 0.025754 |
| `screen_test` | 7.502699 | 4.499999 | 3.0 | 12.0 | 0.500300 | 0.499700 | 0.024904 |
| `darcy_screen_mid_tau5` | 7.486272 | 4.499979 | 3.0 | 12.0 | 0.498475 | 0.501525 | 0.034045 |
| `darcy_screen_soft_beta12` | 7.586417 | 3.421812 | 3.000002 | 11.999999 | 0.000012 | 0.000006 | 0.999771 |
| `darcy_r02_contrast_low4_high11` | 7.480361 | 3.499945 | 4.0 | 11.0 | 0.497194 | 0.502806 | 0.026329 |
| `darcy_r03_contrast_low4_high11p25` | 7.679124 | 3.624596 | 4.0 | 11.25 | 0.507465 | 0.492535 | 0.024543 |
| `darcy_r03_contrast_low4p5_high10p5` | 7.504948 | 2.999996 | 4.5 | 10.5 | 0.500825 | 0.499175 | 0.022451 |

Observed caveat: `edge_density` is directly meaningful for binary coefficient fields. For soft coefficient fields, adjacent values are almost always unequal, so edge density near `1.0` should not be interpreted as binary-interface density.

## Interpretation

Observed:

- Round01 binary parameter-shift datasets were first-order aligned with adversarial gradients by cosine, but their candidate eval losses increased over the 50-step probe.
- Round02 and round03 soft/mild-contrast datasets made eval loss decrease, but cosine stayed in the moderate range, with the best loss-decreasing candidates around `0.874` to `0.880` and `3/50` negative-cosine steps.
- No screened candidate met a strict combined target of loss decrease plus cosine above `0.9`.

Inference:

- The generated datasets are mechanically valid screening datasets, but the current candidate family is not yet an ideal Darcy Flow generalization suite for adversarial training if the bar is both loss decrease and very high gradient-direction similarity.
- The prior observation that Darcy Flow Loss3 adversarial training can reduce accuracy is consistent with a distribution-mismatch explanation: candidates that look close to the clean binary field can have high gradient cosine while actual eval loss rises under the adversarial-training trajectory; candidates whose eval loss decreases are softer or lower-contrast and only moderately aligned with the adversarial-sample gradient.
- The most defensible promotion candidates for a larger follow-up sweep are `darcy_screen_soft_beta12`, `darcy_r03_contrast_low4_high11p25`, and `darcy_r02_contrast_low4_high11`, but they should be treated as best compromises, not solved selections.

## Remaining Work

- Do not promote the full Darcy Flow 50-dataset generalization suite from round01 alone.
- If continuing this line, run an automated local sweep around soft beta `10-14` and binary contrast ranges near `low=4`, `high=10.75-11.25`, using the same two acceptance metrics.
- For official Darcy Flow adversarial training, restore or regenerate the official Darcy train/test/checkpoint artifacts; the current baseline is explicitly a screening baseline created because local official Darcy `.pt` files/checkpoint were missing.
## Clarification On Loss Direction

Observed from the first50 rows of the three `candidate_screen_summary.csv` files: several candidates did make candidate eval loss decrease during the 50-step probe. The issue is not that every candidate loss rose. The issue is that no screened candidate simultaneously had loss decrease and cosine above `0.9`.

Examples of loss-decreasing candidates:

- `darcy_r02_soft_low4_high10_beta10`: cosine `0.760114`, eval loss delta `-2.029294e-07`.
- `darcy_r02_soft_low4_high10_beta8`: cosine `0.760426`, eval loss delta `-2.011733e-07`.
- `darcy_screen_soft_beta6`: cosine `0.825345`, eval loss delta `-1.665820e-07`.
- `darcy_screen_soft_beta12`: cosine `0.879517`, eval loss delta `-2.356520e-08`.
- `darcy_r03_contrast_low4_high11p25`: cosine `0.874474`, eval loss delta `-4.724270e-08`.

Inference: the candidate family contains loss-decreasing datasets, but the strongest loss decreases came with only moderate gradient-direction similarity. The very high-cosine binary parameter shifts were the ones whose eval loss rose.

## Loss-Drop 50 Follow-Up

Observed follow-up result: `docs/darcy_lossdrop50_selected_generalization_suite_20260607.md` records a selected 50-dataset suite where every selected dataset has `first50 eval_loss_delta < 0`.
