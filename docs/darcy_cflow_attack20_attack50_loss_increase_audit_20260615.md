# Darcy CFlow Attack Loss Increase Audit

Date: 2026-06-15.

## What Was Checked

This note separates the real attack-loss-increase artifacts from the earlier
wrong placeholder table.

Definitions:

- `initial loss` = `clean_loss_before_attack`
- `final loss` = `adv_loss_after_attack`
- `attack loss increase` = `attack_loss_gain`
- `relative gain from means` = `mean_attack_loss_gain / mean_clean_loss`

The 20-step table below uses the shared solver-consistent `loss3` attack
objective for every model, so the absolute loss increases are directly
comparable across models.

## Attack20, Full 52 Datasets, 50 Samples Per Dataset, 7 Models

Source:
`analysis_outputs/darcy_attack20_52datasets_50samples_20260613_7models_delta_complete/summary_by_model_split.csv`

Coverage:

- `52` datasets: train, test, and `50` generalization datasets
- `50` samples per dataset
- `2600` samples per model
- `7` models
- `attack_steps = 20`
- `epsilon_fraction = 0.025`

Important caveat: these checkpoints are the `1000/1026/1011/1040/1100` epoch
artifact family, not the later `3000/3500` final time-matched checkpoints.

### All 52 Datasets

| model | trained epochs | samples | initial loss | final loss | attack loss increase | relative gain from means |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 500 | 2600 | 7.69130e-07 | 5.20344e-06 | 4.43431e-06 | 5.76536 |
| loss1 | 1000 | 2600 | 9.38239e-07 | 5.71414e-06 | 4.77590e-06 | 5.09028 |
| loss2 | 1026 | 2600 | 6.89173e-07 | 4.62150e-06 | 3.93232e-06 | 5.70586 |
| loss3 | 1011 | 2600 | 4.61900e-07 | 3.15314e-06 | 2.69124e-06 | 5.82646 |
| physics_loss | 1040 | 2600 | 7.61504e-07 | 4.85337e-06 | 4.09187e-06 | 5.37340 |
| random_clean_y | 1100 | 2600 | 5.07706e-07 | 3.52767e-06 | 3.01997e-06 | 5.94826 |
| random_solver_y | 1100 | 2600 | 8.29141e-07 | 5.30209e-06 | 4.47295e-06 | 5.39468 |

### Generalization Only

| model | trained epochs | samples | initial loss | final loss | attack loss increase | relative gain from means |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 500 | 2500 | 7.99173e-07 | 5.35144e-06 | 4.55227e-06 | 5.69623 |
| loss1 | 1000 | 2500 | 9.74390e-07 | 5.82647e-06 | 4.85208e-06 | 4.97961 |
| loss2 | 1026 | 2500 | 7.15598e-07 | 4.78439e-06 | 4.06879e-06 | 5.68586 |
| loss3 | 1011 | 2500 | 4.76911e-07 | 3.20764e-06 | 2.73073e-06 | 5.72588 |
| physics_loss | 1040 | 2500 | 7.90779e-07 | 4.93586e-06 | 4.14508e-06 | 5.24177 |
| random_clean_y | 1100 | 2500 | 5.26291e-07 | 3.65351e-06 | 3.12722e-06 | 5.94200 |
| random_solver_y | 1100 | 2500 | 8.61425e-07 | 5.45866e-06 | 4.59724e-06 | 5.33678 |

Generalization per-dataset winner counts, lower is better:

- `mean_adv_loss`: loss3 best on `45/50`, random_clean best on `5/50`.
- `mean_attack_loss_gain`: loss3 best on `45/50`, random_clean best on `5/50`.
- `mean_clean_loss`: loss3 best on `28/50`, random_clean best on `20/50`,
  physics best on `2/50`.

## Attack50 Artifacts Found Locally

No complete `52 datasets x 50 samples x 7 models` attack50 table from the later
`3000/3500` final checkpoints was found locally.

Two attack50 subsets are available:

1. Five-model ranked-5 heatmap subset:
   `analysis_outputs/darcy_five_model_batch_ranked_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished/all_50step_batch_attack_results.csv`
2. Seven-model extra-15 curated loss3-advantage subset:
   `analysis_outputs/darcy_seven_model_attack_heatmaps_20260615_loss3_advantage_extra15/summary.csv`

### Attack50, Five-Model Ranked-5 Subset

Coverage: `5` generalization datasets, `50` samples each, `5` models,
`attack_steps = 50`.

| model | datasets | samples | initial loss | final loss | attack loss increase | relative gain from means |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 5 | 250 | 1.52395e-07 | 5.48115e-06 | 5.32875e-06 | 34.9667 |
| loss1 | 5 | 250 | 2.07665e-07 | 5.92051e-06 | 5.71285e-06 | 27.5100 |
| loss2 | 5 | 250 | 1.11825e-07 | 4.79209e-06 | 4.68026e-06 | 41.8536 |
| loss3 | 5 | 250 | 1.03826e-07 | 2.71828e-06 | 2.61446e-06 | 25.1811 |
| physics loss | 5 | 250 | 1.54510e-07 | 5.10175e-06 | 4.94724e-06 | 32.0188 |

### Attack50, Seven-Model Extra-15 Curated Subset

Coverage: `15` selected generalization samples, `7` models,
`attack_steps = 50`. This subset was selected to show loss3-advantage samples,
so it is not an unbiased robustness ranking.

| model | samples | initial loss | final loss | attack loss increase | relative gain from means |
|---|---:|---:|---:|---:|---:|
| baseline | 15 | 1.14672e-06 | 5.62518e-06 | 4.47846e-06 | 3.90546 |
| loss1 | 15 | 1.40913e-06 | 5.94789e-06 | 4.53876e-06 | 3.22096 |
| loss2 | 15 | 1.08275e-06 | 5.08595e-06 | 4.00320e-06 | 3.69726 |
| loss3 | 15 | 6.72584e-07 | 3.51426e-06 | 2.84168e-06 | 4.22501 |
| physics loss | 15 | 1.13127e-06 | 5.12805e-06 | 3.99678e-06 | 3.53299 |
| random clean y | 15 | 7.79376e-07 | 4.24725e-06 | 3.46788e-06 | 4.44956 |
| random solver y | 15 | 1.24972e-06 | 5.59359e-06 | 4.34387e-06 | 3.47587 |

## Correct Conclusion

For the available real 20-step full attack table, `loss3` is best on absolute
attack loss increase and final attacked loss. The earlier "physics is better"
statement came from the wrong placeholder table and should not be used.

For attack50, only selected subsets were found locally. Those subsets also show
lower absolute attack loss increase for `loss3`, but they are not a complete
final robustness sweep.
