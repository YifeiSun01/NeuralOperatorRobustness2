# Burgers Round03 Wall-Clock Analysis

Date: 2026-06-05 UTC.

## Scope

This note explains why the round03 long run says two things at once:

- At the loss1-final wall-clock budget, loss3 has not clearly caught loss1 on generated50 generalization.
- By the loss2-final wall-clock budget, loss3 is clearly better on generated50 generalization.

This is based on completed prediction and gradient-alignment outputs. The final Jacobian/SVD posthoc job is still running, so this note does not use spectral/subspace evidence.

## Source Evidence

Observed from:

- `forensics/burgers_loss3_selective_round03_long_training_comparison_20260605/wall_clock_checkpoint_selection.csv`
- `forensics/burgers_loss3_selective_round03_long_training_comparison_20260605/wall_clock_aligned_split_metrics.csv`
- `forensics/burgers_loss3_selective_round03_long_training_comparison_20260605/wall_clock_pairwise_ratios.csv`
- `forensics/burgers_loss3_selective_round03_long_training_comparison_20260605/epoch_aligned_split_metrics.csv`
- `adversarial_training_runs/burgers_loss3_selective_round03_loss*/burgers/checkpoints.csv`
- `adversarial_training_runs/burgers_loss3_selective_round03_loss*/burgers/train_steps.csv`
- `adversarial_training_runs/burgers_loss3_selective_round03_loss3_500ep_long_20260605/burgers/eval_split_summary.csv`
- `forensics/burgers_loss3_selective_round03_loss123_gradient_alignment_50step_long_pipeline_20260605/gradient_alignment_mean_by_variant.csv`

## Runtime Cost

Observed final wall-clock and average epoch cost:

| loss | final epoch | final wall h | avg sec / epoch | relative to loss1 | relative to loss2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| loss1 | 1000 | 1.39523 | 5.02 | 1.00x | 0.23x |
| loss2 | 500 | 3.02754 | 21.80 | 4.34x | 1.00x |
| loss3 | 500 | 6.63540 | 47.77 | 9.51x | 2.19x |

Observed train-step timing means:

| loss | attack objective | solver forward in attack | solver backward in attack | mean attack sec/step | mean step sec |
| --- | --- | ---: | ---: | ---: | ---: |
| loss1 | loss1 | 0 | 0 | 1.1975 | 1.5625 |
| loss2 | loss2 | 1 | 0 | 6.1675 | 7.1541 |
| loss3 | loss3 | 1 | 1 | 14.7681 | 15.8127 |

Observed inference: the wall-clock asymmetry is mostly real algorithmic cost. Loss3 uses the solver forward and backward path in the adversarial attack, so each adversarial step is much more expensive. Full eval is about 0.31 sec and is too small to explain the gap.

## Same-Epoch Versus Same-Time

Observed same-epoch generated50 generalization RMSE:

| epoch | loss1 | loss2 | loss3 |
| ---: | ---: | ---: | ---: |
| 10 | 0.068311 | 0.067392 | 0.056565 |
| 50 | 0.055702 | 0.056329 | 0.049165 |
| 100 | 0.051914 | 0.051430 | 0.045186 |
| 200 | 0.045326 | 0.045918 | 0.038361 |
| 300 | 0.040174 | 0.041472 | 0.028279 |
| 400 | 0.041277 | 0.041015 | 0.025874 |
| 500 | 0.039417 | 0.039666 | 0.023661 |

Observed inference: per epoch, loss3 improves the round03 generated generalization target faster than loss1/loss2. The catch is that one loss3 epoch costs far more wall-clock time.

## Strict Wall-Clock Comparisons

Observed generated50 generalization metrics at key wall-clock selections:

| comparison | loss | selected epoch | wall h | gen RMSE | gen rel L2 |
| --- | --- | ---: | ---: | ---: | ---: |
| loss1-final budget | loss1 | 1000 | 1.3952 | 0.036557 | 0.065558 |
| loss1-final budget | loss2 | 230 | 1.4002 | 0.044122 | 0.079149 |
| loss1-final budget | loss3 | 104 | 1.3933 | 0.046274 | 0.083018 |
| loss2-final budget | loss1 | 1000 | 1.3952 | 0.036557 | 0.065558 |
| loss2-final budget | loss2 | 500 | 3.0275 | 0.039666 | 0.071147 |
| loss2-final budget | loss3 | 229 | 3.0135 | 0.034025 | 0.061043 |
| loss3-final budget | loss1 | 1000 | 1.3952 | 0.036557 | 0.065558 |
| loss3-final budget | loss2 | 500 | 3.0275 | 0.039666 | 0.071147 |
| loss3-final budget | loss3 | 500 | 6.6354 | 0.023661 | 0.042441 |

Observed inference:

- At the loss1-final budget, loss1 has already completed 1000 cheap epochs, while loss3 has only reached epoch 104. Loss3 is better per epoch, but not enough to beat 1000 loss1 epochs by 1.395 h.
- At the loss2-final budget, loss3 has reached epoch 229. That is enough for its round03 generalization advantage to overcome the extra compute cost.
- At loss3 final, the generated50 advantage is large, but it uses a longer observed budget than loss1/loss2 final runs.

## One-Epoch Volatility Near 1.4 Hours

Observed loss3 generated50 RMSE around the loss1-final wall-clock:

| epoch | gen RMSE | gen rel L2 |
| ---: | ---: | ---: |
| 100 | 0.045186 | 0.081072 |
| 101 | 0.037226 | 0.066805 |
| 102 | 0.043919 | 0.078796 |
| 103 | 0.038026 | 0.068241 |
| 104 | 0.046274 | 0.083018 |
| 105 | 0.038256 | 0.068659 |
| 106 | 0.045843 | 0.082239 |
| 107 | 0.034832 | 0.062516 |
| 108 | 0.043626 | 0.078269 |
| 109 | 0.038739 | 0.069538 |
| 110 | 0.043938 | 0.078822 |
| 111 | 0.038057 | 0.068291 |
| 112 | 0.045794 | 0.082156 |
| 113 | 0.037090 | 0.066578 |
| 114 | 0.040953 | 0.073467 |
| 115 | 0.036481 | 0.065455 |

Observed inference: strict same-wall selection chooses epoch 104 for the loss1-final budget because it is the nearest available checkpoint at that target. One epoch later, epoch 105, loss3 is already below loss2 final but still above loss1 final. Two epochs later, epoch 107 is below both. These are real model-state oscillations in the fixed eval output, not a reason to claim stable loss3 dominance at 1.4 h. The safer statement is that around 1.4 h loss3 is near the transition zone but not stably better than loss1 final.

## Stable Crossover Trend

Observed loss3 generated50 RMSE windows:

| epoch range | mean RMSE | median RMSE | min | max | frac below loss1 final | frac below loss2 final |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1-50 | 0.050353 | 0.049355 | 0.040883 | 0.067299 | 0.00 | 0.00 |
| 51-100 | 0.043386 | 0.043147 | 0.037255 | 0.049409 | 0.00 | 0.06 |
| 101-150 | 0.040096 | 0.039829 | 0.033879 | 0.046274 | 0.18 | 0.50 |
| 151-200 | 0.037903 | 0.037583 | 0.033527 | 0.043826 | 0.32 | 0.76 |
| 201-250 | 0.035700 | 0.035622 | 0.030870 | 0.042077 | 0.68 | 0.98 |
| 251-300 | 0.033488 | 0.033280 | 0.028279 | 0.040854 | 0.88 | 0.98 |
| 301-350 | 0.031003 | 0.030319 | 0.027754 | 0.036966 | 0.96 | 1.00 |
| 351-400 | 0.029096 | 0.028746 | 0.025593 | 0.034908 | 1.00 | 1.00 |
| 401-450 | 0.027089 | 0.026418 | 0.023953 | 0.031833 | 1.00 | 1.00 |
| 451-500 | 0.026252 | 0.026470 | 0.023077 | 0.031024 | 1.00 | 1.00 |

Observed rolling 10-epoch RMSE mean first crosses:

| baseline | baseline final gen RMSE | first 10-epoch loss3 window below baseline |
| --- | ---: | --- |
| loss2 final | 0.039666 | epochs 136-145 |
| loss1 final | 0.036557 | epochs 208-217 |

Inference from checkpoint wall-clock interpolation:

- Epochs 136-145 correspond roughly to about 1.8-1.9 h.
- Epochs 208-217 correspond roughly to about 2.7-2.9 h.
- Therefore the loss2-final budget at about 3.03 h is after the stable loss3 crossover against both final baselines.

## Mechanistic Interpretation

Observed gradient-alignment mean on the round03 generalization target:

| variant | generalization mixed4 cosine |
| --- | ---: |
| loss1_raw | 0.033863 |
| loss2_raw | 0.119059 |
| loss3_raw | 0.276633 |

Inference: loss3 pays a large compute tax because its attack objective is more expensive, but the update direction is much more aligned with the round03 generalization loss. Early in wall-clock time, the compute tax dominates. After enough loss3 epochs, the better OOD-aligned update direction dominates. This explains the transition from "not caught at loss1-final time" to "clear generated50 win by loss2-final time".

## Train/Test Caveat

Observed final train/test RMSE:

| loss | train RMSE | test RMSE |
| --- | ---: | ---: |
| loss1 final | 0.002576 | 0.002681 |
| loss2 final | 0.001921 | 0.002109 |
| loss3 final | 0.005881 | 0.006169 |

Inference: this is not a uniform in-distribution win for loss3. It is a targeted round03 generated-generalization win. The wall-clock story should always say "generated50 generalization" rather than implying train/test superiority.

## Recommended Reporting Language

A precise statement is:

> Loss3 is substantially more expensive per epoch because the adversarial attack uses solver forward and backward computations. Under equal epoch counts it improves round03 generated generalization much faster than loss1/loss2. Under strict equal wall-clock at the loss1-final budget, loss3 has only reached about epoch 104 and has not stably surpassed loss1 final. By the loss2-final budget, loss3 reaches about epoch 229 and is already below both final baselines on generated50 generalization. At its final checkpoint, loss3 is the clear generated50 winner, but with a larger wall-clock budget and worse train/test RMSE.
