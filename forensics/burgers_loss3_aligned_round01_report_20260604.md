# Burgers Loss3-Aligned Aggressive Generalization Probe - 2026-06-04

## Purpose

This run tests the hypothesis that the earlier benchmark favored loss1/loss2 because the generalization set was too close to clean train/test geometry. I generated a more aggressive Burgers generalization set aligned with raw loss3 adversarial directions, then checked whether loss3 adversarial updates align better with the gradient that reduces that new generalization loss.

Important caveat: this is a mechanism probe, not a neutral benchmark. The new generalization set is intentionally loss3-aligned, so it asks whether loss3 becomes useful when the target distribution really points in that direction.

## Generated Generalization Set

Path: `generalization_datasets_burgers_loss3_aligned_search/round_01`

- 50 Burgers datasets.
- 50 samples per dataset.
- Dataset tier: `far_range_pattern`.
- Baseline RMSE on true train/test stays small, but baseline RMSE on this generated set is much larger.

| split | baseline RMSE | baseline relative L2 |
|---|---:|---:|
| train | 0.008984 | 0.016898 |
| test | 0.009544 | 0.017755 |
| generalization | 0.042543 | 0.078831 |
| ALL | 0.041263 | 0.076466 |

Loss3-aligned geometry summary:

| metric | value |
|---|---:|
| delta L2 RMS mean | 0.060071 |
| delta Linf mean | 0.280736 |
| Linf / L2RMS mean | 4.680996 |
| x_adv min | -0.287345 |
| x_adv max | 1.307812 |
| OOB mean | 0.006013 |
| OOB max | 0.313591 |

## Gradient Alignment Validation

Cosine is computed between the parameter gradient from training on an adversarial batch and the parameter gradient that would reduce the evaluation split loss. Positive cosine means the adversarial update is first-order helpful for that split; negative means it pushes against that split.

### 10 Attack-Batch Training Steps

| variant | clean train mean cosine | clean test mean cosine | generalization mean cosine | gen min | gen max | gen negative steps |
|---|---:|---:|---:|---:|---:|---:|
| loss1_raw | 0.937120 | 0.915230 | 0.310066 | -0.733933 | 0.946258 | 4 / 10 |
| loss2_raw | 0.924793 | 0.856638 | 0.442757 | -0.867992 | 0.937828 | 2 / 10 |
| loss3_raw | 0.940144 | 0.826327 | 0.711823 | 0.121155 | 0.997186 | 0 / 10 |

### 50 Attack-Batch Training Steps

| variant | clean train mean cosine | clean test mean cosine | generalization mean cosine | gen min | gen max | gen negative steps |
|---|---:|---:|---:|---:|---:|---:|
| loss1_raw | 0.918757 | 0.922245 | 0.235689 | -0.818357 | 0.955416 | 19 / 50 |
| loss2_raw | 0.853998 | 0.903673 | 0.166701 | -0.870177 | 0.936240 | 22 / 50 |
| loss3_raw | 0.948527 | 0.885902 | 0.735448 | -0.099694 | 0.997186 | 1 / 50 |

This is the key validation: on this constructed aggressive set, loss3 has the largest and most stable generalization-gradient alignment. Loss1/loss2 still align well with clean train/test, but their generalization alignment is weaker and frequently negative.

### 50-Step Generalization Cosine By Step

This is the full per-step generalization-gradient cosine table behind the 50-step mean summary above. The evaluation gradient is computed on `generalization_mixed50`.

| step | loss1 gen cosine | loss2 gen cosine | loss3 gen cosine |
|---:|---:|---:|---:|
| 1 | 0.828362 | 0.527888 | 0.997186 |
| 2 | 0.946911 | 0.936240 | 0.978611 |
| 3 | -0.474875 | 0.355001 | 0.996290 |
| 4 | -0.793538 | -0.651214 | 0.119418 |
| 5 | 0.917841 | 0.824673 | 0.934285 |
| 6 | 0.888584 | 0.687015 | 0.669218 |
| 7 | -0.644915 | -0.868375 | 0.575181 |
| 8 | 0.929395 | 0.895122 | 0.928407 |
| 9 | 0.723557 | 0.760571 | 0.398571 |
| 10 | 0.933273 | 0.933555 | 0.964874 |
| 11 | 0.851440 | 0.881766 | 0.632600 |
| 12 | -0.559805 | 0.852322 | 0.960288 |
| 13 | 0.438901 | 0.446742 | 0.282205 |
| 14 | 0.657179 | -0.049412 | 0.809874 |
| 15 | 0.133888 | -0.659410 | 0.469499 |
| 16 | 0.573804 | -0.416756 | 0.120897 |
| 17 | -0.555446 | -0.347407 | 0.952777 |
| 18 | 0.193462 | -0.722358 | 0.810591 |
| 19 | 0.842880 | 0.235406 | 0.980625 |
| 20 | 0.904207 | 0.844730 | 0.861238 |
| 21 | -0.590579 | -0.544974 | 0.651182 |
| 22 | -0.741461 | -0.668415 | 0.812901 |
| 23 | 0.758434 | 0.820147 | 0.217483 |
| 24 | 0.741881 | -0.768391 | 0.761427 |
| 25 | 0.881940 | 0.838924 | 0.334435 |
| 26 | 0.944449 | 0.856260 | 0.966178 |
| 27 | -0.491694 | 0.736492 | 0.186655 |
| 28 | -0.660163 | -0.598971 | 0.967298 |
| 29 | -0.714476 | -0.556830 | 0.700366 |
| 30 | 0.311289 | 0.880970 | 0.970754 |
| 31 | 0.610701 | -0.227891 | 0.866477 |
| 32 | 0.901149 | 0.017270 | 0.983796 |
| 33 | 0.913673 | -0.517101 | 0.927473 |
| 34 | 0.937948 | 0.931588 | 0.992190 |
| 35 | 0.867538 | 0.881846 | 0.982920 |
| 36 | -0.395231 | -0.437074 | 0.994168 |
| 37 | -0.543874 | 0.781256 | 0.958392 |
| 38 | -0.688924 | 0.087317 | 0.990024 |
| 39 | 0.936898 | 0.770834 | 0.967714 |
| 40 | 0.075347 | 0.902781 | 0.972790 |
| 41 | -0.818357 | 0.850880 | 0.742141 |
| 42 | 0.848087 | 0.825707 | 0.967941 |
| 43 | -0.505443 | -0.424937 | 0.403807 |
| 44 | 0.798542 | -0.512260 | -0.099694 |
| 45 | 0.752361 | -0.496101 | 0.724234 |
| 46 | 0.955416 | -0.371864 | 0.661218 |
| 47 | -0.707605 | -0.870177 | 0.687932 |
| 48 | -0.589702 | 0.690867 | 0.467458 |
| 49 | -0.623749 | -0.671844 | 0.659347 |
| 50 | -0.115070 | -0.337336 | 0.910746 |

Negative generalization-cosine steps:

- loss1: 19 / 50 negative steps: `3, 4, 7, 12, 17, 21, 22, 27, 28, 29, 36, 37, 38, 41, 43, 47, 48, 49, 50`
- loss2: 22 / 50 negative steps: `4, 7, 14, 15, 16, 17, 18, 21, 22, 24, 28, 29, 31, 33, 36, 43, 44, 45, 46, 47, 49, 50`
- loss3: 1 / 50 negative steps: `44`

### Attack Geometry During Probe

| steps | variant | delta L2 RMS | delta Linf | Linf/L2RMS | x min | x max | OOB mean | OOB max |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 10 | loss1_raw | 0.060068 | 0.105691 | 1.758991 | -0.158257 | 1.159651 | 0.002543 | 0.164595 |
| 10 | loss2_raw | 0.060068 | 0.105629 | 1.757682 | -0.152362 | 1.150558 | 0.002614 | 0.161865 |
| 10 | loss3_raw | 0.060068 | 0.250900 | 4.168397 | -0.336902 | 1.333193 | 0.003807 | 0.356450 |
| 50 | loss1_raw | 0.060129 | 0.106460 | 1.769579 | -0.161351 | 1.156321 | 0.002627 | 0.165993 |
| 50 | loss2_raw | 0.060129 | 0.106896 | 1.776463 | -0.162250 | 1.149965 | 0.002625 | 0.166027 |
| 50 | loss3_raw | 0.060129 | 0.257010 | 4.263399 | -0.346803 | 1.344149 | 0.003331 | 0.373151 |

The new generated set deliberately follows this stronger loss3 geometry. That is why loss3 becomes aligned instead of destructive here.

## Long Adversarial Training

Runs:

- loss1: `adversarial_training_runs/burgers_loss3_aligned_round01_loss1_1000ep_20260604`
- loss2: `adversarial_training_runs/burgers_loss3_aligned_round01_loss2_500ep_20260604`
- loss3: `adversarial_training_runs/burgers_loss3_aligned_round01_loss3_500ep_20260604`

Final checkpoints:

| model | final epoch | final checkpoint | wall time hours |
|---|---:|---|---:|
| loss1 | 1000 | `burgers/checkpoints/burgers_epoch1000_step003000.pt` | 1.398 |
| loss2 | 500 | `burgers/checkpoints/burgers_epoch500_step001500.pt` | 3.084 |
| loss3 | 500 | `burgers/checkpoints/burgers_epoch500_step001500.pt` | 6.431 |

### Final Epoch Metrics

| model | epoch | train RMSE | train rel L2 | test RMSE | test rel L2 | gen RMSE | gen rel L2 | ALL RMSE | ALL rel L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| loss1 | 1000 | 0.001969 | 0.003703 | 0.002133 | 0.003968 | 0.009127 | 0.016920 | 0.008854 | 0.016417 |
| loss2 | 500 | 0.002529 | 0.004756 | 0.002668 | 0.004964 | 0.012053 | 0.022336 | 0.011690 | 0.021664 |
| loss3 | 500 | 0.006416 | 0.012067 | 0.006601 | 0.012280 | 0.009939 | 0.018412 | 0.009807 | 0.018172 |

### Best Seen During Training

| model | best train epoch/RMSE | best test epoch/RMSE | best gen epoch/RMSE | best ALL epoch/RMSE |
|---|---:|---:|---:|---:|
| loss1 | 998 / 0.001332 | 998 / 0.001481 | 948 / 0.008785 | 948 / 0.008574 |
| loss2 | 479 / 0.001756 | 479 / 0.001938 | 493 / 0.009930 | 493 / 0.009651 |
| loss3 | 485 / 0.004739 | 485 / 0.005066 | 463 / 0.007302 | 463 / 0.007224 |

## Interpretation

On the original easier/clean-like benchmark, loss1/loss2 looked better because their adversarial samples were closer to the train/test/generalization distribution and their gradients aligned with those evaluation losses.

On this deliberately more aggressive loss3-aligned generalization set, the mechanism flips:

1. Loss3 has much stronger generalization-gradient alignment than loss1/loss2.
2. Loss3 has almost no negative generalization-gradient steps over 50 attack-batch steps.
3. In long training, loss3 reaches the best observed generalization RMSE: 0.007302 at epoch 463.
4. Loss3 final epoch 500 is worse than its best epoch, but still better than loss2 final on the aggressive generalization set and close to loss1 final.
5. Loss1 remains best on clean train/test final accuracy, which makes sense because loss1/loss2 perturbations are less extreme and more clean-like.

So the earlier result does not mean loss3 is intrinsically worse. It means loss3 is only useful when the evaluation distribution actually contains the stronger off-range/peaky geometry that loss3 creates. If the generalization set is too close to clean train/test, loss3 can over-train on a distribution mismatch and look bad.

## Execution Notes

The interactive shell session disconnected near the end, but the original loss3 Python process continued and completed. Final model checkpoint exists at:

`adversarial_training_runs/burgers_loss3_aligned_round01_loss3_500ep_20260604/burgers/checkpoints/burgers_epoch500_step001500.pt`

I also added a small checkpoint-loading/resume option to `tools/adversarial_training.py` as a safety path, but it was not needed for this completed run.

## Relevant Files

- Dataset generator: `tools/generate_burgers_loss3_aligned_generalization.py`
- Gradient probe script: `tools/probe_burgers_p2q2_loss123_50step_gradient_alignment_trajectory.py`
- Long training launcher: `tools/run_burgers_loss3_aligned_round01_long_training_20260604.sh`
- 10-step gradient results: `forensics/burgers_loss3_aligned_round01_loss123_gradient_alignment_10step_20260604`
- 50-step gradient results: `forensics/burgers_loss3_aligned_round01_loss123_gradient_alignment_50step_20260604`
