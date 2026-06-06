# Burgers Round03 Final Jacobian/SVD Interpretation

Date: 2026-06-05T20:30:00Z

## Sources

Observed from:

- `forensics/burgers_loss3_selective_round03_long_final_jacobian_svd_rep20_top100_20260605/round03_long_final_jacobian_svd_summary.md`
- `forensics/burgers_loss3_selective_round03_long_final_jacobian_svd_rep20_top100_20260605/round03_long_final_error_spectral_norm_aggregate.csv`
- `forensics/burgers_loss3_selective_round03_long_final_jacobian_svd_rep20_top100_20260605/round03_long_final_jacobian_spectral_norm_aggregate.csv`
- `forensics/burgers_loss3_selective_round03_long_final_jacobian_svd_rep20_top100_20260605/round03_long_final_solver_similarity_rankwise.csv`
- `forensics/burgers_loss3_selective_round03_long_final_jacobian_svd_rep20_top100_20260605/round03_long_final_solver_similarity_subspaces.csv`

This is the basic/final SVD only: loss1 epoch1000, loss2 epoch500, loss3 epoch500, with train/test/generalization samples. Same-wall/time-matched SVD was cancelled.

## Main Conclusion

Observed evidence supports a targeted round03 generalization advantage for loss3 in local Jacobian geometry, not a uniform train/test advantage.

- On round03 generalization, loss3 has the lowest `J_model - J_solver` spectral norm.
- On train/test, loss1/loss2 still have lower `J_model - J_solver` spectral norm than loss3.
- On generalization singular-vector and subspace alignment, loss3 is strongest, especially for right singular vectors/input perturbation directions and top-10/top-20 subspaces.
- Singular-value magnitudes alone are mixed: loss3 is not simply winning by matching the top singular values better. Its advantage is more about modal direction/subspace geometry plus lower generalization error operator norm.

## Model-Minus-Solver Error Spectral Norm

Observed split means:

| split | loss1 | loss2 | loss3 | interpretation |
| --- | ---: | ---: | ---: | --- |
| generalization | 3.3956 | 3.9365 | 2.6563 | loss3 best |
| test | 0.5002 | 0.3650 | 0.6825 | loss2 best, loss3 worse |
| train | 0.4155 | 0.2885 | 0.8854 | loss2 best, loss3 worse |
| ALL | 1.9267 | 2.1316 | 1.7201 | loss3 best because generalization dominates |

Observed per-sample generalization wins:

- loss3 has lower error spectral norm than loss1 on 8/10 generalization samples.
- loss3 has lower error spectral norm than loss2 on 8/10 generalization samples.
- loss3 is the best of loss1/loss2/loss3 on 7/10 generalization samples.

Inference: the round03 dataset really does expose a local operator-generalization benefit for loss3, but it is not absolute on every sample.

## Solver/Model Spectral Norms

Observed generalization mean model spectral norms:

- solver: 5.4573
- loss1 model: 4.7784
- loss2 model: 4.7913
- loss3 model: 4.7025

Inference: loss3's win is not because its top singular value magnitude is obviously closest to the solver. In fact, loss1/loss2 are slightly closer to the solver's mean top singular value on generalization. The better evidence for loss3 is the full error operator norm and the singular-vector/subspace alignment.

## Singular-Vector/Subspace Similarity To Solver

Observed generalization top-k subspace mean principal cosines:

| top-k | metric | loss1 | loss2 | loss3 | best |
| ---: | --- | ---: | ---: | ---: | --- |
| 5 | right/input subspace | 0.9675 | 0.9491 | 0.9755 | loss3 |
| 10 | right/input subspace | 0.9259 | 0.9298 | 0.9508 | loss3 |
| 20 | right/input subspace | 0.7774 | 0.7747 | 0.8438 | loss3 |
| 5 | left/output subspace | 0.9042 | 0.8696 | 0.9075 | loss3 |
| 10 | left/output subspace | 0.8974 | 0.8898 | 0.9204 | loss3 |
| 20 | left/output subspace | 0.8508 | 0.8492 | 0.8753 | loss3 |

Observed generalization rankwise vector cosines also favor loss3 in the selected ranks. For example:

- Rank-1 right/left: loss3 `0.8108/0.7501`, loss1 `0.8037/0.6625`, loss2 `0.8024/0.5853`.
- Rank-5 right/left: loss3 `0.8102/0.7996`, loss1 `0.7410/0.7382`, loss2 `0.7280/0.7243`.
- Rank-10 right/left: loss3 `0.5392/0.5486`, loss1 `0.4700/0.4843`, loss2 `0.4876/0.5115`.

Inference: loss3 learns local sensitivity directions that are more solver-like on the generated OOD cases. This matches the intended loss3 mechanism better than the prediction-only metrics alone.

## Train/Test Caveat

Observed train/test error spectral norms are worse for loss3:

- train: loss3 0.8854 vs loss1 0.4155 and loss2 0.2885.
- test: loss3 0.6825 vs loss1 0.5002 and loss2 0.3650.

However, loss3 often has stronger top-20 subspace alignment even on train/test. Example top-20 right subspace mean principal cosine:

- train: loss3 0.9501 vs loss1 0.8640 and loss2 0.8630.
- test: loss3 0.9443 vs loss1 0.8482 and loss2 0.8462.

Inference: loss3 is aligning broader modal directions, but it is not preserving in-distribution local error magnitude as well as loss1/loss2. That explains why train/test prediction and Jacobian error can worsen while generated-OOD Jacobian geometry improves.

## Bottom Line

The final SVD supports this statement:

> Loss3 at epoch500 is more solver-like on the round03 generated generalization set in local Jacobian error norm and singular-vector/subspace geometry, especially the right/input singular directions. It does not prove loss3 is uniformly better: loss1/loss2 still win train/test local Jacobian error, and singular-value magnitude matching alone is mixed.

## Baseline-Relative Error Spectral Norm Drop - 2026-06-05T20:34:00Z

Observed from `forensics/burgers_loss3_selective_round03_long_final_jacobian_svd_rep20_top100_20260605/round03_long_final_error_spectral_norm_aggregate.csv`.

Baseline means are the original pre-adversarial-training FNO model compared with the solver, i.e. spectral norm of `J_baseline_model - J_solver`.

| split | baseline error mean | loss1 error mean | loss1 drop | loss2 error mean | loss2 drop | loss3 error mean | loss3 drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 1.08462 | 0.41549 | 0.66913 / 61.69% | 0.28852 | 0.79610 / 73.40% | 0.88538 | 0.19924 / 18.37% |
| test | 1.23788 | 0.50024 | 0.73764 / 59.59% | 0.36496 | 0.87291 / 70.52% | 0.68253 | 0.55534 / 44.86% |
| generalization | 5.77462 | 3.39560 | 2.37902 / 41.20% | 3.93652 | 1.83810 / 31.83% | 2.65625 | 3.11836 / 54.00% |

Inference: loss3 gives the largest baseline-relative reduction on round03 generated generalization, but loss2 gives the largest baseline-relative reduction on train/test.

## Baseline Singular-Vector/Subspace Similarity - 2026-06-05T20:38:00Z

Observed from `round03_long_final_solver_similarity_subspaces.csv`. This compares the pre-adversarial-training baseline model Jacobian `J_baseline_model` against `J_solver`.

| split | top-k | right/input mean cosine | left/output mean cosine | right/input min cosine | left/output min cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 5 | 0.990330 | 0.981085 | 0.964052 | 0.947516 |
| train | 10 | 0.964643 | 0.968771 | 0.731382 | 0.783024 |
| train | 20 | 0.786076 | 0.858951 | 0.064903 | 0.129379 |
| test | 5 | 0.995983 | 0.977436 | 0.989957 | 0.949180 |
| test | 10 | 0.981244 | 0.974644 | 0.876485 | 0.872541 |
| test | 20 | 0.772706 | 0.844117 | 0.001891 | 0.003377 |
| generalization | 5 | 0.914152 | 0.714969 | 0.625861 | 0.175694 |
| generalization | 10 | 0.876361 | 0.770438 | 0.267928 | 0.168905 |
| generalization | 20 | 0.732393 | 0.767185 | 0.008807 | 0.018966 |

Observed from `round03_long_final_solver_similarity_rankwise.csv`, selected baseline rankwise vector cosines on generalization:

| rank | singular value ratio to solver | right/input absdot | left/output absdot |
| ---: | ---: | ---: | ---: |
| 1 | 0.803870 | 0.684916 | 0.244388 |
| 2 | 0.947243 | 0.625427 | 0.575190 |
| 5 | 0.986095 | 0.616740 | 0.605843 |
| 10 | 1.059589 | 0.259214 | 0.286939 |
| 20 | 0.946940 | 0.050181 | 0.071082 |

Inference: baseline is already quite solver-like on train/test top-5/top-10 subspaces, but its generated-generalization singular-vector alignment is much weaker, especially left/output directions and the tail of the top-20 subspace. Loss3's final generalization top-20 right/left mean cosines (`0.843830` / `0.875296`) are therefore not just better than loss1/loss2, but also a large improvement over baseline (`0.732393` / `0.767185`).

## Baseline vs Loss1/Loss2/Loss3 Top-k Subspace Improvement - 2026-06-05T20:42:00Z

Observed from `round03_long_final_solver_similarity_subspaces.csv` and derived CSV `round03_long_final_baseline_vs_loss123_subspace_topk_improvement.csv`.

Definitions:

- `right`: mean principal cosine between model and solver right/input singular subspaces.
- `left`: mean principal cosine between model and solver left/output singular subspaces.
- `delta`: absolute improvement over baseline cosine.
- `gap closed`: `(loss cosine - baseline cosine) / (1 - baseline cosine)`.

### Generalization

| top-k | model | right | right delta | right gap closed | left | left delta | left gap closed |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | baseline | 0.914152 | 0.000000 | 0.00% | 0.714969 | 0.000000 | 0.00% |
| 5 | loss1 | 0.967460 | 0.053308 | 62.10% | 0.904195 | 0.189226 | 66.39% |
| 5 | loss2 | 0.949149 | 0.034997 | 40.77% | 0.869640 | 0.154670 | 54.26% |
| 5 | loss3 | 0.975547 | 0.061395 | 71.52% | 0.907541 | 0.192572 | 67.56% |
| 10 | baseline | 0.876361 | 0.000000 | 0.00% | 0.770438 | 0.000000 | 0.00% |
| 10 | loss1 | 0.925909 | 0.049549 | 40.08% | 0.897435 | 0.126997 | 55.32% |
| 10 | loss2 | 0.929847 | 0.053487 | 43.26% | 0.889833 | 0.119395 | 52.01% |
| 10 | loss3 | 0.950825 | 0.074465 | 60.23% | 0.920445 | 0.150007 | 65.34% |
| 20 | baseline | 0.732393 | 0.000000 | 0.00% | 0.767185 | 0.000000 | 0.00% |
| 20 | loss1 | 0.777392 | 0.044998 | 16.82% | 0.850798 | 0.083613 | 35.91% |
| 20 | loss2 | 0.774694 | 0.042301 | 15.81% | 0.849171 | 0.081986 | 35.22% |
| 20 | loss3 | 0.843830 | 0.111437 | 41.64% | 0.875296 | 0.108110 | 46.44% |

### Train/Test Summary

Observed train top-20 right/left deltas over baseline:

- loss1: `+0.077942` / `+0.101506`.
- loss2: `+0.076909` / `+0.099521`.
- loss3: `+0.164056` / `+0.118186`.

Observed test top-20 right/left deltas over baseline:

- loss1: `+0.075525` / `+0.108290`.
- loss2: `+0.073472` / `+0.105592`.
- loss3: `+0.171584` / `+0.133185`.

Inference: for subspace similarity specifically, loss3 improves over baseline the most at top-20 across train/test/generalization, and also wins generalization top-5/top-10 for both right/input and left/output subspaces. This is a different statement from `J_model - J_solver` error spectral norm, where loss2 remains best on train/test error magnitude.
