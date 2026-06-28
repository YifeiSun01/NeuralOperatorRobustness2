# Burgers p2q2 Loss1/Loss2/Loss3 Solver Alignment Report, 2026-06-03

This report records the complete currently available comparison for the 1D Burgers p2q2 adversarial-training runs discussed on 2026-06-03.

The main conclusion is surprising but clear for the current benchmark: **loss1 at epoch 2000 is the best available model by the combined evidence of prediction loss, model-solver Jacobian error, model Jacobian spectral norm alignment, and top singular-vector alignment with the solver**.

Important caveat: prediction-loss CSVs exist for more epochs than the Jacobian/SVD forensics. In the current local forensics directories, Jacobian/SVD summaries are available for baseline, loss1 epoch 2000, loss2 epoch 900, and loss3 epochs 200/400/600/800/1000. I did not find already-computed Jacobian/SVD CSVs for loss1 epoch 1000 or loss2 epoch 1000.

## Available Model Checkpoints and Metrics

| model | prediction loss available | Jacobian/SVD available |
|---|---|---|
| baseline | epoch 0 | epoch 0 |
| loss1 | epochs 0/1/100/200/400/600/800/900/1000/2000 | epoch 2000 |
| loss2 | epochs 0/1/100/200/400/600/800/900/1000 | epoch 900 |
| loss3 | epochs 0/1/100/200/400/600/800/900/1000 | epochs 200/400/600/800/1000 |

The `score` columns below are taken from `accuracy_score` in `eval_metrics.csv`. These are not classification F1 scores.

## Prediction Loss and Score

| model | epoch | train RMSE | test RMSE | gen RMSE | train relL2 | test relL2 | gen relL2 | train score | test score | gen score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0 | 0.008984 | 0.009544 | 0.017932 | 0.016898 | 0.017755 | 0.031019 | 98.338299 | 98.255480 | 96.992501 |
| loss1 | 1 | 0.006102 | 0.006715 | 0.011179 | 0.011476 | 0.012492 | 0.019371 | 98.865413 | 98.766166 | 98.099995 |
| loss1 | 100 | 0.003381 | 0.003733 | 0.005596 | 0.006358 | 0.006945 | 0.009692 | 99.368189 | 99.310318 | 99.040197 |
| loss1 | 200 | 0.002461 | 0.002597 | 0.006284 | 0.004629 | 0.004832 | 0.010813 | 99.539225 | 99.519129 | 98.930817 |
| loss1 | 400 | 0.002926 | 0.002995 | 0.005492 | 0.005503 | 0.005572 | 0.009474 | 99.452664 | 99.445850 | 99.061811 |
| loss1 | 600 | 0.002248 | 0.002415 | 0.005626 | 0.004228 | 0.004492 | 0.009693 | 99.579002 | 99.552788 | 99.040143 |
| loss1 | 800 | 0.001530 | 0.001686 | 0.004244 | 0.002877 | 0.003137 | 0.007303 | 99.713097 | 99.687255 | 99.275276 |
| loss1 | 900 | 0.002346 | 0.002495 | 0.007126 | 0.004412 | 0.004642 | 0.012233 | 99.560727 | 99.537965 | 98.792228 |
| loss1 | 1000 | 0.001993 | 0.002154 | 0.003951 | 0.003749 | 0.004008 | 0.006823 | 99.626484 | 99.600839 | 99.322380 |
| loss1 | 2000 | 0.001596 | 0.001696 | 0.003788 | 0.003001 | 0.003155 | 0.006521 | 99.700769 | 99.685535 | 99.352370 |
| loss2 | 1 | 0.008220 | 0.008673 | 0.012566 | 0.015461 | 0.016135 | 0.021783 | 98.477444 | 98.412080 | 97.868616 |
| loss2 | 100 | 0.003090 | 0.003383 | 0.007076 | 0.005811 | 0.006293 | 0.012195 | 99.422256 | 99.374621 | 98.795486 |
| loss2 | 200 | 0.005009 | 0.005161 | 0.009585 | 0.009421 | 0.009601 | 0.016510 | 99.066741 | 99.049041 | 98.376191 |
| loss2 | 400 | 0.002361 | 0.002474 | 0.006227 | 0.004440 | 0.004604 | 0.010699 | 99.557934 | 99.541759 | 98.942119 |
| loss2 | 600 | 0.002986 | 0.003018 | 0.007818 | 0.005616 | 0.005615 | 0.013449 | 99.441493 | 99.441608 | 98.673403 |
| loss2 | 800 | 0.001955 | 0.002078 | 0.003066 | 0.003677 | 0.003866 | 0.005310 | 99.633688 | 99.614915 | 99.471938 |
| loss2 | 900 | 0.003272 | 0.003433 | 0.003995 | 0.006155 | 0.006387 | 0.006942 | 99.388300 | 99.365333 | 99.310764 |
| loss2 | 1000 | 0.001575 | 0.001660 | 0.004249 | 0.002963 | 0.003088 | 0.007320 | 99.704605 | 99.692153 | 99.273485 |
| loss3 | 1 | 0.019895 | 0.020725 | 0.039502 | 0.037419 | 0.038557 | 0.068074 | 96.393051 | 96.287431 | 93.629093 |
| loss3 | 100 | 0.013175 | 0.013077 | 0.024255 | 0.024780 | 0.024328 | 0.041742 | 97.581910 | 97.625017 | 95.997370 |
| loss3 | 200 | 0.010953 | 0.011098 | 0.024866 | 0.020601 | 0.020646 | 0.042796 | 97.981478 | 97.977121 | 95.900347 |
| loss3 | 400 | 0.006563 | 0.007013 | 0.018281 | 0.012344 | 0.013047 | 0.031388 | 98.780694 | 98.712141 | 96.962612 |
| loss3 | 600 | 0.004583 | 0.004616 | 0.013558 | 0.008620 | 0.008587 | 0.023252 | 99.145357 | 99.148627 | 97.731683 |
| loss3 | 800 | 0.002922 | 0.003053 | 0.004522 | 0.005496 | 0.005681 | 0.007812 | 99.453358 | 99.435157 | 99.224906 |
| loss3 | 900 | 0.004395 | 0.004488 | 0.010253 | 0.008267 | 0.008350 | 0.017650 | 99.180124 | 99.171915 | 98.266115 |
| loss3 | 1000 | 0.004245 | 0.004391 | 0.005315 | 0.007984 | 0.008169 | 0.009203 | 99.207902 | 99.189755 | 99.088153 |

Prediction-loss reading:

- Among the rows with available prediction metrics, loss1 epoch 2000 has the lowest generalization RMSE among the main comparison checkpoints: 0.003788.
- loss2 epoch 1000 has the lowest train/test RMSE in this table, but its generalization RMSE is 0.004249, above loss1 epoch 2000.
- loss3 improves substantially by epoch 800/1000, but its epoch 1000 generalization RMSE is 0.005315, still worse than loss1 epoch 2000.
- loss3 has a large early degradation at epoch 1: generalization RMSE rises to 0.039502, compared with baseline 0.017932.

## Model Minus Solver Jacobian Error Spectral Norm

This table reports `||J_model - J_solver||_2`. Lower is better.

| model | epoch | all | train | test | generalization |
|---|---:|---:|---:|---:|---:|
| baseline | 0 | 2.377762 | 1.508001 | 0.769029 | 3.543111 |
| loss1 | 2000 | 0.558097 | 0.302427 | 0.290937 | 0.818364 |
| loss2 | 900 | 0.707897 | 0.526293 | 0.446145 | 0.921560 |
| loss3 | 200 | 2.698649 | 1.390560 | 1.514154 | 3.957299 |
| loss3 | 400 | 2.267816 | 0.974671 | 1.117322 | 3.503901 |
| loss3 | 600 | 2.084135 | 1.033413 | 0.879824 | 3.196294 |
| loss3 | 800 | 0.755361 | 0.532506 | 0.457528 | 1.008207 |
| loss3 | 1000 | 0.915005 | 0.783163 | 0.646639 | 1.101456 |

Jacobian-error reading:

- loss1 epoch 2000 is best on all four groups: all, train, test, and generalization.
- loss2 epoch 900 is second best among the available Jacobian/SVD checkpoints.
- loss3 epoch 800 gets close, but remains worse than loss1 epoch 2000 and loss2 epoch 900.
- loss3 epoch 1000 is worse than loss3 epoch 800 on this metric, especially on generalization.

## Model and Solver Jacobian Spectral Norm

This table reports `||J_model||_2` and `||J_solver||_2`. The solver values are the same across rows because the representative samples are the same.

| model | epoch | model all | model train | model test | model gen | solver all | solver train | solver test | solver gen |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0 | 5.288638 | 4.918761 | 4.224217 | 5.936333 | 5.903452 | 5.178373 | 4.383691 | 6.946403 |
| loss1 | 2000 | 5.804817 | 5.171838 | 4.374590 | 6.756696 | 5.903452 | 5.178373 | 4.383691 | 6.946403 |
| loss2 | 900 | 5.796415 | 5.152493 | 4.387838 | 6.746198 | 5.903452 | 5.178373 | 4.383691 | 6.946403 |
| loss3 | 200 | 5.265671 | 4.887456 | 4.131164 | 5.946403 | 5.903452 | 5.178373 | 4.383691 | 6.946403 |
| loss3 | 400 | 5.655792 | 5.149418 | 4.344761 | 6.484029 | 5.903452 | 5.178373 | 4.383691 | 6.946403 |
| loss3 | 600 | 5.697860 | 5.134086 | 4.288071 | 6.600040 | 5.903452 | 5.178373 | 4.383691 | 6.946403 |
| loss3 | 800 | 5.714751 | 5.142720 | 4.268442 | 6.636492 | 5.903452 | 5.178373 | 4.383691 | 6.946403 |
| loss3 | 1000 | 5.743317 | 5.133354 | 4.335563 | 6.672397 | 5.903452 | 5.178373 | 4.383691 | 6.946403 |

Model/solver spectral-norm reading:

- loss1 epoch 2000 and loss2 epoch 900 are closest to the solver spectral norms across the split averages.
- loss3 gradually approaches the solver from epoch 200 to epoch 1000, but the model-minus-solver error table shows that matching the top norm alone is not enough; direction and full error still matter.

## Top-10 Singular Vector Similarity vs Solver

This table averages ranks 1 through 10 for the model Jacobian singular vectors against the solver singular vectors. Higher right/left absolute dot product is better. A singular-value ratio closer to 1 is better.

| model | epoch | right all | left all | sv ratio all | right gen | left gen | sv ratio gen |
|---|---:|---:|---:|---:|---:|---:|---:|
| loss1 | 2000 | 0.952910 | 0.955859 | 0.998993 | 0.923361 | 0.923368 | 1.003778 |
| loss2 | 900 | 0.935283 | 0.937376 | 0.992561 | 0.891817 | 0.891548 | 0.995128 |
| loss3 | 200 | 0.801485 | 0.783421 | 0.966197 | 0.756214 | 0.733094 | 0.957213 |
| loss3 | 400 | 0.810084 | 0.797880 | 0.978727 | 0.706015 | 0.690822 | 0.964209 |
| loss3 | 600 | 0.860858 | 0.855586 | 0.985128 | 0.793307 | 0.787871 | 0.974064 |
| loss3 | 800 | 0.933930 | 0.932947 | 0.992308 | 0.909926 | 0.909287 | 0.986076 |
| loss3 | 1000 | 0.922363 | 0.921171 | 0.988300 | 0.880430 | 0.880654 | 0.977916 |

Singular-vector reading:

- loss1 epoch 2000 has the highest top-10 singular-vector alignment with the solver on both all samples and generalization samples.
- loss2 epoch 900 is second best among available checkpoints.
- loss3 improves by epoch 800, but epoch 1000 is again worse than epoch 800 in this table.

## Overall Conclusion

Across the currently available full comparison:

1. **Best overall checkpoint:** loss1 epoch 2000.
2. **Second strongest available Jacobian/SVD checkpoint:** loss2 epoch 900.
3. **Best loss3 Jacobian/SVD checkpoint among the available series:** loss3 epoch 800, not loss3 epoch 1000.
4. **Most unexpected result:** loss3, which was expected to be more solver/geometric-aligned, is not the best under the current benchmark, perturbation strength, and training distribution.

The evidence for loss1 epoch 2000 being best is not only from prediction loss. It is also supported by:

- Lowest `||J_model - J_solver||_2`.
- Model Jacobian spectral norms closest to solver spectral norms.
- Highest top-10 singular-vector alignment with solver.
- Strong generalization RMSE and relative L2.

This does not prove that loss1 is always better than loss3. It shows that under this exact p2q2 setup, data distribution, perturbation size, and saved checkpoint set, loss1 epoch 2000 is the most solver-aligned model currently measured.

## Interpretation Hypothesis

The most plausible explanation remains that loss3 generates adversarial perturbations that are too strong or too off-manifold for the current train/test/generalization distribution. Earlier audits showed that loss3 has much larger first-epoch perturbations than loss1/loss2:

| model | epoch | delta_linf_mean |
|---|---:|---:|
| loss1 | 1 | about 0.105 |
| loss2 | 1 | about 0.105 |
| loss3 | 1 | about 0.259 |

That explains the large early loss3 jump in prediction loss and the poorer early Jacobian alignment. In this benchmark, loss1/loss2 seem to create milder perturbations that stay closer to the evaluation distribution, while loss3 may push the training samples into regions that are not helpful for this generalization set.

## Missing Follow-up Computations

The following are not present as already-computed local Jacobian/SVD CSVs:

- loss1 epoch 1000 Jacobian/SVD.
- loss2 epoch 1000 Jacobian/SVD.

If these are needed, they should be computed using the same representative-20 sample manifest and the same top-100 SVD pipeline used for:

- `forensics/burgers_p2q2_loss1_epoch2000_jacobian_svd_rep20_top100_20260603`
- `forensics/burgers_p2q2_loss2_epoch900_jacobian_svd_rep20_top100_20260603`
- `forensics/burgers_p2q2_checkpoint_series_jacobian_svd_20260601`
