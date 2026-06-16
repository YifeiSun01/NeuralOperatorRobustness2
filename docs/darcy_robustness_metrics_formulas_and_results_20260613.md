# Darcy Flow Robustness Metrics, Formulas, and Results

Date: 2026-06-13  
Branch target: `vast-ai-darcy-flow`  
Problem: binary 2D Darcy Flow coefficient field, with coefficient values restricted to `{3, 12}`.

This note records the formulas, metric definitions, numeric tables, and current conclusions from the Darcy Flow binary adversarial/self-training analysis.

## 1. Main Conclusions

### 1.1 Loss3 is best on error-aligned robustness metrics

Across the current Darcy Flow robustness results, `loss3` is strongest on the quantities that directly measure error-aligned adversarial vulnerability:

- 20-step binary adversarial attack loss gain.
- clean generalization loss on most of the 50 binary generalization datasets.
- continuous error-aligned sensitivity, measured by \( \|J^\top e\|_2 \), \( \|J^\top e\|_2^2 \), and \( \|J^\top e\|_\infty \).
- binary-feasible first-order loss gain, which respects the discrete flip constraint \(3 \leftrightarrow 12\).
- top singular output-direction error alignment \( |\langle e, u_1\rangle| \).

### 1.2 Loss3 is not best on pure worst-case Jacobian spectral norm

`loss3` does not make the full Jacobian worst-case sensitivity smallest. In the 25-sample metric run, `loss3` has the largest estimated top singular value:

| model | mean \( \sigma_{\max}(J) \) |
|---|---:|
| loss2 | 0.001723 |
| loss1 | 0.001799 |
| physics loss | 0.001861 |
| loss3 | 0.002073 |

This is the key mechanism point: `loss3` does not uniformly shrink the whole Jacobian operator norm. Instead, it reduces the part of the Jacobian sensitivity aligned with the current prediction error and with binary-feasible attack directions.

### 1.3 Correlation result

Within the same initial condition, attack loss gain correlates better with error-aligned quantities than with \( \sigma_{\max}(J) \). In the 25-sample run:

| metric vs attack gain | within-sample Pearson | within-sample Spearman |
|---|---:|---:|
| \( |\langle e,u_1\rangle| \) | 0.725 | 0.629 |
| \( \|J^\top e\|_2 \) | 0.639 | 0.638 |
| binary first-order gain | 0.564 | 0.522 |
| \( \sigma_{\max}(J)\|e\|_2 \) | 0.516 | 0.544 |
| \( \sigma_{\max}(J) \) | -0.689 | -0.599 |

So the better explanatory variable is not the pure largest singular value, but the error-aligned sensitivity.

## 2. Notation

Let the binary Darcy coefficient field be

\[
a \in \{3,12\}^{H \times W}.
\]

Let the neural operator prediction be

\[
F(a),
\]

and the numerical Darcy solver output be

\[
u(a).
\]

The prediction error is

\[
e(a) = F(a) - u(a).
\]

The model Jacobian with respect to the input coefficient is

\[
J(a) = \frac{\partial F(a)}{\partial a}.
\]

The clean squared-error loss is

\[
\mathcal{L}(a) = \frac{1}{2}\|F(a)-u(a)\|_2^2
              = \frac{1}{2}\|e(a)\|_2^2.
\]

Its gradient with respect to the input coefficient field is

\[
\nabla_a \mathcal{L}(a) = J(a)^\top e(a).
\]

This vector \(J^\top e\) is the first-order direction in input space that increases the current model error loss.

## 3. Metric Definitions

### 3.1 Clean loss

The clean loss measures the model-solver mismatch before attack:

\[
\mathcal{L}_{\mathrm{clean}}(a)
= \frac{1}{2}\|F(a)-u(a)\|_2^2.
\]

The clean relative \(L^2\) error is

\[
\mathrm{RelL2}(a)
= \frac{\|F(a)-u(a)\|_2}{\|u(a)\|_2}.
\]

### 3.2 Attack gain

For an adversarial binary perturbation \(\delta\), the attack loss gain is

\[
\Delta \mathcal{L}_{\mathrm{attack}}
= \mathcal{L}(a+\delta)-\mathcal{L}(a).
\]

In the 52-dataset and 25-sample summaries, this is the measured loss increase after a finite-step binary attack, usually 20 attack steps unless stated otherwise.

### 3.3 \(J^\top e\)

The vector

\[
J^\top e
\]

is the gradient of squared error with respect to the coefficient field \(a\). It measures the input direction that locally increases the current prediction error. The reported scalar metrics are:

\[
\|J^\top e\|_2,
\]

\[
\|J^\top e\|_2^2,
\]

and

\[
\|J^\top e\|_\infty.
\]

Smaller values mean the current model error is less sensitive to local input changes.

### 3.4 Binary first-order gain

Because Darcy input is binary, the feasible one-pixel flip is not an arbitrary continuous perturbation. For each pixel \(i\),

\[
a_i \in \{3,12\}.
\]

The only allowed flip is

\[
\delta_i =
\begin{cases}
9, & a_i = 3,\\
-9, & a_i = 12.
\end{cases}
\]

The first-order gain score of flipping pixel \(i\) is

\[
s_i = (J^\top e)_i \delta_i.
\]

For a top-\(k\) binary attack budget, the binary first-order gain is

\[
G_{\mathrm{binary}}
= \sum_{i \in \mathrm{TopK}(s_i)} s_i.
\]

This is a more faithful first-order proxy for the real Darcy binary attack than the unconstrained continuous gradient norm, because it only allows \(3 \leftrightarrow 12\) flips.

### 3.5 Top singular value

The Jacobian spectral norm is

\[
\sigma_{\max}(J)
= \max_{\|v\|_2=1}\|Jv\|_2.
\]

It measures worst-case input-output sensitivity, independent of the current prediction error direction.

### 3.6 Top singular vectors and error alignment

The top right and left singular vectors \(v_1,u_1\) satisfy

\[
Jv_1 = \sigma_1 u_1,
\]

\[
J^\top u_1 = \sigma_1 v_1,
\]

where

\[
\sigma_1 = \sigma_{\max}(J).
\]

The error alignment with the most sensitive output direction is

\[
|\langle e,u_1\rangle|.
\]

If this is small, then even a large \(\sigma_{\max}(J)\) may not lead to large adversarial loss growth, because the worst-case output direction is not aligned with the current model error.

### 3.7 \( \sigma_{\max}(J)\|e\|_2 \)

The product

\[
\sigma_{\max}(J)\|e\|_2
\]

is an upper-bound style quantity because

\[
\|J^\top e\|_2
\le \|J^\top\|_2\|e\|_2
= \sigma_{\max}(J)\|e\|_2.
\]

It combines worst-case Jacobian size with error magnitude, but it still does not know whether the error is aligned with the top singular direction.

### 3.8 Top-1 weighted error energy

The top-1 singular-direction contribution to input-gradient energy is approximately

\[
\sigma_1^2 |\langle e,u_1\rangle|^2.
\]

This is the portion of \( \|J^\top e\|_2^2 \) explained by the top singular output direction.

## 4. Full 3000-Epoch Clean Evaluation

Source:

- `analysis_outputs/darcy_full3000_burgers_image_only_bundle_20260612_work/work_visualizations/comparison_dense/darcy_full50_eval_merged.csv`

Mean clean relative \(L^2\) by split:

| split | baseline | loss1 | loss2 | loss3 | physics loss |
|---|---:|---:|---:|---:|---:|
| generalization | 0.091337 | 0.087929 | 0.072153 | 0.061406 | 0.087820 |
| test | 0.022536 | 0.026950 | 0.029057 | 0.048288 | 0.028503 |
| train | 0.019132 | 0.019201 | 0.027017 | 0.035806 | 0.016197 |

Winner counts among trained models:

| split group | loss1 | loss2 | loss3 | physics loss |
|---|---:|---:|---:|---:|
| all 52 datasets | 1 | 12 | 38 | 1 |
| 50 generalization datasets | 0 | 12 | 38 | 0 |
| train | 0 | 0 | 0 | 1 |
| test | 1 | 0 | 0 | 0 |

Winner counts including baseline:

| split group | baseline | loss1 | loss2 | loss3 | physics loss |
|---|---:|---:|---:|---:|---:|
| all 52 datasets | 1 | 0 | 12 | 38 | 1 |
| 50 generalization datasets | 0 | 0 | 12 | 38 | 0 |
| train | 0 | 0 | 0 | 0 | 1 |
| test | 1 | 0 | 0 | 0 | 0 |

Interpretation: on the 50 binary generalization datasets, `loss3` has the best clean relative \(L^2\) on 38/50 datasets. On the original train/test splits, `loss3` is not best; physics loss wins train and baseline/loss1 are better on test depending on whether baseline is included.

## 5. 52-Dataset 20-Step Attack Results

Sources:

- `analysis_outputs/darcy_attack20_52datasets_50samples_20260612_attack20_1000c_52datasets_50samples/summary_by_model_split.csv`
- `analysis_outputs/darcy_attack20_52datasets_50samples_20260612_attack20_1000c_52datasets_50samples/summary_by_dataset_model.csv`

Mean values by model and split:

| model | split | clean loss | attack gain | attacked loss | relative gain |
|---|---|---:|---:|---:|---:|
| loss1 | all | 9.382394e-7 | 4.775900e-6 | 5.714139e-6 | 18.919861 |
| loss1 | train | 3.162582e-8 | 2.742341e-6 | 2.773967e-6 | 104.367007 |
| loss1 | test | 3.731524e-8 | 3.000252e-6 | 3.037567e-6 | 100.796431 |
| loss1 | generalization | 9.743902e-7 | 4.852084e-6 | 5.826474e-6 | 15.573386 |
| loss2 | all | 6.891727e-7 | 3.932322e-6 | 4.621495e-6 | 15.801060 |
| loss2 | train | 2.777102e-8 | 3.703943e-7 | 3.981653e-7 | 18.891575 |
| loss2 | test | 2.933194e-8 | 6.708336e-7 | 7.001655e-7 | 24.590001 |
| loss2 | generalization | 7.155975e-7 | 4.068791e-6 | 4.784388e-6 | 15.563471 |
| loss3 | all | 4.619002e-7 | 2.691245e-6 | 3.153145e-6 | 12.206663 |
| loss3 | train | 8.286518e-8 | 1.677470e-6 | 1.760335e-6 | 27.561185 |
| loss3 | test | 9.039436e-8 | 1.730591e-6 | 1.820986e-6 | 27.797491 |
| loss3 | generalization | 4.769110e-7 | 2.730733e-6 | 3.207644e-6 | 11.587756 |
| physics loss | all | 7.615042e-7 | 4.091870e-6 | 4.853374e-6 | 20.113254 |
| physics loss | train | 2.576308e-8 | 2.813138e-6 | 2.838901e-6 | 152.539095 |
| physics loss | test | 3.351022e-8 | 2.709854e-6 | 2.743365e-6 | 106.295764 |
| physics loss | generalization | 7.907789e-7 | 4.145085e-6 | 4.935864e-6 | 15.741087 |

Winner counts:

| metric | all 52 winner counts |
|---|---|
| mean clean loss | loss3: 44, loss2: 5, physics loss: 3 |
| mean attack loss gain | loss3: 50, loss2: 2 |
| mean attacked loss | loss3: 50, loss2: 2 |

Split-specific winner counts:

| metric | split | winner counts |
|---|---|---|
| mean clean loss | generalization | loss3: 44, loss2: 4, physics loss: 2 |
| mean clean loss | test | loss2: 1 |
| mean clean loss | train | physics loss: 1 |
| mean attack loss gain | generalization | loss3: 50 |
| mean attack loss gain | test | loss2: 1 |
| mean attack loss gain | train | loss2: 1 |
| mean attacked loss | generalization | loss3: 50 |
| mean attacked loss | test | loss2: 1 |
| mean attacked loss | train | loss2: 1 |

Interpretation: on the 50 binary generalization datasets, `loss3` is best for attack loss gain on 50/50 datasets.

## 6. Early 5-Sample Jacobian/SVD Probe

This was the first small probe before the larger 25-sample run.

Mean 20-step attack gain over 52 datasets:

| model | mean attack gain |
|---|---:|
| loss1 | 4.78e-6 |
| loss2 | 3.93e-6 |
| loss3 | 2.69e-6 |
| physics loss | 4.09e-6 |

Mean \( \|J^\top e\|_2 \) over 5 generalization samples:

| model | mean \( \|J^\top e\|_2 \) |
|---|---:|
| loss1 | 1.608e-4 |
| loss2 | 1.346e-4 |
| loss3 | 1.265e-4 |
| physics loss | 1.533e-4 |

Mean top singular value over 5 samples:

| model | mean \( \sigma_{\max}(J) \) |
|---|---:|
| loss2 | 0.001759 |
| loss1 | 0.001828 |
| physics loss | 0.001890 |
| loss3 | 0.002108 |

Interpretation: even in the early small probe, `loss3` had the smallest attack gain and smallest \( \|J^\top e\|_2 \), while having the largest \( \sigma_{\max}(J) \).

## 7. 10-Sample Correlation Probe

Source:

- `analysis_outputs/darcy_metric_correlation_10samples_20260612_10samples_loss3attack20_corr_chunk64/`

Model means:

| model | mean attack gain | mean \( \|J^\top e\|_2 \) | mean \( \sigma_{\max}(J) \) |
|---|---:|---:|---:|
| loss1 | 4.87872e-06 | 1.25913e-04 | 0.00178995 |
| loss2 | 3.84134e-06 | 1.02028e-04 | 0.00172110 |
| loss3 | 2.64413e-06 | 9.28814e-05 | 0.00206670 |
| physics loss | 4.20670e-06 | 1.17896e-04 | 0.00185268 |

Within-sample Spearman correlation with attack gain:

| metric | within-sample Spearman |
|---|---:|
| \( |\langle e,u_1\rangle| \) | 0.680 |
| \( \|J^\top e\|_2 \) | 0.620 |
| \( \|J^\top e\|_2^2 \) | 0.529 |
| binary first-order gain | 0.523 |
| \( \sigma_{\max}(J)\|e\|_2 \) | 0.514 |
| clean loss | 0.477 |
| \( \sigma_{\max}(J) \) | -0.530 |

Interpretation: the 10-sample run already showed the key structure: error-aligned metrics correlate positively with attack gain, while pure \( \sigma_{\max}(J) \) correlates negatively within sample.

## 8. 25-Sample Jacobian/SVD/Attack Metric Means

Sources:

- `analysis_outputs/darcy_metric_correlation_25samples_20260612_25samples_loss3attack20_corr_chunk64/metrics_by_model_sample.csv`
- `analysis_outputs/darcy_metric_correlation_25samples_20260612_25samples_loss3attack20_corr_chunk64/model_metric_means.csv`

Mean metrics over 25 generalization samples:

| metric | loss1 | loss2 | loss3 | physics loss | best model |
|---|---:|---:|---:|---:|---|
| attack loss gain | 4.788547e-6 | 4.030971e-6 | 2.756509e-6 | 4.139830e-6 | loss3 |
| clean loss before attack | 9.425244e-7 | 7.374211e-7 | 4.733863e-7 | 7.665237e-7 | loss3 |
| relative \(L^2\) | 0.097770 | 0.085630 | 0.071582 | 0.089155 | loss3 |
| \( \|J^\top e\|_2 \) | 1.391878e-4 | 1.166779e-4 | 1.057695e-4 | 1.321585e-4 | loss3 |
| \( \|J^\top e\|_2^2 \) | 2.239106e-8 | 1.632852e-8 | 1.336571e-8 | 2.025079e-8 | loss3 |
| \( \|J^\top e\|_\infty \) | 4.224892e-6 | 3.532118e-6 | 3.316905e-6 | 4.174356e-6 | loss3 |
| binary first-order gain | 1.704681e-6 | 1.423322e-6 | 1.321766e-6 | 1.683077e-6 | loss3 |
| \( \sigma_{\max}(J) \) | 0.001799 | 0.001723 | 0.002073 | 0.001861 | loss2 |
| \( \sigma_{\max}(J)\|e\|_2 \) | 1.612624e-4 | 1.359559e-4 | 1.355216e-4 | 1.523018e-4 | loss3 |
| \( \sigma_{\max}(J)^2\|e\|_2^2 \) | 2.951896e-8 | 2.156929e-8 | 2.078279e-8 | 2.643218e-8 | loss3 |
| \( |\langle e,u_1\rangle| \) | 0.805529 | 0.775132 | 0.674547 | 0.812583 | loss3 |
| \( \sigma_1^2|\langle e,u_1\rangle|^2 \) | 2.031642e-8 | 1.456447e-8 | 1.151381e-8 | 1.845196e-8 | loss3 |

Interpretation:

- `loss3` is best on actual attack gain, clean loss, \(J^\top e\), binary first-order gain, and top-direction error alignment.
- `loss3` is worst on pure \( \sigma_{\max}(J) \).
- \( \sigma_{\max}(J)\|e\|_2 \) makes `loss3` slightly better than `loss2`, but the difference is small.

## 9. Correlations With Attack Gain

Sources:

- `analysis_outputs/darcy_metric_correlation_25samples_20260612_25samples_loss3attack20_corr_chunk64/correlations.csv`
- `analysis_outputs/darcy_metric_correlation_25samples_20260612_25samples_loss3attack20_corr_chunk64/fixed_effect_correlations.csv`

### 9.1 Within-sample correlations

These compare the four trained models within the same initial condition, removing the large model-group effect.

| metric vs attack gain | Pearson | Spearman |
|---|---:|---:|
| \( \|J^\top e\|_2 \) | 0.639 | 0.638 |
| binary first-order gain | 0.564 | 0.522 |
| \( \sigma_{\max}(J)\|e\|_2 \) | 0.516 | 0.544 |
| \( |\langle e,u_1\rangle| \) | 0.725 | 0.629 |
| \( \sigma_{\max}(J) \) | -0.689 | -0.599 |

### 9.2 Raw overall correlations

These mix model-group differences and are therefore less clean mechanistically.

| metric vs attack gain | Pearson | Spearman |
|---|---:|---:|
| \( \|J^\top e\|_2 \) | -0.121 | -0.248 |
| binary first-order gain | -0.131 | -0.256 |
| \( \sigma_{\max}(J)\|e\|_2 \) | -0.202 | -0.328 |
| \( |\langle e,u_1\rangle| \) | 0.434 | 0.205 |
| \( \sigma_{\max}(J) \) | -0.539 | -0.604 |

Interpretation:

- The raw overall correlations are confounded by model group: `loss3` has low attack gain but high \( \sigma_{\max}(J) \).
- The within-sample correlations are the more meaningful mechanism test.
- Within sample, \( \|J^\top e\|_2 \), binary first-order gain, and \( |\langle e,u_1\rangle| \) explain attack gain better than pure \( \sigma_{\max}(J) \).

## 10. Pairwise Metric Correlations

### 10.1 Within-sample pairwise correlations

| metric pair | Pearson | Spearman |
|---|---:|---:|
| \( \|J^\top e\|_2 \) vs binary first-order gain | 0.966 | 0.954 |
| \( \|J^\top e\|_2 \) vs \( \sigma_{\max}(J)\|e\|_2 \) | 0.943 | 0.950 |
| \( \|J^\top e\|_2 \) vs \( \sigma_{\max}(J) \) | -0.531 | -0.312 |
| \( \sigma_{\max}(J) \) vs \( \sigma_{\max}(J)\|e\|_2 \) | -0.310 | -0.145 |
| \( \sigma_{\max}(J) \) vs \( |\langle e,u_1\rangle| \) | -0.576 | -0.393 |

### 10.2 Raw overall pairwise correlations

| metric pair | Pearson | Spearman |
|---|---:|---:|
| \( \|J^\top e\|_2 \) vs binary first-order gain | 0.978 | 0.980 |
| \( \|J^\top e\|_2 \) vs \( \sigma_{\max}(J)\|e\|_2 \) | 0.984 | 0.981 |
| \( \|J^\top e\|_2 \) vs \( \sigma_{\max}(J) \) | 0.315 | 0.354 |
| \( \sigma_{\max}(J) \) vs \( \sigma_{\max}(J)\|e\|_2 \) | 0.406 | 0.430 |
| \( \sigma_{\max}(J) \) vs \( |\langle e,u_1\rangle| \) | -0.189 | -0.114 |

Interpretation:

- \( \|J^\top e\|_2 \) and binary first-order gain are very highly correlated.
- Pure \( \sigma_{\max}(J) \) is not a reliable proxy for \( \|J^\top e\|_2 \) within the same sample.
- \( \sigma_{\max}(J)\|e\|_2 \) tracks \( \|J^\top e\|_2 \) better than \( \sigma_{\max}(J) \) alone, but it is still an upper-bound-style proxy rather than the actual error-aligned sensitivity.

## 11. Vector Angle Results

Sources:

- `analysis_outputs/darcy_jacobian_probe_5gen_20260612_jacobian_5gen_1000c/summary.csv`
- `analysis_outputs/darcy_jacobian_probe_5gen_20260612_jacobian_5gen_1000c/vectors/`
- `analysis_outputs/darcy_five_model_attack_heatmaps_20260612_loss3attack20_five_models_sharedrange/arrays/`

Singular vector signs are arbitrary, so absolute angles are often more meaningful than signed cosines.

### 11.1 Mean absolute angles in degrees

| model | delta vs \(J^\top e\) | delta vs top \(v_1\) | \(J^\top e\) vs top \(v_1\) | error \(e\) vs top \(u_1\) |
|---|---:|---:|---:|---:|
| physics loss | 86.551604 | 86.267758 | 15.857891 | 32.747792 |
| loss1 | 85.867660 | 85.500897 | 15.904716 | 32.811921 |
| loss2 | 85.560257 | 84.671171 | 20.034150 | 38.397552 |
| loss3 | 88.277657 | 88.380983 | 18.435803 | 38.966482 |
| overall mean | 86.564295 | 86.205202 | 17.558140 | 35.730937 |

### 11.2 Mean signed cosines

| model | cos(delta, \(J^\top e\)) | cos(delta, top \(v_1\)) | cos(\(J^\top e\), top \(v_1\)) | cos(error \(e\), top \(u_1\)) |
|---|---:|---:|---:|---:|
| physics loss | -0.019540 | -0.027575 | 0.477821 | 0.403127 |
| loss1 | -0.040351 | -0.078398 | 0.491447 | 0.441271 |
| loss2 | -0.055719 | 0.014953 | -0.015859 | 0.000070 |
| loss3 | 0.022280 | 0.003890 | 0.013059 | 0.016206 |
| overall mean | -0.023332 | -0.021783 | 0.241617 | 0.215168 |

Interpretation:

- Attack deltas are nearly orthogonal to the full continuous \(J^\top e\) and to top \(v_1\) in these vector-angle summaries. This is plausible because the attack is binary and constrained, while \(J^\top e\) and \(v_1\) are continuous full-space vectors.
- \(J^\top e\) and top \(v_1\) can be closer, but this relationship varies strongly by model.
- The angle results should be treated as supporting diagnostics, not the main robustness metric.

## 12. Block SVD and Power Refinement

The block/2 approximation computes a coarse-grid singular value:

\[
\sigma_{\mathrm{block}}
= \sigma_{\max}(PJL),
\]

where \(L\) lifts a \(42 \times 42\) block vector to the full \(84/85\) grid, and \(P\) projects the full output back to the block output space.

The lifted direct estimate uses the block top vector \(v_{\mathrm{block}}\), lifts it,

\[
v_0 = L v_{\mathrm{block}},
\]

and then computes

\[
\sigma_{\mathrm{lift}}
= \|Jv_0\|_2.
\]

The one-step power refinement uses the lifted vector as an initializer:

\[
u_0 = \frac{Jv_0}{\|Jv_0\|_2},
\]

\[
v_1 = \frac{J^\top u_0}{\|J^\top u_0\|_2},
\]

\[
\sigma_{\mathrm{power}}
= \|Jv_1\|_2.
\]

Observed mean relative errors from the 6-sample check:

| approximation | mean error vs full SVD |
|---|---:|
| block/2 sigma | 2.89% |
| lifted direct \( \|Jv_0\|_2 \) | 2.83% |
| one-step power refinement | 0.03% |

Interpretation: the block/2 vector already gives a good direction, but the one-step full-space power refinement corrects the amplitude and direction enough to nearly match the full SVD top singular value.

## 13. Mechanism Explanation

The top singular value measures the worst possible input-output amplification:

\[
\sigma_{\max}(J)
= \max_{\|v\|_2=1}\|Jv\|_2.
\]

But adversarial loss growth for the current sample is controlled at first order by

\[
\nabla_a \mathcal{L}(a)
= J^\top(F(a)-u(a))
= J^\top e(a).
\]

For a small continuous perturbation \(\delta\),

\[
\mathcal{L}(a+\delta)-\mathcal{L}(a)
\approx \langle J^\top e,\delta\rangle.
\]

For the binary Darcy setting, the feasible first-order approximation becomes

\[
\mathcal{L}(a+\delta)-\mathcal{L}(a)
\approx \sum_i (J^\top e)_i \delta_i,
\quad
\delta_i \in \{9,-9\}.
\]

Therefore, if the current error \(e\) is weakly aligned with the sensitive output directions of \(J\), the model can have a large \(\sigma_{\max}(J)\) while still having a small adversarial loss increase.

This is exactly what the current Darcy results show:

- `loss3` has the smallest actual attack gain.
- `loss3` has the smallest \( \|J^\top e\|_2 \) and binary first-order gain.
- `loss3` has the weakest \( |\langle e,u_1\rangle| \).
- `loss3` has the largest \( \sigma_{\max}(J) \).

The clean conclusion is:

\[
\text{Loss3 improves robustness mainly by reducing error-aligned sensitivity, not by uniformly shrinking } \sigma_{\max}(J).
\]

## 14. What Is Fully Supported and What Is Not

Supported:

- On the 50 binary generalization datasets, `loss3` is best for 20-step attack gain in 50/50 cases.
- On the 25-sample Jacobian/SVD probe, `loss3` is best for \( \|J^\top e\|_2 \), binary first-order gain, and attack gain.
- Within-sample correlations support error-aligned sensitivity as a better explanation than pure spectral norm.

Not supported:

- It is not correct to say every possible robustness metric favors `loss3`.
- It is not correct to say `loss3` has the smallest Jacobian spectral norm.
- The pure top singular value \( \sigma_{\max}(J) \) points in the opposite direction in this experiment.

Best short paper-style statement:

Loss3 reduces Darcy Flow adversarial loss growth by reducing the error-aligned input gradient \(J^\top(F(a)-u(a))\) and binary-feasible first-order attack gain, even though it does not reduce the full Jacobian spectral norm. This suggests that practical robustness here is governed more by error-aligned sensitivity than by worst-case Jacobian sensitivity.

