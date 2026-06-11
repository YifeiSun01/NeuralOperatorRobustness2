# Burgers SVD, Attack, and A^T b Robustness Metric Summary - 2026-06-11

This note consolidates the discussion and results after the earlier biased-direction markdown records. It records the mathematical definitions, the 25-sample full `1024 x 1024` SVD/attack experiment, the residual-aware `A^T b` metric, direction-alignment evidence, loss3 advantages, and runtime/complexity comparison.

No credentials are recorded here.

## Source Artifacts

- SVD25 root: `forensics/burgers_wideparam_loss3targeted_full1024_svd_attack25_reuse3_20260611`
- Biased-direction root: `forensics/burgers_wideparam_loss3targeted_full1024_svd_attack25_biased_local_direction_20260611`
- SVD25 report: `docs/burgers_wideparam_loss3targeted_full1024_svd_attack25_reuse3_20260611.md`
- Biased-direction report: `docs/burgers_wideparam_loss3targeted_full1024_svd_attack25_biased_local_direction_20260611.md`
- Earlier mathematical note: `docs/burgers_biased_attack_direction_math_and_results_20260611.md`
- Main metrics table: `forensics/burgers_wideparam_loss3targeted_full1024_svd_attack25_biased_local_direction_20260611/biased_direction_metrics.csv`
- Correlation table: `forensics/burgers_wideparam_loss3targeted_full1024_svd_attack25_biased_local_direction_20260611/biased_direction_correlations.csv`
- Runtime table: `forensics/burgers_wideparam_loss3targeted_full1024_svd_attack25_reuse3_20260611/runtime_components.csv`

## Experiment Scope

The completed SVD/attack analysis used:

- `25` fixed samples.
- `4` models: baseline, loss1, loss2, loss3.
- `100` model-sample pairs.
- Full `1024 x 1024` Jacobians.
- No block projection.
- No randomized SVD.
- No top-k approximation in the saved full-SVD run.
- `15` attack steps for the actual adversarial attack comparison.

Sample composition:

| split | count |
|---|---:|
| generalization | 21 |
| train | 2 |
| test | 2 |

Family composition:

| family | count |
|---|---:|
| powerlaw_fourier | 6 |
| gaussian | 6 |
| matern | 5 |
| original_gaussian | 4 |
| sine_mixture | 4 |

## Local Error Model

Define the model-solver error:

$$
e(x)=f_{\mathrm{model}}(x)-f_{\mathrm{solver}}(x)
$$

At a fixed sample \(x_0\), define:

$$
b=e(x_0)
$$

and:

$$
A=J_e(x_0)=J_{\mathrm{model}}(x_0)-J_{\mathrm{solver}}(x_0)
$$

For a small input perturbation \(\delta\), the local affine approximation is:

$$
e(x_0+\delta)\approx b+A\delta
$$

The local MSE loss is:

$$
L(x)=\frac{1}{n}\|e(x)\|_2^2
$$

Therefore local loss increase is:

$$
\Delta L(\delta)
=
\frac{1}{n}\|b+A\delta\|_2^2-\frac{1}{n}\|b\|_2^2
$$

Expanding:

$$
\Delta L(\delta)
=
\frac{2}{n}b^\top A\delta
+
\frac{1}{n}\delta^\top A^\top A\delta
$$

Let:

$$
c=A^\top b
$$

Then:

$$
\Delta L(\delta)
=
\frac{2}{n}c^\top\delta
+
\frac{1}{n}\delta^\top A^\top A\delta
$$

For small radius \(r=\|\delta\|_2\), the linear term is \(O(r)\) and the quadratic term is \(O(r^2)\). Thus the infinitesimal attack direction is controlled by \(A^\top b\), not by the top singular vector alone.

## Three Robustness Diagnostics

### 1. Actual attack loss increase

The actual attack metric is:

$$
\Delta L_{\mathrm{attack}}
=
L(x+\delta_{\mathrm{attack}})-L(x)
$$

This is the most direct empirical robustness metric, but it requires running iterative attack optimization.

### 2. Error-Jacobian spectral norm

The SVD/spectral diagnostic is:

$$
\sigma_1(A)=\|A\|_2
$$

It measures the largest local amplification of the error Jacobian. It is a local sensitivity or Lipschitz-style quantity. It depends on \(A\), but not directly on the current residual direction \(b\).

### 3. Residual-aware A^T b metric

The residual-aware diagnostic is:

$$
A^\top b
$$

with strength:

$$
\|A^\top b\|_2
$$

This is the input-space dual/gradient direction of the current error residual. In MSE-gradient units:

$$
\|\nabla_x L(x_0)\|_2
=
\frac{2}{n}\|A^\top b\|_2
$$

This quantity is directly derived from the first-order term in adversarial loss growth.

## Difference Between Strength and Local Gain

Four scalar quantities were discussed. They are related but not identical.

| quantity | meaning | depends on radius \(r\)? | includes loss expansion? |
|---|---|---:|---:|
| \(\sigma_1=\|A\|_2\) | top singular value / operator amplification | no | no |
| \(\|A^\top b\|_2\) | residual-aware first-order gradient strength | no | no |
| SVD local gain | predicted local loss increase along top SVD direction | yes | yes |
| A^T b outward local gain | predicted local loss increase along \(A^\top b\) direction | yes | yes |

The SVD direction is:

$$
v_{\mathrm{svd}}=v_1
$$

where \(v_1\) is the top right singular vector of \(A\). The SVD local gain at radius \(r\) is:

$$
G_{\mathrm{svd}}(r)
=
\frac{2r}{n}b^\top A v_{\mathrm{svd}}
+
\frac{r^2}{n}v_{\mathrm{svd}}^\top A^\top A v_{\mathrm{svd}}
$$

If only the pure SVD quadratic term is used, it becomes:

$$
\frac{r^2}{n}\sigma_1^2
$$

However, the recorded `svd_local_gain_eps_mse` includes both the linear and quadratic terms evaluated along the SVD direction.

The \(A^\top b\) outward direction is:

$$
v_{\mathrm{out}}
=
\frac{A^\top b}{\|A^\top b\|_2}
$$

The outward local gain at radius \(r\) is:

$$
G_{\mathrm{out}}(r)
=
\frac{2r}{n}b^\top A v_{\mathrm{out}}
+
\frac{r^2}{n}v_{\mathrm{out}}^\top A^\top A v_{\mathrm{out}}
$$

Since:

$$
b^\top A v_{\mathrm{out}}
=
\|A^\top b\|_2
$$

the first-order term is:

$$
\frac{2r}{n}\|A^\top b\|_2
$$

Thus \(G_{\mathrm{out}}(r)\) is the radius-normalized, loss-scale version of the \(A^\top b\) diagnostic.

## Affine Direction

The affine direction is the finite-radius optimum under the local affine model:

$$
e(x+\delta)\approx b+A\delta
$$

It solves:

$$
\max_{\|\delta\|_2\le r}
\left(
2(A^\top b)^\top\delta
+
\delta^\top A^\top A\delta
\right)
$$

This includes both the first-order \(A^\top b\) term and the second-order SVD/Jacobian term. If \(r\to 0\), it tends toward the \(A^\top b\) direction. At larger finite radius, it can be influenced by the spectral/SVD term.

In the current experiment, the actual nonlinear attack direction is much closer to \(A^\top b\) than to either the top SVD direction or the finite-radius affine direction.

## Direction Similarity Results

Across all `100` model-sample pairs:

| direction comparison | mean cosine | median cosine |
|---|---:|---:|
| SVD direction vs \(A^\top b\) direction | 0.2400 | 0.1667 |
| actual attack delta vs SVD direction | 0.2005 | 0.1121 |
| actual attack delta vs \(A^\top b\) direction | 0.7804 | 0.7869 |
| actual attack delta vs affine direction | 0.2189 | 0.1200 |

The average angle between top SVD direction and \(A^\top b\) direction is:

$$
75.200^\circ
$$

Per-model mean direction agreement:

| model | attack delta vs SVD cosine | attack delta vs \(A^\top b\) cosine | attack delta vs affine cosine | SVD vs \(A^\top b\) angle |
|---|---:|---:|---:|---:|
| baseline | 0.3049 | 0.7935 | 0.3405 | 71.3502 |
| loss1 | 0.2091 | 0.8031 | 0.2266 | 76.6468 |
| loss2 | 0.1965 | 0.7987 | 0.2083 | 73.6884 |
| loss3 | 0.0915 | 0.7265 | 0.1002 | 79.1153 |

Interpretation:

$$
\delta_{\mathrm{attack}}
\text{ is much closer to }
A^\top b
\text{ than to }
v_1
$$

This supports the claim that \(A^\top b\) is a better local proxy for the actual attack direction than the top singular vector.

## Correlations With Actual Attack Loss Increase

All-pair correlations, `n = 100`:

| x | y | Pearson | Spearman |
|---|---|---:|---:|
| error spectral norm \(\|A\|_2\) | attack loss growth | 0.6372 | 0.7778 |
| \(\|A^\top b\|_2\) | attack loss growth | 0.7453 | 0.8507 |
| SVD local gain | attack loss growth | 0.5767 | 0.7802 |
| \(A^\top b\) outward local gain | attack loss growth | 0.6262 | 0.8715 |
| affine local gain | attack loss growth | 0.5776 | 0.7807 |

All-pair correlations with final attacked MSE:

| x | y | Pearson | Spearman |
|---|---|---:|---:|
| error spectral norm \(\|A\|_2\) | final attacked MSE | 0.7240 | 0.8158 |
| \(\|A^\top b\|_2\) | final attacked MSE | 0.8279 | 0.8869 |
| SVD local gain | final attacked MSE | 0.6806 | 0.8179 |
| \(A^\top b\) outward local gain | final attacked MSE | 0.6617 | 0.8947 |
| affine local gain | final attacked MSE | 0.6814 | 0.8183 |

Interpretation:

- The SVD spectral norm is positively correlated with adversarial loss growth.
- The \(A^\top b\) family of metrics is more strongly correlated with attack loss growth and final attacked MSE.
- Directional similarity alone is not the strongest scalar predictor of loss growth, but it clearly shows the attack delta is closer to \(A^\top b\) than to the top SVD direction.

## Loss3 Robustness Advantage

Mean metrics across the `25` samples:

| model | attack loss growth | final attacked MSE | error spectral norm \(\|A\|_2\) | \(\|A^\top b\|_2\) | SVD local gain | \(A^\top b\) outward gain |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.009164 | 0.010259 | 2.367025 | 0.462951 | 0.108082 | 0.032898 |
| loss1 | 0.005838 | 0.006661 | 1.702470 | 0.315854 | 0.061843 | 0.009502 |
| loss2 | 0.005623 | 0.006479 | 1.855112 | 0.343340 | 0.071077 | 0.015741 |
| loss3 | 0.003063 | 0.003307 | 1.271530 | 0.111950 | 0.032733 | 0.005090 |

Loss3 is best on average for all listed metrics.

Per-sample low-is-better wins:

| metric | loss3 wins |
|---|---:|
| attack loss growth | 22 / 25 |
| final attacked MSE | 23 / 25 |
| \(\|A^\top b\|_2\) | 19 / 25 |
| error spectral norm \(\|A\|_2\) | 16 / 25 |
| clean initial MSE | 18 / 25 |

Interpretation:

- Loss3 has the clearest advantage in actual attack metrics.
- Loss3 also has a strong advantage in the residual-aware \(A^\top b\) metric.
- Loss3 has a positive but less absolute advantage in the pure spectral norm metric.

This supports the empirical claim that loss3 adversarial training reduces both actual adversarial vulnerability and local residual-aware vulnerability.

## Runtime and Complexity Comparison

### Completed SVD25 timing

The completed full SVD25 run reported:

| component | wall time |
|---|---:|
| total SVD25 workflow | 14589.0 s |
| total attack time | 183.0 s |
| total Jacobian + SVD time | 14401.2 s |

The SVD25 run used full `1024 x 1024` Jacobians and full SVD. This is stronger than what is needed if only the top spectral norm is required.

### Component timing

For non-reused samples in the SVD25 run:

| component | mean time |
|---|---:|
| solver Jacobian | 292.3 s / sample |
| solver full SVD | 53.7 s / sample |
| model Jacobian | 3.0 s / model-sample |
| model full SVD | 37.1 s / model-sample |
| error full SVD | 35.7 s / model-sample |

The solver Jacobian is much slower than the model Jacobian because the solver is a PDE/numerical-solver path, while the model is a neural network path with fast autograd. The earlier phrase "solver copy" should be understood as solver Jacobian/SVD work, not a simple file copy.

### Top spectral norm does not require full SVD

If the goal is only:

$$
\sigma_1(A)=\|A\|_2
$$

then a full SVD is unnecessary. One can use power iteration, Lanczos, or another top singular value method. This is cheaper than full SVD, but still needs repeated applications of \(A\) and \(A^\top\).

If a full explicit error matrix \(A\) is already available, the approximate costs are:

| method on explicit \(1024 x 1024\) matrix | observed CPU timing |
|---|---:|
| one explicit \(A^\top b\) matrix-vector multiply | 0.08 to 0.096 s |
| 30-step power iteration for \(\sigma_1\) | 4.6 to 5.1 s |
| full SVD | 30 to 60 s |

This CPU micro-benchmark used saved error Jacobian matrices and was stopped after two representative matrices to avoid wasting CPU during the active retraining run. It is a small benchmark, but it gives the right order of magnitude: when \(A\) is explicit, \(A^\top b\) is a single matrix-vector multiply, while spectral norm estimation is iterative and full SVD is much heavier.

### Fastest reasonable algorithms

The fairest comparison is by each metric's fastest reasonable implementation:

| metric | fastest reasonable approach | comments |
|---|---|---|
| \(\sigma_1(A)\) | power iteration/Lanczos using \(A v\) and \(A^\top u\) | no full SVD needed, but iterative |
| \(A^\top b\) | one VJP: \(J_{\mathrm{model}}^\top b - J_{\mathrm{solver}}^\top b\) | no full Jacobian required in principle |
| 15-step attack | iterative attack optimization | roughly 15 gradient/solver/model steps |

The most important computational point is:

$$
A^\top b
=
(J_{\mathrm{model}}-J_{\mathrm{solver}})^\top b
=
J_{\mathrm{model}}^\top b
-
J_{\mathrm{solver}}^\top b
$$

This can be computed by vector-Jacobian products (VJPs), without explicitly constructing the full `1024 x 1024` Jacobian and without doing SVD.

### Runtime interpretation

The runtime claim should be stated carefully:

- Full SVD is not required for the top spectral norm.
- Top spectral norm is still iterative if computed efficiently.
- \(A^\top b\) can be computed as one residual-aware VJP, so it is naturally cheaper than iterative spectral norm estimation and cheaper than multi-step attack.
- In the current posthoc implementation, \(A^\top b\) was computed from saved matrices/Jacobians; the next cleaner benchmark would implement direct VJP timing.

## Proposed Paper Framing

The proposed innovation can be framed as a residual-aware local robustness diagnostic:

$$
\|A^\top b\|_2
$$

or as a local outward gain:

$$
G_{\mathrm{out}}(r)
=
\frac{2r}{n}\|A^\top b\|_2
+
\frac{r^2}{n}v_{\mathrm{out}}^\top A^\top A v_{\mathrm{out}}
$$

with:

$$
v_{\mathrm{out}}
=
\frac{A^\top b}{\|A^\top b\|_2}
$$

Compared with the traditional Jacobian spectral norm, this metric:

- incorporates the current prediction residual \(b\);
- is derived directly from the first-order term of adversarial loss growth;
- aligns much better with the actual attack delta direction;
- correlates better with attack loss growth;
- can be computed without full SVD and, in principle, without explicitly materializing the full Jacobian.

Concise paper-style statement:

> We propose a residual-aware Jacobian-gradient robustness diagnostic, \(\|A^\top b\|_2\), derived from the first-order expansion of adversarial error growth. Unlike the standard Jacobian spectral norm, which measures worst-case operator sensitivity independent of the current prediction residual, this metric directly captures the local direction and magnitude of loss-increasing perturbations. Empirically, it aligns more strongly with actual adversarial perturbation directions and correlates better with adversarial loss growth, while avoiding the expensive full-SVD computation.

## Current Follow-Up Training Status

After the SVD25 analysis, the resumed adversarial retraining workflow was launched with SVD skipped:

```text
RUN_SVD=0
RUN_RETRAIN=1
RETRAIN_ORDER=loss3_then_loss12
```

As of this note, the loss3 retraining process `burgers_wideparam_loss3_1000ep_retrain_20260611` is running. The new retraining results are not included in the SVD25 analysis above, which uses the existing baseline/loss1/loss2/loss3 checkpoints from the completed SVD25 experiment.

## Bottom Line

The completed 25-sample study supports three points:

1. The spectral norm \(\|A\|_2\) is positively correlated with actual attack loss growth.
2. The residual-aware \(A^\top b\) metrics are more directly aligned with attack behavior, both in direction and scalar correlation.
3. Loss3 is the strongest model across actual attack growth, final attacked MSE, \(A^\top b\), and average spectral norm in this experiment.

This makes \(A^\top b\) a promising, cheaper, attack-aligned local robustness proxy for the paper.

