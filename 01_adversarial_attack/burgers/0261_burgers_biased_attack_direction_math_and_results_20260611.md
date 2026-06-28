# Burgers Biased Attack Direction: Math and 3-Sample Results - 2026-06-11

This note records the discussion about why the adversarial attack direction is not necessarily the pure top-SVD direction of the error Jacobian. It summarizes the local mathematics, the direction/scale quantities that were compared, and the current 3-sample full `1024 x 1024` Burgers experiment results.

Related source records:

- `docs/burgers_wideparam_loss3targeted_biased_local_direction_20260611.md`
- `docs/burgers_wideparam_full1024_svd_attack3_detailed_spectrum_similarity_20260611.md`
- `forensics/burgers_wideparam_loss3targeted_full1024_svd_attack3_20260611`
- `forensics/burgers_wideparam_loss3targeted_biased_local_direction_20260611`

## Symbols

Let the model prediction and solver output be:

$$
f_{\mathrm{model}}(x), \qquad f_{\mathrm{solver}}(x)
$$

Define the error function:

$$
e(x)=f_{\mathrm{model}}(x)-f_{\mathrm{solver}}(x)
$$

At a fixed sample \(x_0\), define the clean residual:

$$
b=e(x_0)=f_{\mathrm{model}}(x_0)-f_{\mathrm{solver}}(x_0)
$$

Define the local error Jacobian:

$$
A=J_e(x_0)=J_{\mathrm{model}}(x_0)-J_{\mathrm{solver}}(x_0)
$$

For a small input perturbation \(\delta\), the local affine approximation is:

$$
e(x_0+\delta)\approx b+A\delta
$$

Here \(b\) is already present before attack. This is the key reason the attack-loss problem is not the same as the pure SVD sensitivity problem.

## Local Attack Loss Expansion

Use MSE-style loss over \(n\) output coordinates:

$$
L(x)=\frac{1}{n}\|e(x)\|_2^2
$$

The local attack loss growth is:

$$
\Delta L(\delta)
=
\frac{1}{n}\|b+A\delta\|_2^2-\frac{1}{n}\|b\|_2^2
$$

Expand the square:

$$
\Delta L(\delta)
=
\frac{2}{n}b^\top A\delta
+
\frac{1}{n}\delta^\top A^\top A\delta
$$

Define:

$$
Q=A^\top A
$$

and define the dual/input-space bias-gradient vector:

$$
c=A^\top b
$$

Then:

$$
\Delta L(\delta)
=
\frac{2}{n}c^\top\delta
+
\frac{1}{n}\delta^\top Q\delta
$$

The first term is linear in \(\delta\), while the second term is quadratic in \(\delta\). Therefore, when the perturbation radius \(r=\|\delta\|_2\) tends to zero, the linear term dominates:

$$
\frac{2}{n}c^\top\delta = O(r)
$$

$$
\frac{1}{n}\delta^\top Q\delta = O(r^2)
$$

So in the infinitesimal-radius regime, the direction is mainly controlled by \(c=A^\top b\), not by the top singular vector of \(A\).

## Direction 1: Pure SVD / Rayleigh Direction

The pure SVD sensitivity problem ignores the existing clean residual \(b\) and asks:

$$
\max_{\|\delta\|_2\le r}\|A\delta\|_2^2
$$

Equivalently:

$$
\max_{\|\delta\|_2\le r}\delta^\top A^\top A\delta
$$

Let:

$$
A v_1=\sigma_1 u_1
$$

where \(\sigma_1\) is the largest singular value and \(v_1\) is the top right singular vector. The maximizer is:

$$
\delta_{\mathrm{svd}}=r v_1
$$

The corresponding scalar strength is the top singular value:

$$
\sigma_1=\|A\|_2
$$

If converted to MSE-scale local quadratic gain at radius \(r\), the pure SVD prediction is:

$$
G_{\mathrm{svd}}(r)
=
\frac{r^2}{n}\sigma_1^2
$$

So for the pure SVD problem:

- direction: \(v_1\)
- scalar strength: \(\sigma_1\)
- MSE-scale radius-\(r\) gain: \(\frac{r^2}{n}\sigma_1^2\)

## Direction 2: Biased / Dual / Outward Direction

For the actual local attack-loss problem, the clean residual \(b\) creates the linear term:

$$
\frac{2}{n}c^\top\delta
$$

where:

$$
c=A^\top b
$$

When \(r\to 0\), the optimal direction solves:

$$
\max_{\|\delta\|_2\le r} c^\top\delta
$$

So the infinitesimal attack direction is:

$$
v_{\mathrm{bias}}
=
\frac{c}{\|c\|_2}
=
\frac{A^\top b}{\|A^\top b\|_2}
$$

This is the input-space dual direction of the clean residual. It is also the normalized gradient direction of the MSE loss under the local affine model.

The gradient of the MSE loss with respect to the input is:

$$
\nabla_x L(x_0)
=
\frac{2}{n}A^\top b
=
\frac{2}{n}c
$$

Therefore the scalar analogous to a "singular value" for this biased infinitesimal direction is not a singular value. The natural scalar is the bias-gradient norm:

$$
\|c\|_2=\|A^\top b\|_2
$$

In MSE-gradient units:

$$
\|\nabla_x L(x_0)\|_2
=
\frac{2}{n}\|A^\top b\|_2
$$

The maximum first-order MSE growth at radius \(r\) is:

$$
G_{\mathrm{bias,1st}}(r)
=
\frac{2r}{n}\|A^\top b\|_2
$$

If we also include the quadratic term evaluated along this biased direction, then:

$$
G_{\mathrm{bias,total}}(r)
=
\frac{2r}{n}\|c\|_2
+
\frac{r^2}{n}v_{\mathrm{bias}}^\top Qv_{\mathrm{bias}}
$$

So for the biased infinitesimal problem:

- direction: \(v_{\mathrm{bias}}=\frac{A^\top b}{\|A^\top b\|_2}\)
- scalar strength: \(\|A^\top b\|_2\)
- MSE-gradient norm: \(\frac{2}{n}\|A^\top b\|_2\)
- first-order MSE radius-\(r\) gain: \(\frac{2r}{n}\|A^\top b\|_2\)

## Direction 3: Finite-Radius Affine Trust-Region Direction

At finite attack radius \(r\), the local affine problem keeps both the linear and quadratic terms:

$$
\max_{\|\delta\|_2\le r}
\left(
\delta^\top Q\delta+2c^\top\delta
\right)
$$

with:

$$
Q=A^\top A,\qquad c=A^\top b
$$

The KKT condition gives:

$$
Q\delta+c=\mu\delta
$$

Equivalently:

$$
\delta(\mu)=(\mu I-Q)^{-1}c
$$

with:

$$
\mu>\lambda_{\max}(Q)
$$

and \(\mu\) is chosen so that:

$$
\|\delta(\mu)\|_2=r
$$

The finite-radius affine local gain is:

$$
G_{\mathrm{affine}}(r)
=
\frac{1}{n}
\left(
2c^\top\delta_{\mathrm{affine}}
+
\delta_{\mathrm{affine}}^\top Q\delta_{\mathrm{affine}}
\right)
$$

This direction is more faithful to the local finite-radius attack-loss objective than pure SVD, because it includes both the bias term and the Jacobian curvature term.

However, the current 3-sample experiment showed that the final nonlinear attack delta was much closer to \(A^\top b\) than to this finite-radius affine direction. This can happen because the actual model and solver Jacobians change along the nonlinear attack trajectory.

## Direction 4: Actual Nonlinear Attack Delta

The actual attack is not just the local affine formula. It uses iterative nonlinear optimization, here a 15-step P2Q2 RMS-L2 attack. The final attack perturbation is:

$$
\delta_{\mathrm{attack}}
$$

The experiment compared this final nonlinear attack direction against:

$$
v_1
$$

$$
v_{\mathrm{bias}}=\frac{A^\top b}{\|A^\top b\|_2}
$$

$$
\frac{\delta_{\mathrm{affine}}}{\|\delta_{\mathrm{affine}}\|_2}
$$

## What Was Compared

There are four main directions in the comparison:

| direction | formula | meaning | scalar strength |
|---|---|---|---|
| top SVD direction | \(v_1\) | maximizes \(\|A\delta\|_2^2\) | \(\sigma_1=\|A\|_2\) |
| biased dual direction | \(\frac{A^\top b}{\|A^\top b\|_2}\) | infinitesimal attack-loss growth direction | \(\|A^\top b\|_2\) |
| finite-radius affine direction | \(\delta_{\mathrm{affine}}\) | local finite-radius trust-region direction | \(G_{\mathrm{affine}}(r)\) |
| nonlinear attack direction | \(\delta_{\mathrm{attack}}\) | actual final PGD/attack perturbation | actual attack loss growth |

The direction comparisons used cosine similarity and angles, especially:

$$
|\cos(\delta_{\mathrm{attack}},v_1)|
$$

$$
|\cos(\delta_{\mathrm{attack}},v_{\mathrm{bias}})|
$$

$$
|\cos(\delta_{\mathrm{attack}},\delta_{\mathrm{affine}})|
$$

and:

$$
\angle(v_1,v_{\mathrm{bias}})
$$

The scalar correlation comparisons used:

$$
\sigma_1=\|A\|_2
$$

$$
\|A^\top b\|_2
$$

$$
G_{\mathrm{svd}}(r)
$$

$$
G_{\mathrm{bias,total}}(r)
$$

$$
G_{\mathrm{affine}}(r)
$$

against the actual attack metrics:

$$
L(x_0+\delta_{\mathrm{attack}})-L(x_0)
$$

and:

$$
L(x_0+\delta_{\mathrm{attack}})
$$

## Experiment Context

Source SVD root:

`forensics/burgers_wideparam_loss3targeted_full1024_svd_attack3_20260611`

Biased-direction output root:

`forensics/burgers_wideparam_loss3targeted_biased_local_direction_20260611`

Samples:

| sample | dataset | index | family |
|---:|---|---:|---|
| 0 | `burgers_widevis_l3target_d31` | 10 | `powerlaw_fourier` |
| 1 | `burgers_widevis_l3target_d17` | 177 | `matern` |
| 2 | `burgers_widevis_l3target_d05` | 45 | `gaussian` |

Models:

- baseline
- loss1 adversarial training checkpoint
- loss2 adversarial training checkpoint
- loss3 adversarial training checkpoint

For each sample and model, the run used:

- full \(1024\times1024\) Jacobian, not projected or coarse SVD
- full SVD for \(J_{\mathrm{solver}}\), \(J_{\mathrm{model}}\), and \(J_{\mathrm{error}}\)
- 15-step P2Q2 RMS-L2 attack
- attack radius where final \(\delta_{\mathrm{rms}}\) reached `0.12` in all 12 model-sample pairs

## Headline Direction Results

Across 12 model-sample pairs:

| quantity | value |
|---|---:|
| pairs analyzed | 12 |
| mean angle between top SVD direction and \(A^\top b\) | 80.777 degrees |
| mean abs cosine, attack delta vs top SVD | 0.1537 |
| mean abs cosine, attack delta vs \(A^\top b\) | 0.7730 |
| mean abs cosine, attack delta vs finite-radius affine direction | 0.1679 |

Main interpretation:

$$
\delta_{\mathrm{attack}}
\text{ is much closer to }
\frac{A^\top b}{\|A^\top b\|_2}
\text{ than to }
v_1
$$

So the actual nonlinear attack direction is much better explained by the biased dual direction than by the pure top singular vector direction in this small 3-sample run.

## Per-Model Mean Direction Agreement

| model | angle SVD vs \(A^\top b\) deg | attack cos SVD | attack cos \(A^\top b\) | attack cos affine eps |
|---|---:|---:|---:|---:|
| baseline | 80.7953 | 0.313126 | 0.741439 | 0.340124 |
| loss1 | 78.2556 | 0.154677 | 0.805780 | 0.182628 |
| loss2 | 81.2142 | 0.106322 | 0.807138 | 0.0980208 |
| loss3 | 82.8418 | 0.0406373 | 0.737664 | 0.0507659 |

This means every model shows the same qualitative pattern: the final nonlinear attack delta is much closer to the biased \(A^\top b\) direction than to the top SVD direction.

## Compact Per-Pair Direction and Gain Table

| sample | model | \(\sigma_1\) | angle SVD vs \(A^\top b\) deg | cos attack SVD | cos attack \(A^\top b\) | cos attack affine | actual growth | SVD gain | affine gain |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | baseline | 2.36802 | 86.4495 | 0.0316196 | 0.832821 | 0.0796711 | 0.0333194 | 0.0811511 | 0.0813243 |
| 0 | loss1 | 3.02591 | 74.8808 | 0.237807 | 0.712122 | 0.249762 | 0.0108567 | 0.133060 | 0.133100 |
| 0 | loss2 | 3.07656 | 79.9648 | 0.0647416 | 0.741123 | 0.0471148 | 0.00766046 | 0.137337 | 0.137406 |
| 0 | loss3 | 1.06904 | 84.4276 | 0.00441821 | 0.760831 | 0.0477549 | 0.00632803 | 0.0165894 | 0.0166483 |
| 1 | baseline | 1.71212 | 75.9939 | 0.126244 | 0.758792 | 0.152445 | 0.00437249 | 0.0428246 | 0.0428718 |
| 1 | loss1 | 1.99931 | 78.5502 | 0.125629 | 0.869603 | 0.168368 | 0.00754316 | 0.0581908 | 0.0582664 |
| 1 | loss2 | 1.97707 | 78.0083 | 0.171126 | 0.794364 | 0.186470 | 0.00674436 | 0.0567107 | 0.0567302 |
| 1 | loss3 | 1.70113 | 77.8820 | 0.0919157 | 0.753722 | 0.103135 | 0.00396794 | 0.0419253 | 0.0419342 |
| 2 | baseline | 1.95388 | 79.9425 | 0.781514 | 0.632704 | 0.788255 | 0.00870896 | 0.0552268 | 0.0552365 |
| 2 | loss1 | 1.22419 | 81.3359 | 0.100595 | 0.835616 | 0.129753 | 0.00271454 | 0.0217529 | 0.0217728 |
| 2 | loss2 | 1.21643 | 85.6695 | 0.0830966 | 0.885926 | 0.0604777 | 0.00235327 | 0.0213816 | 0.0213939 |
| 2 | loss3 | 0.420253 | 86.2159 | 0.0255779 | 0.698440 | 0.00140744 | 0.00146813 | 0.00255264 | 0.00255496 |

## Scalar Correlation Results

Across 12 model-sample pairs:

| x | y | n | Pearson | Spearman |
|---|---|---:|---:|---:|
| error spectral norm \(\sigma_1\) | attack loss growth | 12 | 0.476677 | 0.846154 |
| error spectral norm \(\sigma_1\) | final attacked MSE | 12 | 0.512738 | 0.881119 |
| bias-gradient norm \(\|A^\top b\|_2\) | attack loss growth | 12 | 0.749221 | 0.888112 |
| bias-gradient norm \(\|A^\top b\|_2\) | final attacked MSE | 12 | 0.779204 | 0.923077 |
| SVD local gain \(G_{\mathrm{svd}}(r)\) | attack loss growth | 12 | 0.430769 | 0.846154 |
| SVD local gain \(G_{\mathrm{svd}}(r)\) | final attacked MSE | 12 | 0.469991 | 0.881119 |
| outward local gain along \(A^\top b\) | attack loss growth | 12 | 0.751010 | 0.832168 |
| outward local gain along \(A^\top b\) | final attacked MSE | 12 | 0.778240 | 0.874126 |
| affine local gain \(G_{\mathrm{affine}}(r)\) | attack loss growth | 12 | 0.431559 | 0.846154 |
| affine local gain \(G_{\mathrm{affine}}(r)\) | final attacked MSE | 12 | 0.470770 | 0.881119 |

The scalar result supports the direction result: \(\|A^\top b\|_2\) is more correlated with actual attack loss growth than the pure error spectral norm \(\sigma_1\) in this 3-sample test.

## Loss3 Robustness Advantage in This 3-Sample Test

Mean metrics across the three samples:

| model | error spectral norm \(\sigma_1\) | initial MSE | final attacked MSE | attack growth |
|---|---:|---:|---:|---:|
| baseline | 2.01134 | 0.000757474 | 0.0162244 | 0.0154669 |
| loss1 | 2.08314 | 0.000747912 | 0.00778604 | 0.00703813 |
| loss2 | 2.09002 | 0.000764422 | 0.00635045 | 0.00558603 |
| loss3 | 1.06347 | 0.000165457 | 0.00408682 | 0.00392136 |

Loss3 reductions versus the other models:

| metric | reference | loss3 value | reference value | reduction fraction |
|---|---|---:|---:|---:|
| error spectral norm \(\sigma_1\) | baseline | 1.06347 | 2.01134 | 47.1261% |
| error spectral norm \(\sigma_1\) | loss1 | 1.06347 | 2.08314 | 48.9484% |
| error spectral norm \(\sigma_1\) | loss2 | 1.06347 | 2.09002 | 49.1165% |
| final attacked MSE | baseline | 0.00408682 | 0.0162244 | 74.8107% |
| final attacked MSE | loss1 | 0.00408682 | 0.00778604 | 47.5109% |
| final attacked MSE | loss2 | 0.00408682 | 0.00635045 | 35.6452% |
| attack growth | baseline | 0.00392136 | 0.0154669 | 74.6468% |
| attack growth | loss1 | 0.00392136 | 0.00703813 | 44.2840% |
| attack growth | loss2 | 0.00392136 | 0.00558603 | 29.8005% |

In this small 3-sample robustness probe, loss3 has a clear advantage in:

- smaller error spectral norm
- smaller clean/initial MSE
- smaller final attacked MSE
- smaller absolute attack loss growth

## Main Conclusion

The pure SVD direction answers this question:

$$
\text{Which input direction is maximally amplified by the local error Jacobian } A?
$$

The biased attack direction answers a different question:

$$
\text{Which input direction most increases the current loss } \|b+A\delta\|_2^2?
$$

Because the current loss already has a residual \(b\), the attack-loss objective contains a first-order term:

$$
2(A^\top b)^\top\delta
$$

Therefore, at very small radius:

$$
\delta_{\mathrm{attack}}
\text{ should point toward }
A^\top b
\text{ rather than toward }
v_1
$$

The current 3-sample experiment is consistent with that prediction:

$$
|\cos(\delta_{\mathrm{attack}},A^\top b)|
\approx
0.773
$$

while:

$$
|\cos(\delta_{\mathrm{attack}},v_1)|
\approx
0.154
$$

So the final interpretation is:

- SVD spectral norm \(\sigma_1\) is still useful as a local error-sensitivity / Lipschitz-style robustness metric.
- For explaining actual adversarial loss increase and the attack delta direction, the biased dual quantity \(A^\top b\) is more directly relevant.
- The scalar corresponding to \(A^\top b\) is not a singular value; it is the bias-gradient norm \(\|A^\top b\|_2\), or in MSE-gradient units \(\frac{2}{n}\|A^\top b\|_2\).
- In the current 3-sample result, \(\|A^\top b\|_2\) correlates more strongly with attack growth than \(\sigma_1\).

