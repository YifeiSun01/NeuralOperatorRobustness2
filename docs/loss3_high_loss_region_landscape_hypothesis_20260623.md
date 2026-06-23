# Loss3 High-Loss Region Landscape Hypothesis - 2026-06-23

This note records the current working explanation for why `replace`-style
optimizers work well on Burgers/Darcy but fail on NS2D, the evidence we already
have, what the current evidence does and does not prove, and the direct
landscape-volume experiments that should be run next.

## Core Conclusion

The main conclusion is:

\[
\texttt{steepest\_add}
\]

is the more robust general optimizer, while

\[
\texttt{replace}
\]

works well only when the full-budget jump lands in a broad and forgiving
high-loss region.

The evidence points to this mechanism:

\[
\texttt{replace}
\text{ works when the high-loss region on the }\epsilon\text{-boundary is wide,}
\]

and fails when the true high-loss region is narrow, path-dependent, or hard to
hit with one full-budget jump.

This is more precise than saying simply:

\[
\text{Burgers is linear, NS is nonlinear.}
\]

Linearity/nonlinearity matters because it affects direction stability and
predictability, but the more direct explanation is the size and connectivity of
the high-loss region on the constraint boundary.

## Main Optimizer Results

The averaged Loss3 curves give the empirical facts that mechanism experiments
must explain.

| System | Best final method | Final Loss3 evidence |
| --- | --- | --- |
| Burgers 1D | `steepest_add`, nearly tied with `replace` | `steepest_add = 6.80804`, `raw_replace = 6.80783`, `steepest_replace = 6.80783`, `raw_add = 6.41518`, \(N=100\) |
| Darcy Flow | `replace` methods | `raw_replace = steepest_replace = 0.04996`, `raw_add = steepest_add = 0.04518`, \(N=20\) |
| NS2D | `steepest_add` | `steepest_add = 278.54`, `raw_add = 129.86`, `raw_replace = steepest_replace = 96.42`, \(N=20\) |

So `replace` is not universally better.  It is good on Burgers/Darcy, but not on
NS2D.

## Existing Evidence For The High-Loss Region Explanation

### Burgers Boundary Arc Evidence

For Burgers, we compared two final boundary perturbations:

- \(A = \delta_{\texttt{steepest\_replace}}\), with endpoint Loss3 about \(7.069\).
- \(B = \delta_{\texttt{steepest\_add}}\), with endpoint Loss3 about \(6.339\).

The weaker endpoint is \(B\), because:

\[
L(B) < L(A).
\]

We then moved along the boundary arc between \(A\) and \(B\), sampled 17 points,
and evaluated Loss3.

The lowest Loss3 on the arc was:

\[
6.172.
\]

The ratio to the weaker endpoint is:

\[
\frac{6.172}{6.339} \approx 0.974.
\]

This means the worst point on the sampled arc still has \(97.4\%\) of the weaker
endpoint's Loss3.

Also:

\[
100\%
\]

of the 17 sampled arc points were above:

\[
0.95 \times 6.339.
\]

Interpretation:

The high-loss region between `steepest_replace` and `steepest_add` is not a
single sharp point.  It forms a broad high-loss ridge.  `replace` does not need
to hit the exact best direction; many nearby boundary directions still give high
Loss3.

### Burgers 2D Slice Evidence

We also probed 2D slices around the final perturbations.

Around `steepest_replace`:

- \(96\%\) of the grid points reached at least \(90\%\) of the center Loss3.
- \(72\%\) of the grid points reached at least \(95\%\) of the center Loss3.

Around `steepest_add`:

- \(100\%\) of the grid points reached at least \(90\%\) of the center Loss3.
- \(84\%\) of the grid points reached at least \(95\%\) of the center Loss3.

Interpretation:

In these local 2D slices, Burgers has a visibly broad high-loss area near the
final solutions.

### Burgers Ray Evidence

For Burgers `steepest_replace`, ray endpoints along early trajectory directions
quickly approach the final high-loss value:

| Direction source | Endpoint Loss3 |
| --- | ---: |
| trajectory step 5 | \(6.883\) |
| trajectory step 10 | \(7.027\) |
| trajectory step 20 | \(7.105\) |
| final | \(7.069\) |

Interpretation:

`replace` finds a final-like high-loss boundary direction early.  Combined with
the broad ridge evidence, this explains why it performs well on Burgers.

### Burgers First-Order Prediction Evidence

This is important because it argues against the simple explanation:

\[
\text{Burgers is linear, therefore replace is accurate.}
\]

Let:

\[
L(\delta)
\]

be the Loss3 objective, and let the current perturbation be:

\[
\delta_k.
\]

The current gradient is:

\[
g_k = \nabla_\delta L(\delta_k).
\]

For a candidate next perturbation \(\delta_{\mathrm{cand}}\), the true gain is:

\[
\Delta L_{\mathrm{true}}
=
L(\delta_{\mathrm{cand}})-L(\delta_k).
\]

The first-order Taylor prediction is:

\[
\Delta L_{\mathrm{pred}}
=
g_k^\top(\delta_{\mathrm{cand}}-\delta_k).
\]

The relative prediction error is:

\[
\mathrm{RelErr}
=
\frac{
\left|
\Delta L_{\mathrm{pred}}-\Delta L_{\mathrm{true}}
\right|
}{
\left|\Delta L_{\mathrm{true}}\right|+\eta
}.
\]

For `add`, the candidate has the form:

\[
\delta_{\mathrm{add}}
=
\Pi_{\|\delta\|\le \epsilon}
\left(
\delta_k+\alpha d_k
\right).
\]

For `replace`, the candidate has the form:

\[
\delta_{\mathrm{replace}}
=
\Pi_{\|\delta\|\le \epsilon}
\left(
\epsilon d_k
\right).
\]

If Burgers `replace` worked mainly because the loss landscape was locally linear
and `replace` predictions were accurate, then the first-order prediction error
for `replace` should be small.

But the measured Burgers errors were:

| Candidate type | Relative first-order error |
| --- | ---: |
| `add` | \(0.0289\) |
| `replace` | \(3.407\) |

Interpretation:

The full-budget `replace` prediction is not especially accurate on Burgers, yet
`replace` performs well.  This supports the explanation that Burgers `replace`
works because the high-loss boundary region is forgiving, not because the
full-budget first-order prediction is perfectly accurate.

## Existing Evidence From NS2D

### NS2D Boundary Arc Evidence

For NS2D, the boundary arc between `steepest_replace` and `steepest_add` looks
very different.

For the N=3 exact probe:

- `steepest_replace` endpoint: \(85.09\)
- `steepest_add` endpoint: \(255.48\)
- minimum along arc relative to add endpoint:

\[
\frac{85.09}{255.48} \approx 0.333.
\]

Only \(1/9\) sampled arc points reached at least \(95\%\) of the add endpoint.

For the N=2 top-up:

- `steepest_replace` endpoint: \(114.54\)
- `steepest_add` endpoint: \(337.49\)
- minimum along arc relative to add endpoint:

\[
\frac{114.54}{337.49} \approx 0.339.
\]

Only \(2/9\) sampled arc points reached at least \(95\%\) of the add endpoint.

Interpretation:

NS2D does not show a broad shared high-loss ridge between the replace and add
solutions.  The true high-loss region is much more localized near the add
endpoint, while the replace endpoint is far lower.

### NS2D Post-Boundary Gain Evidence

`replace` reaches the boundary quickly, but then does not continue improving
well on the boundary.

N=3 exact probe:

| Method | Post-boundary gain mean |
| --- | ---: |
| `steepest_replace` | \(-51.05\) |
| `steepest_add` | \(+43.43\) |

N=2 top-up:

| Method | Post-boundary gain mean |
| --- | ---: |
| `steepest_replace` | \(-59.43\) |
| `steepest_add` | \(+52.80\) |

Interpretation:

For NS2D, reaching the \(\epsilon\)-boundary is not enough.  The optimizer must
find the right region of the boundary.  `replace` often lands on a poor boundary
region and then fails to correct effectively.

### NS2D JVP/VJP Candidate Evidence

The NS2D JVP/VJP probe tested whether a local Jacobian/top-singular-direction
story explains `replace`.

The mean true gain of candidate moves was:

| Candidate style | Mean true gain |
| --- | ---: |
| add-style candidates | \(+20.68\) |
| replace-style candidates | \(-48.13\) |

Even `replace_top_singular` candidates often had negative true gain.

Interpretation:

Even when using matrix-free Jacobian information, a full-radius replacement move
is not reliable on NS2D.  The issue is not merely that NS2D lacks local ascent
directions; the issue is that full-budget jumps can land in the wrong boundary
region.

## What The Current Evidence Does Not Fully Prove

The current evidence strongly supports the high-loss-region explanation, but it
is not yet a direct measurement of the full high-dimensional boundary volume.

The existing experiments include:

- boundary arcs,
- 2D slices,
- ray profiles,
- first-order prediction tests,
- local linearity tests,
- post-boundary gain tests,
- JVP/VJP candidate tests.

These are informative low-dimensional and path-based probes.  They do not yet
directly answer:

\[
\text{What fraction of the entire }\epsilon\text{-boundary is high-loss?}
\]

So the current evidence is strong but not the most direct possible evidence for
the claim that Burgers has a much larger high-loss boundary region than NS2D.

## Direct Boundary-Volume Experiments To Run Next

The most direct next experiment is to estimate the measure of the high-loss set
on the \(\epsilon\)-boundary.

Define:

\[
S_\epsilon
=
\{\delta:\|\delta\|_2=\epsilon\}.
\]

For each sample, define a reference value:

\[
L_{\max}
=
\max_{\text{optimizers}} L(\delta_{\text{final}}).
\]

For a threshold:

\[
\tau \in \{0.90, 0.95, 0.99\},
\]

define the high-loss set:

\[
H_\tau
=
\{
\delta\in S_\epsilon:
L(\delta)\ge \tau L_{\max}
\}.
\]

Then estimate:

\[
p_\tau
=
\mathbb{P}_{\delta\sim S_\epsilon}
\left[
L(\delta)\ge \tau L_{\max}
\right].
\]

This \(p_\tau\) is the direct "high-loss boundary volume" metric.

### Experiment 1: Global Boundary Volume

Sample random boundary points:

\[
z_i\sim \mathcal{N}(0,I),
\qquad
\delta_i
=
\epsilon \frac{z_i}{\|z_i\|_2}.
\]

Estimate:

\[
\hat p_\tau
=
\frac{1}{N}
\sum_i
\mathbf{1}
\left[
L(\delta_i)\ge \tau L_{\max}
\right].
\]

Expected pattern:

\[
\hat p_\tau^{\text{Burgers}}
\gg
\hat p_\tau^{\text{NS2D}}.
\]

If this holds, it directly supports the claim that Burgers has a broader
high-loss boundary region.

### Experiment 2: Local Endpoint Cap Volume

Global random sampling may miss tiny high-loss regions in high dimensions, so we
should also sample local spherical caps around optimizer endpoints.

Let:

\[
u^*
=
\frac{\delta^*}{\epsilon}.
\]

Sample:

\[
v\perp u^*,
\qquad
\|v\|_2=1.
\]

For angle \(\theta\), define:

\[
\delta(\theta,v)
=
\epsilon
\left(
\cos\theta\,u^*
+
\sin\theta\,v
\right).
\]

Then measure:

\[
p_\tau(\theta)
=
\mathbb{P}_v
\left[
L(\delta(\theta,v))
\ge
\tau L(\delta^*)
\right].
\]

This tells us how far we can move away from an endpoint on the boundary while
remaining high-loss.

Useful width metrics:

\[
\theta_{50}^{(\tau)}
=
\max\theta
\quad
\text{s.t.}
\quad
p_\tau(\theta)\ge 0.5.
\]

and:

\[
W_\tau
=
\int_0^{\theta_{\max}} p_\tau(\theta)\,d\theta.
\]

Expected pattern:

\[
W_\tau^{\text{Burgers}}
\gg
W_\tau^{\text{NS2D}}.
\]

### Experiment 3: High-Loss Basin Connectivity

Given two high-loss boundary points \(\delta_a\) and \(\delta_b\), sample the
geodesic between them and compute:

\[
r_{\mathrm{valley}}
=
\frac{
\min_s L(\mathrm{geodesic}(\delta_a,\delta_b;s))
}{
\min(L(\delta_a),L(\delta_b))
}.
\]

If:

\[
r_{\mathrm{valley}}\approx 1,
\]

then the two points are connected by a high-loss ridge.

If:

\[
r_{\mathrm{valley}}\ll 1,
\]

then the high-loss regions are separated by a low-loss valley.

Expected pattern:

- Burgers: \(r_{\mathrm{valley}}\) near \(1\).
- NS2D: \(r_{\mathrm{valley}}\) often much smaller when connecting replace-like
  and add-like endpoints.

## Final Intended Evidence Table

The next direct study should produce a table like this:

| System | Global \(p_{0.95}\) | Cap width \(W_{0.95}\) | Arc valley ratio | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Burgers | high | large | near \(1\) | high-loss boundary region is broad |
| NS2D | low | small | much below \(1\) | high-loss boundary region is narrow/localized |
| Darcy Flow | discrete analogue | many near-optimal flip sets | high near-optimal multiplicity | many good discrete high-loss choices |

## Current Bottom Line

The best current explanation is:

\[
\texttt{replace}
\text{ succeeds when it does not need to be precise.}
\]

Burgers has a broad high-loss boundary ridge, so a full-budget replacement jump
can land in many good places.  Darcy Flow has a discrete analogue: many
near-optimal flip sets, and replacement-like flip selection finds strong sets
quickly.

NS2D is different.  The true high-loss region is much more localized and
path-dependent.  A full-budget replacement step can reach the boundary but land
in a poor boundary region.  `steepest_add` performs better because it follows
the changing geometry and updates the perturbation path gradually.

Therefore, the current evidence supports this hierarchy:

\[
\text{boundary high-loss region width}
\quad >
\quad
\text{simple linear vs nonlinear explanation}.
\]

Linearity/nonlinearity is a contributing factor, but the more direct variable is
the size, width, and connectivity of the high-loss region on the feasible
boundary.

