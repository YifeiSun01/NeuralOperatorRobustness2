# Loss3 Replace vs Add Theory And Mechanism Unified Note

Updated: 2026-06-22 UTC

This note records the current unified explanation for why `replace` can be very
fast on Burgers, why this is not a universal optimizer theorem, and what
mechanism experiments should be used to verify or falsify the explanation on
Burgers, Darcy Flow, and NS2D.

Concrete validation experiment plan:

```text
docs/loss3_replace_add_linearity_validation_experiment_plan_20260622.md
```

The main point is:

\[
\boxed{
\texttt{replace} \text{ is favored when the attack is locally close to a stable,
boundary-attained, high-gain linear/quadratic problem.}
}
\]

and

\[
\boxed{
\texttt{add} \text{ is favored when the attack landscape is nonlinear,
path-dependent, sharply curved, or has narrow high-loss regions.}
}
\]

This means the Burgers explanation must be connected back to these theory
conditions. The empirical phrases "early good direction", "broad high-loss
boundary", "low curvature", and "boundary rotation" are not separate stories;
they are measurable proxies for the theoretical conditions under which
`replace` should work.

## Target Update Rules

The comparison uses four update rules:

- `raw_add`
- `raw_replace`
- `steepest_add`
- `steepest_replace`

The target objective is true Loss3:

\[
L_3(x+\delta)=\|F_\theta(x+\delta)-G(x+\delta)\|.
\]

The constrained attack is written as

\[
\max_{\delta\in\mathcal B_p(\epsilon)} \Phi(\delta).
\]

At iteration \(k\),

\[
g_k=\nabla_\delta \Phi(\delta_k).
\]

## Theory: Why Replace Can Be Faster

The local first-order model is

\[
\Phi(y)\approx \Phi(\delta_k)+g_k^\top(y-\delta_k).
\]

The `replace` update directly solves the current full-budget linearized
problem:

\[
\delta_{k+1}^{rep}
=
\arg\max_{\delta\in\mathcal B_p(\epsilon)}
g_k^\top \delta.
\]

For \(p=2\),

\[
\delta_{k+1}^{rep}
=
\epsilon \frac{g_k}{\|g_k\|_2}.
\]

The `add` update only takes a small projected step:

\[
\delta_{k+1}^{add}
=
\Pi_{\mathcal B_p(\epsilon)}(\delta_k+\alpha s_k).
\]

Therefore `replace` is more aggressive: it trusts the current local model over a
large portion of the feasible ball. `add` is more conservative: it only trusts
the local model over a step of size roughly \(\alpha\).

If \(\Phi\) is \(L\)-smooth, then

\[
\left|
\Phi(y)-\Phi(\delta_k)-g_k^\top(y-\delta_k)
\right|
\le
\frac{L}{2}\|y-\delta_k\|^2.
\]

For `add`, the local model error is approximately

\[
O(L\alpha^2).
\]

For `replace`, the jump can be \(O(\epsilon)\), or even close to \(2\epsilon\)
from one side of the ball to another, so the model error can be

\[
O(L\epsilon^2).
\]

Thus the central theoretical condition is:

\[
\boxed{
\texttt{replace} \text{ is accurate only when the full-budget linearized model
has small enough error.}
}
\]

If the local effective curvature/nonlinearity is small, `replace` can be fast
and accurate. If the local curvature is large or the direction field rotates
strongly, `replace` can jump in a direction that is good for the current
linearized model but bad for the real nonlinear objective.

## Quadratic / Power-Style Explanation

In a frozen local quadratic model,

\[
\Phi(\delta)=\frac12\delta^\top A\delta,
\qquad
g_k=A\delta_k.
\]

Then, directionally,

\[
\texttt{replace}:\quad \delta_{k+1}\propto A\delta_k,
\]

while

\[
\texttt{add}:\quad \delta_{k+1}\approx (I+\alpha A)\delta_k.
\]

If

\[
A v_i=\lambda_i v_i,\qquad \lambda_1>\lambda_2,
\]

then the non-leading component decays for replacement like

\[
\left(\frac{\lambda_2}{\lambda_1}\right)^k,
\]

whereas for additive updates it decays like

\[
\left(\frac{1+\alpha\lambda_2}{1+\alpha\lambda_1}\right)^k.
\]

Since

\[
\frac{\lambda_2}{\lambda_1}
<
\frac{1+\alpha\lambda_2}{1+\alpha\lambda_1},
\]

the fixed quadratic model predicts that `replace` aligns with the leading
direction faster than `add`.

This is the clean mathematical reason why, in a pure linear/quadratic problem,
`replace` should be faster.

## Why This Does Not Automatically Hold In Our Problem

The actual Loss3 problem is nonlinear. A local residual expansion is

\[
r(\delta)\approx r_0+J\delta,
\]

and the squared local loss is

\[
\frac12\|r_0+J\delta\|_2^2
=
c+b^\top\delta+\frac12\delta^\top A\delta,
\qquad
b=J^\top r_0,\quad A=J^\top J.
\]

Along the real attack path, however,

\[
A=A_k=J(x+\delta_k)^\top J(x+\delta_k),
\]

and

\[
b=b_k=J(x+\delta_k)^\top r(\delta_k)
\]

change with \(k\). Therefore `replace` is not guaranteed to behave like a true
power method. It can become a moving-target process:

\[
\delta_{k+1}^{rep}
\approx
\epsilon\frac{A_k\delta_k+b_k}{\|A_k\delta_k+b_k\|}.
\]

If \(A_k\), \(b_k\), or \(g_k\) rotate strongly, the full-budget step can be too
aggressive. This is the main reason a nonlinear NS2D problem can favor `add`
even though the frozen quadratic theory favors `replace`.

## The Key Continuous Hypothesis

The clean hypothesis is:

\[
\boxed{
\text{The more locally linear/quadratic and direction-stable the attack
landscape is, the larger the advantage of } \texttt{replace}.
}
\]

Equivalently:

\[
\text{linear/quadratic regime}
\quad\Longrightarrow\quad
\texttt{replace} \gg \texttt{add},
\]

\[
\text{mild nonlinear but broad-ridge regime}
\quad\Longrightarrow\quad
\texttt{replace} \gtrsim \texttt{add},
\]

\[
\text{strong nonlinear / rotating direction / narrow-peak regime}
\quad\Longrightarrow\quad
\texttt{replace} \le \texttt{add}.
\]

This is the theory-to-mechanism bridge.

## How The Burgers Evidence Connects To The Theory

The Burgers explanation is not independent of the theory. Each empirical
condition corresponds to one theoretical requirement:

| Theory condition | Burgers empirical proxy | Meaning |
| --- | --- | --- |
| Full-budget Taylor error must be small | scalar Loss3 becomes locally more linear; directional curvature drops after the early transition | `replace` can trust a large step more than it could in a strongly nonlinear region |
| Direction field must be stable | early replacement directions become final-like; gradients/Jacobian subspaces stabilize later | `replace` is not chasing a wildly moving target |
| Frozen quadratic/power-style alignment should be plausible | early ray directions become high-loss quickly; local spectrum becomes more dominated | `replace` can get leading high-gain directions faster |
| Full budget should be useful | optimum is effectively on the perturbation boundary; replacement reaches boundary immediately | the radial aggressiveness of `replace` is an advantage rather than a bug |
| Direction errors should be tolerated | high-loss boundary ridge is broad | even if `replace` is not exact, it still lands in a high-loss region |
| Boundary arrival is not convergence | large tangent residual and post-boundary gain | after reaching the boundary, the method can still improve by rotating direction |

The correct Burgers claim is:

\[
\boxed{
\text{Burgers supports the local conditions under which } \texttt{replace}
\text{ is expected to work.}
}
\]

The incorrect overclaim would be:

\[
\boxed{
\text{Burgers proves } \texttt{replace} \text{ is a true global power method.}
}
\]

The second statement should not be used.

## Current Burgers Evidence

### Formal Curve Evidence

Formal curve root:

\[
\text{analysis\_outputs/optimizer\_ablation\_20260622}
\]

For Burgers:

\[
\epsilon=8,\qquad \alpha=0.3,\qquad \text{steps}=300,\qquad N=100.
\]

Mean Loss3 curve metrics:

| Optimizer | \(N\) | step 0 | final | gain | step to 50% gain | step to 90% gain | mean curve height |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `raw_add` | 100 | 0.297596 | 6.415180 | 6.117584 | 35 | 130 | 5.284226 |
| `raw_replace` | 100 | 0.297596 | 6.807833 | 6.510237 | 2 | 4 | 6.715443 |
| `steepest_add` | 100 | 0.297596 | 6.808037 | 6.510441 | 21 | 78 | 6.088848 |
| `steepest_replace` | 100 | 0.297596 | 6.807833 | 6.510237 | 2 | 4 | 6.715443 |

This supports the behavioral fact that `replace` reaches high loss much faster
than additive methods on Burgers.

The aggregate summary also records, for the 300-step run:

- `raw_replace` / `steepest_replace`: final true Loss3 \(6.807833\), step to
  95% best about \(5.95\);
- `steepest_add`: final true Loss3 \(6.808037\), step to 95% best about
  \(74.39\);
- `raw_add`: final true Loss3 \(6.415180\), step to 95% best about \(115.95\).

So `steepest_add` can eventually reach the same final value, but `replace` gets
there far faster.

### Early Direction Evidence

From the Burgers p2q2 ray/profile mechanism summary:

| step | endpoint/final endpoint ratio for `steepest_replace` |
| ---: | ---: |
| 1 | 0.4074 |
| 5 | 0.9612 |
| 10 | 1.008 |
| 20 | 1.012 |

This means the replacement direction is already nearly final-like by steps 5 to
10. This supports the direction-stability / fast-alignment condition required
by the theory.

### Boundary Ridge Evidence

For p2q2 boundary arcs:

| pair | min over weaker endpoint |
| --- | ---: |
| `raw_add__steepest_add` | 1.0000 |
| `steepest_replace__raw_add` | 0.9863 |
| `steepest_replace__steepest_add` | 0.9861 |

Values near 1 mean the arc between final directions does not dip much below the
weaker endpoint. This supports a broad high-loss ridge rather than a narrow
isolated spike.

This ridge-width condition is not part of the pure quadratic power method
theory. It is an additional nonlinear robustness condition: even if the
replacement direction is not exact, the loss remains high if many nearby
boundary directions are good.

### Boundary Hit And Post-Boundary Gain

For p2q2:

| method | hit step 99 mean | post-boundary gain mean | tangent residual at hit mean | mean angle after hit |
| --- | ---: | ---: | ---: | ---: |
| `raw_add` | 32.41 | 1.28 | 0.397 | 0.2812 deg |
| `raw_replace` | 1 | 3.412 | 0.8552 | 30.68 deg |
| `steepest_add` | 12.75 | 1.467 | 0.4349 | 0.605 deg |
| `steepest_replace` | 1 | 3.412 | 0.8552 | 30.68 deg |

This shows:

\[
\text{boundary hit} \ne \text{convergence}.
\]

Replacement removes the radial phase immediately, then continues optimizing by
changing direction on the boundary.

### Local Linearity Evidence

Finite-difference scalar Loss3 local linearity score \(C_L\) decreased along the
Burgers path:

| region | mean \(C_L\) | median \(C_L\) |
| --- | ---: | ---: |
| early | 0.3314 | 0.0641 |
| middle | 0.0515 | 0.0109 |
| late | 0.0148 | 0.0046 |

Smaller means more locally linear under this finite-difference test. This
supports the claim that the scalar objective becomes more compatible with a
large-step first-order approximation farther out.

Caveat: the residual vector map itself does not uniformly become more affine in
all directions. The safest statement is about scalar Loss3 local linearity, not
global residual-map linearity.

### Curvature Evidence

Directional finite-difference curvature along the path:

| region | mean curvature | median curvature |
| --- | ---: | ---: |
| early | 0.1561 | 0.1141 |
| transition | 0.0402 | 0.0228 |
| middle / low corridor | 0.0267 | 0.0199 |
| late boundary | 0.0451 | 0.0368 |

The path enters a lower-curvature corridor after the early transition, although
the late boundary region is not perfectly flat.

This supports:

\[
\text{Burgers is not globally linear, but it becomes locally easier for
large-step updates after the early transition.}
\]

### Gradient And Jacobian Stability Evidence

PGD gradient adjacent-step rotation decreases:

| region | mean adjacent gradient rotation |
| --- | ---: |
| early | 16.18 deg |
| middle | 9.32 deg |
| late | 4.43 deg |
| very late | 3.23 deg |

The straight-line residual-Jacobian path experiment shows:

- top-1 clean-reference angle is already about \(40.55^\circ\) at \(t=0.1\);
- endpoint top-1 clean-reference angle is about \(57.64^\circ\);
- adjacent previous-reference angle drops to about \(1.89^\circ\) at the
  endpoint;
- \(\sigma_1(J_e)\) grows from about \(0.868\) at \(t=0\) to \(8.495\) at
  \(t=1\).

This means the clean-point geometry is not fixed. The better statement is:

\[
\text{the geometry changes sharply early, then evolves more smoothly in the
outer high-radius region.}
\]

That is compatible with the replace story, but it is not a proof of one fixed
global \(A\).

### Spectrum And Power-Method Caveat

The residual-Jacobian spectrum becomes more dominated after replacement moves:

| state | \(\sigma_1/\sigma_2\) | top-1 energy fraction |
| --- | ---: | ---: |
| clean | 1.148 | 0.4038 |
| `steepest_replace_step5` | 5.094 | 0.9245 |
| `steepest_replace_step10` | 3.984 | 0.8752 |
| `steepest_replace_final` | 3.955 | 0.8751 |

This supports a local high-gain ridge/mode story.

However, the top residual-Jacobian right singular vector is not strongly aligned
with the final replacement delta:

- at `steepest_replace_final`, cosine with `steepest_replace_final` is only
  \(0.281\);
- at `steepest_replace_step5`, cosine with final replacement delta is \(0.193\)
  and with step5 delta is \(0.125\).

Therefore the evidence does not justify saying that the successful optimizer is
literally following the top singular vector of the residual Jacobian.

### Objective-Gradient Replacement vs JVP/VJP Power Variants

In the p2q2 eps8/alpha0.3 probe:

| method family | final mean Loss3 |
| --- | ---: |
| objective-gradient replacement | 6.805 |
| objective-gradient additive | 6.377 |
| raw-gradient additive | 5.390 |
| affine JVP/VJP replacement | 3.842 |
| pure JVP/VJP replacement | 2.324 |

This is important. It means the successful method should be interpreted as
objective-gradient replacement, not as a literal generalized JVP/VJP power
method for the full nonlinear Loss3 objective.

## Current Best Burgers Explanation

The best-supported explanation is:

\[
\boxed{
\text{Burgers replacement works because the path quickly enters a high-gain,
relatively stable, broad high-loss boundary corridor.}
}
\]

More explicitly:

\[
\text{fast full-budget boundary arrival}
+
\text{early final-like direction}
+
\text{broad high-loss boundary ridge}
+
\text{lower effective curvature / more stable outer region}
+
\text{post-boundary direction rotation}
\]

combine to make `replace` fast.

This is consistent with the theory because those conditions make the
full-budget linearized update less dangerous. It is not a proof of a global
fixed quadratic model.

## NS2D Working Hypothesis

The working contrast with NS2D is:

\[
\boxed{
\text{NS2D may be more nonlinear, more path-dependent, and have a narrower
high-loss region, so } \texttt{add} \text{ can outperform } \texttt{replace}.
}
\]

In that regime, `replace` can still reach the perturbation boundary quickly, but
the full-budget direction may be a poor global direction because the local
linearized model is less reliable.

Potential NS2D failure modes for `replace`:

- \(A_k\), \(b_k\), or \(g_k\) rotate strongly;
- the high-loss boundary region is narrow;
- local linear prediction is poor over full \(\epsilon\)-scale jumps;
- surrogate gradient and true Loss3 evaluation are less aligned;
- path history matters, so resetting the perturbation direction loses useful
  accumulated information.

This is why formal NS2D curves can favor `steepest_add` even though a pure
quadratic theory favors `replace`.

## Darcy Flow Working Hypothesis

Darcy Flow is a binary coefficient-flip problem, so continuous-vector geometry
does not directly apply. The analogous questions are:

- are the high-score flip rankings stable across steps?
- do many near-optimal flip sets exist?
- does replacement choose a high-quality flip set immediately?
- do additive/prefix flip sets need path accumulation?

For Darcy, "broad high-loss ridge" should be translated into:

\[
\text{many different flip sets produce similarly high Loss3}.
\]

Direction stability should be translated into:

\[
\text{top-ranked flip scores and selected flip sets have high overlap across
steps}.
\]

## Experiments Needed To Strengthen Or Falsify This Explanation

### Experiment 1: Full-Budget Local Prediction Accuracy

Purpose: test whether the local first-order model predicts the true gain of
`replace` and `add`.

At saved steps

\[
k\in\{0,1,2,5,10,20,50,100\}
\]

and for Burgers also

\[
k\in\{150,200,300\},
\]

record \(g_k\) and evaluate candidate moves:

\[
\delta_{k+1}^{rep}=\epsilon s_k,
\]

\[
\delta_{k+1}^{add}=\Pi(\delta_k+\alpha s_k),
\]

\[
\delta^{final},
\qquad
\delta^{random\ boundary}.
\]

For each candidate, compare predicted gain

\[
\widehat{\Delta L}=g_k^\top(\delta_{\text{candidate}}-\delta_k)
\]

with true gain

\[
\Delta L=L(x+\delta_{\text{candidate}})-L(x+\delta_k).
\]

Metrics:

\[
\mathrm{corr}(\widehat{\Delta L},\Delta L),
\]

\[
\frac{|\widehat{\Delta L}-\Delta L|}{|\Delta L|+\eta},
\]

and whether the candidate ranked best by the local model is also good under true
Loss3.

Expected support for the current explanation:

- Burgers should have better full-budget prediction accuracy than NS2D,
  especially after the early transition.
- NS2D should have worse full-budget prediction accuracy for `replace` jumps.

### Experiment 2: Boundary High-Loss Ridge Width

Purpose: test whether high Loss3 occupies a broad boundary region.

For final directions \(\delta_a,\delta_b\), evaluate boundary arcs:

\[
\delta(\theta)
=
\epsilon\,
\frac{(1-\theta)\delta_a+\theta\delta_b}
{\|(1-\theta)\delta_a+\theta\delta_b\|}.
\]

Also sample tangent perturbations around a final direction:

\[
\delta_{\tan}(\rho,u)
=
\epsilon\,
\frac{\delta^\star+\rho u_\perp}{\|\delta^\star+\rho u_\perp\|}.
\]

Metrics:

\[
\Pr[L(\delta_{\tan})\ge 0.95L(\delta^\star)],
\]

\[
\Pr[L(\delta_{\tan})\ge 0.90L(\delta^\star)],
\]

and arc dip:

\[
\min_\theta
\frac{L(\delta(\theta))}
{\min(L(\delta_a),L(\delta_b))}.
\]

Expected support:

- Burgers should have larger high-loss boundary volume and smaller arc dips.
- NS2D should have narrower high-loss regions if this part of the hypothesis is
  correct.

### Experiment 3: Direction And Jacobian Stability

Purpose: test whether Burgers is more stable locally than NS2D along relevant
attack paths.

Measure along optimizer paths and straight-line paths:

\[
x_k=x+\delta_k,
\qquad
x(t)=x+t\delta^\star.
\]

Metrics:

\[
\cos(g_k,g_{k-1}),
\]

\[
\cos(\delta_k,\delta_{\text{final}}),
\]

\[
\angle(v_1(J_e(x_k)),v_1(J_e(x_{k-1}))),
\]

\[
\angle(\mathrm{span}_r(J_e(x_k)),\mathrm{span}_r(J_e(x_{k-1}))),
\]

\[
\frac{\sigma_1}{\sigma_2}.
\]

Expected support:

- Burgers: early change, then stable outer corridor.
- NS2D: stronger or more persistent direction rotation, if the nonlinear
  path-dependent explanation is correct.

### Experiment 4: Objective-Gradient Replace vs True JVP/VJP Power

Purpose: avoid falsely claiming a literal power-method theorem.

Compare:

- objective-gradient replacement;
- objective-gradient additive;
- affine JVP/VJP power replacement;
- pure JVP/VJP power replacement.

Expected support:

- If objective-gradient replacement still beats JVP/VJP variants, the mechanism
  is a practical loss-gradient replacement surrogate, not a literal residual
  Jacobian top-singular-vector power method.

### Experiment 5: Epsilon / Alpha Sweep

Purpose: test whether replace advantage shrinks as the nonlinear jump scale
gets larger.

Burgers:

\[
\epsilon\in\{1,2,4,8,16\}.
\]

NS2D:

\[
\epsilon\in\{1,4,8,16,32\}.
\]

Compare at least `steepest_add` and `steepest_replace`.

Expected patterns:

- If small \(\epsilon\) makes `replace` good but large \(\epsilon\) makes it
  worse, the main issue is local linearity radius.
- If Burgers keeps `replace` strong even at \(\epsilon=8\), then broad boundary
  ridge and direction stability are likely helping beyond pure local linearity.
- If NS2D `replace` improves at small \(\epsilon\) but fails at large
  \(\epsilon=32\), that strongly supports the nonlinear full-budget error
  explanation.

## Minimal Next Run Package

The minimal useful mechanism package should be:

| Experiment | Burgers | Darcy Flow | NS2D |
| --- | ---: | ---: | ---: |
| local prediction accuracy | \(N=20\) | \(N=20\) | \(N=5\) |
| boundary ridge width / flip-set width | \(N=20\) | \(N=20\) | \(N=5\) |
| direction / Jacobian stability | \(N=5\) | \(N=5\) | \(N=3\) to \(5\) |

This should produce a final verdict table:

| System | local prediction accurate? | boundary ridge broad? | direction stable? | theory predicts replace advantage? | actual curve agrees? |
| --- | --- | --- | --- | --- | --- |
| Burgers | TBD | TBD | TBD | TBD | yes |
| Darcy Flow | TBD | TBD | TBD | TBD | TBD |
| NS2D | TBD | TBD | TBD | TBD | yes/no by formal curve |

## Statements That Are Safe To Use

Safe:

\[
\texttt{replace} \text{ is not universally better; it is favored when the
full-budget linearized step is reliable.}
\]

Safe:

\[
\text{Burgers appears to satisfy several empirical proxies for this reliability:
early final-like directions, broad high-loss boundary regions, lower outer
curvature, and post-boundary direction improvement.}
\]

Safe:

\[
\text{NS2D may violate these conditions through stronger nonlinearity,
direction rotation, path dependence, or narrower high-loss regions.}
\]

Not safe:

\[
\texttt{replace} \text{ is always better than } \texttt{add}.
\]

Not safe:

\[
\text{Burgers proves the optimizer is a true global generalized power method.}
\]

Not safe:

\[
\text{The residual map is globally linear in Burgers.}
\]

## One-Sentence Paper Version

The concise paper-style version is:

> Replacement updates are theoretically advantageous in stable linear or
> quadratic regimes because they solve the full-budget linearized subproblem and
> align rapidly with dominant high-gain directions. Their advantage is not
> universal: in nonlinear or path-dependent landscapes the full-budget
> linearized step can have large approximation error. Burgers appears to be in a
> replace-friendly regime, with early final-like directions, broad high-loss
> boundary ridges, lower effective curvature in the outer region, and useful
> post-boundary direction rotation, whereas NS2D may be more nonlinear and
> narrow-ridged, favoring additive path-following updates.
