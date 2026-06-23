# Loss3 Replace/Add Linearity Validation Experiment Plan

Updated: 2026-06-22 UTC

This document records the concrete experiment plan for validating the current
replace-vs-add mechanism hypothesis.

## Current Run Status

Started on 2026-06-22 UTC:

```text
tools/run_loss3_replace_add_validation_queue_20260622.sh
```

Queue log:

```text
analysis_outputs/mechanism_20260622/replace_add_validation/run_logs/replace_add_validation_queue_20260622.log
```

The queue waits for the currently running NS2D mechanism trace job, then runs:

1. Burgers \(N=5\), `eps=8`, `alpha=0.3`, `steps=100`, four core optimizers,
   with saved trajectories.
2. Burgers first-order full-budget prediction probe:
   `tools/probe_burgers_first_order_prediction_20260622.py`.
3. Burgers landscape/ridge probe:
   `tools/run_loss3_core4_pq_landscape_probe.py`.
4. NS2D direction-stability diagnostics on the completed mechanism trace:
   `tools/analyze_step_sample_direction_stability_20260622.py`.

Core hypothesis:

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

The goal is not only to show which optimizer wins, but to measure whether the
mechanism indicators predict the observed optimizer behavior.

## Systems And Methods

Systems:

- Burgers 1D
- Darcy Flow
- NS2D recurrent

Methods:

- `raw_add`
- `raw_replace`
- `steepest_add`
- `steepest_replace`

Main formal comparison metric:

\[
L_3(x+\delta)=\|F_\theta(x+\delta)-G(x+\delta)\|.
\]

Main performance summaries:

\[
\text{final Loss3},
\qquad
\text{AUC/mean curve height},
\qquad
k_{50},
\qquad
k_{90},
\qquad
k_{95}.
\]

Useful replace advantage scores:

\[
\text{speed advantage}
=
\frac{k_{90}^{add}}{k_{90}^{replace}},
\]

\[
\text{AUC advantage}
=
\frac{\mathrm{AUC}_{replace}}{\mathrm{AUC}_{add}},
\]

\[
\text{final advantage}
=
\frac{L_{T}^{replace}}{L_{T}^{add}}.
\]

For Burgers and NS2D, use continuous-vector diagnostics. For Darcy Flow, use
binary flip-set analogues: flip ranking stability, prefix gain prediction,
selected-set overlap, and near-optimal flip-set volume.

## Experiment 1: First-Order Full-Budget Prediction Accuracy

Purpose:

Test whether the current gradient's first-order model predicts true Loss3 gain,
especially for full-budget `replace` jumps.

At saved step \(k\), current point is \(\delta_k\) and

\[
g_k=\nabla_\delta \Phi(\delta_k).
\]

For any candidate move \(\delta_{\mathrm{cand}}\), the predicted gain is

\[
\widehat{\Delta\Phi}
=
g_k^\top(\delta_{\mathrm{cand}}-\delta_k),
\]

and the true gain is

\[
\Delta\Phi
=
\Phi(\delta_{\mathrm{cand}})-\Phi(\delta_k).
\]

Candidate moves:

\[
\delta_{k+1}^{rep},
\qquad
\delta_{k+1}^{add},
\qquad
\delta^{final},
\qquad
\delta^{random\ boundary}.
\]

For Burgers and NS2D:

\[
\delta_{k+1}^{rep}=\epsilon s_k,
\]

\[
\delta_{k+1}^{add}=\Pi_{\mathcal B_p(\epsilon)}(\delta_k+\alpha s_k).
\]

Metrics:

\[
\mathrm{corr}(\widehat{\Delta\Phi},\Delta\Phi),
\]

\[
\frac{|\widehat{\Delta\Phi}-\Delta\Phi|}
{|\Delta\Phi|+\eta},
\]

\[
\mathrm{rankcorr}(\widehat{\Delta\Phi},\Delta\Phi).
\]

Interpretation:

- If Burgers has high correlation and low error for full-budget moves, it
  supports the claim that Burgers is replace-friendly because its full-budget
  local model is reliable.
- If NS2D has low correlation or high error for full-budget moves, it supports
  the claim that NS2D is too nonlinear/path-dependent for aggressive
  replacement.
- If Burgers full-budget prediction is not very accurate but replacement still
  works, then boundary ridge width is probably the stronger explanation.

Suggested saved steps:

\[
k\in\{0,1,2,5,10,20,50,100\}
\]

and for Burgers 300-step curves:

\[
k\in\{150,200,300\}.
\]

## Experiment 2: Multi-Scale Local Linearity

Purpose:

Measure whether scalar Loss3 and the residual map are locally linear, and at
what radius. This directly tests whether `replace` can safely trust a large
linearized step.

Residual map:

\[
r(\delta)=F_\theta(x+\delta)-G(x+\delta).
\]

For a direction \(u\) and radius \(\rho\), define residual-map linearity score:

\[
C_r(\rho,u)
=
\frac{
\|r(\delta+\rho u)-2r(\delta)+r(\delta-\rho u)\|
}{
\|r(\delta+\rho u)-r(\delta-\rho u)\|+\eta
}.
\]

Define scalar Loss3 linearity score:

\[
C_\Phi(\rho,u)
=
\frac{
|\Phi(\delta+\rho u)-2\Phi(\delta)+\Phi(\delta-\rho u)|
}{
|\Phi(\delta+\rho u)-\Phi(\delta-\rho u)|+\eta
}.
\]

Smaller means more locally linear.

Directions:

- gradient direction;
- radial direction;
- optimizer step direction;
- random tangent directions;
- random full-space directions;
- final-delta direction.

Radii:

\[
\rho/\epsilon\in\{0.01,0.03,0.1,0.3,1.0\}.
\]

Important:

Very small \(\rho\) is not enough. `replace` makes an \(\epsilon\)-scale jump,
so the key question is whether linearity persists at medium and large fractions
of the perturbation budget.

Interpretation:

- Burgers should show lower \(C_\Phi\) and possibly lower directional curvature
  in the outer attack region.
- NS2D should show larger \(C_\Phi\), stronger radius dependence, or stronger
  direction dependence if the nonlinear explanation is correct.

## Experiment 3: Quadratic Surrogate Accuracy

Purpose:

Test whether the residual map is linear enough that squared Loss3 is well
approximated by a local quadratic model.

At point \(\delta\), approximate:

\[
r(\delta+s)\approx r(\delta)+J_\delta s.
\]

Then the local quadratic surrogate is:

\[
\Phi_{\mathrm{quad}}(s)
=
\frac12\|r(\delta)+J_\delta s\|^2.
\]

The true value is:

\[
\Phi_{\mathrm{true}}(s)
=
\frac12\|r(\delta+s)\|^2.
\]

Evaluate candidate \(s\)'s:

\[
s_{rep}=\delta_{k+1}^{rep}-\delta_k,
\]

\[
s_{add}=\delta_{k+1}^{add}-\delta_k,
\]

\[
s_{final}=\delta^{final}-\delta_k,
\]

plus random boundary/tangent moves.

Metrics:

\[
\frac{|\Phi_{\mathrm{quad}}(s)-\Phi_{\mathrm{true}}(s)|}
{|\Phi_{\mathrm{true}}(s)|+\eta},
\]

\[
\mathrm{corr}(\Phi_{\mathrm{quad}}(s),\Phi_{\mathrm{true}}(s)),
\]

\[
\mathrm{rankcorr}(\Phi_{\mathrm{quad}}(s),\Phi_{\mathrm{true}}(s)).
\]

Interpretation:

- If Burgers quadratic surrogate ranks candidate directions correctly and NS2D
  does not, then Burgers is closer to the frozen linear/quadratic theory.
- If the quadratic surrogate fails in Burgers but replacement still succeeds,
  then the successful explanation should emphasize boundary ridge width and
  objective-gradient surrogate behavior rather than strict quadratic theory.

## Experiment 4: Direction Field Stability

Purpose:

Test whether `replace` is chasing a stable direction or a moving target.

Measure along optimizer paths:

\[
\cos(g_k,g_{k-1}),
\]

\[
\cos(s_k,s_{k-1}),
\]

\[
\cos(\delta_k,\delta_T),
\]

where \(\delta_T\) is the final perturbation.

For Jacobian/subspace diagnostics, use:

\[
J_e(x)=J_{F_\theta}(x)-J_G(x).
\]

Measure:

\[
\angle(v_1(J_e(x_k)),v_1(J_e(x_{k-1}))),
\]

\[
\angle(\mathrm{span}_m(J_e(x_k)),\mathrm{span}_m(J_e(x_{k-1}))),
\]

\[
\frac{\sigma_1}{\sigma_2}.
\]

Also measure along straight-line paths:

\[
x(t)=x_0+t\delta^\star,
\qquad t\in[0,1].
\]

Interpretation:

- Burgers should show early change followed by a more stable outer corridor.
- NS2D should show stronger or more persistent direction rotation if the
  nonlinear/path-dependent hypothesis is correct.

## Experiment 5: Power-Method Condition Check

Purpose:

Distinguish true power-method behavior from practical objective-gradient
replacement behavior.

In a real frozen quadratic/power-method regime, we should see:

\[
\lambda_1\gg\lambda_2,
\]

and replacement directions should rapidly align with the top direction:

\[
\cos(\delta_k,v_1)\to 1.
\]

Measure:

\[
\frac{\lambda_1}{\lambda_2}
\quad\text{or}\quad
\frac{\sigma_1}{\sigma_2},
\]

\[
\cos(\delta_k,v_1),
\]

\[
\cos(s_k,v_1),
\]

and compare objective-gradient replacement with explicit JVP/VJP power variants:

- objective-gradient replacement;
- affine JVP/VJP replacement;
- pure JVP/VJP replacement.

Interpretation:

- If objective-gradient replacement wins but JVP/VJP variants fail, the method
  should not be described as a literal generalized power method.
- If both objective-gradient replacement and JVP/VJP variants win, the power
  explanation is stronger.

## Experiment 6: Boundary High-Loss Region Width

Purpose:

Test whether `replace` is robust to direction error because the boundary
high-loss region is broad.

On the boundary:

\[
\delta=\epsilon u,\qquad \|u\|=1.
\]

Around a final direction \(u^\star\), sample tangent perturbations:

\[
u'=\mathrm{normalize}(u^\star+\rho v_\perp),
\]

\[
\delta'=\epsilon u'.
\]

Measure:

\[
\Pr[\Phi(\epsilon u')\ge 0.95\Phi(\epsilon u^\star)],
\]

\[
\Pr[\Phi(\epsilon u')\ge 0.90\Phi(\epsilon u^\star)].
\]

Also measure arcs between method final directions:

\[
\delta(\theta)
=
\epsilon
\frac{(1-\theta)\delta_a+\theta\delta_b}
{\|(1-\theta)\delta_a+\theta\delta_b\|}.
\]

Arc metric:

\[
\min_\theta
\frac{\Phi(\delta(\theta))}
{\min(\Phi(\delta_a),\Phi(\delta_b))}.
\]

Interpretation:

- Burgers broad high-loss boundary supports replacement even if the local model
  is imperfect.
- NS2D narrow peaks would explain why replacement can be fast but not best.

For Darcy Flow, replace boundary-width with flip-set width:

\[
\Pr[\Phi(S')\ge 0.95\Phi(S^\star)]
\]

where \(S'\) is a perturbed flip set with controlled overlap to the best set
\(S^\star\).

## Experiment 7: Cross-System Correlation Analysis

Purpose:

The final validation is not a single diagnostic; it is whether mechanism
metrics predict optimizer advantage across systems/settings.

Build a summary table:

| System | setting | linearity score | quadratic error | direction stability | ridge width | replace advantage |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Burgers | eps/alpha/steps | TBD | TBD | TBD | TBD | TBD |
| Darcy Flow | flips/alpha/steps | TBD | TBD | TBD | TBD | TBD |
| NS2D | eps/alpha/steps | TBD | TBD | TBD | TBD | TBD |

Expected trend:

\[
\text{replace advantage}\uparrow
\quad\text{when}\quad
C_\Phi\downarrow,
\]

\[
\text{replace advantage}\uparrow
\quad\text{when}\quad
\text{first-order prediction error}\downarrow,
\]

\[
\text{replace advantage}\uparrow
\quad\text{when}\quad
\cos(g_k,g_{k-1})\uparrow,
\]

\[
\text{replace advantage}\uparrow
\quad\text{when}\quad
\text{boundary ridge width}\uparrow.
\]

This is the strongest final test of the current hypothesis.

## Minimal Recommended Run

Do not start with the full expensive package. The minimum useful validation is:

| Experiment | Burgers | Darcy Flow | NS2D |
| --- | ---: | ---: | ---: |
| first-order prediction accuracy | \(N=20\) | \(N=20\) | \(N=5\) |
| multi-scale linearity | \(N=10\) | optional flip analogue | \(N=3\) to \(5\) |
| quadratic surrogate accuracy | \(N=5\) | optional flip analogue | \(N=3\) |
| boundary ridge / flip-set width | \(N=20\) | \(N=20\) | \(N=5\) |
| direction stability | \(N=5\) | \(N=5\) flip overlap | \(N=3\) to \(5\) |

Suggested priority:

1. first-order prediction accuracy;
2. multi-scale linearity and quadratic surrogate accuracy;
3. boundary ridge width;
4. direction/Jacobian stability;
5. correlation of mechanism metrics with replace advantage.

## Expected Conclusions To Test

The current hypothesis predicts:

| System | Expected mechanism | Expected optimizer behavior |
| --- | --- | --- |
| Burgers | relatively strong local linear/quadratic approximation after early transition; stable outer direction; broad high-loss boundary | `replace` very fast; `steepest_add` may eventually catch up |
| Darcy Flow | ranking/flip-set stability and many near-optimal flip sets may make replacement-like updates good | replacement-like methods may be fast if flip rankings are stable |
| NS2D | stronger nonlinearity, direction rotation, path dependence, or narrow high-loss region | `steepest_add` can beat `replace` despite slower conservative updates |

If these predictions fail, revise the mechanism. In particular:

- If Burgers is not more linear/stable than NS2D but replacement still wins, the
  main explanation should shift toward boundary ridge width or surrogate/metric
  alignment.
- If NS2D is linear/stable but replacement loses, the main explanation should
  shift toward narrow high-loss regions or path-history dependence.
- If neither mechanism metric correlates with replace advantage, the current
  theory is insufficient and a different mechanism is needed.
