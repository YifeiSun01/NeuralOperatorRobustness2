# Analytic-Solution Hierarchy for Delta Objectives

Date: 2026-05-16

This note records the distinction between three optimization levels that look similar
but should not be mixed:

1. pure local residual movement;
2. local affine endpoint error;
3. true nonlinear finite-radius attack.

Throughout this note, assume the local notation

\[
b=e(x)=f(x)-j(x)
\]

and

\[
A=J_f(x)-J_j(x).
\]

For small perturbations around the clean point \(x\), the residual field has the local
affine approximation

\[
e(x+\delta)=f(x+\delta)-j(x+\delta)\approx b+A\delta.
\]

The important caveat is that this approximation is a local model. It is appropriate
for infinitesimal or small-radius diagnostics, but it should not be treated as the
exact finite-radius attack objective when \(\delta\) is not small.

## 1. Pure Local Residual Movement

The pure residual-movement objective is

\[
\max_{\|\delta\|_2\le \epsilon}\|A\delta\|_2^2.
\]

Expanding gives

\[
\|A\delta\|_2^2=\delta^TA^TA\delta.
\]

Let

\[
Q=A^TA.
\]

Then this is a Rayleigh-quotient problem over the \(L_2\) ball. Its simple analytic
solution is

\[
\delta^*=\epsilon v_1,
\]

where \(v_1\) is a top eigenvector of \(Q=A^TA\), equivalently the top right singular
vector of \(A\). If the largest singular value has multiplicity greater than one, any
unit vector in the top right-singular subspace is also optimal.

Meaning:

- This direction maximizes how much the error field moves:

\[
\|e(x+\delta)-e(x)\|_2 \approx \|A\delta\|_2.
\]

- It does **not** necessarily maximize the final current-error norm

\[
\|e(x+\delta)\|_2.
\]

- It is the correct analytic direction for a local residual-Lipschitz / mismatch-gain
diagnostic, not automatically the final finite-radius attack direction.

## 2. Local Affine Endpoint Error

The local affine endpoint objective is

\[
\max_{\|\delta\|_2\le \epsilon}\|b+A\delta\|_2^2.
\]

Expanding gives

\[
\|b+A\delta\|_2^2
=\|b\|_2^2+2b^TA\delta+\delta^TA^TA\delta.
\]

Using

\[
Q=A^TA,
\qquad
c=A^Tb,
\]

this becomes

\[
\max_{\|\delta\|_2\le \epsilon}
\delta^TQ\delta+2c^T\delta.
\]

The gradient of the squared local endpoint objective is

\[
\nabla_\delta \|b+A\delta\|_2^2
=
2A^Tb+2A^TA\delta.
\]

So, compared with pure residual movement,

\[
\nabla_\delta \|A\delta\|_2^2
=
2A^TA\delta,
\]

there is an additional clean-residual term

\[
2A^Tb.
\]

A boundary KKT stationary point satisfies

\[
A^TA\delta + A^Tb = \mu \delta.
\]

Equivalently,

\[
(Q-\mu I)\delta + c=0.
\]

When written in secular-equation form, one can express candidate solutions as

\[
\delta(\mu)=(\mu I-Q)^{-1}c,
\]

with \(\mu\) chosen so that

\[
\|\delta(\mu)\|_2=\epsilon.
\]

For the usual global maximizer branch of this trust-region-type problem, \(\mu\) is
chosen outside the spectrum of \(Q\), typically

\[
\mu>\lambda_{\max}(Q),
\]

with the standard caveat that special cases such as \(c=0\) or eigenvalue degeneracy
can reduce to the top-eigenspace solution.

Meaning:

- This objective has a KKT / implicit analytic characterization if the full matrix
  \(A\) is available.
- It is **not** just the top right singular vector of \(A\).
- It depends on both the quadratic movement term \(A^TA\delta\) and the clean-residual
  outward term \(A^Tb\).
- For very small \(\epsilon\), the linear term dominates, so the direction approaches

\[
\frac{A^Tb}{\|A^Tb\|_2}
\]

when \(A^Tb\neq0\).

- For larger \(\epsilon\), the quadratic term can bend the solution toward the dominant
  singular-vector structure of \(A\).

This is the local-affine version of endpoint error. It is more endpoint-aware than
pure residual movement, but it still assumes the clean-point approximation

\[
e(x+\delta)\approx b+A\delta.
\]

## 3. True Nonlinear Finite-Radius Attack

The true nonlinear regression attack objective is

\[
\max_{\|\delta\|_2\le \epsilon}
\|f(x+\delta)-j(x+\delta)\|_2.
\]

In general this objective has no closed-form analytic solution. The reason is that
both \(f\) and \(j\) can change nonlinearly along the path from \(x\) to
\(x+\delta\). The clean-point matrix \(A=J_f(x)-J_j(x)\) only describes the first local
linearization at \(x\); it does not describe the whole finite-radius path.

For a finite saved point

\[
z_k=x+\delta_k,
\]

the fair gradient comparison should directly differentiate the true nonlinear losses
at \(z_k\), for example

\[
L_{endpoint}(z_k)=\|f(z_k)-j(z_k)\|_2
\]

and

\[
L_{movement}(z_k)=\|(f(z_k)-j(z_k))-(f(x)-j(x))\|_2.
\]

Then compare

\[
\nabla_{z_k}L_{endpoint}(z_k)
\quad \text{versus} \quad
\nabla_{z_k}L_{movement}(z_k).
\]

This direct nonlinear-gradient comparison does not require pretending that
\(\delta_k\) is infinitesimal. It is still local at the current finite point
\(z_k\), but it uses the actual model and solver there instead of the fixed clean-point
matrix \(A\).

Meaning:

- The true finite-radius attack should be optimized iteratively, for example with PGD
  or LP-steepest PGD.
- The evidence should be based on actual trajectories, final deltas, final losses,
  and same-point nonlinear gradients.
- A top singular vector of \(A\) is a local diagnostic direction, not a general
  analytic solution to the nonlinear attack.
- The KKT solution of \(\|b+A\delta\|^2\) is a local-affine finite-radius diagnostic,
  not a general analytic solution to the nonlinear attack.

## Summary Rule

The three levels should be kept separate:

\[
\max_{\|\delta\|\le\epsilon}\|A\delta\|^2
\quad\Rightarrow\quad
\text{simple SVD direction: top right singular vector of } A.
\]

\[
\max_{\|\delta\|\le\epsilon}\|b+A\delta\|^2
\quad\Rightarrow\quad
\text{KKT / implicit trust-region solution, not a simple singular vector.}
\]

\[
\max_{\|\delta\|\le\epsilon}\|f(x+\delta)-j(x+\delta)\|
\quad\Rightarrow\quad
\text{generally no analytic solution; use iterative optimization and inspect trajectories.}
\]

中文一句话：

- \(\|A\delta\|^2\) 是“误差场移动最大”，有简单 SVD 解。
- \(\|b+A\delta\|^2\) 是“local affine endpoint error 最大”，有 KKT 隐式解，受
  \(A^Tb\) 和 \(A^TA\delta\) 两项共同影响。
- \(\|f(x+\delta)-j(x+\delta)\|\) 是真实 nonlinear finite-radius attack，一般没有
  解析解，只能用实际优化轨迹和 final delta 来判断。
