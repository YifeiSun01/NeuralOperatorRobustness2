# Error-Operator Robustness: Local Lipschitz/Kernel Norm Versus PGD Epsilon-Ball Loss Growth

Date: 2026-06-08

This note records the mathematical relationship between two robustness definitions used in the Burgers/Darcy neural-operator experiments:

1. local robustness defined by the spectral norm of the Frechet derivative of the model-minus-solver error operator;
2. finite-radius robustness defined by the maximum growth of an error-energy loss inside a small input perturbation ball.

The key conclusion is:

**For an \(L^2\) input ball and \(L^2\) output error norm, the local operator norm of the Frechet derivative is the infinitesimal epsilon-ball worst-case error amplification. For the squared error-change energy, the corresponding limit is the squared operator norm, equivalently the top eigenvalue of the local kernel operator. For the practical PGD loss \(\frac12\|\mathcal E(a+h)\|^2\), the same equivalence requires the current residual first-order term to vanish or be negligible.**

## Problem Setup

Let \(X\) be the input function space and \(Y\) be the output function space. In the Burgers case, \(X\) can be viewed as the space of initial conditions and \(Y\) as the space of final-time solution functions. In Darcy Flow, \(X\) is the coefficient/input-field space and \(Y\) is the solution-field space.

Use Hilbert norms, typically \(L^2\)-type norms:

\[
\|\cdot\|_X,\qquad \|\cdot\|_Y.
\]

Let the trained neural operator be

\[
\mathcal M:X\to Y,
\]

and the numerical/PDE solver operator be

\[
\mathcal S:X\to Y.
\]

Define the model-solver error operator

\[
\mathcal E = \mathcal M-\mathcal S:X\to Y.
\]

For an input function \(a\in X\), the output

\[
\mathcal E(a)=\mathcal M(a)-\mathcal S(a)
\]

is the error function.

This is the important object in the experiments: we are not taking the Jacobian of the model alone; we are taking the local derivative of the **error operator** \(\mathcal M-\mathcal S\).

## Frechet Derivative As Continuous Jacobian

Assume \(\mathcal E\) is Frechet differentiable at \(a\). Then there exists a bounded linear operator

\[
A = D\mathcal E(a):X\to Y
\]

such that, for small function perturbations \(h\in X\),

\[
\mathcal E(a+h)=\mathcal E(a)+Ah+r(h),
\]

with

\[
\frac{\|r(h)\|_Y}{\|h\|_X}\to 0
\qquad\text{as}\qquad \|h\|_X\to0.
\]

This \(D\mathcal E(a)\) is the infinite-dimensional analogue of the finite-dimensional Jacobian matrix. It maps an input perturbation function \(h\) to the first-order change of the model-solver error function.

In a discretized computation, \(D\mathcal E(a)\) becomes the matrix

\[
J_e(a)=J_{\text{model}}(a)-J_{\text{solver}}(a).
\]

## Local Lipschitz Constant And Kernel Operator

The local \(L^2\to L^2\) Lipschitz constant of the error operator is

\[
L(a)=\|D\mathcal E(a)\|_{X\to Y}
=
\sup_{\|h\|_X=1}\|D\mathcal E(a)h\|_Y.
\]

This is the continuous operator version of the spectral norm.

Because \(X,Y\) are Hilbert spaces, \(A=D\mathcal E(a)\) has an adjoint

\[
A^*:Y\to X.
\]

Define the local kernel operator

\[
K=A^*A:X\to X.
\]

Then

\[
\|A\|_{X\to Y}^2=\|A^*A\|_{X\to X}=\|K\|_{X\to X}.
\]

If \(A\) is compact, or after finite-dimensional discretization, this can be written as

\[
L(a)^2=\lambda_{\max}(K).
\]

In finite dimensions this is exactly

\[
\sigma_{\max}(J_e)^2=\lambda_{\max}(J_e^T J_e).
\]

In infinite dimensions, one should be slightly careful: a top eigenvalue may fail to exist for a general bounded operator. The always-valid statement is the operator norm identity \(\|A\|^2=\|A^*A\|\). The eigenvalue formula is valid for compact/self-adjoint settings and for the discretized matrices used in the experiments.

## Main Proposition: Infinitesimal Error Amplification

Define the epsilon-ball worst-case error-output change:

\[
R_\epsilon(a)=
\sup_{\|h\|_X\le \epsilon}
\|\mathcal E(a+h)-\mathcal E(a)\|_Y.
\]

Then

\[
\lim_{\epsilon\to0}
\frac{R_\epsilon(a)}{\epsilon}
=
\|D\mathcal E(a)\|_{X\to Y}
=
L(a).
\]

This is the cleanest equivalence. It says that the local spectral norm is exactly the infinitesimal epsilon-ball worst-case amplification of the model-solver error function.

### Proof

Let

\[
A=D\mathcal E(a),\qquad L=\|A\|_{X\to Y}.
\]

By Frechet differentiability,

\[
\mathcal E(a+h)-\mathcal E(a)=Ah+r(h),
\]

where

\[
\|r(h)\|_Y=o(\|h\|_X).
\]

For the upper bound, fix any \(\eta>0\). For sufficiently small \(\epsilon\), whenever \(\|h\|_X\le\epsilon\),

\[
\|r(h)\|_Y\le \eta\|h\|_X.
\]

Therefore

\[
\|\mathcal E(a+h)-\mathcal E(a)\|_Y
\le
\|Ah\|_Y+\|r(h)\|_Y
\le
(L+\eta)\|h\|_X
\le
(L+\eta)\epsilon.
\]

Taking the supremum over \(\|h\|_X\le\epsilon\),

\[
R_\epsilon(a)\le (L+\eta)\epsilon.
\]

Thus

\[
\limsup_{\epsilon\to0}\frac{R_\epsilon(a)}{\epsilon}\le L.
\]

For the lower bound, by the definition of the operator norm, for any \(\eta>0\) there exists \(v_\eta\in X\) such that

\[
\|v_\eta\|_X=1,
\qquad
\|Av_\eta\|_Y\ge L-\eta.
\]

Set

\[
h=\epsilon v_\eta.
\]

Then

\[
R_\epsilon(a)
\ge
\|\mathcal E(a+\epsilon v_\eta)-\mathcal E(a)\|_Y
=
\|\epsilon Av_\eta+r(\epsilon v_\eta)\|_Y.
\]

Using the reverse triangle inequality,

\[
\|\epsilon Av_\eta+r(\epsilon v_\eta)\|_Y
\ge
\epsilon\|Av_\eta\|_Y-\|r(\epsilon v_\eta)\|_Y.
\]

Since

\[
\|r(\epsilon v_\eta)\|_Y=o(\epsilon),
\]

we get

\[
R_\epsilon(a)
\ge
\epsilon(L-\eta)-o(\epsilon).
\]

Hence

\[
\liminf_{\epsilon\to0}\frac{R_\epsilon(a)}{\epsilon}\ge L-\eta.
\]

Letting \(\eta\to0\),

\[
\liminf_{\epsilon\to0}\frac{R_\epsilon(a)}{\epsilon}\ge L.
\]

Combining the upper and lower bounds gives

\[
\lim_{\epsilon\to0}
\frac{R_\epsilon(a)}{\epsilon}=L.
\]

This proves the main equivalence.

## Squared Error-Change Energy

If we define a scalar energy using the output change,

\[
S_\epsilon(a)=
\sup_{\|h\|_X\le\epsilon}
\frac12
\|\mathcal E(a+h)-\mathcal E(a)\|_Y^2,
\]

then

\[
S_\epsilon(a)=\frac12 R_\epsilon(a)^2.
\]

Since

\[
R_\epsilon(a)=\epsilon L(a)+o(\epsilon),
\]

we have

\[
S_\epsilon(a)
=
\frac12\epsilon^2 L(a)^2+o(\epsilon^2).
\]

Equivalently,

\[
\lim_{\epsilon\to0}
\frac{2S_\epsilon(a)}{\epsilon^2}
=
L(a)^2
=
\|D\mathcal E(a)^*D\mathcal E(a)\|.
\]

In the compact/discretized setting,

\[
\lim_{\epsilon\to0}
\frac{2S_\epsilon(a)}{\epsilon^2}
=
\lambda_{\max}\left(D\mathcal E(a)^*D\mathcal E(a)\right).
\]

This is why the local kernel maximum eigenvalue is the squared infinitesimal robustness measure.

## Practical PGD Loss: Error Energy Itself

The attack loss used in the experiments is often closer to

\[
\Phi(a+h)=\frac12\|\mathcal E(a+h)\|_Y^2,
\]

not

\[
\frac12\|\mathcal E(a+h)-\mathcal E(a)\|_Y^2.
\]

Define finite-radius loss growth:

\[
\Delta_\epsilon(a)=
\sup_{\|h\|_X\le\epsilon}
\left[
\Phi(a+h)-\Phi(a)
\right].
\]

Let

\[
e_0=\mathcal E(a),
\qquad
A=D\mathcal E(a).
\]

Using

\[
\mathcal E(a+h)=e_0+Ah+r(h),
\]

we expand:

\[
\Phi(a+h)-\Phi(a)
=
\langle e_0,Ah\rangle_Y
+
\frac12\|Ah\|_Y^2
+
\langle e_0,r(h)\rangle_Y
+
\langle Ah,r(h)\rangle_Y
+
\frac12\|r(h)\|_Y^2.
\]

The first term is

\[
\langle e_0,Ah\rangle_Y
=
\langle A^*e_0,h\rangle_X.
\]

This term is first order in \(\epsilon\). If

\[
A^*e_0\ne0,
\]

then generally

\[
\Delta_\epsilon(a)
=
\epsilon\|A^*e_0\|_X+o(\epsilon).
\]

Therefore, for the practical scalar loss \(\frac12\|\mathcal E(a+h)\|^2\), the infinitesimal limit is not automatically the spectral norm. It first sees the current residual direction back-propagated through the adjoint derivative.

This is the main subtlety.

## When Practical PGD Loss Becomes Equivalent To The Local Kernel Norm

The practical error-energy loss becomes equivalent to the local kernel norm when the first-order residual term does not dominate.

The cleanest sufficient condition is

\[
\mathcal E(a)=0.
\]

Then \(e_0=0\), so

\[
\langle e_0,Ah\rangle_Y=0,
\qquad
\langle e_0,r(h)\rangle_Y=0.
\]

The expansion reduces to

\[
\Phi(a+h)-\Phi(a)
=
\frac12\|Ah\|_Y^2
+
\langle Ah,r(h)\rangle_Y
+
\frac12\|r(h)\|_Y^2.
\]

Since \(r(h)=o(\|h\|_X)\), for \(\|h\|_X\le\epsilon\), the last two terms are \(o(\epsilon^2)\). Thus

\[
\Delta_\epsilon(a)
=
\frac12
\sup_{\|h\|_X\le\epsilon}
\|Ah\|_Y^2
+
o(\epsilon^2).
\]

Because

\[
\sup_{\|h\|_X\le\epsilon}\|Ah\|_Y^2
=
\epsilon^2\|A\|_{X\to Y}^2,
\]

we get

\[
\Delta_\epsilon(a)
=
\frac12\epsilon^2\|D\mathcal E(a)\|_{X\to Y}^2
+
o(\epsilon^2).
\]

Therefore

\[
\lim_{\epsilon\to0}
\frac{2\Delta_\epsilon(a)}{\epsilon^2}
=
\|D\mathcal E(a)\|_{X\to Y}^2.
\]

In the compact/discretized setting,

\[
\lim_{\epsilon\to0}
\frac{2\Delta_\epsilon(a)}{\epsilon^2}
=
\lambda_{\max}\left(D\mathcal E(a)^*D\mathcal E(a)\right).
\]

More generally, equivalence still holds approximately if

\[
\epsilon\|A^*e_0\|_X
\ll
\epsilon^2\|A\|_{X\to Y}^2,
\]

or equivalently

\[
\|A^*e_0\|_X
\ll
\epsilon\|A\|_{X\to Y}^2,
\]

and the residual-remainder term also remains small:

\[
\sup_{\|h\|_X\le\epsilon}
|\langle e_0,r(h)\rangle_Y|
\ll
\epsilon^2\|A\|_{X\to Y}^2.
\]

This is what it means, rigorously, for the scalar loss first-order residual term not to dominate.

## Role Of The Norm: Why \(p=2\) Matters

The spectral norm and the kernel identity above are tied to Hilbert \(L^2\) geometry.

For an \(L^2\) input ball and an \(L^2\) output norm,

\[
\|A\|_{2\to2}=
\sup_{\|h\|_2=1}\|Ah\|_2
\]

is exactly the spectral/operator norm.

If the input perturbation ball is instead \(L^p\), then the infinitesimal vector-output robustness becomes

\[
\|A\|_{L^p\to L^2}.
\]

For scalar loss \(\ell\), the first-order expansion under an \(L^p\) ball is

\[
\max_{\|h\|_p\le\epsilon}
\left[\ell(a+h)-\ell(a)\right]
=
\epsilon\|\nabla\ell(a)\|_q+o(\epsilon),
\]

where \(q\) is the dual exponent:

\[
\frac1p+\frac1q=1.
\]

Thus:

\[
p=2\Rightarrow q=2,
\]

\[
p=\infty\Rightarrow q=1,
\]

\[
p=1\Rightarrow q=\infty.
\]

So if the goal is to compare PGD epsilon-ball robustness directly with ordinary SVD/kernel spectral norm, the natural choice is

\[
p=2.
\]

If the PGD attack uses \(L^\infty\), then the corresponding local operator norm is not the ordinary spectral norm; it is an induced norm such as \(\|A\|_{L^\infty\to L^2}\), or the scalar-loss dual norm \(\|\nabla\ell\|_1\). That is a different robustness geometry.

## Why The Earlier Discussion Had Several Theorems

The mathematical core is the first proposition:

\[
\lim_{\epsilon\to0}
\frac{1}{\epsilon}
\sup_{\|h\|\le\epsilon}
\|\mathcal E(a+h)-\mathcal E(a)\|
=
\|D\mathcal E(a)\|.
\]

That already proves local Lipschitz equivalence for output error amplification.

The extra statements are not independent new ideas. They separate different quantities that are easy to confuse:

1. **Output error change**: \(\|\mathcal E(a+h)-\mathcal E(a)\|\). This gives the local Lipschitz constant directly.
2. **Squared output change energy**: \(\frac12\|\mathcal E(a+h)-\mathcal E(a)\|^2\). This is just the squared version and gives \(\frac12\epsilon^2L^2\).
3. **Practical residual energy loss**: \(\frac12\|\mathcal E(a+h)\|^2\). This has an additional first-order term \(\langle \mathcal E(a),D\mathcal E(a)h\rangle\).
4. **Equivalence condition for practical PGD loss**: the first-order residual term must vanish or be negligible, for example when \(\mathcal E(a)=0\).

So the core theorem is one theorem; the rest are corollaries and caveats needed because the experiment's PGD loss is usually a scalar residual energy, not merely an output-change norm.

## Interpretation For Burgers And Darcy Experiments

For Burgers/Darcy, the relevant error operator is

\[
\mathcal E=\mathcal M-\mathcal S.
\]

The local robustness quantity is

\[
L(a)=\|D(\mathcal M-\mathcal S)(a)\|_{L^2\to L^2}.
\]

This is the local Lipschitz constant of the model-solver error operator. In finite-dimensional discretization, it is the top singular value of

\[
J_{\text{model}}(a)-J_{\text{solver}}(a).
\]

The local kernel is

\[
K(a)=D\mathcal E(a)^*D\mathcal E(a),
\]

and in the discretized/compact case

\[
L(a)^2=\lambda_{\max}(K(a)).
\]

For small \(L^2\) perturbation budget \(\epsilon\), the finite-radius worst-case output error amplification satisfies

\[
\sup_{\|h\|_2\le\epsilon}
\|\mathcal E(a+h)-\mathcal E(a)\|_2
=
\epsilon L(a)+o(\epsilon).
\]

The squared change energy satisfies

\[
\sup_{\|h\|_2\le\epsilon}
\frac12\|\mathcal E(a+h)-\mathcal E(a)\|_2^2
=
\frac12\epsilon^2L(a)^2+o(\epsilon^2).
\]

The practical PGD error-energy loss

\[
\sup_{\|h\|_2\le\epsilon}
\left[
\frac12\|\mathcal E(a+h)\|_2^2
-
\frac12\|\mathcal E(a)\|_2^2
\right]
\]

matches the same \(\frac12\epsilon^2L(a)^2\) asymptotic only when the first-order residual term

\[
D\mathcal E(a)^*\mathcal E(a)
\]

is zero or negligible at the chosen scale.

## Final Statement

A concise mathematically accurate statement is:

**For the model-solver error operator \(\mathcal E=\mathcal M-\mathcal S\), the Frechet derivative \(D\mathcal E(a)\) is the continuous analogue of the error Jacobian. Its \(L^2\to L^2\) operator norm is the local Lipschitz constant and equals the infinitesimal epsilon-ball worst-case amplification of the error function. The associated kernel \(D\mathcal E(a)^*D\mathcal E(a)\) has top eigenvalue equal to the squared local Lipschitz constant in the compact/discretized setting. A finite-radius PGD loss based on \(\frac12\|\mathcal E(a+h)\|^2\) becomes asymptotically equivalent to this kernel norm when \(p=2\), \(\epsilon\to0\), and the first-order residual term \(D\mathcal E(a)^*\mathcal E(a)\) does not dominate. Otherwise PGD loss growth remains a finite-radius scalar-loss robustness measure related to, but not identical with, the local spectral norm.**

Generated at 2026-06-08T03:20:10Z.
