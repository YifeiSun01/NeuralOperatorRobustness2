# Math and Attack Taxonomy Notes

This note records the mathematical language and attack taxonomy for the paper
**Solver-Integrated Adversarial Attacking and Training of Neural Operators**.
It is meant to prevent confusion between vector-level formulas, function-level
formulas, and operator-level formulas.

## 1. Core Distinction: Operator, Function Value, Vector

The most important notation rule is:

An **operator** or **map** does not carry a concrete input. Once we plug in an
input, we get a **function value**. Once we discretize that function value, we
get a **vector**.

### 1.1 Finite-Dimensional Language

Let:

$$
f_\theta:\mathbb{R}^n\to\mathbb{R}^m
$$

be the learned model map, and let:

$$
g:\mathbb{R}^n\to\mathbb{R}^m
$$

be the discretized solver map.

The error map is:

$$
E_\theta=f_\theta-g.
$$

At a concrete input vector \(x\), the error vector is:

$$
e_\theta(x)=E_\theta(x)=f_\theta(x)-g(x)\in\mathbb{R}^m.
$$

### 1.2 Function-Space / Operator Language

Let:

$$
F_\theta:\mathcal{X}\to\mathcal{Y}
$$

be the learned neural operator, and let:

$$
G:\mathcal{X}\to\mathcal{Y}
$$

be the numerical solver or oracle.

The error operator is:

$$
E_\theta=F_\theta-G,\qquad E_\theta:\mathcal{X}\to\mathcal{Y}.
$$

At a concrete input function \(a\in\mathcal{X}\), the error function or error
field is:

$$
e_\theta[a]=E_\theta(a)=F_\theta(a)-G(a)\in\mathcal{Y}.
$$

Therefore:

$$
E_\theta \neq e_\theta[a].
$$

\(E_\theta\) is an operator. \(e_\theta[a]\) is the value of that operator at
the input function \(a\).

### 1.3 Translation Table

| Finite-dimensional language | Function-space / operator language |
|---|---|
| input vector \(x\in\mathbb{R}^n\) | input function \(a\in\mathcal{X}\) |
| perturbation vector \(\delta\in\mathbb{R}^n\) | perturbation function \(\eta\in\mathcal{X}\) |
| output vector \(y\in\mathbb{R}^m\) | output function \(u\in\mathcal{Y}\) |
| learned map \(f_\theta\) | neural operator \(F_\theta\) |
| solver map \(g\) | PDE solver/operator \(G\) |
| error map \(E_\theta=f_\theta-g\) | error operator \(E_\theta=F_\theta-G\) |
| error vector \(e_\theta(x)=E_\theta(x)\) | error function \(e_\theta[a]=E_\theta(a)\) |
| Jacobian \(J_E(x)\) | Frechet derivative \(D E_\theta[a]\) |
| transpose \(J_E(x)^\top\) | adjoint derivative \(D E_\theta[a]^*\) |
| \(J_E(x)^\top e_\theta(x)\) | \(D E_\theta[a]^*e_\theta[a]\) |

## 2. Dot Products, Inner Products, and Dual Pairings

Finite-dimensional dot product:

$$
v^\top w=\sum_i v_i w_i.
$$

Hilbert-space inner product, for scalar fields:

$$
\langle v,w\rangle_{L^2(\Omega)}
=
\int_\Omega v(s)w(s)\,ds.
$$

For vector fields:

$$
\langle v,w\rangle_{L^2(\Omega)}
=
\int_\Omega v(s)\cdot w(s)\,ds.
$$

In a Banach space, one should not assume that every derivative can be written
as a gradient. The derivative is naturally an element of the dual space:

$$
D\Phi[a]\in\mathcal{X}^*.
$$

It acts on a perturbation \(\eta\in\mathcal{X}\) as:

$$
D\Phi[a](\eta)
=
\langle D\Phi[a],\eta\rangle_{\mathcal{X}^*,\mathcal{X}}.
$$

Only in a Hilbert space can we use the Riesz representation to write:

$$
D\Phi[a](\eta)
=
\langle \nabla\Phi[a],\eta\rangle_{\mathcal{X}}.
$$

## 3. Matrix Multiplication and Kernel Operators

Finite-dimensional Jacobian action:

$$
J_E(x)\delta.
$$

Function-space linearized operator action:

$$
D E_\theta[a](\eta).
$$

Here:

$$
D E_\theta[a]:\mathcal{X}\to\mathcal{Y}
$$

is a linear operator. It is the function-space analogue of the Jacobian
matrix.

If \(D E_\theta[a]\) has a kernel representation, then:

$$
D E_\theta[a](\eta)(r)
=
\int_\Omega K_a(r,s)\eta(s)\,ds.
$$

This is the continuous analogue of:

$$
(J_E\delta)_i
=
\sum_j J_{ij}\delta_j.
$$

Thus:

$$
J_{ij}
\quad\leftrightarrow\quad
K_a(r,s).
$$

The adjoint operator satisfies:

$$
\langle A\eta,z\rangle_{\mathcal{Y}}
=
\langle \eta,A^*z\rangle_{\mathcal{X}}.
$$

If:

$$
(A\eta)(r)
=
\int K(r,s)\eta(s)\,ds,
$$

then:

$$
(A^*z)(s)
=
\int K(r,s)^\top z(r)\,dr.
$$

Therefore, the function-space version of \(J_E^\top e\) is:

$$
D E_\theta[a]^*e_\theta[a].
$$

In kernel form:

$$
\big(D E_\theta[a]^*e_\theta[a]\big)(s)
=
\int K_a(r,s)^\top e_\theta[a](r)\,dr.
$$

## 4. Generalization Metrics

Generalization is static: the input is fixed. It measures how close the neural
operator is to the solver at the same input.

### 4.1 Vector Version

MSE:

$$
\mathrm{MSE}(x)
=
\frac{1}{m}\|f_\theta(x)-g(x)\|_2^2.
$$

RMSE:

$$
\mathrm{RMSE}(x)
=
\sqrt{\frac{1}{m}\|f_\theta(x)-g(x)\|_2^2}.
$$

Relative \(L_2\):

$$
\mathrm{RelL2}(x)
=
\frac{\|f_\theta(x)-g(x)\|_2}{\|g(x)\|_2}.
$$

Max error:

$$
\|f_\theta(x)-g(x)\|_\infty.
$$

### 4.2 Function-Space Version

MSE:

$$
\mathrm{MSE}(a)
=
\frac{1}{|\Omega|}
\int_\Omega |F_\theta(a)(s)-G(a)(s)|^2\,ds.
$$

RMSE:

$$
\mathrm{RMSE}(a)
=
\left(
\frac{1}{|\Omega|}
\int_\Omega |F_\theta(a)(s)-G(a)(s)|^2\,ds
\right)^{1/2}.
$$

Relative \(L^2\):

$$
\mathrm{RelL2}(a)
=
\frac{
\|F_\theta(a)-G(a)\|_{L^2(\Omega)}
}{
\|G(a)\|_{L^2(\Omega)}
}.
$$

\(L^\infty\) error:

$$
\|F_\theta(a)-G(a)\|_{L^\infty(\Omega)}
=
\operatorname*{ess\,sup}_{s\in\Omega}
|F_\theta(a)(s)-G(a)(s)|.
$$

Cross entropy is a classification loss:

$$
\mathrm{CE}(x,y)=-\log p_\theta(y|x).
$$

It is not the natural loss for PDE regression unless the output is converted
into a class or probability distribution.

## 5. Robustness Metrics

Robustness is local and dynamic. It studies how the solver-model discrepancy
behaves inside a perturbation ball.

### 5.1 True Solver-Consistent Adversarial Robustness

Vector:

$$
R_{\mathrm{adv}}(x)
=
\max_{\|\delta\|_p\le\varepsilon}
\|e_\theta(x+\delta)\|_q.
$$

Function:

$$
R_{\mathrm{adv}}(a)
=
\sup_{\|\eta\|_{\mathcal{X},p}\le\varepsilon}
\|e_\theta[a+\eta]\|_{\mathcal{Y},q}.
$$

### 5.2 Attack Loss Increase

Vector:

$$
\Delta R(x)
=
\max_{\|\delta\|_p\le\varepsilon}
\left(
\|e_\theta(x+\delta)\|_q-\|e_\theta(x)\|_q
\right).
$$

Function:

$$
\Delta R(a)
=
\sup_{\|\eta\|_{\mathcal{X},p}\le\varepsilon}
\left(
\|e_\theta[a+\eta]\|_{\mathcal{Y},q}
-
\|e_\theta[a]\|_{\mathcal{Y},q}
\right).
$$

### 5.3 Local Operator Sensitivity

Vector:

$$
\|J_E(x)\|_{p\to q}
=
\max_{\|\delta\|_p\le1}
\|J_E(x)\delta\|_q.
$$

Function:

$$
\|D E_\theta[a]\|_{p\to q}
=
\sup_{\|\eta\|_{\mathcal{X},p}\le1}
\|D E_\theta[a](\eta)\|_{\mathcal{Y},q}.
$$

This measures the largest local error change, not necessarily the largest
final error.

### 5.4 Jacobian-Error Proxy

Vector:

$$
\|J_E(x)^\top e_\theta(x)\|_{p^*}.
$$

Function:

$$
\|D E_\theta[a]^*e_\theta[a]\|_{\mathcal{X},p^*}.
$$

For a small perturbation radius \(\varepsilon\), the first-order loss increase
for the squared Hilbert loss is:

$$
\sup_{\|\eta\|_{\mathcal{X},p}\le\varepsilon}
D\Phi[a](\eta)
=
\varepsilon
\|D E_\theta[a]^*e_\theta[a]\|_{\mathcal{X},p^*}.
$$

This proxy is often closer to the true small-\(\varepsilon\) adversarial
direction than the top singular vector, because it incorporates the current
error direction.

### 5.5 How Existing Papers Usually Define Robustness

Existing papers use several different notions of robustness, and they are not
equivalent.

#### Attack-after-loss / adversarial loss increase

Many adversarial-robustness papers define robustness by the loss after an
attack, or by the increase from clean loss to attacked loss:

$$
R_{\mathrm{attack}}(a)
=
\max_{\|\eta\|\le\varepsilon}
\mathcal{L}(F_\theta(a+\eta),\mathrm{ref}),
$$

or

$$
\Delta R_{\mathrm{attack}}(a)
=
\max_{\|\eta\|\le\varepsilon}
\mathcal{L}(F_\theta(a+\eta),\mathrm{ref})
-
\mathcal{L}(F_\theta(a),\mathrm{ref}).
$$

For classification, \(\mathrm{ref}\) is usually the fixed label \(y(a)\). For
PDE/operator regression, this becomes ambiguous because the true solution
changes when the input function changes.

#### Relative error on attacked inputs

Some neural-operator papers report clean relative error and attacked relative
error:

$$
\frac{\|F_\theta(a)-G(a)\|}{\|G(a)\|},
\qquad
\frac{\|F_\theta(a+\eta)-G(a+\eta)\|}{\|G(a+\eta)\|}.
$$

This is useful, but it is still a finite-budget empirical evaluation. It does
not by itself explain which local direction causes the error increase.

#### Local Lipschitz / spectral-norm robustness

Another line defines robustness or stability through a local amplification
constant:

$$
\|D F_\theta[a]\|_{\mathcal{X}\to\mathcal{Y}},
\qquad
\|D E_\theta[a]\|_{\mathcal{X}\to\mathcal{Y}}.
$$

In finite dimensions this is the spectral norm of a Jacobian matrix:

$$
\|J_F(x)\|_{p\to q}
\quad\text{or}\quad
\|J_E(x)\|_{p\to q}.
$$

StablePDENet is close to this view: it connects adversarial training to a
bound on the Frechet derivative and reports a Jacobian spectral norm as a
stability-related diagnostic. This measures maximum local amplification of
output or error changes.

#### The missing metric: Jacobian-error direction

The metric proposed here is different:

$$
\|D E_\theta[a]^* e_\theta[a]\|.
$$

It is not merely the largest singular value of \(D E_\theta[a]\). It is the
gradient of the current squared solver-model error:

$$
\Phi(a)=\frac12\|E_\theta(a)\|_{\mathcal{Y}}^2,
\qquad
\nabla_a\Phi(a)=D E_\theta[a]^*e_\theta[a].
$$

This quantity is more directly tied to the small-\(\varepsilon\) increase of
the true solver-consistent adversarial loss:

$$
\sup_{\|\eta\|\le\varepsilon}
\left[
\Phi(a+\eta)-\Phi(a)
\right]
=
\varepsilon\|D E_\theta[a]^*e_\theta[a]\|_*+o(\varepsilon).
$$

Based on the reviewed literature, existing papers use attacked loss,
relative error under attack, PDE residuals, or Jacobian/spectral-norm
stability metrics. They do not appear to use the solver-consistent
Jacobian-error quantity \(D E_\theta[a]^*e_\theta[a]\) as a central
robustness metric.

## 6. Loss Taxonomy: \(L_1\), \(L_2\), \(L_3\)

The three losses are a **reference taxonomy**. They say what the perturbed
model output is compared against. They are not optimization algorithms.

### 6.1 Vector Version

Model-only output change:

$$
L_1(\delta)
=
\|f_\theta(x+\delta)-f_\theta(x)\|_q.
$$

Fixed-clean-target error:

$$
L_2(\delta)
=
\|f_\theta(x+\delta)-g(x)\|_q.
$$

Solver-consistent perturbed error:

$$
L_3(\delta)
=
\|f_\theta(x+\delta)-g(x+\delta)\|_q
=
\|e_\theta(x+\delta)\|_q.
$$

### 6.2 Function-Space Version

Model-only output change:

$$
L_1(\eta)
=
\|F_\theta(a+\eta)-F_\theta(a)\|_{\mathcal{Y},q}.
$$

Fixed-clean-target error:

$$
L_2(\eta)
=
\|F_\theta(a+\eta)-G(a)\|_{\mathcal{Y},q}.
$$

Solver-consistent perturbed error:

$$
L_3(\eta)
=
\|F_\theta(a+\eta)-G(a+\eta)\|_{\mathcal{Y},q}
=
\|e_\theta[a+\eta]\|_{\mathcal{Y},q}.
$$

### 6.3 Interpretation

\(L_1\) asks how much the model moves.

\(L_2\) asks how far the perturbed model output is from the original reference.

\(L_3\) asks how far the perturbed model output is from the perturbed solver
output.

For PDE regression, \(L_3\) is the correct solver-consistent target because
the true solution generally changes when the input function changes:

$$
G(a+\eta)\neq G(a).
$$

In classification, many classical attacks assume:

$$
y(x+\delta)=y(x).
$$

That fixed-label assumption makes \(L_2\)-like and \(L_3\)-like references
collapse. In PDE regression, they do not collapse.

## 7. Squared Attack Functionals and Gradients

The following formulas assume Hilbert norms and squared losses.

### 7.1 \(L_1\): Model-Only Movement

Vector:

$$
\mathcal{A}_1(\delta)
=
\frac12
\|f_\theta(x+\delta)-f_\theta(x)\|_2^2.
$$

Gradient:

$$
\nabla_\delta \mathcal{A}_1(\delta)
=
J_f(x+\delta)^\top
\big(f_\theta(x+\delta)-f_\theta(x)\big).
$$

Function:

$$
\mathcal{A}_1(\eta)
=
\frac12
\|F_\theta(a+\eta)-F_\theta(a)\|_{\mathcal{Y}}^2.
$$

Gradient:

$$
\nabla_\eta \mathcal{A}_1(\eta)
=
D F_\theta[a+\eta]^*
\big(F_\theta(a+\eta)-F_\theta(a)\big).
$$

At \(\eta=0\), this gradient is zero. Therefore \(L_1\) attacks need a
nonzero random start.

### 7.2 \(L_2\): Fixed Clean Solver Target

Vector:

$$
\mathcal{A}_2(\delta)
=
\frac12
\|f_\theta(x+\delta)-g(x)\|_2^2.
$$

Gradient:

$$
\nabla_\delta \mathcal{A}_2(\delta)
=
J_f(x+\delta)^\top
\big(f_\theta(x+\delta)-g(x)\big).
$$

Function:

$$
\mathcal{A}_2(\eta)
=
\frac12
\|F_\theta(a+\eta)-G(a)\|_{\mathcal{Y}}^2.
$$

Gradient:

$$
\nabla_\eta \mathcal{A}_2(\eta)
=
D F_\theta[a+\eta]^*
\big(F_\theta(a+\eta)-G(a)\big).
$$

This objective uses the solver value \(G(a)\) in the forward loss, but it does
not use the derivative of the solver with respect to \(a+\eta\), because the
solver target is fixed.

### 7.3 \(L_3\): Solver-Consistent Perturbed Error

Vector:

$$
\mathcal{A}_3(\delta)
=
\frac12
\|f_\theta(x+\delta)-g(x+\delta)\|_2^2
=
\frac12\|e_\theta(x+\delta)\|_2^2.
$$

Gradient:

$$
\nabla_\delta \mathcal{A}_3(\delta)
=
J_E(x+\delta)^\top e_\theta(x+\delta).
$$

where:

$$
J_E(x+\delta)
=
J_f(x+\delta)-J_g(x+\delta).
$$

Function:

$$
\mathcal{A}_3(\eta)
=
\frac12
\|e_\theta[a+\eta]\|_{\mathcal{Y}}^2.
$$

First variation:

$$
D\mathcal{A}_3[\eta](\xi)
=
\langle e_\theta[a+\eta],
D E_\theta[a+\eta](\xi)\rangle_{\mathcal{Y}}.
$$

Gradient:

$$
\nabla_\eta \mathcal{A}_3(\eta)
=
D E_\theta[a+\eta]^*e_\theta[a+\eta].
$$

where:

$$
D E_\theta[a+\eta]
=
D F_\theta[a+\eta]-D G[a+\eta].
$$

This is the full solver-integrated backward direction.

## 8. Local Linearization and Infinitesimal Expansion

### 8.1 Linearization

Vector:

$$
e_\theta(x+\delta)
=
e_\theta(x)+J_E(x)\delta+O(\|\delta\|^2).
$$

Function:

$$
e_\theta[a+\eta]
=
e_\theta[a]+D E_\theta[a](\eta)+o(\|\eta\|_{\mathcal{X}}).
$$

Use the shorthand:

$$
b=e_\theta(x),\qquad A=J_E(x)
$$

in finite dimensions, and:

$$
b=e_\theta[a],\qquad A=D E_\theta[a]
$$

in function space.

Then the local model is:

$$
b+A\delta
\quad\text{or}\quad
b+A\eta.
$$

### 8.2 Squared Final Error Expansion

Vector:

$$
\frac12\|b+A\delta\|_2^2
=
\frac12\|b\|_2^2
+\delta^\top A^\top b
+\frac12\delta^\top A^\top A\delta.
$$

The first-order direction is:

$$
A^\top b
=
J_E(x)^\top e_\theta(x).
$$

Function:

$$
\frac12\|b+A\eta\|_{\mathcal{Y}}^2
=
\frac12\|b\|_{\mathcal{Y}}^2
+\langle A^*b,\eta\rangle_{\mathcal{X}}
+\frac12\|A\eta\|_{\mathcal{Y}}^2.
$$

The first-order direction is:

$$
A^*b
=
D E_\theta[a]^*e_\theta[a].
$$

If the full nonlinear operator is expanded to second order, there can also be:

$$
\frac12
\langle e_\theta[a],D^2E_\theta[a](\eta,\eta)\rangle_{\mathcal{Y}}.
$$

The small-\(\varepsilon\) Jacobian-error theory usually ignores this higher
order term.

## 9. SVD, Singular Functions, and Nonlinear Operators

### 9.1 Matrix SVD

For a matrix \(A:\mathbb{R}^n\to\mathbb{R}^m\):

$$
A v_i=\sigma_i u_i,
$$

$$
A^\top u_i=\sigma_i v_i,
$$

$$
A^\top A v_i=\sigma_i^2 v_i.
$$

The largest singular value is:

$$
\sigma_1
=
\|A\|_{2\to2}
=
\max_{\|\delta\|_2=1}\|A\delta\|_2.
$$

### 9.2 Linear Operators

For a compact linear operator \(A:\mathcal{X}\to\mathcal{Y}\) between Hilbert
spaces, there is a singular system:

$$
A v_i=\sigma_i w_i,
$$

$$
A^* w_i=\sigma_i v_i,
$$

$$
A^*A v_i=\sigma_i^2 v_i.
$$

Here \(v_i\) is an input singular function, \(w_i\) is an output singular
function, and \(\sigma_i\) is a singular value.

If \(A\) is not compact, a discrete singular-value expansion may not exist.
One can still discuss the induced operator norm:

$$
\|A\|_{2\to2}
=
\sup_{\|\eta\|_{\mathcal{X}}\le1}
\|A\eta\|_{\mathcal{Y}}.
$$

### 9.3 Kernel Form of \(A^*A\)

If:

$$
(A\eta)(r)
=
\int K(r,s)\eta(s)\,ds,
$$

then:

$$
(A^*A v)(s)
=
\int C(s,s')v(s')\,ds',
$$

where:

$$
C(s,s')
=
\int K(r,s)^\top K(r,s')\,dr.
$$

This is the continuous analogue of:

$$
A^\top A v=\sigma^2 v.
$$

### 9.4 Nonlinear Operators

The neural operator \(F_\theta\), solver \(G\), and error operator
\(E_\theta=F_\theta-G\) are generally nonlinear. A nonlinear operator does not
have a global SVD. We first linearize at an input function \(a\):

$$
e_\theta[a+\eta]
\approx
e_\theta[a]+D E_\theta[a](\eta).
$$

Then we study the singular system or induced norm of the linearized operator:

$$
D E_\theta[a].
$$

The local singular direction is:

$$
v_1
=
\arg\max_{\|\eta\|_{\mathcal{X}}=1}
\|D E_\theta[a](\eta)\|_{\mathcal{Y}}.
$$

This direction maximizes the local error change, not necessarily the final
error.

## 10. Three Important Directions

### 10.1 Spectral / Singular Direction

Vector:

$$
v_1
=
\arg\max_{\|\delta\|_2=1}
\|J_E(x)\delta\|_2.
$$

Function:

$$
v_1
=
\arg\max_{\|\eta\|_{\mathcal{X}}=1}
\|D E_\theta[a](\eta)\|_{\mathcal{Y}}.
$$

Meaning: maximizes local error change.

### 10.2 Jacobian-Error Direction

Vector:

$$
J_E(x)^\top e_\theta(x).
$$

Function:

$$
D E_\theta[a]^*e_\theta[a].
$$

Meaning: maximizes the first-order increase of squared final error.

### 10.3 Finite-Budget Adversarial Direction

Vector:

$$
\delta^*
=
\arg\max_{\|\delta\|_p\le\varepsilon}
\|e_\theta(x+\delta)\|_q.
$$

Function:

$$
\eta^*
=
\arg\sup_{\|\eta\|_{\mathcal{X},p}\le\varepsilon}
\|e_\theta[a+\eta]\|_{\mathcal{Y},q}.
$$

Meaning: directly maximizes the final perturbed solver-model discrepancy.

These three directions are related but not identical:

$$
\text{singular direction}
\neq
\text{Jacobian-error direction}
\neq
\text{finite-budget PGD direction}.
$$

For very small \(\varepsilon\), the finite-budget \(L_3\) attack direction
should align more with the Jacobian-error direction than with the pure
singular direction.

## 11. PGD / PJD Attack in Vector and Function Spaces

PGD/PJD is an optimization method. It is separate from the loss target
\(L_1,L_2,L_3\).

### 11.1 Vector PGD

Generic projected ascent:

$$
\delta_{k+1}
=
\Pi_{\|\delta\|_p\le\varepsilon}
\left(
\delta_k+\alpha s_k
\right).
$$

Raw gradient direction:

$$
s_k=\nabla_\delta \mathcal{A}(\delta_k).
$$

Steepest direction:

$$
s_k
=
\arg\max_{\|s\|_p\le1}
\langle \nabla_\delta\mathcal{A}(\delta_k),s\rangle.
$$

For \(p=2\):

$$
s_k
=
\frac{\nabla_\delta\mathcal{A}}
{\|\nabla_\delta\mathcal{A}\|_2}.
$$

For \(p=\infty\):

$$
s_k=\operatorname{sign}(\nabla_\delta\mathcal{A}).
$$

For \(1<p<\infty\), let \(p^*\) be the dual exponent:

$$
\frac1p+\frac1{p^*}=1.
$$

If \(r=\nabla_\delta\mathcal{A}\), then:

$$
s_i
=
\frac{
\operatorname{sign}(r_i)|r_i|^{p^*-1}
}{
\|r\|_{p^*}^{p^*-1}
}.
$$

### 11.2 Function-Space PGD

Generic projected ascent:

$$
\eta_{k+1}
=
\Pi_{\|\eta\|_{\mathcal{X},p}\le\varepsilon}
\left(
\eta_k+\alpha s_k
\right).
$$

Steepest direction:

$$
s_k
=
\arg\max_{\|s\|_{\mathcal{X},p}\le1}
D\mathcal{A}[\eta_k](s).
$$

In a Hilbert space with \(p=2\):

$$
s_k
=
\frac{\nabla_\eta\mathcal{A}(\eta_k)}
{\|\nabla_\eta\mathcal{A}(\eta_k)\|_{\mathcal{X}}}.
$$

In \(L^p\), if \(r(s)\) denotes the dual gradient:

$$
s_k(s)
=
\frac{
\operatorname{sign}(r(s))|r(s)|^{p^*-1}
}{
\|r\|_{L^{p^*}}^{p^*-1}
}.
$$

### 11.3 Projection

\(L^2\) projection:

$$
\Pi_{\|\eta\|_2\le\varepsilon}(\eta)
=
\begin{cases}
\eta, & \|\eta\|_2\le\varepsilon,\\
\varepsilon\eta/\|\eta\|_2, & \|\eta\|_2>\varepsilon.
\end{cases}
$$

\(L^\infty\) projection:

$$
\Pi_{\|\eta\|_\infty\le\varepsilon}(\eta)(s)
=
\operatorname{clip}(\eta(s),-\varepsilon,\varepsilon).
$$

### 11.4 Add and Replace Variants

Add:

$$
\eta_{k+1}
=
\Pi_{\mathcal{B}_\varepsilon}
\left(
\eta_k+\alpha s_k
\right).
$$

Replace:

$$
\eta_{k+1}
=
\varepsilon
\frac{s_k}{\|s_k\|_{\mathcal{X},p}}.
$$

Add accumulates directions. Replace discards the previous perturbation and
uses the current direction directly at the boundary.

In many \(p=2\) settings, the maximum of a local linear or nearly linear
objective is on the boundary. This explains why replace-style methods can
reach useful perturbation magnitudes quickly. For highly nonlinear or
path-dependent settings such as recurrent Navier-Stokes rollouts, additive
methods can be stronger.

## 12. Constraint Geometry

Perturbation geometry is another axis, separate from the loss target.

\(p\)-norm ball:

$$
\|\delta\|_p\le\varepsilon.
$$

Sparse or \(L^0\)-style constraint:

$$
\|\delta\|_0\le k.
$$

Patch constraint:

$$
\delta=M\odot p.
$$

Physical transform / expectation-over-transformation objective:

$$
\mathbb{E}_{T\sim\mathcal{T}}
L(f_\theta(T(x+\delta)),y).
$$

Query budget:

$$
N_{\mathrm{query}}\le B.
$$

Binary flip constraint, such as Darcy coefficient fields:

$$
a(i)\in\{3,12\}.
$$

In binary Darcy settings, the perturbation is a flip mask with a Hamming
budget, not an ordinary continuous \(p\)-norm vector.

## 13. Black-Box, White-Box, and Loss Target Are Different Axes

Black-box / white-box describes access and optimization, not the reference
target.

White-box methods use gradients:

$$
\delta_{k+1}
=
\Pi_{\|\delta\|\le\varepsilon}
\left(
\delta_k+\alpha\nabla_\delta L(\delta_k)
\right).
$$

Black-box methods may use:

- label queries;
- score queries;
- random search;
- evolutionary search;
- surrogate models.

The same \(L_2\)-like objective can be optimized with white-box PGD or
black-box Square Attack. The same \(L_3\) objective could also be optimized
black-box, but using solver gradients gives the strongest solver-integrated
white-box setting.

Important distinctions:

$$
\text{black-box/white-box} \neq \text{loss target}.
$$

$$
\text{norm/sparse/patch constraint} \neq \text{loss target}.
$$

$$
L_1,L_2,L_3 \text{ are reference-target categories}.
$$

$$
\text{PGD/C\&W/DeepFool/Square are attack mechanisms}.
$$

## 14. Classical Adversarial Attack Taxonomy

Most classical computer-vision adversarial attacks are classification attacks.
They usually assume a fixed semantic label:

$$
y(x+\delta)=y(x).
$$

Therefore they are usually closer to \(L_2\)-like fixed-reference objectives
than to \(L_3\).

| Method | Task type | Reference target | Closest \(L_i\) type | Access / optimization | Perturbation geometry |
|---|---|---|---|---|---|
| Szegedy adversarial examples | classification | fixed label | \(L_2\)-like | white-box optimization | small norm perturbation |
| FGSM | classification | fixed label CE | \(L_2\)-like | white-box one-step gradient | often \(L^\infty\) |
| PGD / Madry | classification | fixed label CE | \(L_2\)-like | white-box iterative gradient | \(p\)-norm ball |
| C&W | classification | fixed class or target-class margin | \(L_2\)-like | white-box optimization | min distortion + margin |
| DeepFool | classification | leave original decision region | \(L_2\)-like boundary | white-box boundary approximation | minimum distance |
| JSMA | classification | fixed target class | \(L_2\)-like | white-box saliency | sparse \(L^0\)-style |
| One-pixel attack | classification | fixed target/original class | \(L_2\)-like | black-box / evolutionary | extremely sparse |
| Boundary Attack | classification | adversarial decision maintained | \(L_2\)-like boundary | black-box decision-only | reduce perturbation norm |
| HopSkipJump | classification | decision boundary | \(L_2\)-like boundary | black-box decision-only | query-efficient boundary search |
| Square Attack | classification | fixed label score or margin | \(L_2\)-like | black-box score-based | random square updates |
| Adversarial Patch | classification | fixed target class | \(L_2\)-like targeted | often white-box | patch + transformations |
| AutoAttack | classification | fixed label robustness evaluation | \(L_2\)-like | ensemble/evaluation protocol | multiple constraints |
| TRADES | classification | clean-vs-perturbed prediction consistency | \(L_1\)-like component | white-box | robustness-accuracy tradeoff |
| VAT | classification or semi-supervised | prediction distribution change | \(L_1\)-like | white-box | local distributional smoothness |

Important caution: these methods are not literally equal to \(L_2\), because
they may use cross entropy, logit margins, decision-boundary constraints, or
minimum-distortion objectives. They are \(L_2\)-like only in the sense that the
reference is fixed.

## 15. Scientific ML, PINN, and Neural-Operator Taxonomy

Many scientific ML papers study regression, but they often use PDE residuals
instead of solver-model discrepancy.

Physics residual objective:

$$
\|\mathcal{P}(F_\theta(a+\eta))\|
$$

Solver-consistent objective:

$$
\|F_\theta(a+\eta)-G(a+\eta)\|.
$$

These are different. A small PDE residual does not always guarantee closeness
to the particular solver trajectory or solution branch, especially when
boundary/initial data, uniqueness, discretization, or inverse-problem
ambiguities matter.

| Method | Task | Model class | Sampling / attack target | Closest type | Solver use | Solver gradient? |
|---|---|---|---|---|---|---|
| AT-PINN / WbAR | PDE solving regression | PINN | PDE residual failure region | physics-loss, not \(L_3\) | no solver target in attack | no |
| RAMS PINN mode | PDE solving regression | PINN | move collocation points by PDE residual gradient | physics-loss | no solver | no |
| RAMS PI operator mode | operator learning regression | PI-DeepONet/operator | residual-based input/sample movement | physics-loss | no solver in acquisition | no |
| RAMS data-driven operator mode | operator learning regression | DeepONet/operator | acquisition by PDE residual, then label selected input with solver/experiment | acquisition is physics-loss; training uses data loss | solver labels selected inputs | no solver gradient in acquisition |
| StablePDENet | operator learning regression | neural operator | physics/stability residual under perturbation | physics-loss | not solver-consistent \(G(a+\eta)\) | no core solver backward |
| Adesoji & Chen FNO robustness | operator regression | FNO | adversarial perturbation and solver-output evaluation | closest neural-operator robustness prior; not full solver-backprop \(L_3\) | solver/reference for evaluation or data | no differentiable solver backward |
| Roy neural-operator digital twins | operator regression | neural operator | sparse vulnerability attacks | robustness evaluation | simulation/ground truth for evaluation | no |
| Roy active learning + denoising | operator regression | neural operator | vulnerability probing, then targeted data/denoising | defense, not \(L_3\) solver-backprop | solver/data generator for labels | no |
| Active operator learning / UQ | operator regression | DeepONet/FNO | uncertainty acquisition | not \(L_1,L_2,L_3\), acquisition score | solver labels selected samples | no |
| MRA-FNO | operator regression | FNO | utility/cost acquisition and resolution choice | active learning / multi-fidelity | simulation labels | no |
| GANO | generative operator learning | neural operator generator/discriminator | GAN distribution loss | GAN-adversarial, not perturbation attack | no solver objective | no |
| Karniadakis adv-NO turbulence | regression/generative flow | neural operator / generative model | discriminator-style adversarial loss for sharper turbulence | GAN-style, not PGD \(L_3\) | simulation data possible | no solver attack gradient |
| This paper | PDE operator regression | neural operator | \(L_3=\|F_\theta(a+\eta)-G(a+\eta)\|\) | solver-consistent \(L_3\) | solver in perturbed objective | yes, strongest version |

## 16. Regression Adversarial Attack Taxonomy

Regression adversarial attacks exist, but they are much less common than
classification attacks. They also usually do not use a perturbed oracle
\(G(x+\delta)\). Most regression attacks fall into one of the following
categories:

1. model-output stability, which is \(L_1\)-like;
2. fixed-target regression error, which is \(L_2\)-like;
3. attacker-chosen target output, which is targeted \(L_2\)-like;
4. distributional output shift, which is \(L_1\)-like or target-distribution
   based;
5. task-specific output degradation, such as steering-angle or depth error.

### 16.1 Nguyen and Raff: Numerical Stability for Regression

Nguyen and Raff, **Adversarial Attacks, Regression, and Numerical Stability
Regularization** (2018), is one of the clearest papers explicitly about
adversarial attacks in neural-network regression.

They point out a key fact that is also important for this paper:

- in classification, small input perturbations are expected not to change the
  correct label;
- in regression, output changes are expected when the input changes.

However, their actual regularization is primarily a model-output stability
penalty:

$$
\ell(y-f_\theta(x))
+
\lambda
\mathbb{E}_{\Delta x:\|\Delta x\|_p<\varepsilon}
\ell\big(f_\theta(x)-f_\theta(x+\Delta x)\big).
$$

This is \(L_1\)-like because it compares the model to itself:

$$
\|f_\theta(x+\Delta x)-f_\theta(x)\|.
$$

They later introduce a nearest-neighbor label difference to avoid penalizing
reasonable regression variation too harshly. The idea is that the model should
not be punished when its local output change is no larger than the local label
change observed in nearby training data. But this is still not a
solver-consistent objective:

$$
\|f_\theta(x+\Delta x)-G(x+\Delta x)\|.
$$

It has no perturbed oracle \(G(x+\Delta x)\).

### 16.2 Linear Regression and Adversarial Training

Linear-regression adversarial training papers study objectives like:

$$
\min_w
\sum_i
\max_{\|\delta_i\|\le\varepsilon}
\big(y_i-w^\top(x_i+\delta_i)\big)^2.
$$

This is fixed-target regression. The label \(y_i\) does not change when the
input is perturbed. Therefore it is \(L_2\)-like:

$$
\|f(x_i+\delta_i)-y_i\|.
$$

Examples include:

- **Adversarial Regression with Multiple Learners** by Tong et al. (2018);
- **Regularization Properties of Adversarially-Trained Linear Regression** by
  Ribeiro et al. (2023).

These papers are useful because they show that regression adversarial training
is mathematically meaningful and can connect to ordinary regularization such
as ridge or lasso. But they do not address the PDE operator setting in which a
new input has a new oracle solution.

### 16.3 Multivariate Time-Series Regression

Mode and Hoque, **Adversarial Examples in Deep Learning for Multivariate Time
Series Regression** (2020), adapt FGSM and BIM from image classification to
time-series regression models such as CNN, LSTM, and GRU.

Their attack uses a regression cost function such as MSE:

$$
\eta
=
\varepsilon\operatorname{sign}
\left(
\nabla_X J_f(X,\widehat F)
\right),
$$

and:

$$
X' = X+\eta.
$$

This is essentially fixed-output regression attack. The attack uses the
gradient of the model's training/evaluation cost with respect to the input
time series. It is therefore closest to \(L_2\)-like regression:

$$
\|f(X+\eta)-\widehat F\|.
$$

It does not compute a new ground-truth future trajectory for the perturbed
time series.

### 16.4 Targeted Time-Series Forecasting Attacks

Targeted time-series forecasting attacks define attacker-chosen output targets.
Govindarajulu et al., **Targeted Attacks on Timeseries Forecasting** (2023),
define directional, amplitudinal, and temporal attacks.

Their targeted attacks optimize a loss of the form:

$$
L(f(x_{\mathrm{adv}}),y_0),
$$

where \(y_0\) is an attacker-chosen target such as:

$$
y_0\in
\{y+\alpha|y|,\ \mathrm{lim}(\tau,y),\ \mathrm{att}(y(t))\}.
$$

This is targeted \(L_2\)-like. The reference is not a perturbed oracle; it is
an adversarial target selected to create a desired output impact.

### 16.5 Probabilistic Forecasting Attacks

Probabilistic forecasting attacks deal with models whose output is a
probability distribution over future sequences.

Dang-Nhu et al., **Adversarial Attacks on Probabilistic Autoregressive
Forecasting Models** (2020), optimize statistics of the forecast distribution:

$$
\arg\min_{\|\delta\|\le\eta}
\left\|
\mathbb{E}_{f(y|x+\delta)}
[\chi(Y)]
-
t_{\mathrm{adv}}
\right\|_2^2.
$$

Here \(t_{\mathrm{adv}}\) is an adversarial target for a statistic of the
forecast distribution. This is not solver-consistent \(L_3\). It is a
targeted distributional attack.

Yoon et al., **Robust Probabilistic Time Series Forecasting** (2022), define
robustness in terms of output-distribution stability, for example using
bounded Wasserstein deviation. This is closer to distributional \(L_1\)-like
robustness:

$$
d\big(f(x+\delta),f(x)\big),
$$

where \(d\) is a distributional distance such as Wasserstein distance.

### 16.6 Monocular Depth Estimation

Monocular depth estimation is dense regression. Zhang et al.,
**Adversarial Attacks on Monocular Depth Estimation** (2020), adapt
classification attacks to depth prediction and study non-targeted, targeted,
and universal attacks. The goal is to create large depth-estimation errors, or
to move the predicted depth of selected objects toward a target depth.

Typical forms are:

$$
\|f(x+\delta)-y_{\mathrm{depth}}\|
$$

or:

$$
\|f(x+\delta)-t_{\mathrm{depth}}\|.
$$

These are fixed-target or attacker-targeted dense regression attacks. They do
not compute a new physical ground-truth depth map for a perturbed scene.

Important nuance: in image sensor attacks, the perturbation may be interpreted
as sensor noise or an image artifact while the underlying scene depth remains
the same. In that setting, a fixed depth target can be reasonable. In PDE
operator learning, perturbing the input function changes the physical PDE
problem itself, so \(G(a+\eta)\) must change.

### 16.7 Steering-Angle / Autonomous Driving Attacks

DeepBillboard studies physical-world adversarial billboards for autonomous
driving. The objective is to maximize the possibility, degree, and duration of
steering-angle errors.

This is a regression/control-output attack:

$$
\max_\delta
\mathrm{SteeringError}(f(x+\delta)).
$$

It is task-specific output degradation. It is not an oracle-consistent
function-regression objective.

### 16.8 Recent Online Time-Series Regression Attacks

INTARG, **Informed Real-Time Adversarial Attack Generation for Time-Series
Regression** (2026), attacks time-series forecasting in an online
bounded-buffer setting. It selectively attacks time steps where the model is
confident and the expected prediction error is large.

This is still forecasting-error maximization. It is useful as evidence that
regression attack literature is growing, but it remains in the ordinary
forecasting setting rather than the solver-integrated PDE setting.

### 16.9 Summary of Regression Attack Losses

| Work / direction | Regression setting | Main attack loss | Taxonomy |
|---|---|---|---|
| Nguyen and Raff 2018 | generic neural-network regression | \(\|f(x+\Delta)-f(x)\|\), with nearest-neighbor flexibility threshold | \(L_1\)-like |
| Tong et al. 2018 | adversarial linear regression with multiple learners | \((y-w^\top(x+\delta))^2\) | fixed-target \(L_2\)-like |
| Ribeiro et al. 2023 | adversarially trained linear regression | \(\max_{\|\delta\|\le\varepsilon}(y-w^\top(x+\delta))^2\) | fixed-target \(L_2\)-like |
| Mode and Hoque 2020 | multivariate time-series regression | MSE/cost \(J_f(X,\widehat F)\) under FGSM/BIM | fixed-target \(L_2\)-like |
| Targeted TSF 2023 | time-series forecasting | \(L(f(x_{\mathrm{adv}}),y_0)\) with attacker-chosen \(y_0\) | targeted \(L_2\)-like |
| Dang-Nhu et al. 2020 | probabilistic forecasting | \(\|\mathbb{E}_{f(y|x+\delta)}[\chi(Y)]-t_{\mathrm{adv}}\|^2\) | targeted distributional |
| Yoon et al. 2022 | probabilistic forecasting robustness | distributional stability, e.g. Wasserstein deviation | distributional \(L_1\)-like |
| Monocular depth attacks | dense visual regression | error to fixed depth ground truth or target depth | fixed/targeted \(L_2\)-like |
| DeepBillboard | steering-angle regression/control | steering-angle error magnitude/duration | task-specific output error |
| This paper | PDE operator regression | \(\|F_\theta(a+\eta)-G(a+\eta)\|\) | solver-consistent \(L_3\) |

### 16.10 Key Takeaway for This Paper

Regression adversarial attacks are not absent, but most of them do one of the
following:

$$
\|f(x+\delta)-f(x)\|
$$

or:

$$
\|f(x+\delta)-y\|
$$

or:

$$
\|f(x+\delta)-t_{\mathrm{adv}}\|.
$$

They rarely use:

$$
\|f(x+\delta)-G(x+\delta)\|.
$$

The reason is that most regression settings do not have a differentiable
solver/oracle that can be queried at the perturbed input. PDE operator learning
does have such a solver, and this makes the solver-consistent \(L_3\)
objective both possible and necessary.

## 17. Solver-Usage Levels

| Level | Meaning | Example |
|---|---|---|
| S0 | no solver in the method objective; may use PDE residual or data distribution only | PINN residual training, GANO |
| S1 | solver generates offline labels or labels selected samples | supervised FNO/DeepONet, active learning, RAMS data-driven mode |
| S2 | solver output is evaluated at the perturbed input in the objective | \(\|F_\theta(a+\eta)-G(a+\eta)\|\) |
| S3 | solver is differentiable and its backward/adjoint gradient is used to construct perturbations | strongest solver-integrated attack/training |

Most existing work is S0 or S1. The proposed paper focuses on S2/S3.

## 18. Why \(L_3\) Matters for PDE Regression

In classification, the fixed-label assumption is often reasonable:

$$
y(x+\delta)=y(x).
$$

A cat image with small noise is still a cat. Therefore classical attack losses
can compare \(f_\theta(x+\delta)\) to the same label \(y(x)\).

In PDE regression, perturbing the input function changes the physical problem:

$$
G(a+\eta)\neq G(a).
$$

Therefore:

$$
\|F_\theta(a+\eta)-G(a)\|
$$

can punish correct physical movement. If the model exactly equals the solver:

$$
F_\theta=G,
$$

then the true solver-consistent error should be:

$$
L_3(\eta)
=
\|F_\theta(a+\eta)-G(a+\eta)\|
=0
$$

for every perturbation. But \(L_1\) and \(L_2\) can still be large because the
true PDE solution itself changes.

This is the cleanest argument for why \(L_3\) is necessary.

## 19. Paper-Ready Summary Statement

Classical adversarial attacks differ in optimization strategy, perturbation
geometry, access model, and success criterion. White-box attacks use
gradients, black-box attacks use queries or random search, and sparse or patch
attacks impose different feasible sets. These differences are orthogonal to
the choice of reference target. Most classical classification attacks use a
fixed-label reference and are therefore \(L_2\)-like in our taxonomy, while
some consistency-based methods resemble \(L_1\)-like objectives.
Physics-informed scientific ML methods often attack PDE residuals rather than
solver-model discrepancy. In contrast, PDE operator regression requires a
perturbed-input oracle reference \(G(a+\eta)\), leading to the
solver-consistent \(L_3\) objective.

## 20. Core Equations to Remember

Error operator:

$$
E_\theta=F_\theta-G.
$$

Error function:

$$
e_\theta[a]=E_\theta(a)=F_\theta(a)-G(a).
$$

Fréchet derivative:

$$
D E_\theta[a]:\mathcal{X}\to\mathcal{Y}.
$$

Adjoint derivative:

$$
D E_\theta[a]^*:\mathcal{Y}\to\mathcal{X}.
$$

Local linearization:

$$
e_\theta[a+\eta]
\approx
e_\theta[a]+D E_\theta[a](\eta).
$$

Jacobian-error proxy:

$$
D E_\theta[a]^*e_\theta[a].
$$

True solver-consistent attack:

$$
\eta^*
=
\arg\sup_{\|\eta\|_{\mathcal{X},p}\le\varepsilon}
\|e_\theta[a+\eta]\|_{\mathcal{Y},q}.
$$

Small-\(\varepsilon\) first-order loss increase:

$$
\sup_{\|\eta\|_{\mathcal{X},p}\le\varepsilon}
D\Phi[a](\eta)
=
\varepsilon
\|D E_\theta[a]^*e_\theta[a]\|_{\mathcal{X},p^*}.
$$
