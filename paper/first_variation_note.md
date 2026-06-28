# First Variation of the Solver-Model Error Loss

## 1. Objects and Notation

We consider two maps between function spaces:

$$
\mathcal{F}_\theta:\mathcal{X}\to\mathcal{Y},
\qquad
\mathcal{G}:\mathcal{X}\to\mathcal{Y}.
$$

Here, \(\mathcal{F}_\theta\) is the learned neural operator surrogate, and
\(\mathcal{G}\) is the numerical solver oracle. For an input function
\(x\in\mathcal{X}\), both maps produce output functions in \(\mathcal{Y}\).

The model-solver error operator is

$$
\mathcal{E}_\theta
=
\mathcal{F}_\theta-\mathcal{G},
\qquad
\mathcal{E}_\theta:\mathcal{X}\to\mathcal{Y}.
$$

For a concrete input \(x\), the corresponding error function is

$$
e_\theta[x]
=
\mathcal{E}_\theta(x)
=
\mathcal{F}_\theta(x)-\mathcal{G}(x).
$$

Thus:

$$
\mathcal{E}_\theta
\quad\text{is an operator,}
\qquad
e_\theta[x]
\quad\text{is an output function in }\mathcal{Y}.
$$

## 2. Finite Perturbation Versus Infinitesimal Direction

It is important to separate two different roles.

The finite adversarial perturbation is denoted by \(\delta\):

$$
\|\delta\|_{\mathcal{X},p}\le \varepsilon.
$$

This \(\delta\) is used in the exact adversarial objectives, such as

$$
\mathcal{L}_3(\delta)
=
\|\mathcal{F}_\theta(x+\delta)-\mathcal{G}(x+\delta)\|_{\mathcal{Y},q}.
$$

It is not necessarily infinitesimal. It is only constrained by the attack
budget \(\varepsilon\).

For derivatives and first variations, we use \(h\) as the infinitesimal
direction. The actual limiting parameter is \(t\to 0\), and the perturbed
input is

$$
x+t h.
$$

So \(h\) is the direction, while \(t\) controls the infinitesimal step size.

## 3. Local Linearization of the Error Operator

If \(\mathcal{E}_\theta\) is Fréchet differentiable at \(x\), then along a
direction \(h\),

$$
e_\theta[x+t h]
=
e_\theta[x]
+
tD\mathcal{E}_\theta[x](h)
+
o(t),
\qquad
t\to 0.
$$

Here,

$$
D\mathcal{E}_\theta[x]:\mathcal{X}\to\mathcal{Y}
$$

is a bounded linear operator. It maps an input-space direction \(h\) to the
corresponding first-order change of the error function in output space.

In finite-dimensional language, this is the analogue of

$$
E(x+t h)
\approx
E(x)+tJ_E(x)h.
$$

## 4. The Squared Error Loss Functional

Define the squared model-solver error loss by

$$
\Phi(x)
=
\frac{1}{2}
\|e_\theta[x]\|_{\mathcal{Y}}^2.
$$

Since \(\mathcal{Y}\) is a Hilbert space, the squared norm can be written as
an inner product:

$$
\Phi(x)
=
\frac{1}{2}
\left\langle
e_\theta[x],
e_\theta[x]
\right\rangle_{\mathcal{Y}}.
$$

Therefore,

$$
\Phi:\mathcal{X}\to\mathbb{R}.
$$

It maps each input function \(x\) to a scalar loss value.

## 5. First Variation

The first variation of \(\Phi\) at \(x\) along direction \(h\) is the
directional derivative:

$$
D\Phi[x](h)
=
\left.
\frac{d}{dt}
\Phi(x+t h)
\right|_{t=0}.
$$

Using the definition of \(\Phi\),

$$
D\Phi[x](h)
=
\left.
\frac{d}{dt}
\frac{1}{2}
\left\langle
e_\theta[x+t h],
e_\theta[x+t h]
\right\rangle_{\mathcal{Y}}
\right|_{t=0}.
$$

Let

$$
z(t)=e_\theta[x+t h].
$$

Then

$$
\Phi(x+t h)
=
\frac{1}{2}
\langle z(t),z(t)\rangle_{\mathcal{Y}}.
$$

Differentiate with respect to \(t\):

$$
\frac{d}{dt}
\frac{1}{2}
\langle z(t),z(t)\rangle_{\mathcal{Y}}
=
\frac{1}{2}
\left(
\langle z'(t),z(t)\rangle_{\mathcal{Y}}
+
\langle z(t),z'(t)\rangle_{\mathcal{Y}}
\right).
$$

In a real Hilbert space, the two terms are equal, so

$$
\frac{d}{dt}
\frac{1}{2}
\langle z(t),z(t)\rangle_{\mathcal{Y}}
=
\langle z(t),z'(t)\rangle_{\mathcal{Y}}.
$$

At \(t=0\),

$$
z(0)=e_\theta[x].
$$

Also,

$$
z'(0)
=
\left.
\frac{d}{dt}
e_\theta[x+t h]
\right|_{t=0}
=
D\mathcal{E}_\theta[x](h).
$$

Therefore,

$$
D\Phi[x](h)
=
\left\langle
e_\theta[x],
D\mathcal{E}_\theta[x](h)
\right\rangle_{\mathcal{Y}}.
$$

This is the first important form: the current output-space error is paired
with the first-order output-space error change.

## 6. Adjoint Form

Let

$$
A = D\mathcal{E}_\theta[x].
$$

Then

$$
A:\mathcal{X}\to\mathcal{Y}.
$$

Its adjoint is

$$
A^*:\mathcal{Y}\to\mathcal{X}.
$$

By definition of the adjoint,

$$
\langle v,A h\rangle_{\mathcal{Y}}
=
\langle A^*v,h\rangle_{\mathcal{X}}.
$$

Taking

$$
v=e_\theta[x],
\qquad
A=D\mathcal{E}_\theta[x],
$$

we obtain

$$
\left\langle
e_\theta[x],
D\mathcal{E}_\theta[x](h)
\right\rangle_{\mathcal{Y}}
=
\left\langle
D\mathcal{E}_\theta[x]^*e_\theta[x],
h
\right\rangle_{\mathcal{X}}.
$$

Thus,

$$
D\Phi[x](h)
=
\left\langle
D\mathcal{E}_\theta[x]^*e_\theta[x],
h
\right\rangle_{\mathcal{X}}.
$$

This does not mean that the \(\mathcal{Y}\)-space inner product and the
\(\mathcal{X}\)-space inner product are naturally the same. Rather, the
adjoint operator lets us write the same scalar in two equivalent ways.

## 7. Linear Algebra Analogy

In finite dimensions, write

$$
E(x)=F_\theta(x)-G(x)\in\mathbb{R}^m.
$$

The squared error loss is

$$
\Phi(x)
=
\frac{1}{2}\|E(x)\|_2^2
=
\frac{1}{2}E(x)^\top E(x).
$$

Let the input move along direction \(h\):

$$
x_t=x+t h.
$$

Then

$$
D\Phi[x](h)
=
\left.
\frac{d}{dt}
\frac{1}{2}
E(x+t h)^\top E(x+t h)
\right|_{t=0}.
$$

By the chain rule,

$$
\left.
\frac{d}{dt}E(x+t h)
\right|_{t=0}
=
J_E(x)h.
$$

Therefore,

$$
D\Phi[x](h)
=
E(x)^\top J_E(x)h.
$$

Using transpose,

$$
E(x)^\top J_E(x)h
=
\left(J_E(x)^\top E(x)\right)^\top h.
$$

So

$$
D\Phi[x](h)
=
\left\langle
J_E(x)^\top E(x),
h
\right\rangle_{\mathbb{R}^n}.
$$

The input-space gradient is therefore

$$
\nabla_x\Phi(x)
=
J_E(x)^\top E(x).
$$

Since

$$
J_E(x)=J_{F_\theta}(x)-J_G(x),
$$

we can also write

$$
\nabla_x\Phi(x)
=
\left(J_{F_\theta}(x)-J_G(x)\right)^\top
\left(F_\theta(x)-G(x)\right).
$$

## 8. Operator-to-Matrix Correspondence

The operator form and matrix form correspond as follows:

$$
D\mathcal{E}_\theta[x]
\quad\longleftrightarrow\quad
J_E(x).
$$

$$
D\mathcal{E}_\theta[x]^*
\quad\longleftrightarrow\quad
J_E(x)^\top.
$$

$$
e_\theta[x]
\quad\longleftrightarrow\quad
E(x).
$$

$$
D\mathcal{E}_\theta[x]^*e_\theta[x]
\quad\longleftrightarrow\quad
J_E(x)^\top E(x).
$$

So the key idea is:

$$
\text{output-space error}
\quad
\xrightarrow{\text{adjoint / transpose Jacobian}}
\quad
\text{input-space gradient}.
$$

In words, the current model-solver error lives in the output space
\(\mathcal{Y}\). The adjoint derivative pulls that error back to the input
space \(\mathcal{X}\), producing the direction in input space that most
strongly increases the squared model-solver discrepancy to first order.
