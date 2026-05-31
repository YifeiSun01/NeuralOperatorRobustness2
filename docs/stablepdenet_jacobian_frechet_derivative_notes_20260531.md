# StablePDENet Jacobian Norm / Frechet Derivative Notes

Date: 2026-05-31

This note records our interpretation of the final Jacobian norm / Frechet derivative norm analysis in the StablePDENet paper.

## 1. Neural Operator Object

StablePDENet uses a neural-operator model, which can be written as

```math
G_{\theta_h}: a \mapsto u.
```

Here `a` is the input function. Depending on the PDE or ODE problem, `a` may be a source term, initial condition, boundary condition, coefficient field, forcing term, or another function-valued input.

The output `u` is the solution function predicted by the neural operator. More explicitly,

```math
G_{\theta_h}(a)(x)
```

means: feed the input function `a` into the model, get an output solution function, and evaluate that output solution at the coordinate point `x`.

## 2. DeepONet / PIDeepONet Structure

The model is based on a DeepONet/PIDeepONet-style architecture, not a plain single-input neural network. DeepONet has two main parts.

The branch net receives sampled values of the input function:

```math
[a(s_1), a(s_2), \dots, a(s_m)].
```

It outputs a feature vector

```math
B_\theta(a) = [b_1(a), \dots, b_p(a)].
```

The trunk net receives the output coordinate point `x`, such as a spatial coordinate or a time-space coordinate. It outputs another feature vector

```math
T_\theta(x) = [t_1(x), \dots, t_p(x)].
```

The final predicted solution value is produced by a dot product:

```math
G_\theta(a)(x) = B_\theta(a) \cdot T_\theta(x) + b_0
               = \sum_{k=1}^{p} b_k(a)t_k(x) + b_0.
```

So the intermediate branch/trunk outputs are not themselves the PDE solution. They are feature vectors. The scalar value at coordinate `x` appears after the branch and trunk features are paired by the dot product.

## 3. Two Different Inputs: `a` Versus `x`

There are two different objects that can look like inputs:

- `a`: the neural operator input function. This is the function being perturbed, attacked, or differentiated with respect to.
- `x`: the coordinate where the output solution function is evaluated. This is the trunk-net input, meaning where we look at the solution value `u(x)` or `u(t, x)`.

The final Jacobian norm analysis is about sensitivity with respect to `a`, not sensitivity with respect to the coordinate `x`.

In other words, the question is not mainly "what happens if I move the evaluation coordinate?" The question is:

```text
If the input function a changes a little, how much can the output solution function change?
```

## 4. Frechet Derivative

Because

```math
G_{\theta_h}: a \mapsto u
```

is a function-to-function map, the correct derivative is a Frechet derivative:

```math
D G_{\theta_h}(a).
```

This derivative describes what happens near a fixed input function `a`. If `a` is perturbed by a small function perturbation `\delta a`, then

```math
G_{\theta_h}(a + \delta a)
\approx
G_{\theta_h}(a) + D G_{\theta_h}(a)[\delta a].
```

The Frechet derivative itself is a linear operator:

```math
D G_{\theta_h}(a): \delta a \mapsto \delta u.
```

So it maps an input-function perturbation to an output-solution perturbation:

```text
input function perturbation -> output solution perturbation.
```

## 5. Discretized Jacobian Matrix

After discretization, the function-to-function map becomes an ordinary vector-to-vector map.

The input function `a` is sampled at `m` sensor points:

```math
a \rightarrow [a(s_1), \dots, a(s_m)] \in \mathbb{R}^m.
```

The output solution function `u` is sampled at `n` grid/evaluation points:

```math
u \rightarrow [u(x_1), \dots, u(x_n)] \in \mathbb{R}^n.
```

Therefore the discretized operator is

```math
G_{\theta_h}: \mathbb{R}^m \to \mathbb{R}^n.
```

The discretized Frechet derivative is the Jacobian matrix

```math
J_{\theta_h}(a)
=
\frac{\partial G_{\theta_h}(a)}{\partial a}
\in \mathbb{R}^{n \times m}.
```

A matrix entry has the meaning

```math
\frac{\partial u(x_j)}{\partial a(s_i)}.
```

That is: if the input function value at the `i`-th sensor point changes a little, how much does the predicted solution at the `j`-th output point change?

The local linearization is

```math
G_{\theta_h}(a + \delta a)
\approx
G_{\theta_h}(a) + J_{\theta_h}(a)\delta a.
```

## 6. Jacobian Spectral Norm

The paper's Jacobian norm analysis is about the spectral norm of this local Jacobian matrix:

```math
\|J_{\theta_h}(a)\|_2 = \sigma_{\max}(J_{\theta_h}(a)).
```

Equivalently,

```math
\|J_{\theta_h}(a)\|_2
=
\max_{\|\delta a\|_2 = 1}
\|J_{\theta_h}(a)\delta a\|_2.
```

Plainly: among all unit-size input-function perturbation directions, find the one that makes the model output change the most. The maximum amplification factor is the Jacobian spectral norm.

The largest singular value is this maximum amplification factor.

The corresponding right singular vector is the most dangerous local input perturbation direction.

The corresponding left singular vector is the output-solution direction in which the response is mainly expressed.

## 7. What This Analysis Is Not

This Jacobian norm analysis is not directly computing the prediction error between model output and ground-truth PDE solution.

It is also not a soft loss, SoftDTW, softmax, or training objective by itself.

It is a local sensitivity/stability diagnostic of the neural operator map

```math
a \mapsto G_{\theta_h}(a).
```

The central question is:

```text
Near this input function a, how much can a small input-function perturbation be amplified in the predicted solution function?
```

If the norm is large, the model is locally sensitive to input perturbations and may be less stable.

If the norm is small, the model is locally less sensitive and therefore more stable in this specific Frechet-derivative sense.

## 8. One-Sentence Summary

StablePDENet's final Jacobian norm analysis computes the Frechet derivative of the neural operator `G_{\theta_h}: a \mapsto u` with respect to the input function `a`; after discretization this becomes a Jacobian matrix `J_{\theta_h}(a) \in \mathbb{R}^{n \times m}`, and the paper measures its spectral norm, i.e. largest singular value, to quantify the maximum local amplification from input-function perturbations to output-solution perturbations.
