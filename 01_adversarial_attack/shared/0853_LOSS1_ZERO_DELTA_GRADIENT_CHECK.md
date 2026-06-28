# Loss1 Zero-Delta Gradient Check

This note records why all three `loss1` objectives have zero gradient at
\(\delta=0\) in the current PyTorch/autograd implementation.

The key point is:

Strictly speaking, a norm is usually non-differentiable at zero. However,
PyTorch/autograd chooses the zero subgradient at zero. Therefore, the attack
can get stuck if it starts from exactly \(\delta=0\).

## Loss1 Definition

Define

\[
r_1(\delta)=f(x+\delta)-f(x).
\]

Then

\[
L_1(\delta)=\|r_1(\delta)\|_q.
\]

When \(r_1(\delta)\neq 0\),

\[
\nabla_\delta L_1(\delta)
=
J_f(x+\delta)^\top
\frac{
|r_1(\delta)|^{q-2}r_1(\delta)
}{
\|r_1(\delta)\|_q^{q-1}
},
\]

where

\[
J_f(x+\delta)=\frac{\partial f(x+\delta)}{\partial \delta}.
\]

At

\[
\delta=0,
\]

we have

\[
r_1(0)=f(x)-f(x)=0.
\]

Therefore,

\[
L_1(0)=0.
\]

In PyTorch/autograd,

\[
\nabla_{r_1}\|r_1\|_q\big|_{r_1=0}=0.
\]

Thus,

\[
\nabla_\delta L_1(0)
=
J_f(x)^\top 0
=
0.
\]

## Loss1 Original Objective

The original objective is

\[
O_{\mathrm{orig}}(\delta)
=
L_1(\delta)
=
\|f(x+\delta)-f(x)\|_q.
\]

Therefore,

\[
\nabla_\delta O_{\mathrm{orig}}(0)=0.
\]

## Loss1 Increment-Ratio Objective

The increment-ratio objective is

\[
O_{\mathrm{inc}}(\delta)
=
\frac{L_1(\delta)-L_1(0)}
{\|\delta\|_p+\eta}.
\]

Since

\[
L_1(0)=0,
\]

we have

\[
O_{\mathrm{inc}}(\delta)
=
\frac{L_1(\delta)}
{\|\delta\|_p+\eta}.
\]

Using the quotient rule,

\[
\nabla_\delta O_{\mathrm{inc}}(\delta)
=
\frac{
(\|\delta\|_p+\eta)\nabla_\delta L_1(\delta)
-
L_1(\delta)\nabla_\delta \|\delta\|_p
}{
(\|\delta\|_p+\eta)^2
}.
\]

At \(\delta=0\),

\[
L_1(0)=0,
\]

\[
\nabla_\delta L_1(0)=0,
\]

and in PyTorch/autograd,

\[
\nabla_\delta \|\delta\|_p\big|_{\delta=0}=0.
\]

Therefore,

\[
\nabla_\delta O_{\mathrm{inc}}(0)
=
\frac{
\eta\cdot 0 - 0\cdot 0
}{
\eta^2
}
=
0.
\]

## Loss1 Regularized Objective

The regularized objective is

\[
O_{\mathrm{reg}}(\delta)
=
L_1(\delta)-c\|\delta\|_p.
\]

Therefore,

\[
\nabla_\delta O_{\mathrm{reg}}(\delta)
=
\nabla_\delta L_1(\delta)
-
c\nabla_\delta \|\delta\|_p.
\]

At \(\delta=0\),

\[
\nabla_\delta L_1(0)=0,
\]

and in PyTorch/autograd,

\[
\nabla_\delta \|\delta\|_p\big|_{\delta=0}=0.
\]

Therefore,

\[
\nabla_\delta O_{\mathrm{reg}}(0)
=
0-c\cdot0
=
0.
\]

## Conclusion

For the three `loss1` objectives,

\[
\nabla_\delta O_{\mathrm{orig}}(0)=0,
\]

\[
\nabla_\delta O_{\mathrm{inc}}(0)=0,
\]

and

\[
\nabla_\delta O_{\mathrm{reg}}(0)=0.
\]

Therefore, in the current code, all three `loss1` objectives get zero gradient
when initialized from exactly \(\delta=0\). This is not only an issue for
`loss1 original`; `loss1 increment_ratio` and `loss1 regularized` also get stuck
from a strict zero initialization.
