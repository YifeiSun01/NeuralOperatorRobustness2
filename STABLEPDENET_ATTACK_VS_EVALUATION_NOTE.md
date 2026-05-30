# StablePDENet Attack Objective vs Evaluation Metric Note

This note records the key distinction we discussed for **StablePDENet: Enhancing Stability of Operator Learning for Solving Differential Equations**.

Paper: https://arxiv.org/abs/2601.06472

## Core Point

StablePDENet's attack/training objective and its evaluation metric are not the same loss.

Short version:

> During attack/training, StablePDENet maximizes a physics-informed residual loss. During evaluation, it reports error against a numerical-solver reference solution.

This distinction is important for positioning solver-consistent adversarial robustness work.

---

## 1. Evaluation Stage: Solver-Reference Error

During evaluation/testing, the paper compares the neural operator prediction against a reference solution produced by a numerical PDE solver.

For a clean input function `a`, the reference is:

```math
u_{ref} = S(a),
```

where `S` is a high-fidelity numerical solver.

For a perturbed/adversarial input:

```math
u_{ref}^{adv} = S(a + \delta).
```

The model prediction is:

```math
u_\theta = G_\theta(a),
```

or under perturbation:

```math
u_\theta^{adv} = G_\theta(a + \delta).
```

The evaluation error is then a supervised-style metric such as relative error or RMS error:

```math
\frac{\|G_\theta(a) - S(a)\|}{\|S(a)\|},
```

or

```math
\frac{\|G_\theta(a+\delta) - S(a+\delta)\|}{\|S(a+\delta)\|}.
```

So evaluation is solver-reference based.

---

## 2. Attack Stage: Physics Residual Loss

During the adversarial attack step, StablePDENet does not directly maximize the solver-reference error.

It does **not** repeatedly solve

```math
S(a + \delta_k)
```

at every PGD step, and it does **not** optimize

```math
\|G_\theta(a+\delta)-S(a+\delta)\|.
```

Instead, the attack objective is a physics-informed loss:

```math
\mathcal{L}_{phys}(G_\theta(a+\delta)).
```

That is, the current neural-operator output is substituted into the PDE, and the PDE residual is computed.

For example, for Poisson:

```math
-\Delta u = f,
```

with input `f + delta`, the residual is roughly:

```math
-\Delta G_\theta(f+\delta) - (f+\delta).
```

For a time-dependent PDE, if the model outputs a full spacetime field `u_theta(t,x)`, one can compute residuals such as:

```math
\partial_t u_\theta - \nu \Delta u_\theta - f.
```

But if the model outputs only a final-time snapshot `u_theta(T,x)`, the full time derivative term is not directly available, so the physics loss cannot fully represent the evolution equation.

---

## 3. PGD Path In StablePDENet

For fixed model parameters `theta`, the attack can be written as:

```math
\delta_{k+1}
=
\Pi_{\|\delta\|\le\epsilon}
\left(
\delta_k
+
\alpha \nabla_\delta
\mathcal{L}_{phys}(G_\theta(a+\delta_k))
\right).
```

The gradient path is:

```math
\delta
\rightarrow
 a+\delta
\rightarrow
G_\theta(a+\delta)
\rightarrow
u_\theta
\rightarrow
\mathcal{L}_{phys}.
```

There is no differentiable numerical-solver path:

```math
\delta
\rightarrow
S(a+\delta).
```

---

## 4. Evaluation Path

In evaluation, the paper uses two paths:

Model path:

```math
a+\delta
\rightarrow
G_\theta(a+\delta)
\rightarrow
u_\theta^{adv}.
```

Reference path:

```math
a+\delta
\rightarrow
S(a+\delta)
\rightarrow
u_{ref}^{adv}.
```

Then it compares:

```math
u_\theta^{adv}
\quad \text{vs.} \quad
u_{ref}^{adv}.
```

So the evaluation metric is solver-reference based, while the attack objective is physics-residual based.

---

## 5. Why This Matters

The two quantities are related but not equivalent:

```math
\text{Attack loss}
=
\mathcal{L}_{phys}(G_\theta(a+\delta)),
```

while

```math
\text{Evaluation error}
=
\frac{\|G_\theta(a+\delta)-S(a+\delta)\|}{\|S(a+\delta)\|}.
```

StablePDENet uses `L_phys` as a computationally cheap surrogate for true solver-consistent error.

This is attractive because solving `S(a+delta_k)` at every PGD step would be expensive.

However, the surrogate can be imperfect:

- physics residual can be small while solution error is not small;
- PDE conditioning can make residual and solution error poorly aligned;
- boundary/initial condition weighting can distort the residual loss;
- stiff PDEs, long-time rollout, chaotic dynamics, and non-normal transient growth can make residual stability and solver-consistent stability diverge;
- time-dependent PDEs are especially tricky if the operator only outputs final snapshots.

---

## 6. Positioning Relative To Our Work

StablePDENet attack/training:

```math
\max_{\|\delta\|\le\epsilon}
\mathcal{L}_{phys}(G_\theta(a+\delta)).
```

Solver-consistent attack/training would instead target:

```math
\max_{\|\delta\|\le\epsilon}
\|G_\theta(a+\delta)-S(a+\delta)\|.
```

This means our potential direction attacks the same quantity that evaluation actually cares about, whereas StablePDENet attacks a physics-informed surrogate.

A concise comparison statement:

> StablePDENet uses a physics-informed residual loss as its adversarial objective, while its robustness evaluation is based on numerical-solver reference errors. Therefore, the adversarial objective and the evaluation metric are not identical.

Another sharper statement:

> StablePDENet studies physics-residual stability; solver-consistent robustness studies stability with respect to the numerical solution operator.

---

## 7. Paper-Ready Wording

StablePDENet performs adversarial training in a self-supervised, physics-informed manner. PGD is used to perturb the input functions by maximizing the PDE residual of the neural-operator prediction. However, the robustness evaluation is conducted against numerical-solver reference solutions, using relative or RMS solution errors. Thus, the adversarial objective used during attack/training and the solver-reference metric used during evaluation are not identical. This distinction motivates solver-consistent robustness objectives that directly attack the discrepancy between the learned operator and the numerical solution operator.

