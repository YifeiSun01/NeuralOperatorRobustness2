# Research Notes: Solver-Integrated Robustness for Neural Operators

For the most complete categorized version of the user's research ideas, see
`structured_research_notes.md`. This file keeps the earlier running notes and
mathematical sketches.

The code-folder Markdown audit and additional missing/correction points are in
`structured_research_notes.md`, Section 26.

## Working Title

Solver-Integrated Adversarial Attacking and Training of Neural Operators

## Core Setting

The paper studies neural operators for PDE solution maps in a setting where a
classical numerical solver is available. The neural operator and the solver
should be treated as two operators acting on function spaces:

$$
F_\theta:\mathcal{X}\to\mathcal{Y},
\qquad
G:\mathcal{X}\to\mathcal{Y}.
$$

Here \(F_\theta\) is the learned neural operator and \(G\) is the numerical
solver or oracle. In implementation, both are discretized into high-dimensional
maps between vectors or tensors, but the theory should first be stated at the
operator/function-space level.

The central question is how to define generalization and robustness when both
a learned model and a solver are present.

## Generalization

Generalization is a static discrepancy measured at a fixed input function. For
an input \(u\), the model is compared directly against the solver:

$$
\mathcal{L}_{\mathrm{gen}}(u)
= \|F_\theta(u)-G(u)\|_q.
$$

This measures whether the neural operator matches the solver at the input
point. It does not study a local perturbation neighborhood.

## Robustness

Robustness is local and dynamic. It asks how the solver-model discrepancy
changes when the input is perturbed within a small norm ball:

$$
\|\delta\|_p \le \varepsilon.
$$

The perturbation radius \(\varepsilon\) should be small enough to preserve a
local interpretation. The output change is measured with a \(q\)-norm. The
choice of \(p\) and \(q\) matters. The \(p=q=2\) case aligns with the usual
spectral-norm and singular-vector intuition. Other choices can produce
different geometry; for example, \(p=\infty\), \(p=1\), or sparse-style
constraints may create sharp pointwise perturbations that are less physically
plausible.

## Error Operator

Define the error operator:

$$
E_\theta(u) = F_\theta(u)-G(u).
$$

The true adversarial robustness loss should measure the final perturbed error:

$$
L_3(u,\delta)
= \|E_\theta(u+\delta)\|_q
= \|F_\theta(u+\delta)-G(u+\delta)\|_q.
$$

This is the most solver-integrated attack loss because the solver is evaluated
at the perturbed input.

## Three Robustness Metrics

### 1. True Adversarial Error Increase

The main robustness metric is the worst final error within an input ball:

$$
R_{\mathrm{adv}}(u)
= \max_{\|\delta\|_p\le\varepsilon}
\|F_\theta(u+\delta)-G(u+\delta)\|_q.
$$

This metric directly asks whether a local input perturbation can make the model
and solver disagree.

### 2. Local Operator Sensitivity

Linearize the error operator around \(u\). The Frechet derivative
\(D E_\theta[u]\) is the operator-level analogue of a Jacobian matrix:

$$
E_\theta(u+\delta)
\approx E_\theta(u) + D E_\theta[u](\delta).
$$

One possible robustness measure is the induced operator norm:

$$
\|D E_\theta[u]\|_{p\to q}
= \sup_{\|\delta\|_p\le 1}
\|D E_\theta[u](\delta)\|_q.
$$

In finite dimensions with \(p=q=2\), this is the largest singular value of the
Jacobian of the error map. The associated singular vector gives the direction
that maximizes the local change in the error.

Important caveat: this metric maximizes the change in error,
\(\|D E_\theta[u](\delta)\|_q\), not the final error
\(\|E_\theta(u)+D E_\theta[u](\delta)\|_q\). If the original error vector is
nonzero, a direction that makes the error change a lot may not make the final
error large.

### 3. Jacobian-Error Proxy

For very small perturbations, the first-order expansion of the squared
\(L_2\)-style loss gives

$$
\|E_\theta(u+\delta)\|_2^2
\approx
\|E_\theta(u)\|_2^2
+ 2\langle D E_\theta[u](\delta), E_\theta(u)\rangle.
$$

In finite dimensions, if \(J_E(u)\) is the Jacobian of the discretized error
map and \(e=E_\theta(u)\), the loss-increasing direction is controlled by

$$
J_E(u)^\top e.
$$

At the operator level this is the adjoint derivative applied to the error:

$$
D E_\theta[u]^* E_\theta(u).
$$

This proxy should be more aligned with true adversarial loss increase than the
largest singular vector when \(\varepsilon\) is very small. It is also cheaper:
it avoids both multi-step adversarial attack and full singular-value
decomposition.

## Optimized Losses in the Figure

The provided loss panel distinguishes several attack targets:

$$
L_3 = \|f(x+\delta)-g(x+\delta)\|.
$$

$$
L_3\ \mathrm{stopgrad}
= \|f(x+\delta)-\mathrm{stopgrad}(g(x+\delta))\|.
$$

Dictionary nearest-neighbor losses:

$$
L_2\ \mathrm{dict}\ N=200
= \|f(x+\delta)-y_{\mathrm{nearest},200}(x+\delta)\|.
$$

$$
L_2\ \mathrm{dict}\ N=2000
= \|f(x+\delta)-y_{\mathrm{nearest},2000}(x+\delta)\|.
$$

$$
L_2\ \mathrm{dict}\ N=20000
= \|f(x+\delta)-y_{\mathrm{nearest},20000}(x+\delta)\|.
$$

Fixed-solver-target loss:

$$
L_2\ \mathrm{fixed}
= \|f(x+\delta)-g(x)\|.
$$

Model-only change loss:

$$
L_1 = \|f(x+\delta)-f(x)\|.
$$

The true-loss panel should use \(L_3\).

## Why Solver Integration Matters

Regression PDE settings differ from classification attacks. In classification,
it can be reasonable to assume that a small input perturbation preserves the
label. In PDE regression, the output is a continuous function and should also
change when the input changes.

Several surrogate objectives can be misleading:

- \(L_1=\|F_\theta(u+\delta)-F_\theta(u)\|\) only makes the model output move.
  The solver may move in the same direction, so the true model-solver error may
  remain small.
- \(L_2=\|F_\theta(u+\delta)-G(u)\|\) uses the solver only at the unperturbed
  input. This includes the initial model error but does not update the solver
  target with the perturbed input.
- Dictionary-based \(L_2\) uses precomputed approximate solver targets rather
  than evaluating the solver at the current perturbed input.
- \(L_3=\|F_\theta(u+\delta)-G(u+\delta)\|\) directly optimizes the true
  perturbed discrepancy.

The expected conclusion is that increasing a surrogate loss does not
necessarily increase the true \(L_3\) loss. The strongest attacks and most
useful adversarial training examples come from optimizing \(L_3\).

## Forward and Backward Solver Use

Solver integration can be shallow or deep:

- No solver during attack: use model-only loss or a precomputed dictionary.
- Solver in forward pass only: compare \(F_\theta(u+\delta)\) to a fixed or
  recomputed solver target, but stop gradients through the solver.
- Solver in both forward and backward passes: optimize
  \(F_\theta(u+\delta)-G(u+\delta)\) and differentiate through both the neural
  operator and solver.

The hypothesis is that stronger solver integration produces more efficient and
physically meaningful attacks, but it also costs more time and memory.

## Optimization Variants

Attack optimization compares two axes:

- Add vs. replace: whether each step adds a small update to the current
  perturbation or directly replaces the perturbation direction.
- Raw vs. steepest: whether the update uses the raw gradient or the steepest
  direction under the chosen norm geometry.

Observed pattern:

- Steepest replace can be very efficient on Burgers and Darcy flow because it
  moves immediately to the \(\varepsilon\)-ball boundary, where the maximizer is
  often located.
- Steepest add can be slower because it may require many steps just to reach
  the boundary.
- Navier--Stokes may be harder for steepest replace because the loss landscape
  is more nonlinear and the optimal perturbation structure is more complex.

## Random Starts

The model-only \(L_1\) attack requires a nonzero random start. If
\(\delta=0\), then

$$
\|F_\theta(u+\delta)-F_\theta(u)\| = 0,
$$

and the attack can get stuck at the zero perturbation.

## Adversarial Training

The proposed adversarial training uses perturbed inputs found by a
solver-integrated attack, then trains the model against the solver output at
the perturbed input:

$$
\min_\theta
\mathbb{E}_{u}
\left[
\ell\left(F_\theta(u+\delta^*(u)), G(u+\delta^*(u))\right)
\right],
$$

where

$$
\delta^*(u)
\approx
\arg\max_{\|\delta\|_p\le\varepsilon}
\|F_\theta(u+\delta)-G(u+\delta)\|_q.
$$

The perturbation budget \(\varepsilon\) can be sampled randomly from a range
rather than fixed. This covers both small and large perturbation scales and
improves diversity.

## Training Baselines

Baselines to compare:

- Clean training.
- Random clean augmentation: randomly perturb \(u\) but keep the original
  target fixed. This is physically inconsistent because the PDE solution should
  change when the input changes.
- Random solver augmentation: randomly perturb \(u\), then recompute
  \(G(u+\delta)\) with the solver. This is physically meaningful but not
  adaptive to the current model.
- Adversarial training with \(L_1\), \(L_2\), physics loss, stop-gradient
  solver loss, dictionary targets, and true \(L_3\).

Expected conclusion:

- Random solver augmentation can be strong but may overfit or degrade after a
  few hours because its perturbation distribution is fixed.
- Solver-integrated adversarial training adapts perturbations to the current
  model weakness.
- During training, discovered perturbations tend to become increasingly
  high-frequency as the model becomes harder to attack with simple low-frequency
  directions.

## Wall-Clock-Time Comparison

Compare methods by wall-clock time rather than epoch count. Solver-integrated
methods can make each epoch much slower, so epoch-based comparisons are unfair.

Important observations to develop:

- For Burgers, \(L_3\) adversarial training can be far more expensive per epoch
  than \(L_1\) or clean training. For example, 1000 epochs of \(L_3\) may take a
  similar time to many more epochs of \(L_1\).
- For Darcy flow, solver-integrated methods have less extreme overhead because
  Darcy flow is steady-state and does not require long temporal discretization.
- For Navier--Stokes, full solver-integrated adversarial training may require
  hundreds or thousands of hours, so only limited training may be feasible.

The paper should explicitly state that the comparison includes solver cost and
therefore already accounts for the disadvantage of using the solver.

## PDE Cases

The three main PDE cases are chosen because they are canonical examples from
the Fourier Neural Operator literature:

- 1D Burgers' equation: time-dependent, initial-condition-driven.
- 2D Darcy flow: steady-state coefficient-field-to-solution map.
- Navier--Stokes: time-dependent and computationally expensive.

Darcy flow is much cheaper for differentiable solver integration because it is
steady-state. Burgers and Navier--Stokes require temporal discretization; the
solver must store many intermediate states during backpropagation, which
increases memory and time dramatically.

## Physics Loss Baseline

Physics loss uses PDE residuals computed from predicted fields, often through
finite-difference approximations. This is related to PINN-style residual
regularization.

For Burgers and Navier--Stokes in this setup, physics loss may be impractical
because the model input/output contains only selected temporal frames rather
than dense time-neighboring states needed for stable time-derivative
estimation. Darcy flow is more suitable because it is steady-state.

A key argument:

- Solver outputs contain at least as much physical information as a residual
  loss in these experiments.
- A low physics residual does not always guarantee closeness to the true solver
  output, especially when PDE constraints admit multiple or poorly constrained
  solutions.
- Therefore, direct solver-target discrepancy can be more informative than
  residual-only physics loss.

## Theoretical-Experimental Link

The theory predicts that for very small \(\varepsilon\), the true adversarial
loss increase should align with

$$
D E_\theta[u]^*E_\theta(u),
$$

or in finite-dimensional notation:

$$
J_E(u)^\top e.
$$

Experiments should check:

- cosine similarity between adversarial perturbations and the
  Jacobian-error direction;
- cosine similarity between adversarial perturbations and the top singular
  vector of \(J_E(u)\);
- correlation between true adversarial loss increase and
  \(\|J_E(u)^\top e\|\);
- correlation between true adversarial loss increase and the spectral norm
  \(\|J_E(u)\|_{p\to q}\);
- how these correlations change as \(\varepsilon\) decreases.

Expected result:

- As \(\varepsilon\to 0\), the Jacobian-error proxy should become more aligned
  with the true adversarial loss increase.
- The spectral norm is still a valid sensitivity metric, but it is less
  directly tied to final error maximization.

## Extra Experiments

Additional experimental threads:

- Compare \(p,q\) choices. \(p=q=2\) appears most physically reasonable and
  stable. Sparse or max-norm constraints can create unrealistic spike
  perturbations.
- Test differentiable output warping/alignment losses for outputs that are
  similar up to shifts, stretching, or local deformation. Early observation:
  warping did not produce a large improvement.
- For Navier--Stokes, perform limited adversarial training because full
  experiments are too expensive, but report generalization loss reduction when
  available.

## Paper Contributions

1. Define generalization and robustness for neural operators in a
   solver-integrated PDE setting, using operator-level language and practical
   finite-dimensional discretizations.
2. Show that the correct adversarial target for PDE regression should optimize
   solver-model discrepancy at the perturbed input, not only model output
   change or a fixed solver target.
3. Develop solver-integrated adversarial attacks and adversarial training, then
   demonstrate improved generalization and robustness on Burgers, Darcy flow,
   and limited Navier--Stokes experiments.
4. Connect adversarial loss increase to the Jacobian-error proxy
   \(D E_\theta[u]^*E_\theta(u)\), explaining why it can be more faithful and
   cheaper than pure spectral-norm sensitivity.

## Related Work To Cite

### Direct adversarial robustness and training for neural operators

- Fourier Neural Operator for Parametric Partial Differential Equations
  \citep{li2021fourier}: foundational FNO reference and the source of the
  canonical Burgers, Darcy flow, and Navier--Stokes benchmark style.
- Evaluating the Adversarial Robustness for Fourier Neural Operators
  \citep{adesoji2022evaluating}: direct prior work on adversarial robustness
  for FNO. It perturbs inputs under norm constraints and compares the FNO
  output with PDE solver output. This is the most direct baseline for the
  adversarial attack/evaluation part.
- StablePDENet: Enhancing Stability of Operator Learning for Solving
  Differential Equations \citep{huang2026stablepdenet}: direct adversarial
  training/stability-aware neural-operator paper. It formulates operator
  learning as a min-max problem against worst-case input perturbations and
  emphasizes fidelity under adversarial perturbations.
- RAMS: Residual-based adversarial-gradient moving sample method for
  scientific machine learning in solving PDEs \citep{ouyang2025rams}: uses an
  adversarial gradient direction to move samples toward high PDE residuals and
  includes operator-learning experiments. This is highly relevant to adaptive
  sampling and adversarial-gradient data generation, although its target is a
  residual rather than the solver-model discrepancy \(L_3\).
- Adversarial Vulnerabilities in Neural Operator Digital Twins
  \citep{roy2026vulnerabilities}: shows that neural-operator digital twins can
  be vulnerable to sparse, physically plausible perturbations, using
  gradient-free differential evolution attacks and Jacobian-style diagnostics.
- Beyond Uniform Sampling: Synergistic Active Learning and Input Denoising for
  Robust Neural Operators \citep{roy2026beyond}: combines attack-driven active
  learning with input denoising to improve robustness of neural operators,
  including Burgers-style settings.

### Brown/Karniadakis-related operator-learning line

- I did not find a clearly George Em Karniadakis-authored paper whose central
  title/claim is exactly adversarial training for neural operators. Do not
  overstate this in the paper.
- Learning nonlinear operators via DeepONet based on the universal
  approximation theorem of operators \citep{lu2021deeponet}: foundational
  DeepONet reference from Lu, Jin, Pang, Zhang, and Karniadakis.
- A comprehensive and fair comparison of two neural operators based on FAIR
  data \citep{lu2021faircomparison}: Karniadakis/coauthor paper comparing
  DeepONet and FNO, including robustness/noisy-data behavior. Useful for the
  claim that noise sensitivity and robustness are already recognized issues in
  operator learning.
- Physics-Informed Deep Neural Operator Networks
  \citep{goswami2022physicsinformed}: Karniadakis/coauthor review/reference for
  physics-informed DeepONet and neural operators.
- Physics-informed neural networks \citep{raissi2019physics}: PINN baseline
  reference for PDE residual or physics-loss discussion.
- Physics-Informed Neural Operator for Learning Partial Differential Equations
  \citep{li2021pino}: core PINO reference; useful when comparing direct
  solver-target discrepancy against PDE residual or physics-loss objectives.

### Peripheral or optional related citations

- Adversarial Autoencoders in Operator Learning \citep{enyeart2024adversarial}:
  studies adversarial additions to DeepONet/Koopman autoencoder-style operator
  learning. It is related through the word "adversarial" and operator learning,
  but it is not primarily about PDE-solver adversarial robustness.
- Towards Universal Solvers: Using PGD Attack in Active Learning to Increase
  Generalizability of Neural Operators as Knowledge Distillation from Numerical
  PDE Solvers \citep{sun2025universal}: very close to solver-supervised
  adversarial active learning. Treat as an optional self/prior-work citation,
  depending on whether the final submission should cite prior work by the same
  author.

### Positioning sentence

The paper should position itself as follows: prior work evaluates adversarial
robustness, proposes stability-aware adversarial training, moves samples by
adversarial residual gradients, or improves robustness through active learning
and denoising. The missing piece is a solver-integrated attack/training
framework for PDE regression in which the true attack target is
\(\|F_\theta(u+\delta)-G(u+\delta)\|\), with explicit analysis of how deeply
the solver participates in the forward and backward loop.
