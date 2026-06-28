# Structured Research Notes

Working title:

**Solver-Integrated Adversarial Attacking and Training of Neural Operators**

These notes organize the core ideas, claims, definitions, experimental
observations, and writing points for the paper. The main purpose is to preserve
the research logic clearly so that each item can later be turned into
Introduction, Method, Theory, Experiments, and Discussion paragraphs.

## 1. Central Problem

The paper studies neural operators for PDE solution maps in a special setting:

- A trusted numerical solver already exists.
- A neural operator is trained as a fast surrogate for the solver.
- Both the neural operator and the numerical solver map input functions to
  output functions.
- The key question is how to define and evaluate generalization and robustness
  when both a learned model and a solver are present.

The setting is different from standard adversarial robustness because:

- The task is regression, not classification.
- The output is a continuous field or function, not a discrete label.
- A small input perturbation should generally change the true output.
- Therefore, one cannot assume that the target label/output remains unchanged.
- The solver provides a physically meaningful oracle for the perturbed input.

## 2. Operator-Level Formulation

The theory should first be written at the function-space/operator level.

Let:

$$
F_\theta:\mathcal{X}\to\mathcal{Y}
$$

be the learned neural operator, and let:

$$
G:\mathcal{X}\to\mathcal{Y}
$$

be the numerical solver or oracle.

In implementation, these operators are discretized as high-dimensional
nonlinear maps between vectors or tensors. For example, the input and output
may be grids with thousands of degrees of freedom. However, the mathematical
language should emphasize that the underlying objects are functions and
operators between function spaces.

The paper should consistently use two complementary languages:

- Operator/function-space language: functions, solution maps, Frechet
  derivatives, adjoint derivatives, induced operator norms.
- Finite-dimensional linear algebra language: vectors, nonlinear maps,
  Jacobians, spectral norms, singular vectors, gradients.

The finite-dimensional version is the discretized implementation of the
operator-level theory.

### 2.1 Functional-Analytic Language to Use Consistently

The paper should treat all primary mathematical objects as functions and
operators first.

Use the following hierarchy:

- Input object: an input function \(a\in\mathcal{X}\), such as an initial
  condition, coefficient field, source term, forcing field, or boundary data.
- Output object: a solution function \(u\in\mathcal{Y}\), such as a spatial
  field or a space-time field.
- Neural model: a nonlinear operator
  \(F_\theta:\mathcal{X}\to\mathcal{Y}\).
- Solver/oracle: a numerical PDE solution operator
  \(G:\mathcal{X}\to\mathcal{Y}\).
- Error operator:
  \(E_\theta=F_\theta-G:\mathcal{X}\to\mathcal{Y}\).
- Error function/field at input \(a\):
  \(e_\theta[a]=E_\theta(a)=F_\theta(a)-G(a)\in\mathcal{Y}\).
- Perturbation: a function \(\eta\in\mathcal{X}\), not just a vector.
- Perturbation budget:
  \(\|\eta\|_{\mathcal{X},p}\le\varepsilon\).
- Output discrepancy:
  \(\|e_\theta[a+\eta]\|_{\mathcal{Y},q}\).
- Energy language: the energy of a perturbation or output error should mean a
  function-space norm or squared norm, often an integral over the spatial or
  space-time domain.

The local linearization should be written as a Frechet derivative:

$$
e_\theta[a+\eta]
=
e_\theta[a]+D E_\theta[a](\eta)+o(\|\eta\|_{\mathcal{X}}).
$$

Here:

$$
D E_\theta[a]:\mathcal{X}\to\mathcal{Y}
$$

is a bounded linear operator, the operator-level analogue of the Jacobian
matrix.

The adjoint derivative is:

$$
D E_\theta[a]^*:\mathcal{Y}\to\mathcal{X}.
$$

For the squared Hilbert loss

$$
\Phi(a)=\frac{1}{2}\|e_\theta[a]\|_{\mathcal{Y}}^2,
$$

the first variation is:

$$
D\Phi[a](\eta)
=
\langle e_\theta[a],D E_\theta[a](\eta)\rangle_{\mathcal{Y}}
=
\langle D E_\theta[a]^*e_\theta[a],\eta\rangle_{\mathcal{X}}.
$$

Therefore the function-space version of the Jacobian-error direction is:

$$
D E_\theta[a]^*e_\theta[a].
$$

After discretization, this becomes:

$$
J_E(x)^\top e_\theta(x).
$$

This is exactly the bridge between the two languages:

| Functional/operator language | Discrete linear algebra language |
|---|---|
| function \(a\in\mathcal{X}\) | vector \(x\in\mathbb{R}^n\) |
| output function \(u\in\mathcal{Y}\) | vector/tensor \(y\in\mathbb{R}^m\) |
| nonlinear operator \(F_\theta\) | nonlinear map \(f_\theta:\mathbb{R}^n\to\mathbb{R}^m\) |
| solver operator \(G\) | solver map \(g:\mathbb{R}^n\to\mathbb{R}^m\) |
| error operator \(E_\theta=F_\theta-G\) | error map \(E_\theta=f_\theta-g\) |
| error function \(e_\theta[a]=E_\theta(a)\) | error vector \(e_\theta(x)=E_\theta(x)\) |
| Frechet derivative \(D E_\theta[a]\) | Jacobian matrix \(J_E(x)\) |
| adjoint derivative \(D E_\theta[a]^*\) | transpose/conjugate transpose \(J_E(x)^\top\) |
| induced norm \(\|D E_\theta[a]\|_{p\to q}\) | matrix/operator norm \(\|J_E(x)\|_{p\to q}\) |
| \(p=q=2\) Hilbert setting | spectral norm / top singular vector |
| \(D E_\theta[a]^*e_\theta[a]\) | \(J_E(x)^\top e_\theta(x)\) |

Important writing rule:

- The theory sections should use the left column first.
- The implementation and experiment sections may use the right column.
- Whenever the paper says "Jacobian", it should clarify that this is the
  discretized representation of a Frechet derivative of an operator between
  function spaces.

## 3. Static Generalization

Generalization is defined as a static discrepancy at a fixed input function.

For an input \(u\), compare the model output and solver output at the same
input:

$$
\mathcal{L}_{\mathrm{gen}}(u)
= \|F_\theta(u)-G(u)\|_q.
$$

This measures whether the neural operator matches the solver at the point
\(u\). It is not a local perturbation measure. The input is fixed.

Important writing point:

- Generalization is about the model-solver discrepancy at a point.
- Robustness is about how this discrepancy behaves in a neighborhood of the
  point.

## 4. Local Robustness

Robustness is local and dynamic. It studies the behavior of the solver-model
discrepancy under a norm-bounded perturbation of the input function.

The perturbation is constrained by:

$$
\|\delta\|_p \le \varepsilon.
$$

The output discrepancy is measured with a \(q\)-norm.

The radius \(\varepsilon\) should be small enough to preserve a local
interpretation. Large perturbations may become distribution shift or data
augmentation rather than local robustness.

The choice of \(p\) and \(q\) matters:

- \(p=q=2\) is the most physically reasonable and stable in the observed
  experiments.
- \(p=q=2\) aligns with standard singular-value and spectral-norm intuition.
- \(p=\infty\), \(p=1\), or sparse-style constraints can create sharp,
  pointwise spike perturbations.
- Such spike perturbations may be mathematically valid but physically
  unnatural for PDE inputs.

## 5. Error Operator

Define the model-solver error operator:

$$
E_\theta=F_\theta-G:\mathcal{X}\to\mathcal{Y}.
$$

At a particular input function \(u\), the error function is:

$$
e_\theta[u]=E_\theta(u)=F_\theta(u)-G(u)\in\mathcal{Y}.
$$

This is the central object for the paper.

The true robustness question is not simply how much \(F_\theta(u)\) moves, but
how much the error operator changes or grows under input perturbation.

## 6. Three Robustness Metrics

### 6.1 True Adversarial Error

The primary robustness metric is the worst final model-solver error in an
input perturbation ball:

$$
R_{\mathrm{adv}}(u)
=
\max_{\|\delta\|_p\le\varepsilon}
\|F_\theta(u+\delta)-G(u+\delta)\|_q.
$$

This is the most important attack target in the paper.

It uses the solver at the perturbed input \(u+\delta\), so the target changes
when the input changes. This is essential for PDE regression.

### 6.2 Local Operator Sensitivity

Linearize the error operator around \(u\):

$$
e_\theta[u+\delta]
\approx
e_\theta[u]+D E_\theta[u](\delta).
$$

Here \(D E_\theta[u]\) is the Frechet derivative of the error operator. In the
finite-dimensional discretized setting, it is the Jacobian matrix of the error
map.

An induced local sensitivity metric is:

$$
\|D E_\theta[u]\|_{p\to q}
=
\sup_{\|\delta\|_p\le 1}
\|D E_\theta[u](\delta)\|_q.
$$

For \(p=q=2\), this becomes the largest singular value of the error Jacobian.
The associated singular vector gives the local direction that maximizes the
change in the error.

Important caveat:

- This metric maximizes the change in error.
- It does not directly maximize the final error.
- If the existing error function \(e_\theta[u]\) is nonzero, a direction that causes a
  large error change may still lead to a final error vector with modest norm.

Vector intuition:

- The initial error is \(A\).
- The local error change is \(\Delta A\).
- The spectral-norm metric maximizes \(\|\Delta A\|\).
- The true adversarial loss maximizes \(\|A+\Delta A\|\).
- These are not the same objective.

### 6.3 Jacobian-Error Proxy

For small perturbations and an \(L_2\)-style squared loss:

$$
\|e_\theta[u+\delta]\|_2^2
\approx
\|e_\theta[u]\|_2^2
+2\langle D E_\theta[u](\delta), e_\theta[u]\rangle.
$$

In finite dimensions, if \(J_E(x)\) is the Jacobian of the discretized error
map and \(e=e_\theta(x)\), the first-order loss-increasing direction is
controlled by:

$$
J_E(x)^\top e.
$$

At the operator level, this is:

$$
D E_\theta[u]^*e_\theta[u],
$$

where \(D E_\theta[u]^*\) is the adjoint derivative.

This proxy is important because:

- It is closer to the true adversarial loss increase than the top singular
  vector when \(\varepsilon\) is very small.
- It directly incorporates the existing error direction.
- It is cheaper than a full adversarial attack.
- It is cheaper than computing a full singular-value decomposition.

The theory predicts that as \(\varepsilon\to 0\), the true adversarial loss
increase should align more strongly with \(D E_\theta[u]^*e_\theta[u]\).

## 7. Correspondence Between Metrics and Directions

The three robustness quantities have corresponding perturbation directions:

- Induced operator norm or spectral norm:
  - Quantity: \(\|D E_\theta[u]\|_{p\to q}\)
  - Direction: top singular vector or induced-norm maximizer.
  - Meaning: maximizes local error change.
- Jacobian-error proxy:
  - Quantity: \(\|D E_\theta[u]^*e_\theta[u]\|\)
  - Direction: adjoint-Jacobian applied to the current error.
  - Meaning: first-order direction for increasing final error.
- True adversarial attack:
  - Quantity: \(\max_{\|\delta\|_p\le\varepsilon}\|e_\theta[u+\delta]\|_q\)
  - Direction: optimized adversarial perturbation.
  - Meaning: directly maximizes final perturbed solver-model discrepancy.

Expected experimental relation:

- The adversarial perturbation is more similar to the Jacobian-error direction
  than to the top singular vector when \(\varepsilon\) is small.
- The true attack loss increase has higher correlation with the
  Jacobian-error proxy than with the pure spectral norm in the small-epsilon
  regime.

### 7.1 Existing Robustness Definitions in Related Work

The related literature uses several different robustness definitions:

- Classical adversarial-learning papers usually use attacked loss, robust
  accuracy, or robust risk:
  \[
  \mathbb{E}\max_{\|\delta\|\le\varepsilon}
  \ell(f_\theta(x+\delta),y).
  \]
  This is mostly a classification setting where the label \(y\) is assumed to
  stay fixed under small perturbations.
- FNO robustness evaluation and neural-operator digital-twin attack papers
  usually report prediction error after attack, such as attacked MSE or
  relative \(L_2\) error against solver/simulation outputs.
- StablePDENet uses a stability view: small input perturbations should produce
  bounded output changes. It connects this to a bounded Frechet derivative and
  reports a Jacobian spectral norm. This is a spectral-norm / local-Lipschitz
  robustness metric.
- AT-PINN/WbAR, PIAT, RAMS, and related residual-adversarial methods use PDE
  residuals to find failure regions or informative samples. Their robustness
  notion is residual/failure-region oriented, not solver-consistent
  teacher-student error.

Important conclusion:

- Some papers define robustness by attacked loss increase.
- Some papers define stability/robustness by spectral norm or Frechet
  derivative norm.
- Some papers report attacked relative error.
- Some papers only show improved generalization/prediction accuracy.
- The reviewed papers do not appear to use the Jacobian-error proxy
  \[
  D E_\theta[u]^*e_\theta[u]
  \]
  as a central robustness metric.

This paper's metric is therefore positioned between finite-budget adversarial
loss and local spectral-norm stability. It is the first-order direction for
increasing the current solver-model error, not merely the direction that
maximizes output change or error-field change.

## 8. Attack Loss Taxonomy

The paper compares several attack objectives.

### 8.1 Model-Only Change Loss \(L_1\)

$$
L_1(u,\delta)=\|F_\theta(u+\delta)-F_\theta(u)\|.
$$

This objective only makes the neural operator output move.

Problem:

- The solver output may move in the same direction.
- Therefore, the true model-solver error may not increase.
- This can be inefficient for PDE regression.

Important implementation detail:

- \(L_1\) needs a nonzero random start.
- If \(\delta=0\), then \(L_1=0\), and the attack can get stuck.

### 8.2 Fixed Solver Target Loss \(L_2\)

$$
L_2(u,\delta)=\|F_\theta(u+\delta)-G(u)\|.
$$

This compares the perturbed model output with the unperturbed solver target.

It uses the solver in the forward pass at the original input, but the solver
target does not change during the attack.

Problem:

- It assumes the true output stays fixed even though the PDE input changes.
- This is more natural in classification than in PDE regression.
- It includes the initial error at \(\delta=0\), but it does not represent the
  true perturbed PDE solution.

Variants:

- Compute \(G(u)\) once and keep it fixed for all attack steps.
- Recompute a solver target periodically or at every step.
- Use stop-gradient through the solver.

### 8.3 Dictionary-Based Approximate Solver Target

Dictionary versions use precomputed input-output pairs and approximate the
solver output at \(u+\delta\) by a nearest neighbor:

$$
\|F_\theta(u+\delta)-y_{\mathrm{nearest}}(u+\delta)\|.
$$

Variants include different dictionary sizes, such as:

- \(N=200\)
- \(N=2000\)
- \(N=20000\)

Problem:

- The solver is not truly evaluated at the current perturbed input.
- The attack depends on approximation quality of the dictionary.
- The solver is not used in the backward pass.

### 8.4 True Solver-Integrated Loss \(L_3\)

$$
L_3(u,\delta)
=
\|F_\theta(u+\delta)-G(u+\delta)\|.
$$

This is the true loss for the paper.

It compares model and solver at the same perturbed input.

It can involve the solver in:

- Forward pass only, if gradients through the solver are stopped.
- Forward and backward pass, if the solver is differentiable and gradients
  flow through \(G(u+\delta)\).

This is the most deeply solver-integrated objective.

Main claim:

- \(L_3\) is the correct target loss for adversarial attack and adversarial
  training in solver-supervised PDE regression.

## 9. Solver Integration Levels

Solver integration is not binary. It has degrees.

Possible levels:

- No solver during attack:
  - Example: model-only \(L_1\).
  - The solver may only have been used to generate the training data.
- Precomputed dictionary:
  - Solver outputs are generated offline.
  - The attack uses nearest-neighbor approximations.
- Forward-only fixed solver target:
  - Compute \(G(u)\), keep it fixed during attack.
  - No solver gradient with respect to \(\delta\).
- Forward recomputed solver target:
  - Compute \(G(u+\delta)\) during the attack.
  - May still stop gradients through the solver.
- Full differentiable solver integration:
  - Compute \(G(u+\delta)\) during the attack.
  - Backpropagate through both \(F_\theta\) and \(G\).
  - This is the strongest integration and the most expensive.

Main conclusion:

- Stronger solver integration tends to produce more meaningful and efficient
  attacks.
- Stronger solver integration also increases time and memory cost.

## 10. Why Surrogate Losses Can Fail

The model and solver are often highly correlated because the model was trained
to imitate the solver.

If an attack only maximizes the movement of \(F_\theta(u+\delta)\), the solver
may move in the same direction:

$$
F_\theta(u+\delta)-F_\theta(u)
\quad\text{large, but}\quad
F_\theta(u+\delta)-G(u+\delta)
\quad\text{small.}
$$

Therefore:

- A surrogate loss can increase strongly.
- The true \(L_3\) loss may not increase much.
- The attack can be inefficient.

The correct adversarial direction should enlarge the difference between the
model and solver, not merely the change in the model output.

This is one of the central messages of the paper.

## 11. Attack Optimization Methods

The attack optimization studies two axes:

### 11.1 Add vs. Replace

Add:

- Each step adds a small update to the current perturbation.
- Similar to standard iterative PGD.
- It may take many steps to reach the boundary of the epsilon ball.

Replace:

- Each step directly replaces the perturbation with the new direction.
- It can reach the epsilon-ball boundary immediately.
- It can converge faster when the maximum lies near the boundary.

### 11.2 Raw vs. Steepest

Raw:

- Uses the raw gradient direction.

Steepest:

- Uses the steepest ascent direction under the chosen norm geometry.
- Depends on the \(p\)-norm constraint.

Compared variants:

- Raw add
- Raw replace
- Steepest add
- Steepest replace

Observed pattern:

- On Burgers and Darcy flow, steepest replace appears very efficient.
- It reaches the boundary quickly and then optimizes on the boundary.
- Steepest add can be slow because it may spend many steps just reaching the
  boundary.
- On Navier-Stokes, steepest replace may be less effective because the loss
  landscape is more nonlinear and the optimal perturbation structure is more
  complex.

## 12. Adversarial Training

The paper extends solver-integrated attack to adversarial training.

For each training input \(u\), first find an adversarial perturbation:

$$
\delta^*(u)
\approx
\arg\max_{\|\delta\|_p\le\varepsilon}
\|F_\theta(u+\delta)-G(u+\delta)\|_q.
$$

Then train the model on the attacked input with the solver target:

$$
\min_\theta
\mathbb{E}_{u}
\left[
\ell(F_\theta(u+\delta^*(u)),G(u+\delta^*(u)))
\right].
$$

The perturbation budget \(\varepsilon\) is sampled from a range rather than
fixed:

- This creates both small and large perturbations.
- It improves coverage of the local neighborhood.
- It makes adversarial training data more diverse.

Main result:

- \(L_3\)-based adversarial training improves both generalization and
  robustness more than less solver-integrated methods.

## 13. Training Baselines

The paper compares several training methods.

### 13.1 Clean Training

Train on the original dataset without adversarial or random perturbations.

### 13.2 Random Clean Augmentation

Randomly perturb the input \(u\), but keep the original target \(G(u)\).

Problem:

- Physically inconsistent.
- If the PDE input changes, the output should usually change too.

### 13.3 Random Solver Augmentation

Randomly perturb the input \(u\), then recompute the target using the solver:

$$
G(u+\delta_{\mathrm{rand}}).
$$

This is physically meaningful, but not adaptive.

Observation:

- It can be a strong baseline.
- On Burgers, it can become the second-best method.
- However, it may overfit or degrade after several hours because the
  perturbation distribution is fixed.

### 13.4 Solver-Integrated Adversarial Training

Generate \(\delta\) through adversarial attack against the current model.

Advantage:

- The perturbations adapt to the current weaknesses of the model.
- As the model becomes more robust, the attack discovers harder perturbations.

Observed phenomenon:

- During \(L_3\)-based adversarial training, perturbations become increasingly
  high-frequency over epochs.
- This suggests that low-frequency weaknesses are corrected first.
- Later attacks must find more subtle/high-frequency directions to separate the
  model from the solver.

## 14. Robustness During Training

Observed trend:

- As adversarial training proceeds, the model becomes harder to attack.
- With the same epsilon budget and the same attack setup, the attack-induced
  loss increase becomes smaller.
- This indicates improved local robustness.

This is an important empirical story:

- The model's evasion capability improves.
- The attack must search for more complex perturbations.
- The discovered perturbations become more high-frequency.

## 15. Wall-Clock-Time Comparison

The paper should compare methods by wall-clock time, not only by epoch count.

Reason:

- Methods have very different per-epoch costs.
- Solver-integrated methods can be much slower per epoch.
- Comparing epoch counts would be unfair.

Important claim:

- Even after accounting for solver overhead, solver-integrated methods can be
  more efficient in terms of robustness/generalization achieved per wall-clock
  time.

Examples to express carefully:

- In Burgers, 1000 epochs of \(L_3\) adversarial training can take time
  comparable to many more epochs of \(L_1\)-based training.
- In Darcy flow, the runtime gap between methods is much smaller.
- Full Navier-Stokes adversarial training may require hundreds or thousands of
  hours, making complete experiments infeasible under current resources.

## 16. Computational Cost and Memory

The main drawback of solver-integrated adversarial training is computational
cost.

Costs come from:

- Running the solver many times during attack/training.
- Differentiating through the solver.
- Storing intermediate solver states for backpropagation.
- Fine temporal and spatial discretization.

Time-dependent PDEs are especially expensive:

- Burgers and Navier-Stokes require temporal evolution.
- Backpropagation through time requires storing many intermediate states.
- This greatly increases memory and runtime.

Darcy flow is cheaper:

- It is steady-state.
- It does not require long temporal discretization.
- It is closer to solving one spatial PDE rather than a long time evolution.

Important correction:

- The heavy temporal-discretization cost applies strongly to Burgers and
  Navier-Stokes.
- Darcy flow is comparatively cheaper because it is steady-state.

## 17. PDE Benchmarks

The paper uses or discusses three canonical PDE settings:

- 1D Burgers' equation:
  - Time-dependent.
  - Initial-condition-to-solution style.
  - Full adversarial training experiments are feasible but costly.
- 2D Darcy flow:
  - Steady-state.
  - Coefficient-field-to-solution map.
  - Solver integration is cheaper than time-dependent systems.
  - Suitable for physics-loss baseline.
- Navier-Stokes:
  - Time-dependent.
  - Highly expensive for full solver-integrated adversarial training.
  - Limited adversarial training can still show generalization loss reduction.

Reason for choosing these PDEs:

- They are canonical examples used in the original FNO literature.
- They cover both time-dependent and steady-state PDEs.
- They include initial-condition-driven and coefficient-field-driven operator
  learning settings.

## 18. Physics Loss Baseline

StablePDENet and related physics-informed methods motivate a physics-loss
baseline.

Physics loss:

- Computes PDE residuals from the predicted field.
- Uses finite differences or automatic differentiation to estimate derivatives.
- Penalizes violation of the PDE equation.

Why physics loss is difficult for some experiments:

- Burgers and Navier-Stokes are time-dependent.
- The available model input/output may contain only selected temporal frames.
- There may not be dense neighboring time frames needed to estimate time
  derivatives reliably.
- Computing dense temporal residuals would require much more memory.

Darcy flow is more suitable:

- It is steady-state.
- Residuals can be computed from spatial derivatives of the output.

Why solver-target loss can be stronger:

- Solver outputs contain the full solution information.
- The solver output should satisfy the PDE.
- A small physics residual does not always guarantee closeness to the true
  solver output.
- PDE constraints may admit multiple or poorly constrained solutions.
- Therefore, direct solver-target discrepancy can be more informative than
  residual-only loss.

Experimental conclusion:

- Physics loss does not increase the true \(L_3\) loss as efficiently as
  directly attacking \(L_3\).
- \(L_3\)-based attack is the most effective way to find model-solver
  disagreement.

## 19. Output Warping Experiment

The paper also considers differentiable output warping/alignment.

Motivation:

- Model and solver outputs may have similar shapes but be shifted, stretched,
  or locally deformed.
- Pointwise comparison may over-penalize small misalignments.
- A differentiable warping loss could align the fields before measuring error.

Question:

- Can output warping generate sharper, more targeted, or more complex attacks?

Observation:

- Initial experiments did not show a large difference.
- Warping is an interesting extension but not a main result.

## 20. Main Experimental Conclusions

The important experimental findings are:

- Attacking the true \(L_3\) loss produces the largest increase in true
  model-solver discrepancy.
- Surrogate losses such as \(L_1\), fixed-target \(L_2\), dictionary losses, or
  physics loss may increase their own objectives without strongly increasing
  \(L_3\).
- \(L_3\)-based adversarial training gives the strongest robustness and
  generalization improvements in Burgers and Darcy flow.
- Random solver augmentation can be strong, but it is not adaptive and can
  degrade after long training.
- Solver-integrated adversarial training adapts perturbations to the current
  model weakness.
- Perturbations become more high-frequency as training progresses.
- Attack-induced loss increases become smaller as the model becomes more
  robust.
- The Jacobian-error proxy correlates better with true adversarial loss
  increase than the pure spectral norm in the small-epsilon regime.
- As \(\varepsilon\) decreases, the match between true attack loss increase and
  the Jacobian-error proxy becomes stronger.

## 21. Main Contributions

The paper's contributions can be framed as:

1. Define generalization and robustness for neural operators in a
   solver-integrated PDE regression setting.
2. Introduce the error operator \(E_\theta=F_\theta-G\) and analyze robustness
   from both operator-level and finite-dimensional perspectives.
3. Show that the correct adversarial target should be the true perturbed
   solver-model discrepancy \(L_3=\|F_\theta(u+\delta)-G(u+\delta)\|\).
4. Explain why model-only or fixed-target surrogate losses can fail in PDE
   regression because the model and solver may move together.
5. Compare solver integration levels, from no solver use to full differentiable
   solver integration.
6. Develop solver-integrated adversarial training and show improved
   generalization and robustness.
7. Connect true adversarial loss increase to the Jacobian-error proxy
   \(D E_\theta[u]^*e_\theta[u]\), showing why it is more faithful than pure
   spectral-norm sensitivity for small perturbations.
8. Study practical attack optimization choices, including raw/add,
   raw/replace, steepest/add, and steepest/replace.
9. Analyze computational tradeoffs, especially solver backpropagation cost in
   time-dependent PDEs.

## 22. Writing Emphasis

The paper should emphasize:

- This is not standard image/classification adversarial robustness.
- The PDE output is continuous and should change with the input.
- A solver is available and should be used in defining robustness.
- The central object is the model-solver discrepancy, not model output change.
- Solver integration has different depths, and deeper integration can improve
  attack/training quality.
- The best method is not free: solver-integrated adversarial training can be
  expensive in wall-clock time and memory.
- The paper is both theoretical and empirical:
  - Theory explains why the Jacobian-error proxy matters.
  - Experiments show that \(L_3\) attack/training works better.

## 23. Literature Positioning

Relevant citation categories:

- Neural operator foundations:
  - Fourier Neural Operator.
  - DeepONet.
  - PINO and physics-informed neural operators.
- Adversarial robustness for neural operators:
  - Evaluating the Adversarial Robustness for Fourier Neural Operators.
  - StablePDENet.
  - RAMS adversarial-gradient moving samples.
  - Recent neural-operator digital twin attack/defense papers.
- Karniadakis-related background:
  - PINNs.
  - DeepONet.
  - DeepONet/FNO comparison and noisy-data robustness.
  - Physics-informed Deep Neural Operator Networks.
  - Oommen, Khodakarami, Bora, Wang, and George Em Karniadakis on `adv-NO`
    for turbulent-flow super-resolution, forecasting, and sparse
    reconstruction.

Important caution:

- A George Em Karniadakis-authored `adv-NO` paper is now verified: Oommen,
  Khodakarami, Bora, Wang, and Karniadakis, "Learning Turbulent Flows with
  Generative Models: Super-resolution, Forecasting, and Sparse Flow
  Reconstruction", arXiv:2509.08752.
- That paper uses adversarial/generative/perceptual losses to improve
  turbulent-flow fields. It should not be described as PGD-style adversarial
  attack, worst-case input perturbation robustness, or solver-integrated
  adversarial training.
- RAMS is very relevant and includes Lu Lu, but should not be incorrectly
  attributed to George Em Karniadakis unless he is actually an author.

## 24. Terms To Standardize

Use these terms consistently:

- "adversarial attack", not "adversarial tag".
- "adversarial training", not "adversarial train" or "serial training".
- "solver", not "sower" or "server".
- "Darcy flow", not "Daciflow" or "Dassie Flow".
- "Burgers' equation", not "Burgess".
- "Navier-Stokes", not "Neverstock" or "Navier Stoke".
- "Jacobian", not "Yakubi".
- "spectral norm", not "spectronorm".
- "singular value" and "singular vector", not "strange value" or similar
  speech-to-text variants.
- "Frechet derivative" or "FrÃ©chet derivative" depending on LaTeX style.

## 25. Possible Paper Structure

Suggested section layout:

1. Introduction
2. Related Work
3. Problem Setting
4. Solver-Integrated Generalization and Robustness Metrics
5. Attack Objectives and Solver Integration Levels
6. Local Linearization and Jacobian-Error Proxy
7. Solver-Integrated Adversarial Training
8. Experiments
   - Benchmarks
   - Attack objective comparison
   - Optimization method comparison
   - Adversarial training comparison
   - Wall-clock and memory cost
   - Jacobian-error proxy validation
   - Optional warping and \(p,q\) studies
9. Limitations
10. Conclusion

## 26. Addendum From Code Markdown Audit

Audit source:

- Local code root:
  `<local NeuralOperatorRobustness2 path>`
- Markdown corpus inspected:
  898 Markdown files, about 8.9 MB total.
- The most important files were conclusion/audit/theory/result records such as:
  `docs/paper_conclusion_audit_and_outline_20260616.md`,
  `docs/article_structure_recommendation_20260616.md`,
  `docs/20260608_generalization_and_robustness_definitions.md`,
  `docs/loss3_original_theory_experiment_plan.md`,
  `docs/delta_loss_formula_taxonomy_20260515.md`,
  `docs/darcy_robustness_metrics_formulas_and_results_20260613.md`,
  `docs/darcy_cflow_residual_correlation_interpretation_20260615.md`,
  `2D_Darcy_FNO2d/DARCY_BINARY_ATTACK_NOTES.md`,
  `STABLEPDENET_PINO_DEEPONET_FNO_DISCUSSION_SUMMARY_20260529.md`,
  and `docs/neural_operator_adversarial_related_work_update_20260617.md`.

The code Markdown does not overturn the central story above. It sharpens the
paper's claims and adds several important cautions.

### 26.1 The Paper Should Be Framed Around Four Questions

The paper should not be presented only as "loss3 works." A stronger framing is:

1. What should be measured in solver-backed neural-operator regression?
2. Which loss should generate adversarial perturbations?
3. Which optimizer/update geometry should be used once the loss is fixed?
4. How should the generated adversarial samples be used for training?

This makes the contribution broader than a single attack objective. It becomes
a framework for solver-integrated robustness evaluation and training.

### 26.2 Separate Objective, Gradient, Update Direction, and Projection

The code notes repeatedly warn that these are different objects:

- Objective: the scalar quantity being maximized, such as \(L_1,L_2,L_3\).
- Gradient: derivative of that scalar with respect to the perturbation.
- Update direction: raw gradient, \(L_p\)-steepest direction, generalized
  power direction, binary flip score, etc.
- Projection/constraint: continuous \(p\)-ball projection or binary Hamming
  budget projection.

The paper should avoid saying an optimizer is "better" unless the target loss,
constraint geometry, step count, and comparison metric are specified.

### 26.3 Three Objective Variants Beyond \(L_1,L_2,L_3\)

The Markdown notes distinguish three variants for each base loss:

- Original endpoint objective:
  \[
  O_i^{\mathrm{orig}}(\delta)=L_i(\delta).
  \]
- Increment-ratio diagnostic:
  \[
  O_i^{\mathrm{ratio}}(\delta)
  =
  \frac{L_i(\delta)-L_i(0)}{\|\delta\|_p+\eta}.
  \]
- Regularized objective:
  \[
  O_i^{\mathrm{reg}}(\delta)
  =
  L_i(\delta)-C\|\delta\|_p.
  \]

The main finite-budget regression attack should remain the original endpoint
\(L_3\) objective. Ratio and regularized forms are useful diagnostics, but they
should not be conflated with the main endpoint attack. In particular, an
increment ratio at a finite nonzero \(\delta\) is a secant quantity, not a pure
local derivative.

### 26.4 Linearization Has Several Layers

The notes separate several related but nonidentical questions:

- True nonlinear finite-radius attack:
  \[
  \max_{\|\delta\|_p\le\varepsilon}
  \|e_\theta[u+\delta]\|_q.
  \]
- Local affine approximation:
  \[
  e_\theta[u+\delta]\approx b + A\delta,
  \quad
  b=e_\theta[u],\quad A=D E_\theta[u].
  \]
- Pure residual movement:
  \[
  \|A\delta\|.
  \]
- Endpoint residual under the local model:
  \[
  \|b+A\delta\|.
  \]
- Same-iterate gradient comparison:
  compare gradients at the same current \(\delta_k\), not at different
  optimizer trajectories.

This is important because the singular vector of \(A\) maximizes residual
movement, while \(A^\ast b\) is the first-order endpoint-loss direction at
\(\delta=0\). They are related but not the same direction.

### 26.5 Difference Jacobian Must Be Treated As A Difference

For residual robustness the relevant local operator is:

\[
J_{\mathrm{res}}(u)
=
J_{F_\theta}(u)-J_G(u).
\]

The spectral norm or singular vectors of this difference operator cannot be
obtained by subtracting the singular values of \(J_{F_\theta}\) and \(J_G\).
They must be computed from the difference operator itself, or estimated through
JVP/VJP or power-iteration style methods.

The paper should distinguish:

- \(J_{F_\theta}\): model sensitivity.
- \(J_G\): solver sensitivity.
- \(J_{F_\theta}-J_G\): residual/model-solver mismatch sensitivity.
- Subspace similarity between \(J_{F_\theta}\) and \(J_G\): whether the model's
  local linear response resembles the solver's.

### 26.6 Metric Hierarchy And Reporting Rule

The code records strongly suggest a reporting rule:

Do not say "model A is more robust" without naming the metric.

Use specific language such as:

- clean RMSE or relative \(L_2\) decreased;
- final attacked \(L_3\) decreased;
- attack loss increase decreased;
- residual \(\|J_{\mathrm{res}}^\ast e\|\) decreased;
- residual \(\sigma_1(J_{\mathrm{res}})\) decreased;
- model-solver top singular subspaces became more aligned;
- attack perturbations shifted toward higher frequencies;
- binary flip attack gain decreased.

Clean generalization and adversarial robustness are related but not identical.
A model can improve clean error without minimizing every local robustness
metric.

### 26.7 Loss3 Needs A Precise, Not Absolute, Claim

The evidence supports \(L_3\)-based training as the strongest overall method on
the main comparable Burgers and Darcy protocols. However, the notes warn
against claiming that \(L_3\) wins every scalar metric.

Important nuance:

- In Darcy, some pure or model-only spectral norm diagnostics can favor another
  model or give the wrong story.
- The correct robustness operator is the residual operator
  \(J_{F_\theta}-J_G\), not \(J_{F_\theta}\) alone.
- Final Darcy CFlow records support loss3 through smaller residual operator
  metrics, smaller final attacked loss, and stronger model-solver subspace
  alignment.
- Some earlier or auxiliary tables are diagnostic only and should not be mixed
  with strict same-manifest, same-protocol results.

Recommended wording:

"Loss3 is the strongest model under the main solver-consistent attack and
training protocols, but different diagnostics illuminate different mechanisms.
Model-only sensitivity is not the same as residual robustness."

### 26.8 Darcy Binary Perturbations Are Not Ordinary Continuous PGD

Darcy flow uses a binary coefficient field:

\[
a(i)\in\{3,12\}.
\]

Therefore the perturbation is naturally a flip mask with a Hamming budget, not
a free continuous \(p\)-norm perturbation. The notes define a binary attack
where a mask chooses which pixels flip from 3 to 12 or 12 to 3.

Consequences:

- "Add" means keep old flips and append new flips.
- "Replace" means discard the old flip mask and choose a new full mask from
  current scores.
- Raw and steepest variants can become similar because each valid flip has the
  same magnitude.
- The binary attack is a combinatorial optimization problem, so greedy
  gradient-based flip scores are heuristics, not exact exhaustive search.

The paper should not describe Darcy binary experiments as if they were the same
continuous PGD geometry used for Burgers or Navier-Stokes.

### 26.9 Epsilon Jitter Correction

The user-level idea that epsilon can be randomized is valid, but the code audit
shows it is not universal across all runs.

Important correction:

- Burgers local adversarial-training runs used epsilon jitter such as
  \(0.75\) to \(1.25\) in some recorded experiments.
- Darcy loss1/loss2/loss3/physics adversarial runs inspected in the audit used
  `eps_jitter_low = eps_jitter_high = 1.0`, i.e. a fixed budget multiplier.
- The current source default may differ from the historical run configuration.
- Random-source Darcy baselines do not run an adversarial optimizer; they draw
  random binary perturbations under their own random-field/kernel/flip-fraction
  logic.

Therefore epsilon randomization should be presented as an optional budget
sampling mechanism, not as an inherent property of \(L_1,L_2,L_3\).

### 26.10 Optimizer Conclusions Are Geometry-Dependent

The code notes add several optimizer subtleties:

- Reaching the constraint boundary is not the same as converging in loss.
  Replacement/generalized-power methods can hit the boundary at step 1 and
  still continue improving by rotating along the boundary.
- In \(p=2\) settings, raw-replace and steepest-replace can collapse to nearly
  the same behavior after normalization.
- In Burgers and binary Darcy, replace-style methods are often very fast.
- In 2D Navier-Stokes recurrent settings, additive \(L_p\)-steepest PGD can be
  stronger because the objective is more nonlinear, path-dependent, and
  frequency-filtered.
- Smaller epsilon makes the behavior more local/linear; larger epsilon exposes
  finite-radius nonlinear effects.

The paper should present optimizer results as an ablation, not as a universal
claim that one optimizer always dominates.

### 26.11 Small-Epsilon Theory Versus Finite-Budget Attack

For squared residual loss, the local expansion is:

\[
\|b+A\delta\|_2^2-\|b\|_2^2
\approx
2 b^\top A\delta + \|A\delta\|_2^2.
\]

The first-order term is controlled by:

\[
A^\top b
=
J_{\mathrm{res}}^\ast e_\theta[u].
\]

The second-order/high-gain term is controlled by singular values of
\(A=J_{\mathrm{res}}\).

Implication:

- As \(\varepsilon\to0\), attack loss increase should align more strongly with
  \(J_{\mathrm{res}}^\ast e\).
- For finite multi-step attacks, residual spectral norm can also become
  predictive because the optimizer can exploit high-gain residual singular
  directions.

The Darcy epsilon-sweep notes support this distinction: very small budgets move
the correlation toward residual JT/error-style quantities, while larger budgets
make residual singular directions more visible.

### 26.12 Physics Loss And Fourier Loss Caveats

StablePDENet and PINO-style methods use physics residuals. The audit clarifies
how this differs from the present paper:

- StablePDENet attacks a physics residual, not the solver-model discrepancy
  \(\|F_\theta(u+\delta)-G(u+\delta)\|\).
- It can generate adversarial inputs without solver labels, but the attack
  target and the evaluation target are not the same as in solver-consistent
  \(L_3\).
- Physics residual losses are easiest for coordinate-query DeepONet or dense
  time-output settings.
- For final-snapshot FNO outputs on time-dependent PDEs, the time derivative
  needed by a PDE residual may be unavailable, so a full physics loss can be
  ill-defined or expensive.
- Darcy is steadier and more compatible with a spatial physics residual.

The Fourier-domain note also adds:

- Physical-space \(L_2\)/MSE and unweighted Fourier energy are equivalent up to
  normalization by Parseval/Plancherel.
- Fourier loss is useful only when frequency-weighting or frequency-specific
  behavior is desired.
- Ordinary pointwise RMSE/relative \(L_2\) should remain the primary loss unless
  the paper explicitly studies frequency-weighted objectives.

### 26.13 Related Work Additions From The Code Notes

The related-work notes identify several citeable categories:

- Direct neural-operator adversarial robustness:
  Adesoji and Chen on FNO adversarial robustness.
- Physics-residual adversarial training for operator learning:
  StablePDENet.
- Residual-based adversarial sample movement:
  RAMS.
- Gradient-free attacks on neural-operator digital twins:
  Roy et al.
- GAN-style/adversarial loss for neural operators:
  Generative Adversarial Neural Operators.
- Brown/Karniadakis-related adversarial neural operator:
  Oommen, Khodakarami, Bora, Wang, and George Em Karniadakis, "Learning
  Turbulent Flows with Generative Models: Super-resolution, Forecasting, and
  Sparse Flow Reconstruction", which uses an adversarially trained neural
  operator (`adv-NO`) for turbulent flow.
- Uncertainty-driven active operator learning:
  Winovich, Daneker, Lu Lu, and Guang Lin, "Active operator learning with
  predictive uncertainty quantification for partial differential equations".
  This uses deterministic predictive UQ, trained with Gaussian NLL, to choose
  high-uncertainty PDE instances for additional solver labels. Its evaluation
  is mainly clean prediction error, calibration, runtime, and data efficiency,
  not adversarial robustness.
- Multi-resolution active learning for FNO:
  "Multi-Resolution Active Learning of Fourier Neural Operators" chooses both
  PDE inputs and solver resolution through a utility/cost acquisition rule. It
  is highly relevant to solver-cost-aware data acquisition, but it is not a
  worst-case perturbation attack or adversarial-training method.
- General adversarial training:
  Goodfellow et al. for FGSM, Madry et al. for PGD robust optimization, TRADES,
  and AutoAttack-style evaluation caution.
- PINN adversarial training:
  AT-PINN / white-box adversarial residual-region refinement papers are related
  because they adversarially sample PDE residual failure regions, but they are
  PINN/residual-focused rather than solver-model-discrepancy-focused.

Important positioning:

The Brown/Karniadakis `adv-NO` paper is adversarial in the generative-loss
sense. It should not be described as the same as PGD-style worst-case input
perturbation robustness or solver-integrated adversarial training.

The active-learning papers should be used to motivate expensive solver labels
and data-efficient acquisition. They do use the solver after selecting samples,
because selected PDE instances need ground-truth solutions, but the acquisition
criterion is uncertainty or utility/cost rather than direct model-solver
discrepancy under a perturbation budget.

### 26.14 Evidence Protocol And Comparability

The code notes repeatedly warn that experiment records should not be mixed
across incompatible protocols.

Use only comparable evidence when making main claims:

- same model family/checkpoint definition;
- same data manifest and split;
- same perturbation budget;
- same attack steps;
- same attack target;
- same solver backend and evaluation script;
- wall-clock time rather than only epoch count when solver usage differs.

Exclude or clearly label:

- smoke runs;
- one-epoch runs;
- partial-smoke artifacts;
- old generalization roots mixed with new ones;
- different solver/checkpoint backends;
- artifacts where evaluation rows are repeated or summed incorrectly.

This matters because solver-integrated methods can have longer epochs. A fair
training comparison should report equal wall-clock budgets or explicitly state
the compute budget.

### 26.15 Dataset Construction Details Worth Preserving

The generalization records contain concrete dataset details that can later be
used in the experiment section:

- Burgers:
  Gaussian random-field initial conditions, periodic domain, high spatial
  resolution, viscosity parameter such as \(\nu=0.001\), and solver-generated
  targets.
- Darcy:
  binary coefficient fields with values \(\{3,12\}\), generated from thresholded
  Gaussian random fields; model resolution and solver resolution may differ.
- Navier-Stokes:
  recurrent/final-state setting with spectrally upsampled initial conditions
  and expensive temporal rollout.

These details matter because they explain why Darcy is a binary combinatorial
attack while Burgers/Navier-Stokes are continuous perturbation problems.

### 26.16 Runtime, Backend, And Memory Caveats

The benchmark notes add a practical explanation for the computational-cost
section:

- FNO forward inference is much faster than numerical solver forward.
- Solver forward may use less memory than FNO inference, but solver backward or
  inverse differentiation can be dramatically slower and more memory-hungry.
- Time-dependent PDE solvers scale strongly with final time and time-step
  count; FNO inference cost is mostly fixed once the architecture and
  input/output windows are fixed.
- PyTorch and JAX memory measurements may use different scopes
  (`torch_peak_allocated_mib` versus global `nvidia-smi`), so framework memory
  numbers should be compared cautiously.
- Backend and checkpoint differences can confound scientific interpretation if
  not separated.

This supports the paper's limitation claim: solver-integrated adversarial
training is scientifically meaningful but can be expensive, especially when
differentiating through time-dependent solvers.

