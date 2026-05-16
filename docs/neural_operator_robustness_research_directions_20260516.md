# Neural Operator Robustness Research Directions

Date: 2026-05-16

This note organizes the next research directions for neural-operator robustness
experiments. The overall theme is to study adversarial robustness across five
levels:

- PDE solver implementation
- neural-operator model implementation
- attack loss definition
- perturbation optimization method
- structure-aware error measurement

The goal is not only to find attacks that increase pointwise error, but also to
understand when an attack creates meaningful solver-level or physics-structure
error.

## Direction 1: Cross-Framework Solver and Model Combinations

### Core Question

How do solver framework, model framework, and framework-conversion overhead
affect PGD attack behavior?

The target PDE settings are:

- 1D Burgers
- 2D Stokes / Navier-Stokes

For each problem, compare four solver/model combinations:

| Combination | Solver | Neural operator model |
| --- | --- | --- |
| JAX solver + JAX model | JAX | JAX |
| JAX solver + PyTorch model | JAX | PyTorch |
| PyTorch solver + JAX model | PyTorch | JAX |
| PyTorch solver + PyTorch model | PyTorch | PyTorch |

### Experimental Design

Run the same PGD attack protocol across all four combinations. Keep the dataset,
initial condition, perturbation budget, step count, and evaluation loss fixed so
that differences come from solver/model framework choices rather than from
attack configuration drift.

The mixed-framework cases should explicitly record tensor-transfer and
conversion costs, for example JAX array to PyTorch tensor, PyTorch tensor to JAX
array, device synchronization, and host-device copies.

### Metrics To Record

- Final true solver-level error
- Attack surrogate loss used during optimization
- Runtime per attack and per PGD step
- GPU memory usage
- Tensor conversion overhead
- Solver evaluation time
- Model evaluation time
- Whether gradients are computed fully in one framework or require detached
  cross-framework calls

### Expected Output

This direction should produce a framework-comparison table showing whether the
best attack behavior comes from a real modeling difference or from framework
runtime and conversion effects.

## Direction 2: Compare Different Attack Losses

### Core Question

Which attack loss is the most useful surrogate for creating real solver-level
error?

Compare three attack objectives:

1. Model-output change:

   $$
   \|f(x+\delta)-f(x)\|_2
   $$

2. Model-vs-dataset-target error:

   $$
   \|f(x+\delta)-y\|_2
   $$

3. Model-vs-perturbed-solver error:

   $$
   \|f(x+\delta)-g(x+\delta)\|_2
   $$

Here, \(f\) is the neural operator model, \(g\) is the numerical solver, \(x\)
is the clean input, \(\delta\) is the perturbation, and \(y\) is the original
dataset target.

### Experimental Design

Use each loss as the optimization objective in separate attacks, but evaluate
all attacks with the same final true loss:

$$
\|f(x+\delta)-g(x+\delta)\|_2
$$

This separates the loss used for optimization from the loss used for final
scientific evaluation.

### Metrics To Record

- Surrogate loss trajectory during optimization
- Final true solver-level loss
- Gap between surrogate improvement and true-error improvement
- Attack direction cosine similarities between loss choices
- Frequency content or structure of the final perturbation
- Runtime and memory cost for each loss

### Expected Output

This direction should clarify whether simpler losses, such as model-output
change, are reliable proxies for true solver-level failure, or whether they can
create false alarms where the model moves but still co-moves with the solver.

## Direction 3: Include Perturbation Size Directly In The Objective

### Core Question

Can the attack keep a large true error while using a smaller perturbation?

The original attack setup usually constrains \(\delta\) within a fixed radius
and then maximizes error. This direction makes perturbation size part of the
objective itself.

### Candidate Objectives

Penalty-style objective:

$$
\mathcal{L}_{true}(x,\delta) - \lambda \|\delta\|
$$

or

$$
\mathcal{L}_{true}(x,\delta) - \lambda \|\delta\|^2
$$

Ratio-style objective:

$$
\frac{\mathcal{L}_{true}(x,\delta)}{\|\delta\|+\eta}
$$

where \(\eta\) is a small stabilizing constant.

### Experimental Design

Compare fixed-radius PGD against perturbation-aware objectives. Sweep penalty
weights or ratio stabilizers and record the tradeoff between perturbation size
and final true error.

The central question is whether the attack is merely strong, or whether it is
efficient: high true error per unit perturbation.

### Metrics To Record

- Final true solver-level error
- Perturbation norm
- Error-to-perturbation ratio
- Local Lipschitz-style estimates
- Direction cosine with known local sensitive directions
- Whether smaller perturbations preserve visible or physical plausibility

### Expected Output

This direction should identify more efficient and less obvious attack
directions, especially directions that approximate local sensitivity rather than
only large-radius adversarial endpoints.

## Direction 4: Compare PGD With Power-Iteration-Like Methods

### Core Question

How do standard PGD and power-iteration-like optimization methods compare for
neural-operator attacks?

The comparison should include:

- PGD
- power iteration
- generalized power iteration
- related local-linear or Jacobian-based update rules

### Experimental Design

Run each optimizer under matched perturbation budgets and matched loss
definitions. For losses with a clear local Jacobian interpretation, compare the
optimizer trajectory against the dominant singular directions of the relevant
linearized map.

The point is not only whether an attack succeeds, but how quickly, stably, and
cheaply it reaches a damaging direction.

### Metrics To Record

- Convergence speed
- Final true solver-level error
- Number of solver and model evaluations
- Runtime
- GPU memory usage
- Stability across random seeds and input samples
- Cosine similarity to local SVD or gradient directions

### Expected Output

This direction should show whether PGD is necessary for finite-radius attacks,
or whether a local power-iteration-style method can find comparable directions
with lower cost or better stability.

## Direction 5: Extend Pointwise L2 Error To Structure-Aware Error

### Core Question

Does pointwise \(L_2\) error capture the physically meaningful difference between
two PDE solution fields?

Pointwise \(L_2\) compares values gridpoint by gridpoint. For PDE solutions and
physical fields, this can miss important structure. Two fields may have modest
pointwise error but very different coherent structures; conversely, two fields
may have larger pointwise error while preserving the same physical pattern.

### Structure-Aware Error Ideas

Potential extensions include:

- spatial alignment-aware error
- errors invariant or robust to global value transformations
- shape difference
- texture difference
- structural similarity
- dominant physical-mode difference
- spectrum-aware or frequency-band-aware error
- PDE-residual-aware discrepancy

### Experimental Design

Evaluate attacks using both pointwise \(L_2\) and one or more structure-aware
metrics. Compare cases where these metrics agree and cases where they disagree.

The important diagnostic is not only the scalar score, but also the visual and
physical interpretation of the perturbed solution field.

### Metrics To Record

- Pointwise \(L_2\) error
- Structure-aware error score
- Frequency-spectrum change
- Spatial shift or alignment score
- Coherent-structure difference
- Visual comparison of clean, perturbed, model, and solver fields
- Whether the attack changes the apparent physical mode

### Expected Output

This direction should prevent over-reliance on pointwise \(L_2\). The final
analysis should distinguish numerical value error from meaningful structural or
physical distortion.

## Suggested Execution Order

1. Start with Direction 2 on the existing 1D Burgers / FNO setup because it is
   closest to the current loss1/loss2/loss3 work.
2. Add Direction 3 for small-epsilon and ratio-style experiments, since it
   directly connects to local sensitivity and perturbation efficiency.
3. Add Direction 4 using local Jacobian/SVD artifacts to compare PGD against
   power-iteration-like updates.
4. Expand to Direction 1 once both JAX and PyTorch solver/model paths are stable
   enough for fair cross-framework timing.
5. Add Direction 5 as the evaluation layer that checks whether attacks change
   physical structure, not only pointwise values.

## Minimal Record-Keeping Requirements

For every experiment derived from these directions, record:

- dataset path
- model checkpoint path
- solver implementation and version
- attack loss used for optimization
- final true evaluation loss
- perturbation norm and budget
- optimizer and hyperparameters
- runtime and memory measurements
- output tables and figure paths
- observed results separately from interpretation
