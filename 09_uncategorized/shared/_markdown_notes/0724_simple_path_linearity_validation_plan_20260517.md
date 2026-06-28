# Simple Path-Linearity Validation Plan

Date: 2026-05-17 UTC

Scope: FNO / 1D Burgers `nu=0.001` first. Use the existing Loss3 trajectory and the current straight-line Loss3 endpoint direction. Do not compute full Hessians. Do not compute full dense Jacobians. Keep the first validation cheap.

## Core Hypothesis

We want to test the following pathwise statement:

$$
\text{Near } x_0, \text{ the local geometry changes sharply; farther along the adversarial path, it becomes locally more linear/stable.}
$$

The precise version should be:

$$
\text{Not } J(z) \approx J(x_0),
$$

but rather

$$
\text{local behavior around } z \text{ becomes better approximated by the current first-order model as } \|z-x_0\| \text{ grows along the attack path.}
$$

So the target evidence is:

1. adjacent gradient directions rotate less later in the path;
2. finite-difference local curvature / nonlinearity scores get smaller later in the path;
3. optionally, small 2D local loss slices look more planar later in the path.

## Existing Data To Reuse

Primary existing trajectory:

`results/fno_nu0p001_loss_gradient_path_steps50_save5_gpu_nocudnn_20260515_200631/`

Existing saved PGD points:

$$
k = 0,5,10,15,20,25,30,35,40,45,50.
$$

Samples:

$$
\{0,7,40,47,115\}.
$$

Primary attack objective:

`loss3_original_pgd`.

Use radius ratio as the main x-axis:

$$
r_k = \frac{\|\delta_k\|_2}{\epsilon}, \qquad \epsilon=8.
$$

This is better than using raw step index because the PGD path does not grow in radius uniformly.

## Experiment 1: True PGD Gradient Rotation

### Purpose

This is the simplest and most direct test of whether the actual optimizer direction field stabilizes along the real PGD trajectory.

It answers:

$$
\text{Does the Loss3 gradient direction change a lot early and less later?}
$$

This is different from the older endpoint-vs-movement angle. This experiment explicitly compares current gradient against previous saved gradient.

### Definition

At each saved PGD point,

$$
z_k=x_0+\delta_k,
$$

compute the true Loss3 gradient

$$
g_k = \nabla_z L_3(z_k).
$$

Adjacent gradient rotation:

$$
\theta_{\mathrm{grad,prev}}(k)
=
\arccos
\frac{\langle g_k,g_{k-5}\rangle}{\|g_k\|_2\|g_{k-5}\|_2}.
$$

Clean-reference accumulated rotation:

$$
\theta_{\mathrm{grad,clean}}(k)
=
\arccos
\frac{\langle g_k,g_0\rangle}{\|g_k\|_2\|g_0\|_2}.
$$

Optional movement alignment:

$$
s_k = \frac{\delta_k-\delta_{k-5}}{\|\delta_k-\delta_{k-5}\|_2},
$$

$$
\theta_{\mathrm{grad,step}}(k)
=
\arccos
\frac{\langle g_k,s_k\rangle}{\|g_k\|_2}.
$$

### Expected Pattern

If the hypothesis is correct:

$$
\theta_{\mathrm{grad,prev}}(k)
\text{ is large early and small late.}
$$

But

$$
\theta_{\mathrm{grad,clean}}(k)
$$

may stay large. That distinction is important: later stability does not mean returning to the clean geometry.

### Minimal Run

Use only:

- model: FNO `nu=0.001`;
- attack path: `loss3_original_pgd`;
- samples: `0, 7, 40, 47, 115`;
- saved steps: every 5 steps from `0` to `50`.

No attack rerun is needed if the saved trajectory contains the points. We only recompute gradients at saved points.

### Output Tables

Save one row per sample and saved step:

| column | meaning |
| --- | --- |
| `sample_index` | sample id |
| `k` | PGD step |
| `radius_ratio` | \(\|\delta_k\|_2/8\) |
| `grad_norm` | \(\|g_k\|_2\) |
| `angle_grad_to_previous_deg` | \(\theta_{\mathrm{grad,prev}}(k)\) |
| `angle_grad_to_clean_deg` | \(\theta_{\mathrm{grad,clean}}(k)\) |
| `angle_grad_to_step_deg` | optional movement alignment |

Aggregate by radius / step:

| k | radius ratio mean | angle prev mean | angle clean mean | grad norm mean |
| ---: | ---: | ---: | ---: | ---: |

### Success Criterion

The evidence is good if:

1. mean `angle_grad_to_previous_deg` decreases from early to late;
2. Spearman correlation between radius ratio and adjacent angle is negative;
3. `angle_grad_to_clean_deg` remains nontrivial, showing late stability is not clean-point recovery.

## Experiment 2: Symmetric Finite-Difference Local Linearity Score

### Purpose

This is the cheapest direct test of local linearity. It does not need a Hessian and does not need a Jacobian.

It asks:

$$
\text{Around the current path point } z, \text{ does the function look more linear later in the path?}
$$

The key trick is to use a symmetric second difference. For a perfectly linear function along direction \(u\), the second difference is zero.

### Scalar Loss Local Linearity

For a scalar loss \(L\), direction \(u\), and small radius \(\rho\), define

$$
C_L(z,u,\rho)
=
\frac{
|L(z+\rho u)-2L(z)+L(z-\rho u)|
}{
|L(z+\rho u)-L(z-\rho u)|+\varepsilon_{\mathrm{num}}
}.
$$

If \(C_L\) is small, the loss is nearly linear along direction \(u\) around \(z\).

Use

$$
L(z)=L_3(z)
$$

for the main run.

### Residual-Map Local Linearity

For the vector residual map \(e(z)=f(z)-j(z)\), use

$$
C_e(z,u,\rho)
=
\frac{
\|e(z+\rho u)-2e(z)+e(z-\rho u)\|_2
}{
\|e(z+\rho u)-e(z-\rho u)\|_2+\varepsilon_{\mathrm{num}}
}.
$$

This measures nonlinear bending of the residual map itself, not just the scalar loss.

### Directions To Test

Keep it simple. At each path point test these directions:

1. current Loss3 gradient direction:

$$
u_g(k)=\frac{g_k}{\|g_k\|_2};
$$

2. radial attack direction:

$$
u_r(k)=\frac{\delta_k}{\|\delta_k\|_2};
$$

3. previous movement direction:

$$
u_s(k)=\frac{\delta_k-\delta_{k-5}}{\|\delta_k-\delta_{k-5}\|_2};
$$

4. a small number of fixed random directions, for example `4` random unit L2 directions with fixed seed.

This separates adversarial-path linearity from generic random-direction linearity.

### Step Sizes

Use two small finite-difference radii:

$$
\rho \in \{0.02\epsilon,\,0.05\epsilon\}
= \{0.16,\,0.40\}.
$$

If runtime is too high, run only \(\rho=0.02\epsilon\) first.

### Minimal Run

Use:

- samples: `0, 7, 40, 47, 115`;
- path: `loss3_original_pgd` saved every 5 steps;
- directions: gradient, radial, step, and 4 random directions;
- radii: start with one radius \(\rho=0.16\).

Approximate cost:

$$
5\text{ samples}\times 11\text{ steps}\times 7\text{ directions}\times 2\text{ function evals}
=770\text{ extra function evaluations}.
$$

This is much cheaper than Hessian or full Jacobian work.

### Output Tables

One row per sample / step / direction / radius:

| column | meaning |
| --- | --- |
| `sample_index` | sample id |
| `k` | PGD step |
| `radius_ratio` | \(\|\delta_k\|_2/8\) |
| `direction_type` | grad, radial, step, random_i |
| `rho` | finite difference radius |
| `loss_linearity_score` | \(C_L(z,u,\rho)\) |
| `residual_linearity_score` | \(C_e(z,u,\rho)\) |
| `loss_plus`, `loss_center`, `loss_minus` | raw scalar values |
| `residual_second_diff_norm` | numerator for \(C_e\) |
| `residual_first_diff_norm` | denominator signal for \(C_e\) |

Aggregate by `k`, `direction_type`, and `rho`.

### Success Criterion

The evidence is strong if:

$$
C_L(z_k,u,\rho)
\quad \text{and/or} \quad
C_e(z_k,u,\rho)
$$

decrease with radius ratio along important directions such as gradient/radial/step.

The strongest version is:

$$
\text{early mean score} > \text{late mean score}
$$

for both scalar loss and residual map.

Suggested bins:

- early: \(r_k \le 0.2\);
- middle: \(0.2 < r_k < 0.6\);
- late: \(r_k \ge 0.6\).

## Experiment 3: Small 2D Local Slice Planarity Plot

### Purpose

This is optional but useful for figures. It produces visual evidence that local loss slices are more warped early and more planar later.

It does not require Hessians or Jacobians. It only needs function evaluations.

### Plane Definition

At selected path points \(z_k\), choose two directions:

1. gradient direction:

$$
u_1 = \frac{g_k}{\|g_k\|_2};
$$

2. radial direction orthogonalized against \(\nu_1\):

$$
\tilde u_2 = \frac{\delta_k}{\|\delta_k\|_2} - \left\langle \frac{\delta_k}{\|\delta_k\|_2},\nu_1\right\rangle \nu_1,
$$

$$
u_2=\frac{\tilde u_2}{\|\tilde u_2\|_2}.
$$

Then evaluate

$$
L_3(z_k+a\nu_1+b\nu_2)
$$

on a small grid.

### Minimal Grid

Use only three saved steps:

$$
k\in\{5,25,50\}.
$$

Use only three samples first:

$$
\{0,40,115\}.
$$

Use a small grid:

$$
a,b\in\{-\rho,-\rho/2,0,\rho/2,\rho\}.
$$

This is a `5 x 5` grid.

With 3 samples and 3 steps:

$$
3\times 3\times 25 = 225
$$

function evaluations.

### Planarity Score

Fit an affine plane to the grid values:

$$
\widehat L(a,b)=c_0+c_1a+c_2b.
$$

Record normalized plane residual:

$$
R_{\mathrm{plane}}(z_k)
=
\frac{
\sqrt{\frac{1}{N}\sum_i(L_i-\widehat L_i)^2}
}{
\mathrm{std}(L_i)+\varepsilon_{\mathrm{num}}
}.
$$

Optional: fit a quadratic surface and record quadratic-to-linear coefficient ratio.

### Success Criterion

The visual and numeric evidence is good if:

$$
R_{\mathrm{plane}}(k=5) > R_{\mathrm{plane}}(k=25) > R_{\mathrm{plane}}(k=50)
$$

on average.

This is not the main proof, but it is useful for explaining the phenomenon.

## Recommended First Run

Do only two main experiments first:

1. Experiment 1: PGD gradient adjacent rotation.
2. Experiment 2: symmetric finite-difference local linearity score.

Skip Experiment 3 until the first two show a clean signal.

Do not run Hessian. Do not run full Jacobian. Do not run top-k SVD on the PGD path yet.

## Why These Are Enough For The First Validation

Experiment 1 tests whether the optimizer's direction field stabilizes:

$$
\angle(g_k,g_{k-5})\downarrow.
$$

Experiment 2 tests whether the local function behavior becomes more linear:

$$
C_L(z_k,u,\rho)\downarrow,
\qquad
C_e(z_k,u,\rho)\downarrow.
$$

Together they directly test the simple version of the claim:

$$
\text{early path: high turning / high local nonlinearity,}
$$

$$
\text{late path: low turning / stronger local linearity.}
$$

## Expected Figures

1. `gradient_adjacent_angle_vs_radius.png`
   - x-axis: radius ratio \(\|\delta_k\|_2/8\)
   - y-axis: \(\angle(g_k,g_{k-5})\)

2. `gradient_clean_angle_vs_radius.png`
   - x-axis: radius ratio
   - y-axis: \(\angle(g_k,g_0)\)

3. `loss_linearity_score_vs_radius.png`
   - x-axis: radius ratio
   - y-axis: \(C_L\)
   - color: direction type

4. `residual_linearity_score_vs_radius.png`
   - x-axis: radius ratio
   - y-axis: \(C_e\)
   - color: direction type

5. Optional: `local_2d_slice_k5_k25_k50.png`

## Interpretation Rules

### Strong Positive Evidence

The claim is strongly supported if:

1. adjacent gradient angle decreases with radius;
2. local finite-difference nonlinearity score decreases with radius;
3. clean-reference gradient angle does not collapse to zero;
4. the effect is strongest on the Loss3 path and weaker or different on controls.

This means:

$$
\text{the path is not returning to clean geometry, but entering a locally more stable region.}
$$

### Ambiguous Evidence

If adjacent gradient angle decreases but finite-difference nonlinearity does not, then we can say optimizer directions stabilize, but we cannot yet claim local linearity improves.

If finite-difference nonlinearity decreases only along the gradient direction but not random directions, then the path becomes more linear in the adversarially relevant direction, not necessarily in all directions.

### Negative Evidence

If both adjacent gradient angle and finite-difference scores stay high or increase with radius, the hypothesis is not supported. Then today's straight-line Jacobian result may be a chord-specific phenomenon rather than a general path geometry property.

## Controls

Keep controls minimal:

1. Loss3 PGD path: main experiment.
2. Random ray with same endpoint norm: sanity control for direction specificity.
3. Optional loss1/loss2 PGD paths: check whether this is Loss3-specific.

Do not start with all controls. Start with Loss3 only, then add controls if the signal is promising.

## Deliverables

Output directory suggestion:

`forensics/loss3_simple_path_linearity_20260517/fno_nu0p001/`

Files:

- `pgd_gradient_rotation_by_sample_k.csv`
- `pgd_gradient_rotation_aggregate_by_k.csv`
- `finite_difference_linearity_by_sample_k_direction.csv`
- `finite_difference_linearity_aggregate_by_k_direction.csv`
- `summary.md`
- `figures/gradient_adjacent_angle_vs_radius.png`
- `figures/loss_linearity_score_vs_radius.png`
- `figures/residual_linearity_score_vs_radius.png`

## Final Recommendation

Run the first version as:

- FNO `nu=0.001` only;
- Loss3 PGD trajectory only;
- 5 samples only;
- saved steps every 5;
- one finite-difference radius \(\rho=0.16\);
- directions: gradient, radial, step, and 4 random directions.

This is simple enough to run without Hessians, without full Jacobians, and without top-k SVD. If it shows the expected early-high / late-low pattern, then we can later harden it with PGD-path Jacobian sketching.
