# Local Jacobian/SVD Direction Taxonomy

Date: 2026-05-15

This note records how the local Jacobian/SVD directions relate to the actual
attack optimization directions.

## Source Records

- `docs/loss3_original_theory_experiment_plan.md` defines
  `b=e(x)=f(x)-j(x)`, `Delta f`, `Delta j`, `Delta e`, and the three losses.
- `docs/loss3_original_theory_experiment_plan.md` records the local expansion
  `e(x+delta) ~= b + A delta` with `A = J_f - J_j`.
- `docs/loss3_original_theory_experiment_plan.md` distinguishes error-field
  movement, outward error-norm growth, final endpoint error, local Lipschitz
  analysis, finite-radius attack, and path rotation.
- `docs/local_jacobian_svd_experiment_purpose_20260515.md` records that right
  singular vectors are input perturbation patterns, singular values are local
  gains, and `J_e` singular vectors are local mismatch directions.
- `docs/deeponet_loss1_vs_loss3_jacobian_attack_tension_20260515.md` records
  that `loss3` optimizes `||b + Delta f - Delta j||`, so local SVD alignment
  does not by itself imply finite-radius `loss1` and `loss3` equivalence.

## Basic Local Formulas

Let

```text
f = model
j = solver / oracle
b = e(x) = f(x) - j(x)
J_f = model Jacobian at x
J_j = solver Jacobian at x
A = J_e = J_f - J_j
```

Then near a fixed clean input `x`,

```text
Delta f ~= J_f delta
Delta j ~= J_j delta
Delta e ~= A delta
e(x + delta) ~= b + A delta
```

For squared local `loss1`,

```text
L1_sq(delta) ~= ||J_f delta||^2
grad_delta L1_sq = 2 J_f^T J_f delta
```

For pure residual movement,

```text
R_sq(delta) = ||A delta||^2
grad_delta R_sq = 2 A^T A delta
```

The right singular vectors of `J_f` or `A` are eigenvectors of
`J_f^T J_f` or `A^T A`.  If `v_k` is a right singular vector of `A`, then

```text
A^T A v_k = sigma_k^2 v_k
```

So a singular vector is not the gradient direction at an arbitrary point.  It is
the constrained maximizer / stationary direction of the homogeneous quadratic
gain problem.

For squared local `loss3_original`,

```text
L3_sq(delta) ~= ||b + A delta||^2
             = ||b||^2 + 2 b^T A delta + delta^T A^T A delta

grad_delta L3_sq = 2 A^T b + 2 A^T A delta
```

With a half-squared convention, the factor `2` disappears.  For the unsquared
norm,

```text
grad_delta ||b + A delta|| = A^T (b + A delta) / ||b + A delta||
```

when the denominator is nonzero.

Therefore `loss3_original` has two local forces:

```text
linear outward term:   A^T b
quadratic gain term:   A^T A delta
```

The SVD direction describes the quadratic gain term.  The clean residual `b`
adds a separate outward-growth term, so the `loss3_original` optimizer need not
choose the top singular vector of `A`.

## Direction Types

There are five core mathematical directions.

| direction | formula | meaning |
|---|---|---|
| model sensitivity | `v_f* = argmax ||J_f v||` | local direction where model output moves most; local `loss1` direction |
| solver sensitivity | `v_j* = argmax ||J_j v||` | physical/oracle sensitive direction; used to test co-movement |
| residual Lipschitz / mismatch | `v_e* = argmax ||(J_f-J_j)v||` | pure error-field movement; top right singular vector of `J_e` |
| outward error growth | `v_out* = argmax <b/||b||, (J_f-J_j)v>` | first-order direction that increases current error norm; L2 case is proportional to `A^T b` |
| finite-radius endpoint | `delta*_orig = argmax_{||delta||<=eps} ||e(x+delta)||` | actual `loss3_original` attack direction after nonlinear path, projection, and endpoint effects |

Experimentally, three more direction families should be tracked.

| direction family | meaning |
|---|---|
| finite-radius `loss1_original` final direction | checks whether high model movement is only solver co-movement / false alert |
| ratio or regularized final directions | checks whether local unit-growth directions remain good when rescaled to the attack budget |
| path-rotated local directions `v_e*(x+t delta)` | checks whether a single clean-point SVD direction remains valid along the finite-radius path |

Random directions are useful as a baseline, but they are not a core mechanism
direction.

## Relationship To The Current FNO/DeepONet Result

For FNO, `J_f` and `J_j` are highly aligned, so `v_f*` often describes
co-movement rather than model error.  The important local object becomes
`J_e = J_f - J_j`, and `v_e*` or `v_out*` can differ strongly from `v_f*`.

For DeepONet/default-net, `J_d` and `J_e=J_d-J_j` have highly aligned dominant
subspaces because the solver response is small in DeepONet's dominant
high-frequency directions.  This makes `v_d*` and `v_e*` closer locally, but it
still does not make finite-radius `loss1_original` and `loss3_original`
equivalent, because `loss3_original` also contains the `A^T b` outward term,
the moving solver target, nonlinear path effects, and projection dynamics.

## Short Answer

The top singular vector answers:

```text
Which unit input direction maximizes ||J delta|| locally?
```

The gradient answers:

```text
Given the current delta, which infinitesimal step most increases the chosen loss?
```

The finite-radius optimizer answers:

```text
Which bounded delta gives the largest endpoint loss after the whole nonlinear path?
```

These are related, but they are not the same direction except in special local
homogeneous quadratic cases.
