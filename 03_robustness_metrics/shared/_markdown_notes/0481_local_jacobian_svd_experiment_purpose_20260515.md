# Local Jacobian/SVD Experiment Purpose

Date: 2026-05-15

## Project-Level Question

The broader project is not trying to show that a neural operator output should
stay invariant under input perturbations.  The project-level claim is:

> Regression robustness means the learned model should co-vary with the true
> solver/oracle.  The dangerous directions are high model-solver mismatch
> directions, not merely high model-sensitivity directions.

In notation:

```text
f = learned model
j = true solver / oracle
e(x) = f(x) - j(x)
Delta f = f(x + delta) - f(x)
Delta j = j(x + delta) - j(x)
Delta e = Delta f - Delta j
```

The primary finite-radius attack objective is therefore:

```text
loss3_original(delta) = || f(x + delta) - j(x + delta) ||
```

not just:

```text
loss1_original(delta) = || f(x + delta) - f(x) ||
```

because a large model movement can be harmless if the solver moves the same way.

## Why This Local Jacobian/SVD Experiment Exists

The local Jacobian/SVD experiment is a mechanism experiment.  Its purpose is to
explain why `loss1_original`, `loss2_original`, and `loss3_original` can point
to different directions.

At a fixed input `x`, the local linear approximation is:

```text
Delta f ~= J_f(x) delta
Delta j ~= J_j(x) delta
Delta e ~= (J_f(x) - J_j(x)) delta
```

So the experiment compares:

```text
model Jacobian: J_f or J_d
solver Jacobian: J_j
error Jacobian: J_e = J_model - J_j
```

The SVD of these Jacobians identifies the locally strongest input perturbation
directions.  Right singular vectors are input perturbation patterns; singular
values measure local gain; `J_e` singular vectors are local mismatch directions.

## What The Experiment Is Testing

The experiment directly tests the following chain:

1. If model and solver Jacobians are similar, then model-sensitive directions
   are also solver-sensitive directions.
2. In that case, a large `||Delta f||` may be mostly co-movement with the
   solver, not a regression error.
3. The true dangerous local object is `J_model - J_solver`.
4. Therefore `loss3_original` is the correct attack objective because it targets
   oracle-relative error on the perturbed input.

This connects the finite-radius PGD evidence to a local-linear explanation.

## What The Current Results Show

### FNO

Both FNO variants are locally solver-like.

For FNO `nu=0.001`:

- model spectral norm: `3.895`
- solver spectral norm: `4.095`
- error spectral norm: `0.868`
- model-vs-solver top-8 subspace cosine: `0.981`
- model/solver response cosine along model top-8 directions: `0.981`

For FNO `nu=0.01`:

- model spectral norm: `1.379`
- solver spectral norm: `1.373`
- error spectral norm: `0.0373`
- model-vs-solver top-8 subspace cosine: `0.9987`
- model/solver response cosine along model top-8 directions: `0.9993`

Interpretation:

> FNO's dominant local directions are mostly physical/solver-like directions.
> Therefore `J_f - J_j` is a smaller residual and its dangerous directions can
> differ from the model's own top sensitivity directions.

This supports the idea that `loss1_original` can over-report risk for FNO:
it can find directions where the model moves a lot, but the solver also moves
with it.

### DeepONet / Default-Net

DeepONet `nu=0.01` is not locally solver-like in its dominant directions.

- DeepONet model spectral norm: `7.202`
- solver spectral norm: `1.373`
- error spectral norm: `7.100`
- model-vs-solver top-8 subspace cosine: `0.170`
- model-vs-error top-8 subspace cosine: `0.981`
- model top-1 `hi128`: `0.761`
- model top-1 zero crossings: `507.2`
- error top-1 `hi128`: `0.803`
- error top-1 zero crossings: `516.0`

Interpretation:

> DeepONet's dominant local directions are high-frequency directions that the
> solver does not share.  Therefore `J_d - J_j` is almost the same object as
> `J_d` in those dominant directions.

This explains why DeepONet's model-sensitive directions and error directions
can look similar: the solver response is too small and too differently shaped
to cancel the DeepONet response.

## How This Serves The Main Project

This experiment does not by itself prove finite-radius attack performance.
Instead, it supplies the mechanism behind the attack objectives:

```text
loss1_original asks: how much does the model move?
loss3_original asks: how much does the model fail to track the solver?
Jacobian/SVD asks: locally, are those two questions the same or different?
```

For FNO, they are different because the model and solver co-move.  For DeepONet,
they are closer because the model's high-frequency response is already a
model-solver mismatch.

This is why the experiment belongs in the project:

1. It supports the central claim that regression robustness is not invariance.
2. It gives a concrete local-linear reason why `loss1` and `loss2` can be
   misleading surrogate objectives.
3. It explains what `loss3_original` is actually exploiting: high mismatch
   directions in the error field.
4. It separates architectural behavior: FNO has solver-like local modes, while
   DeepONet/default-net has high-frequency solver-misaligned modes.
5. It gives interpretable evidence for the paper narrative through singular
   values, singular-vector shapes, Fourier frequency metrics, and subspace
   angles.

## Logic Chain To Use In Writing

The clean narrative is:

1. In operator regression, the oracle changes when the input changes.
2. Therefore attacking `||f(x + delta) - f(x)||` can confuse true physical
   response with model error.
3. The correct finite-radius target is
   `||f(x + delta) - j(x + delta)||`.
4. Locally, this depends on `(J_model - J_solver) delta`, not just
   `J_model delta`.
5. FNO's `J_model` is close to `J_solver`, so model-sensitive directions are
   often co-moving directions.
6. DeepONet's `J_model` is far from `J_solver`, high-frequency, and close to
   its own error Jacobian.
7. Therefore the local Jacobian/SVD results explain when model sensitivity is
   a false alert and when it is already true mismatch.

## Related Source Files

- `docs/loss3_original_theory_experiment_plan.md`
- `docs/main_objective_mechanism_experiment1_result_20260514.md`
- `docs/fno_solver_jacobian_similarity_result_20260514.md`
- `docs/deeponet_solver_jacobian_similarity_result_20260515.md`
- `docs/nine_row_fno001_fno01_deeponet01_svd_interpretation_20260515.md`
