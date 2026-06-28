# Outward-Growth Direction Experiment Plan - 2026-05-15

## Status

Plan/script preparation only. The production experiment is scoped only to FNO `nu=0.001`. GPU smoke tests were run for the new script, but the full five-index experiment has not been run in this turn.

This plan fills the missing row in `Experiment 2: Local Response Decomposition
Table` from `docs/loss3_original_theory_experiment_plan.md`: the local
outward-growth direction

\[
v_{growth}^*
=
\arg\max_{\|v\|_p=1}
\left\langle
\frac{e(x)}{\|e(x)\|_2},
(J_f-J_j)v
\right\rangle.
\]

The existing Jacobian/SVD experiments already covered top model, solver, error,
and random directions. This experiment adds the clean-residual outward direction
and compares it against those existing directions.

## Source Records And Existing Evidence

Observed from current local/R2-backed records:

- `docs/loss3_original_theory_experiment_plan.md` defines Experiment 2 and the
  missing `v_growth*` row.
- `docs/local_jacobian_svd_direction_taxonomy_20260515.md` explains that local
  `loss3_original` has two terms: an outward term `A^T b` and a quadratic gain
  term `A^T A delta`.
- `docs/local_jacobian_svd_experiment_purpose_20260515.md` records why local
  Jacobian/SVD analysis supports the `loss3_original` narrative.
- Existing FNO `nu=0.001` Jacobian/SVD data are available locally under
  `forensics/fno_solver_jacobian_similarity_20260514_raw_recomputed/`.
- FNO `nu=0.01` and DeepONet `nu=0.01` are intentionally out of scope for this run because they make the experiment broader and slower than needed.
- Existing direction-comovement tables use columns such as `fno_gain`,
  `solver_gain`, `mismatch_gain`, `cos_fno_solver_response`, `D_f`, and
  `D_sym`.

## Main Question

The current completed SVD rows answer:

```text
Which unit direction maximizes model movement?       v_f*
Which unit direction maximizes solver movement?      v_j*
Which unit direction maximizes error-field movement? v_e*
```

The missing row answers a different question:

```text
Which unit direction most increases the current clean error norm at first order?
```

This distinction matters because the error field can move without increasing
risk. The direction `v_e*` maximizes `||A v||`, while `v_growth*` maximizes the
component of `A v` pointing outward from the current clean residual `b`.

## Mathematical Setup

Use the local notation:

```text
f = model
j = solver / oracle
e(x) = f(x) - j(x)
b = e(x)
J_f = model Jacobian at x
J_j = solver Jacobian at x
A = J_e = J_f - J_j
```

The local error model is:

\[
e(x+\delta) \approx b + A\delta.
\]

For `q=2` and `b != 0`, the first-order change of the error norm is:

\[
\|b+A\delta\|_2 - \|b\|_2
\approx
\left\langle \frac{b}{\|b\|_2}, A\delta \right\rangle.
\]

Let:

\[
u = \frac{b}{\|b\|_2}.
\]

Then the outward-growth direction for `input_p=2` is:

\[
v_{growth}^* = \frac{A^T u}{\|A^T u\|_2}.
\]

The maximum first-order growth rate is:

\[
\|A^T u\|_2.
\]

For other input norms, use the dual-norm steepest direction:

- `p=2`: `v = normalize_l2(A.T @ u)`.
- `p=inf`: `v = sign(A.T @ u)` with the usual Linf unit constraint.
- `p=1`: put the mass on the coordinate with largest absolute entry of
  `A.T @ u`.

Round 1 of this missing-row experiment should use `input_p=2`, `output_q=2`, to
match the existing local SVD and FNO `nu=0.001` attack evidence.

## Experimental Scope

### Phase 1: FNO `nu=0.001` Main Case

Use the same five indices as the existing FNO Jacobian/SVD experiment:

```text
indices = 0, 7, 40, 47, 115
```

Source Jacobian files:

```text
forensics/fno_solver_jacobian_similarity_20260514_raw_recomputed/index_*/fno/fno_index*_jacobian_svd.npz
forensics/fno_solver_jacobian_similarity_20260514_raw_recomputed/index_*/solver/solver_index*_jacobian_svd.npz
forensics/fno_solver_jacobian_similarity_20260514_raw_recomputed/index_*/error/error_index*_jacobian_svd.npz
```

This is the primary paper-facing version because it matches the existing
Experiment 1 finite-radius mechanism diagnostic.

### Out Of Scope

Do not run the FNO `nu=0.01` or DeepONet/default-net variants in this round. The
user-requested experiment is only the FNO `nu=0.001` missing-row diagnostic.

## Required Clean Residual `b`

The Jacobian `.npz` files store Jacobians and SVD factors, not necessarily the
clean residual `b`. Therefore the script should compute or load clean outputs
as follows:

1. Load the same clean sample `x` used by the Jacobian/SVD run.
2. Run the model once to get `f(x)`.
3. Run the solver/oracle once to get `j(x)`.
4. Save:

```text
clean_model_output.npy
clean_solver_output.npy
clean_residual.npy
clean_residual_norm_l2
```

If equivalent clean outputs already exist in a trusted attack trajectory or
summary directory, they may be reused, but the source path must be recorded in
the new manifest.

## Directions To Compare

For each index, compare these directions:

| direction source | construction | purpose |
|---|---|---|
| `model_top1` / `model_top8` | top right singular vectors of `J_f` | model sensitivity / `loss1` local direction |
| `solver_top1` / `solver_top8` | top right singular vectors of `J_j` | physical solver-sensitive direction |
| `error_top1` / `error_top8` | top right singular vectors of `A=J_f-J_j` | residual Lipschitz / pure mismatch movement |
| `outward_growth` | `normalize(A.T @ (b/||b||))` | first-order `loss3` norm growth direction |
| `negative_outward_growth` | `-outward_growth` | sign sanity check; should reduce first-order error norm |
| `random_mean` | fixed random unit directions | baseline |

The key new row is `outward_growth`. The negative row is not part of the paper
table unless useful, but it is a strong implementation sanity check.

## Linear Metrics To Record

For every direction `v`, compute:

```text
model_gain              = ||J_f v||_2
solver_gain             = ||J_j v||_2
mismatch_gain           = ||A v||_2
cos_model_solver        = cos(J_f v, J_j v)
D_f                     = ||A v||_2 / (||J_f v||_2 + eta)
D_sym                   = ||A v||_2 / (||J_f v||_2 + ||J_j v||_2 + eta)
outward_component       = <b/||b||_2, A v>
outward_component_ratio = outward_component / (||A v||_2 + eta)
```

Also record direction similarities:

```text
cos(v_growth, v_f_top1)
cos(v_growth, v_j_top1)
cos(v_growth, v_e_top1)
cos(v_growth, delta_loss3_original_final)   optional if final attack deltas are available
```

For the aggregate table, report mean, standard deviation, min, and max across
indices.

## Small-Radius Sanity Check

The linear table is the main deliverable, but add a finite-difference check so
that the result is not only algebraic.

For each direction `v`, evaluate small radii:

```text
rho in {1e-4, 1e-3, 1e-2}
```

and compute actual nonlinear quantities:

```text
actual_loss3_growth(rho, v)
  = (||f(x + rho v) - j(x + rho v)||_2 - ||f(x) - j(x)||_2) / rho

actual_residual_movement(rho, v)
  = ||(f(x + rho v)-j(x + rho v)) - (f(x)-j(x))||_2 / rho
```

Compare them with the linear predictions:

```text
predicted_outward_component = <b/||b||, A v>
predicted_residual_movement = ||A v||_2
```

This check should be small enough to stay in the local regime. It is not the
same as the later Small-Epsilon Sweep, which systematically studies stability
across multiple epsilon scales.

## Output Organization

Use a new output root:

```text
forensics/outward_growth_direction_20260515/
```

Suggested layout:

```text
forensics/outward_growth_direction_20260515/
  fno_nu0p001/
    config.json
    manifest.json
    all_direction_response_table.csv
    aggregate_direction_response_summary.csv
    all_direction_similarity_table.csv
    finite_difference_growth_table.csv
    finite_difference_growth_summary.csv
    summary.md
    index_000/
      clean_model_output.npy
      clean_solver_output.npy
      clean_residual.npy
      direction_response_table.csv
      direction_similarity_table.csv
      finite_difference_growth_table.csv
      outward_growth_direction.npy
      plots/
        direction_response_bars.png
        finite_difference_growth_vs_prediction.png
```

For repository documentation, create a result note after running:

```text
docs/outward_growth_direction_result_20260515.md
```

The result note must record exact source paths for all CSV/JSON/NPY artifacts.

## Proposed Script

Add a focused postprocessing script rather than changing the large SVD scripts:

```text
tools/analyze_outward_growth_direction.py
```

Suggested command:

```bash
cd /workspace/NeuralOperatorRobustness2

adv_robust/bin/python tools/analyze_outward_growth_direction.py \
  --sample-indices 0 7 40 47 115 \
  --jacobian-root forensics/fno_solver_jacobian_similarity_20260514_raw_recomputed \
  --output-dir forensics/outward_growth_direction_20260515/fno_nu0p001 \
  --result-doc docs/outward_growth_direction_result_20260515.md \
  --top-k 8 \
  --random-directions 64 \
  --seed 0 \
  --eta 1e-6 \
  --finite-difference-rhos 1e-4 1e-3 1e-2 \
  --finite-difference-top-k 8 \
  --device cuda \
  --runtime-workarounds \
  --prepend-env-ptxas \
  --burgers-nu 0.001 \
  --burgers-jax-solver-dtype float64
```

This command stays on GPU. The runtime workaround only changes PATH so that JAX
uses the virtualenv-local `ptxas` before the system CUDA `ptxas`; it does not
move the solver to CPU.

## Implementation Details

### Direction Formula For `p=2`

For each index:

```python
J_f = load model jacobian
J_j = load solver jacobian
A = J_f - J_j
b = f_clean - j_clean
u = b / (np.linalg.norm(b) + eps)
a = A.T @ u
v_growth = a / (np.linalg.norm(a) + eps)
```

If `||b||_2 < 1e-10` or `||A.T @ u||_2 < 1e-12`, mark the direction as
`degenerate` and do not present it as a meaningful outward-growth direction.

### Sign Convention

Use the sign that maximizes the first-order growth:

```text
outward_component = <u, A v_growth> >= 0
```

The negative direction should satisfy:

```text
<u, A (-v_growth)> <= 0
```

This is a simple sanity check that catches transposes or sign mistakes.

### Random Direction Baseline

Use a fixed seed and record both mean and best random direction:

```text
random_mean
random_best_by_outward_component
random_best_by_mismatch_gain
```

This clarifies whether `v_growth*` is merely average, or actually much stronger
than random for outward error growth.

## Expected Outcomes

### FNO `nu=0.001`

Expected from previous SVD evidence:

- `model_top*` directions should have large model and solver gains, high
  `cos(J_f v, J_j v)`, and relatively small `D_f`.
- `error_top*` directions should have larger mismatch gain than model/solver
  directions, but their outward component may not be maximal.
- `outward_growth` should have the largest `outward_component` by construction,
  but it may have smaller `mismatch_gain` than `error_top1`.

This would complete the conceptual separation:

```text
v_e*      = error field moves most
v_growth* = current error norm grows fastest at first order
loss3_original finite-radius delta = nonlinear endpoint optimum
```

### Out-Of-Scope Cases

FNO `nu=0.01` and DeepONet/default-net are not part of this run. They can be
revived later only if the FNO `nu=0.001` outward-growth result leaves an
architecture-specific question unresolved.

## Success Criteria

The experiment is complete when the following are recorded:

1. A direction-response table containing rows for `model_top`, `solver_top`,
   `error_top`, `outward_growth`, and random directions.
2. `outward_growth` has the largest or tied-largest `outward_component` among
   tested unit directions for each nondegenerate sample.
3. Finite-difference growth at `rho=1e-4` and `rho=1e-3` matches the linear
   outward prediction qualitatively.
4. The final documentation explicitly distinguishes:

```text
residual Lipschitz / mismatch movement: ||A v||
outward clean-risk growth: <b/||b||, A v>
finite-radius endpoint risk: ||e(x + delta)||
```

5. `EXPERIMENT_LEDGER.md` and a dedicated result note under `docs/` are updated
   with exact source paths and observed metrics.

## Failure / Edge Cases

- If `||b||` is near zero, the outward direction is undefined. Mark the sample
  as degenerate and report it separately.
- If `||A.T @ (b/||b||)||` is near zero, then first-order clean error growth is
  locally flat. In that case, second-order or finite-radius terms may dominate.
- If finite differences do not match the linear prediction even at `rho=1e-4`,
  check model/solver dtype, normalization, Jacobian orientation, and whether the
  clean sample exactly matches the Jacobian source.
- Do not interpret this experiment as a full finite-radius attack. It is a
  local first-order diagnostic that completes Experiment 2.

## How This Fits The Paper Narrative

This experiment supplies the missing local row between two already separated
ideas:

```text
J_e top singular vector: largest possible error-field movement
A^T b outward direction: largest first-order increase of current error norm
loss3_original attack: largest finite-radius endpoint error
```

If `v_growth*` differs from both `v_f*` and `v_e*`, it strengthens the argument
that there are multiple non-equivalent local and global notions of danger even
within the FNO `nu=0.001` setting.

## Minimal Deliverable

If time is limited, the minimal version is:

1. Run only FNO `nu=0.001` on indices `0, 7, 40, 47, 115`.
2. Produce `aggregate_direction_response_summary.csv` with rows:

```text
model_top8
solver_top8
error_top8
outward_growth
random_mean
```

3. Produce `finite_difference_growth_table.csv` and
   `finite_difference_growth_summary.csv` for the local linearity sanity check.
4. Write `docs/outward_growth_direction_result_20260515.md` with the conclusion
   and exact source paths.
