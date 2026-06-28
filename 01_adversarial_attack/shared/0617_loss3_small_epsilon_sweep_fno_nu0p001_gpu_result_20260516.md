# Loss3 Small-Epsilon Sweep Result - 2026-05-16

Status: completed for FNO / 1D Burgers `nu=0.001` using a fixed candidate-direction bank.

## Scope

- Model: FNO1d, `nu=0.001`.
- Indices: `[0, 7, 40, 47, 115]`.
- Epsilons: `[0.0001, 0.001, 0.01, 0.1]`.
- Input/output norm: L2.
- Output directory: `forensics/loss3_small_epsilon_sweep_20260516/fno_nu0p001_gpu_v100`.
- Device policy: GPU-only; the script refuses CPU fallback.
- Runtime device: `Tesla V100-SXM2-32GB`.

The candidate bank uses clean-point top right singular vectors from `J_f`, `J_j`, and `J_e=J_f-J_j`, both signs of those directions, outward-growth directions, and seeded random directions.

## GPU Runtime Evidence

Observed from `manifest.json`:

- Device: `cuda:0`, `Tesla V100-SXM2-32GB`.
- Required GPU architecture: `sm_70`.
- PyTorch: `2.8.0+cu126`; supported arch list includes `sm_70`.
- PyTorch CUDA sanity value: `1.0`.
- JAX backend: `gpu`; JAX GPU device: `cuda:0`.
- CPU fallback policy: `gpu_only_no_cpu_fallback`.
- Runtime: `20.40` seconds.

## Chinese Plain-Language Summary

### Did This Show An Epsilon-Refinement / Local-Convergence Phenomenon?

Yes. The closest analogy is a mesh-refinement or grid-independence check, but
with perturbation radius instead of grid spacing.

In a PDE grid-refinement test, the grid is made finer until the numerical answer
no longer changes materially. Here, `epsilon` is made smaller until the finite
perturbation ratios stop changing materially and agree with the clean local
Jacobian references. In this run, the stabilization happens for roughly
`epsilon <= 1e-2`:

- the best finite-epsilon values are essentially equal to the clean local
  Jacobian reference values;
- the selected one-dimensional direction subspaces are unchanged across
  epsilon, up to sign;
- the selected source is stable: `L_f` uses the FNO direction, `L_j` uses the
  solver direction, `L_e` uses the residual/error direction, and `G_e` uses the
  outward-growth direction.

So the answer to the user's question is: yes, this experiment found the
small-radius analogue of convergence. Once `epsilon` is small enough, the
finite-radius diagnostic behaves like the clean local linearized object.

### What Was Optimized?

This experiment did not run PGD and did not train or update the FNO model.
It optimized only over perturbation directions, and only over a fixed finite
candidate bank.

For each clean sample `x`, each radius `epsilon`, and each unit direction `v`,
the script evaluated four quantities:

```text
L_f(epsilon, v) = ||f(x + epsilon v) - f(x)|| / epsilon
L_j(epsilon, v) = ||j(x + epsilon v) - j(x)|| / epsilon
L_e(epsilon, v) = ||e(x + epsilon v) - e(x)|| / epsilon, where e = f - j
G_e(epsilon, v) = (||e(x + epsilon v)|| - ||e(x)||) / epsilon
```

Then, for each objective, it selected the candidate direction with the largest
value:

```text
best L_f direction = argmax_v L_f(epsilon, v)
best L_j direction = argmax_v L_j(epsilon, v)
best L_e direction = argmax_v L_e(epsilon, v)
best G_e direction = argmax_v G_e(epsilon, v)
```

This is a candidate-bank maximization, not a continuous global optimization over
all possible perturbations. The purpose is to test local structure, not to claim
that this is the strongest possible finite-radius attack.

### How Was The Direction Bank Built?

For every tested sample, the candidate bank contained 178 unit directions:

- top-8 right singular directions of the clean FNO Jacobian `J_f`, both signs;
- top-8 right singular directions of the clean solver Jacobian `J_j`, both
  signs;
- top-8 right singular directions of the clean residual/error Jacobian
  `J_e = J_f - J_j`, both signs;
- the clean residual outward-growth direction, both signs;
- 128 seeded random control directions.

This means the experiment compares known local linear directions against random
controls and checks which source wins as `epsilon` changes.

### Under What Conditions Was It Run?

- PDE/model scope: 1D Burgers, FNO only, `nu=0.001`.
- Samples: test indices `0, 7, 40, 47, 115`.
- Radii: `epsilon = 1e-4, 1e-3, 1e-2, 1e-1`.
- Total candidate evaluations: `5 samples x 4 epsilons x 178 directions = 3560`
  forward evaluations of the FNO/solver pair.
- Official run device: V100 GPU only; no CPU fallback.

### What Was The Optimization Result?

Across all 5 samples and all 4 epsilon values, the best direction source was:

| objective | selected source | count |
| --- | --- | --- |
| `L_f` | FNO `J_f` top direction | 20 / 20 |
| `L_j` | solver `J_j` top direction | 20 / 20 |
| `L_e` | residual/error `J_e` top direction | 20 / 20 |
| `G_e` | clean residual outward-growth direction | 20 / 20 |

For `L_f`, `L_j`, and `L_e`, the sign sometimes flips between `plus` and
`minus`, but that is the same one-dimensional direction. The absolute cosine is
`1.0`, so the selected direction subspace is stable.

### What Did The Epsilon Sweep Show?

The clean local references are:

| objective | clean local reference mean |
| --- | --- |
| `L_f` | `3.895` |
| `L_j` | `4.095` |
| `L_e` | `0.8682` |
| `G_e` | `0.1665` |

For `epsilon <= 1e-2`, the finite-epsilon best/local ratios stay very close to
`1`:

| objective | ratio at `1e-4` | ratio at `1e-3` | ratio at `1e-2` | ratio at `1e-1` |
| --- | --- | --- | --- | --- |
| `L_f` | `1.0005` | `1.0000` | `1.0001` | `0.9901` |
| `L_j` | `0.9997` | `1.0000` | `0.9999` | `0.9854` |
| `L_e` | `1.0004` | `1.0002` | `1.0010` | `0.9608` |
| `G_e` | `0.9965` | `1.0005` | `1.0061` | `1.0576` |

This is the main epsilon-convergence result. From `1e-4` to `1e-2`, the values
are essentially stable and match the local Jacobian references. At `epsilon=0.1`,
the selected direction is still stable, but the value starts to drift, especially
for `L_e` and `G_e`. That is the beginning of finite-radius nonlinear behavior.

### What Is The Scientific Meaning?

1. `L_f` and `L_j` are both around `4`, while `L_e` is around `0.87`. This means
   the FNO and the solver both move strongly in their own outputs, but they move
   together enough that the residual/error map moves much less. This supports
   the local co-movement interpretation for FNO `nu=0.001`.
2. `G_e` is only around `0.166`, much smaller than `L_e`. So the direction that
   moves the residual field the most is not the same thing as the direction that
   most increases the current clean residual norm. This is why residual-field
   movement and clean-residual outward growth must be reported separately.
3. The result is local. It supports using small `epsilon` ratio diagnostics as a
   local-structure measurement. It does not prove that PGD at a large radius will
   choose the same path or reach the same endpoint.

## Local References

| objective | local reference mean | local reference std |
| --- | --- | --- |
| G_e | 0.1665 | 0.03533 |
| L_e | 0.8682 | 0.2018 |
| L_f | 3.895 | 0.3886 |
| L_j | 4.095 | 0.4566 |

## Aggregate Sweep Values

| objective | epsilon | best mean | best/local | abs cos local |
| --- | --- | --- | --- | --- |
| G_e | 0.0001 | 0.1658 | 0.9965 | 1 |
| G_e | 0.001 | 0.1666 | 1.001 | 1 |
| G_e | 0.01 | 0.1675 | 1.006 | 1 |
| G_e | 0.1 | 0.1763 | 1.058 | 1 |
| L_e | 0.0001 | 0.8685 | 1 | 1 |
| L_e | 0.001 | 0.8684 | 1 | 1 |
| L_e | 0.01 | 0.8689 | 1.001 | 1 |
| L_e | 0.1 | 0.8309 | 0.9608 | 1 |
| L_f | 0.0001 | 3.897 | 1.001 | 1 |
| L_f | 0.001 | 3.895 | 1 | 1 |
| L_f | 0.01 | 3.895 | 1 | 1 |
| L_f | 0.1 | 3.856 | 0.9901 | 1 |
| L_j | 0.0001 | 4.094 | 0.9997 | 1 |
| L_j | 0.001 | 4.095 | 1 | 1 |
| L_j | 0.01 | 4.095 | 0.9999 | 1 |
| L_j | 0.1 | 4.036 | 0.9854 | 1 |

## Best Direction Sources

Observed from `best_direction_source_counts.csv`:

| objective | selected source at all epsilons | count |
| --- | --- | --- |
| `L_f` | FNO `J_f` top direction | 20 / 20 |
| `L_j` | solver `J_j` top direction | 20 / 20 |
| `L_e` | error/residual `J_e` top direction | 20 / 20 |
| `G_e` | clean residual outward-growth direction | 20 / 20 |

For `L_f`, `L_j`, and `L_e`, some signed directions flip between plus and minus. The absolute cosine is still `1.0`, so the selected one-dimensional direction is stable up to sign. For `G_e`, the selected direction is consistently `outward_growth_r1_plus`.

## Direction Stability

| objective | eps a | eps b | abs cosine mean | abs cosine min |
| --- | --- | --- | --- | --- |
| G_e | 0.0001 | 0.001 | 1 | 1 |
| G_e | 0.0001 | 0.01 | 1 | 1 |
| G_e | 0.0001 | 0.1 | 1 | 1 |
| G_e | 0.001 | 0.01 | 1 | 1 |
| G_e | 0.001 | 0.1 | 1 | 1 |
| G_e | 0.01 | 0.1 | 1 | 1 |
| L_e | 0.0001 | 0.001 | 1 | 1 |
| L_e | 0.0001 | 0.01 | 1 | 1 |
| L_e | 0.0001 | 0.1 | 1 | 1 |
| L_e | 0.001 | 0.01 | 1 | 1 |
| L_e | 0.001 | 0.1 | 1 | 1 |
| L_e | 0.01 | 0.1 | 1 | 1 |
| L_f | 0.0001 | 0.001 | 1 | 1 |
| L_f | 0.0001 | 0.01 | 1 | 1 |
| L_f | 0.0001 | 0.1 | 1 | 1 |
| L_f | 0.001 | 0.01 | 1 | 1 |
| L_f | 0.001 | 0.1 | 1 | 1 |
| L_f | 0.01 | 0.1 | 1 | 1 |
| L_j | 0.0001 | 0.001 | 1 | 1 |
| L_j | 0.0001 | 0.01 | 1 | 1 |
| L_j | 0.0001 | 0.1 | 1 | 1 |
| L_j | 0.001 | 0.01 | 1 | 1 |
| L_j | 0.001 | 0.1 | 1 | 1 |
| L_j | 0.01 | 0.1 | 1 | 1 |

## Visualizations

![small epsilon best values](../forensics/loss3_small_epsilon_sweep_20260516/fno_nu0p001_gpu_v100/figures/small_epsilon_best_values.png)

![small epsilon local reference ratio](../forensics/loss3_small_epsilon_sweep_20260516/fno_nu0p001_gpu_v100/figures/small_epsilon_local_reference_ratio.png)

![small epsilon direction stability heatmap](../forensics/loss3_small_epsilon_sweep_20260516/fno_nu0p001_gpu_v100/figures/small_epsilon_direction_stability_heatmap.png)

![small epsilon best direction sources](../forensics/loss3_small_epsilon_sweep_20260516/fno_nu0p001_gpu_v100/figures/small_epsilon_best_direction_sources.png)

## Conclusion

Observed from the completed GPU sweep:

1. The experiment found an epsilon-refinement / local-convergence phenomenon.
   When `epsilon` is reduced from `1e-1` to `1e-2`, `1e-3`, and `1e-4`, the
   finite-epsilon estimates stop changing materially and match the clean local
   Jacobian references. This is analogous in spirit to a grid-independence check:
   once the radius is small enough, further shrinking the radius does not change
   the measured local object.
2. For `epsilon <= 1e-2`, all four objectives have best/local ratios very close
   to `1`. The strongest deviations at `epsilon <= 1e-2` are still small:
   `G_e` is about `1.006` at `1e-2`, and `L_e` is about `1.001` at `1e-2`.
3. At `epsilon=0.1`, the value begins to show finite-radius drift: `L_e` drops
   to about `0.961` of its local reference, while `G_e` rises to about `1.058`.
   The selected direction remains stable, but the scalar value is no longer a
   purely infinitesimal/local measurement.
4. The selected direction source is objective-specific and stable across all 20
   sample-epsilon cases per objective: `L_f` selects FNO directions, `L_j`
   selects solver directions, `L_e` selects residual/error directions, and `G_e`
   selects the clean residual outward-growth direction.
5. `L_f` and `L_j` are both large, around `4`, while `L_e` is much smaller,
   around `0.87`. This is the concrete evidence that FNO and solver locally
   co-move: the model output can be sensitive without producing equally large
   model-vs-solver residual sensitivity.
6. `G_e` is much smaller than `L_e`, around `0.166` versus `0.868`. Therefore
   residual-field movement and outward growth of the current clean residual norm
   are different diagnostics and should not be collapsed into one number.

Inference:

For FNO / 1D Burgers / `nu=0.001`, the small-epsilon ratio diagnostics are valid
as local-structure diagnostics once `epsilon <= 1e-2` in this candidate-bank
sweep. The experiment does not claim to solve the global finite-radius attack
problem. It says that, at small radius, the finite-difference measurements have
converged to the clean Jacobian/SVD structure, and that the four objectives
measure four genuinely different local objects.

## Output Files

- `candidate_metrics.csv`: every evaluated candidate direction.
- `best_by_objective_epsilon_index.csv`: per-index selected direction/value.
- `aggregate_best_by_objective_epsilon.csv`: aggregate values used above.
- `direction_stability.csv` and `direction_stability_summary.csv`: cross-epsilon selected-direction cosines.
- `local_reference_by_index.csv` and `local_reference_summary.csv`: clean-point local references.
- `manifest.json`: reproducibility manifest.
