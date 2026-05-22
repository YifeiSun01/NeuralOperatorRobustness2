# NS2D Optimizer-Hypothesis Validation Experiment Design

Updated: 2026-05-22 22:36:58 UTC

Status: design only. No model, solver, GPU, attack run, or plotting job was started for this note.

## Hypothesis To Validate

The current working hypothesis is:

```text
1D Burgers favored replacement / generalized-power-style methods because the attack geometry quickly entered a stable, local-linear-like high-loss boundary direction regime.

2D NS recurrent FNO favors `steepest_add` in the current eps32/alpha10 setting because the full attack objective is finite-radius, nonlinear, recurrent/rollout dependent, and solver/mode coupled. In that regime, additive normalized updates can preserve useful perturbation history while replacement discards it each step.
```

This design separates that hypothesis into directly testable pieces.

## Experiment 1: Epsilon Sweep / Local-To-Finite-Radius Test

Question:

```text
Does replacement/GPI become more competitive when epsilon is smaller?
```

Run:

- Same model/checkpoint and same test indices as the current NS2D run.
- Use `p=q=2`.
- Run at least these pairs:
  - `epsilon=4, alpha=1.25`
  - `epsilon=8, alpha=2.5`
  - `epsilon=16, alpha=5`
  - `epsilon=32, alpha=10`
- Use the same methods:
  - `raw_add`
  - `raw_replace`
  - `steepest_add`
  - `steepest_replace`
- Start with key cases:
  - `loss1/all_w`
  - `loss3/all_w`
  - `loss2/all_a_target_w`

Metrics:

- final true loss
- final surrogate/objective loss
- boundary ratio
- first step reaching 25%, 50%, 75%, 100% epsilon
- true loss at matched delta-norm milestones
- `steepest_add / steepest_replace` true-loss ratio

Expected if hypothesis is right:

- At small epsilon, replacement/GPI should be closer to `steepest_add`, possibly tied or better.
- At large epsilon, `steepest_add` should pull away, especially for `loss3` and W-heavy modes.
- This pattern is already partially visible locally: `loss1/all_w` ratio drops from about `1.61` at eps32 to about `1.036` at eps8.

Falsifier:

- If replacement/GPI remains much worse even at very small epsilon, then the issue is not mainly finite-radius nonlinearity. It may be implementation, surrogate mismatch, or mode construction.

## Experiment 2: Frozen-Linearized / Local Jacobian Diagnostic

Question:

```text
If we force the 2D NS problem back into a fixed local-linear problem, does replacement/GPI become strong again?
```

Run:

For one or two samples, freeze the clean point and replace the full nonlinear maps by local linear approximations.

For loss1:

```text
F(x + delta) - F(x) ~= J_F(x) delta
```

For loss3:

```text
e(x + delta) = F(x + delta) - G(x + delta)
e(x + delta) ~= e(x) + J_e(x) delta
```

Then compare the four methods on this frozen objective.

Metrics:

- frozen objective value per step
- boundary arrival
- final delta cosine between methods
- top local direction alignment if SVD/JVP power is available

Expected if hypothesis is right:

- Replacement/GPI should become strongest or at least highly competitive on the frozen-linearized objective.
- If full nonlinear attack still favors `steepest_add`, the difference is due to nonlinear/path-dependent geometry, not because replacement is inherently bad.

Falsifier:

- If `steepest_add` still dominates the frozen-linearized problem, then our current explanation is incomplete. It may mean the implementation's replacement update is not matching the true local steepest/generalized-power direction.

## Experiment 3: Early-To-Final Direction Stability Test

Question:

```text
Does replacement/GPI quickly find a final-like direction in 2D NS, as it did in 1D Burgers?
```

Run:

Save intermediate deltas for replacement and additive methods at:

```text
k = 1, 5, 10, 20, 50, 100
```

For each method/loss/mode/sample, compute:

```text
cos(delta_k, delta_final)
true_loss(delta_k) / true_loss(delta_final)
surrogate_loss(delta_k) / surrogate_loss(delta_final)
```

Compare against the old Burgers evidence, where GPI had:

```text
cos(delta_10, delta_300) ~= 0.984
cos(delta_20, delta_300) ~= 0.995
```

Expected if hypothesis is right:

- In 2D NS, replacement/GPI should be less final-like early, especially for eps32/loss3/W-heavy modes.
- `steepest_add` may have slower but more stable accumulation toward a better final delta.

Falsifier:

- If replacement is already final-like by k=5/k=10 but still has worse true loss, then direction instability is not the main cause. The issue may be true-vs-surrogate mismatch or solver/target coupling.

## Experiment 4: Gradient Rotation / Direction History Diagnostic

Question:

```text
Are 2D NS gradients rotating more strongly than Burgers gradients, making replacement reset too aggressively?
```

Run:

For each method, record per-step angles:

```text
angle(grad_k, grad_(k-1))
angle(update_direction_k, update_direction_(k-1))
angle(delta_k, delta_(k-1))
angle(delta_k, update_direction_k)
```

Use at least:

- `loss1/all_w`, eps8 and eps32
- `loss3/all_w`, eps8 and eps32
- optionally `loss3/dddddwwwww` and `loss3/aaaaaddddw`

Metrics:

- mean/std angle over samples
- late-stage angle after boundary hit
- angle spikes around loss-growth jumps

Expected if hypothesis is right:

- eps32 should have stronger gradient/update-direction rotation than eps8.
- replacement methods should show larger reset/rotation behavior.
- `steepest_add` should show a smoother delta trajectory while still improving true loss.

Falsifier:

- If gradients are stable and replacement still loses, then the reason is not gradient rotation; look instead at surrogate/true loss mismatch or implementation details.

## Experiment 5: Boundary-Matched True-Loss Comparison

Question:

```text
Is `steepest_add` better because it reaches the boundary differently, or because its direction is better at the same perturbation size?
```

Run:

For each method, evaluate true loss at matched delta-norm milestones:

```text
25% epsilon
50% epsilon
75% epsilon
100% epsilon
```

Use interpolation or nearest saved step.

Expected if hypothesis is right:

- `steepest_add` should beat replacement not only at the final step, but also at matched norm milestones in the nonlinear regimes.
- At smaller epsilon or simpler loss2, the gap should shrink.

Falsifier:

- If methods are equal at matched norm and only differ because one reaches boundary sooner, then the explanation should shift toward step-size/budget scheduling rather than nonlinear path geometry.

## Experiment 6: Solver/Mode Ablation: W vs D vs A

Question:

```text
Does solver-gradient / ADW mode complexity control the optimizer ranking?
```

Run:

Compare these mode families:

- `all_w`: solver path differentiable where applicable
- `all_d`: solver forward but detached gradients
- `all_a_target_w`: dictionary/fixed target style
- mixed W/D/A modes already used in the current run

Metrics:

- method ranking by true loss
- final delta FFT fingerprints
- true-vs-surrogate gap
- gradient/update angle diagnostics

Expected if hypothesis is right:

- W-heavy and loss3 modes should favor `steepest_add` more strongly.
- A/fixed-target modes should behave more like simpler/local objectives and may favor replacement or tie.
- FFT solver/dealiasing fingerprints should be stronger in W-heavy gradient paths.

Falsifier:

- If method ranking is unchanged across W/D/A, solver/mode coupling is probably not the main cause.

## Experiment 7: Hybrid Add-Then-Replace / Replace-Then-Add Probe

Question:

```text
Is the useful method really additive history, or just reaching the boundary quickly?
```

Run two hybrid methods:

1. `replace_then_add`:
   - replacement/GPI for first 1-5 steps to hit the boundary
   - then `steepest_add` projected updates on the boundary

2. `add_then_replace`:
   - `steepest_add` for first 10-20 steps
   - then replacement updates

Expected if hypothesis is right:

- `replace_then_add` may combine fast boundary arrival with history-preserving refinement and could beat pure replacement.
- If pure `steepest_add` still wins, early additive path construction is important.
- If `replace_then_add` wins, then the issue is not replacement boundary hit but replacement-only loss of path history after the hit.

## Recommended Priority

Run in this order:

1. Epsilon sweep using existing attack script settings. This is easiest and already partially supported by eps32 vs eps8.
2. Early-to-final direction stability using saved intermediate deltas. This directly compares 2D NS to the old Burgers GPI evidence.
3. Boundary-matched true-loss comparison from the same saved traces.
4. Gradient rotation diagnostics.
5. Frozen-linearized/JVP diagnostic. This is the cleanest proof, but it is likely more implementation work.
6. Solver/mode ablation and hybrid method probes.

## Decision Logic

The hypothesis is strongly supported if all three are true:

```text
small epsilon or frozen-linearized 2D NS -> replacement/GPI improves or wins
large epsilon full 2D NS -> steepest_add wins
eps32/full 2D NS -> replacement early directions are less final-like or true-loss weaker at matched norm
```

The hypothesis is weakened if:

```text
replacement/GPI loses even in frozen-linearized local 2D NS
or gradients are stable but replacement still loses
or replacement has better surrogate loss but consistently worse true loss
```

In that case, the main explanation should shift from `nonlinear/path-dependent optimization geometry` to either `true-vs-surrogate mismatch`, `mode/target construction`, or an implementation-level update mismatch.
