# NS2D Recurrent Attack loss1 Solver Usage Clarification - 2026-05-22

## Summary

The current `loss1` attack code is not a Cartesian `loss1 x ADW target-mode` sweep. The label `loss1/all_w` is misleading, but the computation is intentional under the current initial-condition attack setup.

Better label:

```text
loss1 / canonical_solver_context
```

or:

```text
loss1 / W-context
```

not:

```text
loss1 target mode = all_w
```

## What loss1 Means

The optimized `loss1` objective is:

```text
loss1 = || F(x + delta) - F(x) ||_q
```

where:

- `x` is the initial condition.
- `delta` is the perturbation on the initial condition.
- `F(...)` is the recurrent FNO prediction pipeline.
- `f0 = F(x)` is the clean model output.

There is no solver target `G(x + delta)` inside the optimized `loss1` objective.

## Why loss1 Still Uses The Solver

Even though `loss1` has no ADW target mode, the current 2D recurrent FNO model requires `T_in=10` input frames. The attack variable is only the initial condition, so the code must construct the first ten model input frames before calling the FNO.

For the current canonical `loss1` run, those model-input context frames are generated with the solver:

```text
initial condition -> solver frames 2..10 -> recurrent FNO -> final model prediction
```

So `loss1` uses solver computation for input context, not for an ADW target mode.

Observed in code:

- `active_losses(...)` sets `need_target = loss_type == "loss3"`.
- For `loss1`, `need_target=False`, so the target branch is not used.
- `model_prediction(...)` still builds the FNO input frames. With the canonical `all_w` mode, this rolls the solver through the input-context frames.

Relevant source:

```text
2D_NS_FNO2d_recurrent/perturbation_methods/attack_ns2d_recurrent_core4.py
```

## Why loss1 Also Has True-Loss Solver Logging

The current launch used:

```text
TRUE_LOSS_EVERY=1
```

That means every attack step records a diagnostic true-loss curve by running the all-W solver path to the target frame. This is separate from the optimized `loss1` objective.

This logging call:

```text
true_loss_all_w(x_adv)
```

is no-gradient diagnostic logging. It does not define or backpropagate the `loss1` objective.

So there are two solver uses in the current `loss1` run:

| Solver use | Needed for optimization? | Purpose |
|---|---:|---|
| Solver frames 2..10 | Yes, under current initial-condition attack definition | Build the recurrent FNO input context |
| Solver frame 20 true-loss curve | No | Diagnostic logging because `TRUE_LOSS_EVERY=1` |

## No Duplicate loss1 ADW Sweep

Observed from the active launch log and outputs:

- `loss1` ran only once, with canonical W-context.
- It ran the four update methods: `raw_add`, `raw_replace`, `steepest_add`, `steepest_replace`.
- The launch did not run `loss1` across `all_a`, `all_d`, or mixed A/D/W presets.

Correct conceptual grouping:

```text
loss1: single objective, canonical context only
loss2: fixed/approximate target style, can use dictionary/A-style target semantics
loss3: perturbed target objective, where ADW target-path choices matter
```

## Current Interpretation Of Existing Results

The completed first group from the running launch should be interpreted as:

```text
loss1 / canonical_solver_context / four update methods
```

It should not be described as:

```text
loss1 / all_w target mode
```

## Remaining Cleanup

Recommended future cleanup:

- Rename result labels in analysis text from `loss1/all_w` to `loss1/canonical_solver_context`.
- In future refactors, separate `context_policy` from `target_path_policy` so `loss1` cannot be confused with ADW target modes.
- If true-loss curves are not needed for `loss1`, set `TRUE_LOSS_EVERY=0` or skip true-loss logging for `loss1` to save time.
