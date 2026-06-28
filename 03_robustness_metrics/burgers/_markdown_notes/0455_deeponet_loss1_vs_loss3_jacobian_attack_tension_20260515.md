# DeepONet Loss1 vs Loss3: Jacobian/Attack Tension

Date: 2026-05-15

## Question

The local Jacobian/SVD result says that, for DeepONet/default-net `nu=0.01`, the error Jacobian is close to the DeepONet model Jacobian in the dominant local subspace:

```text
J_e = J_d - J_j ~= J_d
```

A natural hypothesis is therefore:

> If DeepONet model directions and error directions are almost the same, then optimizing `loss1` might produce nearly the same final `loss3` as optimizing `loss3`.

The existing attack-ratio data does **not** support that strong conclusion.

## Loss Definitions Found In Code

Current three-loss script, `run_three_loss_objective_attack.py`:

```text
loss1 = || f(x + delta) - f(x) ||_q
loss2 = || f(x + delta) - g(x) ||_q
loss3 = || f(x + delta) - g(x + delta) ||_q
```

Old five-loss Burgers script, `run_burgers_corrected_oldstyle_5loss.py`, uses squared L2 for optimization:

```text
loss1         = squared_l2(f_adv, f_clean)
loss2_fixed   = squared_l2(f_adv, g_clean)
loss3_stopgrad= squared_l2(f_adv, stopgrad(g_adv))
loss3         = squared_l2(f_adv, g_adv)
```

The clean recomputed ratio summaries evaluate true oracle-relative loss3 after each method.

## Existing DeepONet Ratio Evidence

Source table:

```text
results/clean_recomputed_summary/method_ratio_best_vs_second_tests.csv
results/burgers_loss3_clean_recomputed_summary.md
```

Across DeepONet `nu=0.01` settings:

| group | settings | total n | median loss3/loss1 true-loss ratio | mean loss3/loss1 true-loss ratio | loss1 wins | loss3 wins |
|---|---:|---:|---:|---:|---:|---:|
| DeepONet L2 | 9 | 702 | 19.07 | 35.99 | 0/702 | 624/702 |
| DeepONet Linf | 11 | 902 | 84.35 | 488.71 | 12/902 | 799/902 |

Representative rows:

| norm | epsilon | alpha | loss1 ratio | loss3 ratio | loss3/loss1 | winner |
|---|---:|---:|---:|---:|---:|---|
| L2 | 0.5 | 0.009 | 65.0 | 106.8 | 1.64 | loss3 |
| L2 | 4.0 | 0.075 | 105.9 | 2019.6 | 19.07 | loss3 |
| L2 | 12.0 | 0.225 | 41.2 | 6196.8 | 150.40 | loss3 |
| Linf | 0.025 | 0.001 | 124.0 | 173.5 | 1.40 | loss3 |
| Linf | 0.25 | 0.010 | 38.5 | 3247.4 | 84.35 | loss3 |
| Linf | 1.0 | 0.040 | 5.64 | 14951.3 | 2648.83 | loss3 |

So the observed empirical pattern is:

> DeepONet `loss3` optimization is usually much stronger than `loss1` optimization when evaluated by final true `loss3`.


## Does DeepONet Have A Smaller Loss1-vs-Loss3 Gap Than FNO?

Not as a simple global statement in the old ratio tables.  The gap depends strongly on model, viscosity, norm, epsilon, and alpha.  In the grouped summary, DeepONet has a small `loss3/loss1` gap only at very small perturbation radii; for larger radii the gap becomes huge.

This is exactly the distinction between local Jacobian reasoning and finite-radius attack behavior:

- local/small epsilon: `J_e ~= J_d` can make `loss1` and `loss3` more similar for DeepONet;
- finite/large epsilon: nonlinear path effects, moving oracle target, and outward-error alignment dominate, so `loss3` can strongly outperform `loss1`.

Therefore the existing data supports a weaker, more precise claim:

> DeepONet has model/error local-subspace alignment, but finite-radius `loss1` is still not a reliable substitute for finite-radius `loss3`.

## Why This Does Not Contradict The Jacobian/SVD Result

The Jacobian/SVD result is a local statement at the clean input and mostly about dominant singular subspaces.  It does **not** imply the finite-radius objectives are equivalent.

Important distinctions:

1. `loss1` optimizes only `||Delta f||`.

2. `loss3` optimizes `||b + Delta f - Delta j||`, where `b = f(x)-j(x)` is the clean residual.

3. Even if `J_e ~= J_d` as a high-gain operator, `loss3` cares about whether the movement points outward relative to the current error vector `b`.

4. The SVD comparison says model and error top subspaces are close, but it does not say every vector selected by PGD on `loss1` is the same as the vector selected by PGD on `loss3`.

5. DeepONet is highly nonlinear and high-frequency.  Finite-radius PGD can leave the clean-point linear regime, so local SVD directions can rotate along the path.

6. The old five-loss attacks optimize squared L2 objectives and use projected finite-step PGD.  Optimization dynamics, initialization, projection, and saturation can make two locally related objectives end at very different points.

7. The data shows `loss3_stopgrad` is usually much closer to `loss3` than `loss1` is.  This suggests that including the perturbed solver target value `g(x+delta)` is already important, even before considering solver-gradient backpropagation.

## Best Interpretation

The correct statement is not:

> DeepONet `loss1` and `loss3` should be basically the same.

The better statement is:

> DeepONet has much higher alignment between model-sensitive directions and error directions than FNO does, so the conceptual gap between `J_model` and `J_error` is smaller for DeepONet.  However, finite-radius `loss3_original` can still beat `loss1_original` by a lot because it optimizes the actual oracle-relative endpoint error, including the clean residual direction, the moving solver target, nonlinear path effects, and projection dynamics.

## What To Do Next

The clean follow-up is to run a targeted DeepONet mechanism diagnostic analogous to the FNO mechanism experiment:

1. For DeepONet `nu=0.01`, run the same samples and attack settings with `loss1_original`, `loss2_fixed`, `loss3_stopgrad`, and `loss3_original`.
2. For each final delta, record:
   - `||Delta f||`
   - `||Delta j||`
   - `||Delta f - Delta j||`
   - `||e(x + delta)||`
   - `cos(Delta f, Delta j)`
   - outward component `<e(x)/||e(x)||, Delta f - Delta j>`
   - cosine between final delta and the top singular vectors of `J_d` and `J_e`
3. Compare small epsilon vs large epsilon.  The hypothesis should be:
   - at very small epsilon, `loss1` and `loss3` may become closer for DeepONet than for FNO;
   - at finite/large epsilon, `loss3` remains stronger because it follows the endpoint oracle-relative error geometry.

This experiment would directly resolve the tension between the local SVD result and the old DeepONet attack visualizations.
