# Burgers Epsilon Dependence vs Current NS2D Hypothesis Check

Updated: 2026-05-23 UTC

Status: local record inspection only. No solver call, model inference, attack
step, JAX import, PyTorch import, plotting, or GPU computation was started.

## Question

Did the earlier 1D Burgers experiments already observe the pattern now proposed
for 2D NS: smaller epsilon makes replacement/GPI more competitive, while larger
finite-radius epsilon can favor additive LP-steepest PGD?

## Source Evidence Inspected

- `docs/three_loss_burgers_optimizer_findings_summary_20260521.md`
- `docs/loss1_loss2_1d_burgers_optimizer_speed_lookup_20260521.md`
- `docs/ns2d_vs_1d_burgers_optimizer_hypothesis_check_20260522.md`
- `docs/loss3_small_epsilon_sweep_fno_nu0p001_gpu_result_20260516.md`
- `docs/loss3_alpha_epsilon_core4_p2q2_300steps_result_20260520.md`
- `docs/loss3_current_mechanism_validation_summary_20260521.md`

## Short Answer

Partly yes, but not in exactly the same experiment format.

The Burgers records do show a strong small-epsilon/local-linear phenomenon:
for epsilon values `1e-4`, `1e-3`, and `1e-2`, finite-epsilon diagnostics match
clean local Jacobian references and selected direction subspaces are stable. At
`epsilon=0.1`, scalar values begin to drift, indicating finite-radius nonlinear
effects.

The Burgers records also show that optimizer ranking depends on epsilon/alpha:
in the p2q2 loss3 alpha/epsilon sweep, replacement methods are usually fastest
to boundary and fastest early, while long 300-step final loss can sometimes be
slightly larger for `steepest_add` or `raw_add` depending on epsilon and alpha.

However, the exact NS2D-style sweep over `loss1/all_w`, `loss2/all_a_target_w`,
`loss3/all_w`, four optimizers, and boundary-matched true loss was not already
available for 1D Burgers in the same form.

## Observed Burgers Evidence

### Small-Epsilon Local Stability

Observed from `docs/loss3_small_epsilon_sweep_fno_nu0p001_gpu_result_20260516.md`:

- Tested epsilons: `1e-4`, `1e-3`, `1e-2`, `1e-1`.
- For `epsilon <= 1e-2`, finite-epsilon ratios are very close to the clean local
  Jacobian references.
- Direction subspaces are stable across epsilon values with absolute cosine
  `1.0`.
- At `epsilon=0.1`, selected direction is still stable, but scalar values drift:
  for example `L_e` drops to about `0.9608` of its local reference, while `G_e`
  rises to about `1.0576`.

Inference: Burgers does support the idea that small epsilon behaves locally and
linearly, while larger epsilon begins to show finite-radius effects.

### Alpha/Epsilon Sweep For Loss3

Observed from `docs/loss3_alpha_epsilon_core4_p2q2_300steps_result_20260520.md`:

- Replacement methods reach the boundary at step `1` across the p2q2 sweep.
- Replacement methods are usually the fastest early methods.
- In the method rollup over 20 settings, mean final loss3 values were:
  - `raw_add`: `5.659`
  - `raw_replace`: `5.747`
  - `steepest_add`: `5.827`
  - `steepest_replace`: `5.747`
- The existing Burgers summary explicitly says the strongest claim is early
  speed, not unconditional final-loss dominance; long additive runs can slightly
  exceed replacement/GPI in some alpha/epsilon settings.

Inference: Burgers already showed that alpha/epsilon and number of steps matter.
Replacement/GPI was very fast and strong early, but it was not a theorem that it
always has the largest final loss.

## Comparison To Current 2D NS Pattern

Current 2D NS evidence is sharper for the finite-radius interpretation:

- At `eps32_alpha10`, `steepest_add` is much stronger than replacement for
  `loss1/all_w` and completed `loss3` modes.
- At `eps8_alpha2p5`, the `loss1/all_w` gap nearly collapses and replacement is
  more competitive.

This mirrors the broad Burgers lesson that smaller radius is more local-linear,
but the optimizer ranking differs:

- Burgers: replacement/GPI quickly enters a stable high-loss boundary corridor.
- 2D NS: replacement directions rotate more and additive LP-steepest PGD
  accumulates useful path history, so it wins more clearly at large epsilon.

## Answer To The User's Question

Yes, Burgers had evidence in the same direction at the level of local-linear
small-epsilon behavior. But no, we did not already have an exactly equivalent
Burgers experiment showing the same four-method, three-loss, boundary-matched
ranking transition as the current proposed NS2D validation.

The safer statement is:

```text
Burgers supports the principle that epsilon controls local-linear versus
finite-radius behavior. Burgers also showed optimizer ranking can depend on
alpha/epsilon and run length. But in the tested Burgers setting, replacement/GPI
remained the clearest fast early method, whereas in current 2D NS the large-
epsilon full attack favors additive LP-steepest PGD much more strongly.
```



## Correction: Optimizer Ranking Is A Different Question

The small-epsilon local-Jacobian diagnostic above is not, by itself, an optimizer-ranking experiment. It only says whether a finite-epsilon perturbation is still close to a local linear sensitivity reference. The user's question here is narrower and more important for method choice:

```text
As epsilon/alpha change, which optimizer is fastest, which reaches the boundary fastest, and which gives the largest final loss?
```

For Burgers, the existing evidence that actually addresses optimizer ranking is the p2q2 loss3 alpha/epsilon sweep, not the local-Jacobian sweep.

Observed from the Burgers p2q2 loss3 alpha/epsilon records:

- Replacement/GPI-style methods (`raw_replace` and `steepest_replace`) are the clearest speed winners. They reach the epsilon boundary at step `1` and are usually the fastest early-improvement methods.
- This speed advantage is not a subtle monotone epsilon trend. It is mostly a structural effect of the replacement update: it immediately replaces `delta` with a full-budget normalized direction.
- The final 300-step mean-loss winner is not governed by a clean monotone rule like "smaller epsilon makes replacement best" or "larger epsilon makes additive best."
- In the 20-setting Burgers p2q2 loss3 rollup, strict largest-final-mean winners were mostly `steepest_add`, with `raw_add` in one setting and replacement/GPI tied for largest final mean in a small number of settings.

So the more accurate Burgers law is:

```text
Burgers p2q2 loss3:
replacement/GPI is the strongest early and boundary-arrival optimizer.
The long-run final-loss winner can depend on alpha/epsilon and run length, and is not a simple monotone function of epsilon.
```

This is different from the current NS2D result. In NS2D large-epsilon cases, `steepest_add` is not merely better after a very long run; it is already better under boundary-matched true-loss comparison for `loss1/all_w` and completed `loss3` modes. That means the NS2D difference is stronger than the Burgers caveat: replacement/GPI is not just losing final-loss by a small late-run margin; its directions are rotating too much and failing to become final-like early.

## Direct Comparison: Which NS2D Diagnostics Had Burgers Analogues?

Question: were the same mechanism experiments done in the earlier 1D Burgers
work, and what differed from the current 2D NS results?

Short answer: partially yes. The exact NS2D experiment package was not run in
identical form for Burgers, but the main diagnostic ideas had Burgers analogues.

### 1. Boundary-matched / boundary-arrival behavior

Burgers analogue:

- The Burgers loss1/loss2 and loss3 summaries recorded boundary arrival and
  post-boundary gain.
- In Burgers loss3 p2q2, replacement/GPI reached the boundary at step `1` and
  still improved along the boundary.
- In the loss3 core-four p2q2 sweep, replacement methods were usually fastest to
  boundary and fastest early; long final loss could sometimes be slightly larger
  for additive methods.

Difference from NS2D:

- In NS2D `eps32_alpha10`, boundary-matched true loss strongly favors
  `steepest_add` for `loss1/all_w` and completed `loss3` modes.
- In Burgers, replacement/GPI was a very strong boundary method; in NS2D, merely
  reaching/replacing on the boundary was not enough.

### 2. Early-to-final direction stability

Burgers analogue:

- Burgers records measured whether early replacement/GPI directions became
  final-like.
- In the saved Burgers p2q2 trajectory evidence, replacement/GPI endpoint ratio
  was about `0.961` by step `5`, `1.008` by step `10`, and `1.012` by step `20`.
- The Burgers summary also records mean `cos(delta_k, delta_300)` for GPI around
  `0.8868` at `k=5`, `0.9840` at `k=10`, and `0.9949` at `k=20`.

Difference from NS2D:

- In NS2D `eps32_alpha10 / loss3 / all_w`, replacement directions are not
  final-like early: representative-sample `cos(delta_10, delta_final)` was
  negative for replacement (`-0.0758`), and `cos(delta_20, delta_final)` was
  still negative (`-0.1594`).
- In the same NS2D case, `steepest_add` was much more final-like by `k=10` and
  `k=20` (`0.6694` and `0.8233`).

### 3. Direction/gradient rotation

Burgers analogue:

- Burgers loss3 diagnostics measured boundary angles, tangent residual, and
  direction changes.
- Replacement/GPI showed meaningful post-boundary angular motion, but this was
  beneficial in Burgers because it quickly entered a stable high-loss boundary
  region.

Difference from NS2D:

- In NS2D `eps32_alpha10 / loss3 / all_w`, replacement had median
  `angle(delta_k, delta_{k-1})` about `78.09 deg`, while `steepest_add` was only
  about `11.61 deg`.
- This suggests NS2D replacement is not steadily refining a stable boundary
  direction; it is resetting into substantially rotating directions.

### 4. Small-epsilon/local-linear behavior

Burgers analogue:

- Burgers had a dedicated small-epsilon local diagnostic over `1e-4`, `1e-3`,
  `1e-2`, and `1e-1`.
- For `epsilon <= 1e-2`, finite-epsilon diagnostics matched local Jacobian
  references and direction subspaces were stable with absolute cosine `1.0`.
- At `epsilon=0.1`, scalar values began to drift, showing finite-radius effects.

Difference from NS2D:

- NS2D currently has evidence that reducing from `eps32` to `eps8` shrinks the
  method gap, especially for `loss1/all_w`.
- NS2D still needs the more aggressive small-epsilon real attack cases
  (`eps4`, `eps2`, `eps1`) to test whether replacement/GPI fully catches up.

## Mechanism Difference

Burgers mechanism observed in records:

- The Burgers attack often behaves like it quickly finds a useful boundary
  corridor.
- Replacement/GPI works well because early full-budget directions become
  final-like quickly and remain useful.
- Burgers did show path dependence, but it was path dependence that replacement
  could exploit.

NS2D mechanism observed so far:

- The NS2D recurrent attack has stronger finite-radius nonlinearity and more
  path-dependent direction rotation.
- Replacement/GPI resets delta to the current direction, so it can discard useful
  components accumulated in previous steps.
- Additive LP-steepest PGD preserves previous useful structure and adds new
  directions, which is why it can win at large epsilon.

Bottom line:

```text
Burgers: replacement/GPI quickly becomes final-like and exploits a stable high-loss boundary corridor.
NS2D: replacement/GPI directions rotate too much; additive LP-steepest PGD wins by accumulating a nonlinear path.
```
