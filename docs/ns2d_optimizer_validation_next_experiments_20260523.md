# NS2D Optimizer Validation: Recommended Next Experiments

Updated: 2026-05-23 UTC

Status: recommendation/design only. No solver call, model inference, attack step,
JAX import, PyTorch import, plotting, or GPU computation was started.

## Current Working Explanation

Observed evidence so far suggests that the 2D NS recurrent FNO attack differs
from the earlier 1D Burgers case because the 2D NS objective is more nonlinear
and path-dependent at finite epsilon. Replacement / generalized-power style
updates work best when the objective has a stable local dominant direction.
Additive LP-steepest PGD works better when the useful direction rotates across
steps and useful perturbation components must be accumulated instead of reset.

## Highest-Value Next Experiments

### 1. Epsilon Sweep With Fixed Step-To-Boundary Schedule

Goal: test whether replacement becomes stronger as epsilon gets smaller and the
problem becomes more local-linear.

Recommended grid:

- `epsilon=4`, `alpha=1.25`
- `epsilon=8`, `alpha=2.5`
- `epsilon=16`, `alpha=5`
- `epsilon=32`, `alpha=10`

Keep `alpha / epsilon = 0.3125` fixed, so the additive methods have comparable
step-to-boundary pressure across epsilon values.

Minimal cases:

- `loss1 / all_w`
- `loss2 / all_a_target_w`
- `loss3 / all_w`

Primary readouts:

- final true loss,
- boundary-matched true loss at 25%, 50%, 75%, and 100% epsilon,
- early-to-final cosine,
- gradient/update rotation.

Expected pattern if the hypothesis is right:

- small epsilon: replacement/GPI gets closer to `steepest_add`, possibly wins in
  some cases;
- large epsilon: `steepest_add` separates more clearly.

Priority: highest. This is the cleanest cheap validation.

### 2. Alpha Sweep At Fixed Epsilon

Goal: separate optimizer geometry from step-size artifacts.

Recommended grid at `epsilon=32`:

- `alpha=2.5`
- `alpha=5`
- `alpha=10`
- `alpha=15`
- optionally `alpha=20` only if stable.

Minimal cases:

- `loss1 / all_w`
- `loss3 / all_w`

Primary readouts:

- step when each method reaches 25%, 50%, 75%, 100% boundary,
- boundary-matched true loss,
- true-loss curve after reaching boundary.

Expected pattern if the hypothesis is right:

- If `steepest_add` only wins because alpha is tuned better, changing alpha will
  erase the effect.
- If `steepest_add` wins because it follows a better finite-radius path, it will
  remain strong across a reasonable alpha range, especially at boundary-matched
  comparisons.

Priority: high, but after the epsilon sweep.

### 3. Frozen-Linearized Diagnostic

Goal: directly test whether replacement/GPI wins when the nonlinear objective is
turned into a local-linear problem.

Design:

- Pick one clean input `x` and fixed clean trajectory.
- Freeze the local linearization around `x`:
  - for loss1: approximate `F(x + delta) - F(x)` by a fixed local operator,
  - for loss3: approximate the model-solver error by a fixed local operator.
- Compare the four optimizers on this frozen objective.

Expected pattern if the hypothesis is right:

- frozen/local-linear objective: replacement/GPI becomes strongest or nearly
  strongest;
- full nonlinear objective: `steepest_add` remains strongest.

Priority: highest scientific value, but more implementation time.

### 4. Epsilon Radius With Same Number Of Boundary Steps

Goal: avoid confounding epsilon with how fast additive methods reach the boundary.

Design:

- Choose target boundary arrival times, e.g. 10, 25, 50 steps.
- For each epsilon, set alpha so additive methods reach the boundary at roughly
  the same step count.
- Compare method ranking at matched boundary arrival time.

Expected pattern:

- If ranking changes mostly with arrival speed, this experiment will show it.
- If ranking changes with epsilon even after arrival time is matched, the
  finite-radius nonlinear explanation is stronger.

Priority: medium-high.

### 5. More Samples For Statistical Confidence

Goal: check whether the current pattern is robust across initial conditions.

Design:

- Keep a small but representative experiment set:
  - `eps8_alpha2p5` and `eps32_alpha10`,
  - `loss1/all_w` and `loss3/all_w`,
  - all four methods.
- Increase initial conditions from 10 to 20 or 30 if runtime allows.

Readouts:

- mean and standard error of final true loss,
- mean and standard error of boundary-matched true loss,
- per-sample win rate by method.

Priority: medium. Useful for paper-level confidence, not the first thing to run.

### 6. Mode-Specific Check: W/D/A Sensitivity

Goal: test whether solver gradient involvement controls the optimizer ranking.

Design:

- Compare `loss3/all_w`, `loss3/all_d_target_w`, and mixed W/D modes at the same
  epsilon/alpha.
- Once dictionary mode is fully trusted, include A-heavy modes too.

Readouts:

- method ranking,
- gradient rotation,
- spectral structure of final delta.

Expected pattern:

- More W/solver-gradient involvement may increase nonlinear/path dependence and
  spectral filtering effects.
- D/A-heavy modes may behave more like fixed-target or partially frozen
  objectives.

Priority: medium.

## Practical Run Order

Recommended order for compute efficiency:

1. Finish the current running pair-outer attack.
2. Re-run the CPU-only offline diagnostics on all completed results.
3. Add the missing `epsilon=4, alpha=1.25` minimal sweep for:
   - `loss1/all_w`,
   - `loss2/all_a_target_w`,
   - `loss3/all_w`.
4. Run a small alpha sweep at `epsilon=32` for only:
   - `loss1/all_w`,
   - `loss3/all_w`.
5. Only then implement the frozen-linearized diagnostic.
6. If the pattern is still clear, increase sample count for the most important
   settings instead of expanding every W/D/A combination.

## What Not To Do First

Avoid immediately running the full cross-product of:

- many epsilons,
- many alphas,
- all W/D/A modes,
- all losses,
- many samples.

That would be expensive and harder to interpret. The better strategy is to keep
the first validation grid narrow and hypothesis-targeted.

## Summary Recommendation

The most efficient next validation is:

- epsilon sweep first,
- alpha sweep second,
- frozen-linearized pilot third.

The strongest expected evidence would be:

- replacement/GPI improves as epsilon shrinks,
- `steepest_add` remains better at large epsilon even after boundary matching,
- frozen-linearized 2D NS flips back toward replacement/GPI.

If those three happen together, the explanation that 2D NS is more nonlinear and
path-dependent than the 1D Burgers attack is strongly supported.


## 2026-05-23 Smaller-Epsilon Aggressive Local-Linearity Probe

The user pointed out that `epsilon=32, alpha=10` was already run and
`epsilon=8, alpha=2.5` is already in the active running attack, so the next
validation should push smaller epsilon values instead of repeating those.

Recommended smaller-epsilon grid:

- `epsilon=4`, `alpha=1.25`
- `epsilon=2`, `alpha=0.625`
- `epsilon=1`, `alpha=0.3125`
- optional lower bound: `epsilon=0.5`, `alpha=0.15625`

This keeps `alpha / epsilon = 0.3125`, matching the current `32:10` and `8:2.5`
ratio. That makes the comparison cleaner because additive methods have the same
nominal relative step size at each radius.

Recommended minimal cases:

- `loss1 / all_w`
- `loss2 / all_a_target_w`
- `loss3 / all_w`

Do not rerun `epsilon=32, alpha=10` or `epsilon=8, alpha=2.5` unless the current
records are incomplete. Treat those as already-covered baselines.

Interpretation target:

- If the hypothesis is right, smaller epsilon should make the attack more local
  and more linear. Replacement/GPI should become closer to `steepest_add`, and
  may win on some objectives.
- If `steepest_add` still strongly dominates even at `epsilon=1` or `epsilon=0.5`,
  then the explanation cannot be only finite-radius nonlinearity; it would point
  toward method/objective differences even in a small-radius regime.

Caveat:

- Very small epsilon can make true-loss changes small, so final-loss differences
  may become close to numerical/noise-level effects. For that reason,
  `epsilon=2` and `epsilon=1` are the most useful aggressive probes; `epsilon=0.5`
  is optional if the first two still show a clear trend.


## 2026-05-23 Revision: Frozen-Linearized Test Is Secondary, Not The Main Evidence

The user correctly pointed out that a frozen-linearized test changes the actual
optimization problem. The real attack is nonlinear, finite-radius, recurrent,
and solver/mode dependent. Freezing it into a local linear objective creates an
artificial diagnostic problem, not the real adversarial optimization task.

Revised interpretation of frozen-linearized testing:

- It can still be useful as a sanity check: if replacement/GPI wins on the
  frozen local problem, that shows the implementation and local-linear intuition
  are not broken.
- But it should not be treated as the main evidence for the actual 2D NS attack.
- If the frozen test disagrees with the full nonlinear attack, that disagreement
  is expected and only says the real problem is more nonlinear/path-dependent.
- Therefore it is a secondary mechanism probe, not a priority production run.

The primary evidence should come from the real full nonlinear attack:

1. smaller-epsilon full attacks,
2. alpha/step-size sensitivity in the full attack,
3. boundary-matched true-loss comparisons in the full attack,
4. early-to-final cosine and gradient-rotation diagnostics from real attack
   trajectories.

Sample-count expansion is also deprioritized. The current goal is mechanism
validation, not population-level statistics. Increasing from 10 to 20 or 30
initial conditions is not necessary unless a later result is ambiguous or a
paper-quality uncertainty estimate is needed.

Revised priority order:

1. Run missing smaller-epsilon real attacks:
   - `eps4_alpha1p25`
   - `eps2_alpha0p625`
   - `eps1_alpha0p3125`
   - optional `eps0p5_alpha0p15625`
2. Use only the minimal real cases first:
   - `loss1/all_w`
   - `loss2/all_a_target_w`
   - `loss3/all_w`
3. Re-run CPU-only offline diagnostics:
   - boundary-matched true loss,
   - early-to-final cosine,
   - gradient/update rotation.
4. Run alpha sensitivity only if the epsilon trend is unclear.
5. Frozen-linearized test only if we specifically need a sanity check about the
   local-linear limit; do not treat it as the real attack result.

Bottom line:

The central question is not whether a simplified linearized problem behaves like
replacement/GPI. The central question is whether the actual full nonlinear 2D NS
attack changes optimizer ranking as epsilon becomes smaller. That should be the
main experiment.
