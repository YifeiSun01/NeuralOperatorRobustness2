# Burgers vs 2D NS Unified Optimizer-Mechanism Experiment Plan

Updated: 2026-05-23 UTC

Status: design document only. No solver call, model inference, attack step, JAX import, PyTorch import, plotting, or GPU computation was started for this note.

## Goal

Build a matched experimental comparison between the 1D Burgers FNO attacks and the 2D NS recurrent FNO attacks, so that optimizer differences are explained by measured mechanisms rather than by incompatible settings.

The target question is:

```text
Why do replacement/GPI-style methods look strong in 1D Burgers, while 2D NS large-epsilon attacks currently favor LP-steepest additive PGD?
```

This requires comparing the same optimizer rules, same loss roles, same perturbation-budget normalization, and same diagnostics across both PDE systems.

## Working Hypotheses

### H1. Local-linear versus finite-radius nonlinear regime

Inference from existing records:

- Burgers often behaves closer to a stable local/dominant-direction problem, especially for early optimization and small perturbations.
- 2D NS at large epsilon behaves more like a finite-radius nonlinear path problem.

Prediction:

- As epsilon shrinks, 2D NS replacement/GPI should become more competitive.
- At large epsilon, 2D NS additive LP-steepest PGD should keep an advantage if path accumulation is the real mechanism.

### H2. Direction stability versus direction rotation

Inference from existing records:

- Burgers replacement/GPI directions become final-like quickly.
- NS2D replacement directions rotate strongly and can fail to become final-like early.

Prediction:

- Burgers should show high `cos(delta_k, delta_final)` by early steps.
- NS2D should show lower early-to-final cosine and larger `angle(delta_k, delta_{k-1})`, especially for replacement at large epsilon.

### H3. Solver coupling and recurrent rollout create stronger path dependence in 2D NS

Inference from model structure:

- Burgers attacks are lower-dimensional and usually shorter/simpler in rollout structure.
- NS2D recurrent FNO depends on 10-frame inputs and a solver-produced trajectory; loss3/all-w also differentiates through the solver path.

Prediction:

- NS2D W-heavy modes should show stronger rotation, stronger spectral solver fingerprints, and a larger advantage for additive path accumulation.
- D/A-heavy or fixed-target modes should behave more like partially frozen objectives and may move closer to Burgers behavior.

### H4. Spectral filtering and dimensionality affect optimizer geometry

Inference from current NS2D FFT analysis:

- Solver outputs show dealiasing/frequency-boundary structure more clearly than model outputs.
- Final deltas for loss1/loss3 can inherit spectral fingerprints from the optimization path.

Prediction:

- NS2D final deltas and solver-output changes should show stronger 2D spectral structure than Burgers 1D deltas.
- Low-frequency versus high-frequency energy distribution may explain why some optimizers create smoother, more effective perturbations.

## Matched Experimental Axes

### PDE systems

Run matched diagnostics on:

- 1D Burgers FNO attack.
- 2D NS recurrent FNO attack.

Use each system's established trained model, test split, solver, and attack implementation. Do not compare raw epsilon values directly across dimensions.

### Optimizers

Use exactly the same four update families:

- `raw_add`: raw gradient additive PGD.
- `raw_replace`: normalized raw-gradient replacement.
- `steepest_add`: LP-steepest additive PGD.
- `steepest_replace`: LP-steepest replacement / GPI-style update.

For `p=q=2`, `raw_replace` and `steepest_replace` may coincide or nearly coincide. Still log both names so the comparison stays explicit.

### Loss roles

Use three matched loss roles:

- `loss1`: model sensitivity, `||F(x + delta) - F(x)||`.
- `loss2`: fixed-target model error, `||F(x + delta) - G(x)||`, where the solver target is clean/fixed.
- `loss3`: moving-target model-solver error, `||F(x + delta) - G(x + delta)||`.

For 2D NS, mode labels are not a separate full cross-product for every loss. They define how the solver target is obtained:

- `loss1`: objective is model-only; solver can be logged for diagnostics but is not needed for the objective.
- `loss2`: fixed clean/dictionary target, closest to an A/fixed-target interpretation.
- `loss3/all_w`: moving solver target with differentiable solver path.
- `loss3` mixed W/D/A modes: NS2D-only ablation to isolate solver-gradient and dictionary effects.

### Perturbation-budget normalization

Raw epsilon is not comparable between 1D Burgers and 2D NS. Use normalized budgets:

```text
rho_l2 = epsilon_raw / median(||x_clean||_2)
rms_delta = epsilon_raw / sqrt(number_of_grid_points)
relative_rms = rms_delta / median(rms(x_clean))
```

For each system, report both:

- raw epsilon used by the code;
- normalized epsilon values above.

Recommended matched epsilon grid:

- choose the current NS2D `epsilon=32` as one anchor;
- compute its normalized `rho_l2` and `relative_rms`;
- choose Burgers raw epsilons that match those normalized values;
- then run smaller normalized radii: `1x`, `1/2x`, `1/4x`, `1/8x` of the anchor.

This avoids the false comparison where a raw 2D epsilon and raw 1D epsilon have completely different physical meaning.

### Alpha schedule

Use two complementary alpha controls:

1. Fixed `alpha / epsilon` schedule:

```text
alpha / epsilon = constant
```

This keeps additive methods under comparable step-to-boundary pressure.

2. Target boundary-arrival schedule:

Choose alpha so additive methods reach the boundary around:

- 10 steps,
- 25 steps,
- 50 steps.

This separates optimizer geometry from step-size tuning.

## Experiment Blocks

### Block A. Minimal matched optimizer-ranking sweep

Purpose: establish the clean comparison with the smallest compute.

For both Burgers and NS2D:

- losses: `loss1`, `loss2`, `loss3`;
- methods: four optimizers;
- p/q: `p=2`, `q=2`;
- samples: 10 initial conditions first;
- steps: 100 for NS2D; Burgers can use 100 and optionally 300 for compatibility with old records;
- epsilon grid: normalized anchor, `1/2`, `1/4`, `1/8`;
- alpha: fixed `alpha/epsilon` first.

Primary outputs:

- objective loss curve;
- true loss curve;
- surrogate loss curve when applicable;
- final loss;
- loss AUC over steps;
- boundary-matched true loss at 25%, 50%, 75%, 100% epsilon;
- per-sample mean, std, standard error, and method win rate.

Decision test:

- If NS2D replacement catches up at small normalized epsilon, H1 is supported.
- If NS2D steepest_add remains better at all normalized epsilons, the difference is deeper than only finite-radius size.

### Block B. Boundary-arrival matched sweep

Purpose: rule out the simple explanation that one method only wins because alpha reaches the boundary earlier.

For both PDEs:

- pick representative `loss1` and `loss3`;
- choose alpha values so additive methods reach 100% epsilon around 10, 25, and 50 steps;
- compare all four methods at the same normalized epsilon.

Primary outputs:

- step to 25%, 50%, 75%, 100% boundary;
- true loss at those boundary thresholds;
- true loss after reaching boundary;
- final loss.

Decision test:

- If steepest_add still wins after matching boundary arrival, it is not just a step-size artifact.
- If the ranking flips when boundary arrival is controlled, alpha tuning was a major confound.

### Block C. Direction-stability and rotation diagnostics

Purpose: measure whether each system has stable optimizer directions.

For every completed run, compute offline when trace data exist:

- `cos(delta_1, delta_final)`, `cos(delta_5, delta_final)`, `cos(delta_10, delta_final)`, `cos(delta_20, delta_final)`;
- `angle(delta_k, delta_{k-1})`;
- `angle(update_k, update_{k-1})`;
- `angle(delta_k, update_k)`;
- gradient/update norm and boundary ratio.

Decision test:

- Burgers replacement should become final-like quickly if the old mechanism is real.
- NS2D replacement should show stronger direction rotation at large epsilon if the current explanation is right.

### Block D. Path replay and ray-scan diagnostics

Purpose: separate final direction quality from path-following quality.

For selected samples and losses:

1. Take final delta from each method.
2. Evaluate loss along the same ray:

```text
L(x + t * delta_final), t in [0, 1]
```

3. Cross-evaluate each method's final delta under all loss definitions.
4. Replay additive accumulated deltas at matched boundary ratios.

Decision test:

- If replacement's final direction has a worse ray profile in NS2D, the issue is direction quality.
- If replacement's ray is good but it fails to follow it during optimization, the issue is trajectory/path instability.

### Block E. Spectral and spatial diagnostics

Purpose: compare what kind of perturbations each optimizer constructs.

For Burgers:

- 1D FFT magnitude profile;
- low/mid/high-frequency energy fractions;
- spectral centroid;
- total variation or derivative-energy proxy.

For NS2D:

- 2D FFT heatmaps;
- radial FFT profile;
- low/mid/high-frequency energy fractions;
- dealias-band energy near solver cutoff;
- anisotropy/orientation profile;
- spatial delta heatmaps and final-output difference heatmaps.

Apply these to:

- final delta;
- clean model output;
- adversarial model output;
- clean solver output;
- adversarial solver output;
- model-solver difference.

Decision test:

- If NS2D solver-gradient modes generate deltas with solver spectral fingerprints, H3/H4 are supported.
- If Burgers replacement perturbations are smoother or dominated by stable low-frequency modes, that helps explain why replacement can work quickly there.

### Block F. Solver-involvement ablation

Purpose: identify whether the solver gradient is the source of NS2D optimizer differences.

NS2D cases:

- `loss1`: model-only objective, solver only logged.
- `loss2`: fixed clean/dictionary target.
- `loss3/all_w`: full moving solver target with gradient.
- `loss3/all_d`: moving solver forward but detached backward.
- mixed W/D/A modes: only after the minimal cases are understood.

Burgers analogues:

- model-only sensitivity;
- fixed solver target;
- moving solver target.

Decision test:

- If replacement only fails badly when solver gradients are active, solver path nonlinearity is central.
- If replacement also fails in `loss1`, model/recurrent nonlinearity alone is enough.

### Block G. Statistical confirmation

Purpose: avoid over-interpreting a few samples.

Staged sample counts:

- stage 1: 10 samples for mechanism discovery;
- stage 2: 20 or 30 samples only for the few decisive settings;
- do not expand every mode/epsilon/method combination before the mechanism is clear.

Report:

- mean, std, standard error;
- per-sample method win rate;
- paired comparisons between methods on the same initial conditions.

## Recommended Run Order

### Phase 0. Offline consolidation from existing files

No GPU required if existing traces are present.

- Build one unified comparison table for old Burgers and current NS2D:
  - method;
  - loss role;
  - epsilon raw;
  - epsilon normalized;
  - alpha raw;
  - alpha/epsilon;
  - final loss;
  - boundary threshold steps;
  - boundary-matched losses;
  - early-to-final cosine;
  - rotation metrics.

### Phase 1. Minimal real matched sweep

Run the smallest new matched grid:

- systems: Burgers and NS2D;
- losses: `loss1`, `loss2`, `loss3`;
- methods: four;
- samples: 10;
- normalized epsilon: anchor, `1/4`, `1/8`;
- fixed `alpha/epsilon`.

This is the most important phase.

### Phase 2. Boundary-arrival control

Run only:

- `loss1` and `loss3`;
- anchor epsilon and one smaller epsilon;
- alpha chosen for 10, 25, and 50 boundary steps.

### Phase 3. Mechanism deep dive

Only after Phase 1/2 identify the decisive settings:

- path replay/ray scans;
- spectral analysis;
- solver W/D/A ablation;
- larger sample count.

## Main Evidence Needed For A Strong Conclusion

A convincing answer should contain four aligned plots/tables for both PDEs:

1. Optimizer ranking table:
   - final true loss;
   - boundary-matched true loss;
   - per-sample win rate.

2. Speed table:
   - steps to 25%, 50%, 75%, 100% epsilon;
   - steps to 50%, 90%, 95% of best final loss.

3. Direction table:
   - early-to-final cosine;
   - direction rotation angles;
   - update rotation angles.

4. Spectral/spatial table:
   - frequency energy fractions;
   - spectral centroid;
   - representative delta/output FFT plots.

## Expected Outcomes And Interpretation

If the current explanation is right, the final story should look like this:

```text
Burgers:
replacement/GPI is strong because it reaches the boundary immediately and the early full-budget direction quickly becomes close to the final useful direction.

2D NS:
large-epsilon replacement/GPI is weaker because the useful direction changes across the nonlinear rollout/solver path. Additive LP-steepest PGD wins by accumulating useful components instead of resetting delta every step.
```

If the data do not show this, revise the explanation:

- If NS2D small epsilon still favors steepest_add strongly, dimensional/recurrent structure may dominate even locally.
- If Burgers matched normalized epsilon also favors steepest_add in final and boundary-matched loss, the old Burgers conclusion should be rewritten as a speed/Pareto claim rather than a best-final-loss claim.
- If solver-detached NS2D behaves like Burgers but solver-with-gradient NS2D does not, solver-gradient path dependence is the main driver.

## What Not To Do First

Do not immediately run the full cross-product of:

- many epsilons;
- many alphas;
- all W/D/A modes;
- all losses;
- 30+ samples;
- long step counts.

That will be expensive and hard to interpret. First run narrow matched experiments that answer one mechanism question at a time.

## Short Practical Recommendation

Start with this minimal matched comparison:

```text
systems: Burgers, NS2D
losses: loss1, loss2, loss3
methods: raw_add, raw_replace, steepest_add, steepest_replace
p=q=2
samples: 10
normalized epsilons: anchor, anchor/4, anchor/8
alpha/epsilon: fixed
steps: 100, plus optional Burgers 300-step compatibility run
```

Then analyze:

```text
boundary-matched true loss
steps to boundary
steps to best-loss thresholds
early-to-final cosine
delta/update rotation
FFT/spectral energy of final delta and outputs
```

This gives a direct, fair answer to why the two PDE systems prefer different optimization behavior.
