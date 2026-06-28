# Darcy Loss4 Physics-Residual Attack

This note documents the new Darcy Flow attack objective requested after loss1/loss2/loss3.

## Script

```text
2D_Darcy_FNO2d/perturbation_methods/attack_darcy_binary_physics_loss4.py
```

The script keeps the same binary coefficient-field attack geometry as the existing Darcy attacks:

```math
A \in \{3,12\}^{N\times N}
```

and uses a Hamming flip budget over valid pixels.

## Loss Definitions

Existing losses:

```math
L_1(A)=\|f(A)-f(A_0)\|
```

```math
L_2(A)=\|f(A)-g(A_0)\|
```

```math
L_3(A)=\|f(A)-g(A)\|
```

where:

- `f` is the trained Darcy FNO model;
- `g` is the numerical Darcy solver;
- `A0` is the clean coefficient field;
- `A` is the attacked coefficient field.

New optimized objective:

```math
L_4(A)=\mathcal{L}_{phys}(f(A)).
```

For Darcy Flow, the PDE is:

```math
-\nabla \cdot (A \nabla u)=1,
\qquad u|_{\partial\Omega}=0.
```

The script computes a differentiable torch finite-difference residual:

```math
r(A,u_\theta)= -\nabla\cdot(A\nabla u_\theta)-1,
\qquad u_\theta=f(A).
```

The loss has two logged components:

```math
L_{4,pde}(A)=\frac{\|r(A,f(A))\|_2}{\|1\|_2}
```

```math
L_{4,bc}(A)=
\left(
\frac{1}{|\partial\Omega|}
\sum_{x\in\partial\Omega} |f(A)(x)|^2
\right)^{1/2}
```

The default optimized loss includes the homogeneous Dirichlet boundary
condition:

```math
L_4(A)=L_{4,pde}(A)+\lambda_{bc}L_{4,bc}(A),
\qquad \lambda_{bc}=1.0.
```

The boundary weight can be changed with `--bc-weight`, but the default is not
zero. This is important: the physics loss is not only the differential
equation residual; it also checks the boundary condition.

## StablePDENet Analogy

This matches the StablePDENet-style idea:

```math
\max_\delta \mathcal{L}_{phys}(G_\theta(a+\delta)).
```

The attack objective does not call the numerical solver. The solver is used only for logging/evaluation of loss1/loss2/loss3 at each step.

Therefore this lets us compare:

```math
\text{surrogate attack objective} = L_4(A)
```

against

```math
\text{true solver-consistent error} = L_3(A).
```

## Output Files

Each run writes:

```text
experiment_manifest.json
trace.csv
step_losses_per_sample.csv
step_sample_trace.npz
final_per_sample.csv
final_outputs.pt
summary.json
figures/final_panels.png
figures/loss_curves.png
```

Important columns in `trace.csv`:

- `loss4_mean`: physics residual surrogate being optimized;
- `loss4_pde_mean`: PDE residual component of loss4;
- `loss4_bc_mean`: homogeneous Dirichlet boundary component of loss4;
- `loss1_mean`, `loss2_mean`, `loss3_mean`: recorded comparison losses;
- `loss1_increase_mean`, `loss2_increase_mean`, `loss3_increase_mean`, `loss4_increase_mean`: stepwise increase over clean baseline;
- `true_loss3_mean`: alias for solver-consistent attacked error;
- `flip_count_mean`: binary Hamming budget used.

## Boundary-Including Smoke Test

A one-step smoke test with boundary weight 1.0 completed successfully:

```text
2D_Darcy_FNO2d/perturbation_results/binary_loss4_physics_smoke/smoke_loss4_physics_bc_1step
```

Result summary:

```text
clean loss4 physics mean: 2.443891
clean loss4_pde mean:     2.443783
clean loss4_bc mean:      0.000108
final loss4 physics mean: 2.453661
final loss4_pde mean:     2.453554
final loss4_bc mean:      0.000107
clean true loss3 mean:    0.018309
final true loss3 mean:    0.018279
```

The smoke test shows the physics residual objective increased, while true loss3
did not increase for this tiny one-step, one-sample run. That is exactly the
kind of surrogate-vs-true mismatch this loss4 experiment is meant to measure.

## Completed Full Run Matching The Previous Loss1/Loss2/Loss3 Setup

This run follows the previously documented Darcy binary attack setup:

- resolution: `211 x 211`;
- samples: `50`;
- steps: `100`;
- coefficient field: binary `{3, 12}`;
- Hamming budget: `epsilon=0.01`, giving `437` flips on the valid interior pixels;
- method: `steepest_replace`;
- boundary condition: included with `bc_weight=1.0`.

Run output:

```text
2D_Darcy_FNO2d/perturbation_results/binary_loss4_physics/darcy_loss4_physics_steepest_replace_nx211_N50_eps001_alpha5_steps100_bc_20260529
```

Command:

```bash
/workspace/NeuralOperatorRobustness2/adv_robust/bin/python \
  2D_Darcy_FNO2d/perturbation_methods/attack_darcy_binary_physics_loss4.py \
  --dataset 2D_Darcy_FNO2d/datasets/grf_darcy_20260528_N1500/test/dim2d_darcy_nx211_N300_solver=jaxcg_solve421_alpha2_tau3_binary3-12_f1_seed45_test.pt \
  --num-samples 50 \
  --steps 100 \
  --epsilon 0.01 \
  --alpha-flips 5 \
  --method steepest_replace \
  --trace-sample-index 0 \
  --plot-samples 4 \
  --run-name darcy_loss4_physics_steepest_replace_nx211_N50_eps001_alpha5_steps100_bc_20260529 \
  --output-root 2D_Darcy_FNO2d/perturbation_results/binary_loss4_physics
```

Final mean results:

```text
clean loss1 mean:          0.000000
clean loss2 mean:          0.026588
clean loss3 mean:          0.026588
clean loss4 physics mean: 17.712782
clean loss4_pde mean:     17.712658
clean loss4_bc mean:       0.000124

final loss1 mean:          0.011647
final loss2 mean:          0.028078
final true loss3 mean:     0.025949
final loss4 physics mean: 36.435963
final loss4_pde mean:     36.435841
final loss4_bc mean:       0.000123

loss1 increase mean:       0.011647
loss2 increase mean:       0.001490
loss3 increase mean:      -0.000639
loss4 increase mean:      18.723179
loss4_pde increase mean:  18.723179
loss4_bc increase mean:   -0.000001
```

The main observation is sharp: optimizing the physics residual surrogate made
loss4 much larger, but the solver-consistent true loss3 did not increase in
this run. It decreased slightly on average. This is direct evidence that, for
this Darcy setup, a StablePDENet-style physics-residual attack objective is not
the same as attacking the true solver-referenced error.

## Notes

- `steepest_replace` follows the existing Darcy attack logic and projects directly to the full Hamming budget each step.
- `steepest_add` grows the flip set gradually by `--alpha-flips` per step.
- `loss4` includes both PDE residual and boundary residual by default.
- `loss4` is optimized without solver-in-the-loop, but `loss1/loss2/loss3` are recorded using the solver at each step.
- This directly tests whether a physics-residual attack also increases the solver-consistent error.

## Figure Bundle With Loss4

The original clean Darcy figure bundle was found on R2 under:

```text
analysis_outputs/darcy_flow_figures_clean_bundle_20260528_220215_UTC
```

A new local bundle was generated with loss4 added to the same figure family:

```text
analysis_outputs/darcy_flow_figures_clean_bundle_with_loss4_20260529
analysis_outputs/darcy_flow_figures_clean_bundle_with_loss4_20260529.tar.gz
```

The plotting code is:

```text
tools/plot_darcy_loss4_bundle_20260529.py
```

The new bundle contains:

- `01_multiK_sameK_loss_comparison_steepest_replace/K*_with_loss4`: same-K panels for K=437, 874, 2184, 4370, and 10920. The first row is the original clean baseline, followed by loss1/loss2/loss3/loss4. The `batch_mean` panels average the full batch of 50 samples, not just the displayed sample00-03 examples. In same-key panels, solver and model columns share one global color range across the whole same-key/multi-K bundle, and model-solver panels use one global symmetric color range and show the corresponding rel L2 loss3 value. These ranges are recorded in `01_multiK_sameK_loss_comparison_steepest_replace/shared_color_ranges.json`.
- `02_loss_curves`: three versions of the loss curves: no standard-deviation shading, standard-deviation shading with y-range based on the std band, and standard-deviation shading with y-range based on the line values. All multi-panel figures use a shared y-axis range.
- `03_loss4_gifs`: loss4 step-trace GIFs for the four core methods.
- `loss4_bundle_manifest.json`: machine-readable run/path/metric summary for the 16 core curves.

Additional loss4 `steepest_replace` multi-K runs were generated for K=874, 2184, 4370, and 10920 under:

```text
2D_Darcy_FNO2d/perturbation_results/binary_loss4_physics_budget_sweep
```

The final true loss3 behavior for loss4 `steepest_replace` shows the surrogate/true gap clearly:

```text
K=437:   final loss4 36.435963,  final true loss3 0.025949
K=874:   final loss4 50.943165,  final true loss3 0.025942
K=2184:  final loss4 73.555412,  final true loss3 0.026721
K=4370:  final loss4 101.018448, final true loss3 0.030165
K=10920: final loss4 175.727280, final true loss3 0.052057
```

So small/medium K can strongly increase the physics residual without strongly increasing true solver-consistent loss3. At very large K, true loss3 eventually increases substantially as the binary coefficient field is changed much more aggressively.


All visible plot labels use `loss4`, not `loss4 physics`; the longer directory names only identify the underlying physics-residual run files.


## 2026-05-29 visualization style correction

The loss4 bundle was regenerated after matching the original clean Darcy visualization style more closely:

- A/model/solver heatmaps use `viridis`.
- Signed perturbation and model-solver difference heatmaps use red-blue `coolwarm`.
- Loss and method curve colors use the old blue/orange/green/red palette.
- Visible plot labels use `loss4` rather than `loss4 physics`.
- Formula text is moved to the bottom edge to avoid overlapping the panels.
- Loss4 GIFs now use the old 2x3 step-trace layout: A adv, ΔA, model, solver, model-solver, plus a loss progression panel with current frame/step markers.
- The GIFs are saved under `analysis_outputs/darcy_flow_figures_clean_bundle_with_loss4_20260529/03_loss4_gifs`.
- The four individual loss4 method GIFs now show only true `loss3` in the curve panel, not the optimized surrogate objective.
- The added multi-key GIF uses the maximum budget K=10920 and is `darcy_K10920_steepest_replace_loss1_loss2_loss3_loss4_multikey_true_loss3_step_trace.gif`. It has four rows for loss1/loss2/loss3/loss4 and a bottom true-loss3 curve panel with moving current-step markers.

The apparent missing lines in the loss2/loss3 optimized-objective plots are confirmed overlaps, not a plotting omission. For the old loss-method grid:

| loss | pair | max absolute surrogate-curve difference |
|---|---|---:|
| loss1 | raw_add vs steepest_add | 0.0025854893 |
| loss1 | raw_replace vs steepest_replace | 0.0036976263 |
| loss2 | raw_add vs steepest_add | 0.0 |
| loss2 | raw_replace vs steepest_replace | 0.0 |
| loss3 | raw_add vs steepest_add | 0.0 |
| loss3 | raw_replace vs steepest_replace | 0.0 |

So for loss2/loss3, the add pair and replace pair lie exactly on top of each other in those plots.


## 2026-05-29 loss4 observer curves

I added an offline observer plot that asks: while optimizing loss1/loss2/loss3/loss4, how does the physics loss4 value change along the same attack trajectory?

Output directory:

```text
analysis_outputs/darcy_flow_figures_clean_bundle_with_loss4_20260529/05_loss4_observer_curves
```

Files:

```text
darcy_K10920_steepest_replace_loss4_observer_curves.png
darcy_K10920_steepest_replace_loss4_observer_components.png
darcy_K10920_steepest_replace_loss4_observer_curves.csv
README.md
```

Setting: `K=10920`, `steepest_replace`, sample00 step traces. For each step, `A_adv` and `model_u` are plugged into the same Darcy physics-loss function used by the loss4 attack:

```text
loss4 = loss4_pde + loss4_bc
```

with `bc_weight=1.0` and `physics_metric=rel_l2`.

Final sample00 values:

| optimized loss | step0 loss4 | final loss4 | increase | ratio | final true loss3 |
|---|---:|---:|---:|---:|---:|
| loss1 | 21.479 | 33.7417 | 12.2627 | 1.57091 | 0.0529069 |
| loss2 | 14.6585 | 28.121 | 13.4625 | 1.91841 | 0.0443383 |
| loss3 | 14.6585 | 39.5498 | 24.8913 | 2.69807 | 0.21267 |
| loss4 | 14.6419 | 186.846 | 172.204 | 12.7611 | 0.0784569 |

Interpretation: loss1/loss2/loss3 attacks also move physics loss4 upward, but direct loss4 optimization increases the physics residual much more strongly. In this sample, loss3 gives the largest solver-consistent true loss3, while loss4 gives the largest physics residual; this is another example of the surrogate-vs-true mismatch.
