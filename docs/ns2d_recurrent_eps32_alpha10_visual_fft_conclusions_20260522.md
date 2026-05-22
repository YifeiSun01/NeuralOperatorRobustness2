# NS2D Recurrent FNO eps32 alpha10 Visual/FFT Conclusions - 2026-05-22

## Scope

This note consolidates the main conclusions from the `epsilon=32`, `alpha=10`, `p=q=2` NS2D recurrent FNO attack visualizations and FFT diagnostics. It is based on saved attack outputs under:

`2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/full_adw_b10_pair_outer_baseline_first_20260522/eps32_alpha10`

No new model/solver run is implied by this summary.

## Key Visual Folders

Images were collected into:

`2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_visual_fft_report_package_20260522/images_only`

CSV/JSON/Markdown records were collected separately into:

`2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_visual_fft_report_package_20260522/records`

The image folder intentionally contains only image files.

## Main Conclusions

### 1. Loss2 Perturbations Are the Smoothest / Lowest Frequency

Observed from final-delta FFT metrics and heatmaps: `loss2/all_a_target_w` final deltas are the smoothest and most low-frequency-looking family. In the radial FFT summary for `final_delta`, `loss2` had the lowest spectral centroid among the three loss groups.

Interpretation: `loss2` uses a fixed clean target and dictionary-provided frames in the active attack setup, so it does not contain the same differentiable perturbed-solver target path as `loss3`. This likely contributes to its smoother/lower-frequency perturbations.

### 2. Loss3 Perturbations Carry More Mid-Frequency / Curved Spatial Structure

Observed from final-delta heatmaps and radial spectra: `loss3` shifts more power into low-mid/mid spatial frequencies than `loss1` or `loss2`. Visually, `loss3` deltas look more curved and spatially structured.

Important nuance: this is not a large Nyquist-scale high-frequency explosion. The strict high-frequency band remains small; the shift is mostly from very low frequency into low-mid/mid frequency.

### 3. LP Steepest PGD Is Visually and Quantitatively Important in 2D

Observed from final delta, final output, and loss-curve plots: `steepest_add` / LP Steepest PGD often produces the largest and most distinctive perturbation structures, especially in `loss3/all_w` and related W-heavy `loss3` modes.

Interpretation: in this 2D recurrent NS setting, LP Steepest PGD is choosing a materially different perturbation direction, not merely making the same perturbation larger. This differs from the earlier 1D Burgers intuition where replacement/GPI-style methods looked more dominant.

### 4. Replacement/GPI-Style Methods Are Not Always Best Here

Observed from the `eps32 alpha10` baseline curves: `steepest_replace`/GPI-style methods do not dominate the final true-loss increase in the same way as expected from the earlier 1D Burgers experiments.

Interpretation: the finite-epsilon, recurrent, nonlinear 2D NS attack landscape is not behaving like a fixed local quadratic problem. Additive LP-steepest updates can keep accumulating useful direction changes after reaching the boundary.

### 5. PGD/Raw-Add Norm Curves Can Overlap Replacement Curves While Producing Different Deltas

Observed evidence from earlier numeric checks: in several blocks, `raw_add` has nearly the same `||delta||/epsilon` curve as `raw_replace`/`steepest_replace`, but the final perturbation cosine similarity can be low and the final losses can differ substantially.

Interpretation: overlap in delta norm only means the perturbations have the same magnitude. It does not mean they point in the same spatial direction.

### 6. The 2/3 Cutoff / 3/2-Rule Fingerprint Is Strongest in Solver-Related Fields

Corrected visual interpretation from no-cutoff FFT heatmaps: the clean and adversarial FNO/model output FFTs do not show a visually sharp `2/3` cutoff box in the same strong way as solver outputs or solver-change outputs.

The stronger visual evidence of the cutoff appears in:

- clean/adv solver final outputs,
- solver final change,
- final deltas whose gradients pass through solver/dealiasing paths.

Terminology: this note calls the visible retained-mode boundary the `2/3 cutoff`, associated with the pseudo-spectral `3/2` dealiasing rule. The user sometimes refers to it as the `3/2 line`.

### 7. Model and Solver Outputs Have Similar Main Spectral Directions, But Solver Has the Clear Cutoff

Observed visually from clean model output and clean solver output FFT heatmaps: their dominant spectral shapes/directions are very similar, with comparable angular structure. The important difference is that solver-related fields show a clearer `2/3` cutoff boundary, while model outputs do not show it as sharply by eye.

Interpretation: the FNO learned the main low-frequency/dominant-direction structure well, but not the exact hard spectral projection/dealiasing boundary as a visible box.

### 8. Final Delta Can Inherit Solver Cutoff Structure Through Backpropagation

Observed from final delta FFT heatmaps: `loss1` and `loss3` deltas can show a cutoff-box-like structure even when model output itself does not visibly show the box.

Interpretation: if the differentiable solver path includes a spectral projection/dealiasing mask, the backward/adjoint path can also be shaped by that mask. In simple terms, a forward projection `P` tends to produce a backward `P^T`; for a Fourier mask this is essentially the same frequency selector.

### 9. Why Loss1 Can Show Solver Fingerprints

`loss1` is `||F(x+delta)-F(x)||`, so its target is not `G(x+delta)`. However, in `all_w` mode the recurrent FNO input frames 2-10 are generated by the solver and remain in the differentiable path. Therefore `loss1/all_w` can still inherit solver/dealiasing structure in the gradient.

### 10. Why Loss2 Can Lack the Same Fingerprint

`loss2/all_a_target_w` uses a fixed clean target and dictionary-provided frames in the active attack mode. It does not contain the same differentiable perturbed-solver target path as `loss3`, and it does not use the all-W differentiable solver input path in the same way as `loss1/all_w`.

This helps explain why `loss2` final delta looks smoother and lacks the same obvious cutoff-box structure.

### 11. Initial Condition Has Its Own Lower-Frequency Support

Observed from initial-condition FFT heatmaps: the clean initial condition appears lower-band-limited. This is consistent with a `64 x 64` source field being upsampled to `256 x 256`, which would naturally occupy only a lower-frequency subset on the final grid.

Interpretation: `x_adv = x_clean + delta` can visually combine two frequency structures:

- the original lower-band support from the clean initial condition,
- the solver-gradient-shaped `2/3` cutoff structure from `delta`.

### 12. Batch-Mean FFT Heatmaps Are Better Than Sample-0 Alone

The first no-cutoff FFT heatmaps were sample-position 0 only. The later batch-mean heatmaps are better for robust visual judgment because they compute:

`FFT per sample -> magnitude -> average over 10 samples -> log10 plot`

This avoids phase cancellation from averaging fields before FFT and reduces sample-specific speckle.

### 13. Turbulence Scale Interpretation

For spatial FFT analysis, large physical scales correspond to low wavenumber / low spatial frequency, and small physical scales correspond to high wavenumber / high spatial frequency. In turbulence language, a forward cascade means transfer toward smaller physical scales, i.e. toward higher wavenumbers. For 2D NS, energy and enstrophy cascade directions require extra care, but for these visualizations the FFT frequency interpretation is still: low frequency = large smooth structures; high frequency = small fine structures.

## Important Image Sets

- Baseline overview and loss/delta/angle curves: `images_only/eps32_alpha10_baseline_overview_20260522_v2`.
- Final-delta FFT/radial analysis: `images_only/eps32_alpha10_delta_fft_analysis_20260522`.
- Output FFT/dealiasing analysis with cutoff overlays: `images_only/eps32_alpha10_output_fft_dealias_analysis_20260522`.
- Method-grouped curves: `images_only/eps32_alpha10_method_grouped_curves_20260522`.
- No-cutoff sample-0 FFT heatmaps: `images_only/eps32_alpha10_fft_heatmaps_no_cutoff_20260522`.
- No-cutoff 10-sample batch-mean FFT heatmaps: `images_only/eps32_alpha10_fft_heatmaps_batchmean_no_cutoff_20260522`.

## Remaining Work

- Repeat the same FFT/no-cutoff/batch-mean diagnostics for the other epsilon/alpha pairs after they finish.
- Compare whether the solver-cutoff fingerprint persists or weakens as epsilon/alpha changes.
- Check whether LP Steepest PGD remains dominant for non-baseline pairs and for all `loss3` ADW modes.

### 14. Corrected Model-vs-Solver Cutoff Interpretation From Grouped Curves

Observed from the method-grouped radial FFT curves and no-cutoff heatmaps: the clean and adversarial model/FNO outputs have similar dominant spectral directions to the solver outputs, but the model curves do not show the same hard `2/3` cutoff/drop as the solver. The solver curves and solver-change curves show the cutoff much more clearly.

Corrected interpretation: the FNO learned the main low-frequency and angular spectral structure of the solver outputs, but it did not learn the exact pseudo-spectral hard cutoff as a visually clear box/edge. Therefore, model-output low outside-cutoff energy should not be over-interpreted as the model having learned the solver's `3/2` dealiasing rule exactly.

### 15. Why Loss2 Final-Delta Radial Curve Looks Strange

Observed from the final-delta radial profiles: `loss2/all_a_target_w` concentrates even more strongly in the first radial bin than `loss1/all_w` and `loss3/all_w`. For example, the first radial bin (`rho ~= 0.0052`) contains about `0.91-0.93` of the normalized radial profile for loss2 methods, versus about `0.75-0.82` for loss1 and about `0.55-0.70` for loss3/all_w.

This explains why the curve plot makes loss2 look especially high at the low-frequency starting point. The spatial heatmap does not necessarily look obviously wrong because a very smooth, large-scale perturbation can look visually simple in physical space while being extremely concentrated in the first few Fourier bins.

Important plotting distinction: the curve plot is a radial profile on a log y-axis and makes low-frequency concentration very obvious. The FFT heatmap uses a per-image log magnitude colormap with percentile scaling, so the same concentration can be less obvious by eye.

### 16. Why Solver Radial Spectra Can Rise Near rho 0.47-0.55 Before Dropping Near 2/3

Observed from the method-grouped solver radial FFT curves: solver-related spectra can show a visible feature or slight rise around `rho ~= 0.47-0.55`, before the clearer drop near `rho ~= 2/3`.

Interpretation: this is most likely a geometry/anisotropy effect of radial averaging a square pseudo-spectral dealiasing mask, not a second independent physical cutoff. The solver's `2/3` retained-mode rule is per coordinate: `|kx| <= 1/3` and `|ky| <= 1/3` in normalized FFT coordinates. In the radial coordinate used in these plots:

- the axis intercept of that square occurs at `rho = sqrt(2)/3 ~= 0.471`,
- the diagonal corner of that square occurs at `rho = 2/3 ~= 0.667`.

Therefore the radial profile has a transition band from about `0.47` to `0.67`: below `0.47`, a full radial circle lies inside the retained square; between `0.47` and `0.67`, only some angular sectors of the radial ring remain inside the square; beyond `0.67`, the whole ring is outside the retained square.

Because the fields are anisotropic and have preferred spectral directions, the remaining angular sectors between `rho ~= 0.47` and `rho ~= 0.67` can have stronger average power than neighboring sectors. On a log-y radial plot this can look like a bump or sharp boundary before the final drop. The current radial-profile implementation uses mean power per radial bin and then normalizes the profile, so it highlights average intensity in surviving angular sectors rather than shell-integrated total energy.

Working conclusion: the feature near `rho ~= 0.5` is best interpreted as the radial signature of a square Fourier cutoff plus anisotropic spectral directions. It should not be overread as a separate physical cascade scale without additional angular-sector diagnostics.

### 17. Radial Frequency Normalization: Why 0.471 and 0.667 Appear

The radial FFT plots use a normalized radial coordinate `rho`, not the raw `kx` or `ky` coordinate. In these plots, `rho=1` is the distance from the FFT center to the diagonal corner of the square Fourier domain, not the distance from the center to the horizontal-axis Nyquist point.

The normalized Fourier coordinates are:

```text
kx, ky in [-0.5, 0.5]
```

The maximum radial distance in the square FFT plane is the corner distance:

```text
sqrt(0.5^2 + 0.5^2) = sqrt(2) / 2
```

Therefore the plotted radial coordinate is:

```text
rho = sqrt(kx^2 + ky^2) / (sqrt(2) / 2)
```

The pseudo-spectral `2/3` retained-mode cutoff is applied per coordinate direction:

```text
|kx| <= 0.5 * (2/3) = 1/3
|ky| <= 0.5 * (2/3) = 1/3
```

So the retained region is a square, not a circle.

The first important radial location is the side of this square along an axis, for example `(kx, ky) = (1/3, 0)`:

```text
rho = (1/3) / (sqrt(2) / 2)
    = sqrt(2) / 3
    ~= 0.471
```

The second important radial location is the corner of the retained square, `(kx, ky) = (1/3, 1/3)`:

```text
rho = sqrt((1/3)^2 + (1/3)^2) / (sqrt(2) / 2)
    = (sqrt(2) / 3) / (sqrt(2) / 2)
    = 2/3
    ~= 0.667
```

This is why a square `2/3` cutoff appears in the radial plot as a transition band from about `rho=0.471` to `rho=0.667`, rather than as a single cutoff point.

### 18. Important Normalization Clarification: rho=1 Is the Diagonal, Not the X Axis

The key clarification is:

```text
rho=1 means the FFT square corner distance, not the horizontal-axis Nyquist distance.
```

Equivalently, the plot normalizes by the diagonal radius:

```text
sqrt(0.5^2 + 0.5^2) = sqrt(2) / 2
```

Thus:

```text
(kx, ky) = (0.5, 0)       -> rho = 1 / sqrt(2) ~= 0.707
(kx, ky) = (0.5, 0.5)     -> rho = 1
(kx, ky) = (1/3, 0)       -> rho = sqrt(2) / 3 ~= 0.471
(kx, ky) = (1/3, 1/3)     -> rho = 2/3 ~= 0.667
```

In the user's wording: the plot effectively treats the diagonal distance as `1`, so the horizontal-axis Nyquist point is only `0.707` in this radial normalization.

### 19. Linear-Y Final-Delta Radial FFT Plot for Low-Frequency Separation

Observed after regenerating the `eps32_alpha10 final_delta radial FFT by method/all blocks` figure with a linear y-axis: the low-frequency separation is clearer than in the log-y method-grouped plot. A low-frequency zoom was also generated.

Generated plots:

- `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_final_delta_radial_linear_y_20260522/eps32_alpha10_final-delta_radial_fft_by_method_all_blocks_linear_y.png`
- `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_final_delta_radial_linear_y_20260522/eps32_alpha10_final-delta_radial_fft_by_method_all_blocks_linear_y_lowfreq_zoom.png`

Observed first radial-bin values (`rho ~= 0.0052`) for representative blocks:

| Method | `loss1/all_w` | `loss2/all_a_target_w` | `loss3/all_w` |
|---|---:|---:|---:|
| `raw_add` | 0.821 | 0.926 | 0.700 |
| `raw_replace` | 0.808 | 0.909 | 0.616 |
| `steepest_add` | 0.746 | 0.914 | 0.546 |
| `steepest_replace` | 0.808 | 0.909 | 0.616 |

Interpretation: `loss2` really is more concentrated at the very lowest radial frequency bin. In the log-y method-grouped figure, values such as `0.91`, `0.80`, and `0.62` all sit near `10^0`, so the distinction is visually compressed. The linear-y and low-frequency zoom plots make the difference easier to see.

## Optimizer Behavior: 1D Burgers vs 2D NS Recurrent FNO

Updated: 2026-05-22 22:23:23 UTC

Observed from the eps32/alpha10 2D NS recurrent FNO attack plots and records, the strongest method is often `steepest_add` / LP-steepest PGD, while the prior 1D Burgers experiments often favored `steepest_replace` / generalized-power-style replacement. This is not necessarily a contradiction. It points to a different local optimization geometry.

Observed from current 2D NS records:

- `steepest_add` can produce the largest true-loss increase and visibly strongest final-output changes, especially in W-heavy loss3 cases.
- `steepest_replace` / replacement-style updates are not consistently best, even though they jump directly to the L2 boundary.
- For `p=q=2`, the replacement variants are mathematically close or identical in direction: they replace `delta` by an epsilon-scaled normalized gradient/steepest direction. Their weakness is not that they fail to hit the boundary; rather, they discard trajectory history.
- `raw_add` can be too step-size dependent because it uses raw gradient magnitude. `steepest_add` normalizes the update direction, so it separates direction choice from gradient scale while still accumulating useful direction history.

Inference from the 1D Burgers comparison:

- The 1D Burgers case appears closer to a local affine or quadratic objective, roughly like optimizing `||A delta||` or a Rayleigh-type form over an L2 ball.
- In that regime, a dominant adversarial direction can be stable across steps. Replacement / generalized power iteration is well matched: it repeatedly points `delta` toward the current dominant direction and can converge quickly.
- Additive PGD is less special in that setting because it may spend steps accumulating old directions when the best solution is essentially one dominant vector.

Inference for why 2D NS differs:

- The 2D NS attack is finite-radius, recurrent, and solver-coupled. The objective is not a fixed quadratic in `delta`; the model rollout, solver rollout, ADW mode choice, and final-time supervision all change the gradient field along the trajectory.
- The gradient direction appears to rotate as `delta` changes. In that geometry, replacement can chase the current local direction and throw away components that were useful earlier, while additive PGD keeps and rotates accumulated structure.
- Solver/dealiasing paths introduce spectral filtering into the backward signal. The final deltas show frequency fingerprints that are not just model artifacts. This makes the useful perturbation a mixture of low-frequency structure, solver-filtered bands, and mode-dependent components rather than one clean top singular vector.
- `loss2` is smoother/fixed-target/dictionary-like, while `loss3` includes perturbed solver targets and differentiable solver paths. Therefore optimizer ranking can change by loss/mode; the ADW mode is part of the objective, not merely a label.

Practical interpretation:

- For this 2D NS recurrent FNO setting, `steepest_add` is the current best baseline to trust for throughput-quality tradeoff.
- `steepest_replace` remains an important diagnostic because it tests whether the problem behaves like a dominant-direction/generalized-power problem. Its weaker result is evidence that the 2D NS attack geometry is more path-dependent than the 1D Burgers case.
- To verify the mechanism, the next diagnostic should compare smaller epsilon values (`8`, `16`) and a frozen-linearized/JVP-only objective. If replacement becomes strong in the frozen-linearized case but not in the full solver-coupled attack, that would support the explanation above.

## Validation Plan For The Optimizer-Geometry Hypothesis

Updated: 2026-05-22 22:25:39 UTC

Hypothesis to test: the 1D Burgers attack behaved closer to a locally quadratic / dominant-direction problem, where replacement or generalized-power-style updates are well matched. The 2D NS recurrent FNO attack is more finite-radius, nonlinear, recurrent, solver-coupled, and gradient-rotation dominated, so additive normalized updates (`steepest_add`) can outperform replacement updates.

Validation tests:

1. Frozen-linearized diagnostic.
   - Freeze one clean input and compute a fixed local linear/JVP-VJP objective around it.
   - Compare `raw_add`, `raw_replace`, `steepest_add`, and `steepest_replace` on this frozen objective.
   - Expected result if the hypothesis is right: replacement / generalized-power-style methods should become much more competitive or strongest under the frozen linearized objective.
   - Interpretation: if replacement wins only in the frozen-linearized test but not in the full recurrent solver-coupled attack, the full problem is not behaving like a fixed quadratic.

2. Epsilon sweep.
   - Run the same loss/mode/method set at smaller radii, for example `epsilon = 4, 8, 16, 32`, with alpha scaled so each method reaches the boundary at comparable times.
   - Expected result: at smaller epsilon, the objective should be more locally linear/quadratic, so replacement should improve relative to additive updates. At larger epsilon, `steepest_add` should retain more advantage if nonlinear path dependence is the cause.

3. Gradient rotation / trajectory-angle diagnostic.
   - Record angles or cosine similarities for `grad_k` vs `grad_(k-1)`, update direction `u_k` vs `delta_k`, and `delta_k` vs `delta_(k-1)`.
   - Expected result: if gradients rotate strongly in 2D NS, replacement methods will repeatedly change direction, while additive methods will accumulate and rotate the perturbation more smoothly.

4. Boundary-hitting controlled comparison.
   - Match methods by delta norm and boundary-hitting time, not just by step count.
   - Compare true loss at fixed delta-norm milestones: 25%, 50%, 75%, and 100% of epsilon.
   - Expected result: if `steepest_add` is genuinely better, it should have higher true loss even at matched delta norms, not merely because it reaches the boundary earlier or later.

5. Solver-gradient ablation.
   - Compare W, D, and A-style modes where solver gradients are kept, detached, or replaced by dictionary/fixed targets.
   - Expected result: if solver/dealiasing gradients matter, the optimizer ranking and FFT cutoff fingerprints should change when the solver gradient path is detached or replaced.

6. Spectral projection ablation.
   - Constrain delta/update directions to low-frequency, high-frequency, and dealias-mask bands.
   - Expected result: if the `steepest_add` advantage is tied to accumulating useful multi-band spectral structure, its advantage should shift under these frequency restrictions.

7. True-loss vs surrogate-loss check.
   - For every method, log both the optimized surrogate objective and the full solver-evaluated true loss.
   - Expected result: replacement may look good on a local surrogate but worse on true loss if it chases unstable current gradients.

Priority order:

- First: use existing logs to inspect boundary milestones, method curves, and available angle diagnostics.
- Second: run the epsilon sweep with a small sample count.
- Third: implement frozen-linearized/JVP diagnostic. This is the cleanest test of whether the 2D full attack departs from the quadratic/GPI regime.
