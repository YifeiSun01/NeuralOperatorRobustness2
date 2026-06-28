# Loss3 Surprising Findings Synthesis - 2026-05-20

Status: synthesis from existing local experiment artifacts and visual inspection notes; no new optimizer experiment was run for this note.

## Source Evidence

Observed from:

- `forensics/loss3_alpha_epsilon_core4_analysis_p2q2_300steps_20260520/core4_alpha_epsilon_method_summary.csv`
- `forensics/loss3_alpha_epsilon_core4_visuals_p2q2_300steps_20260520/`
- `forensics/loss3_alpha_epsilon_core4_analysis_pneq_q_stopped_100steps_20260520/core4_alpha_epsilon_method_summary.csv`
- `forensics/loss3_alpha_epsilon_core4_visuals_p1_q2_stopped_100steps_20260520/`
- `forensics/loss3_alpha_epsilon_core4_visuals_p1_qinf_stopped_100steps_20260520/`
- `forensics/loss3_alpha_epsilon_core4_visuals_p2_q1_stopped_100steps_20260520/`
- `forensics/loss3_alpha_epsilon_core4_visuals_p2_qinf_stopped_100steps_20260520/`
- `forensics/loss3_gpi_early_step_comparison_20260520/`
- Clean figure export with std and no-std pairs: `forensics/loss3_visuals_clean_export_20260520_2216_with_no_std/`
- Earlier summary notes: `docs/loss3_gpi_overall_conclusion_20260520.md`, `docs/loss3_alpha_epsilon_core4_p2q2_300steps_result_20260520.md`, and `docs/loss3_alpha_epsilon_core4_pneq_q_stopped_100steps_result_20260520.md`.

## Surprising Findings

### 1. Boundary arrival and loss convergence are not the same event

Observed: replacement/GPI methods reach the p-norm boundary at step 1 in the p2q2 sweeps, but their loss can continue to grow substantially afterward. At `epsilon=8, alpha=0.3` in the 100-step p2q2 evidence, `steepest_replace` hit the boundary at step 1 and later increased mean loss from about `2.720` at boundary hit to `6.805` final.

Inference: hitting `||delta||_p = epsilon` is only radial budget saturation. The real optimization still includes moving the perturbation direction along the boundary.

### 2. GPI/replacement is shockingly fast, but not always the 300-step final-loss winner

Observed: GPI/replacement reaches the boundary immediately and reaches a strong perturbation shape very early. In the saved baseline p2q2 trajectory, mean selected-sample `cos(delta_k, delta_300)` for GPI is `0.8868` at k=5, `0.9840` at k=10, and `0.9949` at k=20.

Observed: in 300-step p2q2 runs, `steepest_add` or raw PGD can match or slightly exceed GPI final mean loss for some alpha/epsilon settings. The 300-step rollup records strict largest-final-mean winners as `steepest_add` in 16 settings, `raw_add` in 1 setting, and replacement/GPI tied for largest final mean in 3 settings.

Inference: GPI's main advantage is speed and stability, not unconditional final-loss dominance after very long optimization.

### 3. LP-steepest additive PGD has nearly linear boundary-ratio growth, while raw PGD bends and slows

Observed from the no-std boundary-ratio curves: LP-steepest additive PGD often grows its perturbation norm in an almost straight line early on. Raw PGD typically rises quickly at first and then curves/slows as it approaches the boundary.

Inference: this is consistent with update geometry. LP-steepest uses a normalized steepest direction, so radial budget use is closer to controlled step-size accumulation. Raw PGD uses the raw gradient scale, so the effective radial step changes across samples and over time.

### 4. Boundary-ratio standard deviation is itself a diagnostic

Observed: for `epsilon=8, alpha=0.3` in p2q2 300-step diagnostics, per-sample 99% boundary steps were raw PGD mean/median/max `73.47/80/126`, LP-steepest additive `29.44/29/34`, and replacement methods `1/1/1`.

Observed: the boundary-ratio std for raw PGD was much larger than LP-steepest and replacement methods. Earlier records show p2q2 aggregate boundary-ratio std max/mean around `0.2954/0.0264` for raw PGD in 300-step runs, `0.0898/0.00114` for LP-steepest, and about `3e-08` for replacement/GPI.

Inference: raw PGD does not just move slower; different samples move to the boundary at very different speeds. LP-steepest synchronizes radial growth more strongly. Replacement/GPI almost eliminates radial variability by construction.

### 5. Fast angular motion seems to be a core reason GPI works

Observed: angle-dynamics figures show replacement/GPI-style methods rotate `delta` much more aggressively from step to step than raw PGD or LP-steepest additive PGD. Raw PGD has the slowest angular movement; LP-steepest is faster than raw PGD but still much less aggressive than replacement.

Inference: replacement/GPI avoids additive-update inertia. It can replace the old direction with a new full-budget direction and therefore explore the boundary surface much faster.

### 6. Final perturbation shapes are often surprisingly similar across methods

Observed: final-delta similarity and representative delta grids show that GPI, raw PGD, and LP-steepest often end with broadly similar perturbation shapes, even though their optimization paths differ substantially.

Observed: in p2q2 similarity records, additive methods cluster together and replacement/GPI methods cluster together, but spectral similarities can still be high. Earlier p2q2 records report raw_add vs steepest_replace spectral cosine about `0.8003`, and steepest_add vs steepest_replace spectral cosine about `0.8501`.

Inference: the methods may be finding similar large-scale structures in the loss landscape. The big difference is how fast and smoothly they reach those structures.

### 7. P/Q geometry changes perturbation realism dramatically

Observed from visual inspection: `p=2,q=1` can produce perturbations that still look physically plausible and not obviously spike-like. By contrast, `q=inf`-style settings, especially `p=1,q=inf` and `p=2,q=inf`, can show sharp localized peaks in representative samples.

Inference: changing Q changes what residual/loss structure is emphasized. `q=inf` can focus on extreme local residual points, which makes spike-like perturbations more likely. This is a geometry effect, not just an optimizer effect.

### 8. Raw replace and steepest replace can collapse to the same behavior in p=2 settings

Observed: for p2q2 records, `raw_replace` and `steepest_replace` are identical in final delta similarity: cosine `1.0000`, centered cosine `1.0000`, spectral cosine `1.0000`, and relative L2 `0.0000`.

Inference: in p=2 geometry, the raw replacement direction and the steepest/GPI replacement direction can become the same normalized direction. This explains why their curves and perturbations often overlap.

### 9. The best interpretation is Pareto, not one absolute winner

Observed: GPI/replacement is the fastest to boundary and often the fastest to useful loss, with low radial variance and often smooth final perturbations. But 300-step additive methods can sometimes produce slightly larger final mean loss.

Inference: the clean claim is not "GPI is always best." The clean claim is that GPI/replacement has a Pareto advantage for fast fixed-budget attacks: immediate budget use, rapid boundary-direction optimization, early final-like perturbation shape, and often comparable final loss/smoothness.

## Practical Takeaway

The most surprising overall story is that the optimization bottleneck is not simply reaching the epsilon boundary. It is how quickly the method can move directionally on that boundary after budget saturation. GPI/replacement is unusually strong because it does both: it reaches the boundary immediately, then rotates aggressively on the boundary. LP-steepest additive PGD fixes the radial step-size problem but still behaves like an additive method. Raw PGD has both radial desynchronization and angular inertia.

For future writeup, report both versions of each curve: mean +/- std to show variability/spike risk, and no-std to see the central trajectory and boundary markers clearly.
