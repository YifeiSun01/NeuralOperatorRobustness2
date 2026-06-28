# Loss3 Optimizer Mechanism Experiment Record And Plan

Updated: 2026-06-22 UTC

This note records the current status and the next mechanism experiments for
understanding why the four optimizer/update methods behave differently on
Burgers, Darcy Flow, and NS2D.

Unified theory-to-evidence note:

```text
docs/loss3_replace_add_theory_mechanism_unified_note_20260622.md
```

Concrete linearity/quadratic validation experiment plan:

```text
docs/loss3_replace_add_linearity_validation_experiment_plan_20260622.md
```

That note connects the replace-vs-add optimization theory, the Burgers mechanism
evidence, the NS2D/Darcy Flow hypotheses, and the next validation experiments.

The target comparison is always the same four update rules:

- `raw_add`
- `raw_replace`
- `steepest_add`
- `steepest_replace`

The target evaluation metric for the main comparison is true Loss3:

```text
Loss3 = || F_theta(x_adv) - G(x_adv) ||
```

For Burgers and NS2D, this is a continuous perturbation problem. For Darcy Flow,
this is a binary coefficient-flip problem, so the mechanism diagnostics must use
flip-set metrics rather than only continuous-vector angles.

## Current Formal Curve Status

Current formal optimizer-ablation root:

```text
analysis_outputs/optimizer_ablation_20260622
```

Current three-panel figure:

```text
analysis_outputs/optimizer_ablation_20260622/figures/loss3_three_system_optimizer_mean_curves.png
analysis_outputs/optimizer_ablation_20260622/figures/loss3_three_system_optimizer_mean_curves.pdf
```

Current sample counts after the NS2D and Darcy Flow top-ups completed:

| Problem | Setting | Current formal four-optimizer curve status |
| --- | --- | --- |
| Burgers 1D | `eps=8`, `alpha=0.3`, `steps=300` | `N=100` per optimizer |
| Darcy Flow | `epsilon_flips=437`, `alpha_flips=5`, `steps=100` | `N=20` per optimizer |
| NS2D Recurrent | `eps=32`, `alpha=10`, `steps=100`, `mode=all_w` | `N=20` per optimizer |

Important distinction:

- Darcy Flow has previous attack-objective and robustness experiments.
- Darcy Flow now also has the formal four-optimizer Loss3 curve data under
  `analysis_outputs/optimizer_ablation_20260622/raw_runs/darcy`.
- The current plotted curve table reports `N=20` for each Darcy Flow optimizer.

The NS2D top-up that completed on 2026-06-22 ran:

```text
start_index=10
num_samples=10
attack_batch_size=4
epsilon=32
alpha=10
steps=100
methods=raw_add raw_replace steepest_add steepest_replace
loss=loss3
mode=all_w
solver_remat=chunk
solver_remat_chunk_steps=20
dictionary_chunk_size=16
```

It completed the missing samples `10..19`. Together with the recovered old R2
samples `0..9`, NS2D now has `N=20` formal curve data.

## Existing Mechanism Evidence

### Burgers

Burgers has substantial mechanism evidence already. Key records:

```text
docs/loss3_current_mechanism_validation_summary_20260521.md
docs/loss3_gpi_mechanism_hypothesis_probe_20260520.md
docs/loss3_ray_profile_corrected_fno_nu0p001_gpu_batch20_result_20260516.md
docs/loss3_core4_pq_landscape_probe_full_20260521.md
docs/loss3_direction_rotation_path_fno_nu0p001_result_20260516.md
docs/loss3_directional_curvature_fno_nu0p001_steps100_samples5_20260517.md
docs/loss3_finite_difference_linearity_fno_nu0p001_steps100_fine_bins_20260517.md
```

Main Burgers mechanism conclusions from the existing records:

- Replacement/GPI-style methods reach the perturbation boundary very quickly.
- Reaching the boundary is not the whole story: there can still be tangent or
  angular motion along the boundary.
- For Burgers p2q2, early replacement directions become close to final
  directions quickly. In the prior landscape summary, `steepest_replace` was
  already near final-like by steps 5 to 10.
- Burgers has evidence for broad high-loss ridges or high-loss boundary regions:
  arcs between some final directions do not dip much.
- Local/small-radius steep directions and best finite-radius endpoint directions
  are not always the same; this is the local-to-global nonlinear gap.

Interpretation to test against NS2D:

```text
Burgers may be relatively forgiving because replacement quickly reaches a
useful high-loss direction and the high-loss boundary region is broad enough.
```

### NS2D

NS2D has existing curve, spectrum, and offline diagnostic evidence, but it does
not yet have a fully matched ray/profile/landscape mechanism experiment
parallel to the Burgers one.

Relevant records:

```text
docs/ns2d_recurrent_eps32_alpha10_baseline_overview_20260522.md
docs/ns2d_recurrent_eps32_alpha10_method_grouped_curves_20260522.md
docs/ns2d_recurrent_eps32_alpha10_delta_fft_analysis_20260522.md
docs/ns2d_recurrent_eps32_alpha10_output_fft_dealias_analysis_20260522.md
docs/ns2d_optimizer_validation_offline_diagnostics_20260523.md
docs/ns2d_vs_1d_burgers_optimizer_hypothesis_check_20260522.md
docs/burgers_vs_ns2d_unified_optimizer_mechanism_experiment_plan_20260523.md
docs/burgers_vs_ns2d_unified_experiment_coverage_20260523.md
```

Existing NS2D conclusions are suggestive but incomplete:

- NS2D at large epsilon appears more path-dependent than Burgers.
- Additive methods, especially `steepest_add`, can benefit from accumulated
  path-following rather than just replacing with a boundary direction.
- Existing NS2D spectral analyses show strong structure in final deltas and
  model/solver differences.
- The matched ray-scan/path-replay and local landscape-width experiment is still
  missing for NS2D.

Interpretation to verify:

```text
NS2D may have a narrower or more curved high-loss region. Replacement can jump
to the boundary but land on a poor boundary direction, while additive methods
can accumulate a better path through the nonlinear loss landscape.
```

### Darcy Flow

Darcy Flow has previous attack-objective, robustness, SVD, and correlation
experiments, but not the formal four-optimizer mechanism package.

Relevant records:

```text
docs/attack_objective_loss123_loss4_summary_20260622.md
docs/darcy_cflow_final_robustness_binary20260611_20260615.md
docs/darcy_cflow_epsilon_sweep_full_record_20260615.md
docs/darcy_cflow_residual_jacobian_svd_20260615.md
docs/darcy_cflow_model_solver_subspace_similarity_20260615.md
outputs/darcy_cflow_final_robustness_20260615/reports/final_robustness_summary_20260615.md
```

Important distinction:

```text
Darcy Flow is binary flip-based. Mechanism should be analyzed through flip-set
stability, overlap, prefix curves, and score ranking, not only continuous
cosine angles.
```

## Mechanism Hypotheses

### H0. Frozen-Jacobian Power-Style Explanation

For a locally linear residual map,

\[
r(\delta) \approx r_0 + J\delta,
\]

the squared local Loss3 has the form

\[
\frac{1}{2}\|r_0+J\delta\|_2^2
=
c+b^\top\delta+\frac{1}{2}\delta^\top A\delta,
\qquad
b=J^\top r_0,\quad A=J^\top J.
\]

If the residual offset \(b\) is ignored, or if the problem is closer to the
homogeneous quadratic model, then additive updates locally behave like

\[
\delta_{k+1}^{add}\approx (I+\alpha A)\delta_k,
\qquad
\delta_k^{add}\approx (I+\alpha A)^k\delta_0.
\]

Replacement updates locally behave more like a normalized power step:

\[
\delta_{k+1}^{replace}
\approx
\epsilon\frac{A\delta_k}{\|A\delta_k\|_2},
\qquad
\text{directionally similar to } A^k\delta_0.
\]

If \(A\) is fixed and has eigenvalues \(\lambda_1>\lambda_2\), then the
non-leading component decays like

\[
\left(\frac{\lambda_2}{\lambda_1}\right)^k
\]

for \(A^k\), but like

\[
\left(\frac{1+\alpha\lambda_2}{1+\alpha\lambda_1}\right)^k
\]

for \((I+\alpha A)^k\). Since

\[
\frac{1+\alpha\lambda_2}{1+\alpha\lambda_1}
>
\frac{\lambda_2}{\lambda_1},
\]

the pure replacement/power-style step should align faster in the frozen linear
model.

This is a useful local explanation for why `steepest_replace` can be very fast
in Burgers. However, it is not a full global theory because in the actual
nonlinear problem

\[
A=A_k=J(x+\delta_k)^\top J(x+\delta_k)
\]

changes along the path, and the offset term

\[
b_k=J(x+\delta_k)^\top r(\delta_k)
\]

also changes. Therefore, the important empirical question is not only whether
the scalar loss is locally linear, but whether the local direction field is
stable enough that the replacement/power-style step keeps pointing toward the
same high-loss boundary region.

Existing Burgers evidence:

- finite-difference linearity records show scalar Loss3 becomes much more
  locally linear along the path;
- early replacement directions become final-like quickly;
- landscape/ray diagnostics show broad high-loss boundary regions in p2q2-like
  settings.

Open NS2D test:

- use the mechanism trace to measure whether \(A_k\)-induced directions or
  objective gradients rotate strongly;
- compare `cos(delta_k, delta_final)`, `cos(update_k, update_{k-1})`, and
  post-boundary gain between replacement and additive methods.

### H0a. Is Strong Local Linearity Enough To Explain Replacement?

Discussion question recorded on 2026-06-22:

If the loss landscape is locally close to a frozen linear/Jacobian model, then
`replace` is closer to repeatedly applying \(A^k\), while `add` is closer to
repeatedly applying \((I+\alpha A)^k\). Since

\[
\left(\frac{\lambda_2}{\lambda_1}\right)^k
<
\left(\frac{1+\alpha\lambda_2}{1+\alpha\lambda_1}\right)^k,
\qquad \lambda_1>\lambda_2>0,
\]

the frozen linear model predicts faster top-direction alignment for replacement.

This is a good candidate explanation for Burgers, but it should be treated as a
hypothesis, not as the whole mechanism. Strong scalar linearity of the loss is
not sufficient by itself. Replacement should be good only when the following are
also true:

- the local direction field is stable, meaning \(A_k\) and \(b_k\) do not rotate
  the objective gradient too much as \(\delta_k\) changes;
- the early replacement direction has high cosine with the final high-loss
  direction;
- the high-loss boundary region is broad enough that a fast boundary jump does
  not miss a narrow peak;
- the offset term \(b_k=J_k^\top r_k\) does not dominate in a way that changes
  the target direction step by step.

Therefore the empirical test is:

\[
\text{Burgers replacement good}
\quad \Longleftrightarrow \quad
\cos(\delta_k,\delta_T) \text{ rises early and direction rotation is small},
\]

while

\[
\text{NS2D replacement weak}
\quad \Longleftrightarrow \quad
\cos(\delta_k,\delta_T) \text{ stays lower or update directions rotate more}.
\]

Added analysis utility:

```text
tools/analyze_step_sample_direction_stability_20260622.py
```

It reads `step_sample_trace.npz` files and outputs:

```text
analysis_outputs/mechanism_20260622/diagnostics/**/step_sample_direction_diagnostics.csv
analysis_outputs/mechanism_20260622/diagnostics/**/step_sample_direction_summary.csv
analysis_outputs/mechanism_20260622/diagnostics/**/step_sample_direction_diagnostics.png
```

The decisive columns are:

- `cos_delta_to_final_delta`;
- `cos_direction_to_final_delta`;
- `cos_direction_to_prev_direction`;
- `cos_grad_to_prev_grad`;
- `first_boundary99_k`;
- `post_boundary_true_loss_gain`.

### H0b. Four-Optimizer Speed And Optimality Framework

The four optimizer variants should be explained as different uses of the same
local first-order model, not as methods with global optimality guarantees.

The attack problem is

\[
\max_{\delta\in\mathcal B_p(\epsilon)} \Phi(\delta),
\qquad
\mathcal B_p(\epsilon)=\{\delta:\|\delta\|_p\le \epsilon\}.
\]

At step \(k\), let

\[
g_k=\nabla_\delta \Phi(\delta_k).
\]

The local first-order approximation is

\[
\Phi(\delta_k+\eta)
\approx
\Phi(\delta_k)+g_k^\top \eta.
\]

The steepest unit direction under the attack norm is

\[
s_k
=
\arg\max_{\|s\|_p\le 1} g_k^\top s,
\qquad
\max_{\|s\|_p\le 1} g_k^\top s
=
\|g_k\|_{p^\ast},
\]

where \(p^\ast\) is the dual norm exponent. For \(p=2\),

\[
s_k=\frac{g_k}{\|g_k\|_2}.
\]

For \(p=\infty\),

\[
s_k=\operatorname{sign}(g_k).
\]

For \(1<p<\infty\), one convenient expression is

\[
(s_k)_i
=
\operatorname{sign}((g_k)_i)
\frac{|(g_k)_i|^{p^\ast-1}}
{\|g_k\|_{p^\ast}^{p^\ast-1}}.
\]

The four methods can then be written as:

| Method | Local direction | Update | Interpretation |
| --- | --- | --- | --- |
| `raw_add` | \(g_k\), optionally normalized by projection | \(\delta_{k+1}=\Pi_{\mathcal B_p(\epsilon)}(\delta_k+\alpha g_k)\) | ordinary additive ascent; keeps path memory |
| `raw_replace` | normalized raw gradient | \(\delta_{k+1}=\epsilon\,g_k/\|g_k\|_p\) when this normalization is well-defined | aggressive boundary jump using current raw gradient |
| `steepest_add` | \(s_k\) | \(\delta_{k+1}=\Pi_{\mathcal B_p(\epsilon)}(\delta_k+\alpha s_k)\) | additive ascent using the norm-correct steepest direction |
| `steepest_replace` | \(s_k\) | \(\delta_{k+1}=\epsilon s_k\) | current linear model's full-budget maximizer |

Therefore:

- `replace` variants are usually faster at reaching the boundary because they
  spend the whole budget each step.
- `add` variants are usually more stable because they retain the previous
  perturbation direction and move gradually.
- `steepest` variants are more geometrically matched to the constraint because
  they maximize the local first-order gain under the attack norm.
- `raw` and `steepest` can coincide for \(p=2\), which explains why some
  curves overlap exactly.

The relevant speed metrics are not only final loss. They should include:

\[
T_{\mathrm{boundary}}(\tau)
=
\min\{k:\|\delta_k\|_p/\epsilon\ge \tau\},
\]

\[
T_{\rho}
=
\min\left\{
k:
\Phi(\delta_k)\ge \Phi(\delta_0)
+\rho\left(\Phi(\delta_T)-\Phi(\delta_0)\right)
\right\},
\]

and the average curve height

\[
\mathrm{AUC}
=
\frac{1}{T+1}\sum_{k=0}^{T}\Phi(\delta_k).
\]

Final attack strength should be reported separately:

\[
\Phi(\delta_T).
\]

This separates four questions:

1. how quickly the method reaches the feasible boundary;
2. how quickly the loss increases;
3. how high the average trajectory stays;
4. how large the final attack loss is.

### H0c. Predictions From The Math

If the problem is locally close to a frozen quadratic model and the leading
eigen-direction is stable, then:

\[
\texttt{replace}
\quad\text{should align faster than}\quad
\texttt{add}.
\]

In this regime, the expected behavior is:

| Regime | Expected ranking |
| --- | --- |
| frozen linear, stable direction, broad boundary high-loss region | `steepest_replace` and `raw_replace` should be fastest and can have the best final loss |
| frozen linear but small \(\alpha\) | `add` should be much slower because \((1+\alpha\lambda_i)\) ratios are close to one |
| strong norm-geometry mismatch | `steepest_*` should outperform `raw_*` |
| strongly nonlinear, rotating \(A_k\) or \(b_k\) | `replace` can be fast but unstable; `add` can have better final loss |
| narrow high-loss peak on the boundary | fast replacement can miss the peak; additive path-following can win |
| broad high-loss boundary plateau | replacement is more likely to work well because many boundary directions are good |

The key failure mode for a naive power-method explanation is that the real
finite-radius attack uses

\[
A_k=J(x+\delta_k)^\top J(x+\delta_k),
\qquad
b_k=J(x+\delta_k)^\top r(\delta_k),
\]

not one fixed matrix \(A\). Therefore the empirical test is whether the
directions actually remain stable:

\[
\cos(s_k,s_{k-1})\approx 1,
\qquad
\cos(\delta_k,\delta_T)\to 1 \text{ early}.
\]

If these cosines are small or oscillatory, then the frozen-\(A\) explanation is
not enough even if the method reaches the boundary quickly.

### H0d. Current Empirical Check Against The Formal Curves

The current formal mean Loss3 curves support the following pattern:

| Problem | Method | \(N\) | Final mean Loss3 | Step to 90% of final increase | Mean curve height |
| --- | --- | ---: | ---: | ---: | ---: |
| Burgers 1D | `raw_add` | 100 | 6.415 | 130 | 5.284 |
| Burgers 1D | `raw_replace` | 100 | 6.808 | 4 | 6.715 |
| Burgers 1D | `steepest_add` | 100 | 6.808 | 78 | 6.089 |
| Burgers 1D | `steepest_replace` | 100 | 6.808 | 4 | 6.715 |
| Darcy Flow | `raw_add` | 20 | 0.0452 | 78 | 0.0369 |
| Darcy Flow | `raw_replace` | 20 | 0.0500 | 7 | 0.0491 |
| Darcy Flow | `steepest_add` | 20 | 0.0452 | 78 | 0.0369 |
| Darcy Flow | `steepest_replace` | 20 | 0.0500 | 7 | 0.0491 |
| NS2D Recurrent | `raw_add` | 20 | 129.862 | 1 | 128.293 |
| NS2D Recurrent | `raw_replace` | 20 | 96.419 | 1 | 106.655 |
| NS2D Recurrent | `steepest_add` | 20 | 278.543 | 19 | 258.159 |
| NS2D Recurrent | `steepest_replace` | 20 | 96.419 | 1 | 106.655 |

Interpretation:

- Burgers and Darcy Flow match the fast-replacement hypothesis: replacement is
  both fast and final-strong.
- NS2D does not match the simple frozen-\(A\) story: replacement is fast, but
  its final mean Loss3 is much lower than `steepest_add`.
- Therefore, NS2D needs a nonlinear/path-dependence explanation rather than a
  pure power-method explanation.

The first completed NS2D mechanism trace, `sample 0 / raw_add`, already shows
strong direction rotation:

\[
T_{\mathrm{boundary}}(0.99)=1,
\]

\[
\operatorname{mean}_k\cos(d_k,d_{k-1})\approx 0.144,
\]

\[
\cos(\delta_{20},\delta_T)\approx 0.067.
\]

This means fast boundary arrival alone is not enough; the direction field is
rotating substantially. The full conclusion requires the matching
`raw_replace`, `steepest_add`, and `steepest_replace` traces for the same sample.

### H0e. Literature Anchors

Use these as framing references, not as direct theorems for the nonlinear PDE
attack:

- Goodfellow, Shlens, and Szegedy, *Explaining and Harnessing Adversarial
  Examples*: supports the idea that local linear behavior can explain fast
  adversarial directions. <https://arxiv.org/abs/1412.6572>
- Madry et al., *Towards Deep Learning Models Resistant to Adversarial Attacks*:
  frames PGD as a first-order adversary for constrained inner maximization.
  <https://arxiv.org/abs/1706.06083>
- Boyd and Vandenberghe, *Convex Optimization*, Section 9.4: steepest descent
  directions depend on the chosen norm and dual norm.
  <https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf>
- ETH power-method notes: power iteration speed is controlled by the ratio of
  the two largest eigenvalues in modulus.
  <https://people.inf.ethz.ch/arbenz/ewp/Lnotes/chapter7.pdf>
- MIT 18.06 power-method notes: power method can be slow when
  \(|\lambda_2/\lambda_1|\) is close to one.
  <https://web.mit.edu/18.06/www/Spring17/Power-Method.pdf>

### H1. Direction Stability

Replacement works well when early directions are already close to the final
high-loss direction.

Predictions:

- Burgers: `cos(delta_k, delta_final)` should become high by steps 5 to 10.
- NS2D: replacement directions should rotate more, especially at `eps=32`.
- Darcy Flow: early flip sets should have high Jaccard overlap with final flip
  sets only if replacement is stable.

### H2. Landscape Width

Replacement works well when a broad region on the perturbation boundary has
similar high loss.

Predictions:

- Burgers: ray profiles and boundary arcs should show broad high-loss regions.
- NS2D: high-loss regions should be narrower; arcs between final method
  directions should dip more.
- Darcy Flow: many different high-scoring flip sets should have similar Loss3
  only if the binary landscape is broad.

### H3. Finite-Radius Nonlinearity

If replacement fails only because epsilon is large, it should improve when
epsilon is reduced.

Predictions:

- NS2D replacement should become more competitive at smaller epsilons if the
  issue is finite-radius nonlinearity.
- If NS2D still favors `steepest_add` at small epsilon, then the difference is
  not only radius size; the direction field or solver-coupled objective is more
  intrinsically path-dependent.

### H4. Boundary Arrival Is Not Enough

Fast boundary arrival is only useful if the boundary direction is good and the
method can keep improving along the boundary.

Predictions:

- Replacement methods will hit boundary fastest.
- The decisive quantity is post-boundary gain and whether the tangent direction
  keeps improving true Loss3.

## Experiment 1: Trace Capture For Direction Stability

Purpose:

```text
Measure whether each method's early perturbation direction is already final-like,
and whether replacement rotates more or less than additive methods.
```

### Burgers Trace Run

Small mechanism run:

```bash
/venv/adv_robust/bin/python -u tools/run_loss3_direction_proposal_ablation.py \
  --out-root analysis_outputs/mechanism_20260622/burgers_eps8_alpha0p3_steps300_N10_trace \
  --methods raw_add raw_replace steepest_add steepest_replace \
  --batch-size 10 \
  --start-index 0 \
  --epsilon 8 \
  --alpha 0.3 \
  --steps 300 \
  --p 2 \
  --q 2 \
  --model-kind fno \
  --burgers-nu 0.001 \
  --save-delta-trajectory \
  --save-trajectory-final-conditions \
  --no-plots
```

Expected outputs:

```text
analysis_outputs/mechanism_20260622/burgers_eps8_alpha0p3_steps300_N10_trace/per_step_metrics.csv
analysis_outputs/mechanism_20260622/burgers_eps8_alpha0p3_steps300_N10_trace/per_sample_step_metrics.csv
analysis_outputs/mechanism_20260622/burgers_eps8_alpha0p3_steps300_N10_trace/final_deltas.npz
analysis_outputs/mechanism_20260622/burgers_eps8_alpha0p3_steps300_N10_trace/*/trajectory_samples.npz
```

### NS2D Trace Run

Mechanism run should use `batch=1`, because the current NS2D code records a
single sample position per batch for step-sample traces. With `batch=1`, every
sample is recordable.

```bash
/venv/adv_robust/bin/python -u 2D_NS_FNO2d_recurrent/perturbation_methods/attack_ns2d_recurrent_core4.py \
  --out-root analysis_outputs/mechanism_20260622/ns2d_eps32_alpha10_steps100_N3_trace \
  --checkpoint 2D_NS_FNO2d_recurrent/saved_models/2D/modes64_modes64_width60_epochs500_Tin10_T10_recurrent_pytorch_20260521_090136_UTC/checkpoints/final.pt \
  --test-path 2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/real_initial_laxmap_single/test/dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_test_ntimepoints21_all_frames.pt \
  --dictionary-path 2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dictionary/dim2d_nx256_N2000_solver=exponax_nu0.000_t20.0_dict_ntimepoints21_batch0_all_frames.pt \
  --start-index 0 \
  --num-samples 3 \
  --attack-batch-size 1 \
  --methods raw_add raw_replace steepest_add steepest_replace \
  --loss-types loss3 \
  --mode-spec all_w \
  --steps 100 \
  --epsilon 32 \
  --alpha 10 \
  --p 2 \
  --q 2 \
  --true-loss-every 1 \
  --record-step-sample-outputs \
  --record-step-sample-gradients \
  --record-step-sample-every 1 \
  --record-final-state-outputs \
  --solver-remat chunk \
  --solver-remat-chunk-steps 20 \
  --dictionary-chunk-size 16 \
  --clear-jax-caches-after-batch
```

Expected outputs:

```text
analysis_outputs/mechanism_20260622/ns2d_eps32_alpha10_steps100_N3_trace/**/per_step_metrics.csv
analysis_outputs/mechanism_20260622/ns2d_eps32_alpha10_steps100_N3_trace/**/per_sample_step_metrics.csv
analysis_outputs/mechanism_20260622/ns2d_eps32_alpha10_steps100_N3_trace/**/step_sample_trace.npz
analysis_outputs/mechanism_20260622/ns2d_eps32_alpha10_steps100_N3_trace/**/final_delta_and_metrics.npz
```

Metrics to compute:

| Metric | Meaning |
| --- | --- |
| `cos(delta_k, delta_final)` | Whether early perturbation is final-like |
| `angle(delta_k, delta_{k-1})` | Direction rotation of the accumulated perturbation |
| `cos(update_k, update_{k-1})` | Stability of update direction |
| `boundary_ratio_k` | How fast the method reaches the perturbation boundary |
| `post_boundary_gain` | Loss gain after first hitting boundary |
| `final_loss3` and AUC | Optimizer outcome |

Decision criteria:

- Burgers replacement supports the old explanation if early-to-final cosine is
  high by steps 5 to 10 and post-boundary loss stays high.
- NS2D supports the path-dependence explanation if replacement has lower
  early-to-final cosine, larger direction rotation, and worse final Loss3 than
  `steepest_add`.

## Experiment 2: Ray Scan And Landscape Width

Purpose:

```text
Separate direction quality from path quality, and test whether high-loss regions
are broad or narrow.
```

For each method final perturbation `delta_final`, evaluate:

```text
L(x + t * delta_final), t in {0, 0.05, 0.10, ..., 1.00}
```

For pairs of final perturbations, evaluate a boundary arc:

```text
delta_arc(s) = project_to_boundary((1 - s) * delta_A + s * delta_B)
s in {0, 0.025, 0.05, ..., 1.00}
```

Metrics:

| Metric | Meaning |
| --- | --- |
| `endpoint_loss` | Loss at full radius |
| `ray_auc` | Average loss along the ray |
| `high_loss_width_90` | Fraction of `t` where loss is at least 90% of max ray loss |
| `arc_min_over_weaker_endpoint` | Whether the arc between two final directions dips |
| `arc_depth` | How much loss is lost between two endpoint directions |

Decision criteria:

- Burgers broad ridge: large `high_loss_width_90`, small arc dip.
- NS2D narrow peak: small `high_loss_width_90`, large arc dip.
- If replacement final direction is poor in NS2D even along its own ray, the
  issue is direction quality.
- If replacement direction is good but its optimization path loses to additive
  methods, the issue is path-following or post-boundary movement.

Implementation note:

- Burgers has existing landscape scripts:

```text
tools/run_loss3_core4_pq_landscape_probe.py
tools/run_loss3_path_directional_curvature.py
tools/plot_loss3_mechanism_validation_summary.py
```

- NS2D needs a matched ray/arc evaluation utility using the saved
  `final_delta_and_metrics.npz` and `step_sample_trace.npz` from Experiment 1.

## Experiment 3: Small-Epsilon Nonlinearity Sweep

Purpose:

```text
Test whether NS2D replacement fails mainly because eps=32 is too large and the
finite-radius landscape is nonlinear.
```

Recommended NS2D sweep, small N first:

| eps | alpha |
| --- | --- |
| 32 | 10 |
| 16 | 5 |
| 8 | 2.5 |
| 4 | 1.25 |
| 2 | 0.625 |
| 1 | 0.3125 |

Run with:

```text
num_samples=3 or 5
attack_batch_size=1 for trace runs, or 4 for curve-only runs
steps=100
methods=raw_add raw_replace steepest_add steepest_replace
loss=loss3
mode=all_w
```

Recommended Burgers sweep:

| eps | alpha |
| --- | --- |
| 8 | 0.3 |
| 4 | 0.15 |
| 2 | 0.075 |
| 1 | 0.0375 |
| 0.5 | 0.01875 |

Decision criteria:

- If NS2D replacement catches up at small epsilon, the main explanation is
  finite-radius nonlinear landscape.
- If NS2D still favors `steepest_add` at small epsilon, the difference is deeper
  than radius size and likely tied to direction rotation, recurrent solver
  coupling, or a narrow high-loss region.
- If Burgers replacement remains competitive across epsilon values, this
  supports the broad-ridge/stable-direction explanation.

## Experiment 4: Darcy Flow Binary Mechanism

Purpose:

```text
Analyze Darcy Flow in its native binary-flip geometry, rather than forcing a
continuous vector-angle explanation.
```

Formal four-optimizer Darcy Flow curve run:

```bash
/venv/adv_robust/bin/python -u 2D_Darcy_FNO2d/perturbation_methods/attack_darcy_binary_loss_method_experiments.py \
  --dataset 2D_Darcy_FNO2d/datasets/grf_darcy_20260528_N1500/test/dim2d_darcy_nx211_N300_solver=jaxcg_solve421_alpha2_tau3_binary3-12_f1_seed45_test.pt \
  --checkpoint 2D_Darcy_FNO2d/saved_models/2D/darcy_N1500_nx85_m64_w60_e500_20260528/best.pt \
  --output-root analysis_outputs/optimizer_ablation_20260622/raw_runs/darcy \
  --run-name darcy_cflow_binary_loss3_methods_epsflips437_alphaflips5_steps100_N20_core4 \
  --experiment loss3_methods \
  --losses loss3 \
  --methods raw_add raw_replace steepest_add steepest_replace \
  --start 0 \
  --num-samples 20 \
  --steps 100 \
  --metric rel_l2 \
  --epsilon-fraction 0.01 \
  --epsilon-flips 437 \
  --alpha-flips 5 \
  --trace-true-loss3-every 1 \
  --trace-sample-index 0
```

Darcy mechanism metrics:

| Metric | Meaning |
| --- | --- |
| `flip_count_k` | How fast the method reaches the flip budget |
| `Jaccard(F_k, F_final)` | Early flip-set stability |
| `Jaccard(F_methodA_final, F_methodB_final)` | Whether methods find similar binary regions |
| `prefix_loss_curve` | Loss as top-ranked flips are added by fraction |
| `post_budget_gain` | Whether loss keeps changing after hitting flip budget |
| `flip_score_margin` | Whether the top flip set is sharply separated or broad/ambiguous |

Decision criteria:

- Darcy replacement is expected to work if its early top flip set is stable and
  many high-scoring flip sets produce similar Loss3.
- Darcy additive methods are expected to work better if ranking changes strongly
  during the path, meaning accumulated path information matters.

## Recommended Execution Order

Do **not** start everything at once. The order should be:

1. Confirm the current formal three-panel figure has Burgers `N=100`, NS2D
   `N=20`, Darcy Flow `N=20`.
2. Run the formal Darcy Flow four-optimizer curve if the immediate goal is the
   three-system comparison figure.
3. Run NS2D mechanism trace `N=3`, `batch=1`, with step-sample gradients.
4. Run Burgers mechanism trace `N=10`, `steps=300`, if old traces are not enough
   or if an exactly matched output root is desired.
5. Build the direction-stability table:
   `cos(delta_k, delta_final)`, update rotation, boundary hit, post-boundary
   gain.
6. Build matched ray/arc landscape diagnostics for Burgers and NS2D.
7. Run small-epsilon NS2D sweep only after the trace/landscape results tell us
   whether radius-size is the likely explanation.
8. Add Darcy binary flip-set mechanism after the formal Darcy curve is present.

## Minimal Claims That These Experiments Can Support

Claim A:

```text
Burgers replacement works because it reaches a useful boundary direction quickly
and the high-loss boundary region is broad.
```

Needed evidence:

- high early-to-final cosine for replacement;
- broad ray high-loss width;
- small boundary-arc dip;
- competitive final Loss3.

Claim B:

```text
NS2D replacement performs worse because the high-loss region is narrower or more
path-dependent, so additive steepest updates accumulate a better path.
```

Needed evidence:

- lower early-to-final cosine for replacement;
- stronger direction/update rotation;
- narrower ray high-loss width;
- larger boundary-arc dip;
- small-epsilon sweep showing whether replacement recovers.

Claim C:

```text
Darcy Flow must be explained through binary flip-set stability, not by directly
copying continuous Burgers/NS2D angle diagnostics.
```

Needed evidence:

- flip-set Jaccard curves;
- prefix loss curves;
- final flip-set overlap across methods;
- score margin or ambiguity analysis.

## Notes And Caveats

- The current NS2D formal N=20 curve run did not record per-step gradients or
  full per-sample step traces. It is enough for mean curves, but not enough for
  the full mechanism analysis.
- NS2D mechanism trace should use `batch=1`; otherwise only one sample position
  per batch gets detailed step trace.
- `raw_replace` and `steepest_replace` can coincide for p=q=2 in some settings.
  Keep both labels in all tables anyway, because the comparison requires four
  named methods.
- Darcy Flow should be labeled "Darcy Flow", not "CFlow", in figures and reports.
- The word "preferred" should not be written into formal figure titles. Figure
  titles should show sample count, epsilon/alpha, and steps.
