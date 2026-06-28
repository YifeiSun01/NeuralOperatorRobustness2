# Loss3 JVP/VJP Integration Experiment Plan - 2026-06-23

This plan defines how the new NS2D JVP/VJP and top singular path probes should
fit into the existing Loss3 optimizer comparison.

## Core Question

We want to explain why the four optimizers behave differently on Burgers,
Darcy Flow, and NS2D, especially:

\[
\texttt{steepest\_replace}
\]

can be strong on Burgers-like cases, but is weaker than

\[
\texttt{steepest\_add}
\]

on NS2D.

The working hypothesis is:

\[
\texttt{replace}
\text{ works when full-budget jumps remain predictive and the high-loss region
is broad/stable, but fails when the landscape is path-dependent, nonlinear,
rotating, or narrow.}
\]

The JVP/VJP experiments are not a separate story. They are the NS2D-specific
test of whether a "generalized power iteration / top singular direction" view
really explains replace.

## Layer 1: Main Performance Curves

This is the formal comparison figure.

| System | Methods | Quantity | Required aggregation |
| --- | --- | --- | --- |
| Burgers 1D | `raw_add`, `raw_replace`, `steepest_add`, `steepest_replace` | Loss3 vs step | mean over all available samples, with N shown in title/legend |
| Darcy Flow | same four methods | Loss3 vs step | mean over all available samples, with N shown |
| NS2D | same four methods | Loss3 vs step | mean over all available samples, with N shown |

Purpose:

- Establish the empirical fact: which optimizer is faster early, which reaches
  larger final Loss3, and whether replace/add differences are stable over N.
- This figure should not be explained by one selected sample. It is the averaged
  outcome that all mechanism experiments must explain.

## Layer 2: Existing Mechanism Evidence

These are the already-running or completed supporting experiments.

| Mechanism block | Burgers | Darcy Flow | NS2D |
| --- | --- | --- | --- |
| First-order/full-budget predictability | `burgers_first_order_prediction_N20` | flip-set speed/overlap analogue | `ns2d_exact_mechanism`, `ns2d_exact_mechanism_N2_topup` |
| Local linearity/nonlinearity | Burgers landscape/ridge/curvature probes | discrete flip analogue | exact NS2D linearity rows |
| Boundary/high-loss width | Burgers ridge and boundary arc | near-optimal flip-set size | NS2D boundary arc |
| Direction/path stability | older Burgers direction/path evidence | final flip-set overlap and trace stability | NS2D direction stability |

Interpretation:

- Burgers replace can be explained if it reaches the boundary quickly and the
  boundary contains a broad high-loss region where many directions are similarly
  good.
- Darcy Flow replace-like behavior can be explained through discrete flip-set
  speed and overlap: fast replacement works when good flip sets are easy to
  identify and many near-optimal sets exist.
- NS2D add advantage is supported if replace reaches the boundary early but then
  loses because full-budget directions are poorly predicted, local linearity is
  weak, or boundary high-loss arcs are narrow.

## Layer 3: New NS2D JVP/VJP / Top Singular Path Probe

This is the new implementation:

- Script: `tools/probe_ns2d_jvp_vjp_spectrum_20260623.py`
- Queue wrapper: `tools/run_ns2d_jvp_vjp_mechanism_queue_20260623.sh`
- Post-run summary: `tools/build_ns2d_jvp_vjp_integration_summary_20260623.py`

Queued runs:

| Run | Input trace | Output |
| --- | --- | --- |
| NS2D N=3 | `analysis_outputs/mechanism_20260622/ns2d/ns2d_eps32_alpha10_steps100_N3_trace` | `analysis_outputs/mechanism_20260622/full_mechanism_validation/ns2d_jvp_vjp_spectrum_N3` |
| NS2D N=2 top-up | `analysis_outputs/mechanism_20260622/full_mechanism_validation/ns2d/ns2d_eps32_alpha10_steps100_N2_topup_trace` | `analysis_outputs/mechanism_20260622/full_mechanism_validation/ns2d_jvp_vjp_spectrum_N2_topup` |

Default settings:

| Setting | Value |
| --- | --- |
| Methods | `steepest_add`, `steepest_replace` |
| Steps \(k\) | `0, 1, 10, 50, 100` |
| Power iterations | `4` |
| Directions tested | `top_singular`, `grad`, `saved_direction`, `final_delta`, `random` |
| Radius fractions | `0.03, 0.1, 0.3, 1.0` |
| JVP mode | `auto`: autograd JVP if possible, centered finite-difference JVP fallback |
| VJP mode | exact reverse-mode |

Why matrix-free:

NS2D residual and perturbation are both \(256 \times 256\).  Materializing the
full Jacobian is unnecessary and too expensive.  The correct test is:

\[
Jv,\qquad J^\top u,\qquad
v_{t+1} = \frac{J^\top J v_t}{\|J^\top J v_t\|_2}.
\]

## Outputs And What They Mean

| Output | Meaning |
| --- | --- |
| `spectrum_path.csv` | top singular value estimate, alignment of top singular direction with gradient, saved optimizer direction, final delta, and current radial direction |
| `power_iteration_rows.csv` | whether the top singular iteration converges stably |
| `jvp_direction_rows.csv` | how strongly each direction changes the residual, and whether `Jv` aligns with the current residual |
| `quadratic_surrogate_rows.csv` | whether \(r(x+\delta+d) \approx r(x+\delta) + Jd\) predicts exact Loss3 gain |
| `candidate_comparison_rows.csv` | exact and predicted gain for add/replace candidates built from top singular, gradient, saved direction, final delta, and random direction |
| `ns2d_jvp_vjp_key_metrics.csv` | compact metrics extracted after both JVP/VJP runs complete |
| `loss3_ns2d_jvp_vjp_integration_summary_20260623.md` | human-readable integration summary |

## How This Tests The Replace Explanation

For a true power-iteration-like explanation of replace on NS2D, we would expect:

1. The saved replace direction aligns with the local top singular direction:

\[
\cos(v_{\text{top}}, d_{\text{replace}}) \text{ is high.}
\]

2. The local residual-linear surrogate remains accurate at replacement-scale
steps:

\[
\|r(\delta+d)\| \approx \|r(\delta) + Jd\|.
\]

3. Replace candidates built from `saved_direction` or `top_singular` have true
gain comparable to or larger than add candidates.

4. These properties stay stable along the path, especially after the boundary is
reached.

If these fail on NS2D, then replace is not failing because it is "too slow"; it
is failing because the full-budget/top-singular picture is not stable enough at
the actual radius used by the attack.

## Expected Evidence Pattern

The current explanation is supported if the results look like this:

| System | Expected mechanism pattern |
| --- | --- |
| Burgers | broad high-loss boundary, forgiving ridge, relatively stable useful directions; replace can jump to a good boundary region quickly |
| Darcy Flow | replacement-like flip updates quickly identify strong flip sets; near-optimal flip sets are broad enough |
| NS2D | exact first-order prediction and full-radius surrogate are worse; top directions or saved directions are less stable; replace candidates underperform add after boundary |

In that case the final explanation should be:

\[
\texttt{steepest\_add}
\text{ is more robust as a general optimizer because it follows the changing
local geometry, while }
\texttt{steepest\_replace}
\text{ is strong when the geometry is stable or forgiving enough for full-budget
jumps.}
\]

## What Would Change The Conclusion

If NS2D JVP/VJP shows high alignment, low surrogate error at radius \(1.0\), and
replace candidates still underperform, then the problem is not mainly local
Jacobian instability.  We would then shift the explanation toward boundary
topology or objective/constraint mismatch.

If NS2D JVP/VJP is already inaccurate at very small radii such as \(0.03\), then
the diagnostic itself needs adjustment: smaller finite-difference step, smaller
radius fractions, or a more direct check of solver/model differentiability.

If `jvp_mode_used` is mostly finite difference, report it honestly as a
matrix-free finite-difference JVP plus exact VJP probe, not as a pure
forward-mode autograd result.

## Run Order

1. Finish the current extended mechanism queue:

   `tools/run_loss3_extended_mechanism_queue_20260623.sh`

2. Automatically run the new JVP/VJP queue:

   `tools/run_ns2d_jvp_vjp_mechanism_queue_20260623.sh`

3. Automatically build the JVP/VJP summary:

   `tools/build_ns2d_jvp_vjp_integration_summary_20260623.py`

4. Re-read:

   `docs/loss3_full_mechanism_validation_summary_20260623.md`

   and combine it with:

   `docs/loss3_ns2d_jvp_vjp_integration_summary_20260623.md`

5. Final write-up should explicitly connect:

   - averaged optimizer curves,
   - Burgers landscape/ridge evidence,
   - Darcy flip-set evidence,
   - NS exact first-order/linearity/boundary evidence,
   - NS JVP/VJP top singular evidence.

6. Run the final automation queue:

   `tools/run_loss3_final_auto_pipeline_20260623.sh`

   This waits for the JVP/VJP queue, then automatically rebuilds the main
   optimizer analysis, regenerates the three-system mean curve figure,
   regenerates mechanism summaries, plots the NS2D JVP/VJP mechanism figures,
   creates a final report bundle, creates a local archive, and uploads/backs up
   the bundle if R2 configuration is available in the environment.

## Acceptance Criteria

The new JVP/VJP block is considered successfully integrated only when:

- Both `ns2d_jvp_vjp_spectrum_N3/manifest.json` and
  `ns2d_jvp_vjp_spectrum_N2_topup/manifest.json` exist with status `completed`,
  unless the top-up root is missing and the queue logs that explicitly.
- `spectrum_path.csv`, `quadratic_surrogate_rows.csv`, and
  `candidate_comparison_rows.csv` have nonzero rows.
- `ns2d_jvp_vjp_key_metrics.csv` is generated.
- The final mechanism summary states whether the JVP/VJP results support,
  weaken, or change the current replace/add explanation.
- `analysis_outputs/loss3_final_auto_pipeline_20260623/FINAL_REPORT.md`
  exists and references the main curve, mechanism summary, JVP/VJP summary,
  figures, local archive, and upload status.
