# NS2D Recurrent eps32 alpha10 Step Trace GIF - 2026-05-23

## Status

Generated a CPU-only animated heat-map GIF from an already-saved NS2D attack
step trace. No solver run, model inference, attack update, PyTorch import, JAX
import, or GPU computation was started for this visualization.

## Source Evidence

Observed from:

- Trace source:
  `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/full_adw_b10_pair_outer_baseline_first_20260522/eps32_alpha10/mode_wwwwwwwwww_p2_q2_20260522_074135_UTC/batch_0000_0009/loss3/steepest_add/step_sample_trace.npz`
- Metrics source:
  `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/full_adw_b10_pair_outer_baseline_first_20260522/eps32_alpha10/mode_wwwwwwwwww_p2_q2_20260522_074135_UTC/batch_0000_0009/loss3/steepest_add/step_sample_trace_metrics.csv`
- Plotting script:
  `tools/plot_ns2d_step_trace_gif.py`

The selected trace is:

- `epsilon=32`, `alpha=10`
- `loss3`
- `mode=wwwwwwwwww` / all W
- optimizer method `steepest_add`
- `sample_position=0`
- `dataset_index=0`
- steps `0..100`, `101` GIF frames

## Output Files

Generated at `2026-05-23 05:24 UTC`:

- GIF:
  `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_step_trace_gifs_20260523/eps32_alpha10_loss3_wwwwwwwwww_steepest_add_sample0_idx0_step_trace.gif`
- Manifest:
  `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_step_trace_gifs_20260523/eps32_alpha10_loss3_wwwwwwwwww_steepest_add_sample0_idx0_step_trace_manifest.json`

Validation:

- `imageio` readback reported `101` frames.
- File size was about `39 MB`.

## GIF Panels

Each GIF frame contains:

- perturbed initial condition `x + delta`
- gradient with respect to the initial condition
- FNO model final output
- solver final output
- model output minus solver output
- loss progression curve with the current step marker

The loss panel plots both `true_loss` and the active objective value from
`step_sample_trace_metrics.csv`.

## Color Range Policy

Observed implementation:

- Every heat-map panel uses its own global range across GIF frames.
- Each heat-map range is symmetric around zero:
  `vmin=-max(abs(values))`, `vmax=max(abs(values))`.
- This ensures zero is the center of the color scale for every heat map.
- Difference-like signed fields use `coolwarm`.
- State/output fields use `viridis`, still with symmetric zero-centered limits.

This avoids the misleading case where a field has, for example, a raw range like
`[-2.5, 3.0]` and zero is not centered in the color mapping.

## Remaining Work

Inference from existing saved traces:

- The same script can render the other eps32 alpha10 saved traces by changing
  `--method-dir`.
- There are saved traces for `loss1`, `loss2`, and multiple `loss3` ADW modes
  under the same eps32 alpha10 run, but this record only generated the requested
  representative GIF for `loss3/all_w/steepest_add`.

## PGD Step-Scale Interpretation - 2026-05-23

Observed from the eps8 alpha2p5 summaries and per-step CSVs:

- In `loss3/all_w`, `raw_add` reaches 100% of the epsilon boundary at step `k=1`.
- In the same block, `steepest_add` reaches 100% boundary at step `k=5`.
- Final true-loss increase for `loss3/all_w`:
  - `raw_add`: `+30.2497`
  - `raw_replace`: `+29.3196`
  - `steepest_add`: `+70.9640`
  - `steepest_replace`: `+29.3196`

Observed from `attack_ns2d_recurrent_core4.py`:

- `raw_add` uses the unnormalized raw gradient direction:
  `delta <- Proj(delta + alpha * grad)`.
- `steepest_add` uses the LP-steepest direction. For `p=2`, this is the L2-normalized gradient:
  `delta <- Proj(delta + alpha * grad / ||grad||_2)`.

Inference:

- The poor-looking `raw_add` result here should not be interpreted as "PGD is bad" in general.
- It is more precise to say that this run's `raw_add` is scale-sensitive unnormalized PGD. With `epsilon=8`, `alpha=2.5`, it effectively takes an over-large first step and immediately projects to the boundary.
- The smoother behavior that the user remembers from older PGD runs may correspond to either a smaller alpha, a normalized-gradient PGD variant, or a different data/loss normalization scale.
- Therefore, `steepest_add` winning here may partly reflect better step normalization, not only a deeper PDE/optimizer phenomenon.

Suggested follow-up experiment:

- Run a PGD alpha sweep for `raw_add`, keeping eps fixed, for example `epsilon=8` with `alpha=0.05, 0.1, 0.25, 0.5, 1.0, 2.5`.
- Add or expose a `unit_raw_add` method for NS2D. For `p=2`, `unit_raw_add` should match `steepest_add`, and that would directly test whether the difference is merely normalization.
- Compare boundary-hit step and final true loss. If smaller-alpha `raw_add` begins to look like historical PGD, then the discrepancy was mostly step-size/scale.

