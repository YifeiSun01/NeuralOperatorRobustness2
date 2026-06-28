# NS2D Recurrent eps8 alpha2p5 Visualization Summary - 2026-05-23

## Status

Generated the eps8/alpha2.5 CPU-only analysis and step-trace GIF visualizations from already-saved attack outputs. No solver run, model inference, attack update, PyTorch import, JAX import, or GPU computation was started for this post-processing.

## Source Evidence

Observed from:

- Pair root: `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/full_adw_b10_pair_outer_baseline_first_20260522/eps8_alpha2p5`
- Per-method `per_step_metrics.csv`, `final_state_outputs.npz`, and `step_sample_trace.npz` files under that pair root
- Overview script: `tools/plot_ns2d_pair_attack_overview.py`
- GIF script: `tools/plot_ns2d_step_trace_gif.py`
- Batch GIF wrapper: `tools/render_ns2d_step_trace_gifs_batch.py`

The completed grid contains 7 loss/mode blocks times 4 optimizer methods = 28 method traces.

## Output Files

- Overview figures/tables: `2D_NS_FNO2d_recurrent/visualizations/eps8_alpha2p5_overview_20260523/`
- Step-trace GIFs: `2D_NS_FNO2d_recurrent/visualizations/eps8_alpha2p5_step_trace_gifs_20260523/`
- Final metric summary CSV: `2D_NS_FNO2d_recurrent/visualizations/eps8_alpha2p5_overview_20260523/eps8_alpha2p5_final_metric_summary.csv`
- Winners JSON: `2D_NS_FNO2d_recurrent/visualizations/eps8_alpha2p5_overview_20260523/eps8_alpha2p5_winners.json`

Generated figures include loss curves with standard-deviation shading, delta/boundary curves, angle diagnostics, final-delta grids, final metric heatmaps, method-grouped curves, and radial FFT spectra.

GIF validation:

- `28` GIF files were generated.
- The GIF directory size is about `545 MB`.
- The representative `loss3/all_w/steepest_add` GIF read back as `101` frames.

Representative strongest GIF:

- `2D_NS_FNO2d_recurrent/visualizations/eps8_alpha2p5_step_trace_gifs_20260523/eps8_alpha2p5_loss3_wwwwwwwwww_steepest_add_sample0_idx0_step_trace.gif`

## Key Observations

Observed from `eps8_alpha2p5_winners.json`:

- Overall largest final true-loss increase: `loss3/all_w` with `steepest_add`, true loss `68.49 -> 139.5; increase `70.96`.
- Overall largest active-objective increase: `loss1/all_w` with `steepest_add`, objective increase `304.8`. This is not the same as the largest true-loss attack effect; its true-loss increase is only `6.763`.
- By final true-loss increase, `steepest_add` wins `6/7` blocks; `raw_replace` wins `1/7`; `raw_add` and `steepest_replace` win `0/7`.
- The eps8 result still supports the same optimizer-rank pattern seen at eps32: LP-steepest additive PGD is the most reliable final true-loss optimizer in this NS2D recurrent setting.
- The main difference from eps32 is scale: eps8 has smaller absolute true-loss growth, as expected from the smaller perturbation radius.

## Winner Table

| block | best true-loss method | true initial | true final | true increase | best active-objective method | objective increase |
|---|---:|---:|---:|---:|---:|---:|
| `loss1/all_w` | `steepest_add` | 68.49 | 75.25 | 6.763 | `steepest_add` | 304.8 |
| `loss2/all_a_target_w` | `raw_replace` | 68.49 | 72.44 | 3.946 | `raw_add` | 0.4605 |
| `loss3/all_w` | `steepest_add` | 68.49 | 139.5 | 70.96 | `steepest_add` | 70.96 |
| `loss3/all_d_target_w` | `steepest_add` | 68.49 | 84.74 | 16.24 | `steepest_add` | 16.24 |
| `loss3/w1_5_d6_9_target_w` | `steepest_add` | 68.49 | 84.83 | 16.34 | `steepest_add` | 16.34 |
| `loss3/d1_5_w6_9_target_w` | `steepest_add` | 68.49 | 132.8 | 64.31 | `steepest_add` | 64.31 |
| `loss3/a1_5_d6_9_target_w` | `steepest_add` | 68.49 | 86.74 | 18.25 | `raw_add` | 36.37 |

## Inference

Inference from the saved eps8 outputs:

- `loss3/all_w` is the strongest eps8 attack block under true solver-evaluated loss.
- `loss1` can have a very large active-objective increase, but that objective is model-output change, not necessarily the largest solver/model mismatch true loss.
- `loss2/all_a_target_w` is the one block where the best true-loss method is `raw_replace`, but the true-loss gain is small compared with the loss3 all-W and D/W mixed cases.

## Remaining Work

- A direct eps8-vs-eps32 comparison table can be generated from the two `*_final_metric_summary.csv` files if needed.
- These GIFs are local visualization artifacts; large generated GIFs should generally stay out of git unless explicitly requested.

## Download Package

Created an image-only download package at:

- `2D_NS_FNO2d_recurrent/visualizations/eps8_alpha2p5_image_gif_download_package_20260523`

Package contents:

- `overview_pngs/`: 11 PNG summary figures
- `step_trace_gifs/`: 28 animated GIF step traces

Validation:

- No non-image files were found in the package.
- Package size is about `554 MB`.

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

