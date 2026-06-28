# NS2D Recurrent FNO Loss2 Full Final-Step Visualization - 2026-05-22

## Scope

This note records the corrected `loss2` final-step visualization. Unlike the earlier fast saved-array panels, this script reran the all-W final-step solver/model evaluation for selected saved final perturbations so the plots include actual model output, solver output, and model-minus-solver difference at the final frame.

## Source Evidence

Observed from:

- Attack output root: `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/full_adw_b10_eps32_alpha1_20260522/mode_aaaaaaaaaw_p2_q2_20260522_030849_UTC/batch_0000_0009/loss2`
- Script: `2D_NS_FNO2d_recurrent/visualizations/plot_loss2_attack_full_final_panels.py`
- Output directory: `2D_NS_FNO2d_recurrent/visualizations/loss2_attack_full_final_panels_20260522/`
- Report: `2D_NS_FNO2d_recurrent/visualizations/loss2_attack_full_final_panels_20260522/loss2_full_final_panel_report.json`
- Example panel: `2D_NS_FNO2d_recurrent/visualizations/loss2_attack_full_final_panels_20260522/loss2_steepest_add_samplepos0_idx0_full_final_panels.png`

The corrected panel layout is:

- Row 1: clean initial condition, perturbed initial condition, final delta, absolute final delta.
- Row 2: clean FNO final, clean solver final, clean FNO-minus-solver, FNO final change.
- Row 3: perturbed FNO final, perturbed solver final, perturbed FNO-minus-solver, solver final change.

## GPU Path

Observed from the generated report:

- PyTorch: `2.8.0+cu126`
- CUDA: `12.6`
- GPU: `NVIDIA A100-SXM4-80GB`, compute capability `sm_80`
- JAX backend: `gpu`
- JAX devices: `cuda:0`
- JAX preallocation disabled: `XLA_PYTHON_CLIENT_PREALLOCATE=false`
- Runtime: `58.28` seconds for 12 panels.
- Peak PyTorch allocated memory: `1.40 GiB`; peak reserved memory: `1.70 GiB`.

## Observed Metrics

Mean over the three plotted samples per method:

| method | delta L2 mean | delta Linf mean | clean FNO-solver L2 | adv FNO-solver L2 | FNO final change L2 | solver final change L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| raw_add | 13.2124 | 0.1421 | 48.8579 | 51.9766 | 204.6881 | 204.6365 |
| raw_replace | 32.0000 | 0.3383 | 48.8579 | 60.2301 | 302.8809 | 301.6959 |
| steepest_add | 32.0000 | 0.3239 | 48.8579 | 51.6706 | 298.5220 | 294.6436 |
| steepest_replace | 32.0000 | 0.3383 | 48.8579 | 60.2301 | 302.8809 | 301.6959 |

## Interpretation

Observed evidence:

- `loss2` perturbations are nonzero and visually affect both model and solver final states.
- `raw_add` is much weaker than the other methods for `alpha=1`, because it does not reach the full `epsilon=32` L2 budget in 100 steps.
- `raw_replace`, `steepest_add`, and `steepest_replace` reach the `epsilon=32` boundary.
- The final perturbations produce large final-state changes in both FNO and solver outputs, while the direct FNO-minus-solver error increase is more moderate.
- For `p=2`, `raw_replace` and `steepest_replace` are effectively the same direction in these saved outputs, so their panels and metrics match.

Inference:

- The earlier fast panels were insufficient for judging final model-vs-solver behavior because they did not rerun final FNO/solver output. The corrected full panels should be used for visual scale inspection.
- `epsilon=32` is visually meaningful and not too small.
- `alpha=1` is too conservative for `raw_add`; for additive methods, `steepest_add` is the better candidate from this single-scale run.
- Replacement methods are useful as boundary-direction tests but are aggressive because they jump directly to the L2 boundary.
- A paired epsilon/alpha sweep is still needed before fixing final attack settings.

## 2026-05-22 Revision

The full final-step panels were regenerated after the first version used one shared color range for all difference/change panels. The revised panels now use independent color ranges for each difference-style image and include per-sample L2 loss values directly in the titles.

Observed changes in the regenerated figures:

- The figure title includes `clean loss`, `adv loss`, `diff`, and `ratio`.
- `clean FNO - solver` uses its own color range and title `L2 loss=...`.
- `adv FNO - solver` uses its own color range and title `L2 loss=..., diff=...`.
- `FNO final change` and `solver final change` each include their own L2 norm in the title.
- The generated report now has `plot_version=independent_difference_ranges_with_loss_titles`.

Example observed sample from `loss2/steepest_add`, sample position `0`, dataset index `0`:

- clean loss: `34.82`
- adversarial loss: `47.04`
- loss increase: `+12.22`
- ratio: `1.351`

The attack inputs for this completed `loss2` run came from the test dataset, with manifest `test_path` pointing to `2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/real_initial_laxmap_single/test/dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_test_ntimepoints21_all_frames.pt` and `indices=0,1,2,3,4,5,6,7,8,9`.

## Remaining Work

- Run the corrected attack after the loss1 random-start patch and epsilon/alpha sweep support.
- Use the full final-step plotting script on the corrected sweep outputs to choose final `epsilon` and `alpha`.
