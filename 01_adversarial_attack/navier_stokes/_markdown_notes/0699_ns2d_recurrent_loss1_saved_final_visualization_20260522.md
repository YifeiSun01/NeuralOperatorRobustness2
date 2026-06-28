# NS2D Recurrent FNO Loss1 Saved Final-State Visualization - 2026-05-22

## Status

Generated CPU-only final-state panels for the completed baseline pair `epsilon=32, alpha=10`, `loss1/all_w` attack block.

## Observed Evidence

- Source attack loss root: `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/full_adw_b10_pair_outer_baseline_first_20260522/eps32_alpha10/mode_wwwwwwwwww_p2_q2_20260522_055315_UTC/batch_0000_0009/loss1`.
- Saved arrays used: each method's `final_state_outputs.npz`.
- Report file: `2D_NS_FNO2d_recurrent/visualizations/loss1_attack_full_final_panels_pair_outer_eps32_alpha10_20260522/loss1_saved_final_state_panel_report.json`.
- Output image directory: `2D_NS_FNO2d_recurrent/visualizations/loss1_attack_full_final_panels_pair_outer_eps32_alpha10_20260522`.
- Generated panel count: `40` PNGs, covering `4` methods times `10` sample positions.
- The plotting command used `CUDA_VISIBLE_DEVICES=''` and the script imports NumPy/matplotlib only; it does not import torch/JAX and does not rerun the model or solver.
- GPU attack was still running after plotting: observed `NVIDIA A100-SXM4-80GB, 47235, 81920, 99` from `nvidia-smi`.

## Panel Layout

Each PNG contains 3 rows by 4 columns:

- Row 1: clean initial condition, perturbed initial condition, final delta, absolute final delta.
- Row 2: clean FNO final output, clean solver final output, clean FNO-minus-solver difference with L2 loss, FNO final change.
- Row 3: adversarial FNO final output, adversarial solver final output, adversarial FNO-minus-solver difference with L2 loss and loss increase, solver final change.

Difference/change panels use independent color ranges so small differences do not collapse to gray.

## Summary Metrics

Observed from `loss1_saved_final_state_panel_report.json` across 10 samples:

| Method | Mean clean true loss | Mean adv true loss | Mean increase | Mean ratio | Mean delta L2 |
|---|---:|---:|---:|---:|---:|
| `raw_add` | 68.491997 | 93.041568 | 24.549571 | 1.422519 | 31.999998 |
| `raw_replace` | 68.491997 | 98.326353 | 29.834356 | 1.586603 | 32.000000 |
| `steepest_add` | 68.491997 | 158.283305 | 89.791308 | 2.568838 | 31.929921 |
| `steepest_replace` | 68.491997 | 98.326353 | 29.834356 | 1.586603 | 32.000000 |

## Representative Files

- `2D_NS_FNO2d_recurrent/visualizations/loss1_attack_full_final_panels_pair_outer_eps32_alpha10_20260522/loss1_raw_add_samplepos0_idx0_full_final_panels.png`
- `2D_NS_FNO2d_recurrent/visualizations/loss1_attack_full_final_panels_pair_outer_eps32_alpha10_20260522/loss1_raw_replace_samplepos0_idx0_full_final_panels.png`
- `2D_NS_FNO2d_recurrent/visualizations/loss1_attack_full_final_panels_pair_outer_eps32_alpha10_20260522/loss1_steepest_add_samplepos0_idx0_full_final_panels.png`
- `2D_NS_FNO2d_recurrent/visualizations/loss1_attack_full_final_panels_pair_outer_eps32_alpha10_20260522/loss1_steepest_replace_samplepos0_idx0_full_final_panels.png`

## Inference

For this baseline `loss1` block, `steepest_add` produced the largest mean true-loss increase among the four methods in the completed saved outputs. This is an inference from the saved final-state outputs and does not yet compare against later `loss2`/`loss3` blocks in the full sweep.
