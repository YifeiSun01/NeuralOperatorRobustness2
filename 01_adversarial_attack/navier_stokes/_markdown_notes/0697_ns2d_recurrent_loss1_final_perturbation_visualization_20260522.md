# NS2D Recurrent loss1 Final Perturbation Visualization - 2026-05-22

## Status

Created fast CPU-only visualizations from saved `loss1` attack arrays. No model or solver was rerun for the final-panel PNGs, so the active GPU attack was not disturbed.

## Source Outputs

Observed source directory:

```text
2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/full_adw_b10_eps32_alpha1_20260522/mode_wwwwwwwwww_p2_q2_20260522_015452_UTC/batch_0000_0009/loss1/
```

For each of the four `loss1` methods, the saved `final_delta_and_metrics.npz` contains:

```text
dataset_indices: [10]
final_delta: [10, 256, 256]
final_x_adv: [10, 256, 256]
```

## Generated Files

Visualization script:

```text
2D_NS_FNO2d_recurrent/visualizations/plot_loss1_attack_fast_panels.py
```

Output directory:

```text
2D_NS_FNO2d_recurrent/visualizations/loss1_attack_fast_panels_20260522/
```

Report:

```text
2D_NS_FNO2d_recurrent/visualizations/loss1_attack_fast_panels_20260522/loss1_fast_panel_report.json
```

Generated 12 PNGs:

```text
4 methods * 3 sample positions = 12 panels
```

The plotted sample positions are `0, 1, 2`, corresponding to test dataset indices `0, 1, 2`.

## Observed Results

Observed from saved `final_delta_and_metrics.npz` and `per_step_metrics.csv`:

| Method | final delta L2 mean | final delta Linf max | final loss1 mean | boundary ratio mean |
|---|---:|---:|---:|---:|
| raw_add | 0.0 | 0.0 | 0.0 | 0.0 |
| raw_replace | 0.0 | 0.0 | 0.0 | 0.0 |
| steepest_add | 0.0 | 0.0 | 0.0 | 0.0 |
| steepest_replace | 0.0 | 0.0 | 0.0 | 0.0 |

Inference:

- The completed `loss1` attack did not move away from the clean initial condition.
- Perturbed initial condition equals clean initial condition in the saved outputs.
- Because the final delta is exactly zero, the perturbed solver final condition is identical to the clean solver final condition without needing a rerun.
- This likely happens because `loss1 = ||F(x + delta) - F(x)||` starts at exactly zero when `delta=0`; the norm objective has zero/undefined-at-zero gradient behavior as implemented by PyTorch, so the current initialization can stall.
- For `loss1`, a nonzero random start, or an objective variant that avoids zero-gradient initialization, may be needed if the goal is to produce a nontrivial perturbation.

## Plot Contents

Each fast panel contains:

- Clean initial condition.
- Final perturbation `delta`.
- Perturbed initial condition `x + delta`.
- Initial difference `x_adv - x_clean`.
- Clean solver final condition from the test dataset.
- Perturbed solver final condition, marked equal to clean because `delta=0`.
- Solver final difference, zero because `delta=0`.
- A text panel noting that model-vs-solver final output was not computed in this CPU-safe fast plot.

## Model-vs-Solver Output

A full model-vs-solver final-output panel requires running FNO inference and, for nonzero perturbations, a perturbed solver rollout. A first CPU-only attempt was stopped because CPU FNO inference consumed many cores and could slow the active GPU attack. This should be done after the long attack run finishes, or on GPU with a tiny batch when it is safe.
