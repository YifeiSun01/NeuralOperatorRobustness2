# NS2D JVP/VJP Integration Summary - 2026-06-23

This file connects the new NS2D matrix-free Jacobian probes to the existing Loss3 mechanism story.

## Status

- Completed probe groups: N3, N2_topup
- Key metrics CSV: `analysis_outputs/mechanism_20260622/full_mechanism_validation/cross_system_summary/ns2d_jvp_vjp_key_metrics.csv`

## Interpretation Rules

- Replacement has a power-iteration-like explanation only if the saved replacement direction aligns with the local top singular direction and the local residual-linear surrogate remains accurate at large radius.
- Add is favored when the top direction is unstable, the full-budget surrogate error is large, or replace candidates have lower true gain than add candidates despite comparable local predicted gain.
- If `jvp_mode_used` is finite-difference-heavy, report the result as a matrix-free finite-difference JVP plus exact VJP probe, not as a pure forward-mode autograd JVP.

## Files To Read After Completion

- N3: `analysis_outputs/mechanism_20260622/full_mechanism_validation/ns2d_jvp_vjp_spectrum_N3`
- N2_topup: `analysis_outputs/mechanism_20260622/full_mechanism_validation/ns2d_jvp_vjp_spectrum_N2_topup`

The final cross-system explanation should combine these metrics with the existing Burgers landscape/ridge evidence, Darcy flip-set evidence, and NS2D exact first-order/linearity/boundary-arc evidence.
