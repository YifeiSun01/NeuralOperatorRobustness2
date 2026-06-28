# Normal-Protocol Loss3 Experiment 4 Ray Profile Plan - 2026-05-16

Scope: FNO / 1D Burgers `nu=0.001`, GPU-only, batch size 100.

This run intentionally does not use best-over-steps or multi-restart attack selection.

## Protocol

- One initialization per attack objective.
- Adam ascent for a fixed number of steps.
- Use the final step direction only.
- Fix that direction and evaluate `x + r v` for radii from `0` to `epsilon`.
- Plot `r -> loss3`, `r -> loss1/loss2/loss3`, and local ratio curves.

## Settings

- Sample count: `4`.
- Sample indices: `[0, 1, 2, 3]`.
- Epsilon: `8.0`.
- Attack steps: `2`.
- Attack learning rate: `0.3`.
- Local residual steps: `2`.
- Official run must use GPU; CPU fallback is refused.
