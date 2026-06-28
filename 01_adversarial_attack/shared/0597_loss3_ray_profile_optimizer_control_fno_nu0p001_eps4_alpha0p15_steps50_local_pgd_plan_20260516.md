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

- Sample count: `100`.
- Sample indices: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]`.
- Epsilon: `4.0`.
- Attack optimizer: `pgd`.
- Attack initialization: `local`.
- Attack steps: `50`.
- Attack learning rate / PGD alpha: `0.15`.
- Local residual steps: `24`.
- Official run must use GPU; CPU fallback is refused.
