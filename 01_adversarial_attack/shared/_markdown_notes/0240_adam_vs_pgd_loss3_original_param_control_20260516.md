# Adam vs Ordinary PGD Loss3 Original Parameter Control - 2026-05-16

Status: completed on GPU. No CPU fallback.

## Purpose

Test whether Adam systematically gives smaller direct `loss3_original` attack values than ordinary PGD when the objective, batch, and step count are held fixed.

This is a lightweight optimizer-dynamics check, not a new ray-profile experiment. It optimizes only direct `loss3_original` on the same 100 FNO / Burgers `nu=0.001` samples.

## Settings

- Samples: dataset indices `0..99`.
- Objective: maximize `loss3_original = ||f(x+delta)-g(x+delta)||`.
- Optimizers: Adam vs ordinary PGD.
- Steps: `50`.
- Initialization: zero delta.
- Norm: L2.
- Parameter settings:
  - `epsilon=4.0`, `alpha=0.15`
  - `epsilon=12.0`, `alpha=0.45`

Output directory:

`forensics/adam_vs_pgd_loss3_original_params_20260516/fno_nu0p001_batch100_steps50_eps4_eps12/`

## Summary

### Endpoint / Boundary `loss3_original`

| setting | optimizer | mean | std | variance | final norm mean |
|---|---|---:|---:|---:|---:|
| eps=4, alpha=0.15 | Adam | 1.6712 | 0.5538 | 0.3066 | 4.0000 |
| eps=4, alpha=0.15 | PGD boundary | 2.3290 | 1.3685 | 1.8727 | 2.6570 final, rescaled to 4.0000 |
| eps=12, alpha=0.45 | Adam | 8.1319 | 1.5485 | 2.3980 | 12.0000 |
| eps=12, alpha=0.45 | PGD boundary | 7.2865 | 4.3876 | 19.2513 | 8.6585 final, rescaled to 12.0000 |

### Paired Difference: PGD - Adam

| setting | metric | mean diff | std diff | p-value | PGD > Adam | Adam > PGD |
|---|---|---:|---:|---:|---:|---:|
| eps=4, alpha=0.15 | final loss3 | +0.3143 | 1.2265 | 1.189e-02 | 36 | 64 |
| eps=4, alpha=0.15 | boundary loss3 | +0.6578 | 0.9864 | 1.487e-09 | 75 | 25 |
| eps=12, alpha=0.45 | final loss3 | -1.5838 | 4.2216 | 2.961e-04 | 44 | 56 |
| eps=12, alpha=0.45 | boundary loss3 | -0.8454 | 3.7434 | 2.612e-02 | 48 | 52 |

## Interpretation

The two new settings do not support a universal statement that Adam always gives a smaller `loss3_original` value.

- At `epsilon=4`, ordinary PGD gives a significantly larger boundary endpoint `loss3_original` than Adam.
- At `epsilon=12`, Adam gives a larger endpoint `loss3_original` than PGD after 50 steps.

The likely reason for the `epsilon=12` reversal is optimizer dynamics and boundary reach. Adam reaches the L2 boundary almost immediately, while ordinary PGD after 50 steps has mean final norm only about `8.66` before boundary rescaling. Thus, at a fixed 50-step budget, Adam can sometimes be more aggressive rather than more conservative.

The safer conclusion is:

> Adam changes the constrained attack dynamics substantially. It can make direct `loss3_original` weaker in some settings, but it is not a monotone or universal rule. The relative behavior depends on epsilon, alpha, step count, and whether the method reaches the epsilon boundary.

For the historical `epsilon=8`, `alpha=0.3` normal-protocol comparison, ordinary PGD is the correct protocol to match the earlier results, and it makes direct `loss3_original` the strongest finite-radius objective. But across arbitrary alpha/epsilon choices, Adam-vs-PGD has no simple one-line rule.
