# Angle Experiment Inventory

Date: 2026-05-16

## Count

Observed angle-related diagnostics in the current records:

- 4 diagnostic batches / experiments.
- 7 distinct angle definitions.
- Step-indexed diagnostics use saved trajectory points every 5 PGD steps:
  `k = 0, 5, 10, ..., 50`.
- Parentheses in numeric tables mean standard deviation, not variance.

## Inventory Table

| batch | angle definition | raw count | main source | headline observed result | what it means |
|---|---|---:|---|---|---|
| Clean-point candidate direction | \(\angle(A^Tb, v_i)\), where \(v_i\) is a top singular direction of \(J_f\), \(J_j\), or \(A=J_f-J_j\) | 135 CSV rows | `forensics/outward_growth_direction_20260515/fno_nu0p001/all_direction_similarity_table.csv` | `outward_growth` vs `error` top-8: `84.68 (19.34) deg`; vs `error` rank-1: `90.87 (8.49) deg` | The first-order outward direction \(A^Tb\) is almost orthogonal to the pure residual-movement singular direction. |
| Local-affine endpoint-vs-movement | \(\angle(A^Tb+A^TA\delta_k, A^TA\delta_k)\) | 165 rows | `forensics/outward_growth_direction_20260515/fno_nu0p001/same_delta_gradient_diagnostics/same_delta_gradient_diagnostics.csv` | on `loss3` path: `21.07 deg` at `k=5`, `1.40 deg` at `k=50` | In the clean-point affine model, the endpoint and movement gradients become aligned quickly as \(A^TA\delta\) dominates. |
| Local-affine bias-vs-movement | \(\angle(A^Tb, A^TA\delta_k)\) | 165 rows | same local-affine CSV | on `loss3` path: `48.61 deg` at `k=5`, `69.59 deg` at `k=50`; loss1/loss2 paths stay near `90-100 deg` | A decomposition check only: how the clean residual term aligns with the movement term. Not a main attack comparison. |
| True nonlinear endpoint-vs-movement | \(\angle(\nabla_z\|f(z)-j(z)\|, \nabla_z\|(f-j)(z)-(f-j)(x)\|)\) | 165 rows | `forensics/outward_growth_direction_20260515/fno_nu0p001/true_nonlinear_endpoint_vs_movement_gradients/true_nonlinear_endpoint_vs_movement_gradients.csv` | on `loss3` path: `45.06 deg` at `k=5`, `36.36 deg` at `k=10`, `22.33 deg` at `k=25`, `8.15 deg` at `k=50` | True finite-point evidence that \(\|b+A\delta\|\)-style endpoint direction and residual-movement direction differ early/mid path, then get closer. |
| Native pairwise loss gradients: L1-L2 | \(\angle(\nabla_\delta L_1, \nabla_\delta L_2)\) | 165 rows | `forensics/three_loss_pairwise_gradients_20260516/fno_nu0p001/pairwise_three_loss_gradients.csv` | all `k>=5`: `3.91 (7.06) deg`; on `loss3` path `k>=5`: `9.60 (10.04) deg`; `k=50` loss3 path: `4.41 (1.54) deg` | `loss1` and `loss2` usually give very similar local update directions once away from zero. |
| Native pairwise loss gradients: L1-L3 | \(\angle(\nabla_\delta L_1, \nabla_\delta L_3)\) | 165 rows | same pairwise CSV | all `k>=5`: `56.51 (18.41) deg`; on `loss3` path `k>=5`: `61.30 (20.48) deg`; `k=50` loss3 path: `47.80 (20.31) deg` | Direct evidence that native `loss3_original` points in a different direction from `loss1_original`. |
| Native pairwise loss gradients: L2-L3 | \(\angle(\nabla_\delta L_2, \nabla_\delta L_3)\) | 165 rows | same pairwise CSV | all `k>=5`: `57.23 (18.41) deg`; on `loss3` path `k>=5`: `63.05 (20.16) deg`; `k=50` loss3 path: `48.55 (20.04) deg` | Direct evidence that native `loss3_original` points in a different direction from `loss2_original`. |

## Short Interpretation

There are two separate claims:

1. Endpoint residual vs residual movement:
   \(\|b+A\delta\|\) and \(\|A\delta\|\) have different local geometry. Evidence:
   clean-point candidate angle near `90 deg`, and true nonlinear endpoint-vs-movement
   angle `45.06 deg` at `k=5` on the `loss3` path.

2. Native objective directions:
   `loss1_original` and `loss2_original` are close to each other, but `loss3_original`
   differs strongly from both. Evidence: for `k>=5`, `L1-L2 = 3.91 (7.06) deg`, while
   `L1-L3 = 56.51 (18.41) deg` and `L2-L3 = 57.23 (18.41) deg`.


## Chat-Presented Numeric Tables

The tables below preserve the numeric data presented in chat. All values are angles in
degrees unless otherwise stated. Parentheses mean standard deviation, not variance.

### Overall Table: 4 Batches, 7 Angle Definitions

| # | angle experiment | angle definition | raw count | key result |
|---:|---|---|---:|---|
| 1 | clean-point candidate | \(\angle(A^Tb, v_i)\) | 135 | `outward_growth` vs `error` top-8: `84.68 (19.34)`; vs rank-1: `90.87 (8.49)` |
| 2 | local-affine endpoint-vs-movement | \(\angle(A^Tb+A^TA\delta_k, A^TA\delta_k)\) | 165 | loss3 path: `21.07` at k=5; `1.40` at k=50 |
| 3 | local-affine bias-vs-movement | \(\angle(A^Tb, A^TA\delta_k)\) | 165 | loss3 path: `48.61` at k=5; `69.59` at k=50 |
| 4 | true nonlinear endpoint-vs-movement | \(\angle(\nabla\|f-j\|,\nabla\|\Delta(f-j)\|)\) | 165 | loss3 path: `45.06` at k=5; `8.15` at k=50 |
| 5 | native L1-L2 | \(\angle(\nabla L_1,\nabla L_2)\) | 165 | all k>=5: `3.91 (7.06)` |
| 6 | native L1-L3 | \(\angle(\nabla L_1,\nabla L_3)\) | 165 | all k>=5: `56.51 (18.41)` |
| 7 | native L2-L3 | \(\angle(\nabla L_2,\nabla L_3)\) | 165 | all k>=5: `57.23 (18.41)` |

### Clean-Point Candidate Direction Angles

| comparison | top-8 mean/std | rank-1 mean/std |
|---|---:|---:|
| \(A^Tb\) vs `error` singular directions | `84.68 (19.34)` | `90.87 (8.49)` |
| \(A^Tb\) vs `fno` singular directions | `89.20 (9.89)` | `88.14 (9.52)` |
| \(A^Tb\) vs `solver` singular directions | `92.72 (10.18)` | `88.09 (9.28)` |

### Endpoint-vs-Movement: loss3 Path

| k | local-affine angle | true nonlinear angle |
|---:|---:|---:|
| 0 | `96.96 (18.01)` | `85.35 (8.76)` |
| 5 | `21.07 (4.48)` | `45.06 (10.72)` |
| 10 | `11.28 (4.10)` | `36.36 (17.23)` |
| 15 | `7.19 (3.54)` | `28.05 (19.99)` |
| 20 | `5.28 (3.05)` | `23.63 (17.93)` |
| 25 | `4.22 (2.57)` | `22.33 (18.85)` |
| 30 | `3.51 (2.21)` | `18.87 (16.09)` |
| 35 | `2.87 (1.92)` | `12.98 (11.97)` |
| 40 | `2.17 (1.54)` | `9.24 (12.02)` |
| 45 | `1.64 (1.31)` | `8.45 (11.76)` |
| 50 | `1.40 (1.21)` | `8.15 (11.32)` |

### Native L1/L2/L3 Pairwise Gradient Angles: Summary

| data range | L1-L2 | L1-L3 | L2-L3 |
|---|---:|---:|---:|
| all saved points | `10.77 (23.04)` | `59.81 (20.61)` | `60.87 (21.34)` |
| all k>=5 | `3.91 (7.06)` | `56.51 (18.41)` | `57.23 (18.41)` |
| loss1 trajectory, k>=5 | `1.01 (0.41)` | `55.79 (15.27)` | `55.98 (15.20)` |
| loss2 trajectory, k>=5 | `1.12 (0.41)` | `52.44 (17.99)` | `52.67 (17.97)` |
| loss3 trajectory, k>=5 | `9.60 (10.04)` | `61.30 (20.48)` | `63.05 (20.16)` |

### Native Pairwise: loss3 Trajectory Every 5 Steps

| k | L1-L2 | L1-L3 | L2-L3 | budget |
|---:|---:|---:|---:|---:|
| 0 | `79.40 (12.54)` | `92.88 (8.94)` | `97.19 (12.97)` | `0.00 (0.00)` |
| 5 | `33.34 (14.37)` | `70.70 (7.13)` | `76.83 (4.83)` | `0.04 (0.01)` |
| 10 | `13.62 (6.10)` | `70.38 (10.62)` | `72.03 (9.16)` | `0.09 (0.04)` |
| 15 | `9.36 (5.08)` | `68.83 (16.41)` | `69.63 (15.57)` | `0.17 (0.08)` |
| 20 | `7.38 (3.32)` | `67.16 (21.81)` | `68.52 (20.26)` | `0.27 (0.14)` |
| 25 | `6.81 (2.23)` | `67.49 (22.41)` | `68.91 (21.28)` | `0.36 (0.18)` |
| 30 | `7.16 (3.86)` | `65.20 (20.81)` | `67.18 (20.64)` | `0.46 (0.22)` |
| 35 | `4.72 (1.04)` | `56.61 (18.83)` | `57.83 (18.15)` | `0.55 (0.26)` |
| 40 | `4.60 (1.25)` | `50.48 (19.40)` | `51.71 (18.87)` | `0.63 (0.27)` |
| 45 | `4.56 (1.49)` | `48.32 (20.09)` | `49.31 (19.73)` | `0.71 (0.26)` |
| 50 | `4.41 (1.54)` | `47.80 (20.31)` | `48.55 (20.04)` | `0.76 (0.22)` |

