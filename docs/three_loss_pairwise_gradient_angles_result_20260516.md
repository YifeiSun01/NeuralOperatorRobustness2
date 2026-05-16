# Three-Loss Pairwise Gradient Angle Result

Date: 2026-05-16

## Question

This experiment directly answers the native objective-direction question:

> At the same fixed perturbation \(\delta_k\), do `loss1_original`, `loss2_original`,
> and `loss3_original` point in the same gradient direction?

This is different from the earlier outward-growth experiment, which compared
\(\|A v\|\) and clean-residual outward growth inside the local `loss3` residual
geometry.

## Losses Compared

For each saved point \(z_k=x+\delta_k\), the experiment computes true nonlinear
autograd gradients of:

\[
L_1(\delta)=\|f(x+\delta)-f(x)\|_2,
\]

\[
L_2(\delta)=\|f(x+\delta)-j(x)\|_2,
\]

\[
L_3(\delta)=\|f(x+\delta)-j(x+\delta)\|_2.
\]

Then it records:

\[
\angle(\nabla_\delta L_1,\nabla_\delta L_2),\quad
\angle(\nabla_\delta L_1,\nabla_\delta L_3),\quad
\angle(\nabla_\delta L_2,\nabla_\delta L_3).
\]

## Run Settings

Observed from:

- Script: `tools/analyze_three_loss_pairwise_gradients.py`
- Command: `adv_robust/bin/python tools/analyze_three_loss_pairwise_gradients.py --device cuda`
- Input trajectories: `results/fno_nu0p001_loss_gradient_path_steps50_save5_gpu_nocudnn_20260515_200631/index_*/loss*_original_pgd/trajectory.npz`
- Output directory: `forensics/three_loss_pairwise_gradients_20260516/fno_nu0p001/`
- Samples: `0, 7, 40, 47, 115`
- Delta sources: `loss1_original_pgd`, `loss2_original_pgd`, `loss3_original_pgd`
- Saved steps: `0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50`
- Device: `cuda`

Output files:

- `forensics/three_loss_pairwise_gradients_20260516/fno_nu0p001/pairwise_three_loss_gradients.csv`
- `forensics/three_loss_pairwise_gradients_20260516/fno_nu0p001/summary_by_delta_source_k.csv`
- `forensics/three_loss_pairwise_gradients_20260516/fno_nu0p001/summary_by_k.csv`
- `forensics/three_loss_pairwise_gradients_20260516/fno_nu0p001/summary_by_delta_source.csv`
- `forensics/three_loss_pairwise_gradients_20260516/fno_nu0p001/manifest.json`

Raw row count: `165` rows = `5 samples x 3 delta sources x 11 saved steps`.

## How The Numbers Are Aggregated

The original attack trajectory saved **every 5 PGD steps**, not every internal PGD
step. The saved `k` values are:

```text
0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50
```

Each table entry is reported as:

```text
mean (standard deviation)
```

The value in parentheses is **standard deviation**, not variance. The script uses
population standard deviation, i.e. `np.std(..., ddof=0)`.

The row counts are:

- `all k>=5`: `150` angle rows = `5 samples x 3 delta sources x 10 saved nonzero k values`.
- `loss1 trajectory, k>=5`: `50` angle rows = `5 samples x 10 saved nonzero k values`.
- `loss2 trajectory, k>=5`: `50` angle rows = `5 samples x 10 saved nonzero k values`.
- `loss3 trajectory, k>=5`: `50` angle rows = `5 samples x 10 saved nonzero k values`.
- a single row such as `k=50, loss3 trajectory`: `5` angle rows = one per sample.

For each raw row, the script fixes the same saved point \(z_k=x+\delta_k\), computes
\(\nabla_\delta L_1\), \(\nabla_\delta L_2\), and \(\nabla_\delta L_3\) by autograd,
flattens the gradients, computes cosine similarity, and converts it to degrees:

\[
\theta_{ij}
=\cos^{-1}\left(
\frac{\langle \nabla L_i,\nabla L_j\rangle}
{\|\nabla L_i\|_2\|\nabla L_j\|_2}
\right)\cdot\frac{180}{\pi}.
\]

## Important k=0 Note

At `k=0`, the saved perturbation is near zero, and `loss1` is essentially zero.
The gradient of a norm at zero or near zero is numerically delicate. Therefore the
main interpretation should emphasize `k >= 5`.

## Main Summary

Entries are mean `(std)` in degrees.

| set | n | L1-L2 angle | L1-L3 angle | L2-L3 angle |
|---|---:|---:|---:|---:|
| all saved points | 165 | 10.77 (23.04) | 59.81 (20.61) | 60.87 (21.34) |
| exclude near-zero k=0 | 150 | 3.91 (7.06) | 56.51 (18.41) | 57.23 (18.41) |

## Summary By Delta Source, Excluding k=0

| delta source | n, k>=5 | L1-L2 angle | L1-L3 angle | L2-L3 angle | budget ratio |
|---|---:|---:|---:|---:|---:|
| `loss1_original_pgd` | 50 | 1.01 (0.41) | 55.79 (15.27) | 55.98 (15.20) | 0.85 (0.23) |
| `loss2_original_pgd` | 50 | 1.12 (0.41) | 52.44 (17.99) | 52.67 (17.97) | 0.85 (0.23) |
| `loss3_original_pgd` | 50 | 9.60 (10.04) | 61.30 (20.48) | 63.05 (20.16) | 0.41 (0.31) |

## Every-5-Step Summary

### Delta Source: `loss1_original_pgd`
| k | L1-L2 angle | L1-L3 angle | L2-L3 angle | budget ratio |
|---:|---:|---:|---:|---:|
| 0 | 79.40 (12.54) | 92.88 (8.94) | 97.19 (12.97) | 0.00 (0.00) |
| 5 | 1.58 (0.55) | 68.24 (15.09) | 68.45 (15.00) | 0.32 (0.03) |
| 10 | 1.17 (0.40) | 72.57 (22.21) | 72.56 (22.18) | 0.54 (0.06) |
| 15 | 1.08 (0.30) | 61.38 (15.09) | 61.58 (15.06) | 0.73 (0.08) |
| 20 | 0.94 (0.33) | 54.46 (14.05) | 54.64 (13.88) | 0.90 (0.08) |
| 25 | 0.92 (0.34) | 49.49 (7.82) | 49.66 (7.63) | 0.99 (0.01) |
| 30 | 0.89 (0.33) | 48.83 (6.92) | 49.03 (6.83) | 1.00 (0.00) |
| 35 | 0.88 (0.32) | 49.84 (8.56) | 50.08 (8.53) | 1.00 (0.00) |
| 40 | 0.88 (0.31) | 50.81 (10.09) | 51.06 (10.07) | 1.00 (0.00) |
| 45 | 0.87 (0.30) | 51.25 (10.88) | 51.49 (10.87) | 1.00 (0.00) |
| 50 | 0.87 (0.30) | 51.00 (11.09) | 51.24 (11.08) | 1.00 (0.00) |

### Delta Source: `loss2_original_pgd`
| k | L1-L2 angle | L1-L3 angle | L2-L3 angle | budget ratio |
|---:|---:|---:|---:|---:|
| 0 | 79.40 (12.54) | 92.88 (8.94) | 97.19 (12.97) | 0.00 (0.00) |
| 5 | 1.68 (0.25) | 83.79 (15.69) | 84.03 (15.44) | 0.32 (0.03) |
| 10 | 1.23 (0.41) | 62.46 (12.79) | 62.63 (12.60) | 0.53 (0.04) |
| 15 | 1.18 (0.39) | 58.23 (14.68) | 58.41 (14.79) | 0.72 (0.06) |
| 20 | 1.11 (0.36) | 59.68 (20.79) | 59.88 (20.92) | 0.89 (0.06) |
| 25 | 1.05 (0.34) | 46.46 (8.61) | 46.69 (8.61) | 1.00 (0.00) |
| 30 | 1.03 (0.34) | 44.43 (8.98) | 44.69 (8.98) | 1.00 (0.00) |
| 35 | 1.01 (0.35) | 43.23 (9.06) | 43.49 (9.09) | 1.00 (0.00) |
| 40 | 0.99 (0.35) | 42.48 (9.20) | 42.75 (9.23) | 1.00 (0.00) |
| 45 | 0.98 (0.35) | 41.98 (9.38) | 42.25 (9.40) | 1.00 (0.00) |
| 50 | 0.97 (0.36) | 41.63 (9.56) | 41.90 (9.55) | 1.00 (0.00) |

### Delta Source: `loss3_original_pgd`
| k | L1-L2 angle | L1-L3 angle | L2-L3 angle | budget ratio |
|---:|---:|---:|---:|---:|
| 0 | 79.40 (12.54) | 92.88 (8.94) | 97.19 (12.97) | 0.00 (0.00) |
| 5 | 33.34 (14.37) | 70.70 (7.13) | 76.83 (4.83) | 0.04 (0.01) |
| 10 | 13.62 (6.10) | 70.38 (10.62) | 72.03 (9.16) | 0.09 (0.04) |
| 15 | 9.36 (5.08) | 68.83 (16.41) | 69.63 (15.57) | 0.17 (0.08) |
| 20 | 7.38 (3.32) | 67.16 (21.81) | 68.52 (20.26) | 0.27 (0.14) |
| 25 | 6.81 (2.23) | 67.49 (22.41) | 68.91 (21.28) | 0.36 (0.18) |
| 30 | 7.16 (3.86) | 65.20 (20.81) | 67.18 (20.64) | 0.46 (0.22) |
| 35 | 4.72 (1.04) | 56.61 (18.83) | 57.83 (18.15) | 0.55 (0.26) |
| 40 | 4.60 (1.25) | 50.48 (19.40) | 51.71 (18.87) | 0.63 (0.27) |
| 45 | 4.56 (1.49) | 48.32 (20.09) | 49.31 (19.73) | 0.71 (0.26) |
| 50 | 4.41 (1.54) | 47.80 (20.31) | 48.55 (20.04) | 0.76 (0.22) |


## Observed Evidence

- For `k >= 5`, `loss1` and `loss2` gradients are almost aligned: the aggregate
  `L1-L2` angle is `3.91 (7.06)` degrees.
- For `k >= 5`, `loss3` gradients remain far from the fixed-target/model-movement
  gradients: aggregate `L1-L3` angle is `56.51 (18.41)` degrees, and aggregate
  `L2-L3` angle is `57.23 (18.41)` degrees.
- Along the `loss3_original_pgd` trajectory, even at `k=50`, the mean angles remain
  large: `L1-L3 = 47.80 (20.31)` degrees and `L2-L3 = 48.55 (20.04)` degrees.

## Inference

This directly supports the claim that native `loss3_original` has a substantially
different local update direction from `loss1_original` and `loss2_original` at the
same perturbation points.

The result also clarifies the relationship among the objectives:

- `loss1` and `loss2` are often locally very similar after the perturbation is nonzero,
  because both use the model response against a fixed clean reference.
- `loss3` differs because its target moves with the perturbed input through
  \(j(x+\delta)\), so its gradient includes the solver response.
- Therefore optimizing `loss1` or `loss2` is not equivalent to optimizing
  `loss3_original`; their next-step directions can differ by roughly `50-60` degrees
  over the saved nonzero trajectory points.
