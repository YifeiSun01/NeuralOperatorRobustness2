# Darcy CFlow Correlation Decomposition Interpretation - 2026-06-15

Observed from:

- Source table: `outputs/darcy_cflow_final_robustness_20260615/data/svd_jacobian_metrics.csv`
- Generated table:
  `outputs/darcy_cflow_final_robustness_20260615/data/final_metric_mean_std_20260615/svd_correlation_decomposition_20260615.csv`

This uses the final seven-model Darcy CFlow run on
`generalization_datasets_darcy_binary_loss3targeted_20260611`.

## Main Point

The pooled correlations are real in the current table, but they should not be
read as a single causal chain. Correlation is not transitive, and this table
mixes model-level differences with sample-level differences.

Let:

- `A = attack_loss_increase`
- `G = jt_error_l2_norm`
- `S = block2_sigma1`

The pooled all-row correlations are:

| scope | pair | Pearson r | Spearman rho |
|---|---|---:|---:|
| all 175 rows | A vs G | 0.382521 | 0.429304 |
| all 175 rows | A vs S | -0.319092 | -0.215947 |
| all 175 rows | G vs S | 0.357818 | 0.408939 |
| generalization 147 rows | A vs G | 0.0649105 | 0.0952595 |
| generalization 147 rows | A vs S | -0.816533 | -0.726549 |
| generalization 147 rows | G vs S | 0.058778 | 0.133322 |

This is not mathematically contradictory. A can be positively correlated with G,
G can be positively correlated with S, and A can still be negatively correlated
with S, especially when the first and third correlations are not both extremely
large.

## Decomposition

### All 175 Rows

| pair | pooled Pearson | between-model Pearson | within-model demeaned Pearson | within-sample demeaned Pearson | two-way demeaned Pearson |
|---|---:|---:|---:|---:|---:|
| A vs G | 0.382521 | 0.315000 | 0.501730 | 0.278064 | 0.223680 |
| A vs S | -0.319092 | -0.884774 | 0.385032 | -0.834184 | -0.537888 |

### Generalization 147 Rows

| pair | pooled Pearson | between-model Pearson | within-model demeaned Pearson | within-sample demeaned Pearson | two-way demeaned Pearson |
|---|---:|---:|---:|---:|---:|
| A vs G | 0.0649105 | 0.322801 | -0.369513 | 0.270124 | -0.0145743 |
| A vs S | -0.816533 | -0.909367 | -0.501759 | -0.873086 | 0.0595332 |

## Why Generalization A vs G Drops

In the generalization-only subset, the model-mean relation between `A` and `G`
is positive, but the within-model relation across the 21 generalization samples
is mostly negative. After removing both model and sample fixed effects, the
Pearson correlation is essentially zero (`-0.0145743`).

So the pooled generalization value near zero is a cancellation of effects, not a
proof that `J^T error` has no role.

Per-model generalization Pearson correlations:

| model | A vs G | A vs S |
|---|---:|---:|
| baseline | -0.737 | -0.285 |
| loss1 | -0.622 | -0.552 |
| loss2 | -0.570 | -0.686 |
| loss3 | 0.334 | -0.424 |
| Physics Loss | -0.420 | -0.724 |
| random clean | -0.480 | -0.577 |
| random solver | -0.598 | -0.801 |

## Why A vs S Is Negative

The strongest effect is between-model:

| model | A mean | G mean | S mean |
|---|---:|---:|---:|
| loss3 | 1.71618e-06 | 6.15680e-05 | 2.28890e-03 |
| loss2 | 3.39291e-06 | 9.22537e-05 | 1.71212e-03 |
| random clean | 3.45293e-06 | 9.11171e-05 | 1.68820e-03 |
| loss1 | 3.64937e-06 | 9.92210e-05 | 1.78090e-03 |
| Physics Loss | 3.71350e-06 | 1.03396e-04 | 1.81750e-03 |
| random solver | 4.43937e-06 | 1.23570e-04 | 1.85964e-03 |
| baseline | 8.94546e-06 | 9.29277e-05 | 1.24975e-03 |

Loss3 has the largest block/2 spectral norm but the smallest attack loss
increase. Baseline has the smallest block/2 spectral norm but the largest attack
loss increase. That model-level pattern alone makes `A vs S` strongly negative.

Inference:

- Spectral norm measures a worst-case local amplification direction, not whether
  the current residual or the constrained binary attack actually uses that
  direction.
- `A` depends on both the local sensitivity and the current residual/objective
  geometry. Loss3 can have a large spectral norm but still a small attack loss
  increase because its residual and `J^T error` are much smaller.
- The binary replacement attack delta is nearly orthogonal to both the top
  singular vector and `J^T error`, so a one-dimensional spectral norm is not a
  reliable predictor of final attack growth here.

