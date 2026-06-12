# Fixed-Effect Correlation Interpretation

This supplement separates raw overall correlations from within-model, within-sample, and model+sample residual correlations. This matters because the raw overall scatter mixes model-level effects with sample-level effects.

## Mean By Model

| model | mean attack gain | mean ||J^T e|| | mean sigma_max | mean binary first-order |
|---|---:|---:|---:|---:|
| loss1 | 4.87872e-06 | 0.000125913 | 0.00178995 | 1.56671e-06 |
| loss2 | 3.84134e-06 | 0.000102028 | 0.0017211 | 1.25888e-06 |
| loss3 | 2.64413e-06 | 9.28814e-05 | 0.0020667 | 1.20221e-06 |
| physics | 4.2067e-06 | 0.000117896 | 0.00185268 | 1.51572e-06 |

## Key Spearman Correlations

| scope | clean loss | ||J^T e|| | ||J^T e||^2 | binary first-order | sigma_max | sigma*||e|| | |<e,u1>| |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw overall | -0.323 | -0.221 | -0.221 | -0.267 | -0.592 | -0.304 | 0.258 |
| within model | -0.716 | -0.545 | -0.582 | -0.599 | -0.377 | -0.556 | -0.006 |
| within sample | 0.477 | 0.620 | 0.529 | 0.523 | -0.530 | 0.514 | 0.680 |
| model+sample residual | -0.537 | -0.418 | -0.522 | -0.346 | 0.224 | -0.453 | 0.421 |

## Interpretation

- Raw overall correlation is dominated by model-level structure: `loss3` tends to have lower attack gain while having relatively high `sigma_max`, so raw `sigma_max` is strongly negative, not a direct mechanistic predictor.
- The most relevant view for comparing robustness on the same initial condition is `within sample`: here `||J^T e||`, binary first-order gain, and top-left error alignment are positively correlated with attack gain, while `sigma_max` is negatively correlated.
- Within-model and two-way residual views are mixed on this 10-sample subset; this is a useful case study but not yet a final statistical proof. The stronger next run should scale to more samples and include fixed-effect plots by default.

- Figure: `visualizations/darcy_metric_correlation_10samples_20260612_10samples_loss3attack20_corr_chunk64/fixed_effect_spearman_correlation_summary.png`
- Table: `analysis_outputs/darcy_metric_correlation_10samples_20260612_10samples_loss3attack20_corr_chunk64/fixed_effect_correlations.csv`
