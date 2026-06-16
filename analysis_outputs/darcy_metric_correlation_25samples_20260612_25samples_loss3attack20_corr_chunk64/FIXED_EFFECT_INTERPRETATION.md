# Fixed-Effect Correlation Interpretation - 25 Samples

This supplement separates raw overall correlations from model-residual, sample-residual, and model+sample residual correlations. Raw overall mixes model-level and sample-level effects.

## Model Means

```text
loss1    attack_gain=4.788547e-06  jt_error=1.391878e-04  sigma=1.799438e-03  binary_first_order=1.704681e-06
loss2    attack_gain=4.030971e-06  jt_error=1.166779e-04  sigma=1.722550e-03  binary_first_order=1.423322e-06
loss3    attack_gain=2.756509e-06  jt_error=1.057695e-04  sigma=2.072769e-03  binary_first_order=1.321766e-06
physics  attack_gain=4.139830e-06  jt_error=1.321585e-04  sigma=1.860728e-03  binary_first_order=1.683077e-06
```

## Spearman Correlations With Attack Gain

```text
raw overall:
  clean loss:           -0.266
  ||J^T e||:            -0.248
  ||J^T e||^2:          -0.248
  binary first-order:   -0.256
  sigma_max:            -0.604
  sigma_max * ||e||:    -0.328
  |<e,u1>|:              0.205

within model residual:
  clean loss:           -0.722
  ||J^T e||:            -0.608
  ||J^T e||^2:          -0.634
  binary first-order:   -0.612
  sigma_max:            -0.460
  sigma_max * ||e||:    -0.646
  |<e,u1>|:             -0.055

within sample residual:
  clean loss:            0.604
  ||J^T e||:             0.638
  ||J^T e||^2:           0.552
  binary first-order:    0.522
  sigma_max:            -0.599
  sigma_max * ||e||:     0.544
  |<e,u1>|:              0.629

model+sample residual:
  clean loss:           -0.572
  ||J^T e||:            -0.350
  ||J^T e||^2:          -0.445
  binary first-order:   -0.231
  sigma_max:             0.195
  sigma_max * ||e||:    -0.417
  |<e,u1>|:              0.292
```

## Interpretation

Within-sample residual correlation is the most relevant view for comparing different models on the same coefficient field. In this 25-sample run, the error-aligned metrics remain more directly tied to attack gain than raw clean-loss-only explanations, while raw overall sigma remains dominated by model-level structure.

Important nuance: the exact ranking is less clean than the 10-sample pilot, but the central mechanism remains the same: sigma_max alone is not a sufficient explanation because it measures worst-case sensitivity, whereas J^T error and binary first-order scores include the current error direction and binary feasible flip geometry.

- Table: `analysis_outputs/darcy_metric_correlation_25samples_20260612_25samples_loss3attack20_corr_chunk64/fixed_effect_correlations.csv`
- Model means: `analysis_outputs/darcy_metric_correlation_25samples_20260612_25samples_loss3attack20_corr_chunk64/model_metric_means.csv`
- Figure: `visualizations/darcy_metric_correlation_25samples_20260612_25samples_loss3attack20_corr_chunk64/fixed_effect_spearman_correlation_summary.png`
