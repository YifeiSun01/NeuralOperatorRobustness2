# Burgers Round03 Loss1/Loss2/Loss3 Extended Training Plot Report

This report joins original and later run directories so all three loss curves extend instead of restarting at zero. The temporary `visualizations/...continuation...` plotting directory was removed after its outputs were copied over the user-facing old plot paths.

## Final And Best Generated Generalization

| loss | final epoch | final wall h | final gen RMSE | final gen rel L2 | best gen epoch | best gen RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| loss1 | 3000 | 4.22027 | 0.0343239 | 0.0615363 | 2673 | 0.032945 |
| loss2 | 1000 | 6.07963 | 0.0365082 | 0.0654749 | 948 | 0.0355695 |
| loss3 | 1000 | 13.0142 | 0.0220984 | 0.0396222 | 893 | 0.0193611 |

## Current User-Facing Outputs

- Dense comparison directory: `visualizations/burgers_loss3_selective_round03_long_training_comparison_dense_20260605`
- Compact comparison directory: `visualizations/burgers_loss3_selective_round03_long_training_comparison_20260605`
- loss1 single-run plot directory: `visualizations/burgers_loss3_selective_round03_loss1_3000ep_long_20260605_plots`
- loss2 single-run plot directory: `visualizations/burgers_loss3_selective_round03_loss2_1000ep_long_20260605_plots`
- loss3 single-run plot directory: `visualizations/burgers_loss3_selective_round03_loss3_1000ep_long_20260605_plots`

## Verification

- The dense comparison CSV `visualizations/burgers_loss3_selective_round03_long_training_comparison_dense_20260605/round03_dense_epoch_metrics_every1.csv` has max epochs loss1 `3000`, loss2 `1000`, loss3 `1000`.
- Its `run_role` values are `original` and `extended`.
- The single-run manifests report max eval/attack epochs loss1 `3000`, loss2 `1000`, loss3 `1000`.
