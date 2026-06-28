# Generalization Evaluation Results

The retained Burgers, Darcy flow, and recurrent Navier-Stokes FNO checkpoints were evaluated on original train/test data plus 50 generated generalization datasets per task.

These PDE tasks are regressions, so the plots use `accuracy_score = 100 / (1 + relative_l2)`; higher is better. Raw `relative_l2`, `RMSE`, and `MAE` are in `metrics.csv`. For recurrent NS, non-finite rollout values are excluded from finite-value metrics and recorded as `invalid_value_fraction`.

Train baseline caps used for runtime: Burgers 200 samples, Darcy 200 samples, NS2D 50 samples. Test and generated datasets were evaluated at their full local sizes.

## Output Files

- `generalization_eval/metrics.csv`
- `generalization_eval/metrics_sorted_by_similarity.csv`
- `generalization_eval/similarity.csv`
- `generalization_eval/metrics.json`
- `generalization_eval/similarity.json`
- `generalization_eval/burgers_generalization_accuracy_barplot.png`
- `generalization_eval/darcy_generalization_accuracy_barplot.png`
- `generalization_eval/ns2d_generalization_accuracy_barplot.png`

## Similarity Ranking

Bars are ordered as train, test, then generated datasets from closest to farthest by `feature_distance_to_train`. The distance uses dataset-level statistics of the input/initial field: mean, std, min, max, RMS, spectral low/mid/high fractions for Burgers/NS, plus high-value area fraction and edge density for binary Darcy coefficients. The CSV also retains the manual generation tier: `near_param_shift`, `mid_kernel_spectrum`, and `far_range_pattern`.

## Summary

- `burgers`: 52 evaluated datasets. Best `train_original_gaussian_corr0p03` accuracy=98.387, rel_l2=0.01639; worst `burgers_far_centered_scale_shift_scale2p5_shiftm0p25` accuracy=41.162, rel_l2=1.4294.
  Closest generated: `burgers_near_gaussian_corr0p025` distance=0.011; farthest generated: `burgers_far_sign_centered_scale1_shift0` distance=3.401.
- `darcy`: 52 evaluated datasets. Best `darcy_far_alpha5_tau2_bin3_12` accuracy=98.881, rel_l2=0.011315; worst `darcy_far_alpha3_tau1p5_bin1_12` accuracy=67.583, rel_l2=0.47967.
  Closest generated: `darcy_near_alpha2_tau3p5` distance=0.005; farthest generated: `darcy_far_alpha1_tau12_bin2_20` distance=1.944.
- `ns2d`: 52 evaluated datasets. Best `ns_far_square_centered_scale1_shift0` accuracy=98.252, rel_l2=0.017792; worst `ns_mid_spectrum_alpha1p5_tau4` accuracy=50.287, rel_l2=0.98857.
  Closest generated: `ns_far_scale_scale1p5_shift0` distance=0.032; farthest generated: `ns_mid_spectrum_alpha0p8_tau5` distance=4.716.
