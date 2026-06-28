# Generalization Datasets

This file records the curated generalization datasets generated for the selected pretrained FNO models. These datasets are intended for out-of-distribution, generalization, and training-distribution-shift evaluation.

## Selected Models

Use the checkpoints recorded in `GENERALIZATION_TEST_MODELS.md` for evaluation.

## Output Location

- Dataset root: `generalization_datasets`
- Machine-readable manifest: `generalization_datasets/manifest.json`
- Machine-readable summary: `generalization_datasets/summary.json`
- File format: each `.pt` file is a `torch.save` payload containing `x`, `y`, and `metadata`; Darcy files also contain `latent`.

## Training References

- 1D Burgers training reference: Gaussian GRF, periodic BC, correlation length `0.03`, `nx=1024`, `nu=0.001`, `t_final=1.0`, seed `45`.
- 2D Darcy Flow training reference: Neumann-Laplacian cosine-mode GRF with `alpha=2`, `tau=3`, thresholded to binary coefficients `3/12`, solve resolution `421`, model resolution `85`, seed `45`.
- 2D Navier-Stokes training reference: Zongyi real initial conditions spectrally upsampled to `256x256`, Exponax rollout with `nu=1e-5`, `t_final=20`, `21` saved frames. The generated generalization sets use controlled GRF/spectrum/range shifts for OOD testing.

## Similarity Tiers

- `near_param_shift`: closest to training; same broad generator family, parameters changed.
- `mid_kernel_spectrum`: medium shift; covariance kernel, spectrum, binary area fraction, or latent morphology changed.
- `far_range_pattern`: far shift; range, sign, nonlinear transform, sawtooth pattern, coefficient contrast, or strong spectrum changes.

## Summary

| Task | Datasets | Samples | Tensor shapes | Size | Tier split |
|---|---:|---:|---|---:|---|
| `burgers` | 50 | 10000 | x/y: `(200, 1024)` | 79M | near 17, mid 13, far 20 |
| `darcy` | 50 | 10000 | x/y: `(200, 85, 85)`, latent included | 828M | near 17, mid 13, far 20 |
| `ns2d` | 50 | 2500 | x: `(50, 256, 256)`, y: `(50, 256, 256, 21)` | 14G | near 17, mid 13, far 20 |

## 1D Burgers Datasets

| ID | Tier | Family | N | Parameters | Path |
|---|---|---|---:|---|---|
| `burgers_near_gaussian_corr0p015` | `near_param_shift` | `gaussian_parameter_shift` | 200 | `{"correlation_length":0.015,"kernel":"gaussian","transform":"identity"}` | `generalization_datasets/burgers/burgers_near_gaussian_corr0p015.pt` |
| `burgers_near_gaussian_corr0p02` | `near_param_shift` | `gaussian_parameter_shift` | 200 | `{"correlation_length":0.02,"kernel":"gaussian","transform":"identity"}` | `generalization_datasets/burgers/burgers_near_gaussian_corr0p02.pt` |
| `burgers_near_gaussian_corr0p025` | `near_param_shift` | `gaussian_parameter_shift` | 200 | `{"correlation_length":0.025,"kernel":"gaussian","transform":"identity"}` | `generalization_datasets/burgers/burgers_near_gaussian_corr0p025.pt` |
| `burgers_near_gaussian_corr0p035` | `near_param_shift` | `gaussian_parameter_shift` | 200 | `{"correlation_length":0.035,"kernel":"gaussian","transform":"identity"}` | `generalization_datasets/burgers/burgers_near_gaussian_corr0p035.pt` |
| `burgers_near_gaussian_corr0p04` | `near_param_shift` | `gaussian_parameter_shift` | 200 | `{"correlation_length":0.04,"kernel":"gaussian","transform":"identity"}` | `generalization_datasets/burgers/burgers_near_gaussian_corr0p04.pt` |
| `burgers_near_gaussian_corr0p05` | `near_param_shift` | `gaussian_parameter_shift` | 200 | `{"correlation_length":0.05,"kernel":"gaussian","transform":"identity"}` | `generalization_datasets/burgers/burgers_near_gaussian_corr0p05.pt` |
| `burgers_near_gaussian_corr0p06` | `near_param_shift` | `gaussian_parameter_shift` | 200 | `{"correlation_length":0.06,"kernel":"gaussian","transform":"identity"}` | `generalization_datasets/burgers/burgers_near_gaussian_corr0p06.pt` |
| `burgers_near_gaussian_corr0p08` | `near_param_shift` | `gaussian_parameter_shift` | 200 | `{"correlation_length":0.08,"kernel":"gaussian","transform":"identity"}` | `generalization_datasets/burgers/burgers_near_gaussian_corr0p08.pt` |
| `burgers_near_gaussian_corr0p1` | `near_param_shift` | `gaussian_parameter_shift` | 200 | `{"correlation_length":0.1,"kernel":"gaussian","transform":"identity"}` | `generalization_datasets/burgers/burgers_near_gaussian_corr0p1.pt` |
| `burgers_near_gaussian_corr0p12` | `near_param_shift` | `gaussian_parameter_shift` | 200 | `{"correlation_length":0.12,"kernel":"gaussian","transform":"identity"}` | `generalization_datasets/burgers/burgers_near_gaussian_corr0p12.pt` |
| `burgers_near_gaussian_corr0p18` | `near_param_shift` | `gaussian_parameter_shift` | 200 | `{"correlation_length":0.18,"kernel":"gaussian","transform":"identity"}` | `generalization_datasets/burgers/burgers_near_gaussian_corr0p18.pt` |
| `burgers_near_gaussian_corr0p24` | `near_param_shift` | `gaussian_parameter_shift` | 200 | `{"correlation_length":0.24,"kernel":"gaussian","transform":"identity"}` | `generalization_datasets/burgers/burgers_near_gaussian_corr0p24.pt` |
| `burgers_near_gaussian_corr0p3` | `near_param_shift` | `gaussian_parameter_shift` | 200 | `{"correlation_length":0.3,"kernel":"gaussian","transform":"identity"}` | `generalization_datasets/burgers/burgers_near_gaussian_corr0p3.pt` |
| `burgers_near_gaussian_corr0p4` | `near_param_shift` | `gaussian_parameter_shift` | 200 | `{"correlation_length":0.4,"kernel":"gaussian","transform":"identity"}` | `generalization_datasets/burgers/burgers_near_gaussian_corr0p4.pt` |
| `burgers_near_gaussian_corr0p5` | `near_param_shift` | `gaussian_parameter_shift` | 200 | `{"correlation_length":0.5,"kernel":"gaussian","transform":"identity"}` | `generalization_datasets/burgers/burgers_near_gaussian_corr0p5.pt` |
| `burgers_near_gaussian_corr0p75` | `near_param_shift` | `gaussian_parameter_shift` | 200 | `{"correlation_length":0.75,"kernel":"gaussian","transform":"identity"}` | `generalization_datasets/burgers/burgers_near_gaussian_corr0p75.pt` |
| `burgers_near_gaussian_corr1` | `near_param_shift` | `gaussian_parameter_shift` | 200 | `{"correlation_length":1.0,"kernel":"gaussian","transform":"identity"}` | `generalization_datasets/burgers/burgers_near_gaussian_corr1.pt` |
| `burgers_mid_matern_corr0p02_nu0p8` | `mid_kernel_spectrum` | `matern_kernel` | 200 | `{"correlation_length":0.02,"kernel":"matern","matern_nu":0.8,"transform":"identity"}` | `generalization_datasets/burgers/burgers_mid_matern_corr0p02_nu0p8.pt` |
| `burgers_mid_matern_corr0p03_nu1` | `mid_kernel_spectrum` | `matern_kernel` | 200 | `{"correlation_length":0.03,"kernel":"matern","matern_nu":1.0,"transform":"identity"}` | `generalization_datasets/burgers/burgers_mid_matern_corr0p03_nu1.pt` |
| `burgers_mid_matern_corr0p04_nu1p5` | `mid_kernel_spectrum` | `matern_kernel` | 200 | `{"correlation_length":0.04,"kernel":"matern","matern_nu":1.5,"transform":"identity"}` | `generalization_datasets/burgers/burgers_mid_matern_corr0p04_nu1p5.pt` |
| `burgers_mid_matern_corr0p06_nu2p5` | `mid_kernel_spectrum` | `matern_kernel` | 200 | `{"correlation_length":0.06,"kernel":"matern","matern_nu":2.5,"transform":"identity"}` | `generalization_datasets/burgers/burgers_mid_matern_corr0p06_nu2p5.pt` |
| `burgers_mid_matern_corr0p08_nu3p5` | `mid_kernel_spectrum` | `matern_kernel` | 200 | `{"correlation_length":0.08,"kernel":"matern","matern_nu":3.5,"transform":"identity"}` | `generalization_datasets/burgers/burgers_mid_matern_corr0p08_nu3p5.pt` |
| `burgers_mid_matern_corr0p12_nu0p5` | `mid_kernel_spectrum` | `matern_kernel` | 200 | `{"correlation_length":0.12,"kernel":"matern","matern_nu":0.5,"transform":"identity"}` | `generalization_datasets/burgers/burgers_mid_matern_corr0p12_nu0p5.pt` |
| `burgers_mid_matern_corr0p16_nu1p2` | `mid_kernel_spectrum` | `matern_kernel` | 200 | `{"correlation_length":0.16,"kernel":"matern","matern_nu":1.2,"transform":"identity"}` | `generalization_datasets/burgers/burgers_mid_matern_corr0p16_nu1p2.pt` |
| `burgers_mid_matern_corr0p24_nu2` | `mid_kernel_spectrum` | `matern_kernel` | 200 | `{"correlation_length":0.24,"kernel":"matern","matern_nu":2.0,"transform":"identity"}` | `generalization_datasets/burgers/burgers_mid_matern_corr0p24_nu2.pt` |
| `burgers_mid_matern_corr0p32_nu3` | `mid_kernel_spectrum` | `matern_kernel` | 200 | `{"correlation_length":0.32,"kernel":"matern","matern_nu":3.0,"transform":"identity"}` | `generalization_datasets/burgers/burgers_mid_matern_corr0p32_nu3.pt` |
| `burgers_mid_matern_corr0p5_nu1` | `mid_kernel_spectrum` | `matern_kernel` | 200 | `{"correlation_length":0.5,"kernel":"matern","matern_nu":1.0,"transform":"identity"}` | `generalization_datasets/burgers/burgers_mid_matern_corr0p5_nu1.pt` |
| `burgers_mid_matern_corr0p75_nu2p5` | `mid_kernel_spectrum` | `matern_kernel` | 200 | `{"correlation_length":0.75,"kernel":"matern","matern_nu":2.5,"transform":"identity"}` | `generalization_datasets/burgers/burgers_mid_matern_corr0p75_nu2p5.pt` |
| `burgers_mid_matern_corr1_nu4` | `mid_kernel_spectrum` | `matern_kernel` | 200 | `{"correlation_length":1.0,"kernel":"matern","matern_nu":4.0,"transform":"identity"}` | `generalization_datasets/burgers/burgers_mid_matern_corr1_nu4.pt` |
| `burgers_mid_matern_corr0p1_nu5` | `mid_kernel_spectrum` | `matern_kernel` | 200 | `{"correlation_length":0.1,"kernel":"matern","matern_nu":5.0,"transform":"identity"}` | `generalization_datasets/burgers/burgers_mid_matern_corr0p1_nu5.pt` |
| `burgers_far_centered_scale_shift_scale1p5_shift0` | `far_range_pattern` | `range_or_pattern_shift` | 200 | `{"correlation_length":0.03,"kernel":"gaussian","scale":1.5,"shift":0.0,"transform":"centered_scale_shift"}` | `generalization_datasets/burgers/burgers_far_centered_scale_shift_scale1p5_shift0.pt` |
| `burgers_far_centered_scale_shift_scale2_shift0` | `far_range_pattern` | `range_or_pattern_shift` | 200 | `{"correlation_length":0.03,"kernel":"gaussian","scale":2.0,"shift":0.0,"transform":"centered_scale_shift"}` | `generalization_datasets/burgers/burgers_far_centered_scale_shift_scale2_shift0.pt` |
| `burgers_far_centered_scale_shift_scale0p5_shift0` | `far_range_pattern` | `range_or_pattern_shift` | 200 | `{"correlation_length":0.03,"kernel":"gaussian","scale":0.5,"shift":0.0,"transform":"centered_scale_shift"}` | `generalization_datasets/burgers/burgers_far_centered_scale_shift_scale0p5_shift0.pt` |
| `burgers_far_positive_shift_scale1_shift0p25` | `far_range_pattern` | `range_or_pattern_shift` | 200 | `{"correlation_length":0.03,"kernel":"gaussian","scale":1.0,"shift":0.25,"transform":"positive_shift"}` | `generalization_datasets/burgers/burgers_far_positive_shift_scale1_shift0p25.pt` |
| `burgers_far_positive_shift_scale1_shift0p5` | `far_range_pattern` | `range_or_pattern_shift` | 200 | `{"correlation_length":0.03,"kernel":"gaussian","scale":1.0,"shift":0.5,"transform":"positive_shift"}` | `generalization_datasets/burgers/burgers_far_positive_shift_scale1_shift0p5.pt` |
| `burgers_far_negative_shift_scale1_shift0p25` | `far_range_pattern` | `range_or_pattern_shift` | 200 | `{"correlation_length":0.03,"kernel":"gaussian","scale":1.0,"shift":0.25,"transform":"negative_shift"}` | `generalization_datasets/burgers/burgers_far_negative_shift_scale1_shift0p25.pt` |
| `burgers_far_negative_shift_scale1_shift0p5` | `far_range_pattern` | `range_or_pattern_shift` | 200 | `{"correlation_length":0.03,"kernel":"gaussian","scale":1.0,"shift":0.5,"transform":"negative_shift"}` | `generalization_datasets/burgers/burgers_far_negative_shift_scale1_shift0p5.pt` |
| `burgers_far_zero_mean_scale1_shift0` | `far_range_pattern` | `range_or_pattern_shift` | 200 | `{"correlation_length":0.03,"kernel":"gaussian","scale":1.0,"shift":0.0,"transform":"zero_mean"}` | `generalization_datasets/burgers/burgers_far_zero_mean_scale1_shift0.pt` |
| `burgers_far_sign_centered_scale1_shift0` | `far_range_pattern` | `range_or_pattern_shift` | 200 | `{"correlation_length":0.03,"kernel":"gaussian","scale":1.0,"shift":0.0,"transform":"sign_centered"}` | `generalization_datasets/burgers/burgers_far_sign_centered_scale1_shift0.pt` |
| `burgers_far_sign_centered_scale0p5_shift0p25` | `far_range_pattern` | `range_or_pattern_shift` | 200 | `{"correlation_length":0.03,"kernel":"gaussian","scale":0.5,"shift":0.25,"transform":"sign_centered"}` | `generalization_datasets/burgers/burgers_far_sign_centered_scale0p5_shift0p25.pt` |
| `burgers_far_sawtooth_add_scale0p15_shift0` | `far_range_pattern` | `range_or_pattern_shift` | 200 | `{"correlation_length":0.03,"kernel":"gaussian","scale":0.15,"shift":0.0,"transform":"sawtooth_add"}` | `generalization_datasets/burgers/burgers_far_sawtooth_add_scale0p15_shift0.pt` |
| `burgers_far_sawtooth_add_scale0p3_shift0` | `far_range_pattern` | `range_or_pattern_shift` | 200 | `{"correlation_length":0.03,"kernel":"gaussian","scale":0.3,"shift":0.0,"transform":"sawtooth_add"}` | `generalization_datasets/burgers/burgers_far_sawtooth_add_scale0p3_shift0.pt` |
| `burgers_far_sawtooth_add_scale0p5_shift0` | `far_range_pattern` | `range_or_pattern_shift` | 200 | `{"correlation_length":0.03,"kernel":"gaussian","scale":0.5,"shift":0.0,"transform":"sawtooth_add"}` | `generalization_datasets/burgers/burgers_far_sawtooth_add_scale0p5_shift0.pt` |
| `burgers_far_centered_scale_shift_scale1_shift0p4` | `far_range_pattern` | `range_or_pattern_shift` | 200 | `{"correlation_length":0.03,"kernel":"gaussian","scale":1.0,"shift":0.4,"transform":"centered_scale_shift"}` | `generalization_datasets/burgers/burgers_far_centered_scale_shift_scale1_shift0p4.pt` |
| `burgers_far_centered_scale_shift_scale1_shiftm0p4` | `far_range_pattern` | `range_or_pattern_shift` | 200 | `{"correlation_length":0.03,"kernel":"gaussian","scale":1.0,"shift":-0.4,"transform":"centered_scale_shift"}` | `generalization_datasets/burgers/burgers_far_centered_scale_shift_scale1_shiftm0p4.pt` |
| `burgers_far_centered_scale_shift_scale2p5_shift0p25` | `far_range_pattern` | `range_or_pattern_shift` | 200 | `{"correlation_length":0.03,"kernel":"gaussian","scale":2.5,"shift":0.25,"transform":"centered_scale_shift"}` | `generalization_datasets/burgers/burgers_far_centered_scale_shift_scale2p5_shift0p25.pt` |
| `burgers_far_centered_scale_shift_scale2p5_shiftm0p25` | `far_range_pattern` | `range_or_pattern_shift` | 200 | `{"correlation_length":0.03,"kernel":"gaussian","scale":2.5,"shift":-0.25,"transform":"centered_scale_shift"}` | `generalization_datasets/burgers/burgers_far_centered_scale_shift_scale2p5_shiftm0p25.pt` |
| `burgers_far_positive_shift_scale1_shift1` | `far_range_pattern` | `range_or_pattern_shift` | 200 | `{"correlation_length":0.03,"kernel":"gaussian","scale":1.0,"shift":1.0,"transform":"positive_shift"}` | `generalization_datasets/burgers/burgers_far_positive_shift_scale1_shift1.pt` |
| `burgers_far_negative_shift_scale1_shift1` | `far_range_pattern` | `range_or_pattern_shift` | 200 | `{"correlation_length":0.03,"kernel":"gaussian","scale":1.0,"shift":1.0,"transform":"negative_shift"}` | `generalization_datasets/burgers/burgers_far_negative_shift_scale1_shift1.pt` |
| `burgers_far_sawtooth_add_scale0p8_shift0` | `far_range_pattern` | `range_or_pattern_shift` | 200 | `{"correlation_length":0.03,"kernel":"gaussian","scale":0.8,"shift":0.0,"transform":"sawtooth_add"}` | `generalization_datasets/burgers/burgers_far_sawtooth_add_scale0p8_shift0.pt` |

## 2D Darcy Flow Datasets

| ID | Tier | Family | N | Parameters | Path |
|---|---|---|---:|---|---|
| `darcy_near_alpha1p7_tau3` | `near_param_shift` | `darcy_grf_parameter_shift` | 200 | `{"alpha":1.7,"high":12.0,"latent_transform":"identity","low":3.0,"tau":3.0,"threshold_bias":0.0}` | `generalization_datasets/darcy/darcy_near_alpha1p7_tau3.pt` |
| `darcy_near_alpha1p85_tau3` | `near_param_shift` | `darcy_grf_parameter_shift` | 200 | `{"alpha":1.85,"high":12.0,"latent_transform":"identity","low":3.0,"tau":3.0,"threshold_bias":0.0}` | `generalization_datasets/darcy/darcy_near_alpha1p85_tau3.pt` |
| `darcy_near_alpha2p15_tau3` | `near_param_shift` | `darcy_grf_parameter_shift` | 200 | `{"alpha":2.15,"high":12.0,"latent_transform":"identity","low":3.0,"tau":3.0,"threshold_bias":0.0}` | `generalization_datasets/darcy/darcy_near_alpha2p15_tau3.pt` |
| `darcy_near_alpha2p3_tau3` | `near_param_shift` | `darcy_grf_parameter_shift` | 200 | `{"alpha":2.3,"high":12.0,"latent_transform":"identity","low":3.0,"tau":3.0,"threshold_bias":0.0}` | `generalization_datasets/darcy/darcy_near_alpha2p3_tau3.pt` |
| `darcy_near_alpha2_tau2` | `near_param_shift` | `darcy_grf_parameter_shift` | 200 | `{"alpha":2.0,"high":12.0,"latent_transform":"identity","low":3.0,"tau":2.0,"threshold_bias":0.0}` | `generalization_datasets/darcy/darcy_near_alpha2_tau2.pt` |
| `darcy_near_alpha2_tau2p5` | `near_param_shift` | `darcy_grf_parameter_shift` | 200 | `{"alpha":2.0,"high":12.0,"latent_transform":"identity","low":3.0,"tau":2.5,"threshold_bias":0.0}` | `generalization_datasets/darcy/darcy_near_alpha2_tau2p5.pt` |
| `darcy_near_alpha2_tau3p5` | `near_param_shift` | `darcy_grf_parameter_shift` | 200 | `{"alpha":2.0,"high":12.0,"latent_transform":"identity","low":3.0,"tau":3.5,"threshold_bias":0.0}` | `generalization_datasets/darcy/darcy_near_alpha2_tau3p5.pt` |
| `darcy_near_alpha2_tau4` | `near_param_shift` | `darcy_grf_parameter_shift` | 200 | `{"alpha":2.0,"high":12.0,"latent_transform":"identity","low":3.0,"tau":4.0,"threshold_bias":0.0}` | `generalization_datasets/darcy/darcy_near_alpha2_tau4.pt` |
| `darcy_near_alpha1p5_tau2p5` | `near_param_shift` | `darcy_grf_parameter_shift` | 200 | `{"alpha":1.5,"high":12.0,"latent_transform":"identity","low":3.0,"tau":2.5,"threshold_bias":0.0}` | `generalization_datasets/darcy/darcy_near_alpha1p5_tau2p5.pt` |
| `darcy_near_alpha2p5_tau3p5` | `near_param_shift` | `darcy_grf_parameter_shift` | 200 | `{"alpha":2.5,"high":12.0,"latent_transform":"identity","low":3.0,"tau":3.5,"threshold_bias":0.0}` | `generalization_datasets/darcy/darcy_near_alpha2p5_tau3p5.pt` |
| `darcy_near_alpha3_tau4` | `near_param_shift` | `darcy_grf_parameter_shift` | 200 | `{"alpha":3.0,"high":12.0,"latent_transform":"identity","low":3.0,"tau":4.0,"threshold_bias":0.0}` | `generalization_datasets/darcy/darcy_near_alpha3_tau4.pt` |
| `darcy_near_alpha1p2_tau3` | `near_param_shift` | `darcy_grf_parameter_shift` | 200 | `{"alpha":1.2,"high":12.0,"latent_transform":"identity","low":3.0,"tau":3.0,"threshold_bias":0.0}` | `generalization_datasets/darcy/darcy_near_alpha1p2_tau3.pt` |
| `darcy_near_alpha3p5_tau3` | `near_param_shift` | `darcy_grf_parameter_shift` | 200 | `{"alpha":3.5,"high":12.0,"latent_transform":"identity","low":3.0,"tau":3.0,"threshold_bias":0.0}` | `generalization_datasets/darcy/darcy_near_alpha3p5_tau3.pt` |
| `darcy_near_alpha2_tau5` | `near_param_shift` | `darcy_grf_parameter_shift` | 200 | `{"alpha":2.0,"high":12.0,"latent_transform":"identity","low":3.0,"tau":5.0,"threshold_bias":0.0}` | `generalization_datasets/darcy/darcy_near_alpha2_tau5.pt` |
| `darcy_near_alpha2_tau6` | `near_param_shift` | `darcy_grf_parameter_shift` | 200 | `{"alpha":2.0,"high":12.0,"latent_transform":"identity","low":3.0,"tau":6.0,"threshold_bias":0.0}` | `generalization_datasets/darcy/darcy_near_alpha2_tau6.pt` |
| `darcy_near_alpha1_tau2` | `near_param_shift` | `darcy_grf_parameter_shift` | 200 | `{"alpha":1.0,"high":12.0,"latent_transform":"identity","low":3.0,"tau":2.0,"threshold_bias":0.0}` | `generalization_datasets/darcy/darcy_near_alpha1_tau2.pt` |
| `darcy_near_alpha4_tau4` | `near_param_shift` | `darcy_grf_parameter_shift` | 200 | `{"alpha":4.0,"high":12.0,"latent_transform":"identity","low":3.0,"tau":4.0,"threshold_bias":0.0}` | `generalization_datasets/darcy/darcy_near_alpha4_tau4.pt` |
| `darcy_mid_identity_biasm0p8` | `mid_kernel_spectrum` | `binary_area_or_latent_transform` | 200 | `{"alpha":2.0,"high":12.0,"latent_transform":"identity","low":3.0,"tau":3.0,"threshold_bias":-0.8}` | `generalization_datasets/darcy/darcy_mid_identity_biasm0p8.pt` |
| `darcy_mid_identity_biasm0p5` | `mid_kernel_spectrum` | `binary_area_or_latent_transform` | 200 | `{"alpha":2.0,"high":12.0,"latent_transform":"identity","low":3.0,"tau":3.0,"threshold_bias":-0.5}` | `generalization_datasets/darcy/darcy_mid_identity_biasm0p5.pt` |
| `darcy_mid_identity_biasm0p25` | `mid_kernel_spectrum` | `binary_area_or_latent_transform` | 200 | `{"alpha":2.0,"high":12.0,"latent_transform":"identity","low":3.0,"tau":3.0,"threshold_bias":-0.25}` | `generalization_datasets/darcy/darcy_mid_identity_biasm0p25.pt` |
| `darcy_mid_identity_bias0p25` | `mid_kernel_spectrum` | `binary_area_or_latent_transform` | 200 | `{"alpha":2.0,"high":12.0,"latent_transform":"identity","low":3.0,"tau":3.0,"threshold_bias":0.25}` | `generalization_datasets/darcy/darcy_mid_identity_bias0p25.pt` |
| `darcy_mid_identity_bias0p5` | `mid_kernel_spectrum` | `binary_area_or_latent_transform` | 200 | `{"alpha":2.0,"high":12.0,"latent_transform":"identity","low":3.0,"tau":3.0,"threshold_bias":0.5}` | `generalization_datasets/darcy/darcy_mid_identity_bias0p5.pt` |
| `darcy_mid_identity_bias0p8` | `mid_kernel_spectrum` | `binary_area_or_latent_transform` | 200 | `{"alpha":2.0,"high":12.0,"latent_transform":"identity","low":3.0,"tau":3.0,"threshold_bias":0.8}` | `generalization_datasets/darcy/darcy_mid_identity_bias0p8.pt` |
| `darcy_mid_negative_bias0` | `mid_kernel_spectrum` | `binary_area_or_latent_transform` | 200 | `{"alpha":2.0,"high":12.0,"latent_transform":"negative","low":3.0,"tau":3.0,"threshold_bias":0.0}` | `generalization_datasets/darcy/darcy_mid_negative_bias0.pt` |
| `darcy_mid_square_centered_bias0` | `mid_kernel_spectrum` | `binary_area_or_latent_transform` | 200 | `{"alpha":2.0,"high":12.0,"latent_transform":"square_centered","low":3.0,"tau":3.0,"threshold_bias":0.0}` | `generalization_datasets/darcy/darcy_mid_square_centered_bias0.pt` |
| `darcy_mid_square_centered_bias0p2` | `mid_kernel_spectrum` | `binary_area_or_latent_transform` | 200 | `{"alpha":2.0,"high":12.0,"latent_transform":"square_centered","low":3.0,"tau":3.0,"threshold_bias":0.2}` | `generalization_datasets/darcy/darcy_mid_square_centered_bias0p2.pt` |
| `darcy_mid_square_centered_biasm0p2` | `mid_kernel_spectrum` | `binary_area_or_latent_transform` | 200 | `{"alpha":2.0,"high":12.0,"latent_transform":"square_centered","low":3.0,"tau":3.0,"threshold_bias":-0.2}` | `generalization_datasets/darcy/darcy_mid_square_centered_biasm0p2.pt` |
| `darcy_mid_log_abs_centered_bias0` | `mid_kernel_spectrum` | `binary_area_or_latent_transform` | 200 | `{"alpha":2.0,"high":12.0,"latent_transform":"log_abs_centered","low":3.0,"tau":3.0,"threshold_bias":0.0}` | `generalization_datasets/darcy/darcy_mid_log_abs_centered_bias0.pt` |
| `darcy_mid_log_abs_centered_bias0p2` | `mid_kernel_spectrum` | `binary_area_or_latent_transform` | 200 | `{"alpha":2.0,"high":12.0,"latent_transform":"log_abs_centered","low":3.0,"tau":3.0,"threshold_bias":0.2}` | `generalization_datasets/darcy/darcy_mid_log_abs_centered_bias0p2.pt` |
| `darcy_mid_log_abs_centered_biasm0p2` | `mid_kernel_spectrum` | `binary_area_or_latent_transform` | 200 | `{"alpha":2.0,"high":12.0,"latent_transform":"log_abs_centered","low":3.0,"tau":3.0,"threshold_bias":-0.2}` | `generalization_datasets/darcy/darcy_mid_log_abs_centered_biasm0p2.pt` |
| `darcy_far_alpha2_tau3_bin2_15` | `far_range_pattern` | `coefficient_contrast_and_spectrum_shift` | 200 | `{"alpha":2.0,"high":15.0,"latent_transform":"identity","low":2.0,"tau":3.0,"threshold_bias":0.0}` | `generalization_datasets/darcy/darcy_far_alpha2_tau3_bin2_15.pt` |
| `darcy_far_alpha2_tau3_bin1_12` | `far_range_pattern` | `coefficient_contrast_and_spectrum_shift` | 200 | `{"alpha":2.0,"high":12.0,"latent_transform":"identity","low":1.0,"tau":3.0,"threshold_bias":0.0}` | `generalization_datasets/darcy/darcy_far_alpha2_tau3_bin1_12.pt` |
| `darcy_far_alpha2_tau3_bin3_20` | `far_range_pattern` | `coefficient_contrast_and_spectrum_shift` | 200 | `{"alpha":2.0,"high":20.0,"latent_transform":"identity","low":3.0,"tau":3.0,"threshold_bias":0.0}` | `generalization_datasets/darcy/darcy_far_alpha2_tau3_bin3_20.pt` |
| `darcy_far_alpha1_tau1p5_bin3_12` | `far_range_pattern` | `coefficient_contrast_and_spectrum_shift` | 200 | `{"alpha":1.0,"high":12.0,"latent_transform":"identity","low":3.0,"tau":1.5,"threshold_bias":0.0}` | `generalization_datasets/darcy/darcy_far_alpha1_tau1p5_bin3_12.pt` |
| `darcy_far_alpha4_tau8_bin3_12` | `far_range_pattern` | `coefficient_contrast_and_spectrum_shift` | 200 | `{"alpha":4.0,"high":12.0,"latent_transform":"identity","low":3.0,"tau":8.0,"threshold_bias":0.0}` | `generalization_datasets/darcy/darcy_far_alpha4_tau8_bin3_12.pt` |
| `darcy_far_alpha0p8_tau10_bin3_12` | `far_range_pattern` | `coefficient_contrast_and_spectrum_shift` | 200 | `{"alpha":0.8,"high":12.0,"latent_transform":"identity","low":3.0,"tau":10.0,"threshold_bias":0.0}` | `generalization_datasets/darcy/darcy_far_alpha0p8_tau10_bin3_12.pt` |
| `darcy_far_alpha5_tau2_bin3_12` | `far_range_pattern` | `coefficient_contrast_and_spectrum_shift` | 200 | `{"alpha":5.0,"high":12.0,"latent_transform":"identity","low":3.0,"tau":2.0,"threshold_bias":0.0}` | `generalization_datasets/darcy/darcy_far_alpha5_tau2_bin3_12.pt` |
| `darcy_far_alpha1p5_tau6_bin2_15` | `far_range_pattern` | `coefficient_contrast_and_spectrum_shift` | 200 | `{"alpha":1.5,"high":15.0,"latent_transform":"identity","low":2.0,"tau":6.0,"threshold_bias":0.0}` | `generalization_datasets/darcy/darcy_far_alpha1p5_tau6_bin2_15.pt` |
| `darcy_far_alpha3_tau1p5_bin1_12` | `far_range_pattern` | `coefficient_contrast_and_spectrum_shift` | 200 | `{"alpha":3.0,"high":12.0,"latent_transform":"identity","low":1.0,"tau":1.5,"threshold_bias":0.0}` | `generalization_datasets/darcy/darcy_far_alpha3_tau1p5_bin1_12.pt` |
| `darcy_far_alpha4p5_tau10_bin3_20` | `far_range_pattern` | `coefficient_contrast_and_spectrum_shift` | 200 | `{"alpha":4.5,"high":20.0,"latent_transform":"identity","low":3.0,"tau":10.0,"threshold_bias":0.0}` | `generalization_datasets/darcy/darcy_far_alpha4p5_tau10_bin3_20.pt` |
| `darcy_far_alpha1_tau12_bin2_20` | `far_range_pattern` | `coefficient_contrast_and_spectrum_shift` | 200 | `{"alpha":1.0,"high":20.0,"latent_transform":"identity","low":2.0,"tau":12.0,"threshold_bias":0.0}` | `generalization_datasets/darcy/darcy_far_alpha1_tau12_bin2_20.pt` |
| `darcy_far_alpha6_tau1p5_bin3_12` | `far_range_pattern` | `coefficient_contrast_and_spectrum_shift` | 200 | `{"alpha":6.0,"high":12.0,"latent_transform":"identity","low":3.0,"tau":1.5,"threshold_bias":0.0}` | `generalization_datasets/darcy/darcy_far_alpha6_tau1p5_bin3_12.pt` |
| `darcy_far_alpha0p7_tau2_bin3_12` | `far_range_pattern` | `coefficient_contrast_and_spectrum_shift` | 200 | `{"alpha":0.7,"high":12.0,"latent_transform":"identity","low":3.0,"tau":2.0,"threshold_bias":0.0}` | `generalization_datasets/darcy/darcy_far_alpha0p7_tau2_bin3_12.pt` |
| `darcy_far_alpha2_tau12_bin1_20` | `far_range_pattern` | `coefficient_contrast_and_spectrum_shift` | 200 | `{"alpha":2.0,"high":20.0,"latent_transform":"identity","low":1.0,"tau":12.0,"threshold_bias":0.0}` | `generalization_datasets/darcy/darcy_far_alpha2_tau12_bin1_20.pt` |
| `darcy_far_alpha5p5_tau12_bin3_12` | `far_range_pattern` | `coefficient_contrast_and_spectrum_shift` | 200 | `{"alpha":5.5,"high":12.0,"latent_transform":"identity","low":3.0,"tau":12.0,"threshold_bias":0.0}` | `generalization_datasets/darcy/darcy_far_alpha5p5_tau12_bin3_12.pt` |
| `darcy_far_alpha1p2_tau8_bin2_15` | `far_range_pattern` | `coefficient_contrast_and_spectrum_shift` | 200 | `{"alpha":1.2,"high":15.0,"latent_transform":"identity","low":2.0,"tau":8.0,"threshold_bias":0.0}` | `generalization_datasets/darcy/darcy_far_alpha1p2_tau8_bin2_15.pt` |
| `darcy_far_alpha3p5_tau6_bin1_12` | `far_range_pattern` | `coefficient_contrast_and_spectrum_shift` | 200 | `{"alpha":3.5,"high":12.0,"latent_transform":"identity","low":1.0,"tau":6.0,"threshold_bias":0.0}` | `generalization_datasets/darcy/darcy_far_alpha3p5_tau6_bin1_12.pt` |
| `darcy_far_alpha0p5_tau15_bin3_20` | `far_range_pattern` | `coefficient_contrast_and_spectrum_shift` | 200 | `{"alpha":0.5,"high":20.0,"latent_transform":"identity","low":3.0,"tau":15.0,"threshold_bias":0.0}` | `generalization_datasets/darcy/darcy_far_alpha0p5_tau15_bin3_20.pt` |
| `darcy_far_alpha6_tau6_bin2_20` | `far_range_pattern` | `coefficient_contrast_and_spectrum_shift` | 200 | `{"alpha":6.0,"high":20.0,"latent_transform":"identity","low":2.0,"tau":6.0,"threshold_bias":0.0}` | `generalization_datasets/darcy/darcy_far_alpha6_tau6_bin2_20.pt` |
| `darcy_far_alpha2_tau1_bin3_12` | `far_range_pattern` | `coefficient_contrast_and_spectrum_shift` | 200 | `{"alpha":2.0,"high":12.0,"latent_transform":"identity","low":3.0,"tau":1.0,"threshold_bias":0.0}` | `generalization_datasets/darcy/darcy_far_alpha2_tau1_bin3_12.pt` |

## 2D Navier-Stokes Datasets

| ID | Tier | Family | N | Parameters | Path |
|---|---|---|---:|---|---|
| `ns_near_grf_alpha2p2_tau7` | `near_param_shift` | `periodic_grf_parameter_shift` | 50 | `{"alpha":2.2,"tau":7.0,"transform":"identity"}` | `generalization_datasets/ns2d/ns_near_grf_alpha2p2_tau7.pt` |
| `ns_near_grf_alpha2p4_tau7` | `near_param_shift` | `periodic_grf_parameter_shift` | 50 | `{"alpha":2.4,"tau":7.0,"transform":"identity"}` | `generalization_datasets/ns2d/ns_near_grf_alpha2p4_tau7.pt` |
| `ns_near_grf_alpha2p6_tau7` | `near_param_shift` | `periodic_grf_parameter_shift` | 50 | `{"alpha":2.6,"tau":7.0,"transform":"identity"}` | `generalization_datasets/ns2d/ns_near_grf_alpha2p6_tau7.pt` |
| `ns_near_grf_alpha2p8_tau7` | `near_param_shift` | `periodic_grf_parameter_shift` | 50 | `{"alpha":2.8,"tau":7.0,"transform":"identity"}` | `generalization_datasets/ns2d/ns_near_grf_alpha2p8_tau7.pt` |
| `ns_near_grf_alpha2p5_tau5` | `near_param_shift` | `periodic_grf_parameter_shift` | 50 | `{"alpha":2.5,"tau":5.0,"transform":"identity"}` | `generalization_datasets/ns2d/ns_near_grf_alpha2p5_tau5.pt` |
| `ns_near_grf_alpha2p5_tau6` | `near_param_shift` | `periodic_grf_parameter_shift` | 50 | `{"alpha":2.5,"tau":6.0,"transform":"identity"}` | `generalization_datasets/ns2d/ns_near_grf_alpha2p5_tau6.pt` |
| `ns_near_grf_alpha2p5_tau8` | `near_param_shift` | `periodic_grf_parameter_shift` | 50 | `{"alpha":2.5,"tau":8.0,"transform":"identity"}` | `generalization_datasets/ns2d/ns_near_grf_alpha2p5_tau8.pt` |
| `ns_near_grf_alpha2p5_tau9` | `near_param_shift` | `periodic_grf_parameter_shift` | 50 | `{"alpha":2.5,"tau":9.0,"transform":"identity"}` | `generalization_datasets/ns2d/ns_near_grf_alpha2p5_tau9.pt` |
| `ns_near_grf_alpha2_tau6` | `near_param_shift` | `periodic_grf_parameter_shift` | 50 | `{"alpha":2.0,"tau":6.0,"transform":"identity"}` | `generalization_datasets/ns2d/ns_near_grf_alpha2_tau6.pt` |
| `ns_near_grf_alpha3_tau8` | `near_param_shift` | `periodic_grf_parameter_shift` | 50 | `{"alpha":3.0,"tau":8.0,"transform":"identity"}` | `generalization_datasets/ns2d/ns_near_grf_alpha3_tau8.pt` |
| `ns_near_grf_alpha3p5_tau10` | `near_param_shift` | `periodic_grf_parameter_shift` | 50 | `{"alpha":3.5,"tau":10.0,"transform":"identity"}` | `generalization_datasets/ns2d/ns_near_grf_alpha3p5_tau10.pt` |
| `ns_near_grf_alpha1p8_tau5` | `near_param_shift` | `periodic_grf_parameter_shift` | 50 | `{"alpha":1.8,"tau":5.0,"transform":"identity"}` | `generalization_datasets/ns2d/ns_near_grf_alpha1p8_tau5.pt` |
| `ns_near_grf_alpha4_tau12` | `near_param_shift` | `periodic_grf_parameter_shift` | 50 | `{"alpha":4.0,"tau":12.0,"transform":"identity"}` | `generalization_datasets/ns2d/ns_near_grf_alpha4_tau12.pt` |
| `ns_near_grf_alpha2p2_tau4` | `near_param_shift` | `periodic_grf_parameter_shift` | 50 | `{"alpha":2.2,"tau":4.0,"transform":"identity"}` | `generalization_datasets/ns2d/ns_near_grf_alpha2p2_tau4.pt` |
| `ns_near_grf_alpha3p2_tau6` | `near_param_shift` | `periodic_grf_parameter_shift` | 50 | `{"alpha":3.2,"tau":6.0,"transform":"identity"}` | `generalization_datasets/ns2d/ns_near_grf_alpha3p2_tau6.pt` |
| `ns_near_grf_alpha1p5_tau7` | `near_param_shift` | `periodic_grf_parameter_shift` | 50 | `{"alpha":1.5,"tau":7.0,"transform":"identity"}` | `generalization_datasets/ns2d/ns_near_grf_alpha1p5_tau7.pt` |
| `ns_near_grf_alpha4p5_tau7` | `near_param_shift` | `periodic_grf_parameter_shift` | 50 | `{"alpha":4.5,"tau":7.0,"transform":"identity"}` | `generalization_datasets/ns2d/ns_near_grf_alpha4p5_tau7.pt` |
| `ns_mid_spectrum_alpha1_tau2` | `mid_kernel_spectrum` | `periodic_grf_spectrum_shift` | 50 | `{"alpha":1.0,"tau":2.0,"transform":"identity"}` | `generalization_datasets/ns2d/ns_mid_spectrum_alpha1_tau2.pt` |
| `ns_mid_spectrum_alpha1p2_tau3` | `mid_kernel_spectrum` | `periodic_grf_spectrum_shift` | 50 | `{"alpha":1.2,"tau":3.0,"transform":"identity"}` | `generalization_datasets/ns2d/ns_mid_spectrum_alpha1p2_tau3.pt` |
| `ns_mid_spectrum_alpha1p5_tau4` | `mid_kernel_spectrum` | `periodic_grf_spectrum_shift` | 50 | `{"alpha":1.5,"tau":4.0,"transform":"identity"}` | `generalization_datasets/ns2d/ns_mid_spectrum_alpha1p5_tau4.pt` |
| `ns_mid_spectrum_alpha2_tau2` | `mid_kernel_spectrum` | `periodic_grf_spectrum_shift` | 50 | `{"alpha":2.0,"tau":2.0,"transform":"identity"}` | `generalization_datasets/ns2d/ns_mid_spectrum_alpha2_tau2.pt` |
| `ns_mid_spectrum_alpha3_tau3` | `mid_kernel_spectrum` | `periodic_grf_spectrum_shift` | 50 | `{"alpha":3.0,"tau":3.0,"transform":"identity"}` | `generalization_datasets/ns2d/ns_mid_spectrum_alpha3_tau3.pt` |
| `ns_mid_spectrum_alpha4_tau4` | `mid_kernel_spectrum` | `periodic_grf_spectrum_shift` | 50 | `{"alpha":4.0,"tau":4.0,"transform":"identity"}` | `generalization_datasets/ns2d/ns_mid_spectrum_alpha4_tau4.pt` |
| `ns_mid_spectrum_alpha5_tau8` | `mid_kernel_spectrum` | `periodic_grf_spectrum_shift` | 50 | `{"alpha":5.0,"tau":8.0,"transform":"identity"}` | `generalization_datasets/ns2d/ns_mid_spectrum_alpha5_tau8.pt` |
| `ns_mid_spectrum_alpha6_tau10` | `mid_kernel_spectrum` | `periodic_grf_spectrum_shift` | 50 | `{"alpha":6.0,"tau":10.0,"transform":"identity"}` | `generalization_datasets/ns2d/ns_mid_spectrum_alpha6_tau10.pt` |
| `ns_mid_spectrum_alpha1_tau10` | `mid_kernel_spectrum` | `periodic_grf_spectrum_shift` | 50 | `{"alpha":1.0,"tau":10.0,"transform":"identity"}` | `generalization_datasets/ns2d/ns_mid_spectrum_alpha1_tau10.pt` |
| `ns_mid_spectrum_alpha2_tau15` | `mid_kernel_spectrum` | `periodic_grf_spectrum_shift` | 50 | `{"alpha":2.0,"tau":15.0,"transform":"identity"}` | `generalization_datasets/ns2d/ns_mid_spectrum_alpha2_tau15.pt` |
| `ns_mid_spectrum_alpha3_tau20` | `mid_kernel_spectrum` | `periodic_grf_spectrum_shift` | 50 | `{"alpha":3.0,"tau":20.0,"transform":"identity"}` | `generalization_datasets/ns2d/ns_mid_spectrum_alpha3_tau20.pt` |
| `ns_mid_spectrum_alpha5_tau20` | `mid_kernel_spectrum` | `periodic_grf_spectrum_shift` | 50 | `{"alpha":5.0,"tau":20.0,"transform":"identity"}` | `generalization_datasets/ns2d/ns_mid_spectrum_alpha5_tau20.pt` |
| `ns_mid_spectrum_alpha0p8_tau5` | `mid_kernel_spectrum` | `periodic_grf_spectrum_shift` | 50 | `{"alpha":0.8,"tau":5.0,"transform":"identity"}` | `generalization_datasets/ns2d/ns_mid_spectrum_alpha0p8_tau5.pt` |
| `ns_far_scale_scale0p5_shift0` | `far_range_pattern` | `range_or_pattern_shift` | 50 | `{"alpha":2.5,"scale":0.5,"shift":0.0,"tau":7.0,"transform":"scale"}` | `generalization_datasets/ns2d/ns_far_scale_scale0p5_shift0.pt` |
| `ns_far_scale_scale1p5_shift0` | `far_range_pattern` | `range_or_pattern_shift` | 50 | `{"alpha":2.5,"scale":1.5,"shift":0.0,"tau":7.0,"transform":"scale"}` | `generalization_datasets/ns2d/ns_far_scale_scale1p5_shift0.pt` |
| `ns_far_scale_scale2_shift0` | `far_range_pattern` | `range_or_pattern_shift` | 50 | `{"alpha":2.5,"scale":2.0,"shift":0.0,"tau":7.0,"transform":"scale"}` | `generalization_datasets/ns2d/ns_far_scale_scale2_shift0.pt` |
| `ns_far_positive_shift_scale1_shift0p25` | `far_range_pattern` | `range_or_pattern_shift` | 50 | `{"alpha":2.5,"scale":1.0,"shift":0.25,"tau":7.0,"transform":"positive_shift"}` | `generalization_datasets/ns2d/ns_far_positive_shift_scale1_shift0p25.pt` |
| `ns_far_positive_shift_scale1_shift0p5` | `far_range_pattern` | `range_or_pattern_shift` | 50 | `{"alpha":2.5,"scale":1.0,"shift":0.5,"tau":7.0,"transform":"positive_shift"}` | `generalization_datasets/ns2d/ns_far_positive_shift_scale1_shift0p5.pt` |
| `ns_far_negative_shift_scale1_shift0p25` | `far_range_pattern` | `range_or_pattern_shift` | 50 | `{"alpha":2.5,"scale":1.0,"shift":0.25,"tau":7.0,"transform":"negative_shift"}` | `generalization_datasets/ns2d/ns_far_negative_shift_scale1_shift0p25.pt` |
| `ns_far_negative_shift_scale1_shift0p5` | `far_range_pattern` | `range_or_pattern_shift` | 50 | `{"alpha":2.5,"scale":1.0,"shift":0.5,"tau":7.0,"transform":"negative_shift"}` | `generalization_datasets/ns2d/ns_far_negative_shift_scale1_shift0p5.pt` |
| `ns_far_scale_shift_scale1p5_shift0p25` | `far_range_pattern` | `range_or_pattern_shift` | 50 | `{"alpha":2.5,"scale":1.5,"shift":0.25,"tau":7.0,"transform":"scale_shift"}` | `generalization_datasets/ns2d/ns_far_scale_shift_scale1p5_shift0p25.pt` |
| `ns_far_scale_shift_scale1p5_shiftm0p25` | `far_range_pattern` | `range_or_pattern_shift` | 50 | `{"alpha":2.5,"scale":1.5,"shift":-0.25,"tau":7.0,"transform":"scale_shift"}` | `generalization_datasets/ns2d/ns_far_scale_shift_scale1p5_shiftm0p25.pt` |
| `ns_far_sign_scale0p5_shift0` | `far_range_pattern` | `range_or_pattern_shift` | 50 | `{"alpha":2.5,"scale":0.5,"shift":0.0,"tau":7.0,"transform":"sign"}` | `generalization_datasets/ns2d/ns_far_sign_scale0p5_shift0.pt` |
| `ns_far_sign_scale1_shift0` | `far_range_pattern` | `range_or_pattern_shift` | 50 | `{"alpha":2.5,"scale":1.0,"shift":0.0,"tau":7.0,"transform":"sign"}` | `generalization_datasets/ns2d/ns_far_sign_scale1_shift0.pt` |
| `ns_far_square_centered_scale1_shift0` | `far_range_pattern` | `range_or_pattern_shift` | 50 | `{"alpha":2.5,"scale":1.0,"shift":0.0,"tau":7.0,"transform":"square_centered"}` | `generalization_datasets/ns2d/ns_far_square_centered_scale1_shift0.pt` |
| `ns_far_square_centered_scale2_shift0` | `far_range_pattern` | `range_or_pattern_shift` | 50 | `{"alpha":2.5,"scale":2.0,"shift":0.0,"tau":7.0,"transform":"square_centered"}` | `generalization_datasets/ns2d/ns_far_square_centered_scale2_shift0.pt` |
| `ns_far_log_abs_centered_scale1_shift0` | `far_range_pattern` | `range_or_pattern_shift` | 50 | `{"alpha":2.5,"scale":1.0,"shift":0.0,"tau":7.0,"transform":"log_abs_centered"}` | `generalization_datasets/ns2d/ns_far_log_abs_centered_scale1_shift0.pt` |
| `ns_far_log_abs_centered_scale2_shift0` | `far_range_pattern` | `range_or_pattern_shift` | 50 | `{"alpha":2.5,"scale":2.0,"shift":0.0,"tau":7.0,"transform":"log_abs_centered"}` | `generalization_datasets/ns2d/ns_far_log_abs_centered_scale2_shift0.pt` |
| `ns_far_sawtooth_add_scale0p1_shift0` | `far_range_pattern` | `range_or_pattern_shift` | 50 | `{"alpha":2.5,"scale":0.1,"shift":0.0,"tau":7.0,"transform":"sawtooth_add"}` | `generalization_datasets/ns2d/ns_far_sawtooth_add_scale0p1_shift0.pt` |
| `ns_far_sawtooth_add_scale0p25_shift0` | `far_range_pattern` | `range_or_pattern_shift` | 50 | `{"alpha":2.5,"scale":0.25,"shift":0.0,"tau":7.0,"transform":"sawtooth_add"}` | `generalization_datasets/ns2d/ns_far_sawtooth_add_scale0p25_shift0.pt` |
| `ns_far_sawtooth_add_scale0p5_shift0` | `far_range_pattern` | `range_or_pattern_shift` | 50 | `{"alpha":2.5,"scale":0.5,"shift":0.0,"tau":7.0,"transform":"sawtooth_add"}` | `generalization_datasets/ns2d/ns_far_sawtooth_add_scale0p5_shift0.pt` |
| `ns_far_scale_shift_scale2_shift0p5` | `far_range_pattern` | `range_or_pattern_shift` | 50 | `{"alpha":2.5,"scale":2.0,"shift":0.5,"tau":7.0,"transform":"scale_shift"}` | `generalization_datasets/ns2d/ns_far_scale_shift_scale2_shift0p5.pt` |
| `ns_far_scale_shift_scale2_shiftm0p5` | `far_range_pattern` | `range_or_pattern_shift` | 50 | `{"alpha":2.5,"scale":2.0,"shift":-0.5,"tau":7.0,"transform":"scale_shift"}` | `generalization_datasets/ns2d/ns_far_scale_shift_scale2_shiftm0p5.pt` |

## Per-Sample Metadata

Each dataset file stores `metadata.sample_metadata`, where every sample records at least `sample_index` and `dataset_id`. The file-level `metadata` also records the similarity tier, family, generation parameters, solver settings, training reference, and timing records.

## Notes

- These are generated test/generalization datasets, not replacement training datasets.
- The curated design intentionally balances close parameter shifts, medium kernel/spectrum shifts, and far range/pattern shifts.
- The canonical model checkpoints for evaluating these files are listed in `GENERALIZATION_TEST_MODELS.md`.
