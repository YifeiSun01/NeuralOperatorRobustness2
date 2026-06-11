# Burgers Loss3-Targeted 52-Dataset Clean Metrics Appendix - 2026-06-11

This appendix expands the 52-dataset clean inference record into one block per dataset. It includes the exact dataset path, dataset identity/description, parameter JSON, observed input statistics, spectral/shape statistics, RMSE, relative L2, and reductions versus the baseline model.

- Source compact CSV: `forensics/burgers_semantic_wideparam_visible_loss3targeted_round00_52dataset_clean_metrics_20260611/per_dataset_52_compact_rmse_relative_l2_reductions.csv`
- Full 266-column CSV: `forensics/burgers_semantic_wideparam_visible_loss3targeted_round00_52dataset_clean_metrics_20260611/per_dataset_52_clean_metrics.csv`
- Sample-level CSV: `forensics/burgers_semantic_wideparam_visible_loss3targeted_round00_52dataset_clean_metrics_20260611/per_sample_52_clean_metrics.csv`

## 01 - burgers_original_train_gaussian_corr0p03_seed45

- group: `train`
- display label: `Original Burgers train; Gaussian GRF corr=0.03; range observed [0,1]`
- family: `train_test_gaussian`
- samples: `1350`
- path: `/workspace/NeuralOperatorRobustness2/1D_Burgers/datasets/1D/Burgers/batched_exponax_splits/dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45/dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45_train.pt`
- description: Original train/test split generated with Gaussian GRF kernel, correlation_length=0.03, periodic BC, nu=0.001, t_final=1.0.
- parameters: `{"base_family": "train_test_gaussian", "bc": "periodic", "correlation_length": 0.03, "kernel": "gaussian", "nu": 0.001, "seed": 45, "solver": "exponax_batched", "split": "train", "t_final": 1.0, "target_max": 1.0, "target_min": 0.0, "zero_mean": false}`
- observed x stats: min `0`, max `1`, mean `0.500417`, std `0.257074`
- spectral/shape stats: centroid `3.47242`, low frac `0.941911`, mid frac `0.0580887`, high frac `3.65047e-09`, total variation `0.00494846`
- RMSE mean: baseline `0.00875634`, loss1 `0.000746925`, loss2 `0.0014018`, loss3 `0.00425008`
- RMSE reduction vs baseline: loss1 `91.47%`, loss2 `83.99%`, loss3 `51.46%`
- relative L2 mean: baseline `0.0168149`, loss1 `0.00144669`, loss2 `0.00270588`, loss3 `0.00811692`
- relative L2 reduction vs baseline: loss1 `91.40%`, loss2 `83.91%`, loss3 `51.73%`
- loss3 clean-RMSE best among all four: `False`

## 02 - burgers_original_test_gaussian_corr0p03_seed45

- group: `test`
- display label: `Original Burgers test; Gaussian GRF corr=0.03; range observed [0,1]`
- family: `train_test_gaussian`
- samples: `150`
- path: `/workspace/NeuralOperatorRobustness2/1D_Burgers/datasets/1D/Burgers/batched_exponax_splits/dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45/dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45_test.pt`
- description: Original train/test split generated with Gaussian GRF kernel, correlation_length=0.03, periodic BC, nu=0.001, t_final=1.0.
- parameters: `{"base_family": "train_test_gaussian", "bc": "periodic", "correlation_length": 0.03, "kernel": "gaussian", "nu": 0.001, "seed": 45, "solver": "exponax_batched", "split": "test", "t_final": 1.0, "target_max": 1.0, "target_min": 0.0, "zero_mean": false}`
- observed x stats: min `0`, max `1`, mean `0.505544`, std `0.257885`
- spectral/shape stats: centroid `3.43381`, low frac `0.94703`, mid frac `0.0529701`, high frac `3.62095e-09`, total variation `0.00487199`
- RMSE mean: baseline `0.00923196`, loss1 `0.000836263`, loss2 `0.00149737`, loss3 `0.00437389`
- RMSE reduction vs baseline: loss1 `90.94%`, loss2 `83.78%`, loss3 `52.62%`
- relative L2 mean: baseline `0.0175141`, loss1 `0.00161192`, loss2 `0.00287989`, loss3 `0.00832509`
- relative L2 reduction vs baseline: loss1 `90.80%`, loss2 `83.56%`, loss3 `52.47%`
- loss3 clean-RMSE best among all four: `False`

## 03 - burgers_widevis_l3target_d00

- group: `generalization`
- display label: `Gaussian GRF corr=0.012; range [-0.2,1.2]`
- family: `gaussian`
- samples: `200`
- path: `generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00/burgers/burgers_widevis_l3target_d00.pt`
- description: Gaussian GRF with visibly varied correlation length; loss3-targeted replacement candidate; Gaussian GRF corr=0.012; range [-0.2,1.2]
- parameters: `{"base_config": "gauss_c0p012", "base_family": "gaussian", "correlation_length": 0.012, "descriptive_name": "Gaussian GRF corr=0.012; range [-0.2,1.2]", "display_label": "Gaussian GRF corr=0.012; range [-0.2,1.2]", "hard_value_max": 1.7, "hard_value_min": -0.7, "loss3_targeted_profile": true, "target_max": 1.2, "target_min": -0.2, "transform": "range_affine", "transform_variant": "range", "wide_parameter_visible_profile": true}`
- observed x stats: min `-0.2`, max `1.2`, mean `0.502465`, std `0.305798`
- spectral/shape stats: centroid `8.07339`, low frac `0.538649`, mid frac `0.461351`, high frac `6.01744e-09`, total variation `0.0140602`
- RMSE mean: baseline `0.0208882`, loss1 `0.0150907`, loss2 `0.0169971`, loss3 `0.0086141`
- RMSE reduction vs baseline: loss1 `27.75%`, loss2 `18.63%`, loss3 `58.76%`
- relative L2 mean: baseline `0.0408522`, loss1 `0.029363`, loss2 `0.0331167`, loss3 `0.0168573`
- relative L2 reduction vs baseline: loss1 `28.12%`, loss2 `18.94%`, loss3 `58.74%`
- loss3 clean-RMSE best among all four: `True`

## 04 - burgers_widevis_l3target_d01

- group: `generalization`
- display label: `Gaussian GRF corr=0.012; range [-0.3,1.3]`
- family: `gaussian`
- samples: `200`
- path: `generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00/burgers/burgers_widevis_l3target_d01.pt`
- description: Gaussian GRF with visibly varied correlation length; loss3-targeted replacement candidate; Gaussian GRF corr=0.012; range [-0.3,1.3]
- parameters: `{"base_config": "gauss_c0p012", "base_family": "gaussian", "correlation_length": 0.012, "descriptive_name": "Gaussian GRF corr=0.012; range [-0.3,1.3]", "display_label": "Gaussian GRF corr=0.012; range [-0.3,1.3]", "hard_value_max": 1.7, "hard_value_min": -0.7, "loss3_targeted_profile": true, "target_max": 1.3, "target_min": -0.3, "transform": "range_affine", "transform_variant": "range", "wide_parameter_visible_profile": true}`
- observed x stats: min `-0.3`, max `1.3`, mean `0.501181`, std `0.348037`
- spectral/shape stats: centroid `8.07865`, low frac `0.536766`, mid frac `0.463234`, high frac `6.08571e-09`, total variation `0.0160336`
- RMSE mean: baseline `0.0276721`, loss1 `0.0200337`, loss2 `0.0224588`, loss3 `0.0114116`
- RMSE reduction vs baseline: loss1 `27.60%`, loss2 `18.84%`, loss3 `58.76%`
- relative L2 mean: baseline `0.0543989`, loss1 `0.0390322`, loss2 `0.043805`, loss3 `0.0223268`
- relative L2 reduction vs baseline: loss1 `28.25%`, loss2 `19.47%`, loss3 `58.96%`
- loss3 clean-RMSE best among all four: `True`

## 05 - burgers_widevis_l3target_d02

- group: `generalization`
- display label: `Gaussian GRF corr=0.012; range [0.15,1.25]`
- family: `gaussian`
- samples: `200`
- path: `generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00/burgers/burgers_widevis_l3target_d02.pt`
- description: Gaussian GRF with visibly varied correlation length; loss3-targeted replacement candidate; Gaussian GRF corr=0.012; range [0.15,1.25]
- parameters: `{"base_config": "gauss_c0p012", "base_family": "gaussian", "correlation_length": 0.012, "descriptive_name": "Gaussian GRF corr=0.012; range [0.15,1.25]", "display_label": "Gaussian GRF corr=0.012; range [0.15,1.25]", "hard_value_max": 1.7, "hard_value_min": -0.7, "loss3_targeted_profile": true, "target_max": 1.25, "target_min": 0.15, "transform": "range_affine", "transform_variant": "range", "wide_parameter_visible_profile": true}`
- observed x stats: min `0.15`, max `1.25`, mean `0.70203`, std `0.241349`
- spectral/shape stats: centroid `7.99354`, low frac `0.551785`, mid frac `0.448215`, high frac `6.10334e-09`, total variation `0.0110632`
- RMSE mean: baseline `0.0251009`, loss1 `0.0117648`, loss2 `0.0129618`, loss3 `0.006844`
- RMSE reduction vs baseline: loss1 `53.13%`, loss2 `48.36%`, loss3 `72.73%`
- relative L2 mean: baseline `0.0345749`, loss1 `0.0165728`, loss2 `0.0183078`, loss3 `0.00957028`
- relative L2 reduction vs baseline: loss1 `52.07%`, loss2 `47.05%`, loss3 `72.32%`
- loss3 clean-RMSE best among all four: `True`

## 06 - burgers_widevis_l3target_d03

- group: `generalization`
- display label: `Gaussian GRF corr=0.012; range [0,1.2]`
- family: `gaussian`
- samples: `200`
- path: `generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00/burgers/burgers_widevis_l3target_d03.pt`
- description: Gaussian GRF with visibly varied correlation length; loss3-targeted replacement candidate; Gaussian GRF corr=0.012; range [0,1.2]
- parameters: `{"base_config": "gauss_c0p012", "base_family": "gaussian", "correlation_length": 0.012, "descriptive_name": "Gaussian GRF corr=0.012; range [0,1.2]", "display_label": "Gaussian GRF corr=0.012; range [0,1.2]", "hard_value_max": 1.7, "hard_value_min": -0.7, "loss3_targeted_profile": true, "target_max": 1.2, "target_min": 0.0, "transform": "range_affine", "transform_variant": "range", "wide_parameter_visible_profile": true}`
- observed x stats: min `0`, max `1.2`, mean `0.602677`, std `0.264736`
- spectral/shape stats: centroid `7.92545`, low frac `0.554623`, mid frac `0.445377`, high frac `6.02552e-09`, total variation `0.0120127`
- RMSE mean: baseline `0.0191863`, loss1 `0.0115113`, loss2 `0.0130524`, loss3 `0.00670663`
- RMSE reduction vs baseline: loss1 `40.00%`, loss2 `31.97%`, loss3 `65.04%`
- relative L2 mean: baseline `0.0310602`, loss1 `0.0189165`, loss2 `0.0214788`, loss3 `0.0109981`
- relative L2 reduction vs baseline: loss1 `39.10%`, loss2 `30.85%`, loss3 `64.59%`
- loss3 clean-RMSE best among all four: `True`

## 07 - burgers_widevis_l3target_d04

- group: `generalization`
- display label: `Gaussian GRF corr=0.009; range [-0.3,1.3]`
- family: `gaussian`
- samples: `200`
- path: `generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00/burgers/burgers_widevis_l3target_d04.pt`
- description: Gaussian GRF with visibly varied correlation length; loss3-targeted replacement candidate; Gaussian GRF corr=0.009; range [-0.3,1.3]
- parameters: `{"base_config": "gauss_c0p009", "base_family": "gaussian", "correlation_length": 0.009, "descriptive_name": "Gaussian GRF corr=0.009; range [-0.3,1.3]", "display_label": "Gaussian GRF corr=0.009; range [-0.3,1.3]", "hard_value_max": 1.7, "hard_value_min": -0.7, "loss3_targeted_profile": true, "target_max": 1.3, "target_min": -0.3, "transform": "range_affine", "transform_variant": "range", "wide_parameter_visible_profile": true}`
- observed x stats: min `-0.3`, max `1.3`, mean `0.492603`, std `0.33422`
- spectral/shape stats: centroid `10.4547`, low frac `0.427543`, mid frac `0.572456`, high frac `4.60501e-07`, total variation `0.0201973`
- RMSE mean: baseline `0.0254725`, loss1 `0.0209863`, loss2 `0.0228983`, loss3 `0.0121777`
- RMSE reduction vs baseline: loss1 `17.61%`, loss2 `10.11%`, loss3 `52.19%`
- relative L2 mean: baseline `0.0513556`, loss1 `0.0423071`, loss2 `0.0461838`, loss3 `0.0246716`
- relative L2 reduction vs baseline: loss1 `17.62%`, loss2 `10.07%`, loss3 `51.96%`
- loss3 clean-RMSE best among all four: `True`

## 08 - burgers_widevis_l3target_d05

- group: `generalization`
- display label: `Gaussian GRF corr=0.009; range [0.15,1.25]`
- family: `gaussian`
- samples: `200`
- path: `generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00/burgers/burgers_widevis_l3target_d05.pt`
- description: Gaussian GRF with visibly varied correlation length; loss3-targeted replacement candidate; Gaussian GRF corr=0.009; range [0.15,1.25]
- parameters: `{"base_config": "gauss_c0p009", "base_family": "gaussian", "correlation_length": 0.009, "descriptive_name": "Gaussian GRF corr=0.009; range [0.15,1.25]", "display_label": "Gaussian GRF corr=0.009; range [0.15,1.25]", "hard_value_max": 1.7, "hard_value_min": -0.7, "loss3_targeted_profile": true, "target_max": 1.25, "target_min": 0.15, "transform": "range_affine", "transform_variant": "range", "wide_parameter_visible_profile": true}`
- observed x stats: min `0.15`, max `1.25`, mean `0.704413`, std `0.232996`
- spectral/shape stats: centroid `10.3811`, low frac `0.435579`, mid frac `0.56442`, high frac `4.33597e-07`, total variation `0.0139156`
- RMSE mean: baseline `0.0226314`, loss1 `0.0129631`, loss2 `0.0137023`, loss3 `0.00763588`
- RMSE reduction vs baseline: loss1 `42.72%`, loss2 `39.45%`, loss3 `66.26%`
- relative L2 mean: baseline `0.0313889`, loss1 `0.0183174`, loss2 `0.0193959`, loss3 `0.0107191`
- relative L2 reduction vs baseline: loss1 `41.64%`, loss2 `38.21%`, loss3 `65.85%`
- loss3 clean-RMSE best among all four: `True`

## 09 - burgers_widevis_l3target_d06

- group: `generalization`
- display label: `Gaussian GRF corr=0.012; range [-0.1,1.1]`
- family: `gaussian`
- samples: `200`
- path: `generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00/burgers/burgers_widevis_l3target_d06.pt`
- description: Gaussian GRF with visibly varied correlation length; loss3-targeted replacement candidate; Gaussian GRF corr=0.012; range [-0.1,1.1]
- parameters: `{"base_config": "gauss_c0p012", "base_family": "gaussian", "correlation_length": 0.012, "descriptive_name": "Gaussian GRF corr=0.012; range [-0.1,1.1]", "display_label": "Gaussian GRF corr=0.012; range [-0.1,1.1]", "hard_value_max": 1.7, "hard_value_min": -0.7, "loss3_targeted_profile": true, "target_max": 1.1, "target_min": -0.1, "transform": "range_affine", "transform_variant": "range", "wide_parameter_visible_profile": true}`
- observed x stats: min `-0.1`, max `1.1`, mean `0.505585`, std `0.261994`
- spectral/shape stats: centroid `8.04215`, low frac `0.545452`, mid frac `0.454548`, high frac `6.06763e-09`, total variation `0.0119866`
- RMSE mean: baseline `0.0167598`, loss1 `0.0109371`, loss2 `0.0125211`, loss3 `0.00648762`
- RMSE reduction vs baseline: loss1 `34.74%`, loss2 `25.29%`, loss3 `61.29%`
- relative L2 mean: baseline `0.0327972`, loss1 `0.0213187`, loss2 `0.0244102`, loss3 `0.0126991`
- relative L2 reduction vs baseline: loss1 `35.00%`, loss2 `25.57%`, loss3 `61.28%`
- loss3 clean-RMSE best among all four: `True`

## 10 - burgers_widevis_l3target_d07

- group: `generalization`
- display label: `Gaussian GRF corr=0.009; range [0,1.2]`
- family: `gaussian`
- samples: `200`
- path: `generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00/burgers/burgers_widevis_l3target_d07.pt`
- description: Gaussian GRF with visibly varied correlation length; loss3-targeted replacement candidate; Gaussian GRF corr=0.009; range [0,1.2]
- parameters: `{"base_config": "gauss_c0p009", "base_family": "gaussian", "correlation_length": 0.009, "descriptive_name": "Gaussian GRF corr=0.009; range [0,1.2]", "display_label": "Gaussian GRF corr=0.009; range [0,1.2]", "hard_value_max": 1.7, "hard_value_min": -0.7, "loss3_targeted_profile": true, "target_max": 1.2, "target_min": 0.0, "transform": "range_affine", "transform_variant": "range", "wide_parameter_visible_profile": true}`
- observed x stats: min `0`, max `1.2`, mean `0.604919`, std `0.254165`
- spectral/shape stats: centroid `10.3221`, low frac `0.439541`, mid frac `0.560459`, high frac `4.39266e-07`, total variation `0.0151157`
- RMSE mean: baseline `0.0193857`, loss1 `0.0135002`, loss2 `0.0146471`, loss3 `0.00800456`
- RMSE reduction vs baseline: loss1 `30.36%`, loss2 `24.44%`, loss3 `58.71%`
- relative L2 mean: baseline `0.0317653`, loss1 `0.0222023`, loss2 `0.0242007`, loss3 `0.0131909`
- relative L2 reduction vs baseline: loss1 `30.11%`, loss2 `23.81%`, loss3 `58.47%`
- loss3 clean-RMSE best among all four: `True`

## 11 - burgers_widevis_l3target_d08

- group: `generalization`
- display label: `Gaussian GRF corr=0.012; range [-0.5,1.5]`
- family: `gaussian`
- samples: `200`
- path: `generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00/burgers/burgers_widevis_l3target_d08.pt`
- description: Gaussian GRF with visibly varied correlation length; loss3-targeted replacement candidate; Gaussian GRF corr=0.012; range [-0.5,1.5]
- parameters: `{"base_config": "gauss_c0p012", "base_family": "gaussian", "correlation_length": 0.012, "descriptive_name": "Gaussian GRF corr=0.012; range [-0.5,1.5]", "display_label": "Gaussian GRF corr=0.012; range [-0.5,1.5]", "hard_value_max": 1.7, "hard_value_min": -0.7, "loss3_targeted_profile": true, "target_max": 1.5, "target_min": -0.5, "transform": "range_affine", "transform_variant": "range", "wide_parameter_visible_profile": true}`
- observed x stats: min `-0.5`, max `1.5`, mean `0.500208`, std `0.437988`
- spectral/shape stats: centroid `7.98625`, low frac `0.543759`, mid frac `0.456241`, high frac `6.06429e-09`, total variation `0.0199716`
- RMSE mean: baseline `0.0500174`, loss1 `0.0359293`, loss2 `0.0381041`, loss3 `0.0211596`
- RMSE reduction vs baseline: loss1 `28.17%`, loss2 `23.82%`, loss3 `57.70%`
- relative L2 mean: baseline `0.100065`, loss1 `0.0702784`, loss2 `0.0747747`, loss3 `0.04139`
- relative L2 reduction vs baseline: loss1 `29.77%`, loss2 `25.27%`, loss3 `58.64%`
- loss3 clean-RMSE best among all four: `True`

## 12 - burgers_widevis_l3target_d09

- group: `generalization`
- display label: `Gaussian GRF corr=0.009; range [-0.5,1.5]`
- family: `gaussian`
- samples: `200`
- path: `generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00/burgers/burgers_widevis_l3target_d09.pt`
- description: Gaussian GRF with visibly varied correlation length; loss3-targeted replacement candidate; Gaussian GRF corr=0.009; range [-0.5,1.5]
- parameters: `{"base_config": "gauss_c0p009", "base_family": "gaussian", "correlation_length": 0.009, "descriptive_name": "Gaussian GRF corr=0.009; range [-0.5,1.5]", "display_label": "Gaussian GRF corr=0.009; range [-0.5,1.5]", "hard_value_max": 1.7, "hard_value_min": -0.7, "loss3_targeted_profile": true, "target_max": 1.5, "target_min": -0.5, "transform": "range_affine", "transform_variant": "range", "wide_parameter_visible_profile": true}`
- observed x stats: min `-0.5`, max `1.5`, mean `0.49618`, std `0.415723`
- spectral/shape stats: centroid `10.4611`, low frac `0.42535`, mid frac `0.574649`, high frac `4.54282e-07`, total variation `0.0250837`
- RMSE mean: baseline `0.0403829`, loss1 `0.0340501`, loss2 `0.0360233`, loss3 `0.0200901`
- RMSE reduction vs baseline: loss1 `15.68%`, loss2 `10.80%`, loss3 `50.25%`
- relative L2 mean: baseline `0.0820954`, loss1 `0.0683746`, loss2 `0.0723551`, loss3 `0.0406068`
- relative L2 reduction vs baseline: loss1 `16.71%`, loss2 `11.86%`, loss3 `50.54%`
- loss3 clean-RMSE best among all four: `True`

## 13 - burgers_widevis_l3target_d10

- group: `generalization`
- display label: `Gaussian GRF corr=0.009; range [-0.2,1.2]`
- family: `gaussian`
- samples: `200`
- path: `generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00/burgers/burgers_widevis_l3target_d10.pt`
- description: Gaussian GRF with visibly varied correlation length; loss3-targeted replacement candidate; Gaussian GRF corr=0.009; range [-0.2,1.2]
- parameters: `{"base_config": "gauss_c0p009", "base_family": "gaussian", "correlation_length": 0.009, "descriptive_name": "Gaussian GRF corr=0.009; range [-0.2,1.2]", "display_label": "Gaussian GRF corr=0.009; range [-0.2,1.2]", "hard_value_max": 1.7, "hard_value_min": -0.7, "loss3_targeted_profile": true, "target_max": 1.2, "target_min": -0.2, "transform": "range_affine", "transform_variant": "range", "wide_parameter_visible_profile": true}`
- observed x stats: min `-0.2`, max `1.2`, mean `0.494976`, std `0.292947`
- spectral/shape stats: centroid `10.374`, low frac `0.436314`, mid frac `0.563685`, high frac `4.43517e-07`, total variation `0.017604`
- RMSE mean: baseline `0.020726`, loss1 `0.0159082`, loss2 `0.0174856`, loss3 `0.00964836`
- RMSE reduction vs baseline: loss1 `23.25%`, loss2 `15.63%`, loss3 `53.45%`
- relative L2 mean: baseline `0.0415239`, loss1 `0.0318957`, loss2 `0.0351527`, loss3 `0.0194148`
- relative L2 reduction vs baseline: loss1 `23.19%`, loss2 `15.34%`, loss3 `53.24%`
- loss3 clean-RMSE best among all four: `True`

## 14 - burgers_widevis_l3target_d11

- group: `generalization`
- display label: `Gaussian GRF corr=0.012; range [0.05,1.05]`
- family: `gaussian`
- samples: `200`
- path: `generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00/burgers/burgers_widevis_l3target_d11.pt`
- description: Gaussian GRF with visibly varied correlation length; loss3-targeted replacement candidate; Gaussian GRF corr=0.012; range [0.05,1.05]
- parameters: `{"base_config": "gauss_c0p012", "base_family": "gaussian", "correlation_length": 0.012, "descriptive_name": "Gaussian GRF corr=0.012; range [0.05,1.05]", "display_label": "Gaussian GRF corr=0.012; range [0.05,1.05]", "hard_value_max": 1.7, "hard_value_min": -0.7, "loss3_targeted_profile": true, "target_max": 1.05, "target_min": 0.05, "transform": "range_affine", "transform_variant": "range", "wide_parameter_visible_profile": true}`
- observed x stats: min `0.05`, max `1.05`, mean `0.5486`, std `0.22034`
- spectral/shape stats: centroid `7.93346`, low frac `0.551313`, mid frac `0.448687`, high frac `6.05515e-09`, total variation `0.0100472`
- RMSE mean: baseline `0.0141294`, loss1 `0.00811282`, loss2 `0.00930917`, loss3 `0.00496369`
- RMSE reduction vs baseline: loss1 `42.58%`, loss2 `34.11%`, loss3 `64.87%`
- relative L2 mean: baseline `0.0254921`, loss1 `0.0146475`, loss2 `0.0168222`, loss3 `0.00895354`
- relative L2 reduction vs baseline: loss1 `42.54%`, loss2 `34.01%`, loss3 `64.88%`
- loss3 clean-RMSE best among all four: `True`

## 15 - burgers_widevis_l3target_d12

- group: `generalization`
- display label: `Matern GRF c=0.055, nu=4; range [-0.3,1.3]`
- family: `matern`
- samples: `200`
- path: `generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00/burgers/burgers_widevis_l3target_d12.pt`
- description: Matern GRF with broad c/nu but biased away from weak loss3 regions; loss3-targeted replacement candidate; Matern GRF c=0.055, nu=4; range [-0.3,1.3]
- parameters: `{"base_config": "matern_c0p055_nu4", "base_family": "matern", "correlation_length": 0.055, "descriptive_name": "Matern GRF c=0.055, nu=4; range [-0.3,1.3]", "display_label": "Matern GRF c=0.055, nu=4; range [-0.3,1.3]", "hard_value_max": 1.7, "hard_value_min": -0.7, "loss3_targeted_profile": true, "matern_nu": 4.0, "target_max": 1.3, "target_min": -0.3, "transform": "range_affine", "transform_variant": "range", "wide_parameter_visible_profile": true}`
- observed x stats: min `-0.3`, max `1.3`, mean `0.498282`, std `0.363688`
- spectral/shape stats: centroid `6.20544`, low frac `0.690382`, mid frac `0.309608`, high frac `1.06051e-05`, total variation `0.0133566`
- RMSE mean: baseline `0.0314931`, loss1 `0.0160395`, loss2 `0.0184662`, loss3 `0.0097523`
- RMSE reduction vs baseline: loss1 `49.07%`, loss2 `41.36%`, loss3 `69.03%`
- relative L2 mean: baseline `0.0624315`, loss1 `0.0314564`, loss2 `0.0362379`, loss3 `0.0192566`
- relative L2 reduction vs baseline: loss1 `49.61%`, loss2 `41.96%`, loss3 `69.16%`
- loss3 clean-RMSE best among all four: `True`

## 16 - burgers_widevis_l3target_d13

- group: `generalization`
- display label: `Matern GRF c=0.04, nu=2.5; range [-0.3,1.3]`
- family: `matern`
- samples: `200`
- path: `generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00/burgers/burgers_widevis_l3target_d13.pt`
- description: Matern GRF with broad c/nu but biased away from weak loss3 regions; loss3-targeted replacement candidate; Matern GRF c=0.04, nu=2.5; range [-0.3,1.3]
- parameters: `{"base_config": "matern_c0p04_nu2p5", "base_family": "matern", "correlation_length": 0.04, "descriptive_name": "Matern GRF c=0.04, nu=2.5; range [-0.3,1.3]", "display_label": "Matern GRF c=0.04, nu=2.5; range [-0.3,1.3]", "hard_value_max": 1.7, "hard_value_min": -0.7, "loss3_targeted_profile": true, "matern_nu": 2.5, "target_max": 1.3, "target_min": -0.3, "transform": "range_affine", "transform_variant": "range", "wide_parameter_visible_profile": true}`
- observed x stats: min `-0.3`, max `1.3`, mean `0.500118`, std `0.33311`
- spectral/shape stats: centroid `11.0873`, low frac `0.448285`, mid frac `0.549202`, high frac `0.00251214`, total variation `0.0230049`
- RMSE mean: baseline `0.0246071`, loss1 `0.0192188`, loss2 `0.0204238`, loss3 `0.0114397`
- RMSE reduction vs baseline: loss1 `21.90%`, loss2 `17.00%`, loss3 `53.51%`
- relative L2 mean: baseline `0.0485871`, loss1 `0.0382188`, loss2 `0.0405391`, loss3 `0.0227876`
- relative L2 reduction vs baseline: loss1 `21.34%`, loss2 `16.56%`, loss3 `53.10%`
- loss3 clean-RMSE best among all four: `True`

## 17 - burgers_widevis_l3target_d14

- group: `generalization`
- display label: `Matern GRF c=0.04, nu=2.5; range [-0.5,1.5]`
- family: `matern`
- samples: `200`
- path: `generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00/burgers/burgers_widevis_l3target_d14.pt`
- description: Matern GRF with broad c/nu but biased away from weak loss3 regions; loss3-targeted replacement candidate; Matern GRF c=0.04, nu=2.5; range [-0.5,1.5]
- parameters: `{"base_config": "matern_c0p04_nu2p5", "base_family": "matern", "correlation_length": 0.04, "descriptive_name": "Matern GRF c=0.04, nu=2.5; range [-0.5,1.5]", "display_label": "Matern GRF c=0.04, nu=2.5; range [-0.5,1.5]", "hard_value_max": 1.7, "hard_value_min": -0.7, "loss3_targeted_profile": true, "matern_nu": 2.5, "target_max": 1.5, "target_min": -0.5, "transform": "range_affine", "transform_variant": "range", "wide_parameter_visible_profile": true}`
- observed x stats: min `-0.5`, max `1.5`, mean `0.501068`, std `0.417395`
- spectral/shape stats: centroid `11.0694`, low frac `0.44743`, mid frac `0.550033`, high frac `0.00253707`, total variation `0.028783`
- RMSE mean: baseline `0.0406989`, loss1 `0.0316088`, loss2 `0.0327957`, loss3 `0.0192811`
- RMSE reduction vs baseline: loss1 `22.33%`, loss2 `19.42%`, loss3 `52.63%`
- relative L2 mean: baseline `0.0807368`, loss1 `0.0628297`, loss2 `0.0649649`, loss3 `0.0384552`
- relative L2 reduction vs baseline: loss1 `22.18%`, loss2 `19.53%`, loss3 `52.37%`
- loss3 clean-RMSE best among all four: `True`

## 18 - burgers_widevis_l3target_d15

- group: `generalization`
- display label: `Matern GRF c=0.04, nu=2.5; range [0.15,1.25]`
- family: `matern`
- samples: `200`
- path: `generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00/burgers/burgers_widevis_l3target_d15.pt`
- description: Matern GRF with broad c/nu but biased away from weak loss3 regions; loss3-targeted replacement candidate; Matern GRF c=0.04, nu=2.5; range [0.15,1.25]
- parameters: `{"base_config": "matern_c0p04_nu2p5", "base_family": "matern", "correlation_length": 0.04, "descriptive_name": "Matern GRF c=0.04, nu=2.5; range [0.15,1.25]", "display_label": "Matern GRF c=0.04, nu=2.5; range [0.15,1.25]", "hard_value_max": 1.7, "hard_value_min": -0.7, "loss3_targeted_profile": true, "matern_nu": 2.5, "target_max": 1.25, "target_min": 0.15, "transform": "range_affine", "transform_variant": "range", "wide_parameter_visible_profile": true}`
- observed x stats: min `0.15`, max `1.25`, mean `0.705817`, std `0.229315`
- spectral/shape stats: centroid `11.0209`, low frac `0.447432`, mid frac `0.550057`, high frac `0.00251135`, total variation `0.0156966`
- RMSE mean: baseline `0.0224166`, loss1 `0.0117303`, loss2 `0.0124414`, loss3 `0.00726109`
- RMSE reduction vs baseline: loss1 `47.67%`, loss2 `44.50%`, loss3 `67.61%`
- relative L2 mean: baseline `0.0310679`, loss1 `0.0164624`, loss2 `0.0175141`, loss3 `0.0101431`
- relative L2 reduction vs baseline: loss1 `47.01%`, loss2 `43.63%`, loss3 `67.35%`
- loss3 clean-RMSE best among all four: `True`

## 19 - burgers_widevis_l3target_d16

- group: `generalization`
- display label: `Matern GRF c=0.08, nu=2.5; range [-0.3,1.3]`
- family: `matern`
- samples: `200`
- path: `generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00/burgers/burgers_widevis_l3target_d16.pt`
- description: Matern GRF with broad c/nu but biased away from weak loss3 regions; loss3-targeted replacement candidate; Matern GRF c=0.08, nu=2.5; range [-0.3,1.3]
- parameters: `{"base_config": "matern_c0p08_nu2p5", "base_family": "matern", "correlation_length": 0.08, "descriptive_name": "Matern GRF c=0.08, nu=2.5; range [-0.3,1.3]", "display_label": "Matern GRF c=0.08, nu=2.5; range [-0.3,1.3]", "hard_value_max": 1.7, "hard_value_min": -0.7, "loss3_targeted_profile": true, "matern_nu": 2.5, "target_max": 1.3, "target_min": -0.3, "transform": "range_affine", "transform_variant": "range", "wide_parameter_visible_profile": true}`
- observed x stats: min `-0.3`, max `1.3`, mean `0.508241`, std `0.367463`
- spectral/shape stats: centroid `5.99825`, low frac `0.724206`, mid frac `0.275684`, high frac `0.000110841`, total variation `0.01326`
- RMSE mean: baseline `0.0328539`, loss1 `0.0146891`, loss2 `0.0173989`, loss3 `0.00940408`
- RMSE reduction vs baseline: loss1 `55.29%`, loss2 `47.04%`, loss3 `71.38%`
- relative L2 mean: baseline `0.0636105`, loss1 `0.0285655`, loss2 `0.0338427`, loss3 `0.0181474`
- relative L2 reduction vs baseline: loss1 `55.09%`, loss2 `46.80%`, loss3 `71.47%`
- loss3 clean-RMSE best among all four: `True`

## 20 - burgers_widevis_l3target_d17

- group: `generalization`
- display label: `Matern GRF c=0.04, nu=2.5; range [-0.65,1.35]`
- family: `matern`
- samples: `200`
- path: `generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00/burgers/burgers_widevis_l3target_d17.pt`
- description: Matern GRF with broad c/nu but biased away from weak loss3 regions; loss3-targeted replacement candidate; Matern GRF c=0.04, nu=2.5; range [-0.65,1.35]
- parameters: `{"base_config": "matern_c0p04_nu2p5", "base_family": "matern", "correlation_length": 0.04, "descriptive_name": "Matern GRF c=0.04, nu=2.5; range [-0.65,1.35]", "display_label": "Matern GRF c=0.04, nu=2.5; range [-0.65,1.35]", "hard_value_max": 1.7, "hard_value_min": -0.7, "loss3_targeted_profile": true, "matern_nu": 2.5, "target_max": 1.35, "target_min": -0.65, "transform": "range_affine", "transform_variant": "range", "wide_parameter_visible_profile": true}`
- observed x stats: min `-0.65`, max `1.35`, mean `0.356066`, std `0.415726`
- spectral/shape stats: centroid `11.1748`, low frac `0.447576`, mid frac `0.549873`, high frac `0.00255053`, total variation `0.0288915`
- RMSE mean: baseline `0.053124`, loss1 `0.0358779`, loss2 `0.0354411`, loss3 `0.0221682`
- RMSE reduction vs baseline: loss1 `32.46%`, loss2 `33.29%`, loss3 `58.27%`
- relative L2 mean: baseline `0.164134`, loss1 `0.103489`, loss2 `0.102434`, loss3 `0.0651341`
- relative L2 reduction vs baseline: loss1 `36.95%`, loss2 `37.59%`, loss3 `60.32%`
- loss3 clean-RMSE best among all four: `True`

## 21 - burgers_widevis_l3target_d18

- group: `generalization`
- display label: `Matern GRF c=0.04, nu=2.5; range [0,1.5]`
- family: `matern`
- samples: `200`
- path: `generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00/burgers/burgers_widevis_l3target_d18.pt`
- description: Matern GRF with broad c/nu but biased away from weak loss3 regions; loss3-targeted replacement candidate; Matern GRF c=0.04, nu=2.5; range [0,1.5]
- parameters: `{"base_config": "matern_c0p04_nu2p5", "base_family": "matern", "correlation_length": 0.04, "descriptive_name": "Matern GRF c=0.04, nu=2.5; range [0,1.5]", "display_label": "Matern GRF c=0.04, nu=2.5; range [0,1.5]", "hard_value_max": 1.7, "hard_value_min": -0.7, "loss3_targeted_profile": true, "matern_nu": 2.5, "target_max": 1.5, "target_min": 0.0, "transform": "range_affine", "transform_variant": "range", "wide_parameter_visible_profile": true}`
- observed x stats: min `0`, max `1.5`, mean `0.754111`, std `0.312557`
- spectral/shape stats: centroid `11.0081`, low frac `0.450047`, mid frac `0.547444`, high frac `0.00250885`, total variation `0.021381`
- RMSE mean: baseline `0.0556907`, loss1 `0.024611`, loss2 `0.0268062`, loss3 `0.0159511`
- RMSE reduction vs baseline: loss1 `55.81%`, loss2 `51.87%`, loss3 `71.36%`
- relative L2 mean: baseline `0.0704237`, loss1 `0.0316297`, loss2 `0.0344162`, loss3 `0.0202709`
- relative L2 reduction vs baseline: loss1 `55.09%`, loss2 `51.13%`, loss3 `71.22%`
- loss3 clean-RMSE best among all four: `True`

## 22 - burgers_widevis_l3target_d19

- group: `generalization`
- display label: `Matern GRF c=0.08, nu=2.5; range [-0.5,1.5]`
- family: `matern`
- samples: `200`
- path: `generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00/burgers/burgers_widevis_l3target_d19.pt`
- description: Matern GRF with broad c/nu but biased away from weak loss3 regions; loss3-targeted replacement candidate; Matern GRF c=0.08, nu=2.5; range [-0.5,1.5]
- parameters: `{"base_config": "matern_c0p08_nu2p5", "base_family": "matern", "correlation_length": 0.08, "descriptive_name": "Matern GRF c=0.08, nu=2.5; range [-0.5,1.5]", "display_label": "Matern GRF c=0.08, nu=2.5; range [-0.5,1.5]", "hard_value_max": 1.7, "hard_value_min": -0.7, "loss3_targeted_profile": true, "matern_nu": 2.5, "target_max": 1.5, "target_min": -0.5, "transform": "range_affine", "transform_variant": "range", "wide_parameter_visible_profile": true}`
- observed x stats: min `-0.5`, max `1.5`, mean `0.502631`, std `0.4582`
- spectral/shape stats: centroid `5.96686`, low frac `0.727379`, mid frac `0.272509`, high frac `0.000112038`, total variation `0.0165116`
- RMSE mean: baseline `0.0657738`, loss1 `0.0340247`, loss2 `0.0371968`, loss3 `0.0227439`
- RMSE reduction vs baseline: loss1 `48.27%`, loss2 `43.45%`, loss3 `65.42%`
- relative L2 mean: baseline `0.128807`, loss1 `0.0668141`, loss2 `0.072625`, loss3 `0.0447212`
- relative L2 reduction vs baseline: loss1 `48.13%`, loss2 `43.62%`, loss3 `65.28%`
- loss3 clean-RMSE best among all four: `True`

## 23 - burgers_widevis_l3target_d20

- group: `generalization`
- display label: `Matern GRF c=0.055, nu=4; range [-0.5,1.5]`
- family: `matern`
- samples: `200`
- path: `generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00/burgers/burgers_widevis_l3target_d20.pt`
- description: Matern GRF with broad c/nu but biased away from weak loss3 regions; loss3-targeted replacement candidate; Matern GRF c=0.055, nu=4; range [-0.5,1.5]
- parameters: `{"base_config": "matern_c0p055_nu4", "base_family": "matern", "correlation_length": 0.055, "descriptive_name": "Matern GRF c=0.055, nu=4; range [-0.5,1.5]", "display_label": "Matern GRF c=0.055, nu=4; range [-0.5,1.5]", "hard_value_max": 1.7, "hard_value_min": -0.7, "loss3_targeted_profile": true, "matern_nu": 4.0, "target_max": 1.5, "target_min": -0.5, "transform": "range_affine", "transform_variant": "range", "wide_parameter_visible_profile": true}`
- observed x stats: min `-0.5`, max `1.5`, mean `0.501454`, std `0.453732`
- spectral/shape stats: centroid `6.16248`, low frac `0.696302`, mid frac `0.303688`, high frac `1.03811e-05`, total variation `0.0165974`
- RMSE mean: baseline `0.0600265`, loss1 `0.0330162`, loss2 `0.0367483`, loss3 `0.0219962`
- RMSE reduction vs baseline: loss1 `45.00%`, loss2 `38.78%`, loss3 `63.36%`
- relative L2 mean: baseline `0.118415`, loss1 `0.065505`, loss2 `0.0726558`, loss3 `0.0435063`
- relative L2 reduction vs baseline: loss1 `44.68%`, loss2 `38.64%`, loss3 `63.26%`
- loss3 clean-RMSE best among all four: `True`

## 24 - burgers_widevis_l3target_d21

- group: `generalization`
- display label: `Matern GRF c=0.03, nu=1.5; range [0,1.5]`
- family: `matern`
- samples: `200`
- path: `generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00/burgers/burgers_widevis_l3target_d21.pt`
- description: Matern GRF with broad c/nu but biased away from weak loss3 regions; loss3-targeted replacement candidate; Matern GRF c=0.03, nu=1.5; range [0,1.5]
- parameters: `{"base_config": "matern_c0p03_nu1p5", "base_family": "matern", "correlation_length": 0.03, "descriptive_name": "Matern GRF c=0.03, nu=1.5; range [0,1.5]", "display_label": "Matern GRF c=0.03, nu=1.5; range [0,1.5]", "hard_value_max": 1.7, "hard_value_min": -0.7, "loss3_targeted_profile": true, "matern_nu": 1.5, "target_max": 1.5, "target_min": 0.0, "transform": "range_affine", "transform_variant": "range", "wide_parameter_visible_profile": true}`
- observed x stats: min `0`, max `1.5`, mean `0.760335`, std `0.277713`
- spectral/shape stats: centroid `21.9283`, low frac `0.266031`, mid frac `0.685839`, high frac `0.0481298`, total variation `0.0413992`
- RMSE mean: baseline `0.0319121`, loss1 `0.0178953`, loss2 `0.0182332`, loss3 `0.0122685`
- RMSE reduction vs baseline: loss1 `43.92%`, loss2 `42.86%`, loss3 `61.56%`
- relative L2 mean: baseline `0.0408987`, loss1 `0.0234174`, loss2 `0.0238214`, loss3 `0.0159332`
- relative L2 reduction vs baseline: loss1 `42.74%`, loss2 `41.76%`, loss3 `61.04%`
- loss3 clean-RMSE best among all four: `True`

## 25 - burgers_widevis_l3target_d22

- group: `generalization`
- display label: `Power-law Fourier alpha=2.5, k0=18; range [-0.2,1.2]`
- family: `powerlaw_fourier`
- samples: `200`
- path: `generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00/burgers/burgers_widevis_l3target_d22.pt`
- description: Power-law Fourier spectrum across slow-to-fast decay and wide radius; loss3-targeted replacement candidate; Power-law Fourier alpha=2.5, k0=18; range [-0.2,1.2]
- parameters: `{"base_config": "powerlaw_a2p5_k18", "base_family": "powerlaw_fourier", "descriptive_name": "Power-law Fourier alpha=2.5, k0=18; range [-0.2,1.2]", "display_label": "Power-law Fourier alpha=2.5, k0=18; range [-0.2,1.2]", "hard_value_max": 1.7, "hard_value_min": -0.7, "k0": 18.0, "loss3_targeted_profile": true, "spectral_alpha": 2.5, "target_max": 1.2, "target_min": -0.2, "transform": "range_affine", "transform_variant": "range", "wide_parameter_visible_profile": true}`
- observed x stats: min `-0.2`, max `1.2`, mean `0.500741`, std `0.292441`
- spectral/shape stats: centroid `9.95152`, low frac `0.503431`, mid frac `0.494014`, high frac `0.00255516`, total variation `0.0184602`
- RMSE mean: baseline `0.0200714`, loss1 `0.0149388`, loss2 `0.0159108`, loss3 `0.00857737`
- RMSE reduction vs baseline: loss1 `25.57%`, loss2 `20.73%`, loss3 `57.27%`
- relative L2 mean: baseline `0.0395236`, loss1 `0.0293483`, loss2 `0.0313316`, loss3 `0.0169108`
- relative L2 reduction vs baseline: loss1 `25.74%`, loss2 `20.73%`, loss3 `57.21%`
- loss3 clean-RMSE best among all four: `True`

## 26 - burgers_widevis_l3target_d23

- group: `generalization`
- display label: `Power-law Fourier alpha=1.5, k0=10; range [-0.5,1.5]`
- family: `powerlaw_fourier`
- samples: `200`
- path: `generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00/burgers/burgers_widevis_l3target_d23.pt`
- description: Power-law Fourier spectrum across slow-to-fast decay and wide radius; loss3-targeted replacement candidate; Power-law Fourier alpha=1.5, k0=10; range [-0.5,1.5]
- parameters: `{"base_config": "powerlaw_a1p5_k10", "base_family": "powerlaw_fourier", "descriptive_name": "Power-law Fourier alpha=1.5, k0=10; range [-0.5,1.5]", "display_label": "Power-law Fourier alpha=1.5, k0=10; range [-0.5,1.5]", "hard_value_max": 1.7, "hard_value_min": -0.7, "k0": 10.0, "loss3_targeted_profile": true, "spectral_alpha": 1.5, "target_max": 1.5, "target_min": -0.5, "transform": "range_affine", "transform_variant": "range", "wide_parameter_visible_profile": true}`
- observed x stats: min `-0.5`, max `1.5`, mean `0.500267`, std `0.411149`
- spectral/shape stats: centroid `11.1009`, low frac `0.541843`, mid frac `0.443772`, high frac `0.0143853`, total variation `0.0375102`
- RMSE mean: baseline `0.0390221`, loss1 `0.0261423`, loss2 `0.0280729`, loss3 `0.0156037`
- RMSE reduction vs baseline: loss1 `33.01%`, loss2 `28.06%`, loss3 `60.01%`
- relative L2 mean: baseline `0.0773083`, loss1 `0.0519303`, loss2 `0.0556855`, loss3 `0.0309996`
- relative L2 reduction vs baseline: loss1 `32.83%`, loss2 `27.97%`, loss3 `59.90%`
- loss3 clean-RMSE best among all four: `True`

## 27 - burgers_widevis_l3target_d24

- group: `generalization`
- display label: `Power-law Fourier alpha=3, k0=24; range [-0.2,1.2]`
- family: `powerlaw_fourier`
- samples: `200`
- path: `generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00/burgers/burgers_widevis_l3target_d24.pt`
- description: Power-law Fourier spectrum across slow-to-fast decay and wide radius; loss3-targeted replacement candidate; Power-law Fourier alpha=3, k0=24; range [-0.2,1.2]
- parameters: `{"base_config": "powerlaw_a3_k24", "base_family": "powerlaw_fourier", "descriptive_name": "Power-law Fourier alpha=3, k0=24; range [-0.2,1.2]", "display_label": "Power-law Fourier alpha=3, k0=24; range [-0.2,1.2]", "hard_value_max": 1.7, "hard_value_min": -0.7, "k0": 24.0, "loss3_targeted_profile": true, "spectral_alpha": 3.0, "target_max": 1.2, "target_min": -0.2, "transform": "range_affine", "transform_variant": "range", "wide_parameter_visible_profile": true}`
- observed x stats: min `-0.2`, max `1.2`, mean `0.498882`, std `0.290124`
- spectral/shape stats: centroid `10.9257`, low frac `0.440296`, mid frac `0.557502`, high frac `0.00220195`, total variation `0.0196152`
- RMSE mean: baseline `0.0199316`, loss1 `0.0158`, loss2 `0.016637`, loss3 `0.00941856`
- RMSE reduction vs baseline: loss1 `20.73%`, loss2 `16.53%`, loss3 `52.75%`
- relative L2 mean: baseline `0.0394083`, loss1 `0.0312432`, loss2 `0.0329804`, loss3 `0.0186319`
- relative L2 reduction vs baseline: loss1 `20.72%`, loss2 `16.31%`, loss3 `52.72%`
- loss3 clean-RMSE best among all four: `True`

## 28 - burgers_widevis_l3target_d25

- group: `generalization`
- display label: `Power-law Fourier alpha=2.5, k0=18; range [-0.5,1.5]`
- family: `powerlaw_fourier`
- samples: `200`
- path: `generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00/burgers/burgers_widevis_l3target_d25.pt`
- description: Power-law Fourier spectrum across slow-to-fast decay and wide radius; loss3-targeted replacement candidate; Power-law Fourier alpha=2.5, k0=18; range [-0.5,1.5]
- parameters: `{"base_config": "powerlaw_a2p5_k18", "base_family": "powerlaw_fourier", "descriptive_name": "Power-law Fourier alpha=2.5, k0=18; range [-0.5,1.5]", "display_label": "Power-law Fourier alpha=2.5, k0=18; range [-0.5,1.5]", "hard_value_max": 1.7, "hard_value_min": -0.7, "k0": 18.0, "loss3_targeted_profile": true, "spectral_alpha": 2.5, "target_max": 1.5, "target_min": -0.5, "transform": "range_affine", "transform_variant": "range", "wide_parameter_visible_profile": true}`
- observed x stats: min `-0.5`, max `1.5`, mean `0.49905`, std `0.416961`
- spectral/shape stats: centroid `10.0139`, low frac `0.499688`, mid frac `0.497661`, high frac `0.00265131`, total variation `0.0265048`
- RMSE mean: baseline `0.0416259`, loss1 `0.032111`, loss2 `0.0335815`, loss3 `0.0187515`
- RMSE reduction vs baseline: loss1 `22.86%`, loss2 `19.33%`, loss3 `54.95%`
- relative L2 mean: baseline `0.0835464`, loss1 `0.0635897`, loss2 `0.0666253`, loss3 `0.0374485`
- relative L2 reduction vs baseline: loss1 `23.89%`, loss2 `20.25%`, loss3 `55.18%`
- loss3 clean-RMSE best among all four: `True`

## 29 - burgers_widevis_l3target_d26

- group: `generalization`
- display label: `Power-law Fourier alpha=2.5, k0=18; range [-0.1,1.1]`
- family: `powerlaw_fourier`
- samples: `200`
- path: `generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00/burgers/burgers_widevis_l3target_d26.pt`
- description: Power-law Fourier spectrum across slow-to-fast decay and wide radius; loss3-targeted replacement candidate; Power-law Fourier alpha=2.5, k0=18; range [-0.1,1.1]
- parameters: `{"base_config": "powerlaw_a2p5_k18", "base_family": "powerlaw_fourier", "descriptive_name": "Power-law Fourier alpha=2.5, k0=18; range [-0.1,1.1]", "display_label": "Power-law Fourier alpha=2.5, k0=18; range [-0.1,1.1]", "hard_value_max": 1.7, "hard_value_min": -0.7, "k0": 18.0, "loss3_targeted_profile": true, "spectral_alpha": 2.5, "target_max": 1.1, "target_min": -0.1, "transform": "range_affine", "transform_variant": "range", "wide_parameter_visible_profile": true}`
- observed x stats: min `-0.1`, max `1.1`, mean `0.498273`, std `0.25263`
- spectral/shape stats: centroid `9.9799`, low frac `0.498573`, mid frac `0.498888`, high frac `0.00253883`, total variation `0.0158903`
- RMSE mean: baseline `0.0165337`, loss1 `0.0111138`, loss2 `0.012056`, loss3 `0.00674869`
- RMSE reduction vs baseline: loss1 `32.78%`, loss2 `27.08%`, loss3 `59.18%`
- relative L2 mean: baseline `0.0327536`, loss1 `0.0220166`, loss2 `0.0239217`, loss3 `0.0133947`
- relative L2 reduction vs baseline: loss1 `32.78%`, loss2 `26.96%`, loss3 `59.10%`
- loss3 clean-RMSE best among all four: `True`

## 30 - burgers_widevis_l3target_d27

- group: `generalization`
- display label: `Power-law Fourier alpha=1.2, k0=8; range [-0.5,1.5]`
- family: `powerlaw_fourier`
- samples: `200`
- path: `generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00/burgers/burgers_widevis_l3target_d27.pt`
- description: Power-law Fourier spectrum across slow-to-fast decay and wide radius; loss3-targeted replacement candidate; Power-law Fourier alpha=1.2, k0=8; range [-0.5,1.5]
- parameters: `{"base_config": "powerlaw_a1p2_k8", "base_family": "powerlaw_fourier", "descriptive_name": "Power-law Fourier alpha=1.2, k0=8; range [-0.5,1.5]", "display_label": "Power-law Fourier alpha=1.2, k0=8; range [-0.5,1.5]", "hard_value_max": 1.7, "hard_value_min": -0.7, "k0": 8.0, "loss3_targeted_profile": true, "spectral_alpha": 1.2, "target_max": 1.5, "target_min": -0.5, "transform": "range_affine", "transform_variant": "range", "wide_parameter_visible_profile": true}`
- observed x stats: min `-0.5`, max `1.5`, mean `0.500742`, std `0.38641`
- spectral/shape stats: centroid `15.0564`, low frac `0.523787`, mid frac `0.440066`, high frac `0.0361465`, total variation `0.0548881`
- RMSE mean: baseline `0.03383`, loss1 `0.0221518`, loss2 `0.022926`, loss3 `0.0135326`
- RMSE reduction vs baseline: loss1 `34.52%`, loss2 `32.23%`, loss3 `60.00%`
- relative L2 mean: baseline `0.0660564`, loss1 `0.0427566`, loss2 `0.0443239`, loss3 `0.0262884`
- relative L2 reduction vs baseline: loss1 `35.27%`, loss2 `32.90%`, loss3 `60.20%`
- loss3 clean-RMSE best among all four: `True`

## 31 - burgers_widevis_l3target_d28

- group: `generalization`
- display label: `Power-law Fourier alpha=2.2, k0=16; range [-0.2,1.2]`
- family: `powerlaw_fourier`
- samples: `200`
- path: `generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00/burgers/burgers_widevis_l3target_d28.pt`
- description: Power-law Fourier spectrum across slow-to-fast decay and wide radius; loss3-targeted replacement candidate; Power-law Fourier alpha=2.2, k0=16; range [-0.2,1.2]
- parameters: `{"base_config": "powerlaw_a2p2_k16", "base_family": "powerlaw_fourier", "descriptive_name": "Power-law Fourier alpha=2.2, k0=16; range [-0.2,1.2]", "display_label": "Power-law Fourier alpha=2.2, k0=16; range [-0.2,1.2]", "hard_value_max": 1.7, "hard_value_min": -0.7, "k0": 16.0, "loss3_targeted_profile": true, "spectral_alpha": 2.2, "target_max": 1.2, "target_min": -0.2, "transform": "range_affine", "transform_variant": "range", "wide_parameter_visible_profile": true}`
- observed x stats: min `-0.2`, max `1.2`, mean `0.509935`, std `0.291866`
- spectral/shape stats: centroid `10.1477`, low frac `0.493754`, mid frac `0.502393`, high frac `0.00385316`, total variation `0.0192881`
- RMSE mean: baseline `0.0192153`, loss1 `0.0137511`, loss2 `0.0149497`, loss3 `0.00835283`
- RMSE reduction vs baseline: loss1 `28.44%`, loss2 `22.20%`, loss3 `56.53%`
- relative L2 mean: baseline `0.0372505`, loss1 `0.0266486`, loss2 `0.0289889`, loss3 `0.0162378`
- relative L2 reduction vs baseline: loss1 `28.46%`, loss2 `22.18%`, loss3 `56.41%`
- loss3 clean-RMSE best among all four: `True`

## 32 - burgers_widevis_l3target_d29

- group: `generalization`
- display label: `Power-law Fourier alpha=3.5, k0=28; range [0.15,1.25]`
- family: `powerlaw_fourier`
- samples: `200`
- path: `generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00/burgers/burgers_widevis_l3target_d29.pt`
- description: Power-law Fourier spectrum across slow-to-fast decay and wide radius; loss3-targeted replacement candidate; Power-law Fourier alpha=3.5, k0=28; range [0.15,1.25]
- parameters: `{"base_config": "powerlaw_a3p5_k28", "base_family": "powerlaw_fourier", "descriptive_name": "Power-law Fourier alpha=3.5, k0=28; range [0.15,1.25]", "display_label": "Power-law Fourier alpha=3.5, k0=28; range [0.15,1.25]", "hard_value_max": 1.7, "hard_value_min": -0.7, "k0": 28.0, "loss3_targeted_profile": true, "spectral_alpha": 3.5, "target_max": 1.25, "target_min": 0.15, "transform": "range_affine", "transform_variant": "range", "wide_parameter_visible_profile": true}`
- observed x stats: min `0.15`, max `1.25`, mean `0.702303`, std `0.229475`
- spectral/shape stats: centroid `11.1901`, low frac `0.43006`, mid frac `0.568401`, high frac `0.0015388`, total variation `0.0156888`
- RMSE mean: baseline `0.0219552`, loss1 `0.0121687`, loss2 `0.0130049`, loss3 `0.00750623`
- RMSE reduction vs baseline: loss1 `44.57%`, loss2 `40.77%`, loss3 `65.81%`
- relative L2 mean: baseline `0.0305766`, loss1 `0.0171564`, loss2 `0.0183737`, loss3 `0.0105579`
- relative L2 reduction vs baseline: loss1 `43.89%`, loss2 `39.91%`, loss3 `65.47%`
- loss3 clean-RMSE best among all four: `True`

## 33 - burgers_widevis_l3target_d30

- group: `generalization`
- display label: `Power-law Fourier alpha=3.5, k0=28; range [-0.5,1.5]`
- family: `powerlaw_fourier`
- samples: `200`
- path: `generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00/burgers/burgers_widevis_l3target_d30.pt`
- description: Power-law Fourier spectrum across slow-to-fast decay and wide radius; loss3-targeted replacement candidate; Power-law Fourier alpha=3.5, k0=28; range [-0.5,1.5]
- parameters: `{"base_config": "powerlaw_a3p5_k28", "base_family": "powerlaw_fourier", "descriptive_name": "Power-law Fourier alpha=3.5, k0=28; range [-0.5,1.5]", "display_label": "Power-law Fourier alpha=3.5, k0=28; range [-0.5,1.5]", "hard_value_max": 1.7, "hard_value_min": -0.7, "k0": 28.0, "loss3_targeted_profile": true, "spectral_alpha": 3.5, "target_max": 1.5, "target_min": -0.5, "transform": "range_affine", "transform_variant": "range", "wide_parameter_visible_profile": true}`
- observed x stats: min `-0.5`, max `1.5`, mean `0.497138`, std `0.4138`
- spectral/shape stats: centroid `11.177`, low frac `0.443328`, mid frac `0.55509`, high frac `0.00158169`, total variation `0.0282809`
- RMSE mean: baseline `0.0391768`, loss1 `0.0307324`, loss2 `0.0321123`, loss3 `0.018782`
- RMSE reduction vs baseline: loss1 `21.55%`, loss2 `18.03%`, loss3 `52.06%`
- relative L2 mean: baseline `0.0790297`, loss1 `0.0611553`, loss2 `0.0633274`, loss3 `0.0375092`
- relative L2 reduction vs baseline: loss1 `22.62%`, loss2 `19.87%`, loss3 `52.54%`
- loss3 clean-RMSE best among all four: `True`

## 34 - burgers_widevis_l3target_d31

- group: `generalization`
- display label: `Power-law Fourier alpha=4, k0=36; range [-0.5,1.5]`
- family: `powerlaw_fourier`
- samples: `200`
- path: `generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00/burgers/burgers_widevis_l3target_d31.pt`
- description: Power-law Fourier spectrum across slow-to-fast decay and wide radius; loss3-targeted replacement candidate; Power-law Fourier alpha=4, k0=36; range [-0.5,1.5]
- parameters: `{"base_config": "powerlaw_a4_k36", "base_family": "powerlaw_fourier", "descriptive_name": "Power-law Fourier alpha=4, k0=36; range [-0.5,1.5]", "display_label": "Power-law Fourier alpha=4, k0=36; range [-0.5,1.5]", "hard_value_max": 1.7, "hard_value_min": -0.7, "k0": 36.0, "loss3_targeted_profile": true, "spectral_alpha": 4.0, "target_max": 1.5, "target_min": -0.5, "transform": "range_affine", "transform_variant": "range", "wide_parameter_visible_profile": true}`
- observed x stats: min `-0.5`, max `1.5`, mean `0.501729`, std `0.405633`
- spectral/shape stats: centroid `13.0266`, low frac `0.366755`, mid frac `0.630576`, high frac `0.00266889`, total variation `0.0318456`
- RMSE mean: baseline `0.0370013`, loss1 `0.0333889`, loss2 `0.0337367`, loss3 `0.0204489`
- RMSE reduction vs baseline: loss1 `9.76%`, loss2 `8.82%`, loss3 `44.73%`
- relative L2 mean: baseline `0.0729226`, loss1 `0.0662119`, loss2 `0.0667295`, loss3 `0.0407514`
- relative L2 reduction vs baseline: loss1 `9.20%`, loss2 `8.49%`, loss3 `44.12%`
- loss3 clean-RMSE best among all four: `True`

## 35 - burgers_widevis_l3target_d32

- group: `generalization`
- display label: `Power-law Fourier alpha=4, k0=36; range [0.15,1.25]`
- family: `powerlaw_fourier`
- samples: `200`
- path: `generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00/burgers/burgers_widevis_l3target_d32.pt`
- description: Power-law Fourier spectrum across slow-to-fast decay and wide radius; loss3-targeted replacement candidate; Power-law Fourier alpha=4, k0=36; range [0.15,1.25]
- parameters: `{"base_config": "powerlaw_a4_k36", "base_family": "powerlaw_fourier", "descriptive_name": "Power-law Fourier alpha=4, k0=36; range [0.15,1.25]", "display_label": "Power-law Fourier alpha=4, k0=36; range [0.15,1.25]", "hard_value_max": 1.7, "hard_value_min": -0.7, "k0": 36.0, "loss3_targeted_profile": true, "spectral_alpha": 4.0, "target_max": 1.25, "target_min": 0.15, "transform": "range_affine", "transform_variant": "range", "wide_parameter_visible_profile": true}`
- observed x stats: min `0.15`, max `1.25`, mean `0.692558`, std `0.224968`
- spectral/shape stats: centroid `12.9111`, low frac `0.365182`, mid frac `0.632209`, high frac `0.0026086`, total variation `0.0176313`
- RMSE mean: baseline `0.0201384`, loss1 `0.01256`, loss2 `0.0132342`, loss3 `0.00773249`
- RMSE reduction vs baseline: loss1 `37.63%`, loss2 `34.28%`, loss3 `61.60%`
- relative L2 mean: baseline `0.0287153`, loss1 `0.0180228`, loss2 `0.0190076`, loss3 `0.0110723`
- relative L2 reduction vs baseline: loss1 `37.24%`, loss2 `33.81%`, loss3 `61.44%`
- loss3 clean-RMSE best among all four: `True`

## 36 - burgers_widevis_l3target_d33

- group: `generalization`
- display label: `Power-law Fourier alpha=3, k0=24; range [-0.1,1.1]`
- family: `powerlaw_fourier`
- samples: `200`
- path: `generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00/burgers/burgers_widevis_l3target_d33.pt`
- description: Power-law Fourier spectrum across slow-to-fast decay and wide radius; loss3-targeted replacement candidate; Power-law Fourier alpha=3, k0=24; range [-0.1,1.1]
- parameters: `{"base_config": "powerlaw_a3_k24", "base_family": "powerlaw_fourier", "descriptive_name": "Power-law Fourier alpha=3, k0=24; range [-0.1,1.1]", "display_label": "Power-law Fourier alpha=3, k0=24; range [-0.1,1.1]", "hard_value_max": 1.7, "hard_value_min": -0.7, "k0": 24.0, "loss3_targeted_profile": true, "spectral_alpha": 3.0, "target_max": 1.1, "target_min": -0.1, "transform": "range_affine", "transform_variant": "range", "wide_parameter_visible_profile": true}`
- observed x stats: min `-0.1`, max `1.1`, mean `0.501759`, std `0.249686`
- spectral/shape stats: centroid `10.8186`, low frac `0.44918`, mid frac `0.548609`, high frac `0.00221059`, total variation `0.01669`
- RMSE mean: baseline `0.016644`, loss1 `0.0117138`, loss2 `0.0126598`, loss3 `0.00717432`
- RMSE reduction vs baseline: loss1 `29.62%`, loss2 `23.94%`, loss3 `56.90%`
- relative L2 mean: baseline `0.0328834`, loss1 `0.0231265`, loss2 `0.0250744`, loss3 `0.0141831`
- relative L2 reduction vs baseline: loss1 `29.67%`, loss2 `23.75%`, loss3 `56.87%`
- loss3 clean-RMSE best among all four: `True`

## 37 - burgers_widevis_l3target_d34

- group: `generalization`
- display label: `Power-law Fourier alpha=1.8, k0=12; range [-0.5,1.5]`
- family: `powerlaw_fourier`
- samples: `200`
- path: `generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00/burgers/burgers_widevis_l3target_d34.pt`
- description: Power-law Fourier spectrum across slow-to-fast decay and wide radius; loss3-targeted replacement candidate; Power-law Fourier alpha=1.8, k0=12; range [-0.5,1.5]
- parameters: `{"base_config": "powerlaw_a1p8_k12", "base_family": "powerlaw_fourier", "descriptive_name": "Power-law Fourier alpha=1.8, k0=12; range [-0.5,1.5]", "display_label": "Power-law Fourier alpha=1.8, k0=12; range [-0.5,1.5]", "hard_value_max": 1.7, "hard_value_min": -0.7, "k0": 12.0, "loss3_targeted_profile": true, "spectral_alpha": 1.8, "target_max": 1.5, "target_min": -0.5, "transform": "range_affine", "transform_variant": "range", "wide_parameter_visible_profile": true}`
- observed x stats: min `-0.5`, max `1.5`, mean `0.506131`, std `0.417499`
- spectral/shape stats: centroid `9.80292`, low frac `0.550351`, mid frac `0.442715`, high frac `0.0069343`, total variation `0.0296198`
- RMSE mean: baseline `0.0422841`, loss1 `0.0276993`, loss2 `0.0292449`, loss3 `0.0169424`
- RMSE reduction vs baseline: loss1 `34.49%`, loss2 `30.84%`, loss3 `59.93%`
- relative L2 mean: baseline `0.0821174`, loss1 `0.054257`, loss2 `0.0572676`, loss3 `0.0331389`
- relative L2 reduction vs baseline: loss1 `33.93%`, loss2 `30.26%`, loss3 `59.64%`
- loss3 clean-RMSE best among all four: `True`

## 38 - burgers_widevis_l3target_d35

- group: `generalization`
- display label: `Power-law Fourier alpha=3, k0=24; range [-0.5,1.5]`
- family: `powerlaw_fourier`
- samples: `200`
- path: `generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00/burgers/burgers_widevis_l3target_d35.pt`
- description: Power-law Fourier spectrum across slow-to-fast decay and wide radius; loss3-targeted replacement candidate; Power-law Fourier alpha=3, k0=24; range [-0.5,1.5]
- parameters: `{"base_config": "powerlaw_a3_k24", "base_family": "powerlaw_fourier", "descriptive_name": "Power-law Fourier alpha=3, k0=24; range [-0.5,1.5]", "display_label": "Power-law Fourier alpha=3, k0=24; range [-0.5,1.5]", "hard_value_max": 1.7, "hard_value_min": -0.7, "k0": 24.0, "loss3_targeted_profile": true, "spectral_alpha": 3.0, "target_max": 1.5, "target_min": -0.5, "transform": "range_affine", "transform_variant": "range", "wide_parameter_visible_profile": true}`
- observed x stats: min `-0.5`, max `1.5`, mean `0.495072`, std `0.414741`
- spectral/shape stats: centroid `10.9121`, low frac `0.441473`, mid frac `0.556359`, high frac `0.00216852`, total variation `0.0279893`
- RMSE mean: baseline `0.0377971`, loss1 `0.0321148`, loss2 `0.0328861`, loss3 `0.0195687`
- RMSE reduction vs baseline: loss1 `15.03%`, loss2 `12.99%`, loss3 `48.23%`
- relative L2 mean: baseline `0.0755834`, loss1 `0.0638614`, loss2 `0.065403`, loss3 `0.0388788`
- relative L2 reduction vs baseline: loss1 `15.51%`, loss2 `13.47%`, loss3 `48.56%`
- loss3 clean-RMSE best among all four: `True`

## 39 - burgers_widevis_l3target_d36

- group: `generalization`
- display label: `Sine mix f=[5, 9, 15, 23], decay=0.25; range [0,1.5]`
- family: `sine_mixture`
- samples: `200`
- path: `generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00/burgers/burgers_widevis_l3target_d36.pt`
- description: Sine-mixture field biased toward mid/high-frequency diverse patterns; loss3-targeted replacement candidate; Sine mix f=[5, 9, 15, 23], decay=0.25; range [0,1.5]
- parameters: `{"base_config": "sine_f5_9_15_23_decay0p25", "base_family": "sine_mixture", "decay": 0.25, "descriptive_name": "Sine mix f=[5, 9, 15, 23], decay=0.25; range [0,1.5]", "display_label": "Sine mix f=[5, 9, 15, 23], decay=0.25; range [0,1.5]", "frequencies": [5, 9, 15, 23], "hard_value_max": 1.7, "hard_value_min": -0.7, "loss3_targeted_profile": true, "target_max": 1.5, "target_min": 0.0, "transform": "range_affine", "transform_variant": "range", "wide_parameter_visible_profile": true}`
- observed x stats: min `0`, max `1.5`, mean `0.750318`, std `0.307049`
- spectral/shape stats: centroid `12.1211`, low frac `0.357393`, mid frac `0.639107`, high frac `0.00350003`, total variation `0.0291777`
- RMSE mean: baseline `0.0375488`, loss1 `0.0245485`, loss2 `0.0245218`, loss3 `0.0106424`
- RMSE reduction vs baseline: loss1 `34.62%`, loss2 `34.69%`, loss3 `71.66%`
- relative L2 mean: baseline `0.049647`, loss1 `0.0324902`, loss2 `0.0324518`, loss3 `0.0140812`
- relative L2 reduction vs baseline: loss1 `34.56%`, loss2 `34.63%`, loss3 `71.64%`
- loss3 clean-RMSE best among all four: `True`

## 40 - burgers_widevis_l3target_d37

- group: `generalization`
- display label: `Sine mix f=[7, 19, 43, 89], decay=0.15; range [0,1.5]`
- family: `sine_mixture`
- samples: `200`
- path: `generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00/burgers/burgers_widevis_l3target_d37.pt`
- description: Sine-mixture field biased toward mid/high-frequency diverse patterns; loss3-targeted replacement candidate; Sine mix f=[7, 19, 43, 89], decay=0.15; range [0,1.5]
- parameters: `{"base_config": "sine_f7_19_43_89_decay0p15", "base_family": "sine_mixture", "decay": 0.15, "descriptive_name": "Sine mix f=[7, 19, 43, 89], decay=0.15; range [0,1.5]", "display_label": "Sine mix f=[7, 19, 43, 89], decay=0.15; range [0,1.5]", "frequencies": [7, 19, 43, 89], "hard_value_max": 1.7, "hard_value_min": -0.7, "loss3_targeted_profile": true, "target_max": 1.5, "target_min": 0.0, "transform": "range_affine", "transform_variant": "range", "wide_parameter_visible_profile": true}`
- observed x stats: min `0`, max `1.5`, mean `0.749172`, std `0.284885`
- spectral/shape stats: centroid `32.1162`, low frac `0.35757`, mid frac `0.47298`, high frac `0.16945`, total variation `0.0642928`
- RMSE mean: baseline `0.0278759`, loss1 `0.0189146`, loss2 `0.0212742`, loss3 `0.00861603`
- RMSE reduction vs baseline: loss1 `32.15%`, loss2 `23.68%`, loss3 `69.09%`
- relative L2 mean: baseline `0.0371069`, loss1 `0.0251775`, loss2 `0.0283181`, loss3 `0.0114676`
- relative L2 reduction vs baseline: loss1 `32.15%`, loss2 `23.68%`, loss3 `69.10%`
- loss3 clean-RMSE best among all four: `True`

## 41 - burgers_widevis_l3target_d38

- group: `generalization`
- display label: `Sine mix f=[5, 9, 15, 23], decay=0.25; range [-0.3,1.3]`
- family: `sine_mixture`
- samples: `200`
- path: `generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00/burgers/burgers_widevis_l3target_d38.pt`
- description: Sine-mixture field biased toward mid/high-frequency diverse patterns; loss3-targeted replacement candidate; Sine mix f=[5, 9, 15, 23], decay=0.25; range [-0.3,1.3]
- parameters: `{"base_config": "sine_f5_9_15_23_decay0p25", "base_family": "sine_mixture", "decay": 0.25, "descriptive_name": "Sine mix f=[5, 9, 15, 23], decay=0.25; range [-0.3,1.3]", "display_label": "Sine mix f=[5, 9, 15, 23], decay=0.25; range [-0.3,1.3]", "frequencies": [5, 9, 15, 23], "hard_value_max": 1.7, "hard_value_min": -0.7, "loss3_targeted_profile": true, "target_max": 1.3, "target_min": -0.3, "transform": "range_affine", "transform_variant": "range", "wide_parameter_visible_profile": true}`
- observed x stats: min `-0.3`, max `1.3`, mean `0.50008`, std `0.32608`
- spectral/shape stats: centroid `12.124`, low frac `0.357383`, mid frac `0.639124`, high frac `0.00349319`, total variation `0.0310047`
- RMSE mean: baseline `0.0226499`, loss1 `0.0206316`, loss2 `0.0216936`, loss3 `0.0101236`
- RMSE reduction vs baseline: loss1 `8.91%`, loss2 `4.22%`, loss3 `55.30%`
- relative L2 mean: baseline `0.0445982`, loss1 `0.0406199`, loss2 `0.0427178`, loss3 `0.019924`
- relative L2 reduction vs baseline: loss1 `8.92%`, loss2 `4.22%`, loss3 `55.33%`
- loss3 clean-RMSE best among all four: `True`

## 42 - burgers_widevis_l3target_d39

- group: `generalization`
- display label: `Sine mix f=[7, 19, 43, 89], decay=0.15; range [-0.3,1.3]`
- family: `sine_mixture`
- samples: `200`
- path: `generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00/burgers/burgers_widevis_l3target_d39.pt`
- description: Sine-mixture field biased toward mid/high-frequency diverse patterns; loss3-targeted replacement candidate; Sine mix f=[7, 19, 43, 89], decay=0.15; range [-0.3,1.3]
- parameters: `{"base_config": "sine_f7_19_43_89_decay0p15", "base_family": "sine_mixture", "decay": 0.15, "descriptive_name": "Sine mix f=[7, 19, 43, 89], decay=0.15; range [-0.3,1.3]", "display_label": "Sine mix f=[7, 19, 43, 89], decay=0.15; range [-0.3,1.3]", "frequencies": [7, 19, 43, 89], "hard_value_max": 1.7, "hard_value_min": -0.7, "loss3_targeted_profile": true, "target_max": 1.3, "target_min": -0.3, "transform": "range_affine", "transform_variant": "range", "wide_parameter_visible_profile": true}`
- observed x stats: min `-0.3`, max `1.3`, mean `0.499189`, std `0.303455`
- spectral/shape stats: centroid `32.1247`, low frac `0.357732`, mid frac `0.472635`, high frac `0.169633`, total variation `0.0684948`
- RMSE mean: baseline `0.0230209`, loss1 `0.0219059`, loss2 `0.022153`, loss3 `0.0105043`
- RMSE reduction vs baseline: loss1 `4.84%`, loss2 `3.77%`, loss3 `54.37%`
- relative L2 mean: baseline `0.0458299`, loss1 `0.0436129`, loss2 `0.0440974`, loss3 `0.0208945`
- relative L2 reduction vs baseline: loss1 `4.84%`, loss2 `3.78%`, loss3 `54.41%`
- loss3 clean-RMSE best among all four: `True`

## 43 - burgers_widevis_l3target_d40

- group: `generalization`
- display label: `Sine mix f=[7, 19, 43, 89], decay=0.15; range [-0.2,1.2]`
- family: `sine_mixture`
- samples: `200`
- path: `generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00/burgers/burgers_widevis_l3target_d40.pt`
- description: Sine-mixture field biased toward mid/high-frequency diverse patterns; loss3-targeted replacement candidate; Sine mix f=[7, 19, 43, 89], decay=0.15; range [-0.2,1.2]
- parameters: `{"base_config": "sine_f7_19_43_89_decay0p15", "base_family": "sine_mixture", "decay": 0.15, "descriptive_name": "Sine mix f=[7, 19, 43, 89], decay=0.15; range [-0.2,1.2]", "display_label": "Sine mix f=[7, 19, 43, 89], decay=0.15; range [-0.2,1.2]", "frequencies": [7, 19, 43, 89], "hard_value_max": 1.7, "hard_value_min": -0.7, "loss3_targeted_profile": true, "target_max": 1.2, "target_min": -0.2, "transform": "range_affine", "transform_variant": "range", "wide_parameter_visible_profile": true}`
- observed x stats: min `-0.2`, max `1.2`, mean `0.499127`, std `0.26553`
- spectral/shape stats: centroid `32.1214`, low frac `0.357668`, mid frac `0.47275`, high frac `0.169583`, total variation `0.0599357`
- RMSE mean: baseline `0.0189297`, loss1 `0.0177194`, loss2 `0.0179467`, loss3 `0.00854363`
- RMSE reduction vs baseline: loss1 `6.39%`, loss2 `5.19%`, loss3 `54.87%`
- relative L2 mean: baseline `0.0377055`, loss1 `0.0352959`, loss2 `0.0357438`, loss3 `0.0170045`
- relative L2 reduction vs baseline: loss1 `6.39%`, loss2 `5.20%`, loss3 `54.90%`
- loss3 clean-RMSE best among all four: `True`

## 44 - burgers_widevis_l3target_d41

- group: `generalization`
- display label: `Sine mix f=[5, 9, 15, 23], decay=0.25; range [-0.65,1.35]`
- family: `sine_mixture`
- samples: `200`
- path: `generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00/burgers/burgers_widevis_l3target_d41.pt`
- description: Sine-mixture field biased toward mid/high-frequency diverse patterns; loss3-targeted replacement candidate; Sine mix f=[5, 9, 15, 23], decay=0.25; range [-0.65,1.35]
- parameters: `{"base_config": "sine_f5_9_15_23_decay0p25", "base_family": "sine_mixture", "decay": 0.25, "descriptive_name": "Sine mix f=[5, 9, 15, 23], decay=0.25; range [-0.65,1.35]", "display_label": "Sine mix f=[5, 9, 15, 23], decay=0.25; range [-0.65,1.35]", "frequencies": [5, 9, 15, 23], "hard_value_max": 1.7, "hard_value_min": -0.7, "loss3_targeted_profile": true, "target_max": 1.35, "target_min": -0.65, "transform": "range_affine", "transform_variant": "range", "wide_parameter_visible_profile": true}`
- observed x stats: min `-0.65`, max `1.35`, mean `0.350773`, std `0.404736`
- spectral/shape stats: centroid `12.1241`, low frac `0.357336`, mid frac `0.639177`, high frac `0.00348727`, total variation `0.0384651`
- RMSE mean: baseline `0.0337055`, loss1 `0.0350339`, loss2 `0.0344868`, loss3 `0.0170079`
- RMSE reduction vs baseline: loss1 `-3.94%`, loss2 `-2.32%`, loss3 `49.54%`
- relative L2 mean: baseline `0.0929582`, loss1 `0.0964855`, loss2 `0.0949278`, loss3 `0.0468649`
- relative L2 reduction vs baseline: loss1 `-3.79%`, loss2 `-2.12%`, loss3 `49.58%`
- loss3 clean-RMSE best among all four: `True`

## 45 - burgers_widevis_l3target_d42

- group: `generalization`
- display label: `Sine mix f=[5, 9, 15, 23], decay=0.25; range [-0.5,1.5]`
- family: `sine_mixture`
- samples: `200`
- path: `generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00/burgers/burgers_widevis_l3target_d42.pt`
- description: Sine-mixture field biased toward mid/high-frequency diverse patterns; loss3-targeted replacement candidate; Sine mix f=[5, 9, 15, 23], decay=0.25; range [-0.5,1.5]
- parameters: `{"base_config": "sine_f5_9_15_23_decay0p25", "base_family": "sine_mixture", "decay": 0.25, "descriptive_name": "Sine mix f=[5, 9, 15, 23], decay=0.25; range [-0.5,1.5]", "display_label": "Sine mix f=[5, 9, 15, 23], decay=0.25; range [-0.5,1.5]", "frequencies": [5, 9, 15, 23], "hard_value_max": 1.7, "hard_value_min": -0.7, "loss3_targeted_profile": true, "target_max": 1.5, "target_min": -0.5, "transform": "range_affine", "transform_variant": "range", "wide_parameter_visible_profile": true}`
- observed x stats: min `-0.5`, max `1.5`, mean `0.500345`, std `0.406655`
- spectral/shape stats: centroid `12.1227`, low frac `0.357418`, mid frac `0.639096`, high frac `0.00348643`, total variation `0.038672`
- RMSE mean: baseline `0.031532`, loss1 `0.0309423`, loss2 `0.03351`, loss3 `0.0152545`
- RMSE reduction vs baseline: loss1 `1.87%`, loss2 `-6.27%`, loss3 `51.62%`
- relative L2 mean: baseline `0.061954`, loss1 `0.0607625`, loss2 `0.0658165`, loss3 `0.0299411`
- relative L2 reduction vs baseline: loss1 `1.92%`, loss2 `-6.23%`, loss3 `51.67%`
- loss3 clean-RMSE best among all four: `True`

## 46 - burgers_widevis_l3target_d43

- group: `generalization`
- display label: `Sine mix f=[5, 9, 15, 23], decay=0.25; range [-0.2,1.2]`
- family: `sine_mixture`
- samples: `200`
- path: `generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00/burgers/burgers_widevis_l3target_d43.pt`
- description: Sine-mixture field biased toward mid/high-frequency diverse patterns; loss3-targeted replacement candidate; Sine mix f=[5, 9, 15, 23], decay=0.25; range [-0.2,1.2]
- parameters: `{"base_config": "sine_f5_9_15_23_decay0p25", "base_family": "sine_mixture", "decay": 0.25, "descriptive_name": "Sine mix f=[5, 9, 15, 23], decay=0.25; range [-0.2,1.2]", "display_label": "Sine mix f=[5, 9, 15, 23], decay=0.25; range [-0.2,1.2]", "frequencies": [5, 9, 15, 23], "hard_value_max": 1.7, "hard_value_min": -0.7, "loss3_targeted_profile": true, "target_max": 1.2, "target_min": -0.2, "transform": "range_affine", "transform_variant": "range", "wide_parameter_visible_profile": true}`
- observed x stats: min `-0.2`, max `1.2`, mean `0.500305`, std `0.285826`
- spectral/shape stats: centroid `12.1252`, low frac `0.357231`, mid frac `0.639275`, high frac `0.00349351`, total variation `0.027173`
- RMSE mean: baseline `0.0195099`, loss1 `0.0160159`, loss2 `0.0166358`, loss3 `0.00824367`
- RMSE reduction vs baseline: loss1 `17.91%`, loss2 `14.73%`, loss3 `57.75%`
- relative L2 mean: baseline `0.0384397`, loss1 `0.0315512`, loss2 `0.0327784`, loss3 `0.0162367`
- relative L2 reduction vs baseline: loss1 `17.92%`, loss2 `14.73%`, loss3 `57.76%`
- loss3 clean-RMSE best among all four: `True`

## 47 - burgers_widevis_l3target_d44

- group: `generalization`
- display label: `Sine mix f=[7, 19, 43, 89], decay=0.15; range [-0.5,1.5]`
- family: `sine_mixture`
- samples: `200`
- path: `generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00/burgers/burgers_widevis_l3target_d44.pt`
- description: Sine-mixture field biased toward mid/high-frequency diverse patterns; loss3-targeted replacement candidate; Sine mix f=[7, 19, 43, 89], decay=0.15; range [-0.5,1.5]
- parameters: `{"base_config": "sine_f7_19_43_89_decay0p15", "base_family": "sine_mixture", "decay": 0.15, "descriptive_name": "Sine mix f=[7, 19, 43, 89], decay=0.15; range [-0.5,1.5]", "display_label": "Sine mix f=[7, 19, 43, 89], decay=0.15; range [-0.5,1.5]", "frequencies": [7, 19, 43, 89], "hard_value_max": 1.7, "hard_value_min": -0.7, "loss3_targeted_profile": true, "target_max": 1.5, "target_min": -0.5, "transform": "range_affine", "transform_variant": "range", "wide_parameter_visible_profile": true}`
- observed x stats: min `-0.5`, max `1.5`, mean `0.498767`, std `0.380342`
- spectral/shape stats: centroid `32.1312`, low frac `0.357699`, mid frac `0.47263`, high frac `0.169671`, total variation `0.0858363`
- RMSE mean: baseline `0.0306223`, loss1 `0.0299367`, loss2 `0.0299964`, loss3 `0.0151803`
- RMSE reduction vs baseline: loss1 `2.24%`, loss2 `2.04%`, loss3 `50.43%`
- relative L2 mean: baseline `0.0609887`, loss1 `0.0596258`, loss2 `0.059732`, loss3 `0.0302076`
- relative L2 reduction vs baseline: loss1 `2.23%`, loss2 `2.06%`, loss3 `50.47%`
- loss3 clean-RMSE best among all four: `True`

## 48 - burgers_widevis_l3target_d45

- group: `generalization`
- display label: `Sine mix f=[5, 10, 20, 40, 80], decay=0.9; range [-0.1,1.1]`
- family: `sine_mixture`
- samples: `200`
- path: `generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00/burgers/burgers_widevis_l3target_d45.pt`
- description: Sine/cosine-like mixtures with visibly different frequencies. Sine mix f=[5, 10, 20, 40, 80], decay=0.9; range [-0.1,1.1]
- parameters: `{"base_config": "cfg_sine_f5_10_20_40_80_decay0p9_rangem0p1to1p1", "base_family": "sine_mixture", "decay": 0.9, "descriptive_name": "Sine mix f=[5, 10, 20, 40, 80], decay=0.9; range [-0.1,1.1]", "display_label": "Sine mix f=[5, 10, 20, 40, 80], decay=0.9; range [-0.1,1.1]", "frequencies": [5, 10, 20, 40, 80], "hard_value_max": 1.7, "hard_value_min": -0.7, "target_max": 1.1, "target_min": -0.1, "transform": "range_affine", "transform_variant": "range", "wide_parameter_visible_profile": true}`
- observed x stats: min `-0.1`, max `1.1`, mean `0.498171`, std `0.27561`
- spectral/shape stats: centroid `22.975`, low frac `0.671409`, mid frac `0.270664`, high frac `0.0579269`, total variation `0.0754126`
- RMSE mean: baseline `0.0111659`, loss1 `0.00815988`, loss2 `0.00861422`, loss3 `0.00426401`
- RMSE reduction vs baseline: loss1 `26.92%`, loss2 `22.85%`, loss3 `61.81%`
- relative L2 mean: baseline `0.022281`, loss1 `0.0164339`, loss2 `0.0173096`, loss3 `0.00874176`
- relative L2 reduction vs baseline: loss1 `26.24%`, loss2 `22.31%`, loss3 `60.77%`
- loss3 clean-RMSE best among all four: `True`

## 49 - burgers_widevis_l3target_d46

- group: `generalization`
- display label: `Sine mix f=[4, 7, 11], decay=0.35; range [-0.3,1.3]`
- family: `sine_mixture`
- samples: `200`
- path: `generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00/burgers/burgers_widevis_l3target_d46.pt`
- description: Sine-mixture field biased toward mid/high-frequency diverse patterns; loss3-targeted replacement candidate; Sine mix f=[4, 7, 11], decay=0.35; range [-0.3,1.3]
- parameters: `{"base_config": "sine_f4_7_11_decay0p35", "base_family": "sine_mixture", "decay": 0.35, "descriptive_name": "Sine mix f=[4, 7, 11], decay=0.35; range [-0.3,1.3]", "display_label": "Sine mix f=[4, 7, 11], decay=0.35; range [-0.3,1.3]", "frequencies": [4, 7, 11], "hard_value_max": 1.7, "hard_value_min": -0.7, "loss3_targeted_profile": true, "target_max": 1.3, "target_min": -0.3, "transform": "range_affine", "transform_variant": "range", "wide_parameter_visible_profile": true}`
- observed x stats: min `-0.3`, max `1.3`, mean `0.492566`, std `0.40615`
- spectral/shape stats: centroid `8.03699`, low frac `0.768142`, mid frac `0.226549`, high frac `0.00530902`, total variation `0.0360028`
- RMSE mean: baseline `0.0294824`, loss1 `0.0141254`, loss2 `0.0187268`, loss3 `0.00854877`
- RMSE reduction vs baseline: loss1 `52.09%`, loss2 `36.48%`, loss3 `71.00%`
- relative L2 mean: baseline `0.0646295`, loss1 `0.0305784`, loss2 `0.0401177`, loss3 `0.0182951`
- relative L2 reduction vs baseline: loss1 `52.69%`, loss2 `37.93%`, loss3 `71.69%`
- loss3 clean-RMSE best among all four: `True`

## 50 - burgers_widevis_l3target_d47

- group: `generalization`
- display label: `Sine mix f=[7, 19, 43, 89], decay=0.15; range [-0.65,1.35]`
- family: `sine_mixture`
- samples: `200`
- path: `generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00/burgers/burgers_widevis_l3target_d47.pt`
- description: Sine-mixture field biased toward mid/high-frequency diverse patterns; loss3-targeted replacement candidate; Sine mix f=[7, 19, 43, 89], decay=0.15; range [-0.65,1.35]
- parameters: `{"base_config": "sine_f7_19_43_89_decay0p15", "base_family": "sine_mixture", "decay": 0.15, "descriptive_name": "Sine mix f=[7, 19, 43, 89], decay=0.15; range [-0.65,1.35]", "display_label": "Sine mix f=[7, 19, 43, 89], decay=0.15; range [-0.65,1.35]", "frequencies": [7, 19, 43, 89], "hard_value_max": 1.7, "hard_value_min": -0.7, "loss3_targeted_profile": true, "target_max": 1.35, "target_min": -0.65, "transform": "range_affine", "transform_variant": "range", "wide_parameter_visible_profile": true}`
- observed x stats: min `-0.65`, max `1.35`, mean `0.349271`, std `0.380248`
- spectral/shape stats: centroid `32.1299`, low frac `0.357712`, mid frac `0.472645`, high frac `0.169642`, total variation `0.085774`
- RMSE mean: baseline `0.02987`, loss1 `0.0292982`, loss2 `0.0275311`, loss3 `0.0160864`
- RMSE reduction vs baseline: loss1 `1.91%`, loss2 `7.83%`, loss3 `46.15%`
- relative L2 mean: baseline `0.0842116`, loss1 `0.0826487`, loss2 `0.0776391`, loss3 `0.0454101`
- relative L2 reduction vs baseline: loss1 `1.86%`, loss2 `7.80%`, loss3 `46.08%`
- loss3 clean-RMSE best among all four: `True`

## 51 - burgers_widevis_l3target_d48

- group: `generalization`
- display label: `Sawtooth freq=2; range [0.15,1.25]`
- family: `sawtooth`
- samples: `200`
- path: `generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00/burgers/burgers_widevis_l3target_d48.pt`
- description: Tiny sawtooth quota for visibly sharp comparison. Sawtooth freq=2; range [0.15,1.25]
- parameters: `{"base_config": "cfg_saw_f2_range0p15to1p25", "base_family": "sawtooth", "descriptive_name": "Sawtooth freq=2; range [0.15,1.25]", "display_label": "Sawtooth freq=2; range [0.15,1.25]", "frequency": 2, "hard_value_max": 1.7, "hard_value_min": -0.7, "target_max": 1.25, "target_min": 0.15, "transform": "range_affine", "transform_variant": "range", "wide_parameter_visible_profile": true}`
- observed x stats: min `0.15`, max `1.25`, mean `0.7`, std `0.318164`
- spectral/shape stats: centroid `8.11653`, low frac `0.827473`, mid frac `0.153459`, high frac `0.019068`, total variation `0.00428824`
- RMSE mean: baseline `0.0159441`, loss1 `0.0159191`, loss2 `0.0128519`, loss3 `0.00457624`
- RMSE reduction vs baseline: loss1 `0.16%`, loss2 `19.39%`, loss3 `71.30%`
- relative L2 mean: baseline `0.0222854`, loss1 `0.0222505`, loss2 `0.0179634`, loss3 `0.00639632`
- relative L2 reduction vs baseline: loss1 `0.16%`, loss2 `19.39%`, loss3 `71.30%`
- loss3 clean-RMSE best among all four: `True`

## 52 - burgers_widevis_l3target_d49

- group: `generalization`
- display label: `Square wave freq=7, duty=0.35; range [0,1.2]`
- family: `square_wave`
- samples: `200`
- path: `generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00/burgers/burgers_widevis_l3target_d49.pt`
- description: Single square-wave comparison. Square wave freq=7, duty=0.35; range [0,1.2]
- parameters: `{"base_config": "cfg_square_f7_d0p35_range0to1p2", "base_family": "square_wave", "descriptive_name": "Square wave freq=7, duty=0.35; range [0,1.2]", "display_label": "Square wave freq=7, duty=0.35; range [0,1.2]", "duty": 0.35, "frequency": 7, "hard_value_max": 1.7, "hard_value_min": -0.7, "target_max": 1.2, "target_min": 0.0, "transform": "range_affine", "transform_variant": "range", "wide_parameter_visible_profile": true}`
- observed x stats: min `0`, max `1.2`, mean `0.45868`, std `0.462891`
- spectral/shape stats: centroid `20.3615`, low frac `0.702175`, mid frac `0.243242`, high frac `0.0545827`, total variation `0.0562604`
- RMSE mean: baseline `0.0370513`, loss1 `0.0222884`, loss2 `0.0275533`, loss3 `0.0097359`
- RMSE reduction vs baseline: loss1 `39.84%`, loss2 `25.63%`, loss3 `73.72%`
- relative L2 mean: baseline `0.0800706`, loss1 `0.0480706`, loss2 `0.0594117`, loss3 `0.0209648`
- relative L2 reduction vs baseline: loss1 `39.96%`, loss2 `25.80%`, loss3 `73.82%`
- loss3 clean-RMSE best among all four: `True`
