# Generalization Relative-L2 Loss Bar Plots V3

This version plots `relative_l2` directly as the loss. Lower is better. It does not convert loss into an accuracy score.

Train and test are first; generated datasets are sorted from closest to farthest by `feature_distance_to_train`. A dashed orange horizontal line marks the original test relative-L2 loss.

## Plots

- `generalization_eval/burgers_generalization_relative_l2_loss_barplot_v3.png`
- `generalization_eval/darcy_generalization_relative_l2_loss_barplot_v3.png`
- `generalization_eval/ns2d_generalization_relative_l2_loss_barplot_v3.png`

## Generated datasets with lower relative-L2 than original test

### burgers
- Original test relative-L2: `0.0177549`
- Generated datasets below test loss: `1`
  - `burgers_near_gaussian_corr0p75`: relative_l2=`0.0167386`, tier=`near_param_shift`, distance=`0.3669`

### darcy
- Original test relative-L2: `0.0227883`
- Generated datasets below test loss: `13`
  - `darcy_far_alpha5_tau2_bin3_12`: relative_l2=`0.0113148`, tier=`far_range_pattern`, distance=`0.03648`
  - `darcy_far_alpha6_tau1p5_bin3_12`: relative_l2=`0.0113635`, tier=`far_range_pattern`, distance=`0.03189`
  - `darcy_near_alpha3p5_tau3`: relative_l2=`0.0152159`, tier=`near_param_shift`, distance=`0.02282`
  - `darcy_near_alpha4_tau4`: relative_l2=`0.0162406`, tier=`near_param_shift`, distance=`0.02864`
  - `darcy_far_alpha2_tau1_bin3_12`: relative_l2=`0.0183382`, tier=`far_range_pattern`, distance=`0.01647`
  - `darcy_near_alpha2p5_tau3p5`: relative_l2=`0.0189807`, tier=`near_param_shift`, distance=`0.02275`
  - `darcy_near_alpha2_tau2`: relative_l2=`0.0194024`, tier=`near_param_shift`, distance=`0.02947`
  - `darcy_near_alpha2p3_tau3`: relative_l2=`0.0194376`, tier=`near_param_shift`, distance=`0.023`
  - `darcy_near_alpha3_tau4`: relative_l2=`0.0195047`, tier=`near_param_shift`, distance=`0.02459`
  - `darcy_near_alpha2p15_tau3`: relative_l2=`0.0212687`, tier=`near_param_shift`, distance=`0.01058`
  - `darcy_far_alpha4_tau8_bin3_12`: relative_l2=`0.0219182`, tier=`far_range_pattern`, distance=`0.02238`
  - `darcy_near_alpha2_tau2p5`: relative_l2=`0.0221209`, tier=`near_param_shift`, distance=`0.01213`
  - `darcy_near_alpha1p85_tau3`: relative_l2=`0.0221464`, tier=`near_param_shift`, distance=`0.02461`

### ns2d
- Original test relative-L2: `0.0957001`
- Generated datasets below test loss: `29`
  - `ns_far_square_centered_scale1_shift0`: relative_l2=`0.0177915`, tier=`far_range_pattern`, distance=`0.3308`
  - `ns_mid_spectrum_alpha4_tau4`: relative_l2=`0.0178179`, tier=`mid_kernel_spectrum`, distance=`0.3575`
  - `ns_mid_spectrum_alpha3_tau3`: relative_l2=`0.0194117`, tier=`mid_kernel_spectrum`, distance=`0.334`
  - `ns_near_grf_alpha4p5_tau7`: relative_l2=`0.0209346`, tier=`near_param_shift`, distance=`0.2939`
  - `ns_mid_spectrum_alpha5_tau8`: relative_l2=`0.0213654`, tier=`mid_kernel_spectrum`, distance=`0.2944`
  - `ns_mid_spectrum_alpha6_tau10`: relative_l2=`0.0217554`, tier=`mid_kernel_spectrum`, distance=`0.291`
  - `ns_far_scale_scale0p5_shift0`: relative_l2=`0.0244275`, tier=`far_range_pattern`, distance=`0.2702`
  - `ns_far_log_abs_centered_scale1_shift0`: relative_l2=`0.0262027`, tier=`far_range_pattern`, distance=`0.2867`
  - `ns_mid_spectrum_alpha2_tau2`: relative_l2=`0.0266004`, tier=`mid_kernel_spectrum`, distance=`0.247`
  - `ns_near_grf_alpha3p2_tau6`: relative_l2=`0.0268117`, tier=`near_param_shift`, distance=`0.2341`
  - `ns_far_square_centered_scale2_shift0`: relative_l2=`0.0289234`, tier=`far_range_pattern`, distance=`0.2367`
  - `ns_near_grf_alpha4_tau12`: relative_l2=`0.0314874`, tier=`near_param_shift`, distance=`0.2115`
  - `ns_near_grf_alpha3p5_tau10`: relative_l2=`0.0319052`, tier=`near_param_shift`, distance=`0.1916`
  - `ns_mid_spectrum_alpha5_tau20`: relative_l2=`0.032385`, tier=`mid_kernel_spectrum`, distance=`0.2038`
  - `ns_near_grf_alpha2p5_tau5`: relative_l2=`0.0336284`, tier=`near_param_shift`, distance=`0.1895`
  - `ns_near_grf_alpha3_tau8`: relative_l2=`0.0351088`, tier=`near_param_shift`, distance=`0.1827`
  - `ns_near_grf_alpha2p8_tau7`: relative_l2=`0.0372537`, tier=`near_param_shift`, distance=`0.1643`
  - `ns_near_grf_alpha2p2_tau4`: relative_l2=`0.0393483`, tier=`near_param_shift`, distance=`0.1582`
  - `ns_near_grf_alpha2p5_tau6`: relative_l2=`0.0400252`, tier=`near_param_shift`, distance=`0.1581`
  - `ns_near_grf_alpha2p6_tau7`: relative_l2=`0.0481324`, tier=`near_param_shift`, distance=`0.1255`
  - ... 9 more; see `metrics_sorted_by_similarity.csv`.

## Interpretation

It is possible for a generated generalization dataset to have lower loss than the original test set. Distribution shift does not automatically mean harder: some shifted sets are smoother, lower amplitude, lower high-frequency energy, or closer to model inductive bias. Therefore these plots should be read together with `similarity.csv` and the dataset parameters, not only with the manual near/mid/far tier.
