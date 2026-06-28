# Deep Cause Analysis

Recomputed with `max_samples=50` for diagnostic sample-level visual/spectral metrics and train/test std bands. Bar heights still use the full evaluator metrics CSV.

## burgers

- train RMSE 0.0086768, rel-L2 0.0163895; test RMSE 0.0095436, rel-L2 0.0177549.
- generated below test: RMSE 2/48, rel-L2 1/48.
- strongest RMSE correlations: target_final_tv_mean_mean=0.632, input_range_mean=0.615, target_final_highfreq_frac_mean=0.404, target_final_range_mean=0.228, target_final_spectral_centroid_mean=0.209
- low RMSE examples: Burgers: Gaussian initial field, correlation length=0.75, Burgers: Gaussian initial field, correlation length=0.025, Burgers: Gaussian initial field, correlation length=0.035, Burgers: Matern initial field, correlation length=0.1, smoothness nu=5, Burgers: Gaussian initial field, correlation length=0.04
- high RMSE examples: Burgers: centered amplitude x2.5, offset -0.25, Burgers: centered amplitude x2, offset 0, Burgers: add negative offset -1, Burgers: centered amplitude x1, offset -0.4, Burgers: add positive offset 1

## darcy

- train RMSE 0.000134776, rel-L2 0.0198393; test RMSE 0.000155193, rel-L2 0.0227883.
- generated below test: RMSE 13/50, rel-L2 13/50.
- strongest RMSE correlations: target_final_range_mean=0.820, target_final_tv_mean_mean=0.792, target_final_rms_mean=0.728, target_final_spectral_centroid_mean=0.558, target_final_highfreq_frac_mean=0.248
- low RMSE examples: Darcy/C-flow: binary coefficient 3/12, GRF alpha=5, tau=2, Darcy/C-flow: binary coefficient 3/12, GRF alpha=6, tau=1.5, Darcy/C-flow: binary coefficient 3/12, GRF alpha=3.5, tau=3, Darcy/C-flow: binary coefficient 3/12, GRF alpha=4, tau=4, Darcy/C-flow: binary coefficient 3/12, GRF alpha=2, tau=1
- high RMSE examples: Darcy/C-flow: binary coefficient 1/12, GRF alpha=3, tau=1.5, Darcy/C-flow: binary coefficient 1/12, GRF alpha=3.5, tau=6, Darcy/C-flow: binary coefficient 1/12, GRF alpha=2, tau=3, Darcy/C-flow: binary coefficient 1/20, GRF alpha=2, tau=12, Darcy/C-flow: threshold bias -0.8

## ns2d

- train RMSE 0.0713213, rel-L2 0.0512957; test RMSE 0.133571, rel-L2 0.0957001.
- generated below test: RMSE 31/47, rel-L2 31/47.
- strongest RMSE correlations: target_final_highfreq_frac_mean=0.696, target_final_tv_mean_mean=0.688, target_final_range_mean=0.680, target_final_spectral_centroid_mean=0.631, input_range_mean=0.547
- low RMSE examples: NS2D: Gaussian random vorticity, alpha=4, tau=4, NS2D: squared-and-centered initial field, scale=1, NS2D: Gaussian random vorticity, alpha=3, tau=3, NS2D: Gaussian random vorticity, alpha=4.5, tau=7, NS2D: Gaussian random vorticity, alpha=5, tau=8
- high RMSE examples: NS2D: scale x2 and offset 0.5, NS2D: scale x2 and offset -0.5, NS2D: add negative vorticity offset -0.5, NS2D: Gaussian random vorticity, alpha=1, tau=2, NS2D: add positive vorticity offset 0.5

## Interpretation

- The 50-sample rerun leaves the main qualitative pattern intact: loss follows final-target range/TV/surviving spectral content much more than raw input high frequency alone.
- The train/test shaded bands in RMSE bar plots are now +/-1 sample RMSE std computed from 50 diagnostic samples, not variance and not the full-metric aggregation std.
