# Loss3 Surprising Findings Cross-PQ Validation - 2026-05-20

Status: post-processing analysis from existing local artifacts; no optimizer experiment was run.

## Scope

This note checks whether the surprising findings are universal or conditional across the available completed alpha/epsilon and P/Q settings.

Observed source sets:

- Full `p=2,q=2`, 20-setting, 300-step sweep.
- Stopped off-diagonal P/Q 100-step sweep: full 20-setting groups for `p=1,q=2`, `p=1,q=inf`, `p=2,q=1`, `p=2,q=inf`, plus partial 2-setting `p=inf,q=1`.

Generated tables:

- Per-setting/method validation: `forensics/loss3_surprising_findings_validation_20260520/per_setting_method_validation.csv`
- P/Q-method rollup: `forensics/loss3_surprising_findings_validation_20260520/pq_method_validation_rollup.csv`
- Final-loss winners: `forensics/loss3_surprising_findings_validation_20260520/final_loss_winner_counts_by_pq.csv`
- Angle-motion winners: `forensics/loss3_surprising_findings_validation_20260520/angle_motion_winner_counts_by_pq.csv`
- Boundary-arrival winners: `forensics/loss3_surprising_findings_validation_20260520/boundary_arrival_winner_counts_by_pq.csv`
- Delta-similarity pair rollup: `forensics/loss3_surprising_findings_validation_20260520/delta_similarity_pair_rollup_by_pq.csv`
- Near-exact equivalent method pairs: `forensics/loss3_surprising_findings_validation_20260520/equivalent_method_pairs_by_pq.csv`

## Claim Checks

### 1. Boundary arrival is not convergence

Supported, but strength varies by method and P/Q. Replacement methods usually reach 99% boundary at step 1 and often still have positive post-boundary loss gain. Additive methods may also keep improving after boundary arrival, but they spend more steps getting there and sometimes do not reach the 99% mean boundary within the stopped 100-step off-diagonal runs.

Replacement-method rollup:

| dataset_tag | pq | method | setting_count | median_step_to_boundary_0p99 | mean_post_boundary_loss_gain | fraction_positive_post_boundary_gain | mean_delta_prev_angle_degrees |
|---|---|---|---|---|---|---|---|
| p2q2_300step | p=2,q=2 | raw_replace | 20 | 1 | 3.412 | 1 | 30.68 |
| p2q2_300step | p=2,q=2 | steepest_replace | 20 | 1 | 3.412 | 1 | 30.68 |
| pneq_100step_stopped | p=1,q=2 | raw_replace | 20 | 1 | 0.05039 | 1 | 4.459 |
| pneq_100step_stopped | p=1,q=2 | steepest_replace | 20 | 1 | -0.05431 | 0.9 | 21.15 |
| pneq_100step_stopped | p=1,q=inf | raw_replace | 20 | 1 | 1.318e-05 | 0.2 | 44.86 |
| pneq_100step_stopped | p=1,q=inf | steepest_replace | 20 | 1 | -0.04598 | 0.1 | 23.43 |
| pneq_100step_stopped | p=2,q=1 | raw_replace | 20 | 1 | 70.32 | 1 | 25.73 |
| pneq_100step_stopped | p=2,q=1 | steepest_replace | 20 | 1 | 70.32 | 1 | 25.73 |
| pneq_100step_stopped | p=2,q=inf | raw_replace | 20 | 1 | 0.1543 | 1 | 66.86 |
| pneq_100step_stopped | p=2,q=inf | steepest_replace | 20 | 1 | 0.1543 | 1 | 66.86 |
| pneq_100step_stopped | p=inf,q=1 | raw_replace | 2 | 1 | 160.8 | 1 | 27.15 |
| pneq_100step_stopped | p=inf,q=1 | steepest_replace | 2 | 1 | 171.6 | 1 | 72.61 |

### 2. GPI/replacement is fast, but not always final-loss winner

Supported. The final-loss winner counts show that replacement is not universally the long-run final-loss winner, especially in the 300-step p2q2 run where additive methods can overtake it.

| dataset_tag | pq | method | winner_count_fractional_ties | setting_count | winner_fraction |
|---|---|---|---|---|---|
| p2q2_300step | p=2,q=2 | steepest_add | 16 | 20 | 0.8 |
| p2q2_300step | p=2,q=2 | raw_replace | 1.5 | 20 | 0.075 |
| p2q2_300step | p=2,q=2 | steepest_replace | 1.5 | 20 | 0.075 |
| p2q2_300step | p=2,q=2 | raw_add | 1 | 20 | 0.05 |
| pneq_100step_stopped | p=1,q=2 | steepest_add | 20 | 20 | 1 |
| pneq_100step_stopped | p=1,q=2 | raw_add | 0 | 20 | 0 |
| pneq_100step_stopped | p=1,q=2 | raw_replace | 0 | 20 | 0 |
| pneq_100step_stopped | p=1,q=2 | steepest_replace | 0 | 20 | 0 |
| pneq_100step_stopped | p=1,q=inf | steepest_add | 20 | 20 | 1 |
| pneq_100step_stopped | p=1,q=inf | raw_add | 0 | 20 | 0 |
| pneq_100step_stopped | p=1,q=inf | raw_replace | 0 | 20 | 0 |
| pneq_100step_stopped | p=1,q=inf | steepest_replace | 0 | 20 | 0 |
| pneq_100step_stopped | p=2,q=1 | raw_add | 14 | 20 | 0.7 |
| pneq_100step_stopped | p=2,q=1 | raw_replace | 2.5 | 20 | 0.125 |
| pneq_100step_stopped | p=2,q=1 | steepest_replace | 2.5 | 20 | 0.125 |
| pneq_100step_stopped | p=2,q=1 | steepest_add | 1 | 20 | 0.05 |
| pneq_100step_stopped | p=2,q=inf | steepest_add | 18 | 20 | 0.9 |
| pneq_100step_stopped | p=2,q=inf | raw_add | 2 | 20 | 0.1 |
| pneq_100step_stopped | p=2,q=inf | raw_replace | 0 | 20 | 0 |
| pneq_100step_stopped | p=2,q=inf | steepest_replace | 0 | 20 | 0 |
| pneq_100step_stopped | p=inf,q=1 | steepest_add | 2 | 2 | 1 |
| pneq_100step_stopped | p=inf,q=1 | raw_add | 0 | 2 | 0 |
| pneq_100step_stopped | p=inf,q=1 | raw_replace | 0 | 2 | 0 |
| pneq_100step_stopped | p=inf,q=1 | steepest_replace | 0 | 2 | 0 |

### 3. LP-steepest boundary-ratio growth is more linear than raw PGD

Mostly supported for p2q2 and many off-diagonal groups, but not a theorem. The linearity check uses pre-boundary mean boundary-ratio `R^2`, increment coefficient of variation, and late/early increment ratio. LP-steepest usually has smaller increment CV and a late/early ratio closer to 1 than raw PGD; raw PGD often has a smaller late/early ratio, matching the visual bending/slowing observation.

| dataset_tag | pq | method | setting_count | median_boundary_linearity_r2 | median_boundary_increment_cv | median_late_over_early_boundary_increment | mean_max_boundary_ratio_std_over_steps |
|---|---|---|---|---|---|---|---|
| p2q2_300step | p=2,q=2 | raw_add | 20 | 0.9305 | 0.5771 | 0.2253 | 0.2815 |
| p2q2_300step | p=2,q=2 | steepest_add | 20 | 0.9915 | 0.2683 | 0.611 | 0.05144 |
| pneq_100step_stopped | p=1,q=2 | raw_add | 20 | 0.8849 | 0.6969 | 0.07249 | 0.1626 |
| pneq_100step_stopped | p=1,q=2 | steepest_add | 20 | 1 | 3.554e-07 | 1 | 0.01885 |
| pneq_100step_stopped | p=1,q=inf | raw_add | 20 | 0.7415 | 1.083 | 0.03239 | 0.2565 |
| pneq_100step_stopped | p=1,q=inf | steepest_add | 20 | 1 | 6.167e-07 | 1 | 0.03563 |
| pneq_100step_stopped | p=2,q=1 | raw_add | 20 | 0.9394 | 0.5875 | 0.1267 | 0.1787 |
| pneq_100step_stopped | p=2,q=1 | steepest_add | 20 | 0.9938 | 0.203 | 0.6968 | 0.04201 |
| pneq_100step_stopped | p=2,q=inf | raw_add | 20 | 0.9333 | 0.7554 | 0.2853 | 0.1486 |
| pneq_100step_stopped | p=2,q=inf | steepest_add | 20 | 0.8935 | 0.7673 | 0.0907 | 0.1268 |
| pneq_100step_stopped | p=inf,q=1 | raw_add | 2 | 0.9534 | 0.5414 | 0.2795 | 0.2628 |
| pneq_100step_stopped | p=inf,q=1 | steepest_add | 2 | 1 | 1.976e-07 | 1 | 0 |

### 4. Boundary-ratio std is a diagnostic

Supported. Raw PGD generally has the largest boundary-ratio standard deviation; LP-steepest is much smaller; replacement methods are often near zero when replacement stays on the p-boundary. In some off-diagonal P/Q settings, final boundary ratios and replacement behavior can deviate, so this should be read alongside final boundary ratio and P/Q geometry.

### 5. GPI has the largest angular motion

Mostly supported, but there are conditional cases. The angle-winner table below counts which method has the largest mean `angle(delta_k, delta_{k-1})` within each setting. Replacement/GPI often dominates, but raw_replace and steepest_replace can tie or overlap in p=2-like replacement geometries, and off-diagonal settings can change the winner.

| dataset_tag | pq | method | winner_count_fractional_ties | setting_count | winner_fraction |
|---|---|---|---|---|---|
| p2q2_300step | p=2,q=2 | raw_replace | 10 | 20 | 0.5 |
| p2q2_300step | p=2,q=2 | steepest_replace | 10 | 20 | 0.5 |
| p2q2_300step | p=2,q=2 | raw_add | 0 | 20 | 0 |
| p2q2_300step | p=2,q=2 | steepest_add | 0 | 20 | 0 |
| pneq_100step_stopped | p=1,q=2 | steepest_replace | 16 | 20 | 0.8 |
| pneq_100step_stopped | p=1,q=2 | raw_replace | 4 | 20 | 0.2 |
| pneq_100step_stopped | p=1,q=2 | raw_add | 0 | 20 | 0 |
| pneq_100step_stopped | p=1,q=2 | steepest_add | 0 | 20 | 0 |
| pneq_100step_stopped | p=1,q=inf | raw_replace | 20 | 20 | 1 |
| pneq_100step_stopped | p=1,q=inf | raw_add | 0 | 20 | 0 |
| pneq_100step_stopped | p=1,q=inf | steepest_add | 0 | 20 | 0 |
| pneq_100step_stopped | p=1,q=inf | steepest_replace | 0 | 20 | 0 |
| pneq_100step_stopped | p=2,q=1 | raw_replace | 10 | 20 | 0.5 |
| pneq_100step_stopped | p=2,q=1 | steepest_replace | 10 | 20 | 0.5 |
| pneq_100step_stopped | p=2,q=1 | raw_add | 0 | 20 | 0 |
| pneq_100step_stopped | p=2,q=1 | steepest_add | 0 | 20 | 0 |
| pneq_100step_stopped | p=2,q=inf | raw_replace | 10 | 20 | 0.5 |
| pneq_100step_stopped | p=2,q=inf | steepest_replace | 10 | 20 | 0.5 |
| pneq_100step_stopped | p=2,q=inf | raw_add | 0 | 20 | 0 |
| pneq_100step_stopped | p=2,q=inf | steepest_add | 0 | 20 | 0 |
| pneq_100step_stopped | p=inf,q=1 | steepest_replace | 2 | 2 | 1 |
| pneq_100step_stopped | p=inf,q=1 | raw_add | 0 | 2 | 0 |
| pneq_100step_stopped | p=inf,q=1 | raw_replace | 0 | 2 | 0 |
| pneq_100step_stopped | p=inf,q=1 | steepest_add | 0 | 2 | 0 |

### 6. Final delta shapes are often similar, but not always identical

Supported as a broad visual/spectral statement, not as a universal exact equality. Pairwise similarity remains high for several pairs, but exact equivalence only occurs for specific method pairs and P/Q geometries.

Near-exact equivalent pairs across all settings in a P/Q group:

| dataset_tag | pq | method_pair | setting_count | mean_cosine | mean_relative_l2 |
|---|---|---|---|---|---|
| p2_q1_100step | p=2,q=1 | raw_replace__steepest_replace | 20 | 1 | 0 |
| p2_qinf_100step | p=2,q=inf | raw_replace__steepest_replace | 20 | 1 | 0 |
| p2q2_300step | p=2,q=2 | raw_replace__steepest_replace | 20 | 1 | 0 |

Selected similarity rollup:

| dataset_tag | pq | method_pair | setting_count | mean_cosine | min_cosine | mean_spectral_cosine | mean_relative_l2 | equivalent_setting_count |
|---|---|---|---|---|---|---|---|---|
| p1_q2_100step | p=1,q=2 | raw_add__steepest_add | 20 | 0.136 | 0.09615 | 0.2573 | 1.859 | 0 |
| p1_q2_100step | p=1,q=2 | raw_add__steepest_replace | 20 | 0.1004 | 0.08412 | 0.2354 | 1.901 | 0 |
| p1_q2_100step | p=1,q=2 | raw_replace__steepest_replace | 20 | 0.09934 | 0.08129 | 0.2346 | 1.901 | 0 |
| p1_q2_100step | p=1,q=2 | steepest_add__steepest_replace | 20 | 0.3167 | 0.1573 | 0.9619 | 1.091 | 0 |
| p1_qinf_100step | p=1,q=inf | raw_add__steepest_add | 20 | 0.18 | 0.06854 | 0.4223 | 1.826 | 0 |
| p1_qinf_100step | p=1,q=inf | raw_add__steepest_replace | 20 | 0.1012 | 0.00312 | 0.4228 | 1.886 | 0 |
| p1_qinf_100step | p=1,q=inf | raw_replace__steepest_replace | 20 | 0.03737 | 0.003339 | 0.4366 | 1.894 | 0 |
| p1_qinf_100step | p=1,q=inf | steepest_add__steepest_replace | 20 | 0.3941 | 0.07355 | 0.9695 | 0.9759 | 0 |
| p2_q1_100step | p=2,q=1 | raw_add__steepest_add | 20 | 0.8153 | 0.6333 | 0.9284 | 0.4276 | 0 |
| p2_q1_100step | p=2,q=1 | raw_add__steepest_replace | 20 | 0.7255 | 0.612 | 0.9247 | 0.5523 | 0 |
| p2_q1_100step | p=2,q=1 | raw_replace__steepest_replace | 20 | 1 | 1 | 1 | 0 | 20 |
| p2_q1_100step | p=2,q=1 | steepest_add__steepest_replace | 20 | 0.6528 | 0.505 | 0.875 | 0.6712 | 0 |
| p2_qinf_100step | p=2,q=inf | raw_add__steepest_add | 20 | 0.5592 | 0.253 | 0.806 | 0.9888 | 0 |
| p2_qinf_100step | p=2,q=inf | raw_add__steepest_replace | 20 | 0.1947 | 0.06799 | 0.6909 | 1.309 | 0 |
| p2_qinf_100step | p=2,q=inf | raw_replace__steepest_replace | 20 | 1 | 1 | 1 | 0 | 20 |
| p2_qinf_100step | p=2,q=inf | steepest_add__steepest_replace | 20 | 0.2223 | 0.08724 | 0.7726 | 1.21 | 0 |
| p2q2_300step | p=2,q=2 | raw_add__steepest_add | 20 | 0.9178 | 0.7809 | 0.968 | 0.1815 | 0 |
| p2q2_300step | p=2,q=2 | raw_add__steepest_replace | 20 | 0.6359 | 0.5168 | 0.8661 | 0.6659 | 0 |
| p2q2_300step | p=2,q=2 | raw_replace__steepest_replace | 20 | 1 | 1 | 1 | 0 | 20 |
| p2q2_300step | p=2,q=2 | steepest_add__steepest_replace | 20 | 0.6587 | 0.5505 | 0.8892 | 0.6354 | 0 |
| pinf_q1_partial_100step | p=inf,q=1 | raw_add__steepest_add | 2 | 0.5118 | 0.4802 | 0.8221 | 0.9345 | 0 |
| pinf_q1_partial_100step | p=inf,q=1 | raw_add__steepest_replace | 2 | 0.1723 | 0.1721 | 0.7091 | 1.272 | 0 |
| pinf_q1_partial_100step | p=inf,q=1 | raw_replace__steepest_replace | 2 | 0.3085 | 0.3085 | 0.7796 | 1.395 | 0 |
| pinf_q1_partial_100step | p=inf,q=1 | steepest_add__steepest_replace | 2 | 0.2124 | 0.1988 | 0.7454 | 1.233 | 0 |

### 7. P/Q geometry affects perturbation realism

Supported, but it should be stated as a combined visual + metric conclusion. `p=2,q=1` looked visually more physical in representative samples, while `q=inf` groups can create localized spikes. The peakiness/high-frequency metrics below help locate where this concern appears quantitatively, but visual representative panels remain important because a single localized spike may be more obvious in the plot than in a rollup average.

| dataset_tag | pq | method | setting_count | mean_peakiness_max_abs_over_rms | max_peakiness_max_abs_over_rms | mean_final_high_frequency_energy_ratio | mean_final_first_derivative_l2 |
|---|---|---|---|---|---|---|---|
| p2q2_300step | p=2,q=2 | raw_add | 20 | 4.128 | 8.176 | 4.460e-06 | 0.3828 |
| p2q2_300step | p=2,q=2 | steepest_add | 20 | 4.048 | 8.172 | 4.337e-06 | 0.3276 |
| p2q2_300step | p=2,q=2 | raw_replace | 20 | 3.633 | 7.17 | 5.060e-09 | 0.2009 |
| p2q2_300step | p=2,q=2 | steepest_replace | 20 | 3.633 | 7.17 | 5.060e-09 | 0.2009 |
| pneq_100step_stopped | p=1,q=2 | steepest_add | 20 | 29.66 | 32 | 0.2295 | 7.274 |
| pneq_100step_stopped | p=1,q=2 | steepest_replace | 20 | 30.05 | 32 | 0.2361 | 7.954 |
| pneq_100step_stopped | p=1,q=2 | raw_add | 20 | 4.305 | 6.359 | 5.427e-08 | 0.02842 |
| pneq_100step_stopped | p=1,q=2 | raw_replace | 20 | 4.268 | 6.271 | 5.468e-08 | 0.02782 |
| pneq_100step_stopped | p=1,q=inf | steepest_add | 20 | 29.54 | 32 | 0.2445 | 6.247 |
| pneq_100step_stopped | p=1,q=inf | steepest_replace | 20 | 30.69 | 32 | 0.2412 | 8.402 |
| pneq_100step_stopped | p=1,q=inf | raw_add | 20 | 8.281 | 21.93 | 0.02153 | 0.1127 |
| pneq_100step_stopped | p=1,q=inf | raw_replace | 20 | 9.069 | 21.84 | 0.02411 | 0.1425 |
| pneq_100step_stopped | p=2,q=1 | raw_add | 20 | 3.532 | 5.948 | 4.821e-06 | 0.2832 |
| pneq_100step_stopped | p=2,q=1 | steepest_add | 20 | 3.653 | 5.94 | 6.269e-06 | 0.3917 |
| pneq_100step_stopped | p=2,q=1 | raw_replace | 20 | 3.509 | 5.818 | 5.991e-06 | 0.2443 |
| pneq_100step_stopped | p=2,q=1 | steepest_replace | 20 | 3.509 | 5.818 | 5.991e-06 | 0.2443 |
| pneq_100step_stopped | p=2,q=inf | raw_replace | 20 | 6.019 | 25.25 | 0.0111 | 1.475 |
| pneq_100step_stopped | p=2,q=inf | steepest_replace | 20 | 6.019 | 25.25 | 0.0111 | 1.475 |
| pneq_100step_stopped | p=2,q=inf | raw_add | 20 | 6.728 | 21.35 | 0.01106 | 0.8387 |
| pneq_100step_stopped | p=2,q=inf | steepest_add | 20 | 5.305 | 21.22 | 0.007185 | 1.114 |
| pneq_100step_stopped | p=inf,q=1 | raw_replace | 2 | 2.832 | 4.18 | 1.364e-06 | 0.3306 |
| pneq_100step_stopped | p=inf,q=1 | steepest_add | 2 | 1.257 | 1.946 | 0.004447 | 5.89 |
| pneq_100step_stopped | p=inf,q=1 | raw_add | 2 | 1.309 | 1.754 | 6.286e-04 | 3.226 |
| pneq_100step_stopped | p=inf,q=1 | steepest_replace | 2 | 1 | 1 | 0.001137 | 4.121 |

## Bottom Line

The earlier summary is directionally right, but not every statement is universal. The most robust cross-setting claim is: the key distinction is radial boundary use plus angular/boundary-surface motion, not boundary arrival alone. GPI/replacement is the most aggressive boundary-direction optimizer and is usually fastest, while additive methods can sometimes win final loss after many steps. P/Q geometry strongly affects whether the final perturbation looks physically plausible or spike-like.

For writing, separate claims into three levels:

1. Strongly supported: replacement reaches boundary immediately in the main p2q2 setting; boundary arrival and convergence are distinct; raw PGD has larger radial variance; no-std and std plots answer different questions.
2. Mostly supported but conditional: LP-steepest is more linear than raw PGD; GPI has the largest angular motion; final deltas share broad shape similarity.
3. Geometry-dependent: perturbation realism and spike behavior, especially in `q=inf` settings and partial `p=inf` evidence.

## Setting-Level Exception Tables

Additional setting-level tables were generated after the rollup to make the conditional claims easier to audit:

- Replacement post-boundary nonpositive gain settings: `forensics/loss3_surprising_findings_validation_20260520/replacement_nonpositive_post_boundary_gain_settings.csv`. Row count: 36. These are concentrated in `p=1,q=inf` (34 rows) and `p=1,q=2` (2 rows), so "boundary arrival then loss grows" is strong for p2 and p=2/q=1-like settings, but not universal for p=1/q=inf geometry.
- Raw-vs-LP-steepest additive linearity comparisons: `forensics/loss3_surprising_findings_validation_20260520/raw_vs_steepest_add_linearity_by_setting.csv`.
- Raw-vs-LP-steepest additive linearity exceptions: `forensics/loss3_surprising_findings_validation_20260520/raw_vs_steepest_add_linearity_exceptions.csv`. Row count: 39. Exceptions are especially common for `p=2,q=inf` (17 rows), so the "LP-steepest is straighter" claim is strongest for p2q2, p1q2, p1qinf, p2q1, and partial pinf/q1, but q=inf can break the simple picture.
- Angle winner by setting: `forensics/loss3_surprising_findings_validation_20260520/angle_winner_by_setting.csv`. Row count: 102. Every setting's largest mean angular motion is in the replacement family, but the winner is not always `steepest_replace`: p2 settings tie `raw_replace` and `steepest_replace`; `p=1,q=inf` is won by `raw_replace`; `p=1,q=2` is mostly `steepest_replace` but has 4 raw-replace wins.
- Exact/near-exact equivalent method-pair settings: `forensics/loss3_surprising_findings_validation_20260520/equivalent_method_pairs_by_setting.csv`. Row count: 65.
- Near-high-cosine pair settings: `forensics/loss3_surprising_findings_validation_20260520/near_high_cosine_method_pairs_by_setting.csv`. Row count: 86.
- Low-cosine selected pair settings: `forensics/loss3_surprising_findings_validation_20260520/low_cosine_selected_pairs_by_setting.csv`. Row count: 181. These low-cosine cases show that "final deltas look similar" is not universal; it is strongest for p2q2 and p2q1, and much weaker for p=1 or q=inf geometries.

## Refined Claim Status

| claim | status after cross-PQ check |
|---|---|
| Boundary arrival is not convergence | True as a mechanism, but positive post-boundary gain is not universal. Strong in p2q2, p2q1, p2qinf, partial pinf/q1; weak or sometimes negative in p=1,q=inf and a few p=1,q=2 replacement cases. |
| GPI/replacement is fast | Strongly true for boundary arrival and angular motion. It is not always the final-loss winner. |
| LP-steepest boundary growth is straighter than raw PGD | Mostly true, but q=inf creates many exceptions. Use this as an empirical tendency, not a theorem. |
| Boundary-ratio std diagnoses radial synchronization | Strongly true. Raw PGD generally has much larger std; LP-steepest smaller; replacement near zero when it truly stays on the p-boundary. |
| GPI has the strongest angular motion | Replacement-family angular motion wins every checked setting. But whether the label is `raw_replace` or `steepest_replace` depends on P geometry. |
| Final delta shapes are similar | Conditional. p2q2 and p2q1 show much stronger similarity; p=1 and q=inf settings can be quite dissimilar. |
| P/Q affects perturbation realism | Strongly supported by visual inspection and peakiness/high-frequency summaries. p2q1 looks comparatively natural; q=inf and p=1 steepest/replacement settings are spike-prone. |
| raw_replace equals steepest_replace when p=2 | Supported across all checked p=2 groups: q=1, q=2, q=inf. Not supported for p=1 or p=inf. |

## Theoretical Caveats: Why Fast GPI Is Suspicious

The empirical result that generalized-power/replacement updates optimize Loss 3 very quickly should not be interpreted as a theorem that Loss 3 is a generalized-power objective. This is one of the least automatic and most important caveats in the current evidence.

Observed from the existing sweeps:
- Replacement/GPI-style methods reach the epsilon boundary immediately or nearly immediately in the main p=2 settings.
- Their loss can continue to grow substantially after boundary arrival, which means the important mechanism is boundary-surface direction change, not just budget saturation.
- Their per-step delta angular motion is much larger than raw PGD or LP-steepest additive PGD.
- In p2q2 and p2q1, the final delta often resembles the additive-method delta, so GPI is often finding a similar broad direction much earlier.
- However, 300-step additive methods sometimes reach equal or larger final mean loss, so GPI is not an unconditional global optimizer.

Inference:
- The most plausible explanation is that GPI is acting as a strong surrogate optimizer for the local linearized or dominant-mode part of Loss 3, not as an exact optimizer for the full nonlinear loss.
- If the neural-operator loss landscape is dominated early by a low-rank, leading-singular-mode, or nearly quadratic component, a power-iteration-like replacement step can look surprisingly effective even when the true objective is not exactly a generalized Rayleigh quotient.
- The result becomes suspicious precisely because Loss 3 is not mathematically guaranteed to match the generalized-power subproblem. The correct interpretation is empirical alignment between the loss geometry and the GPI surrogate in these runs.

Things that remain not fully explained:
- Why the local Loss 3 landscape seems to have such a strong dominant direction in p2q2/p2q1 settings.
- Why replacement can rotate so aggressively on the boundary without producing worse final perturbation shapes in the main p=2 cases.
- Why p=1 and q=inf geometries break the clean story and create spike-like or weak post-boundary-gain exceptions.

Suggested next diagnostics:
- Compare each method step direction with the actual gradient direction by cosine similarity.
- Measure whether the local Hessian/Jacobian spectrum is dominated by one or a few modes near the initial condition and along the boundary trajectory.
- Plot radial profiles L(x + t u) for early GPI directions and final additive directions to see whether GPI identifies a high-value ray early.
- Track loss change predicted by the local linear model versus the actual loss change for each method step.
- Measure boundary-tangent motion: decompose each update into radial and tangent components after boundary arrival.

Paper-safe wording:
- Do not claim that generalized power iteration is the mathematically exact optimizer for Loss 3.
- A safer claim is: in the tested p=2 regimes, the replacement/GPI update appears to be a highly effective surrogate because it saturates the norm budget immediately and rapidly rotates the perturbation direction along the boundary, often reaching a final perturbation shape close to long-run additive methods in only a few steps.

