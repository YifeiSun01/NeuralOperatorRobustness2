# Darcy Binary 20260611 Paired T-Tests for Loss3

Statistical unit: one generalization dataset. `n=50` for each test.

Test definition: paired t-test on `competitor - loss3`. The one-sided alternative is that this mean difference is greater than zero, i.e. Loss3 has lower error/loss than the competitor on the same datasets.

The p-values below are paired-dataset tests. They are not per-sample tests inside each dataset.

## Loss3 vs Second-Best Mean Model

| system | metric_label | competitor_display | n | loss3_mean | competitor_mean | mean_diff_competitor_minus_loss3 | ci95_diff_low | ci95_diff_high | t_stat | df | p_one_sided_loss3_lower | p_one_sided_bh_all_second_tests | cohens_dz | loss3_lower_dataset_count | competitor_lower_dataset_count | loss3_is_best_by_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean_generalization | generalization loss / data MSE | loss2 | 50 | 3.82036e-07 | 8.42343e-07 | 4.60307e-07 | 3.6737e-07 | 5.53245e-07 | 9.95315 | 49 | 1.1761e-13 | 2.11698e-13 | 1.40759 | 47 | 3 | True |
| clean_generalization | RMSE | loss2 | 50 | 0.000594572 | 0.000870154 | 0.000275582 | 0.000232602 | 0.000318561 | 12.8852 | 49 | 1.17752e-17 | 2.64942e-17 | 1.82224 | 47 | 3 | True |
| clean_generalization | Relative L2 | loss2 | 50 | 0.0561539 | 0.0816375 | 0.0254836 | 0.0217596 | 0.0292077 | 13.7516 | 49 | 9.55656e-19 | 2.86697e-18 | 1.94478 | 47 | 3 | True |
| attack50_generalization | attack clean loss | loss2 | 50 | 2.88169e-07 | 6.29486e-07 | 3.41317e-07 | 2.51361e-07 | 4.31272e-07 | 7.62492 | 49 | 3.59309e-10 | 5.38963e-10 | 1.07833 | 44 | 6 | True |
| attack50_generalization | attack adv loss | loss2 | 50 | 1.98716e-06 | 4.19417e-06 | 2.20701e-06 | 2.00881e-06 | 2.40521e-06 | 22.3772 | 49 | 1.11949e-27 | 1.00754e-26 | 3.16461 | 50 | 0 | True |
| attack50_generalization | attack loss increase | loss2 | 50 | 1.699e-06 | 3.56469e-06 | 1.86569e-06 | 1.63309e-06 | 2.0983e-06 | 16.1186 | 49 | 1.59198e-21 | 7.16393e-21 | 2.27951 | 50 | 0 | True |
| attack50_generalization | attack relative increase | random solver | 50 | 8.48128 | 13.3171 | 4.83578 | -0.31517 | 9.98674 | 1.88662 | 49 | 0.0325724 | 0.0418788 | 0.266808 | 18 | 32 | True |
| attack50_generalization | delta L2 RMS | random clean | 50 | 3.68202 | 3.6725 | -0.00952013 | -0.0708616 | 0.0518213 | -0.311884 | 49 | 0.621774 | 0.621774 | -0.0441071 | 24 | 26 | False |
| attack50_generalization | delta Linf | baseline | 50 | 9 | 9 | 0 | 0 | 0 | 0 | 49 | 0.5 | 0.5625 | 0 | 0 | 0 | True |

## Loss3 vs All Other Models

| system | metric_label | competitor_display | n | loss3_mean | competitor_mean | mean_diff_competitor_minus_loss3 | t_stat | p_one_sided_loss3_lower | p_one_sided_bh_all_family | cohens_dz | loss3_lower_dataset_count | competitor_lower_dataset_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean_generalization | generalization loss / data MSE | baseline | 50 | 3.82036e-07 | 1.0453e-06 | 6.63266e-07 | 10.6972 | 1.02465e-14 | 1.15273e-14 | 1.51281 | 50 | 0 |
| clean_generalization | generalization loss / data MSE | loss1 | 50 | 3.82036e-07 | 9.68283e-07 | 5.86247e-07 | 10.8916 | 5.47825e-15 | 6.5739e-15 | 1.5403 | 50 | 0 |
| clean_generalization | generalization loss / data MSE | loss2 | 50 | 3.82036e-07 | 8.42343e-07 | 4.60307e-07 | 9.95315 | 1.1761e-13 | 1.24528e-13 | 1.40759 | 47 | 3 |
| clean_generalization | generalization loss / data MSE | Physics Loss | 50 | 3.82036e-07 | 9.65135e-07 | 5.831e-07 | 11.611 | 5.628e-16 | 7.23601e-16 | 1.64204 | 50 | 0 |
| clean_generalization | generalization loss / data MSE | random clean | 50 | 3.82036e-07 | 8.74675e-07 | 4.92639e-07 | 9.69689 | 2.76827e-13 | 2.76827e-13 | 1.37135 | 48 | 2 |
| clean_generalization | generalization loss / data MSE | random solver | 50 | 3.82036e-07 | 1.2271e-06 | 8.45068e-07 | 13.332 | 3.18744e-18 | 5.73739e-18 | 1.88543 | 50 | 0 |
| clean_generalization | RMSE | baseline | 50 | 0.000594572 | 0.000972351 | 0.000377779 | 16.0269 | 2.01512e-21 | 5.18173e-21 | 2.26655 | 50 | 0 |
| clean_generalization | RMSE | loss1 | 50 | 0.000594572 | 0.000936259 | 0.000341687 | 15.5267 | 7.41779e-21 | 1.669e-20 | 2.1958 | 50 | 0 |
| clean_generalization | RMSE | loss2 | 50 | 0.000594572 | 0.000870154 | 0.000275582 | 12.8852 | 1.17752e-17 | 1.76628e-17 | 1.82224 | 47 | 3 |
| clean_generalization | RMSE | Physics Loss | 50 | 0.000594572 | 0.000938999 | 0.000344427 | 17.2404 | 9.5645e-23 | 2.93608e-22 | 2.43816 | 50 | 0 |
| clean_generalization | RMSE | random clean | 50 | 0.000594572 | 0.000883968 | 0.000289396 | 12.2379 | 8.17927e-17 | 1.13251e-16 | 1.7307 | 48 | 2 |
| clean_generalization | RMSE | random solver | 50 | 0.000594572 | 0.00106732 | 0.000472747 | 21.6497 | 4.8862e-27 | 4.39758e-26 | 3.06173 | 50 | 0 |
| clean_generalization | Relative L2 | baseline | 50 | 0.0561539 | 0.0913369 | 0.035183 | 17.9536 | 1.71251e-23 | 7.7063e-23 | 2.53903 | 50 | 0 |
| clean_generalization | Relative L2 | loss1 | 50 | 0.0561539 | 0.0879285 | 0.0317746 | 17.231 | 9.78694e-23 | 2.93608e-22 | 2.43683 | 50 | 0 |
| clean_generalization | Relative L2 | loss2 | 50 | 0.0561539 | 0.0816375 | 0.0254836 | 13.7516 | 9.55656e-19 | 1.91131e-18 | 1.94478 | 47 | 3 |
| clean_generalization | Relative L2 | Physics Loss | 50 | 0.0561539 | 0.0882931 | 0.0321392 | 19.4669 | 5.2402e-25 | 3.14412e-24 | 2.75303 | 50 | 0 |
| clean_generalization | Relative L2 | random clean | 50 | 0.0561539 | 0.0828807 | 0.0267268 | 12.9759 | 9.0129e-18 | 1.47484e-17 | 1.83507 | 48 | 2 |
| clean_generalization | Relative L2 | random solver | 50 | 0.0561539 | 0.100569 | 0.0444155 | 25.4828 | 3.15053e-30 | 5.67096e-29 | 3.60382 | 50 | 0 |
| attack50_generalization | attack clean loss | baseline | 50 | 2.88169e-07 | 1.4816e-06 | 1.19343e-06 | 10.2306 | 4.69721e-14 | 1.12733e-13 | 1.44682 | 48 | 2 |
| attack50_generalization | attack clean loss | loss1 | 50 | 2.88169e-07 | 7.38492e-07 | 4.50323e-07 | 8.46867 | 1.85086e-11 | 3.91948e-11 | 1.19765 | 44 | 6 |
| attack50_generalization | attack clean loss | loss2 | 50 | 2.88169e-07 | 6.29486e-07 | 3.41317e-07 | 7.62492 | 3.59309e-10 | 6.80796e-10 | 1.07833 | 44 | 6 |
| attack50_generalization | attack clean loss | Physics Loss | 50 | 2.88169e-07 | 7.22671e-07 | 4.34502e-07 | 8.802 | 5.8272e-12 | 1.31112e-11 | 1.24479 | 44 | 6 |
| attack50_generalization | attack clean loss | random clean | 50 | 2.88169e-07 | 6.68449e-07 | 3.8028e-07 | 7.75351 | 2.27928e-10 | 4.55855e-10 | 1.09651 | 44 | 6 |
| attack50_generalization | attack clean loss | random solver | 50 | 2.88169e-07 | 9.31897e-07 | 6.43728e-07 | 10.2999 | 3.74034e-14 | 9.61802e-14 | 1.45662 | 48 | 2 |
| attack50_generalization | attack adv loss | baseline | 50 | 1.98716e-06 | 1.11625e-05 | 9.17536e-06 | 83.5356 | 8.29257e-55 | 2.98532e-53 | 11.8137 | 50 | 0 |
| attack50_generalization | attack adv loss | loss1 | 50 | 1.98716e-06 | 4.66556e-06 | 2.67839e-06 | 23.5137 | 1.21011e-28 | 7.26067e-28 | 3.32534 | 50 | 0 |
| attack50_generalization | attack adv loss | loss2 | 50 | 1.98716e-06 | 4.19417e-06 | 2.20701e-06 | 22.3772 | 1.11949e-27 | 5.75739e-27 | 3.16461 | 50 | 0 |
| attack50_generalization | attack adv loss | Physics Loss | 50 | 1.98716e-06 | 4.69654e-06 | 2.70938e-06 | 29.0416 | 7.69597e-33 | 6.92638e-32 | 4.10711 | 50 | 0 |
| attack50_generalization | attack adv loss | random clean | 50 | 1.98716e-06 | 4.35924e-06 | 2.37207e-06 | 26.4965 | 5.2889e-31 | 3.80801e-30 | 3.74717 | 50 | 0 |
| attack50_generalization | attack adv loss | random solver | 50 | 1.98716e-06 | 5.75595e-06 | 3.76878e-06 | 38.512 | 1.34936e-38 | 1.61923e-37 | 5.44642 | 50 | 0 |
| attack50_generalization | attack loss increase | baseline | 50 | 1.699e-06 | 9.68093e-06 | 7.98193e-06 | 39.1642 | 6.07809e-39 | 1.09406e-37 | 5.53866 | 50 | 0 |
| attack50_generalization | attack loss increase | loss1 | 50 | 1.699e-06 | 3.92706e-06 | 2.22807e-06 | 16.4109 | 7.55221e-22 | 2.26566e-21 | 2.32085 | 50 | 0 |
| attack50_generalization | attack loss increase | loss2 | 50 | 1.699e-06 | 3.56469e-06 | 1.86569e-06 | 16.1186 | 1.59198e-21 | 4.40857e-21 | 2.27951 | 50 | 0 |
| attack50_generalization | attack loss increase | Physics Loss | 50 | 1.699e-06 | 3.97387e-06 | 2.27488e-06 | 18.7637 | 2.57872e-24 | 1.03149e-23 | 2.65359 | 50 | 0 |
| attack50_generalization | attack loss increase | random clean | 50 | 1.699e-06 | 3.69079e-06 | 1.99179e-06 | 18.0431 | 1.38505e-23 | 4.98618e-23 | 2.55168 | 50 | 0 |
| attack50_generalization | attack loss increase | random solver | 50 | 1.699e-06 | 4.82405e-06 | 3.12506e-06 | 21.1879 | 1.27186e-26 | 5.72339e-26 | 2.99642 | 50 | 0 |
| attack50_generalization | attack relative increase | baseline | 50 | 8.48128 | 17.7507 | 9.26937 | 2.6889 | 0.004885 | 0.00764609 | 0.380268 | 26 | 24 |
| attack50_generalization | attack relative increase | loss1 | 50 | 8.48128 | 14.0778 | 5.59654 | 2.63079 | 0.00567749 | 0.00851623 | 0.372049 | 22 | 28 |
| attack50_generalization | attack relative increase | loss2 | 50 | 8.48128 | 16.466 | 7.98472 | 2.89597 | 0.00281666 | 0.00482856 | 0.409552 | 23 | 27 |
| attack50_generalization | attack relative increase | Physics Loss | 50 | 8.48128 | 15.1758 | 6.69449 | 2.85947 | 0.00310882 | 0.00508716 | 0.404391 | 21 | 29 |
| attack50_generalization | attack relative increase | random clean | 50 | 8.48128 | 15.2786 | 6.79734 | 2.95052 | 0.00242719 | 0.00436895 | 0.417266 | 23 | 27 |
| attack50_generalization | attack relative increase | random solver | 50 | 8.48128 | 13.3171 | 4.83578 | 1.88662 | 0.0325724 | 0.0469042 | 0.266808 | 18 | 32 |
| attack50_generalization | delta L2 RMS | baseline | 50 | 3.68202 | 4.19273 | 0.510714 | 17.7573 | 2.73573e-23 | 8.9533e-23 | 2.51126 | 49 | 1 |
| attack50_generalization | delta L2 RMS | loss1 | 50 | 3.68202 | 3.68674 | 0.00471957 | 0.169704 | 0.432971 | 0.529412 | 0.0239998 | 22 | 28 |
| attack50_generalization | delta L2 RMS | loss2 | 50 | 3.68202 | 3.67253 | -0.00948748 | -0.333195 | 0.629796 | 0.629796 | -0.0471209 | 23 | 27 |
| attack50_generalization | delta L2 RMS | Physics Loss | 50 | 3.68202 | 3.69384 | 0.011817 | 0.389531 | 0.349285 | 0.465714 | 0.0550881 | 24 | 26 |
| attack50_generalization | delta L2 RMS | random clean | 50 | 3.68202 | 3.6725 | -0.00952013 | -0.311884 | 0.621774 | 0.629796 | -0.0441071 | 24 | 26 |
| attack50_generalization | delta L2 RMS | random solver | 50 | 3.68202 | 3.72643 | 0.044406 | 1.28772 | 0.101947 | 0.141157 | 0.182111 | 25 | 25 |
| attack50_generalization | delta Linf | baseline | 50 | 9 | 9 | 0 | 0 | 0.5 | 0.529412 | 0 | 0 | 0 |
| attack50_generalization | delta Linf | loss1 | 50 | 9 | 9 | 0 | 0 | 0.5 | 0.529412 | 0 | 0 | 0 |
| attack50_generalization | delta Linf | loss2 | 50 | 9 | 9 | 0 | 0 | 0.5 | 0.529412 | 0 | 0 | 0 |
| attack50_generalization | delta Linf | Physics Loss | 50 | 9 | 9 | 0 | 0 | 0.5 | 0.529412 | 0 | 0 | 0 |
| attack50_generalization | delta Linf | random clean | 50 | 9 | 9 | 0 | 0 | 0.5 | 0.529412 | 0 | 0 | 0 |
| attack50_generalization | delta Linf | random solver | 50 | 9 | 9 | 0 | 0 | 0.5 | 0.529412 | 0 | 0 | 0 |

## Files

- `outputs/darcy_cflow_timematched_organized_release_20260614/data/loss3_ttests_20260615/loss3_vs_second_best_paired_ttests.csv`
- `outputs/darcy_cflow_timematched_organized_release_20260614/data/loss3_ttests_20260615/loss3_vs_all_models_paired_ttests.csv`

## Interpretation Notes

- For clean data MSE, RMSE, and Relative L2, Loss3 is significantly lower than the second-best mean model.
- For attack50 adversarial loss and attack loss increase, Loss3 is also significantly lower than the second-best mean model.
- `delta_linf_mean` is fixed by the epsilon box and all models tie; it is not a meaningful robustness-quality t-test.
- `delta_l2_rms_mean` is a perturbation-size diagnostic, not a direct model-quality metric; Loss3 is not the lowest on that auxiliary quantity.