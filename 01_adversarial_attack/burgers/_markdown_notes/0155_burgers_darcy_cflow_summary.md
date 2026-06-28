# Burgers + Darcy/CFlow Attack Objective Summary

Generated: 2026-06-22 02:47:57 UTC

Scope: completed Burgers 1D and Darcy/CFlow runs only. The still-running NS2D experiment is intentionally excluded.

## Executive conclusion

- Across every completed Burgers setting, optimizing `loss3` gives the largest final true-loss3 mean and wins every paired sample.
- Across every completed Darcy/CFlow setting, optimizing `loss3` also gives the largest final true-loss3 mean; its per-sample win rate rises from 70% at K=100 to 100% at K=875.
- `loss1` and `loss2` are close to each other and often near the clean true-loss3 baseline, while `loss3` scales strongly with budget.

## Completed configurations

Burgers: fixed FNO1D model, nu=0.001, periodic 1D dataset, L2 PGD, 100 steps, 50 samples per setting.

| system | setting | n | loss1 final true-loss3 | loss2 final true-loss3 | loss3 final true-loss3 | loss3 ratio vs clean | loss3 wins | loss3 win rate | best by mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| burgers1d | eps=0.125 | 50 | 0.1027 +/- 0.062 | 0.1021 +/- 0.0624 | 0.1306 +/- 0.0862 | 1.233 | 50 | 1 | loss3 |
| burgers1d | eps=0.25 | 50 | 0.1008 +/- 0.0568 | 0.09994 +/- 0.0578 | 0.1664 +/- 0.114 | 1.563 | 50 | 1 | loss3 |
| burgers1d | eps=0.5 | 50 | 0.09817 +/- 0.0486 | 0.09632 +/- 0.0494 | 0.2793 +/- 0.212 | 2.626 | 50 | 1 | loss3 |
| burgers1d | eps=0.75 | 50 | 0.0976 +/- 0.0448 | 0.09335 +/- 0.0428 | 0.4612 +/- 0.382 | 4.415 | 50 | 1 | loss3 |
| burgers1d | eps=1.0 | 50 | 0.09875 +/- 0.05 | 0.09127 +/- 0.0398 | 0.7262 +/- 0.624 | 7.144 | 50 | 1 | loss3 |

Darcy/CFlow: binary replacement attack, `steepest_replace`, 100 steps, alpha_flips=5, 20 samples per setting.

| system | setting | n | loss1 final true-loss3 | loss2 final true-loss3 | loss3 final true-loss3 | loss3 ratio vs clean | loss3 wins | loss3 win rate | best by mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| darcy_cflow | K=100 | 20 | 0.02537 +/- 0.00741 | 0.02556 +/- 0.00712 | 0.03075 +/- 0.0104 | 1.243 | 14 | 0.7 | loss3 |
| darcy_cflow | K=250 | 20 | 0.02655 +/- 0.00807 | 0.02701 +/- 0.00693 | 0.0403 +/- 0.0142 | 1.64 | 17 | 0.85 | loss3 |
| darcy_cflow | K=437 | 20 | 0.02756 +/- 0.00838 | 0.02845 +/- 0.00681 | 0.04996 +/- 0.0179 | 2.055 | 17 | 0.85 | loss3 |
| darcy_cflow | K=875 | 20 | 0.02877 +/- 0.00859 | 0.03089 +/- 0.00839 | 0.07709 +/- 0.0173 | 3.231 | 20 | 1 | loss3 |

## Figures

![mac_board_main_table](figures/mac_board_main_table.png)

![mac_board_final_true_loss3_means](figures/mac_board_final_true_loss3_means.png)

![mac_board_win_rates](figures/mac_board_win_rates.png)

![mac_board_loss3_ratio_curve](figures/mac_board_loss3_ratio_curve.png)

## Statistical comparison

Positive `loss3_minus_loss*` means the `loss3` attack produced larger final true-loss3.

| system | protocol_id | comparison | n | mean_diff | cohen_dz | paired_ttest_p_value | wilcoxon_p_value | bootstrap_ci95_low | bootstrap_ci95_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| burgers1d | burgers_nu0p001_l2_eps0p125_alpha0p0125_steps100_N50 | loss3_minus_loss1 | 50 | 0.02786 | 1.013 | 3.734e-09 | 1.776e-15 | 0.020975 | 0.035906 |
| burgers1d | burgers_nu0p001_l2_eps0p125_alpha0p0125_steps100_N50 | loss3_minus_loss2 | 50 | 0.028522 | 1.03 | 2.403e-09 | 1.776e-15 | 0.021482 | 0.036715 |
| burgers1d | burgers_nu0p001_l2_eps0p25_alpha0p025_steps100_N50 | loss3_minus_loss1 | 50 | 0.065665 | 1.03 | 2.437e-09 | 1.776e-15 | 0.049028 | 0.084135 |
| burgers1d | burgers_nu0p001_l2_eps0p25_alpha0p025_steps100_N50 | loss3_minus_loss2 | 50 | 0.06651 | 1.033 | 2.252e-09 | 1.776e-15 | 0.050485 | 0.08551 |
| burgers1d | burgers_nu0p001_l2_eps0p5_alpha0p05_steps100_N50 | loss3_minus_loss1 | 50 | 0.1811 | 1.027 | 2.636e-09 | 1.776e-15 | 0.13531 | 0.23314 |
| burgers1d | burgers_nu0p001_l2_eps0p5_alpha0p05_steps100_N50 | loss3_minus_loss2 | 50 | 0.18295 | 1.02 | 3.132e-09 | 1.776e-15 | 0.13704 | 0.23715 |
| burgers1d | burgers_nu0p001_l2_eps0p75_alpha0p075_steps100_N50 | loss3_minus_loss1 | 50 | 0.36361 | 1.026 | 2.657e-09 | 1.776e-15 | 0.27485 | 0.47009 |
| burgers1d | burgers_nu0p001_l2_eps0p75_alpha0p075_steps100_N50 | loss3_minus_loss2 | 50 | 0.36787 | 1.017 | 3.367e-09 | 1.776e-15 | 0.2724 | 0.47154 |
| burgers1d | burgers_nu0p001_l2_eps1p0_alpha0p1_steps100_N50 | loss3_minus_loss1 | 50 | 0.62747 | 1.05 | 1.474e-09 | 1.776e-15 | 0.46977 | 0.79276 |
| burgers1d | burgers_nu0p001_l2_eps1p0_alpha0p1_steps100_N50 | loss3_minus_loss2 | 50 | 0.63496 | 1.039 | 1.933e-09 | 1.776e-15 | 0.47444 | 0.81236 |
| darcy_cflow | darcy_binary_epsfrac0p01_K100_alphaflips5_steps100_methodsteepest_replace_N20 | loss3_minus_loss1 | 20 | 0.0053763 | 1.107 | 8.857e-05 | 0.0001335 | 0.003366 | 0.0075341 |
| darcy_cflow | darcy_binary_epsfrac0p01_K100_alphaflips5_steps100_methodsteepest_replace_N20 | loss3_minus_loss2 | 20 | 0.005184 | 1.035 | 0.0001837 | 0.0007076 | 0.0030651 | 0.0073948 |
| darcy_cflow | darcy_binary_epsfrac0p01_K250_alphaflips5_steps100_methodsteepest_replace_N20 | loss3_minus_loss1 | 20 | 0.013748 | 1.547 | 1.353e-06 | 1.335e-05 | 0.010022 | 0.017418 |
| darcy_cflow | darcy_binary_epsfrac0p01_K250_alphaflips5_steps100_methodsteepest_replace_N20 | loss3_minus_loss2 | 20 | 0.013286 | 1.353 | 8.07e-06 | 2.67e-05 | 0.0092221 | 0.017446 |
| darcy_cflow | darcy_binary_epsfrac0p01_K437_alphaflips5_steps100_methodsteepest_replace_N20 | loss3_minus_loss1 | 20 | 0.022398 | 1.623 | 6.873e-07 | 1.907e-05 | 0.016375 | 0.028247 |
| darcy_cflow | darcy_binary_epsfrac0p01_K437_alphaflips5_steps100_methodsteepest_replace_N20 | loss3_minus_loss2 | 20 | 0.021512 | 1.451 | 3.218e-06 | 2.67e-05 | 0.015255 | 0.027781 |
| darcy_cflow | darcy_binary_epsfrac0p01_K875_alphaflips5_steps100_methodsteepest_replace_N20 | loss3_minus_loss1 | 20 | 0.048327 | 3.212 | 1.175e-11 | 1.907e-06 | 0.041803 | 0.054531 |
| darcy_cflow | darcy_binary_epsfrac0p01_K875_alphaflips5_steps100_methodsteepest_replace_N20 | loss3_minus_loss2 | 20 | 0.046209 | 2.488 | 9.17e-10 | 1.907e-06 | 0.038071 | 0.053614 |

## Files in this bundle

- `tables/main_burgers_darcy_cflow.csv`: compact paper-facing result table.
- `tables/*_burgers_darcy.csv`: filtered source tables for completed Burgers and Darcy/CFlow runs.
- `figures/*.png`: high-resolution board/table figures suitable for pasting into a Mac whiteboard or slide deck.
- `raw_subset/`: raw completed Burgers and Darcy/CFlow run outputs plus logs.
- `burgers_darcy_cflow_summary.tar.gz`: tar.gz archive of this bundle for R2 upload.

## Notes

- The phrase `DataFlow` in the request was treated as Darcy/CFlow, matching the completed result directories.
- Raw NS2D outputs are not included here because that run is still active.
