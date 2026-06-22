# Attack Objective Summary: Burgers, NS2D, and Darcy/CFlow

Date: 2026-06-22

This note records the current attack-objective comparison across the three
systems. The metric in the objective columns is always final true Loss3 after
attacking with that objective:

`true Loss3 = ||F_theta(x_adv) - G(x_adv)||`

Burgers and NS2D have three attack objectives, `loss1`, `loss2`, and `loss3`.
Darcy/CFlow has four attack objectives in this table: `loss1`, `loss2`,
`loss3`, and `loss4_physics`. Darcy/CFlow uses a binary coefficient flip
constraint, not ordinary continuous PGD.

Observed source files:

- `analysis_outputs/attack_objective_true_loss3_comparison_20260622/tables/objective_summary.csv`
- `analysis_outputs/attack_objective_true_loss3_comparison_20260622/tables/winner_summary.csv`
- `analysis_outputs/attack_objective_true_loss3_comparison_20260622/tables/paired_tests.csv`
- `analysis_outputs/attack_objective_true_loss3_comparison_20260622/tables/auxiliary_physics_summary.csv`
- `analysis_outputs/attack_objective_true_loss3_comparison_20260622/tables/with_darcy_physics_as_loss4/paper_ready_main_table_with_darcy_loss4_physics.csv`
- `analysis_outputs/attack_objective_true_loss3_comparison_20260622/tables/with_darcy_physics_as_loss4/winner_summary_darcy_four_objectives.csv`

All numeric entries below are `mean +/- std`. No variance is shown in this
Markdown table.

## Main Table

| System | Protocol | loss1 | loss2 | loss3 | loss4 physics | Best mean | Wins | p L3-L1 | p L3-L2 | p L3-L4 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Burgers | nu=0.001 L2 eps=0.125 alpha=0.0125 steps=100 N=50 | 0.1027 +/- 0.0620 | 0.1021 +/- 0.0624 | 0.1306 +/- 0.0862 | - | loss3 | loss3 50/50 | 3.73e-09 | 2.40e-09 | - |
| Burgers | nu=0.001 L2 eps=0.25 alpha=0.025 steps=100 N=50 | 0.1008 +/- 0.0568 | 0.0999 +/- 0.0578 | 0.1664 +/- 0.1136 | - | loss3 | loss3 50/50 | 2.44e-09 | 2.25e-09 | - |
| Burgers | nu=0.001 L2 eps=0.5 alpha=0.05 steps=100 N=50 | 0.0982 +/- 0.0486 | 0.0963 +/- 0.0494 | 0.2793 +/- 0.2118 | - | loss3 | loss3 50/50 | 2.64e-09 | 3.13e-09 | - |
| Burgers | nu=0.001 L2 eps=0.75 alpha=0.075 steps=100 N=50 | 0.0976 +/- 0.0448 | 0.0934 +/- 0.0428 | 0.4612 +/- 0.3825 | - | loss3 | loss3 50/50 | 2.66e-09 | 3.37e-09 | - |
| Burgers | nu=0.001 L2 eps=1.0 alpha=0.1 steps=100 N=50 | 0.0988 +/- 0.0500 | 0.0913 +/- 0.0398 | 0.7262 +/- 0.6240 | - | loss3 | loss3 50/50 | 1.47e-09 | 1.93e-09 | - |
| Darcy/CFlow | binary K=100 alpha_flips=5 steps=100 N=20 | 0.0254 +/- 0.0074 | 0.0256 +/- 0.0071 | 0.0307 +/- 0.0104 | 0.0245 +/- 0.0071 | loss3 | L1 2, L2 4, L3 14, L4 0 | 8.86e-05 | 0.0002 | 1.81e-05 |
| Darcy/CFlow | binary K=250 alpha_flips=5 steps=100 N=20 | 0.0266 +/- 0.0081 | 0.0270 +/- 0.0069 | 0.0403 +/- 0.0142 | 0.0245 +/- 0.0071 | loss3 | L1 2, L2 1, L3 17, L4 0 | 1.35e-06 | 8.07e-06 | 7.01e-07 |
| Darcy/CFlow | binary K=437 alpha_flips=5 steps=100 N=20 | 0.0276 +/- 0.0084 | 0.0284 +/- 0.0068 | 0.0500 +/- 0.0179 | 0.0244 +/- 0.0071 | loss3 | L1 2, L2 1, L3 17, L4 0 | 6.87e-07 | 3.22e-06 | 2.16e-07 |
| Darcy/CFlow | binary K=875 alpha_flips=5 steps=100 N=20 | 0.0288 +/- 0.0086 | 0.0309 +/- 0.0084 | 0.0771 +/- 0.0173 | 0.0245 +/- 0.0070 | loss3 | L1 0, L2 0, L3 20, L4 0 | 1.18e-11 | 9.17e-10 | 1.03e-12 |
| NS2D | p=q=2 eps=8 alpha=0.25 steps=50 N=20 | 64.979 +/- 28.525 | 59.567 +/- 28.883 | 126.832 +/- 32.477 | - | loss3 | loss3 20/20 | 2.76e-12 | 2.08e-16 | - |
| NS2D | p=q=2 eps=16 alpha=0.5 steps=50 N=20 | 68.913 +/- 26.433 | 61.469 +/- 30.800 | 186.683 +/- 35.291 | - | loss3 | loss3 20/20 | 5.45e-14 | 1.45e-17 | - |
| NS2D | p=q=2 eps=32 alpha=1 steps=50 N=20 | 118.588 +/- 46.057 | 69.505 +/- 30.926 | 276.760 +/- 44.681 | - | loss3 | loss3 20/20 | 6.20e-11 | 8.31e-17 | - |
| NS2D | p=q=2 eps=32 alpha=1 steps=100 N=20 | 136.827 +/- 56.634 | 74.524 +/- 28.541 | 289.772 +/- 46.151 | - | loss3 | loss3 20/20 | 6.99e-09 | 3.23e-16 | - |
| NS2D | p=q=2 eps=64 alpha=2 steps=50 N=20 | 190.810 +/- 90.681 | 88.554 +/- 25.322 | 324.618 +/- 61.349 | - | loss3 | loss3 18/20 | 2.36e-05 | 1.90e-13 | - |

## Darcy/CFlow Four-Objective Winner Summary

| Darcy protocol | n | loss1 wins | loss2 wins | loss3 wins | loss4 physics wins | loss3 win rate | loss4 win rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| binary K=100 alpha_flips=5 steps=100 N=20 | 20 | 2 | 4 | 14 | 0 | 70% | 0% |
| binary K=250 alpha_flips=5 steps=100 N=20 | 20 | 2 | 1 | 17 | 0 | 85% | 0% |
| binary K=437 alpha_flips=5 steps=100 N=20 | 20 | 2 | 1 | 17 | 0 | 85% | 0% |
| binary K=875 alpha_flips=5 steps=100 N=20 | 20 | 0 | 0 | 20 | 0 | 100% | 0% |

## Darcy/CFlow Physics Objective Diagnostic

The `loss4_physics` objective is useful as a physics diagnostic. It strongly
increases the PDE residual/physics loss, but in these runs it does not maximize
the final true Loss3 metric.

Observed from
`analysis_outputs/attack_objective_true_loss3_comparison_20260622/tables/auxiliary_physics_summary.csv`:

| Darcy protocol | clean physics | final physics | physics increase | final true Loss3 |
| --- | --- | --- | --- | --- |
| K=100 alpha_flips=5 steps=100 N=20 | 18.034 | 21.978 +/- 4.433 | 3.945 | 0.0245 +/- 0.0071 |
| K=250 alpha_flips=5 steps=100 N=20 | 18.034 | 29.291 +/- 6.633 | 11.257 | 0.0245 +/- 0.0071 |
| K=437 alpha_flips=5 steps=100 N=20 | 18.034 | 36.662 +/- 8.280 | 18.628 | 0.0244 +/- 0.0071 |
| K=875 alpha_flips=5 steps=100 N=20 | 18.034 | 50.688 +/- 9.619 | 32.654 | 0.0245 +/- 0.0070 |

## Conclusion

Observed evidence:

- Burgers: `loss3` is the strongest attack objective by final true Loss3 for all
  five L2 budgets. Its per-sample win rate is 50/50 at every budget.
- Darcy/CFlow: when `loss4_physics` is included as the fourth objective,
  `loss3` remains the strongest objective by final true Loss3 for every tested
  binary flip budget. `loss4_physics` has 0/20 wins at every tested budget under
  the final true Loss3 evaluation metric.
- NS2D: `loss3` is strongest by mean final true Loss3 across all tested budgets,
  with 20/20 wins except the largest listed budget, where it has 18/20 wins.

Inference from the observed tables:

- For the true Loss3 attack target, `loss3` is currently the consistent winner
  across Burgers, NS2D, and Darcy/CFlow.
- Darcy/CFlow physics loss should be reported as a fourth Darcy objective or as
  a physics diagnostic, depending on the table framing. It should not be
  described as winning the true Loss3 attack comparison in these runs.

Remaining work:

- The separate four-optimizer ablation is still running under
  `analysis_outputs/optimizer_ablation_20260622/` and is not included in this
  table.
