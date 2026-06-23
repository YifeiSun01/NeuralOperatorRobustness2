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
- `analysis_outputs/attack_objective_true_loss3_comparison_20260622/tables/auxiliary_physics_summary.csv`
- `analysis_outputs/attack_objective_true_loss3_comparison_20260622/tables/with_darcy_physics_as_loss4/paper_ready_main_table_with_darcy_loss4_physics.csv`
- `analysis_outputs/attack_objective_true_loss3_comparison_20260622/tables/with_darcy_physics_as_loss4/winner_summary_darcy_four_objectives.csv`

All numeric entries below are `mean +/- std`. No variance is shown in this
Markdown table. The table below is filtered to `steps=100`; earlier `steps=50`
NS2D sweep rows are not shown.

## Main Table

| System | Protocol | loss1 | loss2 | loss3 | loss4 physics | L3 wins |
| --- | --- | --- | --- | --- | --- | --- |
| Burgers | nu=0.001 L2 eps=0.125 alpha=0.0125 steps=100 N=50 | 0.1027 +/- 0.0620 | 0.1021 +/- 0.0624 | 0.1306 +/- 0.0862 | - | 50/50 |
| Burgers | nu=0.001 L2 eps=0.25 alpha=0.025 steps=100 N=50 | 0.1008 +/- 0.0568 | 0.0999 +/- 0.0578 | 0.1664 +/- 0.1136 | - | 50/50 |
| Burgers | nu=0.001 L2 eps=0.5 alpha=0.05 steps=100 N=50 | 0.0982 +/- 0.0486 | 0.0963 +/- 0.0494 | 0.2793 +/- 0.2118 | - | 50/50 |
| Burgers | nu=0.001 L2 eps=0.75 alpha=0.075 steps=100 N=50 | 0.0976 +/- 0.0448 | 0.0934 +/- 0.0428 | 0.4612 +/- 0.3825 | - | 50/50 |
| Burgers | nu=0.001 L2 eps=1.0 alpha=0.1 steps=100 N=50 | 0.0988 +/- 0.0500 | 0.0913 +/- 0.0398 | 0.7262 +/- 0.6240 | - | 50/50 |
| Darcy/CFlow | binary K=100 alpha_flips=5 steps=100 N=20 | 0.0254 +/- 0.0074 | 0.0256 +/- 0.0071 | 0.0307 +/- 0.0104 | 0.0245 +/- 0.0071 | 14/20 |
| Darcy/CFlow | binary K=250 alpha_flips=5 steps=100 N=20 | 0.0266 +/- 0.0081 | 0.0270 +/- 0.0069 | 0.0403 +/- 0.0142 | 0.0245 +/- 0.0071 | 17/20 |
| Darcy/CFlow | binary K=437 alpha_flips=5 steps=100 N=20 | 0.0276 +/- 0.0084 | 0.0284 +/- 0.0068 | 0.0500 +/- 0.0179 | 0.0244 +/- 0.0071 | 17/20 |
| Darcy/CFlow | binary K=875 alpha_flips=5 steps=100 N=20 | 0.0288 +/- 0.0086 | 0.0309 +/- 0.0084 | 0.0771 +/- 0.0173 | 0.0245 +/- 0.0070 | 20/20 |
| NS2D | p=q=2 eps=32 alpha=1 steps=100 N=20 | 136.827 +/- 56.634 | 74.524 +/- 28.541 | 289.772 +/- 46.151 | - | 20/20 |

## Conclusion

Observed evidence:

- Burgers: `loss3` is the strongest attack objective by final true Loss3 for all
  five L2 budgets. Its per-sample win rate is 50/50 at every budget.
- Darcy/CFlow: when `loss4_physics` is included as the fourth objective,
  `loss3` remains the strongest objective by final true Loss3 for every tested
  binary flip budget. `loss4_physics` has 0/20 wins at every tested budget under
  the final true Loss3 evaluation metric.
- NS2D: after filtering this report to `steps=100`, the retained row is
  eps=32, alpha=1, N=20; `loss3` wins 20/20 samples.

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
