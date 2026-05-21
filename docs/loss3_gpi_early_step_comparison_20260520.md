# Loss3 GPI Early-Step Perturbation Comparison

Generated: 2026-05-20T18:41:13.969205+00:00

## Scope

Observed from existing trajectory artifacts only; no neural-operator experiment was rerun.
Source run root: `/workspace/NeuralOperatorRobustness2/forensics/loss3_alpha_epsilon_core4_baseline_giftrace_20260520/fno_nu0p001_eps4_alpha0p4_batch100_steps300_p2_q2`
Method visualized: `steepest_replace`.

The saved trajectory arrays are available for selected dataset indices only. These figures compare early GPI/replacement deltas against the same method's final delta and against final deltas from the other core methods.

## Figures

- delta_grid: `forensics/loss3_gpi_early_step_comparison_20260520/figures/gpi_early_delta_grid_steps_1_5_10_20_100_300.png`
- loss_cosine: `forensics/loss3_gpi_early_step_comparison_20260520/figures/gpi_loss_cosine_to_final_selected_samples.png`
- condition panels:
  - `forensics/loss3_gpi_early_step_comparison_20260520/figures/gpi_early_condition_panels/dataset000_gpi_early_steps.png`
  - `forensics/loss3_gpi_early_step_comparison_20260520/figures/gpi_early_condition_panels/dataset007_gpi_early_steps.png`
  - `forensics/loss3_gpi_early_step_comparison_20260520/figures/gpi_early_condition_panels/dataset040_gpi_early_steps.png`
  - `forensics/loss3_gpi_early_step_comparison_20260520/figures/gpi_early_condition_panels/dataset047_gpi_early_steps.png`

## Numeric Table

- Early-step similarity table: `forensics/loss3_gpi_early_step_comparison_20260520/tables/gpi_early_step_similarity.csv`

## Selected Observations

- dataset `0`, k=`5`: loss=`4.183`, cos_to_gpi_final=`0.8627`, cos_to_pgd_final=`0.0300`, cos_to_lp_steepest_final=`0.0298`.
- dataset `0`, k=`10`: loss=`4.326`, cos_to_gpi_final=`0.9989`, cos_to_pgd_final=`0.0370`, cos_to_lp_steepest_final=`0.0369`.
- dataset `0`, k=`100`: loss=`4.323`, cos_to_gpi_final=`1.0000`, cos_to_pgd_final=`0.0379`, cos_to_lp_steepest_final=`0.0378`.
- dataset `0`, k=`300`: loss=`4.323`, cos_to_gpi_final=`1.0000`, cos_to_pgd_final=`0.0379`, cos_to_lp_steepest_final=`0.0378`.
- dataset `7`, k=`5`: loss=`3.952`, cos_to_gpi_final=`0.9071`, cos_to_pgd_final=`0.9705`, cos_to_lp_steepest_final=`0.9705`.
- dataset `7`, k=`10`: loss=`3.917`, cos_to_gpi_final=`0.9992`, cos_to_pgd_final=`0.9759`, cos_to_lp_steepest_final=`0.9759`.
- dataset `7`, k=`100`: loss=`3.906`, cos_to_gpi_final=`1.0000`, cos_to_pgd_final=`0.9762`, cos_to_lp_steepest_final=`0.9762`.
- dataset `7`, k=`300`: loss=`3.906`, cos_to_gpi_final=`1.0000`, cos_to_pgd_final=`0.9762`, cos_to_lp_steepest_final=`0.9762`.
- dataset `40`, k=`5`: loss=`2.765`, cos_to_gpi_final=`0.8966`, cos_to_pgd_final=`0.8691`, cos_to_lp_steepest_final=`0.8658`.
- dataset `40`, k=`10`: loss=`3.286`, cos_to_gpi_final=`0.9382`, cos_to_pgd_final=`0.9864`, cos_to_lp_steepest_final=`0.9854`.
- dataset `40`, k=`100`: loss=`2.785`, cos_to_gpi_final=`0.9441`, cos_to_pgd_final=`0.9385`, cos_to_lp_steepest_final=`0.9385`.
- dataset `40`, k=`300`: loss=`3.178`, cos_to_gpi_final=`1.0000`, cos_to_pgd_final=`0.9713`, cos_to_lp_steepest_final=`0.9718`.
- dataset `47`, k=`5`: loss=`3.074`, cos_to_gpi_final=`0.8807`, cos_to_pgd_final=`0.9606`, cos_to_lp_steepest_final=`0.9606`.
- dataset `47`, k=`10`: loss=`3.129`, cos_to_gpi_final=`0.9997`, cos_to_pgd_final=`0.9753`, cos_to_lp_steepest_final=`0.9753`.
- dataset `47`, k=`100`: loss=`3.13`, cos_to_gpi_final=`1.0000`, cos_to_pgd_final=`0.9755`, cos_to_lp_steepest_final=`0.9755`.
- dataset `47`, k=`300`: loss=`3.13`, cos_to_gpi_final=`1.0000`, cos_to_pgd_final=`0.9755`, cos_to_lp_steepest_final=`0.9755`.


## Aggregate Observations

Observed from the selected saved trajectory samples (`dataset_index` 0, 7, 40, 47):

- Mean `cos(delta_k, delta_300)` for GPI is `0.8868` at k=5, `0.9840` at k=10, `0.9949` at k=20, and `1.0000` at k=300.
- Mean selected-sample loss is `3.4935` at k=5, `3.6642` at k=10, `3.6791` at k=20, `3.5362` at k=100, and `3.6344` at k=300.
- Dataset 40 is the visibly nonmonotone case: it reaches a higher selected-sample loss around k=20 than at k=100 or k=300, while still having high shape cosine to final.
- Cross-method final-delta similarity is high for datasets 7, 40, and 47, but dataset 0 is an outlier where GPI final and PGD/LP-steepest final deltas have low cosine. Treat the "all methods look similar" impression as mostly true for these representative samples, but not universal sample-by-sample.

Inference: for these saved baseline samples, the GPI perturbation shape is already very close to its final k=300 shape by k=10. The loss can still fluctuate after that, so early stopping should be evaluated by both shape similarity and loss stability.

## Interpretation Guardrail

High early cosine means the perturbation shape is already close to the final stored trajectory shape for the selected samples. It does not by itself prove full-batch equivalence; use the table and representative panels together with loss/smoothness diagnostics.

