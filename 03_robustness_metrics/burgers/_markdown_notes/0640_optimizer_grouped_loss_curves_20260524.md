# Optimizer-Grouped Loss Curves - 2026-05-24

Observed from saved `per_step_metrics.csv` files only. No model inference, solver rollout, or attack update was run.

Each panel fixes the attack target loss and overlays the four optimizer methods.

## Figures

- Burgers: [burgers_optimizer_grouped_target_loss_curves_eps4_alpha0p4.png](docs/optimizer_grouped_loss_curves_20260524/burgers_optimizer_grouped_target_loss_curves_eps4_alpha0p4.png)
- NS2D: [ns2d_optimizer_grouped_target_loss_curves_eps32_vs_eps1.png](docs/optimizer_grouped_loss_curves_20260524/ns2d_optimizer_grouped_target_loss_curves_eps32_vs_eps1.png)

## Interpretation

- For `p = 2`, the steepest direction is an L2-normalized gradient direction. The direction is not fundamentally different from PGD; the main difference is normalization and how the update is combined with the existing perturbation.
- `replace` discards the previous perturbation direction and uses the current proposal after projection. When the gradient direction rotates between steps, this can move the perturbation to a very different point on the epsilon-ball boundary, producing visible oscillation.
- `add` keeps memory of the previous perturbation through `delta + alpha * direction`, so projection tends to smooth the trajectory.
- At very small epsilon, the problem is closer to a local linear regime, so add/replace/raw/steepest can look much more similar.

## Roughness Summary

Roughness is mean absolute one-step loss change divided by the curve range; it is a simple diagnostic for jaggedness, not a new objective.

### Burgers

| target | method | final loss | roughness | negative-step fraction | source |
|---|---|---:|---:|---:|---|
| Loss 1 / all W | Raw Add | 6.97733 | 0.01 | 0 | `forensics/loss1_loss2_core4_p2q2_baseline_marked_angles_20260521/fno_nu0p001_eps4_alpha0p4_batch100_steps100_p2_q2/loss1_original/raw_add/per_step_metrics.csv` |
| Loss 1 / all W | Raw Replace | 6.9416 | 0.01028 | 0.34 | `forensics/loss1_loss2_core4_p2q2_baseline_marked_angles_20260521/fno_nu0p001_eps4_alpha0p4_batch100_steps100_p2_q2/loss1_original/raw_replace/per_step_metrics.csv` |
| Loss 1 / all W | Steepest Add | 6.97673 | 0.01 | 0 | `forensics/loss1_loss2_core4_p2q2_baseline_marked_angles_20260521/fno_nu0p001_eps4_alpha0p4_batch100_steps100_p2_q2/loss1_original/steepest_add/per_step_metrics.csv` |
| Loss 1 / all W | Steepest Replace | 6.9416 | 0.01028 | 0.34 | `forensics/loss1_loss2_core4_p2q2_baseline_marked_angles_20260521/fno_nu0p001_eps4_alpha0p4_batch100_steps100_p2_q2/loss1_original/steepest_replace/per_step_metrics.csv` |
| Loss 2 / all A -> W | Raw Add | 7.06981 | 0.01 | 0 | `forensics/loss1_loss2_core4_p2q2_baseline_marked_angles_20260521/fno_nu0p001_eps4_alpha0p4_batch100_steps100_p2_q2/loss2_original/raw_add/per_step_metrics.csv` |
| Loss 2 / all A -> W | Raw Replace | 7.04519 | 0.01041 | 0.47 | `forensics/loss1_loss2_core4_p2q2_baseline_marked_angles_20260521/fno_nu0p001_eps4_alpha0p4_batch100_steps100_p2_q2/loss2_original/raw_replace/per_step_metrics.csv` |
| Loss 2 / all A -> W | Steepest Add | 7.07425 | 0.01 | 0 | `forensics/loss1_loss2_core4_p2q2_baseline_marked_angles_20260521/fno_nu0p001_eps4_alpha0p4_batch100_steps100_p2_q2/loss2_original/steepest_add/per_step_metrics.csv` |
| Loss 2 / all A -> W | Steepest Replace | 7.04519 | 0.01041 | 0.47 | `forensics/loss1_loss2_core4_p2q2_baseline_marked_angles_20260521/fno_nu0p001_eps4_alpha0p4_batch100_steps100_p2_q2/loss2_original/steepest_replace/per_step_metrics.csv` |
| Loss 3 / all W | Raw Add | 2.87698 | 0.01 | 0 | `forensics/loss3_alpha_epsilon_core4_baseline_giftrace_20260520/fno_nu0p001_eps4_alpha0p4_batch100_steps300_p2_q2/raw_add/per_step_metrics.csv` |
| Loss 3 / all W | Raw Replace | 3.04661 | 0.01939 | 0.41 | `forensics/loss3_alpha_epsilon_core4_baseline_giftrace_20260520/fno_nu0p001_eps4_alpha0p4_batch100_steps300_p2_q2/raw_replace/per_step_metrics.csv` |
| Loss 3 / all W | Steepest Add | 2.98832 | 0.01 | 0 | `forensics/loss3_alpha_epsilon_core4_baseline_giftrace_20260520/fno_nu0p001_eps4_alpha0p4_batch100_steps300_p2_q2/steepest_add/per_step_metrics.csv` |
| Loss 3 / all W | Steepest Replace | 3.04661 | 0.01939 | 0.41 | `forensics/loss3_alpha_epsilon_core4_baseline_giftrace_20260520/fno_nu0p001_eps4_alpha0p4_batch100_steps300_p2_q2/steepest_replace/per_step_metrics.csv` |

### NS2D

| case | target | method | final loss | roughness | negative-step fraction | source |
|---|---|---|---:|---:|---:|---|
| Epsilon = 32, Alpha = 10 | Loss 1 / all W | Raw Add | 445.81 | 0.04172 | 0.49 | `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/full_adw_b10_pair_outer_baseline_first_20260522/eps32_alpha10/mode_wwwwwwwwww_p2_q2_20260522_055315_UTC/batch_0000_0009/loss1/raw_add/per_step_metrics.csv` |
| Epsilon = 32, Alpha = 10 | Loss 1 / all W | Raw Replace | 401.017 | 0.04999 | 0.5 | `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/full_adw_b10_pair_outer_baseline_first_20260522/eps32_alpha10/mode_wwwwwwwwww_p2_q2_20260522_055315_UTC/batch_0000_0009/loss1/raw_replace/per_step_metrics.csv` |
| Epsilon = 32, Alpha = 10 | Loss 1 / all W | Steepest Add | 519.401 | 0.02049 | 0.47 | `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/full_adw_b10_pair_outer_baseline_first_20260522/eps32_alpha10/mode_wwwwwwwwww_p2_q2_20260522_055315_UTC/batch_0000_0009/loss1/steepest_add/per_step_metrics.csv` |
| Epsilon = 32, Alpha = 10 | Loss 1 / all W | Steepest Replace | 401.017 | 0.04999 | 0.5 | `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/full_adw_b10_pair_outer_baseline_first_20260522/eps32_alpha10/mode_wwwwwwwwww_p2_q2_20260522_055315_UTC/batch_0000_0009/loss1/steepest_replace/per_step_metrics.csv` |
| Epsilon = 32, Alpha = 10 | Loss 2 / all A -> W | Raw Add | 297.427 | 0.2664 | 0.48 | `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/full_adw_b10_pair_outer_baseline_first_20260522/eps32_alpha10/mode_aaaaaaaaaw_p2_q2_20260522_070858_UTC/batch_0000_0009/loss2/raw_add/per_step_metrics.csv` |
| Epsilon = 32, Alpha = 10 | Loss 2 / all A -> W | Raw Replace | 307.105 | 0.3664 | 0.5 | `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/full_adw_b10_pair_outer_baseline_first_20260522/eps32_alpha10/mode_aaaaaaaaaw_p2_q2_20260522_070858_UTC/batch_0000_0009/loss2/raw_replace/per_step_metrics.csv` |
| Epsilon = 32, Alpha = 10 | Loss 2 / all A -> W | Steepest Add | 291.119 | 0.2997 | 0.53 | `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/full_adw_b10_pair_outer_baseline_first_20260522/eps32_alpha10/mode_aaaaaaaaaw_p2_q2_20260522_070858_UTC/batch_0000_0009/loss2/steepest_add/per_step_metrics.csv` |
| Epsilon = 32, Alpha = 10 | Loss 2 / all A -> W | Steepest Replace | 307.105 | 0.3664 | 0.5 | `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/full_adw_b10_pair_outer_baseline_first_20260522/eps32_alpha10/mode_aaaaaaaaaw_p2_q2_20260522_070858_UTC/batch_0000_0009/loss2/steepest_replace/per_step_metrics.csv` |
| Epsilon = 32, Alpha = 10 | Loss 3 / all W | Raw Add | 155.712 | 0.1089 | 0.46 | `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/full_adw_b10_pair_outer_baseline_first_20260522/eps32_alpha10/mode_wwwwwwwwww_p2_q2_20260522_074135_UTC/batch_0000_0009/loss3/raw_add/per_step_metrics.csv` |
| Epsilon = 32, Alpha = 10 | Loss 3 / all W | Raw Replace | 106.323 | 0.1237 | 0.51 | `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/full_adw_b10_pair_outer_baseline_first_20260522/eps32_alpha10/mode_wwwwwwwwww_p2_q2_20260522_074135_UTC/batch_0000_0009/loss3/raw_replace/per_step_metrics.csv` |
| Epsilon = 32, Alpha = 10 | Loss 3 / all W | Steepest Add | 304.593 | 0.05353 | 0.42 | `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/full_adw_b10_pair_outer_baseline_first_20260522/eps32_alpha10/mode_wwwwwwwwww_p2_q2_20260522_074135_UTC/batch_0000_0009/loss3/steepest_add/per_step_metrics.csv` |
| Epsilon = 32, Alpha = 10 | Loss 3 / all W | Steepest Replace | 106.323 | 0.1237 | 0.51 | `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/full_adw_b10_pair_outer_baseline_first_20260522/eps32_alpha10/mode_wwwwwwwwww_p2_q2_20260522_074135_UTC/batch_0000_0009/loss3/steepest_replace/per_step_metrics.csv` |
| Epsilon = 1, Alpha = 0.3125 | Loss 1 / all W | Raw Add | 77.6735 | 0.01001 | 0.3 | `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/minimal_mechanism_eps4_eps2_eps1_b10_20260523_111410_UTC/eps1_alpha0p3125/mode_wwwwwwwwww_p2_q2_20260523_185512_UTC/batch_0000_0009/loss1/raw_add/per_step_metrics.csv` |
| Epsilon = 1, Alpha = 0.3125 | Loss 1 / all W | Raw Replace | 77.6732 | 0.01 | 0.31 | `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/minimal_mechanism_eps4_eps2_eps1_b10_20260523_111410_UTC/eps1_alpha0p3125/mode_wwwwwwwwww_p2_q2_20260523_185512_UTC/batch_0000_0009/loss1/raw_replace/per_step_metrics.csv` |
| Epsilon = 1, Alpha = 0.3125 | Loss 1 / all W | Steepest Add | 77.4713 | 0.01 | 0 | `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/minimal_mechanism_eps4_eps2_eps1_b10_20260523_111410_UTC/eps1_alpha0p3125/mode_wwwwwwwwww_p2_q2_20260523_185512_UTC/batch_0000_0009/loss1/steepest_add/per_step_metrics.csv` |
| Epsilon = 1, Alpha = 0.3125 | Loss 1 / all W | Steepest Replace | 77.6732 | 0.01 | 0.31 | `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/minimal_mechanism_eps4_eps2_eps1_b10_20260523_111410_UTC/eps1_alpha0p3125/mode_wwwwwwwwww_p2_q2_20260523_185512_UTC/batch_0000_0009/loss1/steepest_replace/per_step_metrics.csv` |
| Epsilon = 1, Alpha = 0.3125 | Loss 2 / all A -> W | Raw Add | 320.527 | 0.01 | 0.02 | `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/eps1_missing_loss2_loss3_b10_20260524_024533_UTC/eps1_alpha0p3125/mode_aaaaaaaaaw_p2_q2_20260524_024538_UTC/batch_0000_0009/loss2/raw_add/per_step_metrics.csv` |
| Epsilon = 1, Alpha = 0.3125 | Loss 2 / all A -> W | Raw Replace | 308.468 | 0.9894 | 0.5 | `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/eps1_missing_loss2_loss3_b10_20260524_024533_UTC/eps1_alpha0p3125/mode_aaaaaaaaaw_p2_q2_20260524_024538_UTC/batch_0000_0009/loss2/raw_replace/per_step_metrics.csv` |
| Epsilon = 1, Alpha = 0.3125 | Loss 2 / all A -> W | Steepest Add | 320.526 | 0.6233 | 0.31 | `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/eps1_missing_loss2_loss3_b10_20260524_024533_UTC/eps1_alpha0p3125/mode_aaaaaaaaaw_p2_q2_20260524_024538_UTC/batch_0000_0009/loss2/steepest_add/per_step_metrics.csv` |
| Epsilon = 1, Alpha = 0.3125 | Loss 2 / all A -> W | Steepest Replace | 308.468 | 0.9894 | 0.5 | `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/eps1_missing_loss2_loss3_b10_20260524_024533_UTC/eps1_alpha0p3125/mode_aaaaaaaaaw_p2_q2_20260524_024538_UTC/batch_0000_0009/loss2/steepest_replace/per_step_metrics.csv` |
| Epsilon = 1, Alpha = 0.3125 | Loss 3 / all W | Raw Add | 75.6973 | 0.01007 | 0.46 | `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/eps1_missing_loss2_loss3_b10_20260524_024533_UTC/eps1_alpha0p3125/mode_wwwwwwwwww_p2_q2_20260524_031339_UTC/batch_0000_0009/loss3/raw_add/per_step_metrics.csv` |
| Epsilon = 1, Alpha = 0.3125 | Loss 3 / all W | Raw Replace | 75.5452 | 0.0161 | 0.49 | `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/eps1_missing_loss2_loss3_b10_20260524_024533_UTC/eps1_alpha0p3125/mode_wwwwwwwwww_p2_q2_20260524_031339_UTC/batch_0000_0009/loss3/raw_replace/per_step_metrics.csv` |
| Epsilon = 1, Alpha = 0.3125 | Loss 3 / all W | Steepest Add | 75.6979 | 0.01006 | 0.4 | `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/eps1_missing_loss2_loss3_b10_20260524_024533_UTC/eps1_alpha0p3125/mode_wwwwwwwwww_p2_q2_20260524_031339_UTC/batch_0000_0009/loss3/steepest_add/per_step_metrics.csv` |
| Epsilon = 1, Alpha = 0.3125 | Loss 3 / all W | Steepest Replace | 75.5452 | 0.0161 | 0.49 | `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/eps1_missing_loss2_loss3_b10_20260524_024533_UTC/eps1_alpha0p3125/mode_wwwwwwwwww_p2_q2_20260524_031339_UTC/batch_0000_0009/loss3/steepest_replace/per_step_metrics.csv` |
