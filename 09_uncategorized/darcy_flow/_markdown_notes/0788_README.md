# Darcy Flow Candidate Gradient-Alignment Screen

Observed from `gradient_alignment_by_step.csv`: cosine is between the parameter gradient from the adversarial training batch and the parameter gradient that would reduce each evaluation dataset loss.
Inference rule: high positive cosine suggests the candidate dataset is first-order compatible with the adversarial-sample training direction; negative cosine suggests it is a poor generalization target for this attack distribution.

## Top First50 Generalization Candidates

| rank | dataset | cosine mean | negative steps | eval loss delta |
| --- | --- | --- | --- | --- |
| 1 | darcy_screen_mid_tau5 | 0.999412 | 0 | 1.875534e-07 |
| 2 | darcy_screen_mid_alpha2p5_tau3 | 0.988405 | 0 | 1.742665e-07 |
| 3 | darcy_screen_mid_alpha1p5_tau3 | 0.960054 | 1 | 2.088597e-07 |
| 4 | darcy_screen_mid_tau1p8 | 0.959570 | 1 | 2.058244e-07 |
| 5 | darcy_screen_near_tau3p5 | 0.957481 | 1 | 2.165926e-07 |
| 6 | darcy_screen_near_tau2p5 | 0.932928 | 2 | 2.256141e-07 |
| 7 | darcy_screen_soft_beta12 | 0.879517 | 3 | -2.356520e-08 |
| 8 | darcy_screen_soft_beta6 | 0.825345 | 4 | -1.665820e-07 |
| 9 | darcy_screen_contrast_low4_high10 | 0.804450 | 5 | -1.159789e-07 |
| 10 | darcy_screen_contrast_low2_high14 | -0.102057 | 28 | 8.911378e-07 |

## Files

- `gradient_alignment_by_step.csv`
- `optimizer_microsteps.csv`
- `candidate_screen_summary.csv`
