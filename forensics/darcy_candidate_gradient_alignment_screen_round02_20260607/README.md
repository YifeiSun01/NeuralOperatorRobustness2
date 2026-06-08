# Darcy Flow Candidate Gradient-Alignment Screen

Observed from `gradient_alignment_by_step.csv`: cosine is between the parameter gradient from the adversarial training batch and the parameter gradient that would reduce each evaluation dataset loss.
Inference rule: high positive cosine suggests the candidate dataset is first-order compatible with the adversarial-sample training direction; negative cosine suggests it is a poor generalization target for this attack distribution.

## Top First50 Generalization Candidates

| rank | dataset | cosine mean | negative steps | eval loss delta |
| --- | --- | --- | --- | --- |
| 1 | darcy_r02_soft_beta14 | 0.882016 | 3 | 1.248748e-08 |
| 2 | darcy_r02_contrast_low3p5_high10p5 | 0.881670 | 3 | 1.279487e-08 |
| 3 | darcy_r02_contrast_low4_high11 | 0.873782 | 3 | -3.474140e-08 |
| 4 | darcy_r02_soft_beta10 | 0.846222 | 4 | -7.597235e-08 |
| 5 | darcy_r02_soft_tau5_beta8 | 0.842955 | 4 | -7.409177e-08 |
| 6 | darcy_r02_soft_tau4_beta8 | 0.839810 | 4 | -1.062162e-07 |
| 7 | darcy_r02_contrast_low4p5_high10 | 0.835784 | 4 | -1.280352e-07 |
| 8 | darcy_r02_soft_beta8 | 0.834236 | 4 | -1.416262e-07 |
| 9 | darcy_r02_soft_low3p5_high11_beta8 | 0.794071 | 5 | -1.594383e-07 |
| 10 | darcy_r02_contrast_low4_high9p5 | 0.763539 | 6 | -1.363842e-07 |

## Files

- `gradient_alignment_by_step.csv`
- `optimizer_microsteps.csv`
- `candidate_screen_summary.csv`
