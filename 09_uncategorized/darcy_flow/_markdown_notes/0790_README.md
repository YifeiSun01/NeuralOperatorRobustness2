# Darcy Flow Candidate Gradient-Alignment Screen

Observed from `gradient_alignment_by_step.csv`: cosine is between the parameter gradient from the adversarial training batch and the parameter gradient that would reduce each evaluation dataset loss.
Inference rule: high positive cosine suggests the candidate dataset is first-order compatible with the adversarial-sample training direction; negative cosine suggests it is a poor generalization target for this attack distribution.

## Top First50 Generalization Candidates

| rank | dataset | cosine mean | negative steps | eval loss delta |
| --- | --- | --- | --- | --- |
| 1 | darcy_r03_soft_beta12_seed | 0.880580 | 3 | 3.203709e-09 |
| 2 | darcy_r03_contrast_low4_high11p25 | 0.874474 | 3 | -4.724270e-08 |
| 3 | darcy_r03_contrast_low3p75_high10p75 | 0.855212 | 4 | -7.073240e-08 |
| 4 | darcy_r03_contrast_low4_high10p75 | 0.849130 | 4 | -5.766581e-08 |
| 5 | darcy_r03_contrast_low4_high10p5 | 0.841274 | 4 | -6.482272e-08 |
| 6 | darcy_r03_contrast_low4p25_high10p75 | 0.841068 | 4 | -8.489579e-08 |
| 7 | darcy_r03_contrast_low4p5_high10p5 | 0.837256 | 4 | -1.469781e-07 |
| 8 | darcy_r03_soft_low3p5_high11_beta12 | 0.812573 | 5 | -1.140543e-07 |
| 9 | darcy_r03_soft_low4_high11_beta12 | 0.804488 | 5 | -1.659078e-07 |
| 10 | darcy_r03_soft_low4_high10p5_beta12 | 0.767692 | 6 | -1.768891e-07 |

## Files

- `gradient_alignment_by_step.csv`
- `optimizer_microsteps.csv`
- `candidate_screen_summary.csv`
