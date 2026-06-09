# Darcy Flow Candidate Gradient-Alignment Screen

Observed from `gradient_alignment_by_step.csv`: cosine is between the parameter gradient from the adversarial training batch and the parameter gradient that would reduce each evaluation dataset loss.
Inference rule: high positive cosine suggests the candidate dataset is first-order compatible with the adversarial-sample training direction; negative cosine suggests it is a poor generalization target for this attack distribution.

## Top First50 Generalization Candidates

| rank | dataset | cosine mean | negative steps | eval loss delta |
| --- | --- | --- | --- | --- |
| 1 | darcy_lossdrop_pool_soft_l3_h12_b8_03 | 0.879581 | 3 | -5.674352e-08 |
| 2 | darcy_lossdrop_pool_soft_l3_h12_b8_02 | 0.878730 | 3 | -9.856730e-08 |
| 3 | darcy_lossdrop_pool_soft_l3_h12_b8_04 | 0.878595 | 3 | -9.186673e-08 |
| 4 | darcy_lossdrop_pool_soft_l3_h12_b6_02 | 0.877775 | 3 | -1.172274e-07 |
| 5 | darcy_lossdrop_pool_soft_l3_h12_b8_01 | 0.877502 | 3 | -1.337545e-07 |
| 6 | darcy_lossdrop_pool_soft_l3_h12_b6_01 | 0.876909 | 3 | -1.453162e-07 |
| 7 | darcy_lossdrop_pool_soft_l3_h12_b6_04 | 0.866127 | 3 | -1.858915e-07 |
| 8 | darcy_lossdrop_pool_soft_l3_h12_b6_03 | 0.865617 | 3 | -1.836967e-07 |
| 9 | darcy_lossdrop_pool_soft_l4_h10_b14_02 | 0.844586 | 4 | -1.201185e-07 |
| 10 | darcy_lossdrop_pool_soft_l4_h10p5_b12_02 | 0.844402 | 4 | -1.567694e-07 |

## Files

- `gradient_alignment_by_step.csv`
- `optimizer_microsteps.csv`
- `candidate_screen_summary.csv`
