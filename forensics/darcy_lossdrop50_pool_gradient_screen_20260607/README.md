# Darcy Flow Candidate Gradient-Alignment Screen

Observed from `gradient_alignment_by_step.csv`: cosine is between the parameter gradient from the adversarial training batch and the parameter gradient that would reduce each evaluation dataset loss.
Inference rule: high positive cosine suggests the candidate dataset is first-order compatible with the adversarial-sample training direction; negative cosine suggests it is a poor generalization target for this attack distribution.

## Top First50 Generalization Candidates

| rank | dataset | cosine mean | negative steps | eval loss delta |
| --- | --- | --- | --- | --- |
| 1 | darcy_lossdrop_pool_soft_l3_h12_b8_02 | 0.839886 | 4 | -9.992881e-08 |
| 2 | darcy_lossdrop_pool_soft_l3_h12_b8_03 | 0.839469 | 4 | -1.045716e-07 |
| 3 | darcy_lossdrop_pool_soft_l3_h12_b8_01 | 0.839304 | 4 | -1.051125e-07 |
| 4 | darcy_lossdrop_pool_soft_l3_h12_b6_01 | 0.839177 | 4 | -1.197352e-07 |
| 5 | darcy_lossdrop_pool_soft_l3_h12_b8_04 | 0.836979 | 4 | -1.295550e-07 |
| 6 | darcy_lossdrop_pool_soft_l3_h12_b6_02 | 0.836395 | 4 | -1.480994e-07 |
| 7 | darcy_lossdrop_pool_soft_l3_h12_b6_03 | 0.832987 | 4 | -1.501035e-07 |
| 8 | darcy_lossdrop_pool_soft_l3_h12_b6_04 | 0.807931 | 5 | -1.765184e-07 |
| 9 | darcy_lossdrop_pool_soft_l4_h10p5_b12_07 | 0.800289 | 5 | -1.522935e-07 |
| 10 | darcy_lossdrop_pool_soft_l4_h10p5_b12_04 | 0.788122 | 5 | -1.745421e-07 |

## Files

- `gradient_alignment_by_step.csv`
- `optimizer_microsteps.csv`
- `candidate_screen_summary.csv`
