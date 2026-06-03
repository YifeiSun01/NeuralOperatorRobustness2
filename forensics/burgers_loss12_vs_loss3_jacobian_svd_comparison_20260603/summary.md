# Burgers p2q2 loss1/loss2 vs loss3 Jacobian SVD comparison

All rows use the same representative 20 samples and the same reused solver SVD files.

## Output roots
- loss1_epoch2000: `forensics/burgers_p2q2_loss1_epoch2000_jacobian_svd_rep20_top100_20260603`
- loss2_epoch0900: `forensics/burgers_p2q2_loss2_epoch900_jacobian_svd_rep20_top100_20260603`
- loss3_epoch0200: `forensics/burgers_p2q2_checkpoint_series_jacobian_svd_20260601`
- loss3_epoch0400: `forensics/burgers_p2q2_checkpoint_series_jacobian_svd_20260601`

## Wall-clock aligned model-solver Jacobian error

| pair | split | candidate error mean | reference error mean | reference/candidate | candidate baseline ratio | reference baseline ratio |
|---|---:|---:|---:|---:|---:|---:|
| loss1_epoch2000_vs_loss3_epoch0200 | ALL | 0.5581 | 2.699 | 4.835 | 0.304 | 1.388 |
| loss1_epoch2000_vs_loss3_epoch0200 | train | 0.3024 | 1.391 | 4.598 | 0.3069 | 1.245 |
| loss1_epoch2000_vs_loss3_epoch0200 | test | 0.2909 | 1.514 | 5.204 | 0.4128 | 2.063 |
| loss1_epoch2000_vs_loss3_epoch0200 | generalization | 0.8184 | 3.957 | 4.836 | 0.2587 | 1.204 |
| loss2_epoch0900_vs_loss3_epoch0400 | ALL | 0.7079 | 2.268 | 3.204 | 0.4473 | 1.039 |
| loss2_epoch0900_vs_loss3_epoch0400 | train | 0.5263 | 0.9747 | 1.852 | 0.5153 | 0.7661 |
| loss2_epoch0900_vs_loss3_epoch0400 | test | 0.4461 | 1.117 | 2.504 | 0.6654 | 1.366 |
| loss2_epoch0900_vs_loss3_epoch0400 | generalization | 0.9216 | 3.504 | 3.802 | 0.3193 | 1.072 |

## Direction/subspace checkpoints

| experiment | split | kind | top_k | right subspace mean cosine | left subspace mean cosine |
|---|---|---|---:|---:|---:|
| loss1_epoch2000 | train | model | 10 | 0.9846 | 0.9912 |
| loss1_epoch2000 | train | error | 10 | 0.2967 | 0.4432 |
| loss1_epoch2000 | test | model | 10 | 0.9931 | 0.9981 |
| loss1_epoch2000 | test | error | 10 | 0.3233 | 0.4834 |
| loss1_epoch2000 | generalization | model | 10 | 0.9883 | 0.9888 |
| loss1_epoch2000 | generalization | error | 10 | 0.2984 | 0.2914 |
| loss3_epoch0200 | train | model | 10 | 0.9618 | 0.9534 |
| loss3_epoch0200 | train | error | 10 | 0.5594 | 0.4865 |
| loss3_epoch0200 | test | model | 10 | 0.9742 | 0.9574 |
| loss3_epoch0200 | test | error | 10 | 0.6439 | 0.5454 |
| loss3_epoch0200 | generalization | model | 10 | 0.9629 | 0.9419 |
| loss3_epoch0200 | generalization | error | 10 | 0.5081 | 0.4322 |
| loss2_epoch0900 | train | model | 10 | 0.978 | 0.9834 |
| loss2_epoch0900 | train | error | 10 | 0.3686 | 0.4435 |
| loss2_epoch0900 | test | model | 10 | 0.9923 | 0.9964 |
| loss2_epoch0900 | test | error | 10 | 0.3788 | 0.4831 |
| loss2_epoch0900 | generalization | model | 10 | 0.9769 | 0.9771 |
| loss2_epoch0900 | generalization | error | 10 | 0.3371 | 0.3138 |
| loss3_epoch0400 | train | model | 10 | 0.9785 | 0.9727 |
| loss3_epoch0400 | train | error | 10 | 0.5512 | 0.4582 |
| loss3_epoch0400 | test | model | 10 | 0.9955 | 0.9887 |
| loss3_epoch0400 | test | error | 10 | 0.5948 | 0.4911 |
| loss3_epoch0400 | generalization | model | 10 | 0.9709 | 0.9561 |
| loss3_epoch0400 | generalization | error | 10 | 0.5011 | 0.4174 |

## Rankwise top-10 mean vector alignment

| experiment | split | kind | right absdot mean | left absdot mean | singular value ratio mean |
|---|---|---|---:|---:|---:|
| loss1_epoch2000 | train | model | 0.9801 | 0.9866 | 0.9927 |
| loss1_epoch2000 | train | error | 0.07358 | 0.2201 | 0.2023 |
| loss1_epoch2000 | test | model | 0.986 | 0.9909 | 0.9965 |
| loss1_epoch2000 | test | error | 0.0654 | 0.1802 | 0.1937 |
| loss1_epoch2000 | generalization | model | 0.9234 | 0.9234 | 1.004 |
| loss1_epoch2000 | generalization | error | 0.1694 | 0.1184 | 0.2013 |
| loss3_epoch0200 | train | model | 0.8609 | 0.8507 | 0.9743 |
| loss3_epoch0200 | train | error | 0.2145 | 0.1225 | 0.3415 |
| loss3_epoch0200 | test | model | 0.8255 | 0.8082 | 0.9765 |
| loss3_epoch0200 | test | error | 0.2667 | 0.1317 | 0.3466 |
| loss3_epoch0200 | generalization | model | 0.7562 | 0.7331 | 0.9572 |
| loss3_epoch0200 | generalization | error | 0.2585 | 0.1665 | 0.3749 |
| loss2_epoch0900 | train | model | 0.9708 | 0.9759 | 0.9891 |
| loss2_epoch0900 | train | error | 0.1255 | 0.1037 | 0.2335 |
| loss2_epoch0900 | test | model | 0.9907 | 0.9941 | 0.9913 |
| loss2_epoch0900 | test | error | 0.1609 | 0.1841 | 0.2159 |
| loss2_epoch0900 | generalization | model | 0.8918 | 0.8915 | 0.9951 |
| loss2_epoch0900 | generalization | error | 0.21 | 0.09772 | 0.2313 |
| loss3_epoch0400 | train | model | 0.8656 | 0.8574 | 0.9864 |
| loss3_epoch0400 | train | error | 0.146 | 0.08069 | 0.2227 |
| loss3_epoch0400 | test | model | 0.987 | 0.9763 | 1.004 |
| loss3_epoch0400 | test | error | 0.1478 | 0.09648 | 0.2284 |
| loss3_epoch0400 | generalization | model | 0.706 | 0.6908 | 0.9642 |
| loss3_epoch0400 | generalization | error | 0.2141 | 0.1212 | 0.3122 |

## Short read

- For `(J_model - J_solver)` spectral norm, loss1 epoch2000 is much smaller than loss3 epoch0200 at the wall-clock-aligned comparison point.
- Loss2 epoch0900 is also smaller than loss3 epoch0400, especially on the 10 generalization samples.
- The model Jacobian singular subspaces stay very close to the solver subspaces for these aligned checkpoints; the error-Jacobian directions are less solver-aligned, as expected.

