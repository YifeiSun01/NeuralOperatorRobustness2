# Darcy/SIR20 Final 52-Dataset Evaluation

Observed from baseline plus final trained checkpoints on screen train/test and the 50 selected lossdrop50 soft-coefficient generalization datasets.

| method | split | datasets | RMSE mean | Relative L2 mean | accuracy score |
|---|---|---:|---:|---:|---:|
| baseline | train | 1 | 0.00019514086 | 0.028983466 | 97.1833 |
| baseline | test | 1 | 0.00051843767 | 0.0692555 | 93.523 |
| baseline | generalization | 50 | 0.0005867155 | 0.097173241 | 91.1769 |
| loss1 | train | 1 | 0.00042435849 | 0.063028213 | 94.0709 |
| loss1 | test | 1 | 0.00035763485 | 0.047774654 | 95.4404 |
| loss1 | generalization | 50 | 0.00084920592 | 0.14063293 | 87.7052 |
| loss2 | train | 1 | 0.00035955985 | 0.053403938 | 94.9303 |
| loss2 | test | 1 | 0.00039936472 | 0.05334914 | 94.9353 |
| loss2 | generalization | 50 | 0.00078045198 | 0.12925052 | 88.5868 |
| loss3 | train | 1 | 0.00036650486 | 0.054435453 | 94.8375 |
| loss3 | test | 1 | 0.00081657706 | 0.10908245 | 90.1646 |
| loss3 | generalization | 50 | 0.00042530934 | 0.07025369 | 93.4503 |
| physics_loss | train | 1 | 0.00039200946 | 0.058223545 | 94.498 |
| physics_loss | test | 1 | 0.00087904161 | 0.11742678 | 89.4913 |
| physics_loss | generalization | 50 | 0.00035042803 | 0.057945988 | 94.5438 |
| random_clean | train | 1 | 0.00030706662 | 0.045607336 | 95.6382 |
| random_clean | test | 1 | 0.00084379743 | 0.11271869 | 89.87 |
| random_clean | generalization | 50 | 0.00049737021 | 0.082255988 | 92.424 |
| random_solver | train | 1 | 0.00039509854 | 0.058682354 | 94.457 |
| random_solver | test | 1 | 0.00036496762 | 0.048754202 | 95.3512 |
| random_solver | generalization | 50 | 0.00082439769 | 0.13652241 | 88.0219 |
| baseline | ALL | 52 | 0.00057787218 | 0.095325019 | 91.3376 |
| loss1 | ALL | 52 | 0.00083158249 | 0.1373548 | 87.9764 |
| loss2 | ALL | 52 | 0.0007650293 | 0.12633229 | 88.8309 |
| loss3 | ALL | 52 | 0.00043170287 | 0.0706962 | 93.4138 |
| physics_loss | ALL | 52 | 0.00036139332 | 0.059095187 | 94.4457 |
| random_clean | ALL | 52 | 0.00050037259 | 0.082137028 | 92.4367 |
| random_solver | ALL | 52 | 0.00080730674 | 0.13333764 | 88.2866 |

## Files

- Metrics CSV: `outputs/darcy_sir20_timematched_full_20260614_smoke_initial/data/final_eval_metrics.csv`
