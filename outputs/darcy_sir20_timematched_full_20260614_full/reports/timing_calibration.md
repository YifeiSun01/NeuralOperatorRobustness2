# Darcy/SIR20 Timing Calibration

Observed from short calibration runs. Work-clock seconds include attack/random/solver-target generation plus forward/backward/optimizer work, and exclude evaluation/checkpoint/plot/upload.

- Loss3 reference epochs: `3000`
- Loss3 stable sec/epoch: `4.624874`
- Loss3 3000-epoch work-clock budget: `13874.623` seconds (`3.854` hours)

| method | calibration epochs | stable sec/epoch | time-matched epochs | projected work seconds |
|---|---:|---:|---:|---:|
| loss1 | 20 | 4.926671 | 2816 | 13873.505 |
| loss2 | 20 | 4.375749 | 3171 | 13875.499 |
| loss3 | 20 | 4.624874 | 3000 | 13874.623 |
| physics_loss | 20 | 4.033846 | 3440 | 13876.431 |
| random_clean | 20 | 2.891699 | 4798 | 13874.370 |
| random_solver | 20 | 3.010654 | 4609 | 13876.102 |

## Files

- CSV: `outputs/darcy_sir20_timematched_full_20260614_full/data/timing_calibration.csv`
- JSON: `outputs/darcy_sir20_timematched_full_20260614_full/data/timing_calibration.json`
