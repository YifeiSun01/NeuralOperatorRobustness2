# Darcy Stage2 Boundary Check - 2026-06-12

- CSV: `analysis_outputs/darcy_stage2_boundary_check_20260612/boundary_split_summary.csv`
- Plot: `visualizations/darcy_stage2_boundary_check_20260612/darcy_stage2_boundary_relative_l2_zoom.png`

Result: stage1 final evaluation and stage2 resume-before-training evaluation are identical for every method/split checked. The visible boundary change comes from the first stage2 optimizer update, not from restarting from the original Darcy checkpoint.

## Relative L2 Boundary Table

| method | split | resume_epoch | stage1_final_relative_l2 | stage2_resume_relative_l2 | resume_minus_stage1_relative_l2 | stage2_epoch_plus1_relative_l2 | epoch_plus1_minus_resume_relative_l2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| loss1 | ALL | 1000 | 0.0982467585 | 0.0982467585 | 0 | 0.0725994271 | -0.0256473314 |
| loss1 | train | 1000 | 0.0251091366 | 0.0251091366 | 0 | 0.0401799395 | 0.0150708029 |
| loss1 | test | 1000 | 0.0325288461 | 0.0325288461 | 0 | 0.0406008792 | 0.00807203306 |
| loss1 | generalization | 1000 | 0.101023869 | 0.101023869 | 0 | 0.0738877878 | -0.0271360814 |
| loss2 | ALL | 1026 | 0.0840563763 | 0.0840563763 | 0 | 0.0960146298 | 0.0119582535 |
| loss2 | train | 1026 | 0.0227408198 | 0.0227408198 | 0 | 0.0442834646 | 0.0215426448 |
| loss2 | test | 1026 | 0.0268985576 | 0.0268985576 | 0 | 0.0456237901 | 0.0187252325 |
| loss2 | generalization | 1026 | 0.0864258438 | 0.0864258438 | 0 | 0.0980570699 | 0.0116312261 |
| loss3 | ALL | 1011 | 0.0716762016 | 0.0716762016 | 0 | 0.0630824464 | -0.00859375515 |
| loss3 | train | 1011 | 0.0391557526 | 0.0391557526 | 0 | 0.0688543441 | 0.0296985915 |
| loss3 | test | 1011 | 0.0464615969 | 0.0464615969 | 0 | 0.0709107828 | 0.0244491859 |
| loss3 | generalization | 1011 | 0.0728309027 | 0.0728309027 | 0 | 0.0628104418 | -0.0100204609 |
| physics | ALL | 1040 | 0.0896496148 | 0.0896496148 | 0 | 0.0912104497 | 0.0015608349 |
| physics | train | 1040 | 0.0218624859 | 0.0218624859 | 0 | 0.0474724355 | 0.0256099496 |
| physics | test | 1040 | 0.0308495363 | 0.0308495363 | 0 | 0.0529750198 | 0.0221254836 |
| physics | generalization | 1040 | 0.0921813589 | 0.0921813589 | 0 | 0.0928499185 | 0.000668559628 |
