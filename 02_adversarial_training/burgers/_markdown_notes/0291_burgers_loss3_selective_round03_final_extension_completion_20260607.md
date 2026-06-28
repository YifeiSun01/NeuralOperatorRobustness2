# Burgers Round03 Final Extension Completion - 2026-06-07

Status: completed end-to-end. The workflow trained loss1/loss2/loss3 to the requested final epochs, ran final-checkpoint Jacobian/SVD, and refreshed stitched single-run and comparison plots.

## Completion Evidence

Observed from `adversarial_training_runs/burgers_loss3_selective_round03_loss123_final_extension_20260606_logs/driver.log`:

- loss1 `3000 -> 5000` completed at `2026-06-06T17:36:15Z`.
- loss2 `1000 -> 2000` completed at `2026-06-06T23:56:56Z`.
- loss3 `1000 -> 1500` completed at `2026-06-07T06:16:09Z`.
- final-extension Jacobian/SVD completed at `2026-06-07T09:27:18Z`.
- final-extension plots completed at `2026-06-07T09:29:28Z`.
- final-extension workflow completed at `2026-06-07T09:29:28Z`.

No active training, SVD, or plotting process remained at inspection.

## Training Outputs

Run summaries and final checkpoints:

- loss1 epoch5000 summary: `adversarial_training_runs/burgers_loss3_selective_round03_loss1_continue3000to5000_20260606/summary.json`
- loss1 epoch5000 checkpoint: `adversarial_training_runs/burgers_loss3_selective_round03_loss1_continue3000to5000_20260606/burgers/checkpoints/burgers_epoch5000_step015000.pt`
- loss2 epoch2000 summary: `adversarial_training_runs/burgers_loss3_selective_round03_loss2_continue1000to2000_20260606/summary.json`
- loss2 epoch2000 checkpoint: `adversarial_training_runs/burgers_loss3_selective_round03_loss2_continue1000to2000_20260606/burgers/checkpoints/burgers_epoch2000_step006000.pt`
- loss3 epoch1500 summary: `adversarial_training_runs/burgers_loss3_selective_round03_loss3_continue1000to1500_20260606/summary.json`
- loss3 epoch1500 checkpoint: `adversarial_training_runs/burgers_loss3_selective_round03_loss3_continue1000to1500_20260606/burgers/checkpoints/burgers_epoch1500_step004500.pt`

Final generated/generalization metrics from each run's `burgers/eval_split_summary.csv`:

| model | final epoch | generated RMSE | generated relative L2 |
| --- | ---: | ---: | ---: |
| loss1 | 5000 | 0.03245434070606764 | 0.058182682528028995 |
| loss2 | 2000 | 0.03573958119889953 | 0.06408098797752262 |
| loss3 | 1500 | 0.021637501109214533 | 0.03877425041082085 |

Observed from the final plot report, best generated RMSE by stitched history:

| model | best generated epoch | best generated RMSE |
| --- | ---: | ---: |
| loss1 | 4877 | 0.0312228 |
| loss2 | 1784 | 0.034426 |
| loss3 | 893 | 0.0193611 |

## Final SVD/Jacobian Outputs

SVD output directory:

- `forensics/burgers_loss3_selective_round03_loss123_final_extension_jacobian_svd_rep20_top100_20260606`

Primary SVD files:

- `round03_loss123_final_extension_jacobian_svd_summary.md`
- `round03_loss123_final_extension_jacobian_svd_summary.csv`
- `round03_loss123_final_extension_error_spectral_norm_aggregate.csv`
- `round03_loss123_final_extension_jacobian_spectral_norm_aggregate.csv`
- `round03_loss123_final_extension_top_singular_values_long.csv`
- `round03_loss123_final_extension_solver_similarity_rankwise.csv`
- `round03_loss123_final_extension_solver_similarity_subspaces.csv`
- `round03_loss123_final_extension_runtime.csv`
- `round03_loss123_final_extension_sample_manifest.csv`
- `sample_000/` through `sample_019/` contain the per-sample NPZ SVD artifacts.

Generalization model-minus-solver error spectral norm summary from `round03_loss123_final_extension_error_spectral_norm_aggregate.csv`:

| model | n | error mean | baseline error mean | ratio mean | ratio median | smaller than baseline |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| loss1_epoch5000 | 10 | 2.753176126511624 | 5.774617872599661 | 0.49444841170651543 | 0.4696541915071688 | 9 |
| loss2_epoch2000 | 10 | 3.3222567331169137 | 5.774617872599661 | 0.5785477467427029 | 0.5842331915471849 | 9 |
| loss3_epoch1500 | 10 | 2.4995994501058596 | 5.774617872599661 | 0.41154677198027967 | 0.348693952524486 | 9 |

Inference from this SVD evidence: loss3 epoch1500 has the lowest generalization model-minus-solver error spectral norm mean among the three final-extension models, while loss1 and loss2 remain stronger on train/test error shrinkage in the aggregate table.

## Plot Outputs

Single-run stitched plot directories:

- `visualizations/burgers_loss3_selective_round03_loss1_5000ep_long_20260605_plots`
- `visualizations/burgers_loss3_selective_round03_loss2_2000ep_long_20260605_plots`
- `visualizations/burgers_loss3_selective_round03_loss3_1500ep_long_20260605_plots`

Each single-run directory contains refreshed RMSE and relative-L2 raw/MA25 grouped plots plus `polished_report/` figures.

Comparison plot directories refreshed in place:

- Dense comparison: `visualizations/burgers_loss3_selective_round03_long_training_comparison_dense_20260605`
- Compact comparison: `visualizations/burgers_loss3_selective_round03_long_training_comparison_20260605`

Plot report:

- `docs/burgers_loss3_selective_round03_final_extension_plot_report_20260606.md`

Key comparison figures:

- `visualizations/burgers_loss3_selective_round03_long_training_comparison_dense_20260605/round03_dense_same_epoch_rmse_every1_train_test_generalization.png`
- `visualizations/burgers_loss3_selective_round03_long_training_comparison_dense_20260605/round03_dense_wall_clock_rmse_every1_train_test_generalization.png`
- `visualizations/burgers_loss3_selective_round03_long_training_comparison_dense_20260605/round03_dense_same_epoch_relative_l2_every1_train_test_generalization.png`
- `visualizations/burgers_loss3_selective_round03_long_training_comparison_dense_20260605/round03_dense_wall_clock_relative_l2_every1_train_test_generalization.png`
- `visualizations/burgers_loss3_selective_round03_long_training_comparison_20260605/round03_same_epoch_rmse.png`
- `visualizations/burgers_loss3_selective_round03_long_training_comparison_20260605/round03_wall_clock_rmse.png`
- `visualizations/burgers_loss3_selective_round03_long_training_comparison_20260605/round03_final_rmse_bars.png`
- `visualizations/burgers_loss3_selective_round03_long_training_comparison_20260605/round03_final_relative_l2_bars.png`
- `visualizations/burgers_loss3_selective_round03_long_training_comparison_20260605/round03_final_generated50_loss3_advantage_hist.png`

## Remaining Work

- Upload large generated artifacts to R2 if needed.
- Commit/push Markdown and source changes if requested.
- Do not put CSV/NPZ/model/PNG artifacts in Git unless explicitly requested.
