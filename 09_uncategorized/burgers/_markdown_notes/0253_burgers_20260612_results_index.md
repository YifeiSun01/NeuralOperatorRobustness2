# Burgers Wideparam 2026-06-12 Results Index

This index groups the Burgers wide-parameter generalization, retraining, comparison-dense visualization, full-Jacobian SVD, biased ATB direction, and random-field training notes discussed on 2026-06-12.

No credentials or API tokens are stored in these notes. Large binary artifacts such as checkpoints, NPZ traces, datasets, and PNG bundles are referenced by path instead of committed into Git.

## Category Documents

1. [Generalization, retraining, and comparison-dense visual evidence](burgers_20260612_generalization_retrain_visuals_summary.md)
2. [Full 1024 Jacobian SVD, ATB, attack-growth correlations, and direction angles](burgers_20260612_svd_atb_attack_robustness_summary.md)
3. [Training reproducibility, checkpoint differences, and random-field noise training modes](burgers_20260612_training_reproducibility_random_field_summary.md)

## Small Result Tables Added With This Summary

- `forensics/burgers_wideparam_loss3targeted_full1024_svd_attack25_biased_local_direction_20260611/atb_svd_loss_increase_correlation_ranking_20260612.csv`
- `forensics/burgers_wideparam_loss3targeted_full1024_svd_attack25_biased_local_direction_20260611/direction_pairwise_angle_summary_20260612.csv`
- `forensics/burgers_wideparam_loss123_retrain_round00_p2q2_diverse_multi_sample_attack_visuals_batched_20260612/group05_loss3_best/selected_samples.csv`
- `forensics/burgers_wideparam_loss123_retrain_round00_p2q2_diverse_multi_sample_attack_visuals_batched_20260612/group05_loss3_best/summary.json`

## Main Code Added Or Updated

- `tools/run_burgers_wideparam_retrain_round00_p2q2_diverse_multi_visuals_batched_20260612.py`
- `tools/plot_burgers_wideparam_retrain_p2q2_samplewise_overlay_20260612.py`
- `tools/build_burgers_wideparam_loss3_best_group05_20260612.py`
- `tools/plot_burgers_group05_loss3best_samplewise_one_row_logmse_20260612.py`
- `tools/plot_burgers_all_groups_samplewise_one_row_logmse_20260612.py`
- `tools/run_burgers_wideparam_random_field_training_20260612.sh`
- `tools/watch_burgers_retrain_then_random_field_20260612.sh`
- `tools/adversarial_training.py` random-field training support

## Local Artifact Roots

- Latest wideparam loss3-targeted dataset: `generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00`
- Retrained model runs: `adversarial_training_runs/burgers_wideparam_loss*_retrain_20260611`
- Comparison-dense traces: `forensics/burgers_wideparam_loss123_retrain_round00_p2q2_diverse_multi_sample_attack_visuals_batched_20260612`
- Comparison-dense images: `visualizations/burgers_wideparam_loss123_retrain_round00_p2q2_comparison_dense_diverse_multi_sample_batched_20260612`
- Image-only bundle: `visualizations/burgers_wideparam_loss123_retrain_comparison_dense_image_only_bundle_20260612`
- Full SVD/ATB results: `forensics/burgers_wideparam_loss3targeted_full1024_svd_attack25_biased_local_direction_20260611`
