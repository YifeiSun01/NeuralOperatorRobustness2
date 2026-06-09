# R2 Burgers generalization data lookup - 2026-06-08

Status: completed a read-only lookup of the user-provided Cloudflare R2 bucket/prefix using temporary environment variables. No credentials were written to repository files.

Remote prefix inspected:

`neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected`

## Observed Dataset Roots

Observed from `rclone lsf` and `rclone size --json` against the remote prefix:

| user-facing dataset | R2 path under selected prefix | object count | bytes | examples |
|---|---|---:|---:|---|
| first semantic/generalization root | `generalization_datasets/burgers` | 50 | 82205042 | `burgers_near_gaussian_corr0p015.pt`, `burgers_mid_matern_corr0p75_nu2p5.pt`, `burgers_far_sawtooth_add_scale0p3_shift0.pt` |
| second semantic/target-band ns50 root | `generalization_datasets_rmse_1p5_3x_all_ns50/burgers` | 50 | 82206074 | `burgers_target_gaussian_corr0p085.pt`, `burgers_target_matern_corr0p42_nu5.pt`, `burgers_far_centered_scale_shift_scale1_shift0p4.pt` |
| third loss3-selective stress root | `generalization_datasets_burgers_loss3_selective_search/round_03/burgers` | 50 | 20744250 | `burgers_loss3_selective_r03_d00.pt` through `burgers_loss3_selective_r03_d49.pt` |
| local/remote partial round02 root | `generalization_datasets_burgers_loss3_aligned_search/round_02/burgers` | 10 | 4139730 | `burgers_loss3_aligned_r02_d00.pt` through `burgers_loss3_aligned_r02_d09.pt` |

## Observed R2 Forensics

Observed available on R2:

- Full-tag source artifact: `forensics/burgers_round03_full_p2q2_52datasets_4models_finalonly_20step_20260607`, with `21` objects and `160859222` bytes. Listed files include `manifest.json`, `summary.json`, `summary_by_model_dataset.csv`, `progress.jsonl`, `config.json`, and per-model directories `baseline`, `loss1_epoch8000`, `loss2_epoch2000`, `loss3_epoch1500` with NPZ/CSV attack outputs.
- Round03 per-dataset advantage artifact: `forensics/burgers_loss3_selective_round03_per_dataset_loss3_advantage_20260605`, with `4` objects and `57673` bytes. Listed files include `per_dataset_advantage_summary.csv` and `per_dataset_loss3_vs_loss12.csv`.
- Second-root full generalization eval: `generalization_eval_rmse_1p5_3x_all_ns50_full`, containing `GENERALIZATION_EVALUATION.md`, `metrics.csv`, `metrics.json`, `metrics_sorted_by_similarity.csv`, `similarity.csv`, `similarity.json`, and plot PNGs.

Observed missing or not uploaded under this selected prefix at lookup time:

- `forensics/burgers_round03_full52_per_sample_clean_vs_attack_mismatch_20260608`: `0` objects.
- `forensics/burgers_neutral_generalization_final_models_20260608`: `0` objects.
- Newly written local docs such as `docs/burgers_three_generalization_rounds_conclusion_audit_20260608.md` were not observed in the remote `docs` listing during this lookup.

## Inference

Under the user-facing first/second/third naming used in the current discussion:

- the `10200`-sample full-tag source data on R2 belongs to the second semantic/target-band ns50 generalization root, because that source artifact uses `generalization_datasets_rmse_1p5_3x_all_ns50/burgers` for the `10000` generalization samples;
- the third root is the loss3-selective attack-generated/stress root `generalization_datasets_burgers_loss3_selective_search/round_03/burgers`;
- `round_02` is present remotely but only has `10` dataset files, so it is not a complete 50-dataset run in the inspected R2 prefix.

Remaining work:

- Upload or sync the local 2026-06-08 derived mismatch/provenance docs and CSVs if they should live on R2.
- If a different R2 prefix contains another completed `round_02` or full 100-step tag artifact, inspect that prefix separately.
