# Loss3 GitHub and R2 Sync - 2026-05-21

Status: completed.

## GitHub

Observed action:

- Committed code, Markdown, shell scripts, Python scripts, and experiment ledger updates to local git.
- Pushed branch `vast-ai` to `origin`.

Commit:

```text
9dac8f5 Add Loss3 core4 mechanism validation tooling and docs
```

Remote branch:

```text
https://github.com/YifeiSun01/NeuralOperatorRobustness2/tree/vast-ai
```

Committed content type:

- `EXPERIMENT_LEDGER.md`
- Loss3 docs under `docs/`
- Loss3 analysis/probe/plot/run scripts under `tools/`
- `tools/run_loss3_overnight_20260520.sh`

Excluded from GitHub commit:

- Large generated experiment artifacts under `forensics/`

## R2

Observed action:

- Uploaded generated Loss3 data, figures, visual exports, probe outputs, sweep outputs, and summary artifacts to Cloudflare R2.
- Upload used temporary rclone configuration; credentials were not written to the repository.

R2 destination prefix:

```text
s3://neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected/forensics/
```

R2 verification:

```text
Total objects: 14,253
Total size: 16.710 GiB / 17,942,053,213 bytes
```

Important uploaded directories include:

- `loss3_core4_pq_landscape_probe_full_20260521/`
- `loss3_core4_pq_landscape_probe_trajectory_20260521/`
- `loss3_current_core4_jacobian_svd_probe_20260521/`
- `loss3_current_mechanism_validation_summary_20260521/`
- `loss3_all_p2_tangent_geometry_probe_20260520/`
- `loss3_p2q2_tangent_radial_geometry_probe_20260520/`
- `loss3_p2q2_stepwise_gain_geometry_probe_20260520/`
- `loss3_alpha_epsilon_core4_visuals_*`
- `loss3_alpha_epsilon_core4_delta_similarity_*`
- `loss3_visuals_clean_export_*`
- `loss3_visuals_tree_export_*`
- `loss3_alpha_epsilon_core4_sweep_20260519/`
- `loss3_alpha_epsilon_core4_sweep_p2q2_300steps_20260520/`
- `loss3_alpha_epsilon_core4_sweep_pneq_q_100steps_20260520/`
- `loss3_optimizer_direction_proposal_ablation_20260517/`
- `loss3_optimizer_direction_proposal_ablation_raw_replace_backfill_20260518/`

## Notes

- The first attempted full `forensics/` copy was stopped because remote enumeration produced no useful progress. The later selected-directory and large-sweep copies completed successfully.
- A verification listing confirmed the key uploaded R2 directories exist under the destination prefix.
