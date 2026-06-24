# Loss3 replace/add mechanism backup record - 2026-06-23

## Scope

This record covers the final Loss3 replace/add mechanism conclusion, code,
tables, figures, manifests, and compressed archive produced on 2026-06-23.

## GitHub

Branch:

- `merge-vast-ai-darcy-flow`

Latest pushed commit at the time of the first backup pass:

- `a6fd344 Document final Loss3 replace add mechanism conclusion`

Latest pushed commit after recording the backup status:

- `3dd7d24 Record Loss3 mechanism backup completion`

Mechanism-related commits included:

- `794c8d1 Add Lmax-normalized boundary cap analysis`
- `2cf4053 Add dominant subspace iso-projection probe`
- `2ac51d4 Add Jacobian mode causal ablation probe`
- `d5f7c57 Add Jacobian tangent width probe`
- `f8afc7f Fix Burgers tangent-width sample-local evaluation`
- `a6fd344 Document final Loss3 replace add mechanism conclusion`

## R2

Remote prefix:

- `neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected`

Uploaded/updated paths:

- `analysis_outputs/backup_archives/loss3_replace_add_mechanism_final_20260623.tar.gz`
- `analysis_outputs/backup_archives/loss3_replace_add_mechanism_final_20260623.tar.gz.sha256`
- `analysis_outputs/backup_manifests/loss3_replace_add_mechanism_backup_manifest_20260623.json`
- `docs/loss3_*.md`
- `tools/*loss3*20260623.py`
- `tools/run_loss3*20260623.sh`
- `tools/plot_loss3*20260623.py`
- `analysis_outputs/mechanism_20260622/full_mechanism_validation/boundary_volume_probe_20260623`
- `analysis_outputs/mechanism_20260622/full_mechanism_validation/dominant_subspace_iso_projection_20260623`
- `analysis_outputs/mechanism_20260622/full_mechanism_validation/jacobian_mode_causal_ablation_20260623`
- `analysis_outputs/mechanism_20260622/full_mechanism_validation/jacobian_tangent_width_20260623`

Additional full-archive backup pass completed on 2026-06-24 UTC:

- `analysis_outputs/backup_archives/20260623T014210Z/attack_objective_true_loss3_comparison_20260622_20260623T014210Z.tar` - 8,173,240,320 bytes
- `analysis_outputs/backup_archives/20260623T014210Z/mechanism_20260622_partial_20260623T014210Z.tar` - 2,721,730,560 bytes
- `analysis_outputs/backup_archives/20260623T014210Z/optimizer_ablation_20260622_20260623T014210Z.tar` - 1,669,314,560 bytes
- `analysis_outputs/backup_archives/20260623T014210Z/code_docs_tools_commit45a7821_20260623T014210Z.tar` - 15,319,040 bytes
- `analysis_outputs/backup_archives/20260623T014210Z/backup_manifests_20260623T014210Z.tar` - 256,000 bytes
- `analysis_outputs/backup_archives/20260623T014210Z/SHA256SUMS.txt` - 835 bytes

Archive SHA256:

```text
644795e38d819a39fac2902ff2a57b383f1dddc0df1e9aa91a40f5625f17da91
```

## Notes

R2 reported transient `NotImplemented` responses on first upload attempts, but
the rclone retry attempts succeeded and the remote archive listing was verified.
