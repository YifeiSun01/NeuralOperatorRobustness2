# Burgers Full-1024 SVD25 Reuse3 + Retrain Workflow - 2026-06-11

This workflow extends the completed 3-sample full-1024 SVD/attack probe to a unified 25-sample table.

## Sample Design

- Reuse prior completed samples from `forensics/burgers_wideparam_loss3targeted_full1024_svd_attack3_20260611` as sample ids `0`, `1`, and `2`.
- Add `22` non-overlapping samples:
  - `2` train samples from the original Burgers train split.
  - `2` test samples from the original Burgers test split.
  - `18` additional generalization samples from distinct wide-parameter datasets.
- Total manifest: `25` samples, with `21` generalization datasets and no duplicated generalization `dataset_id`.

## Runner

- Python runner: `tools/run_burgers_wideparam_full1024_svd_attack25_reuse3_20260611.py`.
- Shell wrapper: `tools/run_burgers_wideparam_full1024_svd_attack25_reuse3_20260611.sh`.
- Full workflow: `tools/run_burgers_wideparam_svd25_then_retrain_upload_20260611.sh`.

The SVD computation is full dense `1024 x 1024` SVD. It does not use block projection, randomized SVD, or coarse top-k proxy.

## Outputs

- SVD output root: `forensics/burgers_wideparam_loss3targeted_full1024_svd_attack25_reuse3_20260611`.
- SVD report: `docs/burgers_wideparam_loss3targeted_full1024_svd_attack25_reuse3_20260611.md`.
- Master workflow log: `logs/burgers_svd25_then_retrain_upload_20260611.log`.
- Workflow state/upload logs: `forensics/burgers_svd25_then_retrain_upload_state_20260611`.

After SVD finishes, the workflow uploads SVD results to R2, commits code/markdown to GitHub, then starts the Burgers loss1/loss2/loss3 adversarial retraining workflow against the latest loss3-targeted wide-parameter generalization dataset.

## Smoke Test

```bash
SMOKE_ONLY=1 RUN_UPLOAD=0 RUN_GIT=0 bash tools/run_burgers_wideparam_svd25_then_retrain_upload_20260611.sh
```

## Full Run

```bash
bash tools/run_burgers_wideparam_svd25_then_retrain_upload_20260611.sh
```

Credentials are intentionally not stored in the repository. The upload and push steps read credentials from runtime environment variables or temporary files.
