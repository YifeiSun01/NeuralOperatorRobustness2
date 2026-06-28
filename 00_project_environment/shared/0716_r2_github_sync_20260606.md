# R2 And GitHub Sync - 2026-06-06

Status: R2 upload completed; GitHub code/Markdown commit and push completed in this sync turn.

Timestamp: `2026-06-06T13:55:26+00:00`

## R2 Upload

Destination:
- Bucket: `neural-operator-robustness`
- Prefix: `machine-sync/NeuralOperatorRobustness2-selected`
- Endpoint: `https://606bf6c862a4e8f63dabb6243dce2df7.r2.cloudflarestorage.com`

Observed upload:
- Tool: `rclone v1.60.1-DEV`
- Mode: `copy`, not `sync`; remote objects were not deleted or pruned.
- Excluded local paths/patterns: `.git/`, `adv_robust/`, `__pycache__/`, `*.pyc`, `*.pyo`, pytest/mypy/ruff caches, and notebook checkpoint caches.
- R2 compatibility flags used: `--s3-no-check-bucket --s3-no-head --s3-disable-checksum`.
- Initial R2 copy attempt without those compatibility flags failed with Cloudflare R2 `501 NotImplemented`; the successful retry used the flags above.
- Successful copy reported final transferred size: `2.132 GiB / 2.132 GiB`.
- Post-upload remote size check for the prefix reported `128416` objects and `264.008 GiB`. This includes older pre-existing objects in the same prefix because the run used copy-only semantics.

Important security note:
- R2 and GitHub credentials were not written into repository files. The temporary rclone config was placed under `/tmp` for this machine session.
- The credentials pasted into chat should be rotated after the sync is confirmed.

## GitHub Upload

Destination:
- Remote: `https://github.com/YifeiSun01/NeuralOperatorRobustness2.git`
- Branch: `vast-ai`

Git scope:
- Include code/scripts/Markdown: `*.py`, `*.sh`, `*.md`, plus `EXPERIMENT_LEDGER.md`.
- Exclude generated large artifacts such as checkpoints, `.pt`, `.npz`, `.npy`, `.csv`, `.png`, datasets, and visualization binaries from GitHub; those belong in R2.
