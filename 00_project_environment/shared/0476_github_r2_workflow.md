# GitHub + R2 Workflow

This project should use two storage systems with different jobs:

- **GitHub** stores code, small text files, configs, manifests, and reproducible instructions.
- **R2 object storage** stores large datasets, checkpoints, and heavy experiment artifacts.

Do not use R2 as the source of truth for `.git/`. A partially synced `.git/objects/` directory can corrupt the repository and cause errors such as `fatal: bad object HEAD`.

The rules are enforced by these files:

```text
.gitignore                  decides what Git should ignore
.r2exclude                  decides what R2 sync must never upload/download
tools/sync_r2_artifacts.sh  standard wrapper for rclone upload/check/download
```

## Rule Of Thumb

Put files in GitHub if they are small, human-readable, or needed to reproduce an experiment.

Put files in R2 if they are large, binary, generated in bulk, or expensive to keep in Git history.

## Put In GitHub

Track these in GitHub:

```text
tools/
docs/
*.py
*.sh
requirements.txt
README.md
config files
small CSV summaries
small JSON summaries
experiment manifests
```

Examples from this repo:

```text
tools/run_batch_three_loss_loss_only.py
tools/run_and_plot_batch_three_loss_loss_only.py
tools/plot_batch_three_loss_loss_only.py
tools/plot_batch_single_index_loss_curves.py
tools/plot_batch_final_delta_comparison.py
tools/run_three_loss_batch100_full_loss3_sweep.sh
docs/*.md
requirements.txt
loss_attack_common.py
solvers.py
train_burgers_deeponet_deepxde.py
```

Small summaries can also go to GitHub when useful:

```text
results/.../config.json
results/.../summary.json
results/.../final_delta_comparisons/cosine_similarity_summary.csv
```

Only commit small result summaries if they are important for reading the experiment history.

## Put In R2

Store these in R2:

```text
datasets/
checkpoints/
trained models
large .pt files
large .npz files
large result directories
large plots or image batches
full experiment artifact folders
```

Examples from this repo:

```text
1D_Burgers/datasets/
1D_Burgers/trained_models/
deeponet_training_runs/*/checkpoints/
deeponet_training_runs/*/training_logs/output_transform_stats.npz
results/three_loss_batch100_*/
*.pt
*.pth
*.npz
```

## Never Sync These From R2 As Source Of Truth

Do not restore these from a partial R2 sync:

```text
.git/
.git/objects/
.git/refs/
.git/index
```

Always get Git metadata from GitHub by cloning or fetching from GitHub.

## Starting On A New Machine

Start by cloning code from GitHub:

```bash
cd /workspace
git clone --branch vast-ai https://github.com/YifeiSun01/NeuralOperatorRobustness2.git
cd NeuralOperatorRobustness2
```

Then create or activate the Python environment:

```bash
python3 -m venv adv_robust
source adv_robust/bin/activate
pip install -r requirements.txt
```

Then download large artifacts from R2:

```bash
rclone sync r2:YOUR_BUCKET/NeuralOperatorRobustness2/1D_Burgers/datasets 1D_Burgers/datasets --checksum
rclone sync r2:YOUR_BUCKET/NeuralOperatorRobustness2/1D_Burgers/trained_models 1D_Burgers/trained_models --checksum
rclone sync r2:YOUR_BUCKET/NeuralOperatorRobustness2/deeponet_training_runs deeponet_training_runs --checksum
```

Check that the R2 copy is complete:

```bash
rclone check r2:YOUR_BUCKET/NeuralOperatorRobustness2/1D_Burgers/datasets 1D_Burgers/datasets --checksum
rclone check r2:YOUR_BUCKET/NeuralOperatorRobustness2/1D_Burgers/trained_models 1D_Burgers/trained_models --checksum
```

## Running Experiments

Example FNO sweep command:

```bash
cd /workspace/NeuralOperatorRobustness2
OUT_PREFIX=results/three_loss_batch100_full_loss3_delta_rerun_YYYYMMDD \
LOG_DIR=logs/three_loss_batch100_full_loss3_delta_rerun_YYYYMMDD \
bash tools/run_three_loss_batch100_full_loss3_sweep.sh
```

The sweep writes:

```text
results/
logs/
```

Large result folders should be uploaded to R2, not committed to GitHub.

## Saving Code Changes

Commit code and docs to GitHub:

```bash
git status
git add tools docs requirements.txt
git commit -m "a-add"
git push origin vast-ai
```

If only specific files changed, add only those files:

```bash
git add tools/plot_batch_final_delta_comparison.py \
        tools/run_three_loss_batch100_full_loss3_sweep.sh \
        docs/github_r2_workflow.md
git commit -m "a-add"
git push origin vast-ai
```

## Saving Large Results

Upload large results to R2:

```bash
R2_REMOTE=r2:YOUR_BUCKET/NeuralOperatorRobustness2 \
  bash tools/sync_r2_artifacts.sh upload \
  results/three_loss_batch100_full_loss3_delta_rerun_YYYYMMDD
```

Upload logs if needed:

```bash
R2_REMOTE=r2:YOUR_BUCKET/NeuralOperatorRobustness2 \
  bash tools/sync_r2_artifacts.sh upload \
  logs/three_loss_batch100_full_loss3_delta_rerun_YYYYMMDD
```

Verify upload:

```bash
R2_REMOTE=r2:YOUR_BUCKET/NeuralOperatorRobustness2 \
  bash tools/sync_r2_artifacts.sh check \
  results/three_loss_batch100_full_loss3_delta_rerun_YYYYMMDD
```

Download artifacts on a new machine:

```bash
R2_REMOTE=r2:YOUR_BUCKET/NeuralOperatorRobustness2 \
  bash tools/sync_r2_artifacts.sh download \
  1D_Burgers/datasets 1D_Burgers/trained_models deeponet_training_runs
```

## Experiment Manifest

For each important experiment, save a small manifest in GitHub or inside the result folder:

```json
{
  "git_branch": "vast-ai",
  "git_commit": "PUT_COMMIT_HASH_HERE",
  "r2_result_path": "r2://YOUR_BUCKET/NeuralOperatorRobustness2/results/...",
  "dataset_path": "1D_Burgers/datasets/...",
  "checkpoint_path": "1D_Burgers/trained_models/...",
  "command": "OUT_PREFIX=... LOG_DIR=... bash tools/run_three_loss_batch100_full_loss3_sweep.sh",
  "notes": "Short description of the experiment"
}
```

Get the commit hash with:

```bash
git rev-parse HEAD
```

## If `.git` Gets Corrupted

Symptoms:

```text
fatal: bad object HEAD
missing blob
invalid sha1 pointer
```

Do not run destructive commands like `git reset --hard` in the corrupted workspace.

Safe recovery:

```bash
cd /workspace
git clone --branch vast-ai https://github.com/YifeiSun01/NeuralOperatorRobustness2.git NeuralOperatorRobustness2_clean

cp /workspace/NeuralOperatorRobustness2/tools/plot_batch_final_delta_comparison.py \
   /workspace/NeuralOperatorRobustness2_clean/tools/

cp /workspace/NeuralOperatorRobustness2/tools/run_three_loss_batch100_full_loss3_sweep.sh \
   /workspace/NeuralOperatorRobustness2_clean/tools/

cp -r /workspace/NeuralOperatorRobustness2/docs \
      /workspace/NeuralOperatorRobustness2_clean/
```

Then commit from the clean clone:

```bash
cd /workspace/NeuralOperatorRobustness2_clean
git status
git add tools docs
git commit -m "a-add"
git push origin vast-ai
```

Keep the corrupted workspace until all new files have been copied out.

## Optional: Use DVC Later

For a cleaner long-term setup, use DVC with an S3-compatible R2 remote:

```bash
dvc init
dvc remote add -d r2remote s3://YOUR_BUCKET/NeuralOperatorRobustness2/dvc
dvc remote modify r2remote endpointurl https://YOUR_ACCOUNT_ID.r2.cloudflarestorage.com
```

Then large files are tracked by small `.dvc` pointer files in GitHub, while the actual data stays in R2.

This is the cleanest long-term pattern:

```text
GitHub: code + .dvc pointer files
R2: actual datasets, checkpoints, and large results
```
