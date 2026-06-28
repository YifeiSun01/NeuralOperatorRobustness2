# R2 Backup And Burgers Attack Runbook

Last updated: 2026-05-11

This file is the persistent checklist for Burgers attack data, models, dictionaries, commands, and R2 backup targets.

Rule: these artifacts are too large for GitHub and are ignored by `.gitignore`; they must be backed up to Cloudflare R2.

## R2 Target

Primary R2 prefix for this machine copy:

```bash
R2:neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected
```

Critical backup command, assuming an `rclone` remote named `R2` is configured:

```bash
cd /workspace/NeuralOperatorRobustness2

rclone copy 1D_Burgers/datasets R2:neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected/1D_Burgers/datasets --transfers 8 --checkers 16
rclone copy results/dictionaries R2:neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected/results/dictionaries --transfers 8 --checkers 16
rclone copy 1D_Burgers/trained_models R2:neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected/1D_Burgers/trained_models --transfers 8 --checkers 16
rclone copy fno_training_runs/burgers_nu0p01_fno1d_500 R2:neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected/fno_training_runs/burgers_nu0p01_fno1d_500 --transfers 8 --checkers 16
rclone copy deeponet_training_runs/burgers_nu0p01_deeponet_lu_ref_50k R2:neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected/deeponet_training_runs/burgers_nu0p01_deeponet_lu_ref_50k --transfers 8 --checkers 16
rclone copy tmp_old_runner_inputs_b01 R2:neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected/tmp_old_runner_inputs_b01 --transfers 8 --checkers 16
rclone copy results/burgers_loss3_18setting_batch100_random100_losses_parallel6 R2:neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected/results/burgers_loss3_18setting_batch100_random100_losses_parallel6 --transfers 8 --checkers 16
rclone copy ATTACK_RUN_DEPENDENCY_MANIFEST.md R2:neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected/ --transfers 8 --checkers 16
rclone copy R2_BACKUP_AND_ATTACK_RUNBOOK.md R2:neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected/ --transfers 8 --checkers 16
rclone copy run_burgers_loss3_18setting_resume_sequential.sh R2:neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected/ --transfers 8 --checkers 16
```

Current local sizes:

| Path | Size | Purpose |
|---|---:|---|
| `1D_Burgers/datasets` | 47M | Generated Burgers datasets and train/test splits |
| `results/dictionaries` | 426M | Attack dictionaries, including N=200, 2000, 20000 |
| `1D_Burgers/trained_models` | 5.4M | Attack-ready FNO nu=0.001 model and logs |
| `fno_training_runs/burgers_nu0p01_fno1d_500` | 2.5M | FNO nu=0.01 model and logs |
| `deeponet_training_runs/burgers_nu0p01_deeponet_lu_ref_50k` | 2.5M | DeepONet nu=0.01 model and logs |
| `tmp_old_runner_inputs_b01` | 0 | Compatibility symlink path for old FNO attack parser |
| `results/burgers_loss3_18setting_batch100_random100_losses_parallel6` | 105M | Completed and partial attack results |

## Current Artifact Status

Found and validated locally:

| Artifact | Status | Path |
|---|---|---|
| FNO, nu=0.001 | present | `1D_Burgers/trained_models/attack_ready/burgers_nu0.001_fno1d_500/checkpoints/pytorch_fno1d_500.pt` |
| FNO, nu=0.01 | present | `fno_training_runs/burgers_nu0p01_fno1d_500/burgers_1d/checkpoints/fno1d_pytorch.pt` |
| DeepONet, nu=0.01 | present | `deeponet_training_runs/burgers_nu0p01_deeponet_lu_ref_50k/checkpoints/deeponet_burgers_nu0p01.pt` |
| DeepONet, nu=0.001 | not required for current plan | no required checkpoint recorded |
| Burgers N=1500, nu=0.001 | regenerated and split | `1D_Burgers/datasets/1D/Burgers/batched_exponax` and `1D_Burgers/datasets/1D/Burgers/batched_exponax_splits` |
| Burgers N=1500, nu=0.01 | regenerated and split | `1D_Burgers/datasets/1D/Burgers/batched_exponax` and `1D_Burgers/datasets/1D/Burgers/batched_exponax_splits` |
| Dictionary N=200, nu=0.001 | present | `results/dictionaries/burgers_nu0p001_N200_seed1000045` |
| Dictionary N=2000, nu=0.001 | present | `results/dictionaries/burgers_nu0p001_N2000_seed1000045` |
| Dictionary N=20000, nu=0.001 | present | `results/dictionaries/burgers_nu0p001_N20000_seed1000045` |
| Dictionary N=200, nu=0.01 | present | `results/dictionaries/burgers_nu0p01_N200_seed1000045` |
| Dictionary N=2000, nu=0.01 | present | `results/dictionaries/burgers_nu0p01_N2000_seed1000045` |
| Dictionary N=20000, nu=0.01 | present | `results/dictionaries/burgers_nu0p01_N20000_seed1000045` |

Extra dictionary currently present:

```text
results/dictionaries/burgers_nu0p001_N10000_seed1000045
```

## Dataset Files

Generated full datasets:

```text
1D_Burgers/datasets/1D/Burgers/batched_exponax/dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45.pt
1D_Burgers/datasets/1D/Burgers/batched_exponax/dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.01_t1.0_seed45.pt
```

Train/test split roots:

```text
1D_Burgers/datasets/1D/Burgers/batched_exponax_splits/dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45
1D_Burgers/datasets/1D/Burgers/batched_exponax_splits/dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.01_t1.0_seed45
```

Each split root contains:

```text
*_train.pt
*_test.pt
split_manifest.json
```

## Dictionary Files

Each dictionary directory contains an `inputs_*.pt` file and an `outputs_*.pt` file. These are the xy-pair input/output dictionaries used by the attacks.

```text
results/dictionaries/burgers_nu0p001_N200_seed1000045
results/dictionaries/burgers_nu0p001_N2000_seed1000045
results/dictionaries/burgers_nu0p001_N20000_seed1000045
results/dictionaries/burgers_nu0p01_N200_seed1000045
results/dictionaries/burgers_nu0p01_N2000_seed1000045
results/dictionaries/burgers_nu0p01_N20000_seed1000045
```

Regenerate dictionary commands:

```bash
cd /workspace/NeuralOperatorRobustness2
source adv_robust/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python -u generate_burgers_dictionary.py --num_samples 200 --seed 1000045 --batch_size 64 --nu 0.001 --output_dir results/dictionaries/burgers_nu0p001_N200_seed1000045
python -u generate_burgers_dictionary.py --num_samples 2000 --seed 1000045 --batch_size 64 --nu 0.001 --output_dir results/dictionaries/burgers_nu0p001_N2000_seed1000045
python -u generate_burgers_dictionary.py --num_samples 20000 --seed 1000045 --batch_size 64 --nu 0.001 --output_dir results/dictionaries/burgers_nu0p001_N20000_seed1000045

python -u generate_burgers_dictionary.py --num_samples 200 --seed 1000045 --batch_size 64 --nu 0.01 --output_dir results/dictionaries/burgers_nu0p01_N200_seed1000045
python -u generate_burgers_dictionary.py --num_samples 2000 --seed 1000045 --batch_size 64 --nu 0.01 --output_dir results/dictionaries/burgers_nu0p01_N2000_seed1000045
python -u generate_burgers_dictionary.py --num_samples 20000 --seed 1000045 --batch_size 64 --nu 0.01 --output_dir results/dictionaries/burgers_nu0p01_N20000_seed1000045
```

## Dataset Regeneration Commands

Generate the two N=1500 Burgers datasets:

```bash
cd /workspace/NeuralOperatorRobustness2
source adv_robust/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python -u 1D_Burgers/data_generation/generate_burgers_exponax_batched.py \
  --num-records 1500 \
  --nx 1024 \
  --nu 0.01,0.001 \
  --t-final 1.0 \
  --step 0.001 \
  --batch-size 256 \
  --kernel gaussian \
  --correlation-length 0.03 \
  --bc periodic \
  --seed 45 \
  --domain-extent 2.0 \
  --save-dir 1D_Burgers/datasets/1D/Burgers/batched_exponax \
  --overwrite
```

Create train/test splits:

```bash
cd /workspace/NeuralOperatorRobustness2
source adv_robust/bin/activate

python -u tools/split_burgers_dataset.py \
  --source 1D_Burgers/datasets/1D/Burgers/batched_exponax/dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.01_t1.0_seed45.pt \
  --source 1D_Burgers/datasets/1D/Burgers/batched_exponax/dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45.pt \
  --output-root 1D_Burgers/datasets/1D/Burgers/batched_exponax_splits \
  --seed 1234 \
  --test-count 150 \
  --overwrite
```

## Model Training Commands

DeepONet nu=0.01 checkpoint is already present. If it must be regenerated, use the existing training config/logs as the source of truth:

```text
deeponet_training_runs/burgers_nu0p01_deeponet_lu_ref_50k/training_logs/config.json
deeponet_training_runs/burgers_nu0p01_deeponet_lu_ref_50k/training_logs/dataset_info.json
deeponet_training_runs/burgers_nu0p01_deeponet_lu_ref_50k/training_logs/losses.csv
deeponet_training_runs/burgers_nu0p01_deeponet_lu_ref_50k/training_logs/summary.json
```

The recorded training setup is DeepONet, nu=0.01, LU Burgers reference, 50000 iterations, full-batch training, tanh activation, periodic trunk features, output transform, and inverse-time LR decay.

Do not train DeepONet nu=0.001 for the current attack plan unless explicitly requested.

## Attack Runner Files

FNO attack runner:

```text
run_burgers_corrected_oldstyle_5loss.py
```

DeepONet attack runner:

```text
run_burgers_deeponet_corrected_oldstyle_5loss.py
```

Summarizers:

```text
summarize_burgers_corrected_oldstyle_true_losses.py
summarize_burgers_attack_ratios.py
```

Sequential 18-setting resume script:

```text
run_burgers_loss3_18setting_resume_sequential.sh
```

This script runs settings one by one, not six in parallel, and skips a setting if both summary files already exist:

```text
true_loss_summary_aggregate.csv
ratio_summary/burgers_attack_ratio_by_parameter.csv
```

Run all pending settings sequentially:

```bash
cd /workspace/NeuralOperatorRobustness2
bash run_burgers_loss3_18setting_resume_sequential.sh
```

## Future Settings And Loss3-Bad Search Direction

The old summary file records the future search plan in:

```text
BURGERS_BATCH100_18_SETTINGS_FULL_SUMMARY.md
```

Key idea: search parameter regimes where `loss3` becomes weaker, mainly by moving epsilon/alpha toward larger-radius or low-signal regions depending on model/norm.

Experimental unit and metric:

- Each Burgers setting is run as one batch of `100` different initial conditions sampled from the 150 test initial conditions with seed `20260511`.
- For each initial condition, all attack tags/methods are compared: `loss1`, `loss2_fixed`, `loss2_dict_N200`, `loss2_dict_N2000`, `loss2_dict_N20000`, `loss3_stopgrad`, and `loss3`.
- The per-initial-condition winner is the method with the largest final true-loss ratio:

```text
final true loss / initial true loss
```

- The main summary counts how many of the 100 initial conditions each method wins, plus mean/median ratio, mean increase, and mean final true loss.
- The scientific question is how the winner and loss ratio depend on `epsilon`, `alpha`, and attack norm (`Linf` versus `L2`), especially where `loss3` stops dominating.

Main hypotheses recorded there:

- FNO `nu=0.01 L2` may degrade as L2 radius grows beyond existing `eps=7`.
- At fixed FNO `nu=0.01 L2 eps=9`, varying `alpha` may show whether step size itself changes `loss3` dominance.
- DeepONet `nu=0.01` did not yet have a clear bad region, so the plan was to probe very small epsilon and extreme large epsilon.
- If FNO degrades but DeepONet does not, the behavior is architecture/model dependent rather than just PDE/norm dependent.

Proposed future cases from the old roadmap:

| # | Model | nu | Norm | epsilon | alpha | Purpose |
|---:|---|---:|---|---:|---:|---|
| 1 | FNO | 0.01 | L2 | 8.0 | 0.15 | Extend L2 degradation just beyond eps=7. |
| 2 | FNO | 0.01 | L2 | 10.0 | 0.1875 | Test whether `loss3` keeps dropping at larger L2 radius. |
| 3 | FNO | 0.01 | L2 | 12.0 | 0.225 | Large-radius FNO L2 stress test. |
| 4 | FNO | 0.01 | L2 | 16.0 | 0.30 | Extreme L2 radius; check if dictionary/other losses overtake. |
| 5 | FNO | 0.01 | L2 | 9.0 | 0.09 | Fixed eps=9, smaller alpha. |
| 6 | FNO | 0.01 | L2 | 9.0 | 0.17 | Fixed eps=9, proportional alpha reference. |
| 7 | FNO | 0.01 | L2 | 9.0 | 0.30 | Fixed eps=9, larger alpha. |
| 8 | DeepONet | 0.01 | Linf | 0.025 | 0.001 | Very small Linf, low-signal bad-case search. |
| 9 | DeepONet | 0.01 | Linf | 0.05 | 0.002 | Small Linf, bridge to eps=0.125. |
| 10 | DeepONet | 0.01 | Linf | 0.075 | 0.003 | Small-to-mid Linf mapping. |
| 11 | DeepONet | 0.01 | Linf | 1.0 | 0.04 | Extreme Linf; test DeepONet degradation. |
| 12 | DeepONet | 0.01 | L2 | 0.5 | 0.009 | Very small L2, low-epsilon bad-case search. |
| 13 | DeepONet | 0.01 | L2 | 1.0 | 0.01875 | Small L2; compare against FNO L2 eps=1. |
| 14 | DeepONet | 0.01 | L2 | 2.0 | 0.0375 | Bridge between small L2 and existing eps=4. |
| 15 | DeepONet | 0.01 | L2 | 12.0 | 0.225 | Extreme L2; test whether DeepONet remains favorable to `loss3`. |

Recommended old-roadmap execution order:

```text
1. Run FNO nu=0.01 L2 cases first.
2. Then run DeepONet small-epsilon cases.
3. Keep DeepONet extreme-large cases as confirmation points.
```

The current sequential script contains a smaller executable subset:

| Model | nu | Norm | epsilon | alpha | Current status |
|---|---:|---|---:|---:|---|
| FNO | 0.01 | Linf | 0.15 | 0.006 | complete; `loss3` wins 95/100 |
| FNO | 0.01 | Linf | 0.2 | 0.008 | rerun needed |
| FNO | 0.01 | Linf | 0.45 | 0.018 | rerun needed |
| FNO | 0.01 | Linf | 0.75 | 0.03 | rerun needed |
| FNO | 0.01 | Linf | 0.3 | 0.003 | rerun needed; fixed epsilon, small alpha |
| FNO | 0.01 | Linf | 0.3 | 0.03 | rerun needed; fixed epsilon, large alpha |
| FNO | 0.01 | L2 | 2.0 | 0.0375 | not run |
| FNO | 0.01 | L2 | 9.0 | 0.17 | not run; overlaps old-roadmap fixed eps=9 middle reference |
| DeepONet | 0.01 | Linf | 0.05 | 0.002 | not run; overlaps old-roadmap small Linf |
| DeepONet | 0.01 | Linf | 0.75 | 0.03 | not run; large Linf probe |
| DeepONet | 0.01 | L2 | 1.0 | 0.01875 | not run; overlaps old-roadmap small L2 |
| DeepONet | 0.01 | L2 | 12.0 | 0.225 | not run; overlaps old-roadmap extreme L2 |

Executable combined future-search script:

```bash
cd /workspace/NeuralOperatorRobustness2
bash run_burgers_future_loss3_bad_search_sequential.sh
```

This script runs `22` unique future settings sequentially, one at a time. It merges the old 15-case roadmap with the current executable subset, skips settings whose local summaries already exist, and writes outputs to:

```text
results/burgers_future_loss3_bad_search_batch100_random100_sequential
```

Automatic outputs after the script runs:

```text
results/burgers_future_loss3_bad_search_batch100_random100_sequential/run_status.csv
results/burgers_future_loss3_bad_search_batch100_random100_sequential/global_ratio_summary/burgers_attack_ratio_by_parameter.csv
results/burgers_future_loss3_bad_search_batch100_random100_sequential/global_ratio_summary/burgers_attack_ratio_by_parameter.md
results/burgers_future_loss3_bad_search_batch100_random100_sequential/global_ratio_summary/burgers_attack_ratio_runs.csv
results/burgers_future_loss3_bad_search_batch100_random100_sequential/global_ratio_summary/burgers_attack_ratio_runs.md
results/burgers_future_loss3_bad_search_batch100_random100_sequential/FINAL_22_SETTING_SUMMARY.md
```

`FINAL_22_SETTING_SUMMARY.md` records, for every completed setting:

- best method by mean ratio
- best method by mean true-loss increase
- best method by mean final true loss
- wins and win rate for all 7 methods
- mean ratio, ratio standard deviation, mean increase, and mean final true loss for all 7 methods
- per-setting runtime/status records

Estimated runtime for all 22 settings on the current machine: roughly `8-16 hours`, depending on GPU load and whether DeepONet settings run slower than the FNO settings. The script is restartable because it skips settings that already have completed summaries.

## First Completed New Setting

User-requested first pending setting was run and completed:

```text
FNO, nu=0.01, L-infinity, epsilon=0.15, alpha=0.006, batch_size=100, seed=20260511
```

Result root:

```text
results/burgers_loss3_18setting_batch100_random100_losses_parallel6/fno_nu0p01_linf_eps0p15_alpha0p006_batch100_seed20260511
```

Main summaries:

```text
results/burgers_loss3_18setting_batch100_random100_losses_parallel6/fno_nu0p01_linf_eps0p15_alpha0p006_batch100_seed20260511/true_loss_summary_aggregate.csv
results/burgers_loss3_18setting_batch100_random100_losses_parallel6/fno_nu0p01_linf_eps0p15_alpha0p006_batch100_seed20260511/true_loss_summary_aggregate.md
results/burgers_loss3_18setting_batch100_random100_losses_parallel6/fno_nu0p01_linf_eps0p15_alpha0p006_batch100_seed20260511/ratio_summary/burgers_attack_ratio_by_parameter.csv
results/burgers_loss3_18setting_batch100_random100_losses_parallel6/fno_nu0p01_linf_eps0p15_alpha0p006_batch100_seed20260511/ratio_summary/burgers_attack_ratio_by_parameter.md
```

Result highlights:

| Method | Ratio mean | Increase mean | Final true loss mean |
|---|---:|---:|---:|
| loss1 | 4.24387 | 0.0485411 | 0.061574 |
| loss2_fixed | 5.24126 | 0.0498318 | 0.05956 |
| loss2_dict_N200 | 9.22166 | 0.0851484 | 0.0948766 |
| loss2_dict_N2000 | 9.11123 | 0.0802478 | 0.089976 |
| loss2_dict_N20000 | 8.2034 | 0.0807192 | 0.0904474 |
| loss3_stopgrad | 2.71059 | 0.0168314 | 0.0265597 |
| loss3 | 16.0929 | 0.150504 | 0.160233 |

Best method in this setting:

```text
loss3
```

## GitHub Policy

Keep large artifacts ignored by Git:

```text
*.pt
*.pth
*.pkl
*.npy
*.npz
adv_robust/
run_logs/
results/
1D_Burgers/datasets/
fno_training_runs/
deeponet_training_runs/
```

The markdown runbooks and shell scripts should stay in the repository/workspace because they explain how to regenerate, run, and back up the ignored files.
