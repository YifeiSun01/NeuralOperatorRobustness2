# Attack Run Dependency Manifest

Generated on 2026-05-11 in `/workspace/NeuralOperatorRobustness2`.

This file records the files, data, models, commands, outputs, and current status for the Burgers loss-attack sweeps so the run dependencies do not get mixed up.

## Main Record Files

The existing result/setting record that matches the "lNorm / Linf / epsilon / alpha / attack win rate / ratio" description is:

- `BURGERS_BATCH100_18_SETTINGS_FULL_SUMMARY.md`
  - Human summary of 18 batch-100 Burgers settings.
  - Combines previous completed settings from:
    - `results/burgers_loss3_candidate_batch100_random100`
    - `results/burgers_loss3_good_bad_7settings_batch100_random100_losses`
  - Reports method wins, win rates, and final/initial true-loss ratios.

Other closely related summaries:

- `BURGERS_LOSS3_CANDIDATE_BATCH100_FULL_SUMMARY.md`
  - Summary of 11 completed candidate settings.
- `BURGERS_BATCH50_LOSS3_CANDIDATE_SUMMARY.md`
  - Earlier batch-50 candidate summary.
- `results/burgers_attack_ratio_summary/burgers_attack_ratio_by_parameter.md`
  - Ratio summary grouped by parameter.
- `results/burgers_attack_ratio_summary/burgers_attack_ratio_runs.md`
  - Per-run ratio summary.

## Main Run/Summary Scripts

Future settings were encoded in:

- `run_burgers_loss3_18setting_batch100_losses_parallel6.sh`
  - Contains the 18 future/batch100 settings.
  - Original script runs up to 6 jobs in parallel. For future runs, use the sequential command style below instead.

Related previous sweep script:

- `run_burgers_loss3_good_bad_7settings_batch100_losses.sh`
  - Contains 7 "good/bad" settings from an earlier round.

Attack runners:

- `run_burgers_corrected_oldstyle_5loss.py`
  - FNO Burgers runner.
  - Compares 7 attack methods:
    - `loss1`
    - `loss2_fixed`
    - `loss2_dict_N200`
    - `loss2_dict_N2000`
    - `loss2_dict_N20000`
    - `loss3_stopgrad`
    - `loss3`
- `run_burgers_deeponet_corrected_oldstyle_5loss.py`
  - DeepONet wrapper around the FNO runner.

Summary scripts:

- `summarize_burgers_corrected_oldstyle_true_losses.py`
  - Produces:
    - `true_loss_summary_aggregate.csv`
    - `true_loss_summary_aggregate.md`
    - `true_loss_summary_all_indices.csv`
- `summarize_burgers_attack_ratios.py`
  - Produces:
    - `burgers_attack_ratio_runs.csv`
    - `burgers_attack_ratio_runs.md`
    - `burgers_attack_ratio_by_parameter.csv`
    - `burgers_attack_ratio_by_parameter.md`

## Virtual Environment

Use this environment:

- `adv_robust/`

Activation:

```bash
cd /workspace/NeuralOperatorRobustness2
source adv_robust/bin/activate
```

Verified imports:

- `torch` is installed in `adv_robust`
- `jax` is installed in `adv_robust`
- `numpy`, `scipy`, `matplotlib`, and `pandas` are installed in `adv_robust`

## Required Shared Runtime Components

Python source files used directly or indirectly by the attack runners:

- `run_burgers_corrected_oldstyle_5loss.py`
- `run_burgers_deeponet_corrected_oldstyle_5loss.py`
- `summarize_burgers_corrected_oldstyle_true_losses.py`
- `summarize_burgers_attack_ratios.py`
- `loss_attack_common.py`
- `solvers.py`
- `plot_burgers_deeponet_predictions.py`
- `train_burgers_deeponet_deepxde.py`

The runners also use the JAX Burgers solver through `solvers.py`.

## Required Test Data

Both the FNO runner and DeepONet runner call `default_test_path(nu)` from `run_burgers_corrected_oldstyle_5loss.py`.

Expected nu=0.001 test split:

- `1D_Burgers/datasets/1D/Burgers/batched_exponax_splits/dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45/dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45_test.pt`

Current status:

- Present locally.
- Regenerated on 2026-05-11 with:
  - `1D_Burgers/data_generation/generate_burgers_exponax_batched.py --num-records 1500 --nu 0.01,0.001 --batch-size 256 --seed 45`
  - `tools/split_burgers_dataset.py --seed 1234 --test-count 150`
- Shape verified:
  - train: `(1350, 1024)`
  - test: `(150, 1024)`

Expected nu=0.01 test split:

- `1D_Burgers/datasets/1D/Burgers/batched_exponax_splits/dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.01_t1.0_seed45/dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.01_t1.0_seed45_test.pt`

Current status:

- Present locally.
- Regenerated on 2026-05-11 with:
  - `1D_Burgers/data_generation/generate_burgers_exponax_batched.py --num-records 1500 --nu 0.01,0.001 --batch-size 256 --seed 45`
  - `tools/split_burgers_dataset.py --seed 1234 --test-count 150`
- Shape verified:
  - train: `(1350, 1024)`
  - test: `(150, 1024)`

## Required FNO Models

FNO nu=0.001 default model path used by `run_burgers_corrected_oldstyle_5loss.py`:

- `1D_Burgers/trained_models/attack_ready/burgers_nu0.001_fno1d_500/checkpoints/pytorch_fno1d_500.pt`

Current status:

- Present locally.
- This is why the copied nu=0.001 outputs are internally complete.

FNO nu=0.01 default old-runner model path used by `run_burgers_corrected_oldstyle_5loss.py`:

- `tmp_old_runner_inputs_b01/burgers_1d_FNO_model_trainedby_dim1d_nx1024_N1500_solver=exponax_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.01_t1.0_seed45.pth`

Current status:

- Present locally as a compatibility symlink created on 2026-05-11.
- The symlink points to:
  - `fno_training_runs/burgers_nu0p01_fno1d_500/burgers_1d/checkpoints/fno1d_pytorch.pt`
- This compatibility path is needed because the attack runner validates `nu` by parsing the model path string, and the existing `burgers_nu0p01...` checkpoint path is parsed incorrectly by that validator.

Available FNO nu=0.01 checkpoint in the current project:

- `fno_training_runs/burgers_nu0p01_fno1d_500/burgers_1d/checkpoints/fno1d_pytorch.pt`

Current status:

- Present locally.
- Validated on 2026-05-11: `run_burgers_corrected_oldstyle_5loss.load_model()` loads this checkpoint successfully.
- The old default `.pth` path is now a compatibility symlink to this load-compatible checkpoint.

## Required DeepONet Models

DeepONet nu=0.01 default run directory:

- `deeponet_training_runs/burgers_nu0p01_deeponet_lu_ref_50k`

DeepONet nu=0.01 checkpoint:

- `deeponet_training_runs/burgers_nu0p01_deeponet_lu_ref_50k/checkpoints/deeponet_burgers_nu0p01.pt`

DeepONet nu=0.01 config/log files:

- `deeponet_training_runs/burgers_nu0p01_deeponet_lu_ref_50k/training_logs/config.json`
- `deeponet_training_runs/burgers_nu0p01_deeponet_lu_ref_50k/training_logs/dataset_info.json`
- `deeponet_training_runs/burgers_nu0p01_deeponet_lu_ref_50k/training_logs/summary.json`

Current status:

- Present locally.
- DeepONet nu=0.01 model exists.
- Validated on 2026-05-11: `run_burgers_deeponet_corrected_oldstyle_5loss.load_deeponet_model()` loads this checkpoint successfully.

DeepONet nu=0.001 default checkpoint:

- `deeponet_training_runs/burgers_nu0p001_deeponet_lu_ref_50k/checkpoints/deeponet_burgers_nu0.001.pt`

Current status:

- Present locally.
- Validated on 2026-05-11: the FNO attack runner loads the default FNO nu=0.001 checkpoint successfully.

## Required Dictionary Files

The attack runner auto-loads dictionary targets from `results/dictionaries`, using sizes 200, 2000, and 20000 for methods:

- `loss2_dict_N200`
- `loss2_dict_N2000`
- `loss2_dict_N20000`

nu=0.001 dictionary files:

- `results/dictionaries/burgers_nu0p001_N200_seed1000045/inputs_dim1d_nx1024_N200_solver=jax_exponax_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed1000045.pt`
- `results/dictionaries/burgers_nu0p001_N200_seed1000045/outputs_dim1d_nx1024_N200_solver=jax_exponax_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed1000045.pt`
- `results/dictionaries/burgers_nu0p001_N2000_seed1000045/inputs_dim1d_nx1024_N2000_solver=jax_exponax_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed1000045.pt`
- `results/dictionaries/burgers_nu0p001_N2000_seed1000045/outputs_dim1d_nx1024_N2000_solver=jax_exponax_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed1000045.pt`
- `results/dictionaries/burgers_nu0p001_N20000_seed1000045/inputs_dim1d_nx1024_N20000_solver=jax_exponax_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed1000045.pt`
- `results/dictionaries/burgers_nu0p001_N20000_seed1000045/outputs_dim1d_nx1024_N20000_solver=jax_exponax_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed1000045.pt`

nu=0.01 dictionary files:

- `results/dictionaries/burgers_nu0p01_N200_seed1000045/inputs_dim1d_nx1024_N200_solver=jax_exponax_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.01_t1.0_seed1000045.pt`
- `results/dictionaries/burgers_nu0p01_N200_seed1000045/outputs_dim1d_nx1024_N200_solver=jax_exponax_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.01_t1.0_seed1000045.pt`
- `results/dictionaries/burgers_nu0p01_N2000_seed1000045/inputs_dim1d_nx1024_N2000_solver=jax_exponax_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.01_t1.0_seed1000045.pt`
- `results/dictionaries/burgers_nu0p01_N2000_seed1000045/outputs_dim1d_nx1024_N2000_solver=jax_exponax_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.01_t1.0_seed1000045.pt`
- `results/dictionaries/burgers_nu0p01_N20000_seed1000045/inputs_dim1d_nx1024_N20000_solver=jax_exponax_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.01_t1.0_seed1000045.pt`
- `results/dictionaries/burgers_nu0p01_N20000_seed1000045/outputs_dim1d_nx1024_N20000_solver=jax_exponax_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.01_t1.0_seed1000045.pt`

Current status:

- The listed dictionary files are present locally.

## Existing 18-Setting Sweep Status

Output root:

- `results/burgers_loss3_18setting_batch100_random100_losses_parallel6`

Completed settings with summary files:

| Setting | Status |
|---|---|
| `fno_nu0p001_linf_eps0p15_alpha0p015_batch100_seed20260511` | complete |
| `fno_nu0p001_linf_eps0p25_alpha0p025_batch100_seed20260511` | complete |
| `fno_nu0p001_linf_eps0p35_alpha0p035_batch100_seed20260511` | complete |
| `fno_nu0p001_linf_eps0p5_alpha0p005_batch100_seed20260511` | complete |
| `fno_nu0p001_linf_eps0p5_alpha0p08_batch100_seed20260511` | complete |
| `fno_nu0p001_linf_eps1p0_alpha0p04_batch100_seed20260511` | complete |

Partial/failed directories created by an interrupted resume attempt before the missing datasets were regenerated:

| Setting | Status | Blocking issue |
|---|---|---|
| `fno_nu0p01_linf_eps0p15_alpha0p006_batch100_seed20260511` | complete | first sequential rerun completed; `loss3` wins 95/100 |
| `fno_nu0p01_linf_eps0p2_alpha0p008_batch100_seed20260511` | failed before attack | rerun sequentially with explicit `--model_path` |
| `fno_nu0p01_linf_eps0p3_alpha0p003_batch100_seed20260511` | failed before attack | rerun sequentially with explicit `--model_path` |
| `fno_nu0p01_linf_eps0p3_alpha0p03_batch100_seed20260511` | failed before attack | rerun sequentially with explicit `--model_path` |
| `fno_nu0p01_linf_eps0p45_alpha0p018_batch100_seed20260511` | failed before attack | rerun sequentially with explicit `--model_path` |
| `fno_nu0p01_linf_eps0p75_alpha0p03_batch100_seed20260511` | failed before attack | rerun sequentially with explicit `--model_path` |

Not yet attempted in the resumed 18-setting output root:

| Setting | Status | Blocking issue |
|---|---|---|
| `fno_nu0p01_l2_eps2p0_alpha0p0375_batch100_seed20260511` | not run | ready to run sequentially with explicit `--model_path` |
| `fno_nu0p01_l2_eps9p0_alpha0p17_batch100_seed20260511` | not run | ready to run sequentially with explicit `--model_path` |
| `deeponet_nu0p01_linf_eps0p05_alpha0p002_batch100_seed20260511` | not run | ready to run sequentially |
| `deeponet_nu0p01_linf_eps0p75_alpha0p03_batch100_seed20260511` | not run | ready to run sequentially |
| `deeponet_nu0p01_l2_eps1p0_alpha0p01875_batch100_seed20260511` | not run | ready to run sequentially |
| `deeponet_nu0p01_l2_eps12p0_alpha0p225_batch100_seed20260511` | not run | ready to run sequentially |

## Sequential Run Command Template

Use one setting at a time. Do not use the old 6-way parallel script.

Ready-to-use sequential resume script:

```bash
cd /workspace/NeuralOperatorRobustness2
bash run_burgers_loss3_18setting_resume_sequential.sh
```

This script skips settings that already have both:

- `true_loss_summary_aggregate.csv`
- `ratio_summary/burgers_attack_ratio_by_parameter.csv`

For FNO `nu=0.01`, the runner uses the default old-runner path, which is now a compatibility symlink to the existing validated checkpoint.

Common setup:

```bash
cd /workspace/NeuralOperatorRobustness2
source adv_robust/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export BATCH_SIZE=100
export SEED=20260511
export STEPS=100
export OUT_BASE=results/burgers_loss3_18setting_batch100_random100_losses_sequential
export INDICES="$(python3 -c 'import random; random.seed(20260511); print(" ".join(map(str, sorted(random.sample(range(150), 100)))))')"
mkdir -p "$OUT_BASE"
```

FNO command shape:

```bash
python -u run_burgers_corrected_oldstyle_5loss.py \
  --nu NU \
  --domain 2.0 \
  --nx 1024 \
  --t_final 1.0 \
  --dt 0.001 \
  --norm NORM \
  --epsilon EPSILON \
  --alpha ALPHA \
  --steps "$STEPS" \
  --indices $INDICES \
  --output_root "$OUT_BASE/TAG" \
  --tag "TAG_domain2_NORMTAG_epsEPSTAG_alphaALPHATAG_steps100_index{index}_batch100_losses_rs" \
  --batch_tag "TAG_domain2_NORMTAG_epsEPSTAG_alphaALPHATAG_steps100_batch100_losses_seed20260511_rs" \
  --record_steps losses
python -u summarize_burgers_corrected_oldstyle_true_losses.py --result_root "$OUT_BASE/TAG" --outdir "$OUT_BASE/TAG"
python -u summarize_burgers_attack_ratios.py --result_root "$OUT_BASE/TAG" --outdir "$OUT_BASE/TAG/ratio_summary"
```

DeepONet command shape:

```bash
python -u run_burgers_deeponet_corrected_oldstyle_5loss.py \
  --nu NU \
  --domain 2.0 \
  --nx 1024 \
  --t_final 1.0 \
  --dt 0.001 \
  --norm NORM \
  --epsilon EPSILON \
  --alpha ALPHA \
  --steps "$STEPS" \
  --indices $INDICES \
  --output_root "$OUT_BASE/TAG" \
  --tag "TAG_domain2_NORMTAG_epsEPSTAG_alphaALPHATAG_steps100_index{index}_batch100_losses_rs" \
  --batch_tag "TAG_domain2_NORMTAG_epsEPSTAG_alphaALPHATAG_steps100_batch100_losses_seed20260511_rs" \
  --record_steps losses
python -u summarize_burgers_corrected_oldstyle_true_losses.py --result_root "$OUT_BASE/TAG" --outdir "$OUT_BASE/TAG"
python -u summarize_burgers_attack_ratios.py --result_root "$OUT_BASE/TAG" --outdir "$OUT_BASE/TAG/ratio_summary"
```

After all settings finish:

```bash
python -u summarize_burgers_attack_ratios.py \
  --result_root "$OUT_BASE" \
  --outdir "$OUT_BASE/global_ratio_summary"
```

## 18 Future Settings From The Script

Already complete in copied `parallel6` root; do not rerun unless intentionally regenerating:

```text
fno|0.001|inf|0.15|0.015
fno|0.001|inf|0.25|0.025
fno|0.001|inf|0.35|0.035
fno|0.001|inf|1.0|0.04
fno|0.001|inf|0.5|0.005
fno|0.001|inf|0.5|0.08
```

Remaining/current 18-setting script status after dataset/model validation; run incomplete settings with `run_burgers_loss3_18setting_resume_sequential.sh`:

```text
fno|0.01|inf|0.15|0.006|complete
fno|0.01|inf|0.2|0.008|rerun needed
fno|0.01|inf|0.45|0.018|rerun needed
fno|0.01|inf|0.75|0.03|rerun needed
fno|0.01|inf|0.3|0.003|rerun needed
fno|0.01|inf|0.3|0.03|rerun needed
fno|0.01|2|2.0|0.0375|not run
fno|0.01|2|9.0|0.17|not run
deeponet|0.01|inf|0.05|0.002|not run
deeponet|0.01|inf|0.75|0.03|not run
deeponet|0.01|2|1.0|0.01875|not run
deeponet|0.01|2|12.0|0.225|not run
```

## Regenerated Or Validated On 2026-05-11

Regenerated dataset files:

- `1D_Burgers/datasets/1D/Burgers/batched_exponax/dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.01_t1.0_seed45.pt`
- `1D_Burgers/datasets/1D/Burgers/batched_exponax/dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45.pt`
- `1D_Burgers/datasets/1D/Burgers/batched_exponax_splits/dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.01_t1.0_seed45/dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.01_t1.0_seed45_train.pt`
- `1D_Burgers/datasets/1D/Burgers/batched_exponax_splits/dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.01_t1.0_seed45/dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.01_t1.0_seed45_test.pt`
- `1D_Burgers/datasets/1D/Burgers/batched_exponax_splits/dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45/dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45_train.pt`
- `1D_Burgers/datasets/1D/Burgers/batched_exponax_splits/dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45/dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45_test.pt`

Validated models:

- `fno_training_runs/burgers_nu0p01_fno1d_500/burgers_1d/checkpoints/fno1d_pytorch.pt`
- `deeponet_training_runs/burgers_nu0p01_deeponet_lu_ref_50k/checkpoints/deeponet_burgers_nu0p01.pt`
- `1D_Burgers/trained_models/attack_ready/burgers_nu0.001_fno1d_500/checkpoints/pytorch_fno1d_500.pt`

Compatibility symlink created for FNO `nu=0.01`:

- `tmp_old_runner_inputs_b01/burgers_1d_FNO_model_trainedby_dim1d_nx1024_N1500_solver=exponax_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.01_t1.0_seed45.pth`

## R2 Backup Runbook

The authoritative backup checklist and exact R2 copy commands are recorded in:

- `R2_BACKUP_AND_ATTACK_RUNBOOK.md`

The large data/model/result files are intentionally ignored by GitHub, but they must be copied to Cloudflare R2 under:

- `R2:neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected`

New first completed setting after validation:

- `results/burgers_loss3_18setting_batch100_random100_losses_parallel6/fno_nu0p01_linf_eps0p15_alpha0p006_batch100_seed20260511`

For this setting, `loss3` was best with ratio mean `16.0929`, increase mean `0.150504`, and final true loss mean `0.160233`.
