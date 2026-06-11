# Burgers Wideparam Loss1/Loss2/Loss3 Retrain Plan - 2026-06-11

This plan prepares a fresh Burgers adversarial-training sweep against the final
wide-parameter loss3-targeted generalization dataset:

`generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00`

## Training Targets

| run | epochs | objective | default batch | optimizer microbatch | default remat |
|---|---:|---|---:|---:|---|
| `loss1` | `8000` | `MSE(model(x_adv), model(x_clean).detach())` for attack generation | `1350` | `32` | `chunk` |
| `loss2` | `2000` | `MSE(model(x_adv), solver(x_clean).detach())` for attack generation | `1350` | `32` | `chunk` |
| `loss3` | `1000` | `MSE(model(x_adv), solver(x_adv))` with solver gradient | `1350` | `32` | `chunk` |

Training target remains solver-label self-training: the optimizer trains on
`model(x_adv) -> solver(x_adv).detach()` for all three objectives.

## Runtime And VRAM Estimate

Observed old long-run timing with the earlier round03 dataset and batch-480 style
configuration:

| old result | epochs | wall hours | average sec/epoch |
|---|---:|---:|---:|
| loss1 final | `8000` | `11.9919` | `5.396` |
| loss2 final | `2000` | `12.4230` | `22.361` |
| loss3 final | `1500` | `19.3331` | `46.399` |

For the new launcher, the default is max-train-batch `1350` with chunk remat.
A prior Burgers remat sweep observed `batch=1350`, `10` attack steps, chunk `50`
as a successful full-train-batch configuration with about `10.73 GiB` peak
allocated and `12.19 GiB` reserved. The new training uses `5` attack steps by
default, so loss3 should fit with headroom in this max-batch profile.

Practical estimates before running the new VRAM probe:

| objective | expected peak allocated | comment |
|---|---:|---|
| loss1 | `~4-8 GiB` | attack does not use solver backward; solver-label training target still uses solver forward |
| loss2 | `~5-10 GiB` | attack uses clean solver forward but no solver backward |
| loss3 | `~8-12 GiB` with chunk max-batch; `~25-30 GiB` if forced to no-remat memory-filling profile | solver forward/backward is the expensive path |

Because loss1/loss2 should not fill the 32GB V100 individually, the main launcher
defaults to running them concurrently. Loss3 runs alone afterward.

## Commands

VRAM/throughput probe only:

```bash
bash tools/probe_burgers_wideparam_loss123_vram_20260611.sh
```

Dry-run the full workflow command lines:

```bash
DRY_RUN=1 bash tools/run_burgers_wideparam_loss123_retrain_20260611.sh
```

Run the full workflow:

```bash
bash tools/run_burgers_wideparam_loss123_retrain_20260611.sh
```

Force sequential loss1/loss2 if the parallel probe is too tight:

```bash
RUN_LOSS12_PARALLEL=0 bash tools/run_burgers_wideparam_loss123_retrain_20260611.sh
```

Use a memory-filling loss3 profile instead of max-batch chunk remat:

```bash
LOSS3_BATCH=480 LOSS3_BURGERS_SOLVER_REMAT=none bash tools/run_burgers_wideparam_loss123_retrain_20260611.sh
```

## Recorded Outputs

Each run keeps the existing `tools/adversarial_training.py` logging contract:

- `train_steps.csv`
- `attack_batches.csv`
- `attack_epoch_summary.csv`
- `attack_epsilon_bucket_summary.csv`
- `optimizer_steps.csv`
- `eval_metrics.csv`
- `eval_split_summary.csv`
- `evaluation_passes.csv`
- `memory.csv`
- `checkpoints.csv`
- `attack_probe_samples.csv`
- `attack_probe_epochs.csv`
- `attack_probe_samples/*.npz`
- `data_range_summary.json`
- `summary.json`

The plotting script merges those outputs and writes:

- `visualizations/burgers_wideparam_loss123_retrain_20260611/`
- `docs/burgers_wideparam_loss123_retrain_report_20260611.md`

The merged plots include clean train/test/generalization RMSE and relative L2,
attack loss gain, epsilon-bucket/probe diagnostics, fixed-probe delta spectrum
statistics, and CUDA memory curves.
