# Adversarial Training Detailed Logging Update 2026-05-31

This records the current adversarial-training logging contract in `tools/adversarial_training.py`.

## Training Schedule

- Default training length is now `1000` epochs for Burgers, Darcy, and NS2D.
- Clean evaluation runs at baseline epoch `0` and after every epoch. By default `--eval-max-samples 0` evaluates the full samples in each dataset. With the full generalization set this gives `52` dataset-level curves per task: train, test, and 50 generated generalization datasets.
- Checkpoints are saved every `200` epochs by default, plus the final checkpoint. Use `--checkpoint-every-epochs N` to change this.

## Fixed Same-Index Attack Probes

The run fixes a small set of train-source indices, default `5` spread across the train split, or explicit indices from `--attack-probe-indices`. For each probe epoch it saves the attack result for the same source index.

Default attack randomness is reduced for comparability:

- epsilon jitter defaults to fixed `1.0`;
- alpha jitter defaults to fixed `1.0`;
- random start defaults to off;
- Darcy binary score noise defaults to `0.0`;
- Darcy binary candidate pool defaults to deterministic top-k.

The model still changes after every epoch, so the same `x`, same epsilon, and same alpha can produce different adversarial perturbations because the attack gradient comes from the current model.

## Main Output Files

Each task directory contains:

- `eval_metrics.csv`: one row per dataset per evaluation epoch, including `rmse`, `mae`, and `relative_l2`.
- `eval_split_summary.csv`: train/test/generalization/ALL summaries for each evaluation epoch.
- `attack_batches.csv`: one row per attacked training batch, including clean loss before attack, adversarial loss after attack, and attack gain.
- `attack_epoch_summary.csv`: epoch-level averages of attack loss gain, epsilon/alpha, perturbation size, and attack throughput.
- `attack_probe_samples.csv`: fixed-index per-sample attack diagnostics.
- `attack_probe_epochs.csv`: expected/captured probe counts and the matching NPZ path per epoch.
- `attack_probe_samples/*.npz`: arrays for fixed probe samples.

## Attack Gain Definition

For a fixed probe sample and current model at epoch `t`:

```text
clean_loss_before_attack_sample = MSE(model_t(x_clean), solver(x_clean))
adv_loss_after_attack_sample    = MSE(model_t(x_adv),   solver(x_adv))
attack_loss_gain_sample         = adv_loss_after_attack_sample - clean_loss_before_attack_sample
```

The epoch-level attack gain is the batch-size-weighted mean of the same clean-before and adv-after quantities across attacked batches.

## Probe Arrays

The NPZ files always include:

- `x_clean`
- `x_adv`
- `delta = x_adv - x_clean`
- `probe_rank`
- `source_index`
- `attack_global_step`
- `local_batch_idx`

By default they also include:

- `y_clean`
- `y_adv`

For NS2D they additionally include:

- `x0_clean`
- `x0_adv`
- `delta_initial = x0_adv - x0_clean`

This is the part meant to show whether the perturbation becomes harder, sharper, more high-frequency, or otherwise more structured as adversarial self-training proceeds.
