# Adversarial Training Run

- Run directory: `adversarial_training_runs/burgers_p2q2_loss1_save1000_2000_v5_20260602`
- Tasks: `burgers`
- Smoke: `False`
- Label mode: `solver`
- Training data mode: `adv-only`
- Burgers attack loss objective: `loss1`

Each task subdirectory contains `train_steps.csv`, `attack_batches.csv`, `attack_epoch_summary.csv`, `attack_epsilon_bucket_summary.csv`, `optimizer_steps.csv`, `eval_metrics.csv`, `eval_split_summary.csv`, `evaluation_passes.csv`, `memory.csv`, checkpoints, `attack_probe_samples.csv`, `attack_probe_epochs.csv`, `attack_probe_samples/*.npz`, `data_range_summary.json`, and `summary.json`.
Training data modes: `adv-only` uses only attacked solver pairs; `clean-plus-adv` trains each batch on clean solver pairs plus newly attacked solver pairs, doubling the training examples per attack batch.
Default training now uses the full original train split for every epoch; pass `--<task>-train-max N` only for debugging caps, or `0` for full.

Default attack policy:
- Burgers: p=2/q=2 RMS-L2 fast-replace attack in the corrected pipeline. The attack objective can be loss1, loss2, or loss3: loss1 uses no solver; loss2 uses fixed clean solver output without solver backward; loss3 uses attacked solver output with solver backward. Optimizer training remains on attacked solver pairs unless training-data-mode is changed.
- Darcy: binary steepest-replace flips coefficient pixels; default attack batch is 256, optimizer microbatch is 32, flip budget is fixed by epsilon, score noise is off, and top-k replacement is deterministic by default.
- NS2D: L-infinity add attack on the initial vorticity frame; attack batch and optimizer batch stay 1:1 by default, epsilon/alpha are fixed, and random start is off for comparable same-index probes.

Evaluation is clean evaluation on train/test/generated datasets at baseline and after every epoch, giving 52 dataset-level curves per task when all 50 generated sets are present; checkpoints default to every 200 epochs plus final.
`attack_epsilon_bucket_summary.csv` writes one row per epsilon bucket per epoch. With the default 5 buckets, the random epsilon jitter range is split into 0-20%, 20-40%, 40-60%, 60-80%, and 80-100% bands, each with clean loss, attacked loss, loss increase, and relative loss increase statistics.
Attack probes save fixed train-set source indices after their actual training attack each probe epoch, including x_clean, x_adv, delta, y_clean/y_adv targets by default, per-sample clean/adv attack loss gain, and delta high-frequency summary metrics. NS2D probes also save x0_clean, x0_adv, and delta_initial.
