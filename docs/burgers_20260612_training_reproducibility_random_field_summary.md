# Burgers 2026-06-12 Training Reproducibility And Random-Field Noise Training Summary

This note records the retraining schedule, checkpoint reproducibility interpretation, epoch-time estimates, and the new random-field noise training modes.

## Retrained Adversarial Models

Latest retrain runs:

| objective | run directory | final epoch | final checkpoint |
|---|---|---:|---|
| loss1 | `adversarial_training_runs/burgers_wideparam_loss1_8000ep_retrain_20260611` | 8000 | `burgers/checkpoints/burgers_epoch8000_step008000.pt` |
| loss2 | `adversarial_training_runs/burgers_wideparam_loss2_2000ep_retrain_20260611` | 2000 | `burgers/checkpoints/burgers_epoch2000_step002000.pt` |
| loss3 | `adversarial_training_runs/burgers_wideparam_loss3_1000ep_retrain_20260611` | 1000 | `burgers/checkpoints/burgers_epoch1000_step001000.pt` |

A previous final-model SVD workflow used loss3 epoch1500 from the earlier round03 long training series. The newest comparison-dense plots use loss3 epoch1000 from the latest retrain run listed above.

## Twelve-And-A-Half-Hour Epoch Estimate

The working estimate discussed for a roughly 12.5 hour budget was:

| objective | estimated/referenced epoch count near 12.5h |
|---|---:|
| loss1 | 8000 |
| loss2 | 2000 |
| loss3 | around 1000 for latest retrain; older long-run comparison used 1500 |

The exact number depends on batch size, solver rematerialization mode, GPU state, and whether loss1/loss2 run concurrently.

## Why Same Training Method Can Produce Different Checkpoints

The comparison between old and latest retrain plots showed:

- generalization sample manifests were the same across old and new P2Q2 comparison groups;
- baseline traces were exactly identical, because the same baseline checkpoint and deterministic evaluation path were reused;
- loss1/loss2/loss3 traces differed because their checkpoint files differed.

Interpretation: the method can be the same while the resulting checkpoint is not bitwise identical. The training script sets Python, NumPy, PyTorch, and CUDA seeds, and batch order is generated deterministically by epoch seed. However, strict bitwise deterministic CUDA behavior is not forced globally. Long adversarial/self-training trajectories can amplify small numerical differences from GPU reductions, FFTs, solver paths, attack deltas, and optimizer updates.

Therefore, same AutoCell/adversarial training method means same algorithm, not guaranteed identical weights.

## Random-Field Noise Training Modes

The new random-field mode adds random Gaussian or Matern-field perturbations to Burgers input x instead of computing delta by adversarial attack. The perturbation parameters are randomized by epoch/batch/sample, including family, correlation length, and Matern nu choices.

Two target modes were added:

| mode | target definition | interpretation |
|---|---|---|
| clean-y | train model(x plus random delta) toward original clean y | intentionally unrealistic control: y is unchanged even though x changes |
| solver-y | train model(x plus random delta) toward solver(x plus random delta) | physically consistent random-noise augmentation |

The implementation lives in `tools/adversarial_training.py` and is launched through:

`tools/run_burgers_wideparam_random_field_training_20260612.sh`

Watcher/chain script:

`tools/watch_burgers_retrain_then_random_field_20260612.sh`

## Random-Field Run Status Observed Locally

| run | status observed | final checkpoint |
|---|---|---|
| `burgers_wideparam_random_field_clean_y_2000ep_20260612` | completed configured 2000 epochs | `adversarial_training_runs/burgers_wideparam_random_field_clean_y_2000ep_20260612/burgers/checkpoints/burgers_epoch2000_step002000.pt` |
| `burgers_wideparam_random_field_solver_y_2000ep_20260612` | directory and partial records exist; no final summary observed in this audit | not confirmed in this audit |

The completed clean-y run reported about 4034 seconds, or 67.2 minutes, for 2000 epochs on this machine and configuration.

## Random-Field Noise Energy Constraint

The configured intent was to keep random perturbation energy small, around a 5 percent scale. In implementation, random-field deltas are normalized per sample and scaled by the configured epsilon. If a strict exactly-5-percent energy audit is needed, run a follow-up check over `attack_batches.csv` or probe tensors from the random-field run.

## Related Scripts

- `tools/run_burgers_wideparam_loss123_retrain_20260611.sh`
- `tools/run_burgers_wideparam_svd25_then_retrain_upload_20260611.sh`
- `tools/run_burgers_wideparam_random_field_training_20260612.sh`
- `tools/watch_burgers_retrain_then_random_field_20260612.sh`
- `tools/adversarial_training.py`

## Practical Reproducibility Recommendation

For exact reproducibility experiments, run a short duplicate test such as 20 or 50 epochs twice with the same seed, then compare checkpoint hashes, weight differences, clean metrics, and attack traces. For production scientific reporting, store checkpoint SHA256 values and manifest hashes next to each figure bundle.
