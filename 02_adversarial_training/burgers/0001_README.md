# Attack-Ready 1D Burgers FNO1d Models (500 Epochs)

Source run: `fno_training_runs/burgers_profile_500_four_way`

This folder keeps the two models intended for later 1D Burgers attacks:

- PyTorch baseline: `checkpoints/pytorch_fno1d_500.pt`
- JAX real/imag native checkpoint: `checkpoints/jax_real_imag_fno1d_500.pkl`
- JAX real/imag converted to PyTorch state_dict: `checkpoints/jax_real_imag_as_pytorch_fno1d_500.pt`

## Training Setup

- problem: 1D Burgers, nu=0.001
- model: FNO1d
- modes: 16
- width: 64
- num_layers: 4
- epochs: 500
- batch_size: 64
- eval_batch_size: 128
- lr: 1e-3
- weight_decay: 1e-4
- dtype: float32
- seed: 1234
- profile: memory_profile=true, time_profile=true

## Final Metrics

| framework | train relative L2 | test relative L2 | seconds total | sampled GPU peak MiB | train peak MiB | inference peak MiB |
|---|---:|---:|---:|---:|---:|---:|
| pytorch | 0.01681982 | 0.01751890 | 80.94 | 1407 | 505.84 | 36.28 |
| jax_real_imag | 0.01555738 | 0.01598039 | 89.16 | 2735 | 2733.00 | 2735.00 |

## Logs Included

- `training_logs/config.json`
- `training_logs/dataset_info.json`
- `training_logs/metrics_summary_all_four_models.csv`
- `training_logs/losses_pytorch.csv`
- `training_logs/losses_jax_real_imag.csv`
- `training_logs/memory_summary_pytorch.json`
- `training_logs/memory_summary_jax_real_imag.json`
- `training_logs/memory_phases_pytorch.csv`
- `training_logs/memory_phases_jax_real_imag.csv`
- `training_logs/attack_ready_manifest.json`

## About the JAX `.pkl` and Converted `.pt`

Native JAX checkpoints are saved as pickle because the model parameters are a JAX PyTree. For convenience, `jax_real_imag_as_pytorch_fno1d_500.pt` converts the JAX real/imag weights into a PyTorch `FNO1d` state_dict by combining Fourier real/imag tensors into complex spectral weights.

Conversion sanity check on the first 5 test samples:

- max absolute difference between native JAX real/imag inference and converted PyTorch inference: `0.003465116`
- relative L2 difference: `0.000368576`
