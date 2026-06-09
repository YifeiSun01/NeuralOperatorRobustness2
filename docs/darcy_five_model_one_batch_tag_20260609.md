# Darcy/C-flow Five-Model One-Batch Tag Result - 2026-06-09

## Status

Completed one GPU-only Darcy/C-flow tag batch for the five formal models requested by the user:

- `baseline`
- `loss1`
- `loss2`
- `loss3`
- `physical_source`

This was intentionally a one-batch run, not a full 50-file generalization sweep.

## Protocol

Observed from `forensics/darcy_five_model_one_batch_tag_20260609/result.json`:

- Dataset batch source: `generalization_datasets_darcy_lossdrop50_selected_20260607/darcy/darcy_lossdrop_pool_soft_l4_h10_b10_02.pt`
- Dataset sample count: `48`
- Candidate batch sizes after clamping to dataset size: `48, 40, 32, 24, 16, 8`
- Common batch used across all five models: `48`
- Attack objective: shared `loss3`
- Attack type: `binary_steepest_replace`
- Attack steps: `1`
- Epsilon fraction: `0.025`
- Ranking metric: ascending `loss_gain_mean`

The shared `loss3` objective was used for all five models so the final loss gain is measured on the same scale.

## GPU Preflight

Observed from `forensics/darcy_five_model_one_batch_tag_20260609/gpu_preflight.json`:

- GPU: Tesla V100-SXM2-32GB
- Driver/CUDA from `nvidia-smi`: driver `580.76.05`, CUDA `13.0`
- PyTorch: `2.8.0+cu126`
- PyTorch CUDA: `12.6`
- Device capability: `sm_70`
- PyTorch CUDA architecture list includes `sm_70`
- JAX version: `0.10.0`
- JAX backend: `gpu`
- JAX device: `cuda:0`
- `XLA_PYTHON_CLIENT_PREALLOCATE=false`

Observed GPU sanity checks completed in both PyTorch and JAX.

## Result

Observed from `forensics/darcy_five_model_one_batch_tag_20260609/one_batch_summary_by_model.csv`:

| rank | model | clean loss mean | final tag loss mean | loss gain mean | relative gain mean |
|---:|---|---:|---:|---:|---:|
| 1 | loss3 | 9.599584697520906e-08 | 1.1024506788951764e-07 | 1.424922091430858e-08 | 0.17596435946567604 |
| 2 | physical_source | 1.5705951295075238e-07 | 1.835484744934964e-07 | 2.648896154274401e-08 | 0.18621729942969978 |
| 3 | loss1 | 2.9563728087822483e-07 | 3.371411810467369e-07 | 4.1503900168512096e-08 | 0.1633932989401122 |
| 4 | loss2 | 2.424386837951431e-07 | 2.854224743960761e-07 | 4.298379060093301e-08 | 0.2081057421552638 |
| 5 | baseline | 3.277261191314551e-07 | 4.460621176131478e-07 | 1.1833599848169267e-07 | 0.41604626675446826 |

Observed least-burst model by mean loss gain: `loss3`.

## Source Files

- Script: `tools/run_darcy_five_model_one_batch_tag_20260609.py`
- Result JSON: `forensics/darcy_five_model_one_batch_tag_20260609/result.json`
- Summary CSV: `forensics/darcy_five_model_one_batch_tag_20260609/one_batch_summary_by_model.csv`
- Batch probe CSV: `forensics/darcy_five_model_one_batch_tag_20260609/batch_probe_by_model.csv`
- GPU preflight JSON: `forensics/darcy_five_model_one_batch_tag_20260609/gpu_preflight.json`

## Inference

Inference from this single batch: for this sampled generalization batch, the `loss3`-trained Darcy/C-flow model is the least burst under the shared solver-consistent tag objective, because it has the smallest average post-tag loss increase.

This conclusion is limited to one batch of 48 samples. A full generalization sweep over all 50 `.pt` files should be run before making a dataset-level conclusion.
