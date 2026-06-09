# Darcy/C-flow Five-Model One-Dataset 20-Step Tag Timing - 2026-06-09

## Status

Completed a timing run for one Darcy/C-flow generalization dataset with five formal models and 20-step tag attacks.

## Protocol

Observed from `forensics/darcy_five_model_one_dataset_tag20_timing_20260609/result.json`:

- Dataset: `generalization_datasets_darcy_lossdrop50_selected_20260607/darcy/darcy_lossdrop_pool_soft_l4_h10_b10_02.pt`
- Dataset index: `0`
- Batch size: `48`
- Models: `baseline`, `loss1`, `loss2`, `loss3`, `physical_source`
- Attack steps: `20`
- Shared objective: `loss3`
- Attack type: binary steepest replace
- Epsilon fraction: `0.025`

## Runtime

Observed runtime:

- Total wall time for one dataset across all five models: `29.999275830574334` seconds
- Mean attack wall time per model: `3.1563973486423493` seconds
- Estimated 50-dataset wall time by direct multiplication: `1499.9637915287167` seconds
- Estimated 50-dataset wall time in hours: `0.41665660875797683` hours

## Result On This Dataset

Observed ranking by ascending mean loss gain:

| rank | model | attack wall sec | clean loss mean | final tag loss mean | loss gain mean |
|---:|---|---:|---:|---:|---:|
| 1 | physical_source | 2.775472982786596 | 1.5705951295075238e-07 | 2.386436874779463e-07 | 8.158417452719391e-08 |
| 2 | loss2 | 2.7978224605321884 | 2.424386837951431e-07 | 3.602693124567698e-07 | 1.178306286616267e-07 |
| 3 | loss1 | 2.7644626293331385 | 2.9563728087822483e-07 | 4.2230398289433424e-07 | 1.266667020161094e-07 |
| 4 | loss3 | 2.8103448692709208 | 9.599584697520906e-08 | 3.8296407585865683e-07 | 2.869682288834478e-07 |
| 5 | baseline | 4.633883801288903 | 3.277261191314551e-07 | 8.32303507299533e-07 | 5.045773837271857e-07 |

Observed least-burst model for this one 20-step dataset timing run: `physical_source`.

## GPU Preflight

Observed GPU preflight:

- GPU: Tesla V100-SXM2-32GB
- PyTorch: `2.8.0+cu126`
- PyTorch CUDA: `12.6`
- Device capability: `sm_70`
- PyTorch CUDA architecture list includes `sm_70`
- JAX: `0.10.0`
- JAX backend: `gpu`
- JAX device: `cuda:0`

## Source Files

- Script: `tools/time_darcy_five_model_one_dataset_tag20_20260609.py`
- Result JSON: `forensics/darcy_five_model_one_dataset_tag20_timing_20260609/result.json`
- Timing CSV: `forensics/darcy_five_model_one_dataset_tag20_timing_20260609/timing_by_model.csv`
- Partial CSV: `forensics/darcy_five_model_one_dataset_tag20_timing_20260609/timing_by_model_partial.csv`
- GPU preflight JSON: `forensics/darcy_five_model_one_dataset_tag20_timing_20260609/gpu_preflight.json`

## Inference

Inference from this timing run: a full 50-dataset run at the same batch size and attack-step count should take roughly `25` minutes if runtime scales linearly and no other GPU workload interferes. This is a timing estimate, not a completed 50-dataset result.
