# Exponax T20 Dataset GIF Visualization - 2026-05-21

## Status

Generated CPU-only coolwarm heatmap GIFs for randomly selected train and test samples from the 2D recurrent Navier-Stokes Exponax T20 datasets.

## Observed Evidence

- Plotting command ran with `CUDA_VISIBLE_DEVICES=''` and printed `torch_cuda_available=False`.
- Source train dataset: `2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/real_initial_laxmap_single/train/dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_ntimepoints21_all_frames.pt`.
- Source test dataset: `2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/real_initial_laxmap_single/test/dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_test_ntimepoints21_all_frames.pt`.
- Observed train tensor shape: `(1150, 256, 256, 21)`.
- Observed test tensor shape: `(50, 256, 256, 21)`.
- Random seed: `20260521`.
- Selected train sample indices: `131`, `831`, `907`, `927`, `1084`.
- Selected test sample indices: `4`, `30`, `32`, `35`, `45`.
- Output directory: `2D_NS_FNO2d_recurrent/visualizations/exponax_t20_random_samples_coolwarm_20260521_0926`.
- Validation with PIL confirmed every generated GIF has size `(256, 256)` and `21` frames.
- The active GPU training run continued during CPU visualization; after generation, `progress_latest.json` showed epoch `18/500`, batch `50/72`, and `nvidia-smi` still showed `100%` GPU utilization.

## Output Files

- `2D_NS_FNO2d_recurrent/visualizations/exponax_t20_random_samples_coolwarm_20260521_0926/README.md`
- `2D_NS_FNO2d_recurrent/visualizations/exponax_t20_random_samples_coolwarm_20260521_0926/manifest.json`
- `2D_NS_FNO2d_recurrent/visualizations/exponax_t20_random_samples_coolwarm_20260521_0926/train_sample_0131_coolwarm_21frames.gif`
- `2D_NS_FNO2d_recurrent/visualizations/exponax_t20_random_samples_coolwarm_20260521_0926/train_sample_0831_coolwarm_21frames.gif`
- `2D_NS_FNO2d_recurrent/visualizations/exponax_t20_random_samples_coolwarm_20260521_0926/train_sample_0907_coolwarm_21frames.gif`
- `2D_NS_FNO2d_recurrent/visualizations/exponax_t20_random_samples_coolwarm_20260521_0926/train_sample_0927_coolwarm_21frames.gif`
- `2D_NS_FNO2d_recurrent/visualizations/exponax_t20_random_samples_coolwarm_20260521_0926/train_sample_1084_coolwarm_21frames.gif`
- `2D_NS_FNO2d_recurrent/visualizations/exponax_t20_random_samples_coolwarm_20260521_0926/test_sample_0004_coolwarm_21frames.gif`
- `2D_NS_FNO2d_recurrent/visualizations/exponax_t20_random_samples_coolwarm_20260521_0926/test_sample_0030_coolwarm_21frames.gif`
- `2D_NS_FNO2d_recurrent/visualizations/exponax_t20_random_samples_coolwarm_20260521_0926/test_sample_0032_coolwarm_21frames.gif`
- `2D_NS_FNO2d_recurrent/visualizations/exponax_t20_random_samples_coolwarm_20260521_0926/test_sample_0035_coolwarm_21frames.gif`
- `2D_NS_FNO2d_recurrent/visualizations/exponax_t20_random_samples_coolwarm_20260521_0926/test_sample_0045_coolwarm_21frames.gif`

## Rendering Settings

- CPU-only rendering; CUDA hidden from the plotting process.
- Color map: Matplotlib `coolwarm`.
- Frame count: `21` frames per sample.
- Pixel size: `256x256`.
- GIF frame duration: `180 ms`.
- Color scaling: per-sample symmetric range around zero, using the maximum absolute value across the 21 frames of that sample.

## Inference

The sampled train and test files have the expected `(N, 256, 256, 21)` layout for direct 21-frame heatmap animation. The GIF validation confirms the generated visualizations match the requested frame count and pixel size.

## Remaining Work

Inspect the GIFs visually and decide whether additional samples, a shared global color scale, or labeled frame overlays are needed.
