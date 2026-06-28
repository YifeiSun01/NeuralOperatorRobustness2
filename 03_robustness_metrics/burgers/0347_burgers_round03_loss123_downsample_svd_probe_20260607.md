# Burgers Round03 Loss1/Loss2/Loss3 Downsample Jacobian SVD Probe - 2026-06-07

## Scope

Observed evidence: this probe reuses saved full `1024 x 1024` Burgers Jacobian matrices from the round03 final-extension Jacobian/SVD directory. It does not recompute dense Jacobians with autograd.

Observed source directory:

- `forensics/burgers_loss3_selective_round03_loss123_final_extension_jacobian_svd_rep20_top100_20260606`

Observed model labels in that source directory:

- `baseline`
- `loss1_epoch5000`
- `loss2_epoch2000`
- `loss3_epoch1500`

Observed limitation: a search for local `*8000*jacobian_svd*.npz` under `forensics/` returned no matching loss1 epoch8000 full-Jacobian NPZ, so this probe uses the saved round03 `loss1_epoch5000` Jacobians rather than claiming epoch8000 Jacobian evidence.

GPU path recorded by the run:

- PyTorch `2.8.0+cu126`, Torch CUDA `12.6`.
- CUDA available on `Tesla V100-SXM2-32GB`, compute capability `[7, 0]`.
- PyTorch CUDA arch list includes `sm_70`: `True`.
- `nvidia-smi` immediately before the run showed Tesla V100-SXM2-32GB at `0 MiB / 32768 MiB` used.

Primary outputs:

- CSV: `forensics/burgers_round03_loss123_downsample_svd_probe_20260607/downsample_jacobian_svd_comparison.csv`
- Summary: `forensics/burgers_round03_loss123_downsample_svd_probe_20260607/summary.json`
- GPU preflight: `forensics/burgers_round03_loss123_downsample_svd_probe_20260607/gpu_preflight.json`
- Row count: `720`
- Max CUDA allocated/reserved: `19.04 MB / 26.00 MB`

## Method

Compared two coarse proxies for each saved full Jacobian:

- `stride`: literal submatrix `J[::d, ::d]`, with `d=2` for 512 and `d=4` for 256.
- `block_projection`: orthogonal projection onto block-constant input/output subspaces, `P^T J P`.

Reported metrics:

- `raw sigma`: top singular value of the coarse proxy divided by the full 1024 top singular value.
- `scaled sigma`: same as raw sigma for block projection; for stride, raw sigma multiplied by the downsample factor so 512 stride is multiplied by 2 and 256 stride is multiplied by 4.
- `rank1 R/L`: absolute cosine similarity for the top right/left singular vector after projecting the full vector to the coarse space.
- `top10 R/L`: mean principal cosine between projected full top-10 right/left singular subspace and coarse top-10 right/left subspace.

## Block Projection Summary

Observed from the CSV: block projection is the cleanest proxy. The table reports `sigma mean`, `sigma range`, and the weakest top-10 right/left subspace cosine over all 20 samples.

| model | kind | 512 sigma mean | 512 sigma range | 512 min top10 R/L | 256 sigma mean | 256 sigma range | 256 min top10 R/L |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline | model | `0.9952` | `[0.9891, 0.9975]` | `0.9998/0.9998` | `0.9781` | `[0.9706, 0.9878]` | `0.9974/0.9975` |
| loss1_epoch5000 | model | `0.9939` | `[0.9896, 0.9982]` | `1.0000/1.0000` | `0.9722` | `[0.9500, 0.9918]` | `0.9991/0.9995` |
| loss2_epoch2000 | model | `0.9937` | `[0.9885, 0.9973]` | `0.9999/1.0000` | `0.9699` | `[0.9355, 0.9865]` | `0.9982/0.9990` |
| loss3_epoch1500 | model | `0.9943` | `[0.9909, 0.9974]` | `0.9999/1.0000` | `0.9738` | `[0.9644, 0.9872]` | `0.9991/0.9991` |
| baseline | error | `0.9852` | `[0.9550, 0.9976]` | `0.9987/0.9989` | `0.9361` | `[0.8342, 0.9881]` | `0.9193/0.9213` |
| loss1_epoch5000 | error | `0.9848` | `[0.9287, 0.9960]` | `0.9867/0.9878` | `0.9493` | `[0.8548, 0.9799]` | `0.9186/0.9194` |
| loss2_epoch2000 | error | `0.9844` | `[0.9536, 0.9958]` | `0.9653/0.9669` | `0.9366` | `[0.7601, 0.9792]` | `0.9487/0.9575` |
| loss3_epoch1500 | error | `0.9777` | `[0.9541, 0.9936]` | `0.9999/0.9999` | `0.8959` | `[0.7225, 0.9691]` | `0.9991/0.9992` |

## Scaled Stride Summary

Observed from the CSV: scaling stride fixes the expected sampling-density factor in the singular value, but it does not fix lost within-block information or aliasing. This is where loss1/loss2 error Jacobians degrade most clearly.

| model | kind | 512 scaled sigma mean/range | 512 min top10 R/L | 256 scaled sigma mean/range | 256 min rank1 R/L | 256 min top10 R/L |
|---|---|---:|---:|---:|---:|---:|
| baseline | model | `1.0002` / `[1.0000, 1.0006]` | `0.9117/0.9119` | `0.9992` / `[0.9619, 1.0031]` | `0.9866/0.9866` | `0.9038/0.9044` |
| loss1_epoch5000 | model | `1.0005` / `[0.9986, 1.0033]` | `0.9957/0.9958` | `1.0001` / `[0.9828, 1.0229]` | `0.9937/0.9937` | `0.9281/0.9302` |
| loss2_epoch2000 | model | `1.0004` / `[0.9997, 1.0026]` | `0.9960/0.9959` | `1.0040` / `[0.9883, 1.0384]` | `0.9969/0.9969` | `0.9919/0.9918` |
| loss3_epoch1500 | model | `1.0000` / `[0.9997, 1.0003]` | `0.9999/0.9999` | `0.9982` / `[0.9709, 1.0124]` | `0.9998/0.9999` | `0.9994/0.9994` |
| baseline | error | `1.0012` / `[0.9960, 1.0091]` | `0.9351/0.9383` | `1.0040` / `[0.8915, 1.0684]` | `0.9176/0.9330` | `0.7724/0.8168` |
| loss1_epoch5000 | error | `1.0066` / `[0.9742, 1.0225]` | `0.8870/0.8921` | `1.0309` / `[0.8024, 1.2052]` | `0.8566/0.9157` | `0.7091/0.7297` |
| loss2_epoch2000 | error | `1.0048` / `[0.9910, 1.0195]` | `0.9000/0.9027` | `1.0231` / `[0.6282, 1.1963]` | `0.0354/0.0412` | `0.6796/0.6943` |
| loss3_epoch1500 | error | `0.9996` / `[0.9901, 1.0025]` | `0.9962/0.9964` | `1.0108` / `[0.8318, 1.1695]` | `0.9953/0.9957` | `0.9648/0.9671` |

## Focused Error-Jacobian Findings

Observed: for the `model` Jacobians themselves, adversarial/self-trained loss1/loss2/loss3 models are not worse than baseline under these proxies. The 512 block projection stays around `0.9937` to `0.9943` sigma mean for loss1/loss2/loss3 models, and 256 block projection stays around `0.9699` to `0.9738`.

Observed: for `model - solver` error Jacobians, 512 block projection remains usable across all four models. Error sigma means are `0.9852` baseline, `0.9848` loss1, `0.9844` loss2, and `0.9777` loss3. The weakest 512 block top10 right/left subspace cosines are `0.9987/0.9989` baseline, `0.9867/0.9878` loss1, `0.9653/0.9669` loss2, and `0.9999/0.9999` loss3.

Observed: 256 block projection is still a good subspace screen but can underestimate the error-Jacobian spectral norm. Error sigma means/ranges are baseline `0.9361 [0.8342, 0.9881]`, loss1 `0.9493 [0.8548, 0.9799]`, loss2 `0.9366 [0.7601, 0.9792]`, and loss3 `0.8959 [0.7225, 0.9691]`.

Observed: 256 scaled stride is the most unstable for error Jacobians. Ranges are baseline `[0.8915, 1.0684]`, loss1 `[0.8024, 1.2052]`, loss2 `[0.6282, 1.1963]`, and loss3 `[0.8318, 1.1695]`. Direction/subspace degradation is strongest for loss1/loss2 stride error matrices: minimum top10 right/left cosines are loss1 `0.7091/0.7297`, loss2 `0.6796/0.6943`, baseline `0.7724/0.8168`, and loss3 `0.9648/0.9671`.

Inference: the user suspicion is partly supported, but the effect is not simply “all adversarial models are more high-frequency.” The full model Jacobians remain easy to compress. The fragile object is the error Jacobian. Loss1/loss2 error Jacobians are especially bad for 256 stride directions, while loss3 error is more problematic for 256 block-projection sigma magnitude.

Inference: for future cheap Jacobian/SVD screening on these Burgers runs, `512 block_projection` is still the safest proxy. `256 block_projection` is acceptable for top-10 subspace screening but can understate error spectral norm by roughly 10% to 28% in these 20 samples. `256 stride`, even with scaling, should not be trusted for error-Jacobian conclusions.

## Worst 256 Error Rows

Observed worst rows by 256 error-Jacobian top10 right-subspace cosine:

| sample | matrix | method | raw sigma | scaled sigma | rank1 R/L | top10 R/L |
|---|---|---|---:|---:|---:|---:|
| sample_011 | loss2_epoch2000_error | stride | `0.2465` | `0.9860` | `0.9999/0.9999` | `0.6796/0.6943` |
| sample_000 | loss1_epoch5000_error | stride | `0.2661` | `1.0643` | `0.9612/0.9942` | `0.7091/0.7297` |
| sample_006 | loss1_epoch5000_error | stride | `0.2726` | `1.0903` | `0.9544/0.9867` | `0.7426/0.7672` |
| sample_003 | loss1_epoch5000_error | stride | `0.2767` | `1.1069` | `0.9104/0.9475` | `0.7724/0.8236` |
| sample_009 | baseline_error | stride | `0.2642` | `1.0568` | `0.9901/0.9985` | `0.7724/0.8168` |
| sample_002 | loss1_epoch5000_error | stride | `0.2594` | `1.0375` | `0.9612/0.9856` | `0.7746/0.8205` |
| sample_016 | baseline_error | stride | `0.2341` | `0.9363` | `0.9999/1.0000` | `0.8006/0.8199` |
| sample_008 | loss1_epoch5000_error | stride | `0.2729` | `1.0914` | `0.8698/0.9470` | `0.8012/0.8393` |
| sample_016 | loss1_epoch5000_error | stride | `0.2006` | `0.8024` | `0.9926/0.9993` | `0.8156/0.8270` |
| sample_000 | baseline_error | stride | `0.2587` | `1.0350` | `0.9846/0.9976` | `0.8191/0.8505` |
| sample_009 | loss1_epoch5000_error | stride | `0.2824` | `1.1295` | `0.9291/0.9848` | `0.8209/0.8621` |
| sample_001 | loss1_epoch5000_error | stride | `0.2856` | `1.1424` | `0.8566/0.9389` | `0.8285/0.8559` |

Observed worst rows by 256 error-Jacobian absolute scaled-sigma error:

| sample | matrix | method | raw sigma | scaled sigma | rank1 R/L | top10 R/L |
|---|---|---|---:|---:|---:|---:|
| sample_016 | loss2_epoch2000_error | stride | `0.1571` | `0.6282` | `0.9850/0.9954` | `0.8831/0.8955` |
| sample_014 | loss3_epoch1500_error | block_projection | `0.7225` | `0.7225` | `0.9991/1.0000` | `0.9994/0.9997` |
| sample_015 | loss2_epoch2000_error | block_projection | `0.7601` | `0.7601` | `0.9978/0.9996` | `0.9989/0.9992` |
| sample_019 | loss3_epoch1500_error | block_projection | `0.7890` | `0.7890` | `1.0000/1.0000` | `0.9996/0.9998` |
| sample_015 | loss1_epoch5000_error | stride | `0.3013` | `1.2052` | `0.9895/0.9996` | `0.9019/0.9223` |
| sample_016 | loss1_epoch5000_error | stride | `0.2006` | `0.8024` | `0.9926/0.9993` | `0.8156/0.8270` |
| sample_008 | loss2_epoch2000_error | stride | `0.2991` | `1.1963` | `0.6046/0.6780` | `0.8455/0.8711` |
| sample_007 | loss3_epoch1500_error | block_projection | `0.8160` | `0.8160` | `0.9999/1.0000` | `0.9998/1.0000` |
| sample_019 | loss2_epoch2000_error | stride | `0.2041` | `0.8163` | `0.9956/0.9996` | `0.8798/0.8990` |
| sample_002 | loss2_epoch2000_error | stride | `0.2949` | `1.1797` | `0.9476/0.9910` | `0.8979/0.9427` |
| sample_019 | loss1_epoch5000_error | stride | `0.2072` | `0.8289` | `0.9936/0.9995` | `0.8728/0.9043` |
| sample_014 | loss3_epoch1500_error | stride | `0.2924` | `1.1695` | `0.9997/1.0000` | `0.9947/0.9956` |

## Notes

Observed: some rank-1 cosine minima can be misleading when the leading singular values are clustered or when rank ordering swaps inside a near-degenerate top subspace. The top-10 subspace cosine is the more stable diagnostic for that situation.

Inference: if the goal is fast screening before expensive full 1024 Jacobians, use 512 block projection for quantitative claims, and use 256 block projection only as a cheaper early warning signal. For error Jacobians, calibrate 256 against a few full 1024 or 512 block runs before making spectral-norm claims.

## Correction On “Block Projection More Reliable”

Observed clarification: “block projection is more reliable” should not be read as “block projection always gives a closer sigma1 magnitude.” In the expanded round03 20-sample data, 256 scaled stride is actually closer for some error-Jacobian sigma1 means. For `baseline_error`, 256 block projection has sigma mean/range `0.9361 [0.8342, 0.9881]`, while 256 scaled stride has `1.0040 [0.8915, 1.0684]`; for sigma1 magnitude alone, scaled stride is closer here.

Observed counterpoint: scaled stride is weaker for singular directions/subspaces, especially for self-trained error Jacobians. At 256, minimum top10 right/left subspace cosines are baseline stride `0.7724/0.8168`, loss1 stride `0.7091/0.7297`, and loss2 stride `0.6796/0.6943`; the matching block-projection values are baseline `0.9193/0.9213`, loss1 `0.9186/0.9194`, and loss2 `0.9487/0.9575`.

Corrected inference: use scaled stride only as a cheap sigma1-magnitude heuristic, and only after calibration against full or block-projected results. Use block projection when the analysis cares about singular vectors, top-k subspaces, or an `L2`-consistent coarse operator.
