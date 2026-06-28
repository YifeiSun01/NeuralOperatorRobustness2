# Burgers Jacobian Downsample SVD Probe - 2026-06-07

## Scope

Observed from `nvidia-smi` before the run: the Tesla V100-SXM2-32GB was idle enough for the probe; later `nvidia-smi --query-gpu` reported `320 MiB / 32768 MiB` used, `2%` utilization, and temperature `43C`.

Observed GPU path:

- PyTorch `2.8.0+cu126`, Torch CUDA `12.6`.
- CUDA available on `Tesla V100-SXM2-32GB`, compute capability `[7, 0]`.
- PyTorch CUDA arch list includes `sm_70`.
- JAX `0.10.0`, backend `gpu`, device `cuda:0`.

This probe reuses existing saved full `1024 x 1024` Burgers Jacobian matrices. It does not recompute model or solver Jacobians by autograd.

## Source Matrices

Observed source full-Jacobian SVD NPZ files:

- `forensics/burgers_p2q2_checkpoint_series_jacobian_svd_20260601/sample_000/solver/solver_index0_jacobian_svd.npz`
- `forensics/burgers_p2q2_checkpoint_series_jacobian_svd_20260601/sample_000/baseline/baseline_index0_jacobian_svd.npz`
- `forensics/burgers_p2q2_checkpoint_series_jacobian_svd_20260601/sample_000/baseline_error/baseline_error_index0_jacobian_svd.npz`
- `forensics/burgers_p2q2_checkpoint_series_jacobian_svd_20260601/sample_004/solver/solver_index4_jacobian_svd.npz`
- `forensics/burgers_p2q2_checkpoint_series_jacobian_svd_20260601/sample_004/baseline/baseline_index4_jacobian_svd.npz`
- `forensics/burgers_p2q2_checkpoint_series_jacobian_svd_20260601/sample_004/baseline_error/baseline_error_index4_jacobian_svd.npz`
- `forensics/burgers_p2q2_checkpoint_series_jacobian_svd_20260601/sample_006/solver/solver_index6_jacobian_svd.npz`
- `forensics/burgers_p2q2_checkpoint_series_jacobian_svd_20260601/sample_006/baseline/baseline_index6_jacobian_svd.npz`
- `forensics/burgers_p2q2_checkpoint_series_jacobian_svd_20260601/sample_006/baseline_error/baseline_error_index6_jacobian_svd.npz`

Observed each NPZ includes `jacobian`, `singular_values`, `left_singular_vectors`, and `right_singular_vectors`; checked sample matrices had `jacobian` shape `(1024, 1024)`.

## Method

The script `tools/compare_burgers_jacobian_downsample_svd.py` was added for this probe.

Two coarse proxies were compared:

- `stride`: literal submatrix `J[::d, ::d]`, where `d=2` for 512 and `d=4` for 256.
- `block_projection`: orthogonal projection onto block-constant input/output subspaces, computed as `P^T J P`; this is the cleaner low-frequency proxy.

Metrics recorded:

- `coarse_to_full_sigma1_ratio`: top singular value of the coarse proxy divided by the full `1024` top singular value.
- rank-1 absolute cosine between projected full singular vectors and coarse singular vectors.
- top-10 right/left subspace mean principal cosine.

Primary outputs:

- `forensics/burgers_jacobian_downsample_svd_probe_20260607/downsample_jacobian_svd_comparison.csv`
- `forensics/burgers_jacobian_downsample_svd_probe_20260607/gpu_preflight.json`
- `forensics/burgers_jacobian_downsample_svd_probe_20260607/summary.json`
- `forensics/burgers_jacobian_downsample_svd_probe_20260607/README.md`

Observed run summary: `36` result rows, max CUDA allocated `19.04 MB`, max CUDA reserved `26.0 MB`.

## Results

Grouped over `3` samples for each matrix kind:

| matrix | coarse | proxy | sigma1 ratio mean | sigma1 range | rank1 right mean | rank1 left mean | top10 right subspace mean | top10 left subspace mean |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| solver | 512 | stride | `0.5000` | `[0.5000, 0.5000]` | `1.0000` | `1.0000` | `1.0000` | `1.0000` |
| solver | 512 | block_projection | `0.9931` | `[0.9884, 0.9973]` | `1.0000` | `1.0000` | `1.0000` | `1.0000` |
| solver | 256 | stride | `0.2540` | `[0.2483, 0.2638]` | `1.0000` | `1.0000` | `1.0000` | `1.0000` |
| solver | 256 | block_projection | `0.9616` | `[0.9294, 0.9866]` | `1.0000` | `1.0000` | `0.9999` | `0.9999` |
| baseline model | 512 | stride | `0.5001` | `[0.5001, 0.5001]` | `0.9999` | `1.0000` | `0.9998` | `0.9999` |
| baseline model | 512 | block_projection | `0.9947` | `[0.9910, 0.9982]` | `1.0000` | `1.0000` | `1.0000` | `1.0000` |
| baseline model | 256 | stride | `0.2489` | `[0.2460, 0.2504]` | `0.9995` | `0.9998` | `0.9981` | `0.9991` |
| baseline model | 256 | block_projection | `0.9755` | `[0.9605, 0.9912]` | `1.0000` | `1.0000` | `1.0000` | `1.0000` |
| baseline error | 512 | stride | `0.5013` | `[0.5009, 0.5021]` | `0.9993` | `0.9998` | `0.9884` | `0.9913` |
| baseline error | 512 | block_projection | `0.9737` | `[0.9496, 0.9917]` | `0.9990` | `0.9993` | `1.0000` | `1.0000` |
| baseline error | 256 | stride | `0.2660` | `[0.2302, 0.3154]` | `0.9404` | `0.9583` | `0.9198` | `0.9406` |
| baseline error | 256 | block_projection | `0.8908` | `[0.7758, 0.9597]` | `0.9843` | `0.9887` | `0.9989` | `0.9991` |

Overall across all 9 matrices:

| coarse | proxy | sigma1 mean | sigma1 min-max | min rank1 right cosine | min rank1 left cosine | min top10 right subspace cosine | min top10 left subspace cosine |
|---:|---|---:|---:|---:|---:|---:|---:|
| 512 | stride | `0.5005` | `[0.5000, 0.5021]` | `0.9991` | `0.9995` | `0.9812` | `0.9839` |
| 512 | block_projection | `0.9872` | `[0.9496, 0.9982]` | `0.9971` | `0.9979` | `0.9999` | `0.9999` |
| 256 | stride | `0.2563` | `[0.2302, 0.3154]` | `0.8285` | `0.8757` | `0.8767` | `0.9010` |
| 256 | block_projection | `0.9426` | `[0.7758, 0.9912]` | `0.9533` | `0.9661` | `0.9971` | `0.9976` |

## Interpretation

Observed: for these three Burgers samples and the solver/baseline/baseline-error matrices, the dominant singular vector and top-10 singular subspaces are mostly low-frequency/coarse enough that 512 and 256 preserve the directions very well when using `block_projection`.

Observed: literal stride submatrices preserve direction surprisingly well for solver/model matrices, but their top singular values are not comparable to the full matrix. The stride `512` sigma1 is about half of the full sigma1, and stride `256` is roughly one quarter. This is an artifact of taking a submatrix, not evidence that the full operator is four times smaller.

Observed: the hardest case is the baseline model-minus-solver error Jacobian at 256. Even there, `block_projection` preserved top-10 subspaces very well, but top singular value ratios ranged down to `0.7758`, and rank-1 right cosine minimum was `0.9533`.

Inference: for this saved Burgers Jacobian set, a `512` block-projected Jacobian is an excellent proxy for the leading 1024 singular directions and top singular value. A `256` block-projected Jacobian is still a useful quick proxy for leading directions, but it can understate the error-Jacobian spectral norm by about `10%` to `22%` in this probe.

Inference: if the goal is singular-vector/subspace diagnosis, `256` is probably acceptable for a quick screen on similar Burgers matrices. If the goal is exact spectral-norm magnitude, prefer `512` or calibrate the 256 estimate against a small number of full 1024 runs.

Inference: do not use plain `J[::d, ::d]` singular values as a quantitative estimate of the full `1024` spectral norm. Use block projection or another L2-consistent projection.

## Raw Per-Row Key Metrics

Observed raw key metrics from `forensics/burgers_jacobian_downsample_svd_probe_20260607/downsample_jacobian_svd_comparison.csv`:

| sample | matrix | coarse | method | sigma ratio | rank1 R cos | rank1 L cos | top10 R subspace | top10 L subspace |
|---|---|---:|---|---:|---:|---:|---:|---:|
| sample_000 | solver | 512 | stride | `0.5000` | `1.0000` | `1.0000` | `1.0000` | `1.0000` |
| sample_000 | solver | 512 | block_projection | `0.9884` | `1.0000` | `1.0000` | `1.0000` | `1.0000` |
| sample_000 | solver | 256 | stride | `0.2638` | `1.0000` | `1.0000` | `1.0000` | `1.0000` |
| sample_000 | solver | 256 | block_projection | `0.9294` | `1.0000` | `1.0000` | `1.0000` | `1.0000` |
| sample_000 | baseline | 512 | stride | `0.5001` | `0.9999` | `1.0000` | `0.9998` | `0.9999` |
| sample_000 | baseline | 512 | block_projection | `0.9910` | `1.0000` | `1.0000` | `1.0000` | `1.0000` |
| sample_000 | baseline | 256 | stride | `0.2460` | `0.9995` | `1.0000` | `0.9982` | `0.9988` |
| sample_000 | baseline | 256 | block_projection | `0.9605` | `1.0000` | `1.0000` | `1.0000` | `1.0000` |
| sample_000 | baseline_error | 512 | stride | `0.5021` | `0.9994` | `1.0000` | `0.9930` | `0.9957` |
| sample_000 | baseline_error | 512 | block_projection | `0.9496` | `1.0000` | `1.0000` | `1.0000` | `1.0000` |
| sample_000 | baseline_error | 256 | stride | `0.3154` | `0.9974` | `0.9998` | `0.9615` | `0.9805` |
| sample_000 | baseline_error | 256 | block_projection | `0.7758` | `0.9996` | `1.0000` | `0.9996` | `0.9999` |
| sample_004 | solver | 512 | stride | `0.5000` | `1.0000` | `1.0000` | `1.0000` | `1.0000` |
| sample_004 | solver | 512 | block_projection | `0.9935` | `1.0000` | `1.0000` | `1.0000` | `1.0000` |
| sample_004 | solver | 256 | stride | `0.2483` | `1.0000` | `1.0000` | `1.0000` | `1.0000` |
| sample_004 | solver | 256 | block_projection | `0.9687` | `1.0000` | `1.0000` | `0.9998` | `0.9998` |
| sample_004 | baseline | 512 | stride | `0.5001` | `1.0000` | `1.0000` | `0.9998` | `0.9999` |
| sample_004 | baseline | 512 | block_projection | `0.9948` | `1.0000` | `1.0000` | `1.0000` | `1.0000` |
| sample_004 | baseline | 256 | stride | `0.2504` | `0.9998` | `0.9999` | `0.9981` | `0.9993` |
| sample_004 | baseline | 256 | block_projection | `0.9747` | `1.0000` | `1.0000` | `1.0000` | `1.0000` |
| sample_004 | baseline_error | 512 | stride | `0.5010` | `0.9991` | `0.9995` | `0.9911` | `0.9942` |
| sample_004 | baseline_error | 512 | block_projection | `0.9798` | `0.9971` | `0.9979` | `1.0000` | `1.0000` |
| sample_004 | baseline_error | 256 | stride | `0.2302` | `0.8285` | `0.8757` | `0.8767` | `0.9010` |
| sample_004 | baseline_error | 256 | block_projection | `0.9369` | `0.9533` | `0.9661` | `0.9999` | `0.9999` |
| sample_006 | solver | 512 | stride | `0.5000` | `1.0000` | `1.0000` | `1.0000` | `1.0000` |
| sample_006 | solver | 512 | block_projection | `0.9973` | `1.0000` | `1.0000` | `1.0000` | `1.0000` |
| sample_006 | solver | 256 | stride | `0.2500` | `1.0000` | `1.0000` | `1.0000` | `1.0000` |
| sample_006 | solver | 256 | block_projection | `0.9866` | `1.0000` | `1.0000` | `1.0000` | `1.0000` |
| sample_006 | baseline | 512 | stride | `0.5001` | `0.9999` | `1.0000` | `0.9998` | `0.9999` |
| sample_006 | baseline | 512 | block_projection | `0.9982` | `1.0000` | `1.0000` | `1.0000` | `1.0000` |
| sample_006 | baseline | 256 | stride | `0.2503` | `0.9991` | `0.9996` | `0.9980` | `0.9991` |
| sample_006 | baseline | 256 | block_projection | `0.9912` | `1.0000` | `1.0000` | `1.0000` | `1.0000` |
| sample_006 | baseline_error | 512 | stride | `0.5009` | `0.9995` | `0.9999` | `0.9812` | `0.9839` |
| sample_006 | baseline_error | 512 | block_projection | `0.9917` | `1.0000` | `1.0000` | `0.9999` | `0.9999` |
| sample_006 | baseline_error | 256 | stride | `0.2524` | `0.9954` | `0.9994` | `0.9211` | `0.9403` |
| sample_006 | baseline_error | 256 | block_projection | `0.9597` | `1.0000` | `1.0000` | `0.9971` | `0.9976` |

## Metric Definitions And Scaled Stride Check

Definitions used in the tables:

- `sigma ratio` means the coarse top singular value divided by the full `1024 x 1024` top singular value, i.e. \(\sigma_1(J_{\mathrm{coarse}})/\sigma_1(J_{1024})\).
- `sigma mean` is the average of that ratio over the rows in the group.
- `sigma min` and `sigma max` are the smallest and largest ratio in the group.
- `rank1 R cos` is the absolute cosine similarity between the projected full right singular vector for \(\sigma_1\) and the coarse right singular vector for \(\sigma_1\).
- `rank1 L cos` is the same comparison for the left singular vector.
- `top10 R subspace` is the mean principal cosine between the projected full top-10 right singular subspace and the coarse top-10 right singular subspace.
- `top10 L subspace` is the same top-10 subspace comparison for the left singular subspace.

Observed scaled-stride check: for `stride`, multiplying `sigma ratio` by the downsample factor (`2` for 512, `4` for 256) removes the expected sampling-density scaling. This correction changes singular-value magnitude only; it does not fix singular-vector/subspace aliasing or lost within-block information.

| coarse | method | n | raw sigma mean | scaled sigma mean | scaled sigma min-max | min rank1 R | min rank1 L | min top10 R | min top10 L |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | block_projection | 9 | `0.9872` | `0.9872` | `[0.9496,0.9982]` | `0.9971` | `0.9979` | `0.9999` | `0.9999` |
| 512 | stride | 9 | `0.5005` | `1.0009` | `[1.0000,1.0041]` | `0.9991` | `0.9995` | `0.9812` | `0.9839` |
| 256 | block_projection | 9 | `0.9426` | `0.9426` | `[0.7758,0.9912]` | `0.9533` | `0.9661` | `0.9971` | `0.9976` |
| 256 | stride | 9 | `0.2563` | `1.0253` | `[0.9207,1.2617]` | `0.8285` | `0.8757` | `0.8767` | `0.9010` |

Observed matrix-kind breakdown after scaling stride by the downsample factor:

| matrix | coarse | method | raw sigma mean | scaled sigma mean | scaled min-max | min rank1 R | min rank1 L | min top10 R | min top10 L |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| solver | 512 | block_projection | `0.9931` | `0.9931` | `[0.9884,0.9973]` | `1.0000` | `1.0000` | `1.0000` | `1.0000` |
| solver | 512 | stride | `0.5000` | `1.0000` | `[1.0000,1.0001]` | `1.0000` | `1.0000` | `1.0000` | `1.0000` |
| solver | 256 | block_projection | `0.9616` | `0.9616` | `[0.9294,0.9866]` | `1.0000` | `1.0000` | `0.9998` | `0.9998` |
| solver | 256 | stride | `0.2540` | `1.0162` | `[0.9931,1.0552]` | `1.0000` | `1.0000` | `1.0000` | `1.0000` |
| baseline | 512 | block_projection | `0.9947` | `0.9947` | `[0.9910,0.9982]` | `1.0000` | `1.0000` | `1.0000` | `1.0000` |
| baseline | 512 | stride | `0.5001` | `1.0002` | `[1.0001,1.0002]` | `0.9999` | `1.0000` | `0.9998` | `0.9999` |
| baseline | 256 | block_projection | `0.9755` | `0.9755` | `[0.9605,0.9912]` | `1.0000` | `1.0000` | `1.0000` | `1.0000` |
| baseline | 256 | stride | `0.2489` | `0.9956` | `[0.9839,1.0018]` | `0.9991` | `0.9996` | `0.9980` | `0.9988` |
| baseline_error | 512 | block_projection | `0.9737` | `0.9737` | `[0.9496,0.9917]` | `0.9971` | `0.9979` | `0.9999` | `0.9999` |
| baseline_error | 512 | stride | `0.5013` | `1.0026` | `[1.0017,1.0041]` | `0.9991` | `0.9995` | `0.9812` | `0.9839` |
| baseline_error | 256 | block_projection | `0.8908` | `0.8908` | `[0.7758,0.9597]` | `0.9533` | `0.9661` | `0.9971` | `0.9976` |
| baseline_error | 256 | stride | `0.2660` | `1.0640` | `[0.9207,1.2617]` | `0.8285` | `0.8757` | `0.8767` | `0.9010` |

Inference: after multiplying stride by the expected factor, stride is competitive for the smooth solver and baseline model singular value magnitude. It remains worse for baseline-error directions, especially at 256, because rescaling fixes sampling density but cannot recover the within-block coupling and aliasing lost by point sampling.

## Correction On 256 Error-Jacobian Proxy Reliability

Observed clarification after re-reading the `baseline_error` 256 results: for **sigma1 magnitude alone**, scaled stride can be closer than block projection in this small 3-sample probe. The observed `baseline_error` 256 block-projection sigma ratio mean was `0.8908`, i.e. about `10.9%` low, while 256 stride after multiplying by the downsample factor had scaled sigma mean `1.0640`, i.e. about `6.4%` high. So the earlier shorthand “block projection is more reliable” is too broad if it is read as “always more accurate for sigma1 magnitude.”

Observed at the same time: the direction/subspace metrics still favor block projection. For `baseline_error` at 256, block projection had minimum rank1 right/left cosines `0.9533/0.9661` and minimum top10 right/left subspace cosines `0.9971/0.9976`; scaled stride had weaker minimum rank1 right/left cosines `0.8285/0.8757` and minimum top10 right/left subspace cosines `0.8767/0.9010`.

Corrected inference: if the only target is the leading singular value magnitude on this specific smooth-ish baseline-error set, scaled stride is competitive and can be closer in mean. If the target is singular-vector orientation, top-k subspace comparison, or an `L2`-consistent coarse operator, block projection is the safer proxy.
