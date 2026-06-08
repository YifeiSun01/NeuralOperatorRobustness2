# Burgers Round03 SVD Vector Frequency And Downsample Proxy Interpretation - 2026-06-08

## Scope

Observed evidence: this analysis inspects saved full-SVD files from `forensics/burgers_loss3_selective_round03_loss123_final_extension_jacobian_svd_rep20_top100_20260606` and joins their singular-vector frequency/projection-residual metrics with the downsample proxy CSV `forensics/burgers_round03_loss123_downsample_svd_probe_20260607/downsample_jacobian_svd_comparison.csv`.

This does not recompute model, solver, or dense Jacobians. It analyzes existing `1024 x 1024` full-Jacobian SVD artifacts.

GPU path recorded by the run:

- `nvidia-smi` before the run: Tesla V100-SXM2-32GB at `0 MiB / 32768 MiB` used.
- PyTorch `2.8.0+cu126`, Torch CUDA `12.6`.
- CUDA device `Tesla V100-SXM2-32GB`, capability `[7, 0]`, arch list includes `sm_70`.
- Max CUDA allocated/reserved during the analysis: `0.12 MB / 2.00 MB`.

Primary outputs:

- `forensics/burgers_round03_loss123_svd_vector_frequency_probe_20260608/singular_vector_frequency_metrics.csv`
- `forensics/burgers_round03_loss123_svd_vector_frequency_probe_20260608/proxy_error_joined_with_frequency.csv`
- `forensics/burgers_round03_loss123_svd_vector_frequency_probe_20260608/summary_by_model_kind.csv`
- `forensics/burgers_round03_loss123_svd_vector_frequency_probe_20260608/frequency_proxy_correlations.csv`
- `forensics/burgers_round03_loss123_svd_vector_frequency_probe_20260608/summary.json`

## Metrics

For a coarse size such as 256, `top1_block_resid_mean` is the average of left/right top singular-vector residual energy outside the block-constant projection subspace:

\[
\frac{1}{2}\left(\|(I-PP^T)u_1\|^2 + \|(I-PP^T)v_1\|^2\right).
\]

This is the most direct diagnostic for whether the top singular vectors live in the coarse block-projection subspace.

`fft_high` is the Fourier power fraction above the coarse Nyquist cutoff. It is a physical high-frequency diagnostic, but it is not identical to projection residual.

## Observed 256 Summary

| model | kind | top1 block residual | top10 block residual | top1 FFT high | top10 FFT high | block sigma abs err mean/max | stride sigma abs err mean/max | block min top10 | stride min top10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| solver | solver | `0.0320` | `0.0247` | `0.0033` | `0.0024` | `0.0325/0.0726` | `0.0158/0.0772` | `0.9991` | `0.9999` |
| baseline | model | `0.0217` | `0.0165` | `0.0006` | `0.0003` | `0.0219/0.0294` | `0.0033/0.0381` | `0.9974` | `0.9038` |
| loss1_epoch5000 | model | `0.0274` | `0.0213` | `0.0011` | `0.0007` | `0.0278/0.0500` | `0.0066/0.0229` | `0.9991` | `0.9281` |
| loss2_epoch2000 | model | `0.0296` | `0.0223` | `0.0012` | `0.0008` | `0.0301/0.0645` | `0.0078/0.0384` | `0.9982` | `0.9918` |
| loss3_epoch1500 | model | `0.0259` | `0.0194` | `0.0007` | `0.0005` | `0.0262/0.0356` | `0.0054/0.0291` | `0.9991` | `0.9994` |
| baseline | error | `0.0620` | `0.0548` | `0.0111` | `0.0085` | `0.0639/0.1658` | `0.0328/0.1085` | `0.9193` | `0.7724` |
| loss1_epoch5000 | error | `0.0496` | `0.0468` | `0.0185` | `0.0172` | `0.0507/0.1452` | `0.0899/0.2052` | `0.9186` | `0.7091` |
| loss2_epoch2000 | error | `0.0622` | `0.0575` | `0.0145` | `0.0130` | `0.0634/0.2399` | `0.0853/0.3718` | `0.9487` | `0.6796` |
| loss3_epoch1500 | error | `0.0968` | `0.0926` | `0.0206` | `0.0188` | `0.1041/0.2775` | `0.0486/0.1695` | `0.9991` | `0.9648` |

## Observed Correlations For 256 Error Rows

Observed from `frequency_proxy_correlations.csv`:

| proxy | frequency/projection metric | target | Pearson correlation |
|---|---|---|---:|
| block_projection | top1 block residual | sigma abs error | `0.9976` |
| block_projection | top10 block residual | sigma abs error | `0.9724` |
| block_projection | top1 FFT high | sigma abs error | `0.4534` |
| block_projection | top10 FFT high | sigma abs error | `0.4550` |
| stride | top1 FFT high | sigma abs error | `0.5908` |
| stride | top10 FFT high | sigma abs error | `0.5988` |

## Interpretation

Observed: solver and model Jacobians have small 256 block-projection residuals. The model rows are around `0.0217` to `0.0296` for top1 residual, and their 256 block-projection sigma errors are correspondingly small, around `2%` to `3%` mean.

Observed: error Jacobians have much larger 256 block-projection residuals. Baseline error is `0.0620`, loss1 error is `0.0496`, loss2 error is `0.0622`, and loss3 error is `0.0968`. Their block sigma absolute error means are correspondingly larger: baseline `0.0639`, loss1 `0.0507`, loss2 `0.0634`, and loss3 `0.1041`.

Observed: the strongest quantitative relation is not generic FFT high-frequency power; it is the exact projection residual outside the 256 block-constant subspace. Across 256 error rows, top1 block residual correlates with block-projection sigma absolute error at `0.9976`. FFT high power has a weaker but still positive correlation around `0.45` for block-projection sigma error and around `0.59` for stride sigma error.

Inference: the 256 proxy becomes inaccurate mainly because the error Jacobian's leading singular vectors are less well represented by the coarse subspace. This is a low-dimensional projection failure: the vector mass outside `range(P)` is no longer negligible.

Inference: this can be described as a high-frequency effect, but the sharper mathematical statement is projection-residual/cancellation effect. The error Jacobian is a difference `J_model - J_solver`; low-frequency components that both model and solver share can cancel, leaving relatively more mid/high-frequency or localized structure. That structure is precisely what 256 block projection and stride sampling are less able to represent.

Inference: adversarial/self-training does not uniformly make every matrix less accurate. The model Jacobians stay easy to compress. The error Jacobians are the fragile objects. Among the saved round03 results, loss3 error has the largest 256 block residual and the largest 256 block sigma underestimation; loss1/loss2 error are especially bad for 256 stride subspace directions and stride sigma outliers.

Inference: the user statement that 256 can be off by roughly `30%` to `40%` is supported for worst cases, especially stride on loss2 error where scaled sigma absolute error reaches `0.3718`, and block projection on loss3/loss2 error where maximum sigma errors reach `0.2775` and `0.2399`. The mean errors are smaller, but worst-case error Jacobian rows are large enough to invalidate naive 256-only spectral-norm conclusions.
## FFT High Frequency Versus Block Residual Correlation

Observed from `singular_vector_frequency_metrics.csv`: higher FFT high-frequency fraction is positively associated with larger block-projection residual, but they are not identical quantities.

At coarse size 256:

| group | vector metric | n | Pearson corr(FFT high, block residual) | Spearman corr | FFT high mean | block residual mean |
|---|---|---:|---:|---:|---:|---:|
| all matrices | top1 | 180 | `0.5553` | `0.8753` | `0.00796` | `0.04524` |
| all matrices | top10 weighted | 180 | `0.6012` | `0.8998` | `0.00690` | `0.03955` |
| model only | top1 | 80 | `0.5977` | `0.8798` | `0.00089` | `0.02616` |
| model only | top10 weighted | 80 | `0.6600` | `0.8642` | `0.00056` | `0.01989` |
| error only | top1 | 80 | `0.4617` | `0.7936` | `0.01618` | `0.06764` |
| error only | top10 weighted | 80 | `0.4997` | `0.7102` | `0.01438` | `0.06294` |

Observed within individual 256 error families:

| error family | vector metric | Pearson corr | Spearman corr | FFT high mean | block residual mean |
|---|---|---:|---:|---:|---:|
| baseline_error | top1 | `0.2906` | `0.5805` | `0.01110` | `0.06200` |
| baseline_error | top10 weighted | `0.3816` | `0.6075` | `0.00847` | `0.05481` |
| loss1_error | top1 | `0.6502` | `0.8707` | `0.01854` | `0.04956` |
| loss1_error | top10 weighted | `0.6928` | `0.8045` | `0.01723` | `0.04685` |
| loss2_error | top1 | `0.5080` | `0.8421` | `0.01445` | `0.06216` |
| loss2_error | top10 weighted | `0.4642` | `0.7218` | `0.01299` | `0.05751` |
| loss3_error | top1 | `0.5490` | `0.6526` | `0.02064` | `0.09684` |
| loss3_error | top10 weighted | `0.6123` | `0.6797` | `0.01882` | `0.09258` |

Inference: yes, the observed higher-frequency content is associated with larger block residual. The rank-order association is strong, especially at 256: matrices/vectors with larger FFT high-frequency fraction usually have larger residual outside the block-constant subspace.

Inference: the relationship is not one-to-one. FFT high-frequency measures Fourier power above a cutoff, while block residual measures energy not representable by piecewise-constant blocks. A vector can have modest FFT high-frequency power but still have large block residual if it has localized features, phase shifts, or strong within-block variation. Conversely, some high-frequency components can average out in ways that do not affect every block residual equally.

Inference: for this experiment, “error is more high frequency” is a valid physical explanation, but the mathematically sharper statement is: error singular vectors have larger energy outside the 256 block-constant subspace. FFT high frequency helps explain why that happens; block residual is the direct quantity that controls block-projection loss.

