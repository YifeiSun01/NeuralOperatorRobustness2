# Burgers Frequency, Projection Residual, And Error-Jacobian Interpretation - 2026-06-08

## Source Evidence

Observed source report:

- `docs/burgers_round03_svd_vector_frequency_proxy_interpretation_20260608.md`

Observed source CSVs:

- `forensics/burgers_round03_loss123_svd_vector_frequency_probe_20260608/singular_vector_frequency_metrics.csv`
- `forensics/burgers_round03_loss123_svd_vector_frequency_probe_20260608/proxy_error_joined_with_frequency.csv`
- `forensics/burgers_round03_loss123_svd_vector_frequency_probe_20260608/summary_by_model_kind.csv`
- `forensics/burgers_round03_loss123_svd_vector_frequency_probe_20260608/frequency_proxy_correlations.csv`

## Definitions

`FFT high` is Fourier power above the coarse Nyquist cutoff. It is a physical high-frequency diagnostic.

`block residual` is the energy of a singular vector outside the block-constant coarse subspace:

\[
\|(I-PP^T)v\|^2.
\]

The top1 block residual used here averages left and right singular-vector residuals:

\[
\frac{1}{2}\left(\|(I-PP^T)u_1\|^2+\|(I-PP^T)v_1\|^2\right).
\]

These two quantities are related but not identical. A vector can have modest FFT high power but still large block residual due to localized features, phase shifts, or strong within-block variation.

## 256 Summary: Model Versus Error

| model | kind | top1 block residual | top10 block residual | top1 FFT high | top10 FFT high | block sigma abs err mean/max | stride sigma abs err mean/max |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline | model | `0.0217` | `0.0165` | `0.0006` | `0.0003` | `0.0219/0.0294` | `0.0033/0.0381` |
| loss1_epoch5000 | model | `0.0274` | `0.0213` | `0.0011` | `0.0007` | `0.0278/0.0500` | `0.0066/0.0229` |
| loss2_epoch2000 | model | `0.0296` | `0.0223` | `0.0012` | `0.0008` | `0.0301/0.0645` | `0.0078/0.0384` |
| loss3_epoch1500 | model | `0.0259` | `0.0194` | `0.0007` | `0.0005` | `0.0262/0.0356` | `0.0054/0.0291` |
| baseline | error | `0.0620` | `0.0548` | `0.0111` | `0.0085` | `0.0639/0.1658` | `0.0328/0.1085` |
| loss1_epoch5000 | error | `0.0496` | `0.0468` | `0.0185` | `0.0172` | `0.0507/0.1452` | `0.0899/0.2052` |
| loss2_epoch2000 | error | `0.0622` | `0.0575` | `0.0145` | `0.0130` | `0.0634/0.2399` | `0.0853/0.3718` |
| loss3_epoch1500 | error | `0.0968` | `0.0926` | `0.0206` | `0.0188` | `0.1041/0.2775` | `0.0486/0.1695` |

Observed: error Jacobians have much larger FFT high fraction and much larger block residual than model Jacobians.

## FFT High Frequency Versus Block Residual Correlation

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

## Relation To Sigma Error

Observed 256 error rows:

| proxy | metric | target | Pearson correlation |
|---|---|---|---:|
| block_projection | top1 block residual | sigma abs error | `0.9976` |
| block_projection | top10 block residual | sigma abs error | `0.9724` |
| block_projection | top1 FFT high | sigma abs error | `0.4534` |
| block_projection | top10 FFT high | sigma abs error | `0.4550` |
| stride | top1 FFT high | sigma abs error | `0.5908` |
| stride | top10 FFT high | sigma abs error | `0.5988` |

Inference: high frequency is genuinely associated with larger block residual, especially in rank order. But block residual is the more direct mathematical predictor of block-projection sigma loss.

## Mechanistic Explanation

The error Jacobian is

\[
J_{\mathrm{err}}=J_{\mathrm{model}}-J_{\mathrm{solver}}.
\]

If model and solver share a smooth low-frequency part, subtracting them cancels that shared structure and leaves residual mismatch:

\[
J_{\mathrm{model}}\approx A+\Delta_m,\qquad J_{\mathrm{solver}}\approx A+\Delta_s,
\]

\[
J_{\mathrm{err}}\approx \Delta_m-\Delta_s.
\]

This residual mismatch is more localized and higher-frequency. That is why `model - solver` error singular vectors have larger FFT high fraction and larger block residual.

Final interpretation: “error is more high frequency” is a valid physical explanation. The sharper numerical statement is: error singular vectors have more mass outside the 256 block-constant subspace, and that projection residual directly explains block-projection sigma underestimation.
