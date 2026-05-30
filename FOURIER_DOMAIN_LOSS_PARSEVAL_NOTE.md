# Fourier-Domain Loss And Parseval Note

This note records the discussion about computing loss in physical space versus Fourier space for neural operator outputs.

## Core Point

For standard L2 / MSE loss, computing the pointwise loss in physical space is equivalent to computing the energy of the Fourier coefficients, up to a normalization constant.

This is the Parseval / Plancherel theorem.

If

```math
e(x) = u_{pred}(x) - u_{true}(x),
```

then

```math
\|e\|_2^2
\quad \leftrightarrow \quad
C \|\hat e\|_2^2,
```

where `hat e` is the Fourier transform of `e`, and `C` depends on the FFT normalization convention.

Therefore, for ordinary MSE/L2 training loss,

```python
loss = mean((u_pred - u_true) ** 2)
```

already measures the same total error energy as a Fourier-domain squared loss.

## Practical Conclusion

If the goal is only to minimize ordinary MSE or L2 error, there is usually no need to transform the prediction and target into Fourier space before computing the loss.

Physical-space MSE is simpler, cheaper, and mathematically equivalent to total Fourier energy error up to normalization.

## PyTorch FFT Normalization

PyTorch's default FFT normalization is not orthonormal:

```python
torch.fft.fftn(x, norm=None)
```

With this default, the frequency-domain energy differs from physical-space energy by a grid-size factor.

If using

```python
torch.fft.fftn(x, norm="ortho")
```

then Parseval equivalence is more direct:

```math
\sum_x |e(x)|^2
=
\sum_k |\hat e(k)|^2.
```

The exact scaling also depends on whether one uses `fft`, `rfft`, one-sided spectra, and how symmetric conjugate modes are counted.

## When Fourier-Domain Loss Is Useful

Fourier-domain loss becomes meaningful when it is not just ordinary unweighted L2.

Useful cases include:

1. **Frequency-weighted loss**

   Penalize high-frequency errors more strongly:

   ```math
   \sum_k w(k)|\hat u_{pred}(k)-\hat u_{true}(k)|^2.
   ```

2. **Band-specific diagnostics**

   Measure low-, mid-, and high-frequency errors separately.

3. **Spectral-bias analysis**

   Check whether FNO predicts low frequencies accurately but loses high-frequency structures.

4. **Turbulence / sharp-interface problems**

   High-wavenumber content may matter physically, so ordinary MSE can under-emphasize fine structures.

5. **Custom Sobolev-type losses**

   For example,

   ```math
   \sum_k (1 + |k|^2)^s |\hat e(k)|^2,
   ```

   which corresponds to an `H^s` Sobolev norm.

## When Fourier-Domain Loss Is Not Equivalent

The equivalence only holds for unweighted L2/MSE-type losses.

It does not generally hold for:

- L1 loss;
- Huber loss;
- relative L2 computed sample-wise with nontrivial denominators;
- SSIM / perceptual losses;
- thresholded or clipped losses;
- frequency-weighted losses;
- losses computed only on selected Fourier modes.

## Recommendation For The Current Project

For the current Burgers, Darcy, and Navier-Stokes FNO evaluation:

- Use physical-space pointwise MSE/RMSE/relative L2 as the primary loss.
- Use Fourier-domain quantities only for diagnostics, spectral analysis, or intentionally frequency-weighted objectives.
- Do not transform to Fourier space merely to reproduce ordinary L2 loss; it adds complexity without changing the objective.

Short version:

> Physical-space L2 and Fourier-space total energy are the same information under Parseval. Fourier-domain loss is only necessary when we want frequency-specific or frequency-weighted behavior.

