# Top Singular Vector Shape Comparison

Date: 2026-05-15

## Purpose

This note records the direct visual comparison requested after the FNO-vs-solver
and DeepONet-vs-solver local-Jacobian experiments.

For each sampled initial condition, it plots the top 4 **right singular
vectors** of:

```text
FNO nu=0.001
solver nu=0.001
DeepONet nu=0.01
solver nu=0.01
```

Right singular vectors are the input perturbation directions.  These are the
vectors relevant to the observed low-frequency vs high-frequency adversarial
perturbation pattern.

## Inputs

```text
sample indices: 0, 7, 40, 47, 115
FNO SVD root: forensics/local_jacobian_frequency_20260514/01_explicit_jacobian_multi_index/
FNO-solver raw recompute root: forensics/fno_solver_jacobian_similarity_20260514_raw_recomputed/
DeepONet-solver root: forensics/deeponet_solver_jacobian_similarity_20260515/
output root: forensics/top_singular_vector_comparison_20260515/
plotting script: tools/plot_top_singular_vector_comparison.py
```

Display convention:

- singular-vector signs are arbitrary, so signs are aligned for plotting;
- line shapes are scaled by max absolute value, so shape is readable;
- each subplot label reports the actual singular value `sigma`, `hi128`, and
  zero crossings.

## Generated Plots

For each index:

```text
forensics/top_singular_vector_comparison_20260515/index_*/
```

contains:

- `*_top4_right_singular_vectors_lines.png`: 4x4 grid of vector line plots;
- `*_top4_right_singular_vectors_fft.png`: Fourier energy spectrum of each vector;
- `*_top4_right_singular_vectors_overlay.png`: overlay plot for direct shape comparison.

Example, sample 0:

```text
forensics/top_singular_vector_comparison_20260515/index_000/index_000_top4_right_singular_vectors_lines.png
forensics/top_singular_vector_comparison_20260515/index_000/index_000_top4_right_singular_vectors_fft.png
forensics/top_singular_vector_comparison_20260515/index_000/index_000_top4_right_singular_vectors_overlay.png
```

## Key Observations

The visual comparison matches the numeric SVD/frequency summaries:

- FNO `nu=0.001` top right singular vectors are smooth/low-frequency.  Across
  the top-4 vectors and five samples, their `hi128` values are tiny, generally
  around `1e-10` to `1e-5`.
- solver `nu=0.001` top right singular vectors are also low-frequency/smooth.
- DeepONet `nu=0.01` top right singular vectors are visibly high-frequency and
  jagged.  Top-1 `hi128` is about `0.75--0.77`, with about `499--512` zero
  crossings across the five samples.
- solver `nu=0.01` top right singular vectors remain low-frequency/smooth.

So the plot-level conclusion is:

```text
FNO dominant directions look qualitatively like solver dominant directions.
DeepONet dominant directions do not; they are high-frequency directions that
look like the DeepONet-vs-solver error directions.
```

This is exactly the pattern the user wanted to inspect: the singular values can
be read from each subplot, while the vector shapes show whether the local
sensitive modes have the same structure.
