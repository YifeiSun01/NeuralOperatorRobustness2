# Differentiable Loss Experiment Plan

This document records the experimental plan for comparing differentiable loss
functions between 1D signals and 2D tensors/images/fields. The goal is to find
losses that measure real pattern, structure, and texture differences while being
robust to limited transformations such as translation, rotation, scaling, local
stretching, local compression, and small diffeomorphism-like deformation.

The main requirement is not full invariance. The main requirement is finite,
controlled transformation robustness. All selected losses must remain compatible
with PyTorch autograd so that gradients can backpropagate to model outputs,
initial states, perturbations, or model parameters.

## 1. Core Goal

Naive pointwise losses such as MSE or L1 can become very large when two signals
or images contain the same pattern but are slightly shifted, rotated, stretched,
or locally warped. This experiment tests alternatives that are less sensitive to
small allowed transformations but still assign high loss to genuinely different
patterns or textures.

The desired behavior is:

- Same sample with a small allowed transformation: low loss.
- Different sample or genuinely different texture/pattern: high loss.
- Explicit transformation methods must not over-warp negative pairs into low
  loss.
- Loss must be differentiable and usable in a PyTorch training loop.
- Implementation should prefer existing packages over custom algorithms.

Conceptually, many losses follow one of these forms:

```text
L(x, y) = D(T_theta(x), y) + lambda * R(theta)
```

or, for local deformation:

```text
phi(p) = p + u(p)
L(x, y) = D(x o phi, y)
        + lambda_mag * ||u||^2
        + lambda_smooth * ||grad u||^2
```

Here:

- `D` is a differentiable content/structure/texture loss.
- `T_theta` or `phi` is a bounded transform.
- `R(theta)`, `||u||^2`, and `||grad u||^2` prevent transformation abuse.

## 2. Transformation Types

The experiment separates two transformation types.

### 2.1 Value / Intensity Transformations

The spatial or temporal position does not change. Only the value changes.

Examples:

- 2D: brightness shift, contrast scaling, gamma transform, global intensity
  scaling, mild noise.
- 1D: amplitude gain, amplitude bias, normalization change, mild noise.

This corresponds to:

```text
x(p) -> a * x(p) + b
```

### 2.2 Spatial / Geometric Transformations

The value is mostly preserved, but its position changes.

Examples:

- 2D: translation, rotation, scaling, shear, mirror, elastic deformation,
  local diffeomorphism-like warp.
- 1D: time shift, time scaling, local time stretch, local time compression,
  phase shift, local time warp.

This corresponds to:

```text
x(p) -> x(phi(p))
```

The transformation budget must be limited. If arbitrary transformations are
allowed, unrelated samples may be warped into each other and the loss becomes
meaningless.

## 3. Packages

Recommended packages:

| Package | Purpose |
|---|---|
| `piq` | DISTS and other PyTorch image quality metrics |
| `pytorch-msssim` | Fast differentiable SSIM / MS-SSIM |
| `kymatio` | 1D/2D scattering transforms |
| `tslearn` | Soft-DTW loss for time series |
| `geomloss` | Sinkhorn / optimal transport losses with PyTorch autograd |
| `kornia` | Differentiable geometry transforms and warps for PyTorch |
| `torch` | `grid_sample`, `affine_grid`, differentiable tensor operations |

Suggested install command:

```bash
pip install piq pytorch-msssim kymatio tslearn geomloss kornia
```

## 4. Differentiability Requirements

The losses should support:

```python
loss = loss_fn(x, y)
loss.backward()
```

Avoid inside differentiable loss paths:

- hard `argmax`
- hard `argmin`
- `round`
- `detach`
- NumPy conversion
- hard nearest-neighbor indexing for transform selection
- non-differentiable registration solvers

Prefer:

- `torch.nn.functional.grid_sample`
- `torch.nn.functional.affine_grid`
- `torch.logsumexp` for soft-min
- `torch.tanh` to bound transform parameters
- differentiable bilinear sampling
- explicit transform magnitude penalties
- smoothness penalties for deformation fields

## 5. 2D Candidate Losses

Baseline MSE/L1 is not included in the five main methods. It can be recorded as
a sanity check, but the five experimental candidates are:

1. DISTS
2. MS-SSIM
3. Scattering2D + L2
4. bounded affine alignment + DISTS
5. bounded local deformation + DISTS

These methods were selected because they represent five different method
families.

### 5.1 DISTS

Package:

- `piq`

Representative family:

- Deep feature structure + texture similarity.

What it handles:

- Texture and structure similarity.
- Mild misalignment.
- Cases where pointwise loss is too strict.

Why select it:

- DISTS explicitly targets structure and texture similarity.
- It is more aligned with scientific images, PDE fields, texture fields, and
  general 2D tensors than ImageNet-biased perceptual losses in many cases.
- It is available in a PyTorch package.

Similar alternatives:

- LPIPS
- VGG perceptual loss
- Gram/style loss
- A-DISTS
- DeepSSIM

Why not select alternatives first:

- LPIPS and VGG perceptual losses are often more natural-image/ImageNet biased.
- Gram/style losses can lose spatial information.
- DISTS is a direct structure + texture metric and is easy to call through
  `piq`.

Expected input:

```text
x, y: [B, C, H, W]
range: usually [0, 1]
```

Record:

- `loss_P1_photometric`
- `loss_P2_global_geometry`
- `loss_P3_local_deformation`
- `loss_N1_negative`
- gaps
- runtime
- memory
- gradient norm

### 5.2 MS-SSIM

Package:

- `pytorch-msssim` or `piq`

Representative family:

- Local structural statistics.

What it handles:

- Position mostly fixed.
- Brightness, contrast, local structure, blur, and mild noise changes.

Why select it:

- Mature, fast, differentiable, and easy to use.
- Provides a lightweight non-pointwise structural metric.

Similar alternatives:

- SSIM
- CW-SSIM
- FSIM
- GMSD
- VIF
- NLPD

Why not select alternatives first:

- MS-SSIM has the most convenient and stable PyTorch ecosystem among these
  simple structural metrics.
- CW-SSIM may be more robust to small translations/rotations, but package
  support is less convenient.

Typical loss:

```text
loss = 1 - ms_ssim(x, y)
```

Record:

- photometric positive loss
- small-shift positive loss
- negative loss
- runtime
- gradient norm

### 5.3 Scattering2D + L2

Package:

- `kymatio`

Representative family:

- Fixed multi-scale deformation-stable feature loss.

What it handles:

- Small translations.
- Small local deformation.
- Small diffeomorphism-like changes.
- Multi-scale texture similarity.

Why select it:

- It does not require training.
- It is more structured than simple Fourier magnitude loss.
- It is often appropriate for scientific fields, physical simulation outputs,
  PDE solution fields, and texture-like tensors.
- Kymatio provides PyTorch-compatible scattering transforms.

Similar alternatives:

- Wavelet feature loss
- Fourier magnitude loss
- Steerable pyramid loss
- HOG/SIFT feature loss
- Fixed CNN feature loss

Why not select alternatives first:

- Fourier magnitude can discard too much spatial information.
- Manually designed wavelet losses require more design choices.
- CNN feature losses may inject unwanted natural-image bias.
- Scattering is a compact representative of stable multi-scale features.

Typical form:

```text
feat_x = Scattering2D(x)
feat_y = Scattering2D(y)
loss = ||feat_x - feat_y||^2
```

Recommended first parameters:

- `J = 2` for more local sensitivity.
- `J = 3` for more deformation stability.
- Use `J = 3` in the first round unless images are very small.

Record:

- global transform positive loss
- local deformation positive loss
- negative texture loss
- gap
- runtime
- memory
- gradient norm

### 5.4 Bounded Affine Alignment + DISTS

Packages:

- `kornia` or `torch.nn.functional.grid_sample`
- `piq`

Representative family:

- Explicit finite global geometric registration + texture/structure loss.

What it handles:

- Global translation.
- Global rotation.
- Global scaling.
- Mild shear.
- Optional mirror as a finite branch.

Why select it:

- It directly implements the idea: allow only a limited global transformation,
  then compare content.
- The transform budget is interpretable and easy to constrain.
- It is simpler than training a spatial transformer network.

Similar alternatives:

- Phase correlation
- Fourier-Mellin registration
- Spatial transformer networks
- Homography registration
- Classical image registration

Why not select alternatives first:

- Fourier-Mellin requires extra engineering such as log-polar transforms and
  peak extraction.
- Spatial transformer networks require training a localization network.
- Bounded affine alignment is easy to implement and inspect.

Loss form:

```text
L = DISTS(warp(x; theta), y) + lambda * R(theta)
```

Suggested first transform budget:

```text
translation <= 5% image size
rotation <= 10 degrees
scale in [0.9, 1.1]
shear <= 0.05
```

Implementation outline:

1. Parameterize affine parameters with unconstrained variables.
2. Bound them using `tanh`.
3. Build affine matrix.
4. Warp `x` using Kornia or `grid_sample`.
5. Compute DISTS between warped `x` and `y`.
6. Add transform penalty.

Record:

- final loss
- DISTS part
- transform penalty
- translation norm
- rotation angle
- scale deviation
- shear magnitude if used
- runtime
- memory
- gradient norm

Abuse check:

- On negative pairs, if affine alignment reduces loss too much with large
  transform magnitude, the method is over-aligning unrelated samples.

### 5.5 Bounded Local Deformation + DISTS

Packages:

- `torch.nn.functional.grid_sample`
- optionally `kornia`
- `piq`

Representative family:

- Explicit finite local non-rigid deformation / diffeomorphism-like alignment.

What it handles:

- Local translation.
- Local rotation-like motion.
- Local stretch.
- Local compression.
- Elastic deformation.
- Small diffeomorphism-like warp.

Why select it:

- This is the closest candidate to the desired local diffeomorphism-style loss.
- It allows local deformation but makes it finite and regularized.
- It is simpler than VoxelMorph, DeepReg, or LDDMM for a first experiment.

Similar alternatives:

- TPS registration
- Elastic registration
- VoxelMorph
- DeepReg
- LDDMM
- Optical flow alignment
- Diffeomorphic demons

Why not select alternatives first:

- VoxelMorph and DeepReg are more like registration models and require their
  own model/training setup.
- LDDMM is theoretically strong but engineering-heavy.
- A low-resolution displacement grid with `grid_sample` is the simplest first
  version of finite local deformation.

Loss form:

```text
phi(p) = p + u(p)
L = DISTS(x o phi, y)
  + lambda_mag * ||u||^2
  + lambda_smooth * ||grad u||^2
```

Suggested first transform budget:

```text
low-resolution displacement grid: 4x4 or 8x8
max displacement <= 3% image diagonal
second round: max displacement <= 5% image diagonal
smoothness penalty required
```

Implementation outline:

1. Define a low-resolution displacement grid.
2. Bound displacement with `tanh`.
3. Upsample displacement to full image resolution.
4. Add displacement to base sampling grid.
5. Warp `x` with `grid_sample`.
6. Compute DISTS between warped `x` and `y`.
7. Add displacement magnitude and smoothness penalties.

Record:

- final loss
- DISTS part
- displacement norm
- max displacement
- smoothness penalty
- negative pair loss
- runtime
- memory
- gradient norm

Abuse check:

- This method is powerful and can cheat.
- If negative pairs get low loss by using large displacement, reduce max
  displacement or increase regularization.
- Inspect `max_displacement`, displacement norm, and smoothness penalty.

## 6. Relationship Among 2D Methods

| Method | Value changes | Global geometry | Local deformation | Texture difference | Explicit alignment |
|---|---:|---:|---:|---:|---:|
| MS-SSIM | strong | weak | weak | medium | no |
| DISTS | medium | weak/medium | weak/medium | strong | no |
| Scattering2D | medium | medium | medium | strong | no |
| bounded affine + DISTS | medium | strong | weak | strong | yes |
| bounded local deformation + DISTS | medium | medium | strong | strong | yes |

This set covers:

1. Intensity/local structural statistics.
2. Texture + structure perceptual similarity.
3. Stable multi-scale small-deformation features.
4. Bounded global geometric alignment.
5. Bounded local non-rigid deformation alignment.

## 7. 1D Candidate Losses

Baseline MSE/L1 is not included in the five main methods. It can be recorded as
a sanity check only.

The five 1D candidates are:

1. soft-DTW
2. Scattering1D + L2
3. bounded shift/scale + Scattering
4. Sinkhorn / OT on signal points
5. soft-min over finite transforms

### 7.1 Soft-DTW

Package:

- `tslearn`

Representative family:

- Differentiable dynamic time-alignment loss.

What it handles:

- Global time shift.
- Local stretch.
- Local compression.
- Phase drift.
- Speed mismatch.

Why select it:

- DTW is the classic method for time alignment.
- soft-DTW is the differentiable version suitable for gradient-based training.

Similar alternatives:

- DTW
- FastDTW
- derivative DTW
- shapeDTW
- CTC
- other time-warp invariant losses

Why not select alternatives first:

- Plain DTW is not smooth.
- FastDTW is mainly an approximate search method.
- CTC is more suitable for sequence-label alignment.
- soft-DTW is directly designed as a differentiable time-series loss.

Expected input:

```text
x, y: [B, T] or [B, T, D]
```

Recommended parameters:

```text
gamma = 0.05 or 0.1
normalize = True
```

Record:

- positive shift loss
- positive local warp loss
- negative loss
- gap
- runtime
- memory
- gradient norm

Note:

- Complexity is roughly quadratic in sequence length.
- For long sequences, use downsampling, windows, or shorter crops first.

### 7.2 Scattering1D + L2

Package:

- `kymatio`

Representative family:

- Stable multi-scale 1D spectral/texture feature loss.

What it handles:

- Small shifts.
- Local warp.
- Oscillatory patterns.
- Multi-scale frequency/texture structure.

Why select it:

- It compares stable multi-scale wavelet features rather than raw points.
- It is useful for physical signals, waves, spectra, and periodic patterns.
- It is differentiable through Kymatio's PyTorch frontend.

Similar alternatives:

- STFT magnitude loss
- Wavelet loss
- Fourier magnitude loss
- Mel-spectrogram loss
- Spectral loss

Why not select alternatives first:

- FFT magnitude can lose local structure.
- STFT and wavelet losses require more manual design choices.
- Scattering1D is a compact and principled first choice.

Typical form:

```text
feat_x = Scattering1D(x)
feat_y = Scattering1D(y)
loss = ||feat_x - feat_y||^2
```

Recommended parameters:

```text
J = 5 or 6
Q = 8
```

Record:

- positive shift loss
- positive local warp loss
- positive frequency-pattern loss
- negative loss
- gap
- runtime
- gradient norm

### 7.3 Bounded Shift/Scale + Scattering

Packages:

- PyTorch `grid_sample`
- `kymatio`

Representative family:

- Explicit bounded global 1D alignment.

What it handles:

- Global shift.
- Global stretch.
- Global compression.
- Mild global speed scaling.

Why select it:

- It is more constrained and interpretable than soft-DTW.
- It only allows simple global time-axis transformations.
- It is useful when local warp should not be freely allowed.

Similar alternatives:

- Cross-correlation alignment
- Phase alignment
- 1D spatial transformer network
- Fourier phase shift

Why not select alternatives first:

- Hard cross-correlation peak selection is not smoothly differentiable.
- A learned 1D STN adds another model.
- Direct bounded shift/scale with `grid_sample` is simple and controlled.

Loss form:

```text
t -> a * t + b
L = ||S(x(a*t + b)) - S(y)||^2
  + lambda * ((a - 1)^2 + b^2)
```

Suggested first transform budget:

```text
shift <= 5% length
scale in [0.9, 1.1]
```

Record:

- final loss
- scattering part
- shift value
- scale value
- transform penalty
- negative loss
- runtime
- gradient norm

### 7.4 Sinkhorn / OT on Signal Points

Package:

- `geomloss`

Representative family:

- Distribution matching / optimal transport loss.

What it handles:

- Small peak position shifts.
- Loose temporal correspondence.
- Overall shape/energy distribution similarity.

Why select it:

- OT compares distributions by transportation cost.
- It is less rigid than pointwise matching.
- GeomLoss provides PyTorch autograd-compatible Sinkhorn losses.

Similar alternatives:

- Exact Earth Mover's Distance
- Wasserstein distance
- Chamfer distance
- Hausdorff distance
- MMD
- Energy distance

Why not select alternatives first:

- Exact EMD is often too slow.
- Chamfer/Hausdorff are better for point sets or contours.
- MMD does not explicitly encode movement cost.
- Sinkhorn is smooth, differentiable, and package-supported.

Signal-to-point representation:

```text
p_i = (t_i, x_i)
```

Recommended parameters:

```text
blur = 0.05 or 0.1
n_points = 512 or 1024 for long signals
```

Record:

- Sinkhorn loss
- positive shift loss
- positive shape loss
- negative loss
- gap
- runtime
- gradient norm

### 7.5 Soft-Min Over Finite Transforms

Package:

- PyTorch

Representative family:

- Finite candidate transformation invariance.

What it handles:

- Explicitly listed shifts.
- Explicitly listed scales.
- Optional flip/mirror candidates.

Why select it:

- It is transparent and strongly controlled.
- It avoids unlimited transform freedom.
- It uses soft-min instead of hard-min so gradients can flow through multiple
  branches.

Similar alternatives:

- Hard min over augmentations
- Test-time augmentation loss
- Group-invariant pooling
- Equivariant/invariant feature pooling

Why not select alternatives first:

- Hard min / argmin is not smooth.
- Soft-min with logsumexp is easy and differentiable.

Example candidate set:

```text
shift = {-5%, -2.5%, 0, 2.5%, 5%}
scale = {0.95, 1.0, 1.05}
flip = {False, True}  # optional
```

Aggregation:

```text
L_i = D(T_i(x), y)
L = -tau * logsumexp(-L_i / tau)
```

Base loss options:

- Scattering1D loss
- soft-DTW
- L1
- Sinkhorn

Record:

- soft-min loss
- best candidate loss for diagnostics
- candidate weights if useful
- negative loss
- gap
- runtime
- gradient norm

## 8. Relationship Among 1D Methods

| Method | Global shift/scale | Local warp | Frequency/texture | Distribution | Explicit alignment |
|---|---:|---:|---:|---:|---:|
| soft-DTW | strong | strong | medium | weak | yes |
| Scattering1D | medium | medium | strong | medium | no |
| shift/scale + Scattering | strong | weak | strong | medium | yes |
| Sinkhorn signal points | medium | medium | medium | strong | semi-explicit |
| soft-min finite transforms | strong | depends on candidates | depends on base loss | depends on base loss | yes |

This set covers:

1. Dynamic time alignment.
2. Stable multi-scale 1D features.
3. Bounded global time transformation.
4. Distribution / transport distance.
5. Finite candidate transformation invariance.

## 9. Pair Construction

Before integrating losses into the main training loop, run controlled tests on
positive and negative pairs.

### 9.1 2D Pairs

For each 2D sample `x`, construct:

| Pair | Description | Purpose |
|---|---|---|
| `P1_photometric` | Same spatial positions, changed brightness/contrast/gamma/noise | Test value robustness |
| `P2_global_geometry` | Same content, global translation/rotation/scale/shear | Test global geometry robustness |
| `P3_local_deformation` | Same content, small elastic/local warp | Test local deformation robustness |
| `N1_negative` | Different sample/texture/physical state | Test separation from true negatives |

2D transformation strengths:

```text
photometric:
  brightness shift <= 0.1
  contrast scale in [0.9, 1.1]
  gamma in [0.8, 1.2]
  noise std <= 0.02

global geometry:
  translation <= 5% image size
  rotation <= 10 degrees
  scale in [0.9, 1.1]
  shear <= 0.05

local deformation:
  max displacement <= 3% image diagonal
  second round max displacement <= 5%
  smoothness required
```

### 9.2 1D Pairs

For each 1D signal `x`, construct:

| Pair | Description | Purpose |
|---|---|---|
| `P1_amplitude` | Same positions, changed gain/bias/noise | Test amplitude robustness |
| `P2_global_time` | Same pattern, global shift/scale | Test global time-axis robustness |
| `P3_local_warp` | Same pattern, local stretch/compression | Test local warp robustness |
| `N1_negative` | Different signal | Test separation from true negatives |

1D transformation strengths:

```text
amplitude:
  gain in [0.9, 1.1]
  bias <= 0.1 * signal_std
  noise std <= 0.02 * signal_std

global time:
  shift <= 5% length
  scale in [0.9, 1.1]

local warp:
  max displacement <= 3% length
  second round max displacement <= 5%
```

## 10. Metrics to Record

For every candidate loss, record:

```text
loss_P1
loss_P2
loss_P3
loss_N1

gap_P1 = loss_N1 - loss_P1
gap_P2 = loss_N1 - loss_P2
gap_P3 = loss_N1 - loss_P3

time_per_batch
memory_usage
gradient_norm
nan_or_inf_count
```

For explicit transform methods, also record:

```text
transform_magnitude
translation_norm
rotation_angle
scale_deviation
shear_magnitude
displacement_norm
smoothness_penalty
max_displacement
```

Expected behavior:

- Positive losses should be low.
- Negative loss should be high.
- Gaps should be large.
- Gradients should be stable.
- Transform magnitude should remain bounded.
- No NaN/Inf.
- Runtime should be acceptable.

## 11. Abuse Checks

Explicit alignment methods can cheat if they are too flexible. For each
alignment-based method, always inspect negative pairs.

Failure signs:

- Negative pair loss becomes close to positive pair loss.
- Transform magnitude reaches the maximum budget frequently.
- Dense displacement becomes noisy or unsmooth.
- Smoothness penalty is high but content loss is low.
- The method produces low loss by unreasonable warping.

Responses:

- Reduce max displacement.
- Increase transform magnitude penalty.
- Increase smoothness penalty.
- Reduce transform degrees of freedom.
- Use affine-only alignment before dense local deformation.

## 12. Execution Order

Do not implement everything at once.

### Stage 1: 2D Core Losses

Implement and test:

1. DISTS
2. Scattering2D + L2
3. bounded affine + DISTS
4. bounded local deformation + DISTS

Reason:

- These cover texture/structure, small deformation stability, global alignment,
  and local deformation.
- They are the most relevant to 2D diffeomorphism-like robustness.

### Stage 2: 1D Core Losses

Implement and test:

1. soft-DTW
2. Scattering1D + L2
3. bounded shift/scale + Scattering

Reason:

- These cover local time warp, stable multi-scale signal features, and bounded
  global time-axis alignment.

### Stage 3: Complete Candidate Set

Add:

2D:

1. MS-SSIM

1D:

1. Sinkhorn / OT on signal points
2. soft-min over finite transforms

## 13. Suggested Output Files

Suggested experiment directory:

```text
experiments/differentiable_losses/
```

Suggested outputs:

```text
experiments/differentiable_losses/results_2d.csv
experiments/differentiable_losses/results_1d.csv
experiments/differentiable_losses/summary_2d.md
experiments/differentiable_losses/summary_1d.md
experiments/differentiable_losses/config.json
```

Recommended columns for `results_2d.csv`:

```text
method
batch_size
height
width
channels
loss_P1_photometric
loss_P2_global_geometry
loss_P3_local_deformation
loss_N1_negative
gap_P1
gap_P2
gap_P3
time_per_batch_ms
memory_mb
gradient_norm_mean
gradient_norm_max
nan_or_inf_count
translation_norm
rotation_angle_deg
scale_deviation
displacement_norm
max_displacement
smoothness_penalty
```

Recommended columns for `results_1d.csv`:

```text
method
batch_size
length
channels
loss_P1_amplitude
loss_P2_global_time
loss_P3_local_warp
loss_N1_negative
gap_P1
gap_P2
gap_P3
time_per_batch_ms
memory_mb
gradient_norm_mean
gradient_norm_max
nan_or_inf_count
shift_value
scale_value
transform_penalty
```

## 14. Success Criteria

A method is promising if:

- It has low loss on allowed positive transformations.
- It has high loss on negative samples.
- `gap_P2` and `gap_P3` are large.
- Gradients are stable.
- Runtime and memory are acceptable.
- Explicit transforms stay within reasonable bounds.
- It does not over-align unrelated samples.

Most important 2D methods to watch:

1. bounded local deformation + DISTS
2. Scattering2D + L2
3. bounded affine + DISTS
4. DISTS
5. MS-SSIM

Most important 1D methods to watch:

1. soft-DTW
2. Scattering1D + L2
3. bounded shift/scale + Scattering
4. Sinkhorn / OT on signal points
5. soft-min over finite transforms

## 15. Final Recommendation

The most relevant 2D method for local diffeomorphism-like robustness is:

```text
bounded local deformation + DISTS
```

The best supporting 2D methods are:

```text
Scattering2D + L2
bounded affine + DISTS
DISTS
MS-SSIM
```

The most relevant 1D method for local time-warp robustness is:

```text
soft-DTW
```

The best supporting 1D methods are:

```text
Scattering1D + L2
bounded shift/scale + Scattering
Sinkhorn / OT on signal points
soft-min over finite transforms
```

The experiment should determine which method family best matches the data:

- perceptual texture/structure loss,
- stable multi-scale feature loss,
- explicit global alignment,
- explicit local deformation alignment,
- distribution/transport distance,
- or finite candidate transformation invariance.

The central rule is:

```text
Allow limited transformations, penalize transformation magnitude, keep the loss
differentiable, and verify that negative samples cannot be warped into low loss.
```

## 16. References and Package Links

- PIQ DISTS documentation:
  https://piq.readthedocs.io/en/latest/modules.html
- DISTS paper, "Image Quality Assessment: Unifying Structure and Texture Similarity":
  https://arxiv.org/abs/2004.07728
- pytorch-msssim:
  https://github.com/VainF/pytorch-msssim
- Kymatio user guide:
  https://www.kymat.io/userguide.html
- tslearn soft-DTW documentation:
  https://tslearn.readthedocs.io/en/stable/gen_modules/metrics/tslearn.metrics.soft_dtw.html
- soft-DTW paper:
  https://arxiv.org/abs/1703.01541
- GeomLoss PyTorch API:
  https://www.kernel-operations.io/geomloss/api/pytorch-api.html
- Kornia geometry transform documentation:
  https://kornia.readthedocs.io/en/latest/geometry.transform.html
- PyTorch `affine_grid` documentation:
  https://docs.pytorch.org/docs/main/generated/torch.nn.functional.affine_grid.html
