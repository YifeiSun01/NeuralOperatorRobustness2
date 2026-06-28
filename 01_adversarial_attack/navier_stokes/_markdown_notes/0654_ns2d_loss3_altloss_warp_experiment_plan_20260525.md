# NS2D Loss3 Alternative Loss / Warp Experiment Plan and Notes

Date: 2026-05-25  
Repo: `NeuralOperatorRobustness2`  
Task context: NS2D final-state adversarial perturbation experiment, `epsilon = 32`, `alpha = 10`, optimizer `steepest_add`, loss mode `loss3`, all-W / AW meaning **all W only**, not any A/D/dictionary mode.

## 1. Core Clarification

The alternative losses currently fall into two different families.

### Family A: non-warp losses

These methods only replace the distance/loss formula. They do **not** spatially move, rotate, stretch, compress, or warp either image.

Current methods:

- `dists`
- `ms_ssim`
- `scattering2d`

They compute distance directly between:

```text
model_output
solver_output
```

without constructing:

```text
aligned_model = warp(model_output, transform)
```

So these methods are not doing explicit spatial alignment. They may be more robust to small shifts because of feature/statistical comparisons, but there is no actual geometric deformation field.

### Family B: explicit-warp losses

These methods explicitly transform the model output before comparing it with the solver output.

Current methods:

- `affine_dists`
- `local_warp_dists`

They compute something like:

```text
aligned_model = warp(model_output, transform)
loss = DISTS(aligned_model, solver_output) + transform_regularization
```

For these methods, it makes sense to plot:

- `Aligned Model`
- `Aligned Model - Solver`
- `Alignment Warp`
- `Aligned Model - Model Output`

For non-warp methods, these plots should be blank, not copied from the original model output.

## 2. Current Experiment Setup

Current run root:

```text
2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/eps32_alpha10_steepest_add_loss3_allw_alt5_b10_20260525_035657_UTC
```

Experimental condition:

```text
epsilon = 32
alpha = 10
optimizer = steepest_add
NS2D final state
loss type = loss3
mode = all_w
batch size = 10
sample trace = sample_position 0
```

Methods already run:

```text
dists
ms_ssim
scattering2d
affine_dists
local_warp_dists
```

Baseline Loss 3/all-W source:

```text
2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/full_adw_b10_pair_outer_baseline_first_20260522/eps32_alpha10/mode_wwwwwwwwww_p2_q2_20260522_074135_UTC/batch_0000_0009/loss3/steepest_add
```

## 3. Current Methods and What They Mean

### 3.1 `dists`

Full name:

```text
DISTS: Deep Image Structure and Texture Similarity distance
```

Package:

```text
piq
```

Meaning:

DISTS compares structure and texture features. It is not pixelwise MSE/L2 and not W-Q norm. It is a feature/texture/structure distance.

Computation:

```text
loss = DISTS(model_output, solver_output)
```

Explicit warp:

```text
No
```

Important interpretation:

`dists` does not move the image. It only changes how model and solver are compared.

### 3.2 `ms_ssim`

Full name:

```text
MS-SSIM: Multi-Scale Structural Similarity distance
```

Package:

```text
pytorch-msssim
```

Meaning:

MS-SSIM itself is a similarity score: larger means more similar. In this experiment we convert it into a distance:

```text
loss = 1 - MS_SSIM(model_output, solver_output)
```

So larger loss means more dissimilar.

Explicit warp:

```text
No
```

Important interpretation:

`ms_ssim` mostly changes the value/structure comparison rule. It does not rotate, translate, stretch, or warp the image.

### 3.3 `scattering2d`

Full name:

```text
Scattering2D: 2D wavelet scattering feature L2 distance
```

Package:

```text
kymatio
```

Meaning:

The method extracts fixed wavelet scattering features and compares them with L2 distance:

```text
feat_model = Scattering2D(model_output)
feat_solver = Scattering2D(solver_output)
loss = ||feat_model - feat_solver||^2
```

Explicit warp:

```text
No
```

Important interpretation:

Scattering features are more stable to small translations and small deformations, but no actual spatial deformation is computed.

### 3.4 `affine_dists`

Full name:

```text
Affine + DISTS: bounded affine alignment followed by DISTS
```

Package / mechanism:

```text
torch.nn.functional.affine_grid
torch.nn.functional.grid_sample
piq.DISTS
```

Meaning:

This method estimates a bounded affine/similarity transform and warps the model output before applying DISTS.

It can express global transformations such as:

- translation
- rotation
- scaling
- mild shear if included later

Current implementation:

```text
aligned_model = affine_warp(model_output, theta)
loss = DISTS(aligned_model, solver_output) + affine_regularization
```

Explicit warp:

```text
Yes
```

Current budget:

```text
translation <= 5% image size
rotation <= 10 degrees
scale roughly in [0.909, 1.1]
inner_steps = 8
learning_rate = 0.05
regularization weight = 0.01
```

For a 256x256 field, `5%` translation means about 12.8 pixels.

### 3.5 `local_warp_dists`

Full name:

```text
Local warp + DISTS: bounded local deformation / dense-warp alignment followed by DISTS
```

Package / mechanism:

```text
torch.nn.functional.grid_sample
piq.DISTS
```

Meaning:

This method estimates a low-resolution local displacement grid, upsamples it to full resolution, and warps the model output before applying DISTS.

It can express local transformations such as:

- local shift
- local stretch
- local compression
- smooth local bending
- small diffeomorphism-like deformation

Current implementation:

```text
disp_low = learnable low-resolution displacement grid
disp = interpolate(disp_low to image size)
aligned_model = grid_sample(model_output, identity_grid + disp)
loss = DISTS(aligned_model, solver_output)
     + magnitude_penalty
     + smoothness_penalty
```

Explicit warp:

```text
Yes
```

Current budget:

```text
grid_size = 8x8
max_disp_ratio = 0.03
inner_steps = 8
learning_rate = 0.05
magnitude weight = 0.01
smoothness weight = 0.05
```

In the current code, displacement is applied in normalized grid coordinates. The effective maximum is small enough that the final visual warp is subtle.

## 4. Observed Results From Current Plots and Diagnostics

### 4.1 Non-warp methods have no aligned model

For these rows:

```text
Loss 3 baseline
DISTS
MS-SSIM
Scattering2D
```

`Aligned Model`, `Aligned Model - Solver`, and `Alignment Warp` should be blank.

Reason:

```text
aligned_model == model_output
warp == 0
```

for methods without explicit alignment.

The plot script was updated so that these alignment columns are blank for non-warp methods.

### 4.2 Explicit-warp methods are not identical before and after warp

For sample 0, final frame, the direct pixelwise difference:

```text
Aligned Model - Model Output
```

was measured.

| method | nonzero pixels | mean abs(aligned-model) | max abs | warp mean | warp max |
|---|---:|---:|---:|---:|---:|
| Affine + DISTS | 65532 / 65536 | 0.098819 | 2.47925 | 0.01236 | 0.02636 |
| Local warp + DISTS | 65536 / 65536 | 0.229607 | 3.27000 | 0.02254 | 0.03221 |

Across all 101 saved trace frames:

| method | frames with nonzero difference | mean abs average | mean abs range | max abs max | nonzero pixels per frame |
|---|---:|---:|---:|---:|---:|
| Affine + DISTS | 101 / 101 | 0.216059 | 0.03547 - 0.42016 | 4.32638 | 65531 - 65536 |
| Local warp + DISTS | 101 / 101 | 0.188248 | 0.04747 - 0.24846 | 3.62117 | 65532 - 65536 |

Interpretation:

The warp methods do change the model output. The changes are real at pixel level, but visually subtle in the large heatmap because the colormap is dominated by the field's full value range.

### 4.3 Diagnostic plot added

A new diagnostic plot was created to directly compare warp before/after:

```text
Model Output Before Warp
Aligned Model After Warp
Aligned - Model Pointwise Diff
Solver Output
Aligned Model - Solver
Warp Magnitude
```

Output:

```text
docs/ns2d_eps32_alpha10_altloss_original_cleanstyle_20260525/eps32_alpha10_steepest_add_alignment_before_after_diff_dataset0.png
```

Summary CSV:

```text
docs/ns2d_eps32_alpha10_altloss_original_cleanstyle_20260525/alignment_before_after_diff_summary.csv
```

Per-frame pixel-difference CSV:

```text
docs/ns2d_eps32_alpha10_altloss_original_cleanstyle_20260525/alignment_model_minus_model_per_frame_stats.csv
```

## 5. Plotting Corrections and Mistakes to Remember

These were important corrections during the work.

### 5.1 AW means all-W, not all-A-target-W

The user intended:

```text
all_w
```

not:

```text
all_a_target_w
```

Therefore the run does not need dictionary-based A/D lookup. Any script that checks for dictionary only because it sees the letter `a` in `all_w` is wrong.

### 5.2 Loss curves must be comparable

Originally, there was a risk of plotting each method's active optimization loss on one axis. That is not meaningful because DISTS, MS-SSIM, Scattering2D, affine DISTS, and local warp DISTS are different quantities.

Correct policy:

```text
For cross-method comparison, plot the same all-W Loss 3 / W-Q norm for every method.
```

Current top loss curve now uses:

```text
||adv_model_final - adv_solver_final||_2
```

from `step_sample_trace.npz` for sample 0.

### 5.3 Solver output naming

Use:

```text
Solver Output
```

not:

```text
Software Output
```

### 5.4 Non-warp alignment columns must be blank

For non-warp rows, do not show copied `Aligned Model` images and do not show zero warp heatmaps. That is misleading.

Correct policy:

```text
For DISTS / MS-SSIM / Scattering2D, alignment columns are blank.
```

### 5.5 Alignment colorbars must only span explicit-warp rows

If the alignment heatmap cells are blank for non-warp methods, the corresponding colorbars should not start from the first row.

Correct policy:

```text
Alignment column colorbars start at Affine + DISTS row and end at Local warp + DISTS row.
```

This has been fixed.

### 5.6 Current aligned-model visualization is offline recomputed

The current attack run did not save the internal alignment transform for every method as an official attack artifact.

The plot recomputes alignment offline from:

```text
adv_model_final
adv_solver_final
```

using the same style of default alignment settings.

Therefore the diagnostic alignment plot is a visualization of the alignment rule, not an originally saved inner-loop state from the attack.

For future runs, save the actual transform parameters / displacement field during optimization if alignment diagnostics are important.

## 6. Additional Explicit Warp Methods to Add Next

The next experiment should add more methods that actually perform spatial transformation.

Recommended new methods:

```text
homography_dists
tps_dists
elastic_dists
svf_dists
```

Optional later method:

```text
voxelmorph_dists
```

### 6.1 `homography_dists`

Full name:

```text
Homography + DISTS
```

Package:

```text
kornia
```

Possible API / components:

```text
kornia.geometry.transform.warp_perspective
kornia.geometry.transform.get_perspective_transform
```

Meaning:

A homography is a global projective transform. It is stronger than affine. It can represent:

- translation
- rotation
- scale
- shear
- perspective-like four-corner distortion

Suggested loss:

```text
aligned_model = homography_warp(model_output, H)
loss = DISTS(aligned_model, solver_output) + lambda * homography_regularization
```

Suggested strong budget for visual difference:

```text
corner displacement <= 10% image size
inner_steps = 20
regularization = 0.005 to 0.02
```

For 256x256, 10% means about 25 pixels.

Use case:

Good for seeing strong global geometric differences beyond rotation/scale.

Risk:

Can become too powerful if corner displacement is large.

### 6.2 `tps_dists`

Full name:

```text
Thin Plate Spline + DISTS
```

Package:

```text
kornia
```

Possible API / components:

```text
kornia.geometry.transform.get_tps_transform
kornia.geometry.transform.warp_image_tps
```

Meaning:

TPS uses control points to create smooth nonlinear deformation. It is a good middle ground between global affine and dense local warp.

It can represent:

- smooth local bending
- local stretch/compression
- local translation
- control-point based warping

Suggested loss:

```text
control_points_dst = control_points_src + bounded_offsets
kernel_weights, affine_weights = get_tps_transform(...)
aligned_model = warp_image_tps(model_output, ...)
loss = DISTS(aligned_model, solver_output)
     + lambda_offset * ||offsets||^2
     + lambda_bend * bending_or_smoothness_penalty
```

Suggested strong budget:

```text
control grid = 4x4 or 5x5
max control-point displacement = 10% image size
inner_steps = 20
regularization = 0.01 to 0.05
```

Use case:

Probably the best next method for visually interpretable local deformation.

Risk:

If control-point displacement is too large, it can over-align unrelated patterns.

### 6.3 `elastic_dists`

Full name:

```text
Elastic Transform + DISTS
```

Package:

```text
kornia
```

Possible API / components:

```text
kornia.geometry.transform.elastic_transform2d
```

Meaning:

A smooth elastic deformation generated from a noise/displacement field. In our context, the field should be optimized or parameterized, not random at evaluation time.

It can represent:

- rubber-sheet deformation
- local smooth distortion
- local stretch/compression

Suggested loss:

```text
noise_or_field = bounded learnable field
aligned_model = elastic_transform2d(model_output, noise_or_field, alpha, sigma)
loss = DISTS(aligned_model, solver_output)
     + magnitude_penalty
     + smoothness_penalty
```

Suggested strong budget:

```text
alpha equivalent to about 15-25 px max visible displacement
sigma = 8 to 16 px for smoothness
inner_steps = 20
magnitude weight = 0.002 to 0.01
smoothness weight = 0.01 to 0.03
```

Use case:

Good for visibly strong smooth local deformation.

Risk:

Less interpretable than TPS unless the field is carefully visualized.

### 6.4 `svf_dists`

Full name:

```text
Stationary Velocity Field / Diffeomorphic Warp + DISTS
```

Package:

```text
MONAI
```

Possible API / components:

```text
monai.networks.blocks.Warp
monai.networks.blocks.DVF2DDF
```

Meaning:

Instead of directly optimizing displacement, optimize a velocity field. Then integrate it into a displacement field with scaling-and-squaring. This is closer to true diffeomorphic registration.

Suggested loss:

```text
velocity = bounded smooth velocity field
ddf = DVF2DDF(velocity)
aligned_model = Warp(model_output, ddf)
loss = DISTS(aligned_model, solver_output)
     + velocity_magnitude_penalty
     + velocity_smoothness_penalty
     + optional jacobian/folding_penalty
```

Suggested strong budget:

```text
velocity max around 0.06 to 0.08 normalized grid
integration steps = 5 to 7
inner_steps = 20
smoothness weight = 0.01 to 0.05
magnitude weight = 0.002 to 0.01
```

Use case:

Best match to the user's original diffeomorphism idea.

Risk:

More implementation complexity and slower runtime.

### 6.5 Optional later: `voxelmorph_dists`

Package:

```text
voxelmorph
```

Meaning:

A registration network predicts the deformation field. This is powerful but becomes a separate model/training problem.

Recommendation:

Do not add in the immediate next run unless we specifically want to train or optimize a registration network. Use `svf_dists` first as a lighter diffeomorphic route.

## 7. Strong Warp Budget Proposal

The user wants visually obvious differences between methods. The next round should include a strong-budget smoke test.

### Current budget was too mild visually

Current approximate budgets:

```text
affine_dists:
  translation <= 5% image size
  rotation <= 10 degrees
  scale roughly in [0.909, 1.1]
  inner_steps = 8

local_warp_dists:
  grid_size = 8x8
  max_disp_ratio = 0.03
  inner_steps = 8
  mag_weight = 0.01
  smooth_weight = 0.05
```

Observed actual warp max was roughly a few pixels in final visualization, so the main heatmap did not show dramatic differences.

### Proposed strong settings

Use these only for diagnostic/visual-difference experiments first.

```text
affine_dists_very_strong:
  translation <= 20% image size
  rotation <= 45 degrees
  scale in [0.5, 2.0]
  inner_steps = 30
  reg_weight = 0.0005
```

```text
local_warp_dists_very_strong:
  grid_size = 16
  max_disp_ratio = 0.15
  inner_steps = 30
  mag_weight = 0.0005
  smooth_weight = 0.003
```

```text
homography_dists_very_strong:
  corner displacement <= 18% image size
  inner_steps = 30
  reg_weight = 0.001
```

```text
tps_dists_very_strong:
  control grid = 6x6
  max control-point displacement = 18% image size
  inner_steps = 30
  offset_reg = 0.003
  bending/smoothness_reg = 0.006
```

```text
elastic_dists_very_strong:
  target visible displacement about 30-40 px
  kernel = 7, passes = 1
  inner_steps = 30
  mag_weight = 0.0005
  smooth_weight = 0.003
```

```text
svf_dists_very_strong:
  velocity max = 0.14 normalized grid
  integration_steps = 7
  inner_steps = 30
  mag_weight = 0.0005
  smooth_weight = 0.003
```

### Important warning

Strong budgets are useful for seeing differences, but they can make the loss cheat. If the warp is too powerful, unrelated fields can be forcibly aligned.

Therefore run two phases:

1. Strong-budget smoke test for visual diagnosis.
2. Moderate-budget formal run for fair comparison.

## 8. Proposed Next Experiment Matrix

### Phase A: smoke test

Goal:

Estimate runtime and visual behavior.

Run:

```text
epsilon = 32
alpha = 10
optimizer = steepest_add
batch size = 10
steps = 3 to 5
sample trace = index 0
```

Methods:

```text
affine_dists_strong
local_warp_dists_strong
homography_dists_strong
tps_dists_strong
elastic_dists_strong
svf_dists_strong
```

Record:

```text
time_per_step
GPU memory
active loss
common Loss 3 / W-Q norm
delta L2
delta Linf
warp magnitude mean/max
transform parameters
NaN/Inf count
grad norm
```

For plots:

```text
Model Output Before Warp
Aligned Model After Warp
Aligned - Model
Solver Output
Aligned Model - Solver
Warp Magnitude
```

### Phase B: full strong-budget diagnostic run

Goal:

See whether perturbations become visually and mechanistically different.

Run same condition as current full run:

```text
epsilon = 32
alpha = 10
optimizer = steepest_add
steps = same as prior full run
batch size = 10
```

Only after Phase A shows no obvious instability.

### Phase C: moderate-budget formal run

Goal:

Fair comparison without too much alignment cheating.

Use smaller budgets than Phase A, for example:

```text
affine:
  translation <= 6-8%
  rotation <= 15 degrees
  scale in [0.85, 1.2]

local_warp:
  max_disp_ratio = 0.04 to 0.05
  grid_size = 8

tps:
  max control displacement = 5%

homography:
  corner displacement <= 5%

svf:
  velocity max = 0.03 to 0.05
```

## 9. Implementation Tasks

### 9.1 Extend loss choices

File:

```text
2D_NS_FNO2d_recurrent/perturbation_methods/ns2d_alternative_losses.py
```

Add choices:

```text
homography_dists
tps_dists
elastic_dists
svf_dists
```

Optional later:

```text
voxelmorph_dists
```

### 9.2 Add CLI parameters

Suggested new args:

```text
--loss3_homography_inner_steps
--loss3_homography_lr
--loss3_homography_max_corner_ratio
--loss3_homography_reg_weight

--loss3_tps_grid_size
--loss3_tps_inner_steps
--loss3_tps_lr
--loss3_tps_max_disp_ratio
--loss3_tps_offset_weight
--loss3_tps_smooth_weight

--loss3_elastic_inner_steps
--loss3_elastic_lr
--loss3_elastic_max_disp_ratio
--loss3_elastic_sigma
--loss3_elastic_mag_weight
--loss3_elastic_smooth_weight

--loss3_svf_grid_size
--loss3_svf_inner_steps
--loss3_svf_lr
--loss3_svf_max_vel_ratio
--loss3_svf_int_steps
--loss3_svf_mag_weight
--loss3_svf_smooth_weight
```

### 9.3 Save alignment diagnostics during attack

For explicit-warp methods, save at least for sample index 0:

```text
aligned_model_final
alignment_warp_magnitude
alignment_model_minus_model
alignment_model_minus_solver
transform_parameters_or_displacement_field
```

This avoids relying only on offline recomputation.

### 9.4 Update plotting

Main plot policy:

- Non-warp methods have blank alignment columns.
- Explicit-warp methods show alignment columns.
- Alignment colorbars only span explicit-warp rows.
- Curves/spectra that compare methods should share the same axis and same metric.
- Loss curve should be common all-W Loss 3 / W-Q norm, not each method's active loss.

Diagnostic plot policy:

For each explicit-warp method, show:

```text
Model Output Before Warp
Aligned Model After Warp
Aligned - Model
Solver Output
Aligned Model - Solver
Warp Magnitude
```

## 10. Package / Requirement Notes

Required for the current 9-method implementation:

```text
piq
pytorch-msssim
kymatio
kornia
kornia-rs
monai
torch
torchvision
```

Optional if adding VoxelMorph route later:

```text
voxelmorph
```

Recommendation:

Kornia and MONAI are now first-class dependencies for the explicit-warp methods. Avoid adding `voxelmorph` until we intentionally want a registration-network experiment.

## 11. References

Kornia image registration / homography:

```text
https://www.kornia.org/tutorials/nbs/image_registration.html
https://kornia.readthedocs.io/en/latest/geometry.transform.html
```

Kornia TPS:

```text
https://kornia.readthedocs.io/en/latest/geometry.transform.html
```

Kornia elastic transform:

```text
https://kornia.readthedocs.io/en/v0.6.4/_modules/kornia/geometry/transform/elastic_transform.html
```

PyTorch grid warping:

```text
https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
https://docs.pytorch.org/docs/main/generated/torch.nn.functional.affine_grid.html
```

MONAI Warp / DVF2DDF:

```text
https://docs.monai.io/en/latest/networks.html
https://docs.monai.io/en/1.3.0/_modules/monai/networks/blocks/warp.html
```

VoxelMorph:

```text
https://github.com/voxelmorph/voxelmorph
```

## 12. Short Takeaway

The current five methods are not all the same type.

```text
DISTS / MS-SSIM / Scattering2D:
  only replace the loss/distance formula;
  no spatial warp.

Affine + DISTS / Local warp + DISTS:
  explicitly warp model output before comparing with solver output.
```

Next, add more explicit spatial-deformation losses:

```text
homography_dists
tps_dists
elastic_dists
svf_dists
```

First run them with strong budgets to make the deformation visually obvious, then reduce budgets for a fair formal experiment.

## 13. Implemented Next-Round Strong-Warp 9-Method Setup

This section records the concrete implementation prepared after the initial 5-method experiment.

### 13.1 New methods added to code

File updated:

```text
2D_NS_FNO2d_recurrent/perturbation_methods/ns2d_alternative_losses.py
```

New `loss3_metric` choices added:

```text
homography_dists
tps_dists
elastic_dists
svf_dists
```

The full next-round method list is now 9 methods:

```text
dists
ms_ssim
scattering2d
affine_dists
local_warp_dists
homography_dists
tps_dists
elastic_dists
svf_dists
```

The first three methods are still non-warp loss replacements:

```text
dists
ms_ssim
scattering2d
```

The last six methods are explicit spatial-warp methods:

```text
affine_dists
local_warp_dists
homography_dists
tps_dists
elastic_dists
svf_dists
```

### 13.2 Implementation note about official packages

Correction: the 9-method implementation should use official geometry / registration packages for the explicit warp methods, not only hand-written PyTorch helper code. The `adv_robust` environment has now been updated with:

```text
kornia==0.8.3
kornia-rs==0.1.14
monai==1.5.2
```

`requirements.txt` has also been updated with these dependencies.

Current official backend mapping:

```text
affine_dists      -> kornia.geometry.transform.warp_affine
local_warp_dists  -> monai.networks.blocks.Warp
homography_dists  -> kornia.geometry.transform.get_perspective_transform + warp_perspective
tps_dists         -> kornia.geometry.transform.get_tps_transform + warp_image_tps
elastic_dists     -> kornia.geometry.transform.elastic_transform2d
svf_dists         -> monai.networks.blocks.DVF2DDF + monai.networks.blocks.Warp
```

The remaining non-warp methods still use their original packages:

```text
dists        -> piq
ms_ssim      -> pytorch-msssim
scattering2d -> kymatio
```

The inner registration parameters are still optimized locally for each batch/sample, but the actual affine, homography, TPS, elastic, dense warp, and SVF integration operations now go through Kornia or MONAI. A finite-value/clamp guard is applied after inner optimizer steps so registration parameters do not become NaN or invalid before calling official solvers such as Kornia DLT.

### 13.3 New launcher

New launcher script:

```text
tools/run_ns2d_eps32_alpha10_steepest_add_loss3_allw_alt9_strongwarp_steps25_20260525.sh
```

Purpose:

```text
Run NS2D final-state loss3/all-W attack with 9 alternative metrics,
strong warp budgets,
epsilon = 32,
alpha = 10,
optimizer = steepest_add,
batch size = 10,
steps = 25.
```

The output tag is changed from the previous 100-step style to a 25-step style:

```text
eps32_alpha10_steepest_add_loss3_allw_alt9_strongwarp_steps25_b10_<timestamp>_UTC
```

### 13.4 Old vs current very-strong parameter table

The next run is intentionally configured to be visually aggressive. The goal is not a conservative production setting; the goal is to make perturbation/result differences easier to see by eye.

| Method | Earlier setting | Current very-strong setting |
|---|---|---|
| `dists` | no geometric parameter | no geometric parameter; same DISTS distance |
| `ms_ssim` | no geometric parameter | no geometric parameter; same `1 - MS-SSIM` distance |
| `scattering2d` | `J = 3`, then `J = 4` | `J = 5` for broader multi-scale feature comparison |
| `affine_dists` | `inner_steps=20`, `shift<=12%`, `rotation<=25 deg`, `scale<=1.35`, `reg=0.002` | `inner_steps=30`, `shift<=20%`, `rotation<=45 deg`, `scale in [0.5, 2.0]`, `reg=0.0005` |
| `local_warp_dists` | `grid=12`, `inner_steps=20`, `max_disp_ratio=0.08`, `mag=0.002`, `smooth=0.01` | `grid=16`, `inner_steps=30`, `max_disp_ratio=0.15`, `mag=0.0005`, `smooth=0.003` |
| `homography_dists` | `inner_steps=20`, `max_corner_ratio=0.10`, `reg=0.005` | `inner_steps=30`, `max_corner_ratio=0.18`, `reg=0.001` |
| `tps_dists` | `grid=5`, `inner_steps=20`, `max_disp_ratio=0.10`, `offset=0.01`, `smooth=0.02` | `grid=6`, `inner_steps=30`, `max_disp_ratio=0.18`, `offset=0.003`, `smooth=0.006` |
| `elastic_dists` | `grid=16`, `inner_steps=20`, `max_disp_ratio=0.08`, `smooth_kernel=9`, `smooth_passes=2`, `mag=0.002`, `smooth=0.01` | `grid=20`, `inner_steps=30`, `max_disp_ratio=0.15`, `smooth_kernel=7`, `smooth_passes=1`, `mag=0.0005`, `smooth=0.003` |
| `svf_dists` | `grid=12`, `inner_steps=20`, `max_vel_ratio=0.08`, `integration_steps=6`, `mag=0.002`, `smooth=0.01` | `grid=16`, `inner_steps=30`, `max_vel_ratio=0.14`, `integration_steps=7`, `mag=0.0005`, `smooth=0.003` |

### 13.5 Interpretation of very-strong budgets

For 256x256 NS2D fields, rough pixel-scale interpretation:

```text
20% affine translation ~= 51 pixels
18% homography corner / TPS control displacement ~= 46 pixels
max_disp_ratio = 0.15 gives normalized component bound about 0.30 ~= 38 pixels per axis
scale in [0.5, 2.0] allows obvious zoom/shrink behavior
45 degree affine rotation is deliberately large
```

These settings are intentionally strong. The goal is to make perturbation differences and warp effects visually obvious.

This is not necessarily the final fair setting. It is a diagnostic setting.

### 13.6 Attack steps changed

Previous full run:

```text
steps = 100
```

Next strong-warp run:

```text
steps = 25
```

Reason:

- 9 methods are more expensive than 5.
- Strong-warp inner optimization uses 20 registration steps for warp methods.
- 25 attack steps should be enough for a first diagnostic run to see whether perturbations diverge visually.

### 13.7 What to record in the new run

The launcher keeps:

```text
RECORD_FINAL_STATE_OUTPUTS=1
RECORD_STEP_SAMPLE_OUTPUTS=1
RECORD_STEP_SAMPLE_POSITION=0
RECORD_STEP_SAMPLE_EVERY=1
RECORD_STEP_SAMPLE_GRADIENTS=1
```

This records detailed per-step arrays only for sample position 0, while batch-level metrics cover all 10 samples.

The run should record:

```text
per_step_metrics.csv
per_sample_step_metrics.csv
final_state_outputs.npz
final_state_metrics.csv
step_sample_trace.npz
summary.json
batch_memory.csv
```

For explicit-warp methods, metric diagnostics should include method-specific quantities such as:

```text
corner_displacement_mean/max
control_displacement_mean/max
displacement_norm/max_displacement
velocity_norm/svf_max_displacement
content_loss
regularization terms
```

### 13.8 Background launch command

Use:

```bash
cd /workspace/NeuralOperatorRobustness2
nohup bash tools/run_ns2d_eps32_alpha10_steepest_add_loss3_allw_alt9_strongwarp_steps25_20260525.sh > /tmp/ns2d_alt9_strongwarp_steps25_launch.log 2>&1 & disown
```

Monitor:

```bash
tail -f /tmp/ns2d_alt9_strongwarp_steps25_launch.log
```

The script also writes its own run log under the run directory:

```text
2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/<RUN_TAG>/nohup_<RUN_TAG>.log
```

### 13.9 Caution

The strong settings may make explicit-warp losses more visually different, but they also increase the chance of alignment cheating.

After the strong diagnostic run, if results are too warped, run a moderate-budget version before drawing final scientific conclusions.

### 13.10 Smoke test result and numerical fix

After adding the four new metrics, a toy forward/backward smoke test was run on 256x256 tensors for all 9 metrics:

```text
dists
ms_ssim
scattering2d
affine_dists
local_warp_dists
homography_dists
tps_dists
elastic_dists
svf_dists
```

Result:

```text
SMOKE_OK all metrics
```

Each method produced finite loss and finite gradient.

Important fix discovered during smoke test:

The first implementation of `elastic_dists` and `svf_dists` used a regularization pattern like:

```text
sqrt(mean(square(field))).square()
```

At the zero-displacement initialization this can produce NaN gradients because the derivative of `sqrt` at zero is singular.

Fix:

Use the squared mean directly for optimization regularization:

```text
mean(square(field))
```

and only compute the square-rooted norm for detached diagnostics, with a small clamp.

This fix is already applied in:

```text
2D_NS_FNO2d_recurrent/perturbation_methods/ns2d_alternative_losses.py
```

## 15. Smoke Timing for New Official Warp Methods

Date: 2026-05-25 UTC.

Scope:

```text
metrics = homography_dists, tps_dists, elastic_dists, svf_dists
batch_size = 10
epsilon = 32
alpha = 10
optimizer = steepest_add
mode = loss3 / all_w
very-strong warp budgets enabled
```

### 15.1 Micro loss-only smoke

This isolates final-state loss forward/backward on random `[10, 1, 256, 256]` tensors. It does not include FNO rollout or solver cost.

CSV:

```text
2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/loss3_metric_micro_smoke_new4_verystrong_20260525.csv
```

| Metric | seconds / step | seconds / step / sample | peak CUDA MB | Notes |
|---|---:|---:|---:|---|
| `homography_dists` | 0.1785 | 0.01785 | 2322.8 | Kornia homography + DISTS |
| `tps_dists` | 0.1401 | 0.01401 | 2325.6 | Kornia TPS + DISTS |
| `elastic_dists` | 0.1105 | 0.01105 | 2325.6 | Kornia elastic + DISTS |
| `svf_dists` | 3.0506 | 0.30506 | 2379.2 | MONAI DVF2DDF + Warp + DISTS; much slower |

Takeaway: loss-only overhead is small for homography/TPS/elastic but large for SVF.

### 15.2 Actual attack-path smoke, compute-only logging

Run settings:

```text
STEPS=5
TRUE_LOSS_EVERY=0
RECORD_FINAL_STATE_OUTPUTS=0
RECORD_STEP_SAMPLE_OUTPUTS=0
RECORD_STEP_SAMPLE_GRADIENTS=0
```

CSV:

```text
2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/smoke_eps32_alpha10_steepest_add_loss3_allw_new4_verystrong_b10_s5_20260525_080121_UTC/new4_verystrong_smoke_timing_summary.csv
```

| Metric | 5-step wall time | seconds / attack step | estimated 25-step time | peak allocated GPU GB |
|---|---:|---:|---:|---:|
| `homography_dists` | 90 s | 18.0 | 7.50 min | 45.74 |
| `tps_dists` | 89 s | 17.8 | 7.42 min | 45.74 |
| `elastic_dists` | 90 s | 18.0 | 7.50 min | 45.74 |
| `svf_dists` | 109 s | 21.8 | 9.08 min | 45.82 |

Estimated four-method 25-step total without detailed recording: about 31.5 minutes.

### 15.3 Actual attack-path smoke, formal recording enabled

This is closer to the real planned run because it leaves the launcher defaults enabled: final-state outputs, sample-0 step outputs/gradients, and true loss evaluation.

Run settings:

```text
STEPS=3
TRUE_LOSS_EVERY=1
RECORD_FINAL_STATE_OUTPUTS=1
RECORD_STEP_SAMPLE_OUTPUTS=1
RECORD_STEP_SAMPLE_GRADIENTS=1
```

CSV:

```text
2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/smoke_recording_on_eps32_alpha10_loss3_allw_new4_verystrong_b10_s3_20260525_080856_UTC/new4_verystrong_smoke_timing_summary.csv
```

| Metric | 3-step wall time | seconds / attack step | estimated 25-step time | peak allocated GPU GB |
|---|---:|---:|---:|---:|
| `homography_dists` | 75 s | 25.00 | 10.42 min | 45.74 |
| `tps_dists` | 77 s | 25.67 | 10.69 min | 45.74 |
| `elastic_dists` | 75 s | 25.00 | 10.42 min | 45.74 |
| `svf_dists` | 89 s | 29.67 | 12.36 min | 45.82 |

Estimated four-method 25-step total with formal recording: about 43.9 minutes.

Practical estimate:

```text
new four methods only, 25 steps, batch 10: ~44 minutes
new four methods only, 25 steps, minimal logging: ~32 minutes
SVF is the slowest method.
Peak GPU memory is about 46 GB on A100 80GB.
```

## 16. Four-Budget Sweep Plan

The next production sweep runs the same 9 loss3/all-W metrics under four explicit warp-budget presets.

Order is fixed by request:

```text
1. very_strong
2. strong
3. medium
4. weak
```

Each preset uses:

```text
epsilon = 32
alpha = 10
optimizer = steepest_add
mode = loss3 / all_w
batch size = 10
steps = 25
metrics = dists ms_ssim scattering2d affine_dists local_warp_dists homography_dists tps_dists elastic_dists svf_dists
```

Launcher:

```text
tools/run_ns2d_eps32_alpha10_steepest_add_loss3_allw_alt9_budget_sweep_steps25_20260525.sh
```

Default preset values:

| Preset | Scattering J | Affine | Local warp | Homography | TPS | Elastic | SVF |
|---|---:|---|---|---|---|---|---|
| `very_strong` | 5 | inner=30, shift=20%, rot=45 deg, scale=[0.5,2.0], reg=0.0005 | grid=16, inner=30, disp=0.15, mag=0.0005, smooth=0.003 | inner=30, corner=0.18, reg=0.001 | grid=6, inner=30, disp=0.18, offset=0.003, smooth=0.006 | grid=20, inner=30, disp=0.15, kernel=7, passes=1, mag=0.0005, smooth=0.003 | grid=16, inner=30, vel=0.14, int=7, mag=0.0005, smooth=0.003 |
| `strong` | 4 | inner=20, shift=12%, rot=25 deg, scale<=1.35, reg=0.002 | grid=12, inner=20, disp=0.08, mag=0.002, smooth=0.01 | inner=20, corner=0.10, reg=0.005 | grid=5, inner=20, disp=0.10, offset=0.01, smooth=0.02 | grid=16, inner=20, disp=0.08, kernel=9, passes=2, mag=0.002, smooth=0.01 | grid=12, inner=20, vel=0.08, int=6, mag=0.002, smooth=0.01 |
| `medium` | 4 | inner=12, shift=8%, rot=15 deg, scale<=1.2, reg=0.005 | grid=10, inner=12, disp=0.05, mag=0.005, smooth=0.02 | inner=12, corner=0.07, reg=0.008 | grid=4, inner=12, disp=0.07, offset=0.008, smooth=0.03 | grid=16, inner=12, disp=0.06, kernel=9, passes=2, mag=0.005, smooth=0.02 | grid=10, inner=12, vel=0.06, int=5, mag=0.005, smooth=0.03 |
| `weak` | 3 | inner=8, shift=5%, rot=10 deg, scale<=1.1, reg=0.01 | grid=8, inner=8, disp=0.03, mag=0.01, smooth=0.05 | inner=8, corner=0.05, reg=0.01 | grid=4, inner=8, disp=0.05, offset=0.01, smooth=0.05 | grid=16, inner=8, disp=0.05, kernel=9, passes=2, mag=0.01, smooth=0.03 | grid=8, inner=8, vel=0.04, int=5, mag=0.01, smooth=0.05 |

The sweep writes one top-level directory with four subdirectories:

```text
<base_sweep_root>/1_very_strong
<base_sweep_root>/2_strong
<base_sweep_root>/3_medium
<base_sweep_root>/4_weak
```

Run command:

```bash
cd /workspace/NeuralOperatorRobustness2
nohup bash tools/run_ns2d_eps32_alpha10_steepest_add_loss3_allw_alt9_budget_sweep_steps25_20260525.sh > /tmp/ns2d_alt9_budget_sweep_steps25.log 2>&1 & disown
```

Monitor:

```bash
tail -f /tmp/ns2d_alt9_budget_sweep_steps25.log
```

## 2026-05-25 Brutal Warp v2 Parameter Update

Reason for this update:

The first `very_strong` 25-step run showed that `local_warp_dists` produced visibly different aligned-model-vs-solver behavior, but several global or semi-global warp methods still looked too close to identity in the alignment visualization:

- `affine_dists` showed near-zero or visually tiny warp magnitude.
- `homography_dists` also did not show obvious geometric motion.
- Some `aligned model - solver` and `model - solver` panels looked too similar by eye.

Interpretation:

The previous settings increased the allowed warp range, but the detached inner alignment was still initialized at identity and used `loss3_align_objective=l2`. If the local L2 objective is already near a weak identity optimum, increasing the maximum allowed range alone may not make the optimizer actually use that range. Therefore v2 changes three things at once:

1. Increase the allowed geometric budget substantially.
2. Lower transformation regularization weights substantially.
3. Change inner alignment objective from `l2` to `dists` for all budget presets, so the internal registration tries to match structure/texture rather than only pointwise L2.

Important consequence:

This is no longer directly comparable to the earlier completed `very_strong` run. The new run should use a new output tag/root containing `brutalwarp_v2`. Do not mix old `very_strong` outputs with the new parameter set.

### New Brutal Warp v2 Presets

| Preset | Align objective | Scattering J | Affine | Local warp | Homography | TPS | Elastic | SVF |
|---|---:|---:|---|---|---|---|---|---|
| `very_strong` | `dists` | 6 | inner=70, lr=0.10, shift=45%, rot=120 deg, scale=[0.25,4.0], reg=1e-6 | grid=24, inner=70, lr=0.10, disp=0.35, mag=1e-6, smooth=1e-4 | inner=70, lr=0.10, corner=0.45, reg=1e-6 | grid=10, inner=70, lr=0.10, disp=0.45, offset=1e-6, smooth=1e-4 | grid=32, inner=70, lr=0.10, disp=0.35, kernel=3, passes=1, mag=1e-6, smooth=1e-4 | grid=24, inner=70, lr=0.10, vel=0.32, int=8, mag=1e-6, smooth=1e-4 |
| `strong` | `dists` | 5 | inner=50, lr=0.08, shift=30%, rot=75 deg, scale=[0.333,3.0], reg=1e-5 | grid=20, inner=50, lr=0.08, disp=0.25, mag=1e-5, smooth=5e-4 | inner=50, lr=0.08, corner=0.32, reg=1e-5 | grid=8, inner=50, lr=0.08, disp=0.32, offset=1e-5, smooth=5e-4 | grid=28, inner=50, lr=0.08, disp=0.25, kernel=5, passes=1, mag=1e-5, smooth=5e-4 | grid=20, inner=50, lr=0.08, vel=0.24, int=8, mag=1e-5, smooth=5e-4 |
| `medium` | `dists` | 5 | inner=35, lr=0.08, shift=20%, rot=45 deg, scale=[0.5,2.0], reg=1e-4 | grid=16, inner=35, lr=0.08, disp=0.16, mag=1e-4, smooth=0.002 | inner=35, lr=0.08, corner=0.20, reg=1e-4 | grid=6, inner=35, lr=0.08, disp=0.20, offset=1e-4, smooth=0.002 | grid=22, inner=35, lr=0.08, disp=0.16, kernel=5, passes=1, mag=1e-4, smooth=0.002 | grid=16, inner=35, lr=0.08, vel=0.16, int=7, mag=1e-4, smooth=0.002 |
| `weak` | `dists` | 4 | inner=20, lr=0.07, shift=10%, rot=25 deg, scale=[0.741,1.35], reg=0.001 | grid=10, inner=20, lr=0.07, disp=0.08, mag=0.001, smooth=0.01 | inner=20, lr=0.07, corner=0.10, reg=0.001 | grid=5, inner=20, lr=0.07, disp=0.10, offset=0.001, smooth=0.01 | grid=18, inner=20, lr=0.07, disp=0.08, kernel=7, passes=1, mag=0.001, smooth=0.01 | grid=10, inner=20, lr=0.07, vel=0.08, int=6, mag=0.001, smooth=0.01 |

Approximate visual scale on 256x256:

- `very_strong` affine translation budget is about 115 pixels per axis.
- `very_strong` homography corner budget is about 115 pixels.
- `very_strong` TPS control displacement budget is about 115 pixels in normalized coordinates.
- `very_strong` local/elastic dense-warp component budget is about 89 pixels per axis before smoothing/interpolation.

Caveat:

These settings are intentionally aggressive for visual separation. They are useful for seeing whether the methods generate visibly different perturbations, but they are less conservative as a scientifically fair invariance budget. After identifying a visually distinct method, a smaller follow-up budget should be used for a cleaner comparison.

Updated files:

- `tools/run_ns2d_eps32_alpha10_steepest_add_loss3_allw_alt9_budget_sweep_steps25_20260525.sh`
- `tools/run_ns2d_eps32_alpha10_steepest_add_loss3_allw_alt9_strongwarp_steps25_20260525.sh`
- `tools/run_ns2d_alt9_budget_sweep_steps25_then100_upload_20260525.sh`
- `tools/plot_ns2d_eps32_alpha10_altloss_heatmaps_spectrum_loss_curves_cleanstyle_20260525.py`

## 2026-05-25 Very Very Strong / Brutal Warp v3 Update

A new preset named `very_very_strong` has been added above `very_strong`.

Default budget order is now only:

1. `very_very_strong`

The older `very_strong`, `strong`, `medium`, and `weak` presets remain defined in the code for manual override, but the default 25-then-100 launcher no longer runs them.

The default output tag/root now contains `brutalwarp_v3`, so the new runs will not be mixed with the earlier `brutalwarp_v2` configuration.

### Motivation

The first aggressive run still showed several alignment methods with weak or visually tiny warp magnitude, especially:

- `affine_dists`
- `homography_dists`
- some pointwise aligned-model difference panels

The goal of `very_very_strong` is not to be a conservative invariance budget. It is intentionally extreme, mainly to force a visually obvious difference if the method is capable of using the warp freedom.

### Very Very Strong Parameters

| Method family | `very_very_strong` budget |
|---|---|
| Inner alignment objective | `dists` |
| Scattering2D | `J=6` |
| Affine | `inner=100`, `lr=0.12`, `shift=0.65`, `rotation=180 deg`, `log_scale=ln(5)=1.6094379124`, `reg=1e-7` |
| Local dense warp | `grid=32`, `inner=100`, `lr=0.12`, `disp=0.50`, `mag=1e-7`, `smooth=1e-5` |
| Homography | `inner=100`, `lr=0.12`, `corner=0.65`, `reg=1e-7` |
| TPS | `grid=12`, `inner=100`, `lr=0.12`, `disp=0.65`, `offset=1e-7`, `smooth=1e-5` |
| Elastic | `grid=40`, `inner=100`, `lr=0.12`, `disp=0.50`, `kernel=3`, `passes=1`, `mag=1e-7`, `smooth=1e-5` |
| SVF | `grid=32`, `inner=100`, `lr=0.12`, `vel=0.45`, `int=9`, `mag=1e-7`, `smooth=1e-5` |

Approximate scale on a 256x256 field:

- Affine translation budget: about `0.65 * 255 ~= 166` pixels per axis.
- Homography corner budget: about `0.65 * 256 ~= 166` pixels.
- TPS control displacement budget: very large normalized displacement, intentionally allowing strong folding-like visual changes if the optimizer wants it.
- Local/elastic component displacement budget: normalized component bound `1.00`, roughly `128` pixels per axis before interpolation/smoothing.

### Caution

This preset is deliberately extreme. It can produce unrealistic or overly permissive alignment. Use it to make visual differences obvious, not as the final scientifically fair budget. If `very_very_strong` makes the perturbations visually separable, the next step should be to back off toward `strong` or `medium` and identify the smallest budget where the effect remains visible.

Updated files:

- `tools/run_ns2d_alt9_budget_sweep_steps25_then100_upload_20260525.sh`
- `tools/run_ns2d_eps32_alpha10_steepest_add_loss3_allw_alt9_budget_sweep_steps25_20260525.sh`
- `tools/run_ns2d_eps32_alpha10_steepest_add_loss3_allw_alt9_strongwarp_steps25_20260525.sh`
- `tools/watch_then_run_ns2d_alt9_budget_sweep_steps100_after25_20260525.sh`
- `tools/watch_and_plot_ns2d_alt9_budget_sweep_dataset0_20260525.sh`
- `tools/plot_ns2d_eps32_alpha10_altloss_heatmaps_spectrum_loss_curves_cleanstyle_20260525.py`



## 2026-05-25 Very Very Strong Only Run Update

The active default run plan was narrowed to only `very_very_strong`.

The 25-then-100 launcher still runs in this order:

1. `steps25` with `very_very_strong` only.
2. `steps100` with `very_very_strong` only.

The other presets (`very_strong`, `strong`, `medium`, `weak`) are still available if explicitly requested through `BUDGET_PRESETS`, but they are not part of the default run.

Updated defaults:

- `tools/run_ns2d_alt9_budget_sweep_steps25_then100_upload_20260525.sh`: `BUDGET_PRESETS=very_very_strong`
- `tools/run_ns2d_eps32_alpha10_steepest_add_loss3_allw_alt9_budget_sweep_steps25_20260525.sh`: `BUDGET_PRESETS=very_very_strong`
- `tools/watch_then_run_ns2d_alt9_budget_sweep_steps100_after25_20260525.sh`: `BUDGET_PRESETS=very_very_strong`
- `tools/watch_and_plot_ns2d_alt9_budget_sweep_dataset0_20260525.sh`: `BUDGET_PRESET_DIRS=1_very_very_strong`
