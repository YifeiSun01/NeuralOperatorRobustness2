# Loss3 Delta 3D Surface Visualizations - 2026-05-18

## Scope

Generated 3D surface plots from saved `trajectory_samples.npz` files. This is
post-processing only: no optimizer, model, or solver was rerun.

New script:

```text
/workspace/NeuralOperatorRobustness2/tools/plot_delta_3d_surfaces.py
```

Command used:

```bash
./adv_robust/bin/python tools/plot_delta_3d_surfaces.py --space-stride 4 --step-stride 1
```

## Plot Meaning

Each 3D surface figure is the 3D version of the delta heatmap:

- X axis: `space index`
- Y axis: optimizer step `k`, from `0` to `100`
- Z axis: perturbation value `delta[k, space]`
- Color: signed delta value, using a symmetric coolwarm scale

The source trajectories contain all 101 steps. Observed example shape:

```text
k: (101,), first [0,1,2,3,4], last [96,97,98,99,100]
delta: (101, 4, 1024, 1)
dataset_index: [0, 7, 40, 47]
```

The generated plots use `--space-stride 4`, so the surface draws every fourth
space point for readability and runtime. The plotted X coordinates are still the
true space indices.

## Generated Outputs

Output directory under each completed run:

```text
figures/delta_surfaces_3d_20260518/
```

Observed counts:

| Run | Status | 3D surface PNGs |
|---|---|---:|
| `p=1,q=1` | completed | 28 |
| `p=1,q=2` | completed | 28 |
| `p=1,q=inf` | completed | 28 |
| `p=2,q=2` | completed | 44 |
| `p=2,q=1` | run_started, skipped | 0 |

Total generated: `128` PNG files.

Example paths:

- `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p2_q2/figures/delta_surfaces_3d_20260518/raw_add/delta_surface3d_raw_add_sample_000.png`
- `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p2_q2/figures/delta_surfaces_3d_20260518/steepest_replace/delta_surface3d_steepest_replace_sample_000.png`
- `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p1_qinf/figures/delta_surfaces_3d_20260518/steepest_add/delta_surface3d_steepest_add_sample_000.png`

Preservation note: existing delta heatmaps and previous figures were not deleted
or overwritten. These 3D surfaces were written to a new figure directory.

## Reading Guidance

Use these figures to inspect temporal perturbation geometry:

- sudden vertical walls or ridges: abrupt update changes or immediate boundary
  replacement behavior;
- persistent ripples along the space axis: high-frequency perturbation content;
- smooth broad surfaces: lower-frequency, smoother perturbation evolution;
- flat early region followed by growth: gradual additive PGD-style expansion.

These plots are visual diagnostics. Quantitative comparison should still use
saved metrics such as loss curves, roughness/spectrum CSVs, and final-delta
similarity matrices.
