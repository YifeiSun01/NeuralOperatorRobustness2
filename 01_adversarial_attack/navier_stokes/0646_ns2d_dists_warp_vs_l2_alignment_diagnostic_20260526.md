# NS2D DISTS-Warp vs L2 Alignment Diagnostic

Date: 2026-05-26

## Context

In the NS2D final-state loss3 all-W experiments, some explicit warp methods produced a visually strange result:

after applying the learned/optimized warp, the aligned model output looked less similar to the solver output in the heatmap, even though the warp was supposed to align the model output to the solver output.

This note records why this happened and what to test next.

## Key Observation

The observation is real:

**Some warped/aligned outputs are worse under pointwise L2, even though they improve under DISTS.**

The current `+ DISTS` warp methods do not optimize raw pointwise `model - solver` L2. They optimize a DISTS feature/texture/structure distance.

The current inner warp estimation objective is:

$$
\theta^* = \arg\min_{\theta}
\left[
\mathrm{DISTS}(T_{\theta}(W), Q) + \lambda R(\theta)
\right]
$$

It is not:

$$
\theta^* = \arg\min_{\theta}
\left\|T_{\theta}(W)-Q\right\|_2^2
$$

Therefore the following can happen:

- DISTS gets smaller, meaning the fields are closer in DISTS feature space.
- Pointwise L2 gets larger, meaning the raw pixel/grid-point difference gets worse.
- The heatmap can look less similar after warp, because the heatmap is closer to pointwise visual/L2 comparison than to DISTS feature-space comparison.

## Strong/25 Dataset 0 Evidence

From `alignment_before_after_diff_summary.csv` in the strong/25 run:

| Method | Raw L2: model-solver | Aligned L2: aligned model-solver | DISTS raw | DISTS aligned | Interpretation |
|---|---:|---:|---:|---:|---|
| Affine + DISTS | 388.19 | 510.03 | 0.3651 | 0.3567 | L2 worse, DISTS better |
| Local warp + DISTS | 304.63 | 301.50 | 0.3849 | 0.2789 | L2 slightly better, DISTS much better |
| Homography + DISTS | 470.00 | 401.88 | 0.4299 | 0.3664 | both better |
| TPS + DISTS | 306.90 | 292.49 | 0.3569 | 0.3092 | both better |
| Elastic + DISTS | 247.55 | 304.22 | 0.3458 | 0.2677 | L2 worse, DISTS better |
| SVF + DISTS | 182.43 | 233.91 | 0.3034 | 0.2060 | L2 worse, DISTS better |

The important conclusion is:

**The current `+ DISTS` aligned model is not guaranteed to reduce pointwise L2. It is only intended to reduce the DISTS objective.**

## Why It Looks Strange

### 1. DISTS is not pointwise L2

DISTS is a deep feature / texture / structure distance. It compares image-like feature statistics, not exact physical grid-point equality.

So it can decide that a warped field is more similar in structure or texture even if the raw pointwise difference becomes larger.

### 2. Large warp budget and weak regularization can amplify the issue

In the strong and very-very-strong runs, the allowed warp budgets were intentionally large and the regularization was small. That lets the optimizer find aggressive deformations that are useful for DISTS but not necessarily visually intuitive in raw field space.

### 3. NS2D fields are not natural images

DISTS was designed for perceptual image similarity. For NS2D vorticity / physical fields, DISTS similarity is not always the same as physical closeness.

### 4. The outer attack maximizes the selected loss

There are two nested processes:

1. Inner alignment: choose a warp that minimizes the selected alignment objective.
2. Outer adversarial attack: choose a perturbation that maximizes the final loss after alignment.

So after the adversarial perturbation is found, the result is already adversarial against that metric. Weird-looking alignment is not impossible.

## Correct Interpretation

The current `+ DISTS` warp results do not necessarily mean the warp implementation is wrong.

They show that:

$$
\mathrm{DISTS}(T_{\theta^*}(W), Q)
<
\mathrm{DISTS}(W,Q)
$$

can hold while:

$$
\left\|T_{\theta^*}(W)-Q\right\|_2
>
\left\|W-Q\right\|_2
$$

That is a metric mismatch, not automatically a coordinate bug.

## If We Want Warp to Look Better Under Pointwise Heatmaps

If the intended diagnostic is that the warped model output should look closer to the solver output in pointwise heatmaps, then the inner alignment objective should include L2.

A hybrid objective would be:

$$
D_{align}(W,Q)
=
\lambda_1 \mathrm{DISTS}(T_{\theta}W,Q)
+
\lambda_2 \left\|T_{\theta}W-Q\right\|_2^2
+
\lambda_3 R(\theta)
$$

A direct pointwise objective would be:

$$
\theta^* = \arg\min_{\theta}
\left[
\left\|T_{\theta}(W)-Q\right\|_2^2 + \lambda R(\theta)
\right]
$$

This should make the aligned model more consistent with `model - solver` heatmaps.

## Code Change Added After This Diagnosis

The code now supports six explicit `+ L2` warp metrics:

- `affine_l2`
- `local_warp_l2`
- `homography_l2`
- `tps_l2`
- `elastic_l2`
- `svf_l2`

For these metrics, both the inner alignment objective and the final content loss use normalized per-pixel L2/MSE:

$$
\theta^* = \arg\min_{\theta \in \Theta}
\left[
\frac{1}{N}\left\|T_{\theta}(\widetilde W)-\widetilde Q\right\|_2^2
+ \lambda R(\theta)
\right]
$$

and

$$
L_{warp+L2}(W,Q)=
\frac{1}{N}\left\|T_{\theta^*}(\widetilde W)-\widetilde Q\right\|_2^2
+ \lambda R(\theta^*)
$$

Here:

- `W` is the model final state.
- `Q` is the solver final state.
- `T_theta` is the chosen warp family.
- `theta` is the warp parameter.
- `R(theta)` is the warp regularization penalty.
- `tilde W`, `tilde Q` are pair-normalized fields.
- `N` is the number of pixels/grid points.

## Recommended Next Checks

### Check 1: Put both L2 and DISTS numbers on alignment figures

Each alignment figure should display:

- raw L2: `model - solver`
- aligned L2: `aligned model - solver`
- raw DISTS
- aligned DISTS

This prevents confusion between feature-space improvement and pointwise improvement.

### Check 2: Synthetic sanity tests

For each warp family, create a known artificial transform:

- translate a field
- rotate a field
- apply known affine transform
- apply known local warp

Then test whether the corresponding alignment method can recover a warp that reduces L2.

This separates two possible explanations:

1. The objective is doing what it was told to do, but DISTS and L2 disagree.
2. The warp coordinate convention or implementation has a real bug.

## Short Conclusion

The strange-looking `+ DISTS` warp figures are mostly explained by objective mismatch:

**DISTS alignment can improve DISTS while worsening pointwise L2.**

For physically interpretable NS2D heatmap alignment, the new `*_l2` warp metrics should be tested next.
