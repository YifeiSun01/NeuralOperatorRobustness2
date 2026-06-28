# Jacobian Projection Theory And References - 2026-06-08

## Core Objects

For a full Burgers local Jacobian `J` with input/output resolution `1024`, two coarse proxies were discussed.

Block projection uses an orthonormal piecewise-constant basis `P`:

\[
J_{\mathrm{block}} = P^T J P.
\]

For downsample factor `d`, each column of `P` has entries `1/\sqrt d` on one adjacent block. Therefore each coarse matrix entry is a normalized block sum:

\[
(J_{\mathrm{block}})_{ab}=\frac{1}{d}\sum_{i\in B_a}\sum_{j\in B_b}J_{ij}.
\]

Stride uses point sampling:

\[
J_{\mathrm{stride}} = S J S^T,
\]

where `S` has one `1` in each row at the sampled grid point. For factor `d=4`, `S` selects every fourth point. Stride is a submatrix, not an orthogonal Galerkin projection.

## What The Projection Residual Means

For a full top singular pair

\[
Jv_k = \sigma_k u_k,
\]

the amount of the vector not represented by the coarse block subspace is measured by

\[
\|(I-PP^T)v_k\|^2,\qquad \|(I-PP^T)u_k\|^2.
\]

The reported `top1 block residual` is

\[
\frac{1}{2}\left(\|(I-PP^T)u_1\|^2 + \|(I-PP^T)v_1\|^2\right).
\]

Small values mean the full top singular vectors are mostly in the coarse block-constant subspace; large values mean the coarse proxy is discarding part of the dominant singular structure.

A simple magnitude heuristic is: if the left and right vectors retain lengths `a` and `b` in the projected space, then the observed singular value can scale like

\[
\sigma_{\mathrm{coarse}} \approx ab\,\sigma_{\mathrm{full}}.
\]

This explains how a modest loss on both sides can become a larger singular-value loss. For example, `a=b=0.85` gives `ab=0.7225`, matching the size of the worst 256 block-projection underestimates observed for some error Jacobians.

## Rayleigh-Ritz / Galerkin Principle

Rayleigh-Ritz/Galerkin says: if the selected trial subspace contains, or accurately approximates, the invariant/singular subspace of interest, then the projected operator's Ritz values/vectors approximate the full operator's dominant spectral structure.

For SVD-style projection, the relevant small matrix is

\[
B = Q_L^T J Q_R.
\]

After computing

\[
B\hat v = \hat\sigma\hat u,
\]

the lifted vectors are

\[
u_{\mathrm{ritz}} = Q_L\hat u,\qquad v_{\mathrm{ritz}} = Q_R\hat v.
\]

If the full singular vectors are already in `range(Q_L)` and `range(Q_R)`, this can recover the full singular triplet exactly. If the projection residual is small, it can be a good approximation.

Reference: Knyazev and Argentati, *Rayleigh-Ritz majorization error bounds with applications to FEM*, arXiv: https://arxiv.org/abs/math/0701784. The paper describes the Rayleigh-Ritz method as finding Ritz values of the Rayleigh quotient on a trial subspace and relates errors to subspace angles.

## Davis-Kahan And Wedin sin-theta Interpretation

Davis-Kahan is the classic Hermitian/eigen-subspace perturbation result. Wedin is the SVD/singular-subspace analogue. Their practical message here is:

\[
\sin\Theta \lesssim \frac{\mathrm{residual}}{\mathrm{gap}}.
\]

So singular subspaces are stable when:

- the residual left after projection/perturbation is small;
- the target singular value cluster is separated from the rest by a decent gap.

They become unstable when:

- the projected vectors miss important components of the full singular vectors;
- the error Jacobian has clustered singular values, so the gap is small.

Reference: Davis and Kahan, *The Rotation of Eigenvectors by a Perturbation. III*, SIAM J. Numer. Anal., 1970: https://epubs.siam.org/doi/10.1137/0707001.

Reference: Wedin, *Perturbation bounds in connection with singular value decomposition*, 1972: https://cir.nii.ac.jp/crid/1360298342261921536. Modern summaries call this Wedin's sin-theta theorem for singular subspaces.

## How This Explains The Burgers Phenomenon

Observed solver/model Jacobians have small projection residuals, so Rayleigh-Ritz/Galerkin is in its good regime.

Observed error Jacobians have larger projection residuals. The error matrix is

\[
J_{\mathrm{err}} = J_{\mathrm{model}} - J_{\mathrm{solver}}.
\]

If

\[
J_{\mathrm{model}}\approx A+\Delta_m,\qquad J_{\mathrm{solver}}\approx A+\Delta_s,
\]

then

\[
J_{\mathrm{err}}\approx \Delta_m-\Delta_s.
\]

The shared smooth low-frequency structure `A` cancels, leaving more localized/mid-high-frequency residual structure. That pushes singular-vector mass outside the 256 block-constant space and makes Wedin-style residual/gap bounds worse.

## StablePDENet Reference

Reference paper: Huang, Ma, Wang, Xiang, *StablePDENet: Enhancing Stability of Operator Learning for Solving Differential Equations*, arXiv:2601.06472, posted 2026-01-10: https://arxiv.org/abs/2601.06472.

Observed from the arXiv abstract: StablePDENet proposes robust self-supervised neural-operator training against worst-case input perturbations and emphasizes stability under normal and adversarial inputs.

Connection to this repository: the paper's stability narrative motivates looking at physics/PDE residual losses and Jacobian/singular-value behavior as stability diagnostics. In our local Burgers work, the analogous object is the local Jacobian of model/solver/error maps and the spectral behavior of `J_model - J_solver`.

Important caution: the exact StablePDENet Jacobian construction, example dimension, and SVD implementation should be checked directly from the PDF before being used as a factual comparison. The local conclusion here does not depend on their exact implementation details.
