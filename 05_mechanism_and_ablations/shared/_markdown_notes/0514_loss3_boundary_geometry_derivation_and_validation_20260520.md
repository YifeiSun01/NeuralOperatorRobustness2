# Loss 3 Boundary Geometry: Derivations and Validation Status

Date: 2026-05-20

This note clarifies the geometry behind several Loss 3 observations: boundary stationarity, why additive updates have angular inertia, why LP-steepest norm growth is only approximately linear, and what can currently be validated from existing artifacts.

## 1. What does `(I - u u^T) grad L` mean?

For the p=2 constraint

\[
\|\delta\|_2 \le \epsilon,
\]

a boundary point satisfies \(\|\delta\|_2=\epsilon\). Define

\[
u = \frac{\delta}{\|\delta\|_2}.
\]

Here \(u\) is the unit radial direction: it points from the origin to the current perturbation.

\(I\) is the identity matrix. The matrix

\[
u u^T
\]

is the projection onto the radial line spanned by \(u\). For any vector \(v\),

\[
u u^T v = u (u^T v)
\]

is the component of \(v\) parallel to \(u\).

Therefore

\[
(I-u u^T)v
\]

is the component of \(v\) orthogonal to \(u\). At a point on the L2 sphere, the orthogonal-to-u directions are exactly tangent directions along the sphere.

So

\[
(I-u u^T)\nabla L(\delta)
\]

means: the part of the loss gradient that points along the boundary surface instead of radially outward.

## 2. Why does nonzero tangent gradient mean not converged?

At the L2 boundary, a small feasible first-order movement along the sphere must satisfy

\[
u^T v = 0.
\]

That is, \(v\) is tangent to the sphere. The first-order loss change is

\[
D L(\delta)[v] = \nabla L(\delta)^T v.
\]

Let

\[
g_T=(I-u u^T)\nabla L(\delta).
\]

If \(g_T\ne 0\), choose \(v=g_T\). This is a valid tangent direction, and

\[
\nabla L(\delta)^T g_T = \|g_T\|_2^2 > 0.
\]

So loss can still increase while staying on the boundary. Therefore boundary arrival is not convergence.

The first-order boundary-stationary condition for p=2 is

\[
(I-u u^T)\nabla L(\delta)=0,
\]

which means the gradient is parallel to \(u\). Equivalently, the gradient has no tangent component left.

This is also the KKT condition. Maximizing \(L(\delta)\) subject to \(\|\delta\|_2^2=\epsilon^2\) gives

\[
\nabla L(\delta)=\lambda \delta.
\]

So the gradient must point radially, not tangentially.

## 3. Why LP-steepest additive norm growth can look linear, but is not guaranteed

For p=2, let the current perturbation be

\[
\delta_k = r_k u_k, \quad \|u_k\|_2=1,
\]

and let the LP-steepest update direction be unit norm:

\[
\|d_k\|_2=1.
\]

Before projection is active,

\[
\delta_{k+1}=\delta_k+\alpha d_k.
\]

Then

\[
r_{k+1}^2=\|r_k u_k+\alpha d_k\|_2^2
= r_k^2 + 2\alpha r_k \cos\theta_k + \alpha^2,
\]

where

\[
\cos\theta_k = u_k^T d_k.
\]

So the radius increment is approximately

\[
r_{k+1}-r_k \approx \alpha \cos\theta_k
\]

when \(\alpha\) is small relative to \(r_k\).

Therefore LP-steepest is not mathematically guaranteed to grow exactly linearly. It looks linear when:

1. each step has controlled unit size, and
2. \(d_k\) keeps a reasonably stable positive radial component, i.e. \(\cos\theta_k\) does not fluctuate too much.

This is why the observation should be stated as empirical: LP-steepest usually grows much straighter than raw PGD in the checked p2q2/p1q2/p2q1 settings, but q=inf can break the clean pattern.

For raw PGD,

\[
\delta_{k+1}=\delta_k+\alpha g_k,
\]

so

\[
r_{k+1}^2=r_k^2+2\alpha r_k\|g_k\|_2\cos\theta_k + \alpha^2\|g_k\|_2^2.
\]

Now radial progress depends on both \(\|g_k\|\) and the angle. This explains why raw PGD is more curved and sample-dependent.

## 4. Why additive methods have angular inertia

Assume p=2 and the perturbation is already on the boundary:

\[
\delta_k=\epsilon u_k,
\quad \|u_k\|_2=1.
\]

An additive step proposes

\[
\tilde\delta_{k+1}=\epsilon u_k + \alpha d_k.
\]

Projection back to the L2 sphere keeps the direction

\[
u_{k+1}=\frac{u_k+\eta d_k}{\|u_k+\eta d_k\|_2},
\quad \eta=\frac{\alpha}{\epsilon}.
\]

For small \(\eta\), a first-order expansion gives

\[
u_{k+1} \approx u_k + \eta\left(d_k-(u_k^T d_k)u_k\right).
\]

The term

\[
d_k-(u_k^T d_k)u_k
\]

is exactly the tangent component of the update direction. Its norm is at most 1. Therefore the angular change is approximately

\[
\angle(u_{k+1},u_k) \approx \eta\|d_{k,T}\|_2
\le \frac{\alpha}{\epsilon}.
\]

This is the meaning of the earlier claim: when \(\alpha\ll\epsilon\), additive methods can only rotate by a small angle per step after reaching the boundary.

## 5. Why replacement can rotate much more

Replacement proposes

\[
\delta_{k+1}=\Pi_{p,\epsilon}(\epsilon d_k).
\]

If \(d_k\) is already p-unit, then

\[
\delta_{k+1}=\epsilon d_k.
\]

For p=2, the new unit direction is simply

\[
u_{k+1}=d_k.
\]

The angle is

\[
\angle(u_{k+1},u_k)=\arccos(u_k^T d_k).
\]

There is no \(\alpha/\epsilon\) small factor. The method can jump to any boundary direction selected by the current gradient. That is why replacement/GPI can show large per-step angular motion.

## 6. What has already been validated from existing artifacts?

### Boundary motion and angular inertia

Observed from `docs/loss3_gpi_mechanism_hypothesis_probe_20260520.md`:

- In p2q2, replacement hits the 99% boundary at mean step `1`.
- Raw add hits at mean step `49.2`; steepest add hits at mean step `13.45`.
- In p2q2, replacement mean post-boundary angular motion is about `30.68 deg`.
- Raw add is about `0.246 deg`; steepest add is about `0.612 deg`.
- Replacement mean post-boundary loss gain is `3.412`.

Inference: this strongly supports the mechanism that replacement is fast because it removes radial travel and then performs aggressive boundary-direction search.

### Angle motion is not sufficient

Observed:

- Across all P/Q settings, Pearson correlation between post-boundary gain and post-boundary angle motion is only about `0.041`.

Inference: turning hard is not enough. The new direction must also be useful for the current loss landscape. This explains p1qinf: replacement can rotate a lot but still have weak or negative gain.

### Final delta similarity / dominant-direction hypothesis

Observed:

- In the baseline p2q2 giftrace, replacement's mean cosine to final delta is about `0.887` at step 5, `0.984` at step 10, and `0.995` at step 20.
- In the p2q2 eps8/alpha0.3 direction-proposal trajectory, objective-gradient replacement has about `0.788` at step 5, `0.966` at step 10, and `0.988` at step 20.
- p2q2 final-delta similarity rollups show exact `raw_replace = steepest_replace`, high raw-add/steepest-add similarity, and moderate-to-high broad/spectral similarity between additive and replacement families.

Inference: existing data supports a practical dominant-direction story in p2q2: replacement finds a final-like direction very early. However, this does not prove a true spectral dominant mode of the Loss 3 Jacobian/Hessian. That stronger claim requires a GPU diagnostic that computes local Jacobian/Hessian spectrum or randomized leading-mode probes.

### P/Q geometry and spike behavior

Observed from concentration metrics:

- In p1qinf, steepest methods have peakiness around `30` and top-1 energy fraction around `0.86-0.96`.
- In p2q2, methods have peakiness around `3.6-4.1` and top-1 energy fraction around `0.014-0.018`.
- In p1q2, steepest methods are similarly concentrated/spike-like, while raw methods are much smoother.

Inference: this strongly validates the geometry explanation. p=1 steepest directions are sparse extreme points, and q=inf focuses the output residual on extreme coordinates, so spike-like perturbations are expected.

## 7. What is not yet fully verified?

The following remain plausible mechanisms, not fully proven facts:

1. True dominant spectral mode: needs local Jacobian/Hessian or randomized power-spectrum diagnostics.
2. Exact local-linear predicted gain: needs recording \(\nabla L^T \Delta\delta\) versus actual \(L(\delta_{k+1})-L(\delta_k)\).
3. Boundary tangent KKT residual over time: for p2, directly record \(\|(I-u u^T)\nabla L\|/\|\nabla L\|\). This would show whether methods are actually reducing tangent gradient after boundary arrival.

The best next GPU diagnostic is therefore:

\[
\text{tangent residual}_k = \frac{\|(I-u_k u_k^T)\nabla L(\delta_k)\|_2}{\|\nabla L(\delta_k)\|_2}.
\]

If replacement/GPI reduces this residual quickly while additive methods reduce it slowly, that would directly verify the boundary-direction optimization explanation.
