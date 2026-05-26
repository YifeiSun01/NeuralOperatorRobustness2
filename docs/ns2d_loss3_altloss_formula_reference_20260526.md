# NS2D Loss3 Alternative Loss Formula Reference

Date: 2026-05-26

This document records the actual formulas for the NS2D `loss3` alternatives used in the epsilon=32, alpha=10, `steepest_add`, all-W final-state experiments.

The key point is that the current explicit-warp methods use **DISTS after warp**, not ordinary pointwise L2 after warp.

## 1. Objects

Let:

- `x`: clean initial condition.
- `delta`: adversarial perturbation.
- `x_adv = x + delta`.
- `M(x_adv)`: FNO model final-state output.
- `S(x_adv)`: solver final-state output.
- `W = M(x_adv)`.
- `Q = S(x_adv)`.

The attack maximizes the selected final-state discrepancy:

$$
\max_{\delta} L_3(W,Q)
\quad \text{subject to} \quad
\|\delta\|_p \le \epsilon
$$

Current experiment settings:

- `epsilon = 32`
- `alpha = 10`
- `optimizer = steepest_add`
- `mode = all_w`
- `batch size = 10`

## 2. Pair Normalization

For image-style losses, the code first normalizes `W` and `Q` together:

$$
m = \operatorname{detach}\left(\min(W,Q)\right)
$$

$$
M = \operatorname{detach}\left(\max(W,Q)\right)
$$

$$
\widetilde W = \frac{W-m}{\max(M-m,\varepsilon_m)}
$$

$$
\widetilde Q = \frac{Q-m}{\max(M-m,\varepsilon_m)}
$$

So DISTS, MS-SSIM, Scattering2D, and DISTS-based warp losses are computed on normalized image fields, not directly on raw physical values.

## 3. Baseline Loss3 / qnorm

The original baseline loss3 is pointwise final-state norm:

$$
L_{qnorm}(W,Q)=\|\operatorname{vec}(W-Q)\|_q
$$

For the current `q_order = 2` case:

$$
L_{qnorm}(W,Q)=\|W-Q\|_2
$$

This is ordinary pointwise L2. It compares every grid point with the same coordinate in the other field.

## 4. DISTS

Package: `piq`

The implemented loss is:

$$
L_{DISTS}(W,Q)=DISTS(\widetilde W,\widetilde Q)
$$

DISTS is not pointwise L2. It compares deep feature structure and texture. Conceptually:

$$
DISTS(x,y) \approx 1 - \sum_l \left[\alpha_l S_l\left(\phi_l(x),\phi_l(y)\right) + \beta_l T_l\left(\phi_l(x),\phi_l(y)\right)\right]
$$

Here `phi_l` is a fixed CNN feature map, `S_l` is a structure-like similarity term, and `T_l` is a texture/statistics-like similarity term.

Meaning:

- Larger means more dissimilar.
- It is feature/texture/structure based.
- It is not raw pixelwise L2.
- It can improve while raw pointwise L2 gets worse.

## 5. MS-SSIM Distance

Package: `pytorch-msssim`

MS-SSIM is originally a similarity score. The implementation converts it to a distance:

$$
L_{MS\text{-}SSIM}(W,Q)=1-MS\text{-}SSIM(\widetilde W,\widetilde Q)
$$

Meaning:

- Larger means more dissimilar.
- It compares multi-scale local structural statistics.
- It has no explicit spatial warp.
- Pixel coordinates stay fixed.

## 6. Scattering2D Feature L2

Package: `kymatio`

Let `Scat_J` be the 2D scattering transform. The implemented loss is:

$$
L_{Scattering2D}(W,Q)=\frac{1}{N}\left\|Scat_J(\widetilde W)-Scat_J(\widetilde Q)\right\|_2^2
$$

Current setting uses `J = 3` unless overridden.

Meaning:

- Larger means more dissimilar.
- It compares fixed wavelet/scattering features.
- It has no explicit warp.
- Pixel coordinates stay fixed.
- It is more stable to small shifts/deformations than raw L2.

## 7. General Explicit-Warp Template

For explicit warp methods, a spatial transform is first estimated:

$$
\theta^* = \arg\min_{\theta \in \Theta}
\left[
A\left(T_{\theta}(\widetilde W),\widetilde Q\right)+\lambda R(\theta)
\right]
$$

Then the final loss is:

$$
L(W,Q)=DISTS\left(T_{\theta^*}(\widetilde W),\widetilde Q\right)+\lambda R(\theta^*)
$$

The inner alignment objective `A` can be either DISTS or L2. In the current very-very-strong run:

- `loss3_align_objective = dists`

So the inner alignment minimizes:

$$
A\left(T_{\theta}(\widetilde W),\widetilde Q\right)
= DISTS\left(T_{\theta}(\widetilde W),\widetilde Q\right)
$$

not raw pointwise L2.

Therefore it is possible that:

$$
DISTS\left(T_{\theta^*}(\widetilde W),\widetilde Q\right)
< DISTS\left(\widetilde W,\widetilde Q\right)
$$

but also:

$$
\left\|T_{\theta^*}(W)-Q\right\|_2
>
\left\|W-Q\right\|_2
$$

That is exactly what happened for Homography on sample 0.

## 8. Affine + DISTS

Backend: `kornia.warp_affine`

Transform:

$$
T_{\theta}=Affine(t_x,t_y,r,s)
$$

where `t_x,t_y` are translation, `r` is rotation, and `s` is isotropic scale.

Inner optimization:

$$
\theta^* = \arg\min_{\theta}
\left[
DISTS\left(Affine_{\theta}(\widetilde W),\widetilde Q\right)
+\lambda_a R_a(\theta)
\right]
$$

Final loss:

$$
L_{Affine+DISTS}(W,Q)
=
DISTS\left(Affine_{\theta^*}(\widetilde W),\widetilde Q\right)
+
\lambda_a R_a(\theta^*)
$$

Regularization:

$$
R_a(\theta)=t_x^2+t_y^2+r^2+(\log s)^2
$$

Current very-very-strong budget:

- max shift ratio: `0.65`
- max angle: `180 degrees`
- max log scale: `log(5)`
- regularization weight: `1e-7`
- inner steps: `100`
- inner lr: `0.12`

Important: this is warp + DISTS, not warp + raw L2.

## 9. Local Dense Warp + DISTS

Backend: `monai.networks.blocks.Warp`

Transform:

$$
T_u(p)=p+u(p)
$$

where `u` is a dense displacement field upsampled from a low-resolution grid.

Inner optimization:

$$
u^* = \arg\min_u
\left[
DISTS\left(\widetilde W \circ (Id+u),\widetilde Q\right)
+
\lambda_m\|u\|_2^2
+
\lambda_s\|\nabla u\|_2^2
\right]
$$

Final loss:

$$
L_{LocalWarp+DISTS}(W,Q)
=
DISTS\left(\widetilde W \circ (Id+u^*),\widetilde Q\right)
+
\lambda_m\|u^*\|_2^2
+
\lambda_s\|\nabla u^*\|_2^2
$$

Current very-very-strong budget:

- local grid size: `8`
- max displacement ratio: `0.5`
- magnitude regularization weight: `1e-7`
- smoothness weight: `1e-5`
- inner steps: `100`
- inner lr: `0.12`

Important: this is warp + DISTS, not warp + raw L2.

## 10. Homography + DISTS

Backend: `kornia.get_perspective_transform` + `kornia.warp_perspective`

Transform:

$$
T_H(p)=Hp
$$

where `H` is a projective homography determined by four moved image corners.

Inner optimization:

$$
H^* = \arg\min_H
\left[
DISTS\left(H(\widetilde W),\widetilde Q\right)+\lambda_h R_h(H)
\right]
$$

Final loss:

$$
L_{Homography+DISTS}(W,Q)
=
DISTS\left(H^*(\widetilde W),\widetilde Q\right)+\lambda_h R_h(H^*)
$$

Regularization:

$$
R_h(H)=\operatorname{mean}\left[
\left(\frac{\Delta x_{corner}}{width}\right)^2
+
\left(\frac{\Delta y_{corner}}{height}\right)^2
\right]
$$

Current very-very-strong budget:

- max corner displacement ratio: `0.65`
- regularization weight: `1e-7`
- inner steps: `100`
- inner lr: `0.12`

Current observed sample 0 behavior:

$$
\|W-Q\|_2 = 177.85
$$

$$
\|H^*(W)-Q\|_2 = 310.76
$$

$$
DISTS(\widetilde W,\widetilde Q)=0.30985
$$

$$
DISTS(H^*(\widetilde W),\widetilde Q)=0.29654
$$

So Homography improved DISTS but worsened raw W-Q L2.

## 11. TPS + DISTS

Backend: `kornia.get_tps_transform` + `kornia.warp_image_tps`

Transform:

$$
T_{TPS}(p)=p+u_{TPS}(p)
$$

where `u_TPS` is controlled by a sparse control grid.

Inner optimization:

$$
\theta^* = \arg\min_{\theta}
\left[
DISTS\left(TPS_{\theta}(\widetilde W),\widetilde Q\right)
+
\lambda_o\|o_{\theta}\|_2^2
+
\lambda_s\|\nabla o_{\theta}\|_2^2
\right]
$$

Final loss:

$$
L_{TPS+DISTS}(W,Q)
=
DISTS\left(TPS_{\theta^*}(\widetilde W),\widetilde Q\right)
+
\lambda_o\|o_{\theta^*}\|_2^2
+
\lambda_s\|\nabla o_{\theta^*}\|_2^2
$$

Current very-very-strong budget:

- TPS grid size: `4`
- max displacement ratio: `0.65`
- offset weight: `1e-7`
- smoothness weight: `1e-5`
- inner steps: `100`
- inner lr: `0.12`

Important: this is warp + DISTS, not warp + raw L2.

## 12. Elastic + DISTS

Backend: `kornia.elastic_transform2d`

Transform:

$$
T_e(p)=p+e(p)
$$

where `e` is a smoothed displacement/noise field.

Inner optimization:

$$
e^* = \arg\min_e
\left[
DISTS\left(Elastic_e(\widetilde W),\widetilde Q\right)
+
\lambda_m\|e\|_2^2
+
\lambda_s\|\nabla e\|_2^2
\right]
$$

Final loss:

$$
L_{Elastic+DISTS}(W,Q)
=
DISTS\left(Elastic_{e^*}(\widetilde W),\widetilde Q\right)
+
\lambda_m\|e^*\|_2^2
+
\lambda_s\|\nabla e^*\|_2^2
$$

Current very-very-strong budget:

- elastic grid size: `16`
- max displacement ratio: `0.5`
- smooth kernel: `9`
- smooth passes: `2`
- magnitude weight: `1e-7`
- smoothness weight: `1e-5`
- inner steps: `100`
- inner lr: `0.12`

Important: this is warp + DISTS, not warp + raw L2.

## 13. SVF + DISTS

Backend: `monai.DVF2DDF` + `monai.networks.blocks.Warp`

SVF means stationary velocity field. It optimizes a velocity field and integrates it into a displacement field.

Velocity integration:

$$
\nu \mapsto d_{\nu}=Integrate(\nu)
$$

Warp:

$$
T_{d_{\nu}}(p)=p+d_{\nu}(p)
$$

Inner optimization:

$$
\nu^* = \arg\min_{\nu}
\left[
DISTS\left(\widetilde W \circ (Id+d_{\nu}),\widetilde Q\right)
+
\lambda_m\|\nu\|_2^2
+
\lambda_s\|\nabla \nu\|_2^2
\right]
$$

Final loss:

$$
L_{SVF+DISTS}(W,Q)
=
DISTS\left(\widetilde W \circ (Id+d_{\nu^*}),\widetilde Q\right)
+
\lambda_m\|\nu^*\|_2^2
+
\lambda_s\|\nabla \nu^*\|_2^2
$$

Current very-very-strong budget:

- SVF grid size: `8`
- max velocity ratio: `0.45`
- integration steps: `9`
- magnitude weight: `1e-7`
- smoothness weight: `1e-5`
- inner steps: `100`
- inner lr: `0.12`

Important: this is warp + DISTS, not warp + raw L2.

## 14. If We Want Raw Pointwise Alignment

If the intended behavior is to make the warped model output closer to the solver output in pointwise physical space, the inner objective should be raw L2:

$$
A(T(W),Q)=\|T(W)-Q\|_2^2
$$

or a mixed objective:

$$
A(T(W),Q)=DISTS(T(W),Q)+\beta\|T(W)-Q\|_2^2
$$

Then the alignment visualization would be more consistent with the `model - solver` heatmap and raw W-Q L2 diagnostic.

## 15. Summary

The current explicit warp losses are:

$$
\text{bounded warp} + DISTS + \text{small regularization}
$$

They are not:

$$
\text{bounded warp} + \text{ordinary pointwise L2}
$$

unless `loss3_align_objective` is changed to `l2` or the formula is modified to include a raw L2 term.
