# Loss3 replace/add mechanism final conclusion - 2026-06-23

## Final conclusion

The current mechanism evidence supports the following interpretation:

\[
\text{Burgers `steepest_replace` works because it reaches a relatively broad, forgiving, optimizer-relevant high-loss basin.}
\]

\[
\text{NS2D `steepest_replace` fails because it lands outside the high-loss basin reached by `steepest_add`.}
\]

This should not be stated as:

- Burgers has high loss over the whole epsilon boundary.
- Burgers is purely linear, so replace is accurate.
- Burgers is controlled only by one or two top modes.
- The basin is completely explained by top-Jacobian tangent directions.

The safer statement is:

> Experiments strongly support that Burgers `replace` reaches a broad optimizer-relevant high-loss region, while NS2D `replace` remains outside the high-loss basin. Endpoint-local Jacobian/Gauss-Newton modes are causally important for the Burgers replace endpoint, but the basin width is not explained by a simple top-tangent versus orthogonal-tangent split alone.

## Evidence chain

### 1. Main optimizer behavior

Burgers:

- `steepest_add` and `steepest_replace` are nearly tied in final Loss3.

NS2D:

- `steepest_add` is much stronger than `steepest_replace`.
- Representative result: `steepest_add` around `278.5`, `steepest_replace` around `96.4`.

This shows that NS2D replace does not merely converge slowly; it reaches a different and worse region.

### 2. Lmax-normalized endpoint cap volume

Using same-sample \(L_{\max}\) as the reference:

Burgers `steepest_replace` near the endpoint:

\[
p(L \ge 0.95 L_{\max}) \approx 0.4 \text{ to } 0.6
\]

NS2D `steepest_replace` near the endpoint:

\[
p(L \ge 0.95 L_{\max}) = 0,
\qquad
L/L_{\max}\approx 0.27.
\]

This directly supports that Burgers replace is near a high-loss region, while NS2D replace is not.

Files:

- `analysis_outputs/mechanism_20260622/full_mechanism_validation/boundary_volume_probe_20260623/tables/endpoint_cap_volume_lmax_summary.csv`
- `analysis_outputs/mechanism_20260622/full_mechanism_validation/boundary_volume_probe_20260623/figures/endpoint_cap_lmax_p095_by_theta.svg`

### 3. First-order prediction does not explain Burgers replace

Burgers replace has large first-order prediction error:

\[
\text{relative error} \approx 3.407.
\]

Therefore the Burgers result should not be explained as "the problem is linear, so replace accurately predicts the full-radius optimum."

### 4. Dominant-subspace iso-projection

Endpoint-PCA/coherent-structure test:

Burgers `steepest_replace`, \(k=8,\beta=1\):

\[
p_{0.95}=0.600,
\qquad
L/L_{\max}=0.919.
\]

NS2D `steepest_replace`, same test:

\[
p_{0.95}=0,
\qquad
L/L_{\max}=0.274.
\]

This supports a forgiving medium-dimensional local structure in Burgers and confirms that NS2D replace remains low relative to the best loss.

Files:

- `analysis_outputs/mechanism_20260622/full_mechanism_validation/dominant_subspace_iso_projection_20260623/tables/iso_projection_summary.csv`
- `analysis_outputs/mechanism_20260622/full_mechanism_validation/dominant_subspace_iso_projection_20260623/figures/iso_projection_p095_by_beta_k4.svg`

### 5. Jacobian/Gauss-Newton keep/remove causal ablation

Endpoint-local Jacobian/Gauss-Newton basis, Burgers `steepest_replace`:

\[
\text{keep } k=8:
p_{0.95}=0.600,\quad L/L_{\max}=0.944.
\]

\[
\text{remove } k=8:
p_{0.95}=0,\quad L/L_{\max}=0.261.
\]

This is the strongest causal evidence: keeping the endpoint-local Jacobian/GN structure preserves high loss; removing it destroys high loss.

NS2D `steepest_replace`:

\[
\text{keep } k=8:
p_{0.95}=0,\quad L/L_{\max}=0.157.
\]

So NS2D replace cannot be rescued by keeping these modes; it starts outside the high-loss basin.

Files:

- `analysis_outputs/mechanism_20260622/full_mechanism_validation/jacobian_mode_causal_ablation_20260623/tables/jacobian_mode_keep_remove_summary.csv`
- `analysis_outputs/mechanism_20260622/full_mechanism_validation/jacobian_mode_causal_ablation_20260623/figures/jacobian_keep_remove_endpoint_steepest_replace_p095.svg`

### 6. Basin attraction from replace endpoint

Continue steepest-style optimization from the `steepest_replace` endpoint for 20 steps:

Burgers:

\[
L/L_{\max}=1.044,
\qquad
p_{0.95}=0.800.
\]

NS2D:

\[
L/L_{\max}=0.700,
\qquad
p_{0.95}=0.
\]

This supports that Burgers replace is in a basin that can continue to high loss, while NS2D replace remains outside the high-loss basin.

File:

- `analysis_outputs/mechanism_20260622/full_mechanism_validation/jacobian_mode_causal_ablation_20260623/tables/basin_attraction_summary.csv`

### 7. Matched-angle Jacobian tangent-width check

This test compares equal-angle perturbations from the replace endpoint along:

- endpoint-local top-Jacobian tangent directions;
- orthogonal tangent directions.

It does not show a clean, decisive top-tangent advantage in Burgers. Therefore, the final explanation should not claim that basin width is completely determined by top tangent directions.

It does confirm:

- Burgers replace neighborhood retains high loss at small angles.
- NS2D replace neighborhood remains low loss for both top and orthogonal tangent families.

Files:

- `analysis_outputs/mechanism_20260622/full_mechanism_validation/jacobian_tangent_width_20260623/tables/jacobian_tangent_width_summary.csv`
- `analysis_outputs/mechanism_20260622/full_mechanism_validation/jacobian_tangent_width_20260623/figures/jacobian_tangent_width_steepest_replace_k8_p095.svg`

## Reliability

The conclusion is strong experimentally, but should be phrased as evidence rather than proof.

Reliable:

- NS2D replace is outside the high-loss basin.
- Burgers replace reaches a much more forgiving, high-loss optimizer-relevant region.
- Burgers endpoint-local Jacobian/GN modes are causally important for preserving high loss.

Still cautious:

- Current causal experiments use Burgers `N=5` and NS2D `N=2`.
- Endpoint-local Jacobian modes are not the same as a global mechanistic basis.
- Matched-angle tangent width does not reduce the mechanism to a simple top-vs-orth tangent story.

Recommended paper wording:

> The evidence strongly supports that Burgers `steepest_replace` reaches a broad optimizer-relevant high-loss basin, whereas NS2D `steepest_replace` remains outside the high-loss basin reached by `steepest_add`.

Avoid:

> We prove that Burgers is controlled solely by top Jacobian modes.
