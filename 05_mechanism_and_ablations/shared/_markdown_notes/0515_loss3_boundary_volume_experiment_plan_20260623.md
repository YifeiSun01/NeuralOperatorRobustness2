# Loss3 Boundary-Volume Experiment Plan - 2026-06-23

This plan turns the high-loss landscape hypothesis into direct measurements of
high-loss boundary volume, endpoint cap width, and basin connectivity.

## Question

The current hypothesis is:

\[
\texttt{replace}
\text{ works when the }\epsilon\text{-boundary contains a broad high-loss region,}
\]

and fails when the high-loss region is small, narrow, or path-dependent.

The previous evidence used boundary arcs, 2D slices, ray probes, and JVP/VJP
candidate tests.  Those are useful, but they are still low-dimensional probes.
This experiment directly estimates how much of the boundary is high-loss.

## Metrics

### 1. Global Boundary Volume

Let:

\[
S_\epsilon=\{\delta:\|\delta\|_2=\epsilon\}.
\]

For each sample, define:

\[
L_{\max}=\max_{\text{optimizer finals}} L(\delta_{\text{final}}).
\]

For thresholds:

\[
\tau\in\{0.90,0.95,0.99\},
\]

estimate:

\[
p_\tau
=
\mathbb{P}_{\delta\sim S_\epsilon}
\left[
L(\delta)\ge \tau L_{\max}
\right].
\]

This directly measures the fraction of random boundary points that are high
loss.

Expected pattern:

\[
p_\tau^{\text{Burgers}}
\gg
p_\tau^{\text{NS2D}}.
\]

### 2. Endpoint Cap Volume

Around an endpoint:

\[
u^*=\frac{\delta^*}{\epsilon},
\]

sample orthogonal directions \(v\perp u^*\) and form:

\[
\delta(\theta,v)
=
\epsilon
\left(
\cos\theta\,u^*
+
\sin\theta\,v
\right).
\]

Measure:

\[
p_\tau(\theta)
=
\mathbb{P}_{v}
\left[
L(\delta(\theta,v))\ge \tau L(\delta^*)
\right].
\]

This asks: if we move away from an optimizer endpoint by angle \(\theta\), how
often does Loss3 remain high?

Width summaries:

\[
\theta_{50}^{(\tau)}
=
\max\theta
\quad
\text{s.t.}
\quad
p_\tau(\theta)\ge 0.5,
\]

and:

\[
W_\tau
=
\int p_\tau(\theta)\,d\theta.
\]

Expected pattern:

\[
W_\tau^{\text{Burgers}}
\gg
W_\tau^{\text{NS2D}}.
\]

### 3. Basin Connectivity

For two endpoint perturbations \(\delta_a,\delta_b\), sample a boundary geodesic
between them and compute:

\[
r_{\mathrm{valley}}
=
\frac{
\min_s L(\mathrm{geodesic}(\delta_a,\delta_b;s))
}{
\min(L(\delta_a),L(\delta_b))
}.
\]

Interpretation:

- \(r_{\mathrm{valley}}\approx 1\): connected high-loss ridge.
- \(r_{\mathrm{valley}}\ll 1\): high-loss regions are separated by a valley.

## Implementation

Script:

`tools/probe_loss3_boundary_volume_20260623.py`

Default output:

`analysis_outputs/mechanism_20260622/full_mechanism_validation/boundary_volume_probe_20260623`

Main output tables:

- `tables/global_boundary_volume.csv`
- `tables/global_boundary_volume_summary.csv`
- `tables/endpoint_cap_volume.csv`
- `tables/endpoint_cap_volume_summary.csv`
- `tables/endpoint_cap_width_summary.csv`
- `tables/basin_connectivity.csv`
- `tables/basin_connectivity_summary.csv`

Figures:

- `figures/global_boundary_p095.png`
- `figures/endpoint_cap_p095_by_theta.png`
- `figures/connectivity_valley_ratio.png`

## Initial Run Parameters

The first run is intentionally small so it can start immediately and reveal
whether the metric behaves sensibly:

| Parameter | Value |
| --- | --- |
| Burgers samples | 5 |
| NS2D samples | 2 |
| Global boundary random points | 32 per sample |
| Cap samples | 8 per angle |
| Angles | \(0,0.05,0.10,0.20,0.40,0.80\) radians |
| Thresholds | \(0.90,0.95,0.99\) |
| Connectivity points | 9 |
| Endpoint methods | `steepest_add`, `steepest_replace` |
| Connectivity pair | `steepest_replace__steepest_add` |

## Scaling Plan

If the initial run behaves correctly, scale in this order:

1. Increase Burgers samples from 5 to 20.
2. Increase NS2D samples from 2 to 5.
3. Increase global boundary random points from 32 to 256 or 512.
4. Increase cap samples from 8 to 32 or 64 per angle.
5. Add more endpoint pairs:
   - `raw_add__steepest_add`
   - `steepest_replace__raw_add`
   - `raw_replace__steepest_replace`

## Decision Rule

The high-loss-region explanation is directly supported if:

| Metric | Burgers expected | NS2D expected |
| --- | --- | --- |
| global \(p_{0.95}\) | visibly larger | near zero or much smaller |
| cap \(p_{0.95}(\theta)\) | stays high for larger \(\theta\) | decays quickly |
| \(W_{0.95}\) | large | small |
| valley ratio | near 1 | substantially below 1 for replace-to-add |

This would be stronger evidence than the current linearity/nonlinearity
argument, because it directly measures boundary high-loss width.

