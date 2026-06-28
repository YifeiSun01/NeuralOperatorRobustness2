# Burgers Error-Jacobian And Endpoint Growth Final Conclusion

Date: 2026-06-08

## Question

Record the final conclusion about the relationship between:

- residual movement,
- endpoint growth,
- the error-Jacobian spectral norm,
- the top singular direction,
- and the cross term with the clean residual.

This is the cleaned-up conclusion after separating the Rayleigh/SVD direction
from the actual PGD endpoint-attack direction.

## Setup

Define the model-solver error vector:

\[
E(x)=f_\theta(x)-S(x).
\]

At the clean input:

\[
E_0 = E(x).
\]

After an adversarial perturbation:

\[
E_1 = E(x+\delta).
\]

The residual movement is:

\[
\Delta E = E_1-E_0=E(x+\delta)-E(x).
\]

The endpoint growth is:

\[
\Delta L_{\mathrm{end}}
=
\frac{1}{N}\|E(x+\delta)\|^2
-
\frac{1}{N}\|E(x)\|^2.
\]

Since:

\[
E(x+\delta)=E_0+\Delta E,
\]

we have:

\[
\Delta L_{\mathrm{end}}
=
\frac{1}{N}\|\Delta E\|^2
+
\frac{2}{N}\langle E_0,\Delta E\rangle.
\]

Define:

\[
R=\frac{1}{N}\|\Delta E\|^2
\]

and:

\[
C=\frac{2}{N}\langle E_0,\Delta E\rangle.
\]

Then:

\[
\Delta L_{\mathrm{end}}=R+C.
\]

Here \(R\) is always nonnegative. The cross term \(C\) can be positive or
negative.

If:

\[
\langle E_0,\Delta E\rangle>0,
\]

then the residual movement points partly outward along the clean residual, and
endpoint growth gets an extra positive contribution.

If:

\[
\langle E_0,\Delta E\rangle<0,
\]

then the residual movement points partly against the clean residual, and endpoint
growth is partially canceled.

## SVD / Rayleigh Direction

Let:

\[
A=J_E(x)=J_{\mathrm{model}}(x)-J_{\mathrm{solver}}(x).
\]

The residual-movement-only linearized problem is:

\[
\max_{\|\delta\|=r}\|A\delta\|^2.
\]

The solution is the top singular axis:

\[
\delta=\pm r v_1,
\]

where \(v_1\) is the top right singular vector of \(A\).

The sign is not determined by this residual-movement-only objective because:

\[
\|A(rv_1)\|^2=\|A(-rv_1)\|^2.
\]

So Rayleigh/SVD gives the axis:

\[
\mathrm{span}(v_1),
\]

not the endpoint-good sign.

## Endpoint Sign On The Top Singular Axis

The linearized endpoint-growth objective is:

\[
G(\delta)
=
\frac{1}{N}\|A\delta\|^2
+
\frac{2}{N}\langle E_0,A\delta\rangle.
\]

On the top singular axis:

\[
\delta = s r v_1,\qquad s\in\{-1,+1\}.
\]

Then:

\[
G(srv_1)
=
\frac{r^2}{N}\|Av_1\|^2
+
\frac{2sr}{N}\langle E_0,Av_1\rangle.
\]

Therefore, the endpoint-good sign is:

\[
s^\star=\operatorname{sign}\left(\langle E_0,Av_1\rangle\right).
\]

The endpoint-oriented top-singular perturbation is:

\[
\delta_{\mathrm{top,out}}
=
r\,\operatorname{sign}\left(\langle E_0,Av_1\rangle\right)v_1.
\]

Its cross term is:

\[
C_{\mathrm{top,out}}
=
\frac{2r}{N}\left|\langle E_0,Av_1\rangle\right|
\ge 0.
\]

So the raw sign of \(v_1\) from an SVD file should not be interpreted. The
physically meaningful top-singular endpoint comparison is the outward-sign
version above.

## Actual PGD Direction

Actual PGD does not optimize the Rayleigh quotient. It directly optimizes the
nonlinear endpoint objective:

\[
\max_{\|\delta\|\le r}
\left[
\frac{1}{N}\|E(x+\delta)\|^2
-
\frac{1}{N}\|E(x)\|^2
\right].
\]

Its realized residual movement is:

\[
\Delta E_{\mathrm{PGD}}
=
E(x+\delta_{\mathrm{PGD}})-E(x).
\]

The actual PGD cross term is:

\[
C_{\mathrm{PGD}}
=
\frac{2}{N}\langle E_0,\Delta E_{\mathrm{PGD}}\rangle.
\]

This sign is meaningful because \(\delta_{\mathrm{PGD}}\) is the actual
perturbation chosen by the endpoint attack.

## Observed Evidence

Source files:

- direction audit: `docs/burgers_attack_delta_svd_direction_alignment_20260608.md`
- actual-vs-top-singular cross-term audit: `docs/burgers_actual_vs_top_singular_cross_term_20260608.md`
- direction rows: `forensics/burgers_attack_delta_svd_direction_alignment_20260608/attack_delta_vs_error_svd_direction_rows.csv`
- actual-vs-top rows: `forensics/burgers_actual_vs_top_singular_cross_term_20260608/actual_vs_top1_cross_term_rows.csv`
- actual-vs-top summary: `forensics/burgers_actual_vs_top_singular_cross_term_20260608/actual_vs_top1_cross_term_summary.csv`
- scripts:
  - `tools/compute_burgers_attack_delta_svd_direction_alignment_20260608.py`
  - `tools/compute_burgers_actual_vs_top_singular_cross_term_20260608.py`

The saved analyses matched all rows:

\[
n=440,\qquad n_{\mathrm{missing}}=0.
\]

Actual PGD is not close to the top-1 singular vector:

\[
\operatorname{mean}\left(|\cos(\delta_{\mathrm{PGD}},v_1)|\right)=0.322
\]

over all rows, and:

\[
\operatorname{mean}\left(|\cos(\delta_{\mathrm{PGD}},v_1)|\right)=0.350
\]

on generalization rows.

For actual PGD on generalization rows:

\[
C_{\mathrm{PGD}}<0
\]

on:

\[
62.7\%
\]

of rows, with mean:

\[
\operatorname{mean}(C_{\mathrm{PGD}})=-0.0010.
\]

The actual residual movement mean was:

\[
\operatorname{mean}(R_{\mathrm{PGD}})=0.01517.
\]

So the mean cancellation fraction was approximately:

\[
\frac{0.0010}{0.01517}\approx 6.6\%.
\]

Thus, the cross term often partially cancels endpoint growth on generalization
rows, but it does not dominate the endpoint growth.

For the endpoint-good top-singular sign on generalization rows:

\[
\operatorname{mean}(C_{\mathrm{top,out}})=0.002215.
\]

But the corresponding linearized residual movement was much larger:

\[
\operatorname{mean}\left(\frac{1}{N}\|A\delta_{\mathrm{top,out}}\|^2\right)=0.1866.
\]

Also:

\[
\operatorname{mean}\left(|\cos(E_0,Av_1)|\right)=0.0775.
\]

So \(Av_1\) is mostly close to orthogonal to the clean residual \(E_0\). The top
singular direction mainly creates large residual movement, not a large outward
cross term.

## Final Conclusion

The core result is:

\[
\Delta L_{\mathrm{end}}
=
\frac{1}{N}\|\Delta E\|^2
+
\frac{2}{N}\langle E_0,\Delta E\rangle.
\]

Therefore endpoint growth is not determined only by residual movement
\(\|\Delta E\|^2\); it also depends on the orientation of \(\Delta E\) relative
to the clean residual \(E_0\).

However, in the observed Burgers artifacts, the cross term is a correction term,
not the dominant mechanism. The dominant mechanism behind attack damage remains
large residual movement, which is why the error-Jacobian spectral norm remains
strongly predictive of attack damage.

The top singular direction should be interpreted as the direction/axis that
maximizes local residual movement:

\[
\max_{\|\delta\|=r}\|A\delta\|^2.
\]

It should not be interpreted as the actual PGD endpoint direction. Actual PGD
chooses a nonlinear endpoint direction, and empirically it is not close to the
top-1 singular vector.

The endpoint-good top-singular sign is:

\[
\delta_{\mathrm{top,out}}
=
r\,\operatorname{sign}\left(\langle E_0,Av_1\rangle\right)v_1.
\]

With that sign, the top-singular cross term is nonnegative by construction, but
small relative to its residual movement. Thus, the top singular mechanism is
mostly about amplification magnitude, not alignment with \(E_0\).

Paper-ready wording:

> The error-Jacobian spectral norm controls the largest local residual movement.
> Endpoint growth equals this residual movement energy plus an orientation term
> with the clean residual. In the Burgers audits, the orientation term is often
> mildly negative on generalization samples, so it partially cancels endpoint
> growth, but its magnitude is small compared with residual movement. Therefore
> spectral norm remains a strong predictor of adversarial damage, while the
> cross term explains why endpoint growth and residual movement are not exactly
> the same quantity.

## Caveat

This conclusion is based on saved Burgers SVD/attack artifacts for the covered
representative rows. It does not claim that every possible generalization root
has been fully re-SVDed. In particular, this audit uses the existing SVD/attack
coverage described in the source docs above.
