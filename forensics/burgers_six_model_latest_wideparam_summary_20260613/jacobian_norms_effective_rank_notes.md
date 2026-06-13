# Jacobian Norms and Effective Rank Notes

Context: in the Burgers robustness analysis, the main local object is

\[
J_{\mathrm{error}}(x) = J_{\mathrm{model}}(x) - J_{\mathrm{solver}}(x).
\]

The same definitions below apply to any matrix \(A \in \mathbb{R}^{m \times n}\).

## 1. SVD Setup

Let the singular value decomposition be

\[
A = U \Sigma V^\top,
\]

with singular values

\[
\sigma_1 \ge \sigma_2 \ge \cdots \ge \sigma_r > 0,
\]

where \(r = \operatorname{rank}(A)\).

The right singular vectors \(v_k\) are input directions, and the left singular
vectors \(u_k\) are output directions. They satisfy

\[
A v_k = \sigma_k u_k.
\]

So \(\sigma_k\) is the local amplification factor along input direction \(v_k\).

## 2. Spectral Norm

The spectral norm is

\[
\|A\|_2 = \max_{\|x\|_2 = 1} \|Ax\|_2.
\]

In terms of singular values,

\[
\|A\|_2 = \sigma_1.
\]

Interpretation:

- It measures the largest possible local amplification over all unit input
  directions.
- For \(A = J_{\mathrm{error}}\), it measures the worst local error-sensitivity
  direction.

## 3. Frobenius Norm

The Frobenius norm is defined from matrix entries:

\[
\|A\|_F =
\sqrt{
\sum_{i=1}^{m}\sum_{j=1}^{n} A_{ij}^2
}.
\]

Equivalently,

\[
\|A\|_F^2 = \operatorname{tr}(A^\top A).
\]

Why?

\[
(A^\top A)_{jj}
=
\sum_{i=1}^{m} A_{ij}^2.
\]

Therefore,

\[
\operatorname{tr}(A^\top A)
=
\sum_{j=1}^{n} (A^\top A)_{jj}
=
\sum_{j=1}^{n}\sum_{i=1}^{m} A_{ij}^2
=
\|A\|_F^2.
\]

Now plug in the SVD:

\[
A = U\Sigma V^\top.
\]

Then

\[
A^\top A
=
(U\Sigma V^\top)^\top(U\Sigma V^\top)
=
V\Sigma^\top U^\top U\Sigma V^\top.
\]

Since \(U^\top U = I\),

\[
A^\top A = V\Sigma^\top\Sigma V^\top.
\]

Also,

\[
\Sigma^\top\Sigma
=
\operatorname{diag}
(\sigma_1^2,\sigma_2^2,\ldots,\sigma_r^2,0,\ldots,0).
\]

So the eigenvalues of \(A^\top A\) are

\[
\sigma_1^2,\sigma_2^2,\ldots,\sigma_r^2,0,\ldots,0.
\]

Trace equals the sum of eigenvalues, so

\[
\operatorname{tr}(A^\top A)
=
\sum_{k=1}^{r}\sigma_k^2.
\]

Combining the two identities gives

\[
\boxed{
\|A\|_F^2
=
\sum_{i,j} A_{ij}^2
=
\sum_{k=1}^{r}\sigma_k^2
}
\]

and therefore

\[
\boxed{
\|A\|_F
=
\sqrt{\sum_{k=1}^{r}\sigma_k^2}.
}
\]

Interpretation:

- Spectral norm looks only at the largest direction.
- Frobenius norm sums the squared amplification over all singular directions.
- For \(A = J_{\mathrm{error}}\), Frobenius norm measures the total local
  error-Jacobian energy across directions.

## 4. Random Direction Interpretation

If \(x\) is uniformly distributed on the unit sphere \(S^{n-1}\), then

\[
\mathbb{E}_{x\sim S^{n-1}}\|Ax\|_2^2
=
\frac{1}{n}\|A\|_F^2.
\]

Reason:

\[
\|Ax\|_2^2 = x^\top A^\top A x.
\]

Taking expectation,

\[
\mathbb{E}\|Ax\|_2^2
=
\mathbb{E}\left[x^\top A^\top A x\right]
=
\operatorname{tr}
\left(
A^\top A \, \mathbb{E}[xx^\top]
\right).
\]

For a uniform random unit vector,

\[
\mathbb{E}[xx^\top] = \frac{1}{n}I.
\]

Therefore,

\[
\mathbb{E}\|Ax\|_2^2
=
\operatorname{tr}
\left(
A^\top A \frac{1}{n}I
\right)
=
\frac{1}{n}\operatorname{tr}(A^\top A)
=
\frac{1}{n}\|A\|_F^2.
\]

So Frobenius norm is directly tied to average squared amplification over random
directions.

## 5. Relation Between Spectral and Frobenius Norm

Since

\[
\|A\|_2 = \sigma_1
\]

and

\[
\|A\|_F = \sqrt{\sigma_1^2+\sigma_2^2+\cdots+\sigma_r^2},
\]

we always have

\[
\boxed{
\|A\|_2 \le \|A\|_F.
}
\]

If \(A\) is almost rank one, then most energy is in \(\sigma_1\), so

\[
\|A\|_F \approx \|A\|_2.
\]

If many singular values are large, then

\[
\|A\|_F \gg \|A\|_2.
\]

More generally, if \(\operatorname{rank}(A)=r\), then

\[
\boxed{
\|A\|_F \le \sqrt{r}\,\|A\|_2.
}
\]

because

\[
\|A\|_F^2
=
\sum_{k=1}^{r}\sigma_k^2
\le
\sum_{k=1}^{r}\sigma_1^2
=
r\sigma_1^2
=
r\|A\|_2^2.
\]

## 6. Effective Rank / Entropy Rank

In the code, effective rank is computed from singular-value energy:

\[
e_k = \sigma_k^2.
\]

Normalize the energies into a probability distribution:

\[
p_k =
\frac{\sigma_k^2}
{\sum_j \sigma_j^2}.
\]

Then define entropy

\[
H(p)
=
-
\sum_k p_k \log p_k.
\]

The effective rank is

\[
\boxed{
r_{\mathrm{eff}}
=
\exp(H(p))
=
\exp
\left(
-
\sum_k p_k\log p_k
\right).
}
\]

This is also called entropy rank.

Interpretation:

- \(r_{\mathrm{eff}}\) measures how many singular directions effectively carry
  the energy of the matrix.
- It is not the ordinary algebraic rank.
- It is a measure of how spread out the singular-value energy distribution is.

### Extreme Cases

If all energy is in one singular value:

\[
p_1 = 1,\qquad p_{k>1}=0.
\]

Then

\[
H(p)=0
\]

and

\[
r_{\mathrm{eff}} = e^0 = 1.
\]

So effective rank near 1 means the matrix behaves like one dominant direction.

If energy is evenly distributed over \(K\) singular directions:

\[
p_1=p_2=\cdots=p_K=\frac{1}{K}.
\]

Then

\[
H(p)
=
-
\sum_{k=1}^{K}
\frac{1}{K}\log\frac{1}{K}
=
\log K.
\]

Therefore

\[
r_{\mathrm{eff}}
=
\exp(\log K)
=
K.
\]

So effective rank near \(K\) means the matrix energy is spread across roughly
\(K\) directions.

## 7. Is Effective Rank a Measure of Unevenness?

Yes, but with a precise wording:

- Effective rank is high when singular-value energy is spread across many
  directions.
- Effective rank is low when singular-value energy is concentrated in a few
  directions.

So it measures the effective number of active directions, which is the inverse
notion of concentration/unevenness.

High effective rank:

\[
\sigma_1^2,\sigma_2^2,\ldots
\]

are more evenly spread across many directions.

Low effective rank:

\[
\sigma_1^2
\]

dominates, and most other singular values are small.

However, effective rank alone does not say whether the matrix is large or small.
It only describes the shape of the singular-value distribution after normalization.

For example:

\[
\sigma = (100,0,0,0)
\]

has effective rank 1 and huge spectral norm.

But

\[
\sigma = (0.001,0,0,0)
\]

also has effective rank 1, while the matrix is tiny.

Therefore effective rank must be read together with spectral norm and Frobenius
norm.

## 8. How to Read These Metrics Together

For \(A = J_{\mathrm{error}}\):

- \(\|A\|_2\): worst local error-amplifying direction.
- \(\|A\|_F\): total local error-amplification energy across directions.
- \(r_{\mathrm{eff}}\): whether the error energy is concentrated or spread out.

Typical interpretations:

- Large spectral norm, large Frobenius norm, low effective rank:
  one or a few very bad directions dominate.
- Large spectral norm, large Frobenius norm, high effective rank:
  many directions are bad.
- Small spectral norm and small Frobenius norm:
  locally robust, regardless of effective rank.
- High effective rank with small spectral/Frobenius norm:
  error is spread out, but the total error is still small.

In the Burgers six-model summary, spectral norm and Frobenius norm are highly
correlated, so they mostly agree on model ranking. Effective rank is useful for
understanding concentration of error directions, not as a standalone robustness
score.
