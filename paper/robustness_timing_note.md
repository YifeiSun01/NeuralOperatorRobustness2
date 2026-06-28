# Robustness Metric Timing Note

This note records the timing comparison for the three robustness quantities.
It is only a working note; the table is not included in `main.tex`.

## Unit

A `sample-model pair` means one input sample evaluated with one model.
For example, 25 samples and 2 models give

\[
25 \times 2 = 50
\]

sample-model pairs.

The average time is computed as

\[
\text{mean time per pair}
=
\frac{\text{total wall-clock time}}{\text{number of sample-model pairs}}.
\]

## Recorded Timing

| Robustness quantity | Recorded setting | Total time | Pairs | Mean time per pair |
|---|---:|---:|---:|---:|
| Jacobian operator norm / top singular value | top-\(k\) SVD, \(k=20\), not full SVD | 259.42 s | 50 | 5.188 s |
| Finite adversarial attack loss increase | 20 attack steps | 214.70 s | 80 | 2.684 s |
| Jacobian-error metric \(J_{\mathcal E}(x)^\top e_\theta(x)\) | formed from saved local derivative artifacts | 8.33 s | 50 | 0.167 s |

## Interpretation

The operator-norm timing above is not the old full-SVD timing. It is the recorded
top-\(k\) SVD run used to obtain the top singular value.

The finite attack is faster than this top-singular/operator-norm computation in
these records, but it is not faster than the recorded Jacobian-error quantity.

The Jacobian-error timing here is postprocessing from saved local derivative
artifacts. A standalone VJP-only implementation was not separately timed in
these logs, so the paper should avoid claiming an exact standalone VJP runtime
unless that experiment is run separately.
