# Burgers attack-step delta comparison

Same fixed Burgers baseline model and same fixed train samples. The seed is reset before each attack-step run, so each sample uses the same epsilon across steps. Method is `fast_replace_linf`, the current Burgers training attack.

## Key summary vs step 10

| steps | adv loss mean | gain mean | rel L2 diff vs 10 | cosine vs 10 | sign agreement vs 10 | high-freq ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.000350974 | 0.000276317 | 1.0971 | 0.3889 | 0.6945 | 0.0369 |
| 2 | 0.000275709 | 0.000201052 | 0.8309 | 0.6487 | 0.8243 | 0.0364 |
| 3 | 0.000309764 | 0.000235107 | 1.1123 | 0.3672 | 0.6836 | 0.0367 |
| 5 | 0.000305185 | 0.000230528 | 1.1314 | 0.3518 | 0.6759 | 0.0342 |
| 10 | 0.000274494 | 0.000199837 | 0.0000 | 1.0000 | 1.0000 | 0.0369 |

## Interpretation

- `fast_replace_linf` always replaces the perturbation by `epsilon * sign(gradient)` at the current attacked point, so all final deltas sit near the L-infinity boundary.
- More steps can change many signs because the gradient is recomputed at the previously attacked point. It is not guaranteed to be monotonic in loss.
- If 2-step and 10-step have high sign agreement/cosine, then the faster training attack is close to the longer attack. If they have low agreement, then 2-step is a cheaper but meaningfully different attack.

## Files

- `summary_vs_step10.csv`
- `per_sample_vs_step10.csv`
- `pairwise_step_similarity.csv`
- `manifest.json`
