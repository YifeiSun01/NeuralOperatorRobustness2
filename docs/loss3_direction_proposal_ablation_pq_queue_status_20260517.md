# Loss3 Direction-Proposal Ablation P/Q Queue Status - 2026-05-17

Status checked after queue launch.

Observed process state:

- Queue shell process is running: PID `44098`.
- Current Python experiment process is running: PID `44102`.
- Current pair: `p=1`, `q=1`.
- Current method in log: `raw_add`.
- GPU utilization: `97%`.
- GPU memory: `9522 / 32768 MiB`.

Observed logs:

```text
/workspace/NeuralOperatorRobustness2/logs/loss3_direction_proposal_pq_queue.log
/workspace/NeuralOperatorRobustness2/logs/loss3_direction_proposal_p1_q1_batch100.log
```

Observed output directory exists:

```text
/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p1_q1
```

The JAX CUDA executor warning appeared again, but the earlier completed runs also
showed the same warning and continued successfully.  There is no failure in the
current log at this status check.

## Progress Update - 2026-05-18 00:18 UTC

Observed process state:

- Queue shell process is still running: PID `44098`.
- Current Python experiment process is running: PID `48371`.
- Current pair: `p=1`, `q=inf`.
- Current method in log: `steepest_add` after `raw_add` and `unit_raw_add` started/completed.

Observed output directories and figure counts:

- `fno_nu0p001_eps8_alpha0p3_batch100_steps100_p1_q1`: manifest status `completed`, `40` PNG figures.
- `fno_nu0p001_eps8_alpha0p3_batch100_steps100_p1_q2`: manifest status `completed`, `40` PNG figures.
- `fno_nu0p001_eps8_alpha0p3_batch100_steps100_p1_qinf`: manifest status `run_started`, `0` PNG figures at this check.

Observed logs:

```text
/workspace/NeuralOperatorRobustness2/logs/loss3_direction_proposal_pq_queue.log
/workspace/NeuralOperatorRobustness2/logs/loss3_direction_proposal_p1_qinf_batch100.log
```

Inference: the queue is progressing normally through the requested P/Q grid. Do
not interpret `p=1,q=inf` figures yet because this pair has not completed and no
figures existed at this check.

## Progress Update - 2026-05-18 01:18 UTC

Observed process state:

- Queue shell process is still running: PID `44098`.
- Current Python experiment process is running: PID `65319`.
- Current pair: `p=inf`, `q=1`.
- GPU utilization: `97%`.
- GPU memory: `10740 / 32768 MiB`.

Observed queue log:

```text
/workspace/NeuralOperatorRobustness2/logs/loss3_direction_proposal_pq_queue.log
```

The queue has already launched these pairs:

```text
p=1,q=1
p=1,q=2
p=1,q=inf
p=2,q=1
p=2,q=inf
p=inf,q=1  <-- currently running
```

Observed manifest/output status:

| Pair | Manifest status | final_deltas.npz | Figure PNG count |
|---|---|---:|---:|
| `p=1,q=1` | `completed` | yes | 70 |
| `p=1,q=2` | `completed` | yes | 70 |
| `p=1,q=inf` | `completed` | yes | 68 |
| `p=2,q=1` | `completed` | yes | 40 |
| `p=2,q=2` | `completed` | yes | 112 |
| `p=2,q=inf` | `completed` | yes | 40 |
| `p=inf,q=1` | `run_started` | no | 0 |

Observed current `p=inf,q=1` method progress:

| Method | per-step rows | Last k | Summary exists |
|---|---:|---:|---|
| `raw_add` | 101 | 100 | yes |
| `unit_raw_add` | 101 | 100 | yes |
| `steepest_add` | 101 | 100 | yes |
| `steepest_replace` | 101 | 100 | yes |
| `power_replace__objective_gradient` | 101 | 100 | yes |
| `power_replace__pure_jvp_vjp` | 101 | 100 | yes |
| `power_replace__generalized_pq` | 0 | none | no |

Inference: `p=inf,q=1` is still running and is on the final method of the
`pq_key` method set. After `p=inf,q=1`, the queued command still has
`p=inf,q=2` and `p=inf,q=inf` remaining.

No separate plotting or post-processing process was observed running at this
check; the only active experiment process is the queue's current optimizer run.


## Progress Update - 2026-05-18 01:20 UTC

Observed process state:

- Queue shell process is still running: PID `44098`.
- Current Python experiment process is running: PID `71831`.
- Current pair: `p=inf`, `q=2`.
- Current method in log: `raw_add`.
- GPU utilization: `99%`.
- GPU memory: `9522 / 32768 MiB`.

Observed queue log:

```text
/workspace/NeuralOperatorRobustness2/logs/loss3_direction_proposal_pq_queue.log
```

Observed current run log:

```text
/workspace/NeuralOperatorRobustness2/logs/loss3_direction_proposal_pinf_q2_batch100.log
```

Observed status: the queue has advanced past `p=inf,q=1` and is now running
`p=inf,q=2`. The `p=inf,q=inf` pair remains queued after the current run.

No separate plotting or post-processing process was observed at this check; the
active workload is the optimizer/data run.


## Core-Four Data Availability Check - 2026-05-18 01:22 UTC

Observed evidence from local output directories:

| Pair | Core methods visible | Missing core methods | final_deltas.npz |
|---|---|---|---|
| `p=1,q=1` | `raw_add`, `steepest_add`, `steepest_replace` | `raw_replace` | yes |
| `p=1,q=2` | `raw_add`, `steepest_add`, `steepest_replace` | `raw_replace` | yes |
| `p=1,q=inf` | `raw_add`, `steepest_add`, `steepest_replace` | `raw_replace` | yes |
| `p=2,q=1` | `raw_add`, `steepest_add`, `steepest_replace` | `raw_replace` | yes |
| `p=2,q=2` | `raw_add`, `raw_replace`, `steepest_add`, `steepest_replace` | none | yes |
| `p=2,q=inf` | `raw_add`, `steepest_add`, `steepest_replace` | `raw_replace` | yes |
| `p=inf,q=1` | `raw_add`, `steepest_add`, `steepest_replace` | `raw_replace` | yes |
| `p=inf,q=2` | `raw_add` so far | `raw_replace`, `steepest_add`, `steepest_replace` so far | no |

Observed process state: queue PID `44098` is still running; current experiment PID
`71831` is running `p=inf,q=2` with methods `pq_key`. No separate plotting or
post-processing process was observed.

Inference: matching simplified four-line plots for other P/Q pairs require either
a minimal additional run that includes `raw_replace`, or an explicitly marked
inferred duplicate only for `p=2` cases where `raw_replace` and
`steepest_replace` should coincide by the p=2 geometry.


## Progress Update - 2026-05-18 01:30 UTC

Observed process state:

- Queue shell process is still running: PID `44098`.
- Current Python experiment process is still running: PID `71831`.
- Current pair: `p=inf`, `q=2`.
- GPU utilization: `97%`.
- GPU memory: `10744 / 32768 MiB`.

Observed status: the queue is still active; the current workload is still the
optimizer/data run rather than a separate plotting process.


## Progress Update - 2026-05-18 01:33 UTC

Observed process state:

- Queue shell process is still running: PID `44098`.
- Current Python experiment process is still running: PID `71831`.
- Current pair: `p=inf`, `q=2`.
- GPU utilization: `98%`.
- GPU memory: `10744 / 32768 MiB`.

Observed current `p=inf,q=2` method progress:

| Method | per-step rows | Last k | Summary exists |
|---|---:|---:|---|
| `raw_add` | 101 | 100 | no |
| `unit_raw_add` | 101 | 100 | no |
| `steepest_add` | 101 | 100 | no |
| `steepest_replace` | 101 | 100 | no |
| `power_replace__objective_gradient` | 101 | 100 | no |
| `power_replace__pure_jvp_vjp` | 101 | 100 | no |
| `power_replace__generalized_pq` | 0 | none | no |

Observed output status: completed pairs through `p=inf,q=1` have root
`final_deltas.npz`; `p=inf,q=2` is still `run_started` and has no root
`final_deltas.npz` yet. The queued `p=inf,q=inf` pair has not started yet.

Inference: the background queue is not finished. It is currently on the final
method of `p=inf,q=2`, and then still needs to run `p=inf,q=inf`.


## Progress Update - 2026-05-18 01:41 UTC

Observed process state:

- Queue shell process is still running: PID `44098`.
- Current Python experiment process is running: PID `79740`.
- Current pair: `p=inf`, `q=inf`, the final queued pair.
- GPU utilization at the previous check was `97%`; GPU memory was `9522 / 32768 MiB`.

Observed current `p=inf,q=inf` method progress:

| Method | per-step rows | Last k |
|---|---:|---:|
| `raw_add` | 101 | 100 |
| `unit_raw_add` | 101 | 100 |
| `steepest_add` | 101 | 100 |
| `steepest_replace` | directory exists, no per-step rows yet | none |
| `power_replace__objective_gradient` | not started | none |
| `power_replace__pure_jvp_vjp` | not started | none |
| `power_replace__generalized_pq` | not started | none |

Inference from the previous completed `p=inf,q=1` and `p=inf,q=2` timings: the
remaining runtime is approximately 10-12 minutes from this check, so a reasonable
expected finish window is around 2026-05-18 01:51-01:54 UTC if the current speed
continues.


## Progress Update - 2026-05-18 01:48 UTC

Observed process/output state for final queued pair `p=inf,q=inf`:

| Method | per-step rows | Last k |
|---|---:|---:|
| `raw_add` | 101 | 100 |
| `unit_raw_add` | 101 | 100 |
| `steepest_add` | 101 | 100 |
| `steepest_replace` | 101 | 100 |
| `power_replace__objective_gradient` | 101 | 100 |
| `power_replace__pure_jvp_vjp` | 101 | 100 |
| `power_replace__generalized_pq` | directory exists, no per-step rows yet | none |

Observed status: `p=inf,q=inf` is still `run_started`; root `final_deltas.npz`
does not yet exist. The run appears to be on the final method of the final queued
pair.

Inference from prior `p=inf` timings: likely remaining time is only a few
minutes if the final method proceeds similarly.


## Completion/Data Availability Update - 2026-05-18 01:53 UTC

Observed process state:

- No active `run_loss3_direction_proposal_ablation.py` or queue shell process was observed.
- Queue log includes the final queued pair `p=inf,q=inf`.
- Final run log includes `[done] ablation outputs written under ... p=inf_qinf`.

Observed output status:

| Pair | Manifest status | final_deltas.npz | Core methods present | Missing core method |
|---|---|---|---|---|
| `p=1,q=1` | completed | yes | `raw_add`, `steepest_add`, `steepest_replace` | `raw_replace` |
| `p=1,q=2` | completed | yes | `raw_add`, `steepest_add`, `steepest_replace` | `raw_replace` |
| `p=1,q=inf` | completed | yes | `raw_add`, `steepest_add`, `steepest_replace` | `raw_replace` |
| `p=2,q=1` | completed | yes | `raw_add`, `steepest_add`, `steepest_replace` | `raw_replace` |
| `p=2,q=2` | completed | yes | `raw_add`, `raw_replace`, `steepest_add`, `steepest_replace` | none |
| `p=2,q=inf` | completed | yes | `raw_add`, `steepest_add`, `steepest_replace` | `raw_replace` |
| `p=inf,q=1` | completed | yes | `raw_add`, `steepest_add`, `steepest_replace` | `raw_replace` |
| `p=inf,q=2` | completed | yes | `raw_add`, `steepest_add`, `steepest_replace` | `raw_replace` |
| `p=inf,q=inf` | completed | yes | `raw_add`, `steepest_add`, `steepest_replace` | `raw_replace` |

Inference: the queued P/Q experiments are complete, but matching simplified
four-method figures cannot be strictly generated for pairs other than `p=2,q=2`
until `raw_replace` is run. Three-method figures can be generated now for all
completed P/Q pairs. For `p=2` pairs, `raw_replace` can be inferred from
`steepest_replace` by L2 geometry, but that should be labeled as inferred unless
it is actually run.


## Raw-Replace Backfill Progress Update - 2026-05-18 02:04 UTC

Observed process state:

- Backfill queue shell process is running: PID `86272`.
- Current Python experiment process is running: PID `89734`.
- Current pair: `p=2`, `q=inf`.
- Current method: `raw_replace` only.
- GPU utilization observed: `88-99%`.
- GPU memory observed: `9522 / 32768 MiB`.

Observed log/output status:

| Pair | Status | final_deltas.npz | raw_replace per-step rows |
|---|---|---|---:|
| `p=1,q=1` | completed | yes | 101 |
| `p=1,q=2` | completed | yes | 101 |
| `p=1,q=inf` | completed | yes | 101 |
| `p=2,q=1` | completed | yes | 101 |
| `p=2,q=inf` | running | no | none yet |
| `p=inf,q=1` | queued | no | none |
| `p=inf,q=2` | queued | no | none |
| `p=inf,q=inf` | queued | no | none |

Observed note: the expected outer queue log
`logs/loss3_raw_replace_backfill_queue_20260518.log` does not exist, but the
per-pair logs exist and are updating.

Inference from completed backfill pairs: each pair is taking about 1.9 minutes.
At 2026-05-18 02:04 UTC, the queue had been running about 8 minutes and had
completed 4 of 8 pairs; the current fifth pair had just started. Estimated
remaining runtime is about 7-9 minutes, with an expected finish around
2026-05-18 02:11-02:13 UTC if speed stays similar.


## Raw-Replace Backfill Completion - 2026-05-18 02:17 UTC

Observed process state:

- No active `loss3_raw_replace_backfill`, `run_loss3_direction_proposal_ablation.py`, or `adv_robust/bin/python` experiment process was observed.
- GPU utilization was `0%`; GPU memory was `0 / 32768 MiB`.
- The last backfill log, `logs/loss3_raw_replace_backfill_pinf_qinf_batch100.log`, includes `[done] ablation outputs written under ... p_inf_qinf`.

Observed backfill output status:

| Pair | Manifest status | final_deltas.npz | raw_replace per-step rows | Last k |
|---|---|---|---:|---:|
| `p=1,q=1` | completed | yes | 101 | 100 |
| `p=1,q=2` | completed | yes | 101 | 100 |
| `p=1,q=inf` | completed | yes | 101 | 100 |
| `p=2,q=1` | completed | yes | 101 | 100 |
| `p=2,q=inf` | completed | yes | 101 | 100 |
| `p=inf,q=1` | completed | yes | 101 | 100 |
| `p=inf,q=2` | completed | yes | 101 | 100 |
| `p=inf,q=inf` | completed | yes | 101 | 100 |

Inference: the raw-replace backfill is complete. Strict four-core-method plotting
is now possible by combining the original P/Q runs with the corresponding
backfill `raw_replace` outputs.
