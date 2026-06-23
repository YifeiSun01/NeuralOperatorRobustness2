# Loss3 mechanism run coverage - 2026-06-23

This file records what is currently considered **core required mechanism
evidence** for the replace/add explanation, what has already run, what is
running, and what is explicitly treated as optional escalation.

## Core Required Mechanism Evidence

These are the experiments needed for the current claim:

\[
\texttt{replace} \text{ works when the landscape is stable/forgiving enough for
full-budget jumps, and fails when the direction/landscape is path-dependent,
rotating, or narrow.}
\]

| Evidence block | Burgers | Darcy Flow | NS2D | Status |
| --- | --- | --- | --- | --- |
| Formal four-optimizer Loss3 curves | N=100 done | N=20 done | N=20 done | done |
| First-order/full-budget prediction | N=5 current validation + older supporting studies | flip-set analogue through speed/overlap/final set evidence | N=3 exact probe running | running/done |
| Multi-scale local linearity / nonlinearity | older finite-difference/curvature records + current first-order evidence | optional flip analogue | N=3 exact probe running | running/done |
| Boundary high-loss width / near-optimal region | N=5 current landscape/ridge + older landscape records | N=20 near-optimal flip-set/overlap done | N=3 exact boundary arc running | running/done |
| Direction/path stability | older Burgers rotation/path records | N=20 final set overlap + sample trace stability done | N=3 direction trace done | done |
| Cross-system summary tying mechanism to optimizer performance | pending | pending | pending | queued after NS exact probe |

## Running Now

- NS2D exact mechanism probe:
  - PID: `213652`
  - Log: `analysis_outputs/mechanism_20260622/full_mechanism_validation/run_logs/ns2d_exact_mechanism_20260623.log`
  - Output root: `analysis_outputs/mechanism_20260622/full_mechanism_validation/ns2d_exact_mechanism`
  - Covers exact first-order prediction, multi-scale linearity, and boundary arcs on the saved NS2D N=3 trace.

- Auto-summary watcher:
  - PID: `214933`
  - It generates `docs/loss3_full_mechanism_validation_summary_20260623.md` after the NS2D exact probe completes.

- Extended mechanism queue:
  - Script: `tools/run_loss3_extended_mechanism_queue_20260623.sh`
  - Log: `analysis_outputs/mechanism_20260622/full_mechanism_validation/run_logs/extended_mechanism_queue_20260623.log`
  - PID file: `analysis_outputs/mechanism_20260622/full_mechanism_validation/run_logs/extended_mechanism_queue_20260623.pid`
  - It waits for the current NS2D exact probe, then runs the queued expansions below serially.

## Completed New Runs

- Burgers validation trace:
  - `analysis_outputs/mechanism_20260622/replace_add_validation/raw/burgers_eps8_alpha0p3_steps100_N5_trace`
- Burgers first-order prediction:
  - `analysis_outputs/mechanism_20260622/replace_add_validation/diagnostics/burgers_first_order_prediction`
- Burgers landscape/ridge:
  - `analysis_outputs/mechanism_20260622/replace_add_validation/diagnostics/burgers_landscape_ridge_probe`
- NS2D direction stability:
  - `analysis_outputs/mechanism_20260622/replace_add_validation/diagnostics/ns2d_direction_stability`
- Darcy Flow flip-set mechanism:
  - `analysis_outputs/mechanism_20260622/full_mechanism_validation/darcy_flipset_mechanism`

## Queued Escalation Runs

These are no longer just optional notes. The directly runnable items have been
queued so they run after the current NS2D exact probe without interrupting it.

| Queued experiment | Output |
| --- | --- |
| Burgers N=20 core4 trace, eps=8, alpha=0.3, steps=100 | `analysis_outputs/mechanism_20260622/full_mechanism_validation/raw/burgers_eps8_alpha0p3_steps100_N20_trace` |
| Burgers N=20 first-order/full-budget prediction probe | `analysis_outputs/mechanism_20260622/full_mechanism_validation/burgers_first_order_prediction_N20` |
| Burgers N=20 landscape/ridge/boundary-width probe | `analysis_outputs/mechanism_20260622/full_mechanism_validation/burgers_landscape_ridge_probe_N20` |
| NS2D N=2 top-up trace for dataset indices 3,4 | `analysis_outputs/mechanism_20260622/full_mechanism_validation/ns2d/ns2d_eps32_alpha10_steps100_N2_topup_trace` |
| NS2D N=2 top-up direction/path stability | `analysis_outputs/mechanism_20260622/full_mechanism_validation/ns2d_direction_stability_N2_topup` |
| NS2D N=2 top-up exact first-order/linearity/boundary probe | `analysis_outputs/mechanism_20260622/full_mechanism_validation/ns2d_exact_mechanism_N2_topup` |
| Final cross-system mechanism summary rebuild | `docs/loss3_full_mechanism_validation_summary_20260623.md` |

## Not Queued Yet

These are not directly covered by the existing runnable scripts and would require
new implementation rather than simply launching an already-defined experiment.

| Not-yet-queued experiment | Reason |
| --- | --- |
| Explicit JVP/Jacobian quadratic-surrogate accuracy on NS2D | Needs a separate NS2D JVP/VJP probe implementation. |
| NS2D Jacobian spectrum / top singular-vector path study | Needs a dedicated recurrent-NS Jacobian/spectral script. |
| Explicit JVP/VJP replacement variants on NS2D | Needs a literal generalized-power-method variant implementation. |

## Rule Going Forward

The queued items are the current "run everything already runnable" set. If a
not-yet-queued item becomes necessary, that means we are adding a new
implementation, not discovering that an already runnable experiment was forgotten.
