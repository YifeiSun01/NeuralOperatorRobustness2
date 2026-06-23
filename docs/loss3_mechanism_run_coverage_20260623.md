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
| First-order/full-budget prediction | N=20 extended run done | flip-set analogue through speed/overlap/final set evidence | N=3 exact probe done; N=2 top-up queued | running/done |
| Multi-scale local linearity / nonlinearity | older finite-difference/curvature records + N=20 landscape running | optional flip analogue | N=3 exact probe done; N=2 top-up queued | running/done |
| Boundary high-loss width / near-optimal region | N=20 landscape/ridge running | N=20 near-optimal flip-set/overlap done | N=3 exact boundary arc done; N=2 top-up queued | running/done |
| Direction/path stability | older Burgers rotation/path records | N=20 final set overlap + sample trace stability done | N=3 direction trace done | done |
| Cross-system summary tying mechanism to optimizer performance | pending | pending | pending | queued after NS exact probe |
| Explicit Jacobian-vector / vector-Jacobian mechanism | not needed for Burgers core claim; older spectral/Jacobian evidence exists | not currently targeted | matrix-free NS2D implementation queued | queued |
| NS2D top singular-vector path / local surrogate accuracy | not needed for Burgers core claim; older spectral/Jacobian evidence exists | not currently targeted | N=3 + N=2 top-up matrix-free runs queued | queued |

## Running Now

- Extended mechanism queue:
  - PID: `217887`
  - Script: `tools/run_loss3_extended_mechanism_queue_20260623.sh`
  - Log: `analysis_outputs/mechanism_20260622/full_mechanism_validation/run_logs/extended_mechanism_queue_20260623.log`
  - PID file: `analysis_outputs/mechanism_20260622/full_mechanism_validation/run_logs/extended_mechanism_queue_20260623.pid`
  - Current child process: NS2D N=2 top-up trace for dataset indices 3,4.
- NS2D JVP/VJP mechanism queue:
  - PID: `236246`
  - Script: `tools/run_ns2d_jvp_vjp_mechanism_queue_20260623.sh`
  - Log: `analysis_outputs/mechanism_20260622/full_mechanism_validation/run_logs/ns2d_jvp_vjp_mechanism_queue_20260623.log`
  - PID file: `analysis_outputs/mechanism_20260622/full_mechanism_validation/run_logs/ns2d_jvp_vjp_mechanism_queue_20260623.pid`
  - Current state: waiting for extended mechanism queue PID `217887`, then runs the new NS2D JVP/VJP/top-singular probes.

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
- NS2D exact mechanism N=3:
  - `analysis_outputs/mechanism_20260622/full_mechanism_validation/ns2d_exact_mechanism`
  - Rows: first-order 420, linearity 450, boundary arc 81.
- Initial cross-system summary:
  - `docs/loss3_full_mechanism_validation_summary_20260623.md`
- Burgers N=20 core4 trace:
  - `analysis_outputs/mechanism_20260622/full_mechanism_validation/raw/burgers_eps8_alpha0p3_steps100_N20_trace`
- Burgers N=20 first-order/full-budget prediction:
  - `analysis_outputs/mechanism_20260622/full_mechanism_validation/burgers_first_order_prediction_N20`
- Burgers N=20 landscape/ridge/boundary-width:
  - `analysis_outputs/mechanism_20260622/full_mechanism_validation/burgers_landscape_ridge_probe_N20`

## Queued Escalation Runs

These are no longer just optional notes. The directly runnable items have been
queued so they run after the current NS2D exact probe without interrupting it.

| Queued experiment | Output |
| --- | --- |
| Burgers N=20 core4 trace, eps=8, alpha=0.3, steps=100 | done: `analysis_outputs/mechanism_20260622/full_mechanism_validation/raw/burgers_eps8_alpha0p3_steps100_N20_trace` |
| Burgers N=20 first-order/full-budget prediction probe | done: `analysis_outputs/mechanism_20260622/full_mechanism_validation/burgers_first_order_prediction_N20` |
| Burgers N=20 landscape/ridge/boundary-width probe | done: `analysis_outputs/mechanism_20260622/full_mechanism_validation/burgers_landscape_ridge_probe_N20` |
| NS2D N=2 top-up trace for dataset indices 3,4 | running: `analysis_outputs/mechanism_20260622/full_mechanism_validation/ns2d/ns2d_eps32_alpha10_steps100_N2_topup_trace` |
| NS2D N=2 top-up direction/path stability | `analysis_outputs/mechanism_20260622/full_mechanism_validation/ns2d_direction_stability_N2_topup` |
| NS2D N=2 top-up exact first-order/linearity/boundary probe | `analysis_outputs/mechanism_20260622/full_mechanism_validation/ns2d_exact_mechanism_N2_topup` |
| Final cross-system mechanism summary rebuild | `docs/loss3_full_mechanism_validation_summary_20260623.md` |
| NS2D N=3 explicit JVP/VJP, top singular path, local surrogate probe | queued after extended queue: `analysis_outputs/mechanism_20260622/full_mechanism_validation/ns2d_jvp_vjp_spectrum_N3` |
| NS2D N=2 top-up explicit JVP/VJP, top singular path, local surrogate probe | queued after extended queue: `analysis_outputs/mechanism_20260622/full_mechanism_validation/ns2d_jvp_vjp_spectrum_N2_topup` |

## Newly Implemented Matrix-Free NS2D Jacobian Probes

These were the previous "not directly runnable" items. They now have an
implementation and are queued behind the current extended mechanism queue.

| Implemented experiment | Implementation / output |
| --- | --- |
| Explicit JVP/Jacobian local surrogate accuracy on NS2D | `tools/probe_ns2d_jvp_vjp_spectrum_20260623.py`; outputs `quadratic_surrogate_rows.csv` and aggregate tables. |
| NS2D Jacobian spectrum / top singular-vector path study | Same script; outputs `spectrum_path.csv`, `power_iteration_rows.csv`, and `jvp_direction_rows.csv`. |
| Explicit JVP/VJP add-vs-replace candidate variants on NS2D | Same script; outputs `candidate_comparison_rows.csv` and aggregate tables. |
| Queue wrapper | `tools/run_ns2d_jvp_vjp_mechanism_queue_20260623.sh`, currently waiting for PID `217887`. |

## Rule Going Forward

The queued items are the current "run everything already runnable or now
implemented" set. Any future mechanism experiment beyond this document should be
treated as a new extension, not as an already-runnable item that was forgotten.
