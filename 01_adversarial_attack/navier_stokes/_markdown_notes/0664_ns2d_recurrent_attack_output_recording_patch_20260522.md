# NS2D Recurrent Attack Output Recording Patch - 2026-05-22

## Scope

This patch changes the NS2D recurrent FNO core4 attack so future attack runs archive the visual and numeric state needed to inspect whether `epsilon` and `alpha` are reasonable.

No attack experiment was launched while making this patch. The change was checked with Python compilation and shell syntax checks only.

## Modified Files

Observed source changes:

- `2D_NS_FNO2d_recurrent/perturbation_methods/attack_ns2d_recurrent_core4.py`
- `2D_NS_FNO2d_recurrent/perturbation_methods/run_ns2d_recurrent_core4_attack.sh`

## New Default Recording Behavior

For every `loss_type / method` output directory, the attack now records three levels of evidence.

### Existing Scalar Curves

These already existed and remain the primary loss-curve records:

- `per_step_metrics.csv`
- `per_sample_step_metrics.csv`
- `delta_threshold_crossings.csv`

They record surrogate loss, true all-W loss, delta L2/Linf/P norms, boundary ratio, gradient norm, direction norm, growth from step 0, and per-sample curves.

### Full Batch Final State Archive

New files:

- `final_state_outputs.npz`
- `final_state_metrics.csv`

This is saved for every sample in the attack batch, for example all 10 samples when `ATTACK_BATCH_SIZE=10`.

`final_state_outputs.npz` contains:

- `dataset_indices`
- `x_clean`
- `final_delta`
- `x_adv`
- `clean_model_final`
- `clean_solver_final`
- `clean_model_minus_solver`
- `adv_model_final`
- `adv_solver_final`
- `adv_model_minus_solver`
- `model_final_change`
- `solver_final_change`
- `clean_true_loss`
- `adv_true_loss`

`final_state_metrics.csv` contains per-sample scalar summaries including delta L2/Linf, clean and adversarial true loss, true-loss ratio, clean/adv FNO-vs-solver L2, model final change L2, and solver final change L2.

### Per-Step Trace For One Batch Sample

New files:

- `step_sample_trace.npz`
- `step_sample_trace_metrics.csv`

By default this records `sample_position=0` inside each attack batch. For a batch of 10, this means the first sample in that batch gets a full per-step visual trace.

`step_sample_trace.npz` contains:

- `sample_position`
- `dataset_index`
- `k`
- `x_clean`
- `clean_model_final`
- `clean_solver_final`
- `clean_model_minus_solver`
- `clean_true_loss`
- `delta[k]`
- `x_adv[k]`
- `grad[k]`
- `grad_available[k]`
- `direction[k]`
- `direction_available[k]`
- `adv_model_final[k]`
- `adv_solver_final[k]`
- `adv_model_minus_solver[k]`

`step_sample_trace_metrics.csv` mirrors the scalar row records for that selected sample and the recorded steps.

## New CLI Options

Added to `attack_ns2d_recurrent_core4.py`:

```bash
--record-final-state-outputs / --no-record-final-state-outputs
--record-step-sample-outputs / --no-record-step-sample-outputs
--record-step-sample-position 0
--record-step-sample-every 1
--record-step-sample-gradients / --no-record-step-sample-gradients
```

Defaults are intentionally on:

```bash
--record-final-state-outputs
--record-step-sample-outputs
--record-step-sample-position 0
--record-step-sample-every 1
--record-step-sample-gradients
```

The wrapper exposes matching environment variables:

```bash
RECORD_FINAL_STATE_OUTPUTS=1
RECORD_STEP_SAMPLE_OUTPUTS=1
RECORD_STEP_SAMPLE_POSITION=0
RECORD_STEP_SAMPLE_EVERY=1
RECORD_STEP_SAMPLE_GRADIENTS=1
```

## Runtime And Storage Implications

Observed by code inspection:

- Full-batch final recording adds two all-W final solver/model evaluations per `loss_type / method`: one for clean input and one for final adversarial input.
- Per-step sample tracing reuses the all-W true-loss evaluation path when it is already being computed. With `TRUE_LOSS_EVERY=1`, it should not require a separate additional solver call for the selected sample at each attack step.
- It does add CPU-side array storage for the selected sample at each recorded step.

Inference:

- This will increase output size substantially, especially with `RECORD_STEP_SAMPLE_EVERY=1` and gradients enabled.
- If storage or I/O becomes too large, the first conservative reduction should be `RECORD_STEP_SAMPLE_EVERY=5` or `RECORD_STEP_SAMPLE_GRADIENTS=0`; keep `RECORD_FINAL_STATE_OUTPUTS=1` for visual inspection of all batch samples.

## Verification

Observed checks:

```bash
adv_robust/bin/python -m py_compile 2D_NS_FNO2d_recurrent/perturbation_methods/attack_ns2d_recurrent_core4.py
bash -n 2D_NS_FNO2d_recurrent/perturbation_methods/run_ns2d_recurrent_core4_attack.sh
```

Both checks passed.

## Remaining Work

- Run the corrected attack command when ready.
- After the run, use the saved `final_state_outputs.npz` and `step_sample_trace.npz` to generate final panels and per-step GIFs/curves for scale inspection.
