# NS2D Recurrent Attack Alpha/Epsilon Tuning - 2026-05-22

## Scope

This note updates the attack parameter recommendation after inspecting the completed `loss2/all_a_target_w` result from the interrupted full ADW launch. The key question is whether the current `epsilon=32`, `alpha=1`, `steps=100` setting reaches the perturbation boundary quickly enough.

No attack run was launched while writing this note.

## Source Evidence

Observed from:

- `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/full_adw_b10_eps32_alpha1_20260522/mode_aaaaaaaaaw_p2_q2_20260522_030849_UTC/batch_0000_0009/loss2/raw_add/per_step_metrics.csv`
- `2D_NS_FNO2d_recurrent/visualizations/loss2_attack_fast_panels_20260522/loss2_fast_panel_report.json`
- `2D_NS_FNO2d_recurrent/visualizations/loss2_attack_full_final_panels_20260522/loss2_full_final_panel_report.json`

Observed values for the old `raw_add` setting:

- `epsilon=32`
- `alpha=1`
- `steps=100`
- final mean `delta_p = 12.0754`
- final mean boundary ratio `delta_p / epsilon = 0.3774`

## Boundary-Time Estimate

Inference from the observed `raw_add` curve:

- Current setting would need about `100 / 0.3774 = 265` steps to reach the L2 boundary.
- To reach the boundary by about 100 steps, increase `alpha / epsilon` by about `2.65x`.
- To reach the boundary by about 50 steps, increase `alpha / epsilon` by about `5.3x`.
- To reach the boundary by about 25 steps, increase `alpha / epsilon` by about `10.6x`.

This means the old pair `32:1` is too slow for additive methods. The next run should not use `epsilon=32, alpha=1` as the main setting.

## Recommended Pairs

The tuning should use paired values, not a Cartesian product, because `alpha` and `epsilon` must be coupled.

Recommended calibration sweep:

```bash
EPSILON_ALPHA_PAIRS="8:1.25 16:2.5 32:5 32:10"
```

Meaning:

- `8:1.25` keeps the same boundary-time ratio as `32:5`, but with a smaller perturbation budget.
- `16:2.5` is the middle, visually safer setting.
- `32:5` keeps the old perturbation budget but should reach the boundary around 50 steps for raw additive behavior.
- `32:10` is the aggressive setting and should reach the boundary around 25 steps.

If runtime needs to be smaller, run only:

```bash
EPSILON_ALPHA_PAIRS="16:2.5 32:5"
```

If the goal is to keep `alpha=1`, then the equivalent smaller-epsilon sweep is:

```bash
EPSILON_ALPHA_PAIRS="3.2:1 6.4:1 8:1"
```

But this tests a much smaller perturbation budget and may understate the possible attack effect.

## Recommendation

Use `EPSILON_ALPHA_PAIRS="8:1.25 16:2.5 32:5 32:10"` for the next calibration run. For a first short check, use `16:2.5 32:5`.

The earlier recommendation `16:0.5 32:1 64:2` is superseded because it preserves the old too-small `alpha / epsilon` ratio and would remain too slow to reach the boundary in 100 steps.

## Estimated Boundary Crossing For Recommended Pairs

This estimate uses the observed old `loss2` run with `epsilon=32`, `alpha=1`, and `steps=100`.

Observed old threshold evidence:

| method | old boundary evidence |
| --- | --- |
| raw_add | 25% at step 62; final 100-step boundary ratio `0.3774`; estimated 100% around step `265` |
| steepest_add | 25% at step 9; 50% at step 20; 75% at step 31; 100% at step 70 |
| raw_replace | effectively at boundary after the first update; exact 100% row appears by step 10 due floating-point thresholding |
| steepest_replace | same as raw_replace for `p=2`; effectively at boundary after the first update |

For the recommended paired sweep:

| epsilon:alpha | alpha/epsilon vs old | raw_add estimated 100% step | raw_add projected steps after boundary | steepest_add estimated 100% step | steepest_add projected steps after boundary | replace methods |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `8:1.25` | `5x` | about `53` | about `47` | about `14` observed-scaled, ideal `7` | about `86` to `93` | boundary from first update |
| `16:2.5` | `5x` | about `53` | about `47` | about `14` observed-scaled, ideal `7` | about `86` to `93` | boundary from first update |
| `32:5` | `5x` | about `53` | about `47` | about `14` observed-scaled, ideal `7` | about `86` to `93` | boundary from first update |
| `32:10` | `10x` | about `27` | about `73` | about `7` observed-scaled, ideal `4` | about `93` to `96` | boundary from first update |

Interpretation:

- The three `5x` ratio pairs should make raw additive behavior reach the boundary around the middle of a 100-step run, which is the intended calibration target.
- `32:10` is the aggressive setting: raw additive behavior should reach the boundary around the first quarter of the run.
- Replacement methods ignore `alpha` in the replacement step; they use `epsilon` as the radius and jump to the boundary immediately. Their purpose is a max-budget direction test, not step-size calibration.
- After a method reaches the boundary, projection keeps `||delta||_p` at `epsilon`; the remaining iterations optimize by moving along/changing direction on the boundary, not by increasing radial perturbation size.

## Stricter 50-Step Boundary Requirement

The target criterion is now stricter: additive attacks should reach the epsilon boundary within 50 steps, preferably earlier, because `loss3` can optimize more slowly than the completed `loss2` calibration run.

Inference from the old `loss2/raw_add` curve:

- Old `32:1` reaches the boundary at about `265` estimated steps.
- A `5x` ratio such as `32:5` estimates about `53` steps, which is borderline and may be too slow for `loss3`.
- A `10x` ratio such as `32:10` estimates about `27` steps, leaving about `73` projected boundary steps.
- A `15x` ratio such as `32:15` estimates about `18` steps, which is an aggressive backup if `loss3` still reaches the boundary too late.

Updated recommendation:

```bash
EPSILON_ALPHA_PAIRS="8:2.5 16:5 32:10"
```

Optional aggressive add-on:

```bash
EPSILON_ALPHA_PAIRS="8:2.5 16:5 32:10 32:15"
```

The older `8:1.25 16:2.5 32:5` set should be treated as a mild calibration set, not the main run, because it may only reach the raw-add boundary around step 50 or slightly after.

## Remaining Work

- Run a corrected attack calibration using the paired sweep.
- Inspect `delta_threshold_crossings.csv` to confirm the 25%, 50%, 75%, and 100% boundary-crossing steps.
- Use the new `final_state_outputs.npz` and `step_sample_trace.npz` records to check whether the perturbations are visually reasonable.
