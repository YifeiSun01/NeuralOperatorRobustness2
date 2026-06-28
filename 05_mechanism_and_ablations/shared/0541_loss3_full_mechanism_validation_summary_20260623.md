# Loss3 Full Mechanism Validation Summary - 2026-06-23

This note summarizes the current replace/add mechanism evidence across Burgers, Darcy Flow, and NS2D.

## Artifacts

- Burgers validation: `analysis_outputs/mechanism_20260622/replace_add_validation`
- Burgers extended validation when present: `analysis_outputs/mechanism_20260622/full_mechanism_validation/burgers_first_order_prediction_N20`, `analysis_outputs/mechanism_20260622/full_mechanism_validation/burgers_landscape_ridge_probe_N20`
- Darcy flip-set validation: `analysis_outputs/mechanism_20260622/full_mechanism_validation/darcy_flipset_mechanism`
- NS2D exact validation: `analysis_outputs/mechanism_20260622/full_mechanism_validation/ns2d_exact_mechanism`
- NS2D top-up exact validation when present: `analysis_outputs/mechanism_20260622/full_mechanism_validation/ns2d_exact_mechanism_N2_topup`
- Summary CSV: `analysis_outputs/mechanism_20260622/full_mechanism_validation/cross_system_summary/mechanism_summary_rows.csv`

## Summary Table

| System | Evidence | Replace/Add Observation | Mechanism Read |
| --- | --- | --- | --- |
| Burgers 1D | first-order N=5; landscape N=5; formal curves N=100 | fast boundary/high early loss | replacement works mainly because boundary/high-loss region is forgiving; full-budget first-order prediction is not uniformly accurate |
| Darcy Flow | formal/flip-set N=20 | k90 replace 6.5 vs add 78.45 | replacement-like flip selection is much faster and ends higher on average; add and replace often choose different final flip sets |
| NS2D | direction N=3; exact probe N=3 | replacement reaches boundary immediately but loses after boundary in trace | if exact probe confirms poorer full-budget prediction/linearity or narrow arcs, this supports path-dependent add advantage |
| NS2D top-up | direction N=2; exact probe N=2; dataset indices 3,4 | top-up repeat of exact NS mechanism probes on held-out additional indices | additional NS samples check whether the N=3 exact-mechanism pattern repeats |

## Compact Numeric Rows

### Burgers 1D
- `evidence_n`: first-order N=5; landscape N=5; formal curves N=100
- `replace_speed_observation`: fast boundary/high early loss
- `first_order_add_rel_error_mean`: 0.02892
- `first_order_replace_rel_error_mean`: 3.407
- `ridge_min_steepest_replace_to_add`: 6.172
- `mechanism_read`: replacement works mainly because boundary/high-loss region is forgiving; full-budget first-order prediction is not uniformly accurate
- `source_first_order`: analysis_outputs/mechanism_20260622/replace_add_validation/diagnostics/burgers_first_order_prediction
- `source_landscape`: analysis_outputs/mechanism_20260622/full_mechanism_validation/burgers_landscape_ridge_probe_N20

### Darcy Flow
- `evidence_n`: formal/flip-set N=20
- `replace_speed_observation`: k90 replace 6.5 vs add 78.45
- `final_replace_loss_mean`: 0.04996
- `final_add_loss_mean`: 0.04518
- `final_add_replace_jaccard`: 0.2568
- `near95_method_count_mean`: 2.8
- `mechanism_read`: replacement-like flip selection is much faster and ends higher on average; add and replace often choose different final flip sets

### NS2D
- `evidence_n`: direction N=3; exact probe N=3
- `replace_speed_observation`: replacement reaches boundary immediately but loses after boundary in trace
- `first_order_add_rel_error_mean`: 12.45
- `first_order_replace_rel_error_mean`: 26.19
- `linearity_c_phi_steepest_replace_mean`: 14.2
- `arc_min_steepest_replace_to_add`: 85.09
- `replace_post_boundary_gain_mean`: -51.05
- `steepest_add_post_boundary_gain_mean`: 43.43
- `mechanism_read`: if exact probe confirms poorer full-budget prediction/linearity or narrow arcs, this supports path-dependent add advantage

### NS2D top-up
- `evidence_n`: direction N=2; exact probe N=2; dataset indices 3,4
- `replace_speed_observation`: top-up repeat of exact NS mechanism probes on held-out additional indices
- `first_order_add_rel_error_mean`: 7.883
- `first_order_replace_rel_error_mean`: 18.53
- `linearity_c_phi_steepest_replace_mean`: 6.629
- `arc_min_steepest_replace_to_add`: 114.5
- `replace_post_boundary_gain_mean`: -59.43
- `steepest_add_post_boundary_gain_mean`: 52.8
- `mechanism_read`: additional NS samples check whether the N=3 exact-mechanism pattern repeats
- `source_exact`: analysis_outputs/mechanism_20260622/full_mechanism_validation/ns2d_exact_mechanism_N2_topup
