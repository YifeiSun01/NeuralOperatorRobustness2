# NS2D vs 1D Burgers Optimizer Curve Location Check - 2026-05-23

Status: inspected existing Markdown records and local filesystem only. No solver run, model inference, attack update, PyTorch import, JAX import, or GPU computation was started.

## User Question

The user asked where the earlier Wendy/1D Burgers optimizer loss-curve figures are, and whether the old Burgers loss results match the current NS2D observation where LP-steepest additive PGD (`steepest_add`) often looks best.

## Local Figure Availability

Observed from local filesystem checks:

- The documented Burgers figure folder `all_requested_figures_20260521/` is not present in the current local working tree.
- The documented source/output directories are also not present locally:
  - `forensics/loss1_loss2_core4_p2q2_baseline_marked_angles_20260521/`
  - `forensics/loss1_loss2_loss3_core4_combined_0to100_20260521/`
  - `forensics/loss3_alpha_epsilon_core4_baseline_giftrace_20260520/`
  - `forensics/loss3_alpha_epsilon_core4_sweep_p2q2_300steps_20260520/`
  - `forensics/loss3_alpha_epsilon_core4_analysis_p2q2_300steps_20260520/`

Observed from Markdown records, the figure paths should be:

- `all_requested_figures_20260521/`
- `forensics/loss1_loss2_loss3_core4_combined_0to100_20260521/combined_loss_mean_with_std_by_objective_0to100.png`
- `forensics/loss1_loss2_loss3_core4_combined_0to100_20260521/combined_loss_mean_with_std_by_method_0to100.png`
- `forensics/loss1_loss2_loss3_core4_combined_0to100_20260521/combined_loss_normalized_with_std_by_objective_0to100.png`
- `forensics/loss1_loss2_core4_p2q2_baseline_marked_angles_20260521/fno_nu0p001_eps4_alpha0p4_batch100_steps100_p2_q2/figures_0to100_marked_angles/`
- `forensics/loss3_alpha_epsilon_core4_analysis_p2q2_300steps_20260520/figures/` for the p2q2 alpha/epsilon sweep plots.

## Burgers Observed Results From Existing Docs

Observed from `docs/three_loss_burgers_optimizer_findings_summary_20260521.md`:

- In the saved 1D Burgers FNO three-loss records (`epsilon=8`, `alpha=0.3`, `p=q=2`, batch size 100, steps 100), generalized-power/replacement was the fastest early optimizer.
- k95 for original losses:
  - `loss1_original`: PGD `26`, LP-steepest PGD `27`, generalized power `3`.
  - `loss2_original`: PGD `24`, LP-steepest PGD `27`, generalized power `2`.
- The summary explicitly warns that GPI/replacement is the strongest early-speed claim, not unconditional final-loss dominance.

Observed from `docs/loss1_loss2_loss3_core4_combined_0to100_20260521.md`:

- For Burgers baseline `epsilon=4`, `alpha=0.4`, `p=q=2`, `k=0..100`:
  - `loss1`: final raw_add `6.977`, steepest_add `6.977`, raw/steepest_replace `6.942`.
  - `loss2`: final steepest_add `7.074`, raw_add `7.070`, raw/steepest_replace `7.045`.
  - `loss3`: final raw/steepest_replace `3.047`, steepest_add `2.988`, raw_add `2.877`.
- Boundary arrivals for loss3 baseline:
  - raw/steepest replacement hits boundary at step `1`.
  - steepest_add hits boundary at step `11`.
  - raw_add hits boundary at step `43`.

Observed from `docs/loss3_alpha_epsilon_core4_p2q2_300steps_result_20260520.md`:

- Across the 20-setting p2q2 300-step Burgers sweep:
  - mean final loss3_q: raw_add `5.659`, raw_replace `5.747`, steepest_add `5.827`, steepest_replace `5.747`.
  - median step to 95% of setting-best final loss: raw_add `94`, raw/steepest_replace `7`, steepest_add `36`.
  - median step to 99% boundary: raw/steepest_replace `1`, steepest_add `11`, raw_add `42.5`.

## Comparison To Current NS2D

Observed from current NS2D eps8/eps32 records:

- NS2D `eps8_alpha2p5`: by final true-loss increase, `steepest_add` wins 6/7 blocks.
- NS2D `eps32_alpha10`: `steepest_add` is much stronger than replacement in most W-heavy `loss3` modes and in `loss1/all_w`.

Inference:

- The old Burgers results are not identical to the current NS2D results.
- Burgers strongly supported replacement/GPI as the fastest early optimizer, especially at 25%/50%/early-loss milestones and boundary arrival.
- Burgers did have some long-run p2q2 sweep settings where `steepest_add` had the largest final mean loss, but the gap was modest and replacement remained much faster early.
- NS2D shows a much stronger and more persistent final true-loss advantage for `steepest_add`, especially in `loss3` W-heavy modes.

## Practical Interpretation

The user is right to be suspicious of the blanket statement "LP-steepest PGD is best." More precise wording is:

- In 1D Burgers, replacement/GPI-style methods were usually the fastest early optimizers and often very competitive or best for loss3 at 0..100.
- In longer Burgers p2q2 sweeps, `steepest_add` can slightly win final mean loss, but replacement/GPI still wins early speed.
- In current 2D NS, `steepest_add` is much more clearly the final true-loss winner in most tested blocks.
- Some of the apparent PGD discrepancy may also be step-scale/normalization: current `raw_add` is unnormalized PGD, while `steepest_add` is normalized-gradient PGD for `p=2`.

## Next Work

To view the old Burgers figures, restore/fetch the missing figure folders, especially `all_requested_figures_20260521/` or the `forensics/loss1_loss2_loss3_core4_combined_0to100_20260521/` tree, from R2 or the machine where they were generated.
