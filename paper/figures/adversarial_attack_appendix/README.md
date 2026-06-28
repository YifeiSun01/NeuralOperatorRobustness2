# Solver-Integrated Attack and Training Appendix Figures

This directory is the user-corrected, cropped figure set for the
solver-integrated attack and training appendix.

The slide-number mapping here follows the manual corrections from
2026-06-20, not the earlier automatic extraction that shifted several slide
numbers and domains.

## Layout

- `attack/`: solver-integrated attack figures.
  - `attack/burgers/fno/`: Burgers FNO attack figures, split into
    `loss_progression/` and `final_prediction_delta/`.
  - `attack/burgers/deeponet/`: Burgers DeepONet attack figures, split into
    `loss_progression/` and `final_prediction_delta/`.
  - `attack/burgers/optimizer_diagnostics/`: recovered non-slide-named
    Burgers attack diagnostics for boundary-ratio and update-angle behavior.
  - `attack/burgers/svd_diagnostics/`: local SVD singular-vector diagnostics
    comparing FNO, DeepONet, and solver behavior.
  - `attack/cross_benchmark/epsilon_sweeps/`: recovered non-slide-named
    attack epsilon-sweep diagnostics spanning Burgers, Darcy Flow, and NS2D.
  - `attack/darcy_flow/`: Darcy Flow attack figures, split into
    `loss_progression/` and `final_prediction_delta/`.
  - `attack/navier_stokes/`: Navier--Stokes attack figures, split into
    `loss_progression/`, `final_prediction_delta/`, and
    `periodic_boundary_evidence/`.
- `training/`: solver-integrated adversarial-training figures.
  - `training/burgers/`: Burgers training/generalization curves and training
    diagnostics. Slides 48--53 are all Burgers-related. Recovered
    non-slide-named epsilon attack sweeps are under
    `training/burgers/epsilon_attack_sweeps/`.
  - `training/darcy_flow/`: Darcy Flow training/generalization curves and
    training diagnostics. Slide 54 starts the Darcy Flow block, slides 55--56
    remain Darcy Flow, and slide 57 is saved under
    `training/darcy_flow/training_diagnostics/`. Recovered non-slide-named
    epsilon attack sweeps are under `training/darcy_flow/epsilon_attack_sweeps/`.
  - `training/navier_stokes/`: Navier--Stokes/FNO2d training or
    generalization diagnostics. Slide 59 is NS.

## Important Corrections

- Slide 18 is Burgers FNO L2, so the filename explicitly includes `l2`.
- Slide 19 is a Burgers final-state comparison for `Steepest Add`, not a
  boundary/angle diagnostic.
- Slide 25 is Burgers DeepONet Linf loss progression, not a generic Burgers
  plot.
- Slide 24 is the second Burgers DeepONet Linf loss-progression figure and is
  saved next to slide 25 under `attack/burgers/deeponet/loss_progression/`.
- Slide 27 is Darcy Flow, not NS.
  - Slides 28, 30, 34, 35, and 37 are marked in the manifest as part of the
    periodic-boundary evidence group.
  - Current LaTeX Figure 24 uses the NS2D Steepest Add
    $\epsilon=32,\alpha=10$ final-state comparison image requested by the user.
  - The old current Figures 29--30 were cropped to their right-hand `[Auto]`
    columns and combined into one auto-field comparison for $\alpha=50.0$ and
    $\alpha=2.5$.
  - Slide 48 is Burgers adversarial training, not NS, and is named consistently
    with slides 49--50.
- Slide 57 belongs to `Training / Darcy Flow`, not the attack block; it is saved
  under `training/darcy_flow/training_diagnostics/`.
- Non-slide-named PPT-frame sources were audited after the old
  `ppt_last_frames` folder was removed. The recovered figures are saved under
  `attack/burgers/optimizer_diagnostics/`,
  `attack/burgers/svd_diagnostics/`,
  `attack/burgers/fno/loss_progression/`, and
  `attack/cross_benchmark/epsilon_sweeps/`.
- The training epsilon attack sweeps for Burgers and Darcy Flow were recovered
  from the 2026-06-19 final PNG bundle and saved under the corresponding
  `training/*/epsilon_attack_sweeps/` directories.

## Manifests

- `manifest.csv`: corrected filename, crop dimensions, description, and user
  correction note for each cropped output.
- `legacy_name_audit_20260620.csv`: audit from old descriptive filenames to
  the corrected files.
- `non_slide_recovery_audit_20260620.csv`: recovery audit for non-slide-named
  sources that may have been present in the removed PPT-frame folder.
- `excluded_or_ambiguous_sources.csv`: extracted files not used in this
  corrected set because their slide numbers came from a shifted/temporary
  extraction or no local source image was found.
- `recovered_non_slide_contact_sheet_20260620.png`: visual check sheet for the
  recovered non-slide sources.
- `recovered_training_epsilon_sweeps_contact_sheet_20260620.png`: visual check
  sheet for the recovered Burgers and Darcy Flow training epsilon attack
  sweeps.

The previous automatic/misnamed extracted sets have been removed so they are
not used accidentally.
