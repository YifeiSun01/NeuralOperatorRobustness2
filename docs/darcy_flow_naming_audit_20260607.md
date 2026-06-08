# Darcy Flow Naming Audit - 2026-06-07

## Status

Audited current Darcy-related text, scripts, records, and generated filenames for old misspelled or ambiguous Darcy labels. Corrected current user-facing Darcy labels to `Darcy Flow` at `2026-06-07T20:55:27Z`.

## Observed Evidence

- A wrong-spelling search over current `EXPERIMENT_LEDGER.md`, `docs`, `tools`, `visualizations`, `forensics`, and `generalization_datasets_darcy_lossdrop50_selected_20260607` returned no matches after correction, except this audit file before it was sanitized.
- `find` over the same current artifact roots found no filenames containing the known wrong-spelling patterns.
- A stale alternate-flow-label search over current `EXPERIMENT_LEDGER.md`, `docs`, `tools`, `visualizations`, `forensics`, and `generalization_datasets_darcy_lossdrop50_selected_20260607` returned no matches after correction, except this audit file before it was sanitized.
- Current Darcy-related scripts and reports now use `Darcy Flow` in user-facing labels, report titles, and future generated text.
- `find visualizations forensics docs -type f` found no current Darcy PNG files, and no PNG filenames containing known wrong or stale Darcy labels.

## Files Updated

- `EXPERIMENT_LEDGER.md`
- `docs/darcy_lossdrop50_loss3_500ep_launch_20260607.md`
- `docs/darcy_lossdrop50_selected_generalization_suite_20260607.md`
- `docs/darcy_candidate_generalization_gradient_screen_20260607.md`
- `docs/burgers_round03_baseline_vs_final_p2q2_attack_verification_20260607.md`
- `generalization_datasets_darcy_lossdrop50_selected_20260607/README.md`
- Darcy candidate-screen README files under `forensics/`
- Current Darcy helper scripts under `tools/`
- `tools/analyze_generalization_loss_patterns.py`
- `tools/adversarial_training.py`

## Inference

For current generated records and future current-script outputs, the task name is now consistently `Darcy Flow`, not the old misspelling. There were no current Darcy PNGs requiring regeneration, so no image bitmap text needed to be redrawn in this audit.
