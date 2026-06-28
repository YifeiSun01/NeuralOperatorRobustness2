# Darcy Flow Loss-Drop Selected 50 - 2026-06-07

Observed criterion: each selected dataset has `window=first50` and `eval_loss_delta < 0` in the pool gradient screen.

Source screen summary: `forensics/darcy_stockgeneralization_gradient_screen_20260608/candidate_screen_summary.csv`
Source pool manifest: `generalization_datasets_darcy_stockgeneralization_pool_20260608/candidate_manifest.csv`

Files:

- `candidate_manifest.csv`: selected 50 dataset manifest with observed first50 metrics.
- `selection_summary.csv`: loss/cosine evidence used for selection.
- `darcy/*.pt`: selected dataset files, hardlinked when possible from the pool root.
