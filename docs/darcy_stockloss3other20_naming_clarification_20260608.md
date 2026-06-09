# Darcy stockloss3other20 naming clarification - 2026-06-08

Status: clarified the meaning of the local run label `darcy_stockloss3other20training_20260608`. No new training, attack generation, model-forward evaluation, or solver-forward evaluation was run.

Observed evidence:

- Pipeline note: `docs/darcy_other30_stockgeneralization_loss3other20_pipeline_20260608.md`.
- Run summary doc: `docs/darcy_stockloss3other20training_20260608.md`.
- Run config: `adversarial_training_runs/darcy_stockloss3other20training_20260608/run_config.json`.

Observed meaning:

- `stockloss3other20training` is a run name, not a new loss method.
- The run used Darcy attack objective `loss3` for `20` epochs.
- It initialized from the `other30training` Darcy checkpoint: `2D_Darcy_FNO2d/saved_models/2D/darcy_flow_other30training_20260608/best.pt`.
- It evaluated against the `stockgeneralization` root: `generalization_datasets_darcy_stockgeneralization_20260608`.

Inference:

- The label is shorthand for a separate experiment: stock/generalization setup + loss3 adversarial self-training + 20 epochs, initialized from the other30 checkpoint.
- It is not part of the four formal Darcy/C-flow loss-method comparison (`loss1`, `loss2`, `loss3`, `physics/loss4`) and should not be plotted as a fifth method in that comparison.
