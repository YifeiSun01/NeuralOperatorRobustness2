# FNO2d Recurrent Test Prediction GIF Visualization - 2026-05-21

Observed run after correcting the visualization format:

- Checkpoint visualized: `2D_NS_FNO2d_recurrent/saved_models/2D/modes64_modes64_width60_epochs500_Tin10_T10_recurrent_pytorch_20260521_090136_UTC/checkpoints/final.pt`.
- Test samples: `4`, `25`, `28`, `30`, `33`.
- Output directory: `2D_NS_FNO2d_recurrent/visualizations/fno2d_recurrent_test_predictions_20260521_final`.
- Each sample now has exactly one four-panel GIF: initial condition, solver / ground-truth frame, model output frame, and signed `model - solver` difference frame.
- No PNG files remain in the regenerated output directory.

Final checkpoint per-sample metrics:

| sample | relative L2 | MSE | four-panel GIF |
|---:|---:|---:|---|
| 4 | 0.0804970786 | 0.0124655962 | `2D_NS_FNO2d_recurrent/visualizations/fno2d_recurrent_test_predictions_20260521_final/sample_004_initial_solver_model_diff.gif` |
| 25 | 0.1197442338 | 0.0240913928 | `2D_NS_FNO2d_recurrent/visualizations/fno2d_recurrent_test_predictions_20260521_final/sample_025_initial_solver_model_diff.gif` |
| 28 | 0.0816519186 | 0.0111914380 | `2D_NS_FNO2d_recurrent/visualizations/fno2d_recurrent_test_predictions_20260521_final/sample_028_initial_solver_model_diff.gif` |
| 30 | 0.0518293418 | 0.0046042087 | `2D_NS_FNO2d_recurrent/visualizations/fno2d_recurrent_test_predictions_20260521_final/sample_030_initial_solver_model_diff.gif` |
| 33 | 0.1158824265 | 0.0222778656 | `2D_NS_FNO2d_recurrent/visualizations/fno2d_recurrent_test_predictions_20260521_final/sample_033_initial_solver_model_diff.gif` |

Inference from the five visualized samples: sample `30` is the easiest among these five by relative L2, while samples `25` and `33` are the hardest. The full final test metric remains the training-record value: test relative L2 mean `0.08654900729656219`.
