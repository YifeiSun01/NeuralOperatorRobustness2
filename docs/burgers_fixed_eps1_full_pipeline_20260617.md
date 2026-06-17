# Burgers Fixed-Epsilon Full Pipeline, 20260617

Status: prepared/running

This reruns Burgers 1024 adversarial/noise/clean training on the latest 52-dataset wide-parameter loss3-targeted set.

- Dataset root: `generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00`
- Train/test split: `1D_Burgers/datasets/1D/Burgers/batched_exponax_splits/...nu0.001...seed45`
- Baseline: `1D_Burgers/trained_models/attack_ready/burgers_nu0.001_fno1d_500/checkpoints/pytorch_fno1d_500.pt`
- Methods: `loss1`, `loss2`, `loss3`, `clean`, `random_clean_y`, `random_solver_y`
- Epsilon jitter: fixed `low=1.0`, `high=1.0` for every Burgers run
- Work-clock target: about 28800 seconds per training method
- Training curves: `visualizations/burgers_fixed_eps1_six_method_training_curves_20260617`
- Full clean/attack/SVD suite: `forensics/burgers_fixed_eps1_baseline_plus_six_full_suite_20260617`
- Preflight: `forensics/burgers_fixed_eps1_preflight_20260617`
