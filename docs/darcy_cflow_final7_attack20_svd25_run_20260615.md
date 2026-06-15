# Darcy CFlow Final Seven-Model Attack20/SVD25 Run - 2026-06-15

## Status

Complete. The final seven-model robustness/Jacobian/SVD run finished, and the
automatic postprocess summary also finished.

This run is intended to replace the invalid smoke/partial/1000-epoch diagnostics
for robustness claims.

## Formal output folder

Use this folder for the final full run:

`outputs/darcy_cflow_final_robustness_20260615_full_attack20_svd25/`

Do not use the earlier default folder
`outputs/darcy_cflow_final_robustness_20260615/` as a final result folder. It
contains partial files from an intentionally timed-out foreground startup check.

## Configuration

- Attack objective: loss3 solver-consistent Darcy attack.
- Attack steps: `20`.
- Epsilon fraction: `0.025`.
- Requested samples per dataset: `50`.
- Actual samples:
  - screen train: `50`
  - screen test: `50`
  - each lossdrop50 generalization dataset: `48`, because the local dataset
    tensors have shape `[48, 85, 85, 1]`.
- Attack delta storage: one compressed NPZ per method/dataset.
- SVD/Jacobian sample set: fixed `25` samples shared by all seven models:
  train `2`, test `2`, and first `21` generalization datasets x one sample.
- SVD top-k stored: `10`.
- Block Jacobian projection: factor `2`, block Jacobian shape `1764 x 1764`.
- Row chunk: `128`.

Expected final output counts:

- Attack batches / delta NPZ files: `364` = seven models x 52 datasets.
- Attack sample rows: `17,500` = seven models x `(50 + 50 + 50*48)`.
- SVD/Jacobian rows: `175` = seven models x 25 samples.
- SVD vector NPZ files: `175`.

## Final checkpoints

Checkpoint manifest:

`outputs/darcy_cflow_final_robustness_20260615_full_attack20_svd25/checkpoints_manifest/final_7model_checkpoints.json`

The seven checkpoints are:

| method | epoch | checkpoint |
|---|---:|---|
| baseline | 0 | `2D_Darcy_FNO2d/saved_models/2D/darcy_screen_baseline_m64_w60_e50_20260607/best.pt` |
| loss1 | 3000 | `adversarial_training_runs/darcy_binary_loss3targeted_loss1_continue2000ep_from_1000ep_full50_timematched_20260612_stage2_2000_from_1000c/darcy/checkpoints/darcy_epoch3000_step003000.pt` |
| loss2 | 3079 | `adversarial_training_runs/darcy_binary_loss3targeted_loss2_continue2053ep_from_1026ep_full50_timematched_20260612_stage2_2000_from_1000c/darcy/checkpoints/darcy_epoch3079_step003079.pt` |
| loss3 | 3033 | `adversarial_training_runs/darcy_binary_loss3targeted_loss3_continue2022ep_from_1011ep_full50_timematched_20260612_stage2_2000_from_1000c/darcy/checkpoints/darcy_epoch3033_step003033.pt` |
| physics_loss | 3121 | `adversarial_training_runs/darcy_binary_loss3targeted_physics_continue2081ep_from_1040ep_full50_timematched_20260612_stage2_2000_from_1000c/darcy/checkpoints/darcy_epoch3121_step003121.pt` |
| random_clean | 3500 | `adversarial_training_runs/darcy_binary_random_binary_fixed_y_continue_to3500_from_3000_20260614_supervised/darcy/checkpoints/darcy_epoch3500_step003500.pt` |
| random_solver | 3500 | `adversarial_training_runs/darcy_binary_random_binary_solver_y_continue_to3500_from_3000_20260614_supervised/darcy/checkpoints/darcy_epoch3500_step003500.pt` |

## Preflight

Preflight folder:

`outputs/darcy_cflow_final_robustness_20260615_preflight/`

Preflight completed successfully:

- Attack rows: `28` = seven models x two datasets x two samples.
- SVD/Jacobian rows: `14` = seven models x two fixed samples.
- Delta NPZ files and vector NPZ files were written.
- The SVD CSV includes top-k singular values, top-k subspace angles, and
  pairwise cosine/angle/correlation for singular vector, `J^T error`, and attack
  delta.

## Ten-minute running checkpoint

Observed after roughly ten minutes of active tracking:

- Parent PID: `440473`.
- Python worker PID: `440496`.
- Attack batch log lines: `216 / 364`.
- Delta NPZ files: `216 / 364`.
- SVD rows: `0 / 175`; expected, because the script runs all attacks before the
  SVD/Jacobian stage.
- Completed attack stage through baseline, loss1, loss2, and loss3; physics loss
  had started.
- The process was still running.

Log file:

`outputs/darcy_cflow_final_robustness_20260615_full_attack20_svd25/logs/setsid_final7_attack20_svd25.out`

## Completion Evidence

Observed after completion:

- Main worker PID `440496` has exited.
- Postprocess waiter PID `445388` has exited.
- Waiter log recorded completion at `2026-06-15T03:03:56Z`.
- Attack log lines: `364 / 364`.
- SVD log lines: `175 / 175`.
- SVD rows by model:
  baseline `25`, loss1 `25`, loss2 `25`, loss3 `25`,
  Physics Loss `25`, random_clean `25`, random_solver `25`.

Final output counts:

- Attack sample rows:
  `17,500` in
  `data/robustness_attack_52datasets_samples.csv`.
- Attack delta NPZ files:
  `364` in `data/robustness_deltas/`.
- SVD/Jacobian rows:
  `175` in `data/svd_jacobian_metrics.csv`.
- SVD/Jacobian vector NPZ files:
  `175` in `data/svd_jacobian_vectors/`.
- Attack summary rows:
  `28` in `data/attack20_summary_by_model_split.csv`.
- Scalar correlation rows:
  `522` in `data/svd_attack_scalar_correlations.csv`.
- Vector alignment summary rows:
  `28` in `data/vector_alignment_summary_by_model.csv`.

The raw attack/SVD CSVs keep the machine key `physics_loss`; postprocessed
summary tables and reports display that method as `Physics Loss`.

SVD/Jacobian contents verified:

- `data/svd_jacobian_metrics.csv` contains `175` rows and non-null values for:
  `error_l2_norm`, `jt_error_l2_norm`, `sigma_input_right`, `block2_sigma1`,
  `block2_top_singular_values_json`, `attack_loss_increase`, and
  `attack_relative_increase`.
- The same table contains same-sample vector alignment metrics:
  `cos_singular_jt_error`, `angle_singular_jt_error_deg`,
  `corr_singular_jt_error`, `cos_singular_attack_delta`,
  `angle_singular_attack_delta_deg`, `corr_singular_attack_delta`,
  `cos_jt_error_attack_delta`, `angle_jt_error_attack_delta_deg`, and
  `corr_jt_error_attack_delta`.
- It also contains top-k singular-subspace alignment metrics:
  `topk_subspace_cos_jt_error`, `topk_subspace_angle_jt_error_deg`,
  `topk_subspace_cos_attack_delta`, and
  `topk_subspace_angle_attack_delta_deg`.
- Each row points to a vector NPZ in `data/svd_jacobian_vectors/`; those NPZ
  files store `x`, `y`, `pred`, `error`, `jt_error`, `attack_delta`,
  `input_right_singular_vector`, `output_left_singular_vector`,
  `block2_top_singular_values`, `block2_top_right_singular_vectors`,
  `block2_top_left_singular_vectors`, and
  `top_right_singular_vector_basis_full`.
- `data/svd_attack_scalar_correlations.csv` contains same-sample scalar
  correlations among singular-value diagnostics, `J^T error` norm, attack loss
  increase, and attack relative increase.
- `data/vector_alignment_summary_by_model.csv` contains by-model/by-split
  summaries of the singular-vector, `J^T error`, and attack-delta alignments.

## Automatic postprocess

The automatic postprocess waiter completed:

- Waiter log:
  `outputs/darcy_cflow_final_robustness_20260615_full_attack20_svd25/logs/postprocess_waiter.out`
- It waited for worker PID `440496` to exit.
- It then ran:
  `tools/summarize_darcy_final_robustness_20260615.py`

The postprocess created:

- `data/attack20_summary_by_model_split.csv`
- `data/svd_attack_scalar_correlations.csv`
- `data/vector_alignment_summary_by_model.csv`
- `reports/final_robustness_summary.md`

This specifically covers the scalar same-sample correlations among singular
value, `J^T error` norm, and attack loss increase/relative increase.

## Outputs to inspect after completion

- `data/attack_50sample_manifest.csv`
- `data/robustness_attack_52datasets_samples.csv`
- `data/robustness_deltas/*.npz`
- `data/svd_jacobian_25sample_manifest.csv`
- `data/svd_jacobian_metrics.csv`
- `data/svd_jacobian_vectors/*.npz`
- `reports/robustness_and_svd.md`
- `data/robustness_manifest.json`

## Code changes

- `tools/darcy_sir20_robustness.py`
  - Added top-k block SVD singular values/vectors.
  - Added top-k right-singular subspace angle metrics for `J^T error` and attack
    delta.
  - Preserved pairwise cosine, angle, and correlation among top singular vector,
    `J^T error`, and attack delta.
- `tools/run_darcy_cflow_final_robustness_20260615.sh`
  - New launcher that creates the final seven-model checkpoint manifest and runs
    preflight or full final robustness.
- `tools/summarize_darcy_final_robustness_20260615.py`
  - New postprocess summarizer for attack metrics, scalar correlations, vector
    alignment summaries, and the final Markdown robustness report.
