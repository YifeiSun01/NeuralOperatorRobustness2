# Darcy/SIR20 Time-Matched Full Rerun

Status: pipeline implemented and smoke/full launchers prepared for the DarcyFlow-focused rerun.

## Scope

This rerun uses the Darcy screen baseline and six fine-tuning methods:

- `baseline`
- `loss1`
- `loss2`
- `loss3`
- `physics_loss`
- `random_clean`
- `random_solver`

The data root is `generalization_datasets_darcy_lossdrop50_selected_20260607`, with screen train/test overrides:

- Train: `2D_Darcy_FNO2d/datasets/grf_darcy_screen_20260607/train/dim2d_darcy_nx85_N384_solver=jaxcg_solve421_alpha2_tau3_binary3-12_f1_seed45_train.pt`
- Test: `2D_Darcy_FNO2d/datasets/grf_darcy_screen_20260607/test/dim2d_darcy_nx85_N96_solver=jaxcg_solve421_alpha2_tau3_binary3-12_f1_seed45_test.pt`
- Baseline: `2D_Darcy_FNO2d/saved_models/2D/darcy_screen_baseline_m64_w60_e50_20260607/best.pt`

Observed data note: the 50 selected lossdrop generalization datasets are soft-coefficient Darcy fields with 48 samples each. The robustness manifest records `requested_samples_per_dataset=50` and `selected_samples_for_dataset=48` for those datasets rather than inventing missing samples.

## Implementation

- `tools/adversarial_training.py` now writes `work_clock_epoch_summary.csv`; work-clock is `attack/random/solver target generation + forward/backward/optimizer step`, excluding evaluation/checkpoint/plot/upload time.
- `tools/adversarial_training.py` checkpoints now include `model_state_dict`, `optimizer_state_dict`, optimizer/global step, and work-clock metadata. Resume loads AdamW state when present.
- `tools/evaluate_generalization_models.py` now accepts finite Darcy coefficient fields so the soft lossdrop50 datasets evaluate correctly.
- SIR20 modules:
  - `tools/darcy_sir20_calibrate.py`
  - `tools/darcy_sir20_train_launcher.py`
  - `tools/darcy_sir20_evaluate.py`
  - `tools/darcy_sir20_robustness.py`
  - `tools/darcy_sir20_visualize.py`
  - `tools/run_darcy_sir20_timematched_full_20260614.sh`

## Commands

Smoke:

```bash
MODE=smoke TAG=20260614_smoke tools/run_darcy_sir20_timematched_full_20260614.sh
```

Full long run:

```bash
MODE=full TAG=20260614_full UPLOAD_TO_R2=1 AUTO_GIT_PUSH=1 tools/run_darcy_sir20_timematched_full_20260614.sh
```

The full run first calibrates each training method, using loss3 at 3000 epochs as the work-clock reference, then launches time-matched training, final 52-dataset evaluation, fixed-sample robustness, SVD/Jacobian diagnostics, figures, and optional R2/GitHub sync.

Outputs are written under `outputs/darcy_sir20_timematched_full_<tag>/` with:

- `figures/`
- `data/`
- `checkpoints_manifest/`
- `reports/`
- `logs/`
