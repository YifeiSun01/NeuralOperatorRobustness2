# Burgers Wideparam SVD25 -> Analysis -> Retrain Automation - 2026-06-11

This records the current automation contract for the Burgers wide-parameter loss3-targeted workflow.

## Main Driver

`tools/run_burgers_wideparam_svd25_then_retrain_upload_20260611.sh`

Default order:

1. Run full `1024 x 1024` SVD/attack workflow for 25 samples via `tools/run_burgers_wideparam_full1024_svd_attack25_reuse3_20260611.sh`.
2. Use the matched rows in `svd_attack_joined_metrics.csv` to analyze SVD/attack consistency on the same samples.
3. Run biased local direction analysis with `tools/analyze_burgers_biased_local_attack_direction_20260611.py`, computing top-SVD, `A^T b`, finite-radius affine trust-region directions, their angles, local gains, and correlations with attack loss growth/final MSE.
4. Upload SVD outputs, biased-direction outputs, reports, and logs to R2 through `tools/upload_path_to_r2_20260525.sh`.
5. Commit/push tracked code, markdown, and small analysis tables to GitHub when `RUN_GIT=1`.
6. Run Burgers loss retraining using `tools/run_burgers_wideparam_loss123_retrain_20260611.sh`.
7. Plot merged retrain outputs, including the polished report-style figures.
8. Upload retrain runs, logs, visualizations, and report to R2, then commit/push tracked code/markdown.

## Retrain Order

`tools/run_burgers_wideparam_loss123_retrain_20260611.sh` now defaults to:

```bash
RETRAIN_ORDER=loss3_then_loss12
```

That means:

1. Run `loss3` first.
2. Then run `loss1` and `loss2`; by default they run in parallel when `RUN_LOSS12_PARALLEL=1`.

The old order is still available explicitly:

```bash
RETRAIN_ORDER=loss12_then_loss3
```

## Key Switches

- `RUN_SVD=1`: run or reuse SVD25 output.
- `RUN_SVD_ANALYSIS=1`: after SVD25, run biased local direction/correlation analysis.
- `SVD_ANALYSIS_DEVICE=cpu`: default device for residual recomputation in biased analysis.
- `RUN_RETRAIN=1`: run loss1/loss2/loss3 retraining.
- `RUN_UPLOAD=1`: upload generated artifacts to R2.
- `RUN_GIT=1`: commit and push tracked code/markdown/small result tables.
- `SMOKE_ONLY=1`: estimate/check flow without full SVD or full retraining.

## SVD/Attack Correlation Outputs

The SVD25 workflow writes matched sample/model rows, so SVD metrics and attack loss growth use the same samples:

- `forensics/burgers_wideparam_loss3targeted_full1024_svd_attack25_reuse3_20260611/svd_attack_joined_metrics.csv`
- `forensics/burgers_wideparam_loss3targeted_full1024_svd_attack25_reuse3_20260611/svd_attack_correlations.csv`

The biased local direction follow-up writes:

- `forensics/burgers_wideparam_loss3targeted_full1024_svd_attack25_biased_local_direction_20260611/biased_direction_metrics.csv`
- `forensics/burgers_wideparam_loss3targeted_full1024_svd_attack25_biased_local_direction_20260611/biased_direction_correlations.csv`
- `docs/burgers_wideparam_loss3targeted_full1024_svd_attack25_biased_local_direction_20260611.md`

## Note About Current Running Process

The long-running PID launched before this note may already have read portions of the old main driver. The retrain order is still protected because the retrain script itself now defaults to `loss3_then_loss12`. If the already-running main driver does not execute the newly inserted SVD biased-direction hook after SVD25, the same analysis can be run posthoc with `RUN_SVD=0 RUN_RETRAIN=0 RUN_SVD_ANALYSIS=1` or by directly calling `tools/analyze_burgers_biased_local_attack_direction_20260611.py` against the completed SVD25 root.
