# Burgers Generalization Round00/Round01 Direction Analysis

Date: 2026-06-05

This note compares the two existing Burgers generalization-search datasets and records the direction for the next, more aggressive search. No new training run was launched for this analysis. An accidental partial `round_02` generation was stopped and is not used as evidence.

## Evidence Read

- Round00 dataset: `generalization_datasets_burgers_aggressive_loss3_search/round_00`
- Round00 eval: `generalization_eval_burgers_aggressive_loss3_search_round00/metrics_sorted_by_similarity.csv`
- Round00 gradient alignment: `forensics/burgers_aggressive_round00_loss123_gradient_alignment_10step_20260604/`
- Round01 dataset: `generalization_datasets_burgers_loss3_aligned_search/round_01`
- Round01 eval: `generalization_eval_burgers_loss3_aligned_search_round01/metrics_sorted_by_similarity.csv`
- Round01 gradient alignment: `forensics/burgers_loss3_aligned_round01_loss123_gradient_alignment_10step_20260604/` and `forensics/burgers_loss3_aligned_round01_loss123_gradient_alignment_50step_20260604/`
- Round01 training reports: `forensics/burgers_loss3_aligned_round01_report_20260604.md` and `forensics/burgers_loss3_aligned_round01_epoch_wallclock_rmse_report_20260604.md`
- Round01 Jacobian/SVD: `forensics/burgers_loss3_aligned_round01_final_jacobian_svd_rep20_top100_20260604/`
- Mechanism probes: `forensics/burgers_p2q2_loss123_delta_manifold_probe_20260603/`, `forensics/burgers_p2q2_loss123_gradient_alignment_probe_20260604/`, `forensics/burgers_p2q2_epoch1_jump_causal_probe_20260604/`, and `forensics/burgers_p2q2_loss123_50step_input_similarity_probe_20260604/`

## Observed Dataset Differences

Round00 is a hand-crafted aggressive set. Its 50 datasets include centered scale/shift, positive/negative shifts, sawtooth injections, sign-centered patterns, and kernel/spectrum shifts. The baseline retained Burgers model sees very large generalization error on this set: mean generated RMSE `0.3743`, mean relative L2 `0.7405`, with worst relative L2 `1.6078`. Geometry is often far outside the training range; for example centered range datasets have mean `oob_mean = 0.3641`, and sign-pattern datasets have mean `oob_mean = 0.6204`.

Round01 is a loss3-aligned adversarial set. All 50 datasets are generated from raw loss3 adversarial inputs with `epsilon_fraction = 0.06`, 5 attack steps, and train sources. It is much less extreme in ordinary range terms: mean `x_adv` range is about `[-0.2873, 1.3078]`, `oob_mean = 0.0060`, `delta_linf_mean = 0.2807`, and `linf/RMS = 4.6810`. Baseline generalization error is moderate: mean generated RMSE `0.04254`, mean relative L2 `0.07883`.

## Observed Performance

Round00 is hard but not loss3-selective. The 10-step gradient alignment to `generalization_mixed50` is near zero for all three raw variants: loss1 mean cosine `-0.0155`, loss2 `0.0374`, loss3 `0.0342`. Inference from this evidence: round00 mostly creates distribution shifts that are not the loss3 training direction; making loss large alone did not make loss3 first-order helpful.

Round01 is loss3-selective in gradient space. In 10 steps, generalization cosine means are loss1 `0.3101`, loss2 `0.4428`, loss3 `0.7118`; in 50 steps they are loss1 `0.2357`, loss2 `0.1667`, loss3 `0.7354`. Loss3 has only `1/50` negative generalization-cosine steps, while loss1 has `19/50` and loss2 has `22/50`.

Round01 also shows a real prediction-loss signal, but not a clean final-win signal. Observed from `eval_split_summary.csv`: loss3 reaches the best observed round01 generalization RMSE at epoch 463 (`0.007302`), better than loss1's best (`0.008785`) and loss2's best (`0.009930`). But at final checkpoints, loss1 epoch1000 has gen RMSE `0.009127`, loss3 epoch500 has `0.009939`, and loss2 epoch500 has `0.012053`. Equal wall-clock comparisons still favor loss1/loss2 because loss3 is much slower.

Round01 Jacobian/SVD is mixed. On generalization, loss3 has the best top-10/top-20 right model-subspace similarity to the solver: top-10 right mean cosine `0.9907` for loss3 vs `0.9811` for loss1 and `0.9807` for loss2; top-20 right mean cosine `0.9025` for loss3 vs about `0.805` for loss1/loss2. However, `||J_model - J_solver||_2` is still lower for loss1: generalization mean `1.3575` for loss1, `1.6195` for loss3, `1.7846` for loss2.

## Direction For The Next Dataset

The next dataset should be more aggressive than round01, but not in the round00 style of arbitrary large range shifts. The useful target is:

- preserve the raw loss3 adversarial geometry that gave strong gradient cosine;
- increase strength enough that loss1/loss2 cannot ride along on clean-like improvement;
- filter candidates by first-order selectivity, not by raw loss size alone.

Practical target bands for the next search:

- baseline generated RMSE roughly `0.06` to `0.15`, not round00-like `0.3+`;
- `oob_mean` roughly `0.008` to `0.03`, much stronger than round01 but far below round00 centered/sign shifts;
- `x_adv` range around `[-0.45, 1.55]` to `[-0.60, 1.70]`, watched for solver instability;
- `linf/RMS` near `4.5` to `6.0`, because loss3's mechanism is peakier than loss1/loss2;
- loss3 generalization gradient cosine mean above about `0.65`, while loss1/loss2 should be much lower or include negative steps.

Suggested generation strategy before any long training:

1. Generate multiple small candidate pools using loss3 raw attacks with `epsilon_fraction` around `0.08`, `0.10`, and `0.12`, 8 to 12 attack steps, mixed train/test source samples.
2. Score candidate datasets by gradient selectivity: high loss3 cosine to the candidate generalization gradient, low or negative loss1/loss2 cosine.
3. Keep only the top 50 selected datasets for a full candidate round.
4. Run only 5/10 epoch screening first. Require loss3 to improve candidate generalization faster than loss1/loss2 before any 50 epoch training.
5. For Jacobian/SVD, start with rep5/top20 on the new candidate points; do not reuse old solver or baseline Jacobians.

## Partial Round02 State

Observed on 2026-06-05: an accidental partial generation process for `generalization_datasets_burgers_loss3_aligned_search/round_02` was stopped. It left 10 `.pt` files and no manifest/summary. These files are incomplete and are not used in this analysis. A future official round should either overwrite/rebuild `round_02` deliberately or use a fresh round id.
