# Burgers Self-Training Loss1/Loss2/Loss3 Dataset-Dependence Conclusions

Date: 2026-06-08

This note records the qualitative conclusion from the Burgers self-training experiments comparing the baseline model against loss1, loss2, and loss3 self-trained models. The key point is that the apparent advantage of loss3 depends strongly on what generalization dataset is used and on whether the metric is clean generalization loss or finite-budget adversarial robustness.

## Related Records

Observed source/record files:

- `docs/burgers_p2q2_loss1_loss2_loss3_training_pipeline_20260602.md`
- `docs/burgers_round03_clean_loss_ratio_interpretation_20260608.md`
- `forensics/burgers_round03_full_p2q2_52datasets_4models_finalonly_20step_20260607/summary_by_model_dataset.csv`
- `forensics/burgers_round03_full_p2q2_clean_loss_per_dataset_20260608/clean_and_attack_loss_ratios_by_dataset.csv`
- `forensics/burgers_round03_full_p2q2_clean_loss_per_dataset_20260608/sample_weighted_before_after_solver_gap_summary.csv`

## What Was Being Compared

The three Burgers self-training variants differ in the adversarial sample generation objective:

- `loss1`: attack makes `model(x_adv)` differ from `model(x_clean)`.
- `loss2`: attack makes `model(x_adv)` differ from `solver(x_clean)`.
- `loss3`: attack makes `model(x_adv)` differ from `solver(x_adv)`, so this is the solver-consistent attacked-target objective.

The important control is that, after adversarial samples are generated, the self-training update is still solver-consistent: the model is trained on attacked inputs against solver-generated labels. So the comparison is primarily about which adversarial samples are chosen, not about changing the final supervised target.

## Main Qualitative Finding

The headline conclusion is not simply "loss3 always wins." A more accurate conclusion is:

**Loss3 has the clearest advantage when the generalization dataset is meaningfully different from train/test, especially when the input-function range is broader. When the generalization data are too close to the original test distribution, loss1 and loss2 can also reduce clean generalization loss quickly, so loss3's advantage is muted.**

This matters because the first generalization datasets we tried were not far enough from the test distribution. Their input ranges and overall distributional character were close to train/test. Under that setup, loss1 and loss2 both looked surprisingly competitive on clean generalization metrics: they could push generalization loss down quickly, sometimes faster than loss3 in the early epochs.

When we later used a wider-range, more shifted generalization set, loss3's advantage became more visible. That supports the idea that loss3 is more valuable when the selected generalization samples genuinely stress the model/solver relationship, not merely when they are mild variants of the original distribution.

## Early Dataset Issue

Observed problem from the earlier generalization-set experiments:

- The first generalization datasets had ranges that were not very different from the original train/test range.
- Because the generalization samples were close to the test distribution, the clean generalization metric was not a strong discriminator among loss1, loss2, and loss3.
- Loss1/loss2 could reduce the generalization loss quickly even though their attack objectives are less solver-consistent than loss3.
- Therefore, using these mild generalization datasets made loss3 look less special than expected.

Inference:

- If the generalization set is too easy or too close to test, clean loss reduction mostly measures ordinary adaptation to nearby inputs. In that regime, cheaper objectives such as loss1/loss2 can look nearly as good as loss3.
- To reveal the advantage of loss3, the generalization set needs to include samples whose solver response changes in a way that model-only or clean-target attacks do not fully capture.

## First-Epoch Behavior

Observed behavior in the training curves:

- Loss1 and loss2 often begin by directly lowering the clean evaluation loss.
- Loss3 often shows a first-epoch spike: the loss rises sharply at the beginning and then starts decreasing afterward.
- This first-step/first-epoch spike appears not only on generalization curves but also on train/test curves in several runs.

Interpretation:

- Loss3 uses attacked solver targets, so the early training steps can move the model into a harder solver-consistent region before it has adapted to the new attacked distribution.
- Loss1/loss2 choose adversarial samples using easier/fixed targets, so they can look smoother and faster at the beginning.
- The first-epoch spike should not be read as loss3 failing; it is a transient cost of using the harder objective. But it does mean early-epoch clean loss alone can make loss3 look worse than it is.

## Third/Wider-Range Dataset

Observed qualitative change after moving to the wider-range generalization dataset:

- Loss3's advantage became clearer on generalization metrics.
- The advantage was still not equally obvious on train/test, because train/test remain close to the original training distribution.
- This supports the interpretation that loss3 is most useful for distributional stress tests rather than for nearby clean test points.

Inference:

- The more the generalization set forces true solver-consistent extrapolation, the more meaningful loss3 becomes.
- If the dataset only perturbs the original distribution mildly, loss1/loss2 may produce enough regularization to improve clean loss, making loss3's extra solver-gradient cost less visibly beneficial.

## Clean Accuracy Versus Robustness

The later round03 attack evaluation helps separate two questions.

Observed from `docs/burgers_round03_clean_loss_ratio_interpretation_20260608.md` and the 52-dataset p2q2 20-step attack summary:

- Before attack, loss1 has the smallest clean model-solver RMS gap on `52/52` datasets.
- Before attack, loss2 is also better than loss3 on clean RMS gap.
- After the same p2q2 attack, loss3 has the smallest model-solver RMS gap on `52/52` datasets.
- Sample-weighted before-attack RMS ratios versus baseline are approximately loss1 `0.100384`, loss2 `0.162950`, loss3 `0.390539`.
- Sample-weighted after-attack RMS ratios versus baseline are approximately loss1 `0.337486`, loss2 `0.380699`, loss3 `0.238735`.

Interpretation:

- Loss1/loss2 can be better for clean/pre-attack solver gap.
- Loss3 is better for finite-budget attacked/post-attack solver gap.
- So the answer to "which self-training is best" depends on whether we are asking about clean generalization or adversarial robustness.

## Practical Conclusion

The current best summary is:

1. Loss3 is not universally dominant on every clean generalization curve.
2. Loss1 and loss2 can reduce clean generalization loss quickly when the generalization dataset is close to train/test.
3. Loss3 can show an early clean-loss spike before it starts improving.
4. Loss3's advantage becomes clearer when the generalization dataset has a larger input range and is more genuinely out-of-distribution.
5. Loss3 is much more convincing when evaluated under the same finite-budget attack: after attack, loss3 gives the lowest model-solver RMS gap across the 52-dataset round03 evaluation.
6. Therefore, the scientific conclusion depends strongly on generalization dataset construction. Mild generalization datasets understate the value of loss3; wider-range/stress-test datasets reveal it more clearly.

## Recommendation For Future Burgers Reports

Future Burgers self-training comparisons should always report three separated views:

- Clean train/test/generalization loss over epochs.
- Finite-budget attacked loss under the same p2q2 attack protocol.
- Dataset range/distribution statistics for the selected generalization set.

Without the third item, it is too easy to mistake "loss3 has no strong advantage" for a method-level conclusion, when it may actually be a dataset-selection artifact.

Generated at 2026-06-08T02:24:33Z.
