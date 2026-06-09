# Burgers round03 attack-generated generalization validity caveat - 2026-06-08

Status: credibility caveat recorded after re-auditing the round03 dataset provenance.

Observed evidence:
- `tools/generate_burgers_loss3_selective_generalization.py` builds the round03 candidate pool by sampling original Burgers train/test examples and running `attack_batch(..., "loss3")` on them.
- The selected `round_03/burgers` datasets are copied from that attack-generated candidate pool after selection-score sorting.
- `generalization_datasets_burgers_loss3_selective_search/round_03/selected_candidate_scores.csv` shows the 50 selected datasets collapse to six source/epsilon/step configurations: train/test crossed with eps fractions `0.08`, `0.10`, `0.12` and steps `8` or `10`.
- The round03 metadata does not contain neutral master distribution names such as `burgers_far_sawtooth_add_scale0p3_shift0`, `kernel`, `transform`, `scale`, or `shift`. Those semantic distribution names belong to `generalization_datasets_rmse_1p5_3x_all_ns50/burgers`, not to the round03 evaluation root.

Inference from observed evidence:
- The round03 result is not evidence for broad neutral OOD generalization across 50 independently named Burgers distribution shifts.
- The valid interpretation is narrower: round03 tests whether loss3 self-training improves performance on a loss3-selected, attack-generated stress distribution derived from original train/test samples.
- Because the data-generation mechanism is aligned with the loss3 objective, round03 has selection bias toward loss3. It can support a targeted robustness/stress-test claim, but it should not be used alone as a persuasive general generalization claim.
- Any paper, figure caption, or slide should relabel this set as `round03 loss3-selective adversarial stress set` or similar, not simply `50 generalization datasets` without qualification.

Required next evidence for a stronger claim:
- Re-evaluate the same baseline/loss1/loss2/loss3/physics models on the neutral semantic Burgers root `generalization_datasets_rmse_1p5_3x_all_ns50/burgers` with dataset IDs such as `burgers_far_sawtooth_add_scale0p3_shift0`.
- Report train/test separately from neutral semantic generalization and from attack-generated stress sets.
- For any attack-generated set, include an independent holdout generated with different seeds and preferably different attack construction, and describe it as adversarial stress evaluation rather than neutral generalization.
- Use same wall-clock comparisons for model checkpoints when comparing training methods.

## Plain conclusion to preserve

This round03 dataset should not be described as "50 kinds of generalization datasets." Observed evidence shows it is 50 selected datasets, but not 50 independent semantic distribution types. The selected datasets collapse to only six source/epsilon/step generation configurations and were produced by attacking original train/test Burgers samples.

Therefore, this experiment has low persuasive power for claims about generalization loss on neutral OOD data. It is mainly evidence for performance on a loss3-selected attack-generated stress set. Any future report should state this limitation before interpreting the round03 generalization-loss curves.
