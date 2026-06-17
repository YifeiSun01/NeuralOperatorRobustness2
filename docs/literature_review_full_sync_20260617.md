# Literature Review Full Sync - 2026-06-17

Status: completed as a literature-document synchronization. No numerical
experiment, attack run, training run, GPU job, or new empirical result was
performed.

## Scope

The sync covered the repository literature-review and manuscript-note files
that discuss adversarial attack/training, active learning, robustness, and
neural operators:

- `NEURAL_OPERATOR_ADVERSARIAL_RELATED_WORK.md`
- `NEURAL_OPERATOR_ADVERSARIAL_LITERATURE_REVIEW.md`
- `GENERALIZATION_MASTER_RECORD.md`
- root `main.tex`
- `docs/neural_operator_adversarial_related_work_update_20260617.md`
- `paper_literature_notes_bundle_20260617/literature_review_adversarial_active_operator_learning.md`
- `paper_literature_notes_bundle_20260617/main.tex`
- `paper_literature_notes_bundle_20260617/references.bib`
- `paper_literature_notes_bundle_20260617/structured_research_notes.md`
- `paper_literature_notes_bundle_20260617/research_notes.md`
- `paper_literature_notes_bundle_20260617/math_attack_taxonomy_notes.md`

## Synchronized Conclusions

Observed from local synchronized files:

1. The paper's central positioning is now consistently recorded as
   solver-integrated adversarial attack and training of neural operators.
   The main object is not only a neural model, and not only a numerical solver,
   but the discrepancy between a learned neural operator \(F_\theta\) and a
   solver/teacher \(G\).

2. The loss taxonomy is consistently recorded:

   - \(L_1(\delta)=\|F_\theta(a+\delta)-F_\theta(a)\|\): model-only movement.
   - \(L_2(\delta)=\|F_\theta(a+\delta)-G(a)\|\): fixed clean solver target.
   - \(L_3(\delta)=\|F_\theta(a+\delta)-G(a+\delta)\|\): solver-consistent
     perturbed model-solver discrepancy.

   The synced notes consistently emphasize that \(L_3\) is the most faithful
   objective for PDE operator regression because it compares the model and
   solver at the same perturbed input.

3. Generalization and robustness are separated:

   - Generalization is a static prediction metric on clean samples, such as
     relative \(L_1/L_2\), RMSE, energy norm, or output-function error.
   - Robustness is local and dynamic: it measures the response under small
     perturbation budgets, adversarial loss increase, local Jacobian behavior,
     or solver-consistent worst-case error growth.

4. Active-learning papers are now positioned as adjacent but distinct:

   - Winovich, Daneker, Lu Lu, and Guang Lin use deterministic predictive UQ
     and Gaussian NLL to select high-uncertainty PDE instances for additional
     solver labels. Their evaluation focuses on clean prediction error,
     calibration, runtime, and data efficiency, not adversarial robustness.
   - MRA-FNO selects both PDE inputs and simulation resolutions using a
     utility/cost acquisition rule. It is solver-cost-aware active learning,
     not a worst-case perturbation attack or adversarial-training method.

   In both cases, the solver is used after acquisition to label selected
   examples, while our framework uses model-solver discrepancy under a
   perturbation budget as the adversarial objective.

5. GAN-style adversarial neural-operator papers are now positioned as
   adversarial-objective related work, not robustness-equivalent work:

   - GANO learns distributions over functions through a generator neural
     operator and discriminator neural functional.
   - Oommen, Khodakarami, Bora, Wang, and George Em Karniadakis introduce
     `adv-NO` for turbulent-flow super-resolution, forecasting, and sparse
     reconstruction. This is the Brown/Karniadakis-related adversarial
     neural-operator citation.

   Both use adversarial/generative/discriminator losses, not PGD-style
   norm-bounded input attacks or solver-integrated adversarial training.

6. Roy et al.'s neural-operator digital-twin paper is now consistently
   positioned as attack/evaluation related work:

   - It studies sparse, physically plausible, gradient-free attacks on neural
     operator digital twins.
   - It is important for robustness evaluation and local sensitivity
     diagnostics.
   - It does not propose solver-integrated adversarial training in the
     \(F_\theta(a+\delta)\) versus \(G(a+\delta)\) sense.

## What Was Changed

Observed from this sync:

- Root `main.tex` now includes the active-learning and GAN-style related-work
  section for Winovich et al., MRA-FNO, GANO, and Karniadakis/Oommen `adv-NO`.
- The bundle `main.tex` now includes the active-learning distinction in the
  introduction/related-work framing and cites `winovich2025active` and
  `li2024mrafno`.
- `NEURAL_OPERATOR_ADVERSARIAL_RELATED_WORK.md` and
  `NEURAL_OPERATOR_ADVERSARIAL_LITERATURE_REVIEW.md` now include active
  learning, Roy digital-twin attacks, GANO, and Karniadakis/Oommen `adv-NO`
  in the same taxonomy.
- `GENERALIZATION_MASTER_RECORD.md` now correctly names GANO and adds the
  active-learning papers instead of treating all adversarial-objective work as
  robustness training.
- `structured_research_notes.md` and `research_notes.md` no longer contain
  the outdated statement that no Karniadakis-authored adversarial
  neural-operator paper was found. They now record the verified
  Karniadakis/Oommen `adv-NO` paper and its limitation.

## Recommended Manuscript Use

Inference from the synchronized notes:

- Main text should use the solver-integrated story: neural operator as a
  surrogate for a solver, adversarial perturbation as a way to expose
  model-solver discrepancy, and adversarial training as a way to reduce that
  discrepancy.
- Related work should group papers by mechanism:
  neural-operator foundations, residual/physics adversarial sampling,
  neural-operator attack evaluation, active learning/UQ, GAN-style adversarial
  objectives, and general adversarial robustness.
- Appendices should carry detailed taxonomy tables, local Jacobian/operator
  derivations, extra optimizer comparisons, and long literature matrices.

## Verification Notes

Observed command status:

- Local text searches found synchronized mentions of Winovich/UQ active
  learning, MRA-FNO, GANO, Karniadakis/Oommen `adv-NO`, Roy digital-twin
  attacks, and the local \(L_1/L_2/L_3\) solver-integrated framework across
  the target files.
- A `git status --short` check from the workspace returned
  `fatal: not a git repository (or any of the parent directories): .git`.
  Therefore this sync report records file-level evidence but cannot provide a
  git-tracked status summary from this workspace state.

## Remaining Work

No additional literature-review file was identified locally as requiring the
same sync. Remaining work is editorial rather than synchronization:

- choose which manuscript file is the submission source;
- compress the related-work material into the final page budget;
- move long paper-by-paper taxonomy tables and proof details to appendices;
- run a compile/check pass once the final LaTeX source is selected.
