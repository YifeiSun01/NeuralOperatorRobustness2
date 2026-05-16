# Ray Profile Markdown Lookup - 2026-05-15

## Status

Completed lookup only. No numerical experiment was run.

## Source Files

- Observed from `docs/loss3_original_theory_experiment_plan.md`.
- Observed from `git status --short` on 2026-05-15 UTC.

## Output Files

- `docs/ray_profile_markdown_lookup_20260515.md`
- `EXPERIMENT_LEDGER.md`

## Observed Evidence

- Markdown search for whole-word `ray` / `ray profile` / `ray experiment`
  found Ray-related Markdown hits only in
  `docs/loss3_original_theory_experiment_plan.md`.
- The main section is
  `docs/loss3_original_theory_experiment_plan.md:741`:
  `Experiment 4: Ray Profile / Local-to-Global Profile`.
- The compact experiment list also names `ray profile` at
  `docs/loss3_original_theory_experiment_plan.md:994`.
- `stat` reported the file mtime as
  `2026-05-15 15:04:45.684671694 +0000`, so it was not observed locally as a
  May 14 file by filesystem mtime at lookup time.
- `git status --short` showed many deleted tracked experiment artifacts under
  paths including `benchmark_results/`, `fno_training_runs/`,
  `gradient_audit/`, `path_audit/`, and `results/`. Those deletions were not
  interpreted as evidence for the Ray-profile plan.

## Inference

- The Markdown file the user is looking for is most likely
  `docs/loss3_original_theory_experiment_plan.md`.
- The "Ray experiment" refers to Experiment 4, the ray profile /
  local-to-global profile check. Its purpose is to compare fixed directions
  along radii `r in [0, epsilon]` and show that a locally good direction need
  not be the best finite-radius endpoint direction.

## Remaining Work

- None for the file lookup.
- If this becomes an executed experiment, record the exact script, run
  directory, generated tables, and numeric metrics separately.
