# Repository Instructions

## Experiment Logging Rules

This repository contains long-running numerical experiments. Do not rely on
chat history as the only record.

Whenever you run, inspect, summarize, or modify an experiment, update the
experiment records before finishing the turn.

Required records:

- Update `EXPERIMENT_LEDGER.md` with the experiment name, date, status, source
  files, output files, key settings, key metrics, conclusions, and remaining
  work.
- If the experiment has a specific conclusion, create or update a dedicated
  Markdown result file under `docs/`.
- For generated numeric tables, record the exact source path, for example a
  CSV, JSON, log, checkpoint directory, or git object.
- Clearly separate observed evidence from inference. Use wording like
  "Observed from ..." and "Inference from ...".
- If files are missing locally but exist in git, R2, or another machine, say
  that explicitly and do not present them as current local files.
- If the working tree shows deleted tracked experiment artifacts, mention that
  before interpreting results.

Before finalizing any experiment-related task, verify:

- `git status --short` for the affected paths.
- the result Markdown exists in the current working tree, unless the task is
  explicitly read-only.
- the ledger names what has been done, what has not been done, and what should
  be run next.

Never invent experiment results. If a result is not visible in a file, git
object, log, or generated artifact, say it is not currently evidenced.


## Git Recording Discipline

When the user asks to save experiment work, do not rely on `git commit -am`.
Check for untracked files and explicitly stage relevant source and record files,
especially `*.py`, `*.sh`, and `*.md` under the project root, `tools/`, and
`docs/`. Generated large arrays, checkpoints, images, and datasets should stay
in R2 unless the user explicitly asks to force-add them.
