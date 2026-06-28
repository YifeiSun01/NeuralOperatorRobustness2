# Git Large File Guard

`.gitignore` cannot ignore files by size. It only matches path and filename
patterns. To avoid accidentally committing files that GitHub will reject, this
repository uses two layers:

1. `.gitignore` ignores common generated data/model/result file types.
2. `.githooks/` checks file sizes before commit and push.

## GitHub Size Limits

- Above 50 MiB: GitHub warns that the file is large.
- Above 100 MiB: GitHub blocks the file.

The local guard is intentionally stricter than GitHub's own limits:

- warning threshold: `5 MiB`
- blocking threshold: `20 MiB`

The block limit is deliberately low because this repo should track code and
small text artifacts, not generated datasets, checkpoints, model weights, or
large result dumps.

## Enable Hooks

This repository stores hooks in `.githooks/`. Enable them with:

```bash
git config core.hooksPath .githooks
```

The current local checkout has already been configured this way if Codex set it
up during environment preparation.

## What The Hooks Do

Before commit:

```bash
python3 tools/check_large_files.py --mode staged
```

This checks only files staged for commit.

Before push:

```bash
python3 tools/check_large_files.py --mode tracked
```

This checks all currently tracked files.

## Override The Limit

For one command, you can make the limit even stricter:

```bash
GIT_MAX_FILE_MB=10 git commit
```

or:

```bash
GIT_MAX_FILE_MB=10 git push
```

You can also run the checker manually:

```bash
python3 tools/check_large_files.py --mode staged --max-mib 10
python3 tools/check_large_files.py --mode tracked --max-mib 10
```

## Emergency Bypass

Only use this if you know what you are doing:

```bash
SKIP_LARGE_FILE_CHECK=1 git commit
SKIP_LARGE_FILE_CHECK=1 git push
```

## Important Notes

Adding a pattern to `.gitignore` does not remove files that are already tracked.
For an already tracked file, use:

```bash
git rm --cached path/to/large_file
```

That removes it from future commits without deleting the local file.

If a large file already exists in git history, removing it from the latest
commit may not be enough. GitHub can still reject the push if the old large blob
is in the branch history. In that case, history cleanup with `git filter-repo`
or BFG may be needed.

For truly necessary large files, use external storage or Git LFS rather than
normal git tracking.
