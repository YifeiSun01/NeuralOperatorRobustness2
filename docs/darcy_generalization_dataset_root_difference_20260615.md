# Darcy Generalization Dataset Root Difference - 2026-06-15

Status: verified from local `.pt` tensors.

## Roots Compared

1. `generalization_datasets_darcy_lossdrop50_selected_20260607/`
2. `generalization_datasets_darcy_binary_loss3targeted_20260611/`

## Date Ordering

By dataset naming and `.pt` file mtimes:

- `generalization_datasets_darcy_lossdrop50_selected_20260607` is older. Its
  `.pt` files were generated on 2026-06-07.
- `generalization_datasets_darcy_binary_loss3targeted_20260611` is newer. Its
  `.pt` files were generated on 2026-06-11.

The top-level directory mtime for `lossdrop50_selected` was later touched on
2026-06-14, but the actual selected `.pt` dataset files are from 2026-06-07.

## Coefficient Field Check

The Darcy coefficient field is stored in key `x` in the `.pt` files.

Full scan over all 50 `.pt` files showed:

| root | files | x min range | x max range | unique count | binary? | values |
|---|---:|---:|---:|---:|---|---|
| `lossdrop50_selected_20260607` | 50 | 4.0 to 4.001276 | 9.998280 to 10.5 | 305163 to 338459 | no | continuous soft fields |
| `binary_loss3targeted_20260611` | 50 | 3.0 to 3.0 | 12.0 to 12.0 | 2 to 2 | yes | exactly 3.0 and 12.0 |

Therefore:

- `lossdrop50_selected_20260607` is not a binary coefficient-field suite.
- `binary_loss3targeted_20260611` is binary, but its two values are `3` and
  `12`, not `5` and `13`.
- Neither root is exactly a `5/13` binary coefficient-field suite.

## Source Metadata

`lossdrop50_selected_20260607` says it was selected from:

```text
generalization_datasets_darcy_lossdrop50_pool_20260607/
forensics/darcy_lossdrop50_pool_gradient_screen_20260607/
```

Its selected files have names like:

```text
darcy_lossdrop_pool_soft_l4_h10_b10_02.pt
```

and the tensors confirm soft/continuous `x`.

`binary_loss3targeted_20260611/darcy/generation_summary.json` records:

```json
{
  "num_datasets": 50,
  "samples_per_dataset": 50,
  "low": 3.0,
  "high": 12.0,
  "all_binary_verified": true
}
```

The tensor scan agrees with this metadata.

## Practical Consequence

Do not compare rankings from these two roots as if they were the same
generalization suite:

- The organized-release loss curves that showed loss3 strongest used the newer
  binary `20260611` root.
- The final robustness matrix that used `lossdrop50_selected_20260607` was using
  the older continuous soft-field root.

Those are different out-of-distribution test distributions.

## Required Raw Figure Provenance

The figures under:

```text
outputs/darcy_cflow_timematched_organized_release_20260614/figures/diagnostic_existing/required_raw_figures_previous/
```

use source tables:

```text
outputs/darcy_cflow_timematched_organized_release_20260614/data/source_tables/six_method_common_range_eval_metrics.csv
outputs/darcy_cflow_timematched_organized_release_20260614/data/source_tables/six_method_common_range_eval_split_summary.csv
```

The generalization rows in `six_method_common_range_eval_metrics.csv` point only
to:

```text
generalization_datasets_darcy_binary_loss3targeted_20260611/
```

Verification from the source table:

- generalization rows: `962250`
- unique generalization dataset IDs: `50`
- unique generalization root dirs in `path`: `generalization_datasets_darcy_binary_loss3targeted_20260611`
- methods: `loss1`, `loss2`, `loss3`, `physics`, `random_clean`,
  `random_solver`

Therefore the required raw previous loss figures are on the newer binary
`20260611` root, not on the older continuous `lossdrop50_selected_20260607`
root.
