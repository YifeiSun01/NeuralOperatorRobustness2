#!/usr/bin/env python3
"""Validate that a Burgers adversarial-training run saved L2-like deltas.

The failed/ablation run used fast_replace_linf, which makes every coordinate of
`delta` land at approximately +/- epsilon and therefore looks rectangular.  The
intended p=2,q=2 run should use fast_replace_l2/fast_add_l2.  This checker reads
saved fixed-probe NPZ files and writes a small JSON/CSV report with two concrete
sanity checks:

- attack metadata says continuous_l2_rms / p=2 / q=2
- saved delta arrays have many distinct amplitudes, not only +/- epsilon
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(x) for x in value]
    return value


def resolve_burgers_dir(path: Path) -> Path:
    path = path.resolve()
    if (path / 'attack_probe_samples.csv').exists():
        return path
    if (path / 'burgers' / 'attack_probe_samples.csv').exists():
        return path / 'burgers'
    raise FileNotFoundError(f'cannot find attack_probe_samples.csv under {path}')


def relpath_to_repo(path_text: str, burgers_dir: Path) -> Path:
    p = Path(path_text)
    if p.is_absolute():
        return p
    candidates = [PROJECT_ROOT / p, burgers_dir / p, burgers_dir.parent / p]
    for c in candidates:
        if c.exists():
            return c.resolve()
    return (PROJECT_ROOT / p).resolve()


def finite_stats(values: list[float]) -> dict[str, float | None]:
    arr = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if arr.size == 0:
        return {'mean': None, 'std': None, 'min': None, 'median': None, 'max': None}
    return {
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        'min': float(np.min(arr)),
        'median': float(np.median(arr)),
        'max': float(np.max(arr)),
    }


def inspect_delta(delta: np.ndarray, *, decimals: int) -> dict[str, float | int]:
    flat = np.asarray(delta, dtype=np.float64).reshape(-1)
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return {
            'n_values': 0,
            'unique_rounded_values': 0,
            'l2_rms': math.nan,
            'linf': math.nan,
            'abs_mean': math.nan,
            'top_abs_fraction': math.nan,
            'sign_change_fraction': math.nan,
        }
    rounded = np.round(finite, decimals=decimals)
    unique_count = int(np.unique(rounded).size)
    abs_finite = np.abs(finite)
    linf = float(np.max(abs_finite))
    l2_rms = float(np.sqrt(np.mean(finite * finite)))
    tol = max(10.0 ** (-decimals), 1e-4 * linf)
    top_abs_fraction = float(np.mean(np.abs(abs_finite - linf) <= tol)) if linf > 0 else 0.0
    signs = np.sign(finite)
    nonzero = signs != 0
    if np.count_nonzero(nonzero) > 1:
        s = signs[nonzero]
        sign_change_fraction = float(np.mean(s[1:] != s[:-1]))
    else:
        sign_change_fraction = 0.0
    return {
        'n_values': int(finite.size),
        'unique_rounded_values': unique_count,
        'l2_rms': l2_rms,
        'linf': linf,
        'abs_mean': float(np.mean(abs_finite)),
        'top_abs_fraction': top_abs_fraction,
        'sign_change_fraction': sign_change_fraction,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True, help='Run root or run/burgers directory.')
    parser.add_argument('--out-json', type=Path, default=None)
    parser.add_argument('--out-csv', type=Path, default=None)
    parser.add_argument('--max-files', type=int, default=5, help='Inspect first/last probe files up to this count each.')
    parser.add_argument('--round-decimals', type=int, default=6)
    parser.add_argument('--min-median-unique-values', type=int, default=16)
    parser.add_argument('--max-median-top-abs-fraction', type=float, default=0.45)
    parser.add_argument('--fail-on-bad', action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    burgers_dir = resolve_burgers_dir(args.run_dir)
    run_root = burgers_dir.parent
    out_json = args.out_json or (burgers_dir / 'l2_delta_geometry_check.json')
    out_csv = args.out_csv or (burgers_dir / 'l2_delta_geometry_check_samples.csv')

    probe_csv = burgers_dir / 'attack_probe_samples.csv'
    probe_df = pd.read_csv(probe_csv)
    if probe_df.empty:
        raise ValueError(f'{probe_csv} is empty')

    metadata_attack_methods = sorted(str(x) for x in probe_df['attack_method'].dropna().unique()) if 'attack_method' in probe_df else []
    metadata_attack_types = sorted(str(x) for x in probe_df['attack_type'].dropna().unique()) if 'attack_type' in probe_df else []
    metadata_ok = bool(metadata_attack_methods) and all(m.endswith('_l2') for m in metadata_attack_methods) and all(t == 'continuous_l2_rms' for t in metadata_attack_types)

    npz_paths = []
    if 'npz_path' not in probe_df.columns:
        raise ValueError(f'{probe_csv} has no npz_path column')
    for text in probe_df['npz_path'].dropna().astype(str).unique():
        p = relpath_to_repo(text, burgers_dir)
        if p.exists():
            npz_paths.append(p)
    npz_paths = sorted(npz_paths)
    if not npz_paths:
        raise FileNotFoundError('no saved probe NPZ files could be resolved')
    if len(npz_paths) > args.max_files * 2:
        selected_paths = npz_paths[: args.max_files] + npz_paths[-args.max_files :]
    else:
        selected_paths = npz_paths

    rows: list[dict[str, Any]] = []
    for npz_path in selected_paths:
        z = np.load(npz_path)
        if 'delta' not in z.files:
            continue
        delta = np.asarray(z['delta'])
        epoch = int(np.asarray(z['epoch']).item()) if 'epoch' in z.files else None
        probe_ranks = np.asarray(z['probe_rank']).reshape(-1) if 'probe_rank' in z.files else np.arange(delta.shape[0])
        source_indices = np.asarray(z['source_index']).reshape(-1) if 'source_index' in z.files else np.full(delta.shape[0], -1)
        for i in range(delta.shape[0]):
            stats = inspect_delta(delta[i], decimals=args.round_decimals)
            rows.append(
                {
                    'epoch': epoch,
                    'probe_rank': int(probe_ranks[i]) if i < probe_ranks.size else i,
                    'source_index': int(source_indices[i]) if i < source_indices.size else -1,
                    'npz_path': str(npz_path),
                    **stats,
                }
            )

    unique_stats = finite_stats([float(r['unique_rounded_values']) for r in rows])
    top_abs_stats = finite_stats([float(r['top_abs_fraction']) for r in rows])
    l2_rms_stats = finite_stats([float(r['l2_rms']) for r in rows])
    linf_stats = finite_stats([float(r['linf']) for r in rows])
    median_unique = unique_stats['median'] or 0.0
    median_top_abs_fraction = top_abs_stats['median'] or 1.0
    shape_ok = median_unique >= args.min_median_unique_values and median_top_abs_fraction <= args.max_median_top_abs_fraction
    passed = bool(metadata_ok and shape_ok)

    report = {
        'run_root': str(run_root),
        'burgers_dir': str(burgers_dir),
        'probe_csv': str(probe_csv),
        'metadata_attack_methods': metadata_attack_methods,
        'metadata_attack_types': metadata_attack_types,
        'metadata_ok_l2': metadata_ok,
        'inspected_npz_files': [str(p) for p in selected_paths],
        'sample_count': len(rows),
        'unique_rounded_values': unique_stats,
        'top_abs_fraction': top_abs_stats,
        'l2_rms': l2_rms_stats,
        'linf': linf_stats,
        'thresholds': {
            'round_decimals': args.round_decimals,
            'min_median_unique_values': args.min_median_unique_values,
            'max_median_top_abs_fraction': args.max_median_top_abs_fraction,
        },
        'passed_l2_non_rectangular_check': passed,
        'interpretation': (
            'pass means the attack metadata is L2 and saved deltas have many amplitudes; '
            'fail means the run is likely still Linf/sign-like or no valid probes were saved.'
        ),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(to_jsonable(report), indent=2), encoding='utf-8')
    write_csv(out_csv, rows)
    print(json.dumps(to_jsonable(report), indent=2))
    if args.fail_on_bad and not passed:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
