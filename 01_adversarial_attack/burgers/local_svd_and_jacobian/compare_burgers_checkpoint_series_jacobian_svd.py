#!/usr/bin/env python3
"""Compute top-k Jacobian/SVD diagnostics for a Burgers checkpoint series.

This script is the post-training companion for the p=2,q=2 Burgers adversarial
training pipeline.  It reuses fixed representative sample points and reuses the
old solver/baseline NPZ files when available, then computes only the new model
Jacobian and new error Jacobian for each saved checkpoint:

    J_error_checkpoint(x) = J_model_checkpoint(x) - J_solver(x)

Only the leading singular triplets are saved by default (`--top-k 100`).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.attack_framework_matrix import sync_torch  # noqa: E402
from tools.compare_burgers_adversarial_jacobian_svd import (  # noqa: E402
    DEFAULT_BASELINE,
    load_burgers_model_from_checkpoint,
    load_manifest,
    load_x_sample,
    save_configured_svd,
    save_json,
    singular_rows,
    sv_summary,
    write_csv,
)
from tools.analyze_local_jacobian_fno_deeponet import compute_explicit_jacobian  # noqa: E402

EPS = 1e-12
DEFAULT_MANIFEST = PROJECT_ROOT / 'forensics' / 'burgers_adv_training_jacobian_svd_20260531_representative20_same_points' / 'representative20_sample_manifest.csv'
DEFAULT_REUSE_ROOT = PROJECT_ROOT / 'forensics' / 'burgers_adv_training_jacobian_svd_20260531_representative20_same_points'
DEFAULT_OUT_ROOT = PROJECT_ROOT / 'forensics' / 'burgers_p2q2_checkpoint_series_jacobian_svd_20260601'


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


def sanitize_label(label: str) -> str:
    out = re.sub(r'[^A-Za-z0-9_]+', '_', label.strip())
    out = re.sub(r'_+', '_', out).strip('_')
    return out or 'checkpoint'


def read_checkpoints_csv(path: Path, epochs: set[int] | None) -> list[dict[str, Any]]:
    df = pd.read_csv(path)
    path_col = 'path' if 'path' in df.columns else 'checkpoint_path' if 'checkpoint_path' in df.columns else None
    if 'epoch' not in df.columns or path_col is None:
        raise ValueError(f'{path} must contain epoch plus path/checkpoint_path columns')
    rows: list[dict[str, Any]] = []
    for _, row in df.sort_values(['epoch']).iterrows():
        epoch = int(row['epoch'])
        if epochs is not None and epoch not in epochs:
            continue
        ckpt = Path(str(row[path_col]))
        if not ckpt.is_absolute():
            ckpt = (PROJECT_ROOT / ckpt).resolve()
        label = sanitize_label(f'epoch{epoch:04d}')
        rows.append({'label': label, 'epoch': epoch, 'path': ckpt})
    if not rows:
        raise ValueError(f'no checkpoint rows selected from {path}')
    return rows


def load_svd_npz(path: Path) -> dict[str, np.ndarray]:
    z = np.load(path)
    return {
        'jacobian': np.asarray(z['jacobian']),
        'singular_values': np.asarray(z['singular_values'], dtype=np.float64),
        'left_singular_vectors': np.asarray(z['left_singular_vectors']),
        'right_singular_vectors': np.asarray(z['right_singular_vectors']),
    }


def svd_paths(sample_dir: Path, name: str, sample_id: int) -> tuple[Path, Path]:
    folder = sample_dir / name
    return folder, folder / f'{name}_index{sample_id}_jacobian_svd.npz'


def truncate_and_copy_svd(src: Path, dst: Path, *, top_k: int, name: str, sample_id: int) -> dict[str, np.ndarray]:
    data = load_svd_npz(src)
    s = data['singular_values'][:top_k].astype(np.float64)
    U = data['left_singular_vectors']
    Vh = data['right_singular_vectors']
    if U.ndim == 2 and U.shape[1] >= len(s):
        U = U[:, : len(s)]
    elif U.ndim == 2 and U.shape[0] >= len(s):
        U = U[: len(s), :].T
    if Vh.ndim == 2 and Vh.shape[0] >= len(s):
        Vh = Vh[: len(s), :]
    elif Vh.ndim == 2 and Vh.shape[1] >= len(s):
        Vh = Vh[:, : len(s)].T
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dst,
        jacobian=data['jacobian'].astype(np.float32),
        singular_values=s,
        left_singular_vectors=U.astype(np.float32),
        right_singular_vectors=Vh.astype(np.float32),
        svd_method=np.array('reused_truncated'),
        top_k=np.array(len(s), dtype=np.int32),
        source_npz=np.array(str(src)),
    )
    return {'J': data['jacobian'], 's': s, 'U': U, 'Vh': Vh}


def ensure_reused_npz(name: str, sample_id: int, out_sample_dir: Path, reuse_sample_dir: Path, *, top_k: int) -> dict[str, np.ndarray]:
    _, dst = svd_paths(out_sample_dir, name, sample_id)
    if dst.exists():
        data = load_svd_npz(dst)
        return {'J': data['jacobian'], 's': data['singular_values'][:top_k], 'U': data['left_singular_vectors'], 'Vh': data['right_singular_vectors']}
    _, src = svd_paths(reuse_sample_dir, name, sample_id)
    if not src.exists():
        raise FileNotFoundError(f'missing reusable {name} SVD: {src}')
    return truncate_and_copy_svd(src, dst, top_k=top_k, name=name, sample_id=sample_id)


def vector_rows_from_vh(Vh: np.ndarray, top_k: int) -> np.ndarray:
    Vh = np.asarray(Vh, dtype=np.float64)
    if Vh.shape[0] >= top_k:
        return Vh[:top_k, :]
    if Vh.shape[1] >= top_k:
        return Vh[:, :top_k].T
    return Vh


def left_cols(U: np.ndarray, top_k: int) -> np.ndarray:
    U = np.asarray(U, dtype=np.float64)
    if U.shape[1] >= top_k:
        return U[:, :top_k]
    if U.shape[0] >= top_k:
        return U[:top_k, :].T
    return U


def normalize_rows(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    norms = np.linalg.norm(A, axis=1, keepdims=True)
    return A / np.maximum(norms, EPS)


def normalize_cols(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    norms = np.linalg.norm(A, axis=0, keepdims=True)
    return A / np.maximum(norms, EPS)


def absdot(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(abs(np.dot(aa, bb)) / (max(np.linalg.norm(aa), EPS) * max(np.linalg.norm(bb), EPS)))


def subspace_similarity(A: np.ndarray, B: np.ndarray) -> dict[str, float]:
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.size == 0 or B.size == 0:
        return {'mean_principal_cosine': float('nan'), 'min_principal_cosine': float('nan'), 'max_principal_cosine': float('nan')}
    qa, _ = np.linalg.qr(A)
    qb, _ = np.linalg.qr(B)
    m = qa.T @ qb
    s = np.linalg.svd(m, compute_uv=False)
    return {
        'mean_principal_cosine': float(np.mean(s)),
        'min_principal_cosine': float(np.min(s)),
        'max_principal_cosine': float(np.max(s)),
    }


def append_solver_similarity(rows: list[dict[str, Any]], sample: dict[str, Any], label: str, kind: str, model_svd: dict[str, np.ndarray], solver_svd: dict[str, np.ndarray], top_k: int) -> None:
    sample_id = int(sample['sample_id'])
    max_rank = min(top_k, len(model_svd['s']), len(solver_svd['s']))
    model_V = normalize_rows(vector_rows_from_vh(model_svd['Vh'], max_rank))
    solver_V = normalize_rows(vector_rows_from_vh(solver_svd['Vh'], max_rank))
    model_U = normalize_cols(left_cols(model_svd['U'], max_rank))
    solver_U = normalize_cols(left_cols(solver_svd['U'], max_rank))
    for rank in range(max_rank):
        rows.append(
            {
                'sample_id': sample_id,
                'source_split': sample.get('source_split', ''),
                'dataset_id': sample.get('dataset_id', ''),
                'local_index': int(sample.get('local_index', 0)),
                'checkpoint_label': label,
                'jacobian_kind': kind,
                'rank': rank + 1,
                'singular_value_model': float(model_svd['s'][rank]),
                'singular_value_solver': float(solver_svd['s'][rank]),
                'singular_value_ratio_to_solver': float(model_svd['s'][rank] / (solver_svd['s'][rank] + EPS)),
                'right_vector_absdot_solver': absdot(model_V[rank], solver_V[rank]),
                'left_vector_absdot_solver': absdot(model_U[:, rank], solver_U[:, rank]),
            }
        )


def append_subspace_rows(rows: list[dict[str, Any]], sample: dict[str, Any], label: str, kind: str, model_svd: dict[str, np.ndarray], solver_svd: dict[str, np.ndarray], ks: list[int]) -> None:
    sample_id = int(sample['sample_id'])
    for k in ks:
        kk = min(k, len(model_svd['s']), len(solver_svd['s']))
        if kk <= 0:
            continue
        model_V = vector_rows_from_vh(model_svd['Vh'], kk).T
        solver_V = vector_rows_from_vh(solver_svd['Vh'], kk).T
        model_U = left_cols(model_svd['U'], kk)
        solver_U = left_cols(solver_svd['U'], kk)
        r = subspace_similarity(model_V, solver_V)
        l = subspace_similarity(model_U, solver_U)
        rows.append(
            {
                'sample_id': sample_id,
                'source_split': sample.get('source_split', ''),
                'dataset_id': sample.get('dataset_id', ''),
                'checkpoint_label': label,
                'jacobian_kind': kind,
                'top_k': kk,
                'right_subspace_mean_principal_cosine': r['mean_principal_cosine'],
                'right_subspace_min_principal_cosine': r['min_principal_cosine'],
                'right_subspace_max_principal_cosine': r['max_principal_cosine'],
                'left_subspace_mean_principal_cosine': l['mean_principal_cosine'],
                'left_subspace_min_principal_cosine': l['min_principal_cosine'],
                'left_subspace_max_principal_cosine': l['max_principal_cosine'],
            }
        )


def finite_summary(values: list[float]) -> dict[str, float]:
    arr = np.asarray([x for x in values if np.isfinite(x)], dtype=np.float64)
    if arr.size == 0:
        return {'mean': float('nan'), 'std': float('nan'), 'min': float('nan'), 'median': float('nan'), 'max': float('nan')}
    return {
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        'min': float(np.min(arr)),
        'median': float(np.median(arr)),
        'max': float(np.max(arr)),
    }


def aggregate_error(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for label in sorted({r.get('checkpoint_label', '') for r in summary_rows}):
        if not label:
            continue
        for split in ['ALL'] + sorted({str(r.get('source_split', '')) for r in summary_rows if r.get('checkpoint_label') == label}):
            subset = [r for r in summary_rows if r.get('checkpoint_label') == label and str(r.get('jacobian_kind')) == 'error']
            if split != 'ALL':
                subset = [r for r in subset if str(r.get('source_split')) == split]
            if not subset:
                continue
            ratios = [float(r['error_spectral_norm']) / (float(r['baseline_error_spectral_norm']) + EPS) for r in subset]
            errs = [float(r['error_spectral_norm']) for r in subset]
            base = [float(r['baseline_error_spectral_norm']) for r in subset]
            row = {'checkpoint_label': label, 'source_split': split, 'n': len(subset)}
            for prefix, vals in [('error_spectral_norm', errs), ('baseline_error_spectral_norm', base), ('ratio_to_baseline_error', ratios)]:
                stats = finite_summary(vals)
                for key, val in stats.items():
                    row[f'{prefix}_{key}'] = val
            row['count_error_smaller_than_baseline'] = int(sum(r < 1.0 for r in ratios))
            out.append(row)
    return out


def write_summary_md(out_root: Path, checkpoint_specs: list[dict[str, Any]], aggregate_rows: list[dict[str, Any]], config: dict[str, Any]) -> None:
    lines = [
        '# Burgers p2q2 Checkpoint-Series Jacobian/SVD Diagnostics',
        '',
        'Object definitions:',
        '',
        '```text',
        'J_model(x) = d model(x) / d x',
        'J_solver(x) = d solver(x) / d x',
        'J_error(x) = J_model(x) - J_solver(x)',
        '```',
        '',
        'The fixed sample points are reused from the representative20 manifest, so every checkpoint is compared on exactly the same input points.',
        '',
        '## Checkpoints',
        '',
        '| label | epoch | checkpoint |',
        '|---|---:|---|',
    ]
    for spec in checkpoint_specs:
        lines.append(f"| {spec['label']} | {spec['epoch']} | `{spec['path']}` |")
    lines.extend(['', '## Error-Jacobian Spectral Norm Summary', '', '| checkpoint | split | n | error mean | baseline error mean | ratio mean | ratio median | smaller count |', '|---|---|---:|---:|---:|---:|---:|---:|'])
    for row in aggregate_rows:
        lines.append(
            f"| {row['checkpoint_label']} | {row['source_split']} | {row['n']} | "
            f"{row['error_spectral_norm_mean']:.6g} | {row['baseline_error_spectral_norm_mean']:.6g} | "
            f"{row['ratio_to_baseline_error_mean']:.6g} | {row['ratio_to_baseline_error_median']:.6g} | "
            f"{row['count_error_smaller_than_baseline']} |"
        )
    lines.extend(
        [
            '',
            '## Output Files',
            '',
            '- `checkpoint_series_jacobian_svd_summary.csv`: per-sample model and error norms.',
            '- `checkpoint_series_top_singular_values_long.csv`: top singular values for model/error Jacobians.',
            '- `checkpoint_series_solver_similarity_rankwise.csv`: rankwise singular-value and singular-vector similarity to solver.',
            '- `checkpoint_series_solver_similarity_subspaces.csv`: top-k left/right singular subspace similarity to solver.',
            '- `checkpoint_series_error_aggregate.csv`: split-level error-Jacobian summary.',
            '- `sample_*/`: top-k SVD NPZ files. Reused solver/baseline NPZ files are copied/truncated with source metadata.',
            '',
            '## Config',
            '',
            '```json',
            json.dumps(to_jsonable(config), indent=2),
            '```',
        ]
    )
    (out_root / 'checkpoint_series_summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True, help='Training run root or run/burgers directory.')
    parser.add_argument('--out-root', type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument('--sample-manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--reuse-root', type=Path, default=DEFAULT_REUSE_ROOT)
    parser.add_argument('--baseline-checkpoint', type=Path, default=DEFAULT_BASELINE)
    parser.add_argument('--checkpoint-csv', type=Path, default=None)
    parser.add_argument('--checkpoint-epochs', type=int, nargs='*', default=[200, 400, 600, 800, 1000])
    parser.add_argument('--top-k', type=int, default=100)
    parser.add_argument('--svd-method', choices=['topk', 'full'], default='topk')
    parser.add_argument('--svd-solver', choices=['propack', 'arpack', 'lobpcg'], default='propack')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--reuse-existing', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def resolve_burgers_dir(run_dir: Path) -> Path:
    run_dir = run_dir.resolve()
    if (run_dir / 'checkpoints.csv').exists():
        return run_dir
    if (run_dir / 'burgers' / 'checkpoints.csv').exists():
        return run_dir / 'burgers'
    raise FileNotFoundError(f'cannot find Burgers checkpoints.csv under {run_dir}')


def main() -> None:
    args = parse_args()
    burgers_dir = resolve_burgers_dir(args.run_dir)
    checkpoint_csv = args.checkpoint_csv or (burgers_dir / 'checkpoints.csv')
    checkpoint_specs = read_checkpoints_csv(checkpoint_csv, set(args.checkpoint_epochs) if args.checkpoint_epochs else None)
    samples = load_manifest(args.sample_manifest.resolve())
    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    write_csv(out_root / 'representative20_sample_manifest.csv', samples)
    config = {
        'run_dir': burgers_dir.parent,
        'burgers_dir': burgers_dir,
        'checkpoint_csv': checkpoint_csv,
        'checkpoint_specs': checkpoint_specs,
        'sample_manifest': args.sample_manifest,
        'reuse_root': args.reuse_root,
        'top_k': args.top_k,
        'svd_method': args.svd_method,
        'svd_solver': args.svd_solver,
        'baseline_checkpoint': args.baseline_checkpoint,
    }
    save_json(out_root / 'config.json', config)
    if args.dry_run:
        print(json.dumps(to_jsonable(config), indent=2))
        return

    device = torch.device(args.device if args.device and torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.enabled = False

    summary_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    rank_similarity_rows: list[dict[str, Any]] = []
    subspace_similarity_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []

    checkpoint_models: dict[str, Any] = {}
    try:
        for spec in checkpoint_specs:
            checkpoint_models[spec['label']] = load_burgers_model_from_checkpoint(Path(spec['path']).resolve(), device)

        for sample in samples:
            sample_id = int(sample['sample_id'])
            sample_dir = out_root / f'sample_{sample_id:03d}'
            sample_dir.mkdir(parents=True, exist_ok=True)
            save_json(sample_dir / 'sample.json', sample)
            reuse_sample_dir = args.reuse_root.resolve() / f'sample_{sample_id:03d}'

            solver_svd = ensure_reused_npz('solver', sample_id, sample_dir, reuse_sample_dir, top_k=args.top_k)
            baseline_svd = ensure_reused_npz('baseline', sample_id, sample_dir, reuse_sample_dir, top_k=args.top_k)
            baseline_error_svd = ensure_reused_npz('baseline_error', sample_id, sample_dir, reuse_sample_dir, top_k=args.top_k)
            J_solver = solver_svd['J'].astype(np.float64)
            baseline_error_norm = float(baseline_error_svd['s'][0])

            for name, svd_data, kind in [('solver', solver_svd, 'solver'), ('baseline', baseline_svd, 'model'), ('baseline_error', baseline_error_svd, 'error')]:
                summary_rows.append(
                    {
                        **sv_summary(name, svd_data['s'], sample=sample, model_name=name),
                        'checkpoint_label': name,
                        'epoch': 0,
                        'jacobian_kind': kind,
                        'error_spectral_norm': float(svd_data['s'][0]) if kind == 'error' else '',
                        'baseline_error_spectral_norm': baseline_error_norm if kind == 'error' else '',
                    }
                )

            x = load_x_sample(Path(sample['dataset_path']), int(sample['local_index']))
            for spec in checkpoint_specs:
                label = spec['label']
                model = checkpoint_models[label]
                model_name = f'p2q2_{label}'
                model_npz = sample_dir / model_name / f'{model_name}_index{sample_id}_jacobian_svd.npz'
                if args.reuse_existing and model_npz.exists():
                    loaded = load_svd_npz(model_npz)
                    model_svd = {'J': loaded['jacobian'], 's': loaded['singular_values'][: args.top_k], 'U': loaded['left_singular_vectors'], 'Vh': loaded['right_singular_vectors']}
                    model_seconds = 0.0
                    model_source = 'reused'
                else:
                    start = time.perf_counter()
                    J_model = compute_explicit_jacobian(model, x, device, progress_prefix=f'{label}_sample{sample_id}')
                    model_seconds = time.perf_counter() - start
                    model_svd = save_configured_svd(model_name, J_model, sample_dir, sample_id, args)
                    model_source = 'computed'
                runtime_rows.append({'sample_id': sample_id, 'checkpoint_label': label, 'component': 'model', 'seconds': model_seconds, 'source': model_source})

                error_name = f'p2q2_{label}_error'
                error_npz = sample_dir / error_name / f'{error_name}_index{sample_id}_jacobian_svd.npz'
                if args.reuse_existing and error_npz.exists():
                    loaded = load_svd_npz(error_npz)
                    error_svd = {'J': loaded['jacobian'], 's': loaded['singular_values'][: args.top_k], 'U': loaded['left_singular_vectors'], 'Vh': loaded['right_singular_vectors']}
                    error_seconds = 0.0
                    error_source = 'reused'
                else:
                    start = time.perf_counter()
                    J_error = model_svd['J'].astype(np.float64) - J_solver
                    error_seconds = time.perf_counter() - start
                    error_svd = save_configured_svd(error_name, J_error, sample_dir, sample_id, args)
                    error_source = 'computed'
                runtime_rows.append({'sample_id': sample_id, 'checkpoint_label': label, 'component': 'error', 'seconds': error_seconds, 'source': error_source})

                model_summary = sv_summary(model_name, model_svd['s'], sample=sample, model_name=label)
                model_summary.update({'checkpoint_label': label, 'epoch': spec['epoch'], 'jacobian_kind': 'model', 'error_spectral_norm': '', 'baseline_error_spectral_norm': baseline_error_norm})
                error_summary = sv_summary(error_name, error_svd['s'], sample=sample, model_name=label)
                error_summary.update(
                    {
                        'checkpoint_label': label,
                        'epoch': spec['epoch'],
                        'jacobian_kind': 'error',
                        'error_spectral_norm': float(error_svd['s'][0]),
                        'baseline_error_spectral_norm': baseline_error_norm,
                        'error_spectral_norm_ratio_to_baseline': float(error_svd['s'][0] / (baseline_error_norm + EPS)),
                    }
                )
                summary_rows.extend([model_summary, error_summary])
                top_rows.extend(singular_rows(sample, label, 'model', model_svd['s'], int(args.top_k)))
                top_rows.extend(singular_rows(sample, label, 'error', error_svd['s'], int(args.top_k)))
                append_solver_similarity(rank_similarity_rows, sample, label, 'model', model_svd, solver_svd, int(args.top_k))
                append_solver_similarity(rank_similarity_rows, sample, label, 'error', error_svd, solver_svd, int(args.top_k))
                append_subspace_rows(subspace_similarity_rows, sample, label, 'model', model_svd, solver_svd, [1, 5, 10, 20, 50, 100])
                append_subspace_rows(subspace_similarity_rows, sample, label, 'error', error_svd, solver_svd, [1, 5, 10, 20, 50, 100])

                sync_torch(torch, device)
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

            write_csv(out_root / 'checkpoint_series_jacobian_svd_summary.partial.csv', summary_rows)
            write_csv(out_root / 'checkpoint_series_top_singular_values_long.partial.csv', top_rows)
            write_csv(out_root / 'checkpoint_series_solver_similarity_rankwise.partial.csv', rank_similarity_rows)
            write_csv(out_root / 'checkpoint_series_solver_similarity_subspaces.partial.csv', subspace_similarity_rows)
            write_csv(out_root / 'runtime.partial.csv', runtime_rows)
    finally:
        for m in checkpoint_models.values():
            del m
        sync_torch(torch, device)
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    write_csv(out_root / 'checkpoint_series_jacobian_svd_summary.csv', summary_rows)
    write_csv(out_root / 'checkpoint_series_top_singular_values_long.csv', top_rows)
    write_csv(out_root / 'checkpoint_series_solver_similarity_rankwise.csv', rank_similarity_rows)
    write_csv(out_root / 'checkpoint_series_solver_similarity_subspaces.csv', subspace_similarity_rows)
    write_csv(out_root / 'runtime.csv', runtime_rows)
    agg = aggregate_error(summary_rows)
    write_csv(out_root / 'checkpoint_series_error_aggregate.csv', agg)
    write_summary_md(out_root, checkpoint_specs, agg, config)
    print(json.dumps({'out_root': str(out_root), 'samples': len(samples), 'checkpoints': [s['label'] for s in checkpoint_specs], 'top_k': args.top_k}, indent=2))


if __name__ == '__main__':
    main()
