#!/usr/bin/env python3
"""Targeted current core4/PQ local residual-Jacobian SVD probe for Loss3.

This is deliberately small because explicit 1024x1024 Jacobians are expensive.
It evaluates the current core4/PQ deltas, not the old PGD-only path.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('DDE_BACKEND', 'pytorch')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.attack_framework_matrix import (  # noqa: E402
    DEFAULT_BURGERS_MODEL_DIR,
    DEFAULT_BURGERS_TEST,
    load_burgers_torch_model,
    sync_torch,
)
from tools.analyze_fno_solver_jacobian_similarity import compute_solver_jacobian  # noqa: E402
from tools.analyze_local_jacobian_fno_deeponet import compute_explicit_jacobian, vector_frequency_metrics  # noqa: E402
from tools.run_loss3_core4_pq_landscape_probe import (  # noqa: E402
    CORE4,
    DEFAULT_ROOTS,
    load_final_delta,
    load_trajectory_delta,
    parse_root,
)
from tools.run_loss3_ray_profile import load_burgers_indices  # noqa: E402
from tools.run_loss3_small_epsilon_sweep import configure_runtime, finite_json, require_gpu_runtime  # noqa: E402

DEFAULT_OUT = PROJECT_ROOT / 'forensics/loss3_current_core4_jacobian_svd_probe_20260521'
DEFAULT_DOC = PROJECT_ROOT / 'docs/loss3_current_core4_jacobian_svd_probe_20260521.md'
EPS = 1e-12


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    fields: list[str] = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(finite_json(payload), indent=2, sort_keys=True) + '\n', encoding='utf-8')


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open('r', newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except Exception:
        return str(path)


def norm2(v: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(v, dtype=np.float64).reshape(-1)))


def unit(v: np.ndarray) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float64).reshape(-1)
    return arr / (np.linalg.norm(arr) + EPS)


def cosine(a: np.ndarray, b: np.ndarray, *, abs_value: bool = True) -> float:
    au = unit(a); bu = unit(b)
    c = float(np.dot(au, bu))
    return abs(c) if abs_value else c


def angle_from_abs_cos(c: float) -> float:
    if not math.isfinite(c):
        return math.nan
    return float(np.degrees(np.arccos(np.clip(c, 0.0, 1.0))))


def concentration(v: np.ndarray) -> dict[str, float]:
    arr = np.asarray(v, dtype=np.float64).reshape(-1)
    abs_arr = np.abs(arr)
    energy = arr * arr
    return {
        'peakiness': float(abs_arr.max() / (abs_arr.mean() + EPS)),
        'top1_energy_fraction': float(energy.max() / (energy.sum() + EPS)),
        'top5_energy_fraction': float(np.sort(energy)[-5:].sum() / (energy.sum() + EPS)),
    }


def load_delta(root: Path, method: str, kind: str, step: int | None, sample_index: int) -> np.ndarray:
    if kind == 'clean':
        z = np.load(root / 'final_deltas.npz', allow_pickle=True)
        nx = int(np.asarray(z['final_delta']).shape[-2])
        return np.zeros((nx, 1), dtype=np.float32)
    if kind == 'final':
        return load_final_delta(root, method, [sample_index])[0]
    if kind == 'trajectory':
        arr = load_trajectory_delta(root, method, int(step), [sample_index])
        if arr is None:
            raise FileNotFoundError(f'trajectory step={step} method={method} root={root}')
        return arr[0]
    raise ValueError(kind)


def default_probe_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    roots = [p if p.is_absolute() else PROJECT_ROOT / p for p in args.setting_root]
    # Expected order from DEFAULT_ROOTS: p2q2, p2q1, p2qinf, p1qinf.
    by_pq = {}
    for r in roots:
        by_pq[parse_root(r)['pq']] = r
    specs: list[dict[str, Any]] = []
    p2q2 = by_pq.get('p2q2')
    if p2q2 is not None:
        specs.extend([
            {'root': p2q2, 'state_method': 'clean', 'state_kind': 'clean', 'state_step': None},
            {'root': p2q2, 'state_method': 'steepest_replace', 'state_kind': 'trajectory', 'state_step': 1},
            {'root': p2q2, 'state_method': 'steepest_replace', 'state_kind': 'trajectory', 'state_step': 5},
            {'root': p2q2, 'state_method': 'steepest_replace', 'state_kind': 'trajectory', 'state_step': 10},
            {'root': p2q2, 'state_method': 'steepest_replace', 'state_kind': 'final', 'state_step': None},
            {'root': p2q2, 'state_method': 'steepest_add', 'state_kind': 'final', 'state_step': None},
        ])
    p2qinf = by_pq.get('p2qinf')
    if p2qinf is not None:
        specs.extend([
            {'root': p2qinf, 'state_method': 'clean', 'state_kind': 'clean', 'state_step': None},
            {'root': p2qinf, 'state_method': 'steepest_replace', 'state_kind': 'final', 'state_step': None},
            {'root': p2qinf, 'state_method': 'steepest_add', 'state_kind': 'final', 'state_step': None},
        ])
    p1qinf = by_pq.get('p1qinf')
    if p1qinf is not None:
        specs.extend([
            {'root': p1qinf, 'state_method': 'clean', 'state_kind': 'clean', 'state_step': None},
            {'root': p1qinf, 'state_method': 'steepest_replace', 'state_kind': 'final', 'state_step': None},
        ])
    return specs


def state_label(method: str, kind: str, step: int | None) -> str:
    if kind == 'clean':
        return 'clean'
    if kind == 'trajectory':
        return f'{method}_step{step}'
    return f'{method}_final'


def compute_state_svd(
    *,
    model: torch.nn.Module,
    sample_state: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    out_state_dir: Path,
    progress_prefix: str,
) -> dict[str, Any]:
    start = time.perf_counter()
    Jf = compute_explicit_jacobian(model, sample_state, device, progress_prefix=f'fno_{progress_prefix}')
    fno_seconds = time.perf_counter() - start
    start = time.perf_counter()
    Jj = compute_solver_jacobian(sample_state, args, device, progress_prefix=f'solver_{progress_prefix}')
    solver_seconds = time.perf_counter() - start
    Jr = Jf.astype(np.float64) - Jj.astype(np.float64)
    start = time.perf_counter()
    U, s, Vh = np.linalg.svd(Jr, full_matrices=False)
    svd_seconds = time.perf_counter() - start
    out_state_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_state_dir / 'residual_jacobian_svd.npz',
        residual_jacobian=Jr.astype(np.float32),
        fno_jacobian=Jf.astype(np.float32),
        solver_jacobian=Jj.astype(np.float32),
        singular_values=s.astype(np.float64),
        left_singular_vectors=U[:, :32].astype(np.float32),
        right_singular_vectors=Vh[:32].astype(np.float32),
    )
    return {
        'singular_values': s,
        'right_singular_vectors': Vh,
        'fno_seconds': fno_seconds,
        'solver_seconds': solver_seconds,
        'svd_seconds': svd_seconds,
    }


def write_figures(out_dir: Path, summary: list[dict[str, Any]], align: list[dict[str, Any]], vector_rows: list[dict[str, Any]]) -> list[str]:
    fig_dir = out_dir / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    if summary:
        labels = [f"{r['pq']}\n{r['state_label']}" for r in summary]
        vals = [float(r['sigma1_over_sigma2']) for r in summary]
        fig, ax = plt.subplots(figsize=(max(8, 0.75 * len(labels)), 4.8))
        ax.bar(range(len(labels)), vals)
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=65, ha='right', fontsize=8)
        ax.set_ylabel('sigma1 / sigma2')
        ax.set_title('Residual-Jacobian spectral gap by current core4/PQ state')
        ax.grid(True, axis='y', alpha=0.25)
        fig.tight_layout(); p = fig_dir / 'residual_spectral_gap_by_state.png'; fig.savefig(p, dpi=180); plt.close(fig); paths.append(str(p))
    if align:
        # Heatmaps by PQ.
        for pq in sorted({r['pq'] for r in align}):
            rows = [r for r in align if r['pq'] == pq]
            states = sorted({r['state_label'] for r in rows})
            dirs = sorted({r['direction_label'] for r in rows})
            M = np.full((len(states), len(dirs)), np.nan)
            for r in rows:
                i = states.index(r['state_label']); j = dirs.index(r['direction_label'])
                M[i, j] = float(r['abs_cos_top1_right_vs_direction'])
            fig, ax = plt.subplots(figsize=(max(7, 0.55 * len(dirs)), max(4, 0.45 * len(states))))
            im = ax.imshow(M, vmin=0, vmax=1, cmap='viridis', aspect='auto')
            ax.set_xticks(range(len(dirs))); ax.set_xticklabels(dirs, rotation=70, ha='right', fontsize=7)
            ax.set_yticks(range(len(states))); ax.set_yticklabels(states, fontsize=8)
            ax.set_title(f'{pq}: top residual singular direction vs deltas')
            cb = fig.colorbar(im, ax=ax); cb.set_label('abs cosine')
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    if np.isfinite(M[i, j]):
                        ax.text(j, i, f'{M[i,j]:.2f}', ha='center', va='center', color='white' if M[i,j] < 0.55 else 'black', fontsize=7)
            fig.tight_layout(); p = fig_dir / f'top_residual_direction_alignment_{pq}.png'; fig.savefig(p, dpi=180); plt.close(fig); paths.append(str(p))
    if vector_rows:
        # Plot p2q2 top vectors if available.
        rows = [r for r in vector_rows if r['pq'] == 'p2q2']
        if rows:
            fig, ax = plt.subplots(figsize=(9, 4.8))
            for r in rows:
                arr = np.asarray(json.loads(r['top1_right_vector_json']), dtype=float)
                ax.plot(arr, linewidth=1.0, label=r['state_label'])
            ax.set_title('p2q2 residual-Jacobian top right singular vector by state')
            ax.set_xlabel('grid index'); ax.set_ylabel('unit vector value')
            ax.grid(True, alpha=0.2); ax.legend(fontsize=7, ncol=2)
            fig.tight_layout(); p = fig_dir / 'p2q2_top_residual_singular_vectors.png'; fig.savefig(p, dpi=180); plt.close(fig); paths.append(str(p))
    return paths


def md_table(rows: list[dict[str, Any]], fields: list[str], max_rows: int = 80) -> str:
    lines = ['| ' + ' | '.join(fields) + ' |', '| ' + ' | '.join(['---'] * len(fields)) + ' |']
    for r in rows[:max_rows]:
        vals = []
        for f in fields:
            v = r.get(f, '')
            try:
                fv = float(v)
                vals.append(f'{fv:.4g}' if math.isfinite(fv) else str(v))
            except Exception:
                vals.append(str(v))
        lines.append('| ' + ' | '.join(vals) + ' |')
    return '\n'.join(lines)


def write_doc(doc: Path, out_dir: Path, manifest: dict[str, Any], summary: list[dict[str, Any]], align: list[dict[str, Any]]) -> None:
    # Compact alignment view for key directions only.
    key_align = [r for r in align if r['direction_label'] in {'steepest_replace_final', 'steepest_replace_step5', 'steepest_add_final', 'raw_add_final'}]
    key_align.sort(key=lambda r: (r['pq'], r['state_label'], r['direction_label']))
    lines = [
        '# Loss3 Current Core4/PQ Jacobian SVD Probe - 2026-05-21', '',
        'Status: completed targeted GPU explicit residual-Jacobian/SVD probe on current core4/PQ states.', '',
        'This is not the old PGD-only path diagnostic. It uses current core4/PQ deltas and selected perturbed states.', '',
        '## Scope', '',
        f"- Output directory: `{rel(out_dir)}`",
        f"- Sample index: `{manifest['sample_index']}`",
        f"- Probe states: `{manifest['probe_state_count']}`",
        f"- GPU runtime recorded in: `{rel(out_dir / 'manifest.json')}`", '',
        '## Main Tables', '',
        f"- State SVD summary: `{rel(out_dir / 'tables/state_svd_summary.csv')}`",
        f"- Top direction alignment: `{rel(out_dir / 'tables/top_direction_alignment.csv')}`",
        f"- Top vector metrics: `{rel(out_dir / 'tables/top_vector_frequency_metrics.csv')}`", '',
        '## Residual-Jacobian Spectral Summary', '',
        md_table(summary, ['pq', 'state_label', 'sigma1', 'sigma2', 'sigma1_over_sigma2', 'top1_energy_fraction', 'top4_energy_fraction', 'top1_peakiness', 'top1_energy_concentration']), '',
        '## Top Residual Singular Direction Alignment Snapshot', '',
        md_table(key_align, ['pq', 'state_label', 'direction_label', 'abs_cos_top1_right_vs_direction', 'angle_deg_top1_right_vs_direction'], max_rows=120), '',
        '## Interpretation Guide', '',
        '- A large `sigma1_over_sigma2` or large `top1_energy_fraction` supports a dominant-mode story.',
        '- A high cosine between the residual-Jacobian top right singular vector and a final/early delta supports the idea that the optimizer rapidly follows a locally dominant direction.',
        '- Low cosine or weak spectral gap means the previous GPI explanation should stay empirical/local-surrogate rather than being upgraded to a strict spectral theorem.',
        '- High `top1_peakiness` or `top1_energy_concentration` supports the spike/concentration concern in p=1 or q=inf geometries.',
    ]
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def run(args: argparse.Namespace) -> dict[str, Any]:
    configure_runtime(args)
    device, gpu_runtime = require_gpu_runtime(str(args.device or 'cuda'))
    out_dir = args.out_dir if args.out_dir.is_absolute() else PROJECT_ROOT / args.out_dir
    roots = [p if p.is_absolute() else PROJECT_ROOT / p for p in args.setting_root]
    if not roots:
        roots = list(DEFAULT_ROOTS)
    args.setting_root = roots
    sample_index = int(args.sample_index)
    x_clean = load_burgers_indices(args.fno_test_path, [sample_index])[0].astype(np.float32)
    model = load_burgers_torch_model(args.fno_checkpoint, device)
    specs = default_probe_specs(args)
    if args.max_states is not None:
        specs = specs[: int(args.max_states)]
    summary_rows: list[dict[str, Any]] = []
    align_rows: list[dict[str, Any]] = []
    vector_rows: list[dict[str, Any]] = []
    state_paths: list[str] = []
    start_all = time.perf_counter()
    for spec_i, spec in enumerate(specs):
        root = Path(spec['root'])
        meta = parse_root(root)
        method = spec['state_method']; kind = spec['state_kind']; step = spec['state_step']
        label = state_label(method, kind, step)
        delta = load_delta(root, method, kind, step, sample_index)
        x_state = (x_clean + delta).astype(np.float32)
        state_dir = out_dir / 'states' / meta['pq'] / label
        print(f"[state {spec_i+1}/{len(specs)}] {meta['pq']} {label}", flush=True)
        svd = compute_state_svd(model=model, sample_state=x_state, args=args, device=device, out_state_dir=state_dir, progress_prefix=f"{meta['pq']}_{label}")
        s = np.asarray(svd['singular_values'], dtype=np.float64)
        Vh = np.asarray(svd['right_singular_vectors'], dtype=np.float64)
        v1 = Vh[0]
        energy = s * s
        top1_energy = float(energy[0] / (energy.sum() + EPS))
        top4_energy = float(energy[:4].sum() / (energy.sum() + EPS))
        cmetrics = concentration(v1)
        fmetrics = vector_frequency_metrics(v1)
        state_row = {
            **meta,
            'sample_index': sample_index,
            'state_label': label,
            'state_method': method,
            'state_kind': kind,
            'state_step': '' if step is None else int(step),
            'state_delta_l2': norm2(delta),
            'sigma1': float(s[0]),
            'sigma2': float(s[1]),
            'sigma1_over_sigma2': float(s[0] / (s[1] + EPS)),
            'sigma2_over_sigma1': float(s[1] / (s[0] + EPS)),
            'top1_energy_fraction': top1_energy,
            'top4_energy_fraction': top4_energy,
            'top1_peakiness': cmetrics['peakiness'],
            'top1_energy_concentration': cmetrics['top1_energy_fraction'],
            'fno_seconds': svd['fno_seconds'],
            'solver_seconds': svd['solver_seconds'],
            'svd_seconds': svd['svd_seconds'],
            'state_dir': str(state_dir),
        }
        summary_rows.append(state_row)
        vector_rows.append({
            **meta,
            'sample_index': sample_index,
            'state_label': label,
            **fmetrics,
            **{f'top1_{k}': v for k, v in cmetrics.items()},
            # Store for the compact p2q2 vector figure; CSV stays small enough.
            'top1_right_vector_json': json.dumps([float(x) for x in v1.tolist()]),
        })
        # Direction bank from the same root.
        direction_bank: dict[str, np.ndarray] = {}
        for m in CORE4:
            try:
                direction_bank[f'{m}_final'] = load_delta(root, m, 'final', None, sample_index)
            except Exception:
                pass
        for st in (1, 5, 10, 20):
            try:
                direction_bank[f'steepest_replace_step{st}'] = load_delta(root, 'steepest_replace', 'trajectory', st, sample_index)
            except Exception:
                pass
        if norm2(delta) > 1e-9:
            direction_bank['state_delta'] = delta
        for dlabel, dval in direction_bank.items():
            c = cosine(v1, dval, abs_value=True)
            align_rows.append({
                **meta,
                'sample_index': sample_index,
                'state_label': label,
                'direction_label': dlabel,
                'direction_l2': norm2(dval),
                'abs_cos_top1_right_vs_direction': c,
                'angle_deg_top1_right_vs_direction': angle_from_abs_cos(c),
            })
        write_json(state_dir / 'state_summary.json', state_row)
        state_paths.append(str(state_dir))
        sync_torch(torch, device)
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    tables = out_dir / 'tables'
    write_csv(tables / 'state_svd_summary.csv', summary_rows)
    write_csv(tables / 'top_direction_alignment.csv', align_rows)
    write_csv(tables / 'top_vector_frequency_metrics.csv', vector_rows)
    fig_paths = write_figures(out_dir, summary_rows, align_rows, vector_rows)
    manifest = {
        'status': 'completed',
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'sample_index': sample_index,
        'probe_state_count': len(specs),
        'setting_roots': [str(p) for p in roots],
        'state_paths': state_paths,
        'runtime_seconds': time.perf_counter() - start_all,
        'gpu_runtime': gpu_runtime,
        'figure_paths': fig_paths,
        'doc': str(args.doc if args.doc.is_absolute() else PROJECT_ROOT / args.doc),
    }
    write_json(out_dir / 'manifest.json', manifest)
    write_doc(args.doc if args.doc.is_absolute() else PROJECT_ROOT / args.doc, out_dir, manifest, summary_rows, align_rows)
    return manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--setting-root', action='append', type=Path, default=list(DEFAULT_ROOTS))
    p.add_argument('--out-dir', type=Path, default=DEFAULT_OUT)
    p.add_argument('--doc', type=Path, default=DEFAULT_DOC)
    p.add_argument('--sample-index', type=int, default=0)
    p.add_argument('--max-states', type=int, default=None)
    p.add_argument('--device', default='cuda')
    p.add_argument('--runtime-workarounds', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--prepend-env-ptxas', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--fno-test-path', type=Path, default=DEFAULT_BURGERS_TEST)
    p.add_argument('--fno-checkpoint', type=Path, default=DEFAULT_BURGERS_MODEL_DIR / 'checkpoints' / 'pytorch_fno1d_500.pt')
    p.add_argument('--burgers-nx', type=int, default=1024)
    p.add_argument('--burgers-nu', type=float, default=0.001)
    p.add_argument('--burgers-t-final', type=float, default=1.0)
    p.add_argument('--burgers-dt', type=float, default=0.001)
    p.add_argument('--burgers-domain', type=float, default=2.0)
    p.add_argument('--burgers-jax-solver-dtype', choices=['float32', 'float64'], default='float64')
    return p.parse_args()


def main() -> None:
    manifest = run(parse_args())
    print(json.dumps(finite_json(manifest), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
