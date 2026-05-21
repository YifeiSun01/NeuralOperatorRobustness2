#!/usr/bin/env python3
"""Small GPU landscape probe for existing Loss3 core4/PQ deltas.

This does not rerun optimizers. It loads existing final/trajectory deltas and
re-evaluates Loss3_q at new ray, arc, 2D slice, and finite-difference curvature
points on GPU.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
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
    make_burgers_jax_solver,
    make_jax_torch_bridge,
    sync_torch,
)
from tools.run_loss3_small_epsilon_sweep import configure_runtime, require_gpu_runtime, finite_json  # noqa: E402
from tools.run_loss3_ray_profile import evaluate_state, load_burgers_indices  # noqa: E402

CORE4 = ('raw_add','raw_replace','steepest_add','steepest_replace')
ROOT_RE = re.compile(r'eps(?P<eps>[^_]+)_alpha(?P<alpha>[^_]+)_batch(?P<batch>\d+)_steps(?P<steps>\d+)_p(?P<p>[^_]+)_q(?P<q>[^_]+)$')
DEFAULT_ROOTS = [
    PROJECT_ROOT / 'forensics/loss3_alpha_epsilon_core4_baseline_giftrace_20260520/fno_nu0p001_eps4_alpha0p4_batch100_steps300_p2_q2',
    PROJECT_ROOT / 'forensics/loss3_alpha_epsilon_core4_sweep_pneq_q_100steps_20260520/fno_nu0p001_eps4_alpha0p4_batch100_steps100_p2_q1',
    PROJECT_ROOT / 'forensics/loss3_alpha_epsilon_core4_sweep_pneq_q_100steps_20260520/fno_nu0p001_eps4_alpha0p4_batch100_steps100_p2_qinf',
    PROJECT_ROOT / 'forensics/loss3_alpha_epsilon_core4_sweep_pneq_q_100steps_20260520/fno_nu0p001_eps4_alpha0p4_batch100_steps100_p1_qinf',
]
DEFAULT_OUT = PROJECT_ROOT / 'forensics/loss3_core4_pq_landscape_probe_20260520'
DEFAULT_DOC = PROJECT_ROOT / 'docs/loss3_core4_pq_landscape_probe_20260520.md'
EPS = 1e-12

def fnum(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return math.nan

def parse_norm_token(s: str) -> float:
    t=str(s).lower().replace('linf','inf').replace('infinity','inf')
    if t == 'inf':
        return float('inf')
    return float(t.replace('p','.'))

def norm_name(x: float) -> str:
    if math.isinf(x):
        return 'inf'
    if abs(x-round(x))<1e-12:
        return str(int(round(x)))
    return str(x).replace('.','p')

def parse_root(root: Path) -> dict[str, Any]:
    m=ROOT_RE.search(root.name)
    if not m:
        raise ValueError(f'Cannot parse setting root name: {root.name}')
    return {
        'setting_root': root.name,
        'epsilon': parse_norm_token(m.group('eps')),
        'alpha': parse_norm_token(m.group('alpha')),
        'steps': int(m.group('steps')),
        'p': parse_norm_token(m.group('p')),
        'q': parse_norm_token(m.group('q')),
        'pq': f"p{m.group('p')}q{m.group('q')}",
    }

def make_solver_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        burgers_nx=args.burgers_nx,
        burgers_nu=args.burgers_nu,
        burgers_t_final=args.burgers_t_final,
        burgers_dt=args.burgers_dt,
        burgers_domain=args.burgers_domain,
        burgers_jax_solver_dtype=args.burgers_jax_solver_dtype,
    )

def write_csv(path: Path, rows: list[dict[str,Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)

def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(finite_json(payload), indent=2, sort_keys=True) + '\n', encoding='utf-8')

def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except Exception:
        return str(path)

def batch_norm(x: torch.Tensor, order: float) -> torch.Tensor:
    flat=x.reshape(x.shape[0], -1)
    if math.isinf(order):
        return torch.amax(torch.abs(flat), dim=1)
    return torch.linalg.vector_norm(flat, ord=float(order), dim=1)

def normalize_batch(delta: torch.Tensor, order: float) -> torch.Tensor:
    flat=delta.reshape(delta.shape[0], -1)
    if math.isinf(order):
        n=torch.amax(torch.abs(flat), dim=1, keepdim=True).clamp_min(EPS)
    else:
        n=torch.linalg.vector_norm(flat, ord=float(order), dim=1, keepdim=True).clamp_min(EPS)
    return (flat/n).reshape_as(delta)

def normalize_l2(delta: torch.Tensor) -> torch.Tensor:
    return normalize_batch(delta, 2.0)

def load_final_delta(root: Path, method: str, sample_indices: list[int]) -> np.ndarray:
    z=np.load(root/'final_deltas.npz', allow_pickle=True)
    methods=[str(x) for x in z['method'].tolist()]
    ds=[int(x) for x in z['dataset_index'].tolist()]
    mi=methods.index(method)
    pos=[ds.index(int(i)) for i in sample_indices]
    return np.asarray(z['final_delta'][mi, pos], dtype=np.float32)

def load_trajectory_delta(root: Path, method: str, step: int, sample_indices: list[int]) -> np.ndarray | None:
    path=root/method/'trajectory_samples.npz'
    if not path.exists():
        return None
    z=np.load(path, allow_pickle=True)
    ks=[int(x) for x in z['k'].tolist()]
    ds=[int(x) for x in z['dataset_index'].tolist()]
    if step not in ks:
        return None
    ki=ks.index(int(step))
    pos=[ds.index(int(i)) for i in sample_indices if int(i) in ds]
    if len(pos) != len(sample_indices):
        return None
    return np.asarray(z['delta'][ki, pos], dtype=np.float32)

def eval_delta(
    *, model: torch.nn.Module, bridge: Any, solver_fn: Any, x0: torch.Tensor,
    f0: torch.Tensor, j0: torch.Tensor, e0: torch.Tensor, delta: torch.Tensor,
    q: float, label: str,
) -> dict[str, torch.Tensor]:
    x_adv=x0+delta
    f_adv,j_adv,e_adv=evaluate_state(model, bridge, solver_fn, x_adv, allow_solver_grad=False, label=label)
    return {
        'loss3_q': batch_norm(e_adv, q),
        'loss3_l2': batch_norm(e_adv, 2.0),
        'loss3_linf': batch_norm(e_adv, float('inf')),
        'clean_loss3_q': batch_norm(e0, q),
        'delta_l2': batch_norm(delta, 2.0),
        'delta_linf': batch_norm(delta, float('inf')),
        'residual_increment_l2': batch_norm(e_adv-e0, 2.0),
        'model_movement_l2': batch_norm(f_adv-f0, 2.0),
        'solver_movement_l2': batch_norm(j_adv-j0, 2.0),
    }

def tensor_np(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(np.asarray(arr, dtype=np.float32), device=device, dtype=torch.float32)

def add_eval_rows(rows: list[dict[str,Any]], base: dict[str,Any], qvals: dict[str,torch.Tensor], sample_indices: list[int]) -> None:
    cpu={k:v.detach().cpu().numpy().astype(float) for k,v in qvals.items()}
    for i,idx in enumerate(sample_indices):
        row={**base, 'sample_position': i, 'sample_index': int(idx)}
        for k,v in cpu.items():
            row[k]=float(v[i])
        rows.append(row)

def aggregate(rows: list[dict[str,Any]], keys: tuple[str,...], metrics: list[str]) -> list[dict[str,Any]]:
    groups=defaultdict(list)
    for r in rows:
        groups[tuple(r.get(k) for k in keys)].append(r)
    out=[]
    for key,items in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        row={k:v for k,v in zip(keys,key)}
        row['n']=len(items)
        for m in metrics:
            arr=np.asarray([fnum(it.get(m)) for it in items if math.isfinite(fnum(it.get(m)))], dtype=float)
            row[f'{m}_mean']=float(arr.mean()) if arr.size else math.nan
            row[f'{m}_std']=float(arr.std(ddof=0)) if arr.size else math.nan
        out.append(row)
    return out

def run_probe(args: argparse.Namespace) -> dict[str,Any]:
    configure_runtime(args)
    device,gpu_runtime=require_gpu_runtime(str(args.device or 'cuda'))
    roots=[p if p.is_absolute() else PROJECT_ROOT/p for p in args.setting_root]
    sample_indices=[int(x) for x in args.sample_indices]
    x_np=load_burgers_indices(args.fno_test_path, sample_indices)
    x0=torch.as_tensor(x_np, device=device, dtype=torch.float32)
    model=load_burgers_torch_model(args.fno_checkpoint, device)
    solver_fn=make_burgers_jax_solver(make_solver_args(args))
    bridge=make_jax_torch_bridge()
    with torch.no_grad():
        f0,j0,e0=evaluate_state(model, bridge, solver_fn, x0, allow_solver_grad=False, label='clean_landscape_probe')
    sync_torch(torch, device)

    out_dir=args.out_dir if args.out_dir.is_absolute() else PROJECT_ROOT/args.out_dir
    rows_ray=[]; rows_arc=[]; rows_slice=[]; rows_curv=[]
    radii_base=np.linspace(0.0, 1.0, int(args.num_radii))
    grid=np.linspace(-1.0,1.0,int(args.slice_grid))
    arc_s=np.linspace(0.0,1.0,int(args.arc_points))
    methods=list(args.methods)
    early_steps=[int(x) for x in args.early_steps]

    for root in roots:
        meta=parse_root(root)
        eps=float(meta['epsilon']); p=float(meta['p']); q=float(meta['q'])
        # Final and trajectory directions for ray profiles.
        direction_specs=[]
        for method in methods:
            d=load_final_delta(root, method, sample_indices)
            direction_specs.append((method, 'final', None, d))
            for step in early_steps:
                td=load_trajectory_delta(root, method, step, sample_indices)
                if td is not None:
                    direction_specs.append((method, 'trajectory', step, td))
        for method, kind, step, d_np in direction_specs:
            d=tensor_np(d_np, device)
            u=normalize_batch(d, p)
            for frac in radii_base:
                radius=eps*float(frac)
                delta=radius*u
                with torch.no_grad():
                    qv=eval_delta(model=model, bridge=bridge, solver_fn=solver_fn, x0=x0, f0=f0, j0=j0, e0=e0, delta=delta, q=q, label=f"ray_{meta['pq']}_{method}_{kind}_{step}_{frac:.3f}")
                add_eval_rows(rows_ray, {**meta, 'method':method, 'direction_kind':kind, 'trajectory_step': step if step is not None else '', 'radius':radius, 'radius_fraction':float(frac)}, qv, sample_indices)
        # Boundary arcs for p=2 only.
        if abs(p-2.0) < 1e-12:
            pairs=[('steepest_replace','steepest_add'),('steepest_replace','raw_add'),('raw_add','steepest_add')]
            final={m:tensor_np(load_final_delta(root,m,sample_indices),device) for m in methods}
            for a,b in pairs:
                if a not in final or b not in final:
                    continue
                ua=normalize_l2(final[a]); ub=normalize_l2(final[b])
                for s in arc_s:
                    u=normalize_l2((1.0-float(s))*ua + float(s)*ub)
                    delta=eps*u
                    with torch.no_grad():
                        qv=eval_delta(model=model, bridge=bridge, solver_fn=solver_fn, x0=x0, f0=f0, j0=j0, e0=e0, delta=delta, q=q, label=f"arc_{meta['pq']}_{a}_{b}_{s:.3f}")
                    add_eval_rows(rows_arc, {**meta, 'pair':f'{a}__{b}', 's':float(s)}, qv, sample_indices)
        # 2D slices and curvature on a tiny selected set to keep runtime bounded.
        slice_centers=[m for m in ['steepest_replace','steepest_add'] if m in methods]
        final_np={m:load_final_delta(root,m,sample_indices) for m in methods}
        final_t={m:tensor_np(v,device) for m,v in final_np.items()}
        rho=float(args.slice_radius_fraction)*eps
        h=float(args.curvature_h_fraction)*eps
        for center_method in slice_centers:
            center=final_t[center_method]
            u1=normalize_l2(final_t.get('steepest_replace', center))
            u2_seed=normalize_l2(final_t.get('steepest_add', center))
            dot=torch.sum(u1.reshape(u1.shape[0],-1)*u2_seed.reshape(u2_seed.shape[0],-1), dim=1).view(-1,1,1)
            u2=normalize_l2(u2_seed - dot*u1)
            # Slice grid.
            center_q=None
            with torch.no_grad():
                center_q=eval_delta(model=model, bridge=bridge, solver_fn=solver_fn, x0=x0, f0=f0, j0=j0, e0=e0, delta=center, q=q, label=f"slice_center_{meta['pq']}_{center_method}")
            for ai,a in enumerate(grid):
                for bi,b in enumerate(grid):
                    delta=center + rho*float(a)*u1 + rho*float(b)*u2
                    with torch.no_grad():
                        qv=eval_delta(model=model, bridge=bridge, solver_fn=solver_fn, x0=x0, f0=f0, j0=j0, e0=e0, delta=delta, q=q, label=f"slice_{meta['pq']}_{center_method}_{ai}_{bi}")
                    add_eval_rows(rows_slice, {**meta, 'center_method':center_method, 'a':float(a), 'b':float(b), 'slice_radius':rho}, qv, sample_indices)
            # Finite-difference curvature along radial and cross-method directions.
            dirs={'radial':normalize_l2(center)}
            if center_method != 'steepest_add' and 'steepest_add' in final_t:
                dirs['toward_steepest_add']=normalize_l2(final_t['steepest_add']-center)
            if center_method != 'steepest_replace' and 'steepest_replace' in final_t:
                dirs['toward_steepest_replace']=normalize_l2(final_t['steepest_replace']-center)
            for direction_name,v in dirs.items():
                with torch.no_grad():
                    q0=eval_delta(model=model, bridge=bridge, solver_fn=solver_fn, x0=x0, f0=f0, j0=j0, e0=e0, delta=center, q=q, label=f"curv0_{meta['pq']}_{center_method}_{direction_name}")
                    qp=eval_delta(model=model, bridge=bridge, solver_fn=solver_fn, x0=x0, f0=f0, j0=j0, e0=e0, delta=center+h*v, q=q, label=f"curvp_{meta['pq']}_{center_method}_{direction_name}")
                    qm=eval_delta(model=model, bridge=bridge, solver_fn=solver_fn, x0=x0, f0=f0, j0=j0, e0=e0, delta=center-h*v, q=q, label=f"curvm_{meta['pq']}_{center_method}_{direction_name}")
                vals=((qp['loss3_q'] - 2*q0['loss3_q'] + qm['loss3_q'])/(h*h if h>0 else 1.0)).detach().cpu().numpy()
                for i,idx in enumerate(sample_indices):
                    rows_curv.append({**meta, 'center_method':center_method, 'direction':direction_name, 'h':h, 'sample_position':i, 'sample_index':int(idx), 'loss3_q_center':float(q0['loss3_q'][i].detach().cpu()), 'curvature_second_diff':float(vals[i])})
        sync_torch(torch, device)

    tables=out_dir/'tables'
    write_csv(tables/'ray_profile.csv', rows_ray)
    write_csv(tables/'boundary_arc.csv', rows_arc)
    write_csv(tables/'slice_2d_grid.csv', rows_slice)
    write_csv(tables/'curvature_finite_difference.csv', rows_curv)
    write_csv(tables/'ray_profile_aggregate.csv', aggregate(rows_ray, ('pq','method','direction_kind','trajectory_step','radius_fraction'), ['loss3_q','loss3_l2','loss3_linf','delta_l2']))
    write_csv(tables/'boundary_arc_aggregate.csv', aggregate(rows_arc, ('pq','pair','s'), ['loss3_q','loss3_l2','loss3_linf']))
    write_csv(tables/'slice_2d_aggregate.csv', aggregate(rows_slice, ('pq','center_method','a','b'), ['loss3_q','loss3_l2','loss3_linf']))
    write_csv(tables/'curvature_aggregate.csv', aggregate(rows_curv, ('pq','center_method','direction'), ['curvature_second_diff','loss3_q_center']))
    fig_paths=write_figures(out_dir, rows_ray, rows_arc, rows_slice, rows_curv)
    manifest={
        'status':'completed',
        'generated_at_utc':datetime.now(timezone.utc).isoformat(),
        'setting_roots':[str(r) for r in roots],
        'sample_indices':sample_indices,
        'methods':methods,
        'out_dir':str(out_dir),
        'doc':str(args.doc if args.doc.is_absolute() else PROJECT_ROOT/args.doc),
        'row_counts':{'ray':len(rows_ray),'arc':len(rows_arc),'slice':len(rows_slice),'curvature':len(rows_curv)},
        'gpu_runtime':gpu_runtime,
        'figure_paths':fig_paths,
    }
    write_json(out_dir/'manifest.json', manifest)
    write_doc(args.doc if args.doc.is_absolute() else PROJECT_ROOT/args.doc, out_dir, manifest)
    return manifest

def write_figures(out_dir: Path, ray_rows: list[dict[str,Any]], arc_rows: list[dict[str,Any]], slice_rows: list[dict[str,Any]], curv_rows: list[dict[str,Any]]) -> list[str]:
    fig_dir=out_dir/'figures'; fig_dir.mkdir(parents=True, exist_ok=True)
    paths=[]
    # Ray mean curves: final directions only for readability.
    final=[r for r in ray_rows if r.get('direction_kind')=='final']
    groups=defaultdict(list)
    for r in final:
        groups[(r['pq'],r['method'],float(r['radius_fraction']))].append(float(r['loss3_q']))
    for pq in sorted({r['pq'] for r in final}):
        fig,ax=plt.subplots(figsize=(7,4.5))
        for method in CORE4:
            xs=[]; ys=[]; yerr=[]
            for frac in sorted({k[2] for k in groups if k[0]==pq and k[1]==method}):
                vals=np.asarray(groups[(pq,method,frac)], dtype=float)
                xs.append(frac); ys.append(vals.mean()); yerr.append(vals.std(ddof=0))
            if xs:
                ax.plot(xs,ys,label=method,linewidth=1.8)
                ax.fill_between(xs,np.asarray(ys)-np.asarray(yerr),np.asarray(ys)+np.asarray(yerr),alpha=0.13)
        ax.set_title(f'{pq}: ray profile along final deltas')
        ax.set_xlabel('radius / epsilon'); ax.set_ylabel('Loss3 q-norm')
        ax.grid(True,alpha=0.25); ax.legend(fontsize=8,ncol=2)
        fig.tight_layout(); p=fig_dir/f'ray_final_{pq}.png'; fig.savefig(p,dpi=180); plt.close(fig); paths.append(str(p))
    # Early GPI ray for p2q2 if available.
    early=[r for r in ray_rows if r['pq']=='p2q2' and r['method'] in ('raw_replace','steepest_replace') and r.get('direction_kind') in ('trajectory','final')]
    if early:
        fig,ax=plt.subplots(figsize=(7,4.5))
        labels=sorted({(r['method'],r['direction_kind'],str(r.get('trajectory_step'))) for r in early})
        for method,kind,step in labels:
            xs=[]; ys=[]
            for frac in sorted({float(r['radius_fraction']) for r in early if r['method']==method and r['direction_kind']==kind and str(r.get('trajectory_step'))==step}):
                vals=[float(r['loss3_q']) for r in early if r['method']==method and r['direction_kind']==kind and str(r.get('trajectory_step'))==step and abs(float(r['radius_fraction'])-frac)<1e-12]
                xs.append(frac); ys.append(float(np.mean(vals)))
            if xs:
                lab=f'{method}_{kind}{step if step else ""}'
                ax.plot(xs,ys,label=lab,linewidth=1.4)
        ax.set_title('p2q2 replacement early-step ray profiles')
        ax.set_xlabel('radius / epsilon'); ax.set_ylabel('Loss3 q-norm')
        ax.grid(True,alpha=0.25); ax.legend(fontsize=7,ncol=2)
        fig.tight_layout(); p=fig_dir/'ray_p2q2_replacement_early_steps.png'; fig.savefig(p,dpi=180); plt.close(fig); paths.append(str(p))
    # Boundary arcs.
    for pq in sorted({r['pq'] for r in arc_rows}):
        fig,ax=plt.subplots(figsize=(7,4.5))
        for pair in sorted({r['pair'] for r in arc_rows if r['pq']==pq}):
            xs=[]; ys=[]
            for s in sorted({float(r['s']) for r in arc_rows if r['pq']==pq and r['pair']==pair}):
                vals=[float(r['loss3_q']) for r in arc_rows if r['pq']==pq and r['pair']==pair and abs(float(r['s'])-s)<1e-12]
                xs.append(s); ys.append(float(np.mean(vals)))
            ax.plot(xs,ys,label=pair,linewidth=1.8)
        ax.set_title(f'{pq}: p=2 boundary arc interpolation')
        ax.set_xlabel('arc interpolation s'); ax.set_ylabel('Loss3 q-norm')
        ax.grid(True,alpha=0.25); ax.legend(fontsize=8)
        fig.tight_layout(); p=fig_dir/f'boundary_arc_{pq}.png'; fig.savefig(p,dpi=180); plt.close(fig); paths.append(str(p))
    # Curvature bar.
    agg=aggregate(curv_rows, ('pq','center_method','direction'), ['curvature_second_diff'])
    if agg:
        labels=[f"{r['pq']}\n{r['center_method']}\n{r['direction']}" for r in agg]
        vals=[float(r['curvature_second_diff_mean']) for r in agg]
        fig,ax=plt.subplots(figsize=(max(8,0.5*len(labels)),4.8))
        ax.bar(range(len(labels)),vals)
        ax.axhline(0,color='black',linewidth=0.8)
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels,rotation=65,ha='right',fontsize=7)
        ax.set_ylabel('finite-diff curvature')
        ax.set_title('Local finite-difference curvature by PQ/center/direction')
        ax.grid(True,axis='y',alpha=0.25)
        fig.tight_layout(); p=fig_dir/'curvature_summary.png'; fig.savefig(p,dpi=180); plt.close(fig); paths.append(str(p))
    return paths

def md_table(rows: list[dict[str,Any]], fields: list[str], max_rows: int=20) -> str:
    rows=rows[:max_rows]
    lines=['| '+' | '.join(fields)+' |','| '+' | '.join(['---']*len(fields))+' |']
    for r in rows:
        vals=[]
        for f in fields:
            v=r.get(f,'')
            fv=fnum(v)
            vals.append(f'{fv:.4g}' if math.isfinite(fv) else str(v))
        lines.append('| '+' | '.join(vals)+' |')
    return '\n'.join(lines)

def read_csv(path: Path) -> list[dict[str,str]]:
    with path.open('r',newline='',encoding='utf-8') as f:
        return list(csv.DictReader(f))

def write_doc(doc: Path, out_dir: Path, manifest: dict[str,Any]) -> None:
    ray=read_csv(out_dir/'tables/ray_profile_aggregate.csv')
    arc=read_csv(out_dir/'tables/boundary_arc_aggregate.csv')
    curv=read_csv(out_dir/'tables/curvature_aggregate.csv')
    # endpoint final ray rows only
    endpoint=[r for r in ray if r['direction_kind']=='final' and abs(fnum(r['radius_fraction'])-1.0)<1e-12]
    endpoint.sort(key=lambda r:(r['pq'],r['method']))
    lines=[
        '# Loss3 Core4/PQ Landscape Probe - 2026-05-20','',
        'Status: completed small GPU evaluation pilot. It reused existing core4 deltas and did not rerun optimizers.','',
        '## Scope','',
        f"- Setting roots: `{len(manifest['setting_roots'])}`",
        f"- Sample indices: `{manifest['sample_indices']}`",
        f"- Output directory: `{rel(out_dir)}`",
        f"- Row counts: `{manifest['row_counts']}`",'',
        'GPU runtime was verified and recorded in `manifest.json` before evaluating the landscape points.','',
        '## Tables and Figures','',
        f"- Ray profile table: `{rel(out_dir/'tables/ray_profile.csv')}`",
        f"- Boundary arc table: `{rel(out_dir/'tables/boundary_arc.csv')}`",
        f"- 2D slice grid: `{rel(out_dir/'tables/slice_2d_grid.csv')}`",
        f"- Curvature table: `{rel(out_dir/'tables/curvature_finite_difference.csv')}`",
        f"- Figures: `{rel(out_dir/'figures')}`",'',
        '## Endpoint Ray Loss Along Final Delta Directions','',
        md_table(endpoint, ['pq','method','direction_kind','trajectory_step','radius_fraction','loss3_q_mean','loss3_q_std','delta_l2_mean'], max_rows=80),'',
        '## Boundary Arc Aggregate Snapshot','',
        md_table(arc, ['pq','pair','s','loss3_q_mean','loss3_q_std'], max_rows=40),'',
        '## Curvature Aggregate Snapshot','',
        md_table(curv, ['pq','center_method','direction','curvature_second_diff_mean','curvature_second_diff_std','loss3_q_center_mean'], max_rows=80),'',
        '## Interpretation Notes','',
        '- This is the first small landscape probe tied to the current core4/PQ deltas rather than the older PGD-only path experiments.',
        '- Ray profiles test whether early/final GPI directions are already strong along the full radius.',
        '- p=2 boundary arcs test whether method final deltas are connected by a high-loss ridge on the L2 boundary.',
        '- 2D slices and finite-difference curvature are pilot-scale; use them as directional evidence, not final statistics.',
    ]
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text('\n'.join(lines)+'\n', encoding='utf-8')

def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--setting-root', action='append', type=Path, default=[])
    p.add_argument('--out-dir', type=Path, default=DEFAULT_OUT)
    p.add_argument('--doc', type=Path, default=DEFAULT_DOC)
    p.add_argument('--sample-indices', nargs='+', type=int, default=[0,7,40,47])
    p.add_argument('--methods', nargs='+', default=list(CORE4))
    p.add_argument('--early-steps', nargs='+', type=int, default=[1,5,10,20])
    p.add_argument('--num-radii', type=int, default=25)
    p.add_argument('--arc-points', type=int, default=21)
    p.add_argument('--slice-grid', type=int, default=7)
    p.add_argument('--slice-radius-fraction', type=float, default=0.08)
    p.add_argument('--curvature-h-fraction', type=float, default=0.02)
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
    p.add_argument('--burgers-jax-solver-dtype', choices=['float32','float64'], default='float64')
    args=p.parse_args()
    if not args.setting_root:
        args.setting_root=list(DEFAULT_ROOTS)
    return args

def main() -> None:
    start=time.perf_counter()
    manifest=run_probe(parse_args())
    manifest['runtime_seconds']=time.perf_counter()-start
    out_dir=Path(manifest['out_dir'])
    write_json(out_dir/'manifest.json', manifest)
    print(json.dumps(finite_json(manifest), indent=2))

if __name__ == '__main__':
    main()
