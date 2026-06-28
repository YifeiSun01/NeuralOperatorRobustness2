#!/usr/bin/env python3
"""Post-process all available p=2 Loss3 core4 runs for tangent geometry.

This is CSV/NPZ post-processing only. It extends the p2q2 tangent KKT residual
check to every available p=2,q in the current core4 sweeps.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / 'forensics' / 'loss3_all_p2_tangent_geometry_probe_20260520'
DEFAULT_DOC = PROJECT_ROOT / 'docs' / 'loss3_all_p2_tangent_geometry_probe_20260520.md'
CORE4 = ('raw_add','raw_replace','steepest_add','steepest_replace')
DEFAULT_SWEEPS = [
    PROJECT_ROOT / 'forensics' / 'loss3_alpha_epsilon_core4_sweep_p2q2_300steps_20260520',
    PROJECT_ROOT / 'forensics' / 'loss3_alpha_epsilon_core4_sweep_pneq_q_100steps_20260520',
]
ROOT_RE = re.compile(r'eps(?P<eps>[^_]+)_alpha(?P<alpha>[^_]+)_batch(?P<batch>\d+)_steps(?P<steps>\d+)_p(?P<p>[^_]+)_q(?P<q>[^_]+)$')

def fnum(x: Any) -> float:
    if x in (None, '', 'nan', 'NaN'):
        return math.nan
    try:
        return float(x)
    except Exception:
        return math.nan

def finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False

def fmt_token(s: str) -> float:
    return fnum(s.replace('p','.').replace('inf','inf'))

def parse_root(root: Path) -> dict[str, Any]:
    m = ROOT_RE.search(root.name)
    if not m:
        return {'setting_root': root.name}
    q = m.group('q')
    p = m.group('p')
    return {
        'setting_root': root.name,
        'epsilon': fmt_token(m.group('eps')),
        'alpha': fmt_token(m.group('alpha')),
        'steps': int(m.group('steps')),
        'p': p,
        'q': q,
        'pq': f'p{p}q{q}',
    }

def read_csv(path: Path) -> list[dict[str,str]]:
    with path.open('r', newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

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

def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except Exception:
        return str(path)

def tangent_ratio(cosv: float) -> float:
    if not math.isfinite(cosv):
        return math.nan
    c=max(-1.0,min(1.0,cosv))
    return math.sqrt(max(0.0,1.0-c*c))

def stats(vals: list[Any]) -> dict[str,float]:
    arr=np.asarray([fnum(v) for v in vals if finite(v)], dtype=np.float64)
    if arr.size == 0:
        return {'mean':math.nan,'std':math.nan,'median':math.nan,'n':0}
    return {'mean':float(arr.mean()),'std':float(arr.std(ddof=0)),'median':float(np.median(arr)),'n':int(arr.size)}

def pearson(xs: list[Any], ys: list[Any]) -> tuple[float,int]:
    pairs=[]
    for x,y in zip(xs,ys):
        x=fnum(x); y=fnum(y)
        if math.isfinite(x) and math.isfinite(y):
            pairs.append((x,y))
    if len(pairs)<3:
        return math.nan,len(pairs)
    x=np.asarray([p[0] for p in pairs], dtype=np.float64)
    y=np.asarray([p[1] for p in pairs], dtype=np.float64)
    if x.std()<=1e-12 or y.std()<=1e-12:
        return math.nan,len(pairs)
    return float(np.corrcoef(x,y)[0,1]),len(pairs)

def find_roots(sweeps: list[Path]) -> list[Path]:
    roots=[]
    for sweep in sweeps:
        for csv_path in sweep.glob('*/per_sample_step_metrics.csv'):
            meta=parse_root(csv_path.parent)
            if str(meta.get('p')) == '2':
                roots.append(csv_path.parent)
    return sorted(roots, key=lambda p: p.name)

def group_rows(rows: list[dict[str,str]]) -> dict[tuple[str,int], list[dict[str,str]]]:
    groups=defaultdict(list)
    for row in rows:
        method=row.get('method','')
        if method not in CORE4:
            continue
        sample=int(float(row.get('sample_position',0)))
        groups[(method,sample)].append(row)
    for k in groups:
        groups[k].sort(key=lambda r:int(float(r.get('k',0))))
    return groups

def analyze_root(root: Path, threshold: float) -> list[dict[str,Any]]:
    meta=parse_root(root)
    rows=read_csv(root/'per_sample_step_metrics.csv')
    groups=group_rows(rows)
    out=[]
    for (method,sample), group in groups.items():
        if not group:
            continue
        hit_i=None
        for i,row in enumerate(group):
            if fnum(row.get('boundary_ratio')) >= threshold:
                hit_i=i; break
        final=group[-1]
        last_grad = next((r for r in reversed(group) if finite(r.get('cos_delta_grad'))), final)
        hit=group[hit_i] if hit_i is not None else None
        after=group[hit_i:] if hit_i is not None else []
        tangent_after=[tangent_ratio(fnum(r.get('cos_delta_grad'))) for r in after]
        angle_after=[fnum(r.get('delta_prev_angle_degrees')) for r in after if fnum(r.get('k')) > (fnum(hit.get('k')) if hit else math.inf)]
        step_gain_after=[]
        for a,b in zip(after[:-1], after[1:]):
            la=fnum(a.get('loss3_q')); lb=fnum(b.get('loss3_q'))
            if math.isfinite(la) and math.isfinite(lb):
                step_gain_after.append(lb-la)
        row={**meta, 'method':method, 'sample_position':sample}
        row.update({
            'hit_step_99': fnum(hit.get('k')) if hit else math.nan,
            'hit_loss3_q': fnum(hit.get('loss3_q')) if hit else math.nan,
            'final_loss3_q': fnum(final.get('loss3_q')),
            'post_boundary_gain': (fnum(final.get('loss3_q'))-fnum(hit.get('loss3_q'))) if hit else math.nan,
            'tangent_residual_at_hit': tangent_ratio(fnum(hit.get('cos_delta_grad'))) if hit else math.nan,
            'radial_signed_ratio_at_hit': fnum(hit.get('cos_delta_grad')) if hit else math.nan,
            'tangent_residual_last_grad': tangent_ratio(fnum(last_grad.get('cos_delta_grad'))),
            'radial_signed_ratio_last_grad': fnum(last_grad.get('cos_delta_grad')),
            'mean_tangent_residual_after_hit': stats(tangent_after)['mean'],
            'mean_angle_after_hit_deg': stats(angle_after)['mean'],
            'mean_step_loss_gain_after_hit': stats(step_gain_after)['mean'],
            'n_after_hit': len(after),
            'input_csv': str(root/'per_sample_step_metrics.csv'),
        })
        out.append(row)
    return out

def rollup(rows: list[dict[str,Any]], keys: tuple[str,...]) -> list[dict[str,Any]]:
    groups=defaultdict(list)
    for r in rows:
        groups[tuple(r.get(k) for k in keys)].append(r)
    metrics=['hit_step_99','post_boundary_gain','tangent_residual_at_hit','radial_signed_ratio_at_hit','tangent_residual_last_grad','mean_tangent_residual_after_hit','mean_angle_after_hit_deg','mean_step_loss_gain_after_hit','final_loss3_q']
    out=[]
    for key,items in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        row={k:v for k,v in zip(keys,key)}
        row['n_sample_rows']=len(items)
        for m in metrics:
            s=stats([it.get(m) for it in items])
            row[f'{m}_mean']=s['mean']; row[f'{m}_std']=s['std']; row[f'{m}_median']=s['median']
        out.append(row)
    return out

def corr_rows(rows: list[dict[str,Any]]) -> list[dict[str,Any]]:
    groups=defaultdict(list)
    for r in rows:
        groups[(r.get('pq'), r.get('method'))].append(r)
    out=[]
    for (pq,method),items in sorted(groups.items()):
        for metric in ['tangent_residual_at_hit','radial_signed_ratio_at_hit','tangent_residual_last_grad','mean_angle_after_hit_deg','mean_tangent_residual_after_hit']:
            rr,n=pearson([it.get(metric) for it in items],[it.get('post_boundary_gain') for it in items])
            out.append({'pq':pq,'method':method,'metric':metric,'pearson_r_with_post_boundary_gain':rr,'pair_count':n})
    return out

def md_table(rows: list[dict[str,Any]], fields: list[str], max_rows: int|None=None) -> str:
    rows = rows if max_rows is None else rows[:max_rows]
    lines=['| '+' | '.join(fields)+' |','| '+' | '.join(['---']*len(fields))+' |']
    for r in rows:
        vals=[]
        for f in fields:
            v=r.get(f,'')
            fv=fnum(v)
            vals.append(f'{fv:.4g}' if math.isfinite(fv) else str(v))
        lines.append('| '+' | '.join(vals)+' |')
    return '\n'.join(lines)

def write_doc(doc: Path, out_dir: Path, manifest: dict[str,Any], roll_method: list[dict[str,Any]], corr: list[dict[str,Any]]) -> None:
    fields=['pq','method','n_sample_rows','hit_step_99_mean','post_boundary_gain_mean','tangent_residual_at_hit_mean','tangent_residual_last_grad_mean','mean_angle_after_hit_deg_mean','final_loss3_q_mean']
    corr_fields=['pq','method','metric','pearson_r_with_post_boundary_gain','pair_count']
    lines=[
        '# Loss3 All p=2 Tangent Geometry Probe - 2026-05-20','',
        'Status: generated from existing core4 per-sample metrics; no optimizer/model experiment was rerun.','',
        '## Scope','',
        f"- Setting roots processed: `{manifest['setting_root_count']}`",
        f"- Sample/method rows: `{manifest['sample_method_row_count']}`",
        f"- Output directory: `{rel(out_dir)}`",'',
        'This extends the p2q2 tangent KKT residual check to all available p=2 settings in the current core4 sweeps: p2q2, p2q1, and p2qinf where files exist.','',
        'Metric:', '',
        '```text', 'tangent_residual = sqrt(1 - cos(delta, grad)^2)', '```','',
        'For p=2 this equals `||(I-u u^T) grad L|| / ||grad L||`.','',
        '## Rollup By PQ And Method','',
        md_table(roll_method, fields),'',
        '## Correlations With Post-Boundary Gain','',
        md_table(corr, corr_fields, max_rows=80),'',
        '## Interpretation','',
        '- Observed evidence here can support or qualify the p2 boundary-stationarity story across q values.',
        '- This still does not evaluate new landscape points. It only uses the gradients/cosines already saved during the completed optimizer runs.',
        '- If p2qinf has high tangent residual but weak gain or high peakiness elsewhere, that supports the idea that tangent opportunity must align with useful/non-spiky loss geometry.',
    ]
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text('\n'.join(lines)+'\n', encoding='utf-8')

def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sweep-root', action='append', type=Path, default=[])
    ap.add_argument('--out-dir', type=Path, default=DEFAULT_OUT)
    ap.add_argument('--doc', type=Path, default=DEFAULT_DOC)
    ap.add_argument('--threshold', type=float, default=0.99)
    args=ap.parse_args()
    sweeps=args.sweep_root or DEFAULT_SWEEPS
    out_dir=args.out_dir if args.out_dir.is_absolute() else PROJECT_ROOT/args.out_dir
    doc=args.doc if args.doc.is_absolute() else PROJECT_ROOT/args.doc
    roots=find_roots(sweeps)
    sample_rows=[]
    for root in roots:
        sample_rows.extend(analyze_root(root,args.threshold))
    tables=out_dir/'tables'
    write_csv(tables/'p2_tangent_by_setting_method_sample.csv', sample_rows)
    by_setting=rollup(sample_rows, ('pq','setting_root','epsilon','alpha','steps','method'))
    by_method=rollup(sample_rows, ('pq','method'))
    corrs=corr_rows(sample_rows)
    write_csv(tables/'p2_tangent_rollup_by_setting_method.csv', by_setting)
    write_csv(tables/'p2_tangent_rollup_by_pq_method.csv', by_method)
    write_csv(tables/'p2_tangent_correlations_by_pq_method.csv', corrs)
    manifest={
        'status':'completed',
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'sweeps':[str(p) for p in sweeps],
        'out_dir':str(out_dir),
        'doc':str(doc),
        'setting_root_count':len(roots),
        'sample_method_row_count':len(sample_rows),
        'threshold':args.threshold,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir/'manifest.json').write_text(json.dumps(manifest,indent=2), encoding='utf-8')
    write_doc(doc,out_dir,manifest,by_method,corrs)
    print(json.dumps(manifest,indent=2))

if __name__ == '__main__':
    main()
