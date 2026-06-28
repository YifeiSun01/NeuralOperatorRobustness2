#!/usr/bin/env python3
"""Create summary tables/figures for current core4/PQ Loss3 mechanism validation."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LANDSCAPE = PROJECT_ROOT / 'forensics/loss3_core4_pq_landscape_probe_full_20260521'
DEFAULT_TANGENT = PROJECT_ROOT / 'forensics/loss3_all_p2_tangent_geometry_probe_20260520'
DEFAULT_STEPWISE = PROJECT_ROOT / 'forensics/loss3_p2q2_stepwise_gain_geometry_probe_20260520'
DEFAULT_JAC = PROJECT_ROOT / 'forensics/loss3_current_core4_jacobian_svd_probe_20260521'
DEFAULT_EARLY_LANDSCAPE = PROJECT_ROOT / 'forensics/loss3_core4_pq_landscape_probe_trajectory_20260521'
DEFAULT_OUT = PROJECT_ROOT / 'forensics/loss3_current_mechanism_validation_summary_20260521'
DEFAULT_DOC = PROJECT_ROOT / 'docs/loss3_current_mechanism_validation_summary_20260521.md'
EPS=1e-12


def fnum(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return math.nan


def read_csv(path: Path) -> list[dict[str,str]]:
    if not path.exists():
        return []
    with path.open('r', newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str,Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8'); return
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except Exception:
        return str(path)


def stats(vals: list[float]) -> dict[str,float]:
    arr=np.asarray([v for v in vals if math.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {'mean':math.nan,'std':math.nan,'min':math.nan,'max':math.nan}
    return {'mean':float(arr.mean()),'std':float(arr.std(ddof=0)),'min':float(arr.min()),'max':float(arr.max())}


def md_table(rows: list[dict[str,Any]], fields: list[str], max_rows: int=80) -> str:
    lines=['| '+' | '.join(fields)+' |','| '+' | '.join(['---']*len(fields))+' |']
    for r in rows[:max_rows]:
        vals=[]
        for f in fields:
            v=r.get(f,'')
            fv=fnum(v)
            vals.append(f'{fv:.4g}' if math.isfinite(fv) else str(v))
        lines.append('| '+' | '.join(vals)+' |')
    return '\n'.join(lines)


def summarize_ray(ray: list[dict[str,str]]) -> tuple[list[dict[str,Any]], list[dict[str,Any]]]:
    endpoint=[]
    for r in ray:
        if abs(fnum(r.get('radius_fraction'))-1.0)<1e-9:
            endpoint.append({
                'pq':r.get('pq'), 'method':r.get('method'), 'direction_kind':r.get('direction_kind'),
                'trajectory_step':r.get('trajectory_step'), 'loss3_q_mean':fnum(r.get('loss3_q_mean')),
                'loss3_q_std':fnum(r.get('loss3_q_std')), 'delta_l2_mean':fnum(r.get('delta_l2_mean')),
            })
    # early replacement ratios within each pq/method against final endpoint.
    by_key=defaultdict(dict)
    for r in endpoint:
        by_key[(r['pq'], r['method'])][(r['direction_kind'], str(r['trajectory_step']))]=r['loss3_q_mean']
    early=[]
    for (pq,method),d in sorted(by_key.items()):
        final=d.get(('final',''))
        final = fnum(final)
        if not math.isfinite(final):
            continue
        for step in ['1','5','10','20']:
            v=d.get(('trajectory',step))
            v = fnum(v)
            if math.isfinite(v):
                early.append({'pq':pq,'method':method,'step':int(step),'endpoint_loss':v,'final_endpoint_loss':final,'endpoint_to_final_ratio':v/(final+EPS)})
    return endpoint, early


def summarize_arc(arc: list[dict[str,str]]) -> list[dict[str,Any]]:
    groups=defaultdict(list)
    for r in arc:
        groups[(r.get('pq'),r.get('pair'))].append(r)
    out=[]
    for (pq,pair),items in sorted(groups.items()):
        vals=[(fnum(r.get('s')), fnum(r.get('loss3_q_mean')), fnum(r.get('loss3_q_std'))) for r in items]
        vals=[v for v in vals if math.isfinite(v[0]) and math.isfinite(v[1])]
        if not vals:
            continue
        left=next((y for s,y,_ in vals if abs(s)<1e-9), math.nan)
        right=next((y for s,y,_ in vals if abs(s-1)<1e-9), math.nan)
        min_s,min_loss,_=min(vals, key=lambda t:t[1])
        max_s,max_loss,_=max(vals, key=lambda t:t[1])
        weaker=min(left,right) if math.isfinite(left) and math.isfinite(right) else math.nan
        stronger=max(left,right) if math.isfinite(left) and math.isfinite(right) else math.nan
        out.append({'pq':pq,'pair':pair,'left_loss':left,'right_loss':right,'min_arc_loss':min_loss,'min_arc_s':min_s,'max_arc_loss':max_loss,'max_arc_s':max_s,'min_over_weaker_endpoint':min_loss/(weaker+EPS),'min_over_stronger_endpoint':min_loss/(stronger+EPS),'arc_depth_from_weaker':(weaker-min_loss) if math.isfinite(weaker) else math.nan})
    return out


def summarize_slice(slice_rows: list[dict[str,str]]) -> list[dict[str,Any]]:
    groups=defaultdict(list)
    for r in slice_rows:
        groups[(r.get('pq'),r.get('center_method'))].append(r)
    out=[]
    for (pq,center),items in sorted(groups.items()):
        vals=[fnum(r.get('loss3_q_mean')) for r in items]
        s=stats(vals)
        center_vals=[fnum(r.get('loss3_q_mean')) for r in items if abs(fnum(r.get('a')))<1e-9 and abs(fnum(r.get('b')))<1e-9]
        center_loss=center_vals[0] if center_vals else math.nan
        out.append({'pq':pq,'center_method':center,'center_loss':center_loss,'slice_loss_mean':s['mean'],'slice_loss_std':s['std'],'slice_loss_min':s['min'],'slice_loss_max':s['max'],'slice_range':s['max']-s['min'] if math.isfinite(s['max']) and math.isfinite(s['min']) else math.nan,'center_minus_min':center_loss-s['min'] if math.isfinite(center_loss) and math.isfinite(s['min']) else math.nan})
    return out


def write_figures(out_dir: Path, ray_endpoint: list[dict[str,Any]], early: list[dict[str,Any]], arc_summary: list[dict[str,Any]], slice_rows: list[dict[str,str]], curvature: list[dict[str,str]], jac_summary: list[dict[str,str]]) -> list[str]:
    fig_dir=out_dir/'figures'; fig_dir.mkdir(parents=True, exist_ok=True)
    paths=[]
    # Endpoint bar by PQ.
    for pq in sorted({r['pq'] for r in ray_endpoint if r['direction_kind']=='final'}):
        rows=[r for r in ray_endpoint if r['pq']==pq and r['direction_kind']=='final']
        if not rows: continue
        fig,ax=plt.subplots(figsize=(6.5,4.2))
        labels=[r['method'] for r in rows]; vals=[r['loss3_q_mean'] for r in rows]; errs=[r['loss3_q_std'] for r in rows]
        ax.bar(range(len(rows)), vals, yerr=errs, capsize=3)
        ax.set_xticks(range(len(rows))); ax.set_xticklabels(labels, rotation=30, ha='right')
        ax.set_ylabel('endpoint Loss3 q mean'); ax.set_title(f'{pq}: ray endpoint along final deltas')
        ax.grid(True,axis='y',alpha=0.25); fig.tight_layout()
        p=fig_dir/f'ray_endpoint_final_{pq}.png'; fig.savefig(p,dpi=180); plt.close(fig); paths.append(str(p))
    # Early ratio.
    if early:
        rows=[r for r in early if r['method'] in ('steepest_replace','raw_replace')]
        fig,ax=plt.subplots(figsize=(7.5,4.5))
        for (pq,method),items in sorted(defaultdict(list, {}).items()):
            pass
        groups=defaultdict(list)
        for r in rows: groups[(r['pq'],r['method'])].append(r)
        for (pq,method),items in groups.items():
            items=sorted(items,key=lambda r:r['step'])
            ax.plot([r['step'] for r in items],[r['endpoint_to_final_ratio'] for r in items],marker='o',label=f'{pq}:{method}')
        ax.axhline(1.0,color='black',linewidth=0.8)
        ax.set_xlabel('trajectory step'); ax.set_ylabel('ray endpoint / final endpoint')
        ax.set_title('Early replacement directions vs final ray endpoint')
        ax.grid(True,alpha=0.25); ax.legend(fontsize=8,ncol=2)
        fig.tight_layout(); p=fig_dir/'early_replacement_endpoint_ratio.png'; fig.savefig(p,dpi=180); plt.close(fig); paths.append(str(p))
    # Arc dip.
    if arc_summary:
        labels=[f"{r['pq']}\n{r['pair']}" for r in arc_summary]
        vals=[r['min_over_weaker_endpoint'] for r in arc_summary]
        fig,ax=plt.subplots(figsize=(max(8,0.7*len(labels)),4.6))
        ax.bar(range(len(labels)), vals)
        ax.axhline(1.0,color='black',linewidth=0.8)
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels,rotation=65,ha='right',fontsize=8)
        ax.set_ylabel('min arc loss / weaker endpoint loss')
        ax.set_title('Boundary arc dip: values near 1 mean shared high-loss ridge')
        ax.grid(True,axis='y',alpha=0.25); fig.tight_layout()
        p=fig_dir/'boundary_arc_dip_ratio.png'; fig.savefig(p,dpi=180); plt.close(fig); paths.append(str(p))
    # Slice contours.
    groups=defaultdict(list)
    for r in slice_rows:
        groups[(r.get('pq'),r.get('center_method'))].append(r)
    for (pq,center),items in sorted(groups.items()):
        xs=sorted({fnum(r.get('a')) for r in items if math.isfinite(fnum(r.get('a')))}); ys=sorted({fnum(r.get('b')) for r in items if math.isfinite(fnum(r.get('b')))} )
        if not xs or not ys: continue
        Z=np.full((len(ys),len(xs)),np.nan)
        for r in items:
            a=fnum(r.get('a')); b=fnum(r.get('b')); val=fnum(r.get('loss3_q_mean'))
            if a in xs and b in ys: Z[ys.index(b), xs.index(a)]=val
        fig,ax=plt.subplots(figsize=(5.5,4.6))
        im=ax.contourf(xs,ys,Z,levels=18,cmap='viridis')
        ax.contour(xs,ys,Z,levels=8,colors='white',linewidths=0.45,alpha=0.65)
        ax.plot([0],[0],marker='x',color='red',markersize=8,label='center')
        ax.set_xlabel('a direction coefficient'); ax.set_ylabel('b direction coefficient')
        ax.set_title(f'{pq}: 2D loss slice around {center}')
        fig.colorbar(im,ax=ax,label='mean Loss3 q')
        ax.legend(fontsize=8); fig.tight_layout()
        p=fig_dir/f'slice2d_contour_{pq}_{center}.png'; fig.savefig(p,dpi=180); plt.close(fig); paths.append(str(p))
    # Curvature compact.
    if curvature:
        rows=curvature
        labels=[f"{r.get('pq')}\n{r.get('center_method')}\n{r.get('direction')}" for r in rows]
        vals=[fnum(r.get('curvature_second_diff_mean')) for r in rows]
        fig,ax=plt.subplots(figsize=(max(8,0.48*len(labels)),4.5))
        ax.bar(range(len(labels)), vals); ax.axhline(0,color='black',linewidth=0.8)
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels,rotation=65,ha='right',fontsize=7)
        ax.set_ylabel('finite-diff curvature mean'); ax.set_title('Curvature probe by PQ/center/direction')
        ax.grid(True,axis='y',alpha=0.25); fig.tight_layout()
        p=fig_dir/'curvature_probe_summary.png'; fig.savefig(p,dpi=180); plt.close(fig); paths.append(str(p))
    # Jacobian spectral gap if present.
    if jac_summary:
        labels=[f"{r.get('pq')}\n{r.get('state_label')}" for r in jac_summary]
        vals=[fnum(r.get('sigma1_over_sigma2')) for r in jac_summary]
        fig,ax=plt.subplots(figsize=(max(8,0.75*len(labels)),4.5))
        ax.bar(range(len(labels)), vals)
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels,rotation=65,ha='right',fontsize=8)
        ax.set_ylabel('sigma1 / sigma2'); ax.set_title('Residual-Jacobian spectral gap')
        ax.grid(True,axis='y',alpha=0.25); fig.tight_layout()
        p=fig_dir/'jacobian_residual_spectral_gap.png'; fig.savefig(p,dpi=180); plt.close(fig); paths.append(str(p))
    return paths


def write_doc(doc: Path, out_dir: Path, args: argparse.Namespace, endpoint: list[dict[str,Any]], early: list[dict[str,Any]], arc: list[dict[str,Any]], slsum: list[dict[str,Any]], tangent: list[dict[str,str]], stepwise: list[dict[str,str]], jacsum: list[dict[str,str]], jacalign: list[dict[str,str]]) -> None:
    # Hand-picked compact rows.
    p2q2_early=[r for r in early if r['pq']=='p2q2' and r['method']=='steepest_replace']
    p2q2_early=sorted(p2q2_early,key=lambda r:r['step'])
    tang_rows=[r for r in tangent if r.get('pq') in ('p2q1','p2q2','p2qinf') and r.get('method') in ('steepest_replace','raw_replace','steepest_add','raw_add')]
    tang_rows=sorted(tang_rows,key=lambda r:(r.get('pq'),r.get('method')))
    step_rows=[r for r in stepwise if r.get('scope')=='post_boundary_099'] if stepwise else []
    step_rows=sorted(step_rows,key=lambda r:(r.get('method'),r.get('metric')))
    key_jac=[r for r in jacalign if r.get('direction_label') in ('steepest_replace_final','steepest_replace_step5','steepest_add_final','raw_add_final')]
    key_jac=sorted(key_jac,key=lambda r:(r.get('pq'),r.get('state_label'),r.get('direction_label')))
    lines=[
        '# Loss3 Current Mechanism Validation Summary - 2026-05-21','',
        'Status: generated from current core4/PQ landscape, tangent, stepwise, and targeted Jacobian/SVD outputs.', '',
        '## Source Outputs','',
        f"- Landscape output: `{rel(args.landscape_dir)}`",
        f"- Early trajectory landscape output: `{rel(args.early_landscape_dir)}`",
        f"- Tangent output: `{rel(args.tangent_dir)}`",
        f"- Stepwise output: `{rel(args.stepwise_dir)}`",
        f"- Jacobian/SVD output: `{rel(args.jacobian_dir)}`",
        f"- Summary output: `{rel(out_dir)}`", '',
        '## Question 1: Does GPI find the high-loss direction early?', '',
        'Observed from ray profiles. For p2q2 steepest_replace, endpoint loss along early directions divided by endpoint loss along the final direction:', '',
        md_table(p2q2_early, ['pq','method','step','endpoint_loss','final_endpoint_loss','endpoint_to_final_ratio']), '',
        'Interpretation: ratios near 1 by step 5-10 support the claim that replacement/GPI rapidly reaches a final-like high-loss direction.', '',
        '## Question 2: Are method final deltas on a shared high-loss ridge?', '',
        'Observed from p=2 boundary arcs. Values near 1 mean the arc between two final directions does not dip below the weaker endpoint much:', '',
        md_table(arc, ['pq','pair','min_arc_loss','min_over_weaker_endpoint','min_over_stronger_endpoint','arc_depth_from_weaker'], max_rows=80), '',
        'Interpretation: p2q2 arcs near 1 support a shared ridge; lower values or stronger dips are caveats, especially q=inf.', '',
        '## Question 3: Is boundary hit different from boundary convergence?', '',
        'Observed from all-p2 tangent residual rollup:', '',
        md_table(tang_rows, ['pq','method','hit_step_99_mean','post_boundary_gain_mean','tangent_residual_at_hit_mean','mean_angle_after_hit_deg_mean','final_loss3_q_mean'], max_rows=80), '',
        'Interpretation: high tangent residual at boundary hit means there is still sideways gradient on the boundary. Replacement can hit boundary immediately but still have large direction work left.', '',
        '## Question 4: Does tangent size predict immediate next-step gain?', '',
        'Observed from p2q2 stepwise post-boundary correlations:', '',
        md_table(step_rows, ['method','metric','pearson_r_with_next_loss_gain','pair_count'], max_rows=80), '',
        'Interpretation: tangent magnitude is opportunity, not a complete one-step predictor. The actual projected direction and local nonlinearity matter.', '',
        '## Question 5: What do 2D slices and curvature say?', '',
        '2D slice summary:', '',
        md_table(slsum, ['pq','center_method','center_loss','slice_loss_min','slice_loss_max','slice_range','center_minus_min'], max_rows=80), '',
        'Interpretation: these contours show whether the final/center point sits on a broad ridge, a narrow peak, or a geometry-dependent irregular surface.', '',
        '## Question 6: Is there direct dominant-mode evidence?', '',
        'Targeted residual-Jacobian/SVD summary:', '',
        md_table(jacsum, ['pq','state_label','sigma1','sigma2','sigma1_over_sigma2','top1_energy_fraction','top4_energy_fraction','top1_peakiness','top1_energy_concentration'], max_rows=80), '',
        'Top residual singular direction alignment with important deltas:', '',
        md_table(key_jac, ['pq','state_label','direction_label','abs_cos_top1_right_vs_direction','angle_deg_top1_right_vs_direction'], max_rows=160), '',
        'Interpretation: high spectral gap plus high cosine would support a literal dominant residual mode. Weak gap or weak cosine means the safer explanation remains empirical/local-surrogate rather than a strict spectral theorem.', '',
        '## Bottom Line','',
        'The validation separates the original mystery into testable pieces: boundary arrival, boundary rotation, shared ridge, local slice/curvature, and residual-Jacobian dominant-mode evidence. The supported claim should stay conditional: replacement/GPI is a very strong aggressive full-budget surrogate in p2q2-like geometry, but q=inf and p=1 geometries can break the clean story.',
    ]
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--landscape-dir', type=Path, default=DEFAULT_LANDSCAPE)
    ap.add_argument('--tangent-dir', type=Path, default=DEFAULT_TANGENT)
    ap.add_argument('--stepwise-dir', type=Path, default=DEFAULT_STEPWISE)
    ap.add_argument('--jacobian-dir', type=Path, default=DEFAULT_JAC)
    ap.add_argument('--early-landscape-dir', type=Path, default=DEFAULT_EARLY_LANDSCAPE)
    ap.add_argument('--out-dir', type=Path, default=DEFAULT_OUT)
    ap.add_argument('--doc', type=Path, default=DEFAULT_DOC)
    args=ap.parse_args()
    for attr in ('landscape_dir','early_landscape_dir','tangent_dir','stepwise_dir','jacobian_dir','out_dir','doc'):
        p=getattr(args,attr)
        if not p.is_absolute(): setattr(args,attr, PROJECT_ROOT/p)
    out_dir=args.out_dir; tables=out_dir/'tables'; tables.mkdir(parents=True,exist_ok=True)
    ray=read_csv(args.landscape_dir/'tables/ray_profile_aggregate.csv')
    early_ray=read_csv(args.early_landscape_dir/'tables/ray_profile_aggregate.csv')
    arc_rows=read_csv(args.landscape_dir/'tables/boundary_arc_aggregate.csv')
    slice_rows=read_csv(args.landscape_dir/'tables/slice_2d_aggregate.csv')
    curvature=read_csv(args.landscape_dir/'tables/curvature_aggregate.csv')
    tangent=read_csv(args.tangent_dir/'tables/p2_tangent_rollup_by_pq_method.csv')
    stepwise=read_csv(args.stepwise_dir/'tables/stepwise_geometry_correlations_by_method_scope.csv')
    jacsum=read_csv(args.jacobian_dir/'tables/state_svd_summary.csv')
    jacalign=read_csv(args.jacobian_dir/'tables/top_direction_alignment.csv')
    endpoint,early=summarize_ray(ray)
    if not early and early_ray:
        _early_endpoint, early = summarize_ray(early_ray)
    arc=summarize_arc(arc_rows)
    slsum=summarize_slice(slice_rows)
    write_csv(tables/'ray_endpoint_summary.csv',endpoint)
    write_csv(tables/'early_direction_endpoint_ratio.csv',early)
    write_csv(tables/'boundary_arc_dip_summary.csv',arc)
    write_csv(tables/'slice_2d_shape_summary.csv',slsum)
    figs=write_figures(out_dir,endpoint,early,arc,slice_rows,curvature,jacsum)
    manifest={'status':'completed','landscape_dir':str(args.landscape_dir),'early_landscape_dir':str(args.early_landscape_dir),'tangent_dir':str(args.tangent_dir),'stepwise_dir':str(args.stepwise_dir),'jacobian_dir':str(args.jacobian_dir),'out_dir':str(out_dir),'doc':str(args.doc),'figure_paths':figs,'row_counts':{'endpoint':len(endpoint),'early':len(early),'arc':len(arc),'slice_summary':len(slsum),'jac_summary':len(jacsum)}}
    (out_dir/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    write_doc(args.doc,out_dir,args,endpoint,early,arc,slsum,tangent,stepwise,jacsum,jacalign)
    print(json.dumps(manifest,indent=2,sort_keys=True))

if __name__=='__main__':
    main()
