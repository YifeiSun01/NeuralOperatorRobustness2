#!/usr/bin/env python3
"""Fast CPU-only panels from saved loss1 attack arrays, no model/solver rerun."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT=Path(__file__).resolve().parents[2]
NS_ROOT=PROJECT_ROOT/'2D_NS_FNO2d_recurrent'
DEFAULT_TEST=NS_ROOT/'datasets'/'exponax_datasets'/'t20'/'real_initial_laxmap_single'/'test'/'dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_test_ntimepoints21_all_frames.pt'
DEFAULT_LOSS1_ROOT=NS_ROOT/'perturbation_results'/'ns2d_recurrent_core4_attack'/'full_adw_b10_eps32_alpha1_20260522'/'mode_wwwwwwwwww_p2_q2_20260522_015452_UTC'/'batch_0000_0009'/'loss1'
DEFAULT_OUT=NS_ROOT/'visualizations'/'loss1_attack_fast_panels_20260522'

def sym_limits(*arrays, pct=99.5):
    vals=np.concatenate([np.asarray(a,dtype=np.float64).ravel() for a in arrays])
    vals=vals[np.isfinite(vals)]
    if vals.size == 0:
        return -1,1
    vmax=float(np.percentile(np.abs(vals), pct))
    if vmax <= 0:
        vmax=float(np.max(np.abs(vals))) if vals.size else 1.0
    if vmax <= 0:
        vmax=1.0
    return -vmax, vmax

def panel(fig, ax, arr, title, vlim=None):
    if vlim is None:
        vlim=sym_limits(arr)
    im=ax.imshow(arr, cmap='coolwarm', origin='lower', vmin=vlim[0], vmax=vlim[1])
    ax.set_title(title, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

def load_last_metrics(path):
    with open(path, newline='') as f:
        rows=list(csv.DictReader(f))
    return rows[-1] if rows else {}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--loss1-root', type=Path, default=DEFAULT_LOSS1_ROOT)
    ap.add_argument('--test-path', type=Path, default=DEFAULT_TEST)
    ap.add_argument('--out-dir', type=Path, default=DEFAULT_OUT)
    ap.add_argument('--methods', nargs='+', default=['raw_add','raw_replace','steepest_add','steepest_replace'])
    ap.add_argument('--sample-positions', nargs='+', type=int, default=[0,1,2])
    ap.add_argument('--target-frame-index', type=int, default=19)
    args=ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data=torch.load(args.test_path, map_location='cpu', weights_only=False)
    y=data['y'].float()
    x_all=(data['x'].float() if 'x' in data else y[...,0].float())
    rows=[]
    for method in args.methods:
        npz=args.loss1_root/method/'final_delta_and_metrics.npz'
        if not npz.exists():
            continue
        z=np.load(npz)
        ds_idx=z['dataset_indices'].astype(int)
        delta=z['final_delta'].astype(np.float32)
        x_adv=z['final_x_adv'].astype(np.float32)
        pos=[p for p in args.sample_positions if 0 <= p < len(ds_idx)]
        idx=ds_idx[pos]
        clean_x=x_all[idx].numpy()
        solver_final=y[idx,...,args.target_frame_index].numpy()
        d=delta[pos]
        xa=x_adv[pos]
        d_l2=np.linalg.norm(d.reshape(d.shape[0],-1), axis=1)
        d_linf=np.max(np.abs(d.reshape(d.shape[0],-1)), axis=1)
        last=load_last_metrics(args.loss1_root/method/'per_step_metrics.csv')
        for i,p in enumerate(pos):
            state_lim=sym_limits(clean_x[i], xa[i], solver_final[i])
            delta_lim=sym_limits(d[i], xa[i]-clean_x[i])
            fig,axes=plt.subplots(2,4,figsize=(16,8),constrained_layout=True)
            fig.suptitle(f'loss1/{method} sample_pos={p} dataset_index={int(idx[i])} delta_l2={d_l2[i]:.6g} delta_linf={d_linf[i]:.6g}', fontsize=11)
            panel(fig,axes[0,0],clean_x[i],'clean initial x',state_lim)
            panel(fig,axes[0,1],d[i],'final delta',delta_lim)
            panel(fig,axes[0,2],xa[i],'perturbed initial x+delta',state_lim)
            panel(fig,axes[0,3],xa[i]-clean_x[i],'x_adv - x_clean',delta_lim)
            panel(fig,axes[1,0],solver_final[i],'solver final clean',state_lim)
            panel(fig,axes[1,1],solver_final[i],'solver final perturbed\n(equal: delta is zero)',state_lim)
            panel(fig,axes[1,2],np.zeros_like(solver_final[i]),'solver final diff\n(not rerun; zero delta)',(-1,1))
            axes[1,3].axis('off')
            text='\n'.join([
                'model-vs-solver final not computed here',
                'reason: avoid GPU/CPU interference',
                f"final loss1_mean={last.get('loss1_mean','NA')}",
                f"final true_loss_mean={last.get('true_loss_mean','NA')}",
                f"boundary_ratio_mean={last.get('boundary_ratio_mean','NA')}",
            ])
            axes[1,3].text(0.02,0.98,text,ha='left',va='top',fontsize=9,transform=axes[1,3].transAxes)
            out=args.out_dir/f'loss1_{method}_samplepos{p}_idx{int(idx[i])}_fast_panels.png'
            fig.savefig(out,dpi=150)
            plt.close(fig)
            rows.append({'method':method,'sample_position':int(p),'dataset_index':int(idx[i]),'delta_l2':float(d_l2[i]),'delta_linf':float(d_linf[i]),'png':str(out)})
    report={'note':'Fast CPU-only visualization from saved arrays. It does not rerun model or solver. Because loss1 final_delta is zero, perturbed initial/final are identical to clean for these saved outputs.', 'rows': rows}
    (args.out_dir/'loss1_fast_panel_report.json').write_text(json.dumps(report,indent=2))
    print(f'[done] wrote {len(rows)} PNGs under {args.out_dir}')

if __name__=='__main__':
    main()
