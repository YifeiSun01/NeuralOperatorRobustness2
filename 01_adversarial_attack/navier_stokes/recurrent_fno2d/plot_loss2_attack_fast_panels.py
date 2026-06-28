#!/usr/bin/env python3
"""Fast CPU-only panels and curves from saved loss2 attack arrays."""
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
DEFAULT_LOSS2_ROOT=NS_ROOT/'perturbation_results'/'ns2d_recurrent_core4_attack'/'full_adw_b10_eps32_alpha1_20260522'/'mode_aaaaaaaaaw_p2_q2_20260522_030849_UTC'/'batch_0000_0009'/'loss2'
DEFAULT_OUT=NS_ROOT/'visualizations'/'loss2_attack_fast_panels_20260522'
METHODS=['raw_add','raw_replace','steepest_add','steepest_replace']

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

def read_csv(path):
    with open(path,newline='') as f:
        return list(csv.DictReader(f))

def f(row, key):
    try:
        return float(row.get(key,''))
    except Exception:
        return float('nan')

def plot_curves(loss2_root: Path, methods: list[str], out_dir: Path):
    fig, axes=plt.subplots(2,2,figsize=(14,9),constrained_layout=True)
    metrics=[]
    for method in methods:
        csv_path=loss2_root/method/'per_step_metrics.csv'
        if not csv_path.exists():
            continue
        rows=read_csv(csv_path)
        k=np.array([int(r['k']) for r in rows])
        loss2=np.array([f(r,'loss2_mean') for r in rows])
        true=np.array([f(r,'true_loss_mean') for r in rows])
        br=np.array([f(r,'boundary_ratio_mean') for r in rows])
        dp=np.array([f(r,'delta_p_mean') for r in rows])
        loss2_ratio=loss2/loss2[0] if loss2[0] != 0 else np.full_like(loss2,np.nan)
        true_ratio=true/true[0] if true[0] != 0 else np.full_like(true,np.nan)
        axes[0,0].plot(k, br, label=method)
        axes[0,1].plot(k, dp, label=method)
        axes[1,0].plot(k, loss2_ratio, label=method)
        axes[1,1].plot(k, true_ratio, label=method)
        metrics.append({
            'method': method,
            'k_final': int(k[-1]),
            'loss2_start': float(loss2[0]),
            'loss2_final': float(loss2[-1]),
            'loss2_final_ratio': float(loss2_ratio[-1]),
            'true_loss_start': float(true[0]),
            'true_loss_final': float(true[-1]),
            'true_loss_final_ratio': float(true_ratio[-1]),
            'delta_p_final': float(dp[-1]),
            'boundary_ratio_final': float(br[-1]),
            'max_boundary_ratio': float(np.nanmax(br)),
        })
    axes[0,0].set_title('Mean boundary ratio ||delta||_p / epsilon')
    axes[0,0].set_xlabel('step'); axes[0,0].set_ylabel('ratio')
    axes[0,0].axhline(1.0,color='k',lw=0.8,ls='--')
    axes[0,1].set_title('Mean delta_p')
    axes[0,1].set_xlabel('step'); axes[0,1].set_ylabel('||delta||_p')
    axes[1,0].set_title('Surrogate loss2 mean ratio')
    axes[1,0].set_xlabel('step'); axes[1,0].set_ylabel('loss2 / loss2[0]')
    axes[1,1].set_title('True all-W loss mean ratio')
    axes[1,1].set_xlabel('step'); axes[1,1].set_ylabel('true / true[0]')
    for ax in axes.ravel():
        ax.grid(True,alpha=0.25)
        ax.legend(fontsize=8)
    out=out_dir/'loss2_method_curves.png'
    fig.savefig(out,dpi=160)
    plt.close(fig)
    return metrics, out

def plot_panels(args, data):
    y=data['y'].float()
    x_all=(data['x'].float() if 'x' in data else y[...,0].float())
    all_rows=[]
    for method in args.methods:
        npz=args.loss2_root/method/'final_delta_and_metrics.npz'
        if not npz.exists():
            continue
        z=np.load(npz)
        ds_idx=z['dataset_indices'].astype(int)
        delta=z['final_delta'].astype(np.float32)
        x_adv=z['final_x_adv'].astype(np.float32)
        pos=[p for p in args.sample_positions if 0 <= p < len(ds_idx)]
        idx=ds_idx[pos]
        clean_x=x_all[idx].numpy()
        target=y[idx,...,args.target_frame_index].numpy()
        d=delta[pos]
        xa=x_adv[pos]
        d_l2=np.linalg.norm(d.reshape(d.shape[0],-1), axis=1)
        d_linf=np.max(np.abs(d.reshape(d.shape[0],-1)), axis=1)
        step_rows=read_csv(args.loss2_root/method/'per_step_metrics.csv')
        last=step_rows[-1]
        for i,p in enumerate(pos):
            state_lim=sym_limits(clean_x[i], xa[i], target[i])
            delta_lim=sym_limits(d[i], xa[i]-clean_x[i])
            fig,axes=plt.subplots(2,4,figsize=(16,8),constrained_layout=True)
            fig.suptitle(f'loss2/{method} sample_pos={p} dataset_index={int(idx[i])} delta_l2={d_l2[i]:.4g} delta_linf={d_linf[i]:.4g}', fontsize=11)
            panel(fig,axes[0,0],clean_x[i],'clean initial x',state_lim)
            panel(fig,axes[0,1],d[i],'final delta',delta_lim)
            panel(fig,axes[0,2],xa[i],'perturbed initial x+delta',state_lim)
            panel(fig,axes[0,3],xa[i]-clean_x[i],'x_adv - x_clean',delta_lim)
            panel(fig,axes[1,0],target[i],'fixed target / solver final y20',state_lim)
            panel(fig,axes[1,1],np.abs(d[i]),'abs(delta)',(0,float(max(np.max(np.abs(d[i])),1e-8))))
            axes[1,2].axis('off')
            text='\n'.join([
                'model/perturbed solver not rerun here',
                f"final loss2_mean={f(last,'loss2_mean'):.4g}",
                f"final true_loss_mean={f(last,'true_loss_mean'):.4g}",
                f"delta_p_mean={f(last,'delta_p_mean'):.4g}",
                f"boundary_ratio_mean={f(last,'boundary_ratio_mean'):.4g}",
            ])
            axes[1,2].text(0.02,0.98,text,ha='left',va='top',fontsize=9,transform=axes[1,2].transAxes)
            axes[1,3].axis('off')
            text2='\n'.join([
                'Interpretation:',
                'boundary near 1 -> epsilon saturated',
                'boundary well below 1 -> alpha/steps too small',
            ])
            axes[1,3].text(0.02,0.98,text2,ha='left',va='top',fontsize=9,transform=axes[1,3].transAxes)
            out=args.out_dir/f'loss2_{method}_samplepos{p}_idx{int(idx[i])}_fast_panels.png'
            fig.savefig(out,dpi=150)
            plt.close(fig)
            all_rows.append({'method':method,'sample_position':int(p),'dataset_index':int(idx[i]),'delta_l2':float(d_l2[i]),'delta_linf':float(d_linf[i]),'png':str(out)})
    return all_rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--loss2-root',type=Path,default=DEFAULT_LOSS2_ROOT)
    ap.add_argument('--test-path',type=Path,default=DEFAULT_TEST)
    ap.add_argument('--out-dir',type=Path,default=DEFAULT_OUT)
    ap.add_argument('--methods',nargs='+',default=METHODS)
    ap.add_argument('--sample-positions',nargs='+',type=int,default=[0,1,2])
    ap.add_argument('--target-frame-index',type=int,default=19)
    args=ap.parse_args()
    args.out_dir.mkdir(parents=True,exist_ok=True)
    data=torch.load(args.test_path,map_location='cpu',weights_only=False)
    curve_metrics, curve_png = plot_curves(args.loss2_root,args.methods,args.out_dir)
    panel_rows=plot_panels(args,data)
    report={'note':'Fast CPU-only visualization from saved loss2 arrays and CSV curves. Does not rerun model or solver.', 'loss2_root':str(args.loss2_root), 'curve_png':str(curve_png), 'curve_metrics':curve_metrics, 'panels':panel_rows}
    report_path=args.out_dir/'loss2_fast_panel_report.json'
    report_path.write_text(json.dumps(report,indent=2))
    print(f'[done] wrote {len(panel_rows)} panels and curves under {args.out_dir}')
    print(f'[done] report {report_path}')
if __name__=='__main__':
    main()
