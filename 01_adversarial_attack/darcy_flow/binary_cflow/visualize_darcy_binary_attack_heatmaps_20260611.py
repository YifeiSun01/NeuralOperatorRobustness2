
#!/usr/bin/env python3
"""Visualize binary Darcy attack samples as heat maps.

Each selected row shows:
  clean binary coefficient, attack delta, attacked coefficient,
  model output on attacked coefficient, solver output on attacked coefficient,
  and model - solver error.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tools.adversarial_training as adv  # noqa: E402


def parse_int_list(text: str) -> list[int]:
    out: list[int] = []
    for item in text.split(','):
        item = item.strip()
        if not item:
            continue
        out.append(int(item))
    if not out:
        raise ValueError(f"empty integer list: {text!r}")
    return out


def find_dataset_by_index(root: Path, index: int) -> Path:
    pattern = re.compile(rf"_0*{int(index)}_")
    matches = [p for p in sorted(root.glob('*.pt')) if pattern.search(p.stem)]
    if len(matches) != 1:
        names = [p.name for p in matches[:10]]
        raise FileNotFoundError(f"expected one dataset for index {index}, found {len(matches)}: {names}")
    return matches[0]


def rel_l2(pred: torch.Tensor, target: torch.Tensor) -> float:
    diff = pred - target
    return float(torch.linalg.vector_norm(diff.reshape(diff.shape[0], -1), dim=1).div(
        torch.linalg.vector_norm(target.reshape(target.shape[0], -1), dim=1).clamp_min(1e-20)
    ).mean().detach().cpu())


def mse_per_sample(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    diff = pred - target
    return diff.reshape(diff.shape[0], -1).square().mean(dim=1)


def load_selected_batch(dataset_paths: list[Path], sample_indices: list[int]) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    meta_rows: list[dict[str, Any]] = []
    for path, sample_idx in zip(dataset_paths, sample_indices):
        data = adv.torch_load(path)
        x, y = adv.tensor_xy(data, 'darcy')
        if sample_idx < 0 or sample_idx >= int(x.shape[0]):
            raise IndexError(f"sample index {sample_idx} out of range for {path.name} with {x.shape[0]} samples")
        meta = data.get('metadata', {}) if isinstance(data, dict) else {}
        variant = meta.get('variant', {}) if isinstance(meta, dict) else {}
        coeff = meta.get('coefficient', {}) if isinstance(meta, dict) else {}
        xs.append(x[sample_idx : sample_idx + 1])
        ys.append(y[sample_idx : sample_idx + 1])
        meta_rows.append({
            'dataset_path': str(path.relative_to(PROJECT_ROOT)),
            'dataset_id': meta.get('dataset_id', path.stem) if isinstance(meta, dict) else path.stem,
            'dataset_file': path.name,
            'dataset_index': variant.get('index', ''),
            'sample_index': int(sample_idx),
            'family': variant.get('family', ''),
            'target_high_fraction': coeff.get('target_high_fraction', variant.get('target_high_fraction', '')),
            'observed_high_fraction_mean': coeff.get('observed_high_fraction_mean', ''),
            'observed_edge_density_mean': coeff.get('observed_edge_density_mean', ''),
        })
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0), meta_rows


def build_cfg(args: argparse.Namespace) -> dict[str, Any]:
    defaults = adv.DEFAULTS['darcy']
    return {
        'label_mode': 'solver',
        'training_data_mode': 'adv-only',
        'attack_method': 'binary_steepest_replace',
        'attack_loss_objective': args.attack_loss_objective,
        'attack_steps': int(args.attack_steps),
        'epsilon_fraction': float(args.epsilon_fraction),
        'epsilon_abs': 0.0,
        'alpha_ratio': 1.0,
        'alpha_jitter_low': 1.0,
        'alpha_jitter_high': 1.0,
        'random_start_fraction': 0.0,
        'eps_jitter_low': float(args.eps_jitter_low),
        'eps_jitter_high': float(args.eps_jitter_high),
        'binary_pool_multiplier': float(args.binary_pool_multiplier),
        'binary_score_noise': float(args.binary_score_noise),
        'darcy_loss1_random_start': True,
        'darcy_loss1_random_start_fraction': 1.0,
        'darcy_physics_metric': 'rel_l2',
        'darcy_physics_bc_weight': 1.0,
        'darcy_physics_forcing_value': 1.0,
        'eval_batch_size': defaults.eval_batch_size,
    }


def load_model(device: torch.device, checkpoint: Path | None) -> torch.nn.Module:
    model = adv.load_darcy_model(device)
    if checkpoint is not None:
        model.load_state_dict(adv.checkpoint_state(checkpoint), strict=True)
    model.eval()
    return model


def field_np(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu().float()
    if x.ndim == 4 and x.shape[-1] == 1:
        x = x[..., 0]
    return x.numpy()


def plot_grid(
    out_path: Path,
    *,
    x_clean: np.ndarray,
    delta: np.ndarray,
    x_adv: np.ndarray,
    pred_adv: np.ndarray,
    solver_adv: np.ndarray,
    diff: np.ndarray,
    rows: list[dict[str, Any]],
    title: str,
) -> None:
    arrays = [x_clean, delta, x_adv, pred_adv, solver_adv, diff]
    col_titles = [
        'initial coefficient a',
        'delta = a_adv - a',
        'attacked coefficient a_adv',
        'model(a_adv)',
        'solver(a_adv)',
        'model - solver',
    ]
    n = x_clean.shape[0]
    fig, axes = plt.subplots(n, 6, figsize=(18, max(2.7 * n, 4.0)), constrained_layout=True)
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    out_min = float(np.nanmin([pred_adv.min(), solver_adv.min()]))
    out_max = float(np.nanmax([pred_adv.max(), solver_adv.max()]))
    diff_abs = float(np.nanmax(np.abs(diff))) or 1e-12
    specs = [
        dict(cmap='viridis', vmin=3.0, vmax=12.0),
        dict(cmap='coolwarm', vmin=-9.0, vmax=9.0),
        dict(cmap='viridis', vmin=3.0, vmax=12.0),
        dict(cmap='magma', vmin=out_min, vmax=out_max),
        dict(cmap='magma', vmin=out_min, vmax=out_max),
        dict(cmap='coolwarm', vmin=-diff_abs, vmax=diff_abs),
    ]
    images = []
    for r in range(n):
        row_label = (
            f"idx {rows[r]['dataset_index']} sample {rows[r]['sample_index']}\n"
            f"{rows[r]['family']} high={float(rows[r]['high_fraction']):.3f}\n"
            f"flips={rows[r]['flip_pixels']} relL2={float(rows[r]['adv_rel_l2']):.4f}"
        )
        for c in range(6):
            ax = axes[r, c]
            im = ax.imshow(arrays[c][r], interpolation='nearest', **specs[c])
            if r == 0:
                ax.set_title(col_titles[c], fontsize=10)
            if c == 0:
                ax.set_ylabel(row_label, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == n - 1:
                images.append((im, c))
    for im, c in images:
        fig.colorbar(im, ax=axes[:, c], fraction=0.025, pad=0.01)
    fig.suptitle(title, fontsize=13)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', type=Path, default=PROJECT_ROOT / 'generalization_datasets_darcy_binary_loss3targeted_20260611/darcy')
    parser.add_argument('--dataset-indices', default='22,30,38,10')
    parser.add_argument('--sample-indices', default='0', help='Comma list. If one value is given, it is reused for all datasets.')
    parser.add_argument('--checkpoint', type=Path, default=None, help='Optional checkpoint to load after the retained baseline model.')
    parser.add_argument('--checkpoint-label', default='baseline_m64w60')
    parser.add_argument('--attack-loss-objective', choices=['loss1', 'loss2', 'loss3', 'physics'], default='loss3')
    parser.add_argument('--attack-steps', type=int, default=adv.DEFAULTS['darcy'].attack_steps)
    parser.add_argument('--epsilon-fraction', type=float, default=adv.DEFAULTS['darcy'].epsilon_fraction)
    parser.add_argument('--eps-jitter-low', type=float, default=adv.DEFAULTS['darcy'].eps_jitter_low)
    parser.add_argument('--eps-jitter-high', type=float, default=adv.DEFAULTS['darcy'].eps_jitter_high)
    parser.add_argument('--binary-pool-multiplier', type=float, default=1.0)
    parser.add_argument('--binary-score-noise', type=float, default=0.0)
    parser.add_argument('--output-dir', type=Path, default=PROJECT_ROOT / 'visualizations/darcy_binary_attack_heatmaps_20260611')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the Darcy attack/solver visualization path')
    device = torch.device('cuda')
    dataset_indices = parse_int_list(args.dataset_indices)
    sample_indices = parse_int_list(args.sample_indices)
    if len(sample_indices) == 1 and len(dataset_indices) > 1:
        sample_indices = sample_indices * len(dataset_indices)
    if len(sample_indices) != len(dataset_indices):
        raise ValueError('--sample-indices must have length 1 or match --dataset-indices')

    dataset_root = args.dataset_root.resolve()
    dataset_paths = [find_dataset_by_index(dataset_root, idx) for idx in dataset_indices]
    x_cpu, y_cpu, meta_rows = load_selected_batch(dataset_paths, sample_indices)
    x = x_cpu.to(device, non_blocking=True)
    y = y_cpu.to(device, non_blocking=True)

    checkpoint = args.checkpoint.resolve() if args.checkpoint is not None else None
    model = load_model(device, checkpoint)
    cfg = build_cfg(args)

    attack = adv.binary_darcy_replace_attack(
        model,
        x,
        y,
        steps=int(args.attack_steps),
        epsilon_fraction=float(args.epsilon_fraction),
        jitter_low=float(args.eps_jitter_low),
        jitter_high=float(args.eps_jitter_high),
        random_pool_multiplier=float(args.binary_pool_multiplier),
        random_score_noise=float(args.binary_score_noise),
        cfg=cfg,
    )
    x_adv = attack.x_train.detach()
    y_adv = attack.y_train.detach()
    with torch.no_grad():
        pred_adv = model(x_adv).detach()
        pred_clean = model(x).detach()
        y_clean_solver = adv.solver_target_for_model_input('darcy', x, y, cfg, allow_target_grad=False).detach()
        _, clean_obj_losses, _, attack_info = adv.darcy_attack_objective_loss(
            model, x, x, y, cfg, allow_solver_backward=False
        )
        _, adv_obj_losses, _, _ = adv.darcy_attack_objective_loss(
            model, x, x_adv, y, cfg, allow_solver_backward=False
        )
    delta = x_adv - x
    diff = pred_adv - y_adv
    clean_solver_mse = mse_per_sample(pred_clean, y_clean_solver)
    adv_solver_mse = mse_per_sample(pred_adv, y_adv)
    adv_rel_l2_each = torch.linalg.vector_norm(diff.reshape(diff.shape[0], -1), dim=1) / torch.linalg.vector_norm(
        y_adv.reshape(y_adv.shape[0], -1), dim=1
    ).clamp_min(1e-20)

    changed = delta.reshape(delta.shape[0], -1).abs() > 1e-12
    high_frac_clean = (x[..., 0] == 12.0).float().reshape(x.shape[0], -1).mean(dim=1)
    high_frac_adv = (x_adv[..., 0] == 12.0).float().reshape(x_adv.shape[0], -1).mean(dim=1)

    summary_rows: list[dict[str, Any]] = []
    for i, row in enumerate(meta_rows):
        record = dict(row)
        record.update({
            'checkpoint_label': args.checkpoint_label,
            'checkpoint': str(checkpoint.relative_to(PROJECT_ROOT)) if checkpoint is not None else 'retained_baseline',
            'attack_loss_objective': args.attack_loss_objective,
            'attack_objective_definition': attack_info.get('attack_objective_definition', ''),
            'attack_steps': int(args.attack_steps),
            'epsilon_fraction': float(args.epsilon_fraction),
            'clean_high_fraction': float(high_frac_clean[i].detach().cpu()),
            'adv_high_fraction': float(high_frac_adv[i].detach().cpu()),
            'flip_pixels': int(changed[i].sum().detach().cpu()),
            'flip_fraction': float(changed[i].float().mean().detach().cpu()),
            'delta_unique_values': ','.join(str(float(v)) for v in torch.unique(delta[i].detach().cpu()).tolist()),
            'clean_objective_loss': float(clean_obj_losses[i].detach().cpu()),
            'adv_objective_loss': float(adv_obj_losses[i].detach().cpu()),
            'objective_loss_gain': float((adv_obj_losses[i] - clean_obj_losses[i]).detach().cpu()),
            'clean_solver_mse': float(clean_solver_mse[i].detach().cpu()),
            'adv_solver_mse': float(adv_solver_mse[i].detach().cpu()),
            'adv_rel_l2': float(adv_rel_l2_each[i].detach().cpu()),
            'model_output_min': float(pred_adv[i].min().detach().cpu()),
            'model_output_max': float(pred_adv[i].max().detach().cpu()),
            'solver_output_min': float(y_adv[i].min().detach().cpu()),
            'solver_output_max': float(y_adv[i].max().detach().cpu()),
            'diff_absmax': float(diff[i].abs().max().detach().cpu()),
        })
        # convenient numeric field for row labels
        try:
            record['high_fraction'] = float(record['observed_high_fraction_mean'])
        except Exception:
            record['high_fraction'] = float(high_frac_clean[i].detach().cpu())
        summary_rows.append(record)

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"darcy_{args.checkpoint_label}_{args.attack_loss_objective}_indices_{'-'.join(map(str, dataset_indices))}_samples_{'-'.join(map(str, sample_indices))}"
    heatmap_path = out_dir / f"{stem}_heatmaps.png"
    csv_path = out_dir / f"{stem}_summary.csv"
    json_path = out_dir / f"{stem}_summary.json"
    npz_path = out_dir / f"{stem}_arrays.npz"

    plot_grid(
        heatmap_path,
        x_clean=field_np(x),
        delta=field_np(delta),
        x_adv=field_np(x_adv),
        pred_adv=field_np(pred_adv),
        solver_adv=field_np(y_adv),
        diff=field_np(diff),
        rows=summary_rows,
        title=(
            f"Darcy binary {args.attack_loss_objective} attack heatmaps; "
            f"checkpoint={args.checkpoint_label}; epsilon={float(args.epsilon_fraction):g}"
        ),
    )

    keys: list[str] = []
    for row in summary_rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(summary_rows)
    payload = {
        'heatmap': str(heatmap_path.relative_to(PROJECT_ROOT)),
        'summary_csv': str(csv_path.relative_to(PROJECT_ROOT)),
        'arrays_npz': str(npz_path.relative_to(PROJECT_ROOT)),
        'checkpoint_label': args.checkpoint_label,
        'checkpoint': str(checkpoint.relative_to(PROJECT_ROOT)) if checkpoint is not None else 'retained_baseline',
        'attack_info': attack.info,
        'samples': summary_rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    np.savez_compressed(
        npz_path,
        x_clean=field_np(x),
        delta=field_np(delta),
        x_adv=field_np(x_adv),
        pred_adv=field_np(pred_adv),
        solver_adv=field_np(y_adv),
        model_minus_solver=field_np(diff),
        dataset_id=np.asarray([r['dataset_id'] for r in summary_rows], dtype=object),
        dataset_index=np.asarray([r['dataset_index'] for r in summary_rows], dtype=object),
        sample_index=np.asarray([r['sample_index'] for r in summary_rows], dtype=np.int64),
    )
    print('[heatmap]', heatmap_path.relative_to(PROJECT_ROOT))
    print('[summary_csv]', csv_path.relative_to(PROJECT_ROOT))
    print('[summary_json]', json_path.relative_to(PROJECT_ROOT))
    print('[arrays_npz]', npz_path.relative_to(PROJECT_ROOT))
    for row in summary_rows:
        print(
            f"[sample] idx={row['dataset_index']} sample={row['sample_index']} "
            f"family={row['family']} flips={row['flip_pixels']} "
            f"clean_obj={row['clean_objective_loss']:.6e} adv_obj={row['adv_objective_loss']:.6e} "
            f"adv_rel_l2={row['adv_rel_l2']:.6f}"
        )


if __name__ == '__main__':
    main()
