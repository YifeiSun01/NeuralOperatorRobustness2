#!/usr/bin/env python3
"""Run the intended Burgers p=2,q=2 adversarial-training pipeline.

This is the orchestrator for the corrected run whose main difference from the
previous zero/adversarial run is the attack geometry:

- old accidental ablation: fast_replace_linf -> sign/rectangular delta
- intended main run: fast_replace_l2 -> p=2,q=2 RMS-L2 delta

The pipeline can train, validate delta geometry, make the selected polished
figures, run top-100 checkpoint-series Jacobian/SVD diagnostics, and optionally
sync selected artifacts to R2/Git.  Credentials are intentionally not stored in
this script; pass them through environment variables or existing git/rclone
configuration.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
DEFAULT_RUN_NAME = 'burgers_p2q2_advonly_random_jitter_1000ep_bs480_steps5_eps5bucket_20260601'
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / 'adversarial_training_runs'
DEFAULT_VIZ_ROOT = PROJECT_ROOT / 'visualizations' / 'burgers_p2q2_adv_training_20260601'
DEFAULT_FORENSICS_ROOT = PROJECT_ROOT / 'forensics' / 'burgers_p2q2_checkpoint_series_jacobian_svd_20260601'
DEFAULT_ATTACK_GIF_ROOT = PROJECT_ROOT / 'forensics' / 'burgers_p2q2_baseline_vs_epoch1000_attack_visualization_20260602'
DEFAULT_SELECTED_TOP = Path('/workspace/polished_selected_download_burgers_p2q2_20260601')
DEFAULT_SELECTED_ZIP = Path('/workspace/polished_selected_download_burgers_p2q2_20260601.zip')
DEFAULT_R2_PREFIX = 'machine-sync/NeuralOperatorRobustness2-selected/20260601_burgers_p2q2_adv_training_full_pipeline'
IMAGE_SUFFIXES = {'.png', '.gif'}

SELECTED_FIGURES = [
    'corrected_attack_loss_three_lines_plus_buckets.png',
    'corrected_relative_l2_full_heatmap_raw_group_line.png',
    'corrected_rmse_full_heatmap_raw_group_line.png',
    'polished_checkpoint_style_delta_fft_raw_heatmap_25epoch_smoothed_lines.png',
    'relative_l2_grouped_shared_y_distinct_datasets_max5_high_transparency_raw.png',
    'relative_l2_grouped_shared_y_distinct_datasets_max5_high_transparency_ma25.png',
    'rmse_grouped_shared_y_distinct_datasets_max5_high_transparency_raw.png',
    'rmse_grouped_shared_y_distinct_datasets_max5_high_transparency_ma25.png',
]


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(x) for x in value]
    return value


def run_cmd(cmd: list[str | Path], *, cwd: Path = PROJECT_ROOT, dry_run: bool = False, log_path: Path | None = None) -> None:
    printable = ' '.join(str(x) for x in cmd)
    if dry_run:
        print(f'[dry-run] {printable}', flush=True)
        return
    print(f'[run] {printable}', flush=True)
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open('a', encoding='utf-8') as f:
            f.write(f'\n$ {printable}\n')
            f.flush()
            subprocess.run([str(x) for x in cmd], cwd=str(cwd), check=True, stdout=f, stderr=subprocess.STDOUT)
    else:
        subprocess.run([str(x) for x in cmd], cwd=str(cwd), check=True)


def burgers_run_dir(args: argparse.Namespace) -> Path:
    return (args.output_root / args.run_name).resolve()


def training_command(args: argparse.Namespace, *, smoke: bool = False) -> list[str | Path]:
    cmd: list[str | Path] = [
        PYTHON,
        PROJECT_ROOT / 'tools' / 'adversarial_training.py',
        '--tasks', 'burgers',
        '--output-root', args.output_root,
        '--run-name', args.run_name,
        '--seed', str(args.seed),
        '--epochs', '1000' if not smoke else str(args.smoke_epochs),
        '--checkpoint-every-epochs', '200',
        '--training-data-mode', 'adv-only',
        '--label-mode', 'solver',
        '--epsilon-bucket-count', '5',
        '--attack-probe-samples', str(args.attack_probe_samples),
        '--attack-probe-every-n-epochs', '1',
        '--burgers-attack-method', 'fast_replace_l2',
        '--burgers-require-p2q2',
        '--burgers-attack-steps', str(args.attack_steps),
        '--burgers-batch-size', str(args.batch_size),
        '--burgers-optimizer-batch-size', str(args.optimizer_batch_size),
        '--burgers-epsilon-fraction', str(args.epsilon_fraction),
        '--burgers-eps-jitter-low', str(args.eps_jitter_low),
        '--burgers-eps-jitter-high', str(args.eps_jitter_high),
        '--burgers-alpha-ratio', str(args.alpha_ratio),
        '--burgers-alpha-jitter-low', str(args.alpha_jitter_low),
        '--burgers-alpha-jitter-high', str(args.alpha_jitter_high),
    ]
    if smoke:
        cmd.extend(
            [
                '--burgers-train-max', str(max(args.batch_size, args.optimizer_batch_size)),
                '--max-batches-per-epoch', str(args.smoke_batches_per_epoch),
                '--eval-max-samples', str(args.smoke_eval_max_samples),
                '--max-generalization-eval', str(args.smoke_max_generalization_eval),
            ]
        )
    else:
        cmd.extend(['--eval-max-samples', '0'])
    return cmd


def validate_delta(args: argparse.Namespace, run_dir: Path, *, dry_run: bool) -> None:
    cmd = [
        PYTHON,
        PROJECT_ROOT / 'tools' / 'validate_burgers_l2_delta_geometry.py',
        '--run-dir', run_dir,
        '--out-json', run_dir / 'burgers' / 'l2_delta_geometry_check.json',
        '--out-csv', run_dir / 'burgers' / 'l2_delta_geometry_check_samples.csv',
    ]
    run_cmd(cmd, dry_run=dry_run, log_path=run_dir / 'pipeline_logs' / 'validate_delta.log')


def import_plot_module(module_name: str):
    if str(PROJECT_ROOT / 'tools') not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT / 'tools'))
    return importlib.import_module(module_name)


def collect_selected_figures(viz_root: Path, selected_repo: Path, selected_top: Path, zip_repo: Path, zip_top: Path) -> dict[str, Any]:
    selected_repo.mkdir(parents=True, exist_ok=True)
    selected_top.mkdir(parents=True, exist_ok=True)
    for folder in [selected_repo, selected_top]:
        for p in folder.iterdir():
            if p.is_file():
                p.unlink()
    source_dirs = [viz_root / 'polished_report', viz_root]
    copied: list[str] = []
    missing: list[str] = []
    for name in SELECTED_FIGURES:
        src = next((d / name for d in source_dirs if (d / name).exists()), None)
        if src is None:
            missing.append(name)
            continue
        shutil.copy2(src, selected_repo / name)
        shutil.copy2(src, selected_top / name)
        copied.append(name)
    manifest = {
        'selected_repo': selected_repo,
        'selected_top': selected_top,
        'zip_repo': zip_repo,
        'zip_top': zip_top,
        'figures': copied,
        'missing': missing,
    }
    (selected_repo / 'selected_figures_manifest.json').write_text(json.dumps(to_jsonable(manifest), indent=2), encoding='utf-8')
    package = refresh_selected_package(selected_repo, selected_top, zip_repo, zip_top)
    manifest['package'] = package
    return manifest


def refresh_selected_package(selected_repo: Path, selected_top: Path, zip_repo: Path, zip_top: Path) -> dict[str, Any]:
    """Mirror selected image artifacts to /workspace and refresh image-only zips."""
    selected_repo.mkdir(parents=True, exist_ok=True)
    selected_top.mkdir(parents=True, exist_ok=True)
    for p in selected_top.iterdir():
        if p.is_file():
            p.unlink()
    images = sorted(p for p in selected_repo.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    for p in images:
        shutil.copy2(p, selected_top / p.name)
    for zip_path, folder in [(zip_repo, selected_repo), (zip_top, selected_top)]:
        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(folder.iterdir()):
                if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
                    zf.write(p, arcname=p.name)
    return {
        'image_count': len(images),
        'selected_repo': selected_repo,
        'selected_top': selected_top,
        'zip_repo': zip_repo,
        'zip_top': zip_top,
    }


def run_plots(args: argparse.Namespace, run_dir: Path, *, dry_run: bool) -> dict[str, Any]:
    if dry_run:
        print(f'[dry-run] would generate selected polished plots for {run_dir}', flush=True)
        return {}
    viz_root = args.viz_root.resolve()
    report_dir = viz_root / 'polished_report'
    selected_repo = viz_root / 'polished_selected_download_burgers_p2q2_20260601'
    zip_repo = viz_root / 'polished_selected_download_burgers_p2q2_20260601.zip'
    viz_root.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    hybrid = import_plot_module('plot_burgers_checkpoint_style_hybrid_visualizations')
    hybrid.RUN_DIR = run_dir
    hybrid.BURGERS_DIR = run_dir / 'burgers'
    hybrid.OUT_DIR = report_dir
    hybrid.main()

    corrected = import_plot_module('plot_burgers_corrected_selected_visualizations')
    corrected.RUN_DIR = run_dir
    corrected.BURGERS_DIR = run_dir / 'burgers'
    corrected.RAW_VIZ_DIR = viz_root
    corrected.OUT_DIR = report_dir
    corrected.REPO_SELECTED = selected_repo
    corrected.TOP_SELECTED = args.selected_top.resolve()
    corrected.REPO_ZIP = zip_repo
    corrected.TOP_ZIP = args.selected_zip.resolve()
    corrected.main()

    max5 = import_plot_module('plot_burgers_max5_high_transparency')
    max5.RUN_DIR = run_dir
    max5.BURGERS_DIR = run_dir / 'burgers'
    max5.OUT_DIR = viz_root
    max5.main()

    manifest = collect_selected_figures(viz_root, selected_repo, args.selected_top.resolve(), zip_repo, args.selected_zip.resolve())
    (report_dir / 'p2q2_selected_download_manifest.json').write_text(json.dumps(to_jsonable(manifest), indent=2), encoding='utf-8')
    print(json.dumps(to_jsonable(manifest), indent=2), flush=True)
    return manifest


def run_jacobian_svd(args: argparse.Namespace, run_dir: Path, *, dry_run: bool) -> None:
    cmd = [
        PYTHON,
        PROJECT_ROOT / 'tools' / 'compare_burgers_checkpoint_series_jacobian_svd.py',
        '--run-dir', run_dir,
        '--out-root', args.forensics_root,
        '--sample-manifest', args.sample_manifest,
        '--reuse-root', args.reuse_root,
        '--top-k', str(args.svd_top_k),
        '--svd-method', 'topk',
        '--svd-solver', args.svd_solver,
        '--checkpoint-epochs', '200', '400', '600', '800', '1000',
    ]
    run_cmd(cmd, dry_run=dry_run, log_path=run_dir / 'pipeline_logs' / 'jacobian_svd.log')


def run_svd_visualizations(args: argparse.Namespace, run_dir: Path, *, dry_run: bool) -> None:
    if dry_run:
        print(f'[dry-run] would render polished SVD figures from {args.forensics_root}', flush=True)
        return
    selected_repo = args.viz_root.resolve() / 'polished_selected_download_burgers_p2q2_20260601'
    selected_repo.mkdir(parents=True, exist_ok=True)
    mod = import_plot_module('plot_burgers_p2q2_svd_polished_visualizations')
    mod.FORENSICS = args.forensics_root.resolve()
    mod.OUT_DIR = selected_repo
    mod.main()
    refresh_selected_package(
        selected_repo,
        args.selected_top.resolve(),
        args.viz_root.resolve() / 'polished_selected_download_burgers_p2q2_20260601.zip',
        args.selected_zip.resolve(),
    )


def checkpoint_for_epoch(run_dir: Path, epoch: int) -> Path:
    csv_path = run_dir / 'burgers' / 'checkpoints.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f'checkpoint csv not found: {csv_path}')
    with csv_path.open('r', newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f'empty checkpoint csv: {csv_path}')
    path_col = 'path' if 'path' in rows[0] else 'checkpoint_path'
    for row in rows:
        if int(float(row.get('epoch', -1))) == int(epoch) and row.get(path_col):
            p = Path(row[path_col])
            if not p.is_absolute():
                p = PROJECT_ROOT / p
            return p.resolve()
    raise FileNotFoundError(f'epoch {epoch} checkpoint not found in {csv_path}')


def run_attack_gif_visualization(args: argparse.Namespace, run_dir: Path, *, dry_run: bool) -> None:
    if dry_run:
        print(f'[dry-run] would render baseline-vs-checkpoint attack GIF/PNG for epoch {args.attack_gif_checkpoint_epoch}', flush=True)
        return
    selected_repo = args.viz_root.resolve() / 'polished_selected_download_burgers_p2q2_20260601'
    selected_repo.mkdir(parents=True, exist_ok=True)
    mod = import_plot_module('plot_burgers_p2q2_baseline_vs_epoch1000_attack_gif')
    mod.SELECTED_DIR = selected_repo
    mod.OUT_DIR = args.attack_gif_root.resolve()
    mod.EPOCH1000_CKPT = checkpoint_for_epoch(run_dir, args.attack_gif_checkpoint_epoch)
    mod.ATTACK_STEPS = int(args.attack_gif_steps)
    mod.FRAME_EVERY = int(args.attack_gif_frame_every)
    mod.EPSILON_RMS = float(args.attack_gif_epsilon_rms)
    mod.ALPHA_RMS = float(args.attack_gif_alpha_rms) if args.attack_gif_alpha_rms is not None else float(args.attack_gif_epsilon_rms) * float(args.attack_gif_alpha_ratio)
    mod._SOLVER_CACHE = {}
    mod.main()
    refresh_selected_package(
        selected_repo,
        args.selected_top.resolve(),
        args.viz_root.resolve() / 'polished_selected_download_burgers_p2q2_20260601.zip',
        args.selected_zip.resolve(),
    )


def sync_to_r2(args: argparse.Namespace, run_dir: Path, *, dry_run: bool) -> None:
    targets = [
        (run_dir, 'run'),
        (args.viz_root.resolve(), 'visualizations'),
        (args.forensics_root.resolve(), 'forensics'),
        (args.attack_gif_root.resolve(), 'attack_gif_forensics'),
    ]
    upload_script = PROJECT_ROOT / 'tools' / 'upload_path_to_r2_20260525.sh'
    for path, name in targets:
        if not path.exists():
            continue
        cmd = ['env', f'R2_PREFIX={args.r2_prefix}/{name}', str(upload_script), str(path)]
        run_cmd(cmd, dry_run=dry_run, log_path=run_dir / 'pipeline_logs' / 'r2_upload.log')


def git_push_outputs(args: argparse.Namespace, run_dir: Path, *, dry_run: bool) -> None:
    """Push lightweight reproducibility artifacts to GitHub.

    Heavy checkpoint and SVD NPZ payloads are intentionally left for R2.  GitHub
    gets code, docs, selected figures, and CSV/Markdown summaries so the run is
    reproducible without trying to commit multi-GB binary diagnostics.
    """
    viz_root = args.viz_root.resolve()
    selected_repo = viz_root / 'polished_selected_download_burgers_p2q2_20260601'
    paths: list[Path] = [
        PROJECT_ROOT / 'tools' / 'adversarial_training.py',
        PROJECT_ROOT / 'tools' / 'validate_burgers_l2_delta_geometry.py',
        PROJECT_ROOT / 'tools' / 'compare_burgers_checkpoint_series_jacobian_svd.py',
        PROJECT_ROOT / 'tools' / 'plot_burgers_p2q2_svd_polished_visualizations.py',
        PROJECT_ROOT / 'tools' / 'plot_burgers_p2q2_baseline_vs_epoch1000_attack_gif.py',
        PROJECT_ROOT / 'tools' / 'run_burgers_p2q2_full_pipeline.py',
        PROJECT_ROOT / 'tools' / 'run_burgers_p2q2_full_pipeline.sh',
        PROJECT_ROOT / 'tools' / 'plot_burgers_checkpoint_style_hybrid_visualizations.py',
        PROJECT_ROOT / 'tools' / 'plot_burgers_corrected_selected_visualizations.py',
        PROJECT_ROOT / 'docs' / 'burgers_p2q2_full_pipeline_20260601.md',
        run_dir / 'pipeline_logs' / 'pipeline_config.json',
        run_dir / 'pipeline_logs' / 'pipeline_done.json',
        run_dir / 'burgers' / 'l2_delta_geometry_check.json',
        run_dir / 'burgers' / 'l2_delta_geometry_check_samples.csv',
        viz_root / 'polished_selected_download_burgers_p2q2_20260601.zip',
        selected_repo,
        args.forensics_root.resolve() / 'checkpoint_series_summary.md',
        args.forensics_root.resolve() / 'checkpoint_series_jacobian_svd_summary.csv',
        args.forensics_root.resolve() / 'checkpoint_series_error_aggregate.csv',
        args.forensics_root.resolve() / 'checkpoint_series_solver_similarity_rankwise.csv',
        args.forensics_root.resolve() / 'checkpoint_series_solver_similarity_subspaces.csv',
        args.forensics_root.resolve() / 'config.json',
        args.attack_gif_root.resolve() / 'summary.json',
        args.attack_gif_root.resolve() / 'attack_loss_curves.csv',
        args.attack_gif_root.resolve() / 'sample_manifest.json',
    ]
    existing = [p for p in paths if p.exists()]
    if not existing:
        return
    run_cmd(['git', 'add', *existing], dry_run=dry_run)
    msg = args.git_commit_message or 'Add Burgers p2q2 adversarial training pipeline outputs'
    run_cmd(['git', 'commit', '-m', msg], dry_run=dry_run)
    run_cmd(['git', 'push', 'origin', args.git_branch], dry_run=dry_run)


def write_pipeline_config(args: argparse.Namespace, run_dir: Path) -> None:
    payload = {
        'run_name': args.run_name,
        'run_dir': run_dir,
        'attack_geometry': 'p=2,q=2 RMS-L2 via fast_replace_l2',
        'old_linf_failure_mode': 'fast_replace_linf creates coordinatewise sign/rectangular deltas',
        'training': {
            'epochs': 1000,
            'checkpoint_every_epochs': 200,
            'training_data_mode': 'adv-only',
            'label_mode': 'solver',
            'batch_size': args.batch_size,
            'optimizer_batch_size': args.optimizer_batch_size,
            'attack_steps': args.attack_steps,
            'epsilon_fraction': args.epsilon_fraction,
            'epsilon_jitter': [args.eps_jitter_low, args.eps_jitter_high],
            'alpha_ratio': args.alpha_ratio,
            'alpha_jitter': [args.alpha_jitter_low, args.alpha_jitter_high],
            'epsilon_bucket_count': 5,
        },
        'visualizations': args.viz_root,
        'jacobian_svd': {
            'sample_manifest': args.sample_manifest,
            'reuse_root': args.reuse_root,
            'top_k': args.svd_top_k,
            'checkpoint_epochs': [200, 400, 600, 800, 1000],
        },
        'postprocess': {
            'svd_polished_figures': not args.skip_svd_plots,
            'attack_gif': {
                'enabled': not args.skip_attack_gif,
                'out_root': args.attack_gif_root,
                'checkpoint_epoch': args.attack_gif_checkpoint_epoch,
                'epsilon_rms': args.attack_gif_epsilon_rms,
                'alpha_rms': args.attack_gif_alpha_rms,
                'alpha_ratio_if_alpha_rms_unset': args.attack_gif_alpha_ratio,
                'attack_steps': args.attack_gif_steps,
                'frame_every': args.attack_gif_frame_every,
            },
        },
    }
    (run_dir / 'pipeline_logs').mkdir(parents=True, exist_ok=True)
    (run_dir / 'pipeline_logs' / 'pipeline_config.json').write_text(json.dumps(to_jsonable(payload), indent=2), encoding='utf-8')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['full', 'smoke', 'postprocess'], default='full')
    parser.add_argument('--run-name', default=DEFAULT_RUN_NAME)
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--viz-root', type=Path, default=DEFAULT_VIZ_ROOT)
    parser.add_argument('--forensics-root', type=Path, default=DEFAULT_FORENSICS_ROOT)
    parser.add_argument('--attack-gif-root', type=Path, default=DEFAULT_ATTACK_GIF_ROOT)
    parser.add_argument('--sample-manifest', type=Path, default=PROJECT_ROOT / 'forensics' / 'burgers_adv_training_jacobian_svd_20260531_representative20_same_points' / 'representative20_sample_manifest.csv')
    parser.add_argument('--reuse-root', type=Path, default=PROJECT_ROOT / 'forensics' / 'burgers_adv_training_jacobian_svd_20260531_representative20_same_points')
    parser.add_argument('--selected-top', type=Path, default=DEFAULT_SELECTED_TOP)
    parser.add_argument('--selected-zip', type=Path, default=DEFAULT_SELECTED_ZIP)
    parser.add_argument('--seed', type=int, default=20260601)
    parser.add_argument('--batch-size', type=int, default=480)
    parser.add_argument('--optimizer-batch-size', type=int, default=32)
    parser.add_argument('--attack-steps', type=int, default=5)
    parser.add_argument('--epsilon-fraction', type=float, default=0.06)
    parser.add_argument('--eps-jitter-low', type=float, default=0.5)
    parser.add_argument('--eps-jitter-high', type=float, default=2.5)
    parser.add_argument('--alpha-ratio', type=float, default=1.0)
    parser.add_argument('--alpha-jitter-low', type=float, default=0.75)
    parser.add_argument('--alpha-jitter-high', type=float, default=1.25)
    parser.add_argument('--attack-probe-samples', type=int, default=5)
    parser.add_argument('--smoke-epochs', type=int, default=1)
    parser.add_argument('--smoke-batches-per-epoch', type=int, default=1)
    parser.add_argument('--smoke-eval-max-samples', type=int, default=4)
    parser.add_argument('--smoke-max-generalization-eval', type=int, default=3)
    parser.add_argument('--skip-training', action='store_true')
    parser.add_argument('--skip-plots', action='store_true')
    parser.add_argument('--skip-jacobian', action='store_true')
    parser.add_argument('--skip-svd-plots', action='store_true')
    parser.add_argument('--skip-attack-gif', action='store_true')
    parser.add_argument('--skip-r2-upload', action='store_true')
    parser.add_argument('--skip-git-push', action='store_true')
    parser.add_argument('--r2-prefix', default=DEFAULT_R2_PREFIX)
    parser.add_argument('--git-branch', default='vast-ai')
    parser.add_argument('--git-commit-message', default=None)
    parser.add_argument('--svd-top-k', type=int, default=100)
    parser.add_argument('--svd-solver', choices=['propack', 'arpack', 'lobpcg'], default='propack')
    parser.add_argument('--attack-gif-checkpoint-epoch', type=int, default=1000)
    parser.add_argument('--attack-gif-epsilon-rms', type=float, default=0.12)
    parser.add_argument('--attack-gif-alpha-ratio', type=float, default=0.10)
    parser.add_argument('--attack-gif-alpha-rms', type=float, default=None)
    parser.add_argument('--attack-gif-steps', type=int, default=100)
    parser.add_argument('--attack-gif-frame-every', type=int, default=2)
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = burgers_run_dir(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_pipeline_config(args, run_dir)

    started = time.perf_counter()
    if args.mode in {'full', 'smoke'} and not args.skip_training:
        run_cmd(training_command(args, smoke=args.mode == 'smoke'), dry_run=args.dry_run, log_path=run_dir / 'pipeline_logs' / 'training.log')
    validate_delta(args, run_dir, dry_run=args.dry_run)

    if args.mode != 'smoke' and not args.skip_plots:
        run_plots(args, run_dir, dry_run=args.dry_run)
    if args.mode != 'smoke' and not args.skip_jacobian:
        run_jacobian_svd(args, run_dir, dry_run=args.dry_run)
    if args.mode != 'smoke' and not args.skip_svd_plots:
        run_svd_visualizations(args, run_dir, dry_run=args.dry_run)
    if args.mode != 'smoke' and not args.skip_attack_gif:
        run_attack_gif_visualization(args, run_dir, dry_run=args.dry_run)
    if args.mode != 'smoke' and not args.skip_r2_upload:
        sync_to_r2(args, run_dir, dry_run=args.dry_run)
    if args.mode != 'smoke' and not args.skip_git_push:
        git_push_outputs(args, run_dir, dry_run=args.dry_run)

    elapsed = time.perf_counter() - started
    summary = {
        'run_dir': run_dir,
        'mode': args.mode,
        'elapsed_sec': elapsed,
        'geometry': 'fast_replace_l2 / p=2,q=2',
        'next_full_command': [str(x) for x in training_command(args, smoke=False)],
    }
    (run_dir / 'pipeline_logs' / 'pipeline_done.json').write_text(json.dumps(to_jsonable(summary), indent=2), encoding='utf-8')
    print(json.dumps(to_jsonable(summary), indent=2), flush=True)


if __name__ == '__main__':
    main()
