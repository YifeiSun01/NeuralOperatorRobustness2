#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p logs
LOG="logs/loss3_overnight_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$LOG") 2>&1

echo "[start] loss3 overnight run: $(date -u --iso-8601=seconds)"
echo "[log] $LOG"

echo "[gpu] nvidia-smi"
nvidia-smi

echo "[gpu] torch/jax quick check"
adv_robust/bin/python -c 'import torch, jax; print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available()); print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO_CUDA"); print("arch", torch.cuda.get_arch_list() if torch.cuda.is_available() else []); print("jax backend", jax.default_backend()); print("jax devices", jax.devices())'

echo
echo "========== 1. Fill missing p=2,q=2 100-step settings =========="
adv_robust/bin/python tools/run_loss3_alpha_epsilon_core4_sweep.py \
  --base-out forensics/loss3_alpha_epsilon_core4_sweep_20260519 \
  --steps 100 \
  --p 2 --q 2 \
  --skip-completed \
  --trajectory-indices 0 7 40 47 \
  --selected-steps 0 1 2 5 10 25 50 75 100

adv_robust/bin/python tools/analyze_loss3_alpha_epsilon_core4_sweep.py \
  --base forensics/loss3_alpha_epsilon_core4_sweep_20260519 \
  --out-dir forensics/loss3_alpha_epsilon_core4_analysis_20260519 \
  --doc docs/loss3_alpha_epsilon_core4_sweep_result_20260519.md

adv_robust/bin/python tools/plot_loss3_alpha_epsilon_core4_visuals.py \
  --sweep-root forensics/loss3_alpha_epsilon_core4_sweep_20260519 \
  --analysis-root forensics/loss3_alpha_epsilon_core4_analysis_20260519 \
  --out-dir forensics/loss3_alpha_epsilon_core4_visuals_20260520 \
  --doc docs/loss3_alpha_epsilon_core4_visuals_20260520.md \
  --p-filter 2 --q-filter 2 \
  --representative-settings 4:0.4 8:0.3 8:1.6 16:1.6 \
  --dataset-index 40 \
  --delta-grid-indices 0 7 40 47

adv_robust/bin/python tools/analyze_loss3_alpha_epsilon_core4_delta_similarity.py \
  --sweep-root forensics/loss3_alpha_epsilon_core4_sweep_20260519 \
  --out-dir forensics/loss3_alpha_epsilon_core4_delta_similarity_20260520 \
  --doc docs/loss3_alpha_epsilon_core4_delta_similarity_20260520.md \
  --p-filter 2 --q-filter 2

adv_robust/bin/python tools/analyze_loss3_post_boundary_mechanism.py \
  --sweep-root forensics/loss3_alpha_epsilon_core4_sweep_20260519 \
  --out-dir forensics/loss3_post_boundary_mechanism_20260520 \
  --doc docs/loss3_post_boundary_mechanism_diagnostics_20260520.md \
  --p-filter 2 --q-filter 2 \
  --boundary-threshold 0.99

echo
echo "========== 2. Run all 20 p=2,q=2 settings for 300 steps =========="
adv_robust/bin/python tools/run_loss3_alpha_epsilon_core4_sweep.py \
  --base-out forensics/loss3_alpha_epsilon_core4_sweep_p2q2_300steps_20260520 \
  --steps 300 \
  --p 2 --q 2 \
  --skip-completed \
  --trajectory-indices 0 7 40 47 \
  --selected-steps 0 1 2 5 10 25 50 75 100 150 200 250 300 \
  --no-plots

adv_robust/bin/python tools/analyze_loss3_alpha_epsilon_core4_sweep.py \
  --base forensics/loss3_alpha_epsilon_core4_sweep_p2q2_300steps_20260520 \
  --out-dir forensics/loss3_alpha_epsilon_core4_analysis_p2q2_300steps_20260520 \
  --doc docs/loss3_alpha_epsilon_core4_p2q2_300steps_result_20260520.md

adv_robust/bin/python tools/plot_loss3_alpha_epsilon_core4_visuals.py \
  --sweep-root forensics/loss3_alpha_epsilon_core4_sweep_p2q2_300steps_20260520 \
  --analysis-root forensics/loss3_alpha_epsilon_core4_analysis_p2q2_300steps_20260520 \
  --out-dir forensics/loss3_alpha_epsilon_core4_visuals_p2q2_300steps_20260520 \
  --doc docs/loss3_alpha_epsilon_core4_visuals_p2q2_300steps_20260520.md \
  --p-filter 2 --q-filter 2 \
  --representative-settings 4:0.4 8:0.3 8:1.6 16:3.2 \
  --dataset-index 40 \
  --delta-grid-indices 0 7 40 47

adv_robust/bin/python tools/analyze_loss3_alpha_epsilon_core4_delta_similarity.py \
  --sweep-root forensics/loss3_alpha_epsilon_core4_sweep_p2q2_300steps_20260520 \
  --out-dir forensics/loss3_alpha_epsilon_core4_delta_similarity_p2q2_300steps_20260520 \
  --doc docs/loss3_alpha_epsilon_core4_delta_similarity_p2q2_300steps_20260520.md \
  --p-filter 2 --q-filter 2

adv_robust/bin/python tools/analyze_loss3_post_boundary_mechanism.py \
  --sweep-root forensics/loss3_alpha_epsilon_core4_sweep_p2q2_300steps_20260520 \
  --out-dir forensics/loss3_post_boundary_mechanism_p2q2_300steps_20260520 \
  --doc docs/loss3_post_boundary_mechanism_p2q2_300steps_20260520.md \
  --p-filter 2 --q-filter 2 \
  --boundary-threshold 0.99

echo
echo "========== 3. Baseline eps=4 alpha=0.4 detailed GIF trace =========="
adv_robust/bin/python tools/run_loss3_direction_proposal_ablation.py \
  --out-root forensics/loss3_alpha_epsilon_core4_baseline_giftrace_20260520/fno_nu0p001_eps4_alpha0p4_batch100_steps300_p2_q2 \
  --methods raw_add raw_replace steepest_add steepest_replace \
  --batch-size 100 \
  --start-index 0 \
  --epsilon 4 \
  --alpha 0.4 \
  --steps 300 \
  --p 2 --q 2 \
  --seed 0 \
  --device cuda \
  --trajectory-indices 0 7 40 47 \
  --selected-steps 0 1 2 5 10 25 50 75 100 150 200 250 300 \
  --save-trajectory-final-conditions \
  --make-gifs

adv_robust/bin/python tools/analyze_loss3_alpha_epsilon_core4_sweep.py \
  --base forensics/loss3_alpha_epsilon_core4_baseline_giftrace_20260520 \
  --out-dir forensics/loss3_alpha_epsilon_core4_analysis_baseline_giftrace_20260520 \
  --doc docs/loss3_alpha_epsilon_core4_baseline_giftrace_result_20260520.md

adv_robust/bin/python tools/plot_loss3_alpha_epsilon_core4_visuals.py \
  --sweep-root forensics/loss3_alpha_epsilon_core4_baseline_giftrace_20260520 \
  --analysis-root forensics/loss3_alpha_epsilon_core4_analysis_baseline_giftrace_20260520 \
  --out-dir forensics/loss3_alpha_epsilon_core4_visuals_baseline_giftrace_20260520 \
  --doc docs/loss3_alpha_epsilon_core4_visuals_baseline_giftrace_20260520.md \
  --p-filter 2 --q-filter 2 \
  --representative-settings 4:0.4 \
  --dataset-index 40 \
  --delta-grid-indices 0 7 40 47

adv_robust/bin/python tools/plot_loss3_trajectory_gif_panels.py \
  --root forensics/loss3_alpha_epsilon_core4_baseline_giftrace_20260520/fno_nu0p001_eps4_alpha0p4_batch100_steps300_p2_q2 \
  --out-dir forensics/loss3_alpha_epsilon_core4_baseline_giftrace_20260520/trajectory_condition_gifs \
  --dataset-indices 0 7 40 47 \
  --frame-step 1 \
  --duration 0.16

adv_robust/bin/python tools/analyze_loss3_post_boundary_mechanism.py \
  --sweep-root forensics/loss3_alpha_epsilon_core4_baseline_giftrace_20260520 \
  --out-dir forensics/loss3_post_boundary_mechanism_baseline_giftrace_20260520 \
  --doc docs/loss3_post_boundary_mechanism_baseline_giftrace_20260520.md \
  --p-filter 2 --q-filter 2 \
  --boundary-threshold 0.99

echo
echo "========== 4. Run strict p != q settings for 100 steps =========="
adv_robust/bin/python tools/run_loss3_alpha_epsilon_core4_sweep.py \
  --base-out forensics/loss3_alpha_epsilon_core4_sweep_pneq_q_100steps_20260520 \
  --steps 100 \
  --pq-pairs 1:2 1:inf 2:1 2:inf inf:1 inf:2 \
  --skip-completed \
  --trajectory-indices 0 7 40 47 \
  --selected-steps 0 1 2 5 10 25 50 75 100 \
  --no-plots

adv_robust/bin/python tools/analyze_loss3_alpha_epsilon_core4_sweep.py \
  --base forensics/loss3_alpha_epsilon_core4_sweep_pneq_q_100steps_20260520 \
  --out-dir forensics/loss3_alpha_epsilon_core4_analysis_pneq_q_100steps_20260520 \
  --doc docs/loss3_alpha_epsilon_core4_pneq_q_100steps_result_20260520.md

for pq in 1:2 1:inf 2:1 2:inf inf:1 inf:2; do
  p="${pq%:*}"
  q="${pq#*:}"
  tag_p="${p//./p}"
  tag_q="${q//./p}"
  tag_p="${tag_p//inf/inf}"
  tag_q="${tag_q//inf/inf}"

  adv_robust/bin/python tools/plot_loss3_alpha_epsilon_core4_visuals.py \
    --sweep-root forensics/loss3_alpha_epsilon_core4_sweep_pneq_q_100steps_20260520 \
    --analysis-root forensics/loss3_alpha_epsilon_core4_analysis_pneq_q_100steps_20260520 \
    --out-dir "forensics/loss3_alpha_epsilon_core4_visuals_p${tag_p}_q${tag_q}_100steps_20260520" \
    --doc "docs/loss3_alpha_epsilon_core4_visuals_p${tag_p}_q${tag_q}_100steps_20260520.md" \
    --p-filter "${p}" --q-filter "${q}" \
    --representative-settings 4:0.4 8:0.3 8:1.6 16:1.6 \
    --dataset-index 40 \
    --delta-grid-indices 0 7 40 47

  adv_robust/bin/python tools/analyze_loss3_alpha_epsilon_core4_delta_similarity.py \
    --sweep-root forensics/loss3_alpha_epsilon_core4_sweep_pneq_q_100steps_20260520 \
    --out-dir "forensics/loss3_alpha_epsilon_core4_delta_similarity_p${tag_p}_q${tag_q}_100steps_20260520" \
    --doc "docs/loss3_alpha_epsilon_core4_delta_similarity_p${tag_p}_q${tag_q}_100steps_20260520.md" \
    --p-filter "${p}" --q-filter "${q}"

  adv_robust/bin/python tools/analyze_loss3_post_boundary_mechanism.py \
    --sweep-root forensics/loss3_alpha_epsilon_core4_sweep_pneq_q_100steps_20260520 \
    --out-dir "forensics/loss3_post_boundary_mechanism_p${tag_p}_q${tag_q}_100steps_20260520" \
    --doc "docs/loss3_post_boundary_mechanism_p${tag_p}_q${tag_q}_100steps_20260520.md" \
    --p-filter "${p}" --q-filter "${q}" \
    --boundary-threshold 0.99
done

echo
echo "[done] loss3 overnight run finished: $(date -u --iso-8601=seconds)"
echo "[log] $LOG"
