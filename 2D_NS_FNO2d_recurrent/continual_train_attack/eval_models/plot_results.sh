#!/bin/bash
#SBATCH --job-name=plot_results
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1          # 作图不需要GPU；如要CPU版见下
#SBATCH --cpus-per-task=4
#SBATCH --mem=5G
#SBATCH --time=02:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yifeisun@umich.edu
#SBATCH --output=./2D_NS_FNO2d_recurrent/continual_train_attack/eval_models/logs/%x_%j.out
#SBATCH --error=./2D_NS_FNO2d_recurrent/continual_train_attack/eval_models/logs/%x_%j.err

set -euo pipefail
hostname; date; pwd
export XDG_RUNTIME_DIR=${SLURM_TMPDIR}

# ===== env =====
export BASHRCSOURCED=1
source ~/.bashrc || true
conda activate adv_robust
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg   # 无显卡环境也能画图

# ===== paths =====
PROJECT_ROOT=/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO
# 用缺省值避免 PYTHONPATH 未定义时报错
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

PLOT_SCRIPT=$PROJECT_ROOT/2D_NS_FNO2d_recurrent/continual_train_attack/eval_models/plot_results.py
CSV_PATH=$PROJECT_ROOT/2D_NS_FNO2d_recurrent/continual_train_attack/eval_models/performance_results/combined_eval__roots_models.csv
OUTPUT_ROOT=$PROJECT_ROOT/2D_NS_FNO2d_recurrent/continual_train_attack/eval_models/plot_results

echo "=== RUNTIME INFO ==="
python -V
nvidia-smi || true
echo "PROJECT_ROOT : $PROJECT_ROOT"
echo "PLOT_SCRIPT  : $PLOT_SCRIPT"
echo "CSV_PATH     : $CSV_PATH"
echo "OUTPUT_ROOT  : $OUTPUT_ROOT"
echo "==============="

python -u "$PLOT_SCRIPT" \
  --csv "$CSV_PATH" \
  --save-root "$OUTPUT_ROOT/plots/models_rmse_minus" \
  --pivot-csv-name "pivot_rmse_minus_baseline_from_pivot.csv"
