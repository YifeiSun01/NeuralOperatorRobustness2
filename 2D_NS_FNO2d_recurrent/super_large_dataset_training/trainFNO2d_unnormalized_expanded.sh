#!/bin/bash
#SBATCH --job-name=trainFNO2d_allmerge
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=70G
#SBATCH --time=100:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yifeisun@umich.edu
#SBATCH --output=./2D_NS_FNO2d_recurrent/super_large_dataset_training/logs/%x_%j.out
#SBATCH --error=./2D_NS_FNO2d_recurrent/super_large_dataset_training/logs/%x_%j.err

set -euo pipefail
hostname; date; pwd
export XDG_RUNTIME_DIR=${SLURM_TMPDIR}

# env
set +u; source ~/.bashrc || true; set -u
conda activate adv_robust
export PYTHONPATH="/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO:${PYTHONPATH:-}"

mkdir -p ./2D_NS_FNO2d_recurrent/super_large_dataset_training/logs

# ====== 可编辑区 ======
EXP_DIR="./2D_NS_FNO2d_recurrent/datasets/expanded_exponax_datasets/t20/N=1150"
EPOCHS=500
BATCH_SIZE=20
NTEST=100
SEED=1234
NUM_WORKERS=0     # 大数据强烈建议先用 0，防止多进程各自加载同一大文件
CKPT_EVERY=20
# =====================

python -u ./2D_NS_FNO2d_recurrent/super_large_dataset_training/trainFNO2d_unnormalized_expanded.py \
  --expanded_dir "$EXP_DIR" \
  --epochs ${EPOCHS} \
  --batch_size ${BATCH_SIZE} \
  --ntest ${NTEST} \
  --seed ${SEED} \
  --num_workers ${NUM_WORKERS} \
  --ckpt_every ${CKPT_EVERY}


