#!/bin/bash
#SBATCH --job-name=PGD_attack_adam_adaptive
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=30:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yifeisun@umich.edu
#SBATCH --output=./2D_NS_FNO2d_recurrent/perturbation_methods/logs/%x_%A_%a.out
#SBATCH --error=./2D_NS_FNO2d_recurrent/perturbation_methods/logs/%x_%A_%a.err

hostname; date; pwd

# ------------------ 基础环境 ------------------
export XDG_RUNTIME_DIR="${SLURM_TMPDIR}"
source ~/.bashrc
conda activate adv_robust
export PYTHONPATH=/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO:$PYTHONPATH
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# ------------------ 单次运行的 MODE_SPEC ------------------
# 如需更改模式，直接改下面这一行或从外部 export MODE_SPEC 后提交
MODE_SPEC="${MODE_SPEC:-wwwwwwwwww}"
export MODE_SPEC

echo ">>> Running SINGLE job with MODE_SPEC=${MODE_SPEC}"
srun python -u ./2D_NS_FNO2d_recurrent/perturbation_methods/PGD_attack_adam_adaptive.py
