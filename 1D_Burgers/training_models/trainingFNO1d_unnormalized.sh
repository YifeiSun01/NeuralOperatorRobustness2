#!/bin/bash
#SBATCH --job-name=trainingFNO1d_unnormalized     
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=2             # CPU
#SBATCH --mem=2G                     
#SBATCH --time=3:00:00               # max running time hr:min:sec
#SBATCH --mail-type=ALL         
#SBATCH --mail-user=yifeisun@umich.edu
#SBATCH --output=./1D_Burgers/training_models/logs/%x_%j.out  # 自动生成日志文件
#SBATCH --error=./1D_Burgers/training_models/logs/%x_%j.err   # 错误日志

hostname;date;pwd
export XDG_RUNTIME_DIR=${SLURM_TMPDIR}

#activate enviorment
source ~/.bashrc
conda activate adv_robust
export PYTHONPATH=/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO:$PYTHONPATH

python -u ./1D_Burgers/training_models/trainingFNO1d_unnormalized.py
