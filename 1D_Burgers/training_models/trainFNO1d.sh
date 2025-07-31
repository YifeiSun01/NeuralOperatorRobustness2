#!/bin/bash
#SBATCH --job-name=trainFNO1d     
#SBATCH --partition=gpu               # partition name
#SBATCH --gres=gpu:a100:1                 # GPU
#SBATCH --cpus-per-task=2             # CPU
#SBATCH --mem=2G                     
#SBATCH --time=8:00:00               # max running time hr:min:sec
#SBATCH --mail-type=ALL         
#SBATCH --mail-user=yifeisun@umich.edu
#SBATCH --output=./training_models/logs/%x_%j.out  # 自动生成日志文件
#SBATCH --error=./training_models/logs/%x_%j.err   # 错误日志

hostname;date;pwd
export XDG_RUNTIME_DIR=${SLURM_TMPDIR}

#activate enviorment
source ~/.bashrc
conda activate adv_robust
export PYTHONPATH=/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO:$PYTHONPATH

python -u ./training_models/trainingFNO1d_unnormalized.py
