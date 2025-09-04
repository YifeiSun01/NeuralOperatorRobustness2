#!/bin/bash
#SBATCH --job-name=run_generalization    
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=2             # CPU
#SBATCH --mem=10G                     
#SBATCH --time=23:00:00               # max running time hr:min:sec
#SBATCH --mail-type=ALL         
#SBATCH --mail-user=yifeisun@umich.edu
#SBATCH --output=./1D_Burgers_FNO_generalization/saved_models/logs/%x_%j.out  # 自动生成日志文件
#SBATCH --error=./1D_Burgers_FNO_generalization/saved_models/logs/%x_%j.err   # 错误日志

hostname;date;pwd
export XDG_RUNTIME_DIR=${SLURM_TMPDIR}

#activate enviorment
source ~/.bashrc
conda activate adv_robust
export PYTHONPATH=/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO:$PYTHONPATH

python -u ./1D_Burgers_FNO_generalization/saved_models/run_generalization.py
