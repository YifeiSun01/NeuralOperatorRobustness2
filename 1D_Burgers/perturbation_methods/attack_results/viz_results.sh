#!/bin/bash
#SBATCH --job-name=viz_results  
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=2             # CPU
#SBATCH --mem=20G                     
#SBATCH --time=18:00:00               # max running time hr:min:sec
#SBATCH --mail-type=ALL         
#SBATCH --mail-user=yifeisun@umich.edu
#SBATCH --output=./1D_Burgers/perturbation_methods/attack_results/logs_test/%x_%j.out  # 自动生成日志文件
#SBATCH --error=./1D_Burgers/perturbation_methods/attack_results/logs_test/%x_%j.err   # 错误日志

hostname;date;pwd
export XDG_RUNTIME_DIR=${SLURM_TMPDIR}

#activate enviorment
source ~/.bashrc
conda activate adv_robust
export PYTHONPATH=/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO:$PYTHONPATH

python -u ./1D_Burgers/perturbation_methods/attack_results/viz_results.py



