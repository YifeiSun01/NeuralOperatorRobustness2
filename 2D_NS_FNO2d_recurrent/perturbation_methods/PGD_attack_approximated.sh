#!/bin/bash
#SBATCH --job-name=PGD_attack_approximated
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=150G                       
#SBATCH --time=30:00:00         
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yifeisun@umich.edu
#SBATCH --output=./2D_NS_FNO2d_recurrent/perturbation_methods/logs/%x_%j.out
#SBATCH --error=./2D_NS_FNO2d_recurrent/perturbation_methods/logs/%x_%j.err

hostname;date;pwd
export XDG_RUNTIME_DIR=${SLURM_TMPDIR}

#activate enviorment
source ~/.bashrc
conda activate adv_robust
export PYTHONPATH=/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO:$PYTHONPATH

python -u ./2D_NS_FNO2d_recurrent/perturbation_methods/PGD_attack_approximated.py
