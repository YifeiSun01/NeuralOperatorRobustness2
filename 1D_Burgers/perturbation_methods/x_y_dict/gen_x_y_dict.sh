#!/bin/bash
#SBATCH --job-name=gen_x_y_dict    
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=2             # CPU
#SBATCH --mem=30G                     
#SBATCH --time=24:00:00               # max running time hr:min:sec
#SBATCH --mail-type=ALL         
#SBATCH --mail-user=yifeisun@umich.edu
#SBATCH --output=./1D_Burgers/perturbation_methods/x_y_dict/1D/Burgers/logs/%x_%j.out
#SBATCH --error=./1D_Burgers/perturbation_methods/x_y_dict/1D/Burgers/logs/%x_%j.err 

hostname;date;pwd
export XDG_RUNTIME_DIR=${SLURM_TMPDIR}

#activate enviorment
source ~/.bashrc
conda activate adv_robust
export PYTHONPATH=/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO:$PYTHONPATH

python -u ./1D_Burgers/perturbation_methods/x_y_dict/gen_x_y_dict.py
