#!/bin/bash
#SBATCH --job-name=VT_NS_gen_all_frame    
#SBATCH --partition=gpu               # partition name
#SBATCH --gres=gpu:a100:1                 # GPU
#SBATCH --cpus-per-task=2             # CPU
#SBATCH --mem=100G                     
#SBATCH --time=18:00:00               # max running time hr:min:sec
#SBATCH --mail-type=ALL         
#SBATCH --mail-user=yifeisun@umich.edu
#SBATCH --output=./data_generation/logs_gen/%x_%j.out  


hostname;date;pwd
export XDG_RUNTIME_DIR=${SLURM_TMPDIR}

source ~/.bashrc
conda activate adv_robust
export PYTHONPATH=/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO/2D_NS:$PYTHONPATH

python -u ./data_generation/VT_NS_gen_all_frame.py 2>&1

