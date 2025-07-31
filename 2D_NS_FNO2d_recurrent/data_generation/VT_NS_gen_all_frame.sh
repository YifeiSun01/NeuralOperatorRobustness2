#!/bin/bash
#SBATCH --job-name=VT_NS_gen_all_frame
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G                       
#SBATCH --time=30:00:00         
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yifeisun@umich.edu
#SBATCH --output=./data_generation/logs_gen/%x_%j.out

hostname;date;pwd
export XDG_RUNTIME_DIR=${SLURM_TMPDIR}

source ~/.bashrc
conda activate adv_robust
export PYTHONPATH=/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO/2D_NS:$PYTHONPATH

python -u ./data_generation/VT_NS_gen_all_frame.py 2>&1

