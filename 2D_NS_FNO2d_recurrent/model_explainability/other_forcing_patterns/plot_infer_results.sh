#!/bin/bash
#SBATCH --job-name=plot_infer_results
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G                       
#SBATCH --time=20:00:00         
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yifeisun@umich.edu
#SBATCH --output=./2D_NS_FNO2d_recurrent/model_explainability/other_forcing_patterns/logs/%x_%j.out
#SBATCH --error=./2D_NS_FNO2d_recurrent/model_explainability/other_forcing_patterns/logs/%x_%j.err

hostname;date;pwd
export XDG_RUNTIME_DIR=${SLURM_TMPDIR}

#activate enviorment
source ~/.bashrc
conda activate adv_robust
export PYTHONPATH=/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO:$PYTHONPATH

srun -u python ./2D_NS_FNO2d_recurrent/model_explainability/other_forcing_patterns/plot_infer_results.py