#!/bin/bash
#SBATCH --job-name=attack_1d_all_methods_test     
#SBATCH --partition=gpu               # partition name
#SBATCH --gres=gpu:a100:1                 # GPU
#SBATCH --cpus-per-task=2             # CPU
#SBATCH --mem=32G                     
#SBATCH --time=18:00:00               # max running time hr:min:sec
#SBATCH --mail-type=ALL         
#SBATCH --mail-user=yifeisun@umich.edu
#SBATCH --output=./perturbation_methods/logs_test/%x_%j.out  # 自动生成日志文件

hostname;date;pwd
export XDG_RUNTIME_DIR=${SLURM_TMPDIR}

#activate enviorment
source ~/.bashrc
conda activate adv_robust
export PYTHONPATH=/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO:$PYTHONPATH
# module load cuda
# nvidia-smi
# nvcc --version

python -u ./perturbation_methods/runner1d_compare.py \
    --workdir=./perturbation_results/1D \
    --config=./perturbation_methods/configs/1D/burgers.py \
    --config.num_records=50 2>&1

