#!/bin/bash
#SBATCH --job-name=attack_1d_all_methods_2     
#SBATCH --partition=gpu               # partition name
#SBATCH --gres=gpu:a100:1                 # GPU
#SBATCH --cpus-per-task=2             # CPU
#SBATCH --mem=32G                     
#SBATCH --time=18:00:00               # max running time hr:min:sec
#SBATCH --mail-type=ALL         
#SBATCH --mail-user=yifeisun@umich.edu
#SBATCH --output=./perturbation_methods/logs_attack/%x_%j.out  # 自动生成日志文件

hostname;date;pwd
export XDG_RUNTIME_DIR=${SLURM_TMPDIR}

#activate enviorment
source ~/.bashrc
conda activate adv_robust
export PYTHONPATH=/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO:$PYTHONPATH
# module load cuda
# nvidia-smi
# nvcc --version

python -u ./perturbation_methods/runner1d_attack.py \
    --workdir=./perturbation_results/1D \
    --config=./perturbation_methods/configs/1D/burgers.py \
    --config.num_steps_list="(2,5,10,20,50,100)" \
    --config.epsilon_list="(0.005,0.01,0.05,0.1,0.5,1,1.5)" 2>&1
