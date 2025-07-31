#!/bin/bash
#SBATCH --job-name=attack_1d_test     
#SBATCH --partition=gpu               # partition name
#SBATCH --gres=gpu:a100:1                 # GPU
#SBATCH --cpus-per-task=2             # CPU
#SBATCH --mem=32G                     
#SBATCH --time=18:00:00               # max running time hr:min:sec
#SBATCH --mail-type=ALL         
#SBATCH --mail-user=yifeisun@umich.edu
#SBATCH --output=./perturbed_results_as_inputs/logs_test/%x_%j.out  # 自动生成日志文件

hostname;date;pwd
export XDG_RUNTIME_DIR=${SLURM_TMPDIR}

#activate enviorment
source ~/.bashrc
conda activate adv_robust
export PYTHONPATH=/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO:$PYTHONPATH
# module load cuda
# nvidia-smi
# nvcc --version

python -u ./perturbed_results_as_inputs/runner1d_test_gradient_model_gradient_unclear_perturbed_input_no_boundary.py \
    --workdir=./perturbed_results_as_inputs/1D \
    --config=./perturbed_results_as_inputs/configs/1D/burgers_test_2.py 2>&1

