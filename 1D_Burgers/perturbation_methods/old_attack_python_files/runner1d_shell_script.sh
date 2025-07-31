#!/bin/bash
#SBATCH --job-name=attack_1d_all_methods     
#SBATCH --partition=gpu               # partition name
#SBATCH --gres=gpu:a100:1                 # GPU
#SBATCH --cpus-per-task=2             # CPU
#SBATCH --mem=32G                     
#SBATCH --time=18:00:00               # max running time hr:min:sec
#SBATCH --mail-type=ALL         
#SBATCH --mail-user=yifeisun@umich.edu
#SBATCH --output=./perturbation_methods/new_logs/%x_%j.out  # 自动生成日志文件
#SBATCH --error=./perturbation_methods/new_logs/%x_%j.err   # 错误日志


# attack_methods=('PGD with solver gradient'  'PGD without solver gradient' "random")
# attack_methods=('PGD without solver gradient' "random")
attack_methods=('PGD with solver gradient')

hostname;date;pwd
export XDG_RUNTIME_DIR=${SLURM_TMPDIR}

#activate enviorment
source ~/.bashrc
conda activate adv_robust
export PYTHONPATH=/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO:$PYTHONPATH
# module load cuda
# nvidia-smi
# nvcc --version

for attack_method in "${attack_methods[@]}"; do
    echo "=== Running attack: ${attack_method} ==="

    python -u ./perturbation_methods/runner1d.py \
        --workdir=./perturbation_results/1D \
        --config=./perturbation_methods/configs/1D/burgers.py \
        --config.attack_method="${attack_method}" 2>&1
    echo "=== Finished attack: ${attack_method} ==="
done
