#!/bin/bash
#SBATCH --job-name=out_of_distr_1d     
#SBATCH --partition=gpu               # partition name
#SBATCH --gres=gpu:a100:1                 # GPU
#SBATCH --cpus-per-task=2             # CPU
#SBATCH --mem=8G                     
#SBATCH --time=8:00:00               # max running time hr:min:sec
#SBATCH --mail-type=ALL         
#SBATCH --mail-user=yifeisun@umich.edu
#SBATCH --output=./out_of_distr_test/logs/%x_%j.out  # 自动生成日志文件
#SBATCH --error=./out_of_distr_test/logs/%x_%j.err   # 错误日志

hostname;date;pwd
export XDG_RUNTIME_DIR=${SLURM_TMPDIR}

#activate enviorment
source ~/.bashrc
conda activate adv_robust
export PYTHONPATH=/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO:$PYTHONPATH

SCRIPTS=(
  "./out_of_distr_test/gaussian_diff_distr_1d.py"
  "./out_of_distr_test/gaussian_diff_distr_nonperiodic_1d.py"
  "./out_of_distr_test/matern_diff_distr_1d.py"
  "./out_of_distr_test/zigzag_1d.py"
)

# Loop through each script and run with both True and False
for script in "${SCRIPTS[@]}"; do
  for bool_value in "True" "False"; do
    echo "Running $script with $bool_value"
    python -u "$script" "$bool_value"
    echo "----------------------------------"
  done
done