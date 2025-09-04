#!/bin/bash
#SBATCH --job-name=attack_1d_all_methods_test     
#SBATCH --partition=gpu               # partition name
#SBATCH --gres=gpu:a100:1             # GPU
#SBATCH --cpus-per-task=2             # CPU
#SBATCH --mem=32G                     
#SBATCH --time=18:00:00               # max running time hr:min:sec
#SBATCH --mail-type=ALL         
#SBATCH --mail-user=yifeisun@umich.edu
#SBATCH --output=./perturbation_methods/logs_test//%x_%A_%a_%N.out  # 自动生成日志文件
#SBATCH --array=0-7                   # 8 tasks (4 files × 2 num_records)

hostname;date;pwd
export XDG_RUNTIME_DIR=${SLURM_TMPDIR}

#activate environment
source ~/.bashrc
conda activate adv_robust
export PYTHONPATH=/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO:$PYTHONPATH

# Define arrays for parameters
files=(
    "runner1d_test_norm2infmodified_2_use_g"
    "runner1d_test_norm2infmodified_2_use_g_tilt"
    "runner1d_test_norm2infmodified_2_use_g_nosign"
    "runner1d_test_norm2infmodified_2_use_g_tilt_nosign"
)
num_records_list=(10 100)

# Calculate current indices
file_index=$((SLURM_ARRAY_TASK_ID / 2))
records_index=$((SLURM_ARRAY_TASK_ID % 2))

# Get current parameters
file=${files[$file_index]}
num_records=${num_records_list[$records_index]}

# Run the specific task
python -u ./perturbation_methods/"${file}".py \
    --workdir=./perturbation_results/1D \
    --config=./perturbation_methods/configs/1D/burgers.py \
    --config.num_steps_list="(5,10,20,50,100,200)" \
    --config.epsilon_list="(0.001,0.005,0.01,0.05,0.1,0.5,1,2)" \
    --config.num_records="$num_records"

