#!/bin/bash
#SBATCH --job-name=PGD_attack_adam_data_expan
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=90:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yifeisun@umich.edu
#SBATCH --output=./2D_NS_FNO2d_recurrent/dataset_expansion/logs/%x_%j.out
#SBATCH --error=./2D_NS_FNO2d_recurrent/dataset_expansion/logs/%x_%j.err
# ↑ 不再使用 --array；单任务日志用 %j（JobID）

hostname; date; pwd

# ------------------ 基础环境 ------------------
export XDG_RUNTIME_DIR="${SLURM_TMPDIR}"
source ~/.bashrc
conda activate adv_robust
export PYTHONPATH=/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO:$PYTHONPATH

# ------------------ 参数列表（保持你现在这几组） ------------------
modes=( "wwwwwwwwww" )
eps_list=( 0.0036 0.0072 )
alpha_list=( 1 2 )

# 固定参数
norm=2
num_steps=10
train_test="train"   # 或 "test"

echo "Total combos = $(( ${#modes[@]} * ${#eps_list[@]} * ${#alpha_list[@]} ))"

# ------------------ 在一个任务里顺序跑完所有组合 ------------------
for MODE_SPEC in "${modes[@]}"; do
  for EPS_MULT in "${eps_list[@]}"; do
    for ALPHA_MULT in "${alpha_list[@]}"; do
      echo ">>> Running: MODE_SPEC=${MODE_SPEC}  eps_mult=${EPS_MULT}  alpha_mult=${ALPHA_MULT}"
      srun python -u ./2D_NS_FNO2d_recurrent/dataset_expansion/PGD_attack_adam_data_expan.py \
        --mode_spec "${MODE_SPEC}" \
        --eps_mult "${EPS_MULT}" \
        --alpha_mult "${ALPHA_MULT}" \
        --norm "${norm}" \
        --num_steps "${num_steps}" \
        --train_test "${train_test}"
      echo ">>> Finished: MODE_SPEC=${MODE_SPEC} eps_mult=${EPS_MULT} alpha_mult=${ALPHA_MULT}"
      echo "------------------------------------------------------------"
    done
  done
done
