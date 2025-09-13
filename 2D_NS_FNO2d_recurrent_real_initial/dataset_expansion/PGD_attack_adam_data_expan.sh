#!/bin/bash
#SBATCH --job-name=PGD_attack_adam_data_expan
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=70:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yifeisun@umich.edu
#SBATCH --output=./2D_NS_FNO2d_recurrent_real_initial/dataset_expansion/logs/%x_%A_%a.out
#SBATCH --error=./2D_NS_FNO2d_recurrent_real_initial/dataset_expansion/logs/%x_%A_%a.err
#SBATCH --array=0-1   # 2个任务，均匀分配全部组合

hostname; date; pwd

# ------------------ 基础环境 ------------------
export XDG_RUNTIME_DIR="${SLURM_TMPDIR}"
source ~/.bashrc
conda activate adv_robust
export PYTHONPATH=/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO:$PYTHONPATH

# ------------------ 参数列表 ------------------
modes=( "wwwwwwwwww" )
eps_list=( 0.0006 0.0012 )
alpha_list=( 0.10 0.50 )

# 固定参数
norm=2
num_steps=10
train_test="train"   # 或 "test"

# ------------------ 计算总组合数 ------------------
NM=${#modes[@]}
NE=${#eps_list[@]}
NA=${#alpha_list[@]}
total=$(( NM * NE * NA ))

# 任务总数（与 --array 的任务个数一致，用它来做步长）
N_TASKS=2

echo "Total combos = $total"
echo "This array task id = ${SLURM_ARRAY_TASK_ID}"
echo "N_TASKS (stride) = ${N_TASKS}"

# ------------------ 工具函数：线性索引 -> (i_mode, i_eps, i_alpha) ------------------
idx_to_combo () {
  local idx=$1
  local ne_na=$(( NE * NA ))
  i_mode=$(( idx / ne_na ))
  rem=$(( idx % ne_na ))
  i_eps=$(( rem / NA ))
  i_alpha=$(( rem % NA ))
}

# ------------------ 均匀分配：步长=N_TASKS，分别领取 idx = tid, tid+N_TASKS, ... ------------------
for idx in $(seq ${SLURM_ARRAY_TASK_ID} ${N_TASKS} $(( total - 1 ))); do
  idx_to_combo $idx
  MODE_SPEC=${modes[$i_mode]}
  EPS_MULT=${eps_list[$i_eps]}
  ALPHA_MULT=${alpha_list[$i_alpha]}

  echo ">>> Combo idx=$idx: MODE_SPEC=$MODE_SPEC eps_mult=$EPS_MULT alpha_mult=$ALPHA_MULT"

  srun python -u ./2D_NS_FNO2d_recurrent_real_initial/dataset_expansion/PGD_attack_adam_data_expan.py \
    --mode_spec "${MODE_SPEC}" \
    --eps_mult "${EPS_MULT}" \
    --alpha_mult "${ALPHA_MULT}" \
    --norm "${norm}" \
    --num_steps "${num_steps}" \
    --train_test "${train_test}"
done
