#!/bin/bash
#SBATCH --job-name=data_expand
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=18:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yifeisun@umich.edu
#SBATCH --output=./1D_Burgers/dataset_expansion/logs_test/%x_%A_%a.out
#SBATCH --error=./1D_Burgers/dataset_expansion/logs_test/%x_%A_%a.err
#SBATCH --array=0-4

hostname; date; pwd
export XDG_RUNTIME_DIR="${SLURM_TMPDIR}"

# activate environment
source ~/.bashrc
conda activate adv_robust
export PYTHONPATH=/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO:$PYTHONPATH

base_path="/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO"

dataset_pt="$base_path/1D_Burgers/datasets/1D/Burgers/pos/dim1d_nx1024_N1500_solver=exponax_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.01_t1.0_seed45.pt"
model_pth="$base_path/1D_Burgers/dataset_expansion/unnormalized/burgers_1d_FNO_model_trainedby_dim1d_nx1024_N1500_solver=exponax_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.0005_t1.0_seed45.pth"
out_root="$base_path/1D_Burgers/datasets/1D/Burgers/expanded_pos/t1"

# 可选：存在性检查
[[ -f "$dataset_pt" ]] || { echo "Missing dataset_pt: $dataset_pt" >&2; exit 1; }
[[ -f "$model_pth"  ]] || { echo "Missing model_pth:  $model_pth"  >&2; exit 1; }

# 固定参数
nu_val="0.0005"
t_final="1.0"
dt="0.0001"
norm="2"
steps="20"

# 组装全部 15 个组合
EPS_LIST=(10 20 50)
ALPHA_LIST=(0.01 0.05 0.1 0.5 1)

COMBOS=()
for eps in "${EPS_LIST[@]}"; do
  for alpha in "${ALPHA_LIST[@]}"; do
    COMBOS+=("${norm},${eps},${steps},${alpha}")
  done
done

TOTAL=${#COMBOS[@]}
TASKS=5                              # 与 --array=0-4 对应
# 每个子任务的份额（向上取整，保证分完）
PER_TASK=$(( (TOTAL + TASKS - 1) / TASKS ))

# 计算当前子任务负责的区间
START=$(( SLURM_ARRAY_TASK_ID * PER_TASK ))
END=$(( START + PER_TASK - 1 ))
if (( END >= TOTAL )); then END=$(( TOTAL - 1 )); fi

echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} handling indices [${START}, ${END}] out of ${TOTAL}"

# 逐个组合跑（保持你原来的“一次 Python 跑一个组合”的方式）
for (( i=START; i<=END; i++ )); do
  combo="${COMBOS[$i]}"
  IFS=',' read -r norm_val eps_val steps_val alpha_val <<< "$combo"
  echo "=== RUN: norm=${norm_val}, epsilon=${eps_val}, steps=${steps_val}, alpha=${alpha_val} ==="

  python "$base_path/1D_Burgers/dataset_expansion/data_expand.py" \
    --dataset_pt "$dataset_pt" \
    --model_pth "$model_pth" \
    --solver exponax \
    --nu "$nu_val" \
    --t_final "$t_final" \
    --dt "$dt" \
    --out_root "$out_root" \
    --inputs "$combo"

  echo "=== DONE: norm=${norm_val}, epsilon=${eps_val}, steps=${steps_val}, alpha=${alpha_val} ==="
done
