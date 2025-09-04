#!/bin/bash
#SBATCH --job-name=trainFNO2d_unnorm_expanded
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=50:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yifeisun@umich.edu
# 更容易区分 array 子任务的日志
#SBATCH --output=./2D_NS_FNO2d_recurrent/training_models_expanded/logs/%x_%A-%a.out
#SBATCH --error=./2D_NS_FNO2d_recurrent/training_models_expanded/logs/%x_%A-%a.err
# 启动 5 个 array 子任务（索引 0 到 4）
#SBATCH --array=0-4

set -euo pipefail

hostname; date; pwd
export XDG_RUNTIME_DIR=${SLURM_TMPDIR}

# env
set +u
source ~/.bashrc || true
set -u
conda activate adv_robust

: "${PYTHONPATH:=}"   # 若未定义，则设为空串（在 set -u 下安全）
export PYTHONPATH="/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO:$PYTHONPATH"

# 确保日志目录存在
mkdir -p ./2D_NS_FNO2d_recurrent/training_models_expanded/logs

# ========= 可编辑区域 =========
EXP_DIR="./2D_NS_FNO2d_recurrent/datasets/expanded_exponax_datasets/t20/N=1150"   # 新数据目录
PERCENTS=(0.25 0.5 0.75 1.0)
EPOCHS=1500
BATCH_SIZE=20
SEED=1234
# ============================

# 找出该目录下所有 .pt 新数据文件
mapfile -t FILES < <(find "$EXP_DIR" -maxdepth 1 -type f -name "*.pt" | sort)
echo "Expanded files found: ${#FILES[@]}"

# 生成所有 (文件 × 百分比) 组合
COMBOS=()
for f in "${FILES[@]}"; do
  for pct in "${PERCENTS[@]}"; do
    COMBOS+=("$f|$pct")
  done
done

TOTAL=${#COMBOS[@]}
TASKS=${SLURM_ARRAY_TASK_COUNT:-1}   # 这里是 2
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}    # 0 或 1

# 计算每个任务要处理的区间 [START, END)
PER_TASK=$(( (TOTAL + TASKS - 1) / TASKS ))   # 向上取整
START=$(( TASK_ID * PER_TASK ))
END=$(( START + PER_TASK ))
if (( START >= TOTAL )); then
  echo "No work for TASK_ID=${TASK_ID} (START=${START} >= TOTAL=${TOTAL}). Exit."
  exit 0
fi
if (( END > TOTAL )); then END=$TOTAL; fi

echo "TOTAL combos: ${TOTAL}"
echo "TASKS: ${TASKS}, TASK_ID: ${TASK_ID}"
echo "Processing slice: [${START}, ${END})  (size=$((END-START)))"

# 跑本任务负责的这段组合
for (( i=START; i<END; i++ )); do
  IFS='|' read -r f pct <<< "${COMBOS[$i]}"
  echo "==> Running combo $((i+1))/${TOTAL}: file=$f  percent=$pct"
  python -u ./2D_NS_FNO2d_recurrent/training_models_expanded/trainFNO2d_unnormalized_expanded.py \
    --expanded_file "$f" \
    --percent "$pct" \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --seed ${SEED}
done

