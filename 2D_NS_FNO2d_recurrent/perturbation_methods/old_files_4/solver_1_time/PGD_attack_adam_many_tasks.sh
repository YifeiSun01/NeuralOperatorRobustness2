#!/bin/bash
#SBATCH --job-name=PGD_attack_adam
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=30:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yifeisun@umich.edu
#SBATCH --output=./2D_NS_FNO2d_recurrent/perturbation_methods/logs/%x_%A_%a.out
#SBATCH --error=./2D_NS_FNO2d_recurrent/perturbation_methods/logs/%x_%A_%a.err
#SBATCH --array=0-5   # 启动 6 个并行任务

hostname; date; pwd

# ------------------ 基础环境 ------------------
export XDG_RUNTIME_DIR="${SLURM_TMPDIR}"
source ~/.bashrc
conda activate adv_robust
export PYTHONPATH=/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO:$PYTHONPATH

# ------------------ 关键：a/d/w 控制串 ------------------
# modes=(
#  "wwwwwwwwww" "aaaaaaaaaa" "dddddddddd" 
#  "aaaaaddddw" "addddaaaaw" "ddddaaaaaw"
#  "aaaddddwww" "aaddddddww" "addddddddw" "wwwddddaaa" "wwddddddaa" "wdddddddda"
#  "aaaaaaaaaw" "aaaaaaawww" "awawawawaw" "aaaaawwwww" "aaawwwwwww" "awwwwwwwww"
#  "dddddddddw" "dddddddwww" "dwdwdwdwdw" "dddddwwwww" "dddwwwwwww" "dwwwwwwwww"
#  "wwwwwwwwwa"
#  "wwwwdddddw"
# )

modes=(
 "aaaddddwwd" "aaddddddwd" "addddddddd" "wwwddddaad" "wwddddddad" "wddddddddd"
 "aaaaaaaaad" "aaaaaaawwd" "awawawawad" "aaaaawwwwd" "aaawwwwwwd" "awwwwwwwwd"
 "dddddddddd" "dddddddwwd" "dwdwdwdwdd" "dddddwwwwd" "dddwwwwwwd" "dwwwwwwwwd"
)

# 总共有 24 个模式，均匀分配到 6 个任务，每个任务跑 4 个模式
num_modes=${#modes[@]}
modes_per_job=$(( (num_modes + 2 - 1) / 2 ))   # 向上取整，每个任务 4 个

start=$(( SLURM_ARRAY_TASK_ID * modes_per_job ))
end=$(( start + modes_per_job - 1 ))

if [ $end -ge $num_modes ]; then
    end=$((num_modes - 1))
fi

echo ">>> Task ${SLURM_ARRAY_TASK_ID} running modes index $start to $end"

for i in $(seq $start $end); do
    MODE_SPEC=${modes[$i]}
    echo ">>> Running with MODE_SPEC=${MODE_SPEC}"
    export MODE_SPEC
    srun python -u ./2D_NS_FNO2d_recurrent/perturbation_methods/PGD_attack_adam.py
done
