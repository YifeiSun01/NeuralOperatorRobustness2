#!/bin/bash
#SBATCH --job-name=model_shortened_infer
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=20:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yifeisun@umich.edu
#SBATCH --output=./2D_NS_FNO2d_recurrent/model_explainability/other_forcing_patterns/logs/%x_%j.out
#SBATCH --error=./2D_NS_FNO2d_recurrent/model_explainability/other_forcing_patterns/logs/%x_%j.err

hostname; date; pwd
export XDG_RUNTIME_DIR=${SLURM_TMPDIR}

# activate environment
source ~/.bashrc
conda activate adv_robust
export PYTHONPATH=/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO:$PYTHONPATH

# ---------- 路径与参数 ---------- #
MODEL_ROOT="/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO/2D_NS_FNO2d_recurrent/external_forcing_patterns/saved_models"
DATASET_ROOT="/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO/2D_NS_FNO2d_recurrent/external_forcing_patterns/datasets/all_patterns"
OUT_DIR="/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO/2D_NS_FNO2d_recurrent/model_explainability/other_forcing_patterns/infer_results"
BATCH_SIZE=20
MAX_KEEP=4   # 生成 keep=0..4 共 5 个变体

echo "[ARGS] MODEL_ROOT=${MODEL_ROOT}"
echo "[ARGS] DATASET_ROOT=${DATASET_ROOT}"
echo "[ARGS] OUT_DIR=${OUT_DIR}"
echo "[ARGS] BATCH_SIZE=${BATCH_SIZE}"
echo "[ARGS] MAX_KEEP=${MAX_KEEP}"

# ---------- 运行（一次调用，内部循环六个 pattern） ---------- #
srun -u python ./2D_NS_FNO2d_recurrent/model_explainability/other_forcing_patterns/model_shortened_infer.py \
  --model_root "${MODEL_ROOT}" \
  --dataset_root "${DATASET_ROOT}" \
  --out_dir "${OUT_DIR}" \
  --batch_size ${BATCH_SIZE} \
  --max_keep_layers ${MAX_KEEP}
