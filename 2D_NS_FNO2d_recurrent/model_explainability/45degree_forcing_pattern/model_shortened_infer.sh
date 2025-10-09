#!/bin/bash
#SBATCH --job-name=model_shortened_infer
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G                       
#SBATCH --time=20:00:00         
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yifeisun@umich.edu
#SBATCH --output=./2D_NS_FNO2d_recurrent/model_explainability/logs/%x_%j.out
#SBATCH --error=./2D_NS_FNO2d_recurrent/model_explainability/logs/%x_%j.err

hostname;date;pwd
export XDG_RUNTIME_DIR=${SLURM_TMPDIR}

#activate enviorment
source ~/.bashrc
conda activate adv_robust
export PYTHONPATH=/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO:$PYTHONPATH

# ---------- 路径与参数（你可以在这里修改） ----------
MODEL_ROOT="/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO/2D_NS_FNO2d_recurrent/saved_models/2D"
DATASET_PATH="/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO/2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_all_frames.pt"
OUT_DIR="/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO/2D_NS_FNO2d_recurrent/model_explainability/infer_results"   # <== 你指定的保存目录
BATCH_SIZE=20
MAX_KEEP=4   # 保留前 0..4 层，得到 5 个变体

# 打印本次运行的关键参数
echo "[ARGS] MODEL_ROOT=${MODEL_ROOT}"
echo "[ARGS] DATASET_PATH=${DATASET_PATH}"
echo "[ARGS] OUT_DIR=${OUT_DIR}"
echo "[ARGS] BATCH_SIZE=${BATCH_SIZE}"
echo "[ARGS] MAX_KEEP=${MAX_KEEP}"

# ---------- 运行 ----------
# 建议用 srun 启动 job step（便于资源绑定、日志更一致）
srun -u python ./2D_NS_FNO2d_recurrent/model_explainability/model_shortened_infer.py \
  --model_root "${MODEL_ROOT}" \
  --dataset_path "${DATASET_PATH}" \
  --out_dir "${OUT_DIR}" \
  --batch_size ${BATCH_SIZE} \
  --max_keep_layers ${MAX_KEEP}