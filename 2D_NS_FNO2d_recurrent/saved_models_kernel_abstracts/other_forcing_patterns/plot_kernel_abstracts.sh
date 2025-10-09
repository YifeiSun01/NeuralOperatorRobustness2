#!/bin/bash
#SBATCH --job-name=plot_kernel_abstracts
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G                       
#SBATCH --time=10:00:00         
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yifeisun@umich.edu
#SBATCH --output=./2D_NS_FNO2d_recurrent/saved_models_kernel_abstracts/other_forcing_patterns/logs/%x_%j.out
#SBATCH --error=./2D_NS_FNO2d_recurrent/saved_models_kernel_abstracts/other_forcing_patterns/logs/%x_%j.err

hostname; date; pwd
export XDG_RUNTIME_DIR=${SLURM_TMPDIR}

# ========= 参数区（可在此修改，或提交时用 sbatch --export 覆盖）=========
# 输入 *.analysis.pt 所在目录
export INPUT_DIR="/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO/2D_NS_FNO2d_recurrent/saved_models_kernel_abstracts/other_forcing_patterns/kernel_results"
# 输出图片根目录
export OUTPUT_DIR="/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO/2D_NS_FNO2d_recurrent/saved_models_kernel_abstracts/other_forcing_patterns/kernel_plots_val"

# 下面这些参数会传给 plot_kernel_abstracts.py（与上个回合给你的 CLI 定义一致）
export LAYERS="0,1,2,3"
export TOP_SINGVAL_K=4
export DATA_LOG_MODE="none"          # none|log|log1p|signed_log1p
export COLOR_LOG_SCALE=1             # 1=开启 LogNorm, 0=关闭
export COLOR_LOG_EPS="1e-12"         # LogNorm 时的偏移; 设为空字符串表示不加偏移
export TSV_INDICES="0,1,2,3"
export SYMMETRIC=0                   # 1=对称色标[-m,+m]；0=自动[vmin,vmax]
export CMAP="viridis"                # 例如 viridis|magma|cividis，空则用默认
export S2_EPS="1e-12"                # σ1/σ2-1 的分母下限
# =========================================================================

# 环境准备
source ~/.bashrc
conda activate adv_robust
export PYTHONPATH=/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO:$PYTHONPATH

# 线程与数值库：与 --cpus-per-task 对齐（通用 HPC 最佳实践）
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# matplotlib 无交互后端，避免显示相关报错
export MPLBACKEND=Agg

# 将布尔/可选项翻译为命令行参数
COLOR_FLAG=""
if [[ "${COLOR_LOG_SCALE}" == "1" ]]; then
  COLOR_FLAG="--color-log-scale"
fi
SYMMETRIC_FLAG=""
if [[ "${SYMMETRIC}" == "1" ]]; then
  SYMMETRIC_FLAG="--symmetric"
fi
CMAP_FLAG=""
if [[ -n "${CMAP}" ]]; then
  CMAP_FLAG="--cmap ${CMAP}"
fi
EPS_FLAG=""
if [[ -n "${COLOR_LOG_EPS}" ]]; then
  EPS_FLAG="--color-log-eps ${COLOR_LOG_EPS}"
fi

# 组合 A：非 log（原始数据 + 线性色标）
python -u ./2D_NS_FNO2d_recurrent/saved_models_kernel_abstracts/other_forcing_patterns/plot_kernel_abstracts.py \
  --input-dir  "${INPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --recursive \
  --run-name   "dlog=none__clog=off" \
  --layers "${LAYERS}" \
  --top-singval-k ${TOP_SINGVAL_K} \
  --data-log-mode "none" \
  --tsv-indices "${TSV_INDICES}" \
  --s2-eps ${S2_EPS} \
  ${CMAP_FLAG}

# 组合 B：数据级 log1p + 颜色轴对数（LogNorm），并用 eps 兜底
python -u ./2D_NS_FNO2d_recurrent/saved_models_kernel_abstracts/other_forcing_patterns/plot_kernel_abstracts.py \
  --input-dir  "${INPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --recursive \
  --run-name   "dlog=log1p__clog=on__eps=${COLOR_LOG_EPS}" \
  --layers "${LAYERS}" \
  --top-singval-k ${TOP_SINGVAL_K} \
  --data-log-mode "log1p" \
  --color-log-scale \
  --color-log-eps ${COLOR_LOG_EPS} \
  --tsv-indices "${TSV_INDICES}" \
  --s2-eps ${S2_EPS} \
  ${CMAP_FLAG}

python -u ./2D_NS_FNO2d_recurrent/saved_models_kernel_abstracts/other_forcing_patterns/plot_kernel_abstracts.py \
  --input-dir  "${INPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --recursive \
  --run-name   "dlog=log__clog=on__eps=${COLOR_LOG_EPS}" \
  --layers "${LAYERS}" \
  --top-singval-k ${TOP_SINGVAL_K} \
  --data-log-mode "log" \
  --color-log-scale \
  --color-log-eps ${COLOR_LOG_EPS} \
  --tsv-indices "${TSV_INDICES}" \
  --s2-eps ${S2_EPS} \
  ${CMAP_FLAG}