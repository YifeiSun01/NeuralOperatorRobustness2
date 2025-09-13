#!/bin/bash
#SBATCH --job-name=data_expand
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=88:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yifeisun@umich.edu
#SBATCH --output=./1D_Burgers/dataset_expansion/logs_test/%x_%j.out
#SBATCH --error=./1D_Burgers/dataset_expansion/logs_test/%x_%j.err

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
dt="0.001"
norm="2"
steps="20"

# epsilon 和 alpha 的排列组合
for eps in 10 20 50; do
  for alpha in 0.01 0.05 0.1 0.5 1; do
    echo "=== RUN: norm=${norm}, epsilon=${eps}, steps=${steps}, alpha=${alpha} ==="
    python "$base_path/1D_Burgers/dataset_expansion/data_expand.py" \
      --dataset_pt "$dataset_pt" \
      --model_pth "$model_pth" \
      --solver exponax \
      --nu "$nu_val" \
      --t_final "$t_final" \
      --dt "$dt" \
      --out_root "$out_root" \
      --inputs "${norm},${eps},${steps},${alpha}"
    echo "=== DONE: norm=${norm}, epsilon=${eps}, steps=${steps}, alpha=${alpha} ==="
  done
done