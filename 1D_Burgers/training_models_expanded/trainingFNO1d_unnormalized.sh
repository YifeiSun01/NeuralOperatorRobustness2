#!/bin/bash
#SBATCH --job-name=trainingFNO1d_unnormalized     
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=2             # CPU
#SBATCH --mem=2G                     
#SBATCH --time=3:00:00               # max running time hr:min:sec
#SBATCH --mail-type=ALL         
#SBATCH --mail-user=yifeisun@umich.edu
#SBATCH --output=./1D_Burgers/training_models_expanded/logs/%x_%j.out  # 自动生成日志文件
#SBATCH --error=./1D_Burgers/training_models_expanded/logs/%x_%j.err   # 错误日志

hostname;date;pwd
export XDG_RUNTIME_DIR=${SLURM_TMPDIR}

#activate enviorment
source ~/.bashrc
conda activate adv_robust
export PYTHONPATH=/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO:$PYTHONPATH

python -u ./1D_Burgers/training_models_expanded/trainingFNO1d_unnormalized.py \
  --pt_original "/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO/1D_Burgers/datasets/1D/Burgers/pos/dim1d_nx1024_N1500_solver=exponax_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.0005_t1.0_seed45.pt" \
  --attacks_root "/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO/1D_Burgers/datasets/1D/Burgers/expanded_pos/t1" \
  --attack_subdir ALL \
  --mode all \
  --epochs 500 --ntrain 1000 --ntest 100 --batch_size 20 --lr 0.001
