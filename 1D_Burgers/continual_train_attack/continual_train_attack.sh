#!/bin/bash
#SBATCH --job-name=continual_train_attack
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G                       
#SBATCH --time=120:00:00         
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yifeisun@umich.edu
#SBATCH --output=./1D_Burgers/continual_train_attack/logs/%x_%j.out
#SBATCH --error=./1D_Burgers/continual_train_attack/logs/%x_%j.err

hostname;date;pwd
export XDG_RUNTIME_DIR=${SLURM_TMPDIR}

#activate enviorment
source ~/.bashrc
conda activate adv_robust
export PYTHONPATH=/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO:$PYTHONPATH

python -u ./1D_Burgers/continual_train_attack/continual_train_attack.py \
  --dataset_pt ./1D_Burgers/datasets/1D/Burgers/pos/dim1d_nx1024_N1500_solver=exponax_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.0005_t1.0_seed45.pt \
  --rounds 3 --attack_ratio 0.3 \
  --inputs "2,10,20,0.5" \
  --solver exponax --nu 0.0005 --t_final 1.0 --dt 0.001 \
  --modes 64 --width 64 --epochs 500 --batch_size 20 --lr 0.001