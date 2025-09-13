#!/bin/bash
#SBATCH --job-name=continual_train_attack_2
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G                       
#SBATCH --time=120:00:00         
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yifeisun@umich.edu
#SBATCH --output=./2D_NS_FNO2d_recurrent/continual_train_attack/logs/%x_%j.out
#SBATCH --error=./2D_NS_FNO2d_recurrent/continual_train_attack/logs/%x_%j.err

hostname;date;pwd
export XDG_RUNTIME_DIR=${SLURM_TMPDIR}

#activate enviorment
source ~/.bashrc
conda activate adv_robust
export PYTHONPATH=/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO:$PYTHONPATH

python -u ./2D_NS_FNO2d_recurrent/continual_train_attack/continual_train_attack_2.py \
  --train_pt ./2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_all_frames.pt \
  --test_pt  ./2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_test_all_frames.pt  \
  --rounds 6 \
  --epochs 500 \
  --batch_size 20 \
  --modes1 96 --modes2 96 --width 80 --Tin 10 --Tout 10 --step 1 \
  --attack_split train \
  --steps 10 \
  --attack_alpha_list 2.5,5,10,20,50 \
  --attack_epsilon_list 8,15,35,75,105 \
  --attack_ratio 0.25 \
  --mix_ratio 0.25 \
  --seed 1234