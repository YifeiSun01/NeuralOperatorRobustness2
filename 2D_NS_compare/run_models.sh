#!/bin/bash
#SBATCH --job-name=run_models
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G                       
#SBATCH --time=50:00:00         
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yifeisun@umich.edu
#SBATCH --output=./2D_NS_compare/logs/%x_%j.out
#SBATCH --error=./2D_NS_compare/logs/%x_%j.err

hostname;date;pwd
export XDG_RUNTIME_DIR=${SLURM_TMPDIR}

#activate enviorment
source ~/.bashrc
conda activate adv_robust
export PYTHONPATH=/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO:$PYTHONPATH

python -u ./2D_NS_compare/run_models.py \
  --train "./2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_all_frames.pt" \
  --test "./2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_test_all_frames.pt" \
  --generalizability "./2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/generalizability" \
  --expanded "./2D_NS_FNO2d_recurrent/datasets/expanded_exponax_datasets/t20/N=1150" \
  --max-samples 1150 \
  --out-csv "/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO/2D_NS_compare/results_batch_eval_2.csv"
