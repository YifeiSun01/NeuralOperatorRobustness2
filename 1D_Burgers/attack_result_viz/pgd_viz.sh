#!/bin/bash
#SBATCH --job-name=pgd_viz     
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=2             # CPU
#SBATCH --mem=2G                     
#SBATCH --time=1:00:00               # max running time hr:min:sec
#SBATCH --mail-type=ALL         
#SBATCH --mail-user=yifeisun@umich.edu
#SBATCH --output=./1D_Burgers/attack_result_viz/logs/%x_%j.out  # 自动生成日志文件

hostname;date;pwd
export XDG_RUNTIME_DIR=${SLURM_TMPDIR}

#activate enviorment
source ~/.bashrc
conda activate adv_robust
export PYTHONPATH=/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO:$PYTHONPATH

python ./1D_Burgers/attack_result_viz/pgd_viz.py \
  --pickle "/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO/1D_Burgers/attack_result_viz/pickle_files/dt0.001_adaptive/gradient_test_exponax_nu0.0005_nsamples1.pkl" \
  --outdir "/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO/1D_Burgers/attack_result_viz/gifs/dt0.001_adaptive" \
  --fps 5

