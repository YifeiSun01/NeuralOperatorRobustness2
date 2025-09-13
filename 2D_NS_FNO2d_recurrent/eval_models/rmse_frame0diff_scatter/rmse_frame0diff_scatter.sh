#!/bin/bash
#SBATCH --job-name=rmse_frame0diff_scatter
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=1G                       
#SBATCH --time=1:00:00         
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yifeisun@umich.edu
#SBATCH --output=./2D_NS_FNO2d_recurrent/eval_models/rmse_frame0diff_scatter/logs/%x_%j.out
#SBATCH --error=./2D_NS_FNO2d_recurrent/eval_models/rmse_frame0diff_scatter/logs/%x_%j.err

hostname;date;pwd
export XDG_RUNTIME_DIR=${SLURM_TMPDIR}

#activate enviorment
source ~/.bashrc
conda activate adv_robust
export PYTHONPATH=/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO:$PYTHONPATH


python ./2D_NS_FNO2d_recurrent/eval_models/rmse_frame0diff_scatter/discover_and_run_scatter.py \
  --perf-csv ./2D_NS_FNO2d_recurrent/eval_models/pivot_rmse_minus_baseline_from_pivot.csv \
  --sim-csv  ./2D_NS_FNO2d_recurrent/datasets/statistics/similarity_spatial_frame0.csv  \
  --plot-script ./2D_NS_FNO2d_recurrent/eval_models/rmse_frame0diff_scatter/rmse_frame0diff_scatter.py \
  --outdir ./2D_NS_FNO2d_recurrent/eval_models/rmse_frame0diff_scatter/plots/scatter \
  --x-metrics all \
  --run

