#!/bin/bash
#SBATCH --job-name=eval_models
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --time=10:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yifeisun@umich.edu
#SBATCH --output=./2D_NS_FNO2d_recurrent/eval_models/logs/%x_%j.out
#SBATCH --error=./2D_NS_FNO2d_recurrent/eval_models/logs/%x_%j.err

hostname; date; pwd
export XDG_RUNTIME_DIR=${SLURM_TMPDIR}

source ~/.bashrc
conda activate adv_robust
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONUNBUFFERED=1

PROJECT_ROOT=/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

EVAL_SCRIPT=$PROJECT_ROOT/2D_NS_FNO2d_recurrent/eval_models/eval_models.py

TRAIN_PT=$PROJECT_ROOT/2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_all_frames.pt
TEST_PT=$PROJECT_ROOT/2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_test_all_frames.pt
# 这是“目录”，所以用 expanded_dir
EXPANDED_DIR=$PROJECT_ROOT/2D_NS_FNO2d_recurrent/datasets/expanded_exponax_datasets/t20/N=1150
GEN_DIR=$PROJECT_ROOT/2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/generalizability

BASELINE_MODEL=$PROJECT_ROOT/2D_NS_FNO2d_recurrent/saved_models/2D/modes64_width60_epochs500_Tin10_T10/NS_2d_FNO_model_trainedby_dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_all_frames.pth
MODEL_DIR=$PROJECT_ROOT/2D_NS_FNO2d_recurrent/saved_models_expanded

OUTPUT_ROOT=$PROJECT_ROOT/2D_NS_FNO2d_recurrent/eval_models

echo "=== RUNTIME INFO ==="
python -V
nvidia-smi || true
echo "PROJECT_ROOT: $PROJECT_ROOT"
echo "EVAL_SCRIPT : $EVAL_SCRIPT"
echo "OUTPUT_ROOT : $OUTPUT_ROOT"
echo "==============="

python -u "$EVAL_SCRIPT" \
  --train_pt "$TRAIN_PT" \
  --test_pt  "$TEST_PT" \
  --expanded_dir "$EXPANDED_DIR" \
  --gen_dir  "$GEN_DIR" \
  --baseline_model "$BASELINE_MODEL" \
  --model_dir "$MODEL_DIR" \
  --output_root "$OUTPUT_ROOT" \
  --modes1 64 --modes2 64 --width 60 \
  --t_in 10 --t_out 10 --step 1 \
  --device auto

