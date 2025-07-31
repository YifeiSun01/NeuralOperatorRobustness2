#!/bin/bash
#SBATCH --job-name=read_data
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G                       
#SBATCH --time=10:00:00         
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yifeisun@umich.edu
#SBATCH --output=./saved_models/3D/modes1232_modes36_width40_epochs500_Tin10_T10/logs/%x_%j.out
#SBATCH --error=./saved_models/3D/modes1232_modes36_width40_epochs500_Tin10_T10/logs/%x_%j.err

hostname;date;pwd
export XDG_RUNTIME_DIR=${SLURM_TMPDIR}

#activate enviorment
source ~/.bashrc
conda activate adv_robust
export PYTHONPATH=/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO:$PYTHONPATH

# Get the directory where this script is located
SCRIPT_DIR=$(dirname "$(realpath "$0")")

# Get the parent directory name (the folder containing this script)
PARENT_DIR_NAME=$(basename "$(dirname "$SCRIPT_DIR")")

# Run the Python script using the parent directory name
python -u "$SCRIPT_DIR/read_data.py"


