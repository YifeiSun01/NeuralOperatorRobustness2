#!/bin/bash
#SBATCH --job-name=runner1d_test_nu0.01     
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=2             # CPU
#SBATCH --mem=20G                     
#SBATCH --time=18:00:00               # max running time hr:min:sec
#SBATCH --mail-type=ALL         
#SBATCH --mail-user=yifeisun@umich.edu
#SBATCH --output=./perturbation_methods/logs_test/%x_%j.out  # 自动生成日志文件

hostname;date;pwd
export XDG_RUNTIME_DIR=${SLURM_TMPDIR}

#activate enviorment
source ~/.bashrc
conda activate adv_robust
export PYTHONPATH=/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO:$PYTHONPATH
# module load cuda
# nvidia-smi
# nvcc --version

base_path="/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO/1D_Burgers"
nu_val="0.01"

python -u "./perturbation_methods/runner1d_test_nu${nu_val}.py" \
  --workdir="./perturbation_results/1D" \
  --config="./perturbation_methods/configs/1D/burgers_test.py" \
  --config.num_records=1 \
  --config.dict_input_path="${base_path}/perturbation_methods/x_y_dict/1D/Burgers/inputs_dim1d_nx1024_N20_solver=exponax_kernel=gaussian_correlation_length0.03_bcperiodic_nu${nu_val}_t1.0_seed1000045.pt" \
  --config.dict_output_path="${base_path}/perturbation_methods/x_y_dict/1D/Burgers/outputs_dim1d_nx1024_N20_solver=exponax_kernel=gaussian_correlation_length0.03_bcperiodic_nu${nu_val}_t1.0_seed1000045.pt" \
  --config.model_path="${base_path}/saved_models/1D/modes16_width64_epochs500/pos/burgers_1d_FNO_model_trainedby_dim1d_nx1024_N1500_solver=exponax_kernel=gaussian_correlation_length0.03_bcperiodic_nu${nu_val}_t1.0_seed45.pth" \
  2>&1
