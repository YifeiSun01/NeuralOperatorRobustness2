#!/bin/bash
#SBATCH --job-name=runner1d_test_nu0.01     
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=2             # CPU
#SBATCH --mem=20G                     
#SBATCH --time=18:00:00               # max running time hr:min:sec
#SBATCH --mail-type=ALL         
#SBATCH --mail-user=yifeisun@umich.edu
#SBATCH --output=./1D_Burgers/perturbation_methods/logs_test/%x_%j.out  # 自动生成日志文件

hostname;date;pwd
export XDG_RUNTIME_DIR=${SLURM_TMPDIR}

#activate enviorment
source ~/.bashrc
conda activate adv_robust
export PYTHONPATH=/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO:$PYTHONPATH
# module load cuda
# nvidia-smi
# nvcc --version

base_path="/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO"
nu_val="0.01"
# for N in 200 2000 20000; do
for N in 200000; do
  in_path="${base_path}/1D_Burgers/perturbation_methods/x_y_dict/1D/Burgers/inputs_dim1d_nx1024_N${N}_solver=exponax_kernel=gaussian_correlation_length0.03_bcperiodic_nu${nu_val}_t1.0_seed1000045.pt"
  out_path="${base_path}/1D_Burgers/perturbation_methods/x_y_dict/1D/Burgers/outputs_dim1d_nx1024_N${N}_solver=exponax_kernel=gaussian_correlation_length0.03_bcperiodic_nu${nu_val}_t1.0_seed1000045.pt"

  # 可选：存在性检查
  [[ -f "$in_path"  ]] || { echo "Missing: $in_path";  continue; }
  [[ -f "$out_path" ]] || { echo "Missing: $out_path"; continue; }

  echo "=== Running N=${N} ==="
  python -u "./1D_Burgers/perturbation_methods/runner1d_test_nu${nu_val}.py" \
    --workdir="./1D_Burgers/perturbation_results/1D" \
    --config="./1D_Burgers/perturbation_methods/configs/1D/burgers_test.py" \
    --config.num_records=1 \
    --config.dict_input_path="$in_path" \
    --config.dict_output_path="$out_path" \
    --config.model_path="${base_path}/1D_Burgers/saved_models/1D/modes16_width64_epochs500/pos/unnormalized/burgers_1d_FNO_model_trainedby_dim1d_nx1024_N1500_solver=exponax_kernel=gaussian_correlation_length0.03_bcperiodic_nu${nu_val}_t1.0_seed45.pth" \
    2>&1
done