#!/bin/bash
#SBATCH --job-name=data_expand
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G                       
#SBATCH --time=8:00:00         
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yifeisun@umich.edu
#SBATCH --output=./2D_NS_FNO2d_recurrent/external_forcing_patterns/dataset_expansion/logs/%x_%j.out
#SBATCH --error=./2D_NS_FNO2d_recurrent/external_forcing_patterns/dataset_expansion/logs/%x_%j.err

hostname;date;pwd
export XDG_RUNTIME_DIR=${SLURM_TMPDIR}

#activate enviorment
source ~/.bashrc
conda activate adv_robust
export PYTHONPATH=/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO:$PYTHONPATH

ROOT="/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO/2D_NS_FNO2d_recurrent/external_forcing_patterns"
DATA_DIR="${ROOT}/datasets/all_patterns"
MODEL_DIR="${ROOT}/saved_models"
OUT_DIR="${ROOT}/datasets/expanded_datasets"
mkdir -p "${OUT_DIR}"

# 6 种 forcing pattern
# patterns=( ringsL1 ringsLinf sBands none)
patterns=(ringsLinf)
# 同时跑 train 和 test；只想跑 test 可改为：splits=(test)
# splits=(train test)
splits=(train)

# 可以按需调整的 attack 超参
BATCH_SIZE=5
MODE_SPEC=wwwwwwwwww
EPS_MULT=0.002
ALPHA_MULT=0.5
NUM_STEPS=10

for pattern in "${patterns[@]}"; do
  for split in "${splits[@]}"; do
    # train 固定 N=1150；test 固定 N=50
    if [[ "${split}" == "train" ]]; then
      N=1150
    else
      N=50
    fi

    dataset_pt="${DATA_DIR}/${pattern}/dim2d_nx256_N${N}_solver=exponax_nu0.000_t20.0_${split}_forcingPattern${pattern}.pt"
    model_pth="${MODEL_DIR}/${pattern}/modes64_width60_epochs500_Tin10_T10/NS_2d_FNO_trained_on_dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_forcingPattern${pattern}.pth"
    save_to="${OUT_DIR}/${pattern}_attack_N${N}_${split}.pt"

    # 路径存在性检查（缺文件就跳过该组合）
    if [[ ! -f "${dataset_pt}" ]]; then
      echo "[SKIP] dataset missing: ${dataset_pt}"
      continue
    fi
    if [[ ! -f "${model_pth}" ]]; then
      echo "[SKIP] model missing:   ${model_pth}"
      continue
    fi
    mkdir -p "$(dirname "${save_to}")"

    echo "[$(date -Is)] ▶ pattern=${pattern}  split=${split}  N=${N}"
    echo "                ds=${dataset_pt}"
    echo "                mdl=${model_pth}"
    python -u "${ROOT}/dataset_expansion/data_expand.py" \
      --dataset_pt "${dataset_pt}" \
      --model_pth  "${model_pth}" \
      --forcing_pattern "${pattern}" \
      --save_to "${save_to}" \
      --batch_size "${BATCH_SIZE}" \
      --mode_spec "${MODE_SPEC}" \
      --eps_mult "${EPS_MULT}" \
      --alpha_mult "${ALPHA_MULT}" \
      --num_steps "${NUM_STEPS}"
    echo "[$(date -Is)] ✅ done -> ${save_to}"
    echo
  done
done

