#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then
  source "/root/miniconda3/etc/profile.d/conda.sh"
else
  echo "Failed to locate conda.sh for activating starVLA." >&2
  exit 1
fi
conda activate starVLA

export NCCL_NVLS_ENABLE=0
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0

export NCCL_DEBUG=WARN
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1800

###########################################################################################
# === Please modify the following paths according to your environment ===
Framework_name=QwenGR00T
freeze_module_list=''
base_vlm=./playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action
config_yaml=./playground/Pretrained_models/Qwen3VL-GR00T-Bridge-RT-1/config.yaml
oxe_data_root=/inspire/hdd/global_user/gongjingjing-25039/zhdai/datasets
data_mix=bridge_train_cot
run_root_dir=./results/Checkpoints
run_id=428_${data_mix}_qwen3GR00T_include_empty_prompt
# === End of environment variable configuration ===
###########################################################################################


export WANDB_MODE=offline

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/


accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.data_root_dir ${oxe_data_root}\
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 16 \
  --datasets.vla_data.skip_empty_language false \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 100000 \
  --trainer.save_interval 5000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 1000 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_428 \
  --wandb_entity leledaidai-harbin-institute-of-technology \
  # --is_debug True
