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

# NCCL 稳定性设置
export NCCL_NVLS_ENABLE=0
export TOKENIZERS_PARALLELISM=false

export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1

###########################################################################################
# === Please modify the following paths according to your environment ===
Framework_name=QwenGR00TImplicitCoT
freeze_module_list=''
base_vlm=./playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action
# 使用训练时保存的 config，保证架构参数（cross_attention_dim=2560 等）完全一致
config_yaml=./results/Checkpoints/latent_cot_distill_off_bridge_train_cot_qwen3vl4b/config.yaml
oxe_data_root=/inspire/hdd/global_user/gongjingjing-25039/zhdai/datasets
data_mix=bridge_train_cot

# Resume from checkpoint
pretrained_checkpoint=./results/Checkpoints/latent_cot_distill_off_bridge_train_cot_qwen3vl4b/checkpoints/steps_100000_pytorch_model.pt

num_processes=8
per_device_batch_size=8
max_train_steps=200000
save_interval=5000
logging_frequency=100
eval_interval=1000

run_root_dir=./results/Checkpoints
run_id=latent_cot_distill_off_${data_mix}_qwen3vl4b_resume_200k
wandb_project=latent_vla_51
wandb_entity=leledaidai-harbin-institute-of-technology
# === End of environment variable configuration ===
###########################################################################################

export WANDB_MODE=offline

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp "$0" "${output_dir}/"

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes ${num_processes} \
  starVLA/training/cot_trainer.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.data_root_dir ${oxe_data_root} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size ${per_device_batch_size} \
  --trainer.freeze_modules "${freeze_module_list}" \
  --trainer.pretrained_checkpoint ${pretrained_checkpoint} \
  --trainer.max_train_steps ${max_train_steps} \
  --trainer.save_interval ${save_interval} \
  --trainer.logging_frequency ${logging_frequency} \
  --trainer.eval_interval ${eval_interval} \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project ${wandb_project} \
  --wandb_entity ${wandb_entity}
