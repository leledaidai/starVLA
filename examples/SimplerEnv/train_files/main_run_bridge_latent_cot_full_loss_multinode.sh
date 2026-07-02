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

# NCCL settings for multi-node communication
# --- Cluster-specific: adjust to your hardware ---
# Socket interface for initial connection. Auto-detect by default; set explicitly
# if your cluster uses a bonded/特定网卡 (e.g. bond0, eth0).
if [ -n "${NCCL_SOCKET_IFNAME_OVERRIDE}" ]; then
  export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME_OVERRIDE}"
fi
# IB HCAs for RDMA. "auto" if empty; otherwise comma-separated list (e.g. mlx5_2,mlx5_3).
if [ -n "${NCCL_IB_HCA_OVERRIDE}" ]; then
  export NCCL_IB_HCA="${NCCL_IB_HCA_OVERRIDE}"
fi
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000
export NCCL_NVLS_ENABLE=0
export TOKENIZERS_PARALLELISM=false

# Multi-node distributed settings (from server built-in env vars)
GPUS_PER_NODE=${PET_NPROC_PER_NODE:-8}
NUM_NODES=${PET_NNODES:-2}
NODE_RANK=${PET_NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
TOTAL_GPUS=$((GPUS_PER_NODE * NUM_NODES))

###########################################################################################
# === Please modify the following paths according to your environment ===
Framework_name=QwenGR00TImplicitCoT
freeze_module_list=''
base_vlm=./playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action
config_yaml=./starVLA/config/training/train_latent_vla/starvla_bridge_latent_cot_full_loss.yaml
oxe_data_root=/inspire/hdd/global_user/daizihao-CZXS25110035/zhdai/datasets
data_mix=bridge_train_cot

per_device_batch_size=8
max_train_steps=100000
save_interval=5000
logging_frequency=100
eval_interval=1000

run_root_dir=./results/Checkpoints
run_id=latent_cot_full_loss_${data_mix}_qwen3vl4b_16gpu
wandb_project=latent_vla_51
wandb_entity=leledaidai-harbin-institute-of-technology
# === End of environment variable configuration ===
###########################################################################################

export WANDB_MODE=offline

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp "$0" "${output_dir}/"

echo "============================================"
echo "  Multi-Node Training Configuration"
echo "  Master:   ${MASTER_ADDR}:${MASTER_PORT}"
echo "  Nodes:    ${NUM_NODES}"
echo "  GPUs/node: ${GPUS_PER_NODE}"
echo "  Total GPUs: ${TOTAL_GPUS}"
echo "  Node rank: ${NODE_RANK}"
echo "  Global batch size: $((TOTAL_GPUS * per_device_batch_size))"
echo "============================================"

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2_multinode.yaml \
  --main_process_ip ${MASTER_ADDR} \
  --main_process_port ${MASTER_PORT} \
  --machine_rank ${NODE_RANK} \
  --num_machines ${NUM_NODES} \
  --num_processes ${TOTAL_GPUS} \
  starVLA/training/cot_trainer.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.data_root_dir ${oxe_data_root} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size ${per_device_batch_size} \
  --trainer.freeze_modules "${freeze_module_list}" \
  --trainer.max_train_steps ${max_train_steps} \
  --trainer.save_interval ${save_interval} \
  --trainer.logging_frequency ${logging_frequency} \
  --trainer.eval_interval ${eval_interval} \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project ${wandb_project} \
  --wandb_entity ${wandb_entity}
