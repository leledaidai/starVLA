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

export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0

export NCCL_DEBUG=WARN
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1800

Framework_name=QwenGR00TImplicitCoT
freeze_module_list=''
base_vlm=./playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action
config_yaml=./starVLA/config/training/starvla_bridge_latent_cot_small_fit.yaml
manifest_path=./starVLA/config/small_fit/bridge_train_cot_first100_episodes.json
oxe_data_root=/inspire/hdd/global_user/gongjingjing-25039/zhdai/datasets
data_mix=bridge_train_cot

num_processes=8
per_device_batch_size=8
max_train_steps=60000
save_interval=5000
logging_frequency=100
eval_interval=1000

run_root_dir=./results/Checkpoints
run_id=implicit_cot_small_fit_full_loss_${data_mix}_qwen3vl4b
wandb_project=starVLA_implicit_cot_small_fit_full_loss
wandb_entity=leledaidai-harbin-institute-of-technology

export WANDB_MODE=offline

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp "$0" "${output_dir}/"

python starVLA/training/build_bridge_small_fit_manifest.py \
  --config_yaml ${config_yaml} \
  --output_path ${manifest_path} \
  --num_episodes 100 \
  --datasets.vla_data.data_root_dir ${oxe_data_root} \
  --datasets.vla_data.data_mix ${data_mix}

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes ${num_processes} \
  starVLA/training/cot_trainer_small_fit.py \
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
  --wandb_entity ${wandb_entity} \
  --small_fit.episode_manifest_path ${manifest_path}

final_ckpt=${output_dir}/final_model/pytorch_model.pt
echo "Waiting for final checkpoint: ${final_ckpt}"
while [ ! -s "${final_ckpt}" ]; do
  sleep 10
done

last_size=0
stable_rounds=0
while [ "${stable_rounds}" -lt 2 ]; do
  current_size=$(stat -c%s "${final_ckpt}")
  if [ "${current_size}" -eq "${last_size}" ]; then
    stable_rounds=$((stable_rounds + 1))
  else
    stable_rounds=0
    last_size=${current_size}
  fi
  sleep 10
done
echo "Final checkpoint is ready: ${final_ckpt}"

python starVLA/training/eval_cot_small_fit.py \
  --config_yaml ${config_yaml} \
  --checkpoint_path ${final_ckpt} \
  --output_path ${output_dir}/small_fit_eval_full_loss.json \
  --summary_jsonl ${output_dir}/small_fit_eval_full_loss_results.jsonl \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --datasets.vla_data.data_root_dir ${oxe_data_root} \
  --datasets.vla_data.data_mix ${data_mix} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --small_fit.episode_manifest_path ${manifest_path}
