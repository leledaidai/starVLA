
# Allow callers to override NCCL networking, otherwise auto-detect interfaces
# that actually exist on the current machine.
if [[ -z "${NCCL_SOCKET_IFNAME:-}" ]]; then
  if [[ -d /sys/class/net/bond0 ]]; then
    export NCCL_SOCKET_IFNAME=bond0
  elif [[ -d /sys/class/net/eth0 ]]; then
    export NCCL_SOCKET_IFNAME=eth0
  else
    auto_nccl_ifname=$(ls /sys/class/net | grep -E '^(bond|en|eth)' | head -n 1)
    if [[ -n "${auto_nccl_ifname}" ]]; then
      export NCCL_SOCKET_IFNAME="${auto_nccl_ifname}"
    fi
  fi
fi

if [[ -z "${NCCL_IB_HCA:-}" ]] && [[ -d /sys/class/infiniband ]]; then
  auto_nccl_hca=$(ls /sys/class/infiniband | paste -sd, -)
  if [[ -n "${auto_nccl_hca}" ]]; then
    export NCCL_IB_HCA="${auto_nccl_hca}"
  fi
fi

# used for check save when communication
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000  # timeout set to 1 hour (unit: seconds)

export WANDB_MODE=offline

echo "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-<unset>}"
echo "NCCL_IB_HCA=${NCCL_IB_HCA:-<unset>}"

###########################################################################################
# === Please modify the following paths according to your environment ===

Framework_name=QwenGR00T

freeze_module_list=''
base_vlm=./playground/Pretrained_models/Qwen2.5-VL-3B-Instruct-Action

config_yaml=./examples/SimplerEnv/train_files/starvla_train_bridge_groot.yaml

oxe_data_root=/inspire/hdd/global_user/gongjingjing-25039/zhdai/datasets
data_mix=bridge
run_root_dir=./results/Checkpoints
run_id=${Framework_name}_${data_mix}_47
# === End of environment variable configuration ===
###########################################################################################


# export WANDB_MODE=disabled

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/

source /root/miniconda3/etc/profile.d/conda.sh
conda activate starVLA

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
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 100000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 1000 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_test \
  --wandb_entity leledaidai-harbin-institute-of-technology \
  # --is_debug True



##### Multi-Server Multi-GPU training script #####
  # accelerate launch \
  #   --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  #   --main_process_ip $MASTER_ADDR \
  #   --main_process_port $MASTER_PORT \
  #   --machine_rank $SLURM_PROCID \
  #   --num_machines $SLURM_NNODES \
  #   --num_processes=${TOTAL_GPUS} \
  #   starVLA/training/train_starvla.py \
  #   --config_yaml ${config_yaml} \
  #   --framework.name ${Framework_name} \
  #   --framework.qwenvl.base_vlm ${base_vlm} \
  #   --run_root_dir ${run_root_dir} \
  #   --run_id ${run_id} \
  #   --wandb_project your_project \
  #   --wandb_entity your_name
##### Multi-Server Multi-GPU training script #####
