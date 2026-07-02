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

###########################################################################################
# Default checkpoint path. You can override it from the command line:
# bash examples/SimplerEnv/train_files/run_bridge_small_fit_eval_ckpt.sh /abs/path/to/pytorch_model.pt
default_ckpt_path=/inspire/hdd/global_user/daizihao-CZXS25110035/zhdai/starVLA/results/Checkpoints/implicit_cot_small_fit_full_loss_bridge_train_cot_qwen3vl4b/checkpoints/steps_15000_pytorch_model.pt
###########################################################################################

manifest_path=./starVLA/config/small_fit/bridge_train_cot_first100_episodes.json
oxe_data_root=/inspire/hdd/global_user/daizihao-CZXS25110035/zhdai/datasets
data_mix=bridge_train_cot
base_vlm=./playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action
eval_batch_size=8
eval_tag=manual

ckpt_path="${1:-${default_ckpt_path}}"

if [ ! -f "${ckpt_path}" ]; then
  echo "Checkpoint not found: ${ckpt_path}" >&2
  exit 1
fi

run_id="$(basename "$(dirname "$(dirname "${ckpt_path}")")")"
run_root_dir="$(dirname "$(dirname "$(dirname "${ckpt_path}")")")"
output_dir="${run_root_dir}/${run_id}"

if [[ "${run_id}" == *teacher_only* ]]; then
  config_yaml=./starVLA/config/training/starvla_bridge_teacher_only_cot.yaml
  predict_mode=teacher
else
  config_yaml=./starVLA/config/training/starvla_bridge_latent_cot_small_fit.yaml
  predict_mode=student
fi

mkdir -p "${output_dir}"

python starVLA/training/eval_cot_small_fit.py \
  --config_yaml "${config_yaml}" \
  --checkpoint_path "${ckpt_path}" \
  --output_path "${output_dir}/small_fit_eval_${eval_tag}.json" \
  --summary_jsonl "${output_dir}/small_fit_eval_${eval_tag}_results.jsonl" \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}" \
  --datasets.vla_data.data_root_dir "${oxe_data_root}" \
  --datasets.vla_data.data_mix "${data_mix}" \
  --framework.qwenvl.base_vlm "${base_vlm}" \
  --small_fit.episode_manifest_path "${manifest_path}" \
  --small_fit.predict_mode "${predict_mode}" \
  --small_fit.eval_batch_size "${eval_batch_size}"
