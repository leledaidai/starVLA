

cd /inspire/hdd/global_user/gongjingjing-25039/zhdai/starVLA/
export PYTHONPATH=$(pwd):${PYTHONPATH}

#### get parameters #####
if [ -n "$1" ]; then
  your_ckpt="$1"
else
  your_ckpt=/inspire/hdd/global_user/gongjingjing-25039/zhdai/starVLA/playground/Pretrained_models/Qwen3VL-GR00T-Bridge-RT-1/checkpoints/steps_20000_pytorch_model.pt
fi

port=${2:-6678}
gpu_id=${3:-0}
# export DEBUG=true
export star_vla_python=/root/miniconda3/envs/starVLA/bin/python

#### build output directory #####
ckpt_dir=$(dirname "${your_ckpt}")
ckpt_base=$(basename "${your_ckpt}")
ckpt_name="${ckpt_base%.*}"
output_server_dir="${ckpt_dir}/output_server"
mkdir -p "${output_server_dir}"
log_file="${output_server_dir}/${ckpt_name}_policy_server_${port}.log"


#### run server #####
CUDA_VISIBLE_DEVICES=${gpu_id} ${star_vla_python} deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16 \
    2>&1 | tee "${log_file}"