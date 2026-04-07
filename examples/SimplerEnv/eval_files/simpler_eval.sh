#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <model_checkpoint_path> [port] [gpu_id]"
  echo "Example: $0 ./playground/Pretrained_models/Qwen-GR00T-Bridge-RT-1/checkpoints/steps_20000_pytorch_model.pt 6678 0"
  exit 1
fi

MODEL_PATH="$1"
PORT=${2:-6678}
GPU_ID=${3:-0}

echo "=========================================="
echo "SimplerEnv Evaluation"
echo "=========================================="
echo "Model: ${MODEL_PATH}"
echo "Port: ${PORT}"
echo "GPU: ${GPU_ID}"
echo "=========================================="

cd /inspire/hdd/global_user/gongjingjing-25039/zhdai/starVLA/

# Generate result directory name from model path
# Extract model_name/checkpoints/ckpt_file and convert to model_name_checkpoints_ckpt_file
MODEL_DIR=$(dirname "${MODEL_PATH}")
MODEL_NAME=$(basename "${MODEL_DIR}")
PARENT_DIR=$(basename $(dirname "${MODEL_DIR}"))
CKPT_FILE=$(basename "${MODEL_PATH}")
RESULT_NAME="${PARENT_DIR}_${MODEL_NAME}_${CKPT_FILE}"

echo "Step 1: Starting policy server..."
bash examples/SimplerEnv/eval_files/run_policy_server.sh "${MODEL_PATH}" "${PORT}" "${GPU_ID}" &
SERVER_PID=$!

sleep 10

echo "Step 2: Running evaluation..."
bash examples/SimplerEnv/eval_files/start_simpler_env.sh "${MODEL_PATH}" "${PORT}"

echo "Step 3: Waiting for all evaluation tasks to complete..."
# Wait for all Python evaluation processes to finish
while pgrep -f "start_simpler_env.py" > /dev/null; do
    sleep 20
done

echo "Step 4: Stopping policy server..."
kill ${SERVER_PID} 2>/dev/null

echo "Step 5: Calculating metrics..."
RESULT_DIR="/inspire/hdd/global_user/gongjingjing-25039/zhdai/starVLA/results/SIMPLER_EVAL/${RESULT_NAME}"
/root/miniconda3/envs/starVLA/bin/python examples/SimplerEnv/eval_files/calc_simpler_metrics.py --result-dir "${RESULT_DIR}"

echo "✅ Evaluation completed"
