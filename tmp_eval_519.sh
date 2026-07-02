#!/bin/bash
# 运行第一个评估
python examples/SimplerEnv/eval_and_summarize.py \
  --ckpt-path results/Checkpoints/decoder_off_codi_new_bridge_train_cot_qwen3vl4b_bs16_fields_3 \
  --gpus 8 \
  --min-step 20000

echo "第一个评估完成，等待 15 分钟后开始第二个..."
sleep 900   # 15 分钟 = 900 秒

# 运行第二个评估
python examples/SimplerEnv/eval_and_summarize.py \
  --ckpt-path results/Checkpoints/decoder_off_codi_new_bridge_train_cot_qwen3vl4b_bs16_fields_3 \
  --gpus 8 \
  --min-step 20000

echo "第2个评估完成，等待 15 分钟后开始第3个..."
sleep 900   # 15 分钟 = 900 秒

# 运行第二个评估
python examples/SimplerEnv/eval_and_summarize.py \
  --ckpt-path results/Checkpoints/full_loss_codi_new_bridge_train_cot_qwen3vl4b_bs8_fields_3 \
  --gpus 8 \
  --min-step 20000

echo "第3个评估完成，等待 15 分钟后开始第4个..."
sleep 900   # 15 分钟 = 900 秒

# 运行第二个评估
python examples/SimplerEnv/eval_and_summarize.py \
  --ckpt-path results/Checkpoints/full_loss_codi_new_bridge_train_cot_qwen3vl4b_bs8_fields_3 \
  --gpus 8 \
  --min-step 20000

echo "全部完成。"