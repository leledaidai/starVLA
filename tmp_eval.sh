#!/bin/bash
# 运行第一个评估
python examples/SimplerEnv/eval_and_summarize.py \
  --ckpt-path results/Checkpoints/latent_cot_distill_off_test_bridge_train_cot_qwen3vl4b_test_bs24_new_55_decoder_freeze \
  --gpus 8 \
  --min-step 25000

echo "第一个评估完成，等待 15 分钟后开始第二个..."
sleep 900   # 15 分钟 = 900 秒

# 运行第二个评估
python examples/SimplerEnv/eval_and_summarize.py \
  --ckpt-path results/Checkpoints/latent_cot_distill_off_test_bridge_train_cot_qwen3vl4b_test_bs24_new_55_decoder_freeze \
  --gpus 8 \
  --min-step 25000

echo "全部完成。"
