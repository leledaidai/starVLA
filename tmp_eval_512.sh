#!/bin/bash
# 运行第一个评估
python examples/SimplerEnv/eval_and_summarize.py \
  --ckpt-path results/Checkpoints/latent_cot_distill_off_test_bridge_train_cot_qwen3vl4b_test_bs16_new_55 \
  --gpus 8 \
  --min-step 20000

echo "一个评测已完成"
sleep 900   # 15 分钟 = 900 秒

# 运行第二个评估
python examples/SimplerEnv/eval_and_summarize.py \
  --ckpt-path results/Checkpoints/latent_cot_distill_off_test_bridge_train_cot_qwen3vl4b_test_bs16_new_55_decoder_freeze \
  --gpus 8 \
  --min-step 20000

echo "一个评测已完成"
sleep 900   # 15 分钟 = 900 秒

# 运行第二个评估
python examples/SimplerEnv/eval_and_summarize.py \
  --ckpt-path results/Checkpoints/latent_cot_distill_off_test_bridge_train_cot_qwen3vl4b_test_bs16_new_55_decoder_freeze_qwen3vl4b_text \
  --gpus 8 \
  --min-step 20000


echo "一个评测已完成"
sleep 900   # 15 分钟 = 900 秒

# 运行第二个评估
python examples/SimplerEnv/eval_and_summarize.py \
  --ckpt-path results/Checkpoints/latent_cot_distill_off_test_bridge_train_cot_qwen3vl4b_test_bs24_new_55_decoder_freeze \
  --gpus 8 \
  --min-step 20000


echo "一个评测已完成"
sleep 900   # 15 分钟 = 900 秒

# 运行第二个评估
python examples/SimplerEnv/eval_and_summarize.py \
  --ckpt-path results/Checkpoints/latent_cot_full_loss_bridge_train_cot_qwen3vl4b_new_weight_small \
  --gpus 8 \
  --min-step 20000

echo "一个评测已完成"
sleep 900   # 15 分钟 = 900 秒

# 运行第二个评估
python examples/SimplerEnv/eval_and_summarize.py \
  --ckpt-path results/Checkpoints/latent_cot_distill_off_bridge_train_cot_qwen3vl4b_resume_200k \
  --gpus 8 \
  --min-step 20000

echo "全部完成。"
