# 进度记录

## 2026-05-07
- 排查 `main_run_bridge_latent_cot_distill_off_test_bs16_weight_small.sh` 的 NaN/Inf 日志。
- 确认脚本实际使用 `starVLA/config/training/train_latent_vla/starvla_bridge_latent_cot_distill_off_test_weight_small.yaml`，student action 和 decoder loss 均开启，distill loss 关闭。
- 读取保存的 `config.full.yaml`，确认本次 run 为 per-device batch size 16、8 卡、BF16/ZeRO-2、`latent_forward_use_cache: true`、`decoder_loss_weight: 0.1`，且未保存任何 checkpoint。
- 检查 `QwenGR00TImplicitCoT.forward()`、`_student_forward()`、`_forward_latent_cached()`，判断 NaN 已在 student forward hidden states 产生，不是最终 loss 单独异常。
- 发现单机 `ds_config.yaml` 在 BF16 训练下使用 `communication_data_type: fp16`，已改为 `fp32`。
- 在 `cot_trainer.py` 增加 optimizer.step 前非有限梯度保护，避免有限 loss 后的非有限梯度污染参数。
- 运行 `python -m py_compile starVLA/training/cot_trainer.py` 通过。
- 用户重跑后发现 `accelerator.clip_grad_norm_()` 在 DeepSpeed 路径返回 `None`，导致 `float(None)` 报错。已修复为 `grad_norm is None` 时跳过 norm 返回值判断，继续依赖后续逐参数梯度检查。
- 再次运行 `python -m py_compile starVLA/training/cot_trainer.py` 通过。
- 根据用户希望训练和通信都使用 BF16 的要求，将 `starVLA/config/deepseeds/ds_config.yaml` 的 `communication_data_type` 从 `fp32` 调整为 `bf16`，保留 optimizer.step 前非有限梯度保护。

## 2026-05-06
- 读取用户指定的 `compare_wandb_prompt.txt`，确认任务为比较 baseline、fields_3、fields_5、fields_8 四组 wandb 日志和配置。
- 检查四个 checkpoint 目录，确认均有 `config.full.yaml`、运行脚本和 `wandb/.../files/output.log`。
- 重新解析四组 output.log 的 step-level 指标，得到 `student_action_loss` 末段排序 fields_3 > fields_5 > fields_8，且三者都显著高于 baseline。
- 对照 `QwenGR00TImplicitCoT.forward`、collator 和 `FlowmatchingActionHead`，确认 fields 数量会同时改变 latent slot 数、decoder/slot distill 目标和可见 CoT 语义内容，不只是改变文本长度。
- 更新 `findings.md` 和 `task_plan.md`，记录本次 fields_3/5/8 分析结论。

## 2026-05-05
- 已确认四个 checkpoint 目录存在 wandb 离线 run、`config.full.yaml`、`config.yaml`、`summary.jsonl` 和运行脚本。
- 已创建规划文件，开始本地日志与配置分析。
- 已解析四个 `output.log` 中每 100 step 的训练指标，获得 action loss、mse、aux loss 的末段均值/最小值/最终值。
- 已检查 latent loss 组合逻辑：`QwenGR00TImplicitCoT.forward` 中多个 loss 直接加权求和，teacher/student action loss 共用同一个 action head。
- 已对比配置，记录 batch size、warmup、action head `output_dim`、loss 开关差异。
