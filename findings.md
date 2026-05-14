# 发现记录

## 2026-05-07：`main_run_bridge_latent_cot_distill_off_test_bs16_weight_small.sh` NaN/Inf

用户日志显示 step 29 起 `student_action_loss`、`decoder_loss`、`total_loss` 为 NaN/Inf，同时 `student.hidden_states` 和 `student.latent_hidden` 已经出现大量 NaN。关键判断：NaN 已经发生在 student forward 输出，不是 action loss 或 decoder loss 公式最后一步才产生。

脚本与配置：
- 脚本使用 `starVLA/config/training/train_latent_vla/starvla_bridge_latent_cot_distill_off_test_weight_small.yaml`。
- 虽然 run 名叫 `distill_off`，配置仍开启 `enable_student_action_loss: true` 和 `enable_decoder_loss: true`，只是 slot/pool distill 关闭，`decoder_loss_weight: 0.1`。
- 实际保存配置 `results/Checkpoints/latent_cot_distill_off_test_bridge_train_cot_qwen3vl4b_test_bs16_new_55_weight_small/config.full.yaml` 显示 per-device batch size 为 16、8 卡、全局 batch 128、warmup 10000、action LR 1e-4、Qwen-VL/base LR 1e-5。
- 本次 run 目录没有 checkpoint，仅有 wandb 离线目录，说明训练在 5000 step 前已经坏掉。

代码链路：
- `QwenGR00TImplicitCoT.forward()` 只有 student 分支参与本次 loss；teacher loss 为 0 是配置关闭导致。
- `_student_forward()` 调用 `_forward_latent()`，再从 `hidden_states` gather 得到 `latent_hidden`。
- 配置中 `latent_forward_use_cache: true`，实际使用 `_forward_latent_cached()`，它逐个 thinking token 使用 KV cache，并把前一个位置 hidden 经过 `latent_projection` 注入下一个 thinking token embedding。
- 一旦主 Qwen-VL 参数、latent projection 或输入激活被 NaN 污染，student 的完整 hidden states 和 latent_hidden 会一起变 NaN，下游 action/decoder loss 都会 NaN。

最可疑根因：
- `starVLA/config/deepseeds/ds_config.yaml` 使用 BF16 训练，但 `communication_data_type` 原来是 `fp16`。FP16 梯度通信动态范围小，ZeRO-2 reduce/scatter 时更容易溢出；仓库的 multinode ds config 已使用 `fp32`。
- 训练代码原本只在 forward 后发现 `total_loss` NaN 时跳过更新。若某一步 loss 仍有限但 backward/通信/裁剪后的梯度已经非有限，原代码仍会执行 `optimizer.step()`，从而污染参数；下一步 forward 才表现为 hidden states 全 NaN。
- `_build_text_decoder()` 把 Qwen3-1.7B decoder 的 `decoder_language_model` 和 `decoder_lm_head` 全部设为可训练。当前训练同时更新主 Qwen-VL、完整 decoder、decoder projection、latent projection 和 action head，数值风险高于只训练轻量 projection/head。

已做修复：
- 将 `starVLA/config/deepseeds/ds_config.yaml` 的 `communication_data_type` 从 `fp16` 改为 `fp32`。
- 在 `starVLA/training/cot_trainer.py` 增加 optimizer.step 前的非有限梯度检查：backward 后先查梯度，clip 后再查梯度/grad norm，发现 NaN/Inf 就打印参数名并跳过 optimizer/scheduler。

建议下一轮验证：
- 直接重跑原脚本。如果日志出现 `[NaN GRAD]`，说明根因就是某步梯度先坏、原代码随后污染参数。
- 若仍在 forward 阶段直接出现 NaN 且没有 `[NaN GRAD]`，优先临时关闭 `latent_forward_use_cache` 或冻结 `decoder_language_model,decoder_lm_head` 对比。
- 更稳的训练设置是冻结完整 decoder，只训练 `decoder_projection`，或降低 `action_model` LR 到 5e-5。

## 2026-05-06：baseline vs latent fields_3/5/8

按用户指定目录重新解析各 run 的 `wandb/.../files/output.log`，每 100 step 一个训练点。结论以 checkpoint 目录内保存的 `config.full.yaml` 为准，因为源码 config 可能已被后续改动覆盖。

| run | metric | min | mean | 70-90k mean | 90-100k mean | tail10 mean | last | mse tail10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | action_dit_loss | 0.00982 | 0.08947 | 0.03143 | 0.02256 | 0.02179 | 0.02168 | 0.00139 |
| fields_3 | student_action_loss | 0.02317 | 0.11941 | 0.06860 | 0.05966 | 0.06272 | 0.07878 | 0.00405 |
| fields_5 | student_action_loss | 0.02333 | 0.11357 | 0.06137 | 0.05351 | 0.05484 | 0.08060 | 0.00386 |
| fields_8 | student_action_loss | 0.02043 | 0.11003 | 0.05718 | 0.05048 | 0.05312 | 0.07292 | 0.00358 |

本次 fields 结论：
- latent 三组末段 `student_action_loss` 均约为 baseline 的 2.2-2.7 倍，`mse_score` 也约为 baseline 的 2.6-2.9 倍。
- fields 越少 action loss 越高在末段均值上稳定成立：fields_3 > fields_5 > fields_8。不是最后一个 noisy point 造成。
- 短 CoT 的确降低了文本相关 loss：tail10 `teacher_cot_loss` fields_3=0.00839、fields_5=0.09866、fields_8=0.13947；tail10 `decoder_loss` fields_3=0.02086、fields_5=0.18725、fields_8=0.24216。
- 但 action loss 没有随 CoT loss 下降而下降，说明“短 CoT 释放更多梯度给 action”这个假设不符合当前训练机制或当前瓶颈。
- tail10 `teacher_action_loss` 也呈 fields_3=0.05398、fields_5=0.04574、fields_8=0.04222，说明不是 student latent branch 独有问题；teacher 分支带完整可见 CoT 时也更依赖被保留字段的信息。

关键配置/代码证据：
- baseline per-device batch size 为 16，8 卡全局 batch 约 128；latent per-device batch size 为 8，全局 batch 约 64。同样 100k steps 下 baseline 约看过 latent 的 2 倍样本。
- baseline warmup 10000 steps；latent fields_3/5 为 5000 steps。latent 任务更复杂但更快进入较高 LR。fields_8 实际保存配置里 `num_warmup_steps: 5000`。
- 三组 latent full_loss 都开启 teacher_cot、teacher_action、student_action、decoder、slot_distill、pool_distill，权重分别 1/1/1/0.5/1/0.2。
- `QwenGR00TImplicitCoT.forward` 中 teacher_action_loss 和 student_action_loss 都通过同一个 `self.action_model`，total loss 直接加权相加。
- `student` 的 latent token 数等于 `fields` 数；`decoder_loss` 对每个 field slot 监督对应字段文本；distill 也是把 student latent slot 对齐到 teacher field hidden。
- fields_3 只保留 `plan/subtask/move`，fields_5 保留 `task/plan/bboxes/subtask/move`，fields_8 还保留 `subtask_reason/move_reason/gripper`。短 CoT 不只是缩短文本，还改变了 latent slot 数和语义锚点，尤其删除了 bbox/gripper/reason 等可能对 action 更直接有用的信息。
- 三个 latent run 的保存配置均为 `diffusion_model_cfg.output_dim: 2560`，baseline 为 `1024`；`FlowmatchingActionHead.action_decoder` 的输入维度来自 `self.model.config.output_dim`。这使 latent 与 baseline 的 action head 结构/参数化不同，不能把 loss 差异归因于 CoT 长短本身。
- 当前源码中的 `starvla_bridge_latent_cot_full_loss.yaml` 已显示 `output_dim: 1024`，但 run 保存配置是 `2560`，说明必须以 run 目录配置为准。

解释排序的最可能原因：
1. fields 数量减少降低了语言 loss，但同时减少/移除了 action 所需状态变量。fields_8 有 bbox、reason、gripper，fields_5 有 bbox，fields_3 没有 bbox/gripper；所以 action loss 排序更像“动作相关信息量”排序，而不是“文本长度”排序。
2. latent slot 数变少会降低 student 隐变量容量。当前每个 field 对应一个 thinking token/latent slot，action head cross-attend 的条件序列也会变短，不能假设短 CoT 只减少辅助任务负担。
3. 多目标梯度不是简单的总 loss 分配。teacher/student action、decoder、slot distill、pool distill 都在更新共享 VLM/latent/action 相关模块；短 CoT 下 decoder loss 小，不代表 action 梯度更一致，反而可能失去有用中间监督。
4. teacher_action_loss 也随 fields 减少变差，说明可见 CoT 的内容本身影响 action 表征；这比“student 生成 latent 不好”更基础。
5. baseline 仍有明显训练预算和结构优势：更大 batch、更多样本、不同 action head output_dim、可能更接近预训练动作模型结构。

## 四个 run 的关键统计

从各 run 的 `wandb/.../files/output.log` 解析每 100 step 的训练指标。`summary.jsonl` 只记录 checkpoint step，不含 loss。

| run | action last | action tail10 mean | action min | mse last | mse tail10 mean |
|---|---:|---:|---:|---:|---:|
| baseline | 0.02168 | 0.02256 | 0.00982 | 0.00130 | 0.00139 |
| latent decoder_off | student 0.07295 | student 0.05027 | student 0.02074 | 0.00336 | 0.00333 |
| latent distill_off | student 0.06958 | student 0.04868 | student 0.01855 | 0.00309 | 0.00365 |
| latent full_loss | student 0.07292 | student 0.04963 | student 0.02043 | 0.00443 | 0.00363 |

Latent 三个 run 的末段 student action loss 均约为 baseline 的 2.16-2.23 倍；teacher action loss 末段约 0.040-0.042，也显著高于 baseline 0.02256。

## 配置与代码差异

- baseline per-device batch size 为 16，8 卡全局 batch 约 128；latent per-device batch size 为 8，全局 batch 约 64。同样 100k steps 下，baseline 看到的样本数约为 latent 的 2 倍。
- baseline warmup 10000 steps；latent warmup 5000 steps。latent 任务更复杂但更快进入峰值 LR。
- action_model LR 都是 1e-4；qwen_vl_interface LR 都是 1e-5。latent base LR 为 1e-5，baseline base LR 为 3e-5。
- baseline `diffusion_model_cfg.output_dim=1024`；latent 为 2560。当前 `FlowmatchingActionHead` 的 action decoder 输入来自 `self.model.config.output_dim`，所以 latent 的 action head 结构与 baseline 不同。
- latent 总 loss 是 teacher_cot + teacher_action + student_action + decoder * 0.5 + distill loss 的加权和；teacher_action 和 student_action 都通过同一个 `self.action_model`。
- decoder_off / distill_off / full_loss 三个 ablation 的 action loss 近似重合，说明 decoder/distill 开关不是当前 action loss 高的一阶原因。
- baseline run 实际 config 中 `freeze_modules: true` 很可能来自空 CLI flag；训练工具只处理字符串路径列表，所以这大概率不代表真正冻结了模块。latent 的 `freeze_modules: null` 也等价于不冻结。

## 初步判断

latent action loss 高主要不是单个辅助 loss 的问题，而是有效 batch/样本数、latent 与 teacher/student 多上下文共用 action head、未从最强 action checkpoint 初始化、以及 action head 结构不一致共同造成。
