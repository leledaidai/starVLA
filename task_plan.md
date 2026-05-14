# 任务计划：latent CoT VLA fields action_loss 分析

## 目标
对比 baseline、latent fields_3、latent fields_5、latent fields_8 四个训练目录的配置与 wandb 日志，解释 latent CoT VLA 的 `student_action_loss`、`teacher_action_loss` 高于 baseline，以及 fields 越少 action loss 反而更高的可能原因，并给出降低 action loss 的训练策略。

## 阶段
- [complete] 阶段 1：收集本次四个 run 的配置、脚本与 wandb output.log
- [complete] 阶段 2：提取 baseline 与 fields_3/5/8 的 action、cot、decoder、distill 曲线统计
- [complete] 阶段 3：检查 latent forward/loss 组合与字段监督方式，定位可能原因
- [complete] 阶段 4：形成降低 action_loss 与验证 fields 顺序假设的实验建议

## 约束
- 不修改训练代码或 checkpoint。
- 只读取本地日志与配置；若 wandb 二进制日志无法直接解析，优先使用 summary/config 文件和可用本地工具。

## 追加任务：2026-05-07 NaN/Inf 排查

## 目标
排查 `examples/SimplerEnv/train_files/main_run_bridge_latent_cot_distill_off_test_bs16_weight_small.sh` 在 step 29 左右出现 `student.hidden_states` / `student.latent_hidden` 全 NaN，以及 `student_action_loss`、`decoder_loss`、`total_loss` NaN 的原因，并给出可执行修复。

## 阶段
- [complete] 阶段 1：确认脚本、实际配置、DeepSpeed 配置和 run 输出目录
- [complete] 阶段 2：定位 NaN 诊断打印位置和 student latent forward / loss 计算链路
- [complete] 阶段 3：判断 NaN 来源是 forward 激活已污染，还是单个 loss 公式本身
- [complete] 阶段 4：增加 optimizer.step 前的非有限梯度保护，并修正 DeepSpeed 通信精度

## 当前判断
- 问题不是 `student_action_loss` 或 `decoder_loss` 的公式单独产生 NaN；日志里 `student.hidden_states` 和 `student.latent_hidden` 已经是 NaN，loss 只是下游表现。
- 更像是前一次或前几次更新中出现非有限梯度，`optimizer.step()` 把参数污染后，之后所有 batch 的 hidden states 都变 NaN。
