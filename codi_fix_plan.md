# CODI 蒸馏修复方案 v3

## 诊断结论

W&B 曲线分析揭示了 distill_loss 存在**阶梯式下降 + 长期平台**的特征：

```
decoder_off_codi distill_loss:
  0.089 → [快速降] → 0.031 → [25000步平台] → 0.016 → [稳定] → 0.014
```

且 distill_loss 与 student_action_loss 的相关性随时间**崩塌**：

| 阶段 | 相关性 |
|------|--------|
| 早期 (0-10%) | 0.84 |
| 中期 (40-50%) | 0.09 |
| 后期 (80-90%) | 0.10 |

**根本原因**：distill_loss 的梯度路径经过 8 个 cross-attention softmax 的叠加：

```
distill_loss → Σ ∂z_layer_i/∂h    (17个layer的梯度之和)
             = Σ [∂/∂h cross-attn(softmax(h·h^T)) × self-attn × FFN]  (每层经过softmax)
```

softmax 的指数函数使 gradient landscape 充满尖锐局部极小值。8 个 cross-attn 层的 softmax 叠加导致 VLM 被反复卡在平台中。当 distill_loss 卡住而 action_loss 继续改善时，两者梯度方向不一致，distill_loss 成为净干扰。

对比：action_loss 梯度虽然也通过 DiT，但 loss 只在**最终输出**计算 MSE，不约束中间层，landscape 平滑得多。

---

## 修正方案

### 方案 A（最小改动）：修复输入层 bug + 提高权重

只改 2 行代码，跳过 `all_hidden_states[0]`（DiT 输入 embedding，teacher/student 完全相同）。

```python
# _cross_attn_distill_loss: range(num_layers) → range(1, num_layers)
# _cross_attn_distill_loss: / num_layers → / (num_layers - 1)
```

YAML:
```yaml
cross_attn_distill_loss_weight: 0.5    # 0.2 → 0.5
enable_teacher_action_loss: false       # 关闭，减少梯度竞争
```

**预期**：缓解但不能根治平台问题——梯度仍然经过 8 个 softmax。

---

### 方案 B（推荐）：蒸馏最终 DiT output 而非所有中间层

**核心思路**：CODI 在 LLM 中蒸馏 `":"` 位置所有层的 hidden state。但 QwenGR00T 的 DiT 每层都有 softmax cross-attn，中间层的梯度 landscape 过于崎岖。改为**只蒸馏 DiT 最后一层输出**——它是 action_decoder 直接读取的张量，且经过了最后的 LayerNorm + Linear（平滑操作），没有额外的 softmax。

```
当前：distill_loss = Σ_i SmoothL1(z_i_student, z_i_teacher)    i=0..16, 17层
方案B：distill_loss = SmoothL1(z_final_student, z_final_teacher)   只最后1层
```

**具体改动**：

`_action_loss_from_hidden` 已经返回 `all_hidden_states`（list of 17 tensors）。不改这个——只改 `_cross_attn_distill_loss` 让它取最后一个：

```python
def _cross_attn_distill_loss(self, *, student_all_hidden, teacher_all_hidden):
    """
    蒸馏 DiT 最终输出（action_decoder 的输入），而非所有中间层。
    
    all_hidden_states[-1] = 经过所有 16 个 block + norm_out + proj_out 后的最终输出
    形状：[B*rd, S_a, D_dit]
    
    只蒸馏 action token 位置（最后 action_horizon 个），因为 action_decoder 
    最终只读取这部分做 action 预测。
    """
    action_horizon = self.config.framework.action_model.future_action_window_size + 1
    
    # 取最后一层的 action token 部分
    student_final = student_all_hidden[-1][:, -action_horizon:, :]
    teacher_final = teacher_all_hidden[-1][:, -action_horizon:, :]
    
    batch_size = student_final.shape[0]
    mask = torch.ones(batch_size, action_horizon, device=student_final.device, dtype=torch.bool)
    
    return self._vector_distill_loss(student=student_final, teacher=teacher_final, mask=mask)
```

**为什么取最后 action_horizon 个 token**：action_decoder 读取的位置是 `pred[:, -actions.shape[1]:]`（第 311 行），即最后 16 个 token。蒸馏这些 token 的输出直接约束 action 预测质量。

**梯度路径对比**：

```
方案B：distill_loss → ∂(z_final)/∂h = 一条通过全部 DiT 层的梯度链，但 loss 在平滑层计算
当前：distill_loss → Σ_i ∂(z_i)/∂h = 17 条梯度链的和，每条在 softmax 层截断
```

**CODI 对应性**：CODI 蒸馏 `":"` 在各层的 hidden state 是因为 LLM 的 self-attention 不引入额外的 softmax 非线性（hidden state 只做线性投影）。DiT 的 cross-attention 不同——它每次都用 softmax 重加权 VLM features，引入了额外的非线性源。跳过中间层、只在最终输出蒸馏，是在 DiT 架构下对 CODI 原则的合理调整。

**代码改动量**：只改 `_cross_attn_distill_loss` 方法（~10 行），不改其他任何文件。

---

### 方案 C（进阶）：VLM 级蒸馏 + 最终 DiT 输出

在方案 B 基础上，增加 VLM 级蒸馏（thinking token hidden state 与 teacher 对应 field 的 hidden state 匹配）。这需要 dataloader 提供 teacher 侧 field 位置信息，改动范围更大。**建议先验证方案 B 有效后再考虑。**

---

## 实验计划

| 实验 | 方案 | distill 目标 | weight | teacher_action | decoder |
|------|------|-------------|--------|---------------|---------|
| B1 | B | 最终层 action token | 1.0 | false | false |
| B2 | B | 最终层 action token | 0.5 | false | true |
| A1 | A | 所有层（修bug） | 0.5 | false | true |
| Baseline | — | 无 distill | — | false | true |

Baseline 就是 best 模型的配置（student_action + decoder，无 distill，无 teacher loss）。

---

## 验证清单

- [ ] 训练不报错
- [ ] B 方案 distill_loss 曲线呈平滑衰减（无阶梯平台）
- [ ] distill_loss 与 student_action_loss 相关性维持在 0.3+ （不崩塌到 0.1）
- [ ] student_action_loss 正常收敛到 < 0.04
- [ ] SimplerEnv eval 接近或超过 baseline
