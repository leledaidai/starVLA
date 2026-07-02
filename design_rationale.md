# QwenGR00T Latent CoT 蒸馏方案的设计理由

## 三句话总结

**CODI 的思想**：推理在边界 token 的 hidden state 上产生一个 additive shift，蒸馏这个 shift 就能传递推理能力。

**QwenGR00T 架构的特殊性**：VLM 和 DiT 是两个模块，通过 cross-attention 连接。边界不再是单个 token 的 hidden state，而是 DiT 每一层中"VLM 信息进入 action stream"的那个点——即 DiT 每层的 hidden state output。

**方案**：蒸馏 DiT 全部层的 hidden state output（VLM→Action 的完整网关），distill_loss 只更新 VLM（强制 VLM 独立弥合差异），action_loss 正常更新两者。

---

## 1. CODI 在做什么

```
LLM 内部:
  Q → CoT step1 → CoT step2 → ... → ":" → answer
                                       ↑
                                 只蒸馏这一个 token
                                 所有 transformer 层
```

为什么蒸馏 `":"` 就够了？因为在这个位置：

- `h_CoT(":") = h_no_CoT(":") + shift(R)` —— **additive shift**
- shift 汇聚了前面所有 CoT token 通过 self-attention 的累积效应
- LLM 从这个 hidden state 出发自回归地生成答案——它是**推理到输出的完整网关**

CODI **不蒸馏中间步骤**，不蒸馏 CoT 每个 step 的 hidden state。这给模型自由去自己组织 latent space。

---

## 2. QwenGR00T 的特殊性

### 2.1 架构对比

```
CODI:                              QwenGR00T:
  一个模块                           两个模块
  ┌──────────┐                      ┌─────┐    cross-attn    ┌─────┐
  │   LLM    │                      │ VLM │ ──────────────→ │ DiT │ → action
  │          │                      │     │  h (L token)    │     │
  │ CoT → ":"│                      └─────┘                 └─────┘
  └──────────┘
  
  边界在 LLM 内部                    边界跨两个模块的接口
  ":" 是 ONE token                  cross-attn 读取 ALL L tokens
  shift 纯线性 (CODI 推导)           shift = linear + quadratic (见数学推导)
```

### 2.2 三个关键差异

**差异 1：双模块 vs 单模块**

推理和输出在不同模块中，边界跨越模块接口。

**差异 2：并行读取 vs 自回归**

CODI 中 LLM 从 `":"` 的 hidden state 自回归生成答案——后续所有 token 都依赖这一个状态。

QwenGR00T 中 DiT 并行 attend 到 VLM 全部 L 个 token——没有单一的"起始状态"，action 的每个 token 独立从 VLM 各位置读取信息。

**差异 3：shift 的数学形式**

在 self-attention（CODI）中：
```
Δa ≈ W_V·R·R^T·W_K^T·q     ← 仅依赖 R（CoT token），q 固定
```

在 cross-attention（QwenGR00T）中：
```
Δa = W_V·[h·Δh^T + Δh·h^T + Δh·Δh^T]·W_K^T·Q·z
     └─线性交叉项──┘  └─二次自交互─┘
```

cross-attention 的 `h·h^T` 外积引入了 Δh 的二次自交互项。

---

## 3. 方案如何回应这些特殊性

### 3.1 为什么蒸馏 DiT hidden state output

在 DiT 的每一层，cross-attention 把 VLM 的 h 读入 action stream：

```
h (VLM) ──cross-attn──→ z_0 ──self-attn+FFN──→ z_1 ──cross-attn──→ ... → action
                            ↑
                        DiT hidden state
```

`z_0` 是 action stream 在吸收了 VLM 信息后的表示。它是 **VLM→Action 的信息网关**——CODI 中 `":"` 的对等物。

但与 CODI 不同的是：这里不是 1 个 token，而是 **S_a 个 action token × 全部 VLM 位置的加权和**。这就是为什么只蒸馏边界 token 的 K,V 不够——gateway 是整个跨注意力运算的结果，不是其中某一个位置的投影。

### 3.2 为什么蒸馏全部 DiT 层

CODI 蒸馏 LLM 的全部层（attention + FFN）。DiT 同理：

```
Block 0 (cross-attn): teacher输出 = x0t, student输出 = x0s  → 蒸馏 ✓
Block 1 (self-attn):  x0t → FFN → x1t, x0s → FFN → x1s     → 蒸馏 ✓
Block 2 (cross-attn): ...
```

Self-attention 层的输出包含了上一层 cross-attention 差异经 FFN 非线性放大后的结果。跳过 self-attn 层等于放任差异在层间累积——违反 CODI 的逐层蒸馏原则。

### 3.3 为什么 distill_loss 只更新 VLM

蒸馏的目的是：让 VLM 在 latent 模式下产生**功能等价的前缀**，使得 DiT 无需改变就能产出相同的 action。

```
好的蒸馏:     VLM → 适应 DiT（DiT 不变，VLM 被优化到提供等价前缀）
不好的蒸馏:   VLM → DiT 互相补偿（目标偏移，VLM 和 DiT 互相迁就）
```

如果 distill_loss 也更新 DiT，等于让 action head 来"迁就" VLM 的不足——背离了"让 VLM 做好前缀"的设计意图。

技术上：在 `forward()` 内用 `torch.autograd.grad(distill_loss, vlm_params)` 手动计算并写入 VLM 梯度，`distill_loss` 不加入返回的 `total_loss`。之后 Trainer 的 `total_loss.backward()` 只给 DiT 写入 action_loss 贡献的梯度。

### 3.4 为什么合并 forward 是安全的

merge forward 和分开 forward 对参数更新**完全等价**（梯度可加性）：

```
分开: DiT.forward() → L_action → backward() → grad_action  }
       DiT.forward() → L_distill → backward() → grad_distill } → optimizer(grad_action + grad_distill)

合并: DiT.forward() → L_action + L_distill → backward() → optimizer(grad_action + grad_distill)
```

区别只是 engineering：省一次 DiT forward。合并不会造成额外的"梯度干扰"。

### 3.5 关于 shift 的非线性

数学推导显示 cross-attention 的 shift 有线性交叉项和二次自交互项：

```
shift_total(Δh) = linear(Δh) + quadratic(Δh·Δh^T)
```

但这不影响蒸馏的可行性，因为 **additivity 对总量成立**：

```
z(h_cot) = z(h_direct) + shift_total(Δh)
```

蒸馏这个总量，梯度驱动 student 找到一个 `Δh_student` 使得：

```
shift_total(Δh_student) ≈ shift_total(Δh_teacher)
```

虽然 teacher 的 shift 包含二次项，student 不需要显式分解它——只需要 match 总量。这和一个神经网络学任何复杂函数没有本质区别：只要目标是可加的（对于固定的 h_direct），梯度就会引导 student 找到正确的 Δh。

---

## 4. 完整梯度流

```
                    VLM                          DiT
                    ===                          ===
action_loss:        ✓  (via backward)            ✓  (via backward)
distill_loss:       ✓  (via autograd.grad)       ✗  (不更新)
teacher_cot_loss:   ✓  (via backward)            ✗  (DiT 不在该图中)
decoder_loss:       ✓  (via backward)            ✗  (DiT 不在该图中)
slot/pool (legacy): ✓  (via backward)            ✗  (DiT 不在该图中)
```

VLM 得到 distill_loss 的梯度（强制它弥合 teacher-student 差异）加上 action_loss 的梯度（保证它产出对 action 预测有用的表示）。

DiT 只得到 action_loss 的梯度（保证它学会从 VLM 特征解码出正确的 action）。

---

## 5. 方案演进历史

| 版本 | 蒸馏对象 | 问题 |
|------|---------|------|
| V1 | DiT hidden states（所有层） | 方案本身方向对，但当时没有解释清楚为什么非线性的 shift 也可以蒸馏 |
| V2 | 边界 token 的 K,V 投影 | 边界 token 只是 cross-attn 中 L 个 KV pair 之一，信息论上不足 |
| V3（最终） | DiT hidden states（所有层），distill 只更新 VLM | 正确匹配 CODI 的网关概念，合并 forward，梯度流向清晰 |
