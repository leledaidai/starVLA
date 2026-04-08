# QwenGR00T 与 QwenPI 架构说明

本文结合以下实现说明两个框架的结构、训练/推理流程，以及它们的核心差异：

- `starVLA/model/framework/QwenGR00T.py`
- `starVLA/model/framework/QwenPI.py`
- `starVLA/model/modules/action_model/GR00T_ActionHeader.py`
- `starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py`
- `starVLA/model/modules/vlm/QWen2_5.py`

## 1. 两个模型的共同主干

二者都属于 `Qwen-VL + 连续动作头` 的 VLA 框架，整体流程一致：

1. 输入样本由 `image + lang + action (+ state)` 组成。
2. `Qwen-VL` 负责把多视角图像和语言指令编码成视觉语言隐藏状态。
3. 动作头把隐藏状态当作条件，预测未来连续动作序列。
4. 训练时使用 flow matching 风格目标，推理时从随机噪声逐步积分得到动作轨迹。

从框架入口看，两者在 `forward()` 和 `predict_action()` 中都遵循这个套路：

- 先调用 `build_qwenvl_inputs()` 构造 Qwen-VL 的多模态输入。
- 再执行 `self.qwen_vl_interface(..., output_hidden_states=True)` 保留隐藏层输出。
- 最后把视觉语言特征送入各自的动作头。

对应代码：

- `QwenGR00T` 在 [QwenGR00T.py](/inspire/hdd/global_user/gongjingjing-25039/zhdai/starVLA/starVLA/model/framework/QwenGR00T.py#L72) 初始化 Qwen-VL 和动作头。
- `QwenPI` 在 [QwenPI.py](/inspire/hdd/global_user/gongjingjing-25039/zhdai/starVLA/starVLA/model/framework/QwenPI.py#L66) 做相同的事情，但会额外对齐 VLM 隐层维度和层数。
- Qwen-VL 输入构造逻辑在 [QWen2_5.py](/inspire/hdd/global_user/gongjingjing-25039/zhdai/starVLA/starVLA/model/modules/vlm/QWen2_5.py#L181)。

## 2. QwenGR00T 架构解读

### 2.1 整体结构

`QwenGR00T` 可以理解成：

`多视角图像 + 指令 -> Qwen-VL 最后一层隐藏状态 -> 单层条件 Flow Matching 动作头 -> 未来动作序列`

框架初始化逻辑位于 [QwenGR00T.py](/inspire/hdd/global_user/gongjingjing-25039/zhdai/starVLA/starVLA/model/framework/QwenGR00T.py#L58)：

- `self.qwen_vl_interface = get_vlm_model(...)`
- 把动作头的 `cross_attention_dim` 强行对齐到 Qwen-VL 的 `hidden_size`
- `self.action_model = get_action_model(...)`

这里最关键的一点是：`QwenGR00T` 只使用 **Qwen-VL 的最后一层 hidden states** 作为条件特征，而不是把多层隐藏态全部拿来用。代码在 [QwenGR00T.py](/inspire/hdd/global_user/gongjingjing-25039/zhdai/starVLA/starVLA/model/framework/QwenGR00T.py#L100) 到 [QwenGR00T.py](/inspire/hdd/global_user/gongjingjing-25039/zhdai/starVLA/starVLA/model/framework/QwenGR00T.py#L108)：

- 执行 Qwen-VL 前向
- 取 `qwenvl_outputs.hidden_states[-1]`

这意味着它把 VLM 当成一个统一的高层语义编码器，动作头只看最终融合后的 token 表示。

### 2.2 输入和数据流

训练时 `examples` 中主要字段是：

- `image`: 多视角图像列表
- `lang`: 语言指令
- `action`: 动作序列
- `state`: 可选机器人状态

在 [QwenGR00T.py](/inspire/hdd/global_user/gongjingjing-25039/zhdai/starVLA/starVLA/model/framework/QwenGR00T.py#L91) 到 [QwenGR00T.py](/inspire/hdd/global_user/gongjingjing-25039/zhdai/starVLA/starVLA/model/framework/QwenGR00T.py#L130) 中，训练数据流如下：

1. 从 batch 中提取 `image/lang/action/state`。
2. 用 `build_qwenvl_inputs()` 打包成 Qwen 的 chat-style 多模态输入。
3. Qwen-VL 输出最后一层 `last_hidden`。
4. 从动作序列中截取最后 `future_action_window_size + 1` 步作为监督目标。
5. 将目标动作、视觉语言特征、状态重复 `repeated_diffusion_steps` 次。
6. 把重复后的条件送入 `FlowmatchingActionHead` 计算损失。

注意这里虽然类里维护了：

- `past_action_window_size`
- `future_action_window_size`
- `chunk_len = past + 1 + future`

但在当前 `forward()` 实现里，真正送入动作头监督的是：

- `actions[:, -(future_action_window_size + 1):, :]`

也就是说当前训练目标只取“当前步 + 未来步”，并没有把 `past_action_window_size` 那段动作直接作为监督片段送进动作头。

### 2.3 GR00T 动作头怎么工作

动作头主体在 [GR00T_ActionHeader.py](/inspire/hdd/global_user/gongjingjing-25039/zhdai/starVLA/starVLA/model/modules/action_model/GR00T_ActionHeader.py#L216)。

其内部组件包括：

- `state_encoder`: 把机器人状态映射到动作隐空间
- `action_encoder`: 把连续动作序列和时间步编码成 token
- `future_tokens`: 一组可学习 token，像是给未来预测预留的槽位
- `DiT`: 条件 Transformer，用视觉语言特征做 cross-attention
- `action_decoder`: 把 DiT 输出还原成连续动作速度

训练过程在 [GR00T_ActionHeader.py](/inspire/hdd/global_user/gongjingjing-25039/zhdai/starVLA/starVLA/model/modules/action_model/GR00T_ActionHeader.py#L270) 到 [GR00T_ActionHeader.py](/inspire/hdd/global_user/gongjingjing-25039/zhdai/starVLA/starVLA/model/modules/action_model/GR00T_ActionHeader.py#L318)：

1. 对真实动作 `actions` 采样随机噪声 `noise`。
2. 采样连续时间 `t`。
3. 构造带噪轨迹 `noisy_trajectory = (1 - t) * noise + t * actions`。
4. 令目标速度为 `velocity = actions - noise`。
5. 用 `ActionEncoder` 编码带噪动作和离散化时间步。
6. 把 `state_features + future_tokens + action_features` 拼成动作侧序列。
7. 用 DiT 对动作侧序列做更新，其中 `encoder_hidden_states=vl_embs`，也就是和 Qwen-VL 最后一层做 cross-attention。
8. 用 `action_decoder` 输出速度预测。
9. 用 MSE 监督 `pred_velocity` 和 `velocity`。

这个设计本质上是一个 **条件 flow matching / velocity prediction** 模型，而不是传统离散 action token 生成。

### 2.4 推理过程

推理入口在 [QwenGR00T.py](/inspire/hdd/global_user/gongjingjing-25039/zhdai/starVLA/starVLA/model/framework/QwenGR00T.py#L136)，真正的动作采样在 [GR00T_ActionHeader.py](/inspire/hdd/global_user/gongjingjing-25039/zhdai/starVLA/starVLA/model/modules/action_model/GR00T_ActionHeader.py#L320)。

流程如下：

1. 输入图像先按训练分辨率 resize。
2. Qwen-VL 输出最后一层隐藏态 `last_hidden`。
3. 动作头从高斯噪声初始化一段动作轨迹。
4. 进行 `num_inference_timesteps` 次迭代。
5. 每次根据当前动作、时间步、视觉语言条件预测速度。
6. 用 Euler 积分更新 `actions = actions + dt * pred_velocity`。
7. 最终得到连续动作序列。

因此 `QwenGR00T` 的推理不是一次性回归动作，而是一个小步积分的生成过程。

## 3. QwenPI 架构解读

### 3.1 整体结构

`QwenPI` 的总体思路是：

`多视角图像 + 指令 -> Qwen-VL 多层隐藏状态 -> 分层条件 Flow Matching 动作头 -> 未来动作序列`

它和 `QwenGR00T` 最大的不同，是 **不只取最后一层 VLM 特征，而是取最后 N 层隐藏状态，逐层喂给动作头的 Transformer block**。

在初始化阶段 [QwenPI.py](/inspire/hdd/global_user/gongjingjing-25039/zhdai/starVLA/starVLA/model/framework/QwenPI.py#L68) 到 [QwenPI.py](/inspire/hdd/global_user/gongjingjing-25039/zhdai/starVLA/starVLA/model/framework/QwenPI.py#L77) 会先做两件事：

1. 读取 Qwen-VL 的真实 `hidden_size`。
2. 决定动作头要消费多少层 VLM hidden states。

然后把这两个值写回：

- `config.framework.qwenvl.vl_hidden_dim`
- `config.framework.qwenvl.num_vl_layers`

这样 `LayerwiseFlowmatchingActionHead` 在构建时，就知道自己的层数和每层 cross-attention 的维度应该和 VLM 对齐。

### 3.2 输入和数据流

训练主流程在 [QwenPI.py](/inspire/hdd/global_user/gongjingjing-25039/zhdai/starVLA/starVLA/model/framework/QwenPI.py#L106) 到 [QwenPI.py](/inspire/hdd/global_user/gongjingjing-25039/zhdai/starVLA/starVLA/model/framework/QwenPI.py#L145)。

前半段和 `QwenGR00T` 一样，区别出现在 Qwen-VL 输出处理：

- 先取 `all_hidden = qwenvl_outputs.hidden_states`
- 再根据动作头 block 数量得到 `expected_layers`
- 最后取 `all_hidden[-expected_layers:]` 作为 `vl_embs_list`

也就是说，`QwenPI` 会把最后若干层 VLM hidden states 保留下来，每一层都参与动作生成。

动作监督目标仍然是：

- `actions[:, -(future_action_window_size + 1):, :]`

但训练时有一个实现细节：虽然先读取了 `trainer.repeated_diffusion_steps`，后面又被硬编码成了 `2`，见 [QwenPI.py](/inspire/hdd/global_user/gongjingjing-25039/zhdai/starVLA/starVLA/model/framework/QwenPI.py#L129) 到 [QwenPI.py](/inspire/hdd/global_user/gongjingjing-25039/zhdai/starVLA/starVLA/model/framework/QwenPI.py#L133)。这说明当前版本为了控制大动作头的训练开销，实际上固定做 2 次重复。

### 3.3 Layerwise 动作头怎么工作

动作头在 [LayerwiseFM_ActionHeader.py](/inspire/hdd/global_user/gongjingjing-25039/zhdai/starVLA/starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py#L215)。

它和 GR00T 版动作头的共同点是：

- 都有 `state_encoder`
- 都有 `action_encoder`
- 都有 `future_tokens`
- 都做 flow matching 风格的速度监督
- 都用 Euler 积分进行推理

但是它的关键设计变化在于 **分层条件注入**。

初始化时 [LayerwiseFM_ActionHeader.py](/inspire/hdd/global_user/gongjingjing-25039/zhdai/starVLA/starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py#L226) 到 [LayerwiseFM_ActionHeader.py](/inspire/hdd/global_user/gongjingjing-25039/zhdai/starVLA/starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py#L234) 会：

1. 用 `global_config.framework.qwenvl.num_vl_layers` 设置 DiT 的层数。
2. 用 `global_config.framework.qwenvl.vl_hidden_dim` 设置输入维度。
3. 根据隐藏维度自动推导注意力头数。

训练过程在 [LayerwiseFM_ActionHeader.py](/inspire/hdd/global_user/gongjingjing-25039/zhdai/starVLA/starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py#L274) 到 [LayerwiseFM_ActionHeader.py](/inspire/hdd/global_user/gongjingjing-25039/zhdai/starVLA/starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py#L326)：

1. 对真实动作注噪，得到 `noisy_trajectory`。
2. 用 `ActionEncoder` 编码当前动作轨迹。
3. 拼接 `state_features + future_tokens + action_features`。
4. 单独用 `self.model.timestep_encoder()` 编码时间步，得到 `temb`。
5. 遍历动作头自己的每一层 `transformer_blocks`。
6. 第 `layer_idx` 层 block 只接收第 `layer_idx` 个 `vl_embs_list[layer_idx]` 作为 `encoder_hidden_states`。
7. 最后解码出速度并计算 MSE。

这里的核心不是“把多层特征先融合后再送入动作头”，而是：

- **动作头第 1 层看 VLM 第 1 个条件层**
- **动作头第 2 层看 VLM 第 2 个条件层**
- ...

这是一种显式的逐层对齐设计。

### 3.4 推理过程

推理入口在 [QwenPI.py](/inspire/hdd/global_user/gongjingjing-25039/zhdai/starVLA/starVLA/model/framework/QwenPI.py#L150)，采样逻辑在 [LayerwiseFM_ActionHeader.py](/inspire/hdd/global_user/gongjingjing-25039/zhdai/starVLA/starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py#L328)。

这里有一个需要注意的地方：`QwenPI.predict_action()` 的函数注释写着“单次前向直接回归未来动作（无扩散采样）”，但按当前代码实现，真正执行的仍然是从噪声初始化后，多步 Euler 积分更新动作，因此它在实现上依然属于迭代式生成，而不是一次前向直接回归。

过程和 `QwenGR00T` 很像：

1. 从噪声初始化动作序列。
2. 逐步迭代多个时间步。
3. 每一步都重新编码当前动作轨迹。
4. 每个 Transformer block 读取对应层的 VLM 特征。
5. 预测速度并进行 Euler 更新。

所以从训练目标和采样方式看，`QwenPI` 仍然是 flow matching 模型；它和 `QwenGR00T` 的本质差异主要在条件建模方式，而不在损失形式。

## 4. QwenGR00T 与 QwenPI 的核心区别

### 4.1 条件特征来源不同

`QwenGR00T`：

- 只使用 Qwen-VL 最后一层隐藏状态 `hidden_states[-1]`
- 动作头看到的是一个“最终融合后的单层条件表示”

对应代码：

- [QwenGR00T.py](/inspire/hdd/global_user/gongjingjing-25039/zhdai/starVLA/starVLA/model/framework/QwenGR00T.py#L107)
- [GR00T_ActionHeader.py](/inspire/hdd/global_user/gongjingjing-25039/zhdai/starVLA/starVLA/model/modules/action_model/GR00T_ActionHeader.py#L306)

`QwenPI`：

- 使用 Qwen-VL 最后若干层隐藏状态
- 每个动作头 block 对应消费一层 VLM hidden state

对应代码：

- [QwenPI.py](/inspire/hdd/global_user/gongjingjing-25039/zhdai/starVLA/starVLA/model/framework/QwenPI.py#L115)
- [QwenPI.py](/inspire/hdd/global_user/gongjingjing-25039/zhdai/starVLA/starVLA/model/framework/QwenPI.py#L118)
- [LayerwiseFM_ActionHeader.py](/inspire/hdd/global_user/gongjingjing-25039/zhdai/starVLA/starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py#L313)

一句话总结：

- `QwenGR00T` 是“单层 VLM 条件”
- `QwenPI` 是“多层 VLM 条件，逐层注入”

### 4.2 动作头结构不同

`QwenGR00T` 的 `FlowmatchingActionHead`：

- 直接调用一个完整的 `DiT(...)`
- `encoder_hidden_states` 是单个张量 `vl_embs`
- 条件注入更简单，结构更接近“标准 cross-attention 条件生成器”

`QwenPI` 的 `LayerwiseFlowmatchingActionHead`：

- 自己显式遍历 `transformer_blocks`
- 每一层单独传入对应的 `vl_embs_list[layer_idx]`
- 更强调动作头层和 VLM 层之间的对应关系

这通常意味着：

- `QwenPI` 的表达能力更强
- 但配置更敏感，显存和计算开销也通常更高

### 4.3 维度与层数对齐策略不同

`QwenGR00T` 只对齐了 `cross_attention_dim`，见 [QwenGR00T.py](/inspire/hdd/global_user/gongjingjing-25039/zhdai/starVLA/starVLA/model/framework/QwenGR00T.py#L74)。

`QwenPI` 不只对齐维度，还显式设置：

- `vl_hidden_dim`
- `num_vl_layers`

见 [QwenPI.py](/inspire/hdd/global_user/gongjingjing-25039/zhdai/starVLA/starVLA/model/framework/QwenPI.py#L68) 到 [QwenPI.py](/inspire/hdd/global_user/gongjingjing-25039/zhdai/starVLA/starVLA/model/framework/QwenPI.py#L75)。

这意味着 `QwenPI` 对 backbone 隐层结构依赖更强，而 `QwenGR00T` 更像一个把 VLM 当成黑盒编码器的方案。

### 4.4 训练开销和实现复杂度不同

`QwenGR00T`：

- 条件只是一份 `last_hidden`
- 框架简单
- 更容易迁移到不同 VLM

`QwenPI`：

- 要保留多层 hidden states
- 动作头层数要与 VLM 条件层数匹配
- 当前实现里为了控制开销，把 `repeated_diffusion_steps` 固定成 2

所以工程上可以粗略理解为：

- `QwenGR00T` 偏轻量、直接
- `QwenPI` 偏重型、分层建模更强

### 4.5 二者“相同”的地方

不要把这两个模型理解成完全不同范式，它们仍然有大量共性：

- 都使用 Qwen-VL 做图像和指令编码
- 都支持多视角图像输入
- 都可以接可选 `state`
- 都不是离散 token action policy，而是连续动作 flow matching policy
- 都预测 `future_action_window_size + 1` 长度的动作片段
- 都从噪声初始化动作，并通过 Euler 积分逐步生成

所以更准确地说，`QwenPI` 不是推翻 `QwenGR00T`，而是在 **动作条件注入方式** 上做了更强的层级化改造。

## 5. 怎么理解两者的适用场景

如果只从当前代码结构出发，可以这样理解：

### 更适合用 QwenGR00T 的情况

- 你希望结构简单，先快速跑通训练和部署
- 你更关心稳定性和工程可迁移性
- 你希望动作头与具体 VLM 隐层细节耦合更小

### 更适合用 QwenPI 的情况

- 你希望更充分利用 Qwen-VL 的中高层表示
- 你接受更高的算力和显存开销
- 你想探索“VLM 分层特征是否能更好支撑动作生成”

## 6. 最后用一句话概括

`QwenGR00T` 是一个 **基于 Qwen-VL 最后一层特征的单条件 Flow Matching 动作模型**；  
`QwenPI` 是一个 **基于 Qwen-VL 多层隐藏态、逐层 cross-attention 注入的 Layerwise Flow Matching 动作模型**。

如果你只看最本质的差别，就是：

- `QwenGR00T`：最后一层条件
- `QwenPI`：多层条件，逐层对齐
