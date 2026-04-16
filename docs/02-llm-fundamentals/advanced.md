# LLM 原理 · 高级

::: info 学习目标
- 从架构层面理解 Transformer 的完整构成（自注意力、多头、FFN、残差）
- 理解 KV Cache 优化的原理和必要性
- 了解 Flash Attention 的核心思想
- 掌握模型量化的基本概念和实践
- 学完能理解推理引擎的底层优化原理，做出正确的工程决策

预计学习时间：2-3 小时
:::

## Transformer Block 的完整构成

::: tip 为什么要深入理解这些组件？
不是为了让你从零实现 Transformer -- PyTorch 一行 `nn.MultiheadAttention` 就能搞定。深入理解的目的是帮你建立三个工程直觉：

1. **为什么注意力是 O(n^2) 的** -- `Q * K^T` 生成 `(seq_len, seq_len)` 矩阵，长上下文为什么烧显存一目了然
2. **为什么多头比单头好** -- 单头只能学一种关注模式，多头让模型同时关注语法、语义、位置等不同维度
3. **为什么 Transformer 可以并行训练** -- 不像 RNN 必须串行，注意力对所有位置的计算是一次矩阵乘法完成的
:::

下面我们从最小的自注意力开始，逐步讲解完整 Transformer Block 的每个组件。

### 第一步：单头自注意力 -- 理解 O(n^2) 的根源

自注意力（Self-Attention）让序列中的每个 Token 能关注其他所有 Token。它的核心流程分五步：

```
自注意力的完整计算流程：

输入 X (seq_len, d_model)
  ↓
步骤 1: 线性投影
  Q = X * W_q    -- (seq_len, d_k)  "每个位置想查什么"
  K = X * W_k    -- (seq_len, d_k)  "每个位置提供什么标签"
  V = X * W_v    -- (seq_len, d_k)  "每个位置的实际内容"
  ↓
步骤 2: 计算注意力分数
  scores = Q * K^T / sqrt(d_k)    -- (seq_len, seq_len)  ← O(n^2) 的根源！
  ↓
步骤 3: 应用因果掩码（自回归生成时）
  scores[未来位置] = -inf          -- 每个位置只能看到自己和之前的 Token
  ↓
步骤 4: Softmax 归一化
  weights = softmax(scores)        -- 每行是一个概率分布
  ↓
步骤 5: 加权求和
  output = weights * V             -- (seq_len, d_k)
```

**因果掩码**是自回归模型（GPT、Claude）的关键：生成第 3 个 Token 时，不能偷看第 4、5 个 Token 的信息。掩码矩阵把未来位置的分数设为负无穷，经过 softmax 后权重变成 0。

```
因果掩码示例（5 个 Token）：

                Token1  Token2  Token3  Token4  Token5
  Token1 看到:   [OK]    [-inf]  [-inf]  [-inf]  [-inf]   -- 只能看自己
  Token2 看到:   [OK]    [OK]    [-inf]  [-inf]  [-inf]   -- 看自己和前面
  Token3 看到:   [OK]    [OK]    [OK]    [-inf]  [-inf]
  Token4 看到:   [OK]    [OK]    [OK]    [OK]    [-inf]
  Token5 看到:   [OK]    [OK]    [OK]    [OK]    [OK]     -- 可以看所有
```

> **工程直觉**：步骤 2 中 `Q * K^T` 生成了 `(seq_len, seq_len)` 的矩阵，这就是注意力 O(n^2) 复杂度的根源。序列长度从 4K 翻倍到 8K，这个矩阵的大小翻 4 倍。当你看到某个模型宣称支持 200K 上下文时，你应该立刻想到：它一定用了某种方式绕过了这个 N^2 瓶颈（比如后面要讲的 Flash Attention）。

### 第二步：多头注意力 -- 单头的局限是什么？

上面的单头注意力有一个根本问题：**一组 Q/K/V 权重只能学到一种关注模式**。比如它可能学会了关注"最近的几个词"，但同时就没法关注"句子开头的主语"。

多头注意力的解决方案很直接：并行运行多个独立的注意力头，每个头学习不同的关注模式，最后把结果拼起来。

```
多头注意力流程（以 d_model=16, num_heads=4 为例）：

输入 X (seq_len, 16)
       ↓
    维度均分：每头 d_k = 16 / 4 = 4
       ↓
  ┌────────┬────────┬────────┬────────┐
  │  头 1  │  头 2  │  头 3  │  头 4  │
  │ 关注   │ 关注   │ 关注   │ 关注   │
  │ 语法   │ 指代   │ 语义   │ 距离   │
  │ (n,4)  │ (n,4)  │ (n,4)  │ (n,4)  │
  └────┬───┴────┬───┴────┬───┴────┬───┘
       └────────┼────────┼────────┘
                ↓ 拼接
        (seq_len, 16)
                ↓ 线性投影 W_o
        输出 (seq_len, 16)
```

> **工程直觉**：注意每个头独立计算、互不依赖，这意味着多头在 GPU 上可以完全并行。实际模型不会用 for 循环，而是把多头 reshape 成一个大的 batch 矩阵乘法一次算完。另外，`d_model / num_heads` 这个设计很精妙 -- 多头不增加总计算量，只是把同样的维度切分给不同的头，让每个头专注于子空间。

### 第三步：完整的 Transformer Block -- 光有注意力够吗？

注意力层只解决了"Token 之间怎么交互信息"的问题，但还缺两样东西：

- **前馈网络（FFN）**：注意力是线性加权求和，表达能力有限。FFN 提供逐位置的非线性变换，让模型有能力做更复杂的特征提取。你可以理解为：注意力负责"看哪里"，FFN 负责"看到之后怎么想"。
- **残差连接 + LayerNorm**：没有残差连接，深层网络的梯度会消失，训练不动。LayerNorm 让每一层的输入分布稳定，加速收敛。

下面是一个完整 Transformer Block 的数据流：

```
完整 Transformer Block：

输入 X
  ↓
  ├──────────────────┐
  ↓                  │ 残差连接
  多头自注意力        │
  ↓                  │
  + (加回原始输入) ←──┘
  ↓
  LayerNorm
  ↓
  ├──────────────────┐
  ↓                  │ 残差连接
  前馈网络 (FFN)      │
  │  隐层 = ReLU(X * W1)
  │  输出 = 隐层 * W2
  ↓                  │
  + (加回输入) ←──────┘
  ↓
  LayerNorm
  ↓
输出
```

下面这张对比表总结了从单头到完整 Block 的递进关系：

| 组件 | 解决的问题 | 输入 -> 输出 | 关键特征 |
|------|-----------|-------------|---------|
| **单头自注意力** | Token 之间如何交互信息 | (n, d_k) -> (n, d_k) | O(n^2) 复杂度，因果掩码 |
| **多头注意力** | 如何同时关注多种关系 | (n, d_model) -> (n, d_model) | 并行多头 + 拼接 + 投影 |
| **Transformer Block** | 如何构建可堆叠的深层网络 | (n, d_model) -> (n, d_model) | 注意力 + FFN + 残差 + LayerNorm |

> **工程直觉**：现在你看到了一个完整 Transformer Block 的全貌 -- 注意力 + FFN + 残差 + LayerNorm。真实的 LLM（如 Llama 3 70B）就是把这个结构堆叠 80 层。理解这个结构后，你在看模型架构图、推理优化文档时就不会犯怵了：所有优化（KV Cache、量化、Flash Attention）都是在这个基本结构上做文章。

## KV Cache 优化

### 为什么需要 KV Cache

LLM 生成文本时是**一个 Token 一个 Token**地生成。每生成一个新 Token，都需要对之前所有 Token 做注意力计算。如果不做优化，每步都要重新计算所有 Token 的 K 和 V，造成大量重复计算。

```
生成 "我 喜欢 吃 苹果"：

步骤 1: 输入 "我"
  计算 K("我"), V("我") -> 生成 "喜欢"

步骤 2: 输入 "我 喜欢"
  重新计算 K("我"), V("我")  ← 浪费！已经算过了
  计算 K("喜欢"), V("喜欢") -> 生成 "吃"

步骤 3: 输入 "我 喜欢 吃"
  重新计算 K("我"), V("我")     ← 浪费！
  重新计算 K("喜欢"), V("喜欢") ← 浪费！
  计算 K("吃"), V("吃") -> 生成 "苹果"
```

KV Cache 的解决方案：把已经计算过的 K 和 V 缓存起来，每步只计算新 Token 的 K、V。

```
有无 KV Cache 的对比：

【无 Cache】生成第 5 个 Token 时：
  重算 K/V("我") + 重算 K/V("喜欢") + 重算 K/V("吃") + 重算 K/V("苹果") + 新算 K/V("很")
  Q("很") 与 5 个 K 做注意力
  总计算量: 5 * 5 * d_k = 25 * d_k

【有 Cache】生成第 5 个 Token 时：
  从缓存取 K/V("我","喜欢","吃","苹果") + 新算 K/V("很") 追加到缓存
  Q("很") 与 5 个 K 做注意力
  总计算量: 1 * 5 * d_k = 5 * d_k       ← 节省 80%!
```

核心流程：

| 步骤 | 操作 | 说明 |
|------|------|------|
| 1 | 只为新 Token 计算 Q、K、V | 之前 Token 的 K/V 已在缓存中 |
| 2 | 将新的 K、V 追加到缓存 | 缓存随生成逐步增长 |
| 3 | 用新 Token 的 Q 与**所有**缓存的 K 计算注意力 | Q 只有 1 个，K 有 n 个 |
| 4 | 用注意力权重加权所有缓存的 V | 得到新 Token 的输出表示 |

随着生成的 Token 越来越多，KV Cache 的节省越显著。生成第 100 个 Token 时，无 Cache 需要 100*100=10000 次运算，有 Cache 只需 1*100=100 次，**节省 99%**。

::: tip KV Cache 的代价
KV Cache 用空间换时间。对于长上下文（如 200K Token），Cache 占用的显存非常大。这也是为什么长上下文推理需要更多 GPU 显存。
:::

## Flash Attention 简介

### 标准注意力为什么慢？

问题不在计算量，而在**显存搬运**。标准注意力需要把完整的 `(N, N)` 注意力矩阵写入 GPU 的主显存（HBM），然后再读出来做 softmax。当 N=100K 时，这个矩阵有 100 亿个元素 -- 光是搬运数据就成了瓶颈。GPU 的计算单元大部分时间在等数据，而不是在算数。

### Flash Attention 的核心洞察

两个关键想法：

1. **分块计算**：不生成完整的 N*N 矩阵，而是把 Q/K/V 切成小块。每次只在 GPU 的高速缓存（SRAM，比 HBM 快 10 倍以上）中计算一小块注意力，算完就丢弃中间结果。
2. **在线 softmax**：分块后没法一次看到所有分数来做 softmax。Flash Attention 用一个巧妙的数学技巧，在遍历每个块的过程中**增量更新** softmax 的分母，最终结果与标准注意力完全一致（不是近似）。

```
Flash Attention 核心逻辑（伪代码）：

1. 初始化
   output = 全零矩阵(N, d)
   row_max = 全负无穷(N)        -- 在线 softmax: 跟踪每行的最大值
   row_sum = 全零(N)             -- 在线 softmax: 跟踪每行的指数和

2. 分块遍历 K, V（每块大小 block_size）
   for block in split(K, V, block_size):
       -- 只在 SRAM（高速缓存）中计算这一小块
       scores = Q * block.K^T / sqrt(d)

       -- 在线更新 softmax 统计量（增量式，不需要全局信息）
       new_max = max(row_max, scores.max())
       old_scale = exp(row_max - new_max)          -- 修正之前累积的结果
       exp_scores = exp(scores - new_max)
       output = output * old_scale + exp_scores * block.V
       row_sum = row_sum * old_scale + exp_scores.sum()
       row_max = new_max

3. 最终归一化
   return output / row_sum
```

**显存对比**：

| | 标准注意力 | Flash Attention |
|---|---|---|
| 显存占用 | O(N^2) -- 存完整注意力矩阵 | O(N) -- 只存分块的中间结果 |
| N=100K 时 | ~40GB（放不下） | ~400KB（轻松放入 SRAM） |
| 计算结果 | 精确 | 精确（不是近似） |

### 对你的实际影响

你不需要实现 Flash Attention -- PyTorch 2.0+ 和所有主流推理引擎（vLLM、TGI、llama.cpp）都已内置。但你需要关注：

- **选模型/引擎时**：确认是否支持 Flash Attention（或其变体 Flash Attention 2/3），这直接决定了能处理多长的上下文
- **遇到 OOM 时**：如果长文本推理爆显存，首先检查是否启用了 Flash Attention，而不是盲目加显卡
- **理解上下文长度的成本**：即使有 Flash Attention，计算量仍然是 O(N^2)，只是显存从 O(N^2) 降到了 O(N)。200K 上下文仍然比 4K 慢很多

## 模型量化

模型的参数通常以 FP16（16位浮点数）存储。量化就是把参数压缩到更少的位数（如 INT8、INT4），减少显存占用和计算量，代价是精度略有下降。

### 量化的直觉

量化的核心思想非常简单：把高精度浮点数"四舍五入"到低精度整数，就像把精确的 GPS 坐标（39.9042, 116.4074）简化为网格编号（A3 区）。

具体来说，INT8 量化的过程分三步：

1. **找值域**：扫描所有权重，找到最小值和最大值，确定数值范围
2. **映射**：将浮点范围线性映射到 INT8 的 256 个整数值（0~255），计算一个缩放因子（scale）和零点偏移（zero_point）
3. **取整存储**：每个权重 = round(原始值 / scale + zero_point)，存为 INT8

推理时反过来：INT8 值还原为 `(整数 - zero_point) * scale`，得到近似的浮点数。这个过程会引入量化误差，但对于大多数权重来说误差很小，模型质量影响有限。

```
量化示例：

原始权重 (FP32):  [-0.82, 0.47, 1.53, -0.23, 0.91]
值域: [-0.82, 1.53], scale = (1.53 - (-0.82)) / 255 ≈ 0.0092

量化 (INT8):      [0, 140, 255, 64, 188]       -- 存储空间缩小 4 倍
反量化 (FP32):    [-0.82, 0.47, 1.53, -0.23, 0.91]  -- 近似还原，误差 < 0.01
```

### 量化方案对比

先说结论 -- 根据你的场景直接选：

::: tip 量化方案速查
- **在 Mac M2/M3 16GB 上跑本地模型**：选 GGUF Q4_K_M，用 Ollama 一键运行，7B 模型只需 ~4GB 内存
- **有云端 GPU（如 A100），想省点显存多留给上下文**：选 INT8（GPTQ），质量损失极小，显存砍半
- **只有 8GB 消费级显卡（如 RTX 4060），想跑 7B 模型**：选 INT4（GPTQ/AWQ），3.5GB 刚好塞得下
- **什么都不想管，只想用好模型**：直接用 API（Claude / GPT），量化是别人的事
:::

下面是完整的方案对比，帮你理解各选项的 trade-off：

| 方案 | 位数 | 7B 模型大小 | 质量损失 | 适用场景 |
|------|------|-----------|---------|---------|
| FP32 | 32 | 28 GB | 无 | 训练 |
| FP16 | 16 | 14 GB | 极小 | 标准推理 |
| INT8 | 8 | 7 GB | 很小 | 高效推理 |
| INT4 (GPTQ/AWQ) | 4 | 3.5 GB | 可接受 | 消费级 GPU 部署 |
| GGUF Q4_K_M | ~4.5 | 4 GB | 较小 | CPU/Mac 推理（Ollama） |

### 本地跑量化模型

```bash
# 用 Ollama 跑量化模型（最简单的方式）
# 安装 Ollama: https://ollama.com

# 下载并运行 Llama 3 的 Q4 量化版
ollama run llama3         # 约 4GB 显存

# 下载更小的量化版
ollama run llama3:7b-q4   # INT4 量化，约 3.5GB
```

```typescript
// 用 TypeScript 调用本地 Ollama 模型（兼容 OpenAI 接口）
import OpenAI from "openai";

const client = new OpenAI({
  apiKey: "ollama",
  baseURL: "http://localhost:11434/v1",
});

const response = await client.chat.completions.create({
  model: "llama3",
  messages: [{ role: "user", content: "你好" }],
  max_tokens: 256,
});

console.log(response.choices[0].message.content);
```

## 小结

1. **理解注意力组件** 帮助你从架构层面理解 Q/K/V 的计算过程、因果掩码的作用，以及单头到多头的递进关系
2. **KV Cache** 是推理加速的关键优化，缓存已计算的 K/V 避免重复计算，用空间换时间
3. **Flash Attention** 通过分块计算将注意力的显存占用从 O(N^2) 降到 O(N)，是长上下文推理的关键
4. **模型量化** 通过降低精度（FP16->INT8->INT4）减少显存占用，使大模型能在消费级硬件上运行
5. 这些优化你不需要自己实现，但理解原理能帮你做架构决策和性能调优

## 练习

1. **因果掩码推演**：手动画出一个 4*4 的因果掩码矩阵，并模拟 softmax 后的注意力权重分布。第 3 个 Token 能看到哪些位置？权重如何分配？

2. **KV Cache 计算**：假设一个模型有 32 层，每层 32 个注意力头，每头维度 128，用 FP16 存储。计算处理 100K Token 时 KV Cache 的显存占用（单位 GB）。

3. **量化选型**：你有一台 MacBook Pro M3 16GB 内存，想本地运行一个 13B 参数的模型。参考量化方案对比表，分析应该选择哪种量化方案，为什么？

4. **思考题**：为什么 KV Cache 只缓存 K 和 V，不缓存 Q？（提示：思考自回归生成时，每步的 Q 是什么）

## 参考资源

- [FlashAttention 论文](https://arxiv.org/abs/2205.14135)
- [GPTQ 量化论文](https://arxiv.org/abs/2210.17323)
- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/)
- [Ollama 官网](https://ollama.com/)
- [llama.cpp（GGUF 格式量化工具）](https://github.com/ggerganov/llama.cpp)
