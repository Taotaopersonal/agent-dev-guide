# LLM 原理 · 高级

::: info 学习目标
- 用 numpy 手写完整的注意力计算，从代码层面理解 Transformer
- 理解 KV Cache 优化的原理和必要性
- 了解 Flash Attention 的核心思想
- 掌握模型量化的基本概念和实践
- 学完能理解推理引擎的底层优化原理

预计学习时间：3-4 小时
:::

## 用 numpy 手写注意力

::: tip 为什么要手写这些代码？
不是为了让你在生产中用 -- PyTorch 一行 `nn.MultiheadAttention` 就能搞定。手写的目的是帮你建立三个工程直觉：

1. **为什么注意力是 O(n^2) 的** -- 看到 `Q @ K.T` 生成 `(seq_len, seq_len)` 矩阵的那一刻，你就明白长上下文为什么烧显存
2. **为什么多头比单头好** -- 单头只能学一种关注模式，多头让模型同时关注语法、语义、位置等不同维度
3. **为什么 Transformer 可以并行训练** -- 不像 RNN 必须串行，注意力对所有位置的计算是一次矩阵乘法完成的
:::

下面我们从最小的自注意力开始，逐步搭建到完整的 Transformer Block。

### 第一步：单头自注意力 -- 理解 O(n^2) 的根源

```python
import numpy as np

class SelfAttention:
    """自注意力层 -- 从零实现"""

    def __init__(self, d_model: int, d_k: int):
        """
        d_model: 输入维度
        d_k: Query/Key/Value 的维度
        """
        self.d_k = d_k
        # 初始化权重矩阵（实际模型中这些是通过训练学到的）
        self.W_q = np.random.randn(d_model, d_k) * 0.1
        self.W_k = np.random.randn(d_model, d_k) * 0.1
        self.W_v = np.random.randn(d_model, d_k) * 0.1

    def forward(self, x: np.ndarray, mask: np.ndarray = None) -> tuple:
        """
        x: (seq_len, d_model) 输入序列
        mask: (seq_len, seq_len) 因果掩码（可选）
        返回: (output, attention_weights)
        """
        # 步骤 1: 线性投影，得到 Q, K, V
        Q = x @ self.W_q  # (seq_len, d_k)
        K = x @ self.W_k  # (seq_len, d_k)
        V = x @ self.W_v  # (seq_len, d_k)

        # 步骤 2: 计算注意力分数
        scores = Q @ K.T / np.sqrt(self.d_k)  # (seq_len, seq_len)

        # 步骤 3: 应用因果掩码（自回归生成时使用）
        if mask is not None:
            scores = scores + mask  # mask 中 -inf 的位置会被 softmax 变成 0

        # 步骤 4: Softmax
        scores_max = scores.max(axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)

        # 步骤 5: 加权求和
        output = weights @ V  # (seq_len, d_k)

        return output, weights


def create_causal_mask(seq_len: int) -> np.ndarray:
    """创建因果掩码 -- 每个位置只能看到自己和之前的位置"""
    mask = np.full((seq_len, seq_len), -np.inf)
    mask = np.triu(mask, k=1)  # 上三角为 -inf，下三角和对角线为 0
    return mask


# 演示
np.random.seed(42)
seq_len = 5
d_model = 8
d_k = 4

# 模拟 5 个 Token 的嵌入
x = np.random.randn(seq_len, d_model)

# 不带掩码（双向注意力，如 BERT）
attn = SelfAttention(d_model, d_k)
output, weights = attn.forward(x)
print("双向注意力权重:")
print(weights.round(3))

# 带因果掩码（单向注意力，如 GPT/Claude）
mask = create_causal_mask(seq_len)
output_causal, weights_causal = attn.forward(x, mask=mask)
print("\n因果注意力权重（每行只能看到自己和左边）:")
print(weights_causal.round(3))
```

> **工程直觉**：注意第 48 行 `Q @ K.T` -- 这一步生成了 `(seq_len, seq_len)` 的矩阵，这就是注意力 O(n^2) 复杂度的根源。序列长度从 4K 翻倍到 8K，这个矩阵的大小翻 4 倍。当你看到某个模型宣称支持 200K 上下文时，你应该立刻想到：它一定用了某种方式绕过了这个 N^2 瓶颈（比如后面要讲的 Flash Attention）。

### 第二步：多头注意力 -- 单头的局限是什么？

上面的单头注意力有一个根本问题：**一组 Q/K/V 权重只能学到一种关注模式**。比如它可能学会了关注"最近的几个词"，但同时就没法关注"句子开头的主语"。

多头注意力的解决方案很直接：并行运行多个独立的注意力头，每个头学习不同的关注模式，最后把结果拼起来。

```python
import numpy as np

class MultiHeadAttention:
    """多头注意力 -- 完整实现"""

    def __init__(self, d_model: int, num_heads: int):
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 每个头有独立的 Q/K/V 投影矩阵
        self.heads = [
            SelfAttention(d_model, self.d_k)
            for _ in range(num_heads)
        ]
        # 输出投影矩阵
        self.W_o = np.random.randn(d_model, d_model) * 0.1

    def forward(self, x: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        # 每个头独立计算注意力
        head_outputs = []
        for head in self.heads:
            output, _ = head.forward(x, mask)
            head_outputs.append(output)

        # 拼接所有头的输出
        concat = np.concatenate(head_outputs, axis=-1)  # (seq_len, d_model)

        # 最终线性投影
        output = concat @ self.W_o
        return output


# 演示
np.random.seed(42)
mha = MultiHeadAttention(d_model=16, num_heads=4)
x = np.random.randn(5, 16)
mask = create_causal_mask(5)
output = mha.forward(x, mask)
print(f"输入形状: {x.shape}")
print(f"输出形状: {output.shape}")
print(f"输出前两个 Token 的前 4 维:\n{output[:2, :4].round(3)}")
```

> **工程直觉**：注意每个头独立计算、互不依赖（第 118-120 行的循环），这意味着多头在 GPU 上可以完全并行。实际模型不会用 for 循环，而是把多头 reshape 成一个大的 batch 矩阵乘法一次算完。另外，`d_model // num_heads` 这个设计很精妙 -- 多头不增加总计算量，只是把同样的维度切分给不同的头，让每个头专注于子空间。

### 第三步：完整的 Transformer Block -- 光有注意力够吗？

注意力层只解决了"Token 之间怎么交互信息"的问题，但还缺两样东西：

- **前馈网络（FFN）**：注意力是线性加权求和，表达能力有限。FFN 提供逐位置的非线性变换，让模型有能力做更复杂的特征提取。你可以理解为：注意力负责"看哪里"，FFN 负责"看到之后怎么想"。
- **残差连接 + LayerNorm**：没有残差连接，深层网络的梯度会消失，训练不动。LayerNorm 让每一层的输入分布稳定，加速收敛。

```python
import numpy as np

def layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Layer Normalization"""
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return (x - mean) / (std + eps)

class FeedForward:
    """前馈神经网络"""
    def __init__(self, d_model: int, d_ff: int):
        self.W1 = np.random.randn(d_model, d_ff) * 0.1
        self.W2 = np.random.randn(d_ff, d_model) * 0.1

    def forward(self, x: np.ndarray) -> np.ndarray:
        # ReLU 激活
        hidden = np.maximum(0, x @ self.W1)
        return hidden @ self.W2

class TransformerBlock:
    """一个 Transformer 层"""
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)

    def forward(self, x: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        # 自注意力 + 残差连接 + LayerNorm
        attn_output = self.attention.forward(x, mask)
        x = layer_norm(x + attn_output)

        # 前馈网络 + 残差连接 + LayerNorm
        ffn_output = self.ffn.forward(x)
        x = layer_norm(x + ffn_output)

        return x


# 演示：两层 Transformer
np.random.seed(42)
d_model = 32
block1 = TransformerBlock(d_model=d_model, num_heads=4, d_ff=64)
block2 = TransformerBlock(d_model=d_model, num_heads=4, d_ff=64)

x = np.random.randn(5, d_model)  # 5 个 Token
mask = create_causal_mask(5)

h = block1.forward(x, mask)
h = block2.forward(h, mask)
print(f"两层 Transformer 输出形状: {h.shape}")
print(f"最后一个 Token 的表示（前 8 维）: {h[-1, :8].round(3)}")
```

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

```python
import numpy as np

class AttentionWithKVCache:
    """带 KV Cache 的注意力"""

    def __init__(self, d_model: int, d_k: int):
        self.d_k = d_k
        self.W_q = np.random.randn(d_model, d_k) * 0.1
        self.W_k = np.random.randn(d_model, d_k) * 0.1
        self.W_v = np.random.randn(d_model, d_k) * 0.1

        # KV Cache
        self.k_cache: list[np.ndarray] = []
        self.v_cache: list[np.ndarray] = []

    def forward_with_cache(self, x_new: np.ndarray) -> np.ndarray:
        """
        只处理新 Token，复用之前的 KV Cache
        x_new: (1, d_model) 新 Token 的嵌入
        """
        # 只为新 Token 计算 Q, K, V
        q_new = x_new @ self.W_q  # (1, d_k)
        k_new = x_new @ self.W_k  # (1, d_k)
        v_new = x_new @ self.W_v  # (1, d_k)

        # 追加到 Cache
        self.k_cache.append(k_new)
        self.v_cache.append(v_new)

        # 用所有 cached K, V 计算注意力
        K_all = np.vstack(self.k_cache)  # (seq_len, d_k)
        V_all = np.vstack(self.v_cache)  # (seq_len, d_k)

        # 新 Token 的 Q 与所有 K 计算注意力
        scores = q_new @ K_all.T / np.sqrt(self.d_k)
        weights = np.exp(scores) / np.exp(scores).sum(axis=-1, keepdims=True)
        output = weights @ V_all

        return output

    def reset_cache(self):
        self.k_cache = []
        self.v_cache = []


# 对比有无 KV Cache 的计算量
np.random.seed(42)
d_model, d_k = 8, 4
attn_cached = AttentionWithKVCache(d_model, d_k)

tokens = np.random.randn(5, d_model)  # 5 个 Token

print("逐 Token 生成（带 KV Cache）:")
for i, token in enumerate(tokens):
    output = attn_cached.forward_with_cache(token.reshape(1, -1))
    k_ops = (i + 1) * d_k  # 本步的计算量（与所有 K 的点积）
    print(f"  Token {i+1}: cache 大小={i+1}, 点积计算量={k_ops}")

print(f"\n无 Cache 时 Token 5 的计算量: {5 * 5 * d_k}")
print(f"有 Cache 时 Token 5 的计算量: {1 * 5 * d_k}")
print(f"节省: {(1 - 1*5/(5*5)) * 100:.0f}%")
```

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

```python
# 伪代码：Flash Attention 的核心逻辑
# （真实实现是 CUDA kernel，这里只展示思路）

def flash_attention(Q, K, V, block_size):
    output = zeros(N, d)
    row_max = full(N, -inf)    # 在线 softmax: 跟踪每行的最大值
    row_sum = zeros(N)          # 在线 softmax: 跟踪每行的指数和

    for block in split(K, V, block_size):       # 分块遍历 K, V
        scores = Q @ block.K.T / sqrt(d)        # 只在 SRAM 中计算小块
        # 在线更新 softmax 统计量（增量式，不需要全局信息）
        new_max = max(row_max, scores.max())
        old_scale = exp(row_max - new_max)       # 修正之前累积的结果
        exp_scores = exp(scores - new_max)
        output = output * old_scale + exp_scores @ block.V
        row_sum = row_sum * old_scale + exp_scores.sum()
        row_max = new_max

    return output / row_sum                      # 最终归一化
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

```python
import numpy as np

def quantize_int8(weights: np.ndarray) -> tuple:
    """将 FP32 权重量化为 INT8"""
    # 找到值域范围
    w_min, w_max = weights.min(), weights.max()
    scale = (w_max - w_min) / 255  # INT8 有 256 个值
    zero_point = round(-w_min / scale)

    # 量化：FP32 -> INT8
    quantized = np.round(weights / scale + zero_point).astype(np.int8)

    return quantized, scale, zero_point

def dequantize_int8(quantized: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    """将 INT8 反量化回 FP32"""
    return (quantized.astype(np.float32) - zero_point) * scale

# 演示
np.random.seed(42)
original = np.random.randn(4, 4).astype(np.float32)

# 量化
quantized, scale, zp = quantize_int8(original)
# 反量化
recovered = dequantize_int8(quantized, scale, zp)

print("原始权重 (FP32):")
print(original.round(4))
print(f"\n量化后 (INT8):")
print(quantized)
print(f"\n反量化后:")
print(recovered.round(4))
print(f"\n量化误差 (MAE): {np.abs(original - recovered).mean():.6f}")
print(f"存储节省: FP32={original.nbytes}B -> INT8={quantized.nbytes}B ({quantized.nbytes/original.nbytes*100:.0f}%)")
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

```python
# 主要量化方案
QUANTIZATION = {
    "FP32": {
        "bits": 32,
        "7B_model_size_gb": 28,
        "quality_loss": "无",
        "use_case": "训练",
    },
    "FP16": {
        "bits": 16,
        "7B_model_size_gb": 14,
        "quality_loss": "极小",
        "use_case": "标准推理",
    },
    "INT8": {
        "bits": 8,
        "7B_model_size_gb": 7,
        "quality_loss": "很小",
        "use_case": "高效推理",
    },
    "INT4 (GPTQ/AWQ)": {
        "bits": 4,
        "7B_model_size_gb": 3.5,
        "quality_loss": "可接受",
        "use_case": "消费级GPU部署",
    },
    "GGUF Q4_K_M": {
        "bits": 4.5,
        "7B_model_size_gb": 4,
        "quality_loss": "较小",
        "use_case": "CPU/Mac 推理（Ollama）",
    },
}

print(f"{'方案':<20} {'位数':>4} {'7B模型大小':>10} {'质量损失':>8} {'适用场景'}")
print("-" * 70)
for name, info in QUANTIZATION.items():
    print(f"{name:<20} {info['bits']:>4} {info['7B_model_size_gb']:>8.1f}GB {info['quality_loss']:>8} {info['use_case']}")
```

### 本地跑量化模型

```bash
# 用 Ollama 跑量化模型（最简单的方式）
# 安装 Ollama: https://ollama.com

# 下载并运行 Llama 3 的 Q4 量化版
ollama run llama3         # 约 4GB 显存

# 下载更小的量化版
ollama run llama3:7b-q4   # INT4 量化，约 3.5GB
```

```python
# 用 Python 调用本地模型（兼容 OpenAI 接口）
from openai import OpenAI

client = OpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1",
)

response = client.chat.completions.create(
    model="llama3",
    messages=[{"role": "user", "content": "你好"}],
    max_tokens=256,
)
print(response.choices[0].message.content)
```

## 小结

1. **手写注意力** 帮助你从代码层面理解 Q/K/V 的计算过程和因果掩码的作用
2. **KV Cache** 是推理加速的关键优化，缓存已计算的 K/V 避免重复计算，用空间换时间
3. **Flash Attention** 通过分块计算将注意力的显存占用从 O(N^2) 降到 O(N)，是长上下文推理的关键
4. **模型量化** 通过降低精度（FP16->INT8->INT4）减少显存占用，使大模型能在消费级硬件上运行
5. 这些优化你不需要自己实现，但理解原理能帮你做架构决策和性能调优

## 练习

1. **注意力实现**：在 `SelfAttention` 基础上，实现带 dropout 的注意力（训练时随机丢弃一些注意力权重），验证 dropout rate=0.1 时输出的变化。

2. **KV Cache 计算**：假设一个模型有 32 层，每层 32 个注意力头，每头维度 128，用 FP16 存储。计算处理 100K Token 时 KV Cache 的显存占用（单位 GB）。

3. **量化实验**：对一个随机矩阵分别做 INT8 和 INT4 量化，计算两者的量化误差（MAE），验证位数越低误差越大。

4. **思考题**：为什么 KV Cache 只缓存 K 和 V，不缓存 Q？（提示：思考自回归生成时，每步的 Q 是什么）

## 参考资源

- [FlashAttention 论文](https://arxiv.org/abs/2205.14135)
- [GPTQ 量化论文](https://arxiv.org/abs/2210.17323)
- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/)
- [Ollama 官网](https://ollama.com/)
- [llama.cpp（GGUF 格式量化工具）](https://github.com/ggerganov/llama.cpp)
