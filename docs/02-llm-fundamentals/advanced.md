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

把中级篇的公式变成完整的可运行代码，从头实现一个简化版 Transformer 的前向传播。

### 完整的自注意力层

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

### 完整的多头注意力

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

### 简化版 Transformer Block

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

标准注意力的问题是需要存储完整的 `(seq_len, seq_len)` 注意力矩阵。当序列长度 N=100K 时，这个矩阵有 100 亿个元素，GPU 显存根本放不下。

Flash Attention 的核心思想是**分块计算**：不存储完整的注意力矩阵，而是分块计算，每块算完就丢弃中间结果。

```python
import numpy as np

def standard_attention(Q, K, V):
    """标准注意力 -- 需要 O(N^2) 显存"""
    N = Q.shape[0]
    # 这里创建了 N*N 的矩阵，显存 O(N^2)
    scores = Q @ K.T / np.sqrt(Q.shape[-1])
    weights = np.exp(scores) / np.exp(scores).sum(axis=-1, keepdims=True)
    return weights @ V

def flash_attention_simplified(Q, K, V, block_size: int = 2):
    """Flash Attention 的简化演示（仅展示思想，非真实实现）"""
    N = Q.shape[0]
    d = Q.shape[-1]
    output = np.zeros_like(V)
    row_max = np.full(N, -np.inf)
    row_sum = np.zeros(N)

    # 分块处理 -- 每次只加载一小块到"快速显存"
    for j_start in range(0, N, block_size):
        j_end = min(j_start + block_size, N)

        # 只计算一小块的注意力分数
        K_block = K[j_start:j_end]
        V_block = V[j_start:j_end]

        scores_block = Q @ K_block.T / np.sqrt(d)  # (N, block_size)

        # 在线 softmax 更新（数值稳定版）
        block_max = scores_block.max(axis=-1)
        new_max = np.maximum(row_max, block_max)

        # 用新的 max 修正之前的累积和
        old_scale = np.exp(row_max - new_max)
        new_scale = np.exp(block_max - new_max)

        exp_scores = np.exp(scores_block - block_max[:, np.newaxis])
        block_sum = exp_scores.sum(axis=-1)

        # 更新输出（在线 softmax：累积未归一化的加权和，最后再除以总和）
        output = output * old_scale[:, np.newaxis] + exp_scores @ V_block

        # 更新统计量
        row_max = new_max
        row_sum = row_sum * old_scale + block_sum

    # 最终归一化
    return output / row_sum[:, np.newaxis]

# 关键对比
N = 1000
print(f"序列长度 N = {N}")
print(f"标准注意力显存: O(N^2) = {N*N:,} 个浮点数")
print(f"Flash Attention 显存: O(N) = {N:,} 个浮点数")
print(f"显存节省: {(1 - N/(N*N))*100:.1f}%")
```

::: warning 关于 Flash Attention
上面的代码是概念演示。真正的 Flash Attention 是用 CUDA 写的 GPU kernel，利用了 GPU 内存层次（SRAM vs HBM）的速度差异。你不需要自己实现它 -- PyTorch 和推理引擎已经内置了。理解原理就够了。
:::

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
