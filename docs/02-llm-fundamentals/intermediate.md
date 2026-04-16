# LLM 原理 · 中级

::: info 学习目标
- 用直觉理解 Transformer 架构的核心思想
- 搞清楚注意力机制中 Q/K/V 分别在干什么
- 理解位置编码和多头注意力的作用
- 掌握 Temperature 和 Top-p 采样策略的原理
- 学完能解释模型参数调优的原理，理解为什么某些设置能改善输出

预计学习时间：3-4 小时
:::

## Transformer 架构直觉

2017 年 Google 发表了 "Attention Is All You Need" 论文，提出了 Transformer 架构。如今几乎所有的 LLM（GPT、Claude、Llama）都基于 Transformer。

### 为什么需要 Transformer

在 Transformer 之前，处理序列数据（文本、语音）主要用 RNN（循环神经网络）。RNN 的问题是必须**一个词一个词地顺序处理**，像排队一样，前一个处理完才能处理下一个。这导致：

1. **训练慢**：无法并行计算
2. **长距离遗忘**：句子越长，前面的信息越容易丢失

Transformer 用**注意力机制**解决了这两个问题：每个词可以直接"关注"句子中的任何其他词，不需要排队，而且长距离依赖不再是问题。

```
RNN 处理方式（排队）:
"我" -> "今天" -> "去" -> "北京" -> "出差"
 ↓       ↓        ↓       ↓         ↓
 h1  ->  h2  ->  h3  ->  h4  ->   h5
（每步依赖前一步，无法并行）

Transformer 处理方式（并行 + 全局关注）:
"我" "今天" "去" "北京" "出差"
  ↕    ↕    ↕    ↕     ↕
  所有词同时互相关注
（完全并行，每个词都能直接看到所有其他词）
```

### 整体结构

一个用于生成文本的 LLM（如 GPT、Claude）使用的是 Transformer 的 **Decoder** 部分，核心结构是：

```
输入文本
   ↓
[Token 嵌入] -- 把文字转成向量
   ↓
[位置编码] -- 告诉模型每个词的位置
   ↓
┌─────────────────────┐
│  [多头自注意力]      │  ← 核心！每个词关注其他词
│         ↓            │
│  [前馈神经网络]      │  ← 对注意力结果做变换
│         ↓            │
│  (重复 N 层)         │  ← GPT-4 约有 100+ 层
└─────────────────────┘
   ↓
[输出层] -- 预测下一个 Token 的概率
```

## 注意力机制：Q/K/V

注意力机制是 Transformer 的灵魂。它的核心问题是：**当模型在处理某个词时，它应该关注句子中的哪些其他词？**

### 直觉理解

想象你在图书馆找资料：

- **Query（Q，查询）**：你脑子里的问题 -- "我想找关于 Python 异步编程的书"
- **Key（K，键）**：每本书封面上的标签 -- "Python 入门"、"异步编程实战"、"Java 设计模式"
- **Value（V，值）**：书的实际内容

查找过程：
1. 你用 **Query** 和每本书的 **Key** 做匹配，算出相关度
2. 相关度越高的书，你越重视它的 **Value**
3. 最终你得到的信息是所有书 **Value** 的加权组合

```python
# 注意力的伪代码
def attention(query, keys, values):
    # 1. 计算每个 key 与 query 的匹配分数
    scores = [dot_product(query, key) for key in keys]

    # 2. 用 softmax 转成权重（和为 1）
    weights = softmax(scores)

    # 3. 按权重加权求和 values
    output = sum(weight * value for weight, value in zip(weights, values))
    return output
```

### 在句子中的实际效果

```
句子: "小猫坐在垫子上，它很舒服"

当模型处理"它"这个词时：
  Query("它") vs Key("小猫") -> 高分！（"它"指代"小猫"）
  Query("它") vs Key("坐在") -> 低分
  Query("它") vs Key("垫子") -> 中分
  Query("它") vs Key("舒服") -> 低分

结果：模型理解了"它"指的是"小猫"
```

### 注意力公式

让我们把刚才图书馆的每一步翻译成数学语言。在图书馆里，你脑中的问题是 Query，每本书封面的标签是 Key，你用"问题和标签的匹配度"来决定该多认真读哪本书的内容（Value）。注意力公式做的事情完全一样——只不过"匹配度"用向量点积来计算，"该多认真读"用 softmax 概率来表示：

```
Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
```

- `Q * K^T`：计算所有 query 和 key 的匹配分数
- `/ sqrt(d_k)`：缩放因子，防止分数太大导致 softmax 输出过于极端
- `softmax`：把分数转成概率（加和为 1）
- `* V`：用概率加权 value

```python
import numpy as np

def attention(Q, K, V):
    """标准注意力计算"""
    d_k = K.shape[-1]  # key 的维度

    # 步骤 1: Q 和 K 的点积
    scores = Q @ K.T

    # 步骤 2: 缩放
    scores = scores / np.sqrt(d_k)

    # 步骤 3: softmax 转成权重
    weights = np.exp(scores) / np.exp(scores).sum(axis=-1, keepdims=True)

    # 步骤 4: 加权求和 V
    output = weights @ V

    return output, weights

# 示例：3 个 Token，每个 4 维
np.random.seed(42)
Q = np.random.randn(3, 4)  # 3 个查询
K = np.random.randn(3, 4)  # 3 个键
V = np.random.randn(3, 4)  # 3 个值

output, weights = attention(Q, K, V)
print("注意力权重:")
print(weights.round(3))
print("\n输出:")
print(output.round(3))
```

## 位置编码

Transformer 同时处理所有 Token（不像 RNN 有天然的顺序），所以它本身不知道词的先后顺序。"我喜欢你" 和 "你喜欢我" 的意思完全不同，但如果没有位置信息，Transformer 无法区分。

位置编码就是给每个位置生成一个独特的向量，加到 Token 的嵌入向量上，让模型知道"这是第几个词"。

**为什么用正弦和余弦？** 原始 Transformer 选择正弦/余弦函数来生成位置编码，背后的直觉是：不同频率的正弦波组合可以唯一标识任意位置——就像钟表用时针、分针、秒针三根不同转速的指针组合来编码时间一样。秒针转得快，区分相邻的秒；时针转得慢，区分不同的小时。位置编码里的不同维度扮演着类似的角色：高频维度区分相邻位置，低频维度捕捉远距离的位置结构。此外，正弦函数的周期性还带来一个好处——任意两个位置之间的相对偏移可以用线性变换表示，这让模型更容易学到"间隔 3 个词"这类相对位置关系。

```python
import numpy as np

def positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """原始 Transformer 的正弦位置编码"""
    pe = np.zeros((seq_len, d_model))
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    pe[:, 0::2] = np.sin(position * div_term)  # 偶数维度用 sin
    pe[:, 1::2] = np.cos(position * div_term)  # 奇数维度用 cos

    return pe

# 生成 10 个位置的编码，每个 64 维
pe = positional_encoding(10, 64)
print(f"位置编码形状: {pe.shape}")
print(f"位置 0 的前 8 维: {pe[0, :8].round(3)}")
print(f"位置 1 的前 8 维: {pe[1, :8].round(3)}")
# 不同位置的编码是不同的，模型能据此区分位置
```

::: tip 现代模型的位置编码
原始 Transformer 用正弦编码，但现代模型大多改用了 **RoPE**（旋转位置编码，Llama/Qwen 使用）或 **ALiBi**（注意力偏置）。它们要解决正弦编码的一个关键局限：正弦编码在训练时就固定了最大序列长度，面对超出训练长度的文本时无法有效外推。RoPE 通过将位置信息编码为向量旋转角度，使相对位置关系天然地融入注意力计算，在长度外推上表现更好；ALiBi 则更激进地抛弃了显式位置向量，直接根据 Token 之间的距离给注意力分数加上线性惩罚，从而天然支持任意长度的序列。
:::

## 多头注意力

一个注意力头只能捕捉一种类型的关系。多头注意力就是同时用多个注意力头，各自关注不同方面：

```
头 1: 关注语法关系（主语 -> 谓语）
头 2: 关注指代关系（"它" -> "小猫"）
头 3: 关注语义相似（"高兴" <-> "开心"）
头 4: 关注位置距离（相邻词的关系）
...
```

```python
import numpy as np

def multi_head_attention(Q, K, V, num_heads: int = 4):
    """多头注意力（简化版）"""
    d_model = Q.shape[-1]
    d_k = d_model // num_heads

    outputs = []
    all_weights = []

    for h in range(num_heads):
        # 每个头取特征的一个切片
        start = h * d_k
        end = start + d_k

        q_h = Q[:, start:end]
        k_h = K[:, start:end]
        v_h = V[:, start:end]

        # 单头注意力
        scores = q_h @ k_h.T / np.sqrt(d_k)
        weights = np.exp(scores) / np.exp(scores).sum(axis=-1, keepdims=True)
        output = weights @ v_h

        outputs.append(output)
        all_weights.append(weights)

    # 拼接所有头的输出
    concat_output = np.concatenate(outputs, axis=-1)

    print(f"头数: {num_heads}, 每头维度: {d_k}")
    for i, w in enumerate(all_weights):
        print(f"  头 {i+1} 注意力权重: {w[0].round(2)}")

    return concat_output

# 示例
np.random.seed(42)
seq_len, d_model = 4, 16
Q = np.random.randn(seq_len, d_model)
K = np.random.randn(seq_len, d_model)
V = np.random.randn(seq_len, d_model)

output = multi_head_attention(Q, K, V, num_heads=4)
print(f"输出形状: {output.shape}")
```

## 模型推理和采样策略

当 LLM 生成文本时，它实际上是在为"下一个 Token"计算一个概率分布。如何从这个概率分布中选择 Token，就是**采样策略**。

### Temperature

Temperature 控制概率分布的"锐度"：

```python
import numpy as np

def softmax_with_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """带 temperature 的 softmax"""
    scaled = logits / temperature
    exp_scaled = np.exp(scaled - scaled.max())  # 减去最大值防止溢出
    return exp_scaled / exp_scaled.sum()

# 模拟模型输出的原始分数（logits）
# 假设词汇表有 5 个 Token: ["好", "不错", "棒", "差", "一般"]
logits = np.array([2.0, 1.5, 1.8, 0.3, 0.8])
tokens = ["好", "不错", "棒", "差", "一般"]

print("不同 Temperature 下的概率分布:")
for temp in [0.1, 0.5, 1.0, 1.5]:
    probs = softmax_with_temperature(logits, temp)
    print(f"\n  Temperature = {temp}:")
    for token, prob in zip(tokens, probs):
        bar = "█" * int(prob * 50)
        print(f"    {token}: {prob:.3f} {bar}")

# Temperature 0.1 -> 几乎确定选"好"（概率最高的那个）
# Temperature 1.0 -> 各 Token 都有合理概率
# Temperature 1.5 -> 更平均，"差"和"一般"也有机会被选中
```

### Top-p（核采样）

Top-p 不是调整概率分布，而是截取概率最高的几个 Token，使它们的累积概率达到 p：

```python
import numpy as np

def top_p_sampling(probs: np.ndarray, p: float = 0.9) -> np.ndarray:
    """Top-p 采样"""
    # 按概率降序排列
    sorted_indices = np.argsort(-probs)
    sorted_probs = probs[sorted_indices]

    # 计算累积概率
    cumulative = np.cumsum(sorted_probs)

    # 找到累积概率超过 p 的位置
    cutoff_idx = np.searchsorted(cumulative, p) + 1

    # 只保留前 cutoff_idx 个 Token
    selected_indices = sorted_indices[:cutoff_idx]
    selected_probs = probs[selected_indices]

    # 重新归一化
    selected_probs = selected_probs / selected_probs.sum()

    return selected_indices, selected_probs

# 示例
probs = np.array([0.35, 0.25, 0.20, 0.10, 0.05, 0.03, 0.02])
tokens = ["好", "不错", "棒", "还行", "一般", "差", "烂"]

print("Top-p = 0.9 采样结果:")
indices, new_probs = top_p_sampling(probs, p=0.9)
for idx, prob in zip(indices, new_probs):
    print(f"  {tokens[idx]}: {prob:.3f}")
# "差"和"烂"被排除了（累积概率已经到 0.9）
```

### 实际建议

```python
# 不同任务的推荐设置

SETTINGS = {
    "分类/提取/判断": {
        "temperature": 0,
        "top_p": 1.0,
        "说明": "确定性输出，每次结果一致"
    },
    "代码生成": {
        "temperature": 0,
        "top_p": 1.0,
        "说明": "代码需要精确，不要随机"
    },
    "日常对话": {
        "temperature": 0.7,
        "top_p": 0.9,
        "说明": "适度多样化，但不离谱"
    },
    "创意写作": {
        "temperature": 0.9,
        "top_p": 0.95,
        "说明": "鼓励创新表达"
    },
    "头脑风暴": {
        "temperature": 1.0,
        "top_p": 0.95,
        "说明": "最大多样性，生成各种可能"
    },
}

for task, config in SETTINGS.items():
    print(f"{task}: temperature={config['temperature']}, top_p={config['top_p']}")
    print(f"  {config['说明']}")
```

::: warning temperature 和 top_p 不要同时调
通常只调其中一个，另一个保持默认。Anthropic 建议修改 temperature 时保持 top_p=1，反之亦然。同时调两个参数会让效果难以预测。
:::

## 小结

1. **Transformer** 用注意力机制替代了 RNN 的顺序处理，实现了并行计算和长距离依赖
2. **Q/K/V** 就是"查询-匹配-取值"：Query 描述需要什么，Key 描述每个位置有什么，Value 是实际内容
3. **位置编码** 解决了 Transformer 不知道词序的问题，现代模型多用 RoPE
4. **多头注意力** 让模型同时关注多种不同的关系（语法、指代、语义等）
5. **Temperature** 控制输出随机性，**Top-p** 截断低概率 Token；确定性任务用低值，创意任务用高值

## 练习

1. **注意力可视化**：用上面的 `attention` 函数，构造一个 5 个 Token 的例子，手动设置 Q/K 让第 3 个 Token 强烈关注第 1 个 Token，验证注意力权重。

2. **Temperature 实验**：对同一个创意写作 Prompt，用 temperature 0、0.5、1.0 各调用 3 次 Claude API，记录输出多样性的差异。

3. **Top-p 理解**：给定概率分布 `[0.4, 0.3, 0.15, 0.08, 0.04, 0.02, 0.01]`，手动计算 Top-p = 0.8 时会保留哪些 Token。

4. **思考题**：为什么多头注意力要把维度切分给各个头，而不是每个头都用完整维度？（提示：从计算量和参数效率角度思考）

## 参考资源

- [The Illustrated Transformer（经典图解）](https://jalammar.github.io/illustrated-transformer/)
- [Attention Is All You Need（原始论文）](https://arxiv.org/abs/1706.03762)
- [3Blue1Brown - 注意力机制可视化（视频）](https://www.youtube.com/watch?v=eMlx5fFNoYc)
- [The Annotated Transformer（带代码的论文解读）](https://nlp.seas.harvard.edu/annotated-transformer/)
