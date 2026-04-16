# LLM 原理 · 初级

::: info 学习目标
- 理解大语言模型（LLM）是什么，用通俗的方式解释它的工作原理
- 掌握 Token 的概念，学会用 js-tiktoken 实际操作
- 了解主流模型（Claude/GPT/开源模型）的特点和选择策略
- 理解"幻觉"问题的成因和应对方式
- 学完能理解 AI 的基本工作方式，在技术讨论中不露怯

预计学习时间：2-3 小时
:::

## LLM 是什么

大语言模型（Large Language Model）本质上是一个**超级文字接龙机器**。

你输入一段文字，它会预测"接下来最可能出现什么文字"，然后一个 Token 一个 Token 地往后生成。就这么简单 -- 但当模型足够大（几千亿参数）、训练数据足够多（几乎整个互联网）时，这个"接龙"能力就涌现出了惊人的效果：写代码、做翻译、逻辑推理、角色扮演。

```
你的输入: "中国的首都是"
模型预测: "北" -> "京" -> "。" -> [结束]

你的输入: "写一个 Python 函数计算斐波那契数列"
模型预测: "def" -> " fib" -> "(" -> "n" -> ")" -> ...
```

::: tip 核心直觉
LLM 不是"理解"了你的问题然后"思考"出答案。它是在所有训练数据中学到了"这种上下文之后，最可能出现什么"的统计规律。但当这个统计规律足够精细时，它的表现看起来就像在"理解"和"思考"。
:::

## Token 和分词

### 什么是 Token

Token 是 LLM 处理文本的最小单位。模型不是按字或按词读取文本的，而是按 Token。一个 Token 大约是：

- **英文**：大约 3-4 个字符，或者 0.75 个单词
- **中文**：大约 1-2 个汉字

```
"Hello, world!" -> ["Hello", ",", " world", "!"]  = 4 tokens
"你好世界" -> ["你好", "世界"]  = 2 tokens (大约)
```

Token 直接影响两件事：
1. **成本**：API 按 Token 计费，Token 越多越贵
2. **上下文窗口**：每个模型有 Token 上限（如 200K），超过就塞不下了

### 用 js-tiktoken 实际看看

```typescript
/** 用 js-tiktoken 理解 Token 分词 */

// npm install js-tiktoken
import { encodingForModel } from "js-tiktoken";

// cl100k_base 是 GPT-4 使用的编码器
// Claude 有自己的分词器，但概念一致
const enc = encodingForModel("gpt-4o");

// 英文分词
const textEn = "Hello, how are you doing today?";
const tokensEn = enc.encode(textEn);
console.log(`英文: '${textEn}'`);
console.log(`Token 数: ${tokensEn.length}`);
console.log(`Token IDs: [${tokensEn}]`);
console.log(`每个 Token 对应的文字:`);
for (const t of tokensEn) {
  console.log(`  ${t} -> '${new TextDecoder().decode(enc.decode([t]))}'`);
}

console.log();

// 中文分词
const textCn = "你好，今天天气怎么样？";
const tokensCn = enc.encode(textCn);
console.log(`中文: '${textCn}'`);
console.log(`Token 数: ${tokensCn.length}`);
console.log(`每个 Token 对应的文字:`);
for (const t of tokensCn) {
  console.log(`  ${t} -> '${new TextDecoder().decode(enc.decode([t]))}'`);
}

console.log();

// 代码分词
const code = "function fibonacci(n) {\n  if (n < 2) return n;\n  return fibonacci(n - 1) + fibonacci(n - 2);\n}";
const tokensCode = enc.encode(code);
console.log(`代码 Token 数: ${tokensCode.length}`);
console.log(`每个字符约 ${(tokensCode.length / code.length).toFixed(2)} 个 Token`);
```

### Token 成本速算

```typescript
// npm install js-tiktoken
import { encodingForModel } from "js-tiktoken";

interface CostEstimate {
  inputTokens: number;
  outputTokens: number;
  inputCostUsd: string;
  outputCostUsd: string;
  totalCostUsd: string;
}

function estimateCost(
  inputText: string,
  outputTokens: number = 500,
  model: string = "claude-sonnet"
): CostEstimate {
  /** 估算一次 API 调用的成本 */
  const enc = encodingForModel("gpt-4o");
  const inputTokens = enc.encode(inputText).length;

  // 2026 年大致价格（美元 / 百万 Token）
  const pricing: Record<string, { input: number; output: number }> = {
    "claude-sonnet": { input: 3.0, output: 15.0 },
    "claude-haiku": { input: 0.25, output: 1.25 },
    "gpt-4o": { input: 2.5, output: 10.0 },
  };

  const price = pricing[model] ?? pricing["claude-sonnet"];
  const inputCost = (inputTokens * price.input) / 1_000_000;
  const outputCost = (outputTokens * price.output) / 1_000_000;
  const total = inputCost + outputCost;

  return {
    inputTokens,
    outputTokens,
    inputCostUsd: `$${inputCost.toFixed(6)}`,
    outputCostUsd: `$${outputCost.toFixed(6)}`,
    totalCostUsd: `$${total.toFixed(6)}`,
  };
}

// 一次普通对话
console.log(estimateCost("请帮我写一个排序函数", 200));

// 一次长文档分析
const longDoc = "这是一份技术文档。".repeat(5000); // 约 5000 字
console.log(estimateCost(longDoc, 1000));
```

## 主流模型对比

### 先解决一个问题：模型那么多，我该怎么选？

初学者面对模型选择时最常见的困境是：列表看了一大堆，每个都说自己好，然后陷入"选择焦虑"。

其实 90% 的场景只需要回答**一个问题**：

```
你的数据能不能发到外部 API？
├── 不能 → 开源模型本地部署（Llama / DeepSeek / Qwen）
└── 能 → 你的任务有多复杂？
    ├── 复杂推理/编程 → Claude Opus 4 或 o3
    ├── 简单分类/提取/高并发 → Claude Haiku 3.5 或 GPT-4o-mini
    └── 日常开发/原型期 → Claude Sonnet 4 或 GPT-4o（先跑通再说）
```

::: tip 作为初学者，你现在只需要记住这一点
**开发阶段直接用 Claude Sonnet 或 GPT-4o**。它们能力够强、价格适中、调试方便。等你的应用跑通了，再根据成本和性能需求换模型。别在起步阶段纠结选型 -- 模型随时可以换，但迟迟不动手写代码才是最大的浪费。
:::

### 三大阵营一览

有了选择锚点后，再简单了解一下各家模型的核心差异：

**Claude（Anthropic）** -- 指令遵循度高，Agent 场景首选

```
Opus 4   -> 最强推理，复杂任务      （贵）
Sonnet 4 -> 能力与速度平衡          （日常推荐）
Haiku 3.5 -> 最快，简单任务和高并发  （便宜）
```

核心优势：200K 长上下文、工具调用（Tool Use）能力强、代码生成质量高。

**GPT（OpenAI）** -- 生态最成熟，兼容性最广

```
GPT-4o      -> 多模态旗舰
GPT-4o-mini -> 轻量性价比
o1 / o3     -> 推理增强，数学/编程突出
```

核心优势：第三方工具兼容性最好、多模态能力（图片/音频/视频）、结构化输出支持成熟。

**开源模型** -- 数据不出域，成本可控

```
Llama 3 / DeepSeek V3 / Qwen 2.5 / Mistral
```

核心优势：本地部署数据安全、可微调适配垂直领域、大多兼容 OpenAI 接口格式。

## 幻觉问题

LLM 有一个著名的问题：**幻觉（Hallucination）**。它会非常自信地编造不存在的事实。

### 幻觉的成因

LLM 的"知识"来自训练数据中的统计规律。当它遇到不确定的问题时，不会说"我不知道"，而是会根据统计规律"编"一个看起来合理的答案。

```typescript
// npm install @anthropic-ai/sdk
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

// 容易产生幻觉的问题类型

// 1. 具体数字和日期
const response1 = await client.messages.create({
  model: "claude-sonnet-4-20250514",
  max_tokens: 256,
  messages: [{ role: "user", content: "2025年中国GDP是多少？" }],
});
console.log("具体数字:", (response1.content[0] as { type: "text"; text: string }).text.slice(0, 100));
// 模型可能编造一个看似合理的数字

// 2. 学术引用
const response2 = await client.messages.create({
  model: "claude-sonnet-4-20250514",
  max_tokens: 256,
  messages: [{ role: "user", content: "请给出3篇关于LLM幻觉问题的论文，包含作者和年份" }],
});
console.log("学术引用:", (response2.content[0] as { type: "text"; text: string }).text.slice(0, 200));
// 论文标题和作者可能是编的

// 3. 不存在的事物
const response3 = await client.messages.create({
  model: "claude-sonnet-4-20250514",
  max_tokens: 256,
  messages: [{ role: "user", content: "介绍一下TypeScript的frombulate库" }],
});
console.log("虚构库:", (response3.content[0] as { type: "text"; text: string }).text.slice(0, 200));
// 模型可能真的"介绍"一个不存在的库
```

### 应对幻觉的方法

```typescript
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

// 方法 1：在 Prompt 中要求诚实
const honestPrompt = `请回答以下问题。如果你不确定或不知道，请明确说"我不确定"，
不要编造答案。

问题：2025年中国GDP是多少？`;

// 方法 2：要求引用来源
const sourcedPrompt = `请回答以下问题，并标注信息来源。
如果无法确认来源，请标注"未经验证"。

问题：Transformer论文是哪一年发表的？`;

// 方法 3：RAG -- 提供参考资料
const ragPrompt = `根据以下参考资料回答问题。只使用参考资料中的信息，
不要添加资料中没有的内容。

参考资料：
---
Transformer 架构最初在 2017 年的论文 "Attention Is All You Need" 中提出，
作者包括 Vaswani 等人。该论文发表在 NeurIPS 2017。
---

问题：Transformer论文是哪一年发表的？谁写的？`;

const response = await client.messages.create({
  model: "claude-sonnet-4-20250514",
  max_tokens: 256,
  messages: [{ role: "user", content: ragPrompt }],
});
console.log((response.content[0] as { type: "text"; text: string }).text);
```

::: warning Agent 开发中的幻觉风险
在 Agent 系统中，幻觉问题更加严重 -- 如果 Agent "幻觉"出一个不存在的工具调用或错误的参数，整个流程就会出错。这就是为什么 Agent 开发中结构化输出、工具定义和验证机制如此重要。
:::

## 模型的核心参数

### 初学者最常踩的两个坑

在讲参数之前，先看两个真实的"翻车现场"——几乎每个初学者都会遇到：

**坑 1：回复莫名其妙被截断了**

你让模型写一篇技术总结，结果写到一半戛然而止，末尾甚至停在半句话中间。这不是模型"偷懒"，而是 `max_tokens` 设小了。模型生成到 token 上限就会强制停止，不管句子有没有写完。

**坑 2：模型开始胡言乱语**

你让模型提取 JSON 数据，结果它返回了格式混乱、字段名乱编的内容。这往往是因为 `temperature` 设得太高。高 temperature 意味着高随机性——对创意写作是好事，但对需要精确输出的任务是灾难。

### 两个关键参数

理解了上面的坑，参数就很好懂了：

```typescript
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

// max_tokens: 最大输出长度
// 设太小 -> 回复被截断（最常见的新手问题！）
// 设太大 -> 不影响质量，但预留了不必要的空间
const response1 = await client.messages.create({
  model: "claude-sonnet-4-20250514",
  max_tokens: 100, // 短回复
  messages: [{ role: "user", content: "写一首诗" }],
});

// temperature: 随机性（0-1）
// 0 -> 最确定，每次结果几乎一样，适合分类、提取、JSON 输出
// 0.7 -> 适度随机，适合创意写作、对话
// 1.0 -> 高随机，适合头脑风暴（但精确任务千万别用！）

// 确定性任务用低 temperature
const response2 = await client.messages.create({
  model: "claude-sonnet-4-20250514",
  max_tokens: 64,
  temperature: 0, // 确定性输出
  messages: [{ role: "user", content: "1+1等于几？只回答数字。" }],
});
console.log(`确定性: ${(response2.content[0] as { type: "text"; text: string }).text}`);

// 创意任务用高 temperature
const response3 = await client.messages.create({
  model: "claude-sonnet-4-20250514",
  max_tokens: 256,
  temperature: 0.9, // 高创意
  messages: [{ role: "user", content: "用一个比喻解释什么是递归" }],
});
console.log(`创意性: ${(response3.content[0] as { type: "text"; text: string }).text}`);
```

### 你的第一个参数配置建议

如果你刚开始接触 LLM API，不确定参数怎么填，用这个"安全默认值"起步：

```typescript
{
  model: "claude-sonnet-4-20250514",
  max_tokens: 1024,    // 足够大多数场景，不会被截断
  temperature: 0,      // 先求稳，需要创意时再调高
}
```

**工程直觉**：`temperature` 的默认策略是"先 0 后调"。开发阶段一律用 0，确保输出稳定可调试；等功能跑通后，只对明确需要多样性的场景（创意文案、头脑风暴）再逐步上调到 0.7-0.9。大多数 Agent 和工具调用场景，temperature 始终应该保持为 0。

## 小结

1. **LLM 是超级接龙机器** -- 根据上下文预测下一个 Token，足够多的参数和数据让它表现出"智能"
2. **Token 是 LLM 的基本单位** -- 影响成本和上下文窗口，中文约 1-2 字/Token
3. **选模型看场景** -- 复杂任务选大模型，简单任务选小模型，数据敏感选开源本地部署
4. **幻觉是固有问题** -- 通过诚实提示、来源标注、RAG 等方法缓解，不能完全消除
5. **temperature 是关键参数** -- 确定性任务用 0，创意任务用 0.7+

## 练习

1. **Token 实验**：用 js-tiktoken 分别统计一段中文文章（500 字）和一段英文文章（500 词）的 Token 数，计算中英文的 Token 密度差异。

2. **模型对比**：对同一个问题（如"解释什么是微服务"），分别用 `temperature=0`、`0.5`、`1.0` 各调用 3 次，观察并记录输出差异。

3. **幻觉测试**：设计 5 个容易引发幻觉的问题（具体数字、学术引用、虚构实体等），记录模型的回答，然后添加"不确定请说不知道"的约束后再测试，对比差异。

4. **成本估算**：假设你的 Agent 每天处理 1000 个用户请求，每个请求平均 500 input tokens + 300 output tokens，分别计算使用 Claude Sonnet 和 Claude Haiku 的日成本。

## 参考资源

- [Anthropic 模型文档](https://docs.anthropic.com/en/docs/about-claude/models)
- [OpenAI 模型概览](https://platform.openai.com/docs/models)
- [js-tiktoken 库](https://github.com/dqbd/tiktoken)
- [Attention Is All You Need（原始论文）](https://arxiv.org/abs/1706.03762)
