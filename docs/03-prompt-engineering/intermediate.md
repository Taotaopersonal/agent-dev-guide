# Prompt Engineering · 中级

::: info 学习目标
- 深入理解 Chain-of-Thought（CoT）和 Tree-of-Thought（ToT）推理技巧
- 掌握结构化输出（JSON / XML）的可靠实现方法
- 学会动态 Few-shot 和 Self-Consistency 策略
- 能用 Zod 定义输出模型并实现容错解析
- 学完能设计复杂的推理 Prompt 并获得稳定的结构化输出

预计学习时间：3-4 小时
:::

## Chain-of-Thought：让模型"思考"

Chain-of-Thought（CoT，思维链）是近年最有影响力的 Prompt 技巧。核心思想很简单：让模型在给出最终答案前，先展示中间推理步骤。

### 为什么 CoT 有效

LLM 是自回归模型——每次生成一个 token，基于之前的所有 token。直接输出答案只有一次"计算机会"，但先输出推理步骤时，每个中间步骤都成为后续推理的上下文，相当于给了模型更多的"计算空间"。

### Zero-shot CoT

最简单的用法：在 Prompt 末尾加一句"让我们一步步思考"。

```typescript
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

async function ask(prompt: string): Promise<string> {
  const response = await client.messages.create({
    model: "claude-sonnet-4-20250514",
    max_tokens: 1024,
    messages: [{ role: "user", content: prompt }],
  });
  const block = response.content[0];
  return block.type === "text" ? block.text : "";
}

const problem = `一个商店进了一批苹果，第一天卖掉一半多 2 个，
第二天卖掉剩下的一半多 1 个，第三天卖掉剩下的一半少 1 个，
最后还剩 4 个。这批苹果一共有多少个？`;

// 直接回答——模型可能给出错误答案
console.log("=== 直接回答 ===");
console.log(await ask(problem));

// Zero-shot CoT——准确率大幅提升
console.log("\n=== Zero-shot CoT ===");
console.log(await ask(problem + "\n\n让我们一步步思考。"));
```

### Manual CoT

手动编写推理步骤的示例，引导模型学会推理模式。效果通常优于 Zero-shot CoT。

```typescript
const manualCot = `解决数学应用题，请先列出推理步骤。

问题：小明有 15 颗糖，给了小红总数的 1/3，又给了小刚 4 颗，小明还有几颗？
推理：
1. 小明最初有 15 颗糖
2. 给了小红 15 × 1/3 = 5 颗
3. 又给了小刚 4 颗
4. 一共给出 5 + 4 = 9 颗
5. 剩余 15 - 9 = 6 颗
答案：6 颗

问题：一个水池有两个水管，进水管每小时 3 吨，出水管每小时 1 吨。
水池是空的，开两管，6 小时后水池有多少吨水？
推理：
1. 进水速度：3 吨/小时
2. 出水速度：1 吨/小时
3. 净进水速度：3 - 1 = 2 吨/小时
4. 6 小时净进水：2 × 6 = 12 吨
答案：12 吨

问题：一个商店进了一批苹果，第一天卖掉一半多 2 个，
第二天卖掉剩下的一半多 1 个，第三天卖掉剩下的一半少 1 个，
最后还剩 4 个。这批苹果一共有多少个？
推理：`;

console.log(await ask(manualCot));
```

### Auto-CoT

自动为种子问题生成推理步骤，再将这些自动生成的步骤作为 Few-shot 示例。减少手工编写成本。

```typescript
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

async function autoGenerateCot(question: string): Promise<string> {
  /** 自动为简单问题生成推理步骤 */
  const response = await client.messages.create({
    model: "claude-sonnet-4-20250514",
    max_tokens: 512,
    messages: [{
      role: "user",
      content: `请一步步解答以下问题，展示完整推理过程：\n${question}`,
    }],
  });
  const block = response.content[0];
  return block.type === "text" ? block.text : "";
}

async function autoCotSolve(target: string, seeds: string[]): Promise<string> {
  /** 先自动生成示例的推理链，再用于目标问题 */
  // 第一步：为种子问题自动生成推理
  const examples: string[] = [];
  for (const q of seeds) {
    const reasoning = await autoGenerateCot(q);
    examples.push(`问题：${q}\n${reasoning}\n`);
  }

  // 第二步：用自动生成的推理链作为 few-shot
  let prompt = "以下是一些数学推理示例：\n\n";
  prompt += examples.join("\n---\n");
  prompt += `\n---\n问题：${target}\n请用相同方式推理：`;

  const response = await client.messages.create({
    model: "claude-sonnet-4-20250514",
    max_tokens: 1024,
    messages: [{ role: "user", content: prompt }],
  });
  const block = response.content[0];
  return block.type === "text" ? block.text : "";
}

// 使用
const seeds = [
  "小明买了 3 本书，每本 12 元，付了 50 元，找回多少钱？",
  "一列火车 2 小时行驶 240 公里，平均速度是多少？",
];
const target = "甲乙两人从相距 100 公里的两地相向而行，甲每小时 6 公里，乙每小时 4 公里，几小时后相遇？";
console.log(await autoCotSolve(target, seeds));
```

## Self-Consistency：多数投票

对同一个问题多次采样，让模型走不同的推理路径，最终取出现最多的答案。有效减少偶发的推理错误。

```typescript
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

function extractAnswer(text: string): string {
  /** 从推理文本中提取最终答案 */
  const patterns = [
    /答案[是为：:]\s*(\d+)/,
    /一共有?\s*(\d+)\s*个/,
    /最终.*?(\d+)/,
  ];
  for (const pattern of patterns) {
    const match = text.match(pattern);
    if (match) {
      return match[1];
    }
  }
  const lines = text.trim().split("\n");
  return lines[lines.length - 1];
}

async function selfConsistency(question: string, nSamples: number = 5): Promise<string> {
  /** Self-Consistency：多次采样取一致结果 */
  const answers: string[] = [];
  for (let i = 0; i < nSamples; i++) {
    const response = await client.messages.create({
      model: "claude-sonnet-4-20250514",
      max_tokens: 1024,
      temperature: 0.7, // 较高温度增加采样多样性
      messages: [{
        role: "user",
        content: `${question}\n\n请一步步推理，最后给出答案。`,
      }],
    });
    const block = response.content[0];
    const answer = extractAnswer(block.type === "text" ? block.text : "");
    answers.push(answer);
    console.log(`采样 ${i + 1}: 答案 = ${answer}`);
  }

  // 多数投票
  const counter = new Map<string, number>();
  for (const ans of answers) {
    counter.set(ans, (counter.get(ans) ?? 0) + 1);
  }
  const sorted = [...counter.entries()].sort((a, b) => b[1] - a[1]);
  const [best, count] = sorted[0];
  console.log(`\n投票结果：${JSON.stringify(Object.fromEntries(counter))}`);
  console.log(`最终答案：${best}（${count}/${nSamples} 票）`);
  return best;
}

const question = "一个商店进了一批苹果，第一天卖掉一半多 2 个，第二天卖掉剩下的一半多 1 个，第三天卖掉剩下的一半少 1 个，最后还剩 4 个。一共多少个？";
const result = await selfConsistency(question, 5);
```

::: warning 成本注意
Self-Consistency 需要多次 API 调用，成本线性增长。建议仅在对准确性要求极高的场景使用（如数学计算、逻辑推理），日常任务单次 CoT 已足够。
:::

## Tree-of-Thought：思维树

### CoT 的天花板

CoT 虽然强大，但它本质上是一条**单线程**推理链——从头到尾只走一条路。如果第一步的方向选错了，后面再怎么推理也是在错误的基础上继续。

举一个典型例子：24 点游戏。给你数字 1、5、5、5，用加减乘除凑出 24。CoT 可能一上来就尝试 `5 + 5 = 10`，然后发现 `10、1、5` 怎么都凑不出 24，整条推理链就废了。但其实正确路径是 `5 × (5 - 1/5) = 24`——这需要你先**探索多个起手方向**，评估哪条路有戏，再深入展开。

类似的场景还有很多：多步规划问题（旅行路线优化）、开放性设计问题（系统架构方案）、博弈推理（需要考虑多种对手策略）。这些问题的共同特点是：**解空间有分支，需要"看几步再决定走哪条路"**。

这正是 Tree-of-Thought（ToT，思维树）要解决的问题。它在每个推理步骤生成多个候选方向，评估每个方向的前景，剪枝不靠谱的分支，保留最有希望的路径继续深入。

### 三种推理策略的全景对比

在看 ToT 的具体实现之前，先建立一个全局视角——CoT、Self-Consistency、ToT 分别解决什么层次的问题：

::: tip 各技巧适用场景
| 技巧 | 核心思路 | 适用场景 | API 调用数 | 延迟 |
|------|---------|---------|-----------|------|
| Zero-shot | 直接回答 | 简单任务 | 1 | 低 |
| Few-shot | 示例引导 | 格式/模式敏感 | 1 | 低 |
| Zero-shot CoT | 单条推理链 | 需要推理 | 1 | 中 |
| Manual CoT | 手写推理模板 | 有推理模板 | 1 | 中 |
| Self-Consistency | 多条推理链取多数 | 高准确性推理 | N | 高 |
| Tree-of-Thought | 树状搜索 + 剪枝 | 开放性复杂问题 | N * depth | 很高 |

一句话总结：CoT 走一条路，Self-Consistency 走多条路然后投票，ToT 走一步看一步、边走边选路。Agent 开发中最常用的组合是 **Few-shot + CoT**：用示例教模型格式，用 CoT 引导推理。
:::

### ToT 实现

```typescript
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

async function generateThoughts(question: string, context: string, n: number = 3): Promise<string[]> {
  /** 在当前上下文下生成 n 个候选思路 */
  const response = await client.messages.create({
    model: "claude-sonnet-4-20250514",
    max_tokens: 1024,
    messages: [{
      role: "user",
      content: `问题：${question}

当前进展：${context}

请提出 ${n} 个不同的下一步思路（编号 1-${n}），每个用一两句话描述。
格式：
1. [思路]
2. [思路]
3. [思路]`,
    }],
  });
  const thoughts: string[] = [];
  const block = response.content[0];
  const text = block.type === "text" ? block.text : "";
  for (const line of text.trim().split("\n")) {
    const trimmed = line.trim();
    if (trimmed && /^\d/.test(trimmed)) {
      const thought = trimmed.includes(".") ? trimmed.split(".").slice(1).join(".").trim() : trimmed;
      thoughts.push(thought);
    }
  }
  return thoughts.slice(0, n);
}

async function evaluateThought(question: string, thought: string): Promise<number> {
  /** 评估一个思路的前景（0-1 分） */
  const response = await client.messages.create({
    model: "claude-sonnet-4-20250514",
    max_tokens: 128,
    messages: [{
      role: "user",
      content: `问题：${question}
思路：${thought}

这个思路有多大帮助？只回复 0 到 1 之间的小数。`,
    }],
  });
  const block = response.content[0];
  const text = block.type === "text" ? block.text : "";
  const parsed = parseFloat(text.trim());
  return isNaN(parsed) ? 0.5 : parsed;
}

async function treeOfThought(question: string, maxDepth: number = 3, breadth: number = 3): Promise<string> {
  /** 简化版 Tree-of-Thought */
  const bestPath: string[] = [];
  let context = "尚未开始分析";

  for (let depth = 0; depth < maxDepth; depth++) {
    const thoughts = await generateThoughts(question, context, breadth);
    const scored: [string, number][] = [];
    for (const t of thoughts) {
      const score = await evaluateThought(question, t);
      scored.push([t, score]);
    }
    const best = scored.reduce((a, b) => (a[1] >= b[1] ? a : b));
    bestPath.push(best[0]);
    context = bestPath.map((s, i) => `步骤 ${i + 1}: ${s}`).join("\n");
    console.log(`深度 ${depth + 1}: 选择 '${best[0].slice(0, 50)}...' (分数: ${best[1]})`);
  }

  // 根据最佳路径生成最终答案
  const response = await client.messages.create({
    model: "claude-sonnet-4-20250514",
    max_tokens: 1024,
    messages: [{
      role: "user",
      content: `问题：${question}\n\n推理路径：\n${context}\n\n请基于以上推理，给出完整解答。`,
    }],
  });
  const block = response.content[0];
  return block.type === "text" ? block.text : "";
}

// 注意：ToT 消耗较多 API 调用
const result = await treeOfThought("设计一个高并发的秒杀系统，关键技术点有哪些？", 2, 3);
console.log(result);
```


## 动态 Few-shot

### 固定示例的尴尬

假设你在做一个"问题分类"系统，Few-shot 示例是这样的：

```
输入：React 组件白屏
输出：{"area": "frontend", "framework": "react"}

输入：Vue 路由 404
输出：{"area": "frontend", "framework": "vue"}
```

这对前端问题效果很好。但用户突然问了一个"线上 MySQL 主从同步延迟"——你的示例里全是前端的，模型看了这些示例后，可能会强行往前端方向靠（输出 `"area": "frontend"`），或者干脆忽略示例自由发挥。更糟的是，这两个完全不相关的示例白白占了 token，挤压了真正有用的上下文空间。

固定示例集有局限——用户输入千变万化，示例可能与实际输入不匹配。动态 Few-shot 根据输入内容自动选择最相关的示例。

```typescript
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

interface Example {
  category: string;
  inputText: string;
  outputText: string;
}

// 示例库，按类别组织
const EXAMPLE_POOL: Example[] = [
  { category: "前端", inputText: "React 组件渲染白屏", outputText: '{"area": "frontend", "framework": "react", "symptom": "白屏"}' },
  { category: "前端", inputText: "Vue 路由跳转 404", outputText: '{"area": "frontend", "framework": "vue", "symptom": "404"}' },
  { category: "后端", inputText: "接口超时 504", outputText: '{"area": "backend", "type": "timeout", "code": 504}' },
  { category: "后端", inputText: "数据库查询慢", outputText: '{"area": "backend", "type": "slow_query", "component": "database"}' },
  { category: "运维", inputText: "服务器 CPU 100%", outputText: '{"area": "ops", "type": "resource", "metric": "cpu"}' },
  { category: "运维", inputText: "磁盘空间不足", outputText: '{"area": "ops", "type": "resource", "metric": "disk"}' },
];

function selectExamples(userInput: string, n: number = 3): Example[] {
  /** 根据输入关键词选择最相关的示例（简单实现，实际可用向量相似度） */
  const keywordsMap: Record<string, string[]> = {
    "前端": ["react", "vue", "组件", "页面", "渲染", "白屏", "路由", "css"],
    "后端": ["接口", "api", "数据库", "查询", "超时", "服务"],
    "运维": ["cpu", "内存", "磁盘", "服务器", "部署", "日志"],
  };
  const scores: Record<string, number> = {};
  const inputLower = userInput.toLowerCase();
  for (const [cat, keywords] of Object.entries(keywordsMap)) {
    scores[cat] = keywords.filter((kw) => inputLower.includes(kw)).length;
  }

  const topCat = Object.entries(scores).reduce((a, b) => (a[1] >= b[1] ? a : b))[0];
  const relevant = EXAMPLE_POOL.filter((e) => e.category === topCat);
  const others = EXAMPLE_POOL.filter((e) => e.category !== topCat);
  return [...relevant, ...others].slice(0, n);
}

async function dynamicFewShot(userInput: string): Promise<string> {
  /** 动态 Few-shot 推理 */
  const examples = selectExamples(userInput);
  const examplesText = examples
    .map((e) => `输入：${e.inputText}\n输出：${e.outputText}\n`)
    .join("\n");
  const prompt = `将问题描述转换为结构化 JSON 工单。

${examplesText}输入：${userInput}
输出：`;

  const response = await client.messages.create({
    model: "claude-sonnet-4-20250514",
    max_tokens: 256,
    messages: [{ role: "user", content: prompt }],
  });
  const block = response.content[0];
  return block.type === "text" ? block.text : "";
}

console.log(await dynamicFewShot("我们的 Vue 项目首屏加载要 8 秒"));
console.log(await dynamicFewShot("线上服务器内存占用持续增长"));
```

## 结构化输出：JSON

在 Agent 系统中，模型的输出需要被程序解析和执行。如果输出格式不可控，整个系统就会崩溃。JSON 是最常用的结构化输出格式。

### Zod 模型定义 + 输出解析

用 Zod 定义输出结构，实现类型安全的解析和验证：

```typescript
import Anthropic from "@anthropic-ai/sdk";
import { z } from "zod";

const client = new Anthropic();

const Severity = z.enum(["low", "medium", "high", "critical"]);
type Severity = z.infer<typeof Severity>;

const Issue = z.object({
  line: z.number().describe("问题所在行号"),
  severity: Severity.describe("严重程度"),
  category: z.string().describe("问题类别：security/performance/style/logic"),
  description: z.string().describe("问题描述"),
  suggestion: z.string().describe("修改建议"),
});
type Issue = z.infer<typeof Issue>;

const CodeReviewResult = z.object({
  file_name: z.string().describe("被审查的文件名"),
  language: z.string().describe("编程语言"),
  overall_score: z.number().min(0).max(100).describe("总体评分 0-100"),
  issues: z.array(Issue).default([]),
  summary: z.string().describe("一句话总结"),
});
type CodeReviewResult = z.infer<typeof CodeReviewResult>;

// 生成 JSON Schema 供 Prompt 使用（简化版）
const schema = JSON.stringify(
  {
    type: "object",
    properties: {
      file_name: { type: "string" },
      language: { type: "string" },
      overall_score: { type: "number", minimum: 0, maximum: 100 },
      issues: {
        type: "array",
        items: {
          type: "object",
          properties: {
            line: { type: "number" },
            severity: { type: "string", enum: ["low", "medium", "high", "critical"] },
            category: { type: "string" },
            description: { type: "string" },
            suggestion: { type: "string" },
          },
          required: ["line", "severity", "category", "description", "suggestion"],
        },
      },
      summary: { type: "string" },
    },
    required: ["file_name", "language", "overall_score", "issues", "summary"],
  },
  null,
  2
);

async function reviewCode(code: string, filename: string = "unknown.ts"): Promise<CodeReviewResult> {
  /** 使用 LLM 审查代码，返回结构化结果 */
  const response = await client.messages.create({
    model: "claude-sonnet-4-20250514",
    max_tokens: 2048,
    system: `你是一个代码审查工具。
你必须以 JSON 格式输出审查结果，严格遵循以下 JSON Schema：

${schema}

只输出 JSON，不要输出其他内容。`,
    messages: [{
      role: "user",
      content: `审查以下代码（文件名：${filename}）：\n\`\`\`\n${code}\n\`\`\``,
    }],
  });

  let resultText = (response.content[0] as { type: "text"; text: string }).text.trim();
  if (resultText.startsWith("```")) {
    resultText = resultText.split("\n").slice(1).join("\n").replace(/```$/, "").trim();
  }

  return CodeReviewResult.parse(JSON.parse(resultText));
}

// 测试
const code = `
import sqlite3

def get_user(name):
    conn = sqlite3.connect("users.db")
    query = f"SELECT * FROM users WHERE name = '{name}'"
    cursor = conn.cursor()
    cursor.execute(query)
    return cursor.fetchall()
`;

const result = await reviewCode(code, "user_service.py");
console.log(`评分: ${result.overall_score}/100`);
console.log(`总结: ${result.summary}`);
for (const issue of result.issues) {
  console.log(`  [${issue.severity}] 第${issue.line}行 - ${issue.description}`);
}
```

### Anthropic Tool Use 强制结构化输出

Prompt 方式让模型生成 JSON 虽然简单，但模型偶尔会输出格式错误的 JSON——多一个逗号、漏一个引号、在 JSON 前后加上解释文字。这些问题在单次调用中出现的概率不高（也许 5%），但在 Agent 循环中累积起来就很致命：如果 Agent 每轮都要解析 JSON，100 轮下来几乎必然遇到解析失败。如果你需要 100% 可靠的结构化输出，可以用 Tool Use 来强制约束——模型的输出会被 API 层面约束为合法的 JSON，从根本上消除格式错误。

即使不需要"工具"，也可以利用 Claude 的 Tool Use 机制强制输出符合 Schema 的 JSON：

```typescript
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

// 定义一个 "工具"，实际上只是结构化输出的 Schema
const tools: Anthropic.Tool[] = [{
  name: "output_analysis",
  description: "输出文本分析结果",
  input_schema: {
    type: "object" as const,
    properties: {
      sentiment: {
        type: "string",
        enum: ["positive", "negative", "neutral", "mixed"],
      },
      confidence: { type: "number", minimum: 0, maximum: 1 },
      key_phrases: { type: "array", items: { type: "string" } },
    },
    required: ["sentiment", "confidence", "key_phrases"],
  },
}];

const response = await client.messages.create({
  model: "claude-sonnet-4-20250514",
  max_tokens: 1024,
  tools,
  tool_choice: { type: "tool", name: "output_analysis" },
  messages: [{
    role: "user",
    content: "分析：这款手机拍照效果很赞，续航也不错，就是价格偏高。",
  }],
});

for (const block of response.content) {
  if (block.type === "tool_use") {
    console.log(JSON.stringify(block.input, null, 2));
  }
}
```

## 结构化输出：XML

Tool Use 方案虽然可靠，但它依赖模型厂商的特定功能（Anthropic Tool Use、OpenAI Function Calling），不同平台的实现方式不同，切换模型时需要改代码。如果你的场景不需要那么严格的格式保证，还有一种更轻量的替代方案——XML 标签。XML 不依赖任何厂商特性，所有 LLM 都能理解，而且 Claude 对 XML 标签有天然的良好支持。

XML 在混合文本和结构化数据的场景中特别好用：

```typescript
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

async function analyzeWithXml(text: string): Promise<Record<string, unknown>> {
  /** 使用 XML 标签格式获取结构化输出 */
  const response = await client.messages.create({
    model: "claude-sonnet-4-20250514",
    max_tokens: 1024,
    messages: [{
      role: "user",
      content: `分析以下用户反馈，用 XML 格式输出：

反馈：${text}

<analysis>
  <sentiment>正面/负面/中性</sentiment>
  <topics>
    <topic>主题</topic>
  </topics>
  <action_items>
    <item priority="high/medium/low">待办事项</item>
  </action_items>
  <response_draft>建议的回复草稿</response_draft>
</analysis>`,
    }],
  });

  const block = response.content[0];
  const output = block.type === "text" ? block.text : "";

  function extractTag(xml: string, tag: string): string {
    const match = xml.match(new RegExp(`<${tag}>(.*?)</${tag}>`, "s"));
    return match ? match[1].trim() : "";
  }

  function extractAll(xml: string, tag: string): string[] {
    const regex = new RegExp(`<${tag}[^>]*>(.*?)</${tag}>`, "gs");
    const results: string[] = [];
    let match: RegExpExecArray | null;
    while ((match = regex.exec(xml)) !== null) {
      results.push(match[1].trim());
    }
    return results;
  }

  return {
    sentiment: extractTag(output, "sentiment"),
    topics: extractAll(output, "topic"),
    action_items: extractAll(output, "item"),
    response_draft: extractTag(output, "response_draft"),
  };
}

const result = await analyzeWithXml("搜索功能比以前好用了，但加载速度还是慢，希望能优化。");
console.log(JSON.stringify(result, null, 2));
```

::: tip XML vs JSON
- **JSON** 更适合程序解析、API 交互、嵌套数据结构
- **XML** 更适合混合文本内容（推理过程 + 结构化结果）
- Agent 开发中 JSON 是主流选择；XML 标签常用于 Prompt 中分隔不同部分
:::

## 输出解析的容错处理

无论哪种结构化输出方式，都可能遇到解析失败。健壮的 Agent 必须有容错机制。

```typescript
import Anthropic from "@anthropic-ai/sdk";
import { z, ZodType, ZodError } from "zod";

const client = new Anthropic();

async function robustParse<T>(schema: ZodType<T>, rawText: string, maxRetries: number = 2): Promise<T> {
  /** 健壮的输出解析，包含多层容错 */

  // 策略 1：直接解析
  try {
    return schema.parse(JSON.parse(rawText));
  } catch {
    // 继续尝试
  }

  // 策略 2：去除 markdown 代码块后解析
  let cleaned = rawText.trim();
  if (cleaned.startsWith("```")) {
    cleaned = cleaned.replace(/^```\w*\n?/, "").replace(/\n?```$/, "");
  }
  try {
    return schema.parse(JSON.parse(cleaned.trim()));
  } catch {
    // 继续尝试
  }

  // 策略 3：从文本中提取 JSON 片段
  const jsonMatch = rawText.match(/\{[\s\S]*\}/);
  if (jsonMatch) {
    try {
      return schema.parse(JSON.parse(jsonMatch[0]));
    } catch {
      // 继续尝试
    }
  }

  // 策略 4：让 LLM 修复输出
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      const fixResp = await client.messages.create({
        model: "claude-sonnet-4-20250514",
        max_tokens: 1024,
        messages: [{
          role: "user",
          content: `以下文本应该是 JSON 但格式有误，请修复。

原始文本：${rawText}

期望的 JSON 结构包含以下字段：goal (string), steps (string[]), estimated_time (string)

只输出修复后的 JSON。`,
        }],
      });
      let fixed = (fixResp.content[0] as { type: "text"; text: string }).text.trim();
      if (fixed.startsWith("```")) {
        fixed = fixed.replace(/^```\w*\n?/, "").replace(/\n?```$/, "");
      }
      return schema.parse(JSON.parse(fixed.trim()));
    } catch (e) {
      if (attempt === maxRetries - 1) {
        throw new Error(`经过 ${maxRetries} 次修复仍无法解析: ${e}`);
      }
    }
  }

  throw new Error("解析失败");
}

// 使用示例
const TaskPlan = z.object({
  goal: z.string(),
  steps: z.array(z.string()),
  estimated_time: z.string(),
});
type TaskPlan = z.infer<typeof TaskPlan>;

const raw = `好的，以下是计划：
\`\`\`json
{"goal": "搭建 API", "steps": ["设计模型", "实现路由", "添加认证"], "estimated_time": "2天"}
\`\`\`
希望有帮助！`;

const plan = await robustParse(TaskPlan, raw);
console.log(`目标: ${plan.goal}, 步骤: ${JSON.stringify(plan.steps)}`);
```

::: warning 生产环境建议
1. 始终用 Zod 定义输出结构，确保类型安全
2. 在 Prompt 中包含 JSON Schema，不仅仅靠"输出 JSON"的指令
3. 实现多层容错（直接解析 -> 清理后解析 -> 提取 JSON -> LLM 修复）
4. 记录解析失败的日志，用于后续优化 Prompt
5. 关键路径使用 Claude Tool Use 或 OpenAI strict JSON Schema 模式
:::

## 小结

1. **Chain-of-Thought** 通过激发中间推理步骤提升复杂任务表现，Zero-shot CoT 最简单，Manual CoT 效果最好
2. **Self-Consistency** 用多次采样 + 多数投票降低推理错误率，适合高准确性场景
3. **Tree-of-Thought** 在每步探索多个方向，适合开放性复杂问题，但 API 消耗大
4. **动态 Few-shot** 根据输入自动选择最相关的示例，比固定示例集更灵活
5. **结构化输出** 是 Agent 的刚需，Zod + JSON Schema 是最佳实践，容错解析不可或缺

## 练习

1. **CoT 练习**：用 Manual CoT 解决以下问题——给定一段 API 日志，判断是否存在 N+1 查询。手动编写 2 个推理示例（一个存在 N+1，一个不存在），然后测试对新日志的判断。

2. **Self-Consistency 练习**：实现一个包装器 `async function reliableAnswer(question: string, n = 5): Promise<string>`，自动对任意问题进行多次采样和投票。额外挑战：当没有明确多数答案时（5 次得到 5 个不同答案），自动增加采样次数。

3. **结构化输出练习**：定义一个 Zod schema `APIEndpoint`（method、path、description、parameters），用 LLM 从一段 API 文档描述中提取信息。分别用 Prompt JSON 和 Claude Tool Use 实现，对比格式准确率。

4. **综合练习**：为"Git commit message 分类"任务设计一个 Dynamic Few-shot + CoT 的 Prompt，包含示例池和关键词匹配选择逻辑，覆盖 feat、fix、refactor、docs 四种类型。

## 参考资源

- [Chain-of-Thought Prompting（原始论文）](https://arxiv.org/abs/2201.11903)
- [Tree of Thoughts（原始论文）](https://arxiv.org/abs/2305.10601)
- [Self-Consistency（原始论文）](https://arxiv.org/abs/2203.11171)
- [Anthropic Tool Use 文档](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)
