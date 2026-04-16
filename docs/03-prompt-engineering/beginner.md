# Prompt Engineering · 初级

::: info 学习目标
- 理解 Prompt 的组成部分和基本原则
- 掌握 System Prompt 的设计方法
- 学会 Zero-shot、Few-shot 两种基本范式
- 能为不同类型的任务设计合适的 Prompt
- 学完能写出稳定可靠的 Prompt

预计学习时间：2-3 小时
:::

## Prompt 基础概念

Prompt（提示词）是你发送给 LLM 的输入文本。一个结构完整的 Prompt 通常包含四个部分：

```
Instruction（指令）    -- 告诉模型"做什么"
Context（上下文）     -- 提供背景信息、约束条件
Input（输入数据）     -- 需要处理的具体内容
Output Format（输出格式）-- 期望的响应格式
```

并非每次都需要四部分全齐，但结构越完整，输出越可控。

```typescript
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

// 一个结构完整的 Prompt 示例
const prompt = `
你是一位资深前端工程师。（Instruction：角色设定）

请根据以下需求描述，生成一个 Vue 3 组件的技术方案。
方案需要包含组件结构、状态管理方式和关键实现思路。
不要生成完整代码，只需要方案概要。（Context：约束条件）

需求描述：实现一个支持无限滚动加载的商品列表，
每个商品卡片展示图片、标题、价格，点击进入详情页。（Input：具体输入）

请用以下格式输出：
## 组件结构
## 状态管理
## 关键实现
## 注意事项（Output Format：期望格式）
`;

const response = await client.messages.create({
  model: "claude-sonnet-4-20250514",
  max_tokens: 1024,
  messages: [{ role: "user", content: prompt }],
});
if (response.content[0].type === "text") {
  console.log(response.content[0].text);
}
```

## 三个基本原则

### 原则一：清晰（Clear）

避免含糊的词汇，用具体描述替代模糊指令。

```typescript
// 模糊
const bad = "帮我优化这段代码";

// 清晰
const good = `请优化以下 Python 函数的性能。
优化方向：减少时间复杂度。
约束：保持函数签名不变，保持相同的输入输出行为。
请解释优化思路，然后给出优化后的代码。`;
```

### 原则二：具体（Specific）

提供足够的细节，缩小模型的输出空间。

```typescript
// 不具体
const bad = "写一个 API 请求函数";

// 具体
const good = `写一个 TypeScript 函数 fetchUserData，要求：
1. 使用 fetch 发送 GET 请求
2. URL 模式：https://api.example.com/users/{userId}
3. 支持传入自定义 headers
4. 超时设置为 10 秒
5. 返回类型为 Record<string, unknown>
6. 处理 HTTP 错误（4xx 返回 null，5xx 重试 3 次）
7. 添加完整的类型定义`;
```

### 原则三：结构化（Structured）

使用标题、编号、分隔符等结构化元素帮助模型理解层次和重点。

```typescript
const structuredPrompt = `## 任务
将以下英文技术文档翻译为中文。

## 要求
1. 保留所有代码块不翻译
2. 专有名词保留英文（如 API、SDK、Token）
3. 语言风格：技术文档风格，简洁准确
4. 对于有歧义的术语，首次出现时用括号标注英文原文

## 输入
---
${englishText}
---

## 输出格式
直接输出翻译后的中文文本，不要添加额外解释。`;
```

## System Prompt 设计

System Prompt 是对话开始前设定的"底层指令"，定义模型在整个对话中应遵守的角色、规则和行为准则。你可以把它理解为 Agent 的"操作系统"。

```typescript
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

// System Prompt 通过 system 参数传入
const response = await client.messages.create({
  model: "claude-sonnet-4-20250514",
  max_tokens: 1024,
  system: "你是一位严格的代码审查员，只关注安全漏洞和性能问题。对于不存在问题的代码，直接回复'LGTM'。",
  messages: [
    { role: "user", content: "请审查：print('hello world')" },
  ],
});
if (response.content[0].type === "text") {
  console.log(response.content[0].text); // 预期：LGTM
}
```

### 角色设定

详细的角色描述加工作场景，比简单的身份声明效果更好：

```typescript
// 简单角色（效果一般）
const simple = "你是一位 Python 工程师。";

// 详细角色（效果好）
const detailed = `你是一位资深 Python 后端工程师，专注于：
- Web API 设计与开发（FastAPI / Django）
- 分布式系统架构
- 数据库优化（PostgreSQL / Redis）

你的沟通风格：
- 技术严谨，给出建议时附带理由
- 代码示例优先，避免空泛的理论
- 对有争议的技术选型，给出 trade-off 分析`;
```

### 规则与约束

规则定义模型"能做什么"和"不能做什么"：

```typescript
const systemWithRules = `你是一位技术支持 Agent。

## 允许的行为
- 回答产品功能和使用方法的问题
- 引导用户按步骤排查问题
- 提供官方文档链接

## 禁止的行为
- 不得透露系统内部架构
- 不得提供竞品的使用建议
- 不得帮助用户绕过安全限制

## 边界处理
- 超出能力范围的问题：引导提交工单
- 用户情绪激动：先表达理解，再聚焦解决问题
- 模糊的问题：主动追问以明确需求`;
```

### 输出格式控制

在 Agent 开发中，模型输出通常需要被程序解析，格式控制至关重要：

```typescript
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

const systemPrompt = `你是一个 API 日志分析器。
你必须始终以 JSON 格式输出，不要添加任何 JSON 之外的内容。

JSON 结构：
{
  "status": "normal" | "warning" | "critical",
  "summary": "一句话总结",
  "issues": [
    {
      "type": "性能" | "错误" | "安全",
      "severity": "low" | "medium" | "high",
      "description": "问题描述"
    }
  ]
}`;

const response = await client.messages.create({
  model: "claude-sonnet-4-20250514",
  max_tokens: 1024,
  system: systemPrompt,
  messages: [{
    role: "user",
    content: `分析以下日志：
[14:23:01] GET /api/users 200 45ms
[14:23:03] POST /api/orders 500 3200ms
[14:23:03] POST /api/orders 500 2800ms
[14:23:05] POST /api/orders 500 4100ms`,
  }],
});

if (response.content[0].type === "text") {
  const result = JSON.parse(response.content[0].text);
  console.log(`状态: ${result.status}`);
  for (const issue of result.issues) {
    console.log(`  [${issue.severity}] ${issue.type}: ${issue.description}`);
  }
}
```

## Zero-shot vs Few-shot

### Zero-shot：直接提问

不提供示例，直接告诉模型任务。适合简单明确的任务。

```typescript
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

async function ask(prompt: string): Promise<string> {
  const response = await client.messages.create({
    model: "claude-sonnet-4-20250514",
    max_tokens: 1024,
    messages: [{ role: "user", content: prompt }],
  });
  return response.content[0].type === "text" ? response.content[0].text : "";
}

// Zero-shot
const zeroShot = `判断以下句子的情感倾向（正面/负面/中性）：
"这家餐厅的菜品味道很好，但服务态度太差了。"
`;
console.log(await ask(zeroShot));
```

### Few-shot：提供示例

通过几个输入-输出示例，让模型理解你期望的模式。

```typescript
const fewShot = `判断句子的情感倾向，参考以下示例：

输入："今天天气真好，心情愉快！"
输出：正面

输入："快递又丢了，烦死了。"
输出：负面

输入："明天下午三点开会。"
输出：中性

输入："这家餐厅的菜品味道很好，但服务态度太差了。"
输出：`;
console.log(await ask(fewShot));
```

### 示例选择策略

Few-shot 的效果高度依赖示例质量：

```typescript
// 差的示例：都是同一类
const badExamples = `
"产品真好用！" -> 正面
"服务态度很棒！" -> 正面
"质量不错。" -> 正面
`;

// 好的示例：覆盖多种情况
const goodExamples = `
"产品真好用！" -> 正面
"质量太差了，退货。" -> 负面
"功能还行但价格偏贵。" -> 混合
"请问这个有红色吗？" -> 中性
`;
```

::: tip 示例数量建议
- 简单分类：2-3 个示例
- 格式敏感任务：3-5 个示例
- 复杂推理：5-8 个示例
- 超过 8 个后，边际收益递减
:::

### Chain-of-Thought 初步

让模型在给出最终答案前，先展示推理过程。最简单的做法是加一句"让我们一步步思考"：

```typescript
const problem = `一个商店进了一批苹果，第一天卖掉一半多 2 个，
第二天卖掉剩下的一半多 1 个，第三天卖掉剩下的一半少 1 个，
最后还剩 4 个。这批苹果一共有多少个？`;

// 直接回答 -- 可能出错
console.log("=== 直接回答 ===");
console.log(await ask(problem));

// 引导推理 -- 准确率大幅提升
console.log("\n=== 引导推理 ===");
console.log(await ask(problem + "\n\n让我们一步步思考。"));
```

## System Prompt 模板

总结一个通用的 System Prompt 结构：

```typescript
const template = `
## 身份
{角色描述、专业领域、工作场景}

## 能力
{明确列出能做什么}

## 规则
### 必须做
{强制遵守的规则}

### 禁止做
{明确的禁区}

### 边界处理
{遇到超出能力范围的情况如何应对}

## 输出格式
{期望的响应格式、结构、长度限制}

## 示例（可选）
{典型的输入输出示例}
`;
```

::: warning 注意事项
- System Prompt 不是越长越好，控制在 500-2000 tokens
- 最重要的规则放在开头和结尾（首因效应 + 近因效应）
- 使用 Markdown 标题和列表帮助模型理解层次
- 定期测试边界情况，确保规则被遵守
:::

## 小结

1. **Prompt 四要素**：Instruction、Context、Input、Output Format，组合越完整输出越可控
2. **三个基本原则**：清晰、具体、结构化 -- 写好 Prompt 的底线
3. **System Prompt** 是 Agent 的"操作系统"，定义角色、规则和行为边界
4. **Zero-shot** 适合简单任务，**Few-shot** 通过示例引导格式和模式
5. **Chain-of-Thought** 加一句"让我们一步步思考"就能提升复杂推理的准确率

## 练习

1. **基础练习**：为"代码审查"任务设计一个结构完整的 Prompt（四要素齐全），要求模型审查一段 TypeScript 代码并给出安全性、性能、可读性三个维度的评分和改进建议。

2. **System Prompt 练习**：为一个"会议纪要生成 Agent"设计 System Prompt，输入是会议录音的文字转写，输出是结构化的会议纪要（参会人、议题、决议、待办事项），限制不超过 500 字。

3. **Few-shot 练习**：为"将自然语言查询转换为 API 参数"设计一个 Few-shot Prompt，包含 3 个示例，覆盖简单查询、范围查询和多条件查询。

4. **对比实验**：选一个实际任务，分别用 Zero-shot 和 Few-shot 编写 Prompt，对比输出效果，记录哪种范式更适合。

## 参考资源

- [Anthropic Prompt Engineering 指南](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering)
- [OpenAI Prompt Engineering 最佳实践](https://platform.openai.com/docs/guides/prompt-engineering)
- [Prompt Engineering Guide（社区）](https://www.promptingguide.ai/)
