# Tool Use 进阶

> **学习目标**：掌握多工具协同的三种模式（并行、链式、混合），理解模型选择工具的决策机制，学会应对工具数量增长带来的挑战。

学完本节，你将能够：
- 正确处理模型返回的多个 tool_use 块（并行调用）
- 用 `Promise.all` 实现真正的并行工具执行
- 理解链式调用为什么不需要特殊代码
- 通过 description 优化帮助模型做出更好的工具选择
- 为工具数量超过 15 个的系统设计路由策略

## 并行工具调用

在入门篇中，我们处理过模型一次返回一个 tool_use 块的情况。但模型很聪明——当它判断多个工具调用之间**没有依赖关系**时，会在一次响应中返回多个 tool_use 块，这就是并行工具调用。

### 什么时候会触发并行调用

典型场景包括：
- 同时查多个城市的天气
- 同时搜索多个关键词
- 同时获取多个用户的信息
- 独立查询不同数据源

模型返回的响应长这样：

```typescript
response.content = [
  { type: "text", text: "我来同时查询两个城市的天气。" },
  { type: "tool_use", id: "toolu_001", name: "get_weather", input: { city: "北京" } },
  { type: "tool_use", id: "toolu_002", name: "get_weather", input: { city: "上海" } },
];
```

### 处理方式：顺序 vs 并行

处理并行调用时，你需要为**每个** tool_use 块都返回对应的 tool_result。两种执行方式各有优劣：

```typescript
import Anthropic from "@anthropic-ai/sdk";

type ToolMap = Record<string, (...args: any[]) => any>;

/** 方式一：顺序执行（简单，适合本地计算） */
function handleSequential(
  response: Anthropic.Message,
  toolMap: ToolMap
): Anthropic.ToolResultBlockParam[] {
  const toolCalls = response.content.filter(
    (block): block is Anthropic.ToolUseBlock => block.type === "tool_use"
  );
  return toolCalls.map((call) => ({
    type: "tool_result" as const,
    tool_use_id: call.id,
    content: JSON.stringify(toolMap[call.name](call.input)),
  }));
}

/** 方式二：并行执行（快，适合网络请求等 I/O 密集型工具） */
async function handleParallel(
  response: Anthropic.Message,
  toolMap: ToolMap
): Promise<Anthropic.ToolResultBlockParam[]> {
  const toolCalls = response.content.filter(
    (block): block is Anthropic.ToolUseBlock => block.type === "tool_use"
  );
  if (toolCalls.length === 0) return [];

  const resultEntries = await Promise.all(
    toolCalls.map(async (call) => {
      try {
        const result = await toolMap[call.name](call.input);
        return { id: call.id, result };
      } catch (e: any) {
        return { id: call.id, result: { error: e.message } };
      }
    })
  );

  // 按原始顺序返回
  return resultEntries.map((entry) => ({
    type: "tool_result" as const,
    tool_use_id: entry.id,
    content: JSON.stringify(entry.result),
  }));
}
```

::: tip 选择哪种执行方式
- **顺序执行**：代码简单，调试方便。如果每个工具调用只需几毫秒（如本地计算），顺序就够了。
- **并行执行**：适用于网络请求等 I/O 密集型工具。3 个各需 1 秒的调用，并行只需 1 秒而非 3 秒。
- 对模型来说没有区别——它只关心最终收到的 tool_result 是否完整和正确。
:::

## 工具链式调用

链式调用是指**工具 A 的结果作为工具 B 的输入**。比如用户问"北京今天多少度？换算成华氏温度"，模型会先调用 `get_weather` 得到 22°C，看到结果后再调用 `calculate("22 * 9/5 + 32")` 得到 71.6°F。

关键点：**链式调用不需要你写任何特殊代码**。入门篇写的 while 循环已经天然支持——模型在一轮中调用工具 A，你返回结果，模型看到结果后在下一轮调用工具 B，如此循环直到 `end_turn`。

```typescript
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

async function processQuery(userMessage: string): Promise<string> {
  /** 自动支持链式调用的处理循环 */
  const messages: Anthropic.MessageParam[] = [
    { role: "user", content: userMessage },
  ];
  const maxIterations = 10; // 防止无限循环

  for (let i = 0; i < maxIterations; i++) {
    const response = await client.messages.create({
      model: "claude-sonnet-4-20250514",
      max_tokens: 1024,
      tools,
      messages,
    });

    if (response.stop_reason === "end_turn") {
      return response.content
        .filter((b): b is Anthropic.TextBlock => b.type === "text")
        .map((b) => b.text)
        .join("");
    }

    if (response.stop_reason === "tool_use") {
      messages.push({ role: "assistant", content: response.content });
      const toolResults: Anthropic.ToolResultBlockParam[] = [];
      for (const block of response.content) {
        if (block.type === "tool_use") {
          const result = executeTool(block.name, block.input);
          toolResults.push({
            type: "tool_result",
            tool_use_id: block.id,
            content: result,
          });
        }
      }
      messages.push({ role: "user", content: toolResults });
      // 继续循环 — 模型看到结果后决定下一步
    }
  }

  return "超过最大迭代次数";
}
```

`max_iterations` 限制了最大循环次数，防止模型反复调用工具而不给出最终答案，这是必备的安全措施。

## 混合调用场景

实际任务中，并行和链式经常同时出现。比如"对比北京和上海的天气，算出温差"：

1. 模型**并行**调用 `get_weather("北京")` 和 `get_weather("上海")`（两个独立查询）
2. 你返回两个结果
3. 模型看到两个温度后，**链式**调用 `calculate("28 - 22")`
4. 模型用所有信息生成最终回答

你不需要区分这些模式——同一个 while 循环就能处理。

## 模型如何选择工具

当你提供多个工具时，模型根据以下因素做决策，按优先级排列：

1. **工具的 description** — 最重要。模型将用户请求与每个工具的 description 进行语义匹配。
2. **参数的 description** — 辅助判断。参数描述与用户信息吻合度越高，越倾向调用。
3. **工具的 name** — 提供额外语义线索，但不如 description 重要。

### 帮助模型做出更好选择

```typescript
// 技巧 1: 在 description 中说明"何时使用"和"何时不使用"
{
  name: "search_internal_docs",
  description:
    "搜索公司内部文档和知识库。" +
    "当用户询问公司政策、产品规格时使用。" +
    "不适用于公开互联网信息——那种情况请用 web_search。",
}

// 技巧 2: 相似工具通过 description 明确区分
{
  name: "search_users_by_name",
  description: "按姓名搜索用户。当用户提供了人名时使用。",
}
{
  name: "search_users_by_email",
  description: "按邮箱搜索用户。当用户提供了邮箱地址时使用。",
}

// 技巧 3: 引导决策顺序
{
  name: "get_user_profile",
  description:
    "获取用户详细资料，需要用户 ID。" +
    "如果只有姓名，请先用 search_users_by_name 获取 ID。",
}
```

## 工具数量的影响与应对

工具不是越多越好。随着数量增加，模型的选择准确率会下降：

| 工具数量 | 模型表现 | 建议 |
|---------|---------|------|
| 1-5 个 | 选择准确率极高 | 理想范围 |
| 5-15 个 | 表现良好，偶尔犹豫 | 注意 description 质量 |
| 15-30 个 | 开始出现选错的情况 | 考虑分组或动态加载 |
| 30+ 个 | 准确率明显下降 | 必须采用路由策略 |

### 策略 1：工具分组

根据任务类型只传递相关的工具子集：

```typescript
const fileTools = [readFile, writeFile, listDir, searchFile];
const webTools = [webSearch, fetchUrl, parseHtml];
const dataTools = [queryDb, runSql, exportCsv];

// 根据对话主题选择工具子集传给模型
```

### 策略 2：两阶段路由

第一阶段用一个"路由工具"判断类别，第二阶段传入具体工具：

```typescript
// 第一阶段：路由
const routerTools: Anthropic.Tool[] = [
  {
    name: "select_toolset",
    description:
      "根据任务类型选择工具集：file（文件）、web（网络）、data（数据）",
    input_schema: {
      type: "object" as const,
      properties: {
        toolset: {
          type: "string",
          enum: ["file", "web", "data"],
        },
      },
      required: ["toolset"],
    },
  },
];

// 拿到路由结果后，第二阶段传入对应的具体工具
```

## 完整代码：多工具协同 Agent

```typescript
/**
 * 多工具协同示例：搜索、计算、读写文件
 * 运行前：npm install @anthropic-ai/sdk && export ANTHROPIC_API_KEY="your-key"
 */
import Anthropic from "@anthropic-ai/sdk";
import * as fs from "fs";

const client = new Anthropic();

const tools: Anthropic.Tool[] = [
  {
    name: "web_search",
    description: "搜索互联网信息。当用户问事实性问题、最新新闻时使用。",
    input_schema: {
      type: "object" as const,
      properties: {
        query: { type: "string", description: "搜索关键词" },
      },
      required: ["query"],
    },
  },
  {
    name: "calculate",
    description: "执行数学计算。需要精确数值时使用。",
    input_schema: {
      type: "object" as const,
      properties: {
        expression: {
          type: "string",
          description: "数学表达式，如 '(100 * 1.05) - 50'",
        },
      },
      required: ["expression"],
    },
  },
  {
    name: "read_file",
    description: "读取本地文件内容。分析或查看文件时使用。",
    input_schema: {
      type: "object" as const,
      properties: {
        path: { type: "string", description: "文件路径" },
      },
      required: ["path"],
    },
  },
];

// 工具实现
function webSearch(input: { query: string }): Record<string, unknown> {
  return {
    results: [
      {
        title: `关于 ${input.query} 的结果`,
        snippet: `${input.query} 的详细信息...`,
      },
    ],
  };
}

function calculate(input: { expression: string }): Record<string, unknown> {
  try {
    // 注意：生产环境应使用安全的表达式解析库，而非 eval
    const result = Function(`"use strict"; return (${input.expression})`)();
    return { expression: input.expression, result };
  } catch (e: any) {
    return { error: `计算失败: ${e.message}` };
  }
}

function readFile(input: { path: string }): Record<string, unknown> {
  try {
    const content = fs.readFileSync(input.path, "utf-8");
    return { path: input.path, content };
  } catch (e: any) {
    return { error: e.message };
  }
}

const toolMap: Record<string, (input: any) => Record<string, unknown>> = {
  web_search: webSearch,
  calculate: calculate,
  read_file: readFile,
};

async function runAgent(userMessage: string): Promise<string> {
  const messages: Anthropic.MessageParam[] = [
    { role: "user", content: userMessage },
  ];

  for (let i = 0; i < 10; i++) {
    const response = await client.messages.create({
      model: "claude-sonnet-4-20250514",
      max_tokens: 2048,
      tools,
      messages,
    });

    if (response.stop_reason === "end_turn") {
      return response.content
        .filter((b): b is Anthropic.TextBlock => b.type === "text")
        .map((b) => b.text)
        .join("");
    }

    if (response.stop_reason === "tool_use") {
      messages.push({ role: "assistant", content: response.content });

      // 并行执行所有工具调用
      const toolCalls = response.content.filter(
        (b): b is Anthropic.ToolUseBlock => b.type === "tool_use"
      );
      const results = await Promise.all(
        toolCalls.map(async (block) => {
          const func = toolMap[block.name];
          const result = func
            ? func(block.input as Record<string, unknown>)
            : { error: `未知: ${block.name}` };
          return {
            type: "tool_result" as const,
            tool_use_id: block.id,
            content: JSON.stringify(result),
          };
        })
      );

      messages.push({ role: "user", content: results });
    }
  }

  return "达到最大迭代次数";
}

// 并行调用场景
console.log(await runAgent("帮我搜索 Python 3.12 新特性和 Rust 2024 更新"));
// 链式调用场景
console.log(
  await runAgent(
    "地球到太阳约 1.5 亿公里，光速 30 万公里/秒，光走这段距离需要多少分钟？"
  )
);
```

## 小结

- **并行调用**：多个独立的工具调用同时返回，可用 `Promise.all` 真正并行执行
- **链式调用**：工具 A 的结果作为工具 B 的输入，while 循环天然支持
- **混合调用**：一次任务中并行和链式共存，同一个循环处理
- 模型选工具主要靠 **description**，写清楚"何时用"和"何时不用"
- 工具超过 15 个时考虑分组或两阶段路由

## 练习

1. **并行优化**：修改上面的 `runAgent`，使用 `Promise.all` 实现真正的并行工具执行，对比顺序执行和并行执行的耗时。
2. **工具路由**：假设你有 25 个工具分为三组，实现一个两阶段路由系统。
3. **思考题**：如果两个工具的 description 非常相似（如 `search_google` 和 `search_bing`），模型会怎么选择？你如何帮助模型做出更好的选择？

## 参考资源

- [Anthropic Tool Use 文档](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) -- 官方完整指南
- [Anthropic: Parallel Tool Use](https://docs.anthropic.com/en/docs/build-with-claude/tool-use#parallel-tool-use) -- 并行调用文档
- [Anthropic TypeScript SDK](https://github.com/anthropics/anthropic-sdk-typescript) -- 官方 TypeScript SDK
