# Agent 入门：ReAct 模式与 Agent 循环

> **学习目标**：理解 Agent 和聊天机器人的本质区别，掌握 ReAct 模式，手写完整的 Agent 循环，构建一个文件管理 Agent。

学完本节，你将能够：
- 清晰区分 Agent 和普通聊天机器人
- 用自己的话解释 ReAct（Reasoning + Acting）模式
- 手写 while 循环 + stop_reason 的 Agent 核心引擎
- 实现一个完整的 SimpleAgent 类
- 构建一个能读写文件、做计算的实用 Agent

## 什么是 Agent

先搞清楚一个概念：**Agent 不是更高级的聊天机器人**。它们是两种完全不同的东西。

**聊天机器人**：你问一句，它答一句。就像客服热线 -- 你说什么，它回什么，结束了。

**Agent**：你给它一个任务，它**自己想办法完成**。它会拆解任务、选择工具、执行操作、检查结果、遇到问题自己调整策略。就像你派一个实习生去办事 -- 你只说"把这份报告整理好"，他自己决定先做什么后做什么。

用一个表格对比：

| 维度 | 聊天机器人 | Agent |
|------|-----------|-------|
| 交互方式 | 一问一答 | 接受任务，自主执行多步 |
| 决策权 | 用户主导每一步 | Agent 自主决定下一步 |
| 工具使用 | 不用或固定流程 | 自主选择和组合工具 |
| 错误处理 | 报错给用户 | 尝试自己修复 |
| 典型产品 | 客服机器人 | Claude Code、Cursor Agent |

一句话总结：**Agent = LLM + 工具 + 自主决策循环**。

## ReAct 模式详解

ReAct 来自 2022 年的论文 "ReAct: Synergizing Reasoning and Acting in Language Models"，名字就是 **Re**asoning + **Act**ing 的合体。

核心思想极其简单：**让 LLM 交替进行思考和行动**。

- **Thought（思考）**：分析当前情况，决定下一步做什么
- **Action（行动）**：调用工具执行操作
- **Observation（观察）**：看到工具返回的结果

然后回到 Thought，继续思考，继续行动，直到任务完成。

```
用户：帮我查一下北京天气，如果超过25度就提醒我涂防晒

Thought 1: 用户要查北京天气，还要根据温度给建议。先查天气。
Action 1:  get_weather(city="北京")
Observation 1: {"temperature": 28, "condition": "晴", "humidity": 45}

Thought 2: 28度超过25度了，而且是晴天。我应该提醒涂防晒。
Action 2:  [直接回答，不需要工具]

最终回答：北京今天28度，晴天。温度超过25度了，出门记得涂防晒霜！
```

### 为什么不能只思考、或只行动

**只思考不行动**（纯 Chain-of-Thought）：模型不知道北京今天真的多少度，只能靠训练数据猜。猜错了怎么办？

**只行动不思考**（纯 Action）：盲目调工具。比如用户问"分析这个项目的性能问题"，模型没有思考就随便读几个文件，效率极低。

ReAct 的精髓是 **用思考指导行动，用观察验证思考**。

::: tip 在现代 API 中的体现
Anthropic Claude API 里，Thought 是隐式的 -- 模型在返回 tool_use 之前，内部已经做了推理。有时候它也会在 tool_use 前输出一段文字，那就是显式的 Thought。你不需要刻意实现 Thought，API 的 stop_reason 机制天然就是 ReAct。
:::

## 手写 Agent 循环

理解了 ReAct，我们来写代码。Agent 的核心其实就是一个 **while 循环**：

```typescript
/** 最小的 Agent 循环 -- 理解核心机制 */
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

// 工具定义
const tools: Anthropic.Tool[] = [
  {
    name: "calculate",
    description: "执行数学计算，返回精确结果。",
    input_schema: {
      type: "object" as const,
      properties: {
        expression: {
          type: "string",
          description: "数学表达式，如 '2+3*4' 或 'Math.sqrt(144)'",
        },
      },
      required: ["expression"],
    },
  },
];

function calculate(expression: string): Record<string, unknown> {
  /** 安全的数学计算器 */
  try {
    const mathScope = { sqrt: Math.sqrt, pow: Math.pow, abs: Math.abs,
                        sin: Math.sin, cos: Math.cos, PI: Math.PI };
    const fn = new Function(...Object.keys(mathScope), `return (${expression})`);
    const result = fn(...Object.values(mathScope));
    return { result };
  } catch (e) {
    return { error: String(e) };
  }
}

const toolMap: Record<string, Function> = { calculate };

async function agentLoop(userMessage: string): Promise<string> {
  /** Agent 核心循环 */
  const messages: Anthropic.MessageParam[] = [
    { role: "user", content: userMessage },
  ];

  // === 核心：while 循环 ===
  const maxSteps = 10; // 安全阀门，防止无限循环
  let step = 0;

  while (step < maxSteps) {
    step += 1;
    console.log(`\n--- 第 ${step} 轮 ---`);

    // 1. 调用 LLM
    const response = await client.messages.create({
      model: "claude-sonnet-4-20250514",
      max_tokens: 1024,
      tools,
      messages,
    });

    console.log(`stop_reason: ${response.stop_reason}`);

    // 2. 判断：完成了还是需要工具？
    if (response.stop_reason === "end_turn") {
      // 任务完成，返回最终回答
      const text = response.content
        .filter((b): b is Anthropic.TextBlock => b.type === "text")
        .map((b) => b.text)
        .join("");
      console.log(`最终回答: ${text}`);
      return text;
    }

    if (response.stop_reason === "tool_use") {
      // 需要调用工具
      messages.push({ role: "assistant", content: response.content });

      // 3. 执行所有工具调用
      const toolResults: Anthropic.ToolResultBlockParam[] = [];
      for (const block of response.content) {
        if (block.type === "tool_use") {
          console.log(`调用工具: ${block.name}(${JSON.stringify(block.input)})`);
          const func = toolMap[block.name];
          const input = block.input as Record<string, string>;
          const result = func ? func(input.expression) : { error: "未知工具" };
          console.log(`工具结果: ${JSON.stringify(result)}`);

          toolResults.push({
            type: "tool_result",
            tool_use_id: block.id,
            content: JSON.stringify(result),
          });
        }
      }

      // 4. 把结果返回给 LLM，继续循环
      messages.push({ role: "user", content: toolResults });
    }
  }

  return "超过最大步骤数";
}

// 测试
await agentLoop("一个圆的半径是 7 厘米，它的面积和周长分别是多少？保留两位小数。");
```

::: warning Agent 循环的三个关键点
1. **while 循环** -- 不是调一次 API 就结束，而是循环直到 `end_turn`
2. **stop_reason 判断** -- 这是决定"继续"还是"结束"的唯一信号
3. **max_steps 限制** -- 必须有！否则模型可能无限调用工具。一般设 5-15
:::

## 完整的 SimpleAgent 类

把上面的逻辑封装成一个可复用的类：

```typescript
/** SimpleAgent -- 完整的 Agent 基类 */
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

type ToolHandler = (params: Record<string, unknown>) => Record<string, unknown>;

class SimpleAgent {
  /** 简单但完整的 Agent 实现 */
  private systemPrompt: string;
  private model: string;
  private maxSteps: number;
  private tools: Anthropic.Tool[] = [];
  private toolHandlers: Record<string, ToolHandler> = {};
  private messages: Anthropic.MessageParam[] = []; // 对话历史

  constructor(
    systemPrompt: string = "",
    model: string = "claude-sonnet-4-20250514",
    maxSteps: number = 10
  ) {
    this.systemPrompt = systemPrompt;
    this.model = model;
    this.maxSteps = maxSteps;
  }

  addTool(
    name: string,
    description: string,
    parameters: Record<string, unknown>,
    handler: ToolHandler
  ): void {
    /** 注册一个工具 */
    this.tools.push({
      name,
      description,
      input_schema: { type: "object" as const, ...parameters },
    });
    this.toolHandlers[name] = handler;
  }

  private executeTool(name: string, params: Record<string, unknown>): string {
    /** 执行工具，返回 JSON 字符串 */
    const handler = this.toolHandlers[name];
    if (!handler) {
      return JSON.stringify({ error: `未知工具: ${name}` });
    }
    try {
      const result = handler(params);
      return JSON.stringify(result);
    } catch (e) {
      const err = e as Error;
      return JSON.stringify({ error: `${err.name}: ${err.message}` });
    }
  }

  async run(userInput: string): Promise<string> {
    /** 运行 Agent：接受用户输入，自主完成任务 */
    this.messages.push({ role: "user", content: userInput });

    for (let step = 0; step < this.maxSteps; step++) {
      // 调用 LLM
      const response = await client.messages.create({
        model: this.model,
        max_tokens: 4096,
        messages: this.messages,
        ...(this.systemPrompt ? { system: this.systemPrompt } : {}),
        ...(this.tools.length > 0 ? { tools: this.tools } : {}),
      });

      // 判断是否完成
      if (response.stop_reason === "end_turn") {
        this.messages.push({ role: "assistant", content: response.content });
        return response.content
          .filter((b): b is Anthropic.TextBlock => b.type === "text")
          .map((b) => b.text)
          .join("");
      }

      // 处理工具调用
      if (response.stop_reason === "tool_use") {
        this.messages.push({ role: "assistant", content: response.content });
        const results: Anthropic.ToolResultBlockParam[] = [];
        for (const block of response.content) {
          if (block.type === "tool_use") {
            const resultStr = this.executeTool(
              block.name,
              block.input as Record<string, unknown>
            );
            results.push({
              type: "tool_result",
              tool_use_id: block.id,
              content: resultStr,
            });
          }
        }
        this.messages.push({ role: "user", content: results });
      }
    }

    return "达到最大执行步骤，任务未完成。";
  }

  async chat(userInput: string): Promise<string> {
    /** 对话模式：支持多轮对话 */
    return this.run(userInput);
  }

  reset(): void {
    /** 重置对话历史 */
    this.messages = [];
  }
}
```

## 对话历史管理基础

Agent 的对话历史不只是"用户说了什么" -- 它还包含所有的工具调用和结果。消息列表的结构是这样的：

```typescript
const messages: Anthropic.MessageParam[] = [
  // 用户提问
  { role: "user", content: "北京今天天气怎么样？" },

  // LLM 回复（包含思考文本 + 工具调用）
  { role: "assistant", content: [
    { type: "text", text: "我来查一下北京的天气。" },
    { type: "tool_use", id: "toolu_01", name: "get_weather", input: { city: "北京" } },
  ]},

  // 工具结果（注意：放在 user 角色中）
  { role: "user", content: [
    { type: "tool_result", tool_use_id: "toolu_01",
      content: '{"temperature": 22, "condition": "晴"}' },
  ]},

  // LLM 最终回答
  { role: "assistant", content: [
    { type: "text", text: "北京今天22度，晴天，适合出门。" },
  ]},
];
```

::: danger 必须完整保留
- assistant 消息中的 **所有 content 块**（文本 + tool_use）必须完整保留
- tool_result 的 **tool_use_id 必须匹配**对应的 tool_use id
- tool_result 放在 **user 角色**中（这是 Anthropic API 的设计）
- 不能跳过、删除或修改中间的消息

违反这些规则会导致 API 报错或模型行为异常。
:::

## 实战：文件管理 Agent

用 SimpleAgent 构建一个能管理文件的 Agent：

```typescript
/** 文件管理 Agent -- 能读写文件、列出目录、做计算 */
import * as fs from "fs";
import * as path from "path";

// 创建 Agent
const agent = new SimpleAgent(
  "你是一个文件管理助手。可以读写文件、列出目录内容、做数学计算。" +
    "执行操作前先确认意图，操作后报告结果。如果遇到错误，说明原因并建议替代方案。",
  "claude-sonnet-4-20250514",
  8
);

// --- 注册工具 ---

agent.addTool(
  "list_directory",
  "列出指定目录下的文件和子目录。不指定路径时列出当前目录。",
  {
    properties: {
      path: { type: "string", description: "目录路径，默认当前目录" },
    },
  },
  (params) => {
    const dirPath = (params.path as string) || ".";
    const entries = fs.readdirSync(dirPath).sort().slice(0, 50); // 限制返回数量
    return {
      path: path.resolve(dirPath),
      entries: entries.map((entry) => {
        const fullPath = path.join(dirPath, entry);
        const isDir = fs.statSync(fullPath).isDirectory();
        return {
          name: entry,
          type: isDir ? "dir" : "file",
          size: isDir ? null : fs.statSync(fullPath).size,
        };
      }),
    };
  }
);

agent.addTool(
  "read_file",
  "读取文本文件的内容。支持 .txt, .ts, .md, .json 等文本格式。",
  {
    properties: {
      path: { type: "string", description: "文件路径" },
    },
    required: ["path"],
  },
  (params) => {
    const filePath = params.path as string;
    if (!fs.existsSync(filePath)) {
      return { error: `文件不存在: ${filePath}` };
    }
    const content = fs.readFileSync(filePath, "utf-8").slice(0, 20000);
    const lines = fs.readFileSync(filePath, "utf-8").split("\n").length;
    return { path: filePath, content, lines };
  }
);

agent.addTool(
  "write_file",
  "将内容写入文件。如果文件已存在会覆盖。",
  {
    properties: {
      path: { type: "string", description: "文件路径" },
      content: { type: "string", description: "要写入的内容" },
    },
    required: ["path", "content"],
  },
  (params) => {
    const filePath = params.path as string;
    const content = params.content as string;
    fs.writeFileSync(filePath, content, "utf-8");
    return { status: "success", path: filePath, bytes: Buffer.byteLength(content, "utf-8") };
  }
);

agent.addTool(
  "calculate",
  "执行数学计算，返回精确结果。",
  {
    properties: {
      expression: { type: "string", description: "数学表达式" },
    },
    required: ["expression"],
  },
  (params) => {
    const expression = params.expression as string;
    const mathScope = {
      sqrt: Math.sqrt, pow: Math.pow, abs: Math.abs, round: Math.round,
      PI: Math.PI, sin: Math.sin, cos: Math.cos,
    };
    const fn = new Function(...Object.keys(mathScope), `return (${expression})`);
    return { result: fn(...Object.values(mathScope)) };
  }
);

// --- 运行 ---
import * as readline from "readline";

const rl = readline.createInterface({ input: process.stdin, output: process.stdout });

function prompt(question: string): Promise<string> {
  return new Promise((resolve) => rl.question(question, resolve));
}

console.log("文件管理 Agent（输入 quit 退出，输入 reset 重置对话）");

async function main() {
  while (true) {
    const user = (await prompt("\n你: ")).trim();
    if (user === "quit") break;
    if (user === "reset") {
      agent.reset();
      console.log("对话已重置。");
      continue;
    }
    const reply = await agent.chat(user);
    console.log(`\n助手: ${reply}`);
  }
  rl.close();
}

main();
```

试试这些对话：
- "列出当前目录的文件"
- "创建一个 hello.txt，内容是 Hello World"
- "读取 hello.txt，统计它有多少个字符"
- "列出当前目录，找到最大的文件"

## 小结

- **Agent 的本质**：LLM + 工具 + 自主决策循环，关键是"自主"二字
- **ReAct 模式**：思考指导行动，观察验证思考，交替循环直到完成
- **Agent 循环**：while + stop_reason 判断，是所有 Agent 框架的底层核心
- **消息历史**：必须完整保留 assistant 的 tool_use 和 user 的 tool_result
- **SimpleAgent 类**：可复用的 Agent 基类，add_tool 注册工具，run/chat 执行

## 练习

1. **动手做**：运行文件管理 Agent，完成以下任务链："创建一个 notes 目录，在里面写入 3 个文件，然后列出目录内容确认"。观察 Agent 需要几轮工具调用。
2. **扩展 Agent**：给 SimpleAgent 添加一个 `get_current_time` 工具，测试"现在几点了？距离今天结束还有多少小时？"
3. **安全思考**：当前的 `write_file` 工具可以写入任何路径。如果模型被恶意 prompt 注入，可能会覆盖系统文件。你会怎么加安全限制？
4. **对比实验**：同一个任务，分别设置 max_steps=3 和 max_steps=10，比较 Agent 的完成情况。

## 参考资源

- [ReAct 原论文 (arXiv:2210.03629)](https://arxiv.org/abs/2210.03629) -- Yao et al., 2022
- [Anthropic: Building Effective Agents](https://docs.anthropic.com/en/docs/build-with-claude/agentic) -- 官方 Agent 构建指南
- [Anthropic: Tool Use](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) -- Tool Use 文档
- [Lilian Weng: LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) -- 经典综述
