# Agent 进阶：设计模式

> **学习目标**：掌握 Planning、Reflection、Router、Human-in-the-Loop 四大 Agent 设计模式，学会状态管理，构建一个带规划和反思的研究 Agent。

学完本节，你将能够：
- 实现 Plan-then-Execute 模式（先规划再执行）
- 让 Agent 具备自我反思和改进的能力
- 设计 Router Agent 进行任务智能分发
- 在关键操作前加入人类审批流程
- 管理复杂 Agent 的运行状态

## Planning 模式：先规划再执行

入门篇的 Agent 是"走一步看一步"-- 做完一件事再想下一件。这对简单任务够用，但面对复杂任务就容易迷失方向。

Planning 模式的核心思想：**先制定完整计划，再逐步执行**。

```typescript
/** Plan-then-Execute Agent */
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

const tools: Anthropic.Tool[] = [
  {
    name: "search_web",
    description: "搜索互联网获取信息",
    input_schema: {
      type: "object" as const,
      properties: { query: { type: "string", description: "搜索关键词" } },
      required: ["query"],
    },
  },
  {
    name: "write_file",
    description: "将内容写入文件",
    input_schema: {
      type: "object" as const,
      properties: {
        path: { type: "string" },
        content: { type: "string" },
      },
      required: ["path", "content"],
    },
  },
  {
    name: "read_file",
    description: "读取文件内容",
    input_schema: {
      type: "object" as const,
      properties: { path: { type: "string" } },
      required: ["path"],
    },
  },
];

const toolMap: Record<string, (...args: any[]) => any> = {
  search_web: (query: string) => ({
    results: [
      { title: `关于 ${query} 的结果`, content: `${query} 的详细信息...` },
    ],
  }),
  write_file: (path: string, content: string) => ({ status: "saved", path }),
  read_file: (path: string) => ({ content: `文件 ${path} 的内容...` }),
};

class PlanningAgent {
  /** 先制定计划，再逐步执行的 Agent */
  private messages: Anthropic.MessageParam[] = [];

  /** 第一阶段：让 LLM 制定执行计划 */
  async plan(task: string): Promise<string[]> {
    const response = await client.messages.create({
      model: "claude-sonnet-4-20250514",
      max_tokens: 1024,
      messages: [
        {
          role: "user",
          content: `请为以下任务制定一个执行计划。

任务：${task}

要求：
1. 列出 3-7 个具体步骤
2. 每步用一句话描述
3. 按 JSON 数组格式返回，如 ["步骤1", "步骤2", ...]
4. 只返回 JSON，不要其他内容`,
        },
      ],
    });

    let text =
      response.content[0].type === "text"
        ? response.content[0].text.trim()
        : "";
    // 尝试解析 JSON
    try {
      // 处理可能被 markdown 代码块包裹的情况
      if (text.includes("```")) {
        text = text.split("```")[1].replace("json", "").trim();
      }
      const plan = JSON.parse(text);
      return Array.isArray(plan) ? plan : [text];
    } catch {
      return [text];
    }
  }

  /** 第二阶段：执行计划中的一个步骤 */
  async executeStep(step: string, context: string): Promise<string> {
    this.messages = [
      {
        role: "user",
        content: `执行以下步骤：

当前步骤：${step}

之前的执行结果：
${context}

请使用可用工具完成这个步骤。完成后总结本步骤的结果。`,
      },
    ];

    for (let i = 0; i < 5; i++) {
      // 每步最多 5 轮工具调用
      const response = await client.messages.create({
        model: "claude-sonnet-4-20250514",
        max_tokens: 2048,
        tools,
        messages: this.messages,
      });

      if (response.stop_reason === "end_turn") {
        return response.content
          .filter((b): b is Anthropic.TextBlock => b.type === "text")
          .map((b) => b.text)
          .join("");
      }

      if (response.stop_reason === "tool_use") {
        this.messages.push({ role: "assistant", content: response.content });
        const results: Anthropic.ToolResultBlockParam[] = [];
        for (const block of response.content) {
          if (block.type === "tool_use") {
            const func = toolMap[block.name];
            const input = block.input as Record<string, any>;
            const result = func
              ? func(...Object.values(input))
              : { error: "未知工具" };
            results.push({
              type: "tool_result",
              tool_use_id: block.id,
              content: JSON.stringify(result),
            });
          }
        }
        this.messages.push({ role: "user", content: results });
      }
    }

    return "步骤执行超时";
  }

  /** 完整的 Plan-then-Execute 流程 */
  async run(task: string): Promise<string> {
    // 1. 制定计划
    console.log("=".repeat(50));
    console.log("第一阶段：制定计划");
    console.log("=".repeat(50));
    const plan = await this.plan(task);
    plan.forEach((step, i) => console.log(`  步骤 ${i + 1}: ${step}`));

    // 2. 逐步执行
    console.log("\n" + "=".repeat(50));
    console.log("第二阶段：逐步执行");
    console.log("=".repeat(50));

    let context = "";
    for (let i = 0; i < plan.length; i++) {
      console.log(
        `\n--- 执行步骤 ${i + 1}/${plan.length}: ${plan[i]} ---`
      );
      const result = await this.executeStep(plan[i], context);
      context += `\n步骤${i + 1}结果: ${result}`;
      console.log(`结果: ${result.slice(0, 200)}`);
    }

    return context;
  }
}

// 使用
const agent = new PlanningAgent();
agent.run(
  "调研 TypeScript 最新的类型系统特性，整理成一份简要报告保存到 report.md"
);
```

## Reflection 模式：自我反思

执行完任务后，让 Agent "回头看看"自己做得怎么样，有没有遗漏或错误：

```typescript
/** Reflection Agent -- 执行后自我反思 */
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

interface ReflectionResult {
  completeness: string;
  quality: string;
  issues: string[];
  suggestions: string[];
  needs_retry: boolean;
}

class ReflectionAgent {
  /** 带自我反思能力的 Agent */
  private systemPrompt: string;

  constructor(systemPrompt: string = "") {
    this.systemPrompt = systemPrompt;
  }

  /** 执行任务 */
  async execute(
    task: string,
    tools: Anthropic.Tool[],
    toolMap: Record<string, (...args: any[]) => any>
  ): Promise<string> {
    const messages: Anthropic.MessageParam[] = [
      { role: "user", content: task },
    ];

    for (let i = 0; i < 8; i++) {
      const response = await client.messages.create({
        model: "claude-sonnet-4-20250514",
        max_tokens: 2048,
        system: this.systemPrompt,
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
        const results: Anthropic.ToolResultBlockParam[] = [];
        for (const block of response.content) {
          if (block.type === "tool_use") {
            const func = toolMap[block.name];
            const input = block.input as Record<string, any>;
            const result = func
              ? func(...Object.values(input))
              : { error: "未知" };
            results.push({
              type: "tool_result",
              tool_use_id: block.id,
              content: JSON.stringify(result),
            });
          }
        }
        messages.push({ role: "user", content: results });
      }
    }
    return "执行超时";
  }

  /** 自我反思：评估执行结果 */
  async reflect(task: string, result: string): Promise<ReflectionResult> {
    const response = await client.messages.create({
      model: "claude-sonnet-4-20250514",
      max_tokens: 1024,
      messages: [
        {
          role: "user",
          content: `请评估以下任务的执行结果：

原始任务：${task}

执行结果：${result}

请从以下维度评估，返回 JSON 格式：
{
    "completeness": "完整/部分完成/未完成",
    "quality": "高/中/低",
    "issues": ["问题1", "问题2"],
    "suggestions": ["改进建议1", "改进建议2"],
    "needs_retry": true/false
}

只返回 JSON，不要其他内容。`,
        },
      ],
    });

    let text =
      response.content[0].type === "text"
        ? response.content[0].text.trim()
        : "";
    try {
      if (text.includes("```")) {
        text = text.split("```")[1].replace("json", "").trim();
      }
      return JSON.parse(text) as ReflectionResult;
    } catch {
      return {
        completeness: "未知",
        quality: "未知",
        issues: [],
        suggestions: [],
        needs_retry: false,
      };
    }
  }

  /** 执行 + 反思 + 可能的重试 */
  async runWithReflection(
    task: string,
    tools: Anthropic.Tool[],
    toolMap: Record<string, (...args: any[]) => any>,
    maxRetries: number = 2
  ): Promise<string> {
    let currentTask = task;
    let result = "";

    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      console.log(`\n${"=".repeat(40)}`);
      console.log(`第 ${attempt + 1} 次执行`);
      console.log("=".repeat(40));

      // 执行
      result = await this.execute(currentTask, tools, toolMap);
      console.log(`执行结果: ${result.slice(0, 300)}`);

      // 反思
      console.log(`\n--- 自我反思 ---`);
      const reflection = await this.reflect(currentTask, result);
      console.log(`完整度: ${reflection.completeness}`);
      console.log(`质量: ${reflection.quality}`);
      if (reflection.issues.length > 0) {
        console.log(`问题: ${JSON.stringify(reflection.issues)}`);
      }
      if (reflection.suggestions.length > 0) {
        console.log(`建议: ${JSON.stringify(reflection.suggestions)}`);
      }

      // 判断是否需要重试
      if (!reflection.needs_retry) {
        console.log("\n反思通过，任务完成！");
        return result;
      }

      // 需要重试 -- 把反思结果加入下次执行的任务描述中
      console.log("\n反思发现问题，准备重试...");
      currentTask = `${currentTask}

上次执行的问题：${JSON.stringify(reflection.issues)}
改进建议：${JSON.stringify(reflection.suggestions)}
请根据以上反馈改进执行。`;
    }

    return result;
  }
}
```

## 路由分发模式

当你的系统需要处理多种类型的任务时，用一个 Router Agent 先判断类型，再分发给专业的子 Agent：

```typescript
/** Router Agent -- 智能任务分发 */
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

interface Route {
  description: string;
  handler: (task: string) => string;
}

class RouterAgent {
  /** 路由 Agent：根据任务类型分发给专业 Agent */
  private routes: Record<string, Route> = {};

  /** 注册一个处理路由 */
  registerRoute(
    name: string,
    description: string,
    handler: (task: string) => string
  ): void {
    this.routes[name] = { description, handler };
  }

  /** 用 LLM 判断任务类型 */
  async classify(userInput: string): Promise<string> {
    const routeDescriptions = Object.entries(this.routes)
      .map(([name, info]) => `- ${name}: ${info.description}`)
      .join("\n");

    const response = await client.messages.create({
      model: "claude-sonnet-4-20250514",
      max_tokens: 50,
      messages: [
        {
          role: "user",
          content: `将以下用户请求分类到最合适的类别。

可用类别：
${routeDescriptions}

用户请求：${userInput}

只返回类别名称，不要其他内容。`,
        },
      ],
    });

    return response.content[0].type === "text"
      ? response.content[0].text.trim()
      : "";
  }

  /** 分类 + 分发 */
  async handle(userInput: string): Promise<string> {
    const category = await this.classify(userInput);
    console.log(`[路由] 任务分类为: ${category}`);

    const route = this.routes[category];
    if (route) {
      return route.handler(userInput);
    } else {
      // 兜底：用通用方式处理
      console.log(`[路由] 未匹配到路由，使用通用处理`);
      return `无法分类的请求: ${userInput}`;
    }
  }
}

// 使用示例
const router = new RouterAgent();

router.registerRoute(
  "code",
  "编程相关：写代码、调试、代码审查、技术问题",
  (task) => `[代码Agent处理] ${task}`
);
router.registerRoute(
  "writing",
  "写作相关：写文章、翻译、总结、润色",
  (task) => `[写作Agent处理] ${task}`
);
router.registerRoute(
  "research",
  "调研相关：搜索信息、对比分析、整理报告",
  (task) => `[调研Agent处理] ${task}`
);

// 测试
async function main() {
  console.log(await router.handle("帮我写一个 TypeScript 排序算法"));
  console.log(await router.handle("帮我把这篇文章翻译成英文"));
  console.log(await router.handle("调研一下 2026 年 AI 芯片市场"));
}
main();
```

## 人机协同模式

有些决策太重要，不能完全交给 Agent。Human-in-the-Loop 在关键节点引入人类判断：

```typescript
/** Human-in-the-Loop Agent */
import * as readline from "readline";

interface DecisionLogEntry {
  action: string;
  decision: string;
  original?: Record<string, any>;
  modified?: Record<string, any>;
}

class HumanInLoopAgent {
  /**
   * 在关键决策点引入人类判断的 Agent
   * autoApprove: 自动放行的操作集合，不在集合中的操作需要人类确认
   */
  private autoApprove: Set<string>;
  private decisionLog: DecisionLogEntry[] = [];

  constructor(autoApproveActions?: Set<string>) {
    this.autoApprove =
      autoApproveActions ?? new Set(["read_file", "search", "calculate"]);
  }

  /** 请求人类审批（通过 stdin 交互） */
  private async requestApproval(
    action: string,
    params: Record<string, any>
  ): Promise<[boolean, Record<string, any>]> {
    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
    });
    const ask = (q: string) =>
      new Promise<string>((resolve) => rl.question(q, resolve));

    console.log(`\n${"*".repeat(50)}`);
    console.log(`  Agent 请求执行以下操作：`);
    console.log(`  操作: ${action}`);
    console.log(`  参数: ${JSON.stringify(params, null, 2)}`);
    console.log("*".repeat(50));

    const choice = (await ask("  [a]允许  [d]拒绝  [m]修改参数: "))
      .trim()
      .toLowerCase();

    if (choice === "a") {
      this.decisionLog.push({ action, decision: "approved" });
      rl.close();
      return [true, params];
    } else if (choice === "m") {
      const newParamsStr = await ask("  输入新参数 (JSON): ");
      try {
        const newParams = JSON.parse(newParamsStr);
        this.decisionLog.push({
          action,
          decision: "modified",
          original: params,
          modified: newParams,
        });
        rl.close();
        return [true, newParams];
      } catch {
        console.log("  参数格式错误，操作取消");
        rl.close();
        return [false, params];
      }
    } else {
      this.decisionLog.push({ action, decision: "rejected" });
      rl.close();
      return [false, params];
    }
  }

  /** 带审批的工具执行 */
  async executeWithApproval(
    action: string,
    params: Record<string, any>,
    handler: (params: Record<string, any>) => Record<string, any>
  ): Promise<Record<string, any>> {
    if (this.autoApprove.has(action)) {
      return handler(params);
    }

    const [approved, finalParams] = await this.requestApproval(action, params);
    if (approved) {
      return handler(finalParams);
    } else {
      return { status: "rejected", reason: "操作被人类拒绝" };
    }
  }
}
```

## Agent 状态管理

复杂 Agent 需要追踪自己的执行状态：

```typescript
/** Agent 状态管理 */

enum AgentState {
  IDLE = "idle", // 空闲
  PLANNING = "planning", // 规划中
  EXECUTING = "executing", // 执行中
  REFLECTING = "reflecting", // 反思中
  WAITING = "waiting", // 等待人类输入
  COMPLETED = "completed", // 已完成
  FAILED = "failed", // 失败
}

interface StepResult {
  step: string;
  result: string;
  success: boolean;
}

/** Agent 执行上下文 */
interface ExecutionContext {
  task: string;
  state: AgentState;
  plan: string[];
  currentStep: number;
  stepResults: StepResult[];
  totalToolCalls: number;
  totalTokensUsed: number;
  startTime: Date;
  errors: string[];
}

function createExecutionContext(
  task: string = ""
): ExecutionContext {
  return {
    task,
    state: AgentState.IDLE,
    plan: [],
    currentStep: 0,
    stepResults: [],
    totalToolCalls: 0,
    totalTokensUsed: 0,
    startTime: new Date(),
    errors: [],
  };
}

function getElapsedSeconds(ctx: ExecutionContext): number {
  return (Date.now() - ctx.startTime.getTime()) / 1000;
}

function getProgress(ctx: ExecutionContext): string {
  if (ctx.plan.length === 0) return "0%";
  const pct = ((ctx.currentStep / ctx.plan.length) * 100).toFixed(0);
  return `${ctx.currentStep}/${ctx.plan.length} (${pct}%)`;
}

function toSummary(ctx: ExecutionContext): Record<string, any> {
  return {
    task: ctx.task,
    state: ctx.state,
    progress: getProgress(ctx),
    tool_calls: ctx.totalToolCalls,
    elapsed: `${getElapsedSeconds(ctx).toFixed(1)}s`,
    errors: ctx.errors.length,
  };
}
```

## 小结

- **Planning 模式**：先让 LLM 制定计划（纯文本），再逐步执行。适合复杂、多步骤的任务
- **Reflection 模式**：执行后让 LLM 评估结果质量，发现问题自动重试。提升输出可靠性
- **Router 模式**：一个 LLM 分类，分发给专业子 Agent。适合多领域服务
- **Human-in-the-Loop**：高风险操作人类审批，低风险自动放行。安全和效率的平衡
- **状态管理**：用 interface + 工厂函数追踪 Agent 执行进度、错误、资源消耗

::: tip Anthropic 的建议
从最简单的架构开始。只有当单 Agent 无法满足需求时，才引入 Planning、Reflection 等复杂模式。过度设计是 Agent 开发中最常见的错误。
:::

## 练习

1. **Planning + Execution**：用 PlanningAgent 完成"调研 3 种 TypeScript Web 框架的优缺点，写成对比报告"。观察它的计划是否合理。
2. **Reflection 实验**：故意给一个含糊的任务（如"写点东西"），观察 Reflection 如何评估结果并建议改进。
3. **Router 扩展**：给 RouterAgent 添加第 4 种类别"数据分析"，测试分类准确率。
4. **组合模式**：把 Planning + Reflection 组合：规划后执行，执行后反思，反思不通过则重新规划。

## 参考资源

- [Anthropic: Building Effective Agents](https://docs.anthropic.com/en/docs/build-with-claude/agentic) -- 官方 Agent 构建指南
- [Plan-and-Solve Prompting (arXiv:2305.04091)](https://arxiv.org/abs/2305.04091) -- Planning 模式论文
- [Reflexion (arXiv:2303.11366)](https://arxiv.org/abs/2303.11366) -- Reflection 模式论文
- [Andrew Ng: Agentic Design Patterns](https://www.deeplearning.ai/the-batch/how-agents-can-improve-llm-performance/) -- Agent 设计模式讲解
- [LangGraph: Agent Architectures](https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/) -- 架构模式系统总结
