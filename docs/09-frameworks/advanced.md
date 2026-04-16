# Agent 框架高级：源码分析与自建框架

::: tip 学习目标
- 理解主流框架的核心设计模式和源码结构
- 掌握自建 Agent 框架的设计哲学：组合优于继承、接口稳定实现可变、中间件管道
- 完整实现一个约 500 行代码的 Mini Agent Framework，包含 Model Provider、Tool Registry、Memory Manager、Middleware Pipeline 和 Agent Runner

**学完你能做到：** 阅读和理解 LangChain/OpenAI Agents SDK 的核心源码，从零构建一个可投入生产的轻量 Agent 框架。
:::

## 为什么要自建框架

使用第三方框架可以快速起步，但在生产环境中往往遇到瓶颈：

1. **过度抽象**：框架为了通用性引入大量抽象层，调试困难
2. **版本不稳定**：快速迭代的框架频繁 breaking changes
3. **性能开销**：通用化带来的序列化、适配、中间件开销
4. **定制受限**：特殊需求难以在框架约束内实现

::: info 何时该自建
- 你的 Agent 逻辑相对固定，不需要框架的通用调度能力
- 你需要极致的性能和可控性
- 你的团队有能力维护核心模块
- 你希望深入理解 Agent 的底层运作机制（学习目的）
:::

### 现有框架的优缺点

| 框架 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| **LangChain** | 生态丰富、社区大 | 抽象过重、API 变化频繁 | 快速原型 |
| **LangGraph** | 图状态机清晰 | 学习曲线陡 | 复杂工作流 |
| **CrewAI** | 多 Agent 协作直观 | 灵活性不足 | 多 Agent 场景 |
| **OpenAI Agents SDK** | 简洁、类型安全 | 绑定 OpenAI | OpenAI 生态 |
| **自建框架** | 完全可控 | 需自行维护 | 生产定制 |

## 设计哲学

```typescript
/**
 * Mini Agent Framework 设计原则：
 * 1. 简单 -- 核心代码不超过 500 行
 * 2. 可扩展 -- 插件化的 Model/Tool/Memory
 * 3. 可测试 -- 每个模块可独立测试
 * 4. 透明 -- 没有隐藏的魔法，代码即文档
 */
```

### 原则 1：组合优于继承

```typescript
// 坏的设计：深度继承
class BaseAgent { /* ... */ }
class ToolAgent extends BaseAgent { /* ... */ }
class RAGToolAgent extends ToolAgent { /* ... */ }
class RAGToolMemoryAgent extends RAGToolAgent { /* ... */ } // 无穷无尽

// 好的设计：组合
class Agent {
  model: ModelProvider;      // 可插拔的模型
  tools: ToolRegistry;       // 可插拔的工具
  memory: MemoryManager;     // 可插拔的记忆

  constructor(model: ModelProvider, tools: ToolRegistry, memory: MemoryManager) {
    this.model = model;
    this.tools = tools;
    this.memory = memory;
  }
}
```

### 原则 2：接口稳定，实现可变

```typescript
// 稳定的接口
interface ModelProvider {
  generate(
    messages: Message[],
    tools?: ToolDefinition[]
  ): Promise<ModelResponse>;
}

// 可变的实现 -- 换模型只需换实现类
class AnthropicProvider implements ModelProvider { /* ... */ }
class OpenAIProvider implements ModelProvider { /* ... */ }
```

### 原则 3：中间件管道

```typescript
// 类似 Express/Koa 的中间件模式
const pipeline: Middleware[] = [
  new InputValidator(),      // 输入验证
  new CostTracker(),          // 成本追踪
  new RateLimiter(),          // 速率限制
  new AgentRunner(),          // 核心执行
  new OutputValidator(),      // 输出验证
];
```

## 核心抽象层

### 类型定义

```typescript
/** mini-agent/types.ts -- 核心类型定义 */

interface ToolCall {
  id: string;
  name: string;
  arguments: Record<string, unknown>;
}

interface ToolResult {
  toolCallId: string;
  output: string;
  isError: boolean;
}

interface ModelResponse {
  content: string;
  toolCalls: ToolCall[];
  stopReason: string;
  inputTokens: number;
  outputTokens: number;
  model: string;
}

// 便捷计算属性用函数代替
function hasToolCalls(response: ModelResponse): boolean {
  return response.toolCalls.length > 0;
}

interface ToolDefinition {
  name: string;
  description: string;
  parameters: Record<string, unknown>;
  handler?: ToolHandler;
}

type ToolHandler = (
  args: Record<string, unknown>
) => string | Promise<string>;

interface AgentResult {
  output: string;
  iterations: number;
  totalInputTokens: number;
  totalOutputTokens: number;
  toolCallsMade: string[];
  model: string;
}

function totalTokens(result: AgentResult): number {
  return result.totalInputTokens + result.totalOutputTokens;
}
```

### Model Provider

```typescript
/** mini-agent/models.ts -- 模型提供者 */

import Anthropic from "@anthropic-ai/sdk";

interface ModelProvider {
  generate(
    messages: Record<string, unknown>[],
    options?: {
      tools?: ToolDefinition[];
      system?: string;
      maxTokens?: number;
    }
  ): Promise<ModelResponse>;
}

class AnthropicProvider implements ModelProvider {
  private model: string;
  private client: Anthropic;

  constructor(model: string = "claude-sonnet-4-20250514", apiKey?: string) {
    this.model = model;
    this.client = new Anthropic(apiKey ? { apiKey } : undefined);
  }

  private formatTools(tools: ToolDefinition[]): Anthropic.Tool[] {
    return tools.map((t) => ({
      name: t.name,
      description: t.description,
      input_schema: t.parameters as Anthropic.Tool["input_schema"],
    }));
  }

  async generate(
    messages: Anthropic.MessageParam[],
    options: {
      tools?: ToolDefinition[];
      system?: string;
      maxTokens?: number;
    } = {}
  ): Promise<ModelResponse> {
    const createParams: Anthropic.MessageCreateParams = {
      model: this.model,
      max_tokens: options.maxTokens ?? 4096,
      messages,
    };
    if (options.system) createParams.system = options.system;
    if (options.tools)
      createParams.tools = this.formatTools(options.tools);

    const resp = await this.client.messages.create(createParams);

    const textParts: string[] = [];
    const toolCalls: ToolCall[] = [];
    for (const block of resp.content) {
      if (block.type === "text") {
        textParts.push(block.text);
      } else if (block.type === "tool_use") {
        toolCalls.push({
          id: block.id,
          name: block.name,
          arguments: block.input as Record<string, unknown>,
        });
      }
    }

    return {
      content: textParts.join(""),
      toolCalls,
      stopReason: resp.stop_reason ?? "end_turn",
      inputTokens: resp.usage.input_tokens,
      outputTokens: resp.usage.output_tokens,
      model: this.model,
    };
  }
}

class OpenAIProvider implements ModelProvider {
  private model: string;
  private client: unknown; // OpenAI client

  constructor(model: string = "gpt-4o", apiKey?: string) {
    // 动态导入 openai
    const { default: OpenAI } = require("openai");
    this.model = model;
    this.client = new OpenAI(apiKey ? { apiKey } : undefined);
  }

  private formatTools(tools: ToolDefinition[]) {
    return tools.map((t) => ({
      type: "function" as const,
      function: {
        name: t.name,
        description: t.description,
        parameters: t.parameters,
      },
    }));
  }

  async generate(
    messages: Record<string, unknown>[],
    options: {
      tools?: ToolDefinition[];
      system?: string;
      maxTokens?: number;
    } = {}
  ): Promise<ModelResponse> {
    const msgs = options.system
      ? [{ role: "system", content: options.system }, ...messages]
      : [...messages];

    const createParams: Record<string, unknown> = {
      model: this.model,
      max_tokens: options.maxTokens ?? 4096,
      messages: msgs,
    };
    if (options.tools)
      createParams.tools = this.formatTools(options.tools);

    const resp = await (this.client as any).chat.completions.create(
      createParams
    );
    const choice = resp.choices[0];

    const toolCalls: ToolCall[] = [];
    if (choice.message.tool_calls) {
      for (const tc of choice.message.tool_calls) {
        toolCalls.push({
          id: tc.id,
          name: tc.function.name,
          arguments: JSON.parse(tc.function.arguments),
        });
      }
    }

    return {
      content: choice.message.content ?? "",
      toolCalls,
      stopReason: choice.finish_reason,
      inputTokens: resp.usage.prompt_tokens,
      outputTokens: resp.usage.completion_tokens,
      model: this.model,
    };
  }
}
```

### Tool Registry

`tool()` 辅助函数从参数自动生成 JSON Schema，注册表统一管理和执行：

```typescript
/** mini-agent/tools.ts -- 工具注册与执行 */

const TYPE_MAP: Record<string, string> = {
  string: "string",
  number: "number",
  boolean: "boolean",
  object: "object",
};

// 工具构建辅助函数（等价于 Python @tool 装饰器）
function defineTool(config: {
  name: string;
  description: string;
  parameters: Record<
    string,
    { type: string; description?: string; required?: boolean }
  >;
  handler: ToolHandler;
}): ToolDefinition {
  const props: Record<string, unknown> = {};
  const required: string[] = [];

  for (const [paramName, paramConfig] of Object.entries(
    config.parameters
  )) {
    props[paramName] = {
      type: paramConfig.type,
      ...(paramConfig.description
        ? { description: paramConfig.description }
        : {}),
    };
    if (paramConfig.required !== false) {
      required.push(paramName);
    }
  }

  return {
    name: config.name,
    description: config.description,
    parameters: {
      type: "object",
      properties: props,
      required,
    },
    handler: config.handler,
  };
}

class ToolRegistry {
  private tools: Map<string, ToolDefinition> = new Map();

  register(toolDef: ToolDefinition): void {
    this.tools.set(toolDef.name, toolDef);
  }

  get definitions(): ToolDefinition[] {
    return Array.from(this.tools.values());
  }

  async execute(call: ToolCall): Promise<ToolResult> {
    const defn = this.tools.get(call.name);
    if (!defn || !defn.handler) {
      return {
        toolCallId: call.id,
        output: `未知工具: ${call.name}`,
        isError: true,
      };
    }
    try {
      const result = await defn.handler(call.arguments);
      return { toolCallId: call.id, output: String(result), isError: false };
    } catch (e) {
      return {
        toolCallId: call.id,
        output: `执行错误: ${e}`,
        isError: true,
      };
    }
  }

  async executeParallel(calls: ToolCall[]): Promise<ToolResult[]> {
    return Promise.all(calls.map((c) => this.execute(c)));
  }
}
```

### Memory Manager

```typescript
/** mini-agent/memory.ts -- 记忆管理 */

interface MemoryManager {
  add(message: Record<string, unknown>): void;
  getMessages(): Record<string, unknown>[];
  clear(): void;
  readonly size: number;
}

/** 简单缓冲记忆 -- 保留最近 N 条消息 */
class BufferMemory implements MemoryManager {
  private max: number;
  private messages: Record<string, unknown>[] = [];

  constructor(maxMessages: number = 100) {
    this.max = maxMessages;
  }

  add(message: Record<string, unknown>): void {
    this.messages.push(message);
    if (this.messages.length > this.max) {
      this.messages = this.messages.slice(-this.max);
    }
  }

  getMessages(): Record<string, unknown>[] {
    return [...this.messages];
  }

  clear(): void {
    this.messages = [];
  }

  get size(): number {
    return this.messages.length;
  }
}

/** 滑动窗口记忆 -- 保留最近 N 轮对话 */
class WindowMemory implements MemoryManager {
  private window: number;
  private messages: Record<string, unknown>[] = [];

  constructor(windowSize: number = 10) {
    this.window = windowSize;
  }

  add(message: Record<string, unknown>): void {
    this.messages.push(message);
  }

  getMessages(): Record<string, unknown>[] {
    const keep = this.window * 2; // 每轮 = user + assistant
    return this.messages.length > keep
      ? this.messages.slice(-keep)
      : [...this.messages];
  }

  clear(): void {
    this.messages = [];
  }

  get size(): number {
    return this.messages.length;
  }
}
```

## 插件系统

### 中间件管道

借鉴 Express/Koa 的洋葱模型 -- 每个中间件可以在 Agent 执行前后插入逻辑：

```typescript
/** mini-agent/middleware.ts -- 中间件系统 */

interface Context {
  metadata: Record<string, unknown>;
}

function createContext(): Context {
  return { metadata: {} };
}

type NextFn = () => Promise<unknown>;

interface Middleware {
  (ctx: Context, next: NextFn): Promise<unknown>;
}

class Pipeline {
  private middlewares: Middleware[] = [];

  use(mw: Middleware): this {
    this.middlewares.push(mw);
    return this;
  }

  async run<T>(
    ctx: Context,
    coreFn: (ctx: Context) => Promise<T>
  ): Promise<T> {
    let index = 0;

    const next: NextFn = async () => {
      if (index < this.middlewares.length) {
        const mw = this.middlewares[index++];
        return mw(ctx, next);
      } else {
        return coreFn(ctx);
      }
    };

    return (await next()) as T;
  }
}
```

### 内置中间件

```typescript
/** 耗时统计 */
const timingMiddleware: Middleware = async (ctx, next) => {
  const start = performance.now();
  const result = await next();
  ctx.metadata.durationMs = Math.round(performance.now() - start);
  return result;
};

/** 成本追踪 */
const PRICES: Record<string, [number, number]> = {
  "claude-sonnet-4-20250514": [3.0, 15.0],
  "claude-haiku-3-20250414": [0.25, 1.25],
};

const costMiddleware: Middleware = async (ctx, next) => {
  const result = await next();
  if (
    result &&
    typeof result === "object" &&
    "totalInputTokens" in result
  ) {
    const r = result as AgentResult;
    const [pi, po] = PRICES[r.model] ?? [3.0, 15.0];
    ctx.metadata.costUsd =
      Math.round(
        (r.totalInputTokens * pi + r.totalOutputTokens * po) / 1e6 * 1e6
      ) / 1e6;
  }
  return result;
};

/** 迭代次数限制 */
function maxIterationsMiddleware(maxIter: number = 25): Middleware {
  return async (ctx, next) => {
    ctx.metadata.maxIterations = maxIter;
    return next();
  };
}
```

### Hook 系统

比中间件更细粒度的生命周期钩子：

```typescript
/** Hook 系统 -- 细粒度的生命周期钩子 */

type HookHandler = (args: Record<string, unknown>) => unknown | Promise<unknown>;

const VALID_HOOKS = [
  "before_llm_call",    // LLM 调用前
  "after_llm_call",     // LLM 调用后
  "before_tool_call",   // 工具调用前
  "after_tool_call",    // 工具调用后
  "on_error",           // 发生错误时
  "on_iteration",       // 每次迭代时
  "on_complete",        // 任务完成时
] as const;

type HookEvent = (typeof VALID_HOOKS)[number];

class HookManager {
  private hooks: Map<HookEvent, HookHandler[]> = new Map();

  on(event: HookEvent, handler: HookHandler): void {
    if (!VALID_HOOKS.includes(event)) {
      throw new Error(
        `未知钩子: ${event}. 可用: ${VALID_HOOKS.join(", ")}`
      );
    }
    const handlers = this.hooks.get(event) ?? [];
    handlers.push(handler);
    this.hooks.set(event, handlers);
  }

  async emit(
    event: HookEvent,
    args: Record<string, unknown> = {}
  ): Promise<unknown[]> {
    const handlers = this.hooks.get(event) ?? [];
    const results: unknown[] = [];
    for (const handler of handlers) {
      results.push(await handler(args));
    }
    return results;
  }
}

// 使用示例
const hooks = new HookManager();
hooks.on("before_llm_call", ({ messages }) =>
  console.log(
    `即将调用 LLM，消息数: ${(messages as unknown[]).length}`
  )
);
hooks.on("after_tool_call", ({ toolName, result }) =>
  console.log(`工具 ${toolName} 执行完成`)
);
```

### 动态工具注册

支持运行时增减工具，按上下文过滤相关工具：

```typescript
type ChangeCallback = (
  action: "added" | "removed",
  toolName: string
) => void;

class DynamicToolRegistry {
  private tools: Map<string, ToolDefinition> = new Map();
  private onChangeCallbacks: ChangeCallback[] = [];

  register(toolDef: ToolDefinition): void {
    this.tools.set(toolDef.name, toolDef);
    this.notifyChange("added", toolDef.name);
  }

  unregister(name: string): void {
    if (this.tools.has(name)) {
      this.tools.delete(name);
      this.notifyChange("removed", name);
    }
  }

  onChange(callback: ChangeCallback): void {
    this.onChangeCallbacks.push(callback);
  }

  private notifyChange(
    action: "added" | "removed",
    toolName: string
  ): void {
    for (const cb of this.onChangeCallbacks) {
      cb(action, toolName);
    }
  }

  /** 根据上下文返回相关工具（而非全部） */
  getToolsForContext(context?: string): ToolDefinition[] {
    const allTools = Array.from(this.tools.values());
    if (!context) return allTools;

    const keywords = context.toLowerCase().split(/\s+/);
    const relevant = allTools.filter((defn) => {
      const desc = `${defn.name} ${defn.description}`.toLowerCase();
      return keywords.some((kw) => desc.includes(kw));
    });

    return relevant.length > 0 ? relevant : allTools;
  }
}
```

## 完整实现：Mini Agent Framework

将所有模块整合为完整框架。项目结构：

```
mini-agent/
├── index.ts             # 对外导出
├── types.ts             # 类型定义（58 行）
├── models.ts            # Model Provider（95 行）
├── tools.ts             # Tool Registry（90 行）
├── memory.ts            # Memory Manager（60 行）
├── middleware.ts         # 中间件系统（80 行）
├── agent.ts             # Agent Runner（120 行）
总计约 503 行
```

### Agent Runner -- 核心循环

```typescript
/** mini-agent/agent.ts -- Agent Runner 核心循环 */

import Anthropic from "@anthropic-ai/sdk";

class MiniAgent {
  /** Mini Agent Framework 核心类 */
  model: ModelProvider;
  tools: ToolRegistry;
  memory: MemoryManager;
  systemPrompt: string;
  maxIterations: number;
  maxTokens: number;
  pipeline: Pipeline;

  constructor(config: {
    model: ModelProvider;
    tools?: ToolRegistry;
    memory?: MemoryManager;
    systemPrompt?: string;
    maxIterations?: number;
    maxTokens?: number;
  }) {
    this.model = config.model;
    this.tools = config.tools ?? new ToolRegistry();
    this.memory = config.memory ?? new BufferMemory();
    this.systemPrompt = config.systemPrompt ?? "";
    this.maxIterations = config.maxIterations ?? 25;
    this.maxTokens = config.maxTokens ?? 4096;
    this.pipeline = new Pipeline();
  }

  /** 添加中间件 */
  use(middleware: Middleware): this {
    this.pipeline.use(middleware);
    return this;
  }

  /** 运行 Agent */
  async run(message: string): Promise<AgentResult> {
    const ctx = createContext();
    ctx.metadata.maxIterations = this.maxIterations;

    return this.pipeline.run(ctx, (ctx) =>
      this.agentLoop(message, ctx)
    );
  }

  /** Agent 核心循环 */
  private async agentLoop(
    message: string,
    ctx: Context
  ): Promise<AgentResult> {
    this.memory.add({ role: "user", content: message });
    const messages = this.memory.getMessages();
    const toolDefs =
      this.tools.definitions.length > 0
        ? this.tools.definitions
        : undefined;

    let totalIn = 0;
    let totalOut = 0;
    const toolCallsMade: string[] = [];
    const maxIter =
      (ctx.metadata.maxIterations as number) ?? this.maxIterations;

    let lastResponse: ModelResponse | null = null;

    for (let iteration = 0; iteration < maxIter; iteration++) {
      const response = await this.model.generate(
        messages as Anthropic.MessageParam[],
        {
          tools: toolDefs,
          system: this.systemPrompt,
          maxTokens: this.maxTokens,
        }
      );

      totalIn += response.inputTokens;
      totalOut += response.outputTokens;
      lastResponse = response;

      // 没有工具调用 -> 返回最终结果
      if (!hasToolCalls(response)) {
        this.memory.add({
          role: "assistant",
          content: response.content,
        });
        return {
          output: response.content,
          iterations: iteration + 1,
          totalInputTokens: totalIn,
          totalOutputTokens: totalOut,
          toolCallsMade,
          model: response.model,
        };
      }

      // 处理工具调用
      messages.push({
        role: "assistant",
        content: response.content,
      });

      const results = await this.tools.executeParallel(
        response.toolCalls
      );
      toolCallsMade.push(...response.toolCalls.map((tc) => tc.name));

      const toolResultContent = results.map((r) => ({
        type: "tool_result" as const,
        tool_use_id: r.toolCallId,
        content: r.output,
        is_error: r.isError,
      }));
      messages.push({ role: "user", content: toolResultContent });
    }

    return {
      output: "达到最大迭代次数",
      iterations: maxIter,
      totalInputTokens: totalIn,
      totalOutputTokens: totalOut,
      toolCallsMade,
      model: lastResponse?.model ?? "",
    };
  }
}
```

### 对外导出

```typescript
/** mini-agent/index.ts -- 一个约 500 行代码的 Agent 框架 */

export {
  MiniAgent,
  AnthropicProvider,
  OpenAIProvider,
  defineTool,
  ToolRegistry,
  BufferMemory,
  WindowMemory,
  Pipeline,
  HookManager,
};

// 导出类型
export type {
  ModelProvider,
  MemoryManager,
  Middleware,
  Context,
  ToolDefinition,
  ToolHandler,
  AgentResult,
  ModelResponse,
  ToolCall,
  ToolResult,
};
```

### 使用示例

```typescript
/** example.ts -- Mini Agent Framework 使用示例 */

import {
  MiniAgent,
  AnthropicProvider,
  ToolRegistry,
  defineTool,
  BufferMemory,
} from "./mini-agent";

// 1. 定义工具
const calculate = defineTool({
  name: "calculate",
  description: "计算数学表达式",
  parameters: {
    expression: {
      type: "string",
      description: "合法的数学表达式",
    },
  },
  handler: (args) => {
    try {
      return String(eval(args.expression as string));
    } catch (e) {
      return `计算错误: ${e}`;
    }
  },
});

const getTime = defineTool({
  name: "get_time",
  description: "获取当前时间",
  parameters: {},
  handler: () => new Date().toISOString(),
});

// 2. 注册工具
const tools = new ToolRegistry();
tools.register(calculate);
tools.register(getTime);

// 3. 创建 Agent
const agent = new MiniAgent({
  model: new AnthropicProvider("claude-sonnet-4-20250514"),
  tools,
  memory: new BufferMemory(50),
  systemPrompt: "你是一个有用的助手。使用工具来回答问题。",
  maxIterations: 10,
});

// 4. 添加中间件
agent.use(timingMiddleware);
agent.use(costMiddleware);

// 5. 运行
async function main() {
  const result = await agent.run(
    "现在几点了？然后帮我算一下 123 * 456 + 789"
  );
  console.log(`回复: ${result.output}`);
  console.log(`迭代次数: ${result.iterations}`);
  console.log(`Token 用量: ${totalTokens(result)}`);
  console.log(`工具调用: ${result.toolCallsMade.join(", ")}`);
}

main();
```

## 小结

- 自建框架不是重复造轮子，而是为了：深入理解底层机制、获得完全可控的生产系统、按需定制不受框架限制
- 核心设计哲学：组合优于继承、接口稳定实现可变、中间件管道模式
- **Model Provider** 统一不同 LLM 的调用接口，换模型只需换实现类
- **Tool Registry** 用辅助函数自动生成 schema，注册表统一管理和并行执行
- **Memory Manager** 可插拔的记忆后端，从简单缓冲到滑动窗口
- **Middleware Pipeline** 洋葱模型，before/after 逻辑自由组合
- **Hook 系统** 细粒度的生命周期事件，支持动态工具注册
- 500 行代码涵盖了生产级框架的所有核心组件

## 练习题

1. 给 Mini Agent Framework 添加一个 `SummaryMemory`：当消息超过阈值时，自动调用 LLM 压缩历史消息为摘要。
2. 实现一个 `RetryMiddleware`：当 LLM 调用失败时自动重试，支持指数退避和最大重试次数配置。
3. 给框架添加 Streaming 支持：实现 `ModelProvider.stream()` 方法和 `Agent.run_stream()` 方法。
4. 阅读 [OpenAI Agents SDK 源码](https://github.com/openai/openai-agents-python)，对比它的抽象设计与本章实现的异同。

## 参考资源

- [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) -- 约 1000 行的官方框架参考
- [Pydantic AI](https://github.com/pydantic/pydantic-ai) -- 类型安全的 Agent 框架
- [smolagents (Hugging Face)](https://github.com/huggingface/smolagents) -- 轻量级 Agent 库
- [LangChain 源码](https://github.com/langchain-ai/langchain) -- 参考其抽象设计
- [Clean Architecture (Robert C. Martin)](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html) -- 架构设计原则
- [Koa.js 中间件机制](https://koajs.com/) -- 洋葱模型参考
- [arXiv:2308.08155 - A Survey on LLM-based Autonomous Agents](https://arxiv.org/abs/2308.08155) -- Agent 架构综述
- [Anthropic Claude Agent 文档](https://docs.anthropic.com/en/docs/build-with-claude/agentic-systems) -- 官方 Agent 指南
