# 评估高级：可观测性与生产监控

::: tip 学习目标
- 理解 Agent 可观测性的三大支柱：Traces、Metrics、Logs
- 掌握自建 Tracing 方案的实现（Span/Trace 数据结构、TracedAgent）
- 学会使用 LangSmith 和 Phoenix 两个主流可观测性平台
- 能够搭建生产级的监控面板和回归测试追踪器

**学完你能做到：** 为你的 Agent 系统添加完整的执行轨迹追踪，搭建一个包含延迟/Token/错误率的监控面板，以及一个能检测性能退化的回归测试系统。
:::

## 为什么需要可观测性

Agent 的执行过程就像一个黑盒：输入一个问题，输出一个答案，但中间发生了什么？调用了几次 LLM？用了哪些工具？每步花了多少时间和 Token？

没有可观测性，你无法：
- **定位性能瓶颈**：哪一步最慢？
- **排查质量问题**：哪一步出了错？
- **优化成本**：哪些调用是冗余的？
- **理解决策过程**：Agent 为什么选择调用这个工具而不是那个？

## 自建 Tracing 方案

我们先从零搭一个 Tracing 系统，理解核心概念后再看第三方工具。

### Span 和 Trace 数据结构

```typescript
interface SpanData {
  name: string;
  duration_ms: number;
  input: Record<string, unknown>;
  output: Record<string, string>;
  metadata: Record<string, unknown>;
  error: string | null;
  children: SpanData[];
}

class Span {
  /**
   * 一个追踪 Span（执行步骤）
   *
   * Span 是 Tracing 的基本单位，表示一次操作（LLM 调用、工具执行等）。
   * Span 可以嵌套——一个 Agent 运行的 root span 包含多个 LLM 调用和工具调用的子 span。
   */
  name: string;
  startTime: number;
  endTime: number = 0;
  inputData: Record<string, unknown>;
  outputData: Record<string, unknown>;
  metadata: Record<string, unknown>;
  children: Span[];
  error: string | null;

  constructor(params: {
    name: string;
    inputData?: Record<string, unknown>;
    outputData?: Record<string, unknown>;
    metadata?: Record<string, unknown>;
  }) {
    this.name = params.name;
    this.startTime = Date.now();
    this.inputData = params.inputData ?? {};
    this.outputData = params.outputData ?? {};
    this.metadata = params.metadata ?? {};
    this.children = [];
    this.error = null;
  }

  end(): void {
    this.endTime = Date.now();
  }

  get durationMs(): number {
    if (this.endTime) {
      return this.endTime - this.startTime;
    }
    return 0;
  }

  toDict(): SpanData {
    const output: Record<string, string> = {};
    for (const [k, v] of Object.entries(this.outputData)) {
      output[k] = String(v).slice(0, 200);
    }
    return {
      name: this.name,
      duration_ms: Math.round(this.durationMs * 10) / 10,
      input: this.inputData,
      output,
      metadata: this.metadata,
      error: this.error,
      children: this.children.map((c) => c.toDict()),
    };
  }
}

interface TraceData {
  trace_id: string;
  metadata: Record<string, unknown>;
  root: SpanData | null;
  total_duration_ms: number;
}

class Trace {
  /** 一次完整的 Agent 执行追踪 */
  traceId: string;
  rootSpan: Span | null;
  metadata: Record<string, unknown>;

  constructor(traceId: string, metadata: Record<string, unknown> = {}) {
    this.traceId = traceId;
    this.rootSpan = null;
    this.metadata = metadata;
  }

  toDict(): TraceData {
    return {
      trace_id: this.traceId,
      metadata: this.metadata,
      root: this.rootSpan ? this.rootSpan.toDict() : null,
      total_duration_ms: this.rootSpan ? this.rootSpan.durationMs : 0,
    };
  }
}
```

### 带追踪的 Agent

```typescript
import Anthropic from "@anthropic-ai/sdk";
import { randomUUID } from "crypto";

const client = new Anthropic();

class TracedAgent {
  /** 带追踪的 Agent：每次执行都记录完整的 Trace */

  name: string;
  traces: Trace[] = [];

  constructor(name: string) {
    this.name = name;
  }

  async run(question: string): Promise<[string, Trace]> {
    /** 运行并追踪 */
    const trace = new Trace(randomUUID().slice(0, 8), {
      agent: this.name,
      question: question.slice(0, 100),
    });

    const root = new Span({
      name: "agent_run",
      inputData: { question },
    });
    trace.rootSpan = root;

    const messages: Anthropic.MessageParam[] = [
      { role: "user", content: question },
    ];
    const tools: Anthropic.Tool[] = [
      {
        name: "search",
        description: "搜索信息",
        input_schema: {
          type: "object" as const,
          properties: { query: { type: "string" } },
          required: ["query"],
        },
      },
    ];

    let answer = "";
    let step = 0;
    let response: Anthropic.Message | undefined;

    while (step < 5) {
      step++;

      // 追踪 LLM 调用
      const llmSpan = new Span({
        name: `llm_call_${step}`,
        inputData: { messages_count: messages.length },
      });

      try {
        response = await client.messages.create({
          model: "claude-sonnet-4-20250514",
          max_tokens: 1024,
          tools,
          messages,
        });
        llmSpan.outputData = {
          stop_reason: response.stop_reason,
          input_tokens: response.usage.input_tokens,
          output_tokens: response.usage.output_tokens,
        };
        llmSpan.metadata = {
          model: "claude-sonnet-4-20250514",
          total_tokens:
            response.usage.input_tokens + response.usage.output_tokens,
        };
      } catch (e) {
        llmSpan.error = String(e);
      } finally {
        llmSpan.end();
        root.children.push(llmSpan);
      }

      if (!response) break;

      if (response.stop_reason === "end_turn") {
        const firstBlock = response.content[0];
        answer = firstBlock.type === "text" ? firstBlock.text : "";
        root.outputData = { answer };
        break;
      }

      // 追踪工具调用
      for (const block of response.content) {
        if (block.type === "tool_use") {
          const toolSpan = new Span({
            name: `tool_${block.name}`,
            inputData: block.input as Record<string, unknown>,
          });
          let result: string;
          try {
            const query = (block.input as Record<string, string>).query ?? "";
            result = `搜索 '${query}' 的结果...`;
            toolSpan.outputData = { result };
          } catch (e) {
            toolSpan.error = String(e);
            result = `工具错误: ${e}`;
          } finally {
            toolSpan.end();
            root.children.push(toolSpan);
          }

          messages.push({
            role: "assistant",
            content: response.content,
          });
          messages.push({
            role: "user",
            content: [
              {
                type: "tool_result",
                tool_use_id: block.id,
                content: result,
              },
            ],
          });
        }
      }
    }

    root.end();
    this.traces.push(trace);
    return [answer, trace];
  }

  printTrace(trace: Trace): void {
    /** 打印追踪信息——直观展示每步的耗时和 Token 消耗 */
    const data = trace.toDict();
    console.log(`\n${"=".repeat(60)}`);
    console.log(`Trace ID: ${data.trace_id}`);
    console.log(`Total Duration: ${data.total_duration_ms.toFixed(0)}ms`);
    console.log("=".repeat(60));

    if (data.root) {
      this.printSpan(data.root, 0);
    }
  }

  private printSpan(span: SpanData, indent: number): void {
    const prefix = "  ".repeat(indent);
    const duration = span.duration_ms;
    const error = span.error ? " [ERROR]" : "";
    const tokens = (span.metadata as Record<string, unknown>)?.total_tokens;
    const tokensStr = tokens ? ` (${tokens} tokens)` : "";
    console.log(
      `${prefix}|- ${span.name}: ${duration.toFixed(0)}ms${tokensStr}${error}`
    );
    for (const child of span.children ?? []) {
      this.printSpan(child, indent + 1);
    }
  }
}
```

输出示例：

```
============================================================
Trace ID: a1b2c3d4
Total Duration: 2340ms
============================================================
|- agent_run: 2340ms
  |- llm_call_1: 1200ms (580 tokens)
  |- tool_search: 150ms
  |- llm_call_2: 990ms (720 tokens)
```

一眼就能看出：LLM 调用是性能瓶颈，工具调用很快。这就是可观测性的价值。

## 使用 LangSmith

LangSmith 是 LangChain 官方的追踪和评估平台，也是目前最成熟的 LLM 可观测性工具之一。

```typescript
// 安装和配置
// npm install langsmith @langchain/openai
// export LANGCHAIN_TRACING_V2=true
// export LANGCHAIN_API_KEY=your_key
// export LANGCHAIN_PROJECT=my_agent_project

import { traceable } from "langsmith/traceable";
import { ChatOpenAI } from "@langchain/openai";

const myAgent = traceable(
  async function myAgent(question: string): Promise<string> {
    /** LangSmith 会自动追踪这个函数的输入输出 */
    const model = new ChatOpenAI({ modelName: "gpt-4o-mini" });
    const response = await model.invoke(question);
    return response.content as string;
  },
  { name: "my_agent" }
);

const retrieveDocs = traceable(
  async function retrieveDocs(query: string): Promise<string[]> {
    /** 子步骤也会被追踪 */
    return [`Document about ${query}`];
  },
  { name: "retrieval" }
);

const ragQuery = traceable(
  async function ragQuery(question: string): Promise<string> {
    const docs = await retrieveDocs(question);
    const context = docs.join("\n");
    return myAgent(`Context: ${context}\nQuestion: ${question}`);
  },
  { name: "rag_pipeline" }
);

// 调用后，追踪数据自动上传到 LangSmith 平台
const result = await ragQuery("什么是向量数据库？");
```

LangSmith 的核心价值在于它的 Web UI——你可以在浏览器中看到每次 Agent 运行的完整执行树，包括每步的输入输出、耗时、Token 消耗，还能按项目分组管理和对比不同版本。

## 使用 Phoenix (Arize)

Phoenix 是一个开源的 LLM 可观测性工具，可以本地部署，不需要将数据发送到第三方。

```typescript
// npm install arize-phoenix-otel openai

import OpenAI from "openai";

// Phoenix 的 Node.js SDK 通过 OpenTelemetry 集成实现追踪
// 具体配置参考 Phoenix 文档的 TypeScript/Node.js 章节

// 启动 Phoenix 本地服务（命令行）：
// npx arize-phoenix serve

// 在代码中注册 OpenTelemetry 追踪：
// import { registerInstrumentations } from "@opentelemetry/instrumentation";
// import { OpenAIInstrumentation } from "@arizeai/openinference-instrumentation-openai";

const openaiClient = new OpenAI();

const response = await openaiClient.chat.completions.create({
  model: "gpt-4o-mini",
  messages: [{ role: "user", content: "Hello" }],
});

// 打开 Phoenix UI 查看追踪
console.log("Phoenix UI: http://localhost:6006");
```

::: tip LangSmith vs Phoenix 怎么选
- **LangSmith**：云服务，功能最全，适合团队使用，有免费额度
- **Phoenix**：开源，本地部署，数据不出域，适合隐私要求高的场景
- 两者都支持 OpenTelemetry 标准，可以互相迁移
:::

## 关键指标监控面板

```typescript
class MetricsDashboard {
  /** 指标监控面板：聚合多次 Trace 数据 */

  private traces: TraceData[] = [];

  addTrace(trace: Trace): void {
    this.traces.push(trace.toDict());
  }

  getMetrics(): Record<string, unknown> {
    /** 计算关键指标 */
    if (this.traces.length === 0) {
      return {};
    }

    const durations = this.traces.map((t) => t.total_duration_ms);
    const allTokens: number[] = [];
    const llmCalls: number[] = [];
    const toolCalls: number[] = [];
    let errors = 0;

    for (const trace of this.traces) {
      const root = trace.root;
      if (!root) continue;
      for (const child of root.children ?? []) {
        if (child.name.startsWith("llm_call")) {
          llmCalls.push(child.duration_ms);
          const tokens = (child.metadata as Record<string, unknown>)
            ?.total_tokens as number | undefined;
          if (tokens) {
            allTokens.push(tokens);
          }
        } else if (child.name.startsWith("tool_")) {
          toolCalls.push(child.duration_ms);
        }
        if (child.error) {
          errors++;
        }
      }
    }

    const sorted = [...durations].sort((a, b) => a - b);
    return {
      total_traces: this.traces.length,
      latency: {
        avg_ms: durations.reduce((a, b) => a + b, 0) / durations.length,
        p50_ms: sorted[Math.floor(sorted.length / 2)],
        p95_ms:
          sorted.length > 20
            ? sorted[Math.floor(sorted.length * 0.95)]
            : Math.max(...sorted),
        max_ms: Math.max(...sorted),
      },
      tokens: {
        avg_per_trace:
          allTokens.length > 0
            ? allTokens.reduce((a, b) => a + b, 0) / allTokens.length
            : 0,
        total: allTokens.reduce((a, b) => a + b, 0),
      },
      calls: {
        avg_llm_calls: llmCalls.length / this.traces.length,
        avg_tool_calls: toolCalls.length / this.traces.length,
      },
      errors: {
        total: errors,
        rate: errors / Math.max(llmCalls.length + toolCalls.length, 1),
      },
    };
  }
}
```

## 回归测试追踪

每次修改 Agent 后运行回归测试，确保改进不会引入退化。

```typescript
import * as fs from "fs";

interface HistoryEntry {
  version: string;
  timestamp: string;
  summary: Record<string, number>;
}

class RegressionTracker {
  /** 回归测试追踪器：检测版本间的性能变化 */

  private historyPath: string;
  private history: HistoryEntry[];

  constructor(historyPath: string = "./eval_history.json") {
    this.historyPath = historyPath;
    this.history = this.loadHistory();
  }

  private loadHistory(): HistoryEntry[] {
    try {
      const raw = fs.readFileSync(this.historyPath, "utf-8");
      return JSON.parse(raw);
    } catch {
      return [];
    }
  }

  record(version: string, report: Record<string, unknown>): void {
    /** 记录一次评估结果 */
    const entry: HistoryEntry = {
      version,
      timestamp: new Date().toISOString().replace("T", " ").slice(0, 19),
      summary: report.summary as Record<string, number>,
    };
    this.history.push(entry);
    fs.writeFileSync(
      this.historyPath,
      JSON.stringify(this.history, null, 2),
      "utf-8"
    );
  }

  compare(): Record<string, unknown> {
    /** 与上一版本对比 */
    if (this.history.length < 2) {
      return { message: "不够历史数据进行对比" };
    }

    const current = this.history[this.history.length - 1];
    const previous = this.history[this.history.length - 2];

    const comparison: Record<string, unknown> = {};
    const metrics = [
      "avg_auto_score",
      "avg_llm_score",
      "avg_latency_ms",
      "error_rate",
    ];

    for (const metric of metrics) {
      const currVal = current.summary[metric] ?? 0;
      const prevVal = previous.summary[metric] ?? 0;
      const delta = currVal - prevVal;
      // 分数上升=改进，延迟和错误率下降=改进
      const isBetter =
        !["avg_latency_ms", "error_rate"].includes(metric)
          ? delta > 0
          : delta < 0;
      comparison[metric] = {
        current: currVal,
        previous: prevVal,
        delta: Math.round(delta * 10000) / 10000,
        improved: isBetter,
      };
    }

    return comparison;
  }
}
```

::: tip 可观测性的三大支柱
1. **Traces**：完整的执行路径追踪——调试单次执行
2. **Metrics**：聚合的性能指标（P50/P95/P99、Token、错误率）——发现系统性问题
3. **Logs**：详细的运行时日志——深入排查具体问题

Agent 的可观测性需要三者结合才能全面覆盖。
:::

## 小结

- 可观测性是 Agent 生产化的必备能力，没有它你就是在盲飞
- 自建 Tracing 适合简单场景和学习目的，LangSmith/Phoenix 适合生产环境
- 关键指标：延迟（P50/P95/P99）、Token 消耗、错误率、工具调用效率
- 回归测试追踪每次迭代的效果变化，防止改进引入退化
- 建议在 CI/CD 中集成自动评估和回归测试

## 练习

1. 为你的 Agent 添加完整的 Tracing：记录每次 LLM 调用和工具调用的输入输出、耗时和 Token 消耗。
2. 搭建一个 LangSmith 或 Phoenix 项目，观察 Agent 的执行轨迹。
3. 实现一个告警系统：当 P95 延迟超过阈值或错误率超过 5% 时打印告警信息。

## 参考资源

- [LangSmith Documentation](https://docs.smith.langchain.com/) -- LangSmith 官方文档
- [Phoenix (Arize) Documentation](https://docs.arize.com/phoenix/) -- Phoenix 可观测性平台
- [OpenTelemetry for LLM Applications](https://opentelemetry.io/) -- OpenTelemetry 标准
- [Langfuse: Open Source LLM Observability](https://langfuse.com/docs) -- 开源可观测性平台
- [Braintrust: LLM Evaluation and Observability](https://www.braintrust.dev/docs) -- 评估和观测平台
- [Harrison Chase: Observability for LLM Apps](https://www.youtube.com/watch?v=Uv8Y8GgYTuA) -- LangChain 创始人讲解
