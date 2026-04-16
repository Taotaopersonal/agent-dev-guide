# 性能优化 · 进阶篇

::: tip 学习目标
- 掌握 asyncio + semaphore 的并发控制方案
- 理解并实现 Prompt Caching 和工具结果缓存
- 设计连接池以提高资源利用率
- 构建完整的异步 Agent 引擎
:::

::: info 学完你能做到
- 管理多用户并发请求，避免资源耗尽
- 用 Prompt Caching 节省 80%+ 的 LLM 输入成本
- 缓存工具调用结果，避免重复请求
- 构建一个生产级的异步 Agent 引擎
:::

## 并发控制：管理多用户请求

当多个用户同时使用你的 Agent 时，不加控制会导致资源耗尽。

### 全局限流 + 用户级限流

```typescript
// concurrency.ts — 并发请求管理器

class Semaphore {
  /** 简单的信号量实现 */
  private _count: number;
  private _waitQueue: Array<() => void> = [];

  constructor(max: number) {
    this._count = max;
  }

  async acquire(): Promise<void> {
    if (this._count > 0) {
      this._count--;
      return;
    }
    return new Promise<void>((resolve) => {
      this._waitQueue.push(resolve);
    });
  }

  release(): void {
    const next = this._waitQueue.shift();
    if (next) {
      next();
    } else {
      this._count++;
    }
  }
}

class ConcurrencyManager {
  /** 双层并发控制：全局限流 + 每用户限流 */

  private globalSem: Semaphore;
  private userSems: Map<string, Semaphore> = new Map();
  private maxPerUser: number;
  private queueTimeout: number;

  constructor(
    maxConcurrent: number = 50,    // 全局最大并发
    maxPerUser: number = 3,        // 每用户最大并发
    queueTimeout: number = 60_000, // 排队超时（毫秒）
  ) {
    this.globalSem = new Semaphore(maxConcurrent);
    this.maxPerUser = maxPerUser;
    this.queueTimeout = queueTimeout;
  }

  private _getUserSem(userId: string): Semaphore {
    if (!this.userSems.has(userId)) {
      this.userSems.set(userId, new Semaphore(this.maxPerUser));
    }
    return this.userSems.get(userId)!;
  }

  async execute<T>(userId: string, fn: () => Promise<T>): Promise<T> {
    /** 带并发控制的请求执行 */
    const userSem = this._getUserSem(userId);

    const acquireBoth = async () => {
      await this.globalSem.acquire();
      try {
        await userSem.acquire();
      } catch (err) {
        this.globalSem.release();
        throw err;
      }
    };

    // 排队超时控制
    const timeout = new Promise<never>((_, reject) =>
      setTimeout(() => reject(new Error("请求排队超时，请稍后重试")), this.queueTimeout)
    );
    await Promise.race([acquireBoth(), timeout]);

    try {
      return await fn();
    } finally {
      this.globalSem.release();
      userSem.release();
    }
  }
}
```

### 并行工具执行器

```typescript
// parallel_tools.ts — 带并发控制的并行工具执行

interface ToolCallResult {
  toolId: string;
  toolName: string;
  output: string;
  success: boolean;
  durationMs: number;
}

class ParallelToolExecutor {
  /** 并行执行工具调用，带超时和并发控制 */

  private semaphore: Semaphore;
  private timeout: number;

  constructor(maxConcurrent: number = 10, timeout: number = 30_000) {
    this.semaphore = new Semaphore(maxConcurrent);
    this.timeout = timeout;
  }

  async executeAll(toolCalls: Array<{ id: string; name: string; input: Record<string, unknown> }>): Promise<ToolCallResult[]> {
    const tasks = toolCalls.map((call) => this._executeSingle(call));
    return Promise.all(tasks);
  }

  private async _executeSingle(call: { id: string; name: string; input: Record<string, unknown> }): Promise<ToolCallResult> {
    const start = performance.now();
    await this.semaphore.acquire();
    try {
      const result = await Promise.race([
        executeTool(call.name, call.input),
        new Promise<never>((_, reject) =>
          setTimeout(() => reject(new Error("超时")), this.timeout)
        ),
      ]);
      return {
        toolId: call.id, toolName: call.name,
        output: result, success: true,
        durationMs: Math.round(performance.now() - start),
      };
    } catch (err) {
      const isTimeout = err instanceof Error && err.message === "超时";
      return {
        toolId: call.id, toolName: call.name,
        output: isTimeout ? `超时 (${this.timeout / 1000}s)` : `错误: ${err}`,
        success: false,
        durationMs: Math.round(performance.now() - start),
      };
    } finally {
      this.semaphore.release();
    }
  }
}
```

## Prompt Caching：节省 80% 输入成本

Agent 的每轮 LLM 调用都要发送 System Prompt 和工具定义，这些内容每次都一样。Anthropic 的 Prompt Caching 允许你标记这些内容为可缓存，后续请求复用缓存。

### 实现 Prompt Caching

```typescript
// prompt_caching.ts — 带缓存的 LLM 客户端

import Anthropic from "@anthropic-ai/sdk";

class CachedLLMClient {
  /** 支持 Prompt Caching 的 LLM 客户端 */

  private client: Anthropic;
  private model: string;
  private _stats = { hits: 0, misses: 0, tokensSaved: 0 };

  constructor(model: string = "claude-sonnet-4-20250514") {
    this.client = new Anthropic();
    this.model = model;
  }

  async call(
    systemPrompt: string,
    tools: Array<Record<string, unknown>>,
    messages: Array<{ role: string; content: string }>,
  ) {
    // System Prompt 标记为可缓存
    const cachedSystem = [{
      type: "text" as const,
      text: systemPrompt,
      cache_control: { type: "ephemeral" as const },
    }];

    // 最后一个工具定义标记为可缓存
    const cachedTools = tools.map((tool, i) => {
      if (i === tools.length - 1) {
        return { ...tool, cache_control: { type: "ephemeral" as const } };
      }
      return { ...tool };
    });

    const response = await this.client.messages.create({
      model: this.model,
      max_tokens: 4096,
      system: cachedSystem,
      tools: cachedTools as Anthropic.Tool[],
      messages: messages as Anthropic.MessageParam[],
    });

    // 追踪缓存命中
    const cacheRead = (response.usage as Record<string, number>)
      .cache_read_input_tokens ?? 0;
    if (cacheRead > 0) {
      this._stats.hits += 1;
      this._stats.tokensSaved += cacheRead;
    } else {
      this._stats.misses += 1;
    }

    return response;
  }
}
```

### 缓存节省了多少钱

```typescript
// cache_savings.ts — 计算缓存节省的成本

function calculateCacheSavings(
  cacheableTokens: number,      // System Prompt + 工具定义的 token 数
  requestsPerHour: number,      // 每小时请求数
  model: string = "claude-sonnet-4-20250514",
): Record<string, string> {
  // Sonnet 定价
  const inputPrice = 3.0 / 1_000_000;        // $3 / 1M tokens
  const cacheWritePrice = 3.75 / 1_000_000;  // $3.75 / 1M tokens
  const cacheReadPrice = 0.3 / 1_000_000;    // $0.3 / 1M tokens (仅 input 的 10%)

  // 无缓存成本
  const noCache = cacheableTokens * requestsPerHour * inputPrice;

  // 有缓存成本：第一次写入 + 后续读取
  const withCache =
    cacheableTokens * cacheWritePrice +  // 写入一次
    cacheableTokens * (requestsPerHour - 1) * cacheReadPrice;  // 后续读取

  return {
    hourly_without_cache: `$${noCache.toFixed(4)}`,
    hourly_with_cache: `$${withCache.toFixed(4)}`,
    savings_pct: `${((1 - withCache / noCache) * 100).toFixed(1)}%`,
  };
}

// 示例：10000 token 的 System Prompt + 工具定义，每小时 100 次请求
console.log(calculateCacheSavings(10000, 100));
// { hourly_without_cache: '$3.0000',
//   hourly_with_cache: '$0.3345',
//   savings_pct: '88.9%' }
```

## 工具结果缓存

有些工具的结果短时间内不会变化（如数据库表结构、天气查询），可以缓存避免重复调用：

```typescript
// tool_cache.ts — 工具结果缓存

import { createHash } from "crypto";

class ToolResultCache {
  /** 根据工具类型设置不同的缓存时间 */

  static TOOL_TTL: Record<string, number> = {
    get_weather: 600,           // 天气: 10 分钟
    search_documents: 300,       // 文档搜索: 5 分钟
    list_tables: 3600,           // 数据库结构: 1 小时
    describe_table: 3600,        // 表结构: 1 小时
    query_database: 60,          // 数据查询: 1 分钟
  };

  private _cache: Map<string, { result: string; timestamp: number }> = new Map();
  private _hits = 0;
  private _misses = 0;

  private _makeKey(toolName: string, toolInput: Record<string, unknown>): string {
    const content = JSON.stringify(
      { tool: toolName, input: toolInput },
      Object.keys({ tool: toolName, input: toolInput }).sort()
    );
    return createHash("md5").update(content).digest("hex");
  }

  get(toolName: string, toolInput: Record<string, unknown>): string | null {
    const key = this._makeKey(toolName, toolInput);
    const entry = this._cache.get(key);
    if (entry) {
      const ttl = ToolResultCache.TOOL_TTL[toolName] ?? 60;
      if (Date.now() / 1000 - entry.timestamp < ttl) {
        this._hits += 1;
        return entry.result;
      }
      this._cache.delete(key);
    }
    this._misses += 1;
    return null;
  }

  set(toolName: string, toolInput: Record<string, unknown>, result: string): void {
    const key = this._makeKey(toolName, toolInput);
    this._cache.set(key, { result, timestamp: Date.now() / 1000 });
  }

  get hitRate(): number {
    const total = this._hits + this._misses;
    return total > 0 ? this._hits / total : 0.0;
  }
}

// 集成到工具执行流程
const toolCache = new ToolResultCache();

async function cachedToolExecute(
  toolName: string,
  toolInput: Record<string, unknown>,
): Promise<string> {
  /** 带缓存的工具执行 */
  const cached = toolCache.get(toolName, toolInput);
  if (cached !== null) {
    return cached;
  }

  const result = await executeTool(toolName, toolInput);
  toolCache.set(toolName, toolInput, result);
  return result;
}
```

## 连接池

复用 HTTP 和数据库连接，避免每次请求都重新建立连接：

```typescript
// connection_pool.ts — 连接池管理

import { Pool } from "pg";

class ConnectionPool {
  private _db: Pool | null = null;

  async initialize(): Promise<void> {
    this._db = new Pool({
      connectionString: "postgresql://user:pass@localhost/agent_db",
      min: 5,
      max: 20,
      idleTimeoutMillis: 30_000,
      connectionTimeoutMillis: 5_000,
    });
  }

  get db(): Pool {
    if (!this._db) {
      throw new Error("数据库连接池未初始化");
    }
    return this._db;
  }

  async cleanup(): Promise<void> {
    if (this._db) {
      await this._db.end();
    }
  }
}
```

## 完整的异步 Agent 引擎

把以上所有模块整合在一起：

```typescript
// async_engine.ts — 生产级异步 Agent 引擎

import Anthropic from "@anthropic-ai/sdk";

class AsyncAgentEngine {
  /** 整合并发控制、缓存、连接池的 Agent 引擎 */

  private client: Anthropic;
  private tools: ParallelToolExecutor;
  private concurrency: ConcurrencyManager;
  private toolCache: ToolResultCache;
  private pool: ConnectionPool;

  constructor() {
    this.client = new Anthropic({ maxRetries: 3 });
    this.tools = new ParallelToolExecutor(10);
    this.concurrency = new ConcurrencyManager(50);
    this.toolCache = new ToolResultCache();
    this.pool = new ConnectionPool();
  }

  async initialize(): Promise<void> {
    await this.pool.initialize();
  }

  async run(
    userId: string,
    message: string,
    model: string = "claude-sonnet-4-20250514",
  ): Promise<string> {
    return this.concurrency.execute(userId, () =>
      this._agentLoop(message, model)
    );
  }

  private async _agentLoop(message: string, model: string): Promise<string> {
    const messages: Anthropic.MessageParam[] = [
      { role: "user", content: message },
    ];

    for (let i = 0; i < 25; i++) {
      const response = await this.client.messages.create({
        model,
        max_tokens: 4096,
        messages,
      });

      if (response.stop_reason !== "tool_use") {
        return response.content
          .filter((b): b is Anthropic.TextBlock => b.type === "text")
          .map((b) => b.text)
          .join("");
      }

      const toolCalls = response.content
        .filter((b): b is Anthropic.ToolUseBlock => b.type === "tool_use")
        .map((b) => ({ id: b.id, name: b.name, input: b.input as Record<string, unknown> }));

      const results = await this.tools.executeAll(toolCalls);

      messages.push({ role: "assistant", content: response.content });
      messages.push({
        role: "user",
        content: results.map((r) => ({
          type: "tool_result" as const,
          tool_use_id: r.toolId,
          content: r.output,
          is_error: !r.success,
        })),
      });
    }

    return "达到最大迭代次数。";
  }
}
```

## 小结

中级性能优化的四大策略：

1. **并发控制**：全局 + 用户级双层信号量，防止资源耗尽
2. **Prompt Caching**：System Prompt 和工具定义标记为可缓存，节省约 90% 输入成本
3. **工具结果缓存**：按工具类型设置不同 TTL，避免重复调用
4. **连接池复用**：HTTP 和数据库连接池，减少连接建立开销

## 练习

1. 实现 `ConcurrencyManager`，测试同时发送 10 个请求时的行为
2. 对比有无 Prompt Caching 的 API 返回 usage，验证缓存命中
3. 给 `ToolResultCache` 添加 Redis 后端，支持多实例间共享缓存

## 参考资源

- [Anthropic Prompt Caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) -- 缓存文档
- [Node.js 文档](https://nodejs.org/docs/latest/api/) -- 异步编程
- [pg 文档](https://node-postgres.com/) -- PostgreSQL 客户端
- [ioredis 文档](https://github.com/redis/ioredis) -- Redis 客户端
- [Redis 官方文档](https://redis.io/docs/) -- 缓存数据库
