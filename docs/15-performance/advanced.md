# 性能优化 · 高级篇

::: tip 学习目标
- 理解语义缓存的原理，用向量相似度匹配复用 LLM 响应
- 设计多级缓存架构（内存 -> Redis -> 语义缓存）
- 构建智能模型路由器，综合成本、质量、速度做最优选择
- 实现 Fallback 降级链和预测性执行
:::

::: info 学完你能做到
- 让语义相近的问题直接命中缓存，避免重复调用 LLM
- 设计三级缓存架构，逐级降速但容量递增
- 构建一个多因子评分的模型路由器，实时适应模型状态变化
- 在 LLM 推理期间预取工具结果，减少等待时间
:::

## 语义缓存：相似问题不重复调用

进阶篇讲了 Prompt Caching（API 层面）和工具结果缓存（精确匹配）。但有一个更大的优化空间：**用户问的问题语义相近但措辞不同时，能不能复用之前的回答？**

"北京今天天气怎么样" 和 "今天北京天气如何" 含义一样，但精确匹配会当作两个不同的查询。语义缓存通过向量相似度匹配来解决这个问题。

### 语义缓存实现

```typescript
// semantic_cache.ts — 基于嵌入向量的语义缓存

import { createHash } from "crypto";

interface CacheEntry {
  query: string;
  response: string;
  embedding: number[];
  createdAt: number;
  hitCount: number;
  ttl: number; // 默认 1 小时过期
}

class SemanticCache {
  /** 语义缓存——用向量相似度匹配 */

  private threshold: number;
  private maxEntries: number;
  private entries: CacheEntry[] = [];
  private _stats = { hits: 0, misses: 0 };

  constructor(
    similarityThreshold: number = 0.95,
    maxEntries: number = 1000,
  ) {
    this.threshold = similarityThreshold;
    this.maxEntries = maxEntries;
  }

  async getEmbedding(text: string): Promise<number[]> {
    /**
     * 获取文本的嵌入向量
     *
     * 生产环境使用 Voyage AI 或 OpenAI Embeddings API：
     *   const response = await voyageClient.embed([text], { model: "voyage-3" });
     *   return response.embeddings[0];
     *
     * 这里用简化方式演示逻辑：
     */
    const hashBytes = createHash("sha256").update(text).digest();
    // 取前 32 字节，每 4 字节转为一个 float32
    const arr: number[] = [];
    for (let i = 0; i < 32 && i + 3 < hashBytes.length; i += 4) {
      arr.push(hashBytes.readFloatLE(i));
    }
    return arr;
  }

  private _cosineSimilarity(a: number[], b: number[]): number {
    /** 计算余弦相似度 */
    let dot = 0, normA = 0, normB = 0;
    for (let i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }
    normA = Math.sqrt(normA);
    normB = Math.sqrt(normB);
    if (normA === 0 || normB === 0) return 0.0;
    return dot / (normA * normB);
  }

  async get(query: string): Promise<string | null> {
    /** 查找语义相似的缓存条目 */
    const queryEmbedding = await this.getEmbedding(query);
    const now = Date.now() / 1000;

    let bestMatch: CacheEntry | null = null;
    let bestSimilarity = 0.0;

    for (const entry of this.entries) {
      // 跳过过期条目
      if (now - entry.createdAt > entry.ttl) {
        continue;
      }
      const similarity = this._cosineSimilarity(queryEmbedding, entry.embedding);
      if (similarity > this.threshold && similarity > bestSimilarity) {
        bestMatch = entry;
        bestSimilarity = similarity;
      }
    }

    if (bestMatch) {
      bestMatch.hitCount += 1;
      this._stats.hits += 1;
      return bestMatch.response;
    }

    this._stats.misses += 1;
    return null;
  }

  async set(query: string, response: string, ttl: number = 3600): Promise<void> {
    /** 添加缓存条目 */
    const embedding = await this.getEmbedding(query);

    // 容量控制：淘汰最旧的 25% 条目
    if (this.entries.length >= this.maxEntries) {
      this.entries.sort((a, b) => a.createdAt - b.createdAt);
      this.entries = this.entries.slice(Math.floor(this.entries.length / 4));
    }

    this.entries.push({
      query,
      response,
      embedding,
      createdAt: Date.now() / 1000,
      hitCount: 0,
      ttl,
    });
  }

  get hitRate(): number {
    const total = this._stats.hits + this._stats.misses;
    return total > 0 ? this._stats.hits / total : 0.0;
  }
}
```

### 多级缓存架构

单一缓存层往往不够。更好的设计是三级缓存，逐级降速但容量递增：

```typescript
// multi_level_cache.ts — L1(内存) -> L2(Redis) -> L3(语义缓存)

import Redis from "ioredis";

class MultiLevelCache {
  /** 三级缓存架构 */

  private l1: Map<string, { result: string; timestamp: number }> = new Map(); // 内存（最快）
  private l2: Redis;                        // Redis（分布式）
  private l3: SemanticCache;                // 语义（最智能）

  constructor(redisUrl: string = "redis://localhost:6379") {
    this.l2 = new Redis(redisUrl);
    this.l3 = new SemanticCache(0.95);
  }

  async get(query: string, exactKey?: string): Promise<string | null> {
    /** 逐级查找，命中后回填上层 */

    // L1: 精确匹配（内存，微秒级）
    if (exactKey && this.l1.has(exactKey)) {
      const entry = this.l1.get(exactKey)!;
      if (Date.now() / 1000 - entry.timestamp < 300) {  // 5 分钟 TTL
        return entry.result;
      }
    }

    // L2: 精确匹配（Redis，毫秒级）
    if (exactKey) {
      const cached = await this.l2.get(`agent:cache:${exactKey}`);
      if (cached) {
        // 回填 L1
        this.l1.set(exactKey, { result: cached, timestamp: Date.now() / 1000 });
        return cached;
      }
    }

    // L3: 语义匹配（向量计算，十毫秒级）
    const semanticResult = await this.l3.get(query);
    if (semanticResult) {
      return semanticResult;
    }

    return null;
  }

  async set(query: string, response: string, exactKey?: string): Promise<void> {
    /** 写入所有层级 */
    if (exactKey) {
      this.l1.set(exactKey, { result: response, timestamp: Date.now() / 1000 });
      await this.l2.setex(`agent:cache:${exactKey}`, 3600, response);
    }
    await this.l3.set(query, response);
  }
}
```

### 缓存失效策略

缓存最难的部分不是写入，而是失效——数据变了，旧缓存就不对了：

```typescript
// cache_invalidation.ts — 缓存失效策略

class CacheInvalidator {
  /** 当底层数据变更时，主动清除相关缓存 */

  private cache: MultiLevelCache;

  constructor(cache: MultiLevelCache) {
    this.cache = cache;
  }

  async invalidateByTool(toolName: string): Promise<void> {
    /** 工具执行写操作时，清除相关的读缓存 */
    // 写操作 -> 需要清除的读缓存
    const writeToReadMap: Record<string, string[]> = {
      insert_record: ["query", "list_tables"],
      update_record: ["query"],
      delete_record: ["query", "list_tables"],
    };
    const relatedTools = writeToReadMap[toolName] ?? [];
    for (const related of relatedTools) {
      const keysToRemove = [...(this.cache as any).l1.keys()].filter(
        (k: string) => k.startsWith(related)
      );
      for (const k of keysToRemove) {
        (this.cache as any).l1.delete(k);
      }
    }
  }

  async invalidateByPattern(pattern: string): Promise<void> {
    /** 按模式批量清除 Redis 缓存 */
    const stream = (this.cache as any).l2.scanStream({
      match: `agent:cache:${pattern}*`,
    });
    const keys: string[] = [];
    for await (const batch of stream) {
      keys.push(...batch);
    }
    if (keys.length > 0) {
      await (this.cache as any).l2.del(...keys);
    }
  }
}
```

## 智能模型路由器

进阶篇用信号量做并发控制，但模型选择还是固定的。高级路由器综合**成本、质量、速度、可用性**四个维度做实时决策。

### 模型画像 + 多因子评分

```typescript
// smart_router.ts — 智能模型路由器

enum Complexity {
  SIMPLE = "simple",
  MEDIUM = "medium",
  COMPLEX = "complex",
}

interface ModelProfile {
  /** 模型画像：静态属性 + 动态状态 */
  name: string;
  provider: string;
  costPer1kInput: number;
  costPer1kOutput: number;
  avgLatencyMs: number;          // 实时更新
  capabilityScore: number;       // 0-1 能力评分
  isAvailable: boolean;
  errorRate: number;             // 近期错误率（实时更新）
}

interface RoutingDecision {
  model: string;
  provider: string;
  reason: string;
  estimatedCost: number;
  estimatedLatencyMs: number;
}

class SmartModelRouter {
  /** 智能模型路由器——综合多因子做最优选择 */

  models: Record<string, ModelProfile>;

  constructor() {
    this.models = {
      "claude-opus-4-20250514": {
        name: "claude-opus-4-20250514", provider: "anthropic",
        costPer1kInput: 0.015, costPer1kOutput: 0.075,
        avgLatencyMs: 3000, capabilityScore: 1.0,
        isAvailable: true, errorRate: 0.0,
      },
      "claude-sonnet-4-20250514": {
        name: "claude-sonnet-4-20250514", provider: "anthropic",
        costPer1kInput: 0.003, costPer1kOutput: 0.015,
        avgLatencyMs: 1500, capabilityScore: 0.85,
        isAvailable: true, errorRate: 0.0,
      },
      "claude-haiku-3-20250414": {
        name: "claude-haiku-3-20250414", provider: "anthropic",
        costPer1kInput: 0.00025, costPer1kOutput: 0.00125,
        avgLatencyMs: 500, capabilityScore: 0.6,
        isAvailable: true, errorRate: 0.0,
      },
      "gpt-4o": {
        name: "gpt-4o", provider: "openai",
        costPer1kInput: 0.0025, costPer1kOutput: 0.01,
        avgLatencyMs: 1200, capabilityScore: 0.85,
        isAvailable: true, errorRate: 0.0,
      },
      "gpt-4o-mini": {
        name: "gpt-4o-mini", provider: "openai",
        costPer1kInput: 0.00015, costPer1kOutput: 0.0006,
        avgLatencyMs: 400, capabilityScore: 0.55,
        isAvailable: true, errorRate: 0.0,
      },
    };
  }

  route(
    message: string,
    priority: "cost" | "quality" | "speed" | "balanced" = "balanced",
    budgetRemaining?: number,
  ): RoutingDecision {
    /** 智能路由决策 */
    // 1. 判断任务复杂度
    const complexity = this._classify(message);
    const estimatedTokens = Math.floor(message.length / 2);

    // 2. 过滤可用模型（排除故障和高错误率的）
    const available = Object.entries(this.models).filter(
      ([, m]) => m.isAvailable && m.errorRate < 0.3
    );

    // 3. 多因子评分
    const scored: Array<[string, ModelProfile, number]> = [];
    for (const [name, model] of available) {
      const score = this._score(
        model, complexity, priority, estimatedTokens, budgetRemaining
      );
      if (score >= 0) {
        scored.push([name, model, score]);
      }
    }

    scored.sort((a, b) => b[2] - a[2]);
    const [bestName, bestModel] = scored[0];

    const estCost =
      (estimatedTokens * bestModel.costPer1kInput) / 1000 +
      (500 * bestModel.costPer1kOutput) / 1000;

    return {
      model: bestName,
      provider: bestModel.provider,
      reason: `complexity=${complexity}, priority=${priority}`,
      estimatedCost: Math.round(estCost * 1_000_000) / 1_000_000,
      estimatedLatencyMs: bestModel.avgLatencyMs,
    };
  }

  private _classify(message: string): Complexity {
    /** 规则引擎判断复杂度（零额外成本） */
    const msgLen = message.length;
    if (msgLen < 50 || ["翻译", "格式化", "分类"].some((k) => message.includes(k))) {
      return Complexity.SIMPLE;
    }
    if (msgLen > 500 || ["分析", "设计", "推理"].some((k) => message.includes(k))) {
      return Complexity.COMPLEX;
    }
    return Complexity.MEDIUM;
  }

  private _score(
    model: ModelProfile,
    complexity: Complexity,
    priority: string,
    estTokens: number,
    budget?: number,
  ): number {
    /** 多因子评分 */
    // 能力门槛
    const minCap: Record<Complexity, number> = {
      [Complexity.SIMPLE]: 0.4,
      [Complexity.MEDIUM]: 0.7,
      [Complexity.COMPLEX]: 0.9,
    };
    if (model.capabilityScore < minCap[complexity]) {
      return -1; // 能力不足，排除
    }

    // 预算检查
    const estCost = (estTokens * model.costPer1kInput) / 1000;
    if (budget !== undefined && estCost > budget) {
      return -1; // 超预算
    }

    // 权重矩阵
    const weightsMap: Record<string, { cost: number; quality: number; speed: number }> = {
      cost:     { cost: 0.6, quality: 0.2, speed: 0.2 },
      quality:  { cost: 0.1, quality: 0.7, speed: 0.2 },
      speed:    { cost: 0.1, quality: 0.2, speed: 0.7 },
      balanced: { cost: 0.33, quality: 0.34, speed: 0.33 },
    };
    const weights = weightsMap[priority];

    // 归一化评分（0-1）
    const costScore = 1.0 - Math.min(model.costPer1kInput / 0.015, 1.0);
    const qualityScore = model.capabilityScore;
    const speedScore = 1.0 - Math.min(model.avgLatencyMs / 5000, 1.0);

    return (
      weights.cost * costScore +
      weights.quality * qualityScore +
      weights.speed * speedScore
    );
  }

  updateModelStats(model: string, latencyMs: number, success: boolean): void {
    /** 每次调用完成后更新模型实时状态 */
    if (model in this.models) {
      const m = this.models[model];
      // 滑动平均更新延迟
      m.avgLatencyMs = m.avgLatencyMs * 0.9 + latencyMs * 0.1;
      // 更新错误率
      if (!success) {
        m.errorRate = Math.min(m.errorRate + 0.1, 1.0);
      } else {
        m.errorRate = Math.max(m.errorRate - 0.01, 0.0);
      }
    }
  }
}

// 使用示例
const router = new SmartModelRouter();

let decision = router.route("帮我翻译这段话", "cost");
console.log(`选择: ${decision.model}, 原因: ${decision.reason}`);

decision = router.route(
  "请设计一个分布式消息队列系统，要求支持百万级并发...",
  "quality",
);
console.log(`选择: ${decision.model}, 原因: ${decision.reason}`);
```

## Fallback 降级链

当主模型不可用时（超时、限流、宕机），自动降级到备选模型，而不是直接报错：

```typescript
// fallback_chain.ts — 跨提供商 Fallback 降级链

import Anthropic from "@anthropic-ai/sdk";
import OpenAI from "openai";

interface LLMResult {
  text: string;
  model: string;
  usage: { input_tokens: number; output_tokens: number };
  degraded?: boolean;
  original_model?: string;
}

class ModelWithFallback {
  /** 带降级的模型调用——主模型失败自动切换 */

  static FALLBACK_CHAIN = [
    { provider: "anthropic", model: "claude-sonnet-4-20250514" },
    { provider: "openai",    model: "gpt-4o" },
    { provider: "anthropic", model: "claude-haiku-3-20250414" },
    { provider: "openai",    model: "gpt-4o-mini" },
  ];

  private anthropicClient: Anthropic;
  private openaiClient: OpenAI;

  constructor() {
    this.anthropicClient = new Anthropic();
    this.openaiClient = new OpenAI();
  }

  async call(
    messages: Array<{ role: "user" | "assistant"; content: string }>,
    maxFallbacks: number = 3,
  ): Promise<LLMResult> {
    /** 逐级尝试，直到成功 */
    const errors: string[] = [];
    const chain = ModelWithFallback.FALLBACK_CHAIN.slice(0, maxFallbacks + 1);

    for (let i = 0; i < chain.length; i++) {
      const option = chain[i];
      try {
        const result = await Promise.race([
          this._callProvider(option.provider, option.model, messages),
          new Promise<never>((_, reject) =>
            setTimeout(() => reject(new Error("超时")), 30_000)
          ),
        ]);
        if (i > 0) {
          result.degraded = true;
          result.original_model = ModelWithFallback.FALLBACK_CHAIN[0].model;
        }
        return result;
      } catch (err) {
        errors.push(`${option.model}: ${err}`);
        continue;
      }
    }

    throw new Error(`所有模型均不可用: ${errors.join("; ")}`);
  }

  private async _callProvider(
    provider: string,
    model: string,
    messages: Array<{ role: "user" | "assistant"; content: string }>,
  ): Promise<LLMResult> {
    if (provider === "anthropic") {
      const response = await this.anthropicClient.messages.create({
        model,
        max_tokens: 4096,
        messages,
      });
      const block = response.content[0];
      return {
        text: block.type === "text" ? block.text : "",
        model,
        usage: {
          input_tokens: response.usage.input_tokens,
          output_tokens: response.usage.output_tokens,
        },
      };
    } else if (provider === "openai") {
      const response = await this.openaiClient.chat.completions.create({
        model,
        messages,
        max_tokens: 4096,
      });
      return {
        text: response.choices[0].message.content ?? "",
        model,
        usage: {
          input_tokens: response.usage?.prompt_tokens ?? 0,
          output_tokens: response.usage?.completion_tokens ?? 0,
        },
      };
    }
    throw new Error(`未知的提供商: ${provider}`);
  }
}
```

## 预测性执行

在 LLM 推理期间，如果能预判它可能需要的工具结果并提前获取，就可以节省一轮等待时间：

```typescript
// predictive_execution.ts — 预测性工具预取

async function executeTool(toolName: string, toolInput: Record<string, unknown>): Promise<string> {
  /** 执行工具调用（需根据实际工具注册表实现） */
  throw new Error(`请实现工具 ${toolName} 的执行逻辑`);
}

class PredictiveExecutor {
  /** 在 LLM 推理的同时，预取可能需要的工具结果 */

  private prefetchCache: Map<string, Promise<string>> = new Map();

  async prefetch(likelyTools: Array<{ name: string; input: Record<string, unknown> }>): Promise<void> {
    /** 提前发起可能需要的工具调用 */
    for (const tool of likelyTools) {
      const cacheKey = `${tool.name}:${JSON.stringify(tool.input)}`;
      if (!this.prefetchCache.has(cacheKey)) {
        this.prefetchCache.set(
          cacheKey,
          executeTool(tool.name, tool.input)
        );
      }
    }
  }

  async getOrExecute(toolName: string, toolInput: Record<string, unknown>): Promise<string> {
    /** 优先从预取缓存获取，否则正常执行 */
    const cacheKey = `${toolName}:${JSON.stringify(toolInput)}`;
    if (this.prefetchCache.has(cacheKey)) {
      const promise = this.prefetchCache.get(cacheKey)!;
      this.prefetchCache.delete(cacheKey);
      return promise;
    }
    return executeTool(toolName, toolInput);
  }

  clear(): void {
    /** 清理未使用的预取任务（Promise 无法取消，仅清空引用） */
    this.prefetchCache.clear();
  }
}

// 使用场景：
// 用户问"今天北京天气怎么样"
// -> 在 LLM 推理时，预测它大概率会调用 get_weather(city="北京")
// -> 提前发起天气查询
// -> LLM 返回 tool_use 时，结果已经准备好了，省去一轮等待
```

::: warning 预测性执行的代价
预取是有成本的——如果预测错了，工具调用白做了。适合以下场景：
- 工具调用模式高度可预测（如 "天气" -> get_weather）
- 工具调用耗时较长但成本低（如数据库查询）
- 错误的预取不会产生副作用（只读操作）

**不适合**：写操作、高成本 API 调用、不确定的场景。
:::

## 小结

高级性能优化的四大策略：

1. **语义缓存**：用向量相似度匹配复用 LLM 响应，语义相近的问题不重复调用
2. **多级缓存**：L1 内存（微秒）-> L2 Redis（毫秒）-> L3 语义（十毫秒），逐级降速但容量递增
3. **智能路由**：综合成本、质量、速度、可用性四个维度实时评分，每次选最优模型
4. **预测性执行**：在 LLM 推理期间预取工具结果，减少一轮等待时间

## 练习

1. 实现 `SemanticCache`，用真实的嵌入 API（如 Voyage 或 OpenAI Embeddings），测试相似问题的命中率
2. 构建 `SmartModelRouter`，准备不同复杂度的 10 个问题，分别以 "cost"、"quality"、"speed" 优先级路由，验证模型选择是否合理
3. 实现 `ModelWithFallback`，模拟主模型超时的场景，验证降级链是否正常工作

## 参考资源

- [GPTCache](https://github.com/zilliztech/GPTCache) -- 开源语义缓存库
- [arXiv:2311.04934 - Semantic Caching for LLM Applications](https://arxiv.org/abs/2311.04934) -- 语义缓存研究
- [arXiv:2404.14219 - RouterBench](https://arxiv.org/abs/2404.14219) -- 模型路由基准测试
- [arXiv:2402.07625 - RouteLLM](https://arxiv.org/abs/2402.07625) -- 基于偏好的路由
- [OpenRouter](https://openrouter.ai/) -- 多模型统一 API 网关
- [LiteLLM](https://github.com/BerriAI/litellm) -- 开源多模型代理工具
- [Redis 官方文档](https://redis.io/docs/) -- 缓存数据库
