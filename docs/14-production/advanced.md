# 生产工程化 · 高级篇

::: tip 学习目标
- 理解 LLM 调用的成本构成，掌握成本计算方法
- 实现 Prompt Caching 节省 80%+ 的输入成本
- 设计成本感知的模型路由器和预算控制系统
- 构建完整的 CI/CD 流水线和灰度发布机制
:::

::: info 学完你能做到
- 精确计算每次 Agent 任务的成本并设置预算上限
- 用 Prompt Caching 大幅降低高频 Agent 的运行成本
- 让简单任务自动走便宜模型，复杂任务走强模型
- 搭建从测试到部署的自动化流水线，支持灰度发布
:::

## 成本控制：Agent 的隐形杀手

普通 API 调用是一次请求一次回复。而 Agent 的一次任务可能涉及多轮 LLM 调用，成本呈倍数放大：

```
用户请求 -> LLM 决策（5K in + 500 out）
         -> 工具调用 -> 工具结果
         -> LLM 分析（8K in + 800 out）  <- 上下文累积！
         -> 工具调用 -> 工具结果
         -> LLM 总结（12K in + 1K out）  <- 持续增长

总计: 25K input + 2.3K output
实际费用可能是单次调用的 5-10 倍
```

### 成本计算器

```typescript
/** cost_calculator.ts — LLM 调用成本计算 */

// 2025 年主流模型定价参考（每百万 Token，美元）
const PRICING: Record<string, Record<string, number>> = {
    "claude-opus-4-20250514": {
        input: 15.0,
        output: 75.0,
        cache_write: 18.75,
        cache_read: 1.5,
    },
    "claude-sonnet-4-20250514": {
        input: 3.0,
        output: 15.0,
        cache_write: 3.75,
        cache_read: 0.3,
    },
    "claude-haiku-3-20250414": {
        input: 0.25,
        output: 1.25,
        cache_write: 0.3,
        cache_read: 0.03,
    },
    "gpt-4o": {
        input: 2.5,
        output: 10.0,
    },
    "gpt-4o-mini": {
        input: 0.15,
        output: 0.6,
    },
};

function calculateCost(
    model: string,
    inputTokens: number,
    outputTokens: number,
    cacheReadTokens: number = 0,
    cacheWriteTokens: number = 0,
): number {
    /** 计算单次调用成本（美元） */
    const price = PRICING[model];
    const cost =
        (inputTokens - cacheReadTokens) * price.input / 1_000_000
        + outputTokens * price.output / 1_000_000
        + cacheReadTokens * (price.cache_read ?? price.input) / 1_000_000
        + cacheWriteTokens * (price.cache_write ?? price.input) / 1_000_000;
    return Math.round(cost * 1_000_000) / 1_000_000;
}

// 示例
const cost = calculateCost("claude-sonnet-4-20250514", 5000, 1000);
console.log(`单次调用成本: $${cost.toFixed(4)}`);  // $0.0300
```

::: warning 成本警示
一个设计不当的 Agent 可能在单次任务中消耗数美元。面向公众用户时，不做成本控制可能导致账单失控。
:::

### 用户级预算控制

```typescript
/** cost_tracker.ts — 用户级成本追踪与预算控制 */

class CostTracker {
    /** 用户级别的成本追踪器 */

    private dailyCosts: Map<string, number> = new Map();
    private monthlyCosts: Map<string, number> = new Map();

    constructor(
        private dailyLimitCents: number = 100,      // $1.00/天
        private monthlyLimitCents: number = 2000,    // $20.00/月
    ) {}

    record(userId: string, costDollars: number): void {
        /** 记录一次成本 */
        const costCents = costDollars * 100;
        const today = new Date().toISOString().slice(0, 10);     // "YYYY-MM-DD"
        const month = new Date().toISOString().slice(0, 7);      // "YYYY-MM"

        const dailyKey = `${userId}:${today}`;
        const monthlyKey = `${userId}:${month}`;
        this.dailyCosts.set(dailyKey, (this.dailyCosts.get(dailyKey) || 0) + costCents);
        this.monthlyCosts.set(monthlyKey, (this.monthlyCosts.get(monthlyKey) || 0) + costCents);
    }

    checkBudget(userId: string): Record<string, unknown> {
        /** 检查用户是否超出预算 */
        const today = new Date().toISOString().slice(0, 10);
        const month = new Date().toISOString().slice(0, 7);

        const daily = this.dailyCosts.get(`${userId}:${today}`) || 0;
        const monthly = this.monthlyCosts.get(`${userId}:${month}`) || 0;

        return {
            daily_used_cents: Math.round(daily * 100) / 100,
            daily_limit_cents: this.dailyLimitCents,
            daily_remaining_pct: Math.max(0, (1 - daily / this.dailyLimitCents) * 100),
            monthly_used_cents: Math.round(monthly * 100) / 100,
            can_proceed: daily < this.dailyLimitCents && monthly < this.monthlyLimitCents,
            reason: this.getBlockReason(daily, monthly),
        };
    }

    private getBlockReason(daily: number, monthly: number): string | null {
        if (daily >= this.dailyLimitCents) {
            return "已达今日用量上限，请明天再试";
        }
        if (monthly >= this.monthlyLimitCents) {
            return "已达本月用量上限";
        }
        return null;
    }
}
```

## Prompt Caching：节省 80%+ 输入成本

Agent 的每轮 LLM 调用都要发送 System Prompt 和工具定义，这些内容每次都一样。Anthropic 的 Prompt Caching 让你标记这些内容为可缓存，后续请求只需支付缓存读取的费用（仅原价的 10%）。

```typescript
/** prompt_caching.ts — Prompt Caching 实现 */

import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

// 假设有 20+ 个工具定义和一段长 System Prompt
const SYSTEM_PROMPT = "你是一个专业的数据分析助手...";  // 很长的指令
const TOOLS: Anthropic.Tool[] = [
    { name: "search", description: "搜索文档库...", input_schema: { type: "object", properties: {} } },
    // ... 20+ 个工具定义
];

const response = await client.messages.create({
    model: "claude-sonnet-4-20250514",
    max_tokens: 4096,
    system: [{
        type: "text",
        text: SYSTEM_PROMPT,
        cache_control: { type: "ephemeral" },  // 标记可缓存
    }],
    tools: TOOLS.map((tool, i) =>
        i === TOOLS.length - 1
            ? { ...tool, cache_control: { type: "ephemeral" as const } }
            : tool
    ),
    messages: [{ role: "user", content: "分析上周的用户增长数据" }],
});

// 查看缓存命中
const usage = response.usage;
console.log(`输入 Token: ${usage.input_tokens}`);
console.log(`缓存写入: ${(usage as any).cache_creation_input_tokens}`);
console.log(`缓存读取: ${(usage as any).cache_read_input_tokens}`);
```

### 缓存节省了多少钱

```typescript
/** cache_savings.ts — 计算缓存带来的成本节省 */

function calculateCacheSavings(
    cacheableTokens: number,         // System Prompt + 工具定义的 token 数
    requestsPerHour: number,         // 每小时请求数
    model: string = "claude-sonnet-4-20250514",
): Record<string, string> {
    const price = PRICING[model];

    // 无缓存：每次都按 input 定价
    const noCache = cacheableTokens * requestsPerHour * price.input / 1_000_000;

    // 有缓存：第一次写入 + 后续读取
    const withCache =
        cacheableTokens * price.cache_write / 1_000_000
        + cacheableTokens * (requestsPerHour - 1) * price.cache_read / 1_000_000;

    return {
        hourly_without_cache: `$${noCache.toFixed(4)}`,
        hourly_with_cache: `$${withCache.toFixed(4)}`,
        savings_pct: `${((1 - withCache / noCache) * 100).toFixed(1)}%`,
    };
}

// 示例：10000 token 的固定内容，每小时 100 次请求
console.log(calculateCacheSavings(10000, 100));
// { hourly_without_cache: '$3.0000',
//   hourly_with_cache: '$0.3345',
//   savings_pct: '88.9%' }
```

## 模型路由：用对的模型做对的事

不是所有任务都需要最强的模型。翻译一句话用 Opus 就是浪费钱。

### 基于规则的路由（零额外成本）

```typescript
/** model_router.ts — 成本感知的模型路由 */

class RuleBasedRouter {
    /** 基于规则的模型路由器——不需要额外的 LLM 调用 */

    private static readonly SIMPLE_PATTERNS = [
        "翻译", "总结", "格式化", "提取", "分类",
        "是否", "对不对", "帮我改一下",
    ];
    private static readonly COMPLEX_PATTERNS = [
        "写一个完整的", "设计一个系统", "分析", "调试",
        "多步骤", "比较.*优缺点", "为什么",
    ];

    route(message: string, toolCount: number = 0): string {
        const msgLower = message.toLowerCase();
        const msgLen = message.length;

        // 需要工具的任务用 Sonnet
        if (toolCount > 0) {
            return "claude-sonnet-4-20250514";
        }

        // 短消息 + 简单模式 -> 便宜模型
        if (msgLen < 100 && RuleBasedRouter.SIMPLE_PATTERNS.some((p) => msgLower.includes(p))) {
            return "claude-haiku-3-20250414";
        }

        // 长消息或复杂模式 -> 中等模型
        if (msgLen > 500 || RuleBasedRouter.COMPLEX_PATTERNS.some((p) => msgLower.includes(p))) {
            return "claude-sonnet-4-20250514";
        }

        return "claude-haiku-3-20250414";  // 默认用便宜的
    }
}
```

### 对话历史压缩

Agent 的上下文会随轮次越来越长。压缩历史消息可以直接减少 Token 消耗：

```typescript
/** history_compression.ts — 对话历史压缩 */

import Anthropic from "@anthropic-ai/sdk";

async function compressHistory(
    messages: Anthropic.MessageParam[],
    keepRecent: number = 4,
): Promise<Anthropic.MessageParam[]> {
    /** 压缩对话历史：保留最近 N 条，之前的压缩为摘要 */
    if (messages.length <= keepRecent) {
        return messages;
    }

    const oldMessages = messages.slice(0, -keepRecent);
    const recentMessages = messages.slice(-keepRecent);

    // 用小模型生成摘要（便宜）
    const client = new Anthropic();
    const summaryResponse = await client.messages.create({
        model: "claude-haiku-3-20250414",
        max_tokens: 300,
        messages: [{
            role: "user",
            content:
                "用2-3句话总结以下对话的关键信息：\n\n"
                + oldMessages
                    .map((m) => `${m.role}: ${String(m.content).slice(0, 200)}`)
                    .join("\n"),
        }],
    });
    const summary = summaryResponse.content[0].type === "text"
        ? summaryResponse.content[0].text
        : "";

    return [
        { role: "user", content: `[对话历史摘要] ${summary}` },
        { role: "assistant", content: "好的，我了解之前的讨论内容。" },
        ...recentMessages,
    ];
}
```

### 成本感知的完整 Agent

把预算控制、模型路由、成本追踪整合在一起：

```typescript
/** cost_aware_agent.ts — 成本感知的 Agent */

import Anthropic from "@anthropic-ai/sdk";

class CostAwareAgent {
    /** 整合预算控制 + 模型路由 + 成本追踪的 Agent */

    private tracker = new CostTracker();
    private router = new RuleBasedRouter();
    private client = new Anthropic();

    async chat(userId: string, message: string): Promise<string> {
        // 1. 预算检查
        const budget = this.tracker.checkBudget(userId);
        if (!budget.can_proceed) {
            return budget.reason as string;
        }

        // 2. 模型路由
        const model = this.router.route(message);

        // 3. 执行
        const response = await this.client.messages.create({
            model,
            max_tokens: 2048,
            messages: [{ role: "user", content: message }],
        });

        // 4. 记录成本
        const cost = calculateCost(
            model,
            response.usage.input_tokens,
            response.usage.output_tokens,
        );
        this.tracker.record(userId, cost);

        // 5. 用量提醒
        const newBudget = this.tracker.checkBudget(userId);
        let warning = "";
        if ((newBudget.daily_remaining_pct as number) < 20) {
            warning = `\n\n(提示：今日用量已使用 ${(100 - (newBudget.daily_remaining_pct as number)).toFixed(0)}%)`;
        }

        const text = response.content[0].type === "text" ? response.content[0].text : "";
        return text + warning;
    }
}
```

## 灰度发布

新版本的模型或 System Prompt 不应该一下子推给所有用户。灰度发布让你先对小比例用户验证，确认无问题再全量推出：

```typescript
/** gray_release.ts — 灰度发布控制 */

import { createHash } from "crypto";

class GrayRelease {
    /** 基于用户 ID 的灰度发布 */

    constructor(private rolloutPercentage: number = 10) {}

    isInExperiment(userId: string, experiment: string): boolean {
        /** 判断用户是否在灰度范围内 */
        const hashInput = `${userId}:${experiment}`;
        const hashHex = createHash("md5").update(hashInput).digest("hex");
        const bucket = parseInt(hashHex.slice(0, 8), 16) % 100;
        return bucket < this.rolloutPercentage;
    }

    getModel(userId: string): string {
        /** 灰度切换模型版本 */
        if (this.isInExperiment(userId, "new_model_v2")) {
            return "claude-sonnet-4-20250514";       // 新版本（灰度测试中）
        }
        return "claude-haiku-3-20250414";            // 稳定版本
    }

    getSystemPrompt(userId: string): string {
        /** 灰度切换 System Prompt 版本 */
        if (this.isInExperiment(userId, "new_prompt_v3")) {
            return "你是专业的数据分析助手。回答要简洁。（v3）";  // 新 Prompt
        }
        return "你是一个数据分析助手。（v2）";                    // 旧 Prompt
    }
}

// 使用：先 10% 用户用新版，观察指标后逐步扩大
const gray = new GrayRelease(10);
```

## CI/CD 流水线

自动化的测试、构建、部署流程是生产系统的标配：

```yaml
# .github/workflows/deploy.yml
name: Agent Service CI/CD

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: "20"
      - run: npm ci
      - run: npm run test -- --coverage
      - run: npx eslint src/
      - run: npx tsc --noEmit

  build:
    needs: test
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - uses: docker/build-push-action@v5
        with:
          push: true
          tags: ghcr.io/${{ github.repository }}:${{ github.sha }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

  deploy:
    needs: build
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - name: Rolling update
        run: |
          kubectl set image deployment/agent-api \
            agent-api=ghcr.io/${{ github.repository }}:${{ github.sha }}
          kubectl rollout status deployment/agent-api --timeout=300s
```

## 健康检查与监控

```typescript
/** health_check.ts — 多层健康检查 */

import express from "express";
import Anthropic from "@anthropic-ai/sdk";

const app = express();

class HealthChecker {
    private startTime = new Date();

    async checkDatabase(): Promise<Record<string, string>> {
        try {
            await dbPool.query("SELECT 1");
            return { status: "healthy" };
        } catch (e) {
            return { status: "unhealthy", error: String(e) };
        }
    }

    async checkRedis(): Promise<Record<string, string>> {
        try {
            await redisClient.ping();
            return { status: "healthy" };
        } catch (e) {
            return { status: "unhealthy", error: String(e) };
        }
    }

    async checkLLMApi(): Promise<Record<string, string>> {
        try {
            const client = new Anthropic();
            await Promise.race([
                client.messages.create({
                    model: "claude-haiku-3-20250414",
                    max_tokens: 5,
                    messages: [{ role: "user", content: "hi" }],
                }),
                new Promise<never>((_, reject) =>
                    setTimeout(() => reject(new Error("timeout")), 10000)
                ),
            ]);
            return { status: "healthy" };
        } catch (e) {
            return { status: "degraded", error: String(e) };
        }
    }

    async fullCheck(): Promise<Record<string, unknown>> {
        const [db, cache, llm] = await Promise.all([
            this.checkDatabase(),
            this.checkRedis(),
            this.checkLLMApi(),
        ]);
        const allHealthy = [db, cache, llm].every((c) => c.status === "healthy");
        return {
            status: allHealthy ? "healthy" : "degraded",
            uptime_seconds: (Date.now() - this.startTime.getTime()) / 1000,
            checks: { database: db, redis: cache, llm_api: llm },
        };
    }
}

const checker = new HealthChecker();

app.get("/health", async (req, res) => {
    /** 轻量健康检查（K8s liveness probe） */
    res.json({ status: "ok" });
});

app.get("/health/ready", async (req, res) => {
    /** 就绪检查（K8s readiness probe） */
    const result = await checker.fullCheck();
    res.json(result);
});
```

## 小结

高级生产工程化的四大策略：

1. **成本控制**：精确追踪每次调用的 Token 和费用，设置用户级日/月预算上限
2. **Prompt Caching**：标记固定上下文为可缓存，节省 80-90% 的输入成本
3. **模型路由**：简单任务走 Haiku（便宜 12 倍），复杂任务走 Sonnet，按需升级 Opus
4. **发布运维**：CI/CD 自动化（测试-构建-部署）+ 灰度发布（按比例放量）+ 多层健康检查

## 练习

1. 用 `calculate_cost` 计算你的 Agent 单次任务的平均成本，对比有无 Prompt Caching 的差异
2. 实现 `RuleBasedRouter`，准备 10 个不同复杂度的问题，验证路由是否合理
3. 为你的 Agent 设置 CI/CD 流水线，跑通 test + docker build + 部署的全流程

## 参考资源

- [Anthropic Pricing](https://www.anthropic.com/pricing) -- 模型定价
- [Anthropic Prompt Caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) -- 缓存文档
- [Token Counting Guide](https://docs.anthropic.com/en/docs/build-with-claude/token-counting) -- Token 计算方法
- [12 Factor App](https://12factor.net/) -- 现代应用 12 要素
- [GitHub Actions 文档](https://docs.github.com/en/actions) -- CI/CD 平台
- [Kubernetes 官方文档](https://kubernetes.io/docs/) -- 容器编排
