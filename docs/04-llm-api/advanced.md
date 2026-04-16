# LLM API 调用 · 高级

::: info 学习目标
- 了解 LLM API 调用中的常见错误类型和应对策略
- 掌握指数退避重试的原理和实现
- 学会处理速率限制（Rate Limit）和批量请求控制
- 掌握 Token 超限的多种截断策略
- 实现一个生产级的健壮 LLM 客户端
- 学完能在生产环境中安全可靠地使用 LLM API

预计学习时间：2-3 小时
:::

## 常见错误类型

LLM API 调用可能遇到多种错误，按是否可重试分为两大类。

```typescript
import Anthropic from "@anthropic-ai/sdk";

// 错误类型速查
const ERROR_TYPES: Record<number | string, [string, string]> = {
    // 不可重试——需要修改请求或配置
    401: ["AuthenticationError", "API Key 无效"],
    400: ["BadRequestError", "请求格式错误/Token 超限"],
    403: ["PermissionDeniedError", "无权限访问该模型"],
    404: ["NotFoundError", "模型/端点不存在"],

    // 可重试——等一等再试
    429: ["RateLimitError", "速率限制"],
    500: ["InternalServerError", "服务器内部错误"],
    503: ["APIStatusError", "服务暂时不可用"],
    timeout: ["APITimeoutError", "请求超时"],
    network: ["APIConnectionError", "网络连接失败"],
};
```

**判断原则**：认证错误和请求格式错误不应重试（重试也不会成功），速率限制和服务器错误应该重试（等一等就好了）。

## 指数退避重试

对于可重试的错误，指数退避（Exponential Backoff）是标准策略：每次重试等待时间翻倍，加随机抖动避免"惊群效应"。

```typescript
import Anthropic from "@anthropic-ai/sdk";

interface RetryOptions {
    maxRetries?: number;
    baseDelay?: number;
    maxDelay?: number;
    jitter?: boolean;
}

/**
 * 指数退避重试包装函数
 * TypeScript 没有装饰器的运行时参数化用法，用高阶函数实现同等效果
 */
function retryWithBackoff<T>(
    fn: (...args: any[]) => Promise<T>,
    options: RetryOptions = {}
): (...args: any[]) => Promise<T> {
    const {
        maxRetries = 3,
        baseDelay = 1.0,
        maxDelay = 60.0,
        jitter = true,
    } = options;

    return async (...args: any[]): Promise<T> => {
        let lastError: Error | null = null;
        for (let attempt = 0; attempt <= maxRetries; attempt++) {
            try {
                return await fn(...args);
            } catch (e: any) {
                // 请求格式错误不重试
                if (e instanceof Anthropic.BadRequestError) {
                    throw e;
                }
                // 可重试的错误类型
                if (
                    e instanceof Anthropic.RateLimitError ||
                    e instanceof Anthropic.InternalServerError ||
                    e instanceof Anthropic.APITimeoutError ||
                    e instanceof Anthropic.APIConnectionError
                ) {
                    lastError = e;
                    if (attempt === maxRetries) {
                        console.log(`[重试] 达到最大次数 ${maxRetries}，放弃`);
                        throw e;
                    }

                    // 等待时间：base * 2^attempt + 随机抖动
                    let delay = Math.min(baseDelay * (2 ** attempt), maxDelay);
                    if (jitter) {
                        delay *= (0.5 + Math.random());
                    }

                    console.log(`[重试] 第 ${attempt + 1} 次失败: ${e.constructor.name}`);
                    console.log(`[重试] 等待 ${delay.toFixed(1)}s (${attempt + 1}/${maxRetries})`);
                    await new Promise(resolve => setTimeout(resolve, delay * 1000));
                } else {
                    throw e;
                }
            }
        }
        throw lastError;
    };
}

// 使用
const client = new Anthropic();

const reliableChat = retryWithBackoff(
    async (message: string): Promise<string> => {
        const response = await client.messages.create({
            model: "claude-sonnet-4-20250514",
            max_tokens: 1024,
            messages: [{ role: "user", content: message }],
        });
        const block = response.content[0];
        return block.type === "text" ? block.text : "";
    },
    { maxRetries: 3, baseDelay: 1.0 }
);

const result = await reliableChat("什么是指数退避？");
console.log(result);
```

### 异步版本

```typescript
import Anthropic from "@anthropic-ai/sdk";

async function retryAsync<T>(
    fn: (...args: any[]) => Promise<T>,
    args: any[] = [],
    maxRetries: number = 3,
    baseDelay: number = 1.0,
): Promise<T> {
    /** 异步指数退避重试 */
    for (let attempt = 0; attempt <= maxRetries; attempt++) {
        try {
            return await fn(...args);
        } catch (e: any) {
            if (
                (e instanceof Anthropic.RateLimitError ||
                 e instanceof Anthropic.InternalServerError ||
                 e instanceof Anthropic.APITimeoutError) &&
                attempt < maxRetries
            ) {
                const delay = baseDelay * (2 ** attempt) * (0.5 + Math.random());
                console.log(`[异步重试] 等待 ${delay.toFixed(1)}s (${attempt + 1}/${maxRetries})`);
                await new Promise(resolve => setTimeout(resolve, delay * 1000));
            } else {
                throw e;
            }
        }
    }
    throw new Error("重试耗尽"); // TypeScript 需要确保返回
}

// 使用
async function main() {
    const client = new Anthropic();

    async function call(msg: string): Promise<string> {
        const resp = await client.messages.create({
            model: "claude-sonnet-4-20250514",
            max_tokens: 256,
            messages: [{ role: "user", content: msg }],
        });
        const block = resp.content[0];
        return block.type === "text" ? block.text : "";
    }

    const result = await retryAsync(call, ["你好"]);
    console.log(result);
}

// main();
```

## 速率限制处理

Rate Limit 是最常遇到的错误。除了被动重试，还可以主动控制请求频率。

### 批量请求的速率控制

```typescript
import Anthropic from "@anthropic-ai/sdk";

class BatchProcessor {
    /** 批量请求处理器，内置速率控制 */
    private client: Anthropic;
    private rpm: number;
    private interval: number;

    constructor(requestsPerMinute: number = 50) {
        this.client = new Anthropic();
        this.rpm = requestsPerMinute;
        this.interval = 60.0 / requestsPerMinute;
    }

    async processBatch(prompts: string[]): Promise<string[]> {
        const results: string[] = [];
        for (let i = 0; i < prompts.length; i++) {
            const start = Date.now();

            try {
                const response = await this.client.messages.create({
                    model: "claude-sonnet-4-20250514",
                    max_tokens: 256,
                    messages: [{ role: "user", content: prompts[i] }],
                });
                const block = response.content[0];
                results.push(block.type === "text" ? block.text : "");
                console.log(`[${i + 1}/${prompts.length}] 完成`);

            } catch (e: any) {
                if (e instanceof Anthropic.RateLimitError) {
                    console.log(`[${i + 1}] 触发速率限制，等待 60s...`);
                    await new Promise(resolve => setTimeout(resolve, 60000));
                    const response = await this.client.messages.create({
                        model: "claude-sonnet-4-20250514",
                        max_tokens: 256,
                        messages: [{ role: "user", content: prompts[i] }],
                    });
                    const block = response.content[0];
                    results.push(block.type === "text" ? block.text : "");
                } else {
                    throw e;
                }
            }

            // 控制请求频率
            const elapsed = (Date.now() - start) / 1000;
            if (elapsed < this.interval) {
                await new Promise(resolve =>
                    setTimeout(resolve, (this.interval - elapsed) * 1000)
                );
            }
        }
        return results;
    }
}

// const processor = new BatchProcessor(40);
// const results = await processor.processBatch(["问题1", "问题2", "问题3"]);
```

### 感知速率限制的客户端

```typescript
import Anthropic from "@anthropic-ai/sdk";

class RateLimitAwareClient {
    /** 主动感知速率限制的客户端 */
    private client: Anthropic;
    private remainingRequests: number = 999; // 初始假设充足

    constructor() {
        this.client = new Anthropic();
    }

    async chat(
        messages: Array<{ role: string; content: string }>,
        options: { model?: string; maxTokens?: number } = {}
    ): Promise<string> {
        const model = options.model ?? "claude-sonnet-4-20250514";
        const maxTokens = options.maxTokens ?? 1024;

        // 预检：配额即将耗尽时主动等待
        if (this.remainingRequests <= 1) {
            console.log("[速率] 配额即将耗尽，主动等待 5s...");
            await new Promise(resolve => setTimeout(resolve, 5000));
        }

        try {
            const response = await this.client.messages.create({
                model,
                max_tokens: maxTokens,
                messages: messages as any,
            });
            const block = response.content[0];
            return block.type === "text" ? block.text : "";

        } catch (e: any) {
            if (e instanceof Anthropic.RateLimitError) {
                // 从错误中提取 retry-after
                let waitTime = 60; // 默认等待
                const retryAfterHeader = e.headers?.["retry-after"];
                if (retryAfterHeader) {
                    waitTime = parseInt(retryAfterHeader, 10);
                }

                console.log(`[速率限制] 等待 ${waitTime}s...`);
                await new Promise(resolve => setTimeout(resolve, waitTime * 1000));

                const response = await this.client.messages.create({
                    model,
                    max_tokens: maxTokens,
                    messages: messages as any,
                });
                const block = response.content[0];
                return block.type === "text" ? block.text : "";
            }
            throw e;
        }
    }
}
```

## Token 超限处理

当输入超过模型的上下文窗口时，需要截断策略。

```typescript
function countTokensApprox(text: string): number {
    /** 粗略估算 Token 数 */
    let chinese = 0;
    for (const c of text) {
        if (c >= '\u4e00' && c <= '\u9fff') chinese++;
    }
    const other = text.length - chinese;
    return Math.floor(chinese / 1.5 + other / 4);
}

function truncateToFit(
    text: string,
    maxTokens: number = 100000,
    strategy: "tail" | "head" | "middle" = "tail"
): string {
    /** 将文本截断到合适长度 */
    const estimated = countTokensApprox(text);
    if (estimated <= maxTokens) {
        return text;
    }

    const ratio = (maxTokens / estimated) * 0.9; // 留 10% 余量
    const targetChars = Math.floor(text.length * ratio);

    if (strategy === "head") {
        return text.slice(0, targetChars) + "\n\n[... 内容已截断 ...]";
    } else if (strategy === "tail") {
        return "[... 早期内容已截断 ...]\n\n" + text.slice(-targetChars);
    } else if (strategy === "middle") {
        const half = Math.floor(targetChars / 2);
        return text.slice(0, half) + "\n\n[... 中间已截断 ...]\n\n" + text.slice(-half);
    }
    return text;
}

async function smartChat(
    messages: Array<{ role: string; content: string }>,
    maxContextTokens: number = 100000
): Promise<string> {
    /** 智能处理 Token 超限 */
    import Anthropic from "@anthropic-ai/sdk";
    const client = new Anthropic();

    const totalText = messages.map(m => m.content).join(" ");
    let estimated = countTokensApprox(totalText);

    if (estimated > maxContextTokens) {
        console.log(`[Token] 估算 ${estimated}，超过限制 ${maxContextTokens}`);

        // 策略 1：多轮对话删除早期消息
        while (estimated > maxContextTokens && messages.length > 2) {
            messages.shift();
            if (messages.length > 0 && messages[0].role === "assistant") {
                messages.shift();
            }
            const newText = messages.map(m => m.content).join(" ");
            estimated = countTokensApprox(newText);
        }

        // 策略 2：单条消息截断
        if (estimated > maxContextTokens) {
            messages[messages.length - 1].content = truncateToFit(
                messages[messages.length - 1].content,
                Math.floor(maxContextTokens / 2),
                "middle"
            );
        }
    }

    const response = await client.messages.create({
        model: "claude-sonnet-4-20250514",
        max_tokens: 1024,
        messages: messages as any,
    });
    const block = response.content[0];
    return block.type === "text" ? block.text : "";
}
```

## 生产级 LLM 客户端

将重试、速率控制、Token 截断和统计整合为一个完整的生产级封装：

```typescript
import Anthropic from "@anthropic-ai/sdk";

interface RetryConfig {
    maxRetries: number;
    baseDelay: number;
    maxDelay: number;
    jitter: boolean;
}

const defaultRetryConfig: RetryConfig = {
    maxRetries: 3,
    baseDelay: 1.0,
    maxDelay: 60.0,
    jitter: true,
};

interface CallStats {
    totalCalls: number;
    successful: number;
    retried: number;
    failed: number;
    totalInputTokens: number;
    totalOutputTokens: number;
    totalRetries: number;
    errors: Array<[string, string]>;
}

class RobustLLMClient {
    /** 健壮的 LLM 客户端——生产可用 */
    private client: Anthropic;
    private retryConfig: RetryConfig;
    private maxContextTokens: number;
    private stats: CallStats;

    constructor(
        retryConfig: Partial<RetryConfig> = {},
        timeout: number = 120.0,
        maxContextTokens: number = 100000,
    ) {
        this.client = new Anthropic({ timeout: timeout * 1000 });
        this.retryConfig = { ...defaultRetryConfig, ...retryConfig };
        this.maxContextTokens = maxContextTokens;
        this.stats = {
            totalCalls: 0, successful: 0, retried: 0, failed: 0,
            totalInputTokens: 0, totalOutputTokens: 0, totalRetries: 0,
            errors: [],
        };
    }

    async chat(
        messages: Array<{ role: string; content: string }>,
        options: {
            model?: string;
            maxTokens?: number;
            system?: string;
            temperature?: number;
        } = {},
    ): Promise<string | null> {
        /** 带完整错误处理的对话请求 */
        const {
            model = "claude-sonnet-4-20250514",
            maxTokens = 1024,
            system = "",
            temperature = 0,
        } = options;

        this.stats.totalCalls++;

        // 预处理：Token 检查
        messages = this.checkTokens(messages);

        const kwargs: any = {
            model,
            max_tokens: maxTokens,
            messages,
            temperature,
        };
        if (system) {
            kwargs.system = system;
        }

        let lastError: Error | null = null;
        for (let attempt = 0; attempt <= this.retryConfig.maxRetries; attempt++) {
            try {
                const response = await this.client.messages.create(kwargs);
                this.stats.successful++;
                this.stats.totalInputTokens += response.usage.input_tokens;
                this.stats.totalOutputTokens += response.usage.output_tokens;

                if (attempt > 0) {
                    this.stats.retried++;
                    console.log(`第 ${attempt} 次重试成功`);
                }

                if (response.stop_reason === "max_tokens") {
                    console.warn(`响应被截断（达到 ${maxTokens} token）`);
                }

                const block = response.content[0];
                return block.type === "text" ? block.text : "";

            } catch (e: any) {
                if (e instanceof Anthropic.AuthenticationError) {
                    console.error(`认证失败: ${e}`);
                    this.stats.failed++;
                    this.stats.errors.push(["Auth", String(e)]);
                    return null;
                }

                if (e instanceof Anthropic.BadRequestError) {
                    if (String(e).toLowerCase().includes("token")) {
                        console.warn("Token 超限，尝试截断");
                        messages = this.aggressiveTruncate(messages);
                        kwargs.messages = messages;
                        continue;
                    }
                    console.error(`请求错误: ${e}`);
                    this.stats.failed++;
                    return null;
                }

                if (e instanceof Anthropic.RateLimitError) {
                    lastError = e;
                    this.stats.totalRetries++;
                    const wait = this.waitTime(attempt, true);
                    console.warn(`速率限制，等待 ${wait.toFixed(1)}s (${attempt + 1}/${this.retryConfig.maxRetries})`);
                    await new Promise(resolve => setTimeout(resolve, wait * 1000));
                    continue;
                }

                if (
                    e instanceof Anthropic.InternalServerError ||
                    e instanceof Anthropic.APIStatusError ||
                    e instanceof Anthropic.APITimeoutError ||
                    e instanceof Anthropic.APIConnectionError
                ) {
                    lastError = e;
                    this.stats.totalRetries++;
                    const wait = this.waitTime(attempt);
                    console.warn(`${e.constructor.name}，等待 ${wait.toFixed(1)}s (${attempt + 1}/${this.retryConfig.maxRetries})`);
                    await new Promise(resolve => setTimeout(resolve, wait * 1000));
                    continue;
                }

                throw e;
            }
        }

        this.stats.failed++;
        this.stats.errors.push([lastError?.constructor.name ?? "Unknown", String(lastError)]);
        console.error(`所有重试均失败: ${lastError}`);
        return null;
    }

    private waitTime(attempt: number, rateLimited: boolean = false): number {
        const cfg = this.retryConfig;
        const base = rateLimited ? cfg.baseDelay * 2 : cfg.baseDelay;
        let delay = Math.min(base * (2 ** attempt), cfg.maxDelay);
        if (cfg.jitter) {
            delay *= (0.5 + Math.random());
        }
        return delay;
    }

    private checkTokens(
        messages: Array<{ role: string; content: string }>
    ): Array<{ role: string; content: string }> {
        const total = messages
            .filter(m => typeof m.content === "string")
            .map(m => m.content)
            .join(" ");
        if (Math.floor(total.length / 3) > this.maxContextTokens) {
            console.warn("Token 超限，执行截断");
            return this.aggressiveTruncate(messages);
        }
        return messages;
    }

    private aggressiveTruncate(
        messages: Array<{ role: string; content: string }>
    ): Array<{ role: string; content: string }> {
        if (messages.length > 4) {
            messages = messages.slice(-4);
        }
        for (const msg of messages) {
            if (typeof msg.content === "string" && msg.content.length > 50000) {
                msg.content =
                    msg.content.slice(0, 25000) + "\n[...截断...]\n" + msg.content.slice(-25000);
            }
        }
        return messages;
    }

    getStats(): Record<string, any> {
        const s = this.stats;
        return {
            total: s.totalCalls,
            success: s.successful,
            retried: s.retried,
            failed: s.failed,
            success_rate: `${((s.successful / Math.max(s.totalCalls, 1)) * 100).toFixed(1)}%`,
            total_tokens: s.totalInputTokens + s.totalOutputTokens,
            recent_errors: s.errors.slice(-5),
        };
    }
}

// 使用
async function main() {
    const llm = new RobustLLMClient(
        { maxRetries: 3, baseDelay: 1.0 },
        60.0,
    );

    const result = await llm.chat(
        [{ role: "user", content: "什么是指数退避重试？" }],
        { system: "用一句话简洁回答。" },
    );
    if (result) {
        console.log(`回复: ${result}`);
    }

    console.log("统计:", llm.getStats());
}

main();
```

## 生产环境补充策略

上面的 `RobustLLMClient` 覆盖了核心能力，生产环境还可以考虑：

```typescript
// 1. 断路器模式——连续失败时暂停请求
class CircuitBreaker {
    /** 连续失败 N 次后暂停一段时间 */
    private failures: number = 0;
    private threshold: number;
    private timeout: number;
    private lastFailureTime: number = 0;
    private state: "closed" | "open" | "half-open" = "closed";

    constructor(failureThreshold: number = 5, recoveryTimeout: number = 30) {
        this.threshold = failureThreshold;
        this.timeout = recoveryTimeout;
    }

    recordSuccess(): void {
        this.failures = 0;
        this.state = "closed";
    }

    recordFailure(): void {
        this.failures++;
        this.lastFailureTime = Date.now() / 1000;
        if (this.failures >= this.threshold) {
            this.state = "open";
        }
    }

    canProceed(): boolean {
        if (this.state === "closed") {
            return true;
        }
        if (this.state === "open") {
            if (Date.now() / 1000 - this.lastFailureTime > this.timeout) {
                this.state = "half-open";
                return true;
            }
            return false;
        }
        return true; // half-open 允许尝试
    }
}

// 2. 模型降级——主模型不可用时切换备用
const MODEL_FALLBACK: string[] = [
    "claude-sonnet-4-20250514",   // 主模型
    "claude-haiku-3-5-20241022",  // 备用模型（更便宜更快）
];

// 3. 预算控制——每日 Token 上限
class BudgetController {
    private limit: number;
    private used: number = 0;

    constructor(dailyTokenLimit: number = 1_000_000) {
        this.limit = dailyTokenLimit;
    }

    check(estimatedTokens: number): boolean {
        return this.used + estimatedTokens < this.limit;
    }

    record(tokens: number): void {
        this.used += tokens;
    }
}
```

::: warning 生产环境建议
1. **日志与监控**：记录每次调用的延迟、Token 用量、错误类型
2. **断路器模式**：连续失败 N 次后暂停，避免无效重试消耗配额
3. **降级策略**：主模型不可用时自动切换备用模型
4. **预算控制**：设置每日/每月 Token 上限，超限时拒绝而非继续调用
5. **异步队列**：高并发场景用请求队列 + 工作池统一管理速率
:::

## 小结

1. **错误分类**：认证和请求错误不可重试，速率限制和服务器错误可重试
2. **指数退避**：等待时间按 `base * 2^attempt` 增长，加随机抖动避免惊群
3. **速率控制**：主动控制请求频率，读取 retry-after 响应头
4. **Token 截断**：支持头部、尾部、中间三种策略，多轮对话优先删除早期消息
5. **生产封装**：统一错误处理 + 重试 + 截断 + 统计，一个类搞定

## 练习

1. **基础练习**：实现 `retryWithBackoff` 高阶函数，支持配置最大重试次数、基础延迟和可重试异常类型。用 mock 模拟异常写单元测试。

2. **断路器练习**：实现一个完整的断路器——连续失败 5 次进入"断开"状态（拒绝所有请求 30 秒），30 秒后进入"半开"（允许 1 个请求），成功则恢复，失败则继续断开。

3. **降级练习**：基于 `RobustLLMClient`，增加模型降级功能——claude-sonnet 连续失败 3 次时自动降级到 claude-haiku；haiku 也失败时返回预设兜底回复。每 5 分钟尝试恢复主模型。

4. **综合练习**：构建一个处理 100 条文本分类请求的批处理脚本，要求：自动速率控制（不超过 40 RPM）、失败自动重试、结果写入 CSV、完成后输出统计报告（成功率、总 Token、平均延迟）。

## 参考资源

- [Anthropic Rate Limits 文档](https://docs.anthropic.com/en/api/rate-limits)
- [Anthropic Error Handling 文档](https://docs.anthropic.com/en/api/errors)
- [指数退避算法（AWS 文档）](https://docs.aws.amazon.com/general/latest/gr/api-retries.html)
- [断路器模式（Martin Fowler）](https://martinfowler.com/bliki/CircuitBreaker.html)
