# 生产工程化 · 入门篇

::: tip 学习目标
- 理解从 TypeScript 脚本到可部署服务的转变
- 掌握标准的项目结构和配置管理方式
- 学会用环境变量安全地管理 API Key 等敏感信息
- 实现简单但有效的错误处理和日志记录
:::

::: info 学完你能做到
- 把一个原型脚本里的 Agent 改造成标准项目结构
- 用 `.env` 文件和 dotenv + TypeScript interface 管理配置
- 写出规范的错误处理代码，而不是到处 `try { } catch { }`
- 让你的程序有清晰可查的日志输出
:::

## 从脚本到服务：为什么需要工程化

你的 Agent 原型可能长这样：

```typescript
// agent.ts —— 一个典型的原型脚本
import Anthropic from "@anthropic-ai/sdk";
import * as readline from "readline";

const client = new Anthropic();  // API Key 硬编码在环境变量里

const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
const question = (prompt: string) => new Promise<string>((resolve) => rl.question(prompt, resolve));

const userInput = await question("你想问什么？");
const response = await client.messages.create({
    model: "claude-sonnet-4-20250514",
    max_tokens: 4096,
    messages: [{ role: "user", content: userInput }],
});
console.log(response.content[0].type === "text" ? response.content[0].text : "");
rl.close();
```

这个脚本能跑，但距离"别人也能用"差了十万八千里：

- **API Key 怎么管理？** 换台机器就跑不了
- **出错了怎么办？** 网络断了直接崩溃
- **别人怎么调用？** 只能命令行手动输入
- **出了 Bug 怎么查？** 没有日志，全靠 print

接下来我们一步步解决这些问题。

## 标准项目结构

一个可部署的 Agent 服务至少需要这样的结构：

```
my-agent-service/
├── src/
│   ├── main.ts              # 应用入口（Express）
│   ├── agent.ts              # Agent 核心逻辑
│   ├── config.ts             # 配置管理
│   └── tools/                # 工具定义
│       └── search.ts
├── tests/
│   └── agent.test.ts
├── .env                      # 环境变量（不提交到 Git！）
├── .env.example              # 环境变量模板（提交到 Git）
├── .gitignore
├── package.json
├── tsconfig.json
└── README.md
```

::: warning 千万不要提交 .env 文件
`.env` 里存的是 API Key、数据库密码等敏感信息。在 `.gitignore` 中加上 `.env`，只提交 `.env.example` 作为模板。
:::

## 环境变量与配置管理

### 为什么不能硬编码配置

```typescript
// 错误示范 —— 硬编码 API Key
const client = new Anthropic({ apiKey: "sk-ant-api03-xxxx" });  // 泄露！

// 错误示范 —— 硬编码模型名称
const model = "claude-sonnet-4-20250514";  // 想换模型就得改代码
```

### 用 dotenv + TypeScript interface 管理配置

```typescript
/** config.ts — 配置管理模块 */

import dotenv from "dotenv";
dotenv.config();  // 从 .env 文件加载环境变量

interface Settings {
    // API Keys（必填，不提供会启动报错）
    anthropicApiKey: string;
    // 可选配置（有默认值）
    defaultModel: string;
    maxTokens: number;
    agentTimeoutSeconds: number;
    // 日志配置
    logLevel: string;
    // 运行环境
    environment: string;
    debug: boolean;
}

let _settings: Settings | null = null;

export function getSettings(): Settings {
    /** 获取配置单例（整个应用生命周期只加载一次） */
    if (_settings) return _settings;

    const apiKey = process.env.ANTHROPIC_API_KEY;
    if (!apiKey) {
        throw new Error("ANTHROPIC_API_KEY 环境变量未设置");
    }

    _settings = {
        anthropicApiKey: apiKey,
        defaultModel: process.env.DEFAULT_MODEL || "claude-sonnet-4-20250514",
        maxTokens: parseInt(process.env.MAX_TOKENS || "4096", 10),
        agentTimeoutSeconds: parseInt(process.env.AGENT_TIMEOUT_SECONDS || "120", 10),
        logLevel: process.env.LOG_LEVEL || "info",
        environment: process.env.ENVIRONMENT || "development",
        debug: process.env.DEBUG === "true",
    };
    return _settings;
}
```

对应的 `.env` 文件：

```bash
# .env —— 本地开发环境配置
ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
DEFAULT_MODEL=claude-sonnet-4-20250514
LOG_LEVEL=debug
ENVIRONMENT=development
```

`.env.example` 文件（提交到 Git，告诉其他开发者需要哪些配置）：

```bash
# .env.example —— 配置模板
ANTHROPIC_API_KEY=your-api-key-here
DEFAULT_MODEL=claude-sonnet-4-20250514
LOG_LEVEL=info
ENVIRONMENT=development
```

### 在代码中使用配置

```typescript
/** agent.ts — 使用配置的 Agent */

import Anthropic from "@anthropic-ai/sdk";
import { getSettings } from "./config";

const settings = getSettings();

// API Key 从配置中读取，不再硬编码
const client = new Anthropic({ apiKey: settings.anthropicApiKey });

async function chat(message: string): Promise<string> {
    const response = await client.messages.create({
        model: settings.defaultModel,      // 模型名称可配置
        max_tokens: settings.maxTokens,    // Token 限制可配置
        messages: [{ role: "user", content: message }],
    });
    return response.content[0].type === "text" ? response.content[0].text : "";
}
```

## 简单但有效的错误处理

### Agent 系统中常见的错误类型

| 错误类型 | 举例 | 正确处理方式 |
|---------|------|------------|
| 认证失败 | API Key 无效（401） | 不重试，提醒用户检查配置 |
| 请求无效 | 参数错误（400） | 不重试，修复代码 |
| 速率限制 | 请求太频繁（429） | 等一会儿再试 |
| 服务器错误 | API 临时故障（500） | 等一会儿再试 |
| 网络超时 | 网络不通 | 等一会儿再试，多次失败则报错 |

### 基础的重试机制

```typescript
/** retry.ts — 简单的重试工具函数 */

/**
 * 带指数退避的重试函数
 *
 * 指数退避的意思是：第一次等 1 秒，第二次等 2 秒，第三次等 4 秒...
 * 加上随机抖动（jitter），避免所有请求同时重试导致"惊群效应"。
 */
async function retryOnError<T>(
    fn: () => Promise<T>,
    maxRetries: number = 3,
    baseDelay: number = 1.0,
): Promise<T> {
    let lastError: Error | null = null;

    for (let attempt = 0; attempt <= maxRetries; attempt++) {
        try {
            return await fn();
        } catch (e) {
            lastError = e as Error;

            // 最后一次重试也失败了，不再等待
            if (attempt >= maxRetries) break;

            // 计算等待时间：指数递增 + 随机抖动
            const delay = Math.random() * baseDelay * (2 ** attempt);
            console.log(
                `[重试] 第 ${attempt + 1} 次，等待 ${delay.toFixed(1)}s，原因: ${e}`
            );
            await new Promise((resolve) => setTimeout(resolve, delay * 1000));
        }
    }

    throw lastError;
}

// 使用示例
import Anthropic from "@anthropic-ai/sdk";

async function callLLM(messages: Anthropic.MessageParam[]): Promise<string> {
    /** 调用 LLM API（自动重试） */
    return retryOnError(async () => {
        const client = new Anthropic();
        const response = await client.messages.create({
            model: "claude-sonnet-4-20250514",
            max_tokens: 4096,
            messages,
        });
        return response.content[0].type === "text" ? response.content[0].text : "";
    }, 3, 2.0);
}
```

### 区分"该重试"和"不该重试"的错误

```typescript
/** smart_error_handling.ts — 更聪明的错误处理 */

import Anthropic from "@anthropic-ai/sdk";

async function safeChat(message: string): Promise<string> {
    /** 带错误分类的聊天函数 */
    const client = new Anthropic();

    try {
        const response = await client.messages.create({
            model: "claude-sonnet-4-20250514",
            max_tokens: 4096,
            messages: [{ role: "user", content: message }],
        });
        return response.content[0].type === "text" ? response.content[0].text : "";

    } catch (e) {
        if (e instanceof Anthropic.AuthenticationError) {
            // 401 —— API Key 有问题，重试没意义
            return "系统配置错误，请联系管理员。";
        }
        if (e instanceof Anthropic.BadRequestError) {
            // 400 —— 请求参数有问题
            return `请求参数错误: ${e.message}`;
        }
        if (e instanceof Anthropic.RateLimitError) {
            // 429 —— 请求太频繁，应该重试
            // 实际项目中这里会触发重试逻辑
            return "系统繁忙，请稍后再试。";
        }
        if (e instanceof Anthropic.InternalServerError) {
            // 500 —— 服务端临时故障，应该重试
            return "服务暂时不可用，请稍后再试。";
        }
        // 兜底处理
        console.error(`[错误] 未知异常: ${e}`);
        return "处理过程中出现问题，请稍后重试。";
    }
}
```

## 日志：从 print 到结构化记录

### 为什么不能用 print

`print` 的问题：没有时间戳、没有级别、没有上下文、生产环境看不到。

### 基础日志设置

```typescript
/** logging_setup.ts — 结构化日志 */

interface LogData {
    timestamp: string;
    level: string;
    message: string;
    logger: string;
    exception?: string;
    [key: string]: unknown;
}

class JSONLogger {
    /** JSON 格式化日志器 —— 方便日志平台解析 */
    constructor(private name: string) {}

    private format(level: string, message: string, extra?: Record<string, unknown>): string {
        const logData: LogData = {
            timestamp: new Date().toISOString(),
            level,
            message,
            logger: this.name,
            ...extra,
        };
        return JSON.stringify(logData);
    }

    info(message: string, extra?: Record<string, unknown>): void {
        console.log(this.format("INFO", message, extra));
    }

    error(message: string, error?: Error, extra?: Record<string, unknown>): void {
        const errorExtra = error ? { exception: error.stack || error.message, ...extra } : extra;
        console.error(this.format("ERROR", message, errorExtra));
    }

    warn(message: string, extra?: Record<string, unknown>): void {
        console.warn(this.format("WARN", message, extra));
    }

    debug(message: string, extra?: Record<string, unknown>): void {
        console.debug(this.format("DEBUG", message, extra));
    }
}

// 在业务代码中使用
const logger = new JSONLogger("agent");

async function chatWithLogging(message: string): Promise<string> {
    logger.info("收到用户消息", { message_length: message.length });

    try {
        const result = await callLLM([{ role: "user", content: message }]);
        logger.info("LLM 调用成功");
        return result;
    } catch (e) {
        logger.error(`LLM 调用失败: ${e}`, e as Error);
        return "处理出错，请稍后重试。";
    }
}
```

## 用 Express 暴露为 HTTP 服务

最后一步，把你的 Agent 包装成一个 HTTP 服务，让前端或其他系统可以调用：

```typescript
/** main.ts — 应用入口 */

import express from "express";
import { getSettings } from "./config";

const settings = getSettings();
const app = express();
app.use(express.json());

interface ChatRequest {
    message: string;
}

interface ChatResponse {
    reply: string;
}

app.post("/api/chat", async (req, res) => {
    /** 聊天接口 */
    const { message } = req.body as ChatRequest;
    if (!message) {
        res.status(400).json({ error: "message is required" });
        return;
    }
    const reply = await safeChat(message);
    const response: ChatResponse = { reply };
    res.json(response);
});

app.get("/health", async (req, res) => {
    /** 健康检查 */
    res.json({ status: "ok" });
});

app.listen(8000, () => {
    console.log("Server running on http://0.0.0.0:8000");
});
```

启动服务：

```bash
# 安装依赖
npm install express @anthropic-ai/sdk dotenv
npm install -D typescript @types/express @types/node tsx

# 启动服务
npx tsx src/main.ts
```

## 小结

从脚本到服务的关键步骤：

1. **项目结构**：清晰的目录组织，代码、配置、测试分离
2. **配置管理**：dotenv + TypeScript interface + `.env` 文件，敏感信息不入代码
3. **错误处理**：区分可重试和不可重试的错误，用指数退避处理瞬时故障
4. **日志记录**：JSON 格式的结构化日志，方便查问题
5. **HTTP 接口**：Express 暴露服务，提供健康检查端点

## 练习

1. 用上面的项目结构创建一个新项目，实现一个简单的翻译 Agent
2. 给你的 Agent 添加一个 `max_concurrent_requests` 配置项，限制同时处理的请求数
3. 故意把 API Key 改错，观察错误处理是否按预期工作

## 参考资源

- [Express 官方文档](https://expressjs.com/) -- Node.js Web 框架
- [dotenv](https://github.com/motdotla/dotenv) -- 环境变量管理
- [The Twelve-Factor App](https://12factor.net/) -- 现代应用设计 12 原则（尤其是第三条：配置）
- [Anthropic API Error Handling](https://docs.anthropic.com/en/api/errors) -- API 错误码参考
