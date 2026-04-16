# 生产工程化 · 进阶篇

::: tip 学习目标
- 掌握生产级 Agent 的架构设计（单体 vs 微服务）
- 实现完善的错误处理体系：重试、降级、Circuit Breaker
- 检测和处理 Agent 死循环问题
- 用 Docker 容器化部署你的 Agent 服务
:::

::: info 学完你能做到
- 设计一个合理的 Agent 服务架构
- 让你的 Agent 在各种异常情况下都能优雅应对
- 检测 Agent 陷入死循环并安全中断
- 用 docker-compose 一键部署完整的 Agent 系统
:::

## 架构设计：从简单开始

### 单体架构（推荐起步）

```typescript
/** monolith.ts — 单体架构的 Agent 服务 */

import express from "express";
import { WebSocketServer, WebSocket } from "ws";
import http from "http";

const app = express();
app.use(express.json());

// 启动时初始化所有组件
const agentRunner = new AgentRunner();
const toolRegistry = new ToolRegistry();
await toolRegistry.initialize();

// 应用退出时清理
process.on("SIGTERM", async () => {
    await toolRegistry.cleanup();
    process.exit(0);
});

app.post("/api/chat", async (req, res) => {
    const response = await agentRunner.run({
        message: req.body.message,
        conversationId: req.body.conversation_id,
    });
    res.json({ response });
});

const server = http.createServer(app);
const wss = new WebSocketServer({ server, path: "/ws/chat" });

wss.on("connection", (ws: WebSocket) => {
    ws.on("message", async (raw) => {
        const data = JSON.parse(raw.toString());
        for await (const chunk of agentRunner.stream({ message: data.message })) {
            ws.send(JSON.stringify({ chunk }));
        }
        ws.send(JSON.stringify({ done: true }));
    });
});

server.listen(8000);
```

::: warning 架构选型建议
- **DAU < 1000**：单体架构足够，不要过度设计
- **DAU 1000-10000**：考虑将 LLM 调用和工具执行异步化
- **DAU > 10000**：按功能域拆分微服务
:::

### 数据库设计

Agent 系统需要持久化对话、消息和工具执行记录：

```sql
-- 对话表
CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(64) NOT NULL,
    title VARCHAR(256),
    model VARCHAR(64) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- 消息表
CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID REFERENCES conversations(id),
    role VARCHAR(16) NOT NULL,
    content TEXT NOT NULL,
    token_count_input INT DEFAULT 0,
    token_count_output INT DEFAULT 0,
    latency_ms INT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- 工具执行记录表
CREATE TABLE tool_executions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    message_id UUID REFERENCES messages(id),
    tool_name VARCHAR(128) NOT NULL,
    tool_input JSONB NOT NULL,
    tool_output TEXT,
    status VARCHAR(16) NOT NULL,  -- 'success', 'error', 'timeout'
    duration_ms INT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_messages_conv ON messages(conversation_id, created_at);
CREATE INDEX idx_tool_exec_msg ON tool_executions(message_id);
```

## 完善的错误处理体系

### 工具执行失败：重试 + 降级

```typescript
/** tool_fallback.ts — 工具降级策略 */

interface ToolResult {
    success: boolean;
    output: string;
    error?: string;
}

class ToolExecutor {
    /** 支持降级的工具执行器 */

    private fallbacks: Record<string, string[]> = {
        web_search: ["cached_search", "knowledge_base"],
        database_query: ["cached_query", "static_data"],
    };

    async executeWithFallback(
        toolName: string,
        toolInput: Record<string, unknown>,
    ): Promise<ToolResult> {
        /** 执行工具，失败时逐级降级 */
        // 尝试主工具
        const result = await this.tryExecute(toolName, toolInput);
        if (result.success) return result;

        // 逐个尝试备选方案
        for (const fallback of this.fallbacks[toolName] || []) {
            console.log(`[降级] ${toolName} 失败，尝试 ${fallback}`);
            const fallbackResult = await this.tryExecute(fallback, toolInput);
            if (fallbackResult.success) {
                fallbackResult.output = `[降级结果] ${fallbackResult.output}`;
                return fallbackResult;
            }
        }

        return {
            success: false,
            output: "",
            error: `工具 ${toolName} 及所有备选方案均失败`,
        };
    }

    private async tryExecute(
        name: string,
        inputData: Record<string, unknown>,
    ): Promise<ToolResult> {
        try {
            const output = await Promise.race([
                toolRegistry.execute(name, inputData),
                new Promise<never>((_, reject) =>
                    setTimeout(() => reject(new Error("timeout")), 30000)
                ),
            ]);
            return { success: true, output };
        } catch (e) {
            return { success: false, output: "", error: String(e) };
        }
    }
}
```

### Circuit Breaker：防止级联故障

当 LLM API 持续失败时，继续重试只会让情况更糟。熔断器在连续失败达到阈值后"断开电路"，直接拒绝请求：

```typescript
/** circuit_breaker.ts — 熔断器模式 */

enum CircuitState {
    CLOSED = "closed",        // 正常运行
    OPEN = "open",            // 熔断（拒绝请求）
    HALF_OPEN = "half_open",  // 试探性恢复
}

class CircuitBreaker {
    /** 熔断器：连续失败时暂停请求，等服务恢复 */

    private state: CircuitState = CircuitState.CLOSED;
    private failureCount: number = 0;
    private successCount: number = 0;
    private lastFailureTime: number = 0;

    constructor(
        private failureThreshold: number = 5,      // 连续失败几次后熔断
        private recoveryTimeout: number = 60.0,     // 熔断多久后尝试恢复（秒）
        private successThreshold: number = 3,       // 恢复期连续成功几次才算恢复
    ) {}

    canExecute(): boolean {
        if (this.state === CircuitState.CLOSED) {
            return true;
        } else if (this.state === CircuitState.OPEN) {
            if (Date.now() / 1000 - this.lastFailureTime > this.recoveryTimeout) {
                this.state = CircuitState.HALF_OPEN;
                this.successCount = 0;
                return true;
            }
            return false;
        } else {
            // HALF_OPEN
            return true;
        }
    }

    recordSuccess(): void {
        if (this.state === CircuitState.HALF_OPEN) {
            this.successCount++;
            if (this.successCount >= this.successThreshold) {
                this.state = CircuitState.CLOSED;
                this.failureCount = 0;
            }
        } else {
            this.failureCount = 0;
        }
    }

    recordFailure(): void {
        this.failureCount++;
        this.lastFailureTime = Date.now() / 1000;
        if (this.failureCount >= this.failureThreshold) {
            this.state = CircuitState.OPEN;
        }
    }
}
```

### Agent 死循环检测

Agent 可能陷入无限循环——反复调用同一工具或在错误中打转：

```typescript
/** loop_detector.ts — 死循环检测器 */

class LoopDetector {
    /** 检测 Agent 是否陷入死循环 */

    private iterationCount: number = 0;
    private toolHistory: string[] = [];
    private totalTokens: number = 0;

    constructor(
        private maxIterations: number = 25,
        private maxSameToolConsecutive: number = 5,
        private maxTotalTokens: number = 100000,
    ) {}

    check(toolName?: string, tokensUsed: number = 0): string | null {
        /** 返回 null 表示正常，返回字符串表示应中断及原因 */
        this.iterationCount++;
        this.totalTokens += tokensUsed;
        if (toolName) {
            this.toolHistory.push(toolName);
        }

        // 检查 1：最大迭代次数
        if (this.iterationCount > this.maxIterations) {
            return `已达最大迭代次数 (${this.maxIterations})`;
        }

        // 检查 2：连续调用同一工具
        if (toolName && this.toolHistory.length >= this.maxSameToolConsecutive) {
            const recent = this.toolHistory.slice(-this.maxSameToolConsecutive);
            if (new Set(recent).size === 1) {
                return `连续 ${this.maxSameToolConsecutive} 次调用 '${toolName}'`;
            }
        }

        // 检查 3：Token 用量超限
        if (this.totalTokens > this.maxTotalTokens) {
            return `Token 用量超过限制 (${this.maxTotalTokens})`;
        }

        // 检查 4：重复模式（如 A->B->A->B 循环）
        if (this.toolHistory.length >= 6) {
            for (const plen of [2, 3]) {
                const pattern = this.toolHistory.slice(-plen);
                const prev = this.toolHistory.slice(-2 * plen, -plen);
                if (JSON.stringify(pattern) === JSON.stringify(prev)) {
                    return `检测到重复模式: ${JSON.stringify(pattern)}`;
                }
            }
        }

        return null;
    }
}
```

## 容器化部署

### Dockerfile

```dockerfile
# 多阶段构建 —— 最终镜像更小
FROM node:20-slim AS builder

WORKDIR /app
COPY package*.json .
RUN npm ci --omit=dev

FROM node:20-slim

WORKDIR /app
COPY --from=builder /app/node_modules ./node_modules
COPY . .

# 非 root 用户运行（安全最佳实践）
RUN useradd -m appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD node -e "fetch('http://localhost:8000/health').then(r => { if (!r.ok) throw r })"

CMD ["node", "--import", "tsx", "src/main.ts"]
```

### docker-compose.yml

```yaml
version: "3.8"

services:
  agent-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - DATABASE_URL=postgresql://user:pass@postgres:5432/agent_db
      - REDIS_URL=redis://redis:6379
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    deploy:
      replicas: 2
      resources:
        limits:
          memory: 1G
          cpus: "1.0"

  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: agent_db
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U user -d agent_db"]
      interval: 10s

  redis:
    image: redis:7-alpine
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s

volumes:
  pgdata:
```

启动整套系统：

```bash
# 构建并启动
docker-compose up -d

# 查看日志
docker-compose logs -f agent-api

# 扩容 API 实例
docker-compose up -d --scale agent-api=4
```

## 小结

生产级 Agent 的三大工程支柱：

1. **架构设计**：单体起步，数据库 schema 先设计好，按需拆分
2. **容错体系**：重试（指数退避）+ 降级（备选方案）+ 熔断（Circuit Breaker）+ 死循环检测
3. **容器化部署**：Docker + docker-compose，确保环境一致性和可扩展性

## 练习

1. 为你的 Agent 服务添加 Circuit Breaker，模拟 LLM API 连续失败的场景
2. 实现 LoopDetector，测试当 Agent 陷入 A->B->A->B 循环时能否被正确检测
3. 用 docker-compose 部署你的 Agent + PostgreSQL + Redis，验证健康检查是否正常

## 参考资源

- [Circuit Breaker Pattern (Martin Fowler)](https://martinfowler.com/bliki/CircuitBreaker.html) -- 熔断器模式详解
- [Exponential Backoff And Jitter (AWS)](https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/) -- 退避策略最佳实践
- [Docker 官方文档](https://docs.docker.com/) -- 容器化基础
- [Release It! (Michael Nygard)](https://pragprog.com/titles/mnee2/release-it-second-edition/) -- 生产环境稳定性模式经典
- [Express 官方文档](https://expressjs.com/) -- Node.js Web 框架
