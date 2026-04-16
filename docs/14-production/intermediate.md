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

```python
"""monolith.py — 单体架构的 Agent 服务"""

from fastapi import FastAPI, WebSocket
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时初始化所有组件
    app.state.agent_runner = AgentRunner()
    app.state.tool_registry = ToolRegistry()
    await app.state.tool_registry.initialize()
    yield
    await app.state.tool_registry.cleanup()

app = FastAPI(lifespan=lifespan)

@app.post("/api/chat")
async def chat(request: ChatRequest):
    response = await app.state.agent_runner.run(
        message=request.message,
        conversation_id=request.conversation_id,
    )
    return {"response": response}

@app.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_json()
        async for chunk in app.state.agent_runner.stream(
            message=data["message"],
        ):
            await websocket.send_json({"chunk": chunk})
        await websocket.send_json({"done": True})
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

```python
"""tool_fallback.py — 工具降级策略"""

from dataclasses import dataclass

@dataclass
class ToolResult:
    success: bool
    output: str
    error: str | None = None

class ToolExecutor:
    """支持降级的工具执行器"""

    def __init__(self):
        self.fallbacks = {
            "web_search": ["cached_search", "knowledge_base"],
            "database_query": ["cached_query", "static_data"],
        }

    async def execute_with_fallback(
        self, tool_name: str, tool_input: dict
    ) -> ToolResult:
        """执行工具，失败时逐级降级"""
        # 尝试主工具
        result = await self._try_execute(tool_name, tool_input)
        if result.success:
            return result

        # 逐个尝试备选方案
        for fallback in self.fallbacks.get(tool_name, []):
            print(f"[降级] {tool_name} 失败，尝试 {fallback}")
            result = await self._try_execute(fallback, tool_input)
            if result.success:
                result.output = f"[降级结果] {result.output}"
                return result

        return ToolResult(
            success=False, output="",
            error=f"工具 {tool_name} 及所有备选方案均失败",
        )

    async def _try_execute(self, name: str, input_data: dict) -> ToolResult:
        try:
            import asyncio
            output = await asyncio.wait_for(
                tool_registry.execute(name, input_data), timeout=30.0
            )
            return ToolResult(success=True, output=output)
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))
```

### Circuit Breaker：防止级联故障

当 LLM API 持续失败时，继续重试只会让情况更糟。熔断器在连续失败达到阈值后"断开电路"，直接拒绝请求：

```python
"""circuit_breaker.py — 熔断器模式"""

import time
from enum import Enum

class CircuitState(Enum):
    CLOSED = "closed"        # 正常运行
    OPEN = "open"            # 熔断（拒绝请求）
    HALF_OPEN = "half_open"  # 试探性恢复

class CircuitBreaker:
    """熔断器：连续失败时暂停请求，等服务恢复"""

    def __init__(
        self,
        failure_threshold: int = 5,    # 连续失败几次后熔断
        recovery_timeout: float = 60.0, # 熔断多久后尝试恢复
        success_threshold: int = 3,     # 恢复期连续成功几次才算恢复
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0

    def can_execute(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        elif self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                return True
            return False
        else:  # HALF_OPEN
            return True

    def record_success(self):
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
        else:
            self.failure_count = 0

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
```

### Agent 死循环检测

Agent 可能陷入无限循环——反复调用同一工具或在错误中打转：

```python
"""loop_detector.py — 死循环检测器"""

class LoopDetector:
    """检测 Agent 是否陷入死循环"""

    def __init__(
        self,
        max_iterations: int = 25,
        max_same_tool_consecutive: int = 5,
        max_total_tokens: int = 100000,
    ):
        self.max_iterations = max_iterations
        self.max_same_tool_consecutive = max_same_tool_consecutive
        self.max_total_tokens = max_total_tokens
        self.iteration_count = 0
        self.tool_history: list[str] = []
        self.total_tokens = 0

    def check(self, tool_name: str | None = None,
              tokens_used: int = 0) -> str | None:
        """返回 None 表示正常，返回字符串表示应中断及原因"""
        self.iteration_count += 1
        self.total_tokens += tokens_used
        if tool_name:
            self.tool_history.append(tool_name)

        # 检查 1：最大迭代次数
        if self.iteration_count > self.max_iterations:
            return f"已达最大迭代次数 ({self.max_iterations})"

        # 检查 2：连续调用同一工具
        if (tool_name and
            len(self.tool_history) >= self.max_same_tool_consecutive):
            recent = self.tool_history[-self.max_same_tool_consecutive:]
            if len(set(recent)) == 1:
                return f"连续 {self.max_same_tool_consecutive} 次调用 '{tool_name}'"

        # 检查 3：Token 用量超限
        if self.total_tokens > self.max_total_tokens:
            return f"Token 用量超过限制 ({self.max_total_tokens})"

        # 检查 4：重复模式（如 A→B→A→B 循环）
        if len(self.tool_history) >= 6:
            for plen in [2, 3]:
                pattern = self.tool_history[-plen:]
                prev = self.tool_history[-2*plen:-plen]
                if pattern == prev:
                    return f"检测到重复模式: {pattern}"

        return None
```

## 容器化部署

### Dockerfile

```dockerfile
# 多阶段构建 —— 最终镜像更小
FROM python:3.11-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.11-slim

WORKDIR /app
COPY --from=builder /install /usr/local
COPY . .

# 非 root 用户运行（安全最佳实践）
RUN useradd -m appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health')"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
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
- [FastAPI Deployment Guide](https://fastapi.tiangolo.com/deployment/) -- 部署最佳实践
