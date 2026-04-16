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

```python
"""concurrency.py — 并发请求管理器"""

import asyncio

class ConcurrencyManager:
    """双层并发控制：全局限流 + 每用户限流"""

    def __init__(
        self,
        max_concurrent: int = 50,    # 全局最大并发
        max_per_user: int = 3,       # 每用户最大并发
        queue_timeout: float = 60.0, # 排队超时
    ):
        self.global_sem = asyncio.Semaphore(max_concurrent)
        self.user_sems: dict[str, asyncio.Semaphore] = {}
        self.max_per_user = max_per_user
        self.queue_timeout = queue_timeout

    def _get_user_sem(self, user_id: str) -> asyncio.Semaphore:
        if user_id not in self.user_sems:
            self.user_sems[user_id] = asyncio.Semaphore(self.max_per_user)
        return self.user_sems[user_id]

    async def execute(self, user_id: str, coro):
        """带并发控制的请求执行"""
        user_sem = self._get_user_sem(user_id)

        try:
            await asyncio.wait_for(
                self._acquire_both(user_sem),
                timeout=self.queue_timeout,
            )
        except asyncio.TimeoutError:
            raise Exception("请求排队超时，请稍后重试")

        try:
            return await coro
        finally:
            self.global_sem.release()
            user_sem.release()

    async def _acquire_both(self, user_sem: asyncio.Semaphore):
        await self.global_sem.acquire()
        try:
            await user_sem.acquire()
        except Exception:
            self.global_sem.release()
            raise
```

### 并行工具执行器

```python
"""parallel_tools.py — 带并发控制的并行工具执行"""

import asyncio
import time
from dataclasses import dataclass

@dataclass
class ToolCallResult:
    tool_id: str
    tool_name: str
    output: str
    success: bool
    duration_ms: int

class ParallelToolExecutor:
    """并行执行工具调用，带超时和并发控制"""

    def __init__(self, max_concurrent: int = 10, timeout: float = 30.0):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.timeout = timeout

    async def execute_all(self, tool_calls: list[dict]) -> list[ToolCallResult]:
        tasks = [self._execute_single(call) for call in tool_calls]
        return await asyncio.gather(*tasks)

    async def _execute_single(self, call: dict) -> ToolCallResult:
        start = time.monotonic()
        async with self.semaphore:
            try:
                result = await asyncio.wait_for(
                    execute_tool(call["name"], call["input"]),
                    timeout=self.timeout,
                )
                return ToolCallResult(
                    tool_id=call["id"], tool_name=call["name"],
                    output=result, success=True,
                    duration_ms=int((time.monotonic() - start) * 1000),
                )
            except asyncio.TimeoutError:
                return ToolCallResult(
                    tool_id=call["id"], tool_name=call["name"],
                    output=f"超时 ({self.timeout}s)", success=False,
                    duration_ms=int((time.monotonic() - start) * 1000),
                )
            except Exception as e:
                return ToolCallResult(
                    tool_id=call["id"], tool_name=call["name"],
                    output=f"错误: {e}", success=False,
                    duration_ms=int((time.monotonic() - start) * 1000),
                )
```

## Prompt Caching：节省 80% 输入成本

Agent 的每轮 LLM 调用都要发送 System Prompt 和工具定义，这些内容每次都一样。Anthropic 的 Prompt Caching 允许你标记这些内容为可缓存，后续请求复用缓存。

### 实现 Prompt Caching

```python
"""prompt_caching.py — 带缓存的 LLM 客户端"""

import anthropic

class CachedLLMClient:
    """支持 Prompt Caching 的 LLM 客户端"""

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.client = anthropic.AsyncAnthropic()
        self.model = model
        self._stats = {"hits": 0, "misses": 0, "tokens_saved": 0}

    async def call(
        self,
        system_prompt: str,
        tools: list[dict],
        messages: list[dict],
    ) -> dict:
        # System Prompt 标记为可缓存
        cached_system = [{
            "type": "text",
            "text": system_prompt,
            "cache_control": {"type": "ephemeral"},
        }]

        # 最后一个工具定义标记为可缓存
        cached_tools = []
        for i, tool in enumerate(tools):
            t = dict(tool)
            if i == len(tools) - 1:
                t["cache_control"] = {"type": "ephemeral"}
            cached_tools.append(t)

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=cached_system,
            tools=cached_tools,
            messages=messages,
        )

        # 追踪缓存命中
        cache_read = getattr(response.usage, 'cache_read_input_tokens', 0)
        if cache_read > 0:
            self._stats["hits"] += 1
            self._stats["tokens_saved"] += cache_read
        else:
            self._stats["misses"] += 1

        return response
```

### 缓存节省了多少钱

```python
"""cache_savings.py — 计算缓存节省的成本"""

def calculate_cache_savings(
    cacheable_tokens: int,       # System Prompt + 工具定义的 token 数
    requests_per_hour: int,      # 每小时请求数
    model: str = "claude-sonnet-4-20250514",
) -> dict:
    # Sonnet 定价
    input_price = 3.0 / 1_000_000       # $3 / 1M tokens
    cache_write_price = 3.75 / 1_000_000 # $3.75 / 1M tokens
    cache_read_price = 0.3 / 1_000_000   # $0.3 / 1M tokens (仅 input 的 10%)

    # 无缓存成本
    no_cache = cacheable_tokens * requests_per_hour * input_price

    # 有缓存成本：第一次写入 + 后续读取
    with_cache = (
        cacheable_tokens * cache_write_price  # 写入一次
        + cacheable_tokens * (requests_per_hour - 1) * cache_read_price  # 后续读取
    )

    return {
        "hourly_without_cache": f"${no_cache:.4f}",
        "hourly_with_cache": f"${with_cache:.4f}",
        "savings_pct": f"{(1 - with_cache / no_cache) * 100:.1f}%",
    }

# 示例：10000 token 的 System Prompt + 工具定义，每小时 100 次请求
print(calculate_cache_savings(10000, 100))
# {'hourly_without_cache': '$3.0000',
#  'hourly_with_cache': '$0.3345',
#  'savings_pct': '88.9%'}
```

## 工具结果缓存

有些工具的结果短时间内不会变化（如数据库表结构、天气查询），可以缓存避免重复调用：

```python
"""tool_cache.py — 工具结果缓存"""

import hashlib
import json
import time

class ToolResultCache:
    """根据工具类型设置不同的缓存时间"""

    TOOL_TTL = {
        "get_weather": 600,          # 天气: 10 分钟
        "search_documents": 300,      # 文档搜索: 5 分钟
        "list_tables": 3600,          # 数据库结构: 1 小时
        "describe_table": 3600,       # 表结构: 1 小时
        "query_database": 60,         # 数据查询: 1 分钟
    }

    def __init__(self):
        self._cache: dict[str, tuple[str, float]] = {}
        self._hits = 0
        self._misses = 0

    def _make_key(self, tool_name: str, tool_input: dict) -> str:
        content = json.dumps(
            {"tool": tool_name, "input": tool_input}, sort_keys=True
        )
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, tool_name: str, tool_input: dict) -> str | None:
        key = self._make_key(tool_name, tool_input)
        if key in self._cache:
            result, timestamp = self._cache[key]
            ttl = self.TOOL_TTL.get(tool_name, 60)
            if time.time() - timestamp < ttl:
                self._hits += 1
                return result
            del self._cache[key]
        self._misses += 1
        return None

    def set(self, tool_name: str, tool_input: dict, result: str):
        key = self._make_key(tool_name, tool_input)
        self._cache[key] = (result, time.time())

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

# 集成到工具执行流程
tool_cache = ToolResultCache()

async def cached_tool_execute(tool_name: str, tool_input: dict) -> str:
    """带缓存的工具执行"""
    cached = tool_cache.get(tool_name, tool_input)
    if cached is not None:
        return cached

    result = await execute_tool(tool_name, tool_input)
    tool_cache.set(tool_name, tool_input, result)
    return result
```

## 连接池

复用 HTTP 和数据库连接，避免每次请求都重新建立连接：

```python
"""connection_pool.py — 连接池管理"""

import httpx

class ConnectionPool:
    def __init__(self):
        self._http: httpx.AsyncClient | None = None
        self._db = None

    async def initialize(self):
        self._http = httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=100,
                max_keepalive_connections=20,
            ),
            timeout=httpx.Timeout(30.0, connect=5.0),
        )

        import asyncpg
        self._db = await asyncpg.create_pool(
            dsn="postgresql://user:pass@localhost/agent_db",
            min_size=5, max_size=20,
        )

    @property
    def http(self) -> httpx.AsyncClient:
        if not self._http:
            raise RuntimeError("连接池未初始化")
        return self._http

    @property
    def db(self):
        if not self._db:
            raise RuntimeError("数据库连接池未初始化")
        return self._db

    async def cleanup(self):
        if self._http:
            await self._http.aclose()
        if self._db:
            await self._db.close()
```

## 完整的异步 Agent 引擎

把以上所有模块整合在一起：

```python
"""async_engine.py — 生产级异步 Agent 引擎"""

import anthropic
from typing import AsyncIterator

class AsyncAgentEngine:
    """整合并发控制、缓存、连接池的 Agent 引擎"""

    def __init__(self):
        self.client = anthropic.AsyncAnthropic(max_retries=3)
        self.tools = ParallelToolExecutor(max_concurrent=10)
        self.concurrency = ConcurrencyManager(max_concurrent_requests=50)
        self.tool_cache = ToolResultCache()
        self.pool = ConnectionPool()

    async def initialize(self):
        await self.pool.initialize()

    async def run(self, user_id: str, message: str,
                  model: str = "claude-sonnet-4-20250514") -> str:
        return await self.concurrency.execute(
            user_id,
            self._agent_loop(message, model),
        )

    async def _agent_loop(self, message: str, model: str) -> str:
        messages = [{"role": "user", "content": message}]

        for _ in range(25):
            response = await self.client.messages.create(
                model=model, max_tokens=4096, messages=messages,
            )

            if response.stop_reason != "tool_use":
                return "".join(
                    b.text for b in response.content if hasattr(b, "text")
                )

            tool_calls = [
                {"id": b.id, "name": b.name, "input": b.input}
                for b in response.content if b.type == "tool_use"
            ]

            results = await self.tools.execute_all(tool_calls)

            messages.append({"role": "assistant", "content": response.content})
            messages.append({
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": r.tool_id,
                     "content": r.output, "is_error": not r.success}
                    for r in results
                ],
            })

        return "达到最大迭代次数。"
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
- [Python asyncio 文档](https://docs.python.org/3/library/asyncio.html) -- 异步编程
- [httpx 文档](https://www.python-httpx.org/) -- 异步 HTTP 客户端
- [asyncpg 文档](https://magicstack.github.io/asyncpg/) -- 异步 PostgreSQL 客户端
- [Redis 官方文档](https://redis.io/docs/) -- 缓存数据库
