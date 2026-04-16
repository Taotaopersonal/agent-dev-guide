# 性能优化 · 入门篇

::: tip 学习目标
- 分析 Agent 系统延迟的三大来源
- 掌握 Streaming（流式输出）降低感知延迟的方法
- 学会基本的异步调用，让你的 Agent 不再"傻等"
:::

::: info 学完你能做到
- 准确说出你的 Agent 的时间花在了哪里
- 为你的 Agent 添加流式输出，用户不再盯着空白屏幕等待
- 用 Python asyncio 实现基本的异步 LLM 调用
:::

## Agent 的延迟从哪来

用户发了一条消息，Agent 处理了 8 秒才回复。这 8 秒到底花在了哪里？

```
用户点击发送
    |
    v (~5ms)     网络传输 + 请求解析 —— 可以忽略
    |
    v (~200ms)   上下文准备：加载历史消息、组装工具描述
    |
    v (~1000ms)  LLM 首 Token 延迟（TTFT）—— 模型开始"思考"
    |
    v (~2000ms)  LLM 生成完成 —— 逐 Token 输出回答
    |
    v (~2000ms)  工具执行 —— 调用搜索API、查数据库
    |
    v (~1500ms)  LLM 第二轮 —— 分析工具结果，生成最终回答
    |
    v (~5ms)     响应传输
    |
用户看到回复，总计约 6.7 秒
```

三大延迟来源：

| 来源 | 占比 | 特点 |
|------|------|------|
| **LLM 推理** | 60-70% | 单次 1-5 秒，Agent 可能调用多轮 |
| **工具执行** | 20-30% | 取决于外部服务，网络请求 100ms-5s |
| **上下文准备** | 5-10% | 历史消息越长越慢 |

::: info 关键发现
Agent 的延迟主要来自 LLM 推理，而且多轮调用会让延迟线性叠加。一个需要 3 轮 LLM 调用的任务，延迟大约是单次调用的 3 倍。
:::

## 用延迟追踪器找到瓶颈

优化的第一步永远是"度量"——先搞清楚时间花在哪，再决定优化什么：

```python
"""latency_tracker.py — 延迟追踪工具"""

import time
from contextlib import contextmanager
from dataclasses import dataclass

@dataclass
class LatencyBreakdown:
    """延迟分解报告"""
    context_prep_ms: int = 0     # 上下文准备
    llm_total_ms: int = 0        # LLM 推理总耗时
    tool_exec_ms: int = 0        # 工具执行总耗时
    total_ms: int = 0            # 总耗时
    llm_calls: int = 0           # LLM 调用次数

    def __str__(self) -> str:
        return (
            f"总耗时: {self.total_ms}ms | "
            f"LLM: {self.llm_total_ms}ms ({self.llm_calls}次) | "
            f"工具: {self.tool_exec_ms}ms | "
            f"准备: {self.context_prep_ms}ms"
        )

class LatencyTracker:
    """延迟追踪器 —— 记录每个阶段的耗时"""

    def __init__(self):
        self.breakdown = LatencyBreakdown()
        self._start = time.monotonic()

    @contextmanager
    def track(self, stage: str):
        """追踪某个阶段的耗时

        用法:
            with tracker.track("llm"):
                response = await call_llm(messages)
        """
        start = time.monotonic()
        yield
        elapsed_ms = int((time.monotonic() - start) * 1000)

        if stage == "context_prep":
            self.breakdown.context_prep_ms += elapsed_ms
        elif stage == "llm":
            self.breakdown.llm_total_ms += elapsed_ms
            self.breakdown.llm_calls += 1
        elif stage == "tool":
            self.breakdown.tool_exec_ms += elapsed_ms

    def finish(self) -> LatencyBreakdown:
        """结束追踪，返回完整报告"""
        self.breakdown.total_ms = int(
            (time.monotonic() - self._start) * 1000
        )
        return self.breakdown


# 使用示例
async def tracked_agent_run(message: str):
    tracker = LatencyTracker()

    with tracker.track("context_prep"):
        messages = [{"role": "user", "content": message}]

    with tracker.track("llm"):
        response = await call_llm(messages)

    # 如果需要工具调用
    if needs_tool_call(response):
        with tracker.track("tool"):
            tool_result = await execute_tool(response)

        with tracker.track("llm"):
            final = await call_llm(messages + [tool_result])

    report = tracker.finish()
    print(report)
    # 总耗时: 4200ms | LLM: 3500ms (2次) | 工具: 500ms | 准备: 10ms
```

## Streaming：让用户不再干等

### 问题：用户盯着空白屏幕

没有 Streaming 时，用户的体验是：

```
[发送消息] → [等3秒...空白...] → [突然出现一大段文字]
```

有了 Streaming，体验变成：

```
[发送消息] → [0.5秒后开始出字] → [文字逐渐出现] → [完成]
```

**Streaming 不会减少总耗时**，但感知延迟从"等全部生成完"降为"首个 Token 到达的时间"，通常只有 0.5-1 秒。

### 实现流式输出

```python
"""streaming_agent.py — 支持流式输出的 Agent"""

import anthropic
from typing import AsyncIterator

async def stream_chat(message: str) -> AsyncIterator[str]:
    """流式对话 —— 逐块返回文字"""
    client = anthropic.AsyncAnthropic()

    # 用 stream() 替代 create()
    async with client.messages.stream(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": message}],
    ) as stream:
        async for text in stream.text_stream:
            yield text  # 每产生一小段文字就立即返回


# 在终端中测试流式输出
async def demo():
    print("Agent: ", end="", flush=True)
    async for chunk in stream_chat("用三句话介绍 Python"):
        print(chunk, end="", flush=True)
    print()  # 换行

# asyncio.run(demo())
```

### 配合 FastAPI WebSocket 使用

```python
"""streaming_api.py — WebSocket 流式聊天端点"""

from fastapi import FastAPI, WebSocket

app = FastAPI()

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket 流式聊天"""
    await websocket.accept()

    while True:
        # 接收用户消息
        data = await websocket.receive_json()
        message = data["message"]

        # 流式返回回复
        async for chunk in stream_chat(message):
            await websocket.send_json({
                "type": "text_delta",
                "content": chunk,
            })

        # 告诉前端消息结束
        await websocket.send_json({"type": "done"})
```

::: tip 配合 Streaming 的工具调用
当 Agent 需要调用工具时，可以在调用前发一个提示，让用户知道 Agent 在做什么：
```python
await websocket.send_json({
    "type": "status",
    "content": "正在搜索相关资料...",
})
```
这样用户就不会在工具执行期间觉得系统卡住了。
:::

## 基本的异步调用

### 同步 vs 异步

```python
import asyncio
import time

# 同步方式：一个接一个等
def sync_demo():
    """同步执行 3 个 API 调用"""
    start = time.monotonic()
    result1 = call_api_sync("query1")  # 等 1 秒
    result2 = call_api_sync("query2")  # 再等 1 秒
    result3 = call_api_sync("query3")  # 再等 1 秒
    print(f"同步耗时: {time.monotonic() - start:.1f}s")  # 约 3 秒

# 异步方式：同时发出，一起等
async def async_demo():
    """异步并行执行 3 个 API 调用"""
    start = time.monotonic()
    result1, result2, result3 = await asyncio.gather(
        call_api_async("query1"),  # 这三个
        call_api_async("query2"),  # 同时执行
        call_api_async("query3"),  # 一起等结果
    )
    print(f"异步耗时: {time.monotonic() - start:.1f}s")  # 约 1 秒
```

### 为什么 Agent 适合异步

Agent 的核心操作——LLM 调用、HTTP 请求、数据库查询——都是 I/O 密集型。也就是说，大部分时间 CPU 在"等网络响应"而不是"做计算"。异步编程让 CPU 在等待期间去处理其他请求。

### 使用 Anthropic 异步客户端

```python
"""async_agent.py — 异步 Agent 基础"""

import anthropic
import asyncio

# 用 AsyncAnthropic 替代 Anthropic
client = anthropic.AsyncAnthropic()

async def chat(message: str) -> str:
    """异步聊天"""
    response = await client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": message}],
    )
    return response.content[0].text

# 同时处理多个用户的请求
async def handle_multiple_users():
    """并行处理多个用户请求"""
    results = await asyncio.gather(
        chat("用户A的问题：Python是什么？"),
        chat("用户B的问题：JavaScript是什么？"),
        chat("用户C的问题：Rust是什么？"),
    )
    for i, result in enumerate(results):
        print(f"用户{chr(65+i)}: {result[:50]}...")

# asyncio.run(handle_multiple_users())
```

### 并行执行多个工具调用

当 LLM 一次返回多个工具调用请求时，并行执行而不是串行：

```python
async def parallel_tool_execution(tool_calls: list[dict]) -> list[dict]:
    """并行执行多个工具调用

    如果 LLM 同时请求搜索天气和查数据库，
    串行需要 2 秒（各 1 秒），并行只需要 1 秒。
    """
    async def execute_one(call: dict) -> dict:
        try:
            result = await asyncio.wait_for(
                execute_tool(call["name"], call["input"]),
                timeout=30.0,
            )
            return {"tool_use_id": call["id"], "content": result}
        except asyncio.TimeoutError:
            return {
                "tool_use_id": call["id"],
                "content": f"工具 {call['name']} 执行超时",
            }

    # asyncio.gather 让所有工具同时开始执行
    results = await asyncio.gather(
        *[execute_one(call) for call in tool_calls]
    )
    return list(results)
```

## 小结

Agent 性能优化从三件事开始：

1. **度量延迟**：用 LatencyTracker 搞清楚时间花在哪，LLM 推理通常占 60-70%
2. **Streaming 输出**：用 `stream()` 替代 `create()`，感知延迟从秒级降到亚秒级
3. **异步调用**：用 `AsyncAnthropic` 和 `asyncio.gather`，并行处理多个请求和工具调用

这三个改动投入小、收益大，是性能优化的第一步。进阶篇会介绍更系统化的并发控制和缓存策略。

## 练习

1. 给你现有的 Agent 加上 `LatencyTracker`，运行 10 次，统计平均延迟分布
2. 把同步的 `client.messages.create()` 改为流式的 `client.messages.stream()`，对比用户体验的变化
3. 写一个程序，同时向 LLM 发送 5 个不同问题，对比串行和并行的总耗时

## 参考资源

- [Anthropic Streaming API](https://docs.anthropic.com/en/api/messages-streaming) -- Claude 流式 API 文档
- [Python asyncio 文档](https://docs.python.org/3/library/asyncio.html) -- Python 异步编程官方文档
- [Real Python - Async IO](https://realpython.com/async-io-python/) -- asyncio 实战教程
- [Vercel AI SDK](https://sdk.vercel.ai/) -- 前端 AI Streaming 工具链
