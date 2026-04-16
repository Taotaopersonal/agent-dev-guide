# Python 基础 · 中级

::: info 学习目标
- 掌握 Python async/await 异步编程，理解与 JS Event Loop 的区别
- 学会装饰器的原理和自定义写法
- 理解上下文管理器（with 语句）的作用
- 掌握包管理和虚拟环境的使用
- 用 Pydantic 定义数据模型并做运行时类型验证
- 学完能写异步服务，管理项目依赖

预计学习时间：3-4 小时
:::

## async/await 异步编程

这是从 JS 转 Python 最需要注意的差异。JS 的异步是"默认非阻塞"的，Python 的异步需要"显式选择"。

### JS vs Python 异步对比

```javascript
// JS: 所有 I/O 天然异步，Event Loop 是运行时内置的
const response = await fetch(url);
const data = await response.json();
```

```python
# Python: 必须使用 async 库，否则默认阻塞
import aiohttp

async with aiohttp.ClientSession() as session:
    async with session.get(url) as response:
        data = await response.json()
```

::: warning 关键区别
JavaScript 的 `fetch` 天然就是异步的。Python 的 `requests.get()` 是**阻塞的**，你必须显式使用 `aiohttp` 或 `httpx` 的异步版本。在 Agent 开发中，你会频繁调用 LLM API，异步至关重要。
:::

### asyncio 基础

```python
import asyncio

# 定义异步函数
async def fetch_data(url: str) -> dict:
    """异步获取数据"""
    print(f"开始请求 {url}")
    await asyncio.sleep(1)  # 模拟网络请求
    print(f"请求完成 {url}")
    return {"url": url, "status": "ok"}

# 运行单个异步函数
async def main():
    result = await fetch_data("https://api.example.com")
    print(result)

asyncio.run(main())
```

### 并发执行多个任务

这是异步编程最大的价值 -- 同时发起多个请求，而不是一个接一个等：

```python
import asyncio
import time

async def call_llm(prompt: str) -> str:
    """模拟 LLM API 调用（耗时 2 秒）"""
    await asyncio.sleep(2)
    return f"回复: {prompt}"

async def main():
    prompts = ["问题1", "问题2", "问题3"]

    # 串行调用：6 秒
    start = time.time()
    for p in prompts:
        await call_llm(p)
    print(f"串行耗时: {time.time() - start:.1f}s")

    # 并行调用：2 秒
    start = time.time()
    results = await asyncio.gather(
        call_llm("问题1"),
        call_llm("问题2"),
        call_llm("问题3"),
    )
    print(f"并行耗时: {time.time() - start:.1f}s")
    print(f"结果: {results}")

asyncio.run(main())
```

### 实际应用：异步 LLM 调用

```python
import anthropic
import asyncio

async def parallel_calls():
    """并行调用多个 LLM 请求"""
    client = anthropic.AsyncAnthropic()  # 使用异步客户端

    tasks = [
        client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=128,
            messages=[{"role": "user", "content": f"用一句话解释{topic}"}]
        )
        for topic in ["递归", "闭包", "协程"]
    ]

    responses = await asyncio.gather(*tasks)
    for resp in responses:
        print(f"- {resp.content[0].text}")

asyncio.run(parallel_calls())
```

### 异步上下文管理器

```python
import aiohttp

async def fetch_json(url: str) -> dict:
    """异步获取 JSON 数据"""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()
    # session 自动关闭，连接自动释放
```

## 装饰器

装饰器是 Python 的"高阶函数语法糖"，在 FastAPI 和 Agent 框架中大量使用。本质上，装饰器就是一个接收函数、返回函数的函数。

### 理解装饰器原理

```python
import functools
import time

def timer(func):
    """测量函数执行时间的装饰器"""
    @functools.wraps(func)  # 保留原函数的名字和文档
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        print(f"{func.__name__} 耗时 {duration:.2f}s")
        return result
    return wrapper

@timer
def slow_function():
    time.sleep(1)
    return "done"

# @timer 等价于: slow_function = timer(slow_function)
slow_function()  # 输出: slow_function 耗时 1.00s
```

### 带参数的装饰器

```python
import functools
import time

def retry(max_attempts: int = 3, delay: float = 1.0):
    """自动重试装饰器"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    print(f"第 {attempt + 1} 次失败: {e}")
                    if attempt < max_attempts - 1:
                        time.sleep(delay)
            raise last_error
        return wrapper
    return decorator

@retry(max_attempts=3, delay=0.5)
def unreliable_api_call():
    """模拟不稳定的 API"""
    import random
    if random.random() < 0.7:
        raise ConnectionError("连接失败")
    return "成功"
```

### Agent 开发中常见的装饰器

```python
# FastAPI 路由装饰器
from fastapi import FastAPI
app = FastAPI()

@app.get("/api/chat")
async def chat(message: str):
    return {"reply": "Hello!"}

# Pydantic 字段验证器
from pydantic import BaseModel, field_validator

class ToolInput(BaseModel):
    query: str

    @field_validator("query")
    @classmethod
    def query_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("查询不能为空")
        return v.strip()
```

## 上下文管理器

Python 的 `with` 语句用于自动管理资源的获取和释放。退出 `with` 块时自动清理资源，即使发生异常也不会遗漏。

### 基础用法

```python
# 文件操作 -- 最经典的 with 用法
with open("data.txt", "r") as file:
    content = file.read()
# 退出 with 块时自动关闭文件

# 对比 JS 的写法
# let file;
# try { file = await fs.open("data.txt"); ... }
# finally { if (file) await file.close(); }
```

### 自定义上下文管理器

```python
from contextlib import contextmanager
import time

@contextmanager
def timer(label: str):
    """计时器上下文管理器"""
    start = time.time()
    yield  # yield 之前是 __enter__，之后是 __exit__
    duration = time.time() - start
    print(f"{label} 耗时 {duration:.2f}s")

# 使用
with timer("LLM 调用"):
    # 模拟 LLM API 调用
    time.sleep(1.5)
# 输出: LLM 调用 耗时 1.50s
```

### Agent 开发中的实际用法

```python
# Anthropic SDK 的流式输出就用了 with
import anthropic

client = anthropic.Anthropic()

with client.messages.stream(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{"role": "user", "content": "你好"}]
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
# 退出 with 块时自动关闭流连接
```

## 包管理和虚拟环境

### 虚拟环境：为什么需要

JS 有 node_modules 做项目级依赖隔离，Python 用虚拟环境做同样的事：

```bash
# uv 自动管理虚拟环境（推荐）
uv init my-project
cd my-project
uv add anthropic pydantic  # 自动创建 .venv 并安装

# 手动管理（了解原理）
python3 -m venv .venv           # 创建虚拟环境
source .venv/bin/activate       # 激活（macOS/Linux）
pip install anthropic           # 安装到虚拟环境
deactivate                      # 退出虚拟环境
```

### pyproject.toml -- Python 的 package.json

```toml
[project]
name = "my-agent"
version = "0.1.0"
description = "My first Agent project"
requires-python = ">=3.12"
dependencies = [
    "anthropic>=0.40.0",
    "pydantic>=2.0",
    "fastapi>=0.100.0",
    "uvicorn>=0.30.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "ruff>=0.5.0",
]
```

### 常用 uv 命令

```bash
uv add anthropic          # 安装依赖（类似 npm install）
uv add --dev pytest       # 安装开发依赖（类似 npm install -D）
uv remove anthropic       # 移除依赖
uv run python main.py     # 在虚拟环境中运行
uv sync                   # 同步依赖（类似 npm ci）
uv lock                   # 锁定依赖版本（类似 npm lock）
```

## Pydantic 数据模型

Pydantic 是 Python 的运行时类型验证库，在 Agent 开发中地位极其重要。它做的事情是：定义数据结构，传入数据时自动校验类型，不合法就报错。

### 基础用法

```python
from pydantic import BaseModel, Field
from enum import Enum

class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class Message(BaseModel):
    """对话消息"""
    role: Role
    content: str
    name: str | None = None  # 可选字段

# 正确使用
msg = Message(role="user", content="你好")
print(msg.role)       # Role.USER
print(msg.model_dump())  # {'role': 'user', 'content': '你好', 'name': None}

# 类型不对会直接报错
try:
    bad = Message(role="invalid", content=123)
except Exception as e:
    print(f"验证失败: {e}")
```

### 嵌套模型和 Field

```python
from pydantic import BaseModel, Field

class ToolParameter(BaseModel):
    name: str
    type: str = Field(description="参数类型: string/number/boolean")
    required: bool = True
    description: str = ""

class ToolDefinition(BaseModel):
    """工具定义 -- Agent 开发中的核心数据结构"""
    name: str = Field(min_length=1, max_length=64)
    description: str = Field(min_length=1)
    parameters: list[ToolParameter] = Field(default_factory=list)

# 使用
tool = ToolDefinition(
    name="web_search",
    description="搜索互联网获取信息",
    parameters=[
        ToolParameter(name="query", type="string", description="搜索关键词"),
        ToolParameter(name="max_results", type="number", required=False),
    ]
)

# 转为 JSON Schema（LLM 工具定义常用）
import json
print(json.dumps(tool.model_json_schema(), indent=2, ensure_ascii=False))
```

### 从 JSON 解析

```python
from pydantic import BaseModel

class LLMResponse(BaseModel):
    content: str
    model: str
    input_tokens: int
    output_tokens: int

# 从 JSON 字符串解析
json_str = '{"content": "你好", "model": "claude", "input_tokens": 10, "output_tokens": 5}'
resp = LLMResponse.model_validate_json(json_str)
print(resp.content)  # "你好"

# 从 dict 解析
data = {"content": "你好", "model": "claude", "input_tokens": 10, "output_tokens": 5}
resp = LLMResponse.model_validate(data)
```

### 自定义验证器

```python
from pydantic import BaseModel, field_validator

class ChatRequest(BaseModel):
    message: str
    temperature: float = 0.7
    max_tokens: int = 1024

    @field_validator("temperature")
    @classmethod
    def check_temperature(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError("temperature 必须在 0 到 1 之间")
        return v

    @field_validator("message")
    @classmethod
    def check_message(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("消息不能为空")
        return v

# 验证通过
req = ChatRequest(message="你好", temperature=0.5)

# 验证失败
try:
    bad = ChatRequest(message="", temperature=1.5)
except Exception as e:
    print(f"验证失败: {e}")
```

## 综合实战：异步 LLM 客户端

把本节学到的知识串起来，写一个简洁的异步 LLM 客户端：

```python
"""异步 LLM 客户端 -- 综合运用 async、Pydantic、上下文管理器"""

import anthropic
import asyncio
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

# 数据模型
class ChatRequest(BaseModel):
    message: str
    system: str = ""
    max_tokens: int = Field(default=1024, ge=1, le=4096)

class ChatResponse(BaseModel):
    content: str
    input_tokens: int
    output_tokens: int

# 异步客户端
class AsyncLLMClient:
    def __init__(self):
        self.client = anthropic.AsyncAnthropic()

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """发送对话请求"""
        kwargs = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": request.max_tokens,
            "messages": [{"role": "user", "content": request.message}],
        }
        if request.system:
            kwargs["system"] = request.system

        response = await self.client.messages.create(**kwargs)

        return ChatResponse(
            content=response.content[0].text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    async def batch_chat(self, requests: list[ChatRequest]) -> list[ChatResponse]:
        """并行处理多个请求"""
        tasks = [self.chat(req) for req in requests]
        return await asyncio.gather(*tasks)

# 运行
async def main():
    client = AsyncLLMClient()

    # 单个请求
    resp = await client.chat(ChatRequest(message="什么是 Agent？"))
    print(f"回复: {resp.content[:100]}...")
    print(f"Token: {resp.input_tokens} in / {resp.output_tokens} out")

    # 批量并行请求
    requests = [
        ChatRequest(message="什么是 RAG？"),
        ChatRequest(message="什么是 Tool Use？"),
    ]
    results = await client.batch_chat(requests)
    for r in results:
        print(f"- {r.content[:50]}...")

asyncio.run(main())
```

## 小结

1. **async/await** 让你能并发调用多个 LLM API，显著提升吞吐量。记住 Python 的异步需要显式选择异步库
2. **装饰器** 是函数的函数，在 FastAPI 路由、重试逻辑中大量使用
3. **上下文管理器** (with) 确保资源自动清理，流式输出、数据库连接都靠它
4. **uv + pyproject.toml** 是现代 Python 项目管理标配，类比 pnpm + package.json
5. **Pydantic** 提供运行时类型验证，是 Agent 工具定义和 API 响应解析的基石

## 练习

1. **异步练习**：写一个异步函数，用 `asyncio.gather` 同时向 3 个不同的 URL 发起请求（可以用 `asyncio.sleep` 模拟），对比串行和并行的耗时差异。

2. **装饰器练习**：写一个 `@retry(max_attempts=3)` 装饰器，在函数抛出异常时自动重试，支持配置重试次数和延迟时间。

3. **Pydantic 练习**：用 Pydantic 定义一个 `AgentConfig` 模型，包含 `model_name`（枚举类型，只允许 claude/gpt-4o）、`temperature`（0-1 的浮点数）、`tools`（ToolDefinition 列表），要求所有字段都有验证逻辑。

4. **综合练习**：基于上面的 AsyncLLMClient，增加重试逻辑（用装饰器）和请求超时处理。

## 参考资源

- [Python asyncio 文档](https://docs.python.org/3/library/asyncio.html)
- [Pydantic V2 文档](https://docs.pydantic.dev/latest/)
- [Real Python - Async IO](https://realpython.com/async-io-python/)
- [uv 文档](https://docs.astral.sh/uv/)
